//! Experimental FP32 GPU `process_many` executor using chunked affine scans.
//!
//! The accepted CPU fast and oracle paths do not call into this module.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::mem::size_of;
use std::sync::{mpsc, Arc};
use std::time::Instant;

use anyhow::{anyhow, bail, ensure, Context, Result};
use candle_core::{Device, Tensor};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::bulk::feature_prepass_step_into;
use super::gpu::{GpuLayerMeta, PackedLinear, PackedNorm, WeightPacker};
use super::process_payload::parse_process_review_payload;
use super::runtime::validate_num_threads;
use super::state::{FlatNativeRnnModuleState, FlatNativeRnnState, NativeRnnModuleState, ReviewIds};
use super::{py_value_error, NativeRnn};
use crate::gpu::{
    is_gpu_out_of_memory, py_gpu_error, py_gpu_unavailable, py_gpu_unavailable_error, GpuContext,
    GpuOperation,
};
use crate::model_weights::SrsRwkvRnnWeights;
use crate::ops::f32_tensor_data;
use crate::state::{FeatureState, ReviewInput};

pub(super) type GpuScanProcessOutput = (Vec<f64>, Option<Vec<f32>>, Option<Vec<f32>>);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GpuProcessScanMode {
    Predictions,
    Curves,
    StateOnly,
}

impl GpuProcessScanMode {
    fn process(return_curves: bool) -> Self {
        if return_curves {
            Self::Curves
        } else {
            Self::Predictions
        }
    }

    fn rows_per_review(self) -> usize {
        if self == Self::StateOnly {
            1
        } else {
            2
        }
    }

    fn process_row_offset(self) -> usize {
        self.rows_per_review() - 1
    }

    fn returns_predictions(self) -> bool {
        self != Self::StateOnly
    }

    fn returns_curves(self) -> bool {
        self == Self::Curves
    }

    fn label(self) -> &'static str {
        match self {
            Self::Predictions => "predictions",
            Self::Curves => "curves",
            Self::StateOnly => "state_only",
        }
    }
}

const FEATURE_DIM: usize = 92;
const CHANNELS: usize = 128;
const HEADS: usize = 4;
const HEAD_SIZE: usize = 32;
const STATE_LAYER_ELEMENTS: usize = 128 + HEADS * HEAD_SIZE * HEAD_SIZE + 128;
const TRANSFORM_MATRIX_ELEMENTS: usize = HEADS * HEAD_SIZE * HEAD_SIZE;
const TRANSFORM_ELEMENTS: usize = 2 * TRANSFORM_MATRIX_ELEMENTS;
const DEFAULT_CHUNK_REVIEWS: usize = 512;
const MAX_OUTPUT_READBACK_BYTES_PER_REVIEW: usize = (1 + 2 * CHANNELS) * size_of::<f32>();
const PROCESS_STATE_BUFFER_LIMIT_ENV_VAR: &str = "RWKV_SRS_GPU_PROCESS_STATE_BUFFER_LIMIT_BYTES";
const FULLY_RESIDENT_STATE_ENV_VAR: &str = "RWKV_SRS_GPU_PROCESS_FULLY_RESIDENT_STATE";
const FULLY_RESIDENT_SHARD_LIMIT_ENV_VAR: &str =
    "RWKV_SRS_GPU_PROCESS_FULLY_RESIDENT_SHARD_LIMIT_BYTES";
const SHARED_CPU_STATE_VIEWS_ENV_VAR: &str = "RWKV_SRS_GPU_PROCESS_SHARED_CPU_STATE_VIEWS";
const FLAT_CPU_STATE_ENV_VAR: &str = "RWKV_SRS_GPU_PROCESS_FLAT_CPU_STATE";
const PIPELINED_CPU_STATE_SYNC_ENV_VAR: &str = "RWKV_SRS_GPU_PROCESS_PIPELINED_CPU_STATE_SYNC";
const FUSED_ROW_BATCHED_PROJECTIONS_ENV_VAR: &str =
    "RWKV_SRS_GPU_PROCESS_FUSED_ROW_BATCHED_PROJECTIONS";
// Timestamp sweeps showed small batches losing to workgroup underfill. At
// 2,048 interleaved rows the fused GPU-stage total crossed below the retained
// per-row kernels; ordinary max-sized process batches are well above it.
const FUSED_ROW_BATCHED_MIN_ROWS: usize = 2048;
const FULLY_RESIDENT_SYNC_CHUNK_BYTES: usize = 32 * 1024 * 1024;
const INVALID_INDEX: u32 = u32::MAX;
const ROW_HAS_STATE: u32 = 1;
const ROW_LAST_PROCESS: u32 = 2;
const CHUNK_LAST: u32 = 1;
const STATE_SOURCE_RESIDENT: u32 = 1;

#[derive(Clone, Copy)]
#[repr(usize)]
enum GpuTimestampStage {
    FeatureEncoding,
    StateTransfer,
    TopologyGather,
    PrepareLayer,
    BuildTransforms,
    ScanTransforms,
    Replay,
    FinishTime,
    FinishChannel,
    TopologyScatter,
    OutputHeads,
}

impl GpuTimestampStage {
    const COUNT: usize = 11;
    const ALL: [Self; Self::COUNT] = [
        Self::FeatureEncoding,
        Self::StateTransfer,
        Self::TopologyGather,
        Self::PrepareLayer,
        Self::BuildTransforms,
        Self::ScanTransforms,
        Self::Replay,
        Self::FinishTime,
        Self::FinishChannel,
        Self::TopologyScatter,
        Self::OutputHeads,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::FeatureEncoding => "feature",
            Self::StateTransfer => "state_transfer",
            Self::TopologyGather => "gather",
            Self::PrepareLayer => "prepare",
            Self::BuildTransforms => "build_transforms",
            Self::ScanTransforms => "scan_transforms",
            Self::Replay => "replay",
            Self::FinishTime => "finish_time",
            Self::FinishChannel => "finish_channel",
            Self::TopologyScatter => "scatter",
            Self::OutputHeads => "heads",
        }
    }
}

struct GpuTimestampProfile {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    stages: Vec<GpuTimestampStage>,
    next_query: u32,
}

impl GpuTimestampProfile {
    fn new(context: &GpuContext, segments: usize) -> Result<Option<Self>> {
        if !scan_profile_enabled() || !context.timestamp_queries || segments == 0 {
            return Ok(None);
        }
        let Some(query_count) = segments
            .checked_mul(2)
            .and_then(|count| u32::try_from(count).ok())
        else {
            return Ok(None);
        };
        let query_set = context.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("rwkv-srs process scan stage timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: query_count,
        });
        let resolve_buffer = context.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rwkv-srs process scan stage timestamp resolve"),
            size: u64::from(query_count) * size_of::<u64>() as u64,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })?;
        Ok(Some(Self {
            query_set,
            resolve_buffer,
            stages: Vec::with_capacity(segments),
            next_query: 0,
        }))
    }

    fn pair(&mut self, stage: GpuTimestampStage) -> (u32, u32) {
        let pair = (self.next_query, self.next_query + 1);
        self.next_query += 2;
        self.stages.push(stage);
        pair
    }

    fn bytes(&self) -> usize {
        self.next_query as usize * size_of::<u64>()
    }

    fn resolve_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        readback: &wgpu::Buffer,
        offset: usize,
    ) {
        encoder.resolve_query_set(&self.query_set, 0..self.next_query, &self.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            readback,
            offset as u64,
            self.bytes() as u64,
        );
    }

    fn summary(&self, context: &GpuContext, bytes: &[u8]) -> Result<String> {
        let timestamps = bytemuck::try_cast_slice::<u8, u64>(bytes)
            .map_err(|error| anyhow!("malformed GPU process timestamp buffer: {error:?}"))?;
        ensure!(
            timestamps.len() == self.next_query as usize,
            "GPU process timestamp count mismatch"
        );
        let period = f64::from(context.queue.get_timestamp_period());
        let mut totals_ns = [0.0f64; GpuTimestampStage::COUNT];
        let mut counts = [0usize; GpuTimestampStage::COUNT];
        for (pair, stage) in timestamps.chunks_exact(2).zip(&self.stages) {
            totals_ns[*stage as usize] += pair[1].wrapping_sub(pair[0]) as f64 * period;
            counts[*stage as usize] += 1;
        }
        let fields = GpuTimestampStage::ALL
            .into_iter()
            .map(|stage| {
                format!(
                    "{}={:.3}ms/{}",
                    stage.label(),
                    totals_ns[stage as usize] / 1_000_000.0,
                    counts[stage as usize]
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        Ok(format!(
            "gpu_process_scan_stages total={:.3}ms {fields}",
            totals_ns.iter().sum::<f64>() / 1_000_000.0
        ))
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RowMeta {
    previous_process_row: u32,
    entity: u32,
    flags: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TransformMeta {
    start_row: u32,
    review_count: u32,
    entity: u32,
    local_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FinalChunkMeta {
    start_row: u32,
    review_count: u32,
    entity: u32,
    prefix_transform: u32,
    flags: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DispatchParams {
    row_count: u32,
    layer_index: u32,
    local_layer: u32,
    layers: u32,
    scan_offset: u32,
    scan_source_b: u32,
    transform_count: u32,
    final_chunk_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FourU32 {
    first: u32,
    second: u32,
    third: u32,
    fourth: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct StateSource {
    entity: u32,
    flags: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ArenaTransfer {
    arena_entity: u32,
    plan_entity: u32,
    _pad0: u32,
    _pad1: u32,
}

struct PackedScanModel {
    values: Vec<f32>,
    normalization_values: Vec<f32>,
    layer_metadata: Vec<GpuLayerMeta>,
    module_layer_offsets: Vec<usize>,
    features_input: PackedLinear,
    features_norm: PackedNorm,
    features_output: PackedLinear,
    prehead_norm: PackedNorm,
    head_ahead: PackedLinear,
    head_w_input: PackedLinear,
    head_w_norm: PackedNorm,
    head_w_output: PackedLinear,
    head_p: PackedLinear,
    ahead_linear: PackedLinear,
    w_linear: PackedLinear,
    p_linear: PackedLinear,
}

impl PackedScanModel {
    fn from_weights(weights: &SrsRwkvRnnWeights) -> Result<Self> {
        ensure!(
            weights.rwkv_modules.len() == 5,
            "GPU scan processor requires five RWKV modules"
        );
        let mut packer = WeightPacker::default();
        let features_input =
            packer.linear(&weights.features2card.input_linear, FEATURE_DIM, 512)?;
        let features_norm = packer.norm(&weights.features2card.norm, 512)?;
        let features_output = packer.linear(&weights.features2card.output_linear, 512, CHANNELS)?;

        let mut layer_metadata = Vec::with_capacity(weights.block_count());
        let mut module_layer_offsets = Vec::with_capacity(weights.rwkv_modules.len());
        for (module_index, module) in weights.rwkv_modules.iter().enumerate() {
            module_layer_offsets.push(layer_metadata.len());
            let entity_stride = (module.blocks.len() * STATE_LAYER_ELEMENTS) as u32;
            for (local_layer, layer) in module.blocks.iter().enumerate() {
                layer_metadata.push(packer.layer(layer)?.gpu_meta(
                    module_index as u32,
                    entity_stride,
                    local_layer as u32,
                ));
            }
        }
        ensure!(
            layer_metadata.len() == 16,
            "GPU scan shader requires 16 total RWKV layers, got {}",
            layer_metadata.len()
        );

        let prehead_norm = packer.norm(&weights.prehead_norm, CHANNELS)?;
        let head_ahead = packer.linear(&weights.head_ahead_logits, CHANNELS, 512)?;
        let head_w_input = packer.linear(&weights.head_w.input_linear, CHANNELS, CHANNELS)?;
        let head_w_norm = packer.norm(&weights.head_w.norm, CHANNELS)?;
        let head_w_output = packer.linear(&weights.head_w.output_linear, CHANNELS, 512)?;
        let head_p = packer.linear(&weights.head_p, CHANNELS, 512)?;
        let ahead_linear = packer.linear(&weights.ahead_linear, 512, CHANNELS)?;
        let w_linear = packer.linear(&weights.w_linear, 512, CHANNELS)?;
        let p_linear = packer.linear(&weights.p_linear, 512, 4)?;

        Ok(Self {
            values: packer.full_precision_values,
            normalization_values: packer.normalization_values,
            layer_metadata,
            module_layer_offsets,
            features_input,
            features_norm,
            features_output,
            prehead_norm,
            head_ahead,
            head_w_input,
            head_w_norm,
            head_w_output,
            head_p,
            ahead_linear,
            w_linear,
            p_linear,
        })
    }

    fn module_shader_source(&self, transform_workgroup_size: usize) -> Result<String> {
        let mut source = include_str!("gpu_process_scan.wgsl").to_owned();
        let transform_lanes_per_row = transform_workgroup_size / CHANNELS;
        let transform_columns_per_lane = HEAD_SIZE / transform_lanes_per_row;
        for (placeholder, value) in [
            ("/*__TRANSFORM_WORKGROUP_SIZE__*/", transform_workgroup_size),
            ("/*__TRANSFORM_LANES_PER_ROW__*/", transform_lanes_per_row),
            (
                "/*__TRANSFORM_COLUMNS_PER_LANE__*/",
                transform_columns_per_lane,
            ),
        ] {
            source = source.replace(placeholder, &value.to_string());
        }
        let mut lane_sum =
            String::from("fn transform_lane_sum(value: f32) -> f32 {\n    var sum = value;\n");
        let mut lane_delta = 1usize;
        while lane_delta < transform_lanes_per_row {
            lane_sum.push_str(&format!(
                "    sum += subgroupShuffleXor(sum, {lane_delta}u);\n"
            ));
            lane_delta *= 2;
        }
        lane_sum.push_str("    return sum;\n}\n");
        source = source.replace("/*__TRANSFORM_LANE_SUM__*/", &lane_sum);
        source = source.replace(
            "/*__LINEAR_FUNCTIONS__*/",
            &linear_functions(&[
                ("x", "x", "0u"),
                ("norm", "norm_values", "0u"),
                ("tmp0", "tmp0", "0u"),
                ("tmp1", "tmp1", "0u"),
                ("v", "tmp0", "V_OFFSET"),
                ("a", "tmp1", "A_OFFSET"),
                ("d", "tmp1", "D_OFFSET"),
                ("g", "extra", "G_OFFSET"),
                ("out", "extra", "OUT_OFFSET"),
            ]),
        );
        ensure_no_placeholders(source)
    }

    fn head_shader_source(&self) -> Result<String> {
        let mut source = include_str!("gpu_process_heads.wgsl").to_owned();
        source = source.replace(
            "/*__LINEAR_FUNCTIONS__*/",
            &linear_functions(&[
                ("x", "x", "0u"),
                ("norm", "norm_values", "0u"),
                ("tmp0", "tmp0", "0u"),
                ("tmp1", "tmp1", "0u"),
            ]),
        );
        for (placeholder, value) in [
            ("/*__FEATURES_INPUT_W__*/", self.features_input.weight),
            ("/*__FEATURES_INPUT_B__*/", self.features_input.bias),
            ("/*__FEATURES_NORM_W__*/", self.features_norm.weight),
            ("/*__FEATURES_NORM_B__*/", self.features_norm.bias),
            ("/*__FEATURES_OUTPUT_W__*/", self.features_output.weight),
            ("/*__FEATURES_OUTPUT_B__*/", self.features_output.bias),
            ("/*__PREHEAD_W__*/", self.prehead_norm.weight),
            ("/*__PREHEAD_B__*/", self.prehead_norm.bias),
            ("/*__HEAD_AHEAD_W__*/", self.head_ahead.weight),
            ("/*__HEAD_AHEAD_B__*/", self.head_ahead.bias),
            ("/*__HEAD_W_INPUT_W__*/", self.head_w_input.weight),
            ("/*__HEAD_W_INPUT_B__*/", self.head_w_input.bias),
            ("/*__HEAD_W_NORM_W__*/", self.head_w_norm.weight),
            ("/*__HEAD_W_NORM_B__*/", self.head_w_norm.bias),
            ("/*__HEAD_W_OUTPUT_W__*/", self.head_w_output.weight),
            ("/*__HEAD_W_OUTPUT_B__*/", self.head_w_output.bias),
            ("/*__HEAD_P_W__*/", self.head_p.weight),
            ("/*__HEAD_P_B__*/", self.head_p.bias),
            ("/*__AHEAD_LINEAR_W__*/", self.ahead_linear.weight),
            ("/*__AHEAD_LINEAR_B__*/", self.ahead_linear.bias),
            ("/*__W_LINEAR_W__*/", self.w_linear.weight),
            ("/*__W_LINEAR_B__*/", self.w_linear.bias),
            ("/*__P_LINEAR_W__*/", self.p_linear.weight),
            ("/*__P_LINEAR_B__*/", self.p_linear.bias),
        ] {
            source = source.replace(placeholder, &value.to_string());
        }
        ensure_no_placeholders(source)
    }
}

fn ensure_no_placeholders(source: String) -> Result<String> {
    if source.contains("/*__") {
        bail!("internal GPU scan shader has an unresolved placeholder");
    }
    Ok(source)
}

fn linear_functions(sources: &[(&str, &str, &str)]) -> String {
    let mut functions = String::new();
    for (name, values, base) in sources {
        functions.push_str(&format!(
            r#"
fn linear_{name}(
    output: u32,
    input_dim: u32,
    output_dim: u32,
    weight_offset: u32,
    bias_offset: u32,
    lane: u32,
) {{
    var output_index = lane;
    loop {{
        if (output_index >= output_dim) {{ break; }}
        var sum = 0.0;
        if (bias_offset != INVALID_INDEX) {{
            sum = model_weights[bias_offset + output_index];
        }}
        var input_index = 0u;
        loop {{
            if (input_index >= input_dim) {{ break; }}
            let packed_offset = weight_offset
                + input_index * output_dim
                + output_index * 4u;
            sum += {values}[{base} + input_index]
                * model_weights[packed_offset];
            sum += {values}[{base} + input_index + 1u]
                * model_weights[packed_offset + 1u];
            sum += {values}[{base} + input_index + 2u]
                * model_weights[packed_offset + 2u];
            sum += {values}[{base} + input_index + 3u]
                * model_weights[packed_offset + 3u];
            input_index += 4u;
        }}
        set_scratch_value(output, output_index, sum);
        output_index += 128u;
    }}
    workgroupBarrier();
}}
"#
        ));
    }
    functions
}

struct ProcessPipelines {
    encode_features: wgpu::ComputePipeline,
    output_heads: wgpu::ComputePipeline,
    gather: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    prepare_layer: wgpu::ComputePipeline,
    prepare_layer_fused_rows: Option<wgpu::ComputePipeline>,
    build_transforms: wgpu::ComputePipeline,
    build_transforms_state_only: wgpu::ComputePipeline,
    scan_transforms: wgpu::ComputePipeline,
    replay_chunks: wgpu::ComputePipeline,
    replay_chunks_state_only: wgpu::ComputePipeline,
    finish_time: wgpu::ComputePipeline,
    finish_channel: wgpu::ComputePipeline,
    finish_channel_fused_rows: Option<wgpu::ComputePipeline>,
    gather_current_state: wgpu::ComputePipeline,
    gather_evicted_state: wgpu::ComputePipeline,
    gather_arena_state: wgpu::ComputePipeline,
    scatter_arena_state: wgpu::ComputePipeline,
    module_bind_group_layout: wgpu::BindGroupLayout,
    dispatch_bind_group_layout: wgpu::BindGroupLayout,
    state_transfer_bind_group_layout: wgpu::BindGroupLayout,
    arena_transfer_bind_group_layout: wgpu::BindGroupLayout,
}

#[derive(Clone, Copy, Debug, Default)]
struct CpuStateSyncProfile {
    calls: u64,
    chunks: u64,
    entities: u64,
    bytes: u64,
    flat_chunks: u64,
    flat_entities: u64,
    flat_bytes: u64,
    pipelined_chunks: u64,
    pipeline_fallbacks: u64,
    pipeline_extra_buffer_bytes: u64,
    copy_map_submit_ns: u128,
    copy_map_wait_ns: u128,
    pipeline_overlap_ns: u128,
    flat_copy_ns: u128,
    field_copy_ns: u128,
    backing_tensor_ns: u128,
    view_ns: u128,
    map_insert_ns: u128,
    arena_destroy_ns: u128,
    deferred_ns: u128,
    total_ns: u128,
}

impl CpuStateSyncProfile {
    fn add(&mut self, other: Self) {
        self.calls += other.calls;
        self.chunks += other.chunks;
        self.entities += other.entities;
        self.bytes += other.bytes;
        self.flat_chunks += other.flat_chunks;
        self.flat_entities += other.flat_entities;
        self.flat_bytes += other.flat_bytes;
        self.pipelined_chunks += other.pipelined_chunks;
        self.pipeline_fallbacks += other.pipeline_fallbacks;
        self.pipeline_extra_buffer_bytes += other.pipeline_extra_buffer_bytes;
        self.copy_map_submit_ns += other.copy_map_submit_ns;
        self.copy_map_wait_ns += other.copy_map_wait_ns;
        self.pipeline_overlap_ns += other.pipeline_overlap_ns;
        self.flat_copy_ns += other.flat_copy_ns;
        self.field_copy_ns += other.field_copy_ns;
        self.backing_tensor_ns += other.backing_tensor_ns;
        self.view_ns += other.view_ns;
        self.map_insert_ns += other.map_insert_ns;
        self.arena_destroy_ns += other.arena_destroy_ns;
        self.deferred_ns += other.deferred_ns;
        self.total_ns += other.total_ns;
    }
}

struct FullyResidentSyncChunk {
    kind: ModuleKind,
    layers: usize,
    keys: Vec<Option<i64>>,
    source: wgpu::Buffer,
    source_offset: usize,
    state_bytes: usize,
}

struct PendingStateReadback {
    chunk: FullyResidentSyncChunk,
    readback: wgpu::Buffer,
    receiver: mpsc::Receiver<std::result::Result<(), wgpu::BufferAsyncError>>,
}

pub(super) struct GpuProcessScan {
    context: GpuContext,
    packed: PackedScanModel,
    pipelines: ProcessPipelines,
    weight_buffer: wgpu::Buffer,
    metadata_buffer: wgpu::Buffer,
    buffers: ProcessBufferCache,
    deferred_states: DeferredProcessStates,
    resident_states: [Option<ResidentModuleState>; 5],
    fully_resident_states: [Option<FullyResidentModuleState>; 5],
    transform_workgroup_size: usize,
    fused_row_batched_prepare: bool,
    fused_row_batched_channel: bool,
    fused_row_batched_prepare_batches: u64,
    fused_row_batched_channel_batches: u64,
    last_fused_row_batched_prepare: bool,
    last_fused_row_batched_channel: bool,
    process_calls: u64,
    process_batches: u64,
    state_only_calls: u64,
    state_only_batches: u64,
    fully_resident_calls: u64,
    fully_resident_batches: u64,
    adaptive_splits: u64,
    oom_retries: u64,
    resident_oom_recoveries: u64,
    resident_readback_recoveries: u64,
    last_requested_reviews: u64,
    last_process_batches: u64,
    last_mode: GpuProcessScanMode,
    last_fully_resident: bool,
    fully_resident_disabled_after_oom: bool,
    fully_resident_commit_ns: u128,
    last_fully_resident_commit_ns: u128,
    shared_cpu_state_view_chunks: u64,
    shared_cpu_state_view_entities: u64,
    last_shared_cpu_state_view_chunks: u64,
    last_shared_cpu_state_view_entities: u64,
    flat_cpu_state: bool,
    pipelined_cpu_state_sync: bool,
    cpu_state_sync_profile: CpuStateSyncProfile,
    last_cpu_state_sync_profile: CpuStateSyncProfile,
    feature_state_clone_ns: u128,
    feature_prepare_ns: u128,
    last_feature_state_clone_ns: u128,
    last_feature_prepare_ns: u128,
}

impl GpuProcessScan {
    pub(super) fn new(rnn: &NativeRnn) -> Result<Self> {
        let context = GpuContext::new(GpuOperation::Process)?;
        ensure!(
            context.limits.max_storage_buffers_per_shader_stage >= 19,
            "GPU exposes only {} storage buffers per shader stage; associative scan needs 19",
            context.limits.max_storage_buffers_per_shader_stage
        );
        ensure!(
            context.limits.max_compute_invocations_per_workgroup >= 256,
            "GPU compute workgroups are too small for associative scan"
        );
        let packed = PackedScanModel::from_weights(&rnn.weights)?;
        let weight_buffer = context.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rwkv-srs scan FP32 model weights"),
            contents: bytemuck::cast_slice(&packed.values),
            usage: wgpu::BufferUsages::STORAGE,
        })?;
        let mut metadata_values = bytemuck::cast_slice(&packed.layer_metadata).to_vec();
        metadata_values.extend_from_slice(bytemuck::cast_slice(&packed.normalization_values));
        let metadata_buffer = context.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rwkv-srs scan layer metadata"),
            contents: &metadata_values,
            usage: wgpu::BufferUsages::STORAGE,
        })?;
        let transform_workgroup_sizes = transform_workgroup_sizes(&context)?;
        let fused_row_batched_projections = fused_row_batched_projection_selection();
        let mut pipeline_errors = Vec::new();
        let mut selected = None;
        for transform_workgroup_size in transform_workgroup_sizes {
            match ProcessPipelines::new(
                &context,
                &packed,
                transform_workgroup_size,
                fused_row_batched_projections.any(),
            ) {
                Ok(pipelines) => {
                    selected = Some((transform_workgroup_size, pipelines));
                    break;
                }
                Err(error) => {
                    pipeline_errors.push(format!("{transform_workgroup_size} lanes: {error}"))
                }
            }
        }
        let (transform_workgroup_size, pipelines) = selected.ok_or_else(|| {
            anyhow!(
                "GPU could not compile an associative-scan pipeline: {}",
                pipeline_errors.join("; ")
            )
        })?;
        let fused_row_batched_available = pipelines.prepare_layer_fused_rows.is_some()
            && pipelines.finish_channel_fused_rows.is_some();
        Ok(Self {
            buffers: ProcessBufferCache::new(),
            deferred_states: DeferredProcessStates::default(),
            resident_states: std::array::from_fn(|_| None),
            fully_resident_states: std::array::from_fn(|_| None),
            context,
            packed,
            pipelines,
            weight_buffer,
            metadata_buffer,
            transform_workgroup_size,
            fused_row_batched_prepare: fused_row_batched_projections.prepare
                && fused_row_batched_available,
            fused_row_batched_channel: fused_row_batched_projections.channel
                && fused_row_batched_available,
            fused_row_batched_prepare_batches: 0,
            fused_row_batched_channel_batches: 0,
            last_fused_row_batched_prepare: false,
            last_fused_row_batched_channel: false,
            process_calls: 0,
            process_batches: 0,
            state_only_calls: 0,
            state_only_batches: 0,
            fully_resident_calls: 0,
            fully_resident_batches: 0,
            adaptive_splits: 0,
            oom_retries: 0,
            resident_oom_recoveries: 0,
            resident_readback_recoveries: 0,
            last_requested_reviews: 0,
            last_process_batches: 0,
            last_mode: GpuProcessScanMode::Predictions,
            last_fully_resident: false,
            fully_resident_disabled_after_oom: false,
            fully_resident_commit_ns: 0,
            last_fully_resident_commit_ns: 0,
            shared_cpu_state_view_chunks: 0,
            shared_cpu_state_view_entities: 0,
            last_shared_cpu_state_view_chunks: 0,
            last_shared_cpu_state_view_entities: 0,
            flat_cpu_state: flat_cpu_state_enabled(),
            pipelined_cpu_state_sync: pipelined_cpu_state_sync_enabled(),
            cpu_state_sync_profile: CpuStateSyncProfile::default(),
            last_cpu_state_sync_profile: CpuStateSyncProfile::default(),
            feature_state_clone_ns: 0,
            feature_prepare_ns: 0,
            last_feature_state_clone_ns: 0,
            last_feature_prepare_ns: 0,
        })
    }

    fn to_pydict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("operation", "process")?;
        dict.set_item("device_name", &self.context.info.name)?;
        dict.set_item("backend", format!("{:?}", self.context.info.backend))?;
        dict.set_item(
            "device_type",
            format!("{:?}", self.context.info.device_type),
        )?;
        dict.set_item("driver", &self.context.info.driver)?;
        dict.set_item("driver_info", &self.context.info.driver_info)?;
        dict.set_item("shader_f16", self.context.shader_f16)?;
        dict.set_item("timestamp_queries", self.context.timestamp_queries)?;
        dict.set_item("subgroup_operations", self.context.subgroup_operations)?;
        dict.set_item("subgroup_min_size", self.context.info.subgroup_min_size)?;
        dict.set_item("subgroup_max_size", self.context.info.subgroup_max_size)?;
        dict.set_item(
            "max_storage_buffer_binding_size",
            self.context.limits.max_storage_buffer_binding_size,
        )?;
        dict.set_item("max_buffer_size", self.context.limits.max_buffer_size)?;
        dict.set_item(
            "max_storage_buffers_per_shader_stage",
            self.context.limits.max_storage_buffers_per_shader_stage,
        )?;
        dict.set_item(
            "max_compute_invocations_per_workgroup",
            self.context.limits.max_compute_invocations_per_workgroup,
        )?;
        dict.set_item(
            "max_compute_workgroups_per_dimension",
            self.context.limits.max_compute_workgroups_per_dimension,
        )?;
        dict.set_item("state_storage", "float32")?;
        dict.set_item("weight_storage", "float32")?;
        dict.set_item("arithmetic", "float32")?;
        dict.set_item("transform_workgroup_size", self.transform_workgroup_size)?;
        dict.set_item("fused_row_batched_prepare", self.fused_row_batched_prepare)?;
        dict.set_item("fused_row_batched_channel", self.fused_row_batched_channel)?;
        dict.set_item("fused_row_batched_min_rows", FUSED_ROW_BATCHED_MIN_ROWS)?;
        dict.set_item(
            "fused_row_batched_prepare_batches",
            self.fused_row_batched_prepare_batches,
        )?;
        dict.set_item(
            "fused_row_batched_channel_batches",
            self.fused_row_batched_channel_batches,
        )?;
        dict.set_item(
            "last_fused_row_batched_prepare",
            self.last_fused_row_batched_prepare,
        )?;
        dict.set_item(
            "last_fused_row_batched_channel",
            self.last_fused_row_batched_channel,
        )?;
        dict.set_item(
            "scan_chunk_reviews",
            scan_chunk_reviews().map_err(py_value_error)?,
        )?;
        let buffer_capacity_bytes = self.buffers.allocated_bytes();
        let state_sync_primary_readback_capacity_bytes =
            self.buffers.readbacks[0].allocated_bytes();
        let state_sync_secondary_readback_capacity_bytes =
            self.buffers.readbacks[1].allocated_bytes();
        let fully_resident_buffer_bytes = self
            .fully_resident_states
            .iter()
            .flatten()
            .map(FullyResidentModuleState::allocated_bytes)
            .sum::<usize>();
        let model_weight_buffer_bytes = self.weight_buffer.size();
        let model_metadata_buffer_bytes = self.metadata_buffer.size();
        dict.set_item("buffer_capacity_bytes", buffer_capacity_bytes)?;
        dict.set_item(
            "state_sync_primary_readback_capacity_bytes",
            state_sync_primary_readback_capacity_bytes,
        )?;
        dict.set_item(
            "state_sync_secondary_readback_capacity_bytes",
            state_sync_secondary_readback_capacity_bytes,
        )?;
        dict.set_item("fully_resident_buffer_bytes", fully_resident_buffer_bytes)?;
        dict.set_item("model_weight_buffer_bytes", model_weight_buffer_bytes)?;
        dict.set_item("model_metadata_buffer_bytes", model_metadata_buffer_bytes)?;
        dict.set_item(
            "tracked_buffer_capacity_bytes",
            buffer_capacity_bytes as u64
                + fully_resident_buffer_bytes as u64
                + model_weight_buffer_bytes
                + model_metadata_buffer_bytes,
        )?;
        dict.set_item("deferred_state_entities", self.deferred_states.len())?;
        dict.set_item(
            "resident_state_entities",
            self.resident_states
                .iter()
                .flatten()
                .map(|resident| resident.keys.len())
                .sum::<usize>()
                + self
                    .fully_resident_states
                    .iter()
                    .flatten()
                    .map(FullyResidentModuleState::len)
                    .sum::<usize>(),
        )?;
        dict.set_item("process_calls", self.process_calls)?;
        dict.set_item("process_batches", self.process_batches)?;
        dict.set_item("state_only_calls", self.state_only_calls)?;
        dict.set_item("state_only_batches", self.state_only_batches)?;
        dict.set_item("fully_resident_calls", self.fully_resident_calls)?;
        dict.set_item("fully_resident_batches", self.fully_resident_batches)?;
        dict.set_item(
            "fully_resident_state_entities",
            self.fully_resident_states
                .iter()
                .flatten()
                .map(FullyResidentModuleState::len)
                .sum::<usize>(),
        )?;
        dict.set_item(
            "fully_resident_state_shards",
            self.fully_resident_states
                .iter()
                .flatten()
                .map(FullyResidentModuleState::shard_count)
                .sum::<usize>(),
        )?;
        dict.set_item("adaptive_splits", self.adaptive_splits)?;
        dict.set_item("oom_retries", self.oom_retries)?;
        dict.set_item("resident_oom_recoveries", self.resident_oom_recoveries)?;
        dict.set_item(
            "resident_readback_recoveries",
            self.resident_readback_recoveries,
        )?;
        dict.set_item("last_requested_reviews", self.last_requested_reviews)?;
        dict.set_item("last_process_batches", self.last_process_batches)?;
        dict.set_item("last_mode", self.last_mode.label())?;
        dict.set_item("last_fully_resident", self.last_fully_resident)?;
        dict.set_item(
            "fully_resident_disabled_after_oom",
            self.fully_resident_disabled_after_oom,
        )?;
        dict.set_item("fully_resident_commit_ns", self.fully_resident_commit_ns)?;
        dict.set_item(
            "last_fully_resident_commit_ns",
            self.last_fully_resident_commit_ns,
        )?;
        dict.set_item(
            "shared_cpu_state_view_chunks",
            self.shared_cpu_state_view_chunks,
        )?;
        dict.set_item(
            "shared_cpu_state_view_entities",
            self.shared_cpu_state_view_entities,
        )?;
        dict.set_item(
            "last_shared_cpu_state_view_chunks",
            self.last_shared_cpu_state_view_chunks,
        )?;
        dict.set_item(
            "last_shared_cpu_state_view_entities",
            self.last_shared_cpu_state_view_entities,
        )?;
        dict.set_item("flat_cpu_state", self.flat_cpu_state)?;
        dict.set_item("pipelined_cpu_state_sync", self.pipelined_cpu_state_sync)?;
        dict.set_item(
            "fully_resident_sync_chunk_limit_bytes",
            FULLY_RESIDENT_SYNC_CHUNK_BYTES,
        )?;
        add_cpu_state_sync_profile(&dict, "cpu_state_sync_", self.cpu_state_sync_profile)?;
        add_cpu_state_sync_profile(
            &dict,
            "last_cpu_state_sync_",
            self.last_cpu_state_sync_profile,
        )?;
        dict.set_item("feature_state_clone_ns", self.feature_state_clone_ns)?;
        dict.set_item("feature_prepare_ns", self.feature_prepare_ns)?;
        dict.set_item(
            "last_feature_state_clone_ns",
            self.last_feature_state_clone_ns,
        )?;
        dict.set_item("last_feature_prepare_ns", self.last_feature_prepare_ns)?;
        Ok(dict.unbind())
    }
}

impl ProcessPipelines {
    fn new(
        context: &GpuContext,
        packed: &PackedScanModel,
        transform_workgroup_size: usize,
        fused_row_batched_projections: bool,
    ) -> Result<Self> {
        let module_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("rwkv-srs scan module resources"),
                    entries: &(0..19)
                        .map(|binding| wgpu::BindGroupLayoutEntry {
                            binding,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage {
                                    read_only: matches!(binding, 0 | 1 | 12 | 13 | 14 | 15),
                                },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        })
                        .collect::<Vec<_>>(),
                });
        let dispatch_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("rwkv-srs scan dispatch parameters"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });
        let module_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("rwkv-srs scan module pipeline layout"),
                    bind_group_layouts: &[
                        Some(&module_bind_group_layout),
                        Some(&dispatch_bind_group_layout),
                    ],
                    immediate_size: 0,
                });

        let module_source = packed.module_shader_source(transform_workgroup_size)?;
        let module_shader = compile_shader(context, "rwkv-srs associative scan", module_source)?;
        let prepare_layer = compute_pipeline(
            context,
            &module_shader,
            "prepare_layer",
            Some(&module_pipeline_layout),
        )?;
        let build_transforms = compute_pipeline(
            context,
            &module_shader,
            "build_chunk_transforms",
            Some(&module_pipeline_layout),
        )?;
        let build_transforms_state_only = compute_pipeline(
            context,
            &module_shader,
            "build_chunk_transforms_state_only",
            Some(&module_pipeline_layout),
        )?;
        let scan_transforms = compute_pipeline(
            context,
            &module_shader,
            "scan_chunk_transforms",
            Some(&module_pipeline_layout),
        )?;
        let replay_chunks = compute_pipeline(
            context,
            &module_shader,
            "replay_chunks",
            Some(&module_pipeline_layout),
        )?;
        let replay_chunks_state_only = compute_pipeline(
            context,
            &module_shader,
            "replay_chunks_state_only",
            Some(&module_pipeline_layout),
        )?;
        let finish_time = compute_pipeline(
            context,
            &module_shader,
            "finish_time_mixer",
            Some(&module_pipeline_layout),
        )?;
        let finish_channel = compute_pipeline(
            context,
            &module_shader,
            "finish_channel_mixer",
            Some(&module_pipeline_layout),
        )?;
        let (prepare_layer_fused_rows, finish_channel_fused_rows) = if fused_row_batched_projections
        {
            let candidate = (|| -> Result<_> {
                let fused_shader = compile_shader(
                    context,
                    "rwkv-srs fused row-batched projections",
                    include_str!("gpu_process_fused_projections.wgsl").to_owned(),
                )?;
                let prepare = compute_pipeline(
                    context,
                    &fused_shader,
                    "prepare_layer_fused_rows",
                    Some(&module_pipeline_layout),
                )?;
                let channel = compute_pipeline(
                    context,
                    &fused_shader,
                    "finish_channel_mixer_fused_rows",
                    Some(&module_pipeline_layout),
                )?;
                Ok((Some(prepare), Some(channel)))
            })();
            match candidate {
                Ok(pipelines) => pipelines,
                Err(error) => {
                    return Err(error.context(
                        "GPU process mode requires the fused row-batched projection shaders; \
                         use CPU Fast mode when this adapter cannot compile them",
                    ));
                }
            }
        } else {
            (None, None)
        };
        context
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("associative scan module pipeline compilation failed")?;

        let head_shader = compile_shader(
            context,
            "rwkv-srs scan feature and output heads",
            packed.head_shader_source()?,
        )?;
        let encode_features = compute_pipeline(context, &head_shader, "encode_features", None)?;
        let output_heads = compute_pipeline(context, &head_shader, "output_heads", None)?;

        let topology_shader = compile_shader(
            context,
            "rwkv-srs scan topology",
            include_str!("gpu_process_topology.wgsl").to_owned(),
        )?;
        let topology_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("rwkv-srs scan topology resources"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        let topology_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("rwkv-srs scan topology pipeline layout"),
                    bind_group_layouts: &[Some(&topology_bind_group_layout)],
                    immediate_size: 0,
                });
        let gather = compute_pipeline(
            context,
            &topology_shader,
            "gather",
            Some(&topology_pipeline_layout),
        )?;
        let scatter = compute_pipeline(
            context,
            &topology_shader,
            "scatter",
            Some(&topology_pipeline_layout),
        )?;

        let state_transfer_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("rwkv-srs resident state transfer resources"),
                    entries: &(0..7)
                        .map(|binding| wgpu::BindGroupLayoutEntry {
                            binding,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: if binding == 6 {
                                    wgpu::BufferBindingType::Uniform
                                } else {
                                    wgpu::BufferBindingType::Storage {
                                        read_only: !matches!(binding, 3 | 5),
                                    }
                                },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        })
                        .collect::<Vec<_>>(),
                });
        let state_transfer_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("rwkv-srs resident state transfer pipeline layout"),
                    bind_group_layouts: &[Some(&state_transfer_bind_group_layout)],
                    immediate_size: 0,
                });
        let state_transfer_shader = compile_shader(
            context,
            "rwkv-srs resident state transfer",
            include_str!("gpu_process_state.wgsl").to_owned(),
        )?;
        let gather_current_state = compute_pipeline(
            context,
            &state_transfer_shader,
            "gather_current_state",
            Some(&state_transfer_pipeline_layout),
        )?;
        let gather_evicted_state = compute_pipeline(
            context,
            &state_transfer_shader,
            "gather_evicted_state",
            Some(&state_transfer_pipeline_layout),
        )?;

        let arena_transfer_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("rwkv-srs fully resident arena transfer resources"),
                    entries: &(0..4)
                        .map(|binding| wgpu::BindGroupLayoutEntry {
                            binding,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: if binding == 3 {
                                    wgpu::BufferBindingType::Uniform
                                } else {
                                    wgpu::BufferBindingType::Storage {
                                        read_only: binding != 1,
                                    }
                                },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        })
                        .collect::<Vec<_>>(),
                });
        let arena_transfer_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("rwkv-srs fully resident arena transfer pipeline layout"),
                    bind_group_layouts: &[Some(&arena_transfer_bind_group_layout)],
                    immediate_size: 0,
                });
        let arena_transfer_shader = compile_shader(
            context,
            "rwkv-srs fully resident arena transfer",
            include_str!("gpu_process_arena.wgsl").to_owned(),
        )?;
        let gather_arena_state = compute_pipeline(
            context,
            &arena_transfer_shader,
            "gather_arena_state",
            Some(&arena_transfer_pipeline_layout),
        )?;
        let scatter_arena_state = compute_pipeline(
            context,
            &arena_transfer_shader,
            "scatter_arena_state",
            Some(&arena_transfer_pipeline_layout),
        )?;
        context
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("associative scan pipeline compilation failed")?;

        Ok(Self {
            encode_features,
            output_heads,
            gather,
            scatter,
            prepare_layer,
            prepare_layer_fused_rows,
            build_transforms,
            build_transforms_state_only,
            scan_transforms,
            replay_chunks,
            replay_chunks_state_only,
            finish_time,
            finish_channel,
            finish_channel_fused_rows,
            gather_current_state,
            gather_evicted_state,
            gather_arena_state,
            scatter_arena_state,
            module_bind_group_layout,
            dispatch_bind_group_layout,
            state_transfer_bind_group_layout,
            arena_transfer_bind_group_layout,
        })
    }
}

fn compile_shader(
    context: &GpuContext,
    label: &'static str,
    source: String,
) -> Result<wgpu::ShaderModule> {
    let error_scope = context
        .device
        .push_error_scope(wgpu::ErrorFilter::Validation);
    let shader = context
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(source)),
        });
    if let Some(error) = pollster::block_on(error_scope.pop()) {
        bail!("could not compile {label}: {error}");
    }
    Ok(shader)
}

fn compute_pipeline(
    context: &GpuContext,
    shader: &wgpu::ShaderModule,
    entry_point: &'static str,
    layout: Option<&wgpu::PipelineLayout>,
) -> Result<wgpu::ComputePipeline> {
    let error_scope = context
        .device
        .push_error_scope(wgpu::ErrorFilter::Validation);
    let pipeline = context
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry_point),
            layout,
            module: shader,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        });
    if let Some(error) = pollster::block_on(error_scope.pop()) {
        bail!("could not compile GPU scan pipeline {entry_point}: {error}");
    }
    Ok(pipeline)
}

#[derive(Clone, Copy)]
enum ModuleKind {
    Card,
    Deck,
    Note,
    Preset,
    Global,
}

impl ModuleKind {
    const ALL: [Self; 5] = [
        Self::Card,
        Self::Deck,
        Self::Note,
        Self::Preset,
        Self::Global,
    ];

    fn key(self, ids: ReviewIds) -> Option<i64> {
        match self {
            Self::Card => Some(ids.0),
            Self::Deck => Some(ids.2),
            Self::Note => Some(ids.1),
            Self::Preset => Some(ids.3),
            Self::Global => None,
        }
    }

    fn state(self, rnn: &NativeRnn, key: Option<i64>) -> Option<&NativeRnnModuleState> {
        match self {
            Self::Card => rnn.card_states.get(&key.expect("card key")),
            Self::Deck => rnn.deck_states.get(&key.expect("deck key")),
            Self::Note => rnn.note_states.get(&key.expect("note key")),
            Self::Preset => rnn.preset_states.get(&key.expect("preset key")),
            Self::Global => rnn.global_state.as_ref(),
        }
    }

    fn flat_state(self, rnn: &NativeRnn, key: Option<i64>) -> Option<&FlatNativeRnnModuleState> {
        match self {
            Self::Card => rnn.flat_cpu_state.card_states.get(&key.expect("card key")),
            Self::Deck => rnn.flat_cpu_state.deck_states.get(&key.expect("deck key")),
            Self::Note => rnn.flat_cpu_state.note_states.get(&key.expect("note key")),
            Self::Preset => rnn
                .flat_cpu_state
                .preset_states
                .get(&key.expect("preset key")),
            Self::Global => rnn.flat_cpu_state.global_state.as_ref(),
        }
    }

    fn layers(self) -> usize {
        match self {
            Self::Card => 3,
            Self::Deck => 4,
            Self::Note => 2,
            Self::Preset => 3,
            Self::Global => 4,
        }
    }
}

struct ResidentModuleState {
    keys: Vec<Option<i64>>,
    entity_by_key: HashMap<Option<i64>, u32>,
}

impl ResidentModuleState {
    fn new(keys: Vec<Option<i64>>) -> Self {
        let entity_by_key = keys
            .iter()
            .copied()
            .enumerate()
            .map(|(entity, key)| (key, entity as u32))
            .collect();
        Self {
            keys,
            entity_by_key,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ArenaLocation {
    shard: usize,
    entity: u32,
}

struct FullyResidentShard {
    keys: Vec<Option<i64>>,
    buffer: Option<wgpu::Buffer>,
    capacity_entities: usize,
}

impl FullyResidentShard {
    fn new() -> Self {
        Self {
            keys: Vec::new(),
            buffer: None,
            capacity_entities: 0,
        }
    }

    fn ensure_capacity(
        &mut self,
        context: &GpuContext,
        state_stride: usize,
        required_entities: usize,
        max_entities: usize,
    ) -> Result<()> {
        if self.capacity_entities >= required_entities {
            return Ok(());
        }
        ensure!(
            required_entities <= max_entities,
            "fully resident GPU state shard requires {required_entities} entities but supports at most {max_entities}"
        );
        let with_headroom = required_entities
            .checked_add(required_entities / 8)
            .unwrap_or(max_entities)
            .min(max_entities);
        let capacity_entities = with_headroom.max(required_entities);
        let capacity_bytes = capacity_entities
            .checked_mul(state_stride)
            .and_then(|values| values.checked_mul(size_of::<f32>()))
            .context("fully resident GPU state shard size overflow")?;
        let replacement = context.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rwkv-srs fully resident FP32 state shard"),
            size: capacity_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })?;
        if let Some(current) = self.buffer.as_ref() {
            let used_bytes = self
                .keys
                .len()
                .checked_mul(state_stride)
                .and_then(|values| values.checked_mul(size_of::<f32>()))
                .context("fully resident GPU state copy size overflow")?;
            if used_bytes > 0 {
                let mut encoder =
                    context
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("rwkv-srs grow fully resident state shard"),
                        });
                encoder.copy_buffer_to_buffer(current, 0, &replacement, 0, used_bytes as u64);
                context.queue.submit([encoder.finish()]);
            }
        }
        self.buffer = Some(replacement);
        self.capacity_entities = capacity_entities;
        Ok(())
    }

    fn buffer(&self) -> Result<wgpu::Buffer> {
        self.buffer
            .as_ref()
            .cloned()
            .context("fully resident GPU state shard is unallocated")
    }

    fn allocated_bytes(&self, state_stride: usize) -> usize {
        self.capacity_entities * state_stride * size_of::<f32>()
    }
}

struct ArenaShardPlan {
    shard: usize,
    gather_count: usize,
    transfers: Vec<ArenaTransfer>,
    required_entities: usize,
}

struct ArenaModulePlan {
    host_transfers: Vec<ArenaTransfer>,
    shards: Vec<ArenaShardPlan>,
    new_keys: Vec<(Option<i64>, ArenaLocation)>,
}

struct FullyResidentModuleState {
    state_stride: usize,
    max_entities_per_shard: usize,
    shards: Vec<FullyResidentShard>,
    location_by_key: HashMap<Option<i64>, ArenaLocation>,
}

impl FullyResidentModuleState {
    fn new(context: &GpuContext, layers: usize) -> Result<Self> {
        let state_stride = layers * STATE_LAYER_ELEMENTS;
        let advertised_bytes = context
            .limits
            .max_buffer_size
            .min(context.limits.max_storage_buffer_binding_size)
            as usize;
        let shard_bytes = match std::env::var(FULLY_RESIDENT_SHARD_LIMIT_ENV_VAR) {
            Ok(value) => {
                let parsed = value.parse::<usize>().with_context(|| {
                    format!("invalid {FULLY_RESIDENT_SHARD_LIMIT_ENV_VAR}={value:?}")
                })?;
                ensure!(
                    parsed > 0,
                    "{FULLY_RESIDENT_SHARD_LIMIT_ENV_VAR} must be positive"
                );
                parsed.min(advertised_bytes)
            }
            Err(std::env::VarError::NotPresent) => advertised_bytes,
            Err(error) => {
                return Err(error).context(format!(
                    "could not read {FULLY_RESIDENT_SHARD_LIMIT_ENV_VAR}"
                ));
            }
        };
        let entity_bytes = state_stride
            .checked_mul(size_of::<f32>())
            .context("fully resident GPU state entity size overflow")?;
        let max_entities_per_shard = shard_bytes / entity_bytes;
        ensure!(
            max_entities_per_shard > 0,
            "fully resident GPU state shard limit {shard_bytes} is smaller than one {entity_bytes}-byte state"
        );
        Ok(Self {
            state_stride,
            max_entities_per_shard,
            shards: Vec::new(),
            location_by_key: HashMap::new(),
        })
    }

    fn plan(&self, keys: &[Option<i64>], host_entities: &[Option<u32>]) -> Result<ArenaModulePlan> {
        ensure!(
            keys.len() == host_entities.len(),
            "fully resident GPU state source plan mismatch"
        );
        let mut next_counts = self
            .shards
            .iter()
            .map(|shard| shard.keys.len())
            .collect::<Vec<_>>();
        let mut existing = vec![Vec::<ArenaTransfer>::new(); next_counts.len()];
        let mut additions = vec![Vec::<ArenaTransfer>::new(); next_counts.len()];
        let mut host_transfers = Vec::new();
        let mut new_keys = Vec::new();

        for (plan_entity, (key, host_entity)) in keys.iter().copied().zip(host_entities).enumerate()
        {
            if let Some(location) = self.location_by_key.get(&key).copied() {
                existing[location.shard].push(ArenaTransfer {
                    arena_entity: location.entity,
                    plan_entity: plan_entity as u32,
                    _pad0: 0,
                    _pad1: 0,
                });
                continue;
            }

            let shard = match next_counts
                .iter()
                .rposition(|count| *count < self.max_entities_per_shard)
            {
                Some(shard) => shard,
                None => {
                    next_counts.push(0);
                    existing.push(Vec::new());
                    additions.push(Vec::new());
                    next_counts.len() - 1
                }
            };
            let arena_entity = next_counts[shard];
            next_counts[shard] += 1;
            let location = ArenaLocation {
                shard,
                entity: u32::try_from(arena_entity)
                    .context("fully resident GPU state shard index overflow")?,
            };
            let transfer = ArenaTransfer {
                arena_entity: location.entity,
                plan_entity: plan_entity as u32,
                _pad0: 0,
                _pad1: 0,
            };
            additions[shard].push(transfer);
            host_transfers.push(ArenaTransfer {
                arena_entity: host_entity
                    .context("new fully resident GPU state is missing its host source")?,
                ..transfer
            });
            new_keys.push((key, location));
        }

        let shards = next_counts
            .into_iter()
            .enumerate()
            .filter_map(|(shard, required_entities)| {
                let gather_count = existing[shard].len();
                let mut transfers = std::mem::take(&mut existing[shard]);
                transfers.append(&mut additions[shard]);
                (!transfers.is_empty()).then_some(ArenaShardPlan {
                    shard,
                    gather_count,
                    transfers,
                    required_entities,
                })
            })
            .collect();
        Ok(ArenaModulePlan {
            host_transfers,
            shards,
            new_keys,
        })
    }

    fn ensure_plan(&mut self, context: &GpuContext, plan: &ArenaModulePlan) -> Result<()> {
        let required_shards = plan
            .shards
            .iter()
            .map(|plan| plan.shard + 1)
            .max()
            .unwrap_or(self.shards.len());
        while self.shards.len() < required_shards {
            self.shards.push(FullyResidentShard::new());
        }
        for shard_plan in &plan.shards {
            self.shards[shard_plan.shard].ensure_capacity(
                context,
                self.state_stride,
                shard_plan.required_entities,
                self.max_entities_per_shard,
            )?;
        }
        Ok(())
    }

    fn commit(&mut self, plan: &ArenaModulePlan) -> Result<()> {
        for (key, location) in plan.new_keys.iter().copied() {
            let shard = self
                .shards
                .get_mut(location.shard)
                .context("fully resident GPU state commit shard is missing")?;
            ensure!(
                shard.keys.len() == location.entity as usize,
                "fully resident GPU state commit order diverged"
            );
            ensure!(
                self.location_by_key.insert(key, location).is_none(),
                "fully resident GPU state key was committed twice"
            );
            shard.keys.push(key);
        }
        Ok(())
    }

    fn buffers(&self) -> Result<Vec<wgpu::Buffer>> {
        self.shards.iter().map(FullyResidentShard::buffer).collect()
    }

    fn len(&self) -> usize {
        self.location_by_key.len()
    }

    fn shard_count(&self) -> usize {
        self.shards
            .iter()
            .filter(|shard| !shard.keys.is_empty())
            .count()
    }

    fn allocated_bytes(&self) -> usize {
        self.shards
            .iter()
            .map(|shard| shard.allocated_bytes(self.state_stride))
            .sum()
    }
}

#[derive(Default)]
struct DeferredProcessStates {
    card: HashMap<i64, Box<[f32]>>,
    deck: HashMap<i64, Box<[f32]>>,
    note: HashMap<i64, Box<[f32]>>,
    preset: HashMap<i64, Box<[f32]>>,
    global: Option<Box<[f32]>>,
}

impl DeferredProcessStates {
    fn is_empty(&self) -> bool {
        self.card.is_empty()
            && self.deck.is_empty()
            && self.note.is_empty()
            && self.preset.is_empty()
            && self.global.is_none()
    }

    fn len(&self) -> usize {
        self.card.len()
            + self.deck.len()
            + self.note.len()
            + self.preset.len()
            + usize::from(self.global.is_some())
    }

    fn state(&self, kind: ModuleKind, key: Option<i64>) -> Option<&[f32]> {
        match kind {
            ModuleKind::Card => self.card.get(&key.expect("card key")).map(AsRef::as_ref),
            ModuleKind::Deck => self.deck.get(&key.expect("deck key")).map(AsRef::as_ref),
            ModuleKind::Note => self.note.get(&key.expect("note key")).map(AsRef::as_ref),
            ModuleKind::Preset => self
                .preset
                .get(&key.expect("preset key"))
                .map(AsRef::as_ref),
            ModuleKind::Global => self.global.as_deref(),
        }
    }

    fn insert(&mut self, rnn: &mut NativeRnn, kind: ModuleKind, key: Option<i64>, values: &[f32]) {
        let values = values.to_vec().into_boxed_slice();
        match kind {
            ModuleKind::Card => {
                let key = key.expect("card key");
                rnn.card_states.remove(&key);
                rnn.flat_cpu_state.card_states.remove(&key);
                self.card.insert(key, values);
            }
            ModuleKind::Deck => {
                let key = key.expect("deck key");
                rnn.deck_states.remove(&key);
                rnn.flat_cpu_state.deck_states.remove(&key);
                self.deck.insert(key, values);
            }
            ModuleKind::Note => {
                let key = key.expect("note key");
                rnn.note_states.remove(&key);
                rnn.flat_cpu_state.note_states.remove(&key);
                self.note.insert(key, values);
            }
            ModuleKind::Preset => {
                let key = key.expect("preset key");
                rnn.preset_states.remove(&key);
                rnn.flat_cpu_state.preset_states.remove(&key);
                self.preset.insert(key, values);
            }
            ModuleKind::Global => {
                rnn.global_state = None;
                rnn.flat_cpu_state.global_state = None;
                self.global = Some(values);
            }
        }
    }

    fn remove(&mut self, rnn: &mut NativeRnn, kind: ModuleKind, key: Option<i64>) {
        match kind {
            ModuleKind::Card => {
                let key = key.expect("card key");
                self.card.remove(&key);
                rnn.card_states.remove(&key);
                rnn.flat_cpu_state.card_states.remove(&key);
            }
            ModuleKind::Deck => {
                let key = key.expect("deck key");
                self.deck.remove(&key);
                rnn.deck_states.remove(&key);
                rnn.flat_cpu_state.deck_states.remove(&key);
            }
            ModuleKind::Note => {
                let key = key.expect("note key");
                self.note.remove(&key);
                rnn.note_states.remove(&key);
                rnn.flat_cpu_state.note_states.remove(&key);
            }
            ModuleKind::Preset => {
                let key = key.expect("preset key");
                self.preset.remove(&key);
                rnn.preset_states.remove(&key);
                rnn.flat_cpu_state.preset_states.remove(&key);
            }
            ModuleKind::Global => {
                self.global = None;
                rnn.global_state = None;
                rnn.flat_cpu_state.global_state = None;
            }
        }
    }

    fn synchronize(&mut self, rnn: &mut NativeRnn) -> Result<()> {
        for (key, values) in self.card.drain() {
            rnn.card_states.insert(
                key,
                module_state_from_values(&values, ModuleKind::Card.layers())?,
            );
        }
        for (key, values) in self.deck.drain() {
            rnn.deck_states.insert(
                key,
                module_state_from_values(&values, ModuleKind::Deck.layers())?,
            );
        }
        for (key, values) in self.note.drain() {
            rnn.note_states.insert(
                key,
                module_state_from_values(&values, ModuleKind::Note.layers())?,
            );
        }
        for (key, values) in self.preset.drain() {
            rnn.preset_states.insert(
                key,
                module_state_from_values(&values, ModuleKind::Preset.layers())?,
            );
        }
        if let Some(values) = self.global.take() {
            rnn.global_state = Some(module_state_from_values(
                &values,
                ModuleKind::Global.layers(),
            )?);
        }
        Ok(())
    }
}

struct EntityStream {
    key: Option<i64>,
    rows: Vec<usize>,
}

struct ModulePlan {
    kind: ModuleKind,
    layers: usize,
    keys: Vec<Option<i64>>,
    sequence_to_original: Vec<u32>,
    row_metadata: Vec<RowMeta>,
    transform_metadata: Vec<TransformMeta>,
    final_chunk_metadata: Vec<FinalChunkMeta>,
    max_transform_run: usize,
    state_values: Vec<f32>,
    state_sources: Vec<StateSource>,
    evicted_keys: Vec<Option<i64>>,
    evicted_entities: Vec<u32>,
    arena_plan: Option<ArenaModulePlan>,
}

impl ModulePlan {
    fn build(
        rnn: &NativeRnn,
        deferred_states: &DeferredProcessStates,
        ids: &[ReviewIds],
        kind: ModuleKind,
        layers: usize,
        chunk_reviews: usize,
        resident: Option<&ResidentModuleState>,
        fully_resident: Option<&FullyResidentModuleState>,
        mode: GpuProcessScanMode,
    ) -> Result<Self> {
        ensure!(
            resident.is_none() || fully_resident.is_none(),
            "GPU process state cannot use working-set and fully resident storage together"
        );
        let mut stream_indices = HashMap::<Option<i64>, usize>::new();
        let mut streams = Vec::<EntityStream>::new();
        for (row, ids) in ids.iter().copied().enumerate() {
            let key = kind.key(ids);
            let stream = *stream_indices.entry(key).or_insert_with(|| {
                let index = streams.len();
                streams.push(EntityStream {
                    key,
                    rows: Vec::new(),
                });
                index
            });
            streams[stream].rows.push(row);
        }

        let mut keys = Vec::with_capacity(streams.len());
        let rows_per_review = mode.rows_per_review();
        let process_row_offset = mode.process_row_offset();
        let mut sequence_to_original = Vec::with_capacity(ids.len() * rows_per_review);
        let mut row_metadata = Vec::with_capacity(ids.len() * rows_per_review);
        let mut transform_metadata = Vec::new();
        let mut final_chunk_metadata = Vec::new();
        let mut max_transform_run = 0usize;
        let mut state_values = Vec::with_capacity(streams.len() * layers * STATE_LAYER_ELEMENTS);
        let mut state_sources = Vec::with_capacity(streams.len());
        let mut host_entities = Vec::with_capacity(streams.len());

        for (entity, stream) in streams.iter().enumerate() {
            let arena_location = fully_resident
                .and_then(|arena| arena.location_by_key.get(&stream.key))
                .copied();
            let resident_entity = resident
                .and_then(|resident| resident.entity_by_key.get(&stream.key))
                .copied();
            let deferred_state = if resident_entity.is_none() && arena_location.is_none() {
                deferred_states.state(kind, stream.key)
            } else {
                None
            };
            let flat_state = if resident_entity.is_none()
                && arena_location.is_none()
                && deferred_state.is_none()
            {
                kind.flat_state(rnn, stream.key)
            } else {
                None
            };
            let state = if resident_entity.is_none()
                && arena_location.is_none()
                && deferred_state.is_none()
                && flat_state.is_none()
            {
                kind.state(rnn, stream.key)
            } else {
                None
            };
            keys.push(stream.key);
            if let Some(location) = arena_location {
                state_sources.push(StateSource {
                    entity: location.entity,
                    flags: STATE_SOURCE_RESIDENT,
                    _pad0: 0,
                    _pad1: 0,
                });
                host_entities.push(None);
            } else if let Some(resident_entity) = resident_entity {
                state_sources.push(StateSource {
                    entity: resident_entity,
                    flags: STATE_SOURCE_RESIDENT,
                    _pad0: 0,
                    _pad1: 0,
                });
                host_entities.push(None);
            } else {
                let host_entity = state_values.len() / (layers * STATE_LAYER_ELEMENTS);
                if let Some(values) = deferred_state {
                    ensure!(
                        values.len() == layers * STATE_LAYER_ELEMENTS,
                        "deferred GPU scan recurrent state has the wrong size"
                    );
                    state_values.extend_from_slice(values);
                } else if let Some(flat_state) = flat_state {
                    ensure!(
                        flat_state.layers() == layers,
                        "flat GPU scan recurrent state has the wrong layer count"
                    );
                    state_values.extend_from_slice(flat_state.values());
                } else {
                    append_module_state(&mut state_values, state, layers)?;
                }
                state_sources.push(StateSource {
                    entity: host_entity as u32,
                    flags: 0,
                    _pad0: 0,
                    _pad1: 0,
                });
                host_entities.push(Some(host_entity as u32));
            }
            let has_state = arena_location.is_some()
                || resident_entity.is_some()
                || deferred_state.is_some()
                || flat_state.is_some()
                || state.is_some();
            let sequence_start = sequence_to_original.len();
            for (position, original_review) in stream.rows.iter().copied().enumerate() {
                let previous_process_row = if position == 0 {
                    INVALID_INDEX
                } else {
                    u32::try_from(sequence_to_original.len() - 1)
                        .context("GPU sequence is too large")?
                };
                let common_flags = if has_state { ROW_HAS_STATE } else { 0 };
                if mode != GpuProcessScanMode::StateOnly {
                    sequence_to_original.push(
                        u32::try_from(original_review * rows_per_review)
                            .context("GPU row index overflow")?,
                    );
                    row_metadata.push(RowMeta {
                        previous_process_row,
                        entity: entity as u32,
                        flags: common_flags,
                        _pad: 0,
                    });
                }
                sequence_to_original.push(
                    u32::try_from(original_review * rows_per_review + process_row_offset)
                        .context("GPU row index overflow")?,
                );
                row_metadata.push(RowMeta {
                    previous_process_row,
                    entity: entity as u32,
                    flags: common_flags
                        | if position + 1 == stream.rows.len() {
                            ROW_LAST_PROCESS
                        } else {
                            0
                        },
                    _pad: 0,
                });
            }

            let final_chunk_count = stream.rows.len().div_ceil(chunk_reviews);
            let transform_start = transform_metadata.len();
            let transform_count = final_chunk_count.saturating_sub(1);
            max_transform_run = max_transform_run.max(transform_count);
            for chunk in 0..final_chunk_count {
                let review_start = chunk * chunk_reviews;
                let review_count = (stream.rows.len() - review_start).min(chunk_reviews);
                let start_row = sequence_start + review_start * rows_per_review;
                if chunk < transform_count {
                    transform_metadata.push(TransformMeta {
                        start_row: start_row as u32,
                        review_count: review_count as u32,
                        entity: entity as u32,
                        local_index: chunk as u32,
                    });
                }
                final_chunk_metadata.push(FinalChunkMeta {
                    start_row: start_row as u32,
                    review_count: review_count as u32,
                    entity: entity as u32,
                    prefix_transform: if chunk == 0 {
                        INVALID_INDEX
                    } else {
                        (transform_start + chunk - 1) as u32
                    },
                    flags: if chunk + 1 == final_chunk_count {
                        CHUNK_LAST
                    } else {
                        0
                    },
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                });
            }
        }

        ensure!(
            sequence_to_original.len() == ids.len() * rows_per_review,
            "GPU row plan mismatch"
        );
        ensure!(
            row_metadata.len() == ids.len() * rows_per_review,
            "GPU metadata plan mismatch"
        );
        let mut evicted_keys = Vec::new();
        let mut evicted_entities = Vec::new();
        if let Some(resident) = resident {
            for (entity, key) in resident.keys.iter().copied().enumerate() {
                if !stream_indices.contains_key(&key) {
                    evicted_keys.push(key);
                    evicted_entities.push(entity as u32);
                }
            }
        }
        let arena_plan = fully_resident
            .map(|arena| arena.plan(&keys, &host_entities))
            .transpose()?;
        Ok(Self {
            kind,
            layers,
            keys,
            sequence_to_original,
            row_metadata,
            transform_metadata,
            final_chunk_metadata,
            max_transform_run,
            state_values,
            state_sources,
            evicted_keys,
            evicted_entities,
            arena_plan,
        })
    }

    fn state_elements(&self) -> usize {
        self.keys.len() * self.layers * STATE_LAYER_ELEMENTS
    }

    fn evicted_state_elements(&self) -> usize {
        self.evicted_keys.len() * self.layers * STATE_LAYER_ELEMENTS
    }
}

fn append_module_state(
    output: &mut Vec<f32>,
    state: Option<&NativeRnnModuleState>,
    layers: usize,
) -> Result<()> {
    if let Some(state) = state {
        ensure!(
            state.time_x_shift_b1c_by_layer.len() == layers
                && state.time_state_b1hkk_by_layer.len() == layers
                && state.channel_state_b1c_by_layer.len() == layers,
            "GPU scan recurrent state layer count mismatch"
        );
        for layer in 0..layers {
            append_tensor_values(output, &state.time_x_shift_b1c_by_layer[layer], CHANNELS)?;
            append_tensor_values(
                output,
                &state.time_state_b1hkk_by_layer[layer],
                TRANSFORM_MATRIX_ELEMENTS,
            )?;
            append_tensor_values(output, &state.channel_state_b1c_by_layer[layer], CHANNELS)?;
        }
    } else {
        output.resize(output.len() + layers * STATE_LAYER_ELEMENTS, 0.0);
    }
    Ok(())
}

fn append_tensor_values(output: &mut Vec<f32>, tensor: &Tensor, expected: usize) -> Result<()> {
    ensure!(
        tensor.elem_count() == expected,
        "GPU scan state tensor shape mismatch"
    );
    let data = f32_tensor_data(tensor)?;
    output.extend_from_slice(data.as_slice()?);
    Ok(())
}

struct ModuleResources {
    state: wgpu::Buffer,
    next_state: wgpu::Buffer,
    resident_state: Option<wgpu::Buffer>,
    host_state: wgpu::Buffer,
    state_sources: wgpu::Buffer,
    evicted_entities: wgpu::Buffer,
    evicted_state: wgpu::Buffer,
    output_slot: usize,
    row_metadata: wgpu::Buffer,
    transform_metadata: wgpu::Buffer,
    final_chunk_metadata: wgpu::Buffer,
    permutation: wgpu::Buffer,
    arena: Option<ArenaModuleResources>,
}

struct ArenaShardResources {
    state: wgpu::Buffer,
    transfers: wgpu::Buffer,
    gather_count: usize,
    scatter_count: usize,
}

struct ArenaModuleResources {
    host_transfers: wgpu::Buffer,
    host_transfer_count: usize,
    shards: Vec<ArenaShardResources>,
}

struct SharedResources {
    original_x: wgpu::Buffer,
    sequence_x: wgpu::Buffer,
    time_residual_x: wgpu::Buffer,
    v0: wgpu::Buffer,
    r: wgpu::Buffer,
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    w: wgpu::Buffer,
    k_deformed: wgpu::Buffer,
    g: wgpu::Buffer,
    time_output: wgpu::Buffer,
    transforms_a: wgpu::Buffer,
    transforms_b: wgpu::Buffer,
}

struct ReusableBuffer {
    label: &'static str,
    usage: wgpu::BufferUsages,
    buffer: Option<wgpu::Buffer>,
    capacity: usize,
}

impl ReusableBuffer {
    fn new(label: &'static str, usage: wgpu::BufferUsages) -> Self {
        Self {
            label,
            usage,
            buffer: None,
            capacity: 0,
        }
    }

    fn ensure(&mut self, context: &GpuContext, required_bytes: usize) -> Result<()> {
        self.ensure_with_growth(context, required_bytes, true)
    }

    fn ensure_exact(&mut self, context: &GpuContext, required_bytes: usize) -> Result<()> {
        self.ensure_with_growth(context, required_bytes, false)
    }

    fn ensure_with_growth(
        &mut self,
        context: &GpuContext,
        required_bytes: usize,
        add_headroom: bool,
    ) -> Result<()> {
        let required_bytes = required_bytes.max(4);
        if self.capacity >= required_bytes {
            return Ok(());
        }
        let advertised_max_capacity = if self.usage.contains(wgpu::BufferUsages::STORAGE) {
            context
                .limits
                .max_buffer_size
                .min(context.limits.max_storage_buffer_binding_size)
        } else {
            context.limits.max_buffer_size
        } as usize;
        let max_capacity = align_down(advertised_max_capacity, 4);
        ensure!(
            required_bytes <= max_capacity,
            "{} requires {required_bytes} bytes but this GPU supports at most {max_capacity}",
            self.label
        );
        let requested_capacity = if add_headroom {
            required_bytes
                .checked_add(required_bytes / 8)
                .unwrap_or(max_capacity)
                .min(max_capacity)
        } else {
            required_bytes
        };
        let capacity = align_up(requested_capacity.max(required_bytes), 4);
        self.buffer = Some(context.create_buffer(&wgpu::BufferDescriptor {
            label: Some(self.label),
            size: capacity as u64,
            usage: self.usage,
            mapped_at_creation: false,
        })?);
        self.capacity = capacity;
        Ok(())
    }

    fn upload(&mut self, context: &GpuContext, bytes: &[u8]) -> Result<()> {
        self.ensure(context, bytes.len())?;
        if !bytes.is_empty() {
            context.write_buffer(self.label, self.buffer(), 0, bytes)?;
        }
        Ok(())
    }

    fn buffer(&self) -> &wgpu::Buffer {
        self.buffer.as_ref().expect("reusable GPU buffer prepared")
    }

    fn cloned(&self) -> wgpu::Buffer {
        self.buffer().clone()
    }

    fn allocated_bytes(&self) -> usize {
        self.capacity
    }

    fn clear(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            buffer.destroy();
        }
        self.capacity = 0;
    }
}

struct ModuleBufferCache {
    state: ReusableBuffer,
    next_states: [ReusableBuffer; 2],
    resident_slot: Option<usize>,
    host_state: ReusableBuffer,
    state_sources: ReusableBuffer,
    evicted_entities: ReusableBuffer,
    evicted_state: ReusableBuffer,
    arena_host_transfers: ReusableBuffer,
    arena_shard_transfers: Vec<ReusableBuffer>,
    row_metadata: ReusableBuffer,
    transform_metadata: ReusableBuffer,
    final_chunk_metadata: ReusableBuffer,
    permutation: ReusableBuffer,
}

impl ModuleBufferCache {
    fn new() -> Self {
        let upload = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        Self {
            state: ReusableBuffer::new("rwkv-srs cached FP32 recurrent state", upload),
            next_states: [
                ReusableBuffer::new(
                    "rwkv-srs cached FP32 next recurrent state A",
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                ),
                ReusableBuffer::new(
                    "rwkv-srs cached FP32 next recurrent state B",
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                ),
            ],
            resident_slot: None,
            host_state: ReusableBuffer::new("rwkv-srs cached FP32 host state", upload),
            state_sources: ReusableBuffer::new("rwkv-srs cached state sources", upload),
            evicted_entities: ReusableBuffer::new("rwkv-srs cached evicted entities", upload),
            evicted_state: ReusableBuffer::new(
                "rwkv-srs cached evicted FP32 state",
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
            arena_host_transfers: ReusableBuffer::new(
                "rwkv-srs cached fully resident host transfers",
                upload,
            ),
            arena_shard_transfers: Vec::new(),
            row_metadata: ReusableBuffer::new("rwkv-srs cached row metadata", upload),
            transform_metadata: ReusableBuffer::new("rwkv-srs cached transform metadata", upload),
            final_chunk_metadata: ReusableBuffer::new("rwkv-srs cached replay metadata", upload),
            permutation: ReusableBuffer::new("rwkv-srs cached permutation", upload),
        }
    }

    fn prepare(
        &mut self,
        context: &GpuContext,
        plan: &ModulePlan,
        resident_enabled: bool,
        arena_buffers: Option<&[wgpu::Buffer]>,
    ) -> Result<ModuleResources> {
        let full_state_bytes = plan.state_elements() * size_of::<f32>();
        let arena_enabled = arena_buffers.is_some();
        ensure!(
            !resident_enabled || !arena_enabled,
            "GPU process module resources cannot use two resident state modes"
        );
        let resident_state = if resident_enabled {
            self.resident_slot
                .map(|slot| self.next_states[slot].cloned())
        } else {
            None
        };
        if arena_enabled {
            self.state.ensure(context, full_state_bytes)?;
            self.host_state
                .upload(context, bytemuck::cast_slice(&plan.state_values))?;
            self.state_sources
                .ensure(context, size_of::<StateSource>())?;
            self.evicted_entities.ensure(context, size_of::<u32>())?;
        } else if resident_state.is_some() {
            self.state.ensure(context, full_state_bytes)?;
            self.host_state
                .upload(context, bytemuck::cast_slice(&plan.state_values))?;
            self.state_sources
                .upload(context, bytemuck::cast_slice(&plan.state_sources))?;
            self.evicted_entities
                .upload(context, bytemuck::cast_slice(&plan.evicted_entities))?;
        } else {
            self.state
                .upload(context, bytemuck::cast_slice(&plan.state_values))?;
            self.host_state.ensure(context, size_of::<f32>())?;
            self.state_sources
                .ensure(context, size_of::<StateSource>())?;
            self.evicted_entities.ensure(context, size_of::<u32>())?;
        }
        self.evicted_state
            .ensure(context, plan.evicted_state_elements() * size_of::<f32>())?;
        let output_slot = if resident_enabled {
            self.resident_slot.map_or(0, |slot| 1 - slot)
        } else {
            0
        };
        self.next_states[output_slot].ensure(context, full_state_bytes)?;
        self.row_metadata
            .upload(context, bytemuck::cast_slice(&plan.row_metadata))?;
        self.transform_metadata
            .upload(context, bytemuck::cast_slice(&plan.transform_metadata))?;
        self.final_chunk_metadata
            .upload(context, bytemuck::cast_slice(&plan.final_chunk_metadata))?;
        self.permutation
            .upload(context, bytemuck::cast_slice(&plan.sequence_to_original))?;
        let arena = match (plan.arena_plan.as_ref(), arena_buffers) {
            (Some(arena_plan), Some(arena_buffers)) => {
                self.arena_host_transfers
                    .upload(context, bytemuck::cast_slice(&arena_plan.host_transfers))?;
                while self.arena_shard_transfers.len() < arena_plan.shards.len() {
                    self.arena_shard_transfers.push(ReusableBuffer::new(
                        "rwkv-srs cached fully resident shard transfers",
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    ));
                }
                let shards = arena_plan
                    .shards
                    .iter()
                    .zip(&mut self.arena_shard_transfers)
                    .map(|(shard_plan, transfer_buffer)| {
                        transfer_buffer
                            .upload(context, bytemuck::cast_slice(&shard_plan.transfers))?;
                        Ok(ArenaShardResources {
                            state: arena_buffers
                                .get(shard_plan.shard)
                                .cloned()
                                .context("fully resident GPU state shard buffer is missing")?,
                            transfers: transfer_buffer.cloned(),
                            gather_count: shard_plan.gather_count,
                            scatter_count: shard_plan.transfers.len(),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Some(ArenaModuleResources {
                    host_transfers: self.arena_host_transfers.cloned(),
                    host_transfer_count: arena_plan.host_transfers.len(),
                    shards,
                })
            }
            (None, None) => None,
            _ => bail!("fully resident GPU state plan/resource mismatch"),
        };
        Ok(ModuleResources {
            state: self.state.cloned(),
            next_state: self.next_states[output_slot].cloned(),
            resident_state,
            host_state: self.host_state.cloned(),
            state_sources: self.state_sources.cloned(),
            evicted_entities: self.evicted_entities.cloned(),
            evicted_state: self.evicted_state.cloned(),
            output_slot,
            row_metadata: self.row_metadata.cloned(),
            transform_metadata: self.transform_metadata.cloned(),
            final_chunk_metadata: self.final_chunk_metadata.cloned(),
            permutation: self.permutation.cloned(),
            arena,
        })
    }

    fn commit_resident(&mut self, slot: usize) {
        self.resident_slot = Some(slot);
    }

    fn resident_buffer(&self) -> Option<wgpu::Buffer> {
        self.resident_slot
            .map(|slot| self.next_states[slot].cloned())
    }

    fn clear_resident(&mut self) {
        self.resident_slot = None;
    }

    fn allocated_bytes(&self) -> usize {
        self.state.allocated_bytes()
            + self
                .next_states
                .iter()
                .map(ReusableBuffer::allocated_bytes)
                .sum::<usize>()
            + self.host_state.allocated_bytes()
            + self.state_sources.allocated_bytes()
            + self.evicted_entities.allocated_bytes()
            + self.evicted_state.allocated_bytes()
            + self.arena_host_transfers.allocated_bytes()
            + self
                .arena_shard_transfers
                .iter()
                .map(ReusableBuffer::allocated_bytes)
                .sum::<usize>()
            + self.row_metadata.allocated_bytes()
            + self.transform_metadata.allocated_bytes()
            + self.final_chunk_metadata.allocated_bytes()
            + self.permutation.allocated_bytes()
    }

    fn clear_transient_preserving_resident(&mut self) {
        self.state.clear();
        self.host_state.clear();
        self.state_sources.clear();
        self.evicted_entities.clear();
        self.evicted_state.clear();
        self.arena_host_transfers.clear();
        for buffer in &mut self.arena_shard_transfers {
            buffer.clear();
        }
        self.row_metadata.clear();
        self.transform_metadata.clear();
        self.final_chunk_metadata.clear();
        self.permutation.clear();
        for slot in 0..self.next_states.len() {
            if self.resident_slot != Some(slot) {
                self.next_states[slot].clear();
            }
        }
    }
}

struct ProcessBufferCache {
    feature: ReusableBuffer,
    original_x: ReusableBuffer,
    sequence_x: ReusableBuffer,
    time_residual_x: ReusableBuffer,
    v0: ReusableBuffer,
    r: ReusableBuffer,
    k: ReusableBuffer,
    v: ReusableBuffer,
    w: ReusableBuffer,
    k_deformed: ReusableBuffer,
    g: ReusableBuffer,
    time_output: ReusableBuffer,
    transforms_a: ReusableBuffer,
    transforms_b: ReusableBuffer,
    probabilities: ReusableBuffer,
    ahead_logits: ReusableBuffer,
    curve_weights: ReusableBuffer,
    modules: [ModuleBufferCache; 5],
    readbacks: [ReusableBuffer; 2],
}

impl ProcessBufferCache {
    fn new() -> Self {
        let storage = wgpu::BufferUsages::STORAGE;
        let upload = storage | wgpu::BufferUsages::COPY_DST;
        let output = storage | wgpu::BufferUsages::COPY_SRC;
        Self {
            feature: ReusableBuffer::new("rwkv-srs cached scan features", upload),
            original_x: ReusableBuffer::new("rwkv-srs cached original rows", storage),
            sequence_x: ReusableBuffer::new("rwkv-srs cached sequence rows", storage),
            time_residual_x: ReusableBuffer::new("rwkv-srs cached time residual rows", storage),
            v0: ReusableBuffer::new("rwkv-srs cached v0", storage),
            r: ReusableBuffer::new("rwkv-srs cached r", storage),
            k: ReusableBuffer::new("rwkv-srs cached k", storage),
            v: ReusableBuffer::new("rwkv-srs cached v", storage),
            w: ReusableBuffer::new("rwkv-srs cached w", storage),
            k_deformed: ReusableBuffer::new("rwkv-srs cached deformed k", storage),
            g: ReusableBuffer::new("rwkv-srs cached g", storage),
            time_output: ReusableBuffer::new("rwkv-srs cached recurrence output", storage),
            transforms_a: ReusableBuffer::new("rwkv-srs cached transforms A", storage),
            transforms_b: ReusableBuffer::new("rwkv-srs cached transforms B", storage),
            probabilities: ReusableBuffer::new("rwkv-srs cached probabilities", output),
            ahead_logits: ReusableBuffer::new("rwkv-srs cached ahead logits", output),
            curve_weights: ReusableBuffer::new("rwkv-srs cached curve weights", output),
            modules: std::array::from_fn(|_| ModuleBufferCache::new()),
            readbacks: [
                ReusableBuffer::new(
                    "rwkv-srs cached associative scan readback A",
                    wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                ),
                ReusableBuffer::new(
                    "rwkv-srs cached associative scan readback B",
                    wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                ),
            ],
        }
    }

    fn prepare_shared(
        &mut self,
        context: &GpuContext,
        features: &[f32],
        row_bytes: usize,
        transform_bytes: usize,
        probability_bytes: usize,
        curve_bytes: usize,
    ) -> Result<(
        wgpu::Buffer,
        SharedResources,
        wgpu::Buffer,
        wgpu::Buffer,
        wgpu::Buffer,
    )> {
        self.feature
            .upload(context, bytemuck::cast_slice(features))?;
        for buffer in [
            &mut self.original_x,
            &mut self.sequence_x,
            &mut self.time_residual_x,
            &mut self.v0,
            &mut self.r,
            &mut self.k,
            &mut self.v,
            &mut self.w,
            &mut self.k_deformed,
            &mut self.g,
            &mut self.time_output,
        ] {
            buffer.ensure(context, row_bytes)?;
        }
        self.transforms_a.ensure(context, transform_bytes)?;
        self.transforms_b.ensure(context, transform_bytes)?;
        self.probabilities.ensure(context, probability_bytes)?;
        self.ahead_logits.ensure(context, curve_bytes)?;
        self.curve_weights.ensure(context, curve_bytes)?;
        Ok((
            self.feature.cloned(),
            SharedResources {
                original_x: self.original_x.cloned(),
                sequence_x: self.sequence_x.cloned(),
                time_residual_x: self.time_residual_x.cloned(),
                v0: self.v0.cloned(),
                r: self.r.cloned(),
                k: self.k.cloned(),
                v: self.v.cloned(),
                w: self.w.cloned(),
                k_deformed: self.k_deformed.cloned(),
                g: self.g.cloned(),
                time_output: self.time_output.cloned(),
                transforms_a: self.transforms_a.cloned(),
                transforms_b: self.transforms_b.cloned(),
            },
            self.probabilities.cloned(),
            self.ahead_logits.cloned(),
            self.curve_weights.cloned(),
        ))
    }

    fn allocated_bytes(&self) -> usize {
        [
            &self.feature,
            &self.original_x,
            &self.sequence_x,
            &self.time_residual_x,
            &self.v0,
            &self.r,
            &self.k,
            &self.v,
            &self.w,
            &self.k_deformed,
            &self.g,
            &self.time_output,
            &self.transforms_a,
            &self.transforms_b,
            &self.probabilities,
            &self.ahead_logits,
            &self.curve_weights,
        ]
        .into_iter()
        .map(ReusableBuffer::allocated_bytes)
        .sum::<usize>()
            + self
                .readbacks
                .iter()
                .map(ReusableBuffer::allocated_bytes)
                .sum::<usize>()
            + self
                .modules
                .iter()
                .map(ModuleBufferCache::allocated_bytes)
                .sum::<usize>()
    }

    fn clear_transient_preserving_resident(&mut self, context: &GpuContext) -> Result<()> {
        for buffer in [
            &mut self.feature,
            &mut self.original_x,
            &mut self.sequence_x,
            &mut self.time_residual_x,
            &mut self.v0,
            &mut self.r,
            &mut self.k,
            &mut self.v,
            &mut self.w,
            &mut self.k_deformed,
            &mut self.g,
            &mut self.time_output,
            &mut self.transforms_a,
            &mut self.transforms_b,
            &mut self.probabilities,
            &mut self.ahead_logits,
            &mut self.curve_weights,
        ] {
            buffer.clear();
        }
        for readback in &mut self.readbacks {
            readback.clear();
        }
        for module in &mut self.modules {
            module.clear_transient_preserving_resident();
        }
        context.queue.submit([]);
        context
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("GPU allocation recovery did not complete")?;
        Ok(())
    }
}

impl GpuProcessScan {
    fn should_use_fully_resident_state(
        &self,
        defer_cpu_state: bool,
        preferred_for_input: bool,
    ) -> bool {
        if !defer_cpu_state {
            return false;
        }
        if self.fully_resident_disabled_after_oom {
            return false;
        }
        match fully_resident_state_policy() {
            FullyResidentStatePolicy::Enabled => true,
            FullyResidentStatePolicy::Disabled => false,
            FullyResidentStatePolicy::Auto => preferred_for_input,
        }
    }

    fn process(
        &mut self,
        rnn: &mut NativeRnn,
        ids: Vec<ReviewIds>,
        features: Vec<f32>,
        mode: GpuProcessScanMode,
        defer_cpu_state: bool,
        fully_resident_enabled: bool,
    ) -> Result<GpuScanProcessOutput> {
        let total_start = Instant::now();
        self.last_fused_row_batched_prepare = false;
        self.last_fused_row_batched_channel = false;
        debug_assert!(!fully_resident_enabled || defer_cpu_state);
        let resident_enabled = defer_cpu_state && !fully_resident_enabled;
        let switching_residency = (fully_resident_enabled
            && self.resident_states.iter().any(Option::is_some))
            || (!fully_resident_enabled && self.fully_resident_states.iter().any(Option::is_some));
        if switching_residency
            || (!defer_cpu_state
                && (self.resident_states.iter().any(Option::is_some)
                    || self.fully_resident_states.iter().any(Option::is_some)))
        {
            self.synchronize_cpu_state(rnn)?;
        }
        if !defer_cpu_state && !self.deferred_states.is_empty() {
            self.deferred_states.synchronize(rnn)?;
        }
        let plan_start = Instant::now();
        let reviews = ids.len();
        ensure!(reviews > 0, "GPU scan requires at least one review");
        let rows_per_review = mode.rows_per_review();
        let rows = reviews
            .checked_mul(rows_per_review)
            .context("GPU scan row count overflow")?;
        ensure!(
            rows <= self.context.limits.max_compute_workgroups_per_dimension as usize,
            "GPU scan batch has {rows} interleaved rows, exceeding the device workgroup limit"
        );
        let chunk_reviews = scan_chunk_reviews()?;
        if fully_resident_enabled {
            for (module_index, kind) in ModuleKind::ALL.into_iter().enumerate() {
                if self.fully_resident_states[module_index].is_none() {
                    self.fully_resident_states[module_index] =
                        Some(FullyResidentModuleState::new(&self.context, kind.layers())?);
                }
            }
        }
        let mut plans = Vec::with_capacity(5);
        for (module_index, kind) in ModuleKind::ALL.into_iter().enumerate() {
            plans.push(ModulePlan::build(
                rnn,
                &self.deferred_states,
                &ids,
                kind,
                rnn.weights.rwkv_modules[module_index].blocks.len(),
                chunk_reviews,
                if resident_enabled {
                    self.resident_states[module_index].as_ref()
                } else {
                    None
                },
                if fully_resident_enabled {
                    self.fully_resident_states[module_index].as_ref()
                } else {
                    None
                },
                mode,
            )?);
        }
        let max_transforms = plans
            .iter()
            .map(|plan| plan.transform_metadata.len())
            .max()
            .unwrap_or(0);
        ensure!(
            max_transforms <= self.context.limits.max_compute_workgroups_per_dimension as usize,
            "GPU scan requires {max_transforms} transform workgroups, exceeding the device limit"
        );
        let readback_bytes = process_readback_bytes(
            &plans,
            reviews,
            mode,
            resident_enabled,
            fully_resident_enabled,
        )?;
        if readback_bytes > self.context.limits.max_buffer_size as usize
            && resident_enabled
            && self.resident_states.iter().any(Option::is_some)
        {
            self.resident_readback_recoveries += 1;
            self.synchronize_cpu_state(rnn)?;
            return self.process(
                rnn,
                ids,
                features,
                mode,
                defer_cpu_state,
                fully_resident_enabled,
            );
        }
        ensure!(
            readback_bytes <= self.context.limits.max_buffer_size as usize,
            "GPU process readback requires {readback_bytes} bytes but this adapter supports at most {}",
            self.context.limits.max_buffer_size
        );
        let plan_ns = plan_start.elapsed().as_nanos();
        let setup_start = Instant::now();

        ensure!(
            features.len() == rows * FEATURE_DIM,
            "GPU scan flat feature row count mismatch"
        );
        let row_bytes = rows * CHANNELS * size_of::<f32>();
        let transform_bytes = max_transforms.max(1) * TRANSFORM_ELEMENTS * size_of::<f32>();
        let probability_bytes = if mode.returns_predictions() {
            reviews * size_of::<f32>()
        } else {
            0
        };
        let curve_bytes = if mode.returns_curves() {
            reviews * CHANNELS * size_of::<f32>()
        } else {
            size_of::<f32>()
        };
        let (feature_buffer, shared, probabilities, ahead_logits, curve_weights) =
            self.buffers.prepare_shared(
                &self.context,
                &features,
                row_bytes,
                transform_bytes,
                probability_bytes,
                curve_bytes,
            )?;
        let mut arena_buffers = vec![None; plans.len()];
        if fully_resident_enabled {
            for (module_index, plan) in plans.iter().enumerate() {
                let arena_plan = plan
                    .arena_plan
                    .as_ref()
                    .context("fully resident GPU state plan is missing")?;
                let arena = self.fully_resident_states[module_index]
                    .as_mut()
                    .context("fully resident GPU state module is missing")?;
                arena.ensure_plan(&self.context, arena_plan)?;
                arena_buffers[module_index] = Some(arena.buffers()?);
            }
        }
        let module_resources = self
            .buffers
            .modules
            .iter_mut()
            .zip(&plans)
            .zip(&arena_buffers)
            .map(|((cache, plan), arena_buffers)| {
                cache.prepare(
                    &self.context,
                    plan,
                    resident_enabled,
                    arena_buffers.as_deref(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let feature_bind_group =
            self.context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("rwkv-srs scan feature encoding"),
                    layout: &self.pipelines.encode_features.get_bind_group_layout(0),
                    entries: &[
                        binding(0, &self.weight_buffer),
                        binding(1, &self.metadata_buffer),
                        binding(2, &feature_buffer),
                        binding(3, &shared.original_x),
                    ],
                });
        let output_bind_group = if mode.returns_predictions() {
            let head_params = uniform_buffer(
                &self.context,
                "rwkv-srs scan head parameters",
                &FourU32 {
                    first: rows as u32,
                    second: u32::from(mode.returns_curves()),
                    third: 0,
                    fourth: 0,
                },
            )?;
            Some(
                self.context
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("rwkv-srs scan output heads"),
                        layout: &self.pipelines.output_heads.get_bind_group_layout(0),
                        entries: &[
                            binding(0, &self.weight_buffer),
                            binding(1, &self.metadata_buffer),
                            binding(3, &shared.original_x),
                            binding(4, &probabilities),
                            binding(5, &ahead_logits),
                            binding(6, &curve_weights),
                            binding(7, &head_params),
                        ],
                    }),
            )
        } else {
            None
        };
        let setup_ns = setup_start.elapsed().as_nanos();
        let encode_start = Instant::now();

        let timestamp_segments = 1
            + usize::from(mode.returns_predictions())
            + plans
                .iter()
                .zip(&module_resources)
                .map(|(plan, resources)| {
                    let mut scan_levels = 0usize;
                    let mut scan_offset = 1usize;
                    while scan_offset < plan.max_transform_run {
                        scan_levels += 1;
                        scan_offset *= 2;
                    }
                    let transform_passes =
                        usize::from(!plan.transform_metadata.is_empty()) + scan_levels;
                    let state_passes = usize::from(resources.resident_state.is_some())
                        + usize::from(
                            resources.resident_state.is_some() && !plan.evicted_keys.is_empty(),
                        )
                        + resources.arena.as_ref().map_or(0, |arena| {
                            usize::from(arena.host_transfer_count > 0)
                                + arena
                                    .shards
                                    .iter()
                                    .filter(|shard| shard.gather_count > 0)
                                    .count()
                        });
                    2 + state_passes + plan.layers * (4 + transform_passes)
                })
                .sum::<usize>();
        let mut timestamp_profile = GpuTimestampProfile::new(&self.context, timestamp_segments)?;

        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rwkv-srs associative scan process_many"),
                });
        dispatch_profiled(
            &mut encoder,
            "rwkv-srs scan feature encoding",
            &self.pipelines.encode_features,
            &[(&feature_bind_group, 0)],
            rows as u32,
            &mut timestamp_profile,
            GpuTimestampStage::FeatureEncoding,
        );
        let use_fused_row_batched_prepare =
            self.fused_row_batched_prepare && rows >= FUSED_ROW_BATCHED_MIN_ROWS;
        let use_fused_row_batched_channel =
            self.fused_row_batched_channel && rows >= FUSED_ROW_BATCHED_MIN_ROWS;

        for (module_index, ((plan, resources), layer_offset)) in plans
            .iter()
            .zip(&module_resources)
            .zip(&self.packed.module_layer_offsets)
            .enumerate()
        {
            if let Some(resident_state) = resources.resident_state.as_ref() {
                let state_transfer_params = uniform_buffer(
                    &self.context,
                    "rwkv-srs resident state transfer parameters",
                    &FourU32 {
                        first: (plan.layers * STATE_LAYER_ELEMENTS) as u32,
                        second: plan.keys.len() as u32,
                        third: plan.evicted_keys.len() as u32,
                        fourth: 0,
                    },
                )?;
                let state_transfer_bind_group =
                    self.context
                        .device
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("rwkv-srs resident state transfer"),
                            layout: &self.pipelines.state_transfer_bind_group_layout,
                            entries: &[
                                binding(0, resident_state),
                                binding(1, &resources.host_state),
                                binding(2, &resources.state_sources),
                                binding(3, &resources.state),
                                binding(4, &resources.evicted_entities),
                                binding(5, &resources.evicted_state),
                                binding(6, &state_transfer_params),
                            ],
                        });
                dispatch_profiled(
                    &mut encoder,
                    "rwkv-srs gather resident current states",
                    &self.pipelines.gather_current_state,
                    &[(&state_transfer_bind_group, 0)],
                    plan.keys.len() as u32,
                    &mut timestamp_profile,
                    GpuTimestampStage::StateTransfer,
                );
                if !plan.evicted_keys.is_empty() {
                    dispatch_profiled(
                        &mut encoder,
                        "rwkv-srs gather evicted resident states",
                        &self.pipelines.gather_evicted_state,
                        &[(&state_transfer_bind_group, 0)],
                        plan.evicted_keys.len() as u32,
                        &mut timestamp_profile,
                        GpuTimestampStage::StateTransfer,
                    );
                }
            }
            if let Some(arena) = resources.arena.as_ref() {
                let state_stride = plan.layers * STATE_LAYER_ELEMENTS;
                if arena.host_transfer_count > 0 {
                    let params = uniform_buffer(
                        &self.context,
                        "rwkv-srs fully resident host-state transfer parameters",
                        &FourU32 {
                            first: state_stride as u32,
                            second: arena.host_transfer_count as u32,
                            third: 0,
                            fourth: 0,
                        },
                    )?;
                    let bind_group = arena_transfer_bind_group(
                        &self.context,
                        &self.pipelines.arena_transfer_bind_group_layout,
                        &resources.host_state,
                        &resources.state,
                        &arena.host_transfers,
                        &params,
                    );
                    dispatch_profiled_2d(
                        &mut encoder,
                        "rwkv-srs gather fully resident host misses",
                        &self.pipelines.gather_arena_state,
                        &bind_group,
                        state_stride.div_ceil(256) as u32,
                        arena.host_transfer_count as u32,
                        &mut timestamp_profile,
                        GpuTimestampStage::StateTransfer,
                    );
                }
                for shard in &arena.shards {
                    if shard.gather_count == 0 {
                        continue;
                    }
                    let params = uniform_buffer(
                        &self.context,
                        "rwkv-srs fully resident shard-gather parameters",
                        &FourU32 {
                            first: state_stride as u32,
                            second: shard.gather_count as u32,
                            third: 0,
                            fourth: 0,
                        },
                    )?;
                    let bind_group = arena_transfer_bind_group(
                        &self.context,
                        &self.pipelines.arena_transfer_bind_group_layout,
                        &shard.state,
                        &resources.state,
                        &shard.transfers,
                        &params,
                    );
                    dispatch_profiled_2d(
                        &mut encoder,
                        "rwkv-srs gather fully resident state shard",
                        &self.pipelines.gather_arena_state,
                        &bind_group,
                        state_stride.div_ceil(256) as u32,
                        shard.gather_count as u32,
                        &mut timestamp_profile,
                        GpuTimestampStage::StateTransfer,
                    );
                }
            }
            let topology_params = uniform_buffer(
                &self.context,
                "rwkv-srs scan topology parameters",
                &FourU32 {
                    first: rows as u32,
                    second: 0,
                    third: 0,
                    fourth: 0,
                },
            )?;
            let topology_bind_group =
                self.context
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("rwkv-srs scan topology"),
                        layout: &self.pipelines.gather.get_bind_group_layout(0),
                        entries: &[
                            binding(0, &shared.original_x),
                            binding(1, &shared.sequence_x),
                            binding(2, &resources.permutation),
                            binding(3, &topology_params),
                        ],
                    });
            dispatch_profiled(
                &mut encoder,
                "rwkv-srs scan gather",
                &self.pipelines.gather,
                &[(&topology_bind_group, 0)],
                (rows * CHANNELS).div_ceil(256) as u32,
                &mut timestamp_profile,
                GpuTimestampStage::TopologyGather,
            );

            let module_bind_group =
                self.context
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("rwkv-srs scan module"),
                        layout: &self.pipelines.module_bind_group_layout,
                        entries: &[
                            binding(0, &self.weight_buffer),
                            binding(1, &self.metadata_buffer),
                            binding(2, &shared.sequence_x),
                            binding(3, &shared.time_residual_x),
                            binding(4, &shared.v0),
                            binding(5, &shared.r),
                            binding(6, &shared.k),
                            binding(7, &shared.v),
                            binding(8, &shared.w),
                            binding(9, &shared.k_deformed),
                            binding(10, &shared.g),
                            binding(11, &shared.time_output),
                            binding(12, &resources.state),
                            binding(13, &resources.row_metadata),
                            binding(14, &resources.transform_metadata),
                            binding(15, &resources.final_chunk_metadata),
                            binding(16, &shared.transforms_a),
                            binding(17, &shared.transforms_b),
                            binding(18, &resources.next_state),
                        ],
                    });

            for local_layer in 0..plan.layers {
                let base_params = DispatchParams {
                    row_count: rows as u32,
                    layer_index: (*layer_offset + local_layer) as u32,
                    local_layer: local_layer as u32,
                    layers: plan.layers as u32,
                    scan_offset: 0,
                    scan_source_b: 0,
                    transform_count: plan.transform_metadata.len() as u32,
                    final_chunk_count: plan.final_chunk_metadata.len() as u32,
                };
                let base_params_buffer = uniform_buffer(
                    &self.context,
                    "rwkv-srs scan layer parameters",
                    &base_params,
                )?;
                let base_params_bind_group = dispatch_bind_group(
                    &self.context,
                    &self.pipelines.dispatch_bind_group_layout,
                    &base_params_buffer,
                );
                let (prepare_pipeline, prepare_workgroups) = if use_fused_row_batched_prepare {
                    (
                        self.pipelines
                            .prepare_layer_fused_rows
                            .as_ref()
                            .context("fused row-batched preparation pipeline is missing")?,
                        rows.div_ceil(2) as u32,
                    )
                } else {
                    (&self.pipelines.prepare_layer, rows as u32)
                };
                dispatch_profiled(
                    &mut encoder,
                    "rwkv-srs scan prepare layer",
                    prepare_pipeline,
                    &[(&module_bind_group, 0), (&base_params_bind_group, 1)],
                    prepare_workgroups,
                    &mut timestamp_profile,
                    GpuTimestampStage::PrepareLayer,
                );

                if !plan.transform_metadata.is_empty() {
                    let build_transforms = if mode == GpuProcessScanMode::StateOnly {
                        &self.pipelines.build_transforms_state_only
                    } else {
                        &self.pipelines.build_transforms
                    };
                    dispatch_profiled(
                        &mut encoder,
                        "rwkv-srs scan build transforms",
                        build_transforms,
                        &[(&module_bind_group, 0), (&base_params_bind_group, 1)],
                        plan.transform_metadata.len() as u32,
                        &mut timestamp_profile,
                        GpuTimestampStage::BuildTransforms,
                    );
                }

                let mut scan_source_b = false;
                let mut scan_offset = 1usize;
                while scan_offset < plan.max_transform_run {
                    let scan_params = DispatchParams {
                        scan_offset: scan_offset as u32,
                        scan_source_b: u32::from(scan_source_b),
                        ..base_params
                    };
                    let scan_params_buffer = uniform_buffer(
                        &self.context,
                        "rwkv-srs scan level parameters",
                        &scan_params,
                    )?;
                    let scan_params_bind_group = dispatch_bind_group(
                        &self.context,
                        &self.pipelines.dispatch_bind_group_layout,
                        &scan_params_buffer,
                    );
                    dispatch_profiled(
                        &mut encoder,
                        "rwkv-srs parallel associative scan",
                        &self.pipelines.scan_transforms,
                        &[(&module_bind_group, 0), (&scan_params_bind_group, 1)],
                        plan.transform_metadata.len() as u32,
                        &mut timestamp_profile,
                        GpuTimestampStage::ScanTransforms,
                    );
                    scan_source_b = !scan_source_b;
                    scan_offset *= 2;
                }
                let replay_params = DispatchParams {
                    scan_source_b: u32::from(scan_source_b),
                    ..base_params
                };
                let replay_params_buffer = uniform_buffer(
                    &self.context,
                    "rwkv-srs scan replay parameters",
                    &replay_params,
                )?;
                let replay_params_bind_group = dispatch_bind_group(
                    &self.context,
                    &self.pipelines.dispatch_bind_group_layout,
                    &replay_params_buffer,
                );
                let replay_chunks = if mode == GpuProcessScanMode::StateOnly {
                    &self.pipelines.replay_chunks_state_only
                } else {
                    &self.pipelines.replay_chunks
                };
                dispatch_profiled(
                    &mut encoder,
                    "rwkv-srs scan chunk replay",
                    replay_chunks,
                    &[(&module_bind_group, 0), (&replay_params_bind_group, 1)],
                    plan.final_chunk_metadata.len() as u32,
                    &mut timestamp_profile,
                    GpuTimestampStage::Replay,
                );
                dispatch_profiled(
                    &mut encoder,
                    "rwkv-srs scan time output",
                    &self.pipelines.finish_time,
                    &[(&module_bind_group, 0), (&base_params_bind_group, 1)],
                    rows as u32,
                    &mut timestamp_profile,
                    GpuTimestampStage::FinishTime,
                );
                dispatch_profiled(
                    &mut encoder,
                    "rwkv-srs scan channel output",
                    if use_fused_row_batched_channel {
                        self.pipelines
                            .finish_channel_fused_rows
                            .as_ref()
                            .context("fused row-batched ChannelMixer pipeline is missing")?
                    } else {
                        &self.pipelines.finish_channel
                    },
                    &[(&module_bind_group, 0), (&base_params_bind_group, 1)],
                    if use_fused_row_batched_channel {
                        rows.div_ceil(4) as u32
                    } else {
                        rows as u32
                    },
                    &mut timestamp_profile,
                    GpuTimestampStage::FinishChannel,
                );
            }

            dispatch_profiled(
                &mut encoder,
                "rwkv-srs scan scatter",
                &self.pipelines.scatter,
                &[(&topology_bind_group, 0)],
                (rows * CHANNELS).div_ceil(256) as u32,
                &mut timestamp_profile,
                GpuTimestampStage::TopologyScatter,
            );
            if scan_profile_enabled() {
                eprintln!(
                    "gpu_process_scan_plan module={module_index} entities={} resident_hits={} evicted={} transforms={} replay_chunks={} max_scan_run={}",
                    plan.keys.len(),
                    plan.state_sources
                        .iter()
                        .filter(|source| source.flags & STATE_SOURCE_RESIDENT != 0)
                        .count(),
                    plan.evicted_keys.len(),
                    plan.transform_metadata.len(),
                    plan.final_chunk_metadata.len(),
                    plan.max_transform_run,
                );
            }
        }

        if let Some(output_bind_group) = output_bind_group.as_ref() {
            dispatch_profiled(
                &mut encoder,
                "rwkv-srs scan output heads",
                &self.pipelines.output_heads,
                &[(output_bind_group, 0)],
                if mode == GpuProcessScanMode::Predictions {
                    reviews as u32
                } else {
                    rows as u32
                },
                &mut timestamp_profile,
                GpuTimestampStage::OutputHeads,
            );
        }

        let probability_offset = 0usize;
        let ahead_offset = align_up(probability_bytes, 4);
        let weights_offset = ahead_offset
            + if mode.returns_curves() {
                curve_bytes
            } else {
                0
            };
        let mut state_offsets = Vec::with_capacity(module_resources.len());
        let mut readback_size = weights_offset
            + if mode.returns_curves() {
                curve_bytes
            } else {
                0
            };
        for (plan, resources) in plans.iter().zip(&module_resources) {
            readback_size = align_up(readback_size, 4);
            state_offsets.push(readback_size);
            let bytes = if fully_resident_enabled {
                0
            } else if resident_enabled {
                plan.evicted_state_elements()
            } else {
                plan.state_elements()
            } * size_of::<f32>();
            readback_size += bytes;
            let _ = resources;
        }
        let timestamp_offset = align_up(readback_size, std::mem::align_of::<u64>());
        let timestamp_bytes = timestamp_profile
            .as_ref()
            .map_or(0, GpuTimestampProfile::bytes);
        if timestamp_bytes > 0 {
            readback_size = timestamp_offset + timestamp_bytes;
        }
        let needs_completion_sentinel = readback_size == 0 && timestamp_bytes == 0;
        if needs_completion_sentinel {
            readback_size = size_of::<f32>();
        }
        self.buffers.readbacks[0].ensure(&self.context, readback_size)?;
        let readback = self.buffers.readbacks[0].cloned();
        if probability_bytes > 0 {
            encoder.copy_buffer_to_buffer(
                &probabilities,
                0,
                &readback,
                probability_offset as u64,
                probability_bytes as u64,
            );
        }
        if mode.returns_curves() {
            encoder.copy_buffer_to_buffer(
                &ahead_logits,
                0,
                &readback,
                ahead_offset as u64,
                curve_bytes as u64,
            );
            encoder.copy_buffer_to_buffer(
                &curve_weights,
                0,
                &readback,
                weights_offset as u64,
                curve_bytes as u64,
            );
        }
        if needs_completion_sentinel {
            encoder.copy_buffer_to_buffer(&probabilities, 0, &readback, 0, size_of::<f32>() as u64);
        }
        for (((plan, resources), offset), _module) in plans
            .iter()
            .zip(&module_resources)
            .zip(&state_offsets)
            .zip(0..)
        {
            let state_elements = if fully_resident_enabled {
                0
            } else if resident_enabled {
                plan.evicted_state_elements()
            } else {
                plan.state_elements()
            };
            if state_elements > 0 {
                encoder.copy_buffer_to_buffer(
                    if resident_enabled {
                        &resources.evicted_state
                    } else {
                        &resources.next_state
                    },
                    0,
                    &readback,
                    *offset as u64,
                    (state_elements * size_of::<f32>()) as u64,
                );
            }
        }
        if let Some(profile) = timestamp_profile.as_ref() {
            profile.resolve_into(&mut encoder, &readback, timestamp_offset);
        }
        let encode_ns = encode_start.elapsed().as_nanos();

        let submit_start = Instant::now();
        self.context.queue.submit([encoder.finish()]);
        wait_for_readback(&self.context, &readback, readback_size)?;
        let submit_ns = submit_start.elapsed().as_nanos();
        let materialize_start = Instant::now();
        let mapped = readback
            .get_mapped_range(0..readback_size as u64)
            .context("GPU associative scan readback range is unavailable")?;
        let result = decode_process_readback(
            rnn,
            &plans,
            &mapped,
            probability_offset,
            probability_bytes,
            ahead_offset,
            weights_offset,
            curve_bytes,
            mode,
            &state_offsets,
            defer_cpu_state,
            resident_enabled,
            fully_resident_enabled,
            &mut self.deferred_states,
        );
        let timestamp_summary = timestamp_profile
            .as_ref()
            .map(|profile| {
                profile.summary(
                    &self.context,
                    &mapped[timestamp_offset..timestamp_offset + timestamp_bytes],
                )
            })
            .transpose();
        drop(mapped);
        readback.unmap();
        let output = result?;
        if let Some(summary) = timestamp_summary? {
            eprintln!("{summary}");
        }
        let materialize_ns = materialize_start.elapsed().as_nanos();
        let arena_commit_start = Instant::now();
        if fully_resident_enabled {
            let mut scatter_encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("rwkv-srs commit fully resident process state"),
                    });
            let mut no_profile = None;
            for (plan, resources) in plans.iter().zip(&module_resources) {
                let state_stride = plan.layers * STATE_LAYER_ELEMENTS;
                let arena = resources
                    .arena
                    .as_ref()
                    .context("fully resident GPU state resources are missing")?;
                for shard in &arena.shards {
                    let params = uniform_buffer(
                        &self.context,
                        "rwkv-srs fully resident shard-scatter parameters",
                        &FourU32 {
                            first: state_stride as u32,
                            second: shard.scatter_count as u32,
                            third: 0,
                            fourth: 0,
                        },
                    )?;
                    let bind_group = arena_transfer_bind_group(
                        &self.context,
                        &self.pipelines.arena_transfer_bind_group_layout,
                        &resources.next_state,
                        &shard.state,
                        &shard.transfers,
                        &params,
                    );
                    dispatch_profiled_2d(
                        &mut scatter_encoder,
                        "rwkv-srs scatter fully resident state shard",
                        &self.pipelines.scatter_arena_state,
                        &bind_group,
                        state_stride.div_ceil(256) as u32,
                        shard.scatter_count as u32,
                        &mut no_profile,
                        GpuTimestampStage::StateTransfer,
                    );
                }
            }
            self.context.queue.submit([scatter_encoder.finish()]);
            self.context
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .context("fully resident GPU state commit did not complete")?;
            for (module_index, plan) in plans.iter().enumerate() {
                let arena_plan = plan
                    .arena_plan
                    .as_ref()
                    .context("fully resident GPU state commit plan is missing")?;
                self.fully_resident_states[module_index]
                    .as_mut()
                    .context("fully resident GPU state module is missing")?
                    .commit(arena_plan)?;
                for key in plan.keys.iter().copied() {
                    self.deferred_states.remove(rnn, plan.kind, key);
                }
            }
        } else if resident_enabled {
            for (module_index, (plan, resources)) in plans.iter().zip(&module_resources).enumerate()
            {
                self.buffers.modules[module_index].commit_resident(resources.output_slot);
                self.resident_states[module_index] =
                    Some(ResidentModuleState::new(plan.keys.clone()));
            }
        }
        let arena_commit_ns = if fully_resident_enabled {
            arena_commit_start.elapsed().as_nanos()
        } else {
            0
        };
        self.fully_resident_commit_ns += arena_commit_ns;
        self.last_fully_resident_commit_ns = arena_commit_ns;
        self.last_fused_row_batched_prepare = use_fused_row_batched_prepare;
        self.last_fused_row_batched_channel = use_fused_row_batched_channel;
        self.fused_row_batched_prepare_batches += u64::from(use_fused_row_batched_prepare);
        self.fused_row_batched_channel_batches += u64::from(use_fused_row_batched_channel);
        if scan_profile_enabled() {
            eprintln!(
                "gpu_process_scan rows={reviews} chunk_reviews={chunk_reviews} plan_ns={plan_ns} setup_ns={setup_ns} encode_ns={encode_ns} submit_readback_ns={submit_ns} materialize_ns={} arena_commit_ns={arena_commit_ns} total_ns={} mode={} fully_resident={fully_resident_enabled}",
                materialize_ns,
                total_start.elapsed().as_nanos(),
                mode.label(),
            );
        }
        Ok(output)
    }

    fn synchronize_cpu_state(&mut self, rnn: &mut NativeRnn) -> Result<()> {
        let total_start = Instant::now();
        let mut profile = CpuStateSyncProfile {
            calls: 1,
            ..CpuStateSyncProfile::default()
        };
        let result = self.synchronize_cpu_state_profiled(rnn, &mut profile);
        profile.total_ns = total_start.elapsed().as_nanos();
        self.cpu_state_sync_profile.add(profile);
        self.last_cpu_state_sync_profile = profile;
        result
    }

    fn fully_resident_sync_chunks(&self) -> Result<Vec<FullyResidentSyncChunk>> {
        let mut chunks = Vec::new();
        for (module_index, arena) in self.fully_resident_states.iter().enumerate() {
            let Some(arena) = arena.as_ref() else {
                continue;
            };
            let kind = ModuleKind::ALL[module_index];
            let layers = kind.layers();
            let entity_bytes = layers
                .checked_mul(STATE_LAYER_ELEMENTS)
                .and_then(|elements| elements.checked_mul(size_of::<f32>()))
                .context("fully resident CPU synchronization entity size overflow")?;
            let entities_per_chunk = (FULLY_RESIDENT_SYNC_CHUNK_BYTES / entity_bytes).max(1);
            for shard in arena.shards.iter().filter(|shard| !shard.keys.is_empty()) {
                let source = shard.buffer()?;
                for (chunk_index, chunk_keys) in shard.keys.chunks(entities_per_chunk).enumerate() {
                    let state_bytes = chunk_keys
                        .len()
                        .checked_mul(entity_bytes)
                        .context("fully resident CPU synchronization chunk size overflow")?;
                    let source_offset = chunk_index
                        .checked_mul(entities_per_chunk)
                        .and_then(|entities| entities.checked_mul(entity_bytes))
                        .context("fully resident CPU synchronization offset overflow")?;
                    chunks.push(FullyResidentSyncChunk {
                        kind,
                        layers,
                        keys: chunk_keys.to_vec(),
                        source: source.clone(),
                        source_offset,
                        state_bytes,
                    });
                }
            }
        }
        Ok(chunks)
    }

    fn synchronize_cpu_state_profiled(
        &mut self,
        rnn: &mut NativeRnn,
        profile: &mut CpuStateSyncProfile,
    ) -> Result<()> {
        self.last_shared_cpu_state_view_chunks = 0;
        self.last_shared_cpu_state_view_entities = 0;
        let use_shared_cpu_state_views = shared_cpu_state_views_enabled();
        let use_flat_cpu_state = self.flat_cpu_state;
        let mut pending_flat_state = FlatNativeRnnState::default();
        for module_index in 0..self.resident_states.len() {
            let Some(resident) = self.resident_states[module_index].as_ref() else {
                continue;
            };
            let keys = resident.keys.clone();
            let kind = ModuleKind::ALL[module_index];
            let state_bytes = keys.len() * kind.layers() * STATE_LAYER_ELEMENTS * size_of::<f32>();
            self.buffers.readbacks[0].ensure(&self.context, state_bytes)?;
            let readback = self.buffers.readbacks[0].cloned();
            let buffer = self.buffers.modules[module_index]
                .resident_buffer()
                .context("resident GPU process state buffer is unavailable")?;
            let copy_start = Instant::now();
            let mut encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("rwkv-srs resident process state synchronization"),
                    });
            encoder.copy_buffer_to_buffer(&buffer, 0, &readback, 0, state_bytes as u64);
            self.context.queue.submit([encoder.finish()]);
            profile.copy_map_submit_ns += copy_start.elapsed().as_nanos();
            let wait_start = Instant::now();
            wait_for_readback(&self.context, &readback, state_bytes)?;
            let mapped = readback
                .get_mapped_range(0..state_bytes as u64)
                .context("resident GPU process state readback range is unavailable")?;
            profile.copy_map_wait_ns += wait_start.elapsed().as_nanos();
            profile.chunks += 1;
            profile.entities += keys.len() as u64;
            profile.bytes += state_bytes as u64;
            let materialize_start = Instant::now();
            let result = (|| {
                cache_states_for_keys(
                    rnn,
                    &mut self.deferred_states,
                    kind,
                    kind.layers(),
                    &keys,
                    cast_f32_slice(&mapped, "resident recurrent state")?,
                )
            })();
            profile.flat_copy_ns += materialize_start.elapsed().as_nanos();
            drop(mapped);
            readback.unmap();
            result?;
            self.resident_states[module_index] = None;
            self.buffers.modules[module_index].clear_resident();
        }
        let fully_resident_chunks = self.fully_resident_sync_chunks()?;
        let use_pipelined_sync = use_flat_cpu_state
            && self.pipelined_cpu_state_sync
            && fully_resident_chunks.len() > 1
            && prepare_state_readback_ring(
                &self.context,
                &mut self.buffers.readbacks,
                &fully_resident_chunks,
                profile,
            )?;
        if use_pipelined_sync {
            let pipeline_result = synchronize_flat_state_chunks_pipelined(
                &self.context,
                &mut self.buffers.readbacks,
                fully_resident_chunks,
                &mut pending_flat_state,
                profile,
            );
            self.buffers.readbacks[1].clear();
            pipeline_result?;
        } else {
            for chunk in fully_resident_chunks {
                self.buffers.readbacks[0].ensure(&self.context, chunk.state_bytes)?;
                let readback = self.buffers.readbacks[0].cloned();
                let copy_start = Instant::now();
                let mut encoder =
                    self.context
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("rwkv-srs fully resident state synchronization"),
                        });
                encoder.copy_buffer_to_buffer(
                    &chunk.source,
                    chunk.source_offset as u64,
                    &readback,
                    0,
                    chunk.state_bytes as u64,
                );
                self.context.queue.submit([encoder.finish()]);
                profile.copy_map_submit_ns += copy_start.elapsed().as_nanos();
                let wait_start = Instant::now();
                wait_for_readback(&self.context, &readback, chunk.state_bytes)?;
                let mapped = readback
                    .get_mapped_range(0..chunk.state_bytes as u64)
                    .context("fully resident GPU state readback range is unavailable")?;
                profile.copy_map_wait_ns += wait_start.elapsed().as_nanos();
                profile.chunks += 1;
                profile.entities += chunk.keys.len() as u64;
                profile.bytes += chunk.state_bytes as u64;
                let result: Result<()> = (|| {
                    if use_flat_cpu_state {
                        let copy_start = Instant::now();
                        let values = Arc::new(copy_flat_state_values(cast_f32_slice(
                            &mapped,
                            "fully resident recurrent state",
                        )?));
                        profile.flat_copy_ns += copy_start.elapsed().as_nanos();
                        let insert_start = Instant::now();
                        append_flat_states_for_keys(
                            &mut pending_flat_state,
                            chunk.kind,
                            chunk.layers,
                            &chunk.keys,
                            values,
                        )?;
                        profile.map_insert_ns += insert_start.elapsed().as_nanos();
                        profile.flat_chunks += 1;
                        profile.flat_entities += chunk.keys.len() as u64;
                        profile.flat_bytes += chunk.state_bytes as u64;
                    } else if use_shared_cpu_state_views {
                        let values = cast_f32_slice(&mapped, "fully resident recurrent state")?;
                        apply_shared_states_for_keys(
                            rnn,
                            chunk.kind,
                            chunk.layers,
                            &chunk.keys,
                            values,
                            profile,
                        )?;
                    } else {
                        let materialize_start = Instant::now();
                        let values = cast_f32_slice(&mapped, "fully resident recurrent state")?;
                        apply_states_for_keys(rnn, chunk.kind, chunk.layers, &chunk.keys, values)?;
                        profile.backing_tensor_ns += materialize_start.elapsed().as_nanos();
                    }
                    Ok(())
                })();
                drop(mapped);
                readback.unmap();
                result?;
                if !use_flat_cpu_state && use_shared_cpu_state_views {
                    self.shared_cpu_state_view_chunks += 1;
                    self.shared_cpu_state_view_entities += chunk.keys.len() as u64;
                    self.last_shared_cpu_state_view_chunks += 1;
                    self.last_shared_cpu_state_view_entities += chunk.keys.len() as u64;
                }
            }
        }
        let deferred_start = Instant::now();
        let deferred_result = self.deferred_states.synchronize(rnn);
        profile.deferred_ns += deferred_start.elapsed().as_nanos();
        deferred_result?;
        if use_flat_cpu_state && !pending_flat_state.is_empty() {
            let insert_start = Instant::now();
            commit_flat_cpu_state(rnn, pending_flat_state);
            profile.map_insert_ns += insert_start.elapsed().as_nanos();
        }
        let destroy_start = Instant::now();
        for arena in &mut self.fully_resident_states {
            *arena = None;
        }
        profile.arena_destroy_ns += destroy_start.elapsed().as_nanos();
        Ok(())
    }
}

fn scan_chunk_reviews() -> Result<usize> {
    match std::env::var("RWKV_SRS_GPU_PROCESS_SCAN_CHUNK") {
        Ok(value) => {
            let parsed = value
                .parse::<usize>()
                .with_context(|| format!("invalid RWKV_SRS_GPU_PROCESS_SCAN_CHUNK={value:?}"))?;
            ensure!(parsed > 0, "GPU scan chunk size must be positive");
            Ok(parsed)
        }
        Err(std::env::VarError::NotPresent) => Ok(DEFAULT_CHUNK_REVIEWS),
        Err(error) => Err(error).context("could not read GPU scan chunk size"),
    }
}

fn transform_workgroup_supported(context: &GpuContext, workgroup_size: usize) -> bool {
    if workgroup_size > context.limits.max_compute_invocations_per_workgroup as usize
        || workgroup_size > context.limits.max_compute_workgroup_size_x as usize
    {
        return false;
    }
    if workgroup_size == CHANNELS {
        return true;
    }
    let lanes_per_row = workgroup_size / CHANNELS;
    context.subgroup_operations
        && context.info.subgroup_min_size as usize >= lanes_per_row
        && (context.info.subgroup_min_size as usize).is_multiple_of(lanes_per_row)
}

fn transform_workgroup_sizes(context: &GpuContext) -> Result<Vec<usize>> {
    match std::env::var("RWKV_SRS_GPU_PROCESS_TRANSFORM_WORKGROUP") {
        Ok(value) => {
            let parsed = value.parse::<usize>().with_context(|| {
                format!("invalid RWKV_SRS_GPU_PROCESS_TRANSFORM_WORKGROUP={value:?}")
            })?;
            ensure!(
                matches!(parsed, 128 | 256 | 512 | 1024),
                "GPU transform workgroup must be 128, 256, 512, or 1024"
            );
            ensure!(
                transform_workgroup_supported(context, parsed),
                "GPU transform workgroup size {parsed} is incompatible with this adapter"
            );
            Ok(vec![parsed])
        }
        Err(std::env::VarError::NotPresent) => Ok([1024usize, 512, 256, 128]
            .into_iter()
            .filter(|workgroup_size| transform_workgroup_supported(context, *workgroup_size))
            .collect()),
        Err(error) => Err(error).context("could not read GPU transform workgroup size"),
    }
}

fn scan_profile_enabled() -> bool {
    std::env::var("RWKV_SRS_GPU_PROCESS_SCAN_PROFILE")
        .map(|value| matches!(value.as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

fn shared_cpu_state_views_enabled() -> bool {
    match std::env::var(SHARED_CPU_STATE_VIEWS_ENV_VAR) {
        Ok(value) => !matches!(value.as_str(), "0" | "false" | "no" | "off"),
        Err(_) => true,
    }
}

fn flat_cpu_state_enabled() -> bool {
    match std::env::var(FLAT_CPU_STATE_ENV_VAR) {
        Ok(value) => !matches!(value.as_str(), "0" | "false" | "no" | "off"),
        Err(_) => true,
    }
}

fn pipelined_cpu_state_sync_enabled() -> bool {
    match std::env::var(PIPELINED_CPU_STATE_SYNC_ENV_VAR) {
        Ok(value) => !matches!(value.as_str(), "0" | "false" | "no" | "off"),
        Err(_) => true,
    }
}

fn add_cpu_state_sync_profile(
    dict: &Bound<'_, PyDict>,
    prefix: &str,
    profile: CpuStateSyncProfile,
) -> PyResult<()> {
    for (name, value) in [
        ("calls", u128::from(profile.calls)),
        ("chunks", u128::from(profile.chunks)),
        ("entities", u128::from(profile.entities)),
        ("bytes", u128::from(profile.bytes)),
        ("flat_chunks", u128::from(profile.flat_chunks)),
        ("flat_entities", u128::from(profile.flat_entities)),
        ("flat_bytes", u128::from(profile.flat_bytes)),
        ("pipelined_chunks", u128::from(profile.pipelined_chunks)),
        ("pipeline_fallbacks", u128::from(profile.pipeline_fallbacks)),
        (
            "pipeline_extra_buffer_bytes",
            u128::from(profile.pipeline_extra_buffer_bytes),
        ),
        ("copy_map_submit_ns", profile.copy_map_submit_ns),
        ("copy_map_wait_ns", profile.copy_map_wait_ns),
        ("pipeline_overlap_ns", profile.pipeline_overlap_ns),
        ("flat_copy_ns", profile.flat_copy_ns),
        ("field_copy_ns", profile.field_copy_ns),
        ("backing_tensor_ns", profile.backing_tensor_ns),
        ("view_ns", profile.view_ns),
        ("map_insert_ns", profile.map_insert_ns),
        ("arena_destroy_ns", profile.arena_destroy_ns),
        ("deferred_ns", profile.deferred_ns),
        ("total_ns", profile.total_ns),
    ] {
        dict.set_item(format!("{prefix}{name}"), value)?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct FusedRowBatchedProjectionSelection {
    prepare: bool,
    channel: bool,
}

impl FusedRowBatchedProjectionSelection {
    fn any(self) -> bool {
        self.prepare || self.channel
    }
}

fn fused_row_batched_projection_selection() -> FusedRowBatchedProjectionSelection {
    match std::env::var(FUSED_ROW_BATCHED_PROJECTIONS_ENV_VAR) {
        Ok(value) => fused_row_batched_projection_selection_from_value(Some(&value)),
        Err(std::env::VarError::NotPresent) => {
            fused_row_batched_projection_selection_from_value(None)
        }
        Err(_) => FusedRowBatchedProjectionSelection::default(),
    }
}

fn fused_row_batched_projection_selection_from_value(
    value: Option<&str>,
) -> FusedRowBatchedProjectionSelection {
    match value {
        Some("0" | "false" | "no" | "off") => FusedRowBatchedProjectionSelection::default(),
        Some("prepare") => FusedRowBatchedProjectionSelection {
            prepare: true,
            channel: false,
        },
        Some("channel") => FusedRowBatchedProjectionSelection {
            prepare: false,
            channel: true,
        },
        None | Some(_) => FusedRowBatchedProjectionSelection {
            prepare: true,
            channel: true,
        },
    }
}

enum FullyResidentStatePolicy {
    Auto,
    Disabled,
    Enabled,
}

fn fully_resident_state_policy() -> FullyResidentStatePolicy {
    match std::env::var(FULLY_RESIDENT_STATE_ENV_VAR) {
        Ok(value) if matches!(value.as_str(), "1" | "true" | "yes" | "on") => {
            FullyResidentStatePolicy::Enabled
        }
        Ok(value) if matches!(value.as_str(), "0" | "false" | "no" | "off") => {
            FullyResidentStatePolicy::Disabled
        }
        Ok(_) | Err(std::env::VarError::NotPresent) => FullyResidentStatePolicy::Auto,
        Err(_) => FullyResidentStatePolicy::Disabled,
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_process_readback(
    rnn: &mut NativeRnn,
    plans: &[ModulePlan],
    bytes: &[u8],
    probability_offset: usize,
    probability_bytes: usize,
    ahead_offset: usize,
    weights_offset: usize,
    curve_bytes: usize,
    mode: GpuProcessScanMode,
    state_offsets: &[usize],
    defer_cpu_state: bool,
    resident_enabled: bool,
    fully_resident_enabled: bool,
    deferred_states: &mut DeferredProcessStates,
) -> Result<GpuScanProcessOutput> {
    let prediction_probabilities = if mode.returns_predictions() {
        cast_f32_slice(
            &bytes[probability_offset..probability_offset + probability_bytes],
            "probabilities",
        )?
        .iter()
        .copied()
        .map(f64::from)
        .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let curve_ahead_logits = if mode.returns_curves() {
        let values = cast_f32_slice(
            &bytes[ahead_offset..ahead_offset + curve_bytes],
            "ahead logits",
        )?;
        Some(values.to_vec())
    } else {
        None
    };
    let curve_w = if mode.returns_curves() {
        let values = cast_f32_slice(
            &bytes[weights_offset..weights_offset + curve_bytes],
            "curve weights",
        )?;
        Some(values.to_vec())
    } else {
        None
    };

    rnn.invalidate_gpu();
    for (plan, offset) in plans.iter().zip(state_offsets) {
        if fully_resident_enabled {
            continue;
        }
        let state_len = if resident_enabled {
            plan.evicted_state_elements()
        } else {
            plan.state_elements()
        } * size_of::<f32>();
        let values = cast_f32_slice(&bytes[*offset..*offset + state_len], "recurrent state")?;
        if resident_enabled {
            cache_states_for_keys(
                rnn,
                deferred_states,
                plan.kind,
                plan.layers,
                &plan.evicted_keys,
                values,
            )?;
            for key in plan.keys.iter().copied() {
                deferred_states.remove(rnn, plan.kind, key);
            }
        } else if defer_cpu_state {
            cache_final_states(rnn, deferred_states, plan, values)?;
        } else {
            apply_final_states(rnn, plan, values)?;
        }
    }
    Ok((prediction_probabilities, curve_ahead_logits, curve_w))
}

fn cache_final_states(
    rnn: &mut NativeRnn,
    deferred_states: &mut DeferredProcessStates,
    plan: &ModulePlan,
    values: &[f32],
) -> Result<()> {
    cache_states_for_keys(
        rnn,
        deferred_states,
        plan.kind,
        plan.layers,
        &plan.keys,
        values,
    )
}

fn cache_states_for_keys(
    rnn: &mut NativeRnn,
    deferred_states: &mut DeferredProcessStates,
    kind: ModuleKind,
    layers: usize,
    keys: &[Option<i64>],
    values: &[f32],
) -> Result<()> {
    let entity_stride = layers * STATE_LAYER_ELEMENTS;
    ensure!(
        values.len() == keys.len() * entity_stride,
        "GPU scan returned malformed deferred recurrent state"
    );
    for (entity, key) in keys.iter().copied().enumerate() {
        deferred_states.insert(
            rnn,
            kind,
            key,
            &values[entity * entity_stride..][..entity_stride],
        );
    }
    Ok(())
}

fn apply_final_states(rnn: &mut NativeRnn, plan: &ModulePlan, values: &[f32]) -> Result<()> {
    apply_states_for_keys(rnn, plan.kind, plan.layers, &plan.keys, values)
}

fn append_flat_states_for_keys(
    pending: &mut FlatNativeRnnState,
    kind: ModuleKind,
    layers: usize,
    keys: &[Option<i64>],
    values: Arc<Vec<f32>>,
) -> Result<()> {
    let entity_stride = layers
        .checked_mul(STATE_LAYER_ELEMENTS)
        .context("flat CPU recurrent-state stride overflow")?;
    ensure!(
        values.len() == keys.len() * entity_stride,
        "GPU scan returned malformed flat CPU recurrent state"
    );
    for (entity, key) in keys.iter().copied().enumerate() {
        let state = FlatNativeRnnModuleState::from_shared_values(values.clone(), entity, layers)?;
        let previous = match kind {
            ModuleKind::Card => pending.card_states.insert(key.expect("card key"), state),
            ModuleKind::Deck => pending.deck_states.insert(key.expect("deck key"), state),
            ModuleKind::Note => pending.note_states.insert(key.expect("note key"), state),
            ModuleKind::Preset => pending
                .preset_states
                .insert(key.expect("preset key"), state),
            ModuleKind::Global => pending.global_state.replace(state),
        };
        ensure!(
            previous.is_none(),
            "flat CPU state key was synchronized twice"
        );
    }
    Ok(())
}

fn copy_flat_state_values(values: &[f32]) -> Vec<f32> {
    // Large libc memcpy implementations may select non-temporal operations
    // that underperform against wgpu's mapped staging memory. Preserve one
    // entity-interleaved allocation while copying in the same bounded block
    // size as the dominant recurrent-state field.
    let mut copied = Vec::with_capacity(values.len());
    for chunk in values.chunks(TRANSFORM_MATRIX_ELEMENTS) {
        copied.extend_from_slice(chunk);
    }
    copied
}

fn submit_state_readback(
    context: &GpuContext,
    readback_cache: &mut ReusableBuffer,
    chunk: FullyResidentSyncChunk,
    profile: &mut CpuStateSyncProfile,
) -> Result<PendingStateReadback> {
    readback_cache.ensure_exact(context, chunk.state_bytes)?;
    let readback = readback_cache.cloned();
    let submit_start = Instant::now();
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rwkv-srs pipelined fully resident state synchronization"),
        });
    encoder.copy_buffer_to_buffer(
        &chunk.source,
        chunk.source_offset as u64,
        &readback,
        0,
        chunk.state_bytes as u64,
    );
    context.queue.submit([encoder.finish()]);
    let (sender, receiver) = mpsc::sync_channel(1);
    readback.clone().map_async(
        wgpu::MapMode::Read,
        0..chunk.state_bytes as u64,
        move |result| {
            let _ = sender.send(result);
        },
    );
    profile.copy_map_submit_ns += submit_start.elapsed().as_nanos();
    Ok(PendingStateReadback {
        chunk,
        readback,
        receiver,
    })
}

fn prepare_state_readback_ring(
    context: &GpuContext,
    readbacks: &mut [ReusableBuffer; 2],
    chunks: &[FullyResidentSyncChunk],
    profile: &mut CpuStateSyncProfile,
) -> Result<bool> {
    let required_bytes = chunks
        .iter()
        .map(|chunk| chunk.state_bytes)
        .max()
        .unwrap_or(0);
    readbacks[0].ensure_exact(context, required_bytes)?;
    match readbacks[1].ensure_exact(context, required_bytes) {
        Ok(()) => {
            profile.pipeline_extra_buffer_bytes += required_bytes as u64;
            Ok(true)
        }
        Err(error) if is_gpu_out_of_memory(&error) => {
            readbacks[1].clear();
            profile.pipeline_fallbacks += 1;
            Ok(false)
        }
        Err(error) => Err(error),
    }
}

fn await_state_readback(
    context: &GpuContext,
    pending: &PendingStateReadback,
    profile: &mut CpuStateSyncProfile,
) -> Result<()> {
    let wait_start = Instant::now();
    let poll_result = context
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .context("pipelined GPU state readback did not complete");
    if let Err(error) = poll_result {
        profile.copy_map_wait_ns += wait_start.elapsed().as_nanos();
        return Err(error);
    }
    let map_result = pending
        .receiver
        .recv()
        .context("pipelined GPU state readback callback was not invoked")?;
    profile.copy_map_wait_ns += wait_start.elapsed().as_nanos();
    map_result.context("pipelined GPU state readback failed")?;
    Ok(())
}

fn materialize_flat_state_readback(
    pending: &PendingStateReadback,
    pending_flat_state: &mut FlatNativeRnnState,
    profile: &mut CpuStateSyncProfile,
) -> Result<()> {
    let mapped = match pending
        .readback
        .get_mapped_range(0..pending.chunk.state_bytes as u64)
    {
        Ok(mapped) => mapped,
        Err(error) => {
            pending.readback.unmap();
            return Err(error)
                .context("pipelined fully resident GPU state readback range is unavailable");
        }
    };
    profile.chunks += 1;
    profile.entities += pending.chunk.keys.len() as u64;
    profile.bytes += pending.chunk.state_bytes as u64;
    profile.pipelined_chunks += 1;
    let result = (|| {
        let copy_start = Instant::now();
        let values = Arc::new(copy_flat_state_values(cast_f32_slice(
            &mapped,
            "pipelined fully resident recurrent state",
        )?));
        profile.flat_copy_ns += copy_start.elapsed().as_nanos();
        let insert_start = Instant::now();
        append_flat_states_for_keys(
            pending_flat_state,
            pending.chunk.kind,
            pending.chunk.layers,
            &pending.chunk.keys,
            values,
        )?;
        profile.map_insert_ns += insert_start.elapsed().as_nanos();
        profile.flat_chunks += 1;
        profile.flat_entities += pending.chunk.keys.len() as u64;
        profile.flat_bytes += pending.chunk.state_bytes as u64;
        Ok(())
    })();
    drop(mapped);
    pending.readback.unmap();
    result
}

fn discard_state_readback(
    context: &GpuContext,
    pending: &PendingStateReadback,
    profile: &mut CpuStateSyncProfile,
) -> Result<()> {
    let result = await_state_readback(context, pending, profile);
    pending.readback.unmap();
    result
}

fn synchronize_flat_state_chunks_pipelined(
    context: &GpuContext,
    readbacks: &mut [ReusableBuffer; 2],
    chunks: Vec<FullyResidentSyncChunk>,
    pending_flat_state: &mut FlatNativeRnnState,
    profile: &mut CpuStateSyncProfile,
) -> Result<()> {
    let mut chunks = chunks.into_iter();
    let Some(first) = chunks.next() else {
        return Ok(());
    };
    let mut current = submit_state_readback(context, &mut readbacks[0], first, profile)?;
    let mut next_slot = 1usize;
    loop {
        if let Err(error) = await_state_readback(context, &current, profile) {
            current.readback.unmap();
            return Err(error);
        }
        let next = match chunks.next() {
            Some(chunk) => {
                match submit_state_readback(context, &mut readbacks[next_slot], chunk, profile) {
                    Ok(pending) => Some(pending),
                    Err(error) => {
                        current.readback.unmap();
                        return Err(error);
                    }
                }
            }
            None => None,
        };
        let materialize_start = Instant::now();
        let result = materialize_flat_state_readback(&current, pending_flat_state, profile);
        if next.is_some() {
            profile.pipeline_overlap_ns += materialize_start.elapsed().as_nanos();
        }
        if let Err(error) = result {
            if let Some(pending) = next.as_ref() {
                if let Err(cleanup_error) = discard_state_readback(context, pending, profile) {
                    return Err(error.context(format!(
                        "also failed to recover pipelined GPU readback: {cleanup_error:#}"
                    )));
                }
            }
            return Err(error);
        }
        let Some(pending) = next else {
            break;
        };
        current = pending;
        next_slot = 1 - next_slot;
    }
    Ok(())
}

fn commit_flat_cpu_state(rnn: &mut NativeRnn, pending: FlatNativeRnnState) {
    for (identity, state) in pending.card_states {
        rnn.card_states.remove(&identity);
        rnn.flat_cpu_state.card_states.insert(identity, state);
    }
    for (identity, state) in pending.note_states {
        rnn.note_states.remove(&identity);
        rnn.flat_cpu_state.note_states.insert(identity, state);
    }
    for (identity, state) in pending.deck_states {
        rnn.deck_states.remove(&identity);
        rnn.flat_cpu_state.deck_states.insert(identity, state);
    }
    for (identity, state) in pending.preset_states {
        rnn.preset_states.remove(&identity);
        rnn.flat_cpu_state.preset_states.insert(identity, state);
    }
    if let Some(state) = pending.global_state {
        rnn.global_state = None;
        rnn.flat_cpu_state.global_state = Some(state);
    }
}

fn apply_states_for_keys(
    rnn: &mut NativeRnn,
    kind: ModuleKind,
    layers: usize,
    keys: &[Option<i64>],
    values: &[f32],
) -> Result<()> {
    let entity_stride = layers * STATE_LAYER_ELEMENTS;
    ensure!(
        values.len() == keys.len() * entity_stride,
        "GPU scan returned malformed recurrent state"
    );
    for (entity, key) in keys.iter().copied().enumerate() {
        let state =
            module_state_from_values(&values[entity * entity_stride..][..entity_stride], layers)?;
        insert_module_state(rnn, kind, key, state);
    }
    Ok(())
}

fn apply_shared_states_for_keys(
    rnn: &mut NativeRnn,
    kind: ModuleKind,
    layers: usize,
    keys: &[Option<i64>],
    values: &[f32],
    profile: &mut CpuStateSyncProfile,
) -> Result<()> {
    let states = module_states_from_shared_values(values, layers, keys.len(), profile)?;
    let insert_start = Instant::now();
    for (key, state) in keys.iter().copied().zip(states) {
        insert_module_state(rnn, kind, key, state);
    }
    profile.map_insert_ns += insert_start.elapsed().as_nanos();
    Ok(())
}

fn insert_module_state(
    rnn: &mut NativeRnn,
    kind: ModuleKind,
    key: Option<i64>,
    state: NativeRnnModuleState,
) {
    match kind {
        ModuleKind::Card => {
            let key = key.expect("card key");
            rnn.flat_cpu_state.card_states.remove(&key);
            rnn.card_states.insert(key, state);
        }
        ModuleKind::Deck => {
            let key = key.expect("deck key");
            rnn.flat_cpu_state.deck_states.remove(&key);
            rnn.deck_states.insert(key, state);
        }
        ModuleKind::Note => {
            let key = key.expect("note key");
            rnn.flat_cpu_state.note_states.remove(&key);
            rnn.note_states.insert(key, state);
        }
        ModuleKind::Preset => {
            let key = key.expect("preset key");
            rnn.flat_cpu_state.preset_states.remove(&key);
            rnn.preset_states.insert(key, state);
        }
        ModuleKind::Global => {
            rnn.flat_cpu_state.global_state = None;
            rnn.global_state = Some(state);
        }
    }
}

fn module_states_from_shared_values(
    values: &[f32],
    layers: usize,
    entities: usize,
    profile: &mut CpuStateSyncProfile,
) -> Result<Vec<NativeRnnModuleState>> {
    let expected_values = entities
        .checked_mul(layers)
        .and_then(|count| count.checked_mul(STATE_LAYER_ELEMENTS))
        .context("shared CPU recurrent-state size overflow")?;
    ensure!(
        values.len() == expected_values,
        "shared CPU recurrent-state layout mismatch"
    );
    if entities == 0 {
        return Ok(Vec::new());
    }

    // Transpose the entity/layer-interleaved GPU arena into the canonical
    // field/layer families once per bounded readback chunk. Each family then
    // owns one batched Candle storage, while the final per-entity tensors are
    // single narrow views. Keeping field shapes in the backing tensors avoids
    // the much more expensive chain of entity/layer/field views and reshapes.
    let field_copy_start = Instant::now();
    let mut time_x_shift_values = (0..layers)
        .map(|_| Vec::with_capacity(entities * CHANNELS))
        .collect::<Vec<_>>();
    let mut time_state_values = (0..layers)
        .map(|_| Vec::with_capacity(entities * TRANSFORM_MATRIX_ELEMENTS))
        .collect::<Vec<_>>();
    let mut channel_state_values = (0..layers)
        .map(|_| Vec::with_capacity(entities * CHANNELS))
        .collect::<Vec<_>>();
    for entity in 0..entities {
        for layer in 0..layers {
            let base = (entity * layers + layer) * STATE_LAYER_ELEMENTS;
            time_x_shift_values[layer].extend_from_slice(&values[base..base + CHANNELS]);
            let recurrent_base = base + CHANNELS;
            time_state_values[layer].extend_from_slice(
                &values[recurrent_base..recurrent_base + TRANSFORM_MATRIX_ELEMENTS],
            );
            let channel_base = recurrent_base + TRANSFORM_MATRIX_ELEMENTS;
            channel_state_values[layer]
                .extend_from_slice(&values[channel_base..channel_base + CHANNELS]);
        }
    }
    profile.field_copy_ns += field_copy_start.elapsed().as_nanos();
    let backing_start = Instant::now();
    let time_x_shift_batch_by_layer = time_x_shift_values
        .into_iter()
        .map(|values| Tensor::from_vec(values, (entities, 1usize, CHANNELS), &Device::Cpu))
        .collect::<Result<Vec<_>, _>>()?;
    let time_state_batch_by_layer = time_state_values
        .into_iter()
        .map(|values| {
            Tensor::from_vec(
                values,
                (entities, 1usize, HEADS, HEAD_SIZE, HEAD_SIZE),
                &Device::Cpu,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let channel_state_batch_by_layer = channel_state_values
        .into_iter()
        .map(|values| Tensor::from_vec(values, (entities, 1usize, CHANNELS), &Device::Cpu))
        .collect::<Result<Vec<_>, _>>()?;
    profile.backing_tensor_ns += backing_start.elapsed().as_nanos();

    let view_start = Instant::now();
    let mut states = Vec::with_capacity(entities);
    for entity in 0..entities {
        let mut time_x_shift_b1c_by_layer = Vec::with_capacity(layers);
        let mut time_state_b1hkk_by_layer = Vec::with_capacity(layers);
        let mut channel_state_b1c_by_layer = Vec::with_capacity(layers);
        for tensor in &time_x_shift_batch_by_layer {
            time_x_shift_b1c_by_layer.push(tensor.narrow(0, entity, 1)?);
        }
        for tensor in &time_state_batch_by_layer {
            time_state_b1hkk_by_layer.push(tensor.narrow(0, entity, 1)?);
        }
        for tensor in &channel_state_batch_by_layer {
            channel_state_b1c_by_layer.push(tensor.narrow(0, entity, 1)?);
        }
        states.push(NativeRnnModuleState {
            time_x_shift_b1c_by_layer,
            time_state_b1hkk_by_layer,
            channel_state_b1c_by_layer,
        });
    }
    profile.view_ns += view_start.elapsed().as_nanos();
    Ok(states)
}

fn module_state_from_values(values: &[f32], layers: usize) -> Result<NativeRnnModuleState> {
    ensure!(
        values.len() == layers * STATE_LAYER_ELEMENTS,
        "GPU scan state layer stride mismatch"
    );
    let mut time_x_shift_b1c_by_layer = Vec::with_capacity(layers);
    let mut time_state_b1hkk_by_layer = Vec::with_capacity(layers);
    let mut channel_state_b1c_by_layer = Vec::with_capacity(layers);
    for layer in 0..layers {
        let base = layer * STATE_LAYER_ELEMENTS;
        time_x_shift_b1c_by_layer.push(Tensor::from_vec(
            values[base..base + CHANNELS].to_vec(),
            (1usize, 1usize, CHANNELS),
            &Device::Cpu,
        )?);
        let state_base = base + CHANNELS;
        time_state_b1hkk_by_layer.push(Tensor::from_vec(
            values[state_base..state_base + TRANSFORM_MATRIX_ELEMENTS].to_vec(),
            (1usize, 1usize, HEADS, HEAD_SIZE, HEAD_SIZE),
            &Device::Cpu,
        )?);
        let channel_base = state_base + TRANSFORM_MATRIX_ELEMENTS;
        channel_state_b1c_by_layer.push(Tensor::from_vec(
            values[channel_base..channel_base + CHANNELS].to_vec(),
            (1usize, 1usize, CHANNELS),
            &Device::Cpu,
        )?);
    }
    Ok(NativeRnnModuleState {
        time_x_shift_b1c_by_layer,
        time_state_b1hkk_by_layer,
        channel_state_b1c_by_layer,
    })
}

fn uniform_buffer<T: bytemuck::Pod>(
    context: &GpuContext,
    label: &'static str,
    value: &T,
) -> Result<wgpu::Buffer> {
    context.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(value),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

fn dispatch_bind_group(
    context: &GpuContext,
    layout: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rwkv-srs scan dispatch parameters"),
            layout,
            entries: &[binding(0, buffer)],
        })
}

fn arena_transfer_bind_group(
    context: &GpuContext,
    layout: &wgpu::BindGroupLayout,
    source: &wgpu::Buffer,
    target: &wgpu::Buffer,
    transfers: &wgpu::Buffer,
    params: &wgpu::Buffer,
) -> wgpu::BindGroup {
    context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rwkv-srs fully resident state transfer"),
            layout,
            entries: &[
                binding(0, source),
                binding(1, target),
                binding(2, transfers),
                binding(3, params),
            ],
        })
}

fn binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_profiled(
    encoder: &mut wgpu::CommandEncoder,
    label: &'static str,
    pipeline: &wgpu::ComputePipeline,
    bind_groups: &[(&wgpu::BindGroup, u32)],
    workgroups: u32,
    profile: &mut Option<GpuTimestampProfile>,
    stage: GpuTimestampStage,
) {
    if workgroups == 0 {
        return;
    }
    let pair = profile.as_mut().map(|profile| profile.pair(stage));
    let timestamp_writes = profile.as_ref().zip(pair).map(
        |(profile, (beginning_of_pass_write_index, end_of_pass_write_index))| {
            wgpu::ComputePassTimestampWrites {
                query_set: &profile.query_set,
                beginning_of_pass_write_index: Some(beginning_of_pass_write_index),
                end_of_pass_write_index: Some(end_of_pass_write_index),
            }
        },
    );
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes,
    });
    pass.set_pipeline(pipeline);
    for (bind_group, index) in bind_groups {
        pass.set_bind_group(*index, *bind_group, &[]);
    }
    pass.dispatch_workgroups(workgroups, 1, 1);
}

#[allow(clippy::too_many_arguments)]
fn dispatch_profiled_2d(
    encoder: &mut wgpu::CommandEncoder,
    label: &'static str,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    workgroups_x: u32,
    workgroups_y: u32,
    profile: &mut Option<GpuTimestampProfile>,
    stage: GpuTimestampStage,
) {
    if workgroups_x == 0 || workgroups_y == 0 {
        return;
    }
    let pair = profile.as_mut().map(|profile| profile.pair(stage));
    let timestamp_writes = profile.as_ref().zip(pair).map(
        |(profile, (beginning_of_pass_write_index, end_of_pass_write_index))| {
            wgpu::ComputePassTimestampWrites {
                query_set: &profile.query_set,
                beginning_of_pass_write_index: Some(beginning_of_pass_write_index),
                end_of_pass_write_index: Some(end_of_pass_write_index),
            }
        },
    );
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}

fn wait_for_readback(
    context: &GpuContext,
    buffer: &wgpu::Buffer,
    readback_size: usize,
) -> Result<()> {
    let (sender, receiver) = mpsc::sync_channel(1);
    buffer.clone().map_async(
        wgpu::MapMode::Read,
        0..readback_size as u64,
        move |result| {
            let _ = sender.send(result);
        },
    );
    context
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .context("GPU associative scan readback did not complete")?;
    receiver
        .recv()
        .context("GPU associative scan readback callback was not invoked")?
        .context("GPU associative scan readback failed")?;
    Ok(())
}

fn cast_f32_slice<'a>(bytes: &'a [u8], name: &str) -> Result<&'a [f32]> {
    bytemuck::try_cast_slice(bytes)
        .map_err(|error| anyhow!("malformed GPU scan {name} buffer: {error:?}"))
}

fn align_up(value: usize, alignment: usize) -> usize {
    value.next_multiple_of(alignment)
}

fn align_down(value: usize, alignment: usize) -> usize {
    value - value % alignment
}

fn process_readback_bytes(
    plans: &[ModulePlan],
    reviews: usize,
    mode: GpuProcessScanMode,
    resident_enabled: bool,
    fully_resident_enabled: bool,
) -> Result<usize> {
    let output_bytes_per_review = match mode {
        GpuProcessScanMode::Curves => MAX_OUTPUT_READBACK_BYTES_PER_REVIEW,
        GpuProcessScanMode::Predictions => size_of::<f32>(),
        GpuProcessScanMode::StateOnly => 0,
    };
    let mut bytes = reviews
        .checked_mul(output_bytes_per_review)
        .context("GPU process output readback size overflow")?;
    for plan in plans {
        bytes = align_up(bytes, 4);
        let state_elements = if fully_resident_enabled {
            0
        } else if resident_enabled {
            plan.evicted_state_elements()
        } else {
            plan.state_elements()
        };
        bytes = bytes
            .checked_add(
                state_elements
                    .checked_mul(size_of::<f32>())
                    .context("GPU process state readback size overflow")?,
            )
            .context("GPU process readback size overflow")?;
    }
    Ok(bytes)
}

#[derive(Clone, Copy)]
struct ProcessResourceLimits {
    max_storage_buffer_bytes: usize,
    max_buffer_bytes: usize,
    max_workgroups: usize,
}

impl ProcessResourceLimits {
    fn from_context(context: &GpuContext) -> Result<Self> {
        let advertised_storage = context
            .limits
            .max_buffer_size
            .min(context.limits.max_storage_buffer_binding_size)
            as usize;
        let advertised_buffer = context.limits.max_buffer_size as usize;
        let configured = match std::env::var(PROCESS_STATE_BUFFER_LIMIT_ENV_VAR) {
            Ok(value) => {
                let parsed = value.parse::<usize>().with_context(|| {
                    format!("invalid {PROCESS_STATE_BUFFER_LIMIT_ENV_VAR}={value:?}")
                })?;
                ensure!(
                    parsed > 0,
                    "{PROCESS_STATE_BUFFER_LIMIT_ENV_VAR} must be positive"
                );
                Some(parsed)
            }
            Err(std::env::VarError::NotPresent) => None,
            Err(error) => {
                return Err(error).context(format!(
                    "could not read {PROCESS_STATE_BUFFER_LIMIT_ENV_VAR}"
                ));
            }
        };
        Ok(Self {
            max_storage_buffer_bytes: configured
                .map_or(advertised_storage, |limit| limit.min(advertised_storage)),
            max_buffer_bytes: advertised_buffer,
            max_workgroups: context.limits.max_compute_workgroups_per_dimension as usize,
        })
    }
}

fn max_safe_process_prefix(
    ids: &[ReviewIds],
    limits: ProcessResourceLimits,
    chunk_reviews: usize,
    mode: GpuProcessScanMode,
) -> usize {
    debug_assert!(chunk_reviews > 0);
    let state_layer_bytes = STATE_LAYER_ELEMENTS * size_of::<f32>();
    if 4 * state_layer_bytes > limits.max_storage_buffer_bytes {
        return 0;
    }
    let transform_bytes = TRANSFORM_ELEMENTS * size_of::<f32>();
    let max_transforms = limits.max_storage_buffer_bytes / transform_bytes;
    let rows_per_review = mode.rows_per_review();
    let mut max_reviews = ids.len().min(limits.max_workgroups / rows_per_review);
    for bytes_per_review in [
        rows_per_review * CHANNELS * size_of::<f32>(),
        rows_per_review * FEATURE_DIM * size_of::<f32>(),
        CHANNELS * size_of::<f32>(),
    ] {
        max_reviews = max_reviews.min(limits.max_storage_buffer_bytes / bytes_per_review);
    }

    let mut identities: [HashSet<i64>; 4] = std::array::from_fn(|_| HashSet::new());
    let mut stream_reviews: [HashMap<i64, usize>; 4] = std::array::from_fn(|_| HashMap::new());
    let mut module_transforms = [0usize; 4];
    let module_layers = [3usize, 4, 2, 3];
    let mut combined_state_bytes = 4 * state_layer_bytes;
    let output_bytes_per_review = match mode {
        GpuProcessScanMode::Curves => MAX_OUTPUT_READBACK_BYTES_PER_REVIEW,
        GpuProcessScanMode::Predictions => size_of::<f32>(),
        GpuProcessScanMode::StateOnly => 0,
    };
    for (index, ids) in ids.iter().copied().take(max_reviews).enumerate() {
        let keys = [ids.0, ids.2, ids.1, ids.3];
        let mut new_state_bytes = 0usize;
        let mut next_module_transforms = module_transforms;
        for module in 0..identities.len() {
            let previous_reviews = stream_reviews[module]
                .get(&keys[module])
                .copied()
                .unwrap_or(0);
            let next_reviews = previous_reviews + 1;
            next_module_transforms[module] += next_reviews.saturating_sub(1) / chunk_reviews
                - previous_reviews.saturating_sub(1) / chunk_reviews;
            if next_module_transforms[module] > max_transforms {
                return index;
            }
            if identities[module].contains(&keys[module]) {
                continue;
            }
            let next_count = identities[module].len() + 1;
            let Some(required_bytes) = next_count
                .checked_mul(module_layers[module])
                .and_then(|count| count.checked_mul(state_layer_bytes))
            else {
                return index;
            };
            if required_bytes > limits.max_storage_buffer_bytes {
                return index;
            }
            let Some(module_bytes) = module_layers[module].checked_mul(state_layer_bytes) else {
                return index;
            };
            let Some(updated_bytes) = new_state_bytes.checked_add(module_bytes) else {
                return index;
            };
            new_state_bytes = updated_bytes;
        }
        let Some(next_state_bytes) = combined_state_bytes.checked_add(new_state_bytes) else {
            return index;
        };
        let Some(output_bytes) = (index + 1).checked_mul(output_bytes_per_review) else {
            return index;
        };
        if next_state_bytes
            .checked_add(output_bytes)
            .is_none_or(|readback_bytes| readback_bytes > limits.max_buffer_bytes)
        {
            return index;
        }
        let global_transforms = index / chunk_reviews;
        if global_transforms > max_transforms {
            return index;
        }
        for module in 0..identities.len() {
            identities[module].insert(keys[module]);
            *stream_reviews[module].entry(keys[module]).or_default() += 1;
        }
        module_transforms = next_module_transforms;
        combined_state_bytes = next_state_bytes;
    }
    max_reviews
}

fn prepare_process_feature_candidate(
    deterministic: &FeatureState,
    inputs: &[ReviewInput],
    mode: GpuProcessScanMode,
) -> PyResult<(FeatureState, Vec<ReviewIds>, Vec<f32>, u128, u128)> {
    let clone_start = Instant::now();
    let mut candidate = deterministic.clone();
    let clone_ns = clone_start.elapsed().as_nanos();
    let feature_start = Instant::now();
    let mut ids = Vec::with_capacity(inputs.len());
    let feature_values = inputs
        .len()
        .checked_mul(mode.rows_per_review() * FEATURE_DIM)
        .ok_or_else(|| py_value_error("GPU process feature capacity overflow"))?;
    let mut features = Vec::with_capacity(feature_values);
    for input in inputs {
        ids.push(if mode == GpuProcessScanMode::StateOnly {
            candidate
                .append_process_feature_only(input, &mut features)
                .map_err(py_value_error)?
        } else {
            feature_prepass_step_into(&mut candidate, input, &mut features)?
        });
    }
    Ok((
        candidate,
        ids,
        features,
        clone_ns,
        feature_start.elapsed().as_nanos(),
    ))
}

fn append_process_output(
    target: &mut GpuScanProcessOutput,
    source: GpuScanProcessOutput,
    mode: GpuProcessScanMode,
) -> PyResult<()> {
    target.0.extend(source.0);
    if mode.returns_curves() {
        target.1.as_mut().expect("curve output initialized").extend(
            source
                .1
                .ok_or_else(|| py_value_error("GPU process omitted requested ahead logits"))?,
        );
        target.2.as_mut().expect("curve output initialized").extend(
            source
                .2
                .ok_or_else(|| py_value_error("GPU process omitted requested curve weights"))?,
        );
    }
    Ok(())
}

impl NativeRnn {
    pub(super) fn ensure_gpu_process_scan(&mut self) -> Result<()> {
        self.invalidate_gpu();
        if self.gpu_process_scan.is_none() {
            self.gpu_process_scan = Some(GpuProcessScan::new(self)?);
        }
        Ok(())
    }

    pub(super) fn gpu_process_profile_pydict(
        &self,
        py: Python<'_>,
    ) -> PyResult<Option<Py<PyDict>>> {
        self.gpu_process_scan
            .as_ref()
            .map(|processor| processor.to_pydict(py))
            .transpose()
    }

    pub(super) fn release_gpu_process_cache(&mut self) -> bool {
        self.gpu_process_scan.take().is_some()
    }
}

pub(super) fn process_reviews_gpu_scan_with_state(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    payload: &[u8],
    return_curves: bool,
    num_threads: Option<usize>,
    defer_cpu_state: bool,
    committed_rows: &mut usize,
    output: &mut GpuScanProcessOutput,
) -> PyResult<()> {
    let mode = GpuProcessScanMode::process(return_curves);
    *committed_rows = 0;
    *output = (
        Vec::new(),
        mode.returns_curves().then(Vec::new),
        mode.returns_curves().then(Vec::new),
    );
    let total_start = Instant::now();
    rnn.invalidate_gpu();
    validate_num_threads(num_threads)?;
    let parse_start = Instant::now();
    let inputs = parse_process_review_payload(payload).map_err(py_value_error)?;
    let parse_ns = parse_start.elapsed().as_nanos();
    process_review_inputs_gpu_scan_with_state_impl(
        rnn,
        deterministic,
        &inputs,
        mode,
        defer_cpu_state,
        false,
        committed_rows,
        output,
        total_start,
        parse_ns,
    )
}

pub(super) fn process_review_inputs_gpu_scan_with_state(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    inputs: &[ReviewInput],
    return_curves: bool,
    num_threads: Option<usize>,
    defer_cpu_state: bool,
    fully_resident_state: bool,
    committed_rows: &mut usize,
    output: &mut GpuScanProcessOutput,
) -> PyResult<()> {
    let mode = GpuProcessScanMode::process(return_curves);
    *committed_rows = 0;
    *output = (
        Vec::new(),
        mode.returns_curves().then(Vec::new),
        mode.returns_curves().then(Vec::new),
    );
    let total_start = Instant::now();
    rnn.invalidate_gpu();
    validate_num_threads(num_threads)?;
    process_review_inputs_gpu_scan_with_state_impl(
        rnn,
        deterministic,
        inputs,
        mode,
        defer_cpu_state,
        fully_resident_state,
        committed_rows,
        output,
        total_start,
        0,
    )
}

pub(super) fn build_state_only_review_inputs_gpu_scan_with_state(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    inputs: &[ReviewInput],
    num_threads: Option<usize>,
    defer_cpu_state: bool,
    fully_resident_state: bool,
    committed_rows: &mut usize,
) -> PyResult<()> {
    *committed_rows = 0;
    let mut output = (Vec::new(), None, None);
    let total_start = Instant::now();
    rnn.invalidate_gpu();
    validate_num_threads(num_threads)?;
    process_review_inputs_gpu_scan_with_state_impl(
        rnn,
        deterministic,
        inputs,
        GpuProcessScanMode::StateOnly,
        defer_cpu_state,
        fully_resident_state,
        committed_rows,
        &mut output,
        total_start,
        0,
    )?;
    if !output.0.is_empty() || output.1.is_some() || output.2.is_some() {
        return Err(py_value_error(
            "GPU state-only scan unexpectedly materialized prediction output",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn process_review_inputs_gpu_scan_with_state_impl(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    inputs: &[ReviewInput],
    mode: GpuProcessScanMode,
    defer_cpu_state: bool,
    fully_resident_state: bool,
    committed_rows: &mut usize,
    output: &mut GpuScanProcessOutput,
    total_start: Instant,
    parse_ns: u128,
) -> PyResult<()> {
    if inputs.is_empty() {
        return Ok(());
    }
    let init_start = Instant::now();
    let mut processor = match rnn.gpu_process_scan.take() {
        Some(processor) => processor,
        None => GpuProcessScan::new(rnn).map_err(py_gpu_unavailable_error)?,
    };
    let init_ns = init_start.elapsed().as_nanos();
    processor.process_calls += 1;
    let fully_resident_requested =
        processor.should_use_fully_resident_state(defer_cpu_state, fully_resident_state);
    let mut fully_resident_active = fully_resident_requested;
    if mode == GpuProcessScanMode::StateOnly {
        processor.state_only_calls += 1;
    }
    if fully_resident_requested {
        processor.fully_resident_calls += 1;
    }
    processor.last_requested_reviews = inputs.len() as u64;
    processor.last_process_batches = 0;
    processor.last_mode = mode;
    processor.last_fully_resident = fully_resident_requested;
    processor.last_fully_resident_commit_ns = 0;
    processor.last_feature_state_clone_ns = 0;
    processor.last_feature_prepare_ns = 0;
    let normalized_ids = inputs
        .iter()
        .map(FeatureState::normalized_review_ids)
        .collect::<Vec<_>>();
    let feature_start = Instant::now();
    let process_start = Instant::now();
    let result = (|| -> PyResult<()> {
        let resource_limits =
            ProcessResourceLimits::from_context(&processor.context).map_err(py_value_error)?;
        let chunk_reviews = scan_chunk_reviews().map_err(py_value_error)?;
        output.0.reserve(inputs.len());
        let mut offset = 0usize;
        while offset < inputs.len() {
            let remaining_ids = &normalized_ids[offset..];
            let safe_prefix =
                max_safe_process_prefix(remaining_ids, resource_limits, chunk_reviews, mode);
            if safe_prefix == 0 {
                return Err(py_gpu_unavailable(
                    "one GPU process review exceeds this adapter's buffer or dispatch limits",
                ));
            }
            let mut attempt = safe_prefix.min(remaining_ids.len());
            let mut final_cleanup_attempted = false;
            if attempt < remaining_ids.len() {
                processor.adaptive_splits += 1;
            }
            loop {
                let end = offset + attempt;
                let (candidate, ids, features, clone_ns, prepare_ns) =
                    prepare_process_feature_candidate(deterministic, &inputs[offset..end], mode)?;
                processor.feature_state_clone_ns += clone_ns;
                processor.feature_prepare_ns += prepare_ns;
                processor.last_feature_state_clone_ns += clone_ns;
                processor.last_feature_prepare_ns += prepare_ns;
                match processor.process(
                    rnn,
                    ids,
                    features,
                    mode,
                    defer_cpu_state,
                    fully_resident_active,
                ) {
                    Ok(batch_output) => {
                        *deterministic = candidate;
                        offset = end;
                        *committed_rows = offset;
                        append_process_output(output, batch_output, mode)?;
                        processor.process_batches += 1;
                        if mode == GpuProcessScanMode::StateOnly {
                            processor.state_only_batches += 1;
                        }
                        if fully_resident_active {
                            processor.fully_resident_batches += 1;
                        }
                        processor.last_process_batches += 1;
                        break;
                    }
                    Err(error) if is_gpu_out_of_memory(&error) && fully_resident_active => {
                        processor.oom_retries += 1;
                        processor.resident_oom_recoveries += 1;
                        processor
                            .buffers
                            .clear_transient_preserving_resident(&processor.context)
                            .map_err(py_gpu_error)?;
                        processor.synchronize_cpu_state(rnn).map_err(py_gpu_error)?;
                        processor.buffers = ProcessBufferCache::new();
                        processor.fully_resident_disabled_after_oom = true;
                        processor.last_fully_resident = false;
                        fully_resident_active = false;
                    }
                    Err(error) if is_gpu_out_of_memory(&error) && attempt > 1 => {
                        processor.oom_retries += 1;
                        processor.adaptive_splits += 1;
                        processor
                            .buffers
                            .clear_transient_preserving_resident(&processor.context)
                            .map_err(py_gpu_error)?;
                        attempt = (attempt / 2).max(1);
                    }
                    Err(error) if is_gpu_out_of_memory(&error) && !final_cleanup_attempted => {
                        processor.oom_retries += 1;
                        final_cleanup_attempted = true;
                        processor
                            .buffers
                            .clear_transient_preserving_resident(&processor.context)
                            .map_err(py_gpu_error)?;
                        if processor.resident_states.iter().any(Option::is_some)
                            || processor.fully_resident_states.iter().any(Option::is_some)
                        {
                            processor.resident_oom_recoveries += 1;
                            processor.synchronize_cpu_state(rnn).map_err(py_gpu_error)?;
                            processor.buffers = ProcessBufferCache::new();
                        }
                        processor
                            .context
                            .device
                            .poll(wgpu::PollType::wait_indefinitely())
                            .context("GPU resident-state OOM recovery did not complete")
                            .map_err(py_gpu_error)?;
                    }
                    Err(error) => return Err(py_gpu_error(error)),
                }
            }
        }
        Ok(())
    })();
    let feature_ns = feature_start.elapsed().as_nanos();
    let process_ns = process_start.elapsed().as_nanos();
    rnn.gpu_process_scan = Some(processor);
    if scan_profile_enabled() {
        eprintln!(
            "gpu_process_scan_host rows={} parse_ns={parse_ns} feature_ns={feature_ns} init_ns={init_ns} process_ns={process_ns} total_ns={} mode={} fully_resident={fully_resident_requested}",
            inputs.len(),
            total_start.elapsed().as_nanos(),
            mode.label(),
        );
    }
    result
}

pub(super) fn synchronize_gpu_process_state_with_state(rnn: &mut NativeRnn) -> Result<u128> {
    let Some(mut processor) = rnn.gpu_process_scan.take() else {
        return Ok(0);
    };
    let synchronized_entities = processor.deferred_states.len()
        + processor
            .resident_states
            .iter()
            .flatten()
            .map(|resident| resident.keys.len())
            .sum::<usize>()
        + processor
            .fully_resident_states
            .iter()
            .flatten()
            .map(FullyResidentModuleState::len)
            .sum::<usize>();
    let start = Instant::now();
    let result = processor.synchronize_cpu_state(rnn);
    let elapsed = start.elapsed().as_nanos();
    let profile = processor.last_cpu_state_sync_profile;
    rnn.gpu_process_scan = Some(processor);
    result?;
    if scan_profile_enabled() {
        eprintln!(
            "gpu_process_scan_sync entities={synchronized_entities} chunks={} bytes={} flat_chunks={} flat_entities={} flat_bytes={} pipelined_chunks={} pipeline_fallbacks={} pipeline_extra_buffer_bytes={} copy_map_submit_ns={} copy_map_wait_ns={} pipeline_overlap_ns={} flat_copy_ns={} field_copy_ns={} backing_tensor_ns={} view_ns={} map_insert_ns={} arena_destroy_ns={} deferred_ns={} total_ns={elapsed}",
            profile.chunks,
            profile.bytes,
            profile.flat_chunks,
            profile.flat_entities,
            profile.flat_bytes,
            profile.pipelined_chunks,
            profile.pipeline_fallbacks,
            profile.pipeline_extra_buffer_bytes,
            profile.copy_map_submit_ns,
            profile.copy_map_wait_ns,
            profile.pipeline_overlap_ns,
            profile.flat_copy_ns,
            profile.field_copy_ns,
            profile.backing_tensor_ns,
            profile.view_ns,
            profile.map_insert_ns,
            profile.arena_destroy_ns,
            profile.deferred_ns,
        );
    }
    Ok(elapsed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::MaybeId;

    fn ids(index: i64) -> ReviewIds {
        (index, index + 10_000, index + 20_000, index + 30_000)
    }

    #[test]
    fn shared_cpu_state_views_preserve_flat_gpu_layout() -> Result<()> {
        let layers = 2;
        let entities = 3;
        let values = (0..entities * layers * STATE_LAYER_ELEMENTS)
            .map(|index| index as f32)
            .collect::<Vec<_>>();
        let mut profile = CpuStateSyncProfile::default();
        let states = module_states_from_shared_values(&values, layers, entities, &mut profile)?;
        assert_eq!(states.len(), entities);

        for (entity, state) in states.iter().enumerate() {
            for layer in 0..layers {
                let base = (entity * layers + layer) * STATE_LAYER_ELEMENTS;
                assert_eq!(
                    state.time_x_shift_b1c_by_layer[layer]
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                    values[base..base + CHANNELS]
                );
                let recurrent_base = base + CHANNELS;
                assert_eq!(
                    state.time_state_b1hkk_by_layer[layer]
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                    values[recurrent_base..recurrent_base + TRANSFORM_MATRIX_ELEMENTS]
                );
                let channel_base = recurrent_base + TRANSFORM_MATRIX_ELEMENTS;
                assert_eq!(
                    state.channel_state_b1c_by_layer[layer]
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                    values[channel_base..channel_base + CHANNELS]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn process_prefix_respects_identity_state_binding_limits() {
        let two_card_states = 2 * 3 * STATE_LAYER_ELEMENTS * size_of::<f32>();
        let limits = ProcessResourceLimits {
            max_storage_buffer_bytes: two_card_states,
            max_buffer_bytes: usize::MAX,
            max_workgroups: usize::MAX,
        };
        let distinct_cards = [(1, 101, 7, 8), (2, 102, 7, 8), (3, 103, 7, 8)];
        assert_eq!(
            max_safe_process_prefix(
                &distinct_cards,
                limits,
                DEFAULT_CHUNK_REVIEWS,
                GpuProcessScanMode::Curves,
            ),
            2
        );

        let repeated = vec![(1, 101, 7, 8); 20];
        assert_eq!(
            max_safe_process_prefix(
                &repeated,
                limits,
                DEFAULT_CHUNK_REVIEWS,
                GpuProcessScanMode::Curves,
            ),
            repeated.len()
        );
    }

    #[test]
    fn process_prefix_respects_interleaved_workgroup_limit() {
        let limits = ProcessResourceLimits {
            max_storage_buffer_bytes: usize::MAX,
            max_buffer_bytes: usize::MAX,
            max_workgroups: 7,
        };
        let rows = (0..10).map(ids).collect::<Vec<_>>();
        assert_eq!(
            max_safe_process_prefix(
                &rows,
                limits,
                DEFAULT_CHUNK_REVIEWS,
                GpuProcessScanMode::Curves,
            ),
            3
        );
        assert_eq!(
            max_safe_process_prefix(
                &rows,
                limits,
                DEFAULT_CHUNK_REVIEWS,
                GpuProcessScanMode::StateOnly,
            ),
            7
        );
    }

    #[test]
    fn process_prefix_respects_combined_state_and_output_readback_limit() {
        let one_review_state_bytes = (4 + 3 + 4 + 2 + 3) * STATE_LAYER_ELEMENTS * size_of::<f32>();
        let limits = ProcessResourceLimits {
            max_storage_buffer_bytes: usize::MAX,
            max_buffer_bytes: one_review_state_bytes + MAX_OUTPUT_READBACK_BYTES_PER_REVIEW,
            max_workgroups: usize::MAX,
        };
        assert_eq!(
            max_safe_process_prefix(
                &[ids(1), ids(2)],
                limits,
                DEFAULT_CHUNK_REVIEWS,
                GpuProcessScanMode::Curves,
            ),
            1
        );
    }

    #[test]
    fn process_prefix_rejects_an_impossible_single_global_state() {
        let limits = ProcessResourceLimits {
            max_storage_buffer_bytes: 4 * STATE_LAYER_ELEMENTS * size_of::<f32>() - 1,
            max_buffer_bytes: usize::MAX,
            max_workgroups: usize::MAX,
        };
        assert_eq!(
            max_safe_process_prefix(
                &[ids(1)],
                limits,
                DEFAULT_CHUNK_REVIEWS,
                GpuProcessScanMode::Curves,
            ),
            0
        );
    }

    #[test]
    fn process_prefix_respects_transform_binding_limit_for_small_scan_chunks() {
        let transform_bytes = TRANSFORM_ELEMENTS * size_of::<f32>();
        let limits = ProcessResourceLimits {
            max_storage_buffer_bytes: 4 * transform_bytes,
            max_buffer_bytes: usize::MAX,
            max_workgroups: usize::MAX,
        };
        let repeated = vec![(1, 101, 7, 8); 20];

        assert_eq!(
            max_safe_process_prefix(&repeated, limits, 1, GpuProcessScanMode::Curves),
            5
        );
        assert_eq!(
            max_safe_process_prefix(&repeated, limits, 2, GpuProcessScanMode::Curves),
            10
        );
    }

    #[test]
    fn process_feature_candidate_leaves_original_state_untouched_until_commit() {
        let original = FeatureState::with_torch_seed(12_345);
        let input = ReviewInput {
            review_id: 1,
            card_id: 2,
            note_id: MaybeId::Present(3),
            deck_id: MaybeId::Present(4),
            preset_id: MaybeId::Present(5),
            day_offset: 0.0,
            elapsed_days: 0.0,
            elapsed_seconds: 0.0,
            rating: Some(3),
            duration: Some(1_000.0),
            state: Some(0.0),
        };

        let (candidate, candidate_ids, features, _, _) = prepare_process_feature_candidate(
            &original,
            std::slice::from_ref(&input),
            GpuProcessScanMode::Predictions,
        )
        .unwrap();
        assert_eq!(original.i, 0);
        assert_eq!(original.card_count, 0);
        assert!(original.card_set.is_empty());
        assert_eq!(candidate.i, 1);
        assert_eq!(candidate.card_count, 1);
        assert!(candidate.card_set.contains(&2));
        assert_eq!(candidate_ids, vec![(2, 3, 4, 5)]);
        assert_eq!(features.len(), 2 * FEATURE_DIM);

        let (_, _, state_only_features, _, _) = prepare_process_feature_candidate(
            &original,
            std::slice::from_ref(&input),
            GpuProcessScanMode::StateOnly,
        )
        .unwrap();
        assert_eq!(state_only_features, features[FEATURE_DIM..]);
    }

    #[test]
    fn buffer_capacity_limit_rounds_down_without_exceeding_adapter_limit() {
        assert_eq!(align_down(u32::MAX as usize, 4), 4_294_967_292);
        assert_eq!(align_down(2_147_483_648, 4), 2_147_483_648);
    }

    #[test]
    fn fused_row_batched_projection_selector_defaults_on_and_keeps_diagnostics() {
        let both = FusedRowBatchedProjectionSelection {
            prepare: true,
            channel: true,
        };
        assert_eq!(
            fused_row_batched_projection_selection_from_value(None),
            both
        );
        assert_eq!(
            fused_row_batched_projection_selection_from_value(Some("1")),
            both
        );
        assert_eq!(
            fused_row_batched_projection_selection_from_value(Some("0")),
            FusedRowBatchedProjectionSelection::default()
        );
        assert_eq!(
            fused_row_batched_projection_selection_from_value(Some("prepare")),
            FusedRowBatchedProjectionSelection {
                prepare: true,
                channel: false,
            }
        );
        assert_eq!(
            fused_row_batched_projection_selection_from_value(Some("channel")),
            FusedRowBatchedProjectionSelection {
                prepare: false,
                channel: true,
            }
        );
    }
}
