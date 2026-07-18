use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::sync::mpsc;
use std::time::Instant;

use anyhow::{anyhow, bail, ensure, Context, Result};
use half::f16;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::state::{FlatNativeRnnModuleState, NativeRnnModuleState, ReviewIds};
use super::NativeRnn;
use crate::gpu::{GpuContext, GpuOperation};
use crate::model_weights::{
    LayerNormWeights, LinearWeights, Rwkv7RnnLayerWeights, SrsRwkvRnnWeights,
};
use crate::ops::f32_tensor_data;

const INVALID_OFFSET: u32 = u32::MAX;
const CHANNELS: usize = 128;
const FEATURE_DIM: usize = 92;
const STATE_LAYER_ELEMENTS: usize = 128 + 4 * 32 * 32 + 128;
const INITIAL_UPLOAD_CHUNK_BYTES: usize = 16 * 1024 * 1024;
const EXTRA_VALUES_PER_ROW: usize = 448;
const AMD_VENDOR_ID: u32 = 0x1002;
const AMD_LEGACY_LANE_ROWS: std::ops::RangeInclusive<usize> = 1_792..=2_112;
const GPU_PRECISION_ENV_VAR: &str = "RWKV_SRS_GPU_PRECISION";
const GPU_PREDICTION_KERNEL_ENV_VAR: &str = "RWKV_SRS_GPU_PREDICTION_KERNEL";
const GPU_PREDICTION_PIPELINE_ENV_VAR: &str = "RWKV_SRS_GPU_PREDICTION_PIPELINE";
const SHARED_ROWS_MIN_ROWS: usize = 2_048;
// The fused shader aliases scratch values whose lifetimes do not overlap. The
// physical layout is documented beside the WGSL declarations.
const WORKGROUP_BYTES_PER_ROW: usize =
    (2 * CHANNELS + 2 * 512 + EXTRA_VALUES_PER_ROW) * std::mem::size_of::<f32>();

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GpuPrecision {
    Mixed,
    StateF16FullF32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PredictionKernel {
    SingleRow,
    SharedRows2,
}

impl PredictionKernel {
    fn name(self) -> &'static str {
        match self {
            Self::SingleRow => "single-row",
            Self::SharedRows2 => "shared-rows-2",
        }
    }

    fn from_env() -> Result<Option<Self>> {
        match std::env::var(GPU_PREDICTION_KERNEL_ENV_VAR) {
            Err(std::env::VarError::NotPresent) => Ok(None),
            Ok(value) if value.is_empty() || value == "default" => Ok(None),
            Ok(value) if value == "single-row" => Ok(Some(Self::SingleRow)),
            Ok(value) if value == "shared-rows-2" => Ok(Some(Self::SharedRows2)),
            Ok(value) => bail!(
                "{GPU_PREDICTION_KERNEL_ENV_VAR} must be default, single-row, or shared-rows-2; got {value:?}"
            ),
            Err(error) => {
                Err(error).context(format!("could not read {GPU_PREDICTION_KERNEL_ENV_VAR}"))
            }
        }
    }
}

fn select_prediction_kernel(
    default: PredictionKernel,
    configured: Option<PredictionKernel>,
    shape_override: bool,
    row_count: usize,
) -> PredictionKernel {
    configured.unwrap_or_else(|| {
        if shape_override
            || (default == PredictionKernel::SharedRows2 && row_count < SHARED_ROWS_MIN_ROWS)
        {
            PredictionKernel::SingleRow
        } else {
            default
        }
    })
}

pub(super) fn gpu_prediction_pipeline_enabled() -> Result<bool> {
    match std::env::var(GPU_PREDICTION_PIPELINE_ENV_VAR) {
        Err(std::env::VarError::NotPresent) => Ok(true),
        Ok(value) if value.is_empty() || value == "default" || value == "1" => Ok(true),
        Ok(value) if value == "0" => Ok(false),
        Ok(value) => {
            bail!("{GPU_PREDICTION_PIPELINE_ENV_VAR} must be default, 0, or 1; got {value:?}")
        }
        Err(error) => {
            Err(error).context(format!("could not read {GPU_PREDICTION_PIPELINE_ENV_VAR}"))
        }
    }
}

impl GpuPrecision {
    fn from_env() -> Result<Self> {
        match std::env::var(GPU_PRECISION_ENV_VAR) {
            Err(std::env::VarError::NotPresent) => Ok(Self::StateF16FullF32),
            Ok(value) if value.is_empty() || value == "state-f16-full-f32" => {
                Ok(Self::StateF16FullF32)
            }
            Ok(value) if value == "mixed" => Ok(Self::Mixed),
            Ok(value) => {
                bail!("{GPU_PRECISION_ENV_VAR} must be state-f16-full-f32 or mixed; got {value:?}")
            }
            Err(error) => Err(error).context(format!("could not read {GPU_PRECISION_ENV_VAR}")),
        }
    }

    fn full_f32_math(self) -> bool {
        self != Self::Mixed
    }

    fn fp32_inputs(self) -> bool {
        self == Self::StateF16FullF32
    }

    fn name(self) -> &'static str {
        match self {
            Self::Mixed => "mixed",
            Self::StateF16FullF32 => "state-f16-full-f32",
        }
    }
}

fn choose_max_rows_per_group(
    max_invocations: usize,
    max_size_x: usize,
    max_storage_bytes: usize,
    lanes_per_row: usize,
) -> Option<usize> {
    [4usize, 2, 1].into_iter().find(|rows| {
        rows * lanes_per_row <= max_invocations
            && rows * lanes_per_row <= max_size_x
            && rows * WORKGROUP_BYTES_PER_ROW <= max_storage_bytes
    })
}

fn shared_rows_supported(
    max_invocations: usize,
    max_size_x: usize,
    max_storage_bytes: usize,
) -> bool {
    max_invocations >= 128 && max_size_x >= 128 && max_storage_bytes >= 2 * WORKGROUP_BYTES_PER_ROW
}

#[derive(Clone, Copy)]
pub(super) struct PackedLinear {
    pub(super) weight: u32,
    pub(super) bias: u32,
}

#[derive(Clone, Copy)]
pub(super) struct PackedNorm {
    pub(super) weight: u32,
    pub(super) bias: u32,
}

#[derive(Clone, Copy)]
pub(super) struct PackedLayer {
    pub(super) ln: PackedNorm,
    pub(super) lerp: u32,
    pub(super) bonus: u32,
    pub(super) wr: PackedLinear,
    pub(super) wk: PackedLinear,
    pub(super) wv: PackedLinear,
    pub(super) wo: PackedLinear,
    pub(super) k_scale: PackedLinear,
    pub(super) v_scale: PackedLinear,
    pub(super) v_lora_a: PackedLinear,
    pub(super) v_lora_b: PackedLinear,
    pub(super) a_lora_a: PackedLinear,
    pub(super) a_lora_b: PackedLinear,
    pub(super) d_lora_a: PackedLinear,
    pub(super) d_lora_b: PackedLinear,
    pub(super) gate_lora_a: PackedLinear,
    pub(super) gate_lora_b: PackedLinear,
    pub(super) group_norm: PackedNorm,
    pub(super) channel_norm: PackedNorm,
    pub(super) channel_lerp: u32,
    pub(super) channel_wk: PackedLinear,
    pub(super) channel_wv: PackedLinear,
    pub(super) channel_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct GpuLayerMeta {
    ln_w: u32,
    ln_b: u32,
    lerp: u32,
    bonus: u32,
    wr: u32,
    wk: u32,
    wv: u32,
    wo: u32,
    ks_w: u32,
    ks_b: u32,
    vs_w: u32,
    vs_b: u32,
    vl_a: u32,
    vl_b: u32,
    vl_bias: u32,
    al_a: u32,
    al_b: u32,
    al_bias: u32,
    dl_a: u32,
    dl_b: u32,
    dl_bias: u32,
    gl_a: u32,
    gl_b: u32,
    gn_w: u32,
    gn_b: u32,
    cln_w: u32,
    cln_b: u32,
    clerp: u32,
    cwk: u32,
    cwv: u32,
    channel_dim: u32,
    module_index: u32,
    entity_stride: u32,
    local_layer: u32,
}

struct PackedModel {
    values: Vec<f16>,
    full_precision_values: Vec<f32>,
    normalization_values: Vec<f32>,
    features_input: PackedLinear,
    features_norm: PackedNorm,
    features_output: PackedLinear,
    layers: Vec<Vec<PackedLayer>>,
    prehead_norm: PackedNorm,
    head_p: PackedLinear,
    p_linear: PackedLinear,
}

impl PackedModel {
    fn from_weights(weights: &SrsRwkvRnnWeights) -> Result<Self> {
        ensure!(
            weights.rwkv_modules.len() == 5,
            "GPU predictor requires five RWKV modules, got {}",
            weights.rwkv_modules.len()
        );
        let mut packer = WeightPacker::default();
        let features_input =
            packer.linear(&weights.features2card.input_linear, FEATURE_DIM, 512)?;
        let features_norm = packer.norm(&weights.features2card.norm, 512)?;
        let features_output = packer.linear(&weights.features2card.output_linear, 512, CHANNELS)?;
        let mut layers = Vec::with_capacity(weights.rwkv_modules.len());
        for module in &weights.rwkv_modules {
            let mut packed_layers = Vec::with_capacity(module.blocks.len());
            for layer in &module.blocks {
                packed_layers.push(packer.layer(layer)?);
            }
            layers.push(packed_layers);
        }
        let prehead_norm = packer.norm(&weights.prehead_norm, CHANNELS)?;
        let head_p = packer.linear(&weights.head_p, CHANNELS, 512)?;
        let p_linear = packer.linear(&weights.p_linear, 512, 4)?;
        if packer.values.len() % 2 != 0 {
            packer.values.push(f16::ZERO);
            packer.full_precision_values.push(0.0);
        }
        debug_assert_eq!(packer.values.len(), packer.full_precision_values.len());
        Ok(Self {
            values: packer.values,
            full_precision_values: packer.full_precision_values,
            normalization_values: packer.normalization_values,
            features_input,
            features_norm,
            features_output,
            layers,
            prehead_norm,
            head_p,
            p_linear,
        })
    }

    fn shader_source(
        &self,
        rows_per_group: usize,
        lanes_per_row: usize,
        precision: GpuPrecision,
        kernel: PredictionKernel,
    ) -> Result<String> {
        let mut source = include_str!("gpu_predict.wgsl").to_owned();
        let workgroup_size = match kernel {
            PredictionKernel::SingleRow => rows_per_group * lanes_per_row,
            PredictionKernel::SharedRows2 => 128,
        };
        for (placeholder, value) in [
            ("/*__ROWS_PER_GROUP__*/", rows_per_group),
            ("/*__SMALL_VALUES__*/", rows_per_group * 128),
            ("/*__LARGE_VALUES__*/", rows_per_group * 512),
            (
                "/*__EXTRA_VALUES__*/",
                rows_per_group * EXTRA_VALUES_PER_ROW,
            ),
            ("/*__EXTRA_STRIDE__*/", EXTRA_VALUES_PER_ROW),
            ("/*__LANES_PER_ROW__*/", lanes_per_row),
            ("/*__WORKGROUP_SIZE__*/", workgroup_size),
        ] {
            source = source.replace(placeholder, &value.to_string());
        }
        source = source.replace(
            "/*__MODEL_VALUE_TYPE__*/",
            if precision.fp32_inputs() {
                "f32"
            } else {
                "f16"
            },
        );
        source = source.replace(
            "/*__FEATURE_VALUE_TYPE__*/",
            if precision.fp32_inputs() {
                "f32"
            } else {
                "f16"
            },
        );
        source = source.replace(
            "/*__FULL_F32_MATH__*/",
            if precision.full_f32_math() {
                "true"
            } else {
                "false"
            },
        );
        let mut linear_functions = String::new();
        for (name, values, base) in [
            ("x", "x", "small_base"),
            ("norm", "norm_values", "small_base"),
            ("tmp0", "tmp0", "large_base"),
            ("tmp1", "tmp1", "large_base"),
            ("v", "tmp0", "large_base + V_OFFSET"),
            ("d", "tmp1", "large_base + D_OFFSET"),
            ("a", "tmp1", "large_base + A_OFFSET"),
            ("g", "extra", "extra_base + G_OFFSET"),
            ("out", "extra", "extra_base + OUT_OFFSET"),
        ] {
            linear_functions.push_str(&format!(
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
        if (bias_offset != INVALID_SLOT) {{
            sum = f32(model_weights[bias_offset + output_index]);
        }}
        var input_index = 0u;
        loop {{
            if (input_index >= input_dim) {{ break; }}
            let packed_offset = weight_offset
                + input_index * output_dim
                + output_index * 4u;
            sum += matrix_product(
                {values}[{base} + input_index],
                f32(model_weights[packed_offset]),
            );
            sum += matrix_product(
                {values}[{base} + input_index + 1u],
                f32(model_weights[packed_offset + 1u]),
            );
            sum += matrix_product(
                {values}[{base} + input_index + 2u],
                f32(model_weights[packed_offset + 2u]),
            );
            sum += matrix_product(
                {values}[{base} + input_index + 3u],
                f32(model_weights[packed_offset + 3u]),
            );
            input_index += 4u;
        }}
        set_scratch_value(output, output_index, sum);
        output_index += LANES_PER_ROW;
    }}
    workgroupBarrier();
}}
"#
            ));
        }
        source = source.replace("/*__LINEAR_FUNCTIONS__*/", &linear_functions);
        let shared_row_functions = if kernel == PredictionKernel::SharedRows2 {
            ensure!(
                rows_per_group == 2 && lanes_per_row == 128,
                "shared-row predictor requires two rows and 128 projection lanes"
            );
            let mut shared = include_str!("gpu_predict_shared_rows.wgsl").to_owned();
            let mut shared_linear_functions = String::new();
            for (name, values, base0, base1) in [
                ("x", "x", "0u", "SMALL_STRIDE"),
                ("norm", "norm_values", "0u", "SMALL_STRIDE"),
                ("tmp0", "tmp0", "0u", "LARGE_STRIDE"),
                ("tmp1", "tmp1", "0u", "LARGE_STRIDE"),
                ("v", "tmp0", "V_OFFSET", "LARGE_STRIDE + V_OFFSET"),
                ("d", "tmp1", "D_OFFSET", "LARGE_STRIDE + D_OFFSET"),
                ("a", "tmp1", "A_OFFSET", "LARGE_STRIDE + A_OFFSET"),
                ("g", "extra", "G_OFFSET", "EXTRA_STRIDE + G_OFFSET"),
                ("out", "extra", "OUT_OFFSET", "EXTRA_STRIDE + OUT_OFFSET"),
            ] {
                shared_linear_functions.push_str(&format!(
                    r#"
fn shared_linear_{name}(
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
        var sum0 = 0.0;
        var sum1 = 0.0;
        if (bias_offset != INVALID_SLOT) {{
            let bias = f32(model_weights[bias_offset + output_index]);
            sum0 = bias;
            sum1 = bias;
        }}
        var input_index = 0u;
        loop {{
            if (input_index >= input_dim) {{ break; }}
            let packed_offset = weight_offset
                + input_index * output_dim
                + output_index * 4u;
            let weight0 = f32(model_weights[packed_offset]);
            let weight1 = f32(model_weights[packed_offset + 1u]);
            let weight2 = f32(model_weights[packed_offset + 2u]);
            let weight3 = f32(model_weights[packed_offset + 3u]);
            sum0 += matrix_product({values}[{base0} + input_index], weight0);
            sum0 += matrix_product({values}[{base0} + input_index + 1u], weight1);
            sum0 += matrix_product({values}[{base0} + input_index + 2u], weight2);
            sum0 += matrix_product({values}[{base0} + input_index + 3u], weight3);
            sum1 += matrix_product({values}[{base1} + input_index], weight0);
            sum1 += matrix_product({values}[{base1} + input_index + 1u], weight1);
            sum1 += matrix_product({values}[{base1} + input_index + 2u], weight2);
            sum1 += matrix_product({values}[{base1} + input_index + 3u], weight3);
            input_index += 4u;
        }}
        shared_set_scratch_value(output, 0u, output_index, sum0);
        shared_set_scratch_value(output, 1u, output_index, sum1);
        output_index += SHARED_LANES;
    }}
    workgroupBarrier();
}}
"#
                ));
            }
            shared = shared.replace("/*__SHARED_LINEAR_FUNCTIONS__*/", &shared_linear_functions);
            shared
        } else {
            String::new()
        };
        source = source.replace("/*__SHARED_ROW_FUNCTIONS__*/", &shared_row_functions);
        for (placeholder, value) in [
            ("/*__FEATURES_INPUT_W__*/", self.features_input.weight),
            ("/*__FEATURES_INPUT_B__*/", self.features_input.bias),
            ("/*__FEATURES_NORM_W__*/", self.features_norm.weight),
            ("/*__FEATURES_NORM_B__*/", self.features_norm.bias),
            ("/*__FEATURES_OUTPUT_W__*/", self.features_output.weight),
            ("/*__FEATURES_OUTPUT_B__*/", self.features_output.bias),
            ("/*__PREHEAD_W__*/", self.prehead_norm.weight),
            ("/*__PREHEAD_B__*/", self.prehead_norm.bias),
            ("/*__HEAD_P_W__*/", self.head_p.weight),
            ("/*__HEAD_P_B__*/", self.head_p.bias),
            ("/*__P_LINEAR_W__*/", self.p_linear.weight),
            ("/*__P_LINEAR_B__*/", self.p_linear.bias),
        ] {
            source = source.replace(placeholder, &value.to_string());
        }

        if source.contains("/*__") {
            bail!("internal GPU shader template has an unresolved placeholder");
        }
        Ok(source)
    }

    fn layer_metadata(&self) -> Vec<GpuLayerMeta> {
        let mut metadata = Vec::new();
        for (module_index, layers) in self.layers.iter().enumerate() {
            let entity_stride = (layers.len() * STATE_LAYER_ELEMENTS) as u32;
            for (local_layer, layer) in layers.iter().copied().enumerate() {
                metadata.push(layer.gpu_meta(
                    module_index as u32,
                    entity_stride,
                    local_layer as u32,
                ));
            }
        }
        metadata
    }
}

impl PackedLayer {
    pub(super) fn gpu_meta(
        self,
        module_index: u32,
        entity_stride: u32,
        local_layer: u32,
    ) -> GpuLayerMeta {
        GpuLayerMeta {
            ln_w: self.ln.weight,
            ln_b: self.ln.bias,
            lerp: self.lerp,
            bonus: self.bonus,
            wr: self.wr.weight,
            wk: self.wk.weight,
            wv: self.wv.weight,
            wo: self.wo.weight,
            ks_w: self.k_scale.weight,
            ks_b: self.k_scale.bias,
            vs_w: self.v_scale.weight,
            vs_b: self.v_scale.bias,
            vl_a: self.v_lora_a.weight,
            vl_b: self.v_lora_b.weight,
            vl_bias: self.v_lora_b.bias,
            al_a: self.a_lora_a.weight,
            al_b: self.a_lora_b.weight,
            al_bias: self.a_lora_b.bias,
            dl_a: self.d_lora_a.weight,
            dl_b: self.d_lora_b.weight,
            dl_bias: self.d_lora_b.bias,
            gl_a: self.gate_lora_a.weight,
            gl_b: self.gate_lora_b.weight,
            gn_w: self.group_norm.weight,
            gn_b: self.group_norm.bias,
            cln_w: self.channel_norm.weight,
            cln_b: self.channel_norm.bias,
            clerp: self.channel_lerp,
            cwk: self.channel_wk.weight,
            cwv: self.channel_wv.weight,
            channel_dim: self.channel_dim,
            module_index,
            entity_stride,
            local_layer,
        }
    }
}

#[derive(Default)]
pub(super) struct WeightPacker {
    pub(super) values: Vec<f16>,
    pub(super) full_precision_values: Vec<f32>,
    pub(super) normalization_values: Vec<f32>,
}

impl WeightPacker {
    fn tensor(&mut self, tensor: &candle_core::Tensor, expected: usize, name: &str) -> Result<u32> {
        ensure!(
            tensor.elem_count() == expected,
            "{name} expected {expected} values, got {} with shape {:?}",
            tensor.elem_count(),
            tensor.dims()
        );
        let offset =
            u32::try_from(self.values.len()).context("model weight buffer is too large")?;
        let data = f32_tensor_data(tensor)?;
        for value in data.as_slice()?.iter().copied() {
            self.values.push(f16::from_f32(value));
            self.full_precision_values.push(value);
        }
        Ok(offset)
    }

    fn normalization_tensor(
        &mut self,
        tensor: &candle_core::Tensor,
        expected: usize,
        name: &str,
    ) -> Result<u32> {
        ensure!(
            tensor.elem_count() == expected,
            "{name} expected {expected} values, got {} with shape {:?}",
            tensor.elem_count(),
            tensor.dims()
        );
        let offset = u32::try_from(self.normalization_values.len())
            .context("FP32 normalization weight buffer is too large")?;
        let data = f32_tensor_data(tensor)?;
        self.normalization_values
            .extend_from_slice(data.as_slice()?);
        Ok(offset)
    }

    pub(super) fn linear(
        &mut self,
        linear: &LinearWeights,
        input: usize,
        output: usize,
    ) -> Result<PackedLinear> {
        ensure!(
            linear.weight.dims() == [output, input],
            "linear expected weight shape [{output}, {input}], got {:?}",
            linear.weight.dims()
        );
        ensure!(
            input.is_multiple_of(4),
            "GPU linear input dimension must be divisible by four"
        );
        let weight =
            u32::try_from(self.values.len()).context("model weight buffer is too large")?;
        let data = f32_tensor_data(&linear.weight)?;
        let values = data.as_slice()?;
        for input_base in (0..input).step_by(4) {
            for output_index in 0..output {
                let row = &values[output_index * input..][..input];
                for value in row[input_base..input_base + 4].iter().copied() {
                    self.values.push(f16::from_f32(value));
                    self.full_precision_values.push(value);
                }
            }
        }
        let bias = match linear.bias.as_ref() {
            Some(bias) => self.tensor(bias, output, "linear.bias")?,
            None => INVALID_OFFSET,
        };
        Ok(PackedLinear { weight, bias })
    }

    pub(super) fn norm(&mut self, norm: &LayerNormWeights, size: usize) -> Result<PackedNorm> {
        Ok(PackedNorm {
            weight: self.normalization_tensor(&norm.weight, size, "norm.weight")?,
            bias: self.tensor(&norm.bias, size, "norm.bias")?,
        })
    }

    pub(super) fn layer(&mut self, layer: &Rwkv7RnnLayerWeights) -> Result<PackedLayer> {
        let time = &layer.time_mixer;
        let channel = &layer.channel_mixer;
        ensure!(
            time.n_heads == 4 && time.head_size == 32,
            "GPU predictor requires 4x32 heads"
        );
        ensure!(
            channel.channel_dim <= 512,
            "channel hidden dimension exceeds GPU scratch size"
        );
        let ln = self.norm(&time.layer_norm, CHANNELS)?;
        let lerp = self.tensor(&time.rkvdag_lerp, 8 * CHANNELS, "time lerp")?;
        let bonus = self.tensor(&time.bonus, CHANNELS, "time bonus")?;
        let wr = self.linear(&time.w_r, CHANNELS, CHANNELS)?;
        let wk = self.linear(&time.w_k, CHANNELS, CHANNELS)?;
        let wv = self.linear(&time.w_v, CHANNELS, CHANNELS)?;
        let wo = self.linear(&time.w_o, CHANNELS, CHANNELS)?;
        let k_scale = self.linear(&time.k_scale_linear, CHANNELS, 4)?;
        let v_scale = self.linear(&time.v_scale_linear, CHANNELS, 4)?;
        let v_lora_a = self.linear(&time.v_lora_simple.a, CHANNELS, 8)?;
        let v_lora_b = self.linear(&time.v_lora_simple.b_and_lamb, 8, CHANNELS)?;
        let a_lora_a = self.linear(&time.a_lora_simple.a, CHANNELS, 16)?;
        let a_lora_b = self.linear(&time.a_lora_simple.b_and_lamb, 16, CHANNELS)?;
        let d_lora_a = self.linear(&time.d_lora_mlp.a, CHANNELS, 16)?;
        let d_lora_b = self.linear(&time.d_lora_mlp.b_and_lamb, 16, CHANNELS)?;
        let gate_lora_a = self.linear(&time.gate_lora.a, CHANNELS, 16)?;
        let gate_lora_b = self.linear(&time.gate_lora.b, 16, CHANNELS)?;
        let group_norm = PackedNorm {
            weight: self.normalization_tensor(
                &time.out_group_norm.weight,
                CHANNELS,
                "group norm weight",
            )?,
            bias: self.tensor(&time.out_group_norm.bias, CHANNELS, "group norm bias")?,
        };
        let channel_norm = self.norm(&channel.layer_norm, CHANNELS)?;
        let channel_lerp = self.tensor(&channel.lerp_k, CHANNELS, "channel lerp")?;
        let channel_wk = self.linear(&channel.w_k, CHANNELS, channel.channel_dim)?;
        let channel_wv = self.linear(&channel.w_v, channel.channel_dim, CHANNELS)?;
        Ok(PackedLayer {
            ln,
            lerp,
            bonus,
            wr,
            wk,
            wv,
            wo,
            k_scale,
            v_scale,
            v_lora_a,
            v_lora_b,
            a_lora_a,
            a_lora_b,
            d_lora_a,
            d_lora_b,
            gate_lora_a,
            gate_lora_b,
            group_norm,
            channel_norm,
            channel_lerp,
            channel_wk,
            channel_wv,
            channel_dim: channel.channel_dim as u32,
        })
    }
}

struct StateArena {
    name: &'static str,
    buffer: wgpu::Buffer,
    slots: HashMap<i64, u32>,
    layers: usize,
    capacity: usize,
    max_capacity: usize,
}

#[derive(Default)]
struct StateArenaUndo {
    uploaded_bytes: u64,
    topology_changed: bool,
    retired_slot: bool,
}

fn retire_tail_slot(
    slots: &mut HashMap<i64, u32>,
    identity: i64,
    arena_name: &str,
) -> Result<bool> {
    let Some(slot) = slots.get(&identity).copied() else {
        return Ok(false);
    };
    let expected_tail = slots
        .len()
        .checked_sub(1)
        .context("GPU state arena slot map is unexpectedly empty")? as u32;
    ensure!(
        slot == expected_tail,
        "cannot retire non-tail {arena_name} GPU state slot {slot}; expected {expected_tail}"
    );
    slots.remove(&identity);
    Ok(true)
}

impl StateArena {
    fn new(
        context: &GpuContext,
        name: &'static str,
        layers: usize,
        states: &BTreeMap<i64, NativeRnnModuleState>,
        flat_states: &BTreeMap<i64, FlatNativeRnnModuleState>,
    ) -> Result<(Self, u64)> {
        let entry_bytes = layers * STATE_LAYER_ELEMENTS * std::mem::size_of::<f16>();
        let max_buffer_bytes = context
            .limits
            .max_buffer_size
            .min(context.limits.max_storage_buffer_binding_size)
            as usize;
        let max_capacity = max_buffer_bytes / entry_bytes;
        let state_count = states
            .len()
            .checked_add(flat_states.len())
            .context("GPU state arena entry count overflow")?;
        ensure!(
            state_count <= max_capacity,
            "{name} FP16 state requires {state_count} entries but this GPU supports at most {max_capacity}"
        );
        // Keep the persistent allocation close to the selected checkpoint
        // scope. Large full checkpoints already consume multiple GiB in FP16;
        // a large geometric reserve would defeat partial loading's purpose.
        let headroom = (state_count / 64).max(8);
        let capacity = (state_count + headroom).clamp(1, max_capacity.max(1));
        let buffer = context.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name),
            size: (capacity * entry_bytes).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })?;
        let mut slots = HashMap::with_capacity(state_count);
        let chunk_entries = (INITIAL_UPLOAD_CHUNK_BYTES / entry_bytes).max(1);
        let mut chunk = Vec::with_capacity(chunk_entries * layers * STATE_LAYER_ELEMENTS);
        let mut chunk_start = 0usize;
        let mut uploaded_bytes = 0u64;
        let mut states = states.iter().peekable();
        let mut flat_states = flat_states.iter().peekable();
        let mut slot = 0usize;
        loop {
            let (identity, state, flat_state) = match (states.peek(), flat_states.peek()) {
                (Some((state_id, _)), Some((flat_id, _))) => match state_id.cmp(flat_id) {
                    std::cmp::Ordering::Less => {
                        let (identity, state) = states.next().expect("peeked canonical state");
                        (identity, Some(state), None)
                    }
                    std::cmp::Ordering::Greater => {
                        let (identity, state) = flat_states.next().expect("peeked flat state");
                        (identity, None, Some(state))
                    }
                    std::cmp::Ordering::Equal => {
                        bail!("{name} identity {state_id} exists in flat and canonical CPU state");
                    }
                },
                (Some(_), None) => {
                    let (identity, state) = states.next().expect("peeked canonical state");
                    (identity, Some(state), None)
                }
                (None, Some(_)) => {
                    let (identity, state) = flat_states.next().expect("peeked flat state");
                    (identity, None, Some(state))
                }
                (None, None) => break,
            };
            if chunk.is_empty() {
                chunk_start = slot;
            }
            slots.insert(*identity, slot as u32);
            if let Some(state) = state {
                append_state_f16(&mut chunk, state, layers, name)?;
            } else {
                append_flat_state_f16(
                    &mut chunk,
                    flat_state.expect("selected flat state exists"),
                    layers,
                    name,
                )?;
            }
            if chunk.len() >= chunk_entries * layers * STATE_LAYER_ELEMENTS {
                context.write_buffer(
                    "initial FP16 state upload",
                    &buffer,
                    (chunk_start * entry_bytes) as u64,
                    bytemuck::cast_slice(&chunk),
                )?;
                uploaded_bytes += (chunk.len() * std::mem::size_of::<f16>()) as u64;
                context.queue.submit([]);
                context
                    .device
                    .poll(wgpu::PollType::Poll)
                    .context("FP16 state upload polling failed")?;
                chunk.clear();
            }
            slot += 1;
        }
        if !chunk.is_empty() {
            context.write_buffer(
                "initial FP16 state upload",
                &buffer,
                (chunk_start * entry_bytes) as u64,
                bytemuck::cast_slice(&chunk),
            )?;
            uploaded_bytes += (chunk.len() * std::mem::size_of::<f16>()) as u64;
            context.queue.submit([]);
            context
                .device
                .poll(wgpu::PollType::Poll)
                .context("FP16 state upload polling failed")?;
        }
        Ok((
            Self {
                name,
                buffer,
                slots,
                layers,
                capacity,
                max_capacity,
            },
            uploaded_bytes,
        ))
    }

    fn singleton(
        context: &GpuContext,
        name: &'static str,
        layers: usize,
        state: Option<&NativeRnnModuleState>,
        flat_state: Option<&FlatNativeRnnModuleState>,
    ) -> Result<(Self, u64)> {
        let states = state
            .map(|state| BTreeMap::from([(0i64, state.clone())]))
            .unwrap_or_default();
        let flat_states = flat_state
            .map(|state| BTreeMap::from([(0i64, state.clone())]))
            .unwrap_or_default();
        Self::new(context, name, layers, &states, &flat_states)
    }

    fn slot(&self, identity: i64) -> u32 {
        self.slots.get(&identity).copied().unwrap_or(INVALID_OFFSET)
    }

    fn entry_bytes(&self) -> usize {
        self.layers * STATE_LAYER_ELEMENTS * std::mem::size_of::<f16>()
    }

    fn allocated_bytes(&self) -> usize {
        self.capacity * self.entry_bytes()
    }

    fn update(
        &mut self,
        context: &GpuContext,
        identity: i64,
        state: &NativeRnnModuleState,
    ) -> Result<(u64, bool)> {
        let mut resized = false;
        let slot = if let Some(slot) = self.slots.get(&identity).copied() {
            slot
        } else {
            if self.slots.len() == self.capacity {
                self.grow(context)?;
                resized = true;
            }
            let slot = self.slots.len() as u32;
            self.slots.insert(identity, slot);
            slot
        };
        let mut values = Vec::with_capacity(self.layers * STATE_LAYER_ELEMENTS);
        append_state_f16(&mut values, state, self.layers, self.name)?;
        context.write_buffer(
            "incremental FP16 state upload",
            &self.buffer,
            u64::from(slot) * self.entry_bytes() as u64,
            bytemuck::cast_slice(&values),
        )?;
        Ok(((values.len() * std::mem::size_of::<f16>()) as u64, resized))
    }

    fn restore_after_undo(
        &mut self,
        context: &GpuContext,
        identity: i64,
        state: Option<&NativeRnnModuleState>,
    ) -> Result<StateArenaUndo> {
        if let Some(state) = state {
            let topology_changed = !self.slots.contains_key(&identity);
            let (uploaded_bytes, _) = self.update(context, identity, state)?;
            return Ok(StateArenaUndo {
                uploaded_bytes,
                topology_changed,
                retired_slot: false,
            });
        }

        if !retire_tail_slot(&mut self.slots, identity, self.name)? {
            // The failed process path can restore an undo frame before the GPU
            // update reached this arena. In that case there is nothing to do.
            return Ok(StateArenaUndo::default());
        }
        // The allocation and old bytes deliberately remain. LIFO undo makes
        // this slot the next insertion point, avoiding a buffer resize/copy.
        Ok(StateArenaUndo {
            uploaded_bytes: 0,
            topology_changed: true,
            retired_slot: true,
        })
    }

    fn grow(&mut self, context: &GpuContext) -> Result<()> {
        ensure!(
            self.capacity < self.max_capacity,
            "{} FP16 state buffer is at this GPU's binding-size limit",
            self.name
        );
        let new_capacity = (self.capacity + self.capacity / 8 + 8).min(self.max_capacity);
        let new_buffer = context.create_buffer(&wgpu::BufferDescriptor {
            label: Some(self.name),
            size: (new_capacity * self.entry_bytes()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })?;
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rwkv-srs grow state arena"),
            });
        let used_bytes = self.slots.len() * self.entry_bytes();
        if used_bytes != 0 {
            encoder.copy_buffer_to_buffer(&self.buffer, 0, &new_buffer, 0, used_bytes as u64);
        }
        context.queue.submit([encoder.finish()]);
        self.buffer = new_buffer;
        self.capacity = new_capacity;
        Ok(())
    }
}

fn append_state_f16(
    output: &mut Vec<f16>,
    state: &NativeRnnModuleState,
    layers: usize,
    name: &str,
) -> Result<()> {
    ensure!(
        state.time_x_shift_b1c_by_layer.len() == layers
            && state.time_state_b1hkk_by_layer.len() == layers
            && state.channel_state_b1c_by_layer.len() == layers,
        "{name} expected {layers} layers of each recurrent state"
    );
    let start_len = output.len();
    for layer in 0..layers {
        append_tensor_f16(output, &state.time_x_shift_b1c_by_layer[layer], 128, name)?;
        append_tensor_f16(
            output,
            &state.time_state_b1hkk_by_layer[layer],
            4 * 32 * 32,
            name,
        )?;
        append_tensor_f16(output, &state.channel_state_b1c_by_layer[layer], 128, name)?;
    }
    ensure!(
        output.len() - start_len == layers * STATE_LAYER_ELEMENTS,
        "{name} packed state length mismatch"
    );
    Ok(())
}

fn append_flat_state_f16(
    output: &mut Vec<f16>,
    state: &FlatNativeRnnModuleState,
    layers: usize,
    name: &str,
) -> Result<()> {
    ensure!(
        state.layers() == layers,
        "{name} expected {layers} flat recurrent-state layers, got {}",
        state.layers()
    );
    let expected = layers * STATE_LAYER_ELEMENTS;
    ensure!(
        state.values().len() == expected,
        "{name} expected {expected} flat recurrent-state values, got {}",
        state.values().len()
    );
    output.extend(state.values().iter().copied().map(f16::from_f32));
    Ok(())
}

fn append_tensor_f16(
    output: &mut Vec<f16>,
    tensor: &candle_core::Tensor,
    expected: usize,
    name: &str,
) -> Result<()> {
    ensure!(
        tensor.elem_count() == expected,
        "{name} state tensor expected {expected} values, got {} with shape {:?}",
        tensor.elem_count(),
        tensor.dims()
    );
    let data = f32_tensor_data(tensor)?;
    output.extend(data.as_slice()?.iter().copied().map(f16::from_f32));
    Ok(())
}

#[derive(Default, Clone)]
pub(super) struct GpuProfile {
    pub(super) initialization_ns: u128,
    pub(super) weight_upload_bytes: u64,
    pub(super) fp16_weight_upload_bytes: u64,
    pub(super) fp32_weight_upload_bytes: u64,
    pub(super) normalization_weight_upload_bytes: u64,
    pub(super) initial_state_upload_bytes: u64,
    pub(super) initial_state_upload_ns: u128,
    pub(super) update_count: u64,
    pub(super) update_bytes: u64,
    pub(super) update_ns: u128,
    pub(super) undo_update_count: u64,
    pub(super) undo_update_bytes: u64,
    pub(super) undo_update_ns: u128,
    pub(super) undo_retired_slots: u64,
    pub(super) synchronization_count: u64,
    pub(super) synchronization_wait_ns: u128,
    pub(super) last_synchronization_wait_ns: u128,
    pub(super) prediction_calls: u64,
    pub(super) prediction_rows: u64,
    pub(super) host_input_parse_ns: u128,
    pub(super) host_scope_validation_ns: u128,
    pub(super) host_feature_build_ns: u128,
    pub(super) host_fallback_rows: u64,
    pub(super) last_host_input_parse_ns: u128,
    pub(super) last_host_scope_validation_ns: u128,
    pub(super) last_host_feature_build_ns: u128,
    pub(super) last_host_fallback_rows: u64,
    pub(super) state_slot_resolution_ns: u128,
    pub(super) feature_upload_ns: u128,
    pub(super) command_setup_ns: u128,
    pub(super) submit_to_readback_ns: u128,
    pub(super) kernel_ns: u128,
    pub(super) kernel_timestamp_samples: u64,
    pub(super) result_conversion_ns: u128,
    pub(super) python_result_ns: u128,
    pub(super) python_binding_total_ns: u128,
    pub(super) prediction_total_ns: u128,
    pub(super) last_prediction_total_ns: u128,
    pub(super) last_state_slot_resolution_ns: u128,
    pub(super) last_feature_upload_ns: u128,
    pub(super) last_command_setup_ns: u128,
    pub(super) last_submit_to_readback_ns: u128,
    pub(super) last_kernel_ns: Option<u128>,
    pub(super) last_result_conversion_ns: u128,
    pub(super) last_python_result_ns: u128,
    pub(super) last_python_binding_total_ns: u128,
    pub(super) last_prediction_rows: u64,
    pub(super) last_padded_rows: u64,
    pub(super) last_rows_per_workgroup: u64,
    pub(super) last_lanes_per_row: u64,
    pub(super) last_workgroup_size: u64,
    pub(super) last_shared_rows_kernel: bool,
    pub(super) slot_cache_hits: u64,
    pub(super) slot_cache_misses: u64,
    pub(super) last_slot_cache_hits: u64,
    pub(super) last_slot_cache_misses: u64,
    pub(super) prediction_pipeline_calls: u64,
    pub(super) prediction_pipeline_submitted_chunks: u64,
    pub(super) prediction_pipeline_blocked_wait_ns: u128,
    pub(super) prediction_pipeline_overlap_ns: u128,
    pub(super) prediction_pipeline_fallbacks: u64,
    pub(super) prediction_pipeline_total_ns: u128,
    pub(super) prediction_pipeline_high_water_bytes: u64,
    pub(super) last_prediction_pipeline_submitted_chunks: u64,
    pub(super) last_prediction_pipeline_blocked_wait_ns: u128,
    pub(super) last_prediction_pipeline_overlap_ns: u128,
    pub(super) last_prediction_pipeline_fallbacks: u64,
    pub(super) last_prediction_pipeline_total_ns: u128,
    pub(super) last_prediction_pipeline_high_water_bytes: u64,
    pub(super) adaptive_splits: u64,
    pub(super) oom_retries: u64,
}

pub(super) struct PendingGpuPrediction {
    // Keep every transient resource alive until its submission and mapped
    // readback have completed. Two pending values therefore describe the
    // pipeline's complete transient GPU high-water set.
    _feature_buffer: wgpu::Buffer,
    _slot_buffer: wgpu::Buffer,
    _output_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    receiver: mpsc::Receiver<std::result::Result<(), wgpu::BufferAsyncError>>,
    submission_index: wgpu::SubmissionIndex,
    total_start: Instant,
    submit_start: Instant,
    state_slot_resolution_ns: u128,
    feature_upload_ns: u128,
    command_setup_ns: u128,
    rows: usize,
    padded_rows: usize,
    rows_per_group: usize,
    lanes_per_row: usize,
    workgroup_size: usize,
    shared_rows_kernel: bool,
    slot_cache_hits: u64,
    slot_cache_misses: u64,
    output_size: usize,
    timestamp_offset: usize,
    timestamp_bytes: usize,
    transient_bytes: u64,
    pipelined: bool,
}

impl PendingGpuPrediction {
    pub(super) fn transient_bytes(&self) -> u64 {
        self.transient_bytes
    }
}

struct PipelineVariant {
    kernel: PredictionKernel,
    rows_per_group: usize,
    lanes_per_row: usize,
    workgroup_size: usize,
    pipeline: wgpu::ComputePipeline,
}

fn compile_prediction_pipeline(
    context: &GpuContext,
    packed: &PackedModel,
    precision: GpuPrecision,
    kernel: PredictionKernel,
    lanes_per_row: usize,
    rows_per_group: usize,
) -> Result<PipelineVariant> {
    let shader_source = packed.shader_source(rows_per_group, lanes_per_row, precision, kernel)?;
    let (entry_point, workgroup_size) = match kernel {
        PredictionKernel::SingleRow => ("predict", rows_per_group * lanes_per_row),
        PredictionKernel::SharedRows2 => ("predict_shared_rows", 128),
    };
    let error_scope = context
        .device
        .push_error_scope(wgpu::ErrorFilter::Validation);
    let shader = context
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rwkv-srs fused predictor"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader_source)),
        });
    let pipeline = context
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rwkv-srs fused predictor"),
            layout: None,
            module: &shader,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        });
    if let Some(error) = pollster::block_on(error_scope.pop()) {
        bail!(
            "could not compile {} prediction shader ({lanes_per_row} lanes/row, {rows_per_group} rows/workgroup): {error}",
            kernel.name(),
        );
    }
    Ok(PipelineVariant {
        kernel,
        rows_per_group,
        lanes_per_row,
        workgroup_size,
        pipeline,
    })
}

fn select_prediction_shape(
    shapes: &[(usize, usize)],
    lanes_per_row: usize,
    forced_rows_per_group: Option<usize>,
    row_count: usize,
) -> Option<(usize, usize)> {
    if let Some(rows_per_group) = forced_rows_per_group {
        return shapes
            .contains(&(lanes_per_row, rows_per_group))
            .then_some((lanes_per_row, rows_per_group));
    }
    let preferred_rows = if lanes_per_row == 64 && row_count <= 256 {
        4
    } else if lanes_per_row == 64 {
        2
    } else {
        1
    };
    shapes
        .iter()
        .copied()
        .filter(|(lanes, rows)| *lanes == lanes_per_row && *rows <= preferred_rows)
        .max_by_key(|(_, rows)| *rows)
        .or_else(|| {
            shapes
                .iter()
                .copied()
                .filter(|(lanes, _)| *lanes == lanes_per_row)
                .min_by_key(|(_, rows)| *rows)
        })
}

pub(super) struct GpuPredictor {
    context: GpuContext,
    precision: GpuPrecision,
    pipelines: Vec<PipelineVariant>,
    default_kernel: PredictionKernel,
    default_lanes_per_row: usize,
    wide_lane_pipeline: bool,
    weight_buffer: wgpu::Buffer,
    layer_metadata_buffer: wgpu::Buffer,
    timestamp_query: Option<wgpu::QuerySet>,
    timestamp_resolve_buffer: Option<wgpu::Buffer>,
    card_states: StateArena,
    deck_states: StateArena,
    note_states: StateArena,
    preset_states: StateArena,
    global_states: StateArena,
    slot_cache: HashMap<ReviewIds, [u32; 5]>,
    profile: GpuProfile,
}

impl GpuPredictor {
    fn new(rnn: &NativeRnn) -> Result<Self> {
        let initialization_start = Instant::now();
        let context = GpuContext::new(GpuOperation::Predict)?;
        let precision = GpuPrecision::from_env()?;
        ensure!(
            context.limits.max_storage_buffers_per_shader_stage >= 10,
            "GPU supports only {} storage buffers per shader stage; predictor needs 10",
            context.limits.max_storage_buffers_per_shader_stage
        );
        let packed = PackedModel::from_weights(&rnn.weights)?;
        let fp16_weight_upload_bytes = if precision.fp32_inputs() {
            0
        } else {
            (packed.values.len() * std::mem::size_of::<f16>()) as u64
        };
        let fp32_weight_upload_bytes = if precision.fp32_inputs() {
            (packed.full_precision_values.len() * std::mem::size_of::<f32>()) as u64
        } else {
            0
        };
        let normalization_weight_upload_bytes =
            (packed.normalization_values.len() * std::mem::size_of::<f32>()) as u64;
        let weight_upload_bytes =
            fp16_weight_upload_bytes + fp32_weight_upload_bytes + normalization_weight_upload_bytes;
        ensure!(
            fp16_weight_upload_bytes + fp32_weight_upload_bytes
                <= context.limits.max_storage_buffer_binding_size,
            "model weights exceed this GPU's storage binding limit"
        );
        let layer_metadata = packed.layer_metadata();
        let weight_buffer = if precision.fp32_inputs() {
            context.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rwkv-srs FP32 model weights"),
                contents: bytemuck::cast_slice(&packed.full_precision_values),
                usage: wgpu::BufferUsages::STORAGE,
            })?
        } else {
            context.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rwkv-srs FP16 model weights"),
                contents: bytemuck::cast_slice(&packed.values),
                usage: wgpu::BufferUsages::STORAGE,
            })?
        };
        let mut metadata_values = bytemuck::cast_slice(&layer_metadata).to_vec();
        metadata_values.extend_from_slice(bytemuck::cast_slice(&packed.normalization_values));
        let layer_metadata_buffer =
            context.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rwkv-srs GPU layer metadata and FP32 normalization weights"),
                contents: &metadata_values,
                usage: wgpu::BufferUsages::STORAGE,
            })?;
        let (timestamp_query, timestamp_resolve_buffer) = if context.timestamp_queries {
            let query = context.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("rwkv-srs prediction timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            });
            let resolve = context.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rwkv-srs prediction timestamp resolve"),
                size: 2 * std::mem::size_of::<u64>() as u64,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })?;
            (Some(query), Some(resolve))
        } else {
            (None, None)
        };

        let mut pipeline_shapes = Vec::new();
        for lanes_per_row in [128usize, 64] {
            let max_rows_per_group = choose_max_rows_per_group(
                context.limits.max_compute_invocations_per_workgroup as usize,
                context.limits.max_compute_workgroup_size_x as usize,
                context.limits.max_compute_workgroup_storage_size as usize,
                lanes_per_row,
            );
            let Some(max_rows_per_group) = max_rows_per_group else {
                continue;
            };
            let rows = if lanes_per_row == 128 {
                vec![1]
            } else {
                [4usize, 2, 1]
                    .into_iter()
                    .filter(|rows| *rows <= max_rows_per_group)
                    .collect()
            };
            pipeline_shapes.extend(
                rows.into_iter()
                    .map(|rows_per_group| (lanes_per_row, rows_per_group)),
            );
        }
        ensure!(
            !pipeline_shapes.is_empty(),
            "GPU workgroup limits are too small for the predictor"
        );
        let mut pipelines = Vec::with_capacity(pipeline_shapes.len());
        let mut pipeline_errors = Vec::new();
        for (lanes_per_row, rows_per_group) in pipeline_shapes {
            match compile_prediction_pipeline(
                &context,
                &packed,
                precision,
                PredictionKernel::SingleRow,
                lanes_per_row,
                rows_per_group,
            ) {
                Ok(pipeline) => pipelines.push(pipeline),
                Err(error) => pipeline_errors.push(error.to_string()),
            }
        }
        if shared_rows_supported(
            context.limits.max_compute_invocations_per_workgroup as usize,
            context.limits.max_compute_workgroup_size_x as usize,
            context.limits.max_compute_workgroup_storage_size as usize,
        ) {
            match compile_prediction_pipeline(
                &context,
                &packed,
                precision,
                PredictionKernel::SharedRows2,
                128,
                2,
            ) {
                Ok(pipeline) => pipelines.push(pipeline),
                Err(error) => pipeline_errors.push(error.to_string()),
            }
        }
        ensure!(
            !pipelines.is_empty(),
            "GPU could not compile any prediction pipeline variant: {}",
            pipeline_errors.join("; ")
        );
        ensure!(
            pipelines
                .iter()
                .any(|pipeline| pipeline.kernel == PredictionKernel::SingleRow),
            "GPU could not compile a single-row prediction pipeline: {}",
            pipeline_errors.join("; ")
        );
        let wide_lane_pipeline = pipelines.iter().any(|pipeline| {
            pipeline.kernel == PredictionKernel::SingleRow && pipeline.lanes_per_row == 128
        });
        let legacy_lane_pipeline = pipelines.iter().any(|pipeline| {
            pipeline.kernel == PredictionKernel::SingleRow && pipeline.lanes_per_row == 64
        });
        let shared_rows_pipeline = pipelines
            .iter()
            .any(|pipeline| pipeline.kernel == PredictionKernel::SharedRows2);
        let default_lanes_per_row = if context.info.vendor == AMD_VENDOR_ID && wide_lane_pipeline {
            128
        } else if legacy_lane_pipeline {
            64
        } else {
            128
        };
        let default_kernel = if context.info.vendor == AMD_VENDOR_ID
            && precision == GpuPrecision::StateF16FullF32
            && shared_rows_pipeline
        {
            PredictionKernel::SharedRows2
        } else {
            PredictionKernel::SingleRow
        };

        let state_start = Instant::now();
        let (card_states, card_bytes) = StateArena::new(
            &context,
            "rwkv-srs card FP16 states",
            rnn.weights.rwkv_modules[0].blocks.len(),
            &rnn.card_states,
            &rnn.flat_cpu_state.card_states,
        )?;
        let (deck_states, deck_bytes) = StateArena::new(
            &context,
            "rwkv-srs deck FP16 states",
            rnn.weights.rwkv_modules[1].blocks.len(),
            &rnn.deck_states,
            &rnn.flat_cpu_state.deck_states,
        )?;
        let (note_states, note_bytes) = StateArena::new(
            &context,
            "rwkv-srs note FP16 states",
            rnn.weights.rwkv_modules[2].blocks.len(),
            &rnn.note_states,
            &rnn.flat_cpu_state.note_states,
        )?;
        let (preset_states, preset_bytes) = StateArena::new(
            &context,
            "rwkv-srs preset FP16 states",
            rnn.weights.rwkv_modules[3].blocks.len(),
            &rnn.preset_states,
            &rnn.flat_cpu_state.preset_states,
        )?;
        let (global_states, global_bytes) = StateArena::singleton(
            &context,
            "rwkv-srs global FP16 state",
            rnn.weights.rwkv_modules[4].blocks.len(),
            rnn.global_state.as_ref(),
            rnn.flat_cpu_state.global_state.as_ref(),
        )?;
        context
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("initial FP16 state upload did not complete")?;
        let initial_state_upload_ns = state_start.elapsed().as_nanos();
        let initial_state_upload_bytes =
            card_bytes + deck_bytes + note_bytes + preset_bytes + global_bytes;
        let initialization_ns = initialization_start.elapsed().as_nanos();
        Ok(Self {
            context,
            precision,
            pipelines,
            default_kernel,
            default_lanes_per_row,
            wide_lane_pipeline,
            weight_buffer,
            layer_metadata_buffer,
            timestamp_query,
            timestamp_resolve_buffer,
            card_states,
            deck_states,
            note_states,
            preset_states,
            global_states,
            slot_cache: HashMap::new(),
            profile: GpuProfile {
                initialization_ns,
                weight_upload_bytes,
                fp16_weight_upload_bytes,
                fp32_weight_upload_bytes,
                normalization_weight_upload_bytes,
                initial_state_upload_bytes,
                initial_state_upload_ns,
                ..Default::default()
            },
        })
    }

    fn query_slots(&mut self, ids: ReviewIds, use_cache: bool) -> ([u32; 5], bool) {
        if use_cache {
            if let Some(slots) = self.slot_cache.get(&ids).copied() {
                return (slots, true);
            }
        }
        let slots = [
            self.card_states.slot(ids.0),
            self.deck_states.slot(ids.2),
            self.note_states.slot(ids.1),
            self.preset_states.slot(ids.3),
            self.global_states.slot(0),
        ];
        if use_cache && slots.iter().all(|slot| *slot != INVALID_OFFSET) {
            self.slot_cache.insert(ids, slots);
        }
        (slots, false)
    }

    fn predict(&mut self, features: &[Vec<f32>], ids: &[ReviewIds]) -> Result<Vec<f64>> {
        ensure!(
            features.len() == ids.len(),
            "feature and identity row counts do not match"
        );
        if features.is_empty() {
            return Ok(Vec::new());
        }
        let pending = self.submit_prediction(features, ids, false)?;
        self.collect_prediction(pending)
    }

    fn submit_prediction(
        &mut self,
        features: &[Vec<f32>],
        ids: &[ReviewIds],
        pipelined: bool,
    ) -> Result<PendingGpuPrediction> {
        let total_start = Instant::now();
        ensure!(
            features.len() == ids.len(),
            "feature and identity row counts do not match"
        );
        ensure!(
            !features.is_empty(),
            "cannot submit an empty GPU prediction"
        );
        let configured_lanes_per_row = std::env::var("RWKV_SRS_GPU_LANES_PER_ROW")
            .ok()
            .and_then(|value| value.parse::<usize>().ok());
        let forced_rows_per_group = std::env::var("RWKV_SRS_GPU_ROWS_PER_GROUP")
            .ok()
            .and_then(|value| value.parse::<usize>().ok());
        let requested_kernel = select_prediction_kernel(
            self.default_kernel,
            PredictionKernel::from_env()?,
            configured_lanes_per_row.is_some() || forced_rows_per_group.is_some(),
            features.len(),
        );
        let compiled_shapes = self
            .pipelines
            .iter()
            .filter(|variant| variant.kernel == PredictionKernel::SingleRow)
            .map(|variant| (variant.lanes_per_row, variant.rows_per_group))
            .collect::<Vec<_>>();
        let variant_index = match requested_kernel {
            PredictionKernel::SharedRows2 => self
                .pipelines
                .iter()
                .position(|variant| variant.kernel == PredictionKernel::SharedRows2)
                .ok_or_else(|| {
                    anyhow!("GPU prediction kernel shared-rows-2 is unavailable on this adapter")
                })?,
            PredictionKernel::SingleRow => {
                let forced_lanes_per_row = configured_lanes_per_row.unwrap_or_else(|| {
                    if self.precision == GpuPrecision::Mixed
                        && self.default_lanes_per_row == 128
                        && AMD_LEGACY_LANE_ROWS.contains(&features.len())
                        && self.pipelines.iter().any(|pipeline| {
                            pipeline.kernel == PredictionKernel::SingleRow
                                && pipeline.lanes_per_row == 64
                        })
                    {
                        64
                    } else {
                        self.default_lanes_per_row
                    }
                });
                let (_, rows_per_group) = select_prediction_shape(
                    &compiled_shapes,
                    forced_lanes_per_row,
                    forced_rows_per_group,
                    features.len(),
                )
                .ok_or_else(|| {
                    anyhow!(
                        "GPU prediction shape for {forced_lanes_per_row} lanes/row is unavailable; compiled variants are {compiled_shapes:?}"
                    )
                })?;
                self.pipelines
                    .iter()
                    .position(|variant| {
                        variant.kernel == PredictionKernel::SingleRow
                            && variant.lanes_per_row == forced_lanes_per_row
                            && variant.rows_per_group == rows_per_group
                    })
                    .ok_or_else(|| {
                        anyhow!(
                            "GPU prediction shape {forced_lanes_per_row} lanes/row x {rows_per_group} rows/workgroup is unavailable; compiled variants are {compiled_shapes:?}"
                        )
                    })?
            }
        };
        let rows_per_group = self.pipelines[variant_index].rows_per_group;
        let padded_rows = features.len().div_ceil(rows_per_group) * rows_per_group;
        let workgroups = padded_rows / rows_per_group;
        ensure!(
            workgroups <= self.context.limits.max_compute_workgroups_per_dimension as usize,
            "GPU prediction batch of {} requires {workgroups} workgroups, exceeding device limit {}",
            features.len(),
            self.context.limits.max_compute_workgroups_per_dimension
        );
        for (row_index, row) in features.iter().enumerate() {
            ensure!(
                row.len() == FEATURE_DIM,
                "GPU feature row {row_index} has {} values, expected {FEATURE_DIM}",
                row.len()
            );
        }
        let slot_resolution_start = Instant::now();
        let mut slots = Vec::with_capacity(padded_rows * 5);
        let use_slot_cache = std::env::var("RWKV_SRS_GPU_SLOT_CACHE").as_deref() != Ok("0");
        if use_slot_cache {
            self.slot_cache.reserve(ids.len());
        }
        let mut slot_cache_hits = 0u64;
        let mut slot_cache_misses = 0u64;
        for (row_index, ids) in ids.iter().copied().enumerate() {
            let (row_slots, cache_hit) = self.query_slots(ids, use_slot_cache);
            if cache_hit {
                slot_cache_hits += 1;
            } else {
                slot_cache_misses += 1;
            }
            ensure!(
                row_slots.iter().all(|slot| *slot != INVALID_OFFSET),
                "GPU prediction row {row_index} is missing a recurrent state slot"
            );
            slots.extend(row_slots);
        }
        slots.resize(padded_rows * 5, INVALID_OFFSET);
        let state_slot_resolution_ns = slot_resolution_start.elapsed().as_nanos();
        self.profile.state_slot_resolution_ns += state_slot_resolution_ns;

        let upload_start = Instant::now();
        let feature_buffer = if self.precision.fp32_inputs() {
            let mut feature_values = Vec::with_capacity(padded_rows * FEATURE_DIM);
            for row in features {
                feature_values.extend_from_slice(row);
            }
            feature_values.resize(padded_rows * FEATURE_DIM, 0.0);
            self.context
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("rwkv-srs FP32 prediction features"),
                    contents: bytemuck::cast_slice(&feature_values),
                    usage: wgpu::BufferUsages::STORAGE,
                })?
        } else {
            let mut feature_values = Vec::with_capacity(padded_rows * FEATURE_DIM);
            for row in features {
                feature_values.extend(row.iter().copied().map(f16::from_f32));
            }
            feature_values.resize(padded_rows * FEATURE_DIM, f16::ZERO);
            self.context
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("rwkv-srs FP16 prediction features"),
                    contents: bytemuck::cast_slice(&feature_values),
                    usage: wgpu::BufferUsages::STORAGE,
                })?
        };
        let slot_buffer = self
            .context
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rwkv-srs prediction state slots"),
                contents: bytemuck::cast_slice(&slots),
                usage: wgpu::BufferUsages::STORAGE,
            })?;
        let output_size = padded_rows * std::mem::size_of::<f32>();
        let output_buffer = self.context.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rwkv-srs probabilities"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })?;
        let timestamp_bytes = if self.timestamp_query.is_some() {
            2 * std::mem::size_of::<u64>()
        } else {
            0
        };
        let timestamp_offset = output_size.next_multiple_of(std::mem::align_of::<u64>());
        let readback_buffer = self.context.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rwkv-srs prediction readback"),
            size: (timestamp_offset + timestamp_bytes) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })?;
        let feature_upload_ns = upload_start.elapsed().as_nanos();
        self.profile.feature_upload_ns += feature_upload_ns;

        let command_setup_start = Instant::now();
        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rwkv-srs fused predictor bindings"),
                layout: &self.pipelines[variant_index]
                    .pipeline
                    .get_bind_group_layout(0),
                entries: &[
                    binding(0, &self.weight_buffer),
                    binding(1, &feature_buffer),
                    binding(2, &slot_buffer),
                    binding(3, &output_buffer),
                    binding(4, &self.card_states.buffer),
                    binding(5, &self.deck_states.buffer),
                    binding(6, &self.note_states.buffer),
                    binding(7, &self.preset_states.buffer),
                    binding(8, &self.global_states.buffer),
                    binding(9, &self.layer_metadata_buffer),
                ],
            });
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rwkv-srs fused prediction"),
                });
        {
            let timestamp_writes =
                self.timestamp_query
                    .as_ref()
                    .map(|query_set| wgpu::ComputePassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rwkv-srs fused prediction"),
                timestamp_writes,
            });
            pass.set_pipeline(&self.pipelines[variant_index].pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_size as u64);
        if let (Some(query), Some(resolve)) = (
            self.timestamp_query.as_ref(),
            self.timestamp_resolve_buffer.as_ref(),
        ) {
            encoder.resolve_query_set(query, 0..2, resolve, 0);
            encoder.copy_buffer_to_buffer(
                resolve,
                0,
                &readback_buffer,
                timestamp_offset as u64,
                timestamp_bytes as u64,
            );
        }
        let command_buffer = encoder.finish();
        let command_setup_ns = command_setup_start.elapsed().as_nanos();
        self.profile.command_setup_ns += command_setup_ns;
        let submit_start = Instant::now();
        let submission_index = self.context.queue.submit([command_buffer]);
        let (sender, receiver) = mpsc::sync_channel(1);
        readback_buffer
            .clone()
            .map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = sender.send(result);
            });
        let feature_bytes_per_value = if self.precision.fp32_inputs() {
            std::mem::size_of::<f32>()
        } else {
            std::mem::size_of::<f16>()
        };
        let transient_bytes = padded_rows
            .checked_mul(FEATURE_DIM)
            .and_then(|values| values.checked_mul(feature_bytes_per_value))
            .and_then(|bytes| bytes.checked_add(padded_rows * 5 * std::mem::size_of::<u32>()))
            .and_then(|bytes| bytes.checked_add(output_size))
            .and_then(|bytes| bytes.checked_add(timestamp_offset + timestamp_bytes))
            .context("GPU prediction transient byte count overflow")?
            as u64;
        if pipelined {
            self.profile.prediction_pipeline_submitted_chunks += 1;
            self.profile.last_prediction_pipeline_submitted_chunks += 1;
        }
        Ok(PendingGpuPrediction {
            _feature_buffer: feature_buffer,
            _slot_buffer: slot_buffer,
            _output_buffer: output_buffer,
            readback_buffer,
            receiver,
            submission_index,
            total_start,
            submit_start,
            state_slot_resolution_ns,
            feature_upload_ns,
            command_setup_ns,
            rows: features.len(),
            padded_rows,
            rows_per_group,
            lanes_per_row: self.pipelines[variant_index].lanes_per_row,
            workgroup_size: self.pipelines[variant_index].workgroup_size,
            shared_rows_kernel: self.pipelines[variant_index].kernel
                == PredictionKernel::SharedRows2,
            slot_cache_hits,
            slot_cache_misses,
            output_size,
            timestamp_offset,
            timestamp_bytes,
            transient_bytes,
            pipelined,
        })
    }

    fn collect_prediction(&mut self, pending: PendingGpuPrediction) -> Result<Vec<f64>> {
        let blocked_wait_start = Instant::now();
        let poll_result = self
            .context
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(pending.submission_index.clone()),
                timeout: None,
            })
            .context("GPU prediction readback did not complete");
        let map_result = if poll_result.is_ok() {
            pending
                .receiver
                .recv()
                .context("GPU prediction readback callback was not invoked")
                .and_then(|result| result.context("GPU prediction readback failed"))
        } else {
            Ok(())
        };
        let blocked_wait_ns = blocked_wait_start.elapsed().as_nanos();
        if pending.pipelined {
            self.profile.prediction_pipeline_blocked_wait_ns += blocked_wait_ns;
            self.profile.last_prediction_pipeline_blocked_wait_ns += blocked_wait_ns;
        }
        if let Err(error) = poll_result {
            pending.readback_buffer.unmap();
            return Err(error);
        }
        if let Err(error) = map_result {
            pending.readback_buffer.unmap();
            return Err(error);
        }
        let bytes = match pending.readback_buffer.get_mapped_range(..) {
            Ok(mapped) => mapped.to_vec(),
            Err(error) => {
                pending.readback_buffer.unmap();
                return Err(error).context("could not access mapped GPU prediction readback");
            }
        };
        pending.readback_buffer.unmap();

        let submit_to_readback_ns = pending.submit_start.elapsed().as_nanos();
        self.profile.submit_to_readback_ns += submit_to_readback_ns;
        self.profile.prediction_calls += 1;
        self.profile.prediction_rows += pending.rows as u64;
        let prediction_total_ns = pending.total_start.elapsed().as_nanos();
        if !pending.pipelined {
            self.profile.prediction_total_ns += prediction_total_ns;
        }
        self.profile.last_prediction_total_ns = prediction_total_ns;
        self.profile.last_state_slot_resolution_ns = pending.state_slot_resolution_ns;
        self.profile.last_feature_upload_ns = pending.feature_upload_ns;
        self.profile.last_command_setup_ns = pending.command_setup_ns;
        self.profile.last_submit_to_readback_ns = submit_to_readback_ns;
        self.profile.last_prediction_rows = pending.rows as u64;
        self.profile.last_padded_rows = pending.padded_rows as u64;
        self.profile.last_rows_per_workgroup = pending.rows_per_group as u64;
        self.profile.last_lanes_per_row = pending.lanes_per_row as u64;
        self.profile.last_workgroup_size = pending.workgroup_size as u64;
        self.profile.last_shared_rows_kernel = pending.shared_rows_kernel;
        self.profile.slot_cache_hits += pending.slot_cache_hits;
        self.profile.slot_cache_misses += pending.slot_cache_misses;
        self.profile.last_slot_cache_hits = pending.slot_cache_hits;
        self.profile.last_slot_cache_misses = pending.slot_cache_misses;
        if pending.timestamp_bytes != 0 {
            let timestamp_values = bytemuck::try_cast_slice::<u8, u64>(
                &bytes
                    [pending.timestamp_offset..pending.timestamp_offset + pending.timestamp_bytes],
            )
            .map_err(|error| anyhow!("malformed GPU timestamp buffer: {error:?}"))?;
            let ticks = timestamp_values[1].wrapping_sub(timestamp_values[0]);
            let kernel_ns = (ticks as f64 * f64::from(self.context.queue.get_timestamp_period()))
                .round() as u128;
            self.profile.kernel_ns += kernel_ns;
            self.profile.kernel_timestamp_samples += 1;
            self.profile.last_kernel_ns = Some(kernel_ns);
        } else {
            self.profile.last_kernel_ns = None;
        }
        let values = bytemuck::try_cast_slice::<u8, f32>(&bytes[..pending.output_size])
            .map_err(|error| anyhow!("malformed GPU probability buffer: {error:?}"))?;
        let result_conversion_start = Instant::now();
        let probabilities = values[..pending.rows]
            .iter()
            .map(|value| f64::from(*value))
            .collect();
        let result_conversion_ns = result_conversion_start.elapsed().as_nanos();
        self.profile.result_conversion_ns += result_conversion_ns;
        self.profile.last_result_conversion_ns = result_conversion_ns;
        Ok(probabilities)
    }

    #[allow(clippy::too_many_arguments)]
    fn update_review_states(
        &mut self,
        card_id: i64,
        card: &NativeRnnModuleState,
        note_id: i64,
        note: &NativeRnnModuleState,
        deck_id: i64,
        deck: &NativeRnnModuleState,
        preset_id: i64,
        preset: &NativeRnnModuleState,
        global: &NativeRnnModuleState,
    ) -> Result<()> {
        let start = Instant::now();
        let (card_bytes, _) = self.card_states.update(&self.context, card_id, card)?;
        let (deck_bytes, _) = self.deck_states.update(&self.context, deck_id, deck)?;
        let (note_bytes, _) = self.note_states.update(&self.context, note_id, note)?;
        let (preset_bytes, _) = self
            .preset_states
            .update(&self.context, preset_id, preset)?;
        let (global_bytes, _) = self.global_states.update(&self.context, 0, global)?;
        self.context.queue.submit([]);
        self.profile.update_count += 1;
        self.profile.update_bytes +=
            card_bytes + deck_bytes + note_bytes + preset_bytes + global_bytes;
        self.profile.update_ns += start.elapsed().as_nanos();
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn restore_review_states_after_undo(
        &mut self,
        card_id: i64,
        card: Option<&NativeRnnModuleState>,
        note_id: i64,
        note: Option<&NativeRnnModuleState>,
        deck_id: i64,
        deck: Option<&NativeRnnModuleState>,
        preset_id: i64,
        preset: Option<&NativeRnnModuleState>,
        global: Option<&NativeRnnModuleState>,
    ) -> Result<()> {
        let start = Instant::now();
        let card_undo = self
            .card_states
            .restore_after_undo(&self.context, card_id, card)?;
        let deck_undo = self
            .deck_states
            .restore_after_undo(&self.context, deck_id, deck)?;
        let note_undo = self
            .note_states
            .restore_after_undo(&self.context, note_id, note)?;
        let preset_undo =
            self.preset_states
                .restore_after_undo(&self.context, preset_id, preset)?;
        let global_undo = self
            .global_states
            .restore_after_undo(&self.context, 0, global)?;

        let uploaded_bytes = card_undo.uploaded_bytes
            + deck_undo.uploaded_bytes
            + note_undo.uploaded_bytes
            + preset_undo.uploaded_bytes
            + global_undo.uploaded_bytes;
        if uploaded_bytes != 0 {
            // Queue submission order guarantees that these reverse writes run
            // after the process writes and before a subsequent prediction.
            self.context.queue.submit([]);
        }

        if global_undo.topology_changed {
            self.slot_cache.clear();
        } else if card_undo.topology_changed
            || note_undo.topology_changed
            || deck_undo.topology_changed
            || preset_undo.topology_changed
        {
            self.slot_cache.retain(
                |&(cached_card, cached_note, cached_deck, cached_preset), _| {
                    (!card_undo.topology_changed || cached_card != card_id)
                        && (!note_undo.topology_changed || cached_note != note_id)
                        && (!deck_undo.topology_changed || cached_deck != deck_id)
                        && (!preset_undo.topology_changed || cached_preset != preset_id)
                },
            );
        }

        self.profile.undo_update_count += 1;
        self.profile.undo_update_bytes += uploaded_bytes;
        self.profile.undo_retired_slots += [
            card_undo.retired_slot,
            note_undo.retired_slot,
            deck_undo.retired_slot,
            preset_undo.retired_slot,
            global_undo.retired_slot,
        ]
        .into_iter()
        .filter(|retired| *retired)
        .count() as u64;
        self.profile.undo_update_ns += start.elapsed().as_nanos();
        Ok(())
    }

    fn to_pydict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("device_name", &self.context.info.name)?;
        dict.set_item("backend", format!("{:?}", self.context.info.backend))?;
        dict.set_item(
            "device_type",
            format!("{:?}", self.context.info.device_type),
        )?;
        dict.set_item("driver", &self.context.info.driver)?;
        dict.set_item("driver_info", &self.context.info.driver_info)?;
        let default_variant = self
            .pipelines
            .iter()
            .find(|variant| {
                variant.kernel == self.default_kernel
                    && variant.lanes_per_row == self.default_lanes_per_row
            })
            .expect("default GPU lane shape has a compiled pipeline");
        dict.set_item("prediction_kernel", self.default_kernel.name())?;
        let mut kernel_variants = self
            .pipelines
            .iter()
            .map(|variant| variant.kernel.name())
            .collect::<Vec<_>>();
        kernel_variants.sort_unstable();
        kernel_variants.dedup();
        dict.set_item("prediction_kernel_variants", kernel_variants)?;
        dict.set_item(
            "shared_rows_pipeline",
            self.pipelines
                .iter()
                .any(|variant| variant.kernel == PredictionKernel::SharedRows2),
        )?;
        dict.set_item("rows_per_workgroup", default_variant.rows_per_group)?;
        dict.set_item("lanes_per_row", self.default_lanes_per_row)?;
        dict.set_item("wide_lane_pipeline", self.wide_lane_pipeline)?;
        dict.set_item(
            "wide_lane_legacy_rows",
            (*AMD_LEGACY_LANE_ROWS.start(), *AMD_LEGACY_LANE_ROWS.end()),
        )?;
        let mut row_tile_variants = self
            .pipelines
            .iter()
            .filter(|variant| {
                variant.kernel == self.default_kernel
                    && variant.lanes_per_row == self.default_lanes_per_row
            })
            .map(|variant| variant.rows_per_group)
            .collect::<Vec<_>>();
        if self.default_kernel == PredictionKernel::SharedRows2 {
            for rows in self
                .pipelines
                .iter()
                .filter(|variant| {
                    variant.kernel == PredictionKernel::SingleRow
                        && variant.lanes_per_row == self.default_lanes_per_row
                })
                .map(|variant| variant.rows_per_group)
            {
                if !row_tile_variants.contains(&rows) {
                    row_tile_variants.push(rows);
                }
            }
        }
        dict.set_item("row_tile_variants", row_tile_variants)?;
        dict.set_item("shared_rows_min_rows", SHARED_ROWS_MIN_ROWS)?;
        dict.set_item(
            "pipeline_shapes",
            self.pipelines
                .iter()
                .filter(|variant| variant.kernel == PredictionKernel::SingleRow)
                .map(|variant| (variant.lanes_per_row, variant.rows_per_group))
                .collect::<Vec<_>>(),
        )?;
        dict.set_item("precision", self.precision.name())?;
        dict.set_item("state_storage", "float16")?;
        dict.set_item(
            "weight_storage",
            if self.precision.fp32_inputs() {
                "float32"
            } else {
                "mixed"
            },
        )?;
        dict.set_item(
            "fp16_weight_storage",
            if self.precision.fp32_inputs() {
                "unused"
            } else {
                "float16"
            },
        )?;
        dict.set_item(
            "feature_storage",
            if self.precision.fp32_inputs() {
                "float32"
            } else {
                "float16"
            },
        )?;
        dict.set_item(
            "arithmetic",
            if self.precision.full_f32_math() {
                "float32"
            } else {
                "mixed"
            },
        )?;
        dict.set_item("normalization_weight_storage", "float32")?;
        dict.set_item(
            "matrix_input_quantization",
            if self.precision.full_f32_math() {
                "float32"
            } else {
                "float16"
            },
        )?;
        dict.set_item("matrix_accumulation", "float32")?;
        dict.set_item("activation_carriers", "float32")?;
        dict.set_item("initialization_ns", self.profile.initialization_ns)?;
        dict.set_item("weight_upload_bytes", self.profile.weight_upload_bytes)?;
        dict.set_item(
            "fp16_weight_upload_bytes",
            self.profile.fp16_weight_upload_bytes,
        )?;
        dict.set_item(
            "fp32_weight_upload_bytes",
            self.profile.fp32_weight_upload_bytes,
        )?;
        dict.set_item(
            "normalization_weight_upload_bytes",
            self.profile.normalization_weight_upload_bytes,
        )?;
        dict.set_item(
            "initial_state_upload_bytes",
            self.profile.initial_state_upload_bytes,
        )?;
        dict.set_item(
            "initial_state_upload_ns",
            self.profile.initial_state_upload_ns,
        )?;
        dict.set_item("update_count", self.profile.update_count)?;
        dict.set_item("update_bytes", self.profile.update_bytes)?;
        dict.set_item("update_enqueue_ns", self.profile.update_ns)?;
        dict.set_item("undo_update_count", self.profile.undo_update_count)?;
        dict.set_item("undo_update_bytes", self.profile.undo_update_bytes)?;
        dict.set_item("undo_update_enqueue_ns", self.profile.undo_update_ns)?;
        dict.set_item("undo_retired_slots", self.profile.undo_retired_slots)?;
        dict.set_item("synchronization_count", self.profile.synchronization_count)?;
        dict.set_item(
            "synchronization_wait_ns",
            self.profile.synchronization_wait_ns,
        )?;
        dict.set_item(
            "last_synchronization_wait_ns",
            self.profile.last_synchronization_wait_ns,
        )?;
        dict.set_item("prediction_calls", self.profile.prediction_calls)?;
        dict.set_item("prediction_rows", self.profile.prediction_rows)?;
        dict.set_item(
            "prediction_pipeline_enabled",
            gpu_prediction_pipeline_enabled().map_err(super::py_value_error)?,
        )?;
        dict.set_item("prediction_pipeline_max_pending_chunks", 2)?;
        dict.set_item(
            "prediction_pipeline_calls",
            self.profile.prediction_pipeline_calls,
        )?;
        dict.set_item(
            "prediction_pipeline_submitted_chunks",
            self.profile.prediction_pipeline_submitted_chunks,
        )?;
        dict.set_item(
            "prediction_pipeline_blocked_wait_ns",
            self.profile.prediction_pipeline_blocked_wait_ns,
        )?;
        dict.set_item(
            "prediction_pipeline_overlap_ns",
            self.profile.prediction_pipeline_overlap_ns,
        )?;
        dict.set_item(
            "prediction_pipeline_fallbacks",
            self.profile.prediction_pipeline_fallbacks,
        )?;
        dict.set_item(
            "prediction_pipeline_total_ns",
            self.profile.prediction_pipeline_total_ns,
        )?;
        dict.set_item(
            "prediction_pipeline_high_water_bytes",
            self.profile.prediction_pipeline_high_water_bytes,
        )?;
        dict.set_item(
            "last_prediction_pipeline_submitted_chunks",
            self.profile.last_prediction_pipeline_submitted_chunks,
        )?;
        dict.set_item(
            "last_prediction_pipeline_blocked_wait_ns",
            self.profile.last_prediction_pipeline_blocked_wait_ns,
        )?;
        dict.set_item(
            "last_prediction_pipeline_overlap_ns",
            self.profile.last_prediction_pipeline_overlap_ns,
        )?;
        dict.set_item(
            "last_prediction_pipeline_fallbacks",
            self.profile.last_prediction_pipeline_fallbacks,
        )?;
        dict.set_item(
            "last_prediction_pipeline_total_ns",
            self.profile.last_prediction_pipeline_total_ns,
        )?;
        dict.set_item(
            "last_prediction_pipeline_high_water_bytes",
            self.profile.last_prediction_pipeline_high_water_bytes,
        )?;
        dict.set_item("host_input_parse_ns", self.profile.host_input_parse_ns)?;
        dict.set_item(
            "host_scope_validation_ns",
            self.profile.host_scope_validation_ns,
        )?;
        dict.set_item("host_feature_build_ns", self.profile.host_feature_build_ns)?;
        dict.set_item("host_fallback_rows", self.profile.host_fallback_rows)?;
        dict.set_item(
            "last_host_input_parse_ns",
            self.profile.last_host_input_parse_ns,
        )?;
        dict.set_item(
            "last_host_scope_validation_ns",
            self.profile.last_host_scope_validation_ns,
        )?;
        dict.set_item(
            "last_host_feature_build_ns",
            self.profile.last_host_feature_build_ns,
        )?;
        dict.set_item(
            "last_host_fallback_rows",
            self.profile.last_host_fallback_rows,
        )?;
        dict.set_item("prediction_total_ns", self.profile.prediction_total_ns)?;
        dict.set_item(
            "state_slot_resolution_ns",
            self.profile.state_slot_resolution_ns,
        )?;
        dict.set_item("feature_upload_ns", self.profile.feature_upload_ns)?;
        dict.set_item("command_setup_ns", self.profile.command_setup_ns)?;
        dict.set_item("submit_to_readback_ns", self.profile.submit_to_readback_ns)?;
        dict.set_item("kernel_ns", self.profile.kernel_ns)?;
        dict.set_item("result_conversion_ns", self.profile.result_conversion_ns)?;
        dict.set_item("python_result_ns", self.profile.python_result_ns)?;
        dict.set_item(
            "python_binding_total_ns",
            self.profile.python_binding_total_ns,
        )?;
        dict.set_item(
            "kernel_timestamp_samples",
            self.profile.kernel_timestamp_samples,
        )?;
        dict.set_item(
            "last_prediction_total_ns",
            self.profile.last_prediction_total_ns,
        )?;
        dict.set_item(
            "last_state_slot_resolution_ns",
            self.profile.last_state_slot_resolution_ns,
        )?;
        dict.set_item(
            "last_feature_upload_ns",
            self.profile.last_feature_upload_ns,
        )?;
        dict.set_item("last_command_setup_ns", self.profile.last_command_setup_ns)?;
        dict.set_item(
            "last_submit_to_readback_ns",
            self.profile.last_submit_to_readback_ns,
        )?;
        dict.set_item("last_kernel_ns", self.profile.last_kernel_ns)?;
        dict.set_item(
            "last_result_conversion_ns",
            self.profile.last_result_conversion_ns,
        )?;
        dict.set_item("last_python_result_ns", self.profile.last_python_result_ns)?;
        dict.set_item(
            "last_python_binding_total_ns",
            self.profile.last_python_binding_total_ns,
        )?;
        dict.set_item("last_prediction_rows", self.profile.last_prediction_rows)?;
        dict.set_item("last_padded_rows", self.profile.last_padded_rows)?;
        dict.set_item(
            "last_rows_per_workgroup",
            self.profile.last_rows_per_workgroup,
        )?;
        dict.set_item("last_lanes_per_row", self.profile.last_lanes_per_row)?;
        dict.set_item("last_workgroup_size", self.profile.last_workgroup_size)?;
        dict.set_item(
            "last_prediction_kernel",
            if self.profile.last_shared_rows_kernel {
                PredictionKernel::SharedRows2.name()
            } else {
                PredictionKernel::SingleRow.name()
            },
        )?;
        dict.set_item("slot_cache_entries", self.slot_cache.len())?;
        dict.set_item("slot_cache_hits", self.profile.slot_cache_hits)?;
        dict.set_item("slot_cache_misses", self.profile.slot_cache_misses)?;
        dict.set_item("last_slot_cache_hits", self.profile.last_slot_cache_hits)?;
        dict.set_item(
            "last_slot_cache_misses",
            self.profile.last_slot_cache_misses,
        )?;
        dict.set_item("adaptive_splits", self.profile.adaptive_splits)?;
        dict.set_item("oom_retries", self.profile.oom_retries)?;

        let states = PyDict::new_bound(py);
        for (name, arena) in [
            ("card", &self.card_states),
            ("deck", &self.deck_states),
            ("note", &self.note_states),
            ("preset", &self.preset_states),
            ("global", &self.global_states),
        ] {
            let value = PyDict::new_bound(py);
            value.set_item("entries", arena.slots.len())?;
            value.set_item("capacity", arena.capacity)?;
            value.set_item("entry_bytes", arena.entry_bytes())?;
            value.set_item("allocated_bytes", arena.allocated_bytes())?;
            states.set_item(name, value)?;
        }
        dict.set_item("states", states)?;
        Ok(dict.unbind())
    }
}

fn binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

impl NativeRnn {
    pub(super) fn recover_gpu_prediction_oom(&mut self) -> Result<()> {
        if let Some(predictor) = self.gpu.as_mut() {
            predictor.profile.adaptive_splits += 1;
            predictor.profile.oom_retries += 1;
            predictor
                .context
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .context("GPU prediction allocation recovery did not complete")?;
        }
        Ok(())
    }

    pub(super) fn record_gpu_host_preparation(
        &mut self,
        input_parse_ns: u128,
        scope_validation_ns: u128,
        feature_build_ns: u128,
        fallback_rows: u64,
    ) {
        let Some(predictor) = self.gpu.as_mut() else {
            return;
        };
        predictor.profile.host_input_parse_ns += input_parse_ns;
        predictor.profile.host_scope_validation_ns += scope_validation_ns;
        predictor.profile.host_feature_build_ns += feature_build_ns;
        predictor.profile.host_fallback_rows += fallback_rows;
        predictor.profile.last_host_input_parse_ns = input_parse_ns;
        predictor.profile.last_host_scope_validation_ns = scope_validation_ns;
        predictor.profile.last_host_feature_build_ns = feature_build_ns;
        predictor.profile.last_host_fallback_rows = fallback_rows;
    }

    pub(super) fn record_gpu_python_result(
        &mut self,
        python_result_ns: u128,
        python_binding_total_ns: u128,
    ) {
        let Some(predictor) = self.gpu.as_mut() else {
            return;
        };
        predictor.profile.python_result_ns += python_result_ns;
        predictor.profile.python_binding_total_ns += python_binding_total_ns;
        predictor.profile.last_python_result_ns = python_result_ns;
        predictor.profile.last_python_binding_total_ns = python_binding_total_ns;
    }

    pub(super) fn ensure_gpu(&mut self) -> Result<()> {
        if self.gpu.is_none() {
            self.gpu = Some(GpuPredictor::new(self)?);
        }
        Ok(())
    }

    pub(super) fn predict_gpu(
        &mut self,
        features: &[Vec<f32>],
        ids: &[ReviewIds],
    ) -> Result<Vec<f64>> {
        self.ensure_gpu()?;
        self.gpu
            .as_mut()
            .expect("GPU predictor initialized")
            .predict(features, ids)
    }

    pub(super) fn begin_gpu_prediction_pipeline(&mut self) -> Result<()> {
        self.ensure_gpu()?;
        let predictor = self.gpu.as_mut().expect("GPU predictor initialized");
        predictor.profile.prediction_pipeline_calls += 1;
        Self::reset_gpu_prediction_pipeline_last_profile(predictor);
        Ok(())
    }

    fn reset_gpu_prediction_pipeline_last_profile(predictor: &mut GpuPredictor) {
        predictor.profile.last_prediction_pipeline_submitted_chunks = 0;
        predictor.profile.last_prediction_pipeline_blocked_wait_ns = 0;
        predictor.profile.last_prediction_pipeline_overlap_ns = 0;
        predictor.profile.last_prediction_pipeline_fallbacks = 0;
        predictor.profile.last_prediction_pipeline_total_ns = 0;
        predictor.profile.last_prediction_pipeline_high_water_bytes = 0;
    }

    pub(super) fn reset_gpu_prediction_pipeline_last(&mut self) {
        let Some(predictor) = self.gpu.as_mut() else {
            return;
        };
        Self::reset_gpu_prediction_pipeline_last_profile(predictor);
    }

    pub(super) fn submit_gpu_prediction(
        &mut self,
        features: &[Vec<f32>],
        ids: &[ReviewIds],
    ) -> Result<PendingGpuPrediction> {
        self.ensure_gpu()?;
        self.gpu
            .as_mut()
            .expect("GPU predictor initialized")
            .submit_prediction(features, ids, true)
    }

    pub(super) fn collect_gpu_prediction(
        &mut self,
        pending: PendingGpuPrediction,
    ) -> Result<Vec<f64>> {
        self.gpu
            .as_mut()
            .context("GPU predictor was released with a pending prediction")?
            .collect_prediction(pending)
    }

    pub(super) fn record_gpu_prediction_pipeline_high_water(&mut self, bytes: u64) {
        let Some(predictor) = self.gpu.as_mut() else {
            return;
        };
        predictor.profile.prediction_pipeline_high_water_bytes = predictor
            .profile
            .prediction_pipeline_high_water_bytes
            .max(bytes);
        predictor.profile.last_prediction_pipeline_high_water_bytes = predictor
            .profile
            .last_prediction_pipeline_high_water_bytes
            .max(bytes);
    }

    pub(super) fn record_gpu_prediction_pipeline_overlap(&mut self, overlap_ns: u128) {
        let Some(predictor) = self.gpu.as_mut() else {
            return;
        };
        predictor.profile.prediction_pipeline_overlap_ns += overlap_ns;
        predictor.profile.last_prediction_pipeline_overlap_ns += overlap_ns;
    }

    pub(super) fn record_gpu_prediction_pipeline_fallback(&mut self) {
        let Some(predictor) = self.gpu.as_mut() else {
            return;
        };
        predictor.profile.prediction_pipeline_fallbacks += 1;
        predictor.profile.last_prediction_pipeline_fallbacks += 1;
    }

    pub(super) fn finish_gpu_prediction_pipeline(&mut self, total_ns: u128, succeeded: bool) {
        let Some(predictor) = self.gpu.as_mut() else {
            return;
        };
        predictor.profile.prediction_pipeline_total_ns += total_ns;
        predictor.profile.last_prediction_pipeline_total_ns = total_ns;
        if succeeded {
            predictor.profile.prediction_total_ns += total_ns;
            predictor.profile.last_prediction_total_ns = total_ns;
        }
    }

    pub(super) fn invalidate_gpu(&mut self) {
        self.gpu = None;
    }

    pub(super) fn release_gpu(&mut self) -> bool {
        let prediction_released = self.gpu.take().is_some();
        let process_scan_released = self.gpu_process_scan.take().is_some();
        prediction_released || process_scan_released
    }

    pub(super) fn synchronize_gpu(&mut self) -> Result<u128> {
        let Some(predictor) = self.gpu.as_mut() else {
            return Ok(0);
        };
        let start = Instant::now();
        predictor
            .context
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("GPU state synchronization failed")?;
        let elapsed_ns = start.elapsed().as_nanos();
        predictor.profile.synchronization_count += 1;
        predictor.profile.synchronization_wait_ns += elapsed_ns;
        predictor.profile.last_synchronization_wait_ns = elapsed_ns;
        Ok(elapsed_ns)
    }

    pub(super) fn sync_gpu_review_states(
        &mut self,
        card_id: i64,
        note_id: i64,
        deck_id: i64,
        preset_id: i64,
    ) -> Result<()> {
        let Some(mut predictor) = self.gpu.take() else {
            return Ok(());
        };
        let updated = predictor.update_review_states(
            card_id,
            self.card_states
                .get(&card_id)
                .context("updated card state is missing")?,
            note_id,
            self.note_states
                .get(&note_id)
                .context("updated note state is missing")?,
            deck_id,
            self.deck_states
                .get(&deck_id)
                .context("updated deck state is missing")?,
            preset_id,
            self.preset_states
                .get(&preset_id)
                .context("updated preset state is missing")?,
            self.global_state
                .as_ref()
                .context("updated global state is missing")?,
        );
        if updated.is_ok() {
            self.gpu = Some(predictor);
        }
        // Canonical FP32 CPU state is already committed. Any allocation,
        // upload, or topology failure drops the optional prediction cache so
        // later GPU prediction can rebuild it from that canonical state.
        Ok(())
    }

    pub(super) fn restore_gpu_review_states_after_undo(
        &mut self,
        card_id: i64,
        note_id: i64,
        deck_id: i64,
        preset_id: i64,
    ) {
        let Some(mut predictor) = self.gpu.take() else {
            return;
        };
        let restored = predictor.restore_review_states_after_undo(
            card_id,
            self.card_states.get(&card_id),
            note_id,
            self.note_states.get(&note_id),
            deck_id,
            self.deck_states.get(&deck_id),
            preset_id,
            self.preset_states.get(&preset_id),
            self.global_state.as_ref(),
        );
        if restored.is_ok() {
            self.gpu = Some(predictor);
        }
        // Any unexpected topology or upload failure leaves canonical CPU state
        // restored and falls back to lazy full GPU reconstruction.
    }

    pub(super) fn gpu_profile_pydict(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        self.gpu
            .as_ref()
            .map(|predictor| predictor.to_pydict(py))
            .transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn state_layout_is_four_byte_aligned() {
        for layers in [2usize, 3, 4] {
            assert_eq!((layers * STATE_LAYER_ELEMENTS * 2) % 4, 0);
        }
    }

    #[test]
    fn undo_slot_retirement_is_missing_safe_and_strictly_lifo() {
        let mut slots = HashMap::from([(10i64, 0u32), (11, 1), (12, 2)]);
        assert!(!retire_tail_slot(&mut slots, 99, "test").unwrap());
        assert_eq!(slots.len(), 3);

        let error = retire_tail_slot(&mut slots, 11, "test").unwrap_err();
        assert!(error.to_string().contains("cannot retire non-tail"));
        assert_eq!(slots.len(), 3);

        assert!(retire_tail_slot(&mut slots, 12, "test").unwrap());
        assert!(retire_tail_slot(&mut slots, 11, "test").unwrap());
        assert!(retire_tail_slot(&mut slots, 10, "test").unwrap());
        assert!(slots.is_empty());
    }

    #[test]
    fn model_packer_retains_fp32_values_for_default_and_fp16_fallback() {
        let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("crate lives under rust/rwkv-srs-cpu")
            .join("tests/fixtures/models/RWKV_trained_on_101_4999.safetensors");
        let rnn = NativeRnn::from_checkpoint(model_path).unwrap();
        let packed = PackedModel::from_weights(&rnn.weights).unwrap();
        assert_eq!(packed.values.len(), packed.full_precision_values.len());
        let norm = &rnn.weights.features2card.norm;
        let expected_weight = f32_tensor_data(&norm.weight).unwrap();
        let expected_bias = f32_tensor_data(&norm.bias).unwrap();
        let weight_offset = packed.features_norm.weight as usize;
        let bias_offset = packed.features_norm.bias as usize;

        assert_eq!(
            &packed.normalization_values[weight_offset..weight_offset + 512],
            expected_weight.as_slice().unwrap()
        );
        assert_eq!(
            packed.values[bias_offset].to_f32(),
            f16::from_f32(expected_bias.as_slice().unwrap()[0]).to_f32()
        );
        assert_eq!(
            packed.full_precision_values[bias_offset],
            expected_bias.as_slice().unwrap()[0]
        );
    }

    #[test]
    fn shader_template_supports_all_portable_row_tiles() {
        let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("crate lives under rust/rwkv-srs-cpu")
            .join("tests/fixtures/models/RWKV_trained_on_101_4999.safetensors");
        let rnn = NativeRnn::from_checkpoint(model_path).unwrap();
        let packed = PackedModel::from_weights(&rnn.weights).unwrap();
        assert_eq!(packed.layer_metadata().len(), 16);

        for precision in [GpuPrecision::Mixed, GpuPrecision::StateF16FullF32] {
            for (lanes, rows) in [(64usize, 1usize), (64, 2), (64, 4), (128, 1)] {
                let source = packed
                    .shader_source(rows, lanes, precision, PredictionKernel::SingleRow)
                    .unwrap();
                assert!(!source.contains("/*__"));
                assert!(source.contains(&format!("@compute @workgroup_size({})", rows * lanes)));
                assert!(source.contains(&format!("const ROWS_PER_GROUP: u32 = {rows}u;")));
                assert!(source.contains(&format!("const LANES_PER_ROW: u32 = {lanes}u;")));
            }

            let shared = packed
                .shader_source(2, 128, precision, PredictionKernel::SharedRows2)
                .unwrap();
            assert!(!shared.contains("/*__"));
            assert!(shared.contains("fn predict_shared_rows("));
            assert!(shared.contains("fn shared_linear_tmp0("));
            assert!(shared.contains("var sum0 = 0.0;"));
            assert!(shared.contains("var sum1 = 0.0;"));
            assert!(shared.contains("@compute @workgroup_size(128)"));
        }
    }

    #[test]
    fn row_tile_selection_respects_each_compute_limit() {
        assert_eq!(choose_max_rows_per_group(1024, 1024, 65_536, 128), Some(4));
        assert_eq!(choose_max_rows_per_group(256, 256, 65_536, 128), Some(2));
        assert_eq!(choose_max_rows_per_group(1024, 128, 65_536, 128), Some(1));
        assert_eq!(choose_max_rows_per_group(128, 128, 32_768, 128), Some(1));
        assert_eq!(choose_max_rows_per_group(64, 64, 32_768, 128), None);
    }

    #[test]
    fn shared_rows_fit_webgpu_minimum_workgroup_limits() {
        assert!(shared_rows_supported(128, 128, 16_384));
        assert!(!shared_rows_supported(127, 128, 16_384));
        assert!(!shared_rows_supported(128, 127, 16_384));
        assert!(!shared_rows_supported(
            128,
            128,
            2 * WORKGROUP_BYTES_PER_ROW - 1,
        ));
    }

    #[test]
    fn prediction_shape_falls_back_to_a_compiled_row_tile() {
        let limited = [(64usize, 2usize), (64, 1), (128, 1)];
        assert_eq!(
            select_prediction_shape(&limited, 64, None, 256),
            Some((64, 2))
        );
        assert_eq!(
            select_prediction_shape(&limited, 64, None, 257),
            Some((64, 2))
        );

        let minimum = [(64usize, 1usize)];
        assert_eq!(
            select_prediction_shape(&minimum, 64, None, 128),
            Some((64, 1))
        );
        assert_eq!(
            select_prediction_shape(&minimum, 64, None, 8_192),
            Some((64, 1))
        );
    }

    #[test]
    fn shared_rows_default_preserves_the_single_row_kernel_for_small_calls() {
        assert_eq!(
            select_prediction_kernel(PredictionKernel::SharedRows2, None, false, 1_024),
            PredictionKernel::SingleRow,
        );
        assert_eq!(
            select_prediction_kernel(PredictionKernel::SharedRows2, None, false, 2_048),
            PredictionKernel::SharedRows2,
        );
        assert_eq!(
            select_prediction_kernel(PredictionKernel::SharedRows2, None, true, 8_192),
            PredictionKernel::SingleRow,
        );
        assert_eq!(
            select_prediction_kernel(
                PredictionKernel::SharedRows2,
                Some(PredictionKernel::SharedRows2),
                true,
                1,
            ),
            PredictionKernel::SharedRows2,
        );
    }

    #[test]
    fn explicitly_forced_prediction_tile_must_exist() {
        let shapes = [(64usize, 2usize), (64, 1)];
        assert_eq!(
            select_prediction_shape(&shapes, 64, Some(2), 64),
            Some((64, 2))
        );
        assert_eq!(select_prediction_shape(&shapes, 64, Some(4), 64), None);
        assert_eq!(select_prediction_shape(&shapes, 128, None, 64), None);
    }
}
