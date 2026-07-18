use std::sync::OnceLock;

pub(crate) const CPU_PROFILE_ENV_VAR: &str = "RWKV_SRS_CPU_PROFILE";
pub(crate) const SIMD_BACKEND_ENV_VAR: &str = "RWKV_SRS_SIMD_BACKEND";

const DEFAULT_BULK_CARD_NOTE_PERSISTENT_FRONTIER_BATCH_SIZE: usize = 512;
const DEFAULT_BULK_LANE_PARALLEL_THREADS: usize = 2;
const DEFAULT_LIGHTNING_DECAY_LUT_STEP: f32 = 0.1;
const DEFAULT_PIPELINE_CAPACITY: usize = 64;
const DEFAULT_PIPELINE_COMPUTE_BATCH: usize = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CpuProfile {
    Oracle,
    Fast,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SimdBackendPreference {
    Auto,
    Avx2,
    Pulp,
}

impl SimdBackendPreference {
    fn parse(value: Option<String>) -> Self {
        match value.as_deref().map(str::trim) {
            Some(value) if value.eq_ignore_ascii_case("avx2") => Self::Avx2,
            Some(value) if value.eq_ignore_ascii_case("pulp") => Self::Pulp,
            _ => Self::Auto,
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Avx2 => "avx2",
            Self::Pulp => "pulp",
        }
    }
}

impl CpuProfile {
    fn parse(value: &str) -> Option<Self> {
        if value.eq_ignore_ascii_case("oracle") {
            Some(Self::Oracle)
        } else if value.eq_ignore_ascii_case("fast") {
            Some(Self::Fast)
        } else {
            None
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Oracle => "oracle",
            Self::Fast => "fast",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct CpuRuntimeConfig {
    profile: CpuProfile,
    core: CoreConfig,
    linear: LinearConfig,
    channel_mixer: ChannelMixerConfig,
    time_mixer: TimeMixerConfig,
    layer: LayerConfig,
    bulk: BulkConfig,
    pipeline: PipelineConfig,
    lightning: LightningConfig,
}

#[derive(Clone, Debug, PartialEq)]
struct CoreConfig {
    simd_disabled: bool,
    simd_backend: SimdBackendPreference,
    state_simd: bool,
    native_layer_norm: bool,
    fast_avx2_horizontal_sum: bool,
    trusted_hot_loop_shapes: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct LinearConfig {
    kernels: bool,
    dot4_same_x: bool,
    dot8_same_x: bool,
    dot12_same_x: bool,
    dot12_pointer_rows: bool,
    dot12_direct_horizontal_sum: bool,
    blocked_input8_dot12_layout: bool,
    shared_direct_horizontal_sum: bool,
    fixed_in_dim_same_x: bool,
    batched_same_x: bool,
    batched_four_row_same_x: bool,
    batched_pointer_rows: bool,
    fused_bias_add: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct ChannelMixerConfig {
    kernels: bool,
    residual_scratch: bool,
    prelude_residual_scratch: bool,
    flat_buffer_reuse: bool,
    inplace_output_values: bool,
    direct_relu_square: bool,
    flat_working_state: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct TimeMixerConfig {
    projection_kernels: bool,
    v_mix_kernels: bool,
    output_kernels: bool,
    output_batched_projection: bool,
    output_batched_projection_min8: bool,
    output_batched_projection_min4: bool,
    output_buffer_reuse: bool,
    output_head32_stack: bool,
    lora_decay_scratch: bool,
    lora_dot4_projections: bool,
    lora_activation_fusion: bool,
    v_mix_project_dot4: bool,
    post_lora_scratch: bool,
    v_mix_values_scratch: bool,
    lerp_projection_values_scratch: bool,
    projection_values_buffer_reuse: bool,
    batched_lerp_projection_values: bool,
    paired_projection_dots: bool,
    middle_all_values_scratch: bool,
    layer0_v_values_scratch: bool,
    lerp_projection_scratch: bool,
    reshape_recurrence_scratch: bool,
    middle_output_scratch: bool,
    middle_buffer_reuse: bool,
    group_output_scratch: bool,
    group_output_cached_aux: bool,
    output_one_pass_group_norm: bool,
    flat_input_values: bool,
    predict_x_buffer_reuse: bool,
    zero_state_recurrence: bool,
    zero_state_stack_row: bool,
    direct_activation_scalars: bool,
    flat_working_state: bool,
    recurrence_reuse_deformed_key_values: bool,
    recurrence_output_sums: bool,
    recurrence_output_variances: bool,
    recurrence_direct_horizontal_sum: bool,
    time_decay_kernels: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct LayerConfig {
    time_channel_values_handoff: bool,
    values_carrier: bool,
    rnn_borrow_first_input_values: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct BulkConfig {
    curve_heads_batch: bool,
    prediction_head_batch: bool,
    lane_parallel: bool,
    lane_parallel_stream_install: bool,
    lane_parallel_borrow_state: bool,
    module_output_buffer: bool,
    module_values_carrier: bool,
    module_raw_values_handoff: bool,
    new_singleton_batch_values: bool,
    profile_fast_paths: bool,
    new_singleton_state_views: bool,
    deck_persistent_frontier_values: bool,
    card_note_persistent_frontier_values: bool,
    persistent_frontier_in_place_state_compaction: bool,
    persistent_frontier_input_buffer_reuse: bool,
    card_note_persistent_frontier_batch_size: usize,
    lane_parallel_threads: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
struct PipelineConfig {
    threads: Option<usize>,
    capacity: usize,
    compute_batch: usize,
    profile: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct LightningConfig {
    parallel_batches: bool,
    modulewise_state_pack: bool,
    skip_deformation: bool,
    deformation_left_threshold: Option<f32>,
    algebraic_wkv: bool,
    decay_lut_step: Option<f32>,
    activation_lut_step: Option<f32>,
    lora_rank_limit: Option<usize>,
    direct_decay_lut_loop: bool,
}

static CPU_CONFIG: OnceLock<CpuRuntimeConfig> = OnceLock::new();

impl CpuRuntimeConfig {
    fn from_env() -> Self {
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    fn from_lookup(mut lookup: impl FnMut(&str) -> Option<String>) -> Self {
        let profile = match lookup(CPU_PROFILE_ENV_VAR).as_deref() {
            Some(value) if value.eq_ignore_ascii_case("fast") => CpuProfile::Fast,
            _ => CpuProfile::Oracle,
        };
        Self::from_profile_and_lookup(profile, lookup)
    }

    fn from_profile(profile: CpuProfile) -> Self {
        Self::from_profile_and_lookup(profile, |name| std::env::var(name).ok())
    }

    fn from_profile_and_lookup(
        profile: CpuProfile,
        mut lookup: impl FnMut(&str) -> Option<String>,
    ) -> Self {
        let fast = profile == CpuProfile::Fast;

        let core = CoreConfig {
            simd_disabled: flag(&mut lookup, "RWKV_SRS_DISABLE_SIMD", false),
            simd_backend: SimdBackendPreference::parse(lookup(SIMD_BACKEND_ENV_VAR)),
            state_simd: flag(&mut lookup, "RWKV_SRS_ENABLE_STATE_SIMD", fast),
            native_layer_norm: flag(&mut lookup, "RWKV_SRS_ENABLE_NATIVE_LAYER_NORM", fast),
            fast_avx2_horizontal_sum: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_FAST_AVX2_HORIZONTAL_SUM",
                fast,
            ),
            trusted_hot_loop_shapes: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TRUSTED_HOT_LOOP_SHAPES",
                fast,
            ),
        };

        let linear = LinearConfig {
            kernels: flag(&mut lookup, "RWKV_SRS_ENABLE_LINEAR_KERNELS", fast),
            dot4_same_x: flag(&mut lookup, "RWKV_SRS_ENABLE_LINEAR_DOT4_SAME_X", fast),
            dot8_same_x: flag(&mut lookup, "RWKV_SRS_ENABLE_LINEAR_DOT8_SAME_X", fast),
            dot12_same_x: flag(&mut lookup, "RWKV_SRS_ENABLE_LINEAR_DOT12_SAME_X", fast),
            dot12_pointer_rows: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_LINEAR_DOT12_POINTER_ROWS",
                fast,
            ),
            dot12_direct_horizontal_sum: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_LINEAR_DOT12_DIRECT_HORIZONTAL_SUM",
                fast,
            ),
            blocked_input8_dot12_layout: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_LINEAR_BLOCKED_INPUT8_DOT12_LAYOUT",
                fast,
            ),
            shared_direct_horizontal_sum: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_LINEAR_SHARED_DIRECT_HORIZONTAL_SUM",
                fast,
            ),
            fixed_in_dim_same_x: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_LINEAR_FIXED_IN_DIM_SAME_X",
                fast,
            ),
            batched_same_x: flag(&mut lookup, "RWKV_SRS_ENABLE_LINEAR_BATCHED_SAME_X", fast),
            batched_four_row_same_x: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_LINEAR_BATCHED_FOUR_ROW_SAME_X",
                fast,
            ),
            batched_pointer_rows: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_LINEAR_BATCHED_POINTER_ROWS",
                fast,
            ),
            fused_bias_add: flag(&mut lookup, "RWKV_SRS_ENABLE_LINEAR_FUSED_BIAS_ADD", fast),
        };

        let recurrence_output_sums = flag(
            &mut lookup,
            "RWKV_SRS_ENABLE_TIME_MIXER_RECURRENCE_OUTPUT_SUMS",
            fast,
        );
        let mut time_mixer = TimeMixerConfig {
            projection_kernels: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_PROJECTION_KERNELS",
                fast,
            ),
            v_mix_kernels: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_V_MIX_KERNELS",
                fast,
            ),
            output_kernels: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_OUTPUT_KERNELS",
                fast,
            ),
            output_batched_projection: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_OUTPUT_BATCHED_PROJECTION",
                fast,
            ),
            output_batched_projection_min8: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_OUTPUT_BATCHED_PROJECTION_MIN8",
                fast,
            ),
            output_batched_projection_min4: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_OUTPUT_BATCHED_PROJECTION_MIN4",
                fast,
            ),
            output_buffer_reuse: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_OUTPUT_BUFFER_REUSE",
                fast,
            ),
            output_head32_stack: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_OUTPUT_HEAD32_STACK",
                fast,
            ),
            lora_decay_scratch: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_LORA_DECAY_SCRATCH",
                fast,
            ),
            lora_dot4_projections: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_LORA_DOT4_PROJECTIONS",
                fast,
            ),
            lora_activation_fusion: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_LORA_ACTIVATION_FUSION",
                fast,
            ),
            v_mix_project_dot4: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_V_MIX_PROJECT_DOT4",
                fast,
            ),
            post_lora_scratch: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_POST_LORA_SCRATCH",
                fast,
            ),
            v_mix_values_scratch: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_V_MIX_VALUES_SCRATCH",
                fast,
            ),
            lerp_projection_values_scratch: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_LERP_PROJECTION_VALUES_SCRATCH",
                fast,
            ),
            projection_values_buffer_reuse: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_PROJECTION_VALUES_BUFFER_REUSE",
                fast,
            ),
            batched_lerp_projection_values: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_BATCHED_LERP_PROJECTION_VALUES",
                fast,
            ),
            paired_projection_dots: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_PAIRED_PROJECTION_DOTS",
                fast,
            ),
            middle_all_values_scratch: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_MIDDLE_ALL_VALUES_SCRATCH",
                fast,
            ),
            layer0_v_values_scratch: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_LAYER0_V_VALUES_SCRATCH",
                fast,
            ),
            lerp_projection_scratch: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_LERP_PROJECTION_SCRATCH",
                fast,
            ),
            reshape_recurrence_scratch: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_RESHAPE_RECURRENCE_SCRATCH",
                fast,
            ),
            middle_output_scratch: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_MIDDLE_OUTPUT_SCRATCH",
                fast,
            ),
            middle_buffer_reuse: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_MIDDLE_BUFFER_REUSE",
                fast,
            ),
            group_output_scratch: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_GROUP_OUTPUT_SCRATCH",
                fast,
            ),
            group_output_cached_aux: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_GROUP_OUTPUT_CACHED_AUX",
                fast,
            ),
            output_one_pass_group_norm: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_OUTPUT_ONE_PASS_GROUP_NORM",
                fast,
            ),
            flat_input_values: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_FLAT_INPUT_VALUES",
                fast,
            ),
            predict_x_buffer_reuse: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_PREDICT_X_BUFFER_REUSE",
                fast,
            ),
            zero_state_recurrence: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_ZERO_STATE_RECURRENCE",
                fast,
            ),
            zero_state_stack_row: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_ZERO_STATE_STACK_ROW",
                fast,
            ),
            direct_activation_scalars: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_DIRECT_ACTIVATION_SCALARS",
                fast,
            ),
            flat_working_state: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_FLAT_WORKING_STATE",
                fast,
            ),
            recurrence_reuse_deformed_key_values: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_RECURRENCE_REUSE_DEFORMED_KEY_VALUES",
                fast,
            ),
            recurrence_output_sums,
            recurrence_output_variances: recurrence_output_sums
                && flag(
                    &mut lookup,
                    "RWKV_SRS_ENABLE_TIME_MIXER_RECURRENCE_OUTPUT_VARIANCES",
                    fast,
                ),
            recurrence_direct_horizontal_sum: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_TIME_MIXER_RECURRENCE_DIRECT_HORIZONTAL_SUM",
                fast,
            ),
            time_decay_kernels: flag(&mut lookup, "RWKV_SRS_ENABLE_TIME_DECAY_KERNELS", fast),
        };
        time_mixer.flat_working_state = time_mixer.flat_working_state
            && time_mixer.lerp_projection_values_scratch
            && time_mixer.lerp_projection_scratch
            && time_mixer.v_mix_values_scratch
            && time_mixer.v_mix_kernels
            && time_mixer.post_lora_scratch
            && time_mixer.lora_decay_scratch
            && time_mixer.middle_output_scratch
            && time_mixer.output_kernels
            && time_mixer.group_output_scratch
            && time_mixer.middle_all_values_scratch
            && time_mixer.group_output_cached_aux
            && time_mixer.layer0_v_values_scratch;

        let mut channel_mixer = ChannelMixerConfig {
            kernels: flag(&mut lookup, "RWKV_SRS_ENABLE_CHANNEL_MIXER_KERNELS", fast),
            residual_scratch: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_CHANNEL_MIXER_RESIDUAL_SCRATCH",
                fast,
            ),
            prelude_residual_scratch: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_CHANNEL_MIXER_PRELUDE_RESIDUAL_SCRATCH",
                fast,
            ),
            flat_buffer_reuse: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_CHANNEL_MIXER_FLAT_BUFFER_REUSE",
                fast,
            ),
            inplace_output_values: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_CHANNEL_MIXER_INPLACE_OUTPUT_VALUES",
                fast,
            ),
            direct_relu_square: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_CHANNEL_MIXER_DIRECT_RELU_SQUARE",
                fast,
            ),
            flat_working_state: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_CHANNEL_MIXER_FLAT_WORKING_STATE",
                fast,
            ),
        };
        channel_mixer.flat_working_state = channel_mixer.flat_working_state
            && time_mixer.flat_working_state
            && channel_mixer.kernels
            && channel_mixer.residual_scratch
            && channel_mixer.prelude_residual_scratch;

        let layer = LayerConfig {
            time_channel_values_handoff: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_LAYER_TIME_CHANNEL_VALUES_HANDOFF",
                fast,
            ),
            values_carrier: flag(&mut lookup, "RWKV_SRS_ENABLE_LAYER_VALUES_CARRIER", fast),
            rnn_borrow_first_input_values: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_RNN_BORROW_FIRST_INPUT_VALUES",
                fast,
            ),
        };

        let module_values_carrier = flag(
            &mut lookup,
            "RWKV_SRS_ENABLE_BULK_MODULE_VALUES_CARRIER",
            fast,
        );
        let new_singleton_batch_values = flag(
            &mut lookup,
            "RWKV_SRS_ENABLE_BULK_NEW_SINGLETON_BATCH_VALUES",
            fast,
        );
        let bulk = BulkConfig {
            curve_heads_batch: flag(&mut lookup, "RWKV_SRS_ENABLE_BULK_CURVE_HEADS_BATCH", fast),
            prediction_head_batch: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_BULK_PREDICTION_HEAD_BATCH",
                fast,
            ),
            lane_parallel: flag(&mut lookup, "RWKV_SRS_ENABLE_BULK_LANE_PARALLEL", fast),
            lane_parallel_stream_install: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_BULK_LANE_PARALLEL_STREAM_INSTALL",
                fast,
            ),
            lane_parallel_borrow_state: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_BULK_LANE_PARALLEL_BORROW_STATE",
                fast,
            ),
            module_output_buffer: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_BULK_MODULE_OUTPUT_BUFFER",
                fast,
            ),
            module_values_carrier,
            module_raw_values_handoff: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_BULK_MODULE_RAW_VALUES_HANDOFF",
                fast,
            ),
            new_singleton_batch_values,
            profile_fast_paths: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_BULK_PROFILE_FAST_PATHS",
                false,
            ),
            new_singleton_state_views: new_singleton_batch_values
                && flag(
                    &mut lookup,
                    "RWKV_SRS_ENABLE_BULK_NEW_SINGLETON_STATE_VIEWS",
                    fast,
                ),
            deck_persistent_frontier_values: module_values_carrier
                && flag(
                    &mut lookup,
                    "RWKV_SRS_ENABLE_BULK_DECK_PERSISTENT_FRONTIER_VALUES",
                    fast,
                ),
            card_note_persistent_frontier_values: module_values_carrier
                && flag(
                    &mut lookup,
                    "RWKV_SRS_ENABLE_BULK_CARD_NOTE_PERSISTENT_FRONTIER_VALUES",
                    fast,
                ),
            persistent_frontier_in_place_state_compaction: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_BULK_PERSISTENT_FRONTIER_IN_PLACE_STATE_COMPACTION",
                fast,
            ),
            persistent_frontier_input_buffer_reuse: flag(
                &mut lookup,
                "RWKV_SRS_ENABLE_BULK_PERSISTENT_FRONTIER_INPUT_BUFFER_REUSE",
                fast,
            ),
            card_note_persistent_frontier_batch_size: usize_at_least(
                &mut lookup,
                "RWKV_SRS_BULK_CARD_NOTE_PERSISTENT_FRONTIER_BATCH_SIZE",
                2,
            )
            .unwrap_or(DEFAULT_BULK_CARD_NOTE_PERSISTENT_FRONTIER_BATCH_SIZE),
            lane_parallel_threads: usize_at_least(
                &mut lookup,
                "RWKV_SRS_BULK_LANE_PARALLEL_THREADS",
                DEFAULT_BULK_LANE_PARALLEL_THREADS,
            )
            .or(fast.then_some(DEFAULT_BULK_LANE_PARALLEL_THREADS)),
        };

        let pipeline = PipelineConfig {
            threads: usize_at_least(&mut lookup, "RWKV_SRS_PIPELINE_THREADS", 1),
            capacity: usize_at_least(&mut lookup, "RWKV_SRS_PIPELINE_CAPACITY", 1)
                .unwrap_or(DEFAULT_PIPELINE_CAPACITY),
            compute_batch: usize_at_least(&mut lookup, "RWKV_SRS_PIPELINE_COMPUTE_BATCH", 1)
                .unwrap_or(DEFAULT_PIPELINE_COMPUTE_BATCH),
            profile: flag(&mut lookup, "RWKV_SRS_PIPELINE_PROFILE", false),
        };

        let lightning = LightningConfig {
            parallel_batches: flag(
                &mut lookup,
                "RWKV_SRS_PREDICT_MANY_LIGHTNING_PARALLEL_BATCHES",
                true,
            ),
            modulewise_state_pack: flag(
                &mut lookup,
                "RWKV_SRS_PREDICT_MANY_LIGHTNING_MODULEWISE_STATE_PACK",
                true,
            ),
            skip_deformation: flag(
                &mut lookup,
                "RWKV_SRS_PREDICT_MANY_LIGHTNING_SKIP_DEFORMATION",
                false,
            ),
            deformation_left_threshold: finite_f32_above(
                &mut lookup,
                "RWKV_SRS_PREDICT_MANY_LIGHTNING_DEFORMATION_LEFT_THRESHOLD",
                0.0,
                None,
            ),
            algebraic_wkv: flag(
                &mut lookup,
                "RWKV_SRS_PREDICT_MANY_LIGHTNING_ALGEBRAIC_WKV",
                true,
            ),
            decay_lut_step: finite_f32_at_least(
                &mut lookup,
                "RWKV_SRS_PREDICT_MANY_LIGHTNING_DECAY_LUT_STEP",
                1e-6,
                Some(DEFAULT_LIGHTNING_DECAY_LUT_STEP),
            ),
            activation_lut_step: finite_f32_at_least(
                &mut lookup,
                "RWKV_SRS_PREDICT_MANY_LIGHTNING_ACTIVATION_LUT_STEP",
                1e-6,
                None,
            ),
            lora_rank_limit: usize_value(&mut lookup, "RWKV_SRS_PREDICT_MANY_LIGHTNING_LORA_RANK"),
            direct_decay_lut_loop: flag(
                &mut lookup,
                "RWKV_SRS_PREDICT_MANY_LIGHTNING_DIRECT_DECAY_LUT_LOOP",
                true,
            ),
        };

        Self {
            profile,
            core,
            linear,
            channel_mixer,
            time_mixer,
            layer,
            bulk,
            pipeline,
            lightning,
        }
    }
}

fn flag(lookup: &mut impl FnMut(&str) -> Option<String>, name: &str, default: bool) -> bool {
    lookup(name)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

fn usize_value(lookup: &mut impl FnMut(&str) -> Option<String>, name: &str) -> Option<usize> {
    lookup(name)?.parse::<usize>().ok()
}

fn usize_at_least(
    lookup: &mut impl FnMut(&str) -> Option<String>,
    name: &str,
    minimum: usize,
) -> Option<usize> {
    usize_value(lookup, name).filter(|value| *value >= minimum)
}

fn finite_f32_above(
    lookup: &mut impl FnMut(&str) -> Option<String>,
    name: &str,
    minimum_exclusive: f32,
    default: Option<f32>,
) -> Option<f32> {
    let Some(raw) = lookup(name) else {
        return default;
    };
    let value = raw.parse::<f32>().ok()?;
    (value.is_finite() && value > minimum_exclusive).then_some(value)
}

fn finite_f32_at_least(
    lookup: &mut impl FnMut(&str) -> Option<String>,
    name: &str,
    minimum: f32,
    default: Option<f32>,
) -> Option<f32> {
    let Some(raw) = lookup(name) else {
        return default;
    };
    let value = raw.parse::<f32>().ok()?;
    (value.is_finite() && value >= minimum).then_some(value)
}

fn cpu_config() -> &'static CpuRuntimeConfig {
    CPU_CONFIG.get_or_init(CpuRuntimeConfig::from_env)
}

pub(crate) fn claim_cpu_profile(value: &str) -> Result<(), String> {
    let requested = CpuProfile::parse(value)
        .ok_or_else(|| format!("CPU profile must be 'oracle' or 'fast', got {value:?}"))?;

    if CPU_CONFIG.get().is_none() {
        let _ = CPU_CONFIG.set(CpuRuntimeConfig::from_profile(requested));
    }
    let initialized = CPU_CONFIG
        .get()
        .expect("CPU_CONFIG is initialized by get or set above")
        .profile;
    if initialized == requested {
        Ok(())
    } else {
        Err(format!(
            "Rust CPU profile is already initialized as '{}'; start a fresh interpreter to use '{}'",
            initialized.as_str(),
            requested.as_str(),
        ))
    }
}

macro_rules! bool_accessor {
    ($name:ident, $section:ident.$field:ident) => {
        #[inline]
        pub(crate) fn $name() -> bool {
            cpu_config().$section.$field
        }
    };
}

bool_accessor!(simd_disabled, core.simd_disabled);
bool_accessor!(native_linear_simd_disabled, core.simd_disabled);
bool_accessor!(state_simd_enabled, core.state_simd);
bool_accessor!(native_centered_layer_norm_enabled, core.native_layer_norm);
bool_accessor!(
    fast_avx2_horizontal_sum_enabled,
    core.fast_avx2_horizontal_sum
);
bool_accessor!(
    trusted_hot_loop_shapes_enabled,
    core.trusted_hot_loop_shapes
);

#[inline]
pub(crate) fn simd_backend_preference() -> SimdBackendPreference {
    cpu_config().core.simd_backend
}

bool_accessor!(native_linear_kernel_enabled, linear.kernels);
bool_accessor!(native_linear_dot4_same_x_enabled, linear.dot4_same_x);
bool_accessor!(native_linear_dot8_same_x_enabled, linear.dot8_same_x);
bool_accessor!(native_linear_dot12_same_x_enabled, linear.dot12_same_x);
bool_accessor!(
    native_linear_dot12_pointer_rows_enabled,
    linear.dot12_pointer_rows
);
bool_accessor!(
    native_linear_dot12_direct_horizontal_sum_enabled,
    linear.dot12_direct_horizontal_sum
);
bool_accessor!(
    native_linear_blocked_input8_dot12_layout_enabled,
    linear.blocked_input8_dot12_layout
);
bool_accessor!(
    native_linear_shared_direct_horizontal_sum_enabled,
    linear.shared_direct_horizontal_sum
);
bool_accessor!(
    native_linear_fixed_in_dim_same_x_enabled,
    linear.fixed_in_dim_same_x
);
bool_accessor!(native_linear_batched_same_x_enabled, linear.batched_same_x);
bool_accessor!(
    native_linear_batched_four_row_same_x_enabled,
    linear.batched_four_row_same_x
);
bool_accessor!(
    native_linear_batched_pointer_rows_enabled,
    linear.batched_pointer_rows
);
bool_accessor!(native_linear_fused_bias_add_enabled, linear.fused_bias_add);

bool_accessor!(
    native_channel_mixer_projection_enabled,
    channel_mixer.kernels
);
bool_accessor!(
    native_channel_mixer_residual_scratch_enabled,
    channel_mixer.residual_scratch
);
bool_accessor!(
    native_channel_mixer_prelude_residual_scratch_enabled,
    channel_mixer.prelude_residual_scratch
);
bool_accessor!(
    native_channel_mixer_flat_buffer_reuse_enabled,
    channel_mixer.flat_buffer_reuse
);
bool_accessor!(
    native_channel_mixer_inplace_output_values_enabled,
    channel_mixer.inplace_output_values
);
bool_accessor!(
    native_channel_mixer_direct_relu_square_enabled,
    channel_mixer.direct_relu_square
);
bool_accessor!(
    channel_mixer_flat_working_state_enabled,
    channel_mixer.flat_working_state
);

bool_accessor!(
    native_time_mixer_projection_enabled,
    time_mixer.projection_kernels
);
bool_accessor!(native_time_mixer_v_mix_enabled, time_mixer.v_mix_kernels);
bool_accessor!(native_time_mixer_output_enabled, time_mixer.output_kernels);
bool_accessor!(
    native_time_mixer_output_batched_projection_enabled,
    time_mixer.output_batched_projection
);
bool_accessor!(
    native_time_mixer_output_batched_projection_min8_enabled,
    time_mixer.output_batched_projection_min8
);
bool_accessor!(
    native_time_mixer_output_batched_projection_min4_enabled,
    time_mixer.output_batched_projection_min4
);
bool_accessor!(
    native_time_mixer_output_buffer_reuse_enabled,
    time_mixer.output_buffer_reuse
);
bool_accessor!(
    native_time_mixer_output_head32_stack_enabled,
    time_mixer.output_head32_stack
);
bool_accessor!(
    native_time_mixer_lora_decay_scratch_enabled,
    time_mixer.lora_decay_scratch
);
bool_accessor!(
    native_time_mixer_lora_dot4_projections_enabled,
    time_mixer.lora_dot4_projections
);
bool_accessor!(
    native_time_mixer_lora_activation_fusion_enabled,
    time_mixer.lora_activation_fusion
);
bool_accessor!(
    native_time_mixer_v_mix_project_dot4_enabled,
    time_mixer.v_mix_project_dot4
);
bool_accessor!(
    native_time_mixer_post_lora_scratch_enabled,
    time_mixer.post_lora_scratch
);
bool_accessor!(
    native_time_mixer_v_mix_values_scratch_enabled,
    time_mixer.v_mix_values_scratch
);
bool_accessor!(
    native_time_mixer_lerp_projection_values_scratch_enabled,
    time_mixer.lerp_projection_values_scratch
);
bool_accessor!(
    native_time_mixer_projection_values_buffer_reuse_enabled,
    time_mixer.projection_values_buffer_reuse
);
bool_accessor!(
    native_time_mixer_batched_lerp_projection_values_enabled,
    time_mixer.batched_lerp_projection_values
);
bool_accessor!(
    native_time_mixer_paired_projection_dots_enabled,
    time_mixer.paired_projection_dots
);
bool_accessor!(
    native_time_mixer_middle_all_values_scratch_enabled,
    time_mixer.middle_all_values_scratch
);
bool_accessor!(
    native_time_mixer_layer0_v_values_scratch_enabled,
    time_mixer.layer0_v_values_scratch
);
bool_accessor!(
    native_time_mixer_lerp_projection_scratch_enabled,
    time_mixer.lerp_projection_scratch
);
bool_accessor!(
    native_time_mixer_reshape_recurrence_scratch_enabled,
    time_mixer.reshape_recurrence_scratch
);
bool_accessor!(
    native_time_mixer_middle_output_scratch_enabled,
    time_mixer.middle_output_scratch
);
bool_accessor!(
    time_mixer_middle_buffer_reuse_enabled,
    time_mixer.middle_buffer_reuse
);
bool_accessor!(
    native_time_mixer_group_output_scratch_enabled,
    time_mixer.group_output_scratch
);
bool_accessor!(
    native_time_mixer_group_output_cached_aux_enabled,
    time_mixer.group_output_cached_aux
);
bool_accessor!(
    native_time_mixer_output_one_pass_group_norm_enabled,
    time_mixer.output_one_pass_group_norm
);
bool_accessor!(
    native_time_mixer_flat_input_values_enabled,
    time_mixer.flat_input_values
);
bool_accessor!(
    native_time_mixer_predict_x_buffer_reuse_enabled,
    time_mixer.predict_x_buffer_reuse
);
bool_accessor!(
    native_time_mixer_zero_state_recurrence_enabled,
    time_mixer.zero_state_recurrence
);
bool_accessor!(
    time_mixer_zero_state_stack_row_enabled,
    time_mixer.zero_state_stack_row
);
bool_accessor!(
    direct_time_mixer_activation_scalars_enabled,
    time_mixer.direct_activation_scalars
);
bool_accessor!(
    time_mixer_flat_working_state_enabled,
    time_mixer.flat_working_state
);
bool_accessor!(
    recurrence_reuse_deformed_key_values_enabled,
    time_mixer.recurrence_reuse_deformed_key_values
);
bool_accessor!(
    time_mixer_recurrence_output_sums_enabled,
    time_mixer.recurrence_output_sums
);
bool_accessor!(
    time_mixer_recurrence_output_variances_enabled,
    time_mixer.recurrence_output_variances
);
bool_accessor!(
    recurrence_direct_horizontal_sum_enabled,
    time_mixer.recurrence_direct_horizontal_sum
);
bool_accessor!(fast_time_decay_enabled, time_mixer.time_decay_kernels);

bool_accessor!(
    native_layer_time_channel_values_handoff_enabled,
    layer.time_channel_values_handoff
);
bool_accessor!(
    layer_time_channel_values_handoff_enabled,
    layer.time_channel_values_handoff
);
bool_accessor!(native_layer_values_carrier_enabled, layer.values_carrier);
bool_accessor!(layer_values_carrier_enabled, layer.values_carrier);
bool_accessor!(
    native_rnn_borrow_first_input_values_enabled,
    layer.rnn_borrow_first_input_values
);

bool_accessor!(bulk_curve_heads_batch_enabled, bulk.curve_heads_batch);
bool_accessor!(
    bulk_prediction_head_batch_enabled,
    bulk.prediction_head_batch
);
bool_accessor!(bulk_lane_parallel_enabled, bulk.lane_parallel);
bool_accessor!(
    bulk_lane_parallel_stream_install_enabled,
    bulk.lane_parallel_stream_install
);
bool_accessor!(
    bulk_lane_parallel_borrow_state_enabled,
    bulk.lane_parallel_borrow_state
);
bool_accessor!(bulk_module_output_buffer_enabled, bulk.module_output_buffer);
bool_accessor!(
    bulk_module_values_carrier_enabled,
    bulk.module_values_carrier
);
bool_accessor!(
    bulk_module_raw_values_handoff_enabled,
    bulk.module_raw_values_handoff
);
bool_accessor!(
    bulk_new_singleton_batch_values_enabled,
    bulk.new_singleton_batch_values
);
bool_accessor!(bulk_profile_fast_paths_enabled, bulk.profile_fast_paths);
bool_accessor!(
    bulk_new_singleton_state_views_enabled,
    bulk.new_singleton_state_views
);
bool_accessor!(
    bulk_deck_persistent_frontier_values_enabled,
    bulk.deck_persistent_frontier_values
);
bool_accessor!(
    bulk_card_note_persistent_frontier_values_enabled,
    bulk.card_note_persistent_frontier_values
);
bool_accessor!(
    bulk_persistent_frontier_in_place_state_compaction_enabled,
    bulk.persistent_frontier_in_place_state_compaction
);
bool_accessor!(
    bulk_persistent_frontier_input_buffer_reuse_enabled,
    bulk.persistent_frontier_input_buffer_reuse
);

#[inline]
pub(crate) fn bulk_card_note_persistent_frontier_batch_size() -> usize {
    cpu_config().bulk.card_note_persistent_frontier_batch_size
}

#[inline]
pub(crate) fn bulk_lane_parallel_threads(num_threads: Option<usize>) -> usize {
    cpu_config()
        .bulk
        .lane_parallel_threads
        .or(num_threads)
        .unwrap_or(DEFAULT_BULK_LANE_PARALLEL_THREADS)
        .max(DEFAULT_BULK_LANE_PARALLEL_THREADS)
}

#[inline]
pub(crate) fn pipeline_threads_override() -> Option<usize> {
    cpu_config().pipeline.threads
}

#[inline]
pub(crate) fn pipeline_capacity() -> usize {
    cpu_config().pipeline.capacity
}

#[inline]
pub(crate) fn pipeline_compute_batch() -> usize {
    cpu_config().pipeline.compute_batch
}

bool_accessor!(pipeline_profile_enabled, pipeline.profile);

bool_accessor!(
    predict_many_lightning_parallel_batches_enabled,
    lightning.parallel_batches
);
bool_accessor!(
    predict_many_lightning_modulewise_state_pack_enabled,
    lightning.modulewise_state_pack
);
bool_accessor!(
    lightning_skip_deformation_configured,
    lightning.skip_deformation
);
bool_accessor!(lightning_algebraic_wkv_configured, lightning.algebraic_wkv);
bool_accessor!(
    predict_many_lightning_direct_decay_lut_loop_enabled,
    lightning.direct_decay_lut_loop
);

#[inline]
pub(crate) fn lightning_deformation_left_threshold_configured() -> Option<f32> {
    cpu_config().lightning.deformation_left_threshold
}

#[inline]
pub(crate) fn predict_many_lightning_decay_lut_step() -> Option<f32> {
    cpu_config().lightning.decay_lut_step
}

#[inline]
pub(crate) fn predict_many_lightning_activation_lut_step() -> Option<f32> {
    cpu_config().lightning.activation_lut_step
}

#[inline]
pub(crate) fn predict_many_lightning_lora_rank_limit_configured() -> Option<usize> {
    cpu_config().lightning.lora_rank_limit
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    fn config(values: &[(&str, &str)]) -> CpuRuntimeConfig {
        let values = values
            .iter()
            .map(|(name, value)| ((*name).to_string(), (*value).to_string()))
            .collect::<BTreeMap<_, _>>();
        CpuRuntimeConfig::from_lookup(|name| values.get(name).cloned())
    }

    #[test]
    fn oracle_profile_keeps_promoted_optimizations_disabled() {
        let config = config(&[]);

        assert_eq!(config.profile, CpuProfile::Oracle);
        assert_eq!(config.core.simd_backend, SimdBackendPreference::Auto);
        assert!(!config.linear.kernels);
        assert!(!config.channel_mixer.kernels);
        assert!(!config.time_mixer.output_kernels);
        assert!(!config.time_mixer.flat_working_state);
        assert!(!config.channel_mixer.flat_working_state);
        assert!(!config.bulk.lane_parallel);
        assert!(config.lightning.parallel_batches);
        assert!(config.lightning.modulewise_state_pack);
    }

    #[test]
    fn fast_profile_enables_the_promoted_typed_groups() {
        let config = config(&[(CPU_PROFILE_ENV_VAR, "fast")]);

        assert_eq!(config.profile, CpuProfile::Fast);
        assert!(config.core.state_simd);
        assert!(config.core.native_layer_norm);
        assert!(config.core.fast_avx2_horizontal_sum);
        assert!(config.core.trusted_hot_loop_shapes);
        assert!(config.linear.kernels);
        assert!(config.linear.dot4_same_x);
        assert!(config.linear.dot8_same_x);
        assert!(config.linear.dot12_same_x);
        assert!(config.linear.dot12_pointer_rows);
        assert!(config.linear.dot12_direct_horizontal_sum);
        assert!(config.linear.blocked_input8_dot12_layout);
        assert!(config.linear.shared_direct_horizontal_sum);
        assert!(config.linear.fixed_in_dim_same_x);
        assert!(config.linear.batched_same_x);
        assert!(config.linear.batched_four_row_same_x);
        assert!(config.linear.batched_pointer_rows);
        assert!(config.linear.fused_bias_add);
        assert!(config.channel_mixer.kernels);
        assert!(config.channel_mixer.residual_scratch);
        assert!(config.channel_mixer.prelude_residual_scratch);
        assert!(config.channel_mixer.flat_buffer_reuse);
        assert!(config.channel_mixer.inplace_output_values);
        assert!(config.channel_mixer.direct_relu_square);
        assert!(config.channel_mixer.flat_working_state);
        assert!(config.time_mixer.projection_kernels);
        assert!(config.time_mixer.v_mix_kernels);
        assert!(config.time_mixer.output_kernels);
        assert!(config.time_mixer.flat_working_state);
        assert!(config.time_mixer.recurrence_output_sums);
        assert!(config.time_mixer.recurrence_output_variances);
        assert!(config.time_mixer.time_decay_kernels);
        assert!(config.layer.time_channel_values_handoff);
        assert!(config.layer.values_carrier);
        assert!(config.layer.rnn_borrow_first_input_values);
        assert!(config.bulk.curve_heads_batch);
        assert!(config.bulk.prediction_head_batch);
        assert!(config.bulk.lane_parallel);
        assert!(config.bulk.lane_parallel_stream_install);
        assert!(config.bulk.lane_parallel_borrow_state);
        assert!(config.bulk.module_output_buffer);
        assert!(config.bulk.module_values_carrier);
        assert!(config.bulk.module_raw_values_handoff);
        assert!(config.bulk.new_singleton_state_views);
        assert!(config.bulk.deck_persistent_frontier_values);
        assert!(config.bulk.card_note_persistent_frontier_values);
        assert!(config.bulk.persistent_frontier_in_place_state_compaction);
        assert!(config.bulk.persistent_frontier_input_buffer_reuse);
        assert_eq!(
            config.bulk.lane_parallel_threads,
            Some(DEFAULT_BULK_LANE_PARALLEL_THREADS)
        );
        assert!(!config.bulk.profile_fast_paths);
    }

    #[test]
    fn simd_backend_preference_is_typed_and_defaults_safely() {
        assert_eq!(
            config(&[(SIMD_BACKEND_ENV_VAR, "pulp")]).core.simd_backend,
            SimdBackendPreference::Pulp
        );
        assert_eq!(
            config(&[(SIMD_BACKEND_ENV_VAR, "AVX2")]).core.simd_backend,
            SimdBackendPreference::Avx2
        );
        assert_eq!(
            config(&[(SIMD_BACKEND_ENV_VAR, "not-a-backend")])
                .core
                .simd_backend,
            SimdBackendPreference::Auto
        );
    }

    #[test]
    fn fast_profile_matches_the_legacy_promoted_flag_set() {
        let fast = config(&[(CPU_PROFILE_ENV_VAR, "fast")]);
        let mut legacy = CpuRuntimeConfig::from_lookup(|name| {
            if name.starts_with("RWKV_SRS_ENABLE_")
                && name != "RWKV_SRS_ENABLE_BULK_PROFILE_FAST_PATHS"
            {
                Some("1".to_string())
            } else if name == "RWKV_SRS_BULK_LANE_PARALLEL_THREADS" {
                Some(DEFAULT_BULK_LANE_PARALLEL_THREADS.to_string())
            } else {
                None
            }
        });
        legacy.profile = CpuProfile::Fast;

        assert_eq!(fast, legacy);
    }

    #[test]
    fn explicit_overrides_disable_fast_paths_and_dependencies() {
        let config = config(&[
            (CPU_PROFILE_ENV_VAR, "fast"),
            ("RWKV_SRS_ENABLE_LINEAR_KERNELS", "0"),
            ("RWKV_SRS_ENABLE_TIME_MIXER_OUTPUT_KERNELS", "false"),
            ("RWKV_SRS_ENABLE_BULK_MODULE_VALUES_CARRIER", "off"),
            ("RWKV_SRS_BULK_LANE_PARALLEL_THREADS", "7"),
        ]);

        assert!(!config.linear.kernels);
        assert!(!config.time_mixer.output_kernels);
        assert!(!config.time_mixer.flat_working_state);
        assert!(!config.channel_mixer.flat_working_state);
        assert!(!config.bulk.module_values_carrier);
        assert!(!config.bulk.deck_persistent_frontier_values);
        assert!(!config.bulk.card_note_persistent_frontier_values);
        assert_eq!(config.bulk.lane_parallel_threads, Some(7));
    }

    #[test]
    fn legacy_individual_flags_still_work_without_a_profile() {
        let config = config(&[
            ("RWKV_SRS_ENABLE_LINEAR_KERNELS", "yes"),
            ("RWKV_SRS_ENABLE_TIME_DECAY_KERNELS", "on"),
            ("RWKV_SRS_ENABLE_BULK_PROFILE_FAST_PATHS", "1"),
        ]);

        assert_eq!(config.profile, CpuProfile::Oracle);
        assert!(config.linear.kernels);
        assert!(config.time_mixer.time_decay_kernels);
        assert!(config.bulk.profile_fast_paths);
        assert!(!config.channel_mixer.kernels);
    }
}
