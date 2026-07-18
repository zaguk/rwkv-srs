//! TimeMixer forward paths, optimized projections, recurrence output, and scratch state.

use super::*;

type TimeMixerLerpProjectionScratchOutput = (
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
);
pub(super) struct TimeMixerLerpProjectionScratchValues {
    pub(super) batch_size: usize,
    pub(super) channels: usize,
    pub(super) heads: usize,
    pub(super) r_values: Vec<f32>,
    pub(super) k_values: Vec<f32>,
    pub(super) k_scale_values: Vec<f32>,
    pub(super) v_scale_values: Vec<f32>,
    pub(super) v_values: Vec<f32>,
    pub(super) d_values: Vec<f32>,
    pub(super) a_values: Vec<f32>,
    pub(super) g_values: Vec<f32>,
}
#[derive(Default)]
struct TimeMixerLerpProjectionScratchPool {
    x_values: Vec<f32>,
    r_input_values: Vec<f32>,
    k_input_values: Vec<f32>,
    r_values: Vec<f32>,
    k_values: Vec<f32>,
    k_scale_values: Vec<f32>,
    v_scale_values: Vec<f32>,
    v_values: Vec<f32>,
    d_values: Vec<f32>,
    a_values: Vec<f32>,
    g_values: Vec<f32>,
    r_row: Vec<f32>,
    k_row: Vec<f32>,
    k_scale_row: Vec<f32>,
    v_scale_row: Vec<f32>,
}
pub(super) struct TimeMixerMiddleOutputScratch {
    pub(super) middle: TimeMixerMiddleScratchOutput,
    pub(super) dot_kernel: NativeLinearDotKernel,
}
pub(super) struct TimeMixerLoraDecayScratchValues {
    pub(super) batch_size: usize,
    pub(super) channels: usize,
    pub(super) a_values: Vec<f32>,
    pub(super) g_values: Vec<f32>,
    pub(super) w_values: Vec<f32>,
}
pub(super) struct TimeMixerVMixScratchValues {
    pub(super) batch_size: usize,
    pub(super) channels: usize,
    pub(super) values: Vec<f32>,
}
#[derive(Clone, Debug)]
pub(crate) struct TimeMixerFlatLayerState {
    pub(super) x_shift_values: Vec<f32>,
    pub(super) state_values: Vec<f32>,
    pub(super) channels: usize,
    pub(super) heads: usize,
    pub(super) head_size: usize,
}

#[derive(Default)]
struct TimeMixerOutputScratchPool {
    hidden_values: Vec<f32>,
    hidden_batch_values: Vec<f32>,
}
thread_local! {
    static TIME_MIXER_LERP_PROJECTION_SCRATCH_POOL: RefCell<TimeMixerLerpProjectionScratchPool> =
        RefCell::new(TimeMixerLerpProjectionScratchPool::default());
    static TIME_MIXER_OUTPUT_SCRATCH_POOL: RefCell<TimeMixerOutputScratchPool> =
        RefCell::new(TimeMixerOutputScratchPool::default());
}

static PREDICT_MANY_LIGHTNING_APPROXIMATIONS_ACTIVE: AtomicUsize = AtomicUsize::new(0);
struct PredictManyLightningActivityGuard;

impl PredictManyLightningActivityGuard {
    fn enter() -> Self {
        PREDICT_MANY_LIGHTNING_APPROXIMATIONS_ACTIVE.fetch_add(1, Ordering::AcqRel);
        Self
    }
}

impl Drop for PredictManyLightningActivityGuard {
    fn drop(&mut self) {
        PREDICT_MANY_LIGHTNING_APPROXIMATIONS_ACTIVE.fetch_sub(1, Ordering::AcqRel);
    }
}

struct TimeDecayLut {
    min: f32,
    max: f32,
    inv_step: f32,
    values: Vec<f32>,
}

struct ActivationLut {
    min: f32,
    max: f32,
    inv_step: f32,
    sigmoid_values: Vec<f32>,
    tanh_values: Vec<f32>,
}

fn predict_many_lightning_decay_lut() -> Option<&'static TimeDecayLut> {
    static LUT: OnceLock<Option<TimeDecayLut>> = OnceLock::new();
    LUT.get_or_init(|| {
        let step = predict_many_lightning_decay_lut_step()?;

        let min = -PREDICT_MANY_LIGHTNING_DECAY_LUT_LIMIT;
        let max = PREDICT_MANY_LIGHTNING_DECAY_LUT_LIMIT;
        let bins = ((max - min) / step).ceil() as usize + 1;
        let mut values = Vec::with_capacity(bins);
        for index in 0..bins {
            let d = (min + index as f32 * step).min(max);
            values.push(time_decay_w_scalar(d));
        }

        Some(TimeDecayLut {
            min,
            max,
            inv_step: 1.0 / step,
            values,
        })
    })
    .as_ref()
}

pub(super) fn predict_many_lightning_time_decay_w_scalar(d: f32) -> f32 {
    if PREDICT_MANY_LIGHTNING_APPROXIMATIONS_ACTIVE.load(Ordering::Relaxed) == 0 {
        return time_decay_w_scalar(d);
    }

    let Some(lut) = predict_many_lightning_decay_lut() else {
        return time_decay_w_scalar(d);
    };
    if !d.is_finite() || d < lut.min || d > lut.max {
        return time_decay_w_scalar(d);
    }

    let index = ((d - lut.min) * lut.inv_step).round() as usize;
    lut.values
        .get(index)
        .copied()
        .unwrap_or_else(|| time_decay_w_scalar(d))
}

fn predict_many_lightning_activation_lut() -> Option<&'static ActivationLut> {
    static LUT: OnceLock<Option<ActivationLut>> = OnceLock::new();
    LUT.get_or_init(|| {
        let step = predict_many_lightning_activation_lut_step()?;

        let min = -PREDICT_MANY_LIGHTNING_ACTIVATION_LUT_LIMIT;
        let max = PREDICT_MANY_LIGHTNING_ACTIVATION_LUT_LIMIT;
        let bins = ((max - min) / step).ceil() as usize + 1;
        let mut sigmoid_values = Vec::with_capacity(bins);
        let mut tanh_values = Vec::with_capacity(bins);
        for index in 0..bins {
            let value = (min + index as f32 * step).min(max);
            sigmoid_values.push(sigmoid_scalar(value));
            tanh_values.push(value.tanh());
        }

        Some(ActivationLut {
            min,
            max,
            inv_step: 1.0 / step,
            sigmoid_values,
            tanh_values,
        })
    })
    .as_ref()
}

pub(super) fn predict_many_lightning_sigmoid_scalar(value: f32) -> f32 {
    if PREDICT_MANY_LIGHTNING_APPROXIMATIONS_ACTIVE.load(Ordering::Relaxed) == 0 {
        return sigmoid_scalar(value);
    }

    let Some(lut) = predict_many_lightning_activation_lut() else {
        return sigmoid_scalar(value);
    };
    if !value.is_finite() {
        return sigmoid_scalar(value);
    }
    if value <= lut.min {
        return 0.0;
    }
    if value >= lut.max {
        return 1.0;
    }

    let index = ((value - lut.min) * lut.inv_step).round() as usize;
    lut.sigmoid_values
        .get(index)
        .copied()
        .unwrap_or_else(|| sigmoid_scalar(value))
}

pub(super) fn predict_many_lightning_tanh_scalar(value: f32) -> f32 {
    if PREDICT_MANY_LIGHTNING_APPROXIMATIONS_ACTIVE.load(Ordering::Relaxed) == 0 {
        return value.tanh();
    }

    let Some(lut) = predict_many_lightning_activation_lut() else {
        return value.tanh();
    };
    if !value.is_finite() {
        return value.tanh();
    }
    if value <= lut.min {
        return -1.0;
    }
    if value >= lut.max {
        return 1.0;
    }

    let index = ((value - lut.min) * lut.inv_step).round() as usize;
    lut.tanh_values
        .get(index)
        .copied()
        .unwrap_or_else(|| value.tanh())
}

#[inline]
pub(super) fn use_direct_time_mixer_activation_scalars() -> bool {
    direct_time_mixer_activation_scalars_enabled()
        && PREDICT_MANY_LIGHTNING_APPROXIMATIONS_ACTIVE.load(Ordering::Relaxed) == 0
}

#[inline]
pub(super) fn time_mixer_sigmoid_scalar(value: f32, use_direct_activation_scalars: bool) -> f32 {
    if use_direct_activation_scalars {
        sigmoid_scalar(value)
    } else {
        predict_many_lightning_sigmoid_scalar(value)
    }
}

#[inline]
pub(super) fn time_mixer_tanh_scalar(value: f32, use_direct_activation_scalars: bool) -> f32 {
    if use_direct_activation_scalars {
        value.tanh()
    } else {
        predict_many_lightning_tanh_scalar(value)
    }
}

#[inline]
pub(super) fn time_mixer_time_decay_w_scalar(
    value: f32,
    use_direct_activation_scalars: bool,
) -> f32 {
    if use_direct_activation_scalars {
        time_decay_w_scalar(value)
    } else {
        predict_many_lightning_time_decay_w_scalar(value)
    }
}

pub(super) fn predict_many_lightning_time_decay_w_b1c(d: &Tensor) -> Result<Tensor> {
    if PREDICT_MANY_LIGHTNING_APPROXIMATIONS_ACTIVE.load(Ordering::Relaxed) == 0 {
        return time_decay_w_b1c(d);
    }

    let (batch_size, one, channels) = d.dims3()?;
    if one != 1 {
        bail!(
            "time_decay_w_b1c expected d shape [B, 1, C], got {:?}",
            d.dims()
        );
    }

    let d_data = f32_tensor_data(d)?;
    let d_values = d_data.as_slice()?;
    let mut w_values = Vec::with_capacity(d_values.len());
    let direct_lut_loop = predict_many_lightning_direct_decay_lut_loop_enabled();
    if direct_lut_loop {
        if let Some(lut) = predict_many_lightning_decay_lut() {
            for &value in d_values {
                let w = if value.is_finite() && value >= lut.min && value <= lut.max {
                    let index = ((value - lut.min) * lut.inv_step).round() as usize;
                    lut.values
                        .get(index)
                        .copied()
                        .unwrap_or_else(|| time_decay_w_scalar(value))
                } else {
                    time_decay_w_scalar(value)
                };
                w_values.push(w);
            }
            return Tensor::from_vec(w_values, (batch_size, 1usize, channels), d.device());
        }
    }
    for d in d_values {
        w_values.push(predict_many_lightning_time_decay_w_scalar(*d));
    }
    Tensor::from_vec(w_values, (batch_size, 1usize, channels), d.device())
}

pub(super) fn predict_many_lightning_lora_rank_limit() -> Option<usize> {
    if PREDICT_MANY_LIGHTNING_APPROXIMATIONS_ACTIVE.load(Ordering::Relaxed) == 0 {
        return None;
    }

    predict_many_lightning_lora_rank_limit_configured()
}

pub(super) fn predict_many_lightning_lora_effective_rank(lora_dim: usize) -> usize {
    predict_many_lightning_lora_rank_limit()
        .map(|limit| limit.min(lora_dim))
        .unwrap_or(lora_dim)
}

pub(super) fn with_predict_many_lightning_approximations<T>(
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let _activity = PredictManyLightningActivityGuard::enter();
    f()
}

pub(super) const TIME_MIXER_GROUP_NORM_EPS: f64 = 64e-5;
pub(super) const PREDICT_MANY_LIGHTNING_DECAY_LUT_LIMIT: f32 = 64.0;
pub(super) const PREDICT_MANY_LIGHTNING_ACTIVATION_LUT_LIMIT: f32 = 16.0;
pub(super) const TIME_MIXER_OUTPUT_HEAD32_CHANNELS: usize = 128;
pub(super) const TIME_MIXER_OUTPUT_HEAD32_HEADS: usize = 4;
pub(super) const TIME_MIXER_OUTPUT_HEAD32_HEAD_SIZE: usize = 32;

pub fn time_mixer_forward(
    weights: &Rwkv7RnnTimeMixerWeights,
    in_bc: &Tensor,
    v0_bc: &Tensor,
    state: Option<(&Tensor, &Tensor)>,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    time_mixer_forward_profiled(weights, in_bc, v0_bc, state, None)
}

pub(super) fn record_time_mixer_lora_temporary(
    profile: &mut Option<&mut ForwardProfile>,
    tensor: &Tensor,
) {
    if let Some(profile) = profile.as_deref_mut() {
        profile.time_mixer.lora_decay_temporary_tensors += 1;
        profile.time_mixer.lora_decay_temporary_values += tensor.elem_count();
    }
}

pub(super) fn time_mixer_lora_decay(
    weights: &Rwkv7RnnTimeMixerWeights,
    a_input_b1c: &Tensor,
    gate_input_b1c: &Tensor,
    d_input_b1c: &Tensor,
    profile: &mut Option<&mut ForwardProfile>,
) -> Result<(Tensor, Tensor, Tensor)> {
    if native_time_mixer_lora_decay_scratch_enabled() {
        if let Some((a_b1c, g_b1c, w_b1c)) = time_mixer_lora_decay_scratch_native(
            weights,
            a_input_b1c,
            gate_input_b1c,
            d_input_b1c,
            profile,
        )? {
            if let Some(profile) = profile.as_deref_mut() {
                let rows = a_b1c.dims3()?.0;
                profile.time_mixer.native_lora_decay_scratch_calls += 1;
                profile.time_mixer.native_lora_decay_scratch_rows += rows;
            }
            record_time_mixer_lora_temporary(profile, &a_b1c);
            record_time_mixer_lora_temporary(profile, &g_b1c);
            record_time_mixer_lora_temporary(profile, &w_b1c);
            return Ok((a_b1c, g_b1c, w_b1c));
        }
    }

    time_mixer_lora_decay_candle(weights, a_input_b1c, gate_input_b1c, d_input_b1c, profile)
}

pub(super) fn time_mixer_lora_decay_candle(
    weights: &Rwkv7RnnTimeMixerWeights,
    a_input_b1c: &Tensor,
    gate_input_b1c: &Tensor,
    d_input_b1c: &Tensor,
    profile: &mut Option<&mut ForwardProfile>,
) -> Result<(Tensor, Tensor, Tensor)> {
    let a_b1c;
    let mut d_b1c = d_input_b1c.clone();

    let a_lora_start = ProfileTimer::start(profile.is_some());
    let start = ProfileTimer::start(profile.is_some());
    let a_lora_a = linear_profiled(a_input_b1c, &weights.a_lora_simple.a, profile)?;
    let elapsed_ns = start.elapsed_ns();
    profile_add!(profile, time_mixer.a_lora_a_projection_ns, elapsed_ns);
    profile_add!(profile, time_mixer.lora_first_projection_ns, elapsed_ns);
    record_time_mixer_lora_temporary(profile, &a_lora_a);
    let start = ProfileTimer::start(profile.is_some());
    let a_lora_b = linear_profiled(&a_lora_a, &weights.a_lora_simple.b_and_lamb, profile)?;
    let elapsed_ns = start.elapsed_ns();
    profile_add!(profile, time_mixer.a_lora_b_projection_ns, elapsed_ns);
    profile_add!(profile, time_mixer.lora_second_projection_ns, elapsed_ns);
    record_time_mixer_lora_temporary(profile, &a_lora_b);
    let activation_start = ProfileTimer::start(profile.is_some());
    a_b1c = nn_ops::sigmoid(&a_lora_b)?;
    record_time_mixer_lora_temporary(profile, &a_b1c);
    profile_add!(
        profile,
        time_mixer.lora_a_activation_ns,
        activation_start.elapsed_ns()
    );
    profile_add!(profile, time_mixer.lora_a_ns, a_lora_start.elapsed_ns());

    let gate_lora_start = ProfileTimer::start(profile.is_some());
    let start = ProfileTimer::start(profile.is_some());
    let gate_a = linear_profiled(gate_input_b1c, &weights.gate_lora.a, profile)?;
    let elapsed_ns = start.elapsed_ns();
    profile_add!(profile, time_mixer.gate_lora_a_projection_ns, elapsed_ns);
    profile_add!(profile, time_mixer.lora_first_projection_ns, elapsed_ns);
    record_time_mixer_lora_temporary(profile, &gate_a);
    let activation_start = ProfileTimer::start(profile.is_some());
    let gate_sigmoid = nn_ops::sigmoid(&gate_a)?;
    record_time_mixer_lora_temporary(profile, &gate_sigmoid);
    profile_add!(
        profile,
        time_mixer.lora_gate_activation_ns,
        activation_start.elapsed_ns()
    );
    let start = ProfileTimer::start(profile.is_some());
    let g_b1c = linear_profiled(&gate_sigmoid, &weights.gate_lora.b, profile)?;
    let elapsed_ns = start.elapsed_ns();
    profile_add!(profile, time_mixer.gate_lora_b_projection_ns, elapsed_ns);
    profile_add!(profile, time_mixer.lora_second_projection_ns, elapsed_ns);
    record_time_mixer_lora_temporary(profile, &g_b1c);
    profile_add!(
        profile,
        time_mixer.lora_gate_ns,
        gate_lora_start.elapsed_ns()
    );

    let d_lora_start = ProfileTimer::start(profile.is_some());
    let start = ProfileTimer::start(profile.is_some());
    d_b1c = linear_profiled(&d_b1c, &weights.d_lora_mlp.a, profile)?;
    let elapsed_ns = start.elapsed_ns();
    profile_add!(profile, time_mixer.d_lora_a_projection_ns, elapsed_ns);
    profile_add!(profile, time_mixer.lora_first_projection_ns, elapsed_ns);
    record_time_mixer_lora_temporary(profile, &d_b1c);
    let activation_start = ProfileTimer::start(profile.is_some());
    d_b1c = d_b1c.tanh()?;
    record_time_mixer_lora_temporary(profile, &d_b1c);
    profile_add!(
        profile,
        time_mixer.lora_d_activation_ns,
        activation_start.elapsed_ns()
    );
    let start = ProfileTimer::start(profile.is_some());
    d_b1c = linear_profiled(&d_b1c, &weights.d_lora_mlp.b_and_lamb, profile)?;
    let elapsed_ns = start.elapsed_ns();
    profile_add!(profile, time_mixer.d_lora_b_projection_ns, elapsed_ns);
    profile_add!(profile, time_mixer.lora_second_projection_ns, elapsed_ns);
    record_time_mixer_lora_temporary(profile, &d_b1c);
    profile_add!(profile, time_mixer.lora_d_ns, d_lora_start.elapsed_ns());

    let start = ProfileTimer::start(profile.is_some());
    let w_b1c = predict_many_lightning_time_decay_w_b1c(&d_b1c)?;
    record_time_mixer_lora_temporary(profile, &w_b1c);
    profile_add!(
        profile,
        time_mixer.lora_decay_elementwise_ns,
        start.elapsed_ns()
    );

    Ok((a_b1c, g_b1c, w_b1c))
}

pub(super) fn time_mixer_lora_decay_scratch_native(
    weights: &Rwkv7RnnTimeMixerWeights,
    a_input_b1c: &Tensor,
    gate_input_b1c: &Tensor,
    d_input_b1c: &Tensor,
    profile: &mut Option<&mut ForwardProfile>,
) -> Result<Option<(Tensor, Tensor, Tensor)>> {
    let Some(dot_kernel) = native_linear_dot_kernel() else {
        return Ok(None);
    };
    time_mixer_lora_decay_scratch_native_with_kernel(
        weights,
        a_input_b1c,
        gate_input_b1c,
        d_input_b1c,
        dot_kernel,
        profile,
    )
}

pub(super) fn validate_lora_linear_pair(
    first: &LinearF32Weights,
    second: &LinearF32Weights,
    input_dim: usize,
    output_dim: usize,
    first_bias: bool,
    second_bias: bool,
) -> bool {
    first.in_dim == input_dim
        && second.out_dim == output_dim
        && second.in_dim == first.out_dim
        && first.bias.is_some() == first_bias
        && second.bias.is_some() == second_bias
        && first.weight.len() == first.out_dim * first.in_dim
        && second.weight.len() == second.out_dim * second.in_dim
        && first
            .bias
            .as_ref()
            .map_or(true, |bias| bias.len() == first.out_dim)
        && second
            .bias
            .as_ref()
            .map_or(true, |bias| bias.len() == second.out_dim)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn time_mixer_lora_decay_scratch_native_with_kernel(
    weights: &Rwkv7RnnTimeMixerWeights,
    a_input_b1c: &Tensor,
    gate_input_b1c: &Tensor,
    d_input_b1c: &Tensor,
    dot_kernel: NativeLinearDotKernel,
    profile: &mut Option<&mut ForwardProfile>,
) -> Result<Option<(Tensor, Tensor, Tensor)>> {
    let Some(values) = time_mixer_lora_decay_scratch_values_native_with_kernel(
        weights,
        a_input_b1c,
        gate_input_b1c,
        d_input_b1c,
        dot_kernel,
        profile,
    )?
    else {
        return Ok(None);
    };

    Ok(Some((
        Tensor::from_vec(
            values.a_values,
            (values.batch_size, 1usize, values.channels),
            a_input_b1c.device(),
        )?,
        Tensor::from_vec(
            values.g_values,
            (values.batch_size, 1usize, values.channels),
            gate_input_b1c.device(),
        )?,
        Tensor::from_vec(
            values.w_values,
            (values.batch_size, 1usize, values.channels),
            d_input_b1c.device(),
        )?,
    )))
}

pub(super) fn repeat_channel_values(
    channel_values: &[f32],
    batch_size: usize,
    channels: usize,
) -> Vec<f32> {
    let mut values = Vec::with_capacity(batch_size * channels);
    for _ in 0..batch_size {
        values.extend_from_slice(channel_values);
    }
    values
}

pub(super) fn time_mixer_lora_decay_rank_zero_values(
    a_lora_b: &LinearF32Weights,
    d_lora_b: &LinearF32Weights,
    batch_size: usize,
    channels: usize,
) -> Option<TimeMixerLoraDecayScratchValues> {
    if predict_many_lightning_lora_rank_limit() != Some(0) {
        return None;
    }
    let a_bias = a_lora_b.bias.as_deref()?;
    let d_bias = d_lora_b.bias.as_deref()?;
    if a_bias.len() != channels || d_bias.len() != channels {
        return None;
    }

    let use_direct_activation_scalars = use_direct_time_mixer_activation_scalars();
    let a_channel_values = a_bias
        .iter()
        .map(|value| time_mixer_sigmoid_scalar(*value, use_direct_activation_scalars))
        .collect::<Vec<_>>();
    let w_channel_values = d_bias
        .iter()
        .map(|value| time_mixer_time_decay_w_scalar(*value, use_direct_activation_scalars))
        .collect::<Vec<_>>();
    Some(TimeMixerLoraDecayScratchValues {
        batch_size,
        channels,
        a_values: repeat_channel_values(&a_channel_values, batch_size, channels),
        g_values: vec![0.0; batch_size * channels],
        w_values: repeat_channel_values(&w_channel_values, batch_size, channels),
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn time_mixer_lora_project_row(
    dot_kernel: NativeLinearDotKernel,
    input_row: &[f32],
    weight_values: &[f32],
    out_dim: usize,
    weight_row_stride: usize,
    output_count: usize,
    bias_values: Option<&[f32]>,
    output_row: &mut [f32],
) {
    debug_assert!(output_count <= out_dim);
    debug_assert!(input_row.len() <= weight_row_stride);
    debug_assert!(weight_values.len() >= out_dim * weight_row_stride);
    debug_assert!(output_row.len() >= out_dim);
    debug_assert!(bias_values.map_or(true, |bias| bias.len() >= out_dim));

    if native_time_mixer_lora_dot4_projections_enabled()
        && native_linear_dot4_same_x_enabled()
        && output_count == out_dim
        && input_row.len() == weight_row_stride
    {
        linear_project_row_same_x(
            dot_kernel,
            input_row,
            weight_values,
            out_dim,
            weight_row_stride,
            &mut output_row[..out_dim],
        );
        if let Some(bias_values) = bias_values {
            for output_index in 0..out_dim {
                output_row[output_index] += bias_values[output_index];
            }
        }
        return;
    }

    for output_index in 0..output_count {
        let weight_row = &weight_values
            [output_index * weight_row_stride..output_index * weight_row_stride + input_row.len()];
        let mut output = linear_dot(dot_kernel, input_row, weight_row);
        if let Some(bias_values) = bias_values {
            output += bias_values[output_index];
        }
        output_row[output_index] = output;
    }
}

#[derive(Clone, Copy)]
pub(super) enum TimeMixerLoraProjectionActivation {
    Sigmoid,
    Tanh,
    TimeDecayW,
}

#[inline]
pub(super) fn apply_time_mixer_lora_projection_activation(
    value: f32,
    activation: TimeMixerLoraProjectionActivation,
    use_direct_activation_scalars: bool,
) -> f32 {
    match activation {
        TimeMixerLoraProjectionActivation::Sigmoid => {
            time_mixer_sigmoid_scalar(value, use_direct_activation_scalars)
        }
        TimeMixerLoraProjectionActivation::Tanh => {
            time_mixer_tanh_scalar(value, use_direct_activation_scalars)
        }
        TimeMixerLoraProjectionActivation::TimeDecayW => {
            time_mixer_time_decay_w_scalar(value, use_direct_activation_scalars)
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn time_mixer_lora_project_row_activated(
    dot_kernel: NativeLinearDotKernel,
    input_row: &[f32],
    weight_values: &[f32],
    out_dim: usize,
    weight_row_stride: usize,
    output_count: usize,
    bias_values: Option<&[f32]>,
    output_row: &mut [f32],
    activation: TimeMixerLoraProjectionActivation,
    use_direct_activation_scalars: bool,
) {
    time_mixer_lora_project_row(
        dot_kernel,
        input_row,
        weight_values,
        out_dim,
        weight_row_stride,
        output_count,
        bias_values,
        output_row,
    );
    for value in &mut output_row[..output_count] {
        *value = apply_time_mixer_lora_projection_activation(
            *value,
            activation,
            use_direct_activation_scalars,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn time_mixer_lora_decay_scratch_values_native_with_kernel(
    weights: &Rwkv7RnnTimeMixerWeights,
    a_input_b1c: &Tensor,
    gate_input_b1c: &Tensor,
    d_input_b1c: &Tensor,
    dot_kernel: NativeLinearDotKernel,
    profile: &mut Option<&mut ForwardProfile>,
) -> Result<Option<TimeMixerLoraDecayScratchValues>> {
    let (batch_size, one, channels) = a_input_b1c.dims3()?;
    if one != 1
        || gate_input_b1c.dims() != [batch_size, 1, channels]
        || d_input_b1c.dims() != [batch_size, 1, channels]
    {
        return Ok(None);
    }

    let cached_weights = time_mixer_native_f32_weights(weights)?;
    let a_lora_a = &cached_weights.a_lora_a;
    let a_lora_b = &cached_weights.a_lora_b;
    let d_lora_a = &cached_weights.d_lora_a;
    let d_lora_b = &cached_weights.d_lora_b;
    let gate_lora_a = &cached_weights.gate_lora_a;
    let gate_lora_b = &cached_weights.gate_lora_b;

    if !validate_lora_linear_pair(a_lora_a, a_lora_b, channels, channels, false, true)
        || !validate_lora_linear_pair(d_lora_a, d_lora_b, channels, channels, false, true)
        || !validate_lora_linear_pair(gate_lora_a, gate_lora_b, channels, channels, false, false)
    {
        return Ok(None);
    }

    if let Some(values) =
        time_mixer_lora_decay_rank_zero_values(a_lora_b, d_lora_b, batch_size, channels)
    {
        return Ok(Some(values));
    }

    let a_input_data = f32_tensor_data(a_input_b1c)?;
    let gate_input_data = f32_tensor_data(gate_input_b1c)?;
    let d_input_data = f32_tensor_data(d_input_b1c)?;
    let a_input_values = a_input_data.as_slice()?;
    let gate_input_values = gate_input_data.as_slice()?;
    let d_input_values = d_input_data.as_slice()?;
    let bxc_len = batch_size * channels;
    if a_input_values.len() != bxc_len
        || gate_input_values.len() != bxc_len
        || d_input_values.len() != bxc_len
    {
        return Ok(None);
    }

    if let Some(values) =
        time_mixer_lora_decay_rank_zero_values(a_lora_b, d_lora_b, batch_size, channels)
    {
        return Ok(Some(values));
    }

    let a_lora_dim = a_lora_a.out_dim;
    let a_lora_effective_dim = predict_many_lightning_lora_effective_rank(a_lora_dim);
    let use_direct_activation_scalars = use_direct_time_mixer_activation_scalars();
    let fuse_lora_activations = native_time_mixer_lora_activation_fusion_enabled();
    let mut a_hidden = vec![0.0f32; batch_size * a_lora_dim];
    let a_total_start = ProfileTimer::start(profile.is_some());
    let start = ProfileTimer::start(profile.is_some());
    for batch_index in 0..batch_size {
        let input_row = &a_input_values[batch_index * channels..(batch_index + 1) * channels];
        let hidden_row = &mut a_hidden[batch_index * a_lora_dim..(batch_index + 1) * a_lora_dim];
        time_mixer_lora_project_row(
            dot_kernel,
            input_row,
            &a_lora_a.weight,
            a_lora_dim,
            channels,
            a_lora_effective_dim,
            None,
            hidden_row,
        );
    }
    let elapsed_ns = start.elapsed_ns();
    profile_add!(profile, time_mixer.a_lora_a_projection_ns, elapsed_ns);
    profile_add!(profile, time_mixer.lora_first_projection_ns, elapsed_ns);

    let mut a_values = vec![0.0f32; bxc_len];
    let a_bias = a_lora_b.bias.as_deref().expect("validated a_lora_b bias");
    if fuse_lora_activations {
        let start = ProfileTimer::start(profile.is_some());
        for batch_index in 0..batch_size {
            let hidden_row = &a_hidden[batch_index * a_lora_dim..(batch_index + 1) * a_lora_dim];
            let out_row = &mut a_values[batch_index * channels..(batch_index + 1) * channels];
            time_mixer_lora_project_row_activated(
                dot_kernel,
                &hidden_row[..a_lora_effective_dim],
                &a_lora_b.weight,
                channels,
                a_lora_dim,
                channels,
                Some(a_bias),
                out_row,
                TimeMixerLoraProjectionActivation::Sigmoid,
                use_direct_activation_scalars,
            );
        }
        let elapsed_ns = start.elapsed_ns();
        profile_add!(profile, time_mixer.a_lora_b_projection_ns, elapsed_ns);
        profile_add!(profile, time_mixer.lora_second_projection_ns, elapsed_ns);
    } else {
        let start = ProfileTimer::start(profile.is_some());
        let mut a_raw = vec![0.0f32; bxc_len];
        for batch_index in 0..batch_size {
            let hidden_row = &a_hidden[batch_index * a_lora_dim..(batch_index + 1) * a_lora_dim];
            let out_row = &mut a_raw[batch_index * channels..(batch_index + 1) * channels];
            time_mixer_lora_project_row(
                dot_kernel,
                &hidden_row[..a_lora_effective_dim],
                &a_lora_b.weight,
                channels,
                a_lora_dim,
                channels,
                Some(a_bias),
                out_row,
            );
        }
        let elapsed_ns = start.elapsed_ns();
        profile_add!(profile, time_mixer.a_lora_b_projection_ns, elapsed_ns);
        profile_add!(profile, time_mixer.lora_second_projection_ns, elapsed_ns);

        let start = ProfileTimer::start(profile.is_some());
        for index in 0..bxc_len {
            a_values[index] =
                time_mixer_sigmoid_scalar(a_raw[index], use_direct_activation_scalars);
        }
        profile_add!(profile, time_mixer.lora_a_activation_ns, start.elapsed_ns());
    }
    profile_add!(profile, time_mixer.lora_a_ns, a_total_start.elapsed_ns());

    let gate_total_start = ProfileTimer::start(profile.is_some());
    let start = ProfileTimer::start(profile.is_some());
    let gate_lora_dim = gate_lora_a.out_dim;
    let gate_lora_effective_dim = predict_many_lightning_lora_effective_rank(gate_lora_dim);
    let mut gate_hidden = vec![0.0f32; batch_size * gate_lora_dim];
    for batch_index in 0..batch_size {
        let input_row = &gate_input_values[batch_index * channels..(batch_index + 1) * channels];
        let hidden_row =
            &mut gate_hidden[batch_index * gate_lora_dim..(batch_index + 1) * gate_lora_dim];
        if fuse_lora_activations {
            time_mixer_lora_project_row_activated(
                dot_kernel,
                input_row,
                &gate_lora_a.weight,
                gate_lora_dim,
                channels,
                gate_lora_effective_dim,
                None,
                hidden_row,
                TimeMixerLoraProjectionActivation::Sigmoid,
                use_direct_activation_scalars,
            );
        } else {
            time_mixer_lora_project_row(
                dot_kernel,
                input_row,
                &gate_lora_a.weight,
                gate_lora_dim,
                channels,
                gate_lora_effective_dim,
                None,
                hidden_row,
            );
        }
    }
    let elapsed_ns = start.elapsed_ns();
    profile_add!(profile, time_mixer.gate_lora_a_projection_ns, elapsed_ns);
    profile_add!(profile, time_mixer.lora_first_projection_ns, elapsed_ns);

    if !fuse_lora_activations {
        let start = ProfileTimer::start(profile.is_some());
        for value in &mut gate_hidden {
            *value = time_mixer_sigmoid_scalar(*value, use_direct_activation_scalars);
        }
        profile_add!(
            profile,
            time_mixer.lora_gate_activation_ns,
            start.elapsed_ns()
        );
    }

    let start = ProfileTimer::start(profile.is_some());
    let mut g_values = vec![0.0f32; bxc_len];
    for batch_index in 0..batch_size {
        let hidden_row =
            &gate_hidden[batch_index * gate_lora_dim..(batch_index + 1) * gate_lora_dim];
        let out_row = &mut g_values[batch_index * channels..(batch_index + 1) * channels];
        time_mixer_lora_project_row(
            dot_kernel,
            &hidden_row[..gate_lora_effective_dim],
            &gate_lora_b.weight,
            channels,
            gate_lora_dim,
            channels,
            None,
            out_row,
        );
    }
    let elapsed_ns = start.elapsed_ns();
    profile_add!(profile, time_mixer.gate_lora_b_projection_ns, elapsed_ns);
    profile_add!(profile, time_mixer.lora_second_projection_ns, elapsed_ns);
    profile_add!(
        profile,
        time_mixer.lora_gate_ns,
        gate_total_start.elapsed_ns()
    );

    let d_total_start = ProfileTimer::start(profile.is_some());
    let start = ProfileTimer::start(profile.is_some());
    let d_lora_dim = d_lora_a.out_dim;
    let d_lora_effective_dim = predict_many_lightning_lora_effective_rank(d_lora_dim);
    let mut d_hidden = vec![0.0f32; batch_size * d_lora_dim];
    for batch_index in 0..batch_size {
        let input_row = &d_input_values[batch_index * channels..(batch_index + 1) * channels];
        let hidden_row = &mut d_hidden[batch_index * d_lora_dim..(batch_index + 1) * d_lora_dim];
        if fuse_lora_activations {
            time_mixer_lora_project_row_activated(
                dot_kernel,
                input_row,
                &d_lora_a.weight,
                d_lora_dim,
                channels,
                d_lora_effective_dim,
                None,
                hidden_row,
                TimeMixerLoraProjectionActivation::Tanh,
                use_direct_activation_scalars,
            );
        } else {
            time_mixer_lora_project_row(
                dot_kernel,
                input_row,
                &d_lora_a.weight,
                d_lora_dim,
                channels,
                d_lora_effective_dim,
                None,
                hidden_row,
            );
        }
    }
    let elapsed_ns = start.elapsed_ns();
    profile_add!(profile, time_mixer.d_lora_a_projection_ns, elapsed_ns);
    profile_add!(profile, time_mixer.lora_first_projection_ns, elapsed_ns);

    if !fuse_lora_activations {
        let start = ProfileTimer::start(profile.is_some());
        for value in &mut d_hidden {
            *value = time_mixer_tanh_scalar(*value, use_direct_activation_scalars);
        }
        profile_add!(profile, time_mixer.lora_d_activation_ns, start.elapsed_ns());
    }

    let mut w_values = vec![0.0f32; bxc_len];
    let d_bias = d_lora_b.bias.as_deref().expect("validated d_lora_b bias");
    if fuse_lora_activations {
        let start = ProfileTimer::start(profile.is_some());
        for batch_index in 0..batch_size {
            let hidden_row = &d_hidden[batch_index * d_lora_dim..(batch_index + 1) * d_lora_dim];
            let out_row = &mut w_values[batch_index * channels..(batch_index + 1) * channels];
            time_mixer_lora_project_row_activated(
                dot_kernel,
                &hidden_row[..d_lora_effective_dim],
                &d_lora_b.weight,
                channels,
                d_lora_dim,
                channels,
                Some(d_bias),
                out_row,
                TimeMixerLoraProjectionActivation::TimeDecayW,
                use_direct_activation_scalars,
            );
        }
        let elapsed_ns = start.elapsed_ns();
        profile_add!(profile, time_mixer.d_lora_b_projection_ns, elapsed_ns);
        profile_add!(profile, time_mixer.lora_second_projection_ns, elapsed_ns);
    } else {
        let start = ProfileTimer::start(profile.is_some());
        let mut d_raw = vec![0.0f32; bxc_len];
        for batch_index in 0..batch_size {
            let hidden_row = &d_hidden[batch_index * d_lora_dim..(batch_index + 1) * d_lora_dim];
            let out_row = &mut d_raw[batch_index * channels..(batch_index + 1) * channels];
            time_mixer_lora_project_row(
                dot_kernel,
                &hidden_row[..d_lora_effective_dim],
                &d_lora_b.weight,
                channels,
                d_lora_dim,
                channels,
                Some(d_bias),
                out_row,
            );
        }
        let elapsed_ns = start.elapsed_ns();
        profile_add!(profile, time_mixer.d_lora_b_projection_ns, elapsed_ns);
        profile_add!(profile, time_mixer.lora_second_projection_ns, elapsed_ns);

        let start = ProfileTimer::start(profile.is_some());
        for index in 0..bxc_len {
            w_values[index] =
                time_mixer_time_decay_w_scalar(d_raw[index], use_direct_activation_scalars);
        }
        profile_add!(
            profile,
            time_mixer.lora_decay_elementwise_ns,
            start.elapsed_ns()
        );
    }
    profile_add!(profile, time_mixer.lora_d_ns, d_total_start.elapsed_ns());

    Ok(Some(TimeMixerLoraDecayScratchValues {
        batch_size,
        channels,
        a_values,
        g_values,
        w_values,
    }))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn time_mixer_lora_decay_scratch_values_native_from_input_values_with_kernel(
    weights: &Rwkv7RnnTimeMixerWeights,
    a_input_values: &[f32],
    gate_input_values: &[f32],
    d_input_values: &[f32],
    batch_size: usize,
    channels: usize,
    dot_kernel: NativeLinearDotKernel,
    profile: &mut Option<&mut ForwardProfile>,
) -> Result<Option<TimeMixerLoraDecayScratchValues>> {
    let bxc_len = batch_size * channels;
    if a_input_values.len() != bxc_len
        || gate_input_values.len() != bxc_len
        || d_input_values.len() != bxc_len
    {
        return Ok(None);
    }

    let cached_weights = time_mixer_native_f32_weights(weights)?;
    let a_lora_a = &cached_weights.a_lora_a;
    let a_lora_b = &cached_weights.a_lora_b;
    let d_lora_a = &cached_weights.d_lora_a;
    let d_lora_b = &cached_weights.d_lora_b;
    let gate_lora_a = &cached_weights.gate_lora_a;
    let gate_lora_b = &cached_weights.gate_lora_b;

    if !validate_lora_linear_pair(a_lora_a, a_lora_b, channels, channels, false, true)
        || !validate_lora_linear_pair(d_lora_a, d_lora_b, channels, channels, false, true)
        || !validate_lora_linear_pair(gate_lora_a, gate_lora_b, channels, channels, false, false)
    {
        return Ok(None);
    }

    let a_lora_dim = a_lora_a.out_dim;
    let a_lora_effective_dim = predict_many_lightning_lora_effective_rank(a_lora_dim);
    let use_direct_activation_scalars = use_direct_time_mixer_activation_scalars();
    let fuse_lora_activations = native_time_mixer_lora_activation_fusion_enabled();
    let mut a_hidden = vec![0.0f32; batch_size * a_lora_dim];
    let a_total_start = ProfileTimer::start(profile.is_some());
    let start = ProfileTimer::start(profile.is_some());
    for batch_index in 0..batch_size {
        let input_row = &a_input_values[batch_index * channels..(batch_index + 1) * channels];
        let hidden_row = &mut a_hidden[batch_index * a_lora_dim..(batch_index + 1) * a_lora_dim];
        time_mixer_lora_project_row(
            dot_kernel,
            input_row,
            &a_lora_a.weight,
            a_lora_dim,
            channels,
            a_lora_effective_dim,
            None,
            hidden_row,
        );
    }
    let elapsed_ns = start.elapsed_ns();
    profile_add!(profile, time_mixer.a_lora_a_projection_ns, elapsed_ns);
    profile_add!(profile, time_mixer.lora_first_projection_ns, elapsed_ns);

    let mut a_values = vec![0.0f32; bxc_len];
    let a_bias = a_lora_b.bias.as_deref().expect("validated a_lora_b bias");
    if fuse_lora_activations {
        let start = ProfileTimer::start(profile.is_some());
        for batch_index in 0..batch_size {
            let hidden_row = &a_hidden[batch_index * a_lora_dim..(batch_index + 1) * a_lora_dim];
            let out_row = &mut a_values[batch_index * channels..(batch_index + 1) * channels];
            time_mixer_lora_project_row_activated(
                dot_kernel,
                &hidden_row[..a_lora_effective_dim],
                &a_lora_b.weight,
                channels,
                a_lora_dim,
                channels,
                Some(a_bias),
                out_row,
                TimeMixerLoraProjectionActivation::Sigmoid,
                use_direct_activation_scalars,
            );
        }
        let elapsed_ns = start.elapsed_ns();
        profile_add!(profile, time_mixer.a_lora_b_projection_ns, elapsed_ns);
        profile_add!(profile, time_mixer.lora_second_projection_ns, elapsed_ns);
    } else {
        let start = ProfileTimer::start(profile.is_some());
        let mut a_raw = vec![0.0f32; bxc_len];
        for batch_index in 0..batch_size {
            let hidden_row = &a_hidden[batch_index * a_lora_dim..(batch_index + 1) * a_lora_dim];
            let out_row = &mut a_raw[batch_index * channels..(batch_index + 1) * channels];
            time_mixer_lora_project_row(
                dot_kernel,
                &hidden_row[..a_lora_effective_dim],
                &a_lora_b.weight,
                channels,
                a_lora_dim,
                channels,
                Some(a_bias),
                out_row,
            );
        }
        let elapsed_ns = start.elapsed_ns();
        profile_add!(profile, time_mixer.a_lora_b_projection_ns, elapsed_ns);
        profile_add!(profile, time_mixer.lora_second_projection_ns, elapsed_ns);

        let start = ProfileTimer::start(profile.is_some());
        for index in 0..bxc_len {
            a_values[index] =
                time_mixer_sigmoid_scalar(a_raw[index], use_direct_activation_scalars);
        }
        profile_add!(profile, time_mixer.lora_a_activation_ns, start.elapsed_ns());
    }
    profile_add!(profile, time_mixer.lora_a_ns, a_total_start.elapsed_ns());

    let gate_total_start = ProfileTimer::start(profile.is_some());
    let start = ProfileTimer::start(profile.is_some());
    let gate_lora_dim = gate_lora_a.out_dim;
    let gate_lora_effective_dim = predict_many_lightning_lora_effective_rank(gate_lora_dim);
    let mut gate_hidden = vec![0.0f32; batch_size * gate_lora_dim];
    for batch_index in 0..batch_size {
        let input_row = &gate_input_values[batch_index * channels..(batch_index + 1) * channels];
        let hidden_row =
            &mut gate_hidden[batch_index * gate_lora_dim..(batch_index + 1) * gate_lora_dim];
        if fuse_lora_activations {
            time_mixer_lora_project_row_activated(
                dot_kernel,
                input_row,
                &gate_lora_a.weight,
                gate_lora_dim,
                channels,
                gate_lora_effective_dim,
                None,
                hidden_row,
                TimeMixerLoraProjectionActivation::Sigmoid,
                use_direct_activation_scalars,
            );
        } else {
            time_mixer_lora_project_row(
                dot_kernel,
                input_row,
                &gate_lora_a.weight,
                gate_lora_dim,
                channels,
                gate_lora_effective_dim,
                None,
                hidden_row,
            );
        }
    }
    let elapsed_ns = start.elapsed_ns();
    profile_add!(profile, time_mixer.gate_lora_a_projection_ns, elapsed_ns);
    profile_add!(profile, time_mixer.lora_first_projection_ns, elapsed_ns);

    if !fuse_lora_activations {
        let start = ProfileTimer::start(profile.is_some());
        for value in &mut gate_hidden {
            *value = time_mixer_sigmoid_scalar(*value, use_direct_activation_scalars);
        }
        profile_add!(
            profile,
            time_mixer.lora_gate_activation_ns,
            start.elapsed_ns()
        );
    }

    let start = ProfileTimer::start(profile.is_some());
    let mut g_values = vec![0.0f32; bxc_len];
    for batch_index in 0..batch_size {
        let hidden_row =
            &gate_hidden[batch_index * gate_lora_dim..(batch_index + 1) * gate_lora_dim];
        let out_row = &mut g_values[batch_index * channels..(batch_index + 1) * channels];
        time_mixer_lora_project_row(
            dot_kernel,
            &hidden_row[..gate_lora_effective_dim],
            &gate_lora_b.weight,
            channels,
            gate_lora_dim,
            channels,
            None,
            out_row,
        );
    }
    let elapsed_ns = start.elapsed_ns();
    profile_add!(profile, time_mixer.gate_lora_b_projection_ns, elapsed_ns);
    profile_add!(profile, time_mixer.lora_second_projection_ns, elapsed_ns);
    profile_add!(
        profile,
        time_mixer.lora_gate_ns,
        gate_total_start.elapsed_ns()
    );

    let d_total_start = ProfileTimer::start(profile.is_some());
    let start = ProfileTimer::start(profile.is_some());
    let d_lora_dim = d_lora_a.out_dim;
    let d_lora_effective_dim = predict_many_lightning_lora_effective_rank(d_lora_dim);
    let mut d_hidden = vec![0.0f32; batch_size * d_lora_dim];
    for batch_index in 0..batch_size {
        let input_row = &d_input_values[batch_index * channels..(batch_index + 1) * channels];
        let hidden_row = &mut d_hidden[batch_index * d_lora_dim..(batch_index + 1) * d_lora_dim];
        if fuse_lora_activations {
            time_mixer_lora_project_row_activated(
                dot_kernel,
                input_row,
                &d_lora_a.weight,
                d_lora_dim,
                channels,
                d_lora_effective_dim,
                None,
                hidden_row,
                TimeMixerLoraProjectionActivation::Tanh,
                use_direct_activation_scalars,
            );
        } else {
            time_mixer_lora_project_row(
                dot_kernel,
                input_row,
                &d_lora_a.weight,
                d_lora_dim,
                channels,
                d_lora_effective_dim,
                None,
                hidden_row,
            );
        }
    }
    let elapsed_ns = start.elapsed_ns();
    profile_add!(profile, time_mixer.d_lora_a_projection_ns, elapsed_ns);
    profile_add!(profile, time_mixer.lora_first_projection_ns, elapsed_ns);

    if !fuse_lora_activations {
        let start = ProfileTimer::start(profile.is_some());
        for value in &mut d_hidden {
            *value = time_mixer_tanh_scalar(*value, use_direct_activation_scalars);
        }
        profile_add!(profile, time_mixer.lora_d_activation_ns, start.elapsed_ns());
    }

    let mut w_values = vec![0.0f32; bxc_len];
    let d_bias = d_lora_b.bias.as_deref().expect("validated d_lora_b bias");
    if fuse_lora_activations {
        let start = ProfileTimer::start(profile.is_some());
        for batch_index in 0..batch_size {
            let hidden_row = &d_hidden[batch_index * d_lora_dim..(batch_index + 1) * d_lora_dim];
            let out_row = &mut w_values[batch_index * channels..(batch_index + 1) * channels];
            time_mixer_lora_project_row_activated(
                dot_kernel,
                &hidden_row[..d_lora_effective_dim],
                &d_lora_b.weight,
                channels,
                d_lora_dim,
                channels,
                Some(d_bias),
                out_row,
                TimeMixerLoraProjectionActivation::TimeDecayW,
                use_direct_activation_scalars,
            );
        }
        let elapsed_ns = start.elapsed_ns();
        profile_add!(profile, time_mixer.d_lora_b_projection_ns, elapsed_ns);
        profile_add!(profile, time_mixer.lora_second_projection_ns, elapsed_ns);
    } else {
        let start = ProfileTimer::start(profile.is_some());
        let mut d_raw = vec![0.0f32; bxc_len];
        for batch_index in 0..batch_size {
            let hidden_row = &d_hidden[batch_index * d_lora_dim..(batch_index + 1) * d_lora_dim];
            let out_row = &mut d_raw[batch_index * channels..(batch_index + 1) * channels];
            time_mixer_lora_project_row(
                dot_kernel,
                &hidden_row[..d_lora_effective_dim],
                &d_lora_b.weight,
                channels,
                d_lora_dim,
                channels,
                Some(d_bias),
                out_row,
            );
        }
        let elapsed_ns = start.elapsed_ns();
        profile_add!(profile, time_mixer.d_lora_b_projection_ns, elapsed_ns);
        profile_add!(profile, time_mixer.lora_second_projection_ns, elapsed_ns);

        let start = ProfileTimer::start(profile.is_some());
        for index in 0..bxc_len {
            w_values[index] =
                time_mixer_time_decay_w_scalar(d_raw[index], use_direct_activation_scalars);
        }
        profile_add!(
            profile,
            time_mixer.lora_decay_elementwise_ns,
            start.elapsed_ns()
        );
    }
    profile_add!(profile, time_mixer.lora_d_ns, d_total_start.elapsed_ns());

    Ok(Some(TimeMixerLoraDecayScratchValues {
        batch_size,
        channels,
        a_values,
        g_values,
        w_values,
    }))
}

pub(super) fn time_mixer_forward_profiled(
    weights: &Rwkv7RnnTimeMixerWeights,
    in_bc: &Tensor,
    v0_bc: &Tensor,
    state: Option<(&Tensor, &Tensor)>,
    mut profile: Option<&mut ForwardProfile>,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let (out_bc, next_v0_bc, next_time_x, next_time_state) =
        time_mixer_forward_profiled_options(weights, in_bc, v0_bc, state, true, profile.take())?;
    Ok((
        out_bc,
        next_v0_bc,
        next_time_x.expect("state-producing time mixer returns x-shift state"),
        next_time_state.expect("state-producing time mixer returns recurrent state"),
    ))
}

pub(super) fn time_mixer_forward_profiled_options(
    weights: &Rwkv7RnnTimeMixerWeights,
    in_bc: &Tensor,
    v0_bc: &Tensor,
    state: Option<(&Tensor, &Tensor)>,
    return_state: bool,
    mut profile: Option<&mut ForwardProfile>,
) -> Result<(Tensor, Tensor, Option<Tensor>, Option<Tensor>)> {
    let total_start = ProfileTimer::start(profile.is_some());
    profile_inc!(profile, time_mixer.calls);

    let start = ProfileTimer::start(profile.is_some());
    let (batch_size, channels) = in_bc.dims2()?;
    validate_time_mixer_shapes(weights, channels)?;
    if v0_bc.dims() != [batch_size, channels] {
        bail!(
            "time_mixer expected v0_BC shape [{batch_size}, {channels}], got {:?}",
            v0_bc.dims()
        );
    }
    profile_add!(profile, time_mixer.validate_ns, start.elapsed_ns());

    let heads = weights.n_heads;
    let head_size = weights.head_size;
    let start = ProfileTimer::start(profile.is_some());
    let in_b1c = in_bc.unsqueeze(1)?;
    let x_b1c = layer_norm(&in_b1c, &weights.layer_norm)?;
    profile_add!(profile, time_mixer.layer_norm_ns, start.elapsed_ns());

    let start = ProfileTimer::start(profile.is_some());
    let zero_state = if state.is_none() {
        Some(Tensor::zeros(
            (batch_size, 1usize, heads, head_size, head_size),
            DType::F32,
            in_bc.device(),
        )?)
    } else {
        None
    };
    let (x_shift_b1c, state_b1hkk) = match state {
        Some((x_shift_b1c, state_b1hkk)) => {
            validate_time_mixer_state_shapes(
                x_shift_b1c,
                state_b1hkk,
                batch_size,
                channels,
                heads,
                head_size,
            )?;
            (x_shift_b1c, state_b1hkk)
        }
        None => (&x_b1c, zero_state.as_ref().expect("zero state is set")),
    };
    profile_add!(profile, time_mixer.state_ns, start.elapsed_ns());

    if native_time_mixer_lerp_projection_values_scratch_enabled()
        && native_time_mixer_lerp_projection_scratch_enabled()
        && native_time_mixer_v_mix_values_scratch_enabled()
        && native_time_mixer_v_mix_enabled()
        && native_time_mixer_post_lora_scratch_enabled()
        && native_time_mixer_lora_decay_scratch_enabled()
        && native_time_mixer_middle_output_scratch_enabled()
        && native_time_mixer_output_enabled()
        && native_time_mixer_group_output_scratch_enabled()
        && (weights.layer_id != 0 || native_time_mixer_layer0_v_values_scratch_enabled())
    {
        if let Some(dot_kernel) = native_linear_dot_kernel() {
            let projection_start = ProfileTimer::start(profile.is_some());
            if let Some(projection_values) =
                time_mixer_lerp_projection_scratch_values_native_with_kernel(
                    weights,
                    &x_b1c,
                    x_shift_b1c,
                    dot_kernel,
                )?
            {
                if projection_values.batch_size == batch_size
                    && projection_values.channels == channels
                    && projection_values.heads == heads
                {
                    if let Some(profile) = profile.as_deref_mut() {
                        profile.time_mixer.projection_ns += projection_start.elapsed_ns();
                        profile.time_mixer.native_projection_group_calls += 1;
                        profile.time_mixer.native_projection_group_rows += batch_size;
                    }

                    let start = ProfileTimer::start(profile.is_some());
                    if let Some(v_mix_values) = if weights.layer_id == 0 {
                        time_mixer_layer0_v_values_native_from_input_values_with_kernel(
                            &projection_values.v_values,
                            batch_size,
                            channels,
                            weights,
                            dot_kernel,
                        )?
                    } else {
                        time_mixer_v_mix_values_native_from_input_values_with_kernel(
                            &projection_values.v_values,
                            batch_size,
                            channels,
                            v0_bc,
                            weights,
                            dot_kernel,
                        )?
                    } {
                        if let Some(profile) = profile.as_deref_mut() {
                            profile.time_mixer.native_v_mix_calls += 1;
                            profile.time_mixer.native_v_mix_rows += v_mix_values.batch_size;
                        }
                        profile_add!(profile, time_mixer.v_mix_ns, start.elapsed_ns());

                        let lora_decay_start = ProfileTimer::start(profile.is_some());
                        if let Some(lora_values) =
                            time_mixer_lora_decay_scratch_values_native_from_input_values_with_kernel(
                                weights,
                                &projection_values.a_values,
                                &projection_values.g_values,
                                &projection_values.d_values,
                                batch_size,
                                channels,
                                dot_kernel,
                                &mut profile,
                            )?
                        {
                            if let Some(profile) = profile.as_deref_mut() {
                                profile.time_mixer.native_lora_decay_scratch_calls += 1;
                                profile.time_mixer.native_lora_decay_scratch_rows += lora_values.batch_size;
                            }
                            profile_add!(
                                profile,
                                time_mixer.lora_decay_ns,
                                lora_decay_start.elapsed_ns()
                            );

                            if native_time_mixer_middle_all_values_scratch_enabled() {
                                let start = ProfileTimer::start(profile.is_some());
                                let middle_scratch = if let Some(profile) = profile.as_deref_mut() {
                                    time_mixer_middle_output_scratch_native_from_projection_and_lora_values(
                                        &in_b1c,
                                        &projection_values,
                                        &v_mix_values,
                                        &lora_values,
                                        state_b1hkk,
                                        weights,
                                        heads,
                                        head_size,
                                        return_state,
                                        false,
                                        Some(&mut profile.time_mixer.single_timestep),
                                    )?
                                } else {
                                    time_mixer_middle_output_scratch_native_from_projection_and_lora_values(
                                        &in_b1c,
                                        &projection_values,
                                        &v_mix_values,
                                        &lora_values,
                                        state_b1hkk,
                                        weights,
                                        heads,
                                        head_size,
                                        return_state,
                                        false,
                                        None,
                                    )?
                                };
                                profile_add!(profile, time_mixer.recurrence_ns, start.elapsed_ns());

                                if let Some(mut middle_scratch) = middle_scratch {
                                    let start = ProfileTimer::start(profile.is_some());
                                    let output = if native_time_mixer_group_output_cached_aux_enabled()
                                    {
                                        time_mixer_group_output_scratch_native_from_cached_aux_values(
                                            in_bc,
                                            &projection_values.r_values,
                                            &lora_values,
                                            weights,
                                            &middle_scratch,
                                        )?
                                    } else {
                                        time_mixer_group_output_scratch_native_from_lora_and_r_values(
                                            &in_b1c,
                                            &projection_values.r_values,
                                            &weights.bonus,
                                            &lora_values,
                                            weights,
                                            &middle_scratch,
                                        )?
                                    };
                                    if let Some(out_bc) = output {
                                        if let Some(profile) = profile.as_deref_mut() {
                                            profile.time_mixer.native_output_calls += 1;
                                            profile.time_mixer.native_output_rows += batch_size;
                                        }
                                        let next_state_b1hkk = middle_scratch
                                            .middle
                                            .next_state_bhkk
                                            .take()
                                            .map(|next_state_bhkk| next_state_bhkk.unsqueeze(1))
                                            .transpose()?;
                                        let next_v0_bc = if weights.layer_id == 0 {
                                            Tensor::from_vec(
                                                v_mix_values.values.clone(),
                                                (batch_size, channels),
                                                in_bc.device(),
                                            )?
                                        } else {
                                            v0_bc.clone()
                                        };
                                        recycle_time_mixer_middle_scratch_output(
                                            middle_scratch.middle,
                                        );
                                        recycle_time_mixer_lerp_projection_scratch_values(
                                            projection_values,
                                        );
                                        profile_add!(profile, time_mixer.output_ns, start.elapsed_ns());
                                        profile_add!(
                                            profile,
                                            time_mixer.total_ns,
                                            total_start.elapsed_ns()
                                        );

                                        return Ok((
                                            out_bc,
                                            next_v0_bc,
                                            return_state.then_some(x_b1c),
                                            next_state_b1hkk,
                                        ));
                                    }
                                    recycle_time_mixer_middle_scratch_output(middle_scratch.middle);
                                }
                            }

                            let r_b1c = Tensor::from_vec(
                                projection_values.r_values,
                                (batch_size, 1usize, channels),
                                in_bc.device(),
                            )?;
                            let k_b1c = Tensor::from_vec(
                                projection_values.k_values,
                                (batch_size, 1usize, channels),
                                in_bc.device(),
                            )?;
                            let k_scale_b1h = Tensor::from_vec(
                                projection_values.k_scale_values,
                                (batch_size, 1usize, heads),
                                in_bc.device(),
                            )?;
                            let v_scale_b1h = Tensor::from_vec(
                                projection_values.v_scale_values,
                                (batch_size, 1usize, heads),
                                in_bc.device(),
                            )?;

                            let start = ProfileTimer::start(profile.is_some());
                            let middle_scratch = if let Some(profile) = profile.as_deref_mut() {
                                time_mixer_middle_output_scratch_native_from_lora_and_v_values(
                                    &in_b1c,
                                    &r_b1c,
                                    &k_b1c,
                                    &v_mix_values,
                                    &lora_values,
                                    &k_scale_b1h,
                                    &v_scale_b1h,
                                    state_b1hkk,
                                    weights,
                                    heads,
                                    head_size,
                                    return_state,
                                    false,
                                    Some(&mut profile.time_mixer.single_timestep),
                                )?
                            } else {
                                time_mixer_middle_output_scratch_native_from_lora_and_v_values(
                                    &in_b1c,
                                    &r_b1c,
                                    &k_b1c,
                                    &v_mix_values,
                                    &lora_values,
                                    &k_scale_b1h,
                                    &v_scale_b1h,
                                    state_b1hkk,
                                    weights,
                                    heads,
                                    head_size,
                                    return_state,
                                    false,
                                    None,
                                )?
                            };
                            profile_add!(profile, time_mixer.recurrence_ns, start.elapsed_ns());

                            if let Some(mut middle_scratch) = middle_scratch {
                                let start = ProfileTimer::start(profile.is_some());
                                if let Some(out_bc) =
                                    time_mixer_group_output_scratch_native_from_lora_values(
                                        &in_b1c,
                                        &r_b1c,
                                        &weights.bonus,
                                        &lora_values,
                                        weights,
                                        &middle_scratch,
                                    )?
                                {
                                    if let Some(profile) = profile.as_deref_mut() {
                                        profile.time_mixer.native_output_calls += 1;
                                        profile.time_mixer.native_output_rows += batch_size;
                                    }
                                    let next_state_b1hkk = middle_scratch
                                        .middle
                                        .next_state_bhkk
                                        .take()
                                        .map(|next_state_bhkk| next_state_bhkk.unsqueeze(1))
                                        .transpose()?;
                                    recycle_time_mixer_middle_scratch_output(middle_scratch.middle);
                                    profile_add!(profile, time_mixer.output_ns, start.elapsed_ns());
                                    profile_add!(
                                        profile,
                                        time_mixer.total_ns,
                                        total_start.elapsed_ns()
                                    );

                                    return Ok((
                                        out_bc,
                                        v0_bc.clone(),
                                        return_state.then_some(x_b1c),
                                        next_state_b1hkk,
                                    ));
                                }
                                recycle_time_mixer_middle_scratch_output(middle_scratch.middle);
                            }
                        }
                    }
                }
            }
        }
    }

    let fused_lerp_projection = if native_time_mixer_lerp_projection_scratch_enabled() {
        let start = ProfileTimer::start(profile.is_some());
        let output = time_mixer_lerp_projection_scratch_native(weights, &x_b1c, x_shift_b1c)?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.time_mixer.projection_ns += start.elapsed_ns();
            if output.is_some() {
                profile.time_mixer.native_projection_group_calls += 1;
                profile.time_mixer.native_projection_group_rows += batch_size;
            }
        }
        output
    } else {
        None
    };

    let (r_b1c, k_b1c, k_scale_b1h, v_scale_b1h, mut v_b1c, d_b1c, a_b1c, g_b1c) =
        if let Some(output) = fused_lerp_projection {
            output
        } else {
            let start = ProfileTimer::start(profile.is_some());
            let rkvdag_parts = time_lerp_parts_b1c(&x_b1c, x_shift_b1c, &weights.rkvdag_lerp)?;
            profile_add!(profile, time_mixer.lerp_split_ns, start.elapsed_ns());

            let start = ProfileTimer::start(profile.is_some());
            let (r_b1c, k_b1c, k_scale_b1h, v_scale_b1h) =
                time_mixer_projection_group(weights, &rkvdag_parts, &mut profile)?;
            let v_b1c = rkvdag_parts[2].clone();
            let d_b1c = rkvdag_parts[3].clone();
            let a_b1c = rkvdag_parts[4].clone();
            let g_b1c = rkvdag_parts[5].clone();
            profile_add!(profile, time_mixer.projection_ns, start.elapsed_ns());
            (
                r_b1c,
                k_b1c,
                k_scale_b1h,
                v_scale_b1h,
                v_b1c,
                d_b1c,
                a_b1c,
                g_b1c,
            )
        };

    if native_time_mixer_v_mix_values_scratch_enabled()
        && native_time_mixer_v_mix_enabled()
        && native_time_mixer_post_lora_scratch_enabled()
        && native_time_mixer_lora_decay_scratch_enabled()
        && native_time_mixer_middle_output_scratch_enabled()
        && native_time_mixer_output_enabled()
        && native_time_mixer_group_output_scratch_enabled()
        && weights.layer_id != 0
    {
        if let Some(dot_kernel) = native_linear_dot_kernel() {
            let start = ProfileTimer::start(profile.is_some());
            if let Some(v_mix_values) =
                time_mixer_v_mix_values_native_with_kernel(&v_b1c, v0_bc, weights, dot_kernel)?
            {
                if let Some(profile) = profile.as_deref_mut() {
                    profile.time_mixer.native_v_mix_calls += 1;
                    profile.time_mixer.native_v_mix_rows += v_mix_values.batch_size;
                }
                profile_add!(profile, time_mixer.v_mix_ns, start.elapsed_ns());

                let lora_decay_start = ProfileTimer::start(profile.is_some());
                if let Some(lora_values) = time_mixer_lora_decay_scratch_values_native_with_kernel(
                    weights,
                    &a_b1c,
                    &g_b1c,
                    &d_b1c,
                    dot_kernel,
                    &mut profile,
                )? {
                    if let Some(profile) = profile.as_deref_mut() {
                        profile.time_mixer.native_lora_decay_scratch_calls += 1;
                        profile.time_mixer.native_lora_decay_scratch_rows += lora_values.batch_size;
                    }
                    profile_add!(
                        profile,
                        time_mixer.lora_decay_ns,
                        lora_decay_start.elapsed_ns()
                    );

                    let start = ProfileTimer::start(profile.is_some());
                    let middle_scratch = if let Some(profile) = profile.as_deref_mut() {
                        time_mixer_middle_output_scratch_native_from_lora_and_v_values(
                            &in_b1c,
                            &r_b1c,
                            &k_b1c,
                            &v_mix_values,
                            &lora_values,
                            &k_scale_b1h,
                            &v_scale_b1h,
                            state_b1hkk,
                            weights,
                            heads,
                            head_size,
                            return_state,
                            false,
                            Some(&mut profile.time_mixer.single_timestep),
                        )?
                    } else {
                        time_mixer_middle_output_scratch_native_from_lora_and_v_values(
                            &in_b1c,
                            &r_b1c,
                            &k_b1c,
                            &v_mix_values,
                            &lora_values,
                            &k_scale_b1h,
                            &v_scale_b1h,
                            state_b1hkk,
                            weights,
                            heads,
                            head_size,
                            return_state,
                            false,
                            None,
                        )?
                    };
                    profile_add!(profile, time_mixer.recurrence_ns, start.elapsed_ns());

                    if let Some(mut middle_scratch) = middle_scratch {
                        let start = ProfileTimer::start(profile.is_some());
                        if let Some(out_bc) =
                            time_mixer_group_output_scratch_native_from_lora_values(
                                &in_b1c,
                                &r_b1c,
                                &weights.bonus,
                                &lora_values,
                                weights,
                                &middle_scratch,
                            )?
                        {
                            if let Some(profile) = profile.as_deref_mut() {
                                profile.time_mixer.native_output_calls += 1;
                                profile.time_mixer.native_output_rows += batch_size;
                            }
                            let next_state_b1hkk = middle_scratch
                                .middle
                                .next_state_bhkk
                                .take()
                                .map(|next_state_bhkk| next_state_bhkk.unsqueeze(1))
                                .transpose()?;
                            recycle_time_mixer_middle_scratch_output(middle_scratch.middle);
                            profile_add!(profile, time_mixer.output_ns, start.elapsed_ns());
                            profile_add!(profile, time_mixer.total_ns, total_start.elapsed_ns());

                            return Ok((
                                out_bc,
                                v0_bc.clone(),
                                return_state.then_some(x_b1c),
                                next_state_b1hkk,
                            ));
                        }
                        recycle_time_mixer_middle_scratch_output(middle_scratch.middle);
                    }
                }
            }
        }
    }

    let start = ProfileTimer::start(profile.is_some());
    let (mixed_v_b1c, next_v0_bc) = time_mixer_v_mix(weights, &v_b1c, v0_bc, &mut profile)?;
    v_b1c = mixed_v_b1c;
    profile_add!(profile, time_mixer.v_mix_ns, start.elapsed_ns());

    if native_time_mixer_post_lora_scratch_enabled()
        && native_time_mixer_lora_decay_scratch_enabled()
        && native_time_mixer_middle_output_scratch_enabled()
        && native_time_mixer_output_enabled()
        && native_time_mixer_group_output_scratch_enabled()
    {
        if let Some(dot_kernel) = native_linear_dot_kernel() {
            let lora_decay_start = ProfileTimer::start(profile.is_some());
            if let Some(lora_values) = time_mixer_lora_decay_scratch_values_native_with_kernel(
                weights,
                &a_b1c,
                &g_b1c,
                &d_b1c,
                dot_kernel,
                &mut profile,
            )? {
                if let Some(profile) = profile.as_deref_mut() {
                    profile.time_mixer.native_lora_decay_scratch_calls += 1;
                    profile.time_mixer.native_lora_decay_scratch_rows += lora_values.batch_size;
                }
                profile_add!(
                    profile,
                    time_mixer.lora_decay_ns,
                    lora_decay_start.elapsed_ns()
                );

                let start = ProfileTimer::start(profile.is_some());
                let middle_scratch = if let Some(profile) = profile.as_deref_mut() {
                    time_mixer_middle_output_scratch_native_from_lora_values(
                        &in_b1c,
                        &r_b1c,
                        &k_b1c,
                        &v_b1c,
                        &lora_values,
                        &k_scale_b1h,
                        &v_scale_b1h,
                        state_b1hkk,
                        weights,
                        heads,
                        head_size,
                        return_state,
                        false,
                        Some(&mut profile.time_mixer.single_timestep),
                    )?
                } else {
                    time_mixer_middle_output_scratch_native_from_lora_values(
                        &in_b1c,
                        &r_b1c,
                        &k_b1c,
                        &v_b1c,
                        &lora_values,
                        &k_scale_b1h,
                        &v_scale_b1h,
                        state_b1hkk,
                        weights,
                        heads,
                        head_size,
                        return_state,
                        false,
                        None,
                    )?
                };
                profile_add!(profile, time_mixer.recurrence_ns, start.elapsed_ns());

                if let Some(mut middle_scratch) = middle_scratch {
                    let start = ProfileTimer::start(profile.is_some());
                    if let Some(out_bc) = time_mixer_group_output_scratch_native_from_lora_values(
                        &in_b1c,
                        &r_b1c,
                        &weights.bonus,
                        &lora_values,
                        weights,
                        &middle_scratch,
                    )? {
                        if let Some(profile) = profile.as_deref_mut() {
                            profile.time_mixer.native_output_calls += 1;
                            profile.time_mixer.native_output_rows += batch_size;
                        }
                        let next_state_b1hkk = middle_scratch
                            .middle
                            .next_state_bhkk
                            .take()
                            .map(|next_state_bhkk| next_state_bhkk.unsqueeze(1))
                            .transpose()?;
                        recycle_time_mixer_middle_scratch_output(middle_scratch.middle);
                        profile_add!(profile, time_mixer.output_ns, start.elapsed_ns());
                        profile_add!(profile, time_mixer.total_ns, total_start.elapsed_ns());

                        return Ok((
                            out_bc,
                            next_v0_bc,
                            return_state.then_some(x_b1c),
                            next_state_b1hkk,
                        ));
                    }
                    recycle_time_mixer_middle_scratch_output(middle_scratch.middle);
                }
            }
        }
    }

    let lora_decay_start = ProfileTimer::start(profile.is_some());
    let (a_b1c, g_b1c, w_b1c) =
        time_mixer_lora_decay(weights, &a_b1c, &g_b1c, &d_b1c, &mut profile)?;
    profile_add!(
        profile,
        time_mixer.lora_decay_ns,
        lora_decay_start.elapsed_ns()
    );

    if native_time_mixer_middle_output_scratch_enabled() && native_time_mixer_output_enabled() {
        let start = ProfileTimer::start(profile.is_some());
        let middle_scratch = if let Some(profile) = profile.as_deref_mut() {
            time_mixer_middle_output_scratch_native(
                &in_b1c,
                &r_b1c,
                &k_b1c,
                &v_b1c,
                &w_b1c,
                &a_b1c,
                &k_scale_b1h,
                &v_scale_b1h,
                state_b1hkk,
                weights,
                heads,
                head_size,
                return_state,
                !native_time_mixer_group_output_scratch_enabled(),
                Some(&mut profile.time_mixer.single_timestep),
            )?
        } else {
            time_mixer_middle_output_scratch_native(
                &in_b1c,
                &r_b1c,
                &k_b1c,
                &v_b1c,
                &w_b1c,
                &a_b1c,
                &k_scale_b1h,
                &v_scale_b1h,
                state_b1hkk,
                weights,
                heads,
                head_size,
                return_state,
                !native_time_mixer_group_output_scratch_enabled(),
                None,
            )?
        };
        profile_add!(profile, time_mixer.recurrence_ns, start.elapsed_ns());

        if let Some(mut middle_scratch) = middle_scratch {
            if native_time_mixer_group_output_scratch_enabled() {
                let start = ProfileTimer::start(profile.is_some());
                if let Some(out_bc) = time_mixer_group_output_scratch_native(
                    &in_b1c,
                    &r_b1c,
                    &weights.bonus,
                    &g_b1c,
                    weights,
                    &middle_scratch,
                )? {
                    if let Some(profile) = profile.as_deref_mut() {
                        profile.time_mixer.native_output_calls += 1;
                        profile.time_mixer.native_output_rows += batch_size;
                    }
                    let next_state_b1hkk = middle_scratch
                        .middle
                        .next_state_bhkk
                        .take()
                        .map(|next_state_bhkk| next_state_bhkk.unsqueeze(1))
                        .transpose()?;
                    recycle_time_mixer_middle_scratch_output(middle_scratch.middle);
                    profile_add!(profile, time_mixer.output_ns, start.elapsed_ns());
                    profile_add!(profile, time_mixer.total_ns, total_start.elapsed_ns());

                    return Ok((
                        out_bc,
                        next_v0_bc,
                        return_state.then_some(x_b1c),
                        next_state_b1hkk,
                    ));
                }
            }

            if !native_time_mixer_group_output_scratch_enabled() {
                let start = ProfileTimer::start(profile.is_some());
                let out_b1hk = middle_scratch
                    .middle
                    .out_bhk
                    .as_ref()
                    .expect(
                        "middle/output scratch materializes out_bhk without group/output scratch",
                    )
                    .unsqueeze(1)?;
                record_reshape_layout(&mut profile, &out_b1hk);
                let out_group_bc = group_norm_2d(
                    &out_b1hk.reshape((batch_size, channels))?,
                    weights.n_heads,
                    &weights.out_group_norm.weight,
                    &weights.out_group_norm.bias,
                    TIME_MIXER_GROUP_NORM_EPS,
                )?;
                record_reshape_layout(&mut profile, &out_group_bc);
                let out_b1c = out_group_bc.reshape((batch_size, 1usize, channels))?;
                profile_add!(profile, time_mixer.group_norm_ns, start.elapsed_ns());

                let start = ProfileTimer::start(profile.is_some());
                let out_bc = time_mixer_output_native_from_middle_scratch(
                    &in_b1c,
                    &r_b1c,
                    &weights.bonus,
                    &out_b1c,
                    &g_b1c,
                    weights,
                    &middle_scratch,
                )?
                .expect("middle/output scratch validates native output before recurrence");
                if let Some(profile) = profile.as_deref_mut() {
                    profile.time_mixer.native_output_calls += 1;
                    profile.time_mixer.native_output_rows += batch_size;
                }
                let next_state_b1hkk = middle_scratch
                    .middle
                    .next_state_bhkk
                    .take()
                    .map(|next_state_bhkk| next_state_bhkk.unsqueeze(1))
                    .transpose()?;
                recycle_time_mixer_middle_scratch_output(middle_scratch.middle);
                profile_add!(profile, time_mixer.output_ns, start.elapsed_ns());
                profile_add!(profile, time_mixer.total_ns, total_start.elapsed_ns());

                return Ok((
                    out_bc,
                    next_v0_bc,
                    return_state.then_some(x_b1c),
                    next_state_b1hkk,
                ));
            }
        }
    }

    let start = ProfileTimer::start(profile.is_some());
    record_reshape_layout(&mut profile, &k_b1c);
    let mut k_b1hk = l2_normalize_scaled_b1hk(
        &k_b1c.reshape((batch_size, 1usize, heads, head_size))?,
        &k_scale_b1h,
    )?;
    record_reshape_layout(&mut profile, &r_b1c);
    let r_b1hk = r_b1c.reshape((batch_size, 1usize, heads, head_size))?;
    record_reshape_layout(&mut profile, &v_b1c);
    let v_b1hk = l2_normalize_scaled_b1hk(
        &v_b1c.reshape((batch_size, 1usize, heads, head_size))?,
        &v_scale_b1h,
    )?;
    record_reshape_layout(&mut profile, &w_b1c);
    let w_b1hk = w_b1c.reshape((batch_size, 1usize, heads, head_size))?;
    record_reshape_layout(&mut profile, &a_b1c);
    let a_b1hk = a_b1c.reshape((batch_size, 1usize, heads, head_size))?;
    let k_deformed_b1hk = k_b1hk.clone();
    k_b1hk = k_b1hk.broadcast_mul(&a_b1hk)?;
    profile_add!(profile, time_mixer.reshape_norm_ns, start.elapsed_ns());

    let (out_bhk, next_state_bhkk) = if native_time_mixer_reshape_recurrence_scratch_enabled() {
        let start = ProfileTimer::start(profile.is_some());
        let (out_bhk, next_state_bhkk) = if return_state {
            let (out_bhk, next_state_bhkk) = if let Some(profile) = profile.as_deref_mut() {
                single_timestep_b1c_profiled(
                    &r_b1c,
                    &k_b1c,
                    &v_b1c,
                    &w_b1c,
                    &a_b1c,
                    &k_scale_b1h,
                    &v_scale_b1h,
                    state_b1hkk,
                    heads,
                    head_size,
                    Some(&mut profile.time_mixer.single_timestep),
                )?
            } else {
                single_timestep_b1c_profiled(
                    &r_b1c,
                    &k_b1c,
                    &v_b1c,
                    &w_b1c,
                    &a_b1c,
                    &k_scale_b1h,
                    &v_scale_b1h,
                    state_b1hkk,
                    heads,
                    head_size,
                    None,
                )?
            };
            (out_bhk, Some(next_state_bhkk))
        } else {
            let out_bhk = if let Some(profile) = profile.as_deref_mut() {
                single_timestep_output_b1c_profiled(
                    &r_b1c,
                    &k_b1c,
                    &v_b1c,
                    &w_b1c,
                    &a_b1c,
                    &k_scale_b1h,
                    &v_scale_b1h,
                    state_b1hkk,
                    heads,
                    head_size,
                    Some(&mut profile.time_mixer.single_timestep),
                )?
            } else {
                single_timestep_output_b1c_profiled(
                    &r_b1c,
                    &k_b1c,
                    &v_b1c,
                    &w_b1c,
                    &a_b1c,
                    &k_scale_b1h,
                    &v_scale_b1h,
                    state_b1hkk,
                    heads,
                    head_size,
                    None,
                )?
            };
            (out_bhk, None)
        };
        profile_add!(profile, time_mixer.recurrence_ns, start.elapsed_ns());
        (out_bhk, next_state_bhkk)
    } else {
        let start = ProfileTimer::start(profile.is_some());
        let r_bhk = r_b1hk.squeeze(1)?;
        let k_bhk = k_b1hk.squeeze(1)?;
        let v_bhk = v_b1hk.squeeze(1)?;
        let w_bhk = w_b1hk.squeeze(1)?;
        let a_bhk = a_b1hk.squeeze(1)?;
        let k_deformed_bhk = k_deformed_b1hk.squeeze(1)?;
        let state_bhkk = state_b1hkk.squeeze(1)?;
        profile_add!(
            profile,
            time_mixer.recurrence_squeeze_ns,
            start.elapsed_ns()
        );

        let start = ProfileTimer::start(profile.is_some());
        let (out_bhk, next_state_bhkk) = if return_state {
            let (out_bhk, next_state_bhkk) = if let Some(profile) = profile.as_deref_mut() {
                single_timestep_profiled(
                    &r_bhk,
                    &k_bhk,
                    &v_bhk,
                    &w_bhk,
                    &a_bhk,
                    &k_deformed_bhk,
                    &state_bhkk,
                    Some(&mut profile.time_mixer.single_timestep),
                )?
            } else {
                single_timestep_profiled(
                    &r_bhk,
                    &k_bhk,
                    &v_bhk,
                    &w_bhk,
                    &a_bhk,
                    &k_deformed_bhk,
                    &state_bhkk,
                    None,
                )?
            };
            (out_bhk, Some(next_state_bhkk))
        } else {
            let out_bhk = if let Some(profile) = profile.as_deref_mut() {
                single_timestep_output_profiled(
                    &r_bhk,
                    &k_bhk,
                    &v_bhk,
                    &w_bhk,
                    &a_bhk,
                    &k_deformed_bhk,
                    &state_bhkk,
                    Some(&mut profile.time_mixer.single_timestep),
                )?
            } else {
                single_timestep_output_profiled(
                    &r_bhk,
                    &k_bhk,
                    &v_bhk,
                    &w_bhk,
                    &a_bhk,
                    &k_deformed_bhk,
                    &state_bhkk,
                    None,
                )?
            };
            (out_bhk, None)
        };
        profile_add!(profile, time_mixer.recurrence_ns, start.elapsed_ns());
        (out_bhk, next_state_bhkk)
    };

    let start = ProfileTimer::start(profile.is_some());
    let out_b1hk = out_bhk.unsqueeze(1)?;
    record_reshape_layout(&mut profile, &out_b1hk);
    let out_group_bc = group_norm_2d(
        &out_b1hk.reshape((batch_size, channels))?,
        weights.n_heads,
        &weights.out_group_norm.weight,
        &weights.out_group_norm.bias,
        TIME_MIXER_GROUP_NORM_EPS,
    )?;
    record_reshape_layout(&mut profile, &out_group_bc);
    let out_b1c = out_group_bc.reshape((batch_size, 1usize, channels))?;
    profile_add!(profile, time_mixer.group_norm_ns, start.elapsed_ns());

    let start = ProfileTimer::start(profile.is_some());
    let out_bc = if native_time_mixer_output_enabled() {
        if let Some(out_bc) = time_mixer_output_native(
            &in_b1c,
            &r_b1hk,
            &k_b1hk,
            &v_b1hk,
            &weights.bonus,
            &out_b1c,
            &g_b1c,
            weights,
        )? {
            if let Some(profile) = profile.as_deref_mut() {
                profile.time_mixer.native_output_calls += 1;
                profile.time_mixer.native_output_rows += batch_size;
            }
            out_bc
        } else {
            time_mixer_output_candle(
                &in_b1c,
                &r_b1hk,
                &k_b1hk,
                &v_b1hk,
                &weights.bonus,
                &out_b1c,
                &g_b1c,
                &weights.w_o,
                &mut profile,
            )?
        }
    } else {
        time_mixer_output_candle(
            &in_b1c,
            &r_b1hk,
            &k_b1hk,
            &v_b1hk,
            &weights.bonus,
            &out_b1c,
            &g_b1c,
            &weights.w_o,
            &mut profile,
        )?
    };
    let next_state_b1hkk = next_state_bhkk
        .map(|next_state_bhkk| next_state_bhkk.unsqueeze(1))
        .transpose()?;
    profile_add!(profile, time_mixer.output_ns, start.elapsed_ns());
    profile_add!(profile, time_mixer.total_ns, total_start.elapsed_ns());

    Ok((
        out_bc,
        next_v0_bc,
        return_state.then_some(x_b1c),
        next_state_b1hkk,
    ))
}

pub(super) fn time_mixer_forward_flat_state_profiled_options(
    weights: &Rwkv7RnnTimeMixerWeights,
    in_bc: &Tensor,
    v0_bc: &Tensor,
    state: Option<&TimeMixerFlatLayerState>,
    return_state: bool,
    mut profile: Option<&mut ForwardProfile>,
) -> Result<Option<(Tensor, Tensor, Option<TimeMixerFlatLayerState>)>> {
    if !(native_time_mixer_lerp_projection_values_scratch_enabled()
        && native_time_mixer_lerp_projection_scratch_enabled()
        && native_time_mixer_v_mix_values_scratch_enabled()
        && native_time_mixer_v_mix_enabled()
        && native_time_mixer_post_lora_scratch_enabled()
        && native_time_mixer_lora_decay_scratch_enabled()
        && native_time_mixer_middle_output_scratch_enabled()
        && native_time_mixer_output_enabled()
        && native_time_mixer_group_output_scratch_enabled()
        && native_time_mixer_middle_all_values_scratch_enabled()
        && native_time_mixer_group_output_cached_aux_enabled()
        && (weights.layer_id != 0 || native_time_mixer_layer0_v_values_scratch_enabled()))
    {
        return Ok(None);
    }
    let Some(dot_kernel) = native_linear_dot_kernel() else {
        return Ok(None);
    };

    if native_time_mixer_flat_input_values_enabled() {
        return time_mixer_forward_flat_state_values_profiled_options(
            weights,
            in_bc,
            v0_bc,
            state,
            return_state,
            dot_kernel,
            profile,
        );
    }

    let total_start = ProfileTimer::start(profile.is_some());
    profile_inc!(profile, time_mixer.calls);

    let start = ProfileTimer::start(profile.is_some());
    let (batch_size, channels) = in_bc.dims2()?;
    if !trusted_hot_loop_shapes_enabled() {
        validate_time_mixer_shapes(weights, channels)?;
    }
    if v0_bc.dims() != [batch_size, channels] {
        bail!(
            "time_mixer expected v0_BC shape [{batch_size}, {channels}], got {:?}",
            v0_bc.dims()
        );
    }
    let heads = weights.n_heads;
    let head_size = weights.head_size;
    if let Some(state) = state {
        if state.channels != channels
            || state.heads != heads
            || state.head_size != head_size
            || state.x_shift_values.len() != batch_size * channels
            || state.state_values.len() != batch_size * heads * head_size * head_size
        {
            return Ok(None);
        }
    }
    profile_add!(profile, time_mixer.validate_ns, start.elapsed_ns());

    let start = ProfileTimer::start(profile.is_some());
    let in_b1c = in_bc.unsqueeze(1)?;
    let x_b1c = layer_norm(&in_b1c, &weights.layer_norm)?;
    let x_data = f32_tensor_data(&x_b1c)?;
    let x_values = x_data.as_slice()?;
    if x_values.len() != batch_size * channels {
        return Ok(None);
    }
    profile_add!(profile, time_mixer.layer_norm_ns, start.elapsed_ns());

    let zero_state_recurrence =
        state.is_none() && native_time_mixer_zero_state_recurrence_enabled();
    let zero_state_values;
    let x_shift_values = if let Some(state) = state {
        state.x_shift_values.as_slice()
    } else {
        x_values
    };
    let state_values = if let Some(state) = state {
        state.state_values.as_slice()
    } else if zero_state_recurrence {
        &[] as &[f32]
    } else {
        zero_state_values = vec![0.0f32; batch_size * heads * head_size * head_size];
        zero_state_values.as_slice()
    };

    let projection_start = ProfileTimer::start(profile.is_some());
    let Some(projection_values) =
        time_mixer_lerp_projection_scratch_values_native_from_values_with_kernel(
            weights,
            x_values,
            x_shift_values,
            batch_size,
            channels,
            dot_kernel,
        )?
    else {
        return Ok(None);
    };
    if projection_values.batch_size != batch_size
        || projection_values.channels != channels
        || projection_values.heads != heads
    {
        recycle_time_mixer_lerp_projection_scratch_values(projection_values);
        return Ok(None);
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.time_mixer.projection_ns += projection_start.elapsed_ns();
        profile.time_mixer.native_projection_group_calls += 1;
        profile.time_mixer.native_projection_group_rows += batch_size;
    }

    let start = ProfileTimer::start(profile.is_some());
    let Some(v_mix_values) = (if weights.layer_id == 0 {
        time_mixer_layer0_v_values_native_from_input_values_with_kernel(
            &projection_values.v_values,
            batch_size,
            channels,
            weights,
            dot_kernel,
        )?
    } else {
        time_mixer_v_mix_values_native_from_input_values_with_kernel(
            &projection_values.v_values,
            batch_size,
            channels,
            v0_bc,
            weights,
            dot_kernel,
        )?
    }) else {
        recycle_time_mixer_lerp_projection_scratch_values(projection_values);
        return Ok(None);
    };
    if let Some(profile) = profile.as_deref_mut() {
        profile.time_mixer.native_v_mix_calls += 1;
        profile.time_mixer.native_v_mix_rows += v_mix_values.batch_size;
    }
    profile_add!(profile, time_mixer.v_mix_ns, start.elapsed_ns());

    let lora_decay_start = ProfileTimer::start(profile.is_some());
    let Some(lora_values) =
        time_mixer_lora_decay_scratch_values_native_from_input_values_with_kernel(
            weights,
            &projection_values.a_values,
            &projection_values.g_values,
            &projection_values.d_values,
            batch_size,
            channels,
            dot_kernel,
            &mut profile,
        )?
    else {
        recycle_time_mixer_lerp_projection_scratch_values(projection_values);
        return Ok(None);
    };
    if let Some(profile) = profile.as_deref_mut() {
        profile.time_mixer.native_lora_decay_scratch_calls += 1;
        profile.time_mixer.native_lora_decay_scratch_rows += lora_values.batch_size;
    }
    profile_add!(
        profile,
        time_mixer.lora_decay_ns,
        lora_decay_start.elapsed_ns()
    );

    let start = ProfileTimer::start(profile.is_some());
    let middle = if let Some(profile) = profile.as_deref_mut() {
        time_mixer_middle_scratch_all_values_flat_state_profiled(
            &projection_values.r_values,
            &projection_values.k_values,
            &v_mix_values.values,
            &lora_values.w_values,
            &lora_values.a_values,
            &projection_values.k_scale_values,
            &projection_values.v_scale_values,
            state_values,
            zero_state_recurrence,
            batch_size,
            heads,
            head_size,
            return_state,
            false,
            in_bc.device(),
            Some(&mut profile.time_mixer.single_timestep),
        )?
    } else {
        time_mixer_middle_scratch_all_values_flat_state_profiled(
            &projection_values.r_values,
            &projection_values.k_values,
            &v_mix_values.values,
            &lora_values.w_values,
            &lora_values.a_values,
            &projection_values.k_scale_values,
            &projection_values.v_scale_values,
            state_values,
            zero_state_recurrence,
            batch_size,
            heads,
            head_size,
            return_state,
            false,
            in_bc.device(),
            None,
        )?
    };
    profile_add!(profile, time_mixer.recurrence_ns, start.elapsed_ns());

    let mut middle_scratch = TimeMixerMiddleOutputScratch { middle, dot_kernel };
    let start = ProfileTimer::start(profile.is_some());
    let output = time_mixer_group_output_scratch_native_from_cached_aux_values(
        in_bc,
        &projection_values.r_values,
        &lora_values,
        weights,
        &middle_scratch,
    )?;
    let Some(out_bc) = output else {
        recycle_time_mixer_middle_scratch_output(middle_scratch.middle);
        recycle_time_mixer_lerp_projection_scratch_values(projection_values);
        return Ok(None);
    };
    if let Some(profile) = profile.as_deref_mut() {
        profile.time_mixer.native_output_calls += 1;
        profile.time_mixer.native_output_rows += batch_size;
    }
    let next_state_values = middle_scratch.middle.next_state_values.take();
    recycle_time_mixer_middle_scratch_output(middle_scratch.middle);
    recycle_time_mixer_lerp_projection_scratch_values(projection_values);
    profile_add!(profile, time_mixer.output_ns, start.elapsed_ns());
    profile_add!(profile, time_mixer.total_ns, total_start.elapsed_ns());

    let next_v0_bc = if weights.layer_id == 0 {
        Tensor::from_vec(v_mix_values.values, (batch_size, channels), in_bc.device())?
    } else {
        v0_bc.clone()
    };
    let next_flat_state = if return_state {
        Some(TimeMixerFlatLayerState {
            x_shift_values: x_values.to_vec(),
            state_values: next_state_values
                .expect("flat middle scratch returns next state values when return_state=true"),
            channels,
            heads,
            head_size,
        })
    } else {
        None
    };

    Ok(Some((out_bc, next_v0_bc, next_flat_state)))
}

pub(super) fn time_mixer_forward_flat_state_values_profiled_options(
    weights: &Rwkv7RnnTimeMixerWeights,
    in_bc: &Tensor,
    v0_bc: &Tensor,
    state: Option<&TimeMixerFlatLayerState>,
    return_state: bool,
    dot_kernel: NativeLinearDotKernel,
    profile: Option<&mut ForwardProfile>,
) -> Result<Option<(Tensor, Tensor, Option<TimeMixerFlatLayerState>)>> {
    let Some((output_values, next_v0_bc, next_flat_state)) =
        time_mixer_forward_flat_state_values_raw_output_profiled_options(
            weights,
            in_bc,
            v0_bc,
            state,
            return_state,
            dot_kernel,
            profile,
        )?
    else {
        return Ok(None);
    };
    let (batch_size, channels) = in_bc.dims2()?;
    let out_bc = Tensor::from_vec(output_values, (batch_size, channels), in_bc.device())?;
    Ok(Some((out_bc, next_v0_bc, next_flat_state)))
}

pub(super) fn time_mixer_forward_flat_state_values_raw_output_profiled_options(
    weights: &Rwkv7RnnTimeMixerWeights,
    in_bc: &Tensor,
    v0_bc: &Tensor,
    state: Option<&TimeMixerFlatLayerState>,
    return_state: bool,
    dot_kernel: NativeLinearDotKernel,
    mut profile: Option<&mut ForwardProfile>,
) -> Result<Option<(Vec<f32>, Tensor, Option<TimeMixerFlatLayerState>)>> {
    let total_start = ProfileTimer::start(profile.is_some());
    profile_inc!(profile, time_mixer.calls);

    let start = ProfileTimer::start(profile.is_some());
    let (batch_size, channels) = in_bc.dims2()?;
    if !trusted_hot_loop_shapes_enabled() {
        validate_time_mixer_shapes(weights, channels)?;
    }
    if v0_bc.dims() != [batch_size, channels] {
        bail!(
            "time_mixer expected v0_BC shape [{batch_size}, {channels}], got {:?}",
            v0_bc.dims()
        );
    }
    let heads = weights.n_heads;
    let head_size = weights.head_size;
    if let Some(state) = state {
        if state.channels != channels
            || state.heads != heads
            || state.head_size != head_size
            || state.x_shift_values.len() != batch_size * channels
            || state.state_values.len() != batch_size * heads * head_size * head_size
        {
            return Ok(None);
        }
    }
    let in_data = f32_tensor_data(in_bc)?;
    let v0_data = f32_tensor_data(v0_bc)?;
    let in_values = in_data.as_slice()?;
    let v0_values = v0_data.as_slice()?;
    let bxc_len = batch_size * channels;
    if in_values.len() != bxc_len || v0_values.len() != bxc_len {
        return Ok(None);
    }
    profile_add!(profile, time_mixer.validate_ns, start.elapsed_ns());

    let start = ProfileTimer::start(profile.is_some());
    let reuse_predict_x_values =
        !return_state && native_time_mixer_predict_x_buffer_reuse_enabled();
    let Some(x_values) = (if reuse_predict_x_values {
        layer_norm_values_native_centered_predict_x_buffer(
            in_values,
            batch_size,
            channels,
            &weights.layer_norm,
        )?
    } else {
        layer_norm_values_native_centered(in_values, batch_size, channels, &weights.layer_norm)?
    }) else {
        return Ok(None);
    };
    profile_add!(profile, time_mixer.layer_norm_ns, start.elapsed_ns());

    let zero_state_recurrence =
        state.is_none() && native_time_mixer_zero_state_recurrence_enabled();
    let zero_state_values;
    let x_shift_values = if let Some(state) = state {
        state.x_shift_values.as_slice()
    } else {
        x_values.as_slice()
    };
    let state_values = if let Some(state) = state {
        state.state_values.as_slice()
    } else if zero_state_recurrence {
        &[] as &[f32]
    } else {
        zero_state_values = vec![0.0f32; batch_size * heads * head_size * head_size];
        zero_state_values.as_slice()
    };

    let projection_start = ProfileTimer::start(profile.is_some());
    let Some(projection_values) =
        time_mixer_lerp_projection_scratch_values_native_from_values_with_kernel(
            weights,
            &x_values,
            x_shift_values,
            batch_size,
            channels,
            dot_kernel,
        )?
    else {
        if reuse_predict_x_values {
            recycle_time_mixer_predict_x_values(x_values);
        }
        return Ok(None);
    };
    if projection_values.batch_size != batch_size
        || projection_values.channels != channels
        || projection_values.heads != heads
    {
        recycle_time_mixer_lerp_projection_scratch_values(projection_values);
        if reuse_predict_x_values {
            recycle_time_mixer_predict_x_values(x_values);
        }
        return Ok(None);
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.time_mixer.projection_ns += projection_start.elapsed_ns();
        profile.time_mixer.native_projection_group_calls += 1;
        profile.time_mixer.native_projection_group_rows += batch_size;
    }

    let start = ProfileTimer::start(profile.is_some());
    let Some(v_mix_values) = (if weights.layer_id == 0 {
        time_mixer_layer0_v_values_native_from_input_values_with_kernel(
            &projection_values.v_values,
            batch_size,
            channels,
            weights,
            dot_kernel,
        )?
    } else {
        time_mixer_v_mix_values_native_from_flat_v0_with_kernel(
            &projection_values.v_values,
            v0_values,
            batch_size,
            channels,
            weights,
            dot_kernel,
        )?
    }) else {
        recycle_time_mixer_lerp_projection_scratch_values(projection_values);
        if reuse_predict_x_values {
            recycle_time_mixer_predict_x_values(x_values);
        }
        return Ok(None);
    };
    if let Some(profile) = profile.as_deref_mut() {
        profile.time_mixer.native_v_mix_calls += 1;
        profile.time_mixer.native_v_mix_rows += v_mix_values.batch_size;
    }
    profile_add!(profile, time_mixer.v_mix_ns, start.elapsed_ns());

    let lora_decay_start = ProfileTimer::start(profile.is_some());
    let Some(lora_values) =
        time_mixer_lora_decay_scratch_values_native_from_input_values_with_kernel(
            weights,
            &projection_values.a_values,
            &projection_values.g_values,
            &projection_values.d_values,
            batch_size,
            channels,
            dot_kernel,
            &mut profile,
        )?
    else {
        recycle_time_mixer_lerp_projection_scratch_values(projection_values);
        if reuse_predict_x_values {
            recycle_time_mixer_predict_x_values(x_values);
        }
        return Ok(None);
    };
    if let Some(profile) = profile.as_deref_mut() {
        profile.time_mixer.native_lora_decay_scratch_calls += 1;
        profile.time_mixer.native_lora_decay_scratch_rows += lora_values.batch_size;
    }
    profile_add!(
        profile,
        time_mixer.lora_decay_ns,
        lora_decay_start.elapsed_ns()
    );

    let start = ProfileTimer::start(profile.is_some());
    let middle = if let Some(profile) = profile.as_deref_mut() {
        time_mixer_middle_scratch_all_values_flat_state_profiled(
            &projection_values.r_values,
            &projection_values.k_values,
            &v_mix_values.values,
            &lora_values.w_values,
            &lora_values.a_values,
            &projection_values.k_scale_values,
            &projection_values.v_scale_values,
            state_values,
            zero_state_recurrence,
            batch_size,
            heads,
            head_size,
            return_state,
            false,
            in_bc.device(),
            Some(&mut profile.time_mixer.single_timestep),
        )?
    } else {
        time_mixer_middle_scratch_all_values_flat_state_profiled(
            &projection_values.r_values,
            &projection_values.k_values,
            &v_mix_values.values,
            &lora_values.w_values,
            &lora_values.a_values,
            &projection_values.k_scale_values,
            &projection_values.v_scale_values,
            state_values,
            zero_state_recurrence,
            batch_size,
            heads,
            head_size,
            return_state,
            false,
            in_bc.device(),
            None,
        )?
    };
    profile_add!(profile, time_mixer.recurrence_ns, start.elapsed_ns());

    let mut middle_scratch = TimeMixerMiddleOutputScratch { middle, dot_kernel };
    let start = ProfileTimer::start(profile.is_some());
    let output = time_mixer_group_output_scratch_native_from_cached_aux_input_values(
        in_values,
        batch_size,
        channels,
        &projection_values.r_values,
        &lora_values,
        weights,
        &middle_scratch,
    )?;
    let Some(out_bc) = output else {
        recycle_time_mixer_middle_scratch_output(middle_scratch.middle);
        recycle_time_mixer_lerp_projection_scratch_values(projection_values);
        if reuse_predict_x_values {
            recycle_time_mixer_predict_x_values(x_values);
        }
        return Ok(None);
    };
    if let Some(profile) = profile.as_deref_mut() {
        profile.time_mixer.native_output_calls += 1;
        profile.time_mixer.native_output_rows += batch_size;
    }
    let next_state_values = middle_scratch.middle.next_state_values.take();
    recycle_time_mixer_middle_scratch_output(middle_scratch.middle);
    recycle_time_mixer_lerp_projection_scratch_values(projection_values);
    profile_add!(profile, time_mixer.output_ns, start.elapsed_ns());
    profile_add!(profile, time_mixer.total_ns, total_start.elapsed_ns());

    let next_v0_bc = if weights.layer_id == 0 {
        Tensor::from_vec(v_mix_values.values, (batch_size, channels), in_bc.device())?
    } else {
        v0_bc.clone()
    };
    let next_flat_state = if return_state {
        Some(TimeMixerFlatLayerState {
            x_shift_values: x_values,
            state_values: next_state_values
                .expect("flat middle scratch returns next state values when return_state=true"),
            channels,
            heads,
            head_size,
        })
    } else {
        if reuse_predict_x_values {
            recycle_time_mixer_predict_x_values(x_values);
        }
        None
    };

    Ok(Some((out_bc, next_v0_bc, next_flat_state)))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn time_mixer_forward_flat_state_input_values_raw_output_profiled_options(
    weights: &Rwkv7RnnTimeMixerWeights,
    in_values: &[f32],
    batch_size: usize,
    channels: usize,
    device: &Device,
    v0_values: &[f32],
    state: Option<&TimeMixerFlatLayerState>,
    return_state: bool,
    dot_kernel: NativeLinearDotKernel,
    mut profile: Option<&mut ForwardProfile>,
) -> Result<Option<(Vec<f32>, Option<Vec<f32>>, Option<TimeMixerFlatLayerState>)>> {
    let total_start = ProfileTimer::start(profile.is_some());
    profile_inc!(profile, time_mixer.calls);

    let start = ProfileTimer::start(profile.is_some());
    if !trusted_hot_loop_shapes_enabled() {
        validate_time_mixer_shapes(weights, channels)?;
    }
    let heads = weights.n_heads;
    let head_size = weights.head_size;
    let bxc_len = batch_size * channels;
    if in_values.len() != bxc_len || v0_values.len() != bxc_len {
        return Ok(None);
    }
    if let Some(state) = state {
        if state.channels != channels
            || state.heads != heads
            || state.head_size != head_size
            || state.x_shift_values.len() != bxc_len
            || state.state_values.len() != batch_size * heads * head_size * head_size
        {
            return Ok(None);
        }
    }
    profile_add!(profile, time_mixer.validate_ns, start.elapsed_ns());

    let start = ProfileTimer::start(profile.is_some());
    let reuse_predict_x_values =
        !return_state && native_time_mixer_predict_x_buffer_reuse_enabled();
    let Some(x_values) = (if reuse_predict_x_values {
        layer_norm_values_native_centered_predict_x_buffer(
            in_values,
            batch_size,
            channels,
            &weights.layer_norm,
        )?
    } else {
        layer_norm_values_native_centered(in_values, batch_size, channels, &weights.layer_norm)?
    }) else {
        return Ok(None);
    };
    profile_add!(profile, time_mixer.layer_norm_ns, start.elapsed_ns());

    let zero_state_recurrence =
        state.is_none() && native_time_mixer_zero_state_recurrence_enabled();
    let zero_state_values;
    let x_shift_values = if let Some(state) = state {
        state.x_shift_values.as_slice()
    } else {
        x_values.as_slice()
    };
    let state_values = if let Some(state) = state {
        state.state_values.as_slice()
    } else if zero_state_recurrence {
        &[] as &[f32]
    } else {
        zero_state_values = vec![0.0f32; batch_size * heads * head_size * head_size];
        zero_state_values.as_slice()
    };

    let projection_start = ProfileTimer::start(profile.is_some());
    let Some(projection_values) =
        time_mixer_lerp_projection_scratch_values_native_from_values_with_kernel(
            weights,
            &x_values,
            x_shift_values,
            batch_size,
            channels,
            dot_kernel,
        )?
    else {
        if reuse_predict_x_values {
            recycle_time_mixer_predict_x_values(x_values);
        }
        return Ok(None);
    };
    if projection_values.batch_size != batch_size
        || projection_values.channels != channels
        || projection_values.heads != heads
    {
        recycle_time_mixer_lerp_projection_scratch_values(projection_values);
        if reuse_predict_x_values {
            recycle_time_mixer_predict_x_values(x_values);
        }
        return Ok(None);
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.time_mixer.projection_ns += projection_start.elapsed_ns();
        profile.time_mixer.native_projection_group_calls += 1;
        profile.time_mixer.native_projection_group_rows += batch_size;
    }

    let start = ProfileTimer::start(profile.is_some());
    let Some(v_mix_values) = (if weights.layer_id == 0 {
        time_mixer_layer0_v_values_native_from_input_values_with_kernel(
            &projection_values.v_values,
            batch_size,
            channels,
            weights,
            dot_kernel,
        )?
    } else {
        time_mixer_v_mix_values_native_from_flat_v0_with_kernel(
            &projection_values.v_values,
            v0_values,
            batch_size,
            channels,
            weights,
            dot_kernel,
        )?
    }) else {
        recycle_time_mixer_lerp_projection_scratch_values(projection_values);
        if reuse_predict_x_values {
            recycle_time_mixer_predict_x_values(x_values);
        }
        return Ok(None);
    };
    if let Some(profile) = profile.as_deref_mut() {
        profile.time_mixer.native_v_mix_calls += 1;
        profile.time_mixer.native_v_mix_rows += v_mix_values.batch_size;
    }
    profile_add!(profile, time_mixer.v_mix_ns, start.elapsed_ns());

    let lora_decay_start = ProfileTimer::start(profile.is_some());
    let Some(lora_values) =
        time_mixer_lora_decay_scratch_values_native_from_input_values_with_kernel(
            weights,
            &projection_values.a_values,
            &projection_values.g_values,
            &projection_values.d_values,
            batch_size,
            channels,
            dot_kernel,
            &mut profile,
        )?
    else {
        recycle_time_mixer_lerp_projection_scratch_values(projection_values);
        if reuse_predict_x_values {
            recycle_time_mixer_predict_x_values(x_values);
        }
        return Ok(None);
    };
    if let Some(profile) = profile.as_deref_mut() {
        profile.time_mixer.native_lora_decay_scratch_calls += 1;
        profile.time_mixer.native_lora_decay_scratch_rows += lora_values.batch_size;
    }
    profile_add!(
        profile,
        time_mixer.lora_decay_ns,
        lora_decay_start.elapsed_ns()
    );

    let start = ProfileTimer::start(profile.is_some());
    let middle = if let Some(profile) = profile.as_deref_mut() {
        time_mixer_middle_scratch_all_values_flat_state_profiled(
            &projection_values.r_values,
            &projection_values.k_values,
            &v_mix_values.values,
            &lora_values.w_values,
            &lora_values.a_values,
            &projection_values.k_scale_values,
            &projection_values.v_scale_values,
            state_values,
            zero_state_recurrence,
            batch_size,
            heads,
            head_size,
            return_state,
            false,
            device,
            Some(&mut profile.time_mixer.single_timestep),
        )?
    } else {
        time_mixer_middle_scratch_all_values_flat_state_profiled(
            &projection_values.r_values,
            &projection_values.k_values,
            &v_mix_values.values,
            &lora_values.w_values,
            &lora_values.a_values,
            &projection_values.k_scale_values,
            &projection_values.v_scale_values,
            state_values,
            zero_state_recurrence,
            batch_size,
            heads,
            head_size,
            return_state,
            false,
            device,
            None,
        )?
    };
    profile_add!(profile, time_mixer.recurrence_ns, start.elapsed_ns());

    let mut middle_scratch = TimeMixerMiddleOutputScratch { middle, dot_kernel };
    let start = ProfileTimer::start(profile.is_some());
    let output = time_mixer_group_output_scratch_native_from_cached_aux_input_values(
        in_values,
        batch_size,
        channels,
        &projection_values.r_values,
        &lora_values,
        weights,
        &middle_scratch,
    )?;
    let Some(out_values) = output else {
        recycle_time_mixer_middle_scratch_output(middle_scratch.middle);
        recycle_time_mixer_lerp_projection_scratch_values(projection_values);
        if reuse_predict_x_values {
            recycle_time_mixer_predict_x_values(x_values);
        }
        return Ok(None);
    };
    if let Some(profile) = profile.as_deref_mut() {
        profile.time_mixer.native_output_calls += 1;
        profile.time_mixer.native_output_rows += batch_size;
    }
    let next_state_values = middle_scratch.middle.next_state_values.take();
    recycle_time_mixer_middle_scratch_output(middle_scratch.middle);
    recycle_time_mixer_lerp_projection_scratch_values(projection_values);
    profile_add!(profile, time_mixer.output_ns, start.elapsed_ns());
    profile_add!(profile, time_mixer.total_ns, total_start.elapsed_ns());

    let next_v0_values = (weights.layer_id == 0).then_some(v_mix_values.values);
    let next_flat_state = if return_state {
        Some(TimeMixerFlatLayerState {
            x_shift_values: x_values,
            state_values: next_state_values
                .expect("flat middle scratch returns next state values when return_state=true"),
            channels,
            heads,
            head_size,
        })
    } else {
        if reuse_predict_x_values {
            recycle_time_mixer_predict_x_values(x_values);
        }
        None
    };

    Ok(Some((out_values, next_v0_values, next_flat_state)))
}

pub(super) fn time_mixer_native_f32_weights(
    weights: &Rwkv7RnnTimeMixerWeights,
) -> Result<&TimeMixerNativeF32Weights> {
    if weights.native_f32_cache.get().is_none() {
        let rkvdag_lerp_data = f32_tensor_data(&weights.rkvdag_lerp)?;
        let rkvdag_lerp = rkvdag_lerp_data.as_slice()?.to_vec();
        let bonus_data = f32_tensor_data(&weights.bonus)?;
        let bonus = bonus_data.as_slice()?.to_vec();
        let out_group_norm_weight_data = f32_tensor_data(&weights.out_group_norm.weight)?;
        let out_group_norm_weight = out_group_norm_weight_data.as_slice()?.to_vec();
        let out_group_norm_bias_data = f32_tensor_data(&weights.out_group_norm.bias)?;
        let out_group_norm_bias = out_group_norm_bias_data.as_slice()?.to_vec();
        let cache = TimeMixerNativeF32Weights {
            rkvdag_lerp,
            bonus,
            out_group_norm_weight,
            out_group_norm_bias,
            w_r: linear_f32_weights(&weights.w_r)?,
            w_k: linear_f32_weights(&weights.w_k)?,
            w_v: linear_f32_weights_with_blocked(&weights.w_v, true)?,
            w_o: linear_f32_weights_with_blocked(&weights.w_o, true)?,
            k_scale_linear: linear_f32_weights(&weights.k_scale_linear)?,
            v_scale_linear: linear_f32_weights(&weights.v_scale_linear)?,
            v_lora_a: linear_f32_weights(&weights.v_lora_simple.a)?,
            v_lora_b: linear_f32_weights(&weights.v_lora_simple.b_and_lamb)?,
            a_lora_a: linear_f32_weights(&weights.a_lora_simple.a)?,
            a_lora_b: linear_f32_weights(&weights.a_lora_simple.b_and_lamb)?,
            d_lora_a: linear_f32_weights(&weights.d_lora_mlp.a)?,
            d_lora_b: linear_f32_weights(&weights.d_lora_mlp.b_and_lamb)?,
            gate_lora_a: linear_f32_weights(&weights.gate_lora.a)?,
            gate_lora_b: linear_f32_weights(&weights.gate_lora.b)?,
        };
        let _ = weights.native_f32_cache.set(cache);
    }
    Ok(weights
        .native_f32_cache
        .get()
        .expect("time mixer native f32 cache is initialized"))
}

pub(super) fn time_mixer_projection_group(
    weights: &Rwkv7RnnTimeMixerWeights,
    rkvdag_parts: &[Tensor],
    profile: &mut Option<&mut ForwardProfile>,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    if native_time_mixer_projection_enabled() {
        if let Some(output) = time_mixer_projection_group_native(rkvdag_parts, weights)? {
            if let Some(profile) = profile.as_deref_mut() {
                profile.time_mixer.native_projection_group_calls += 1;
                profile.time_mixer.native_projection_group_rows += rkvdag_parts[0].dims3()?.0;
            }
            return Ok(output);
        }
    }

    time_mixer_projection_group_candle(weights, rkvdag_parts, profile)
}

pub(super) fn time_mixer_lerp_projection_scratch_native(
    weights: &Rwkv7RnnTimeMixerWeights,
    x_b1c: &Tensor,
    x_shift_b1c: &Tensor,
) -> Result<Option<TimeMixerLerpProjectionScratchOutput>> {
    let Some(dot_kernel) = native_linear_dot_kernel() else {
        return Ok(None);
    };
    time_mixer_lerp_projection_scratch_native_with_kernel(weights, x_b1c, x_shift_b1c, dot_kernel)
}

pub(super) fn time_mixer_lerp_projection_scratch_native_with_kernel(
    weights: &Rwkv7RnnTimeMixerWeights,
    x_b1c: &Tensor,
    x_shift_b1c: &Tensor,
    dot_kernel: NativeLinearDotKernel,
) -> Result<Option<TimeMixerLerpProjectionScratchOutput>> {
    let (batch_size, one, channels) = x_b1c.dims3()?;
    if one != 1 || x_shift_b1c.dims() != [batch_size, 1, channels] {
        return Ok(None);
    }

    let heads = weights.n_heads;
    let cached_weights = time_mixer_native_f32_weights(weights)?;
    let lerp_values = cached_weights.rkvdag_lerp.as_slice();
    let w_r = &cached_weights.w_r;
    let w_k = &cached_weights.w_k;
    let k_scale = &cached_weights.k_scale_linear;
    let v_scale = &cached_weights.v_scale_linear;
    if lerp_values.len() != 8 * channels
        || w_r.out_dim != channels
        || w_r.in_dim != channels
        || w_k.out_dim != channels
        || w_k.in_dim != channels
        || k_scale.out_dim != heads
        || k_scale.in_dim != channels
        || v_scale.out_dim != heads
        || v_scale.in_dim != channels
        || w_r.bias.is_some()
        || w_k.bias.is_some()
    {
        return Ok(None);
    }
    let Some(k_scale_bias_values) = k_scale.bias.as_deref() else {
        return Ok(None);
    };
    let Some(v_scale_bias_values) = v_scale.bias.as_deref() else {
        return Ok(None);
    };
    let w_r_values = w_r.weight.as_slice();
    let w_k_values = w_k.weight.as_slice();
    let k_scale_weight_values = k_scale.weight.as_slice();
    let v_scale_weight_values = v_scale.weight.as_slice();
    let bxc_len = batch_size * channels;
    if w_r_values.len() != channels * channels
        || w_k_values.len() != channels * channels
        || k_scale_weight_values.len() != heads * channels
        || v_scale_weight_values.len() != heads * channels
        || k_scale_bias_values.len() != heads
        || v_scale_bias_values.len() != heads
    {
        return Ok(None);
    }

    let x_data = f32_tensor_data(x_b1c)?;
    let x_shift_data = f32_tensor_data(x_shift_b1c)?;
    let x_values = x_data.as_slice()?;
    let x_shift_values = x_shift_data.as_slice()?;
    if x_values.len() != bxc_len || x_shift_values.len() != bxc_len {
        return Ok(None);
    }

    let mut r_output_values = vec![0.0f32; bxc_len];
    let mut k_output_values = vec![0.0f32; bxc_len];
    let mut k_scale_values = vec![0.0f32; batch_size * heads];
    let mut v_scale_values = vec![0.0f32; batch_size * heads];
    let mut v_values = vec![0.0f32; bxc_len];
    let mut d_values = vec![0.0f32; bxc_len];
    let mut a_values = vec![0.0f32; bxc_len];
    let mut g_values = vec![0.0f32; bxc_len];

    let mut r_row = vec![0.0f32; channels];
    let mut k_row = vec![0.0f32; channels];
    let mut k_scale_row = vec![0.0f32; channels];
    let mut v_scale_row = vec![0.0f32; channels];
    let use_direct_activation_scalars = use_direct_time_mixer_activation_scalars();
    for batch_index in 0..batch_size {
        let batch_base = batch_index * channels;
        for channel_index in 0..channels {
            let value_index = batch_base + channel_index;
            let x = x_values[value_index];
            let delta = x_shift_values[value_index] - x;
            r_row[channel_index] = x + delta * lerp_values[channel_index];
            k_row[channel_index] = x + delta * lerp_values[channels + channel_index];
            v_values[value_index] = x + delta * lerp_values[2 * channels + channel_index];
            d_values[value_index] = x + delta * lerp_values[3 * channels + channel_index];
            a_values[value_index] = x + delta * lerp_values[4 * channels + channel_index];
            g_values[value_index] = x + delta * lerp_values[5 * channels + channel_index];
            k_scale_row[channel_index] = x + delta * lerp_values[6 * channels + channel_index];
            v_scale_row[channel_index] = x + delta * lerp_values[7 * channels + channel_index];
        }

        let r_output_row = &mut r_output_values[batch_base..batch_base + channels];
        let k_output_row = &mut k_output_values[batch_base..batch_base + channels];
        if native_time_mixer_paired_projection_dots_enabled() {
            for channel_index in 0..channels {
                let r_weight_row =
                    &w_r_values[channel_index * channels..(channel_index + 1) * channels];
                let k_weight_row =
                    &w_k_values[channel_index * channels..(channel_index + 1) * channels];
                let (r_value, k_value) =
                    linear_dot_pair(dot_kernel, &r_row, r_weight_row, &k_row, k_weight_row);
                r_output_row[channel_index] = r_value;
                k_output_row[channel_index] = k_value;
            }
        } else {
            for channel_index in 0..channels {
                let r_weight_row =
                    &w_r_values[channel_index * channels..(channel_index + 1) * channels];
                let k_weight_row =
                    &w_k_values[channel_index * channels..(channel_index + 1) * channels];
                r_output_row[channel_index] = linear_dot(dot_kernel, &r_row, r_weight_row);
                k_output_row[channel_index] = linear_dot(dot_kernel, &k_row, k_weight_row);
            }
        }

        let scale_base = batch_index * heads;
        if native_time_mixer_paired_projection_dots_enabled() {
            for head_index in 0..heads {
                let k_scale_weight_row =
                    &k_scale_weight_values[head_index * channels..(head_index + 1) * channels];
                let v_scale_weight_row =
                    &v_scale_weight_values[head_index * channels..(head_index + 1) * channels];
                let (k_scale_value, v_scale_value) = linear_dot_pair(
                    dot_kernel,
                    &k_scale_row,
                    k_scale_weight_row,
                    &v_scale_row,
                    v_scale_weight_row,
                );
                k_scale_values[scale_base + head_index] = time_mixer_sigmoid_scalar(
                    k_scale_bias_values[head_index] + k_scale_value,
                    use_direct_activation_scalars,
                );
                v_scale_values[scale_base + head_index] = time_mixer_sigmoid_scalar(
                    v_scale_bias_values[head_index] + v_scale_value,
                    use_direct_activation_scalars,
                );
            }
        } else {
            for head_index in 0..heads {
                let k_scale_weight_row =
                    &k_scale_weight_values[head_index * channels..(head_index + 1) * channels];
                let v_scale_weight_row =
                    &v_scale_weight_values[head_index * channels..(head_index + 1) * channels];
                k_scale_values[scale_base + head_index] = time_mixer_sigmoid_scalar(
                    k_scale_bias_values[head_index]
                        + linear_dot(dot_kernel, &k_scale_row, k_scale_weight_row),
                    use_direct_activation_scalars,
                );
                v_scale_values[scale_base + head_index] = time_mixer_sigmoid_scalar(
                    v_scale_bias_values[head_index]
                        + linear_dot(dot_kernel, &v_scale_row, v_scale_weight_row),
                    use_direct_activation_scalars,
                );
            }
        }
    }

    Ok(Some((
        Tensor::from_vec(
            r_output_values,
            (batch_size, 1usize, channels),
            x_b1c.device(),
        )?,
        Tensor::from_vec(
            k_output_values,
            (batch_size, 1usize, channels),
            x_b1c.device(),
        )?,
        Tensor::from_vec(k_scale_values, (batch_size, 1usize, heads), x_b1c.device())?,
        Tensor::from_vec(v_scale_values, (batch_size, 1usize, heads), x_b1c.device())?,
        Tensor::from_vec(v_values, (batch_size, 1usize, channels), x_b1c.device())?,
        Tensor::from_vec(d_values, (batch_size, 1usize, channels), x_b1c.device())?,
        Tensor::from_vec(a_values, (batch_size, 1usize, channels), x_b1c.device())?,
        Tensor::from_vec(g_values, (batch_size, 1usize, channels), x_b1c.device())?,
    )))
}

pub(super) fn time_mixer_lerp_projection_scratch_values_native_with_kernel(
    weights: &Rwkv7RnnTimeMixerWeights,
    x_b1c: &Tensor,
    x_shift_b1c: &Tensor,
    dot_kernel: NativeLinearDotKernel,
) -> Result<Option<TimeMixerLerpProjectionScratchValues>> {
    let (batch_size, one, channels) = x_b1c.dims3()?;
    if one != 1 || x_shift_b1c.dims() != [batch_size, 1, channels] {
        return Ok(None);
    }

    let heads = weights.n_heads;
    let cached_weights = time_mixer_native_f32_weights(weights)?;
    let lerp_values = cached_weights.rkvdag_lerp.as_slice();
    let w_r = &cached_weights.w_r;
    let w_k = &cached_weights.w_k;
    let k_scale = &cached_weights.k_scale_linear;
    let v_scale = &cached_weights.v_scale_linear;
    if lerp_values.len() != 8 * channels
        || w_r.out_dim != channels
        || w_r.in_dim != channels
        || w_k.out_dim != channels
        || w_k.in_dim != channels
        || k_scale.out_dim != heads
        || k_scale.in_dim != channels
        || v_scale.out_dim != heads
        || v_scale.in_dim != channels
        || w_r.bias.is_some()
        || w_k.bias.is_some()
    {
        return Ok(None);
    }
    let Some(k_scale_bias_values) = k_scale.bias.as_deref() else {
        return Ok(None);
    };
    let Some(v_scale_bias_values) = v_scale.bias.as_deref() else {
        return Ok(None);
    };
    let w_r_values = w_r.weight.as_slice();
    let w_k_values = w_k.weight.as_slice();
    let k_scale_weight_values = k_scale.weight.as_slice();
    let v_scale_weight_values = v_scale.weight.as_slice();
    let bxc_len = batch_size * channels;
    if w_r_values.len() != channels * channels
        || w_k_values.len() != channels * channels
        || k_scale_weight_values.len() != heads * channels
        || v_scale_weight_values.len() != heads * channels
        || k_scale_bias_values.len() != heads
        || v_scale_bias_values.len() != heads
    {
        return Ok(None);
    }

    let x_data = f32_tensor_data(x_b1c)?;
    let x_shift_data = f32_tensor_data(x_shift_b1c)?;
    let x_values = x_data.as_slice()?;
    let x_shift_values = x_shift_data.as_slice()?;
    if x_values.len() != bxc_len || x_shift_values.len() != bxc_len {
        return Ok(None);
    }

    let reuse_buffers = native_time_mixer_projection_values_buffer_reuse_enabled();
    let scale_len = batch_size * heads;
    let (
        mut r_output_values,
        mut k_output_values,
        mut k_scale_values,
        mut v_scale_values,
        mut v_values,
        mut d_values,
        mut a_values,
        mut g_values,
        mut r_row,
        mut k_row,
        mut k_scale_row,
        mut v_scale_row,
    ) = if reuse_buffers {
        TIME_MIXER_LERP_PROJECTION_SCRATCH_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            (
                take_time_mixer_projection_vec(&mut pool.r_values, bxc_len),
                take_time_mixer_projection_vec(&mut pool.k_values, bxc_len),
                take_time_mixer_projection_vec(&mut pool.k_scale_values, scale_len),
                take_time_mixer_projection_vec(&mut pool.v_scale_values, scale_len),
                take_time_mixer_projection_vec(&mut pool.v_values, bxc_len),
                take_time_mixer_projection_vec(&mut pool.d_values, bxc_len),
                take_time_mixer_projection_vec(&mut pool.a_values, bxc_len),
                take_time_mixer_projection_vec(&mut pool.g_values, bxc_len),
                take_time_mixer_projection_vec(&mut pool.r_row, channels),
                take_time_mixer_projection_vec(&mut pool.k_row, channels),
                take_time_mixer_projection_vec(&mut pool.k_scale_row, channels),
                take_time_mixer_projection_vec(&mut pool.v_scale_row, channels),
            )
        })
    } else {
        (
            vec![0.0f32; bxc_len],
            vec![0.0f32; bxc_len],
            vec![0.0f32; scale_len],
            vec![0.0f32; scale_len],
            vec![0.0f32; bxc_len],
            vec![0.0f32; bxc_len],
            vec![0.0f32; bxc_len],
            vec![0.0f32; bxc_len],
            vec![0.0f32; channels],
            vec![0.0f32; channels],
            vec![0.0f32; channels],
            vec![0.0f32; channels],
        )
    };
    let use_direct_activation_scalars = use_direct_time_mixer_activation_scalars();
    for batch_index in 0..batch_size {
        let batch_base = batch_index * channels;
        for channel_index in 0..channels {
            let value_index = batch_base + channel_index;
            let x = x_values[value_index];
            let delta = x_shift_values[value_index] - x;
            r_row[channel_index] = x + delta * lerp_values[channel_index];
            k_row[channel_index] = x + delta * lerp_values[channels + channel_index];
            v_values[value_index] = x + delta * lerp_values[2 * channels + channel_index];
            d_values[value_index] = x + delta * lerp_values[3 * channels + channel_index];
            a_values[value_index] = x + delta * lerp_values[4 * channels + channel_index];
            g_values[value_index] = x + delta * lerp_values[5 * channels + channel_index];
            k_scale_row[channel_index] = x + delta * lerp_values[6 * channels + channel_index];
            v_scale_row[channel_index] = x + delta * lerp_values[7 * channels + channel_index];
        }

        let r_output_row = &mut r_output_values[batch_base..batch_base + channels];
        let k_output_row = &mut k_output_values[batch_base..batch_base + channels];
        if native_time_mixer_paired_projection_dots_enabled() {
            for channel_index in 0..channels {
                let r_weight_row =
                    &w_r_values[channel_index * channels..(channel_index + 1) * channels];
                let k_weight_row =
                    &w_k_values[channel_index * channels..(channel_index + 1) * channels];
                let (r_value, k_value) =
                    linear_dot_pair(dot_kernel, &r_row, r_weight_row, &k_row, k_weight_row);
                r_output_row[channel_index] = r_value;
                k_output_row[channel_index] = k_value;
            }
        } else {
            for channel_index in 0..channels {
                let r_weight_row =
                    &w_r_values[channel_index * channels..(channel_index + 1) * channels];
                let k_weight_row =
                    &w_k_values[channel_index * channels..(channel_index + 1) * channels];
                r_output_row[channel_index] = linear_dot(dot_kernel, &r_row, r_weight_row);
                k_output_row[channel_index] = linear_dot(dot_kernel, &k_row, k_weight_row);
            }
        }

        let scale_base = batch_index * heads;
        if native_time_mixer_paired_projection_dots_enabled() {
            for head_index in 0..heads {
                let k_scale_weight_row =
                    &k_scale_weight_values[head_index * channels..(head_index + 1) * channels];
                let v_scale_weight_row =
                    &v_scale_weight_values[head_index * channels..(head_index + 1) * channels];
                let (k_scale_value, v_scale_value) = linear_dot_pair(
                    dot_kernel,
                    &k_scale_row,
                    k_scale_weight_row,
                    &v_scale_row,
                    v_scale_weight_row,
                );
                k_scale_values[scale_base + head_index] = time_mixer_sigmoid_scalar(
                    k_scale_bias_values[head_index] + k_scale_value,
                    use_direct_activation_scalars,
                );
                v_scale_values[scale_base + head_index] = time_mixer_sigmoid_scalar(
                    v_scale_bias_values[head_index] + v_scale_value,
                    use_direct_activation_scalars,
                );
            }
        } else {
            for head_index in 0..heads {
                let k_scale_weight_row =
                    &k_scale_weight_values[head_index * channels..(head_index + 1) * channels];
                let v_scale_weight_row =
                    &v_scale_weight_values[head_index * channels..(head_index + 1) * channels];
                k_scale_values[scale_base + head_index] = time_mixer_sigmoid_scalar(
                    k_scale_bias_values[head_index]
                        + linear_dot(dot_kernel, &k_scale_row, k_scale_weight_row),
                    use_direct_activation_scalars,
                );
                v_scale_values[scale_base + head_index] = time_mixer_sigmoid_scalar(
                    v_scale_bias_values[head_index]
                        + linear_dot(dot_kernel, &v_scale_row, v_scale_weight_row),
                    use_direct_activation_scalars,
                );
            }
        }
    }
    if reuse_buffers {
        recycle_time_mixer_lerp_projection_temp_rows(r_row, k_row, k_scale_row, v_scale_row);
    }

    Ok(Some(TimeMixerLerpProjectionScratchValues {
        batch_size,
        channels,
        heads,
        r_values: r_output_values,
        k_values: k_output_values,
        k_scale_values,
        v_scale_values,
        v_values,
        d_values,
        a_values,
        g_values,
    }))
}

pub(super) fn time_mixer_lerp_projection_scratch_values_native_from_values_with_kernel(
    weights: &Rwkv7RnnTimeMixerWeights,
    x_values: &[f32],
    x_shift_values: &[f32],
    batch_size: usize,
    channels: usize,
    dot_kernel: NativeLinearDotKernel,
) -> Result<Option<TimeMixerLerpProjectionScratchValues>> {
    if let Some(values) =
        time_mixer_batched_lerp_projection_scratch_values_native_from_values_with_kernel(
            weights,
            x_values,
            x_shift_values,
            batch_size,
            channels,
            dot_kernel,
        )?
    {
        return Ok(Some(values));
    }

    let heads = weights.n_heads;
    let cached_weights = time_mixer_native_f32_weights(weights)?;
    let lerp_values = cached_weights.rkvdag_lerp.as_slice();
    let w_r = &cached_weights.w_r;
    let w_k = &cached_weights.w_k;
    let k_scale = &cached_weights.k_scale_linear;
    let v_scale = &cached_weights.v_scale_linear;
    if lerp_values.len() != 8 * channels
        || w_r.out_dim != channels
        || w_r.in_dim != channels
        || w_k.out_dim != channels
        || w_k.in_dim != channels
        || k_scale.out_dim != heads
        || k_scale.in_dim != channels
        || v_scale.out_dim != heads
        || v_scale.in_dim != channels
        || w_r.bias.is_some()
        || w_k.bias.is_some()
    {
        return Ok(None);
    }
    let Some(k_scale_bias_values) = k_scale.bias.as_deref() else {
        return Ok(None);
    };
    let Some(v_scale_bias_values) = v_scale.bias.as_deref() else {
        return Ok(None);
    };
    let w_r_values = w_r.weight.as_slice();
    let w_k_values = w_k.weight.as_slice();
    let k_scale_weight_values = k_scale.weight.as_slice();
    let v_scale_weight_values = v_scale.weight.as_slice();
    let bxc_len = batch_size * channels;
    if x_values.len() != bxc_len
        || x_shift_values.len() != bxc_len
        || w_r_values.len() != channels * channels
        || w_k_values.len() != channels * channels
        || k_scale_weight_values.len() != heads * channels
        || v_scale_weight_values.len() != heads * channels
        || k_scale_bias_values.len() != heads
        || v_scale_bias_values.len() != heads
    {
        return Ok(None);
    }

    let reuse_buffers = native_time_mixer_projection_values_buffer_reuse_enabled();
    let scale_len = batch_size * heads;
    let (
        mut r_output_values,
        mut k_output_values,
        mut k_scale_values,
        mut v_scale_values,
        mut v_values,
        mut d_values,
        mut a_values,
        mut g_values,
        mut r_row,
        mut k_row,
        mut k_scale_row,
        mut v_scale_row,
    ) = if reuse_buffers {
        TIME_MIXER_LERP_PROJECTION_SCRATCH_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            (
                take_time_mixer_projection_vec(&mut pool.r_values, bxc_len),
                take_time_mixer_projection_vec(&mut pool.k_values, bxc_len),
                take_time_mixer_projection_vec(&mut pool.k_scale_values, scale_len),
                take_time_mixer_projection_vec(&mut pool.v_scale_values, scale_len),
                take_time_mixer_projection_vec(&mut pool.v_values, bxc_len),
                take_time_mixer_projection_vec(&mut pool.d_values, bxc_len),
                take_time_mixer_projection_vec(&mut pool.a_values, bxc_len),
                take_time_mixer_projection_vec(&mut pool.g_values, bxc_len),
                take_time_mixer_projection_vec(&mut pool.r_row, channels),
                take_time_mixer_projection_vec(&mut pool.k_row, channels),
                take_time_mixer_projection_vec(&mut pool.k_scale_row, channels),
                take_time_mixer_projection_vec(&mut pool.v_scale_row, channels),
            )
        })
    } else {
        (
            vec![0.0f32; bxc_len],
            vec![0.0f32; bxc_len],
            vec![0.0f32; scale_len],
            vec![0.0f32; scale_len],
            vec![0.0f32; bxc_len],
            vec![0.0f32; bxc_len],
            vec![0.0f32; bxc_len],
            vec![0.0f32; bxc_len],
            vec![0.0f32; channels],
            vec![0.0f32; channels],
            vec![0.0f32; channels],
            vec![0.0f32; channels],
        )
    };

    let use_direct_activation_scalars = use_direct_time_mixer_activation_scalars();
    for batch_index in 0..batch_size {
        let batch_base = batch_index * channels;
        for channel_index in 0..channels {
            let value_index = batch_base + channel_index;
            let x = x_values[value_index];
            let delta = x_shift_values[value_index] - x;
            r_row[channel_index] = x + delta * lerp_values[channel_index];
            k_row[channel_index] = x + delta * lerp_values[channels + channel_index];
            v_values[value_index] = x + delta * lerp_values[2 * channels + channel_index];
            d_values[value_index] = x + delta * lerp_values[3 * channels + channel_index];
            a_values[value_index] = x + delta * lerp_values[4 * channels + channel_index];
            g_values[value_index] = x + delta * lerp_values[5 * channels + channel_index];
            k_scale_row[channel_index] = x + delta * lerp_values[6 * channels + channel_index];
            v_scale_row[channel_index] = x + delta * lerp_values[7 * channels + channel_index];
        }

        let r_output_row = &mut r_output_values[batch_base..batch_base + channels];
        let k_output_row = &mut k_output_values[batch_base..batch_base + channels];
        if native_time_mixer_paired_projection_dots_enabled() {
            for channel_index in 0..channels {
                let r_weight_row =
                    &w_r_values[channel_index * channels..(channel_index + 1) * channels];
                let k_weight_row =
                    &w_k_values[channel_index * channels..(channel_index + 1) * channels];
                let (r_value, k_value) =
                    linear_dot_pair(dot_kernel, &r_row, r_weight_row, &k_row, k_weight_row);
                r_output_row[channel_index] = r_value;
                k_output_row[channel_index] = k_value;
            }
        } else {
            for channel_index in 0..channels {
                let r_weight_row =
                    &w_r_values[channel_index * channels..(channel_index + 1) * channels];
                let k_weight_row =
                    &w_k_values[channel_index * channels..(channel_index + 1) * channels];
                r_output_row[channel_index] = linear_dot(dot_kernel, &r_row, r_weight_row);
                k_output_row[channel_index] = linear_dot(dot_kernel, &k_row, k_weight_row);
            }
        }

        let scale_base = batch_index * heads;
        if native_time_mixer_paired_projection_dots_enabled() {
            for head_index in 0..heads {
                let k_scale_weight_row =
                    &k_scale_weight_values[head_index * channels..(head_index + 1) * channels];
                let v_scale_weight_row =
                    &v_scale_weight_values[head_index * channels..(head_index + 1) * channels];
                let (k_scale_value, v_scale_value) = linear_dot_pair(
                    dot_kernel,
                    &k_scale_row,
                    k_scale_weight_row,
                    &v_scale_row,
                    v_scale_weight_row,
                );
                k_scale_values[scale_base + head_index] = time_mixer_sigmoid_scalar(
                    k_scale_bias_values[head_index] + k_scale_value,
                    use_direct_activation_scalars,
                );
                v_scale_values[scale_base + head_index] = time_mixer_sigmoid_scalar(
                    v_scale_bias_values[head_index] + v_scale_value,
                    use_direct_activation_scalars,
                );
            }
        } else {
            for head_index in 0..heads {
                let k_scale_weight_row =
                    &k_scale_weight_values[head_index * channels..(head_index + 1) * channels];
                let v_scale_weight_row =
                    &v_scale_weight_values[head_index * channels..(head_index + 1) * channels];
                k_scale_values[scale_base + head_index] = time_mixer_sigmoid_scalar(
                    k_scale_bias_values[head_index]
                        + linear_dot(dot_kernel, &k_scale_row, k_scale_weight_row),
                    use_direct_activation_scalars,
                );
                v_scale_values[scale_base + head_index] = time_mixer_sigmoid_scalar(
                    v_scale_bias_values[head_index]
                        + linear_dot(dot_kernel, &v_scale_row, v_scale_weight_row),
                    use_direct_activation_scalars,
                );
            }
        }
    }
    if reuse_buffers {
        recycle_time_mixer_lerp_projection_temp_rows(r_row, k_row, k_scale_row, v_scale_row);
    }

    Ok(Some(TimeMixerLerpProjectionScratchValues {
        batch_size,
        channels,
        heads,
        r_values: r_output_values,
        k_values: k_output_values,
        k_scale_values,
        v_scale_values,
        v_values,
        d_values,
        a_values,
        g_values,
    }))
}

pub(super) fn time_mixer_batched_lerp_projection_scratch_values_native_from_values_with_kernel(
    weights: &Rwkv7RnnTimeMixerWeights,
    x_values: &[f32],
    x_shift_values: &[f32],
    batch_size: usize,
    channels: usize,
    dot_kernel: NativeLinearDotKernel,
) -> Result<Option<TimeMixerLerpProjectionScratchValues>> {
    if !native_time_mixer_batched_lerp_projection_values_enabled() || batch_size < 2 {
        return Ok(None);
    }

    let heads = weights.n_heads;
    if heads < 4 {
        return Ok(None);
    }

    let cached_weights = time_mixer_native_f32_weights(weights)?;
    let lerp_values = cached_weights.rkvdag_lerp.as_slice();
    let w_r = &cached_weights.w_r;
    let w_k = &cached_weights.w_k;
    let k_scale = &cached_weights.k_scale_linear;
    let v_scale = &cached_weights.v_scale_linear;
    if lerp_values.len() != 8 * channels
        || w_r.out_dim != channels
        || w_r.in_dim != channels
        || w_k.out_dim != channels
        || w_k.in_dim != channels
        || k_scale.out_dim != heads
        || k_scale.in_dim != channels
        || v_scale.out_dim != heads
        || v_scale.in_dim != channels
        || w_r.bias.is_some()
        || w_k.bias.is_some()
    {
        return Ok(None);
    }
    let Some(k_scale_bias_values) = k_scale.bias.as_deref() else {
        return Ok(None);
    };
    let Some(v_scale_bias_values) = v_scale.bias.as_deref() else {
        return Ok(None);
    };
    let w_r_values = w_r.weight.as_slice();
    let w_k_values = w_k.weight.as_slice();
    let k_scale_weight_values = k_scale.weight.as_slice();
    let v_scale_weight_values = v_scale.weight.as_slice();
    let bxc_len = batch_size * channels;
    let scale_len = batch_size * heads;
    if x_values.len() != bxc_len
        || x_shift_values.len() != bxc_len
        || w_r_values.len() != channels * channels
        || w_k_values.len() != channels * channels
        || k_scale_weight_values.len() != heads * channels
        || v_scale_weight_values.len() != heads * channels
        || k_scale_bias_values.len() != heads
        || v_scale_bias_values.len() != heads
    {
        return Ok(None);
    }

    let reuse_buffers = native_time_mixer_projection_values_buffer_reuse_enabled();
    let (
        mut r_input_values,
        mut k_input_values,
        mut r_output_values,
        mut k_output_values,
        mut k_scale_values,
        mut v_scale_values,
        mut v_values,
        mut d_values,
        mut a_values,
        mut g_values,
    ) = if reuse_buffers {
        TIME_MIXER_LERP_PROJECTION_SCRATCH_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            (
                take_time_mixer_projection_vec(&mut pool.r_input_values, bxc_len),
                take_time_mixer_projection_vec(&mut pool.k_input_values, bxc_len),
                take_time_mixer_projection_vec(&mut pool.r_values, bxc_len),
                take_time_mixer_projection_vec(&mut pool.k_values, bxc_len),
                take_time_mixer_projection_vec(&mut pool.k_scale_values, scale_len),
                take_time_mixer_projection_vec(&mut pool.v_scale_values, scale_len),
                take_time_mixer_projection_vec(&mut pool.v_values, bxc_len),
                take_time_mixer_projection_vec(&mut pool.d_values, bxc_len),
                take_time_mixer_projection_vec(&mut pool.a_values, bxc_len),
                take_time_mixer_projection_vec(&mut pool.g_values, bxc_len),
            )
        })
    } else {
        (
            vec![0.0f32; bxc_len],
            vec![0.0f32; bxc_len],
            vec![0.0f32; bxc_len],
            vec![0.0f32; bxc_len],
            vec![0.0f32; scale_len],
            vec![0.0f32; scale_len],
            vec![0.0f32; bxc_len],
            vec![0.0f32; bxc_len],
            vec![0.0f32; bxc_len],
            vec![0.0f32; bxc_len],
        )
    };

    for batch_index in 0..batch_size {
        let batch_base = batch_index * channels;
        for channel_index in 0..channels {
            let value_index = batch_base + channel_index;
            let x = x_values[value_index];
            let delta = x_shift_values[value_index] - x;
            r_input_values[value_index] = x + delta * lerp_values[channel_index];
            k_input_values[value_index] = x + delta * lerp_values[channels + channel_index];
            v_values[value_index] = x + delta * lerp_values[2 * channels + channel_index];
            d_values[value_index] = x + delta * lerp_values[6 * channels + channel_index];
            a_values[value_index] = x + delta * lerp_values[7 * channels + channel_index];
            g_values[value_index] = x + delta * lerp_values[5 * channels + channel_index];
        }
    }

    let projected = linear_project_batch_same_x(
        dot_kernel,
        &r_input_values,
        batch_size,
        channels,
        w_r_values,
        channels,
        &mut r_output_values,
    ) && linear_project_batch_same_x(
        dot_kernel,
        &k_input_values,
        batch_size,
        channels,
        w_k_values,
        channels,
        &mut k_output_values,
    ) && linear_project_batch_same_x(
        dot_kernel,
        &d_values,
        batch_size,
        channels,
        k_scale_weight_values,
        heads,
        &mut k_scale_values,
    ) && linear_project_batch_same_x(
        dot_kernel,
        &a_values,
        batch_size,
        channels,
        v_scale_weight_values,
        heads,
        &mut v_scale_values,
    );

    if !projected {
        if reuse_buffers {
            recycle_time_mixer_lerp_projection_input_values(r_input_values, k_input_values);
            recycle_time_mixer_lerp_projection_scratch_values(
                TimeMixerLerpProjectionScratchValues {
                    batch_size,
                    channels,
                    heads,
                    r_values: r_output_values,
                    k_values: k_output_values,
                    k_scale_values,
                    v_scale_values,
                    v_values,
                    d_values,
                    a_values,
                    g_values,
                },
            );
        }
        return Ok(None);
    }

    let use_direct_activation_scalars = use_direct_time_mixer_activation_scalars();
    for batch_index in 0..batch_size {
        let scale_base = batch_index * heads;
        for head_index in 0..heads {
            k_scale_values[scale_base + head_index] = time_mixer_sigmoid_scalar(
                k_scale_bias_values[head_index] + k_scale_values[scale_base + head_index],
                use_direct_activation_scalars,
            );
            v_scale_values[scale_base + head_index] = time_mixer_sigmoid_scalar(
                v_scale_bias_values[head_index] + v_scale_values[scale_base + head_index],
                use_direct_activation_scalars,
            );
        }

        let batch_base = batch_index * channels;
        for channel_index in 0..channels {
            let value_index = batch_base + channel_index;
            let x = x_values[value_index];
            let delta = x_shift_values[value_index] - x;
            d_values[value_index] = x + delta * lerp_values[3 * channels + channel_index];
            a_values[value_index] = x + delta * lerp_values[4 * channels + channel_index];
        }
    }

    if reuse_buffers {
        recycle_time_mixer_lerp_projection_input_values(r_input_values, k_input_values);
    }

    Ok(Some(TimeMixerLerpProjectionScratchValues {
        batch_size,
        channels,
        heads,
        r_values: r_output_values,
        k_values: k_output_values,
        k_scale_values,
        v_scale_values,
        v_values,
        d_values,
        a_values,
        g_values,
    }))
}

pub(super) fn time_mixer_projection_group_candle(
    weights: &Rwkv7RnnTimeMixerWeights,
    rkvdag_parts: &[Tensor],
    profile: &mut Option<&mut ForwardProfile>,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let r_b1c = linear_profiled(&rkvdag_parts[0], &weights.w_r, profile)?;
    let k_b1c = linear_profiled(&rkvdag_parts[1], &weights.w_k, profile)?;
    let k_scale_b1h = nn_ops::sigmoid(&linear_profiled(
        &rkvdag_parts[6],
        &weights.k_scale_linear,
        profile,
    )?)?;
    let v_scale_b1h = nn_ops::sigmoid(&linear_profiled(
        &rkvdag_parts[7],
        &weights.v_scale_linear,
        profile,
    )?)?;
    Ok((r_b1c, k_b1c, k_scale_b1h, v_scale_b1h))
}

pub(super) fn time_mixer_projection_group_native(
    rkvdag_parts: &[Tensor],
    weights: &Rwkv7RnnTimeMixerWeights,
) -> Result<Option<(Tensor, Tensor, Tensor, Tensor)>> {
    let Some(dot_kernel) = native_linear_dot_kernel() else {
        return Ok(None);
    };
    time_mixer_projection_group_native_with_kernel(rkvdag_parts, weights, dot_kernel)
}

pub(super) fn time_mixer_projection_group_native_with_kernel(
    rkvdag_parts: &[Tensor],
    weights: &Rwkv7RnnTimeMixerWeights,
    dot_kernel: NativeLinearDotKernel,
) -> Result<Option<(Tensor, Tensor, Tensor, Tensor)>> {
    if rkvdag_parts.len() != 8 {
        return Ok(None);
    }
    let (batch_size, one, channels) = rkvdag_parts[0].dims3()?;
    if one != 1 {
        return Ok(None);
    }
    for part in [&rkvdag_parts[1], &rkvdag_parts[6], &rkvdag_parts[7]] {
        if part.dims() != [batch_size, 1, channels] {
            return Ok(None);
        }
    }

    let heads = weights.n_heads;
    let cached_weights = time_mixer_native_f32_weights(weights)?;
    let w_r = &cached_weights.w_r;
    let w_k = &cached_weights.w_k;
    let k_scale = &cached_weights.k_scale_linear;
    let v_scale = &cached_weights.v_scale_linear;
    if w_r.out_dim != channels
        || w_r.in_dim != channels
        || w_k.out_dim != channels
        || w_k.in_dim != channels
        || k_scale.out_dim != heads
        || k_scale.in_dim != channels
        || v_scale.out_dim != heads
        || v_scale.in_dim != channels
        || w_r.bias.is_some()
        || w_k.bias.is_some()
    {
        return Ok(None);
    };
    let Some(k_scale_bias_values) = k_scale.bias.as_deref() else {
        return Ok(None);
    };
    let Some(v_scale_bias_values) = v_scale.bias.as_deref() else {
        return Ok(None);
    };

    let r_input_data = f32_tensor_data(&rkvdag_parts[0])?;
    let k_input_data = f32_tensor_data(&rkvdag_parts[1])?;
    let k_scale_input_data = f32_tensor_data(&rkvdag_parts[6])?;
    let v_scale_input_data = f32_tensor_data(&rkvdag_parts[7])?;
    let r_input_values = r_input_data.as_slice()?;
    let k_input_values = k_input_data.as_slice()?;
    let k_scale_input_values = k_scale_input_data.as_slice()?;
    let v_scale_input_values = v_scale_input_data.as_slice()?;
    let w_r_values = w_r.weight.as_slice();
    let w_k_values = w_k.weight.as_slice();
    let k_scale_weight_values = k_scale.weight.as_slice();
    let v_scale_weight_values = v_scale.weight.as_slice();

    let bxc_len = batch_size * channels;
    if r_input_values.len() != bxc_len
        || k_input_values.len() != bxc_len
        || k_scale_input_values.len() != bxc_len
        || v_scale_input_values.len() != bxc_len
        || w_r_values.len() != channels * channels
        || w_k_values.len() != channels * channels
        || k_scale_weight_values.len() != heads * channels
        || v_scale_weight_values.len() != heads * channels
        || k_scale_bias_values.len() != heads
        || v_scale_bias_values.len() != heads
    {
        return Ok(None);
    }

    let mut r_output_values = vec![0.0f32; bxc_len];
    let mut k_output_values = vec![0.0f32; bxc_len];
    let mut k_scale_values = vec![0.0f32; batch_size * heads];
    let mut v_scale_values = vec![0.0f32; batch_size * heads];
    let use_direct_activation_scalars = use_direct_time_mixer_activation_scalars();
    for batch_index in 0..batch_size {
        let batch_base = batch_index * channels;
        let r_row = &r_input_values[batch_base..batch_base + channels];
        let k_row = &k_input_values[batch_base..batch_base + channels];
        let k_scale_row = &k_scale_input_values[batch_base..batch_base + channels];
        let v_scale_row = &v_scale_input_values[batch_base..batch_base + channels];
        let r_output_row = &mut r_output_values[batch_base..batch_base + channels];
        let k_output_row = &mut k_output_values[batch_base..batch_base + channels];

        for channel_index in 0..channels {
            let r_weight_row =
                &w_r_values[channel_index * channels..(channel_index + 1) * channels];
            let k_weight_row =
                &w_k_values[channel_index * channels..(channel_index + 1) * channels];
            r_output_row[channel_index] = linear_dot(dot_kernel, r_row, r_weight_row);
            k_output_row[channel_index] = linear_dot(dot_kernel, k_row, k_weight_row);
        }

        let scale_base = batch_index * heads;
        for head_index in 0..heads {
            let k_scale_weight_row =
                &k_scale_weight_values[head_index * channels..(head_index + 1) * channels];
            let v_scale_weight_row =
                &v_scale_weight_values[head_index * channels..(head_index + 1) * channels];
            k_scale_values[scale_base + head_index] = time_mixer_sigmoid_scalar(
                k_scale_bias_values[head_index]
                    + linear_dot(dot_kernel, k_scale_row, k_scale_weight_row),
                use_direct_activation_scalars,
            );
            v_scale_values[scale_base + head_index] = time_mixer_sigmoid_scalar(
                v_scale_bias_values[head_index]
                    + linear_dot(dot_kernel, v_scale_row, v_scale_weight_row),
                use_direct_activation_scalars,
            );
        }
    }

    Ok(Some((
        Tensor::from_vec(
            r_output_values,
            (batch_size, 1usize, channels),
            rkvdag_parts[0].device(),
        )?,
        Tensor::from_vec(
            k_output_values,
            (batch_size, 1usize, channels),
            rkvdag_parts[1].device(),
        )?,
        Tensor::from_vec(
            k_scale_values,
            (batch_size, 1usize, heads),
            rkvdag_parts[6].device(),
        )?,
        Tensor::from_vec(
            v_scale_values,
            (batch_size, 1usize, heads),
            rkvdag_parts[7].device(),
        )?,
    )))
}

pub(super) fn take_time_mixer_projection_vec(slot: &mut Vec<f32>, len: usize) -> Vec<f32> {
    let mut values = std::mem::take(slot);
    values.resize(len, 0.0);
    values
}

pub(super) fn recycle_time_mixer_projection_vec(slot: &mut Vec<f32>, mut values: Vec<f32>) {
    values.clear();
    if values.capacity() >= slot.capacity() {
        *slot = values;
    }
}

pub(super) fn take_time_mixer_predict_x_values(len: usize) -> Vec<f32> {
    if !native_time_mixer_predict_x_buffer_reuse_enabled() {
        return vec![0.0f32; len];
    }
    TIME_MIXER_LERP_PROJECTION_SCRATCH_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        take_time_mixer_projection_vec(&mut pool.x_values, len)
    })
}

pub(super) fn recycle_time_mixer_predict_x_values(values: Vec<f32>) {
    if !native_time_mixer_predict_x_buffer_reuse_enabled() {
        return;
    }
    TIME_MIXER_LERP_PROJECTION_SCRATCH_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        recycle_time_mixer_projection_vec(&mut pool.x_values, values);
    });
}

pub(super) fn recycle_time_mixer_lerp_projection_temp_rows(
    r_row: Vec<f32>,
    k_row: Vec<f32>,
    k_scale_row: Vec<f32>,
    v_scale_row: Vec<f32>,
) {
    TIME_MIXER_LERP_PROJECTION_SCRATCH_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        recycle_time_mixer_projection_vec(&mut pool.r_row, r_row);
        recycle_time_mixer_projection_vec(&mut pool.k_row, k_row);
        recycle_time_mixer_projection_vec(&mut pool.k_scale_row, k_scale_row);
        recycle_time_mixer_projection_vec(&mut pool.v_scale_row, v_scale_row);
    });
}

pub(super) fn recycle_time_mixer_lerp_projection_input_values(
    r_input_values: Vec<f32>,
    k_input_values: Vec<f32>,
) {
    TIME_MIXER_LERP_PROJECTION_SCRATCH_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        recycle_time_mixer_projection_vec(&mut pool.r_input_values, r_input_values);
        recycle_time_mixer_projection_vec(&mut pool.k_input_values, k_input_values);
    });
}

pub(super) fn recycle_time_mixer_lerp_projection_scratch_values(
    values: TimeMixerLerpProjectionScratchValues,
) {
    if !native_time_mixer_projection_values_buffer_reuse_enabled() {
        return;
    }

    TIME_MIXER_LERP_PROJECTION_SCRATCH_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        recycle_time_mixer_projection_vec(&mut pool.r_values, values.r_values);
        recycle_time_mixer_projection_vec(&mut pool.k_values, values.k_values);
        recycle_time_mixer_projection_vec(&mut pool.k_scale_values, values.k_scale_values);
        recycle_time_mixer_projection_vec(&mut pool.v_scale_values, values.v_scale_values);
        recycle_time_mixer_projection_vec(&mut pool.v_values, values.v_values);
        recycle_time_mixer_projection_vec(&mut pool.d_values, values.d_values);
        recycle_time_mixer_projection_vec(&mut pool.a_values, values.a_values);
        recycle_time_mixer_projection_vec(&mut pool.g_values, values.g_values);
    });
}

pub(super) fn take_time_mixer_output_scratch_vec(slot: &mut Vec<f32>, len: usize) -> Vec<f32> {
    if !native_time_mixer_output_buffer_reuse_enabled() {
        return vec![0.0f32; len];
    }
    let mut values = std::mem::take(slot);
    values.resize(len, 0.0);
    values
}

pub(super) fn time_mixer_output_scratch_vectors(
    hidden_len: usize,
    hidden_batch_len: usize,
) -> (Vec<f32>, Vec<f32>) {
    TIME_MIXER_OUTPUT_SCRATCH_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        (
            take_time_mixer_output_scratch_vec(&mut pool.hidden_values, hidden_len),
            take_time_mixer_output_scratch_vec(&mut pool.hidden_batch_values, hidden_batch_len),
        )
    })
}

pub(super) fn recycle_time_mixer_output_scratch_vec(slot: &mut Vec<f32>, mut values: Vec<f32>) {
    if !native_time_mixer_output_buffer_reuse_enabled() {
        return;
    }
    values.clear();
    if values.capacity() >= slot.capacity() {
        *slot = values;
    }
}

pub(super) fn recycle_time_mixer_output_scratch(
    hidden_values: Vec<f32>,
    hidden_batch_values: Vec<f32>,
) {
    if !native_time_mixer_output_buffer_reuse_enabled() {
        return;
    }
    TIME_MIXER_OUTPUT_SCRATCH_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        recycle_time_mixer_output_scratch_vec(&mut pool.hidden_values, hidden_values);
        recycle_time_mixer_output_scratch_vec(&mut pool.hidden_batch_values, hidden_batch_values);
    });
}

pub(super) fn time_mixer_v_mix(
    weights: &Rwkv7RnnTimeMixerWeights,
    v_b1c: &Tensor,
    v0_bc: &Tensor,
    profile: &mut Option<&mut ForwardProfile>,
) -> Result<(Tensor, Tensor)> {
    if weights.layer_id == 0 {
        let projected_v_b1c = linear_profiled(v_b1c, &weights.w_v, profile)?;
        let next_v0_bc = projected_v_b1c.squeeze(1)?;
        return Ok((projected_v_b1c, next_v0_bc));
    }

    if native_time_mixer_v_mix_enabled() {
        if let Some(mixed_v_b1c) = time_mixer_v_mix_native(v_b1c, v0_bc, weights)? {
            if let Some(profile) = profile.as_deref_mut() {
                profile.time_mixer.native_v_mix_calls += 1;
                profile.time_mixer.native_v_mix_rows += v_b1c.dims3()?.0;
            }
            return Ok((mixed_v_b1c, v0_bc.clone()));
        }
    }

    time_mixer_v_mix_candle(weights, v_b1c, v0_bc, profile)
}

pub(super) fn time_mixer_v_mix_candle(
    weights: &Rwkv7RnnTimeMixerWeights,
    v_b1c: &Tensor,
    v0_bc: &Tensor,
    profile: &mut Option<&mut ForwardProfile>,
) -> Result<(Tensor, Tensor)> {
    let v_lora_a = linear_profiled(v_b1c, &weights.v_lora_simple.a, profile)?;
    let v_lerp_b1c = nn_ops::sigmoid(&linear_profiled(
        &v_lora_a,
        &weights.v_lora_simple.b_and_lamb,
        profile,
    )?)?;
    let projected_v_b1c = linear_profiled(v_b1c, &weights.w_v, profile)?;
    let mixed_v_b1c = projected_v_b1c.broadcast_add(
        &v0_bc
            .unsqueeze(1)?
            .broadcast_sub(&projected_v_b1c)?
            .broadcast_mul(&v_lerp_b1c)?,
    )?;
    Ok((mixed_v_b1c, v0_bc.clone()))
}

pub(super) fn time_mixer_v_mix_native(
    v_b1c: &Tensor,
    v0_bc: &Tensor,
    weights: &Rwkv7RnnTimeMixerWeights,
) -> Result<Option<Tensor>> {
    let Some(dot_kernel) = native_linear_dot_kernel() else {
        return Ok(None);
    };
    time_mixer_v_mix_native_with_kernel(v_b1c, v0_bc, weights, dot_kernel)
}

pub(super) fn time_mixer_v_mix_native_with_kernel(
    v_b1c: &Tensor,
    v0_bc: &Tensor,
    weights: &Rwkv7RnnTimeMixerWeights,
    dot_kernel: NativeLinearDotKernel,
) -> Result<Option<Tensor>> {
    let Some(values) =
        time_mixer_v_mix_values_native_with_kernel(v_b1c, v0_bc, weights, dot_kernel)?
    else {
        return Ok(None);
    };
    Tensor::from_vec(
        values.values,
        (values.batch_size, 1usize, values.channels),
        v_b1c.device(),
    )
    .map(Some)
}

pub(super) fn time_mixer_v_mix_values_native_with_kernel(
    v_b1c: &Tensor,
    v0_bc: &Tensor,
    weights: &Rwkv7RnnTimeMixerWeights,
    dot_kernel: NativeLinearDotKernel,
) -> Result<Option<TimeMixerVMixScratchValues>> {
    let (batch_size, one, channels) = v_b1c.dims3()?;
    if one != 1 || v0_bc.dims() != [batch_size, channels] {
        return Ok(None);
    }

    let cached_weights = time_mixer_native_f32_weights(weights)?;
    let lora_a = &cached_weights.v_lora_a;
    let lora_b = &cached_weights.v_lora_b;
    let w_v = &cached_weights.w_v;
    let lora_dim = lora_a.out_dim;
    if lora_a.in_dim != channels
        || lora_b.out_dim != channels
        || lora_b.in_dim != lora_dim
        || w_v.out_dim != channels
        || w_v.in_dim != channels
        || lora_a.bias.is_some()
        || w_v.bias.is_some()
    {
        return Ok(None);
    };
    let Some(lora_b_bias_values) = lora_b.bias.as_deref() else {
        return Ok(None);
    };

    let v_data = f32_tensor_data(v_b1c)?;
    let v0_data = f32_tensor_data(v0_bc)?;
    let v_values = v_data.as_slice()?;
    let v0_values = v0_data.as_slice()?;
    let lora_a_values = lora_a.weight.as_slice();
    let lora_b_values = lora_b.weight.as_slice();
    let w_v_values = w_v.weight.as_slice();

    let bxc_len = batch_size * channels;
    if v_values.len() != bxc_len
        || v0_values.len() != bxc_len
        || lora_a_values.len() != lora_dim * channels
        || lora_b_values.len() != channels * lora_dim
        || lora_b_bias_values.len() != channels
        || w_v_values.len() != channels * channels
    {
        return Ok(None);
    }

    let mut output_values = vec![0.0f32; bxc_len];
    let mut lora_hidden = vec![0.0f32; lora_dim];
    let use_direct_activation_scalars = use_direct_time_mixer_activation_scalars();
    if time_mixer_v_mix_project_dot4_values(
        &mut output_values,
        &mut lora_hidden,
        v_values,
        v0_values,
        batch_size,
        channels,
        lora_dim,
        lora_a_values,
        lora_b_values,
        lora_b_bias_values,
        w_v,
        dot_kernel,
        use_direct_activation_scalars,
    ) {
        return Ok(Some(TimeMixerVMixScratchValues {
            batch_size,
            channels,
            values: output_values,
        }));
    }

    for batch_index in 0..batch_size {
        let batch_base = batch_index * channels;
        let v_row = &v_values[batch_base..batch_base + channels];

        for lora_index in 0..lora_dim {
            let weight_row = &lora_a_values[lora_index * channels..(lora_index + 1) * channels];
            lora_hidden[lora_index] = linear_dot(dot_kernel, v_row, weight_row);
        }

        let output_row = &mut output_values[batch_base..batch_base + channels];
        for channel_index in 0..channels {
            let lora_weight_row =
                &lora_b_values[channel_index * lora_dim..(channel_index + 1) * lora_dim];
            let lora_value = lora_b_bias_values[channel_index]
                + linear_dot(dot_kernel, &lora_hidden, lora_weight_row);
            let v_lerp = time_mixer_sigmoid_scalar(lora_value, use_direct_activation_scalars);

            let projection_weight_row =
                &w_v_values[channel_index * channels..(channel_index + 1) * channels];
            let projected_v = linear_dot(dot_kernel, v_row, projection_weight_row);
            output_row[channel_index] =
                projected_v + (v0_values[batch_base + channel_index] - projected_v) * v_lerp;
        }
    }

    Ok(Some(TimeMixerVMixScratchValues {
        batch_size,
        channels,
        values: output_values,
    }))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn time_mixer_v_mix_values_native_from_input_values_with_kernel(
    v_values: &[f32],
    batch_size: usize,
    channels: usize,
    v0_bc: &Tensor,
    weights: &Rwkv7RnnTimeMixerWeights,
    dot_kernel: NativeLinearDotKernel,
) -> Result<Option<TimeMixerVMixScratchValues>> {
    if v0_bc.dims() != [batch_size, channels] {
        return Ok(None);
    }

    let cached_weights = time_mixer_native_f32_weights(weights)?;
    let lora_a = &cached_weights.v_lora_a;
    let lora_b = &cached_weights.v_lora_b;
    let w_v = &cached_weights.w_v;
    let lora_dim = lora_a.out_dim;
    if lora_a.in_dim != channels
        || lora_b.out_dim != channels
        || lora_b.in_dim != lora_dim
        || w_v.out_dim != channels
        || w_v.in_dim != channels
        || lora_a.bias.is_some()
        || w_v.bias.is_some()
    {
        return Ok(None);
    };
    let Some(lora_b_bias_values) = lora_b.bias.as_deref() else {
        return Ok(None);
    };

    let v0_data = f32_tensor_data(v0_bc)?;
    let v0_values = v0_data.as_slice()?;
    let lora_a_values = lora_a.weight.as_slice();
    let lora_b_values = lora_b.weight.as_slice();
    let w_v_values = w_v.weight.as_slice();

    let bxc_len = batch_size * channels;
    if v_values.len() != bxc_len
        || v0_values.len() != bxc_len
        || lora_a_values.len() != lora_dim * channels
        || lora_b_values.len() != channels * lora_dim
        || lora_b_bias_values.len() != channels
        || w_v_values.len() != channels * channels
    {
        return Ok(None);
    }

    let mut output_values = vec![0.0f32; bxc_len];
    let mut lora_hidden = vec![0.0f32; lora_dim];
    let use_direct_activation_scalars = use_direct_time_mixer_activation_scalars();
    if time_mixer_v_mix_project_dot4_values(
        &mut output_values,
        &mut lora_hidden,
        v_values,
        v0_values,
        batch_size,
        channels,
        lora_dim,
        lora_a_values,
        lora_b_values,
        lora_b_bias_values,
        w_v,
        dot_kernel,
        use_direct_activation_scalars,
    ) {
        return Ok(Some(TimeMixerVMixScratchValues {
            batch_size,
            channels,
            values: output_values,
        }));
    }

    for batch_index in 0..batch_size {
        let batch_base = batch_index * channels;
        let v_row = &v_values[batch_base..batch_base + channels];

        for lora_index in 0..lora_dim {
            let weight_row = &lora_a_values[lora_index * channels..(lora_index + 1) * channels];
            lora_hidden[lora_index] = linear_dot(dot_kernel, v_row, weight_row);
        }

        let output_row = &mut output_values[batch_base..batch_base + channels];
        for channel_index in 0..channels {
            let lora_weight_row =
                &lora_b_values[channel_index * lora_dim..(channel_index + 1) * lora_dim];
            let lora_value = lora_b_bias_values[channel_index]
                + linear_dot(dot_kernel, &lora_hidden, lora_weight_row);
            let v_lerp = time_mixer_sigmoid_scalar(lora_value, use_direct_activation_scalars);

            let projection_weight_row =
                &w_v_values[channel_index * channels..(channel_index + 1) * channels];
            let projected_v = linear_dot(dot_kernel, v_row, projection_weight_row);
            output_row[channel_index] =
                projected_v + (v0_values[batch_base + channel_index] - projected_v) * v_lerp;
        }
    }

    Ok(Some(TimeMixerVMixScratchValues {
        batch_size,
        channels,
        values: output_values,
    }))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn time_mixer_v_mix_values_native_from_flat_v0_with_kernel(
    v_values: &[f32],
    v0_values: &[f32],
    batch_size: usize,
    channels: usize,
    weights: &Rwkv7RnnTimeMixerWeights,
    dot_kernel: NativeLinearDotKernel,
) -> Result<Option<TimeMixerVMixScratchValues>> {
    let cached_weights = time_mixer_native_f32_weights(weights)?;
    let lora_a = &cached_weights.v_lora_a;
    let lora_b = &cached_weights.v_lora_b;
    let w_v = &cached_weights.w_v;
    let lora_dim = lora_a.out_dim;
    if lora_a.in_dim != channels
        || lora_b.out_dim != channels
        || lora_b.in_dim != lora_dim
        || w_v.out_dim != channels
        || w_v.in_dim != channels
        || lora_a.bias.is_some()
        || w_v.bias.is_some()
    {
        return Ok(None);
    };
    let Some(lora_b_bias_values) = lora_b.bias.as_deref() else {
        return Ok(None);
    };

    let lora_a_values = lora_a.weight.as_slice();
    let lora_b_values = lora_b.weight.as_slice();
    let w_v_values = w_v.weight.as_slice();

    let bxc_len = batch_size * channels;
    if v_values.len() != bxc_len
        || v0_values.len() != bxc_len
        || lora_a_values.len() != lora_dim * channels
        || lora_b_values.len() != channels * lora_dim
        || lora_b_bias_values.len() != channels
        || w_v_values.len() != channels * channels
    {
        return Ok(None);
    }

    let mut output_values = vec![0.0f32; bxc_len];
    let mut lora_hidden = vec![0.0f32; lora_dim];
    let use_direct_activation_scalars = use_direct_time_mixer_activation_scalars();
    if time_mixer_v_mix_project_dot4_values(
        &mut output_values,
        &mut lora_hidden,
        v_values,
        v0_values,
        batch_size,
        channels,
        lora_dim,
        lora_a_values,
        lora_b_values,
        lora_b_bias_values,
        w_v,
        dot_kernel,
        use_direct_activation_scalars,
    ) {
        return Ok(Some(TimeMixerVMixScratchValues {
            batch_size,
            channels,
            values: output_values,
        }));
    }

    for batch_index in 0..batch_size {
        let batch_base = batch_index * channels;
        let v_row = &v_values[batch_base..batch_base + channels];

        for lora_index in 0..lora_dim {
            let weight_row = &lora_a_values[lora_index * channels..(lora_index + 1) * channels];
            lora_hidden[lora_index] = linear_dot(dot_kernel, v_row, weight_row);
        }

        let output_row = &mut output_values[batch_base..batch_base + channels];
        for channel_index in 0..channels {
            let lora_weight_row =
                &lora_b_values[channel_index * lora_dim..(channel_index + 1) * lora_dim];
            let lora_value = lora_b_bias_values[channel_index]
                + linear_dot(dot_kernel, &lora_hidden, lora_weight_row);
            let v_lerp = time_mixer_sigmoid_scalar(lora_value, use_direct_activation_scalars);

            let projection_weight_row =
                &w_v_values[channel_index * channels..(channel_index + 1) * channels];
            let projected_v = linear_dot(dot_kernel, v_row, projection_weight_row);
            output_row[channel_index] =
                projected_v + (v0_values[batch_base + channel_index] - projected_v) * v_lerp;
        }
    }

    Ok(Some(TimeMixerVMixScratchValues {
        batch_size,
        channels,
        values: output_values,
    }))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn time_mixer_v_mix_project_dot4_values(
    output_values: &mut [f32],
    lora_hidden: &mut [f32],
    v_values: &[f32],
    v0_values: &[f32],
    batch_size: usize,
    channels: usize,
    lora_dim: usize,
    lora_a_values: &[f32],
    lora_b_values: &[f32],
    lora_b_bias_values: &[f32],
    w_v: &LinearF32Weights,
    dot_kernel: NativeLinearDotKernel,
    use_direct_activation_scalars: bool,
) -> bool {
    if !(native_time_mixer_v_mix_project_dot4_enabled() && native_linear_dot4_same_x_enabled()) {
        return false;
    }

    let w_v_values = w_v.weight.as_slice();
    let bxc_len = batch_size * channels;
    if output_values.len() != bxc_len
        || v_values.len() != bxc_len
        || v0_values.len() != bxc_len
        || lora_hidden.len() != lora_dim
        || lora_a_values.len() != lora_dim * channels
        || lora_b_values.len() != channels * lora_dim
        || lora_b_bias_values.len() != channels
        || w_v_values.len() != channels * channels
    {
        return false;
    }

    for batch_index in 0..batch_size {
        let batch_base = batch_index * channels;
        let v_row = &v_values[batch_base..batch_base + channels];

        for lora_index in 0..lora_dim {
            let weight_row = &lora_a_values[lora_index * channels..(lora_index + 1) * channels];
            lora_hidden[lora_index] = linear_dot(dot_kernel, v_row, weight_row);
        }

        let output_row = &mut output_values[batch_base..batch_base + channels];
        if batch_size == 1 {
            linear_project_f32_row_same_x(dot_kernel, v_row, w_v, output_row);
        } else {
            linear_project_row_same_x(
                dot_kernel, v_row, w_v_values, channels, channels, output_row,
            );
        }

        for channel_index in 0..channels {
            let lora_weight_row =
                &lora_b_values[channel_index * lora_dim..(channel_index + 1) * lora_dim];
            let lora_value = lora_b_bias_values[channel_index]
                + linear_dot(dot_kernel, lora_hidden, lora_weight_row);
            let v_lerp = time_mixer_sigmoid_scalar(lora_value, use_direct_activation_scalars);
            let projected_v = output_row[channel_index];
            output_row[channel_index] =
                projected_v + (v0_values[batch_base + channel_index] - projected_v) * v_lerp;
        }
    }

    true
}

pub(super) fn time_mixer_layer0_v_values_native_from_input_values_with_kernel(
    v_values: &[f32],
    batch_size: usize,
    channels: usize,
    weights: &Rwkv7RnnTimeMixerWeights,
    dot_kernel: NativeLinearDotKernel,
) -> Result<Option<TimeMixerVMixScratchValues>> {
    if weights.layer_id != 0 {
        return Ok(None);
    }

    let cached_weights = time_mixer_native_f32_weights(weights)?;
    let w_v = &cached_weights.w_v;
    if w_v.out_dim != channels || w_v.in_dim != channels || w_v.bias.is_some() {
        return Ok(None);
    }

    let weight_values = w_v.weight.as_slice();
    let bxc_len = batch_size * channels;
    if v_values.len() != bxc_len || weight_values.len() != channels * channels {
        return Ok(None);
    }

    let mut output_values = vec![0.0f32; bxc_len];
    linear_project_f32_batch_same_x(dot_kernel, v_values, batch_size, w_v, &mut output_values);

    Ok(Some(TimeMixerVMixScratchValues {
        batch_size,
        channels,
        values: output_values,
    }))
}

pub(super) fn sigmoid_scalar(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn time_mixer_output_candle(
    in_b1c: &Tensor,
    r_b1hk: &Tensor,
    k_b1hk: &Tensor,
    v_b1hk: &Tensor,
    bonus: &Tensor,
    out_b1c: &Tensor,
    g_b1c: &Tensor,
    w_o: &LinearWeights,
    profile: &mut Option<&mut ForwardProfile>,
) -> Result<Tensor> {
    let (batch_size, _, channels) = in_b1c.dims3()?;
    let bonus_bc = r_b1hk
        .broadcast_mul(bonus)?
        .broadcast_mul(k_b1hk)?
        .sum_keepdim(D::Minus1)?
        .broadcast_mul(v_b1hk)?;
    record_reshape_layout(profile, &bonus_bc);
    let bonus_b1c = bonus_bc.reshape((batch_size, 1usize, channels))?;
    let out_b1c = linear_profiled(&g_b1c.broadcast_mul(&(out_b1c + bonus_b1c)?)?, w_o, profile)?;
    (in_b1c + out_b1c)?.squeeze(1)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn time_mixer_middle_output_scratch_native(
    in_b1c: &Tensor,
    r_b1c: &Tensor,
    k_b1c: &Tensor,
    v_b1c: &Tensor,
    w_b1c: &Tensor,
    a_b1c: &Tensor,
    k_scale_b1h: &Tensor,
    v_scale_b1h: &Tensor,
    state_b1hkk: &Tensor,
    weights: &Rwkv7RnnTimeMixerWeights,
    heads: usize,
    head_size: usize,
    return_state: bool,
    materialize_out_tensor: bool,
    profile: Option<&mut crate::profile::SingleTimestepProfile>,
) -> Result<Option<TimeMixerMiddleOutputScratch>> {
    let Some(dot_kernel) = native_linear_dot_kernel() else {
        return Ok(None);
    };

    let (batch_size, one, channels) = in_b1c.dims3()?;
    if one != 1 || channels != heads * head_size {
        return Ok(None);
    }
    if r_b1c.dims() != [batch_size, 1, channels]
        || k_b1c.dims() != [batch_size, 1, channels]
        || v_b1c.dims() != [batch_size, 1, channels]
        || w_b1c.dims() != [batch_size, 1, channels]
        || a_b1c.dims() != [batch_size, 1, channels]
        || k_scale_b1h.dims() != [batch_size, 1, heads]
        || v_scale_b1h.dims() != [batch_size, 1, heads]
        || state_b1hkk.dims() != [batch_size, 1, heads, head_size, head_size]
        || weights.bonus.dims() != [1, 1, heads, head_size]
    {
        return Ok(None);
    }
    let cached_weights = time_mixer_native_f32_weights(weights)?;
    let w_o = &cached_weights.w_o;
    if w_o.out_dim != channels || w_o.in_dim != channels || w_o.bias.is_some() {
        return Ok(None);
    }

    let middle = time_mixer_middle_scratch_profiled(
        r_b1c,
        k_b1c,
        v_b1c,
        w_b1c,
        a_b1c,
        k_scale_b1h,
        v_scale_b1h,
        state_b1hkk,
        heads,
        head_size,
        return_state,
        materialize_out_tensor,
        profile,
    )?;
    Ok(Some(TimeMixerMiddleOutputScratch { middle, dot_kernel }))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn time_mixer_middle_output_scratch_native_from_lora_values(
    in_b1c: &Tensor,
    r_b1c: &Tensor,
    k_b1c: &Tensor,
    v_b1c: &Tensor,
    lora_values: &TimeMixerLoraDecayScratchValues,
    k_scale_b1h: &Tensor,
    v_scale_b1h: &Tensor,
    state_b1hkk: &Tensor,
    weights: &Rwkv7RnnTimeMixerWeights,
    heads: usize,
    head_size: usize,
    return_state: bool,
    materialize_out_tensor: bool,
    profile: Option<&mut crate::profile::SingleTimestepProfile>,
) -> Result<Option<TimeMixerMiddleOutputScratch>> {
    let Some(dot_kernel) = native_linear_dot_kernel() else {
        return Ok(None);
    };

    let (batch_size, one, channels) = in_b1c.dims3()?;
    if one != 1
        || channels != heads * head_size
        || lora_values.batch_size != batch_size
        || lora_values.channels != channels
        || lora_values.a_values.len() != batch_size * channels
        || lora_values.w_values.len() != batch_size * channels
    {
        return Ok(None);
    }
    if r_b1c.dims() != [batch_size, 1, channels]
        || k_b1c.dims() != [batch_size, 1, channels]
        || v_b1c.dims() != [batch_size, 1, channels]
        || k_scale_b1h.dims() != [batch_size, 1, heads]
        || v_scale_b1h.dims() != [batch_size, 1, heads]
        || state_b1hkk.dims() != [batch_size, 1, heads, head_size, head_size]
        || weights.bonus.dims() != [1, 1, heads, head_size]
    {
        return Ok(None);
    }
    let cached_weights = time_mixer_native_f32_weights(weights)?;
    let w_o = &cached_weights.w_o;
    if w_o.out_dim != channels || w_o.in_dim != channels || w_o.bias.is_some() {
        return Ok(None);
    }

    let middle = time_mixer_middle_scratch_values_profiled(
        r_b1c,
        k_b1c,
        v_b1c,
        &lora_values.w_values,
        &lora_values.a_values,
        k_scale_b1h,
        v_scale_b1h,
        state_b1hkk,
        heads,
        head_size,
        return_state,
        materialize_out_tensor,
        profile,
    )?;
    Ok(Some(TimeMixerMiddleOutputScratch { middle, dot_kernel }))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn time_mixer_middle_output_scratch_native_from_lora_and_v_values(
    in_b1c: &Tensor,
    r_b1c: &Tensor,
    k_b1c: &Tensor,
    v_mix_values: &TimeMixerVMixScratchValues,
    lora_values: &TimeMixerLoraDecayScratchValues,
    k_scale_b1h: &Tensor,
    v_scale_b1h: &Tensor,
    state_b1hkk: &Tensor,
    weights: &Rwkv7RnnTimeMixerWeights,
    heads: usize,
    head_size: usize,
    return_state: bool,
    materialize_out_tensor: bool,
    profile: Option<&mut crate::profile::SingleTimestepProfile>,
) -> Result<Option<TimeMixerMiddleOutputScratch>> {
    let Some(dot_kernel) = native_linear_dot_kernel() else {
        return Ok(None);
    };

    let (batch_size, one, channels) = in_b1c.dims3()?;
    if one != 1
        || channels != heads * head_size
        || v_mix_values.batch_size != batch_size
        || v_mix_values.channels != channels
        || v_mix_values.values.len() != batch_size * channels
        || lora_values.batch_size != batch_size
        || lora_values.channels != channels
        || lora_values.a_values.len() != batch_size * channels
        || lora_values.w_values.len() != batch_size * channels
    {
        return Ok(None);
    }
    if r_b1c.dims() != [batch_size, 1, channels]
        || k_b1c.dims() != [batch_size, 1, channels]
        || k_scale_b1h.dims() != [batch_size, 1, heads]
        || v_scale_b1h.dims() != [batch_size, 1, heads]
        || state_b1hkk.dims() != [batch_size, 1, heads, head_size, head_size]
        || weights.bonus.dims() != [1, 1, heads, head_size]
    {
        return Ok(None);
    }
    let cached_weights = time_mixer_native_f32_weights(weights)?;
    let w_o = &cached_weights.w_o;
    if w_o.out_dim != channels || w_o.in_dim != channels || w_o.bias.is_some() {
        return Ok(None);
    }

    let middle = time_mixer_middle_scratch_wav_values_profiled(
        r_b1c,
        k_b1c,
        &v_mix_values.values,
        &lora_values.w_values,
        &lora_values.a_values,
        k_scale_b1h,
        v_scale_b1h,
        state_b1hkk,
        heads,
        head_size,
        return_state,
        materialize_out_tensor,
        profile,
    )?;
    Ok(Some(TimeMixerMiddleOutputScratch { middle, dot_kernel }))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn time_mixer_middle_output_scratch_native_from_projection_and_lora_values(
    in_b1c: &Tensor,
    projection_values: &TimeMixerLerpProjectionScratchValues,
    v_mix_values: &TimeMixerVMixScratchValues,
    lora_values: &TimeMixerLoraDecayScratchValues,
    state_b1hkk: &Tensor,
    weights: &Rwkv7RnnTimeMixerWeights,
    heads: usize,
    head_size: usize,
    return_state: bool,
    materialize_out_tensor: bool,
    profile: Option<&mut crate::profile::SingleTimestepProfile>,
) -> Result<Option<TimeMixerMiddleOutputScratch>> {
    let Some(dot_kernel) = native_linear_dot_kernel() else {
        return Ok(None);
    };

    let (batch_size, one, channels) = in_b1c.dims3()?;
    if one != 1
        || channels != heads * head_size
        || projection_values.batch_size != batch_size
        || projection_values.channels != channels
        || projection_values.heads != heads
        || projection_values.r_values.len() != batch_size * channels
        || projection_values.k_values.len() != batch_size * channels
        || projection_values.k_scale_values.len() != batch_size * heads
        || projection_values.v_scale_values.len() != batch_size * heads
        || v_mix_values.batch_size != batch_size
        || v_mix_values.channels != channels
        || v_mix_values.values.len() != batch_size * channels
        || lora_values.batch_size != batch_size
        || lora_values.channels != channels
        || lora_values.a_values.len() != batch_size * channels
        || lora_values.w_values.len() != batch_size * channels
    {
        return Ok(None);
    }
    if state_b1hkk.dims() != [batch_size, 1, heads, head_size, head_size]
        || weights.bonus.dims() != [1, 1, heads, head_size]
    {
        return Ok(None);
    }
    let cached_weights = time_mixer_native_f32_weights(weights)?;
    let w_o = &cached_weights.w_o;
    if w_o.out_dim != channels || w_o.in_dim != channels || w_o.bias.is_some() {
        return Ok(None);
    }

    let middle = time_mixer_middle_scratch_all_values_profiled(
        &projection_values.r_values,
        &projection_values.k_values,
        &v_mix_values.values,
        &lora_values.w_values,
        &lora_values.a_values,
        &projection_values.k_scale_values,
        &projection_values.v_scale_values,
        state_b1hkk,
        batch_size,
        heads,
        head_size,
        return_state,
        materialize_out_tensor,
        profile,
    )?;
    Ok(Some(TimeMixerMiddleOutputScratch { middle, dot_kernel }))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn time_mixer_output_native(
    in_b1c: &Tensor,
    r_b1hk: &Tensor,
    k_b1hk: &Tensor,
    v_b1hk: &Tensor,
    bonus: &Tensor,
    out_b1c: &Tensor,
    g_b1c: &Tensor,
    weights: &Rwkv7RnnTimeMixerWeights,
) -> Result<Option<Tensor>> {
    let Some(dot_kernel) = native_linear_dot_kernel() else {
        return Ok(None);
    };
    time_mixer_output_native_with_kernel(
        in_b1c, r_b1hk, k_b1hk, v_b1hk, bonus, out_b1c, g_b1c, weights, dot_kernel,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn time_mixer_output_native_with_kernel(
    in_b1c: &Tensor,
    r_b1hk: &Tensor,
    k_b1hk: &Tensor,
    v_b1hk: &Tensor,
    bonus: &Tensor,
    out_b1c: &Tensor,
    g_b1c: &Tensor,
    weights: &Rwkv7RnnTimeMixerWeights,
    dot_kernel: NativeLinearDotKernel,
) -> Result<Option<Tensor>> {
    let (batch_size, one, channels) = in_b1c.dims3()?;
    if one != 1 {
        return Ok(None);
    }
    if out_b1c.dims() != [batch_size, 1, channels] || g_b1c.dims() != [batch_size, 1, channels] {
        return Ok(None);
    }
    let (r_batch_size, r_one, heads, head_size) = r_b1hk.dims4()?;
    if r_batch_size != batch_size || r_one != 1 || channels != heads * head_size {
        return Ok(None);
    }
    if k_b1hk.dims() != [batch_size, 1, heads, head_size]
        || v_b1hk.dims() != [batch_size, 1, heads, head_size]
        || bonus.dims() != [1, 1, heads, head_size]
    {
        return Ok(None);
    }
    let cached_weights = time_mixer_native_f32_weights(weights)?;
    let w_o = &cached_weights.w_o;
    if w_o.out_dim != channels || w_o.in_dim != channels || w_o.bias.is_some() {
        return Ok(None);
    }

    let in_data = f32_tensor_data(in_b1c)?;
    let r_data = f32_tensor_data(r_b1hk)?;
    let k_data = f32_tensor_data(k_b1hk)?;
    let v_data = f32_tensor_data(v_b1hk)?;
    let bonus_data = f32_tensor_data(bonus)?;
    let out_data = f32_tensor_data(out_b1c)?;
    let gate_data = f32_tensor_data(g_b1c)?;
    let in_values = in_data.as_slice()?;
    let r_values = r_data.as_slice()?;
    let k_values = k_data.as_slice()?;
    let v_values = v_data.as_slice()?;
    let bonus_values = bonus_data.as_slice()?;
    let out_values = out_data.as_slice()?;
    let gate_values = gate_data.as_slice()?;
    let weight_values = w_o.weight.as_slice();

    let bxc_len = batch_size * channels;
    let bxhk_len = batch_size * heads * head_size;
    if in_values.len() != bxc_len
        || out_values.len() != bxc_len
        || gate_values.len() != bxc_len
        || r_values.len() != bxhk_len
        || k_values.len() != bxhk_len
        || v_values.len() != bxhk_len
        || bonus_values.len() != heads * head_size
        || weight_values.len() != channels * channels
    {
        return Ok(None);
    }

    let mut output_values = vec![0.0f32; bxc_len];
    let mut hidden_values = vec![0.0f32; channels];
    let mut head_bonus = vec![0.0f32; heads];
    for batch_index in 0..batch_size {
        for (head_index, head_bonus_value) in head_bonus.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for head_offset in 0..head_size {
                let channel_index = head_index * head_size + head_offset;
                let value_index = batch_index * channels + channel_index;
                sum += r_values[value_index] * bonus_values[channel_index] * k_values[value_index];
            }
            *head_bonus_value = sum;
        }

        let batch_base = batch_index * channels;
        for head_index in 0..heads {
            let head_bonus_value = head_bonus[head_index];
            for head_offset in 0..head_size {
                let channel_index = head_index * head_size + head_offset;
                let value_index = batch_base + channel_index;
                let bonus_value = head_bonus_value * v_values[value_index];
                hidden_values[channel_index] =
                    gate_values[value_index] * (out_values[value_index] + bonus_value);
            }
        }

        let output_row = &mut output_values[batch_base..batch_base + channels];
        linear_project_row_same_x(
            dot_kernel,
            &hidden_values,
            weight_values,
            channels,
            channels,
            output_row,
        );
        for out_index in 0..channels {
            output_row[out_index] = in_values[batch_base + out_index] + output_row[out_index];
        }
    }

    Tensor::from_vec(output_values, (batch_size, channels), in_b1c.device()).map(Some)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn time_mixer_output_native_from_middle_scratch(
    in_b1c: &Tensor,
    r_b1c: &Tensor,
    bonus: &Tensor,
    out_b1c: &Tensor,
    g_b1c: &Tensor,
    weights: &Rwkv7RnnTimeMixerWeights,
    scratch: &TimeMixerMiddleOutputScratch,
) -> Result<Option<Tensor>> {
    let (batch_size, one, channels) = in_b1c.dims3()?;
    if one != 1 {
        return Ok(None);
    }
    if r_b1c.dims() != [batch_size, 1, channels]
        || out_b1c.dims() != [batch_size, 1, channels]
        || g_b1c.dims() != [batch_size, 1, channels]
    {
        return Ok(None);
    }
    let heads = weights.n_heads;
    if heads == 0 || channels % heads != 0 {
        return Ok(None);
    }
    let head_size = channels / heads;
    if bonus.dims() != [1, 1, heads, head_size]
        || scratch.middle.k_values.len() != batch_size * channels
        || scratch.middle.v_values.len() != batch_size * channels
    {
        return Ok(None);
    }
    let cached_weights = time_mixer_native_f32_weights(weights)?;
    let w_o = &cached_weights.w_o;
    if w_o.out_dim != channels || w_o.in_dim != channels || w_o.bias.is_some() {
        return Ok(None);
    }

    let in_data = f32_tensor_data(in_b1c)?;
    let r_data = f32_tensor_data(r_b1c)?;
    let bonus_data = f32_tensor_data(bonus)?;
    let out_data = f32_tensor_data(out_b1c)?;
    let gate_data = f32_tensor_data(g_b1c)?;
    let in_values = in_data.as_slice()?;
    let r_values = r_data.as_slice()?;
    let bonus_values = bonus_data.as_slice()?;
    let out_values = out_data.as_slice()?;
    let gate_values = gate_data.as_slice()?;
    let k_values = scratch.middle.k_values.as_slice();
    let v_values = scratch.middle.v_values.as_slice();
    let weight_values = w_o.weight.as_slice();

    let bxc_len = batch_size * channels;
    if in_values.len() != bxc_len
        || r_values.len() != bxc_len
        || out_values.len() != bxc_len
        || gate_values.len() != bxc_len
        || bonus_values.len() != channels
        || weight_values.len() != channels * channels
    {
        return Ok(None);
    }

    let mut output_values = vec![0.0f32; bxc_len];
    let mut hidden_values = vec![0.0f32; channels];
    let mut head_bonus = vec![0.0f32; heads];
    for batch_index in 0..batch_size {
        let batch_base = batch_index * channels;
        for (head_index, head_bonus_value) in head_bonus.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for head_offset in 0..head_size {
                let channel_index = head_index * head_size + head_offset;
                let value_index = batch_base + channel_index;
                sum += r_values[value_index] * bonus_values[channel_index] * k_values[value_index];
            }
            *head_bonus_value = sum;
        }

        for (head_index, head_bonus_value) in head_bonus.iter().copied().enumerate() {
            for head_offset in 0..head_size {
                let channel_index = head_index * head_size + head_offset;
                let value_index = batch_base + channel_index;
                let bonus_value = head_bonus_value * v_values[value_index];
                hidden_values[channel_index] =
                    gate_values[value_index] * (out_values[value_index] + bonus_value);
            }
        }

        let output_row = &mut output_values[batch_base..batch_base + channels];
        linear_project_row_same_x(
            scratch.dot_kernel,
            &hidden_values,
            weight_values,
            channels,
            channels,
            output_row,
        );
        for out_index in 0..channels {
            output_row[out_index] = in_values[batch_base + out_index] + output_row[out_index];
        }
    }

    Tensor::from_vec(output_values, (batch_size, channels), in_b1c.device()).map(Some)
}

pub(super) fn time_mixer_group_output_mean(
    scratch: &TimeMixerMiddleOutputScratch,
    batch_index: usize,
    head_index: usize,
    heads: usize,
    head_size: usize,
    group: &[f32],
) -> f32 {
    if let Some(out_sum_values) = scratch.middle.out_sum_values.as_ref() {
        let sum_index = batch_index * heads + head_index;
        if out_sum_values.len() >= sum_index + 1 {
            return out_sum_values[sum_index] / head_size as f32;
        }
    }
    group.iter().copied().sum::<f32>() / head_size as f32
}

pub(super) fn time_mixer_group_output_variance(
    scratch: &TimeMixerMiddleOutputScratch,
    batch_index: usize,
    head_index: usize,
    heads: usize,
    head_size: usize,
    group: &[f32],
    mean: f32,
) -> f32 {
    if let Some(out_variance_values) = scratch.middle.out_variance_values.as_ref() {
        let variance_index = batch_index * heads + head_index;
        if out_variance_values.len() >= variance_index + 1 {
            return out_variance_values[variance_index];
        }
    }
    group
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<f32>()
        / head_size as f32
}

pub(super) fn time_mixer_group_output_mean_variance(
    scratch: &TimeMixerMiddleOutputScratch,
    batch_index: usize,
    head_index: usize,
    heads: usize,
    head_size: usize,
    group: &[f32],
) -> (f32, f32) {
    if native_time_mixer_output_one_pass_group_norm_enabled() {
        let mut sum = 0.0f32;
        let mut square_sum = 0.0f32;
        for value in group {
            sum += *value;
            square_sum += *value * *value;
        }
        let mean = sum / head_size as f32;
        let variance = square_sum / head_size as f32 - mean * mean;
        return (mean, variance);
    }

    let mean =
        time_mixer_group_output_mean(scratch, batch_index, head_index, heads, head_size, group);
    let variance = time_mixer_group_output_variance(
        scratch,
        batch_index,
        head_index,
        heads,
        head_size,
        group,
        mean,
    );
    (mean, variance)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn time_mixer_group_output_scratch_native(
    in_b1c: &Tensor,
    r_b1c: &Tensor,
    bonus: &Tensor,
    g_b1c: &Tensor,
    weights: &Rwkv7RnnTimeMixerWeights,
    scratch: &TimeMixerMiddleOutputScratch,
) -> Result<Option<Tensor>> {
    let (batch_size, one, channels) = in_b1c.dims3()?;
    if one != 1 {
        return Ok(None);
    }
    if r_b1c.dims() != [batch_size, 1, channels] || g_b1c.dims() != [batch_size, 1, channels] {
        return Ok(None);
    }
    let heads = weights.n_heads;
    if heads == 0 || channels % heads != 0 {
        return Ok(None);
    }
    let head_size = channels / heads;
    if bonus.dims() != [1, 1, heads, head_size]
        || scratch.middle.out_values.len() != batch_size * channels
        || scratch.middle.k_values.len() != batch_size * channels
        || scratch.middle.v_values.len() != batch_size * channels
        || weights.out_group_norm.weight.dims() != [channels]
        || weights.out_group_norm.bias.dims() != [channels]
    {
        return Ok(None);
    }
    let cached_weights = time_mixer_native_f32_weights(weights)?;
    let w_o = &cached_weights.w_o;
    if w_o.out_dim != channels || w_o.in_dim != channels || w_o.bias.is_some() {
        return Ok(None);
    }

    let in_data = f32_tensor_data(in_b1c)?;
    let r_data = f32_tensor_data(r_b1c)?;
    let bonus_data = f32_tensor_data(bonus)?;
    let gate_data = f32_tensor_data(g_b1c)?;
    let group_weight_data = f32_tensor_data(&weights.out_group_norm.weight)?;
    let group_bias_data = f32_tensor_data(&weights.out_group_norm.bias)?;
    let in_values = in_data.as_slice()?;
    let r_values = r_data.as_slice()?;
    let bonus_values = bonus_data.as_slice()?;
    let gate_values = gate_data.as_slice()?;
    let group_weight_values = group_weight_data.as_slice()?;
    let group_bias_values = group_bias_data.as_slice()?;
    let out_values = scratch.middle.out_values.as_slice();
    let k_values = scratch.middle.k_values.as_slice();
    let v_values = scratch.middle.v_values.as_slice();
    let weight_values = w_o.weight.as_slice();

    let bxc_len = batch_size * channels;
    if in_values.len() != bxc_len
        || r_values.len() != bxc_len
        || gate_values.len() != bxc_len
        || bonus_values.len() != channels
        || group_weight_values.len() != channels
        || group_bias_values.len() != channels
        || weight_values.len() != channels * channels
    {
        return Ok(None);
    }

    let eps = TIME_MIXER_GROUP_NORM_EPS as f32;
    if native_time_mixer_output_head32_stack_enabled()
        && channels == TIME_MIXER_OUTPUT_HEAD32_CHANNELS
        && heads == TIME_MIXER_OUTPUT_HEAD32_HEADS
        && head_size == TIME_MIXER_OUTPUT_HEAD32_HEAD_SIZE
    {
        let output_values = time_mixer_group_output_cached_aux_head32_stack_values(
            in_values,
            batch_size,
            r_values,
            gate_values,
            out_values,
            k_values,
            v_values,
            bonus_values,
            group_weight_values,
            group_bias_values,
            w_o,
            scratch,
            scratch.dot_kernel,
            false,
            false,
            eps,
        );
        return Tensor::from_vec(output_values, (batch_size, channels), in_b1c.device()).map(Some);
    }

    let mut output_values = vec![0.0f32; bxc_len];
    let mut hidden_values = vec![0.0f32; channels];
    let mut head_bonus = vec![0.0f32; heads];
    for batch_index in 0..batch_size {
        let batch_base = batch_index * channels;
        for (head_index, head_bonus_value) in head_bonus.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for head_offset in 0..head_size {
                let channel_index = head_index * head_size + head_offset;
                let value_index = batch_base + channel_index;
                sum += r_values[value_index] * bonus_values[channel_index] * k_values[value_index];
            }
            *head_bonus_value = sum;
        }

        for (head_index, head_bonus_value) in head_bonus.iter().copied().enumerate() {
            let group_base = batch_base + head_index * head_size;
            let group = &out_values[group_base..group_base + head_size];
            let (mean, variance) = time_mixer_group_output_mean_variance(
                scratch,
                batch_index,
                head_index,
                heads,
                head_size,
                group,
            );
            let inv_std = 1.0f32 / (variance + eps).sqrt();
            for head_offset in 0..head_size {
                let channel_index = head_index * head_size + head_offset;
                let value_index = batch_base + channel_index;
                let grouped =
                    (out_values[value_index] - mean) * inv_std * group_weight_values[channel_index]
                        + group_bias_values[channel_index];
                let bonus_value = head_bonus_value * v_values[value_index];
                hidden_values[channel_index] = gate_values[value_index] * (grouped + bonus_value);
            }
        }

        let output_row = &mut output_values[batch_base..batch_base + channels];
        linear_project_row_same_x(
            scratch.dot_kernel,
            &hidden_values,
            weight_values,
            channels,
            channels,
            output_row,
        );
        for out_index in 0..channels {
            output_row[out_index] = in_values[batch_base + out_index] + output_row[out_index];
        }
    }

    Tensor::from_vec(output_values, (batch_size, channels), in_b1c.device()).map(Some)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn time_mixer_group_output_scratch_native_from_lora_values(
    in_b1c: &Tensor,
    r_b1c: &Tensor,
    bonus: &Tensor,
    lora_values: &TimeMixerLoraDecayScratchValues,
    weights: &Rwkv7RnnTimeMixerWeights,
    scratch: &TimeMixerMiddleOutputScratch,
) -> Result<Option<Tensor>> {
    let (batch_size, one, channels) = in_b1c.dims3()?;
    if one != 1 {
        return Ok(None);
    }
    if r_b1c.dims() != [batch_size, 1, channels]
        || lora_values.batch_size != batch_size
        || lora_values.channels != channels
        || lora_values.g_values.len() != batch_size * channels
    {
        return Ok(None);
    }
    let heads = weights.n_heads;
    if heads == 0 || channels % heads != 0 {
        return Ok(None);
    }
    let head_size = channels / heads;
    if bonus.dims() != [1, 1, heads, head_size]
        || scratch.middle.out_values.len() != batch_size * channels
        || scratch.middle.k_values.len() != batch_size * channels
        || scratch.middle.v_values.len() != batch_size * channels
        || weights.out_group_norm.weight.dims() != [channels]
        || weights.out_group_norm.bias.dims() != [channels]
    {
        return Ok(None);
    }
    let cached_weights = time_mixer_native_f32_weights(weights)?;
    let w_o = &cached_weights.w_o;
    if w_o.out_dim != channels || w_o.in_dim != channels || w_o.bias.is_some() {
        return Ok(None);
    }

    let in_data = f32_tensor_data(in_b1c)?;
    let r_data = f32_tensor_data(r_b1c)?;
    let bonus_data = f32_tensor_data(bonus)?;
    let group_weight_data = f32_tensor_data(&weights.out_group_norm.weight)?;
    let group_bias_data = f32_tensor_data(&weights.out_group_norm.bias)?;
    let in_values = in_data.as_slice()?;
    let r_values = r_data.as_slice()?;
    let bonus_values = bonus_data.as_slice()?;
    let group_weight_values = group_weight_data.as_slice()?;
    let group_bias_values = group_bias_data.as_slice()?;
    let gate_values = lora_values.g_values.as_slice();
    let out_values = scratch.middle.out_values.as_slice();
    let k_values = scratch.middle.k_values.as_slice();
    let v_values = scratch.middle.v_values.as_slice();
    let weight_values = w_o.weight.as_slice();

    let bxc_len = batch_size * channels;
    if in_values.len() != bxc_len
        || r_values.len() != bxc_len
        || gate_values.len() != bxc_len
        || bonus_values.len() != channels
        || group_weight_values.len() != channels
        || group_bias_values.len() != channels
        || weight_values.len() != channels * channels
    {
        return Ok(None);
    }

    let eps = TIME_MIXER_GROUP_NORM_EPS as f32;
    if native_time_mixer_output_head32_stack_enabled()
        && channels == TIME_MIXER_OUTPUT_HEAD32_CHANNELS
        && heads == TIME_MIXER_OUTPUT_HEAD32_HEADS
        && head_size == TIME_MIXER_OUTPUT_HEAD32_HEAD_SIZE
    {
        let output_values = time_mixer_group_output_cached_aux_head32_stack_values(
            in_values,
            batch_size,
            r_values,
            gate_values,
            out_values,
            k_values,
            v_values,
            bonus_values,
            group_weight_values,
            group_bias_values,
            w_o,
            scratch,
            scratch.dot_kernel,
            false,
            false,
            eps,
        );
        return Tensor::from_vec(output_values, (batch_size, channels), in_b1c.device()).map(Some);
    }

    let mut output_values = vec![0.0f32; bxc_len];
    let mut hidden_values = vec![0.0f32; channels];
    let mut head_bonus = vec![0.0f32; heads];
    for batch_index in 0..batch_size {
        let batch_base = batch_index * channels;
        for (head_index, head_bonus_value) in head_bonus.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for head_offset in 0..head_size {
                let channel_index = head_index * head_size + head_offset;
                let value_index = batch_base + channel_index;
                sum += r_values[value_index] * bonus_values[channel_index] * k_values[value_index];
            }
            *head_bonus_value = sum;
        }

        for (head_index, head_bonus_value) in head_bonus.iter().copied().enumerate() {
            let group_base = batch_base + head_index * head_size;
            let group = &out_values[group_base..group_base + head_size];
            let (mean, variance) = time_mixer_group_output_mean_variance(
                scratch,
                batch_index,
                head_index,
                heads,
                head_size,
                group,
            );
            let inv_std = 1.0f32 / (variance + eps).sqrt();
            for head_offset in 0..head_size {
                let channel_index = head_index * head_size + head_offset;
                let value_index = batch_base + channel_index;
                let grouped =
                    (out_values[value_index] - mean) * inv_std * group_weight_values[channel_index]
                        + group_bias_values[channel_index];
                let bonus_value = head_bonus_value * v_values[value_index];
                hidden_values[channel_index] = gate_values[value_index] * (grouped + bonus_value);
            }
        }

        let output_row = &mut output_values[batch_base..batch_base + channels];
        linear_project_row_same_x(
            scratch.dot_kernel,
            &hidden_values,
            weight_values,
            channels,
            channels,
            output_row,
        );
        for out_index in 0..channels {
            output_row[out_index] = in_values[batch_base + out_index] + output_row[out_index];
        }
    }

    Tensor::from_vec(output_values, (batch_size, channels), in_b1c.device()).map(Some)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn time_mixer_group_output_scratch_native_from_lora_and_r_values(
    in_b1c: &Tensor,
    r_values: &[f32],
    bonus: &Tensor,
    lora_values: &TimeMixerLoraDecayScratchValues,
    weights: &Rwkv7RnnTimeMixerWeights,
    scratch: &TimeMixerMiddleOutputScratch,
) -> Result<Option<Tensor>> {
    let (batch_size, one, channels) = in_b1c.dims3()?;
    if one != 1
        || r_values.len() != batch_size * channels
        || lora_values.batch_size != batch_size
        || lora_values.channels != channels
        || lora_values.g_values.len() != batch_size * channels
    {
        return Ok(None);
    }
    let heads = weights.n_heads;
    if heads == 0 || channels % heads != 0 {
        return Ok(None);
    }
    let head_size = channels / heads;
    if bonus.dims() != [1, 1, heads, head_size]
        || scratch.middle.out_values.len() != batch_size * channels
        || scratch.middle.k_values.len() != batch_size * channels
        || scratch.middle.v_values.len() != batch_size * channels
        || weights.out_group_norm.weight.dims() != [channels]
        || weights.out_group_norm.bias.dims() != [channels]
    {
        return Ok(None);
    }
    let cached_weights = time_mixer_native_f32_weights(weights)?;
    let w_o = &cached_weights.w_o;
    if w_o.out_dim != channels || w_o.in_dim != channels || w_o.bias.is_some() {
        return Ok(None);
    }

    let in_data = f32_tensor_data(in_b1c)?;
    let bonus_data = f32_tensor_data(bonus)?;
    let group_weight_data = f32_tensor_data(&weights.out_group_norm.weight)?;
    let group_bias_data = f32_tensor_data(&weights.out_group_norm.bias)?;
    let in_values = in_data.as_slice()?;
    let bonus_values = bonus_data.as_slice()?;
    let group_weight_values = group_weight_data.as_slice()?;
    let group_bias_values = group_bias_data.as_slice()?;
    let gate_values = lora_values.g_values.as_slice();
    let out_values = scratch.middle.out_values.as_slice();
    let k_values = scratch.middle.k_values.as_slice();
    let v_values = scratch.middle.v_values.as_slice();
    let weight_values = w_o.weight.as_slice();

    let bxc_len = batch_size * channels;
    if in_values.len() != bxc_len
        || gate_values.len() != bxc_len
        || bonus_values.len() != channels
        || group_weight_values.len() != channels
        || group_bias_values.len() != channels
        || weight_values.len() != channels * channels
    {
        return Ok(None);
    }

    let eps = TIME_MIXER_GROUP_NORM_EPS as f32;
    let mut output_values = vec![0.0f32; bxc_len];
    let mut hidden_values = vec![0.0f32; channels];
    let mut head_bonus = vec![0.0f32; heads];
    for batch_index in 0..batch_size {
        let batch_base = batch_index * channels;
        for (head_index, head_bonus_value) in head_bonus.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for head_offset in 0..head_size {
                let channel_index = head_index * head_size + head_offset;
                let value_index = batch_base + channel_index;
                sum += r_values[value_index] * bonus_values[channel_index] * k_values[value_index];
            }
            *head_bonus_value = sum;
        }

        for (head_index, head_bonus_value) in head_bonus.iter().copied().enumerate() {
            let group_base = batch_base + head_index * head_size;
            let group = &out_values[group_base..group_base + head_size];
            let (mean, variance) = time_mixer_group_output_mean_variance(
                scratch,
                batch_index,
                head_index,
                heads,
                head_size,
                group,
            );
            let inv_std = 1.0f32 / (variance + eps).sqrt();
            for head_offset in 0..head_size {
                let channel_index = head_index * head_size + head_offset;
                let value_index = batch_base + channel_index;
                let grouped =
                    (out_values[value_index] - mean) * inv_std * group_weight_values[channel_index]
                        + group_bias_values[channel_index];
                let bonus_value = head_bonus_value * v_values[value_index];
                hidden_values[channel_index] = gate_values[value_index] * (grouped + bonus_value);
            }
        }

        let output_row = &mut output_values[batch_base..batch_base + channels];
        linear_project_row_same_x(
            scratch.dot_kernel,
            &hidden_values,
            weight_values,
            channels,
            channels,
            output_row,
        );
        for out_index in 0..channels {
            output_row[out_index] = in_values[batch_base + out_index] + output_row[out_index];
        }
    }

    Tensor::from_vec(output_values, (batch_size, channels), in_b1c.device()).map(Some)
}

pub(super) fn time_mixer_group_output_scratch_native_from_cached_aux_values(
    in_bc: &Tensor,
    r_values: &[f32],
    lora_values: &TimeMixerLoraDecayScratchValues,
    weights: &Rwkv7RnnTimeMixerWeights,
    scratch: &TimeMixerMiddleOutputScratch,
) -> Result<Option<Tensor>> {
    let (batch_size, channels) = in_bc.dims2()?;
    if r_values.len() != batch_size * channels
        || lora_values.batch_size != batch_size
        || lora_values.channels != channels
        || lora_values.g_values.len() != batch_size * channels
    {
        return Ok(None);
    }
    let heads = weights.n_heads;
    if heads == 0 || channels % heads != 0 {
        return Ok(None);
    }
    let head_size = channels / heads;
    if weights.bonus.dims() != [1, 1, heads, head_size]
        || scratch.middle.out_values.len() != batch_size * channels
        || scratch.middle.k_values.len() != batch_size * channels
        || scratch.middle.v_values.len() != batch_size * channels
        || weights.out_group_norm.weight.dims() != [channels]
        || weights.out_group_norm.bias.dims() != [channels]
    {
        return Ok(None);
    }
    let cached_weights = time_mixer_native_f32_weights(weights)?;
    let w_o = &cached_weights.w_o;
    if w_o.out_dim != channels || w_o.in_dim != channels || w_o.bias.is_some() {
        return Ok(None);
    }

    let in_data = f32_tensor_data(in_bc)?;
    let in_values = in_data.as_slice()?;
    let bonus_values = cached_weights.bonus.as_slice();
    let group_weight_values = cached_weights.out_group_norm_weight.as_slice();
    let group_bias_values = cached_weights.out_group_norm_bias.as_slice();
    let gate_values = lora_values.g_values.as_slice();
    let out_values = scratch.middle.out_values.as_slice();
    let k_values = scratch.middle.k_values.as_slice();
    let v_values = scratch.middle.v_values.as_slice();
    let weight_values = w_o.weight.as_slice();

    let bxc_len = batch_size * channels;
    if in_values.len() != bxc_len
        || gate_values.len() != bxc_len
        || bonus_values.len() != channels
        || group_weight_values.len() != channels
        || group_bias_values.len() != channels
        || weight_values.len() != channels * channels
    {
        return Ok(None);
    }

    let eps = TIME_MIXER_GROUP_NORM_EPS as f32;
    if native_time_mixer_output_head32_stack_enabled()
        && channels == TIME_MIXER_OUTPUT_HEAD32_CHANNELS
        && heads == TIME_MIXER_OUTPUT_HEAD32_HEADS
        && head_size == TIME_MIXER_OUTPUT_HEAD32_HEAD_SIZE
    {
        let output_values = time_mixer_group_output_cached_aux_head32_stack_values(
            in_values,
            batch_size,
            r_values,
            gate_values,
            out_values,
            k_values,
            v_values,
            bonus_values,
            group_weight_values,
            group_bias_values,
            w_o,
            scratch,
            scratch.dot_kernel,
            false,
            false,
            eps,
        );
        return Tensor::from_vec(output_values, (batch_size, channels), in_bc.device()).map(Some);
    }

    let mut output_values = vec![0.0f32; bxc_len];
    let mut hidden_values = vec![0.0f32; channels];
    let mut head_bonus = vec![0.0f32; heads];
    for batch_index in 0..batch_size {
        let batch_base = batch_index * channels;
        for (head_index, head_bonus_value) in head_bonus.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for head_offset in 0..head_size {
                let channel_index = head_index * head_size + head_offset;
                let value_index = batch_base + channel_index;
                sum += r_values[value_index] * bonus_values[channel_index] * k_values[value_index];
            }
            *head_bonus_value = sum;
        }

        for (head_index, head_bonus_value) in head_bonus.iter().copied().enumerate() {
            let group_base = batch_base + head_index * head_size;
            let group = &out_values[group_base..group_base + head_size];
            let (mean, variance) = time_mixer_group_output_mean_variance(
                scratch,
                batch_index,
                head_index,
                heads,
                head_size,
                group,
            );
            let inv_std = 1.0f32 / (variance + eps).sqrt();
            for head_offset in 0..head_size {
                let channel_index = head_index * head_size + head_offset;
                let value_index = batch_base + channel_index;
                let grouped =
                    (out_values[value_index] - mean) * inv_std * group_weight_values[channel_index]
                        + group_bias_values[channel_index];
                let bonus_value = head_bonus_value * v_values[value_index];
                hidden_values[channel_index] = gate_values[value_index] * (grouped + bonus_value);
            }
        }

        let output_row = &mut output_values[batch_base..batch_base + channels];
        linear_project_row_same_x(
            scratch.dot_kernel,
            &hidden_values,
            weight_values,
            channels,
            channels,
            output_row,
        );
        for out_index in 0..channels {
            output_row[out_index] = in_values[batch_base + out_index] + output_row[out_index];
        }
    }

    Tensor::from_vec(output_values, (batch_size, channels), in_bc.device()).map(Some)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn time_mixer_group_output_scratch_native_from_cached_aux_input_values(
    in_values: &[f32],
    batch_size: usize,
    channels: usize,
    r_values: &[f32],
    lora_values: &TimeMixerLoraDecayScratchValues,
    weights: &Rwkv7RnnTimeMixerWeights,
    scratch: &TimeMixerMiddleOutputScratch,
) -> Result<Option<Vec<f32>>> {
    if r_values.len() != batch_size * channels
        || lora_values.batch_size != batch_size
        || lora_values.channels != channels
        || lora_values.g_values.len() != batch_size * channels
    {
        return Ok(None);
    }
    let heads = weights.n_heads;
    if heads == 0 || channels % heads != 0 {
        return Ok(None);
    }
    let head_size = channels / heads;
    if weights.bonus.dims() != [1, 1, heads, head_size]
        || scratch.middle.out_values.len() != batch_size * channels
        || scratch.middle.k_values.len() != batch_size * channels
        || scratch.middle.v_values.len() != batch_size * channels
        || weights.out_group_norm.weight.dims() != [channels]
        || weights.out_group_norm.bias.dims() != [channels]
    {
        return Ok(None);
    }
    let cached_weights = time_mixer_native_f32_weights(weights)?;
    let w_o = &cached_weights.w_o;
    if w_o.out_dim != channels || w_o.in_dim != channels || w_o.bias.is_some() {
        return Ok(None);
    }

    let bonus_values = cached_weights.bonus.as_slice();
    let group_weight_values = cached_weights.out_group_norm_weight.as_slice();
    let group_bias_values = cached_weights.out_group_norm_bias.as_slice();
    let gate_values = lora_values.g_values.as_slice();
    let out_values = scratch.middle.out_values.as_slice();
    let k_values = scratch.middle.k_values.as_slice();
    let v_values = scratch.middle.v_values.as_slice();
    let weight_values = w_o.weight.as_slice();

    let bxc_len = batch_size * channels;
    if in_values.len() != bxc_len
        || gate_values.len() != bxc_len
        || bonus_values.len() != channels
        || group_weight_values.len() != channels
        || group_bias_values.len() != channels
        || weight_values.len() != channels * channels
    {
        return Ok(None);
    }

    let batched_projection_min_rows = if native_time_mixer_output_batched_projection_min4_enabled()
    {
        4
    } else if native_time_mixer_output_batched_projection_min8_enabled() {
        8
    } else {
        16
    };
    let use_batched_projection = native_time_mixer_output_batched_projection_enabled()
        && batch_size >= batched_projection_min_rows;
    let use_output_buffer_reuse = native_time_mixer_output_buffer_reuse_enabled();
    let eps = TIME_MIXER_GROUP_NORM_EPS as f32;
    if native_time_mixer_output_head32_stack_enabled()
        && channels == TIME_MIXER_OUTPUT_HEAD32_CHANNELS
        && heads == TIME_MIXER_OUTPUT_HEAD32_HEADS
        && head_size == TIME_MIXER_OUTPUT_HEAD32_HEAD_SIZE
    {
        let output_values = time_mixer_group_output_cached_aux_head32_stack_values(
            in_values,
            batch_size,
            r_values,
            gate_values,
            out_values,
            k_values,
            v_values,
            bonus_values,
            group_weight_values,
            group_bias_values,
            w_o,
            scratch,
            scratch.dot_kernel,
            use_batched_projection,
            use_output_buffer_reuse,
            eps,
        );
        return Ok(Some(output_values));
    }

    let mut output_values = vec![0.0f32; bxc_len];
    let (mut hidden_values, mut hidden_batch_values) = if use_output_buffer_reuse {
        time_mixer_output_scratch_vectors(
            if use_batched_projection { 0 } else { channels },
            if use_batched_projection { bxc_len } else { 0 },
        )
    } else if use_batched_projection {
        (Vec::new(), vec![0.0f32; bxc_len])
    } else {
        (vec![0.0f32; channels], Vec::new())
    };
    let mut head_bonus = if use_output_buffer_reuse {
        Vec::new()
    } else {
        vec![0.0f32; heads]
    };
    for batch_index in 0..batch_size {
        let batch_base = batch_index * channels;
        {
            let hidden_row = if use_batched_projection {
                &mut hidden_batch_values[batch_base..batch_base + channels]
            } else {
                hidden_values.as_mut_slice()
            };
            if use_output_buffer_reuse {
                for head_index in 0..heads {
                    let mut head_bonus_value = 0.0f32;
                    for head_offset in 0..head_size {
                        let channel_index = head_index * head_size + head_offset;
                        let value_index = batch_base + channel_index;
                        head_bonus_value += r_values[value_index]
                            * bonus_values[channel_index]
                            * k_values[value_index];
                    }
                    let group_base = batch_base + head_index * head_size;
                    let group = &out_values[group_base..group_base + head_size];
                    let (mean, variance) = time_mixer_group_output_mean_variance(
                        scratch,
                        batch_index,
                        head_index,
                        heads,
                        head_size,
                        group,
                    );
                    let inv_std = 1.0f32 / (variance + eps).sqrt();
                    for head_offset in 0..head_size {
                        let channel_index = head_index * head_size + head_offset;
                        let value_index = batch_base + channel_index;
                        let grouped = (out_values[value_index] - mean)
                            * inv_std
                            * group_weight_values[channel_index]
                            + group_bias_values[channel_index];
                        let bonus_value = head_bonus_value * v_values[value_index];
                        hidden_row[channel_index] =
                            gate_values[value_index] * (grouped + bonus_value);
                    }
                }
            } else {
                for (head_index, head_bonus_value) in head_bonus.iter_mut().enumerate() {
                    let mut sum = 0.0f32;
                    for head_offset in 0..head_size {
                        let channel_index = head_index * head_size + head_offset;
                        let value_index = batch_base + channel_index;
                        sum += r_values[value_index]
                            * bonus_values[channel_index]
                            * k_values[value_index];
                    }
                    *head_bonus_value = sum;
                }

                for (head_index, head_bonus_value) in head_bonus.iter().copied().enumerate() {
                    let group_base = batch_base + head_index * head_size;
                    let group = &out_values[group_base..group_base + head_size];
                    let (mean, variance) = time_mixer_group_output_mean_variance(
                        scratch,
                        batch_index,
                        head_index,
                        heads,
                        head_size,
                        group,
                    );
                    let inv_std = 1.0f32 / (variance + eps).sqrt();
                    for head_offset in 0..head_size {
                        let channel_index = head_index * head_size + head_offset;
                        let value_index = batch_base + channel_index;
                        let grouped = (out_values[value_index] - mean)
                            * inv_std
                            * group_weight_values[channel_index]
                            + group_bias_values[channel_index];
                        let bonus_value = head_bonus_value * v_values[value_index];
                        hidden_row[channel_index] =
                            gate_values[value_index] * (grouped + bonus_value);
                    }
                }
            }
        }

        if !use_batched_projection {
            let output_row = &mut output_values[batch_base..batch_base + channels];
            linear_project_row_same_x(
                scratch.dot_kernel,
                &hidden_values,
                weight_values,
                channels,
                channels,
                output_row,
            );
            for out_index in 0..channels {
                output_row[out_index] = in_values[batch_base + out_index] + output_row[out_index];
            }
        }
    }

    if use_batched_projection {
        if !linear_project_batch_same_x(
            scratch.dot_kernel,
            &hidden_batch_values,
            batch_size,
            channels,
            weight_values,
            channels,
            &mut output_values,
        ) {
            for batch_index in 0..batch_size {
                let batch_base = batch_index * channels;
                let hidden_row = &hidden_batch_values[batch_base..batch_base + channels];
                let output_row = &mut output_values[batch_base..batch_base + channels];
                linear_project_row_same_x(
                    scratch.dot_kernel,
                    hidden_row,
                    weight_values,
                    channels,
                    channels,
                    output_row,
                );
            }
        }
        for value_index in 0..bxc_len {
            output_values[value_index] += in_values[value_index];
        }
    }

    if use_output_buffer_reuse {
        recycle_time_mixer_output_scratch(hidden_values, hidden_batch_values);
    }

    Ok(Some(output_values))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn time_mixer_group_output_cached_aux_head32_stack_values(
    in_values: &[f32],
    batch_size: usize,
    r_values: &[f32],
    gate_values: &[f32],
    out_values: &[f32],
    k_values: &[f32],
    v_values: &[f32],
    bonus_values: &[f32],
    group_weight_values: &[f32],
    group_bias_values: &[f32],
    w_o: &LinearF32Weights,
    scratch: &TimeMixerMiddleOutputScratch,
    dot_kernel: NativeLinearDotKernel,
    use_batched_projection: bool,
    use_output_buffer_reuse: bool,
    eps: f32,
) -> Vec<f32> {
    let channels = TIME_MIXER_OUTPUT_HEAD32_CHANNELS;
    let weight_values = w_o.weight.as_slice();
    let bxc_len = batch_size * channels;
    debug_assert_eq!(in_values.len(), bxc_len);
    debug_assert_eq!(r_values.len(), bxc_len);
    debug_assert_eq!(gate_values.len(), bxc_len);
    debug_assert_eq!(out_values.len(), bxc_len);
    debug_assert_eq!(k_values.len(), bxc_len);
    debug_assert_eq!(v_values.len(), bxc_len);
    debug_assert_eq!(bonus_values.len(), channels);
    debug_assert_eq!(group_weight_values.len(), channels);
    debug_assert_eq!(group_bias_values.len(), channels);
    debug_assert_eq!(weight_values.len(), channels * channels);

    let mut output_values = vec![0.0f32; bxc_len];
    let mut hidden_values = [0.0f32; TIME_MIXER_OUTPUT_HEAD32_CHANNELS];
    let (pooled_hidden_values, mut hidden_batch_values) =
        if use_batched_projection && use_output_buffer_reuse {
            time_mixer_output_scratch_vectors(0, bxc_len)
        } else if use_batched_projection {
            (Vec::new(), vec![0.0f32; bxc_len])
        } else {
            (Vec::new(), Vec::new())
        };

    for batch_index in 0..batch_size {
        let batch_base = batch_index * channels;
        let mut head_bonus = [0.0f32; TIME_MIXER_OUTPUT_HEAD32_HEADS];
        for (head_index, head_bonus_value) in head_bonus.iter_mut().enumerate() {
            let channel_base = head_index * TIME_MIXER_OUTPUT_HEAD32_HEAD_SIZE;
            let mut sum = 0.0f32;
            for head_offset in 0..TIME_MIXER_OUTPUT_HEAD32_HEAD_SIZE {
                let channel_index = channel_base + head_offset;
                let value_index = batch_base + channel_index;
                sum += r_values[value_index] * bonus_values[channel_index] * k_values[value_index];
            }
            *head_bonus_value = sum;
        }

        {
            let hidden_row = if use_batched_projection {
                &mut hidden_batch_values[batch_base..batch_base + channels]
            } else {
                &mut hidden_values[..]
            };
            for (head_index, head_bonus_value) in head_bonus.iter().copied().enumerate() {
                let channel_base = head_index * TIME_MIXER_OUTPUT_HEAD32_HEAD_SIZE;
                let group_base = batch_base + channel_base;
                let group =
                    &out_values[group_base..group_base + TIME_MIXER_OUTPUT_HEAD32_HEAD_SIZE];
                let (mean, variance) = time_mixer_group_output_mean_variance(
                    scratch,
                    batch_index,
                    head_index,
                    TIME_MIXER_OUTPUT_HEAD32_HEADS,
                    TIME_MIXER_OUTPUT_HEAD32_HEAD_SIZE,
                    group,
                );
                let inv_std = 1.0f32 / (variance + eps).sqrt();
                for head_offset in 0..TIME_MIXER_OUTPUT_HEAD32_HEAD_SIZE {
                    let channel_index = channel_base + head_offset;
                    let value_index = batch_base + channel_index;
                    let grouped = (out_values[value_index] - mean)
                        * inv_std
                        * group_weight_values[channel_index]
                        + group_bias_values[channel_index];
                    let bonus_value = head_bonus_value * v_values[value_index];
                    hidden_row[channel_index] = gate_values[value_index] * (grouped + bonus_value);
                }
            }
        }

        if !use_batched_projection {
            let output_row = &mut output_values[batch_base..batch_base + channels];
            if batch_size == 1 {
                linear_project_f32_row_same_x(dot_kernel, &hidden_values, w_o, output_row);
            } else {
                linear_project_row_same_x(
                    dot_kernel,
                    &hidden_values,
                    weight_values,
                    channels,
                    channels,
                    output_row,
                );
            }
            for out_index in 0..channels {
                output_row[out_index] = in_values[batch_base + out_index] + output_row[out_index];
            }
        }
    }

    if use_batched_projection {
        if !linear_project_batch_same_x(
            dot_kernel,
            &hidden_batch_values,
            batch_size,
            channels,
            weight_values,
            channels,
            &mut output_values,
        ) {
            for batch_index in 0..batch_size {
                let batch_base = batch_index * channels;
                let hidden_row = &hidden_batch_values[batch_base..batch_base + channels];
                let output_row = &mut output_values[batch_base..batch_base + channels];
                linear_project_row_same_x(
                    dot_kernel,
                    hidden_row,
                    weight_values,
                    channels,
                    channels,
                    output_row,
                );
            }
        }
        for value_index in 0..bxc_len {
            output_values[value_index] += in_values[value_index];
        }
    }

    if use_batched_projection && use_output_buffer_reuse {
        recycle_time_mixer_output_scratch(pooled_hidden_values, hidden_batch_values);
    }

    output_values
}
