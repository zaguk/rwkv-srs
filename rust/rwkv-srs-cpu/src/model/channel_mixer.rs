//! ChannelMixer forward paths, optimized projections, residuals, and scratch state.

use super::*;

#[derive(Clone, Debug)]
pub(crate) struct ChannelMixerFlatLayerState {
    pub(super) state_values: Vec<f32>,
    pub(super) channels: usize,
}
pub(super) struct ChannelMixerPreludeResidualScratchOutput {
    pub(super) out_bc: Tensor,
    pub(super) next_state_b1c: Tensor,
    pub(super) layer_norm_ns: u128,
    pub(super) lerp_ns: u128,
    pub(super) projection_ns: u128,
}
pub(super) struct ChannelMixerPreludeResidualFlatStateOutput {
    pub(super) output_values: Vec<f32>,
    pub(super) next_state_values: Option<Vec<f32>>,
    pub(super) layer_norm_ns: u128,
    pub(super) lerp_ns: u128,
    pub(super) projection_ns: u128,
}
#[derive(Default)]
struct ChannelMixerFlatScratchPool {
    x_values: Vec<f32>,
    xk_values: Vec<f32>,
    hidden_values: Vec<f32>,
}

thread_local! {
    static CHANNEL_MIXER_FLAT_SCRATCH_POOL: RefCell<ChannelMixerFlatScratchPool> =
        RefCell::new(ChannelMixerFlatScratchPool::default());
}

pub(super) fn take_channel_mixer_flat_scratch_vec(values: &mut Vec<f32>, len: usize) -> Vec<f32> {
    if !native_channel_mixer_flat_buffer_reuse_enabled() {
        return vec![0.0f32; len];
    }
    let mut scratch = std::mem::take(values);
    scratch.resize(len, 0.0);
    scratch
}

pub(super) fn channel_mixer_flat_scratch_vectors(
    bxc_len: usize,
    hidden_dim: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    CHANNEL_MIXER_FLAT_SCRATCH_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        (
            take_channel_mixer_flat_scratch_vec(&mut pool.x_values, bxc_len),
            take_channel_mixer_flat_scratch_vec(&mut pool.xk_values, bxc_len),
            take_channel_mixer_flat_scratch_vec(&mut pool.hidden_values, hidden_dim),
        )
    })
}

pub(super) fn recycle_channel_mixer_flat_scratch_vec(target: &mut Vec<f32>, mut values: Vec<f32>) {
    if !native_channel_mixer_flat_buffer_reuse_enabled() {
        return;
    }
    values.clear();
    if values.capacity() > target.capacity() {
        *target = values;
    }
}

pub(super) fn recycle_channel_mixer_flat_scratch(
    x_values: Option<Vec<f32>>,
    xk_values: Vec<f32>,
    hidden_values: Vec<f32>,
) {
    if !native_channel_mixer_flat_buffer_reuse_enabled() {
        return;
    }
    CHANNEL_MIXER_FLAT_SCRATCH_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if let Some(x_values) = x_values {
            recycle_channel_mixer_flat_scratch_vec(&mut pool.x_values, x_values);
        }
        recycle_channel_mixer_flat_scratch_vec(&mut pool.xk_values, xk_values);
        recycle_channel_mixer_flat_scratch_vec(&mut pool.hidden_values, hidden_values);
    });
}

pub fn channel_mixer_forward(
    weights: &Rwkv7RnnChannelMixerWeights,
    in_bc: &Tensor,
    state_b1c: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    channel_mixer_forward_profiled(weights, in_bc, state_b1c, None)
}

pub(super) fn channel_mixer_forward_profiled(
    weights: &Rwkv7RnnChannelMixerWeights,
    in_bc: &Tensor,
    state_b1c: Option<&Tensor>,
    mut profile: Option<&mut ForwardProfile>,
) -> Result<(Tensor, Tensor)> {
    let total_start = ProfileTimer::start(profile.is_some());
    profile_inc!(profile, channel_mixer.calls);

    let start = ProfileTimer::start(profile.is_some());
    let (batch_size, channels) = in_bc.dims2()?;
    validate_channel_mixer_shapes(weights, channels)?;
    profile_add!(profile, channel_mixer.validate_ns, start.elapsed_ns());

    if native_channel_mixer_projection_enabled()
        && native_channel_mixer_residual_scratch_enabled()
        && native_channel_mixer_prelude_residual_scratch_enabled()
    {
        if let Some(output) = channel_mixer_prelude_projection_residual_native(
            in_bc,
            state_b1c,
            weights,
            profile.is_some(),
        )? {
            if let Some(profile) = profile.as_deref_mut() {
                profile.channel_mixer.native_projection_calls += 1;
                profile.channel_mixer.native_projection_rows += batch_size;
            }
            profile_add!(profile, channel_mixer.layer_norm_ns, output.layer_norm_ns);
            profile_add!(profile, channel_mixer.lerp_ns, output.lerp_ns);
            profile_add!(profile, channel_mixer.projection_ns, output.projection_ns);
            profile_add!(profile, channel_mixer.total_ns, total_start.elapsed_ns());
            return Ok((output.out_bc, output.next_state_b1c));
        }
    }

    let start = ProfileTimer::start(profile.is_some());
    let in_b1c = in_bc.unsqueeze(1)?;
    let x_b1c = layer_norm(&in_b1c, &weights.layer_norm)?;
    profile_add!(profile, channel_mixer.layer_norm_ns, start.elapsed_ns());

    let start = ProfileTimer::start(profile.is_some());
    let x_shift_b1c = match state_b1c {
        Some(state) => {
            if state.dims() != [batch_size, 1, channels] {
                bail!(
                    "channel_mixer expected state shape [{batch_size}, 1, {channels}], got {:?}",
                    state.dims()
                );
            }
            state
        }
        None => &x_b1c,
    };
    profile_add!(profile, channel_mixer.state_ns, start.elapsed_ns());

    let start = ProfileTimer::start(profile.is_some());
    let xk_b1c = x_b1c.broadcast_add(
        &x_shift_b1c
            .broadcast_sub(&x_b1c)?
            .broadcast_mul(&weights.lerp_k)?,
    )?;
    profile_add!(profile, channel_mixer.lerp_ns, start.elapsed_ns());

    let start = ProfileTimer::start(profile.is_some());
    if native_channel_mixer_projection_enabled() && native_channel_mixer_residual_scratch_enabled()
    {
        if let Some((out_bc, _kernel)) =
            channel_mixer_projection_residual_native(in_bc, &xk_b1c, weights)?
        {
            if let Some(profile) = profile.as_deref_mut() {
                profile.channel_mixer.native_projection_calls += 1;
                profile.channel_mixer.native_projection_rows += batch_size;
            }
            profile_add!(profile, channel_mixer.projection_ns, start.elapsed_ns());
            profile_add!(profile, channel_mixer.total_ns, total_start.elapsed_ns());
            return Ok((out_bc, x_b1c));
        }
    }

    let o_b1c = if native_channel_mixer_projection_enabled() {
        if let Some((output, _kernel)) = channel_mixer_projection_native(&xk_b1c, weights)? {
            let rows = xk_b1c.dims3()?.0;
            if let Some(profile) = profile.as_deref_mut() {
                profile.channel_mixer.native_projection_calls += 1;
                profile.channel_mixer.native_projection_rows += rows;
            }
            output
        } else {
            let k_b1k = linear_profiled(&xk_b1c, &weights.w_k, &mut profile)?;
            linear_profiled(&k_b1k.relu()?.sqr()?, &weights.w_v, &mut profile)?
        }
    } else {
        let k_b1k = linear_profiled(&xk_b1c, &weights.w_k, &mut profile)?;
        linear_profiled(&k_b1k.relu()?.sqr()?, &weights.w_v, &mut profile)?
    };
    profile_add!(profile, channel_mixer.projection_ns, start.elapsed_ns());

    let start = ProfileTimer::start(profile.is_some());
    let out_bc = (in_b1c + o_b1c)?.squeeze(1)?;
    profile_add!(profile, channel_mixer.output_ns, start.elapsed_ns());
    profile_add!(profile, channel_mixer.total_ns, total_start.elapsed_ns());

    Ok((out_bc, x_b1c))
}

pub(super) fn channel_mixer_forward_flat_state_profiled(
    weights: &Rwkv7RnnChannelMixerWeights,
    in_bc: &Tensor,
    state: Option<&ChannelMixerFlatLayerState>,
    produce_next_state: bool,
    mut profile: Option<&mut ForwardProfile>,
) -> Result<Option<(Tensor, Option<ChannelMixerFlatLayerState>)>> {
    let total_start = ProfileTimer::start(profile.is_some());
    profile_inc!(profile, channel_mixer.calls);

    let start = ProfileTimer::start(profile.is_some());
    let (batch_size, channels) = in_bc.dims2()?;
    if !trusted_hot_loop_shapes_enabled() {
        validate_channel_mixer_shapes(weights, channels)?;
    }
    if let Some(state) = state {
        if state.channels != channels || state.state_values.len() != batch_size * channels {
            return Ok(None);
        }
    }
    profile_add!(profile, channel_mixer.validate_ns, start.elapsed_ns());

    if !(native_channel_mixer_projection_enabled()
        && native_channel_mixer_residual_scratch_enabled()
        && native_channel_mixer_prelude_residual_scratch_enabled())
    {
        return Ok(None);
    }

    let Some(output) = channel_mixer_prelude_projection_residual_native_from_values(
        in_bc,
        state,
        weights,
        profile.is_some(),
        produce_next_state,
    )?
    else {
        return Ok(None);
    };

    if let Some(profile) = profile.as_deref_mut() {
        profile.channel_mixer.native_projection_calls += 1;
        profile.channel_mixer.native_projection_rows += batch_size;
    }
    profile_add!(profile, channel_mixer.layer_norm_ns, output.layer_norm_ns);
    profile_add!(profile, channel_mixer.lerp_ns, output.lerp_ns);
    profile_add!(profile, channel_mixer.projection_ns, output.projection_ns);
    profile_add!(profile, channel_mixer.total_ns, total_start.elapsed_ns());

    let next_state = output
        .next_state_values
        .map(|state_values| ChannelMixerFlatLayerState {
            state_values,
            channels,
        });
    let out_bc = Tensor::from_vec(output.output_values, (batch_size, channels), in_bc.device())?;
    Ok(Some((out_bc, next_state)))
}

pub(super) fn channel_mixer_forward_flat_state_from_input_values_profiled(
    weights: &Rwkv7RnnChannelMixerWeights,
    input_values: &[f32],
    batch_size: usize,
    channels: usize,
    device: &Device,
    state: Option<&ChannelMixerFlatLayerState>,
    produce_next_state: bool,
    mut profile: Option<&mut ForwardProfile>,
) -> Result<Option<(Tensor, Option<ChannelMixerFlatLayerState>)>> {
    let total_start = ProfileTimer::start(profile.is_some());
    profile_inc!(profile, channel_mixer.calls);

    let start = ProfileTimer::start(profile.is_some());
    if !trusted_hot_loop_shapes_enabled() {
        validate_channel_mixer_shapes(weights, channels)?;
    }
    if input_values.len() != batch_size * channels {
        return Ok(None);
    }
    if let Some(state) = state {
        if state.channels != channels || state.state_values.len() != batch_size * channels {
            return Ok(None);
        }
    }
    profile_add!(profile, channel_mixer.validate_ns, start.elapsed_ns());

    if !(native_channel_mixer_projection_enabled()
        && native_channel_mixer_residual_scratch_enabled()
        && native_channel_mixer_prelude_residual_scratch_enabled())
    {
        return Ok(None);
    }

    let Some(dot_kernel) = native_linear_dot_kernel() else {
        return Ok(None);
    };
    let Some(output) =
        channel_mixer_prelude_projection_residual_native_from_input_values_with_kernel(
            input_values,
            batch_size,
            channels,
            device,
            state,
            weights,
            dot_kernel,
            profile.is_some(),
            produce_next_state,
        )?
    else {
        return Ok(None);
    };

    if let Some(profile) = profile.as_deref_mut() {
        profile.channel_mixer.native_projection_calls += 1;
        profile.channel_mixer.native_projection_rows += batch_size;
    }
    profile_add!(profile, channel_mixer.layer_norm_ns, output.layer_norm_ns);
    profile_add!(profile, channel_mixer.lerp_ns, output.lerp_ns);
    profile_add!(profile, channel_mixer.projection_ns, output.projection_ns);
    profile_add!(profile, channel_mixer.total_ns, total_start.elapsed_ns());

    let next_state = output
        .next_state_values
        .map(|state_values| ChannelMixerFlatLayerState {
            state_values,
            channels,
        });
    let out_bc = Tensor::from_vec(output.output_values, (batch_size, channels), device)?;
    Ok(Some((out_bc, next_state)))
}

pub(super) fn channel_mixer_forward_flat_state_from_input_values_raw_output_profiled(
    weights: &Rwkv7RnnChannelMixerWeights,
    input_values: &[f32],
    batch_size: usize,
    channels: usize,
    device: &Device,
    state: Option<&ChannelMixerFlatLayerState>,
    produce_next_state: bool,
    mut profile: Option<&mut ForwardProfile>,
) -> Result<Option<(Vec<f32>, Option<ChannelMixerFlatLayerState>)>> {
    let total_start = ProfileTimer::start(profile.is_some());
    profile_inc!(profile, channel_mixer.calls);

    let start = ProfileTimer::start(profile.is_some());
    if !trusted_hot_loop_shapes_enabled() {
        validate_channel_mixer_shapes(weights, channels)?;
    }
    if input_values.len() != batch_size * channels {
        return Ok(None);
    }
    if let Some(state) = state {
        if state.channels != channels || state.state_values.len() != batch_size * channels {
            return Ok(None);
        }
    }
    profile_add!(profile, channel_mixer.validate_ns, start.elapsed_ns());

    if !(native_channel_mixer_projection_enabled()
        && native_channel_mixer_residual_scratch_enabled()
        && native_channel_mixer_prelude_residual_scratch_enabled())
    {
        return Ok(None);
    }

    let Some(dot_kernel) = native_linear_dot_kernel() else {
        return Ok(None);
    };
    let Some(output) =
        channel_mixer_prelude_projection_residual_native_from_input_values_with_kernel(
            input_values,
            batch_size,
            channels,
            device,
            state,
            weights,
            dot_kernel,
            profile.is_some(),
            produce_next_state,
        )?
    else {
        return Ok(None);
    };

    if let Some(profile) = profile.as_deref_mut() {
        profile.channel_mixer.native_projection_calls += 1;
        profile.channel_mixer.native_projection_rows += batch_size;
    }
    profile_add!(profile, channel_mixer.layer_norm_ns, output.layer_norm_ns);
    profile_add!(profile, channel_mixer.lerp_ns, output.lerp_ns);
    profile_add!(profile, channel_mixer.projection_ns, output.projection_ns);
    profile_add!(profile, channel_mixer.total_ns, total_start.elapsed_ns());

    let next_state = output
        .next_state_values
        .map(|state_values| ChannelMixerFlatLayerState {
            state_values,
            channels,
        });
    Ok(Some((output.output_values, next_state)))
}

pub(super) fn channel_mixer_forward_flat_state_from_owned_input_values_raw_output_profiled(
    weights: &Rwkv7RnnChannelMixerWeights,
    input_values: Vec<f32>,
    batch_size: usize,
    channels: usize,
    device: &Device,
    state: Option<&ChannelMixerFlatLayerState>,
    produce_next_state: bool,
    mut profile: Option<&mut ForwardProfile>,
) -> Result<Option<(Vec<f32>, Option<ChannelMixerFlatLayerState>)>> {
    let total_start = ProfileTimer::start(profile.is_some());
    profile_inc!(profile, channel_mixer.calls);

    let start = ProfileTimer::start(profile.is_some());
    if !trusted_hot_loop_shapes_enabled() {
        validate_channel_mixer_shapes(weights, channels)?;
    }
    if input_values.len() != batch_size * channels {
        return Ok(None);
    }
    if let Some(state) = state {
        if state.channels != channels || state.state_values.len() != batch_size * channels {
            return Ok(None);
        }
    }
    profile_add!(profile, channel_mixer.validate_ns, start.elapsed_ns());

    if !(native_channel_mixer_projection_enabled()
        && native_channel_mixer_residual_scratch_enabled()
        && native_channel_mixer_prelude_residual_scratch_enabled()
        && native_channel_mixer_inplace_output_values_enabled())
    {
        return Ok(None);
    }

    let Some(dot_kernel) = native_linear_dot_kernel() else {
        return Ok(None);
    };
    let Some(output) =
        channel_mixer_prelude_projection_residual_native_from_owned_input_values_with_kernel(
            input_values,
            batch_size,
            channels,
            device,
            state,
            weights,
            dot_kernel,
            profile.is_some(),
            produce_next_state,
        )?
    else {
        return Ok(None);
    };

    if let Some(profile) = profile.as_deref_mut() {
        profile.channel_mixer.native_projection_calls += 1;
        profile.channel_mixer.native_projection_rows += batch_size;
    }
    profile_add!(profile, channel_mixer.layer_norm_ns, output.layer_norm_ns);
    profile_add!(profile, channel_mixer.lerp_ns, output.lerp_ns);
    profile_add!(profile, channel_mixer.projection_ns, output.projection_ns);
    profile_add!(profile, channel_mixer.total_ns, total_start.elapsed_ns());

    let next_state = output
        .next_state_values
        .map(|state_values| ChannelMixerFlatLayerState {
            state_values,
            channels,
        });
    Ok(Some((output.output_values, next_state)))
}

pub(super) fn channel_mixer_native_f32_weights(
    weights: &Rwkv7RnnChannelMixerWeights,
) -> Result<&ChannelMixerNativeF32Weights> {
    if weights.native_f32_cache.get().is_none() {
        let cache = ChannelMixerNativeF32Weights {
            w_k: linear_f32_weights_with_blocked(&weights.w_k, true)?,
            w_v: linear_f32_weights_with_blocked(&weights.w_v, true)?,
        };
        let _ = weights.native_f32_cache.set(cache);
    }
    Ok(weights
        .native_f32_cache
        .get()
        .expect("channel mixer native f32 cache is initialized"))
}

pub(super) fn channel_mixer_projection_native(
    xk_b1c: &Tensor,
    weights: &Rwkv7RnnChannelMixerWeights,
) -> Result<Option<(Tensor, NativeLinearDotKernel)>> {
    let Some(dot_kernel) = native_linear_dot_kernel() else {
        return Ok(None);
    };
    channel_mixer_projection_native_with_kernel(xk_b1c, weights, dot_kernel)
        .map(|output| output.map(|output| (output, dot_kernel)))
}

pub(super) fn channel_mixer_projection_residual_native(
    in_bc: &Tensor,
    xk_b1c: &Tensor,
    weights: &Rwkv7RnnChannelMixerWeights,
) -> Result<Option<(Tensor, NativeLinearDotKernel)>> {
    let Some(dot_kernel) = native_linear_dot_kernel() else {
        return Ok(None);
    };
    channel_mixer_projection_residual_native_with_kernel(in_bc, xk_b1c, weights, dot_kernel)
        .map(|output| output.map(|output| (output, dot_kernel)))
}

pub(super) fn channel_mixer_prelude_projection_residual_native(
    in_bc: &Tensor,
    state_b1c: Option<&Tensor>,
    weights: &Rwkv7RnnChannelMixerWeights,
    profile_enabled: bool,
) -> Result<Option<ChannelMixerPreludeResidualScratchOutput>> {
    let Some(dot_kernel) = native_linear_dot_kernel() else {
        return Ok(None);
    };
    channel_mixer_prelude_projection_residual_native_with_kernel(
        in_bc,
        state_b1c,
        weights,
        dot_kernel,
        profile_enabled,
    )
}

pub(super) fn channel_mixer_prelude_projection_residual_native_from_values(
    in_bc: &Tensor,
    state: Option<&ChannelMixerFlatLayerState>,
    weights: &Rwkv7RnnChannelMixerWeights,
    profile_enabled: bool,
    produce_next_state: bool,
) -> Result<Option<ChannelMixerPreludeResidualFlatStateOutput>> {
    let Some(dot_kernel) = native_linear_dot_kernel() else {
        return Ok(None);
    };
    channel_mixer_prelude_projection_residual_native_from_values_with_kernel(
        in_bc,
        state,
        weights,
        dot_kernel,
        profile_enabled,
        produce_next_state,
    )
}

#[inline(always)]
pub(super) fn channel_mixer_relu_square_in_place(values: &mut [f32], direct_relu_square: bool) {
    if direct_relu_square {
        for value in values {
            *value = if *value > 0.0 { *value * *value } else { 0.0 };
        }
    } else {
        for value in values {
            *value = value.max(0.0).powi(2);
        }
    }
}

#[inline(always)]
pub(super) fn channel_mixer_project_row(
    dot_kernel: NativeLinearDotKernel,
    input_row: &[f32],
    row_major_weights: &[f32],
    out_dim: usize,
    in_dim: usize,
    cached_weights: Option<&LinearF32Weights>,
    output_row: &mut [f32],
) {
    if let Some(cached_weights) = cached_weights {
        linear_project_f32_row_same_x(dot_kernel, input_row, cached_weights, output_row);
    } else {
        linear_project_row_same_x(
            dot_kernel,
            input_row,
            row_major_weights,
            out_dim,
            in_dim,
            output_row,
        );
    }
}

pub(super) fn channel_mixer_projection_native_with_kernel(
    xk_b1c: &Tensor,
    weights: &Rwkv7RnnChannelMixerWeights,
    dot_kernel: NativeLinearDotKernel,
) -> Result<Option<Tensor>> {
    let (batch_size, one, channels) = xk_b1c.dims3()?;
    if one != 1 {
        return Ok(None);
    }
    let (hidden_dim, w_k_in_dim) = weights.w_k.weight.dims2()?;
    let (w_v_out_dim, w_v_in_dim) = weights.w_v.weight.dims2()?;
    if w_k_in_dim != channels || w_v_out_dim != channels || w_v_in_dim != hidden_dim {
        return Ok(None);
    }
    if weights.w_k.bias.is_some() || weights.w_v.bias.is_some() {
        return Ok(None);
    }

    let x_data = f32_tensor_data(xk_b1c)?;
    let w_k_data = f32_tensor_data(&weights.w_k.weight)?;
    let w_v_data = f32_tensor_data(&weights.w_v.weight)?;
    let x_values = x_data.as_slice()?;
    let w_k_values = w_k_data.as_slice()?;
    let w_v_values = w_v_data.as_slice()?;
    if x_values.len() != batch_size * channels
        || w_k_values.len() != hidden_dim * channels
        || w_v_values.len() != channels * hidden_dim
    {
        return Ok(None);
    }

    let mut output_values = vec![0.0f32; batch_size * channels];
    let mut hidden_values = vec![0.0f32; hidden_dim];
    let direct_relu_square = native_channel_mixer_direct_relu_square_enabled();
    for batch_index in 0..batch_size {
        let x_row = &x_values[batch_index * channels..(batch_index + 1) * channels];
        linear_project_row_same_x(
            dot_kernel,
            x_row,
            w_k_values,
            hidden_dim,
            channels,
            &mut hidden_values,
        );
        channel_mixer_relu_square_in_place(&mut hidden_values, direct_relu_square);

        let output_row = &mut output_values[batch_index * channels..(batch_index + 1) * channels];
        linear_project_row_same_x(
            dot_kernel,
            &hidden_values,
            w_v_values,
            channels,
            hidden_dim,
            output_row,
        );
    }

    Tensor::from_vec(
        output_values,
        (batch_size, 1usize, channels),
        xk_b1c.device(),
    )
    .map(Some)
}

pub(super) fn channel_mixer_projection_residual_native_with_kernel(
    in_bc: &Tensor,
    xk_b1c: &Tensor,
    weights: &Rwkv7RnnChannelMixerWeights,
    dot_kernel: NativeLinearDotKernel,
) -> Result<Option<Tensor>> {
    let (batch_size, channels) = in_bc.dims2()?;
    let (x_batch_size, one, x_channels) = xk_b1c.dims3()?;
    if x_batch_size != batch_size || one != 1 || x_channels != channels {
        return Ok(None);
    }
    let (hidden_dim, w_k_in_dim) = weights.w_k.weight.dims2()?;
    let (w_v_out_dim, w_v_in_dim) = weights.w_v.weight.dims2()?;
    if w_k_in_dim != channels || w_v_out_dim != channels || w_v_in_dim != hidden_dim {
        return Ok(None);
    }
    if weights.w_k.bias.is_some() || weights.w_v.bias.is_some() {
        return Ok(None);
    }

    let in_data = f32_tensor_data(in_bc)?;
    let x_data = f32_tensor_data(xk_b1c)?;
    let w_k_data = f32_tensor_data(&weights.w_k.weight)?;
    let w_v_data = f32_tensor_data(&weights.w_v.weight)?;
    let in_values = in_data.as_slice()?;
    let x_values = x_data.as_slice()?;
    let w_k_values = w_k_data.as_slice()?;
    let w_v_values = w_v_data.as_slice()?;
    if in_values.len() != batch_size * channels
        || x_values.len() != batch_size * channels
        || w_k_values.len() != hidden_dim * channels
        || w_v_values.len() != channels * hidden_dim
    {
        return Ok(None);
    }

    let mut output_values = vec![0.0f32; batch_size * channels];
    let mut hidden_values = vec![0.0f32; hidden_dim];
    let direct_relu_square = native_channel_mixer_direct_relu_square_enabled();
    for batch_index in 0..batch_size {
        let batch_base = batch_index * channels;
        let x_row = &x_values[batch_base..batch_base + channels];
        linear_project_row_same_x(
            dot_kernel,
            x_row,
            w_k_values,
            hidden_dim,
            channels,
            &mut hidden_values,
        );
        channel_mixer_relu_square_in_place(&mut hidden_values, direct_relu_square);

        let output_row = &mut output_values[batch_base..batch_base + channels];
        linear_project_row_same_x(
            dot_kernel,
            &hidden_values,
            w_v_values,
            channels,
            hidden_dim,
            output_row,
        );
        for channel_index in 0..channels {
            output_row[channel_index] =
                in_values[batch_base + channel_index] + output_row[channel_index];
        }
    }

    Tensor::from_vec(output_values, (batch_size, channels), xk_b1c.device()).map(Some)
}

pub(super) fn channel_mixer_prelude_projection_residual_native_with_kernel(
    in_bc: &Tensor,
    state_b1c: Option<&Tensor>,
    weights: &Rwkv7RnnChannelMixerWeights,
    dot_kernel: NativeLinearDotKernel,
    profile_enabled: bool,
) -> Result<Option<ChannelMixerPreludeResidualScratchOutput>> {
    let (batch_size, channels) = in_bc.dims2()?;
    if let Some(state) = state_b1c {
        if state.dims() != [batch_size, 1, channels] {
            return Ok(None);
        }
    }
    if weights.layer_norm.weight.dims() != [channels]
        || weights.layer_norm.bias.dims() != [channels]
        || weights.lerp_k.dims() != [1, 1, channels]
    {
        return Ok(None);
    }

    let (hidden_dim, w_k_in_dim) = weights.w_k.weight.dims2()?;
    let (w_v_out_dim, w_v_in_dim) = weights.w_v.weight.dims2()?;
    if w_k_in_dim != channels || w_v_out_dim != channels || w_v_in_dim != hidden_dim {
        return Ok(None);
    }
    if weights.w_k.bias.is_some() || weights.w_v.bias.is_some() {
        return Ok(None);
    }

    let in_data = f32_tensor_data(in_bc)?;
    let state_data = state_b1c.map(f32_tensor_data).transpose()?;
    let norm_weight_data = f32_tensor_data(&weights.layer_norm.weight)?;
    let norm_bias_data = f32_tensor_data(&weights.layer_norm.bias)?;
    let lerp_data = f32_tensor_data(&weights.lerp_k)?;
    let w_k_data = f32_tensor_data(&weights.w_k.weight)?;
    let w_v_data = f32_tensor_data(&weights.w_v.weight)?;

    let in_values = in_data.as_slice()?;
    let state_values = state_data
        .as_ref()
        .map(|data| data.as_slice())
        .transpose()?;
    let norm_weight_values = norm_weight_data.as_slice()?;
    let norm_bias_values = norm_bias_data.as_slice()?;
    let lerp_values = lerp_data.as_slice()?;
    let w_k_values = w_k_data.as_slice()?;
    let w_v_values = w_v_data.as_slice()?;

    let bxc_len = batch_size * channels;
    if in_values.len() != bxc_len
        || state_values.map_or(false, |values| values.len() != bxc_len)
        || norm_weight_values.len() != channels
        || norm_bias_values.len() != channels
        || lerp_values.len() != channels
        || w_k_values.len() != hidden_dim * channels
        || w_v_values.len() != channels * hidden_dim
    {
        return Ok(None);
    }

    let layer_norm_start = ProfileTimer::start(profile_enabled);
    let mut next_state_values = vec![0.0f32; bxc_len];
    for batch_index in 0..batch_size {
        let batch_base = batch_index * channels;
        let input_row = &in_values[batch_base..batch_base + channels];
        let mean = input_row.iter().copied().sum::<f32>() / channels as f32;
        let variance = input_row
            .iter()
            .map(|value| {
                let centered = *value - mean;
                centered * centered
            })
            .sum::<f32>()
            / channels as f32;
        let inv_std = 1.0f32 / (variance + LAYER_NORM_EPS).sqrt();
        let state_row = &mut next_state_values[batch_base..batch_base + channels];
        for channel_index in 0..channels {
            state_row[channel_index] =
                (input_row[channel_index] - mean) * inv_std * norm_weight_values[channel_index]
                    + norm_bias_values[channel_index];
        }
    }
    let layer_norm_ns = layer_norm_start.elapsed_ns();

    let lerp_start = ProfileTimer::start(profile_enabled);
    let mut xk_values = vec![0.0f32; bxc_len];
    for batch_index in 0..batch_size {
        let batch_base = batch_index * channels;
        let x_row = &next_state_values[batch_base..batch_base + channels];
        let x_shift_row = state_values
            .map(|values| &values[batch_base..batch_base + channels])
            .unwrap_or(x_row);
        let xk_row = &mut xk_values[batch_base..batch_base + channels];
        for channel_index in 0..channels {
            let x = x_row[channel_index];
            xk_row[channel_index] =
                x + (x_shift_row[channel_index] - x) * lerp_values[channel_index];
        }
    }
    let lerp_ns = lerp_start.elapsed_ns();

    let projection_start = ProfileTimer::start(profile_enabled);
    let mut output_values = vec![0.0f32; bxc_len];
    let mut hidden_values = vec![0.0f32; hidden_dim];
    let direct_relu_square = native_channel_mixer_direct_relu_square_enabled();
    for batch_index in 0..batch_size {
        let batch_base = batch_index * channels;
        let x_row = &xk_values[batch_base..batch_base + channels];
        linear_project_row_same_x(
            dot_kernel,
            x_row,
            w_k_values,
            hidden_dim,
            channels,
            &mut hidden_values,
        );
        channel_mixer_relu_square_in_place(&mut hidden_values, direct_relu_square);

        let output_row = &mut output_values[batch_base..batch_base + channels];
        linear_project_row_same_x(
            dot_kernel,
            &hidden_values,
            w_v_values,
            channels,
            hidden_dim,
            output_row,
        );
        for channel_index in 0..channels {
            output_row[channel_index] =
                in_values[batch_base + channel_index] + output_row[channel_index];
        }
    }
    let projection_ns = projection_start.elapsed_ns();

    Ok(Some(ChannelMixerPreludeResidualScratchOutput {
        out_bc: Tensor::from_vec(output_values, (batch_size, channels), in_bc.device())?,
        next_state_b1c: Tensor::from_vec(
            next_state_values,
            (batch_size, 1usize, channels),
            in_bc.device(),
        )?,
        layer_norm_ns,
        lerp_ns,
        projection_ns,
    }))
}

pub(super) fn channel_mixer_prelude_projection_residual_native_from_values_with_kernel(
    in_bc: &Tensor,
    state: Option<&ChannelMixerFlatLayerState>,
    weights: &Rwkv7RnnChannelMixerWeights,
    dot_kernel: NativeLinearDotKernel,
    profile_enabled: bool,
    produce_next_state: bool,
) -> Result<Option<ChannelMixerPreludeResidualFlatStateOutput>> {
    let (batch_size, channels) = in_bc.dims2()?;
    if let Some(state) = state {
        if state.channels != channels || state.state_values.len() != batch_size * channels {
            return Ok(None);
        }
    }
    if weights.layer_norm.weight.dims() != [channels]
        || weights.layer_norm.bias.dims() != [channels]
        || weights.lerp_k.dims() != [1, 1, channels]
    {
        return Ok(None);
    }

    let (hidden_dim, w_k_in_dim) = weights.w_k.weight.dims2()?;
    let (w_v_out_dim, w_v_in_dim) = weights.w_v.weight.dims2()?;
    if w_k_in_dim != channels || w_v_out_dim != channels || w_v_in_dim != hidden_dim {
        return Ok(None);
    }
    if weights.w_k.bias.is_some() || weights.w_v.bias.is_some() {
        return Ok(None);
    }

    let in_data = f32_tensor_data(in_bc)?;
    let norm_weight_data = f32_tensor_data(&weights.layer_norm.weight)?;
    let norm_bias_data = f32_tensor_data(&weights.layer_norm.bias)?;
    let lerp_data = f32_tensor_data(&weights.lerp_k)?;
    let w_k_data = f32_tensor_data(&weights.w_k.weight)?;
    let w_v_data = f32_tensor_data(&weights.w_v.weight)?;

    let in_values = in_data.as_slice()?;
    let norm_weight_values = norm_weight_data.as_slice()?;
    let norm_bias_values = norm_bias_data.as_slice()?;
    let lerp_values = lerp_data.as_slice()?;
    let w_k_values = w_k_data.as_slice()?;
    let w_v_values = w_v_data.as_slice()?;

    let bxc_len = batch_size * channels;
    if in_values.len() != bxc_len
        || norm_weight_values.len() != channels
        || norm_bias_values.len() != channels
        || lerp_values.len() != channels
        || w_k_values.len() != hidden_dim * channels
        || w_v_values.len() != channels * hidden_dim
    {
        return Ok(None);
    }

    let cached_projection =
        if batch_size == 1 && native_linear_blocked_input8_dot12_layout_enabled() {
            Some(channel_mixer_native_f32_weights(weights)?)
        } else {
            None
        };
    let use_batched_projection = native_linear_batched_same_x_enabled() && batch_size >= 2;
    let hidden_len = if use_batched_projection {
        batch_size * hidden_dim
    } else {
        hidden_dim
    };
    let (mut x_values, mut xk_values, mut hidden_values) =
        channel_mixer_flat_scratch_vectors(bxc_len, hidden_len);

    let layer_norm_start = ProfileTimer::start(profile_enabled);
    for batch_index in 0..batch_size {
        let batch_base = batch_index * channels;
        let input_row = &in_values[batch_base..batch_base + channels];
        let mean = input_row.iter().copied().sum::<f32>() / channels as f32;
        let variance = input_row
            .iter()
            .map(|value| {
                let centered = *value - mean;
                centered * centered
            })
            .sum::<f32>()
            / channels as f32;
        let inv_std = 1.0f32 / (variance + LAYER_NORM_EPS).sqrt();
        let x_row = &mut x_values[batch_base..batch_base + channels];
        for channel_index in 0..channels {
            x_row[channel_index] =
                (input_row[channel_index] - mean) * inv_std * norm_weight_values[channel_index]
                    + norm_bias_values[channel_index];
        }
    }
    let layer_norm_ns = layer_norm_start.elapsed_ns();

    let lerp_start = ProfileTimer::start(profile_enabled);
    for batch_index in 0..batch_size {
        let batch_base = batch_index * channels;
        let x_row = &x_values[batch_base..batch_base + channels];
        let x_shift_row = state
            .map(|state| &state.state_values[batch_base..batch_base + channels])
            .unwrap_or(x_row);
        let xk_row = &mut xk_values[batch_base..batch_base + channels];
        for channel_index in 0..channels {
            let x = x_row[channel_index];
            xk_row[channel_index] =
                x + (x_shift_row[channel_index] - x) * lerp_values[channel_index];
        }
    }
    let lerp_ns = lerp_start.elapsed_ns();

    let projection_start = ProfileTimer::start(profile_enabled);
    let mut output_values = vec![0.0f32; bxc_len];
    let direct_relu_square = native_channel_mixer_direct_relu_square_enabled();
    let projected_batched = use_batched_projection
        && linear_project_batch_same_x(
            dot_kernel,
            &xk_values,
            batch_size,
            channels,
            w_k_values,
            hidden_dim,
            &mut hidden_values,
        )
        && {
            channel_mixer_relu_square_in_place(&mut hidden_values, direct_relu_square);
            linear_project_batch_same_x(
                dot_kernel,
                &hidden_values,
                batch_size,
                hidden_dim,
                w_v_values,
                channels,
                &mut output_values,
            )
        };

    if projected_batched {
        for index in 0..bxc_len {
            output_values[index] += in_values[index];
        }
    } else {
        for batch_index in 0..batch_size {
            let batch_base = batch_index * channels;
            let x_row = &xk_values[batch_base..batch_base + channels];
            let hidden_row = &mut hidden_values[..hidden_dim];
            channel_mixer_project_row(
                dot_kernel,
                x_row,
                w_k_values,
                hidden_dim,
                channels,
                cached_projection.map(|cache| &cache.w_k),
                hidden_row,
            );
            channel_mixer_relu_square_in_place(hidden_row, direct_relu_square);

            let output_row = &mut output_values[batch_base..batch_base + channels];
            channel_mixer_project_row(
                dot_kernel,
                hidden_row,
                w_v_values,
                channels,
                hidden_dim,
                cached_projection.map(|cache| &cache.w_v),
                output_row,
            );
            for channel_index in 0..channels {
                output_row[channel_index] =
                    in_values[batch_base + channel_index] + output_row[channel_index];
            }
        }
    }
    let projection_ns = projection_start.elapsed_ns();

    let next_state_values = if produce_next_state {
        recycle_channel_mixer_flat_scratch(None, xk_values, hidden_values);
        Some(x_values)
    } else {
        recycle_channel_mixer_flat_scratch(Some(x_values), xk_values, hidden_values);
        None
    };

    Ok(Some(ChannelMixerPreludeResidualFlatStateOutput {
        output_values,
        next_state_values,
        layer_norm_ns,
        lerp_ns,
        projection_ns,
    }))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn channel_mixer_prelude_projection_residual_native_from_input_values_with_kernel(
    in_values: &[f32],
    batch_size: usize,
    channels: usize,
    _device: &Device,
    state: Option<&ChannelMixerFlatLayerState>,
    weights: &Rwkv7RnnChannelMixerWeights,
    dot_kernel: NativeLinearDotKernel,
    profile_enabled: bool,
    produce_next_state: bool,
) -> Result<Option<ChannelMixerPreludeResidualFlatStateOutput>> {
    if let Some(state) = state {
        if state.channels != channels || state.state_values.len() != batch_size * channels {
            return Ok(None);
        }
    }
    if weights.layer_norm.weight.dims() != [channels]
        || weights.layer_norm.bias.dims() != [channels]
        || weights.lerp_k.dims() != [1, 1, channels]
    {
        return Ok(None);
    }

    let (hidden_dim, w_k_in_dim) = weights.w_k.weight.dims2()?;
    let (w_v_out_dim, w_v_in_dim) = weights.w_v.weight.dims2()?;
    if w_k_in_dim != channels || w_v_out_dim != channels || w_v_in_dim != hidden_dim {
        return Ok(None);
    }
    if weights.w_k.bias.is_some() || weights.w_v.bias.is_some() {
        return Ok(None);
    }

    let norm_weight_data = f32_tensor_data(&weights.layer_norm.weight)?;
    let norm_bias_data = f32_tensor_data(&weights.layer_norm.bias)?;
    let lerp_data = f32_tensor_data(&weights.lerp_k)?;
    let w_k_data = f32_tensor_data(&weights.w_k.weight)?;
    let w_v_data = f32_tensor_data(&weights.w_v.weight)?;

    let norm_weight_values = norm_weight_data.as_slice()?;
    let norm_bias_values = norm_bias_data.as_slice()?;
    let lerp_values = lerp_data.as_slice()?;
    let w_k_values = w_k_data.as_slice()?;
    let w_v_values = w_v_data.as_slice()?;

    let bxc_len = batch_size * channels;
    if in_values.len() != bxc_len
        || norm_weight_values.len() != channels
        || norm_bias_values.len() != channels
        || lerp_values.len() != channels
        || w_k_values.len() != hidden_dim * channels
        || w_v_values.len() != channels * hidden_dim
    {
        return Ok(None);
    }

    let cached_projection =
        if batch_size == 1 && native_linear_blocked_input8_dot12_layout_enabled() {
            Some(channel_mixer_native_f32_weights(weights)?)
        } else {
            None
        };
    let use_batched_projection = native_linear_batched_same_x_enabled() && batch_size >= 2;
    let hidden_len = if use_batched_projection {
        batch_size * hidden_dim
    } else {
        hidden_dim
    };
    let (mut x_values, mut xk_values, mut hidden_values) =
        channel_mixer_flat_scratch_vectors(bxc_len, hidden_len);

    let layer_norm_start = ProfileTimer::start(profile_enabled);
    for batch_index in 0..batch_size {
        let batch_base = batch_index * channels;
        let input_row = &in_values[batch_base..batch_base + channels];
        let mean = input_row.iter().copied().sum::<f32>() / channels as f32;
        let variance = input_row
            .iter()
            .map(|value| {
                let centered = *value - mean;
                centered * centered
            })
            .sum::<f32>()
            / channels as f32;
        let inv_std = 1.0f32 / (variance + LAYER_NORM_EPS).sqrt();
        let x_row = &mut x_values[batch_base..batch_base + channels];
        for channel_index in 0..channels {
            x_row[channel_index] =
                (input_row[channel_index] - mean) * inv_std * norm_weight_values[channel_index]
                    + norm_bias_values[channel_index];
        }
    }
    let layer_norm_ns = layer_norm_start.elapsed_ns();

    let lerp_start = ProfileTimer::start(profile_enabled);
    for batch_index in 0..batch_size {
        let batch_base = batch_index * channels;
        let x_row = &x_values[batch_base..batch_base + channels];
        let x_shift_row = state
            .map(|state| &state.state_values[batch_base..batch_base + channels])
            .unwrap_or(x_row);
        let xk_row = &mut xk_values[batch_base..batch_base + channels];
        for channel_index in 0..channels {
            let x = x_row[channel_index];
            xk_row[channel_index] =
                x + (x_shift_row[channel_index] - x) * lerp_values[channel_index];
        }
    }
    let lerp_ns = lerp_start.elapsed_ns();

    let projection_start = ProfileTimer::start(profile_enabled);
    let mut output_values = vec![0.0f32; bxc_len];
    let direct_relu_square = native_channel_mixer_direct_relu_square_enabled();
    let projected_batched = use_batched_projection
        && linear_project_batch_same_x(
            dot_kernel,
            &xk_values,
            batch_size,
            channels,
            w_k_values,
            hidden_dim,
            &mut hidden_values,
        )
        && {
            channel_mixer_relu_square_in_place(&mut hidden_values, direct_relu_square);
            linear_project_batch_same_x(
                dot_kernel,
                &hidden_values,
                batch_size,
                hidden_dim,
                w_v_values,
                channels,
                &mut output_values,
            )
        };

    if projected_batched {
        for index in 0..bxc_len {
            output_values[index] += in_values[index];
        }
    } else {
        for batch_index in 0..batch_size {
            let batch_base = batch_index * channels;
            let x_row = &xk_values[batch_base..batch_base + channels];
            let hidden_row = &mut hidden_values[..hidden_dim];
            channel_mixer_project_row(
                dot_kernel,
                x_row,
                w_k_values,
                hidden_dim,
                channels,
                cached_projection.map(|cache| &cache.w_k),
                hidden_row,
            );
            channel_mixer_relu_square_in_place(hidden_row, direct_relu_square);

            let output_row = &mut output_values[batch_base..batch_base + channels];
            channel_mixer_project_row(
                dot_kernel,
                hidden_row,
                w_v_values,
                channels,
                hidden_dim,
                cached_projection.map(|cache| &cache.w_v),
                output_row,
            );
            for channel_index in 0..channels {
                output_row[channel_index] =
                    in_values[batch_base + channel_index] + output_row[channel_index];
            }
        }
    }
    let projection_ns = projection_start.elapsed_ns();

    let next_state_values = if produce_next_state {
        recycle_channel_mixer_flat_scratch(None, xk_values, hidden_values);
        Some(x_values)
    } else {
        recycle_channel_mixer_flat_scratch(Some(x_values), xk_values, hidden_values);
        None
    };

    Ok(Some(ChannelMixerPreludeResidualFlatStateOutput {
        output_values,
        next_state_values,
        layer_norm_ns,
        lerp_ns,
        projection_ns,
    }))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn channel_mixer_prelude_projection_residual_native_from_owned_input_values_with_kernel(
    mut input_values: Vec<f32>,
    batch_size: usize,
    channels: usize,
    _device: &Device,
    state: Option<&ChannelMixerFlatLayerState>,
    weights: &Rwkv7RnnChannelMixerWeights,
    dot_kernel: NativeLinearDotKernel,
    profile_enabled: bool,
    produce_next_state: bool,
) -> Result<Option<ChannelMixerPreludeResidualFlatStateOutput>> {
    if let Some(state) = state {
        if state.channels != channels || state.state_values.len() != batch_size * channels {
            return Ok(None);
        }
    }
    if weights.layer_norm.weight.dims() != [channels]
        || weights.layer_norm.bias.dims() != [channels]
        || weights.lerp_k.dims() != [1, 1, channels]
    {
        return Ok(None);
    }

    let (hidden_dim, w_k_in_dim) = weights.w_k.weight.dims2()?;
    let (w_v_out_dim, w_v_in_dim) = weights.w_v.weight.dims2()?;
    if w_k_in_dim != channels || w_v_out_dim != channels || w_v_in_dim != hidden_dim {
        return Ok(None);
    }
    if weights.w_k.bias.is_some() || weights.w_v.bias.is_some() {
        return Ok(None);
    }

    let norm_weight_data = f32_tensor_data(&weights.layer_norm.weight)?;
    let norm_bias_data = f32_tensor_data(&weights.layer_norm.bias)?;
    let lerp_data = f32_tensor_data(&weights.lerp_k)?;
    let w_k_data = f32_tensor_data(&weights.w_k.weight)?;
    let w_v_data = f32_tensor_data(&weights.w_v.weight)?;

    let norm_weight_values = norm_weight_data.as_slice()?;
    let norm_bias_values = norm_bias_data.as_slice()?;
    let lerp_values = lerp_data.as_slice()?;
    let w_k_values = w_k_data.as_slice()?;
    let w_v_values = w_v_data.as_slice()?;

    let bxc_len = batch_size * channels;
    if input_values.len() != bxc_len
        || norm_weight_values.len() != channels
        || norm_bias_values.len() != channels
        || lerp_values.len() != channels
        || w_k_values.len() != hidden_dim * channels
        || w_v_values.len() != channels * hidden_dim
    {
        return Ok(None);
    }

    let cached_projection =
        if batch_size == 1 && native_linear_blocked_input8_dot12_layout_enabled() {
            Some(channel_mixer_native_f32_weights(weights)?)
        } else {
            None
        };
    let use_batched_projection = native_linear_batched_same_x_enabled() && batch_size >= 2;
    let hidden_len = if use_batched_projection {
        batch_size * hidden_dim
    } else {
        hidden_dim
    };
    let (mut x_values, mut xk_values, mut hidden_values) =
        channel_mixer_flat_scratch_vectors(bxc_len, hidden_len);

    let layer_norm_start = ProfileTimer::start(profile_enabled);
    for batch_index in 0..batch_size {
        let batch_base = batch_index * channels;
        let input_row = &input_values[batch_base..batch_base + channels];
        let mean = input_row.iter().copied().sum::<f32>() / channels as f32;
        let variance = input_row
            .iter()
            .map(|value| {
                let centered = *value - mean;
                centered * centered
            })
            .sum::<f32>()
            / channels as f32;
        let inv_std = 1.0f32 / (variance + LAYER_NORM_EPS).sqrt();
        let x_row = &mut x_values[batch_base..batch_base + channels];
        for channel_index in 0..channels {
            x_row[channel_index] =
                (input_row[channel_index] - mean) * inv_std * norm_weight_values[channel_index]
                    + norm_bias_values[channel_index];
        }
    }
    let layer_norm_ns = layer_norm_start.elapsed_ns();

    let lerp_start = ProfileTimer::start(profile_enabled);
    for batch_index in 0..batch_size {
        let batch_base = batch_index * channels;
        let x_row = &x_values[batch_base..batch_base + channels];
        let x_shift_row = state
            .map(|state| &state.state_values[batch_base..batch_base + channels])
            .unwrap_or(x_row);
        let xk_row = &mut xk_values[batch_base..batch_base + channels];
        for channel_index in 0..channels {
            let x = x_row[channel_index];
            xk_row[channel_index] =
                x + (x_shift_row[channel_index] - x) * lerp_values[channel_index];
        }
    }
    let lerp_ns = lerp_start.elapsed_ns();

    let projection_start = ProfileTimer::start(profile_enabled);
    let direct_relu_square = native_channel_mixer_direct_relu_square_enabled();
    let projected_batched = use_batched_projection
        && linear_project_batch_same_x(
            dot_kernel,
            &xk_values,
            batch_size,
            channels,
            w_k_values,
            hidden_dim,
            &mut hidden_values,
        )
        && {
            channel_mixer_relu_square_in_place(&mut hidden_values, direct_relu_square);
            linear_project_batch_same_x(
                dot_kernel,
                &hidden_values,
                batch_size,
                hidden_dim,
                w_v_values,
                channels,
                &mut xk_values,
            )
        };

    if projected_batched {
        for index in 0..bxc_len {
            input_values[index] += xk_values[index];
        }
    } else {
        for batch_index in 0..batch_size {
            let batch_base = batch_index * channels;
            {
                let x_row = &xk_values[batch_base..batch_base + channels];
                let hidden_row = &mut hidden_values[..hidden_dim];
                channel_mixer_project_row(
                    dot_kernel,
                    x_row,
                    w_k_values,
                    hidden_dim,
                    channels,
                    cached_projection.map(|cache| &cache.w_k),
                    hidden_row,
                );
                channel_mixer_relu_square_in_place(hidden_row, direct_relu_square);
            }

            let output_row = &mut xk_values[batch_base..batch_base + channels];
            channel_mixer_project_row(
                dot_kernel,
                &hidden_values[..hidden_dim],
                w_v_values,
                channels,
                hidden_dim,
                cached_projection.map(|cache| &cache.w_v),
                output_row,
            );
            for channel_index in 0..channels {
                input_values[batch_base + channel_index] += output_row[channel_index];
            }
        }
    }
    let projection_ns = projection_start.elapsed_ns();

    let next_state_values = if produce_next_state {
        recycle_channel_mixer_flat_scratch(None, xk_values, hidden_values);
        Some(x_values)
    } else {
        recycle_channel_mixer_flat_scratch(Some(x_values), xk_values, hidden_values);
        None
    };

    Ok(Some(ChannelMixerPreludeResidualFlatStateOutput {
        output_values: input_values,
        next_state_values,
        layer_norm_ns,
        lerp_ns,
        projection_ns,
    }))
}
