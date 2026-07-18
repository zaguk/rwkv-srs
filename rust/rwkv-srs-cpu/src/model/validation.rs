use candle_core::{bail, Result, Tensor};

use crate::model_weights::{Rwkv7RnnChannelMixerWeights, Rwkv7RnnTimeMixerWeights};

pub(super) fn validate_channel_mixer_shapes(
    weights: &Rwkv7RnnChannelMixerWeights,
    channels: usize,
) -> Result<()> {
    if weights.layer_norm.weight.dims() != [channels]
        || weights.layer_norm.bias.dims() != [channels]
    {
        bail!(
            "channel_mixer expected layer_norm weight/bias shape [{channels}], got {:?} and {:?}",
            weights.layer_norm.weight.dims(),
            weights.layer_norm.bias.dims()
        );
    }
    if weights.lerp_k.dims() != [1, 1, channels] {
        bail!(
            "channel_mixer expected lerp_k shape [1, 1, {channels}], got {:?}",
            weights.lerp_k.dims()
        );
    }
    if weights.w_k.weight.dims() != [weights.channel_dim, channels] {
        bail!(
            "channel_mixer expected W_k weight shape [{}, {channels}], got {:?}",
            weights.channel_dim,
            weights.w_k.weight.dims()
        );
    }
    if weights.w_v.weight.dims() != [channels, weights.channel_dim] {
        bail!(
            "channel_mixer expected W_v weight shape [{channels}, {}], got {:?}",
            weights.channel_dim,
            weights.w_v.weight.dims()
        );
    }
    if weights.w_k.bias.is_some() || weights.w_v.bias.is_some() {
        bail!("channel_mixer W_k and W_v must not have biases");
    }
    Ok(())
}

pub(super) fn validate_time_mixer_shapes(
    weights: &Rwkv7RnnTimeMixerWeights,
    channels: usize,
) -> Result<()> {
    let heads = weights.n_heads;
    let head_size = weights.head_size;
    if channels != heads * head_size {
        bail!(
            "time_mixer expected channels to equal heads * head_size ({} * {}), got {channels}",
            heads,
            head_size
        );
    }
    if weights.layer_norm.weight.dims() != [channels]
        || weights.layer_norm.bias.dims() != [channels]
    {
        bail!(
            "time_mixer expected layer_norm weight/bias shape [{channels}], got {:?} and {:?}",
            weights.layer_norm.weight.dims(),
            weights.layer_norm.bias.dims()
        );
    }
    if weights.rkvdag_lerp.dims() != [8, 1, 1, channels] {
        bail!(
            "time_mixer expected rkvdag_lerp shape [8, 1, 1, {channels}], got {:?}",
            weights.rkvdag_lerp.dims()
        );
    }
    if weights.bonus.dims() != [1, 1, heads, head_size] {
        bail!(
            "time_mixer expected bonus shape [1, 1, {heads}, {head_size}], got {:?}",
            weights.bonus.dims()
        );
    }
    Ok(())
}

pub(super) fn validate_time_mixer_state_shapes(
    x_shift_b1c: &Tensor,
    state_b1hkk: &Tensor,
    batch_size: usize,
    channels: usize,
    heads: usize,
    head_size: usize,
) -> Result<()> {
    if x_shift_b1c.dims() != [batch_size, 1, channels] {
        bail!(
            "time_mixer expected x_shift_B1C shape [{batch_size}, 1, {channels}], got {:?}",
            x_shift_b1c.dims()
        );
    }
    if state_b1hkk.dims() != [batch_size, 1, heads, head_size, head_size] {
        bail!(
            "time_mixer expected state_B1HKK shape [{batch_size}, 1, {heads}, {head_size}, {head_size}], got {:?}",
            state_b1hkk.dims()
        );
    }
    Ok(())
}

pub(super) fn validate_srs_review_state_modules(
    expected_modules: usize,
    time_x_shift_b1c_by_module: Option<&[Vec<Tensor>]>,
    time_state_b1hkk_by_module: Option<&[Vec<Tensor>]>,
    channel_state_b1c_by_module: Option<&[Vec<Tensor>]>,
) -> Result<()> {
    let provided_group_count = [
        time_x_shift_b1c_by_module.is_some(),
        time_state_b1hkk_by_module.is_some(),
        channel_state_b1c_by_module.is_some(),
    ]
    .into_iter()
    .filter(|provided| *provided)
    .count();

    if provided_group_count != 0 && provided_group_count != 3 {
        bail!(
            "srs_review requires time_x_shift_b1c_by_module, time_state_b1hkk_by_module, and channel_state_b1c_by_module together, or none"
        );
    }

    for (name, states) in [
        ("time_x_shift_b1c_by_module", time_x_shift_b1c_by_module),
        ("time_state_b1hkk_by_module", time_state_b1hkk_by_module),
        ("channel_state_b1c_by_module", channel_state_b1c_by_module),
    ] {
        if let Some(states) = states {
            if states.len() != expected_modules {
                bail!(
                    "srs_review expected {name} length {expected_modules}, got {}",
                    states.len()
                );
            }
        }
    }

    if let (
        Some(time_x_shift_b1c_by_module),
        Some(time_state_b1hkk_by_module),
        Some(channel_state_b1c_by_module),
    ) = (
        time_x_shift_b1c_by_module,
        time_state_b1hkk_by_module,
        channel_state_b1c_by_module,
    ) {
        for module_index in 0..expected_modules {
            let provided = [
                !time_x_shift_b1c_by_module[module_index].is_empty(),
                !time_state_b1hkk_by_module[module_index].is_empty(),
                !channel_state_b1c_by_module[module_index].is_empty(),
            ];
            if provided.iter().any(|value| *value) && !provided.iter().all(|value| *value) {
                bail!(
                    "srs_review expected module {module_index} state to include time x-shift, time recurrent, and channel states together, or none"
                );
            }
        }
    }

    Ok(())
}

pub(super) fn validate_rnn_state_layers(
    expected_layers: usize,
    time_x_shift_b1c_by_layer: Option<&[Tensor]>,
    time_state_b1hkk_by_layer: Option<&[Tensor]>,
    channel_state_b1c_by_layer: Option<&[Tensor]>,
) -> Result<()> {
    let provided_group_count = [
        time_x_shift_b1c_by_layer.is_some(),
        time_state_b1hkk_by_layer.is_some(),
        channel_state_b1c_by_layer.is_some(),
    ]
    .into_iter()
    .filter(|provided| *provided)
    .count();

    if provided_group_count != 0 && provided_group_count != 3 {
        bail!(
            "rwkv_rnn_forward requires time_x_shift_b1c_by_layer, time_state_b1hkk_by_layer, and channel_state_b1c_by_layer together, or none"
        );
    }

    for (name, states) in [
        ("time_x_shift_b1c_by_layer", time_x_shift_b1c_by_layer),
        ("time_state_b1hkk_by_layer", time_state_b1hkk_by_layer),
        ("channel_state_b1c_by_layer", channel_state_b1c_by_layer),
    ] {
        if let Some(states) = states {
            if states.len() != expected_layers {
                bail!(
                    "rwkv_rnn_forward expected {name} length {expected_layers}, got {}",
                    states.len()
                );
            }
        }
    }

    Ok(())
}
