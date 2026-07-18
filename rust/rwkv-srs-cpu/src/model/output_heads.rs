//! Feature projection, review-module orchestration, and prediction/curve output heads.

use super::*;

pub(super) const CARD_STATE_INDEX: usize = 0;
pub(super) const NOTE_STATE_INDEX: usize = 1;
pub(super) const DECK_STATE_INDEX: usize = 2;
pub(super) const PRESET_STATE_INDEX: usize = 3;
pub(super) const GLOBAL_STATE_INDEX: usize = 4;
pub(super) const PREDICT_MANY_LIGHTNING_MODULES_ENV_VAR: &str =
    "RWKV_SRS_PREDICT_MANY_LIGHTNING_MODULES";

pub(super) fn features2card_forward_profiled(
    weights: &Features2CardWeights,
    card_features: &Tensor,
    profile: &mut Option<&mut ForwardProfile>,
) -> Result<Tensor> {
    let x = nn_ops::silu(&linear_profiled(
        card_features,
        &weights.input_linear,
        profile,
    )?)?;
    let x = layer_norm(&x, &weights.norm)?;
    nn_ops::silu(&linear_profiled(&x, &weights.output_linear, profile)?)
}

pub(super) fn srs_review_predict_forward_profiled(
    weights: &SrsRwkvRnnWeights,
    card_features: &Tensor,
    time_x_shift_b1c_by_module: Option<&[Vec<Tensor>]>,
    time_state_b1hkk_by_module: Option<&[Vec<Tensor>]>,
    channel_state_b1c_by_module: Option<&[Vec<Tensor>]>,
    mut profile: Option<&mut ForwardProfile>,
) -> Result<Tensor> {
    let total_start = ProfileTimer::start(profile.is_some());
    if let Some(profile) = profile.as_deref_mut() {
        profile.calls += 1;
    }

    let start = ProfileTimer::start(profile.is_some());
    validate_srs_review_state_modules(
        weights.rwkv_modules.len(),
        time_x_shift_b1c_by_module,
        time_state_b1hkk_by_module,
        channel_state_b1c_by_module,
    )?;
    if weights.rwkv_modules.len() != SRS_REVIEW_STATE_MODULES {
        bail!(
            "srs_review expected {SRS_REVIEW_STATE_MODULES} RWKV modules, got {}",
            weights.rwkv_modules.len()
        );
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.validate_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    let card_rwkv_input =
        features2card_forward_profiled(&weights.features2card, card_features, &mut profile)?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.features2card_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    let card_encoding = rwkv_rnn_predict_forward_profiled(
        &weights.rwkv_modules[0],
        &card_rwkv_input,
        module_state(time_x_shift_b1c_by_module, CARD_STATE_INDEX),
        module_state(time_state_b1hkk_by_module, CARD_STATE_INDEX),
        module_state(channel_state_b1c_by_module, CARD_STATE_INDEX),
        profile.as_deref_mut(),
    )?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.card_module_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    let deck_encoding = rwkv_rnn_predict_forward_profiled(
        &weights.rwkv_modules[1],
        &card_encoding,
        module_state(time_x_shift_b1c_by_module, DECK_STATE_INDEX),
        module_state(time_state_b1hkk_by_module, DECK_STATE_INDEX),
        module_state(channel_state_b1c_by_module, DECK_STATE_INDEX),
        profile.as_deref_mut(),
    )?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.deck_module_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    let note_encoding = rwkv_rnn_predict_forward_profiled(
        &weights.rwkv_modules[2],
        &deck_encoding,
        module_state(time_x_shift_b1c_by_module, NOTE_STATE_INDEX),
        module_state(time_state_b1hkk_by_module, NOTE_STATE_INDEX),
        module_state(channel_state_b1c_by_module, NOTE_STATE_INDEX),
        profile.as_deref_mut(),
    )?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.note_module_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    let preset_encoding = rwkv_rnn_predict_forward_profiled(
        &weights.rwkv_modules[3],
        &note_encoding,
        module_state(time_x_shift_b1c_by_module, PRESET_STATE_INDEX),
        module_state(time_state_b1hkk_by_module, PRESET_STATE_INDEX),
        module_state(channel_state_b1c_by_module, PRESET_STATE_INDEX),
        profile.as_deref_mut(),
    )?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.preset_module_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    let global_encoding = rwkv_rnn_predict_forward_profiled(
        &weights.rwkv_modules[4],
        &preset_encoding,
        module_state(time_x_shift_b1c_by_module, GLOBAL_STATE_INDEX),
        module_state(time_state_b1hkk_by_module, GLOBAL_STATE_INDEX),
        module_state(channel_state_b1c_by_module, GLOBAL_STATE_INDEX),
        profile.as_deref_mut(),
    )?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.global_module_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    let x = layer_norm(&global_encoding, &weights.prehead_norm)?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.prehead_norm_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    let x_p = linear_profiled(&x, &weights.head_p, &mut profile)?.relu()?;
    let out_p_logits = linear_profiled(&x_p, &weights.p_linear, &mut profile)?;
    if let Some(profile) = profile {
        profile.p_head_ns += start.elapsed_ns();
        profile.total_ns += total_start.elapsed_ns();
    }
    Ok(out_p_logits)
}

pub(super) fn predict_many_lightning_module_limit() -> Result<usize> {
    let raw =
        std::env::var(PREDICT_MANY_LIGHTNING_MODULES_ENV_VAR).unwrap_or_else(|_| "5".to_string());
    let module_limit = raw.parse::<usize>().map_err(|error| {
        candle_core::Error::msg(format!(
            "{PREDICT_MANY_LIGHTNING_MODULES_ENV_VAR} must be an integer from 0 to {SRS_REVIEW_STATE_MODULES}; got {raw:?}: {error}"
        ))
    })?;
    if module_limit > SRS_REVIEW_STATE_MODULES {
        bail!(
            "{PREDICT_MANY_LIGHTNING_MODULES_ENV_VAR} must be from 0 to {SRS_REVIEW_STATE_MODULES}; got {module_limit}"
        );
    }
    Ok(module_limit)
}

pub(super) fn srs_review_predict_forward_lightning_profiled(
    weights: &SrsRwkvRnnWeights,
    card_features: &Tensor,
    time_x_shift_b1c_by_module: Option<&[Vec<Tensor>]>,
    time_state_b1hkk_by_module: Option<&[Vec<Tensor>]>,
    channel_state_b1c_by_module: Option<&[Vec<Tensor>]>,
    profile: Option<&mut ForwardProfile>,
) -> Result<Tensor> {
    with_predict_many_lightning_approximations(|| {
        with_lightning_recurrence_approximations(|| {
            srs_review_predict_forward_lightning_profiled_inner(
                weights,
                card_features,
                time_x_shift_b1c_by_module,
                time_state_b1hkk_by_module,
                channel_state_b1c_by_module,
                profile,
            )
        })
    })
}

pub(super) fn srs_review_predict_forward_lightning_modulewise<F>(
    weights: &SrsRwkvRnnWeights,
    card_features: &Tensor,
    mut state_for_module: F,
) -> Result<Tensor>
where
    F: FnMut(usize) -> Result<(Vec<Tensor>, Vec<Tensor>, Vec<Tensor>)>,
{
    with_predict_many_lightning_approximations(|| {
        with_lightning_recurrence_approximations(|| {
            if weights.rwkv_modules.len() != SRS_REVIEW_STATE_MODULES {
                bail!(
                    "srs_review expected {SRS_REVIEW_STATE_MODULES} RWKV modules, got {}",
                    weights.rwkv_modules.len()
                );
            }
            let module_limit = predict_many_lightning_module_limit()?;
            let mut profile = None;
            let mut encoding = features2card_forward_profiled(
                &weights.features2card,
                card_features,
                &mut profile,
            )?;

            for (module_index, module_weights) in
                weights.rwkv_modules.iter().enumerate().take(module_limit)
            {
                let (time_x_shift, time_state, channel_state) = state_for_module(module_index)?;
                encoding = rwkv_rnn_predict_forward_profiled(
                    module_weights,
                    &encoding,
                    Some(time_x_shift.as_slice()),
                    Some(time_state.as_slice()),
                    Some(channel_state.as_slice()),
                    None,
                )?;
            }

            let x = layer_norm(&encoding, &weights.prehead_norm)?;
            let x_p = linear_profiled(&x, &weights.head_p, &mut profile)?.relu()?;
            linear_profiled(&x_p, &weights.p_linear, &mut profile)
        })
    })
}

pub(super) fn srs_review_predict_forward_lightning_profiled_inner(
    weights: &SrsRwkvRnnWeights,
    card_features: &Tensor,
    time_x_shift_b1c_by_module: Option<&[Vec<Tensor>]>,
    time_state_b1hkk_by_module: Option<&[Vec<Tensor>]>,
    channel_state_b1c_by_module: Option<&[Vec<Tensor>]>,
    mut profile: Option<&mut ForwardProfile>,
) -> Result<Tensor> {
    let total_start = ProfileTimer::start(profile.is_some());
    if let Some(profile) = profile.as_deref_mut() {
        profile.calls += 1;
    }

    let start = ProfileTimer::start(profile.is_some());
    validate_srs_review_state_modules(
        weights.rwkv_modules.len(),
        time_x_shift_b1c_by_module,
        time_state_b1hkk_by_module,
        channel_state_b1c_by_module,
    )?;
    if weights.rwkv_modules.len() != SRS_REVIEW_STATE_MODULES {
        bail!(
            "srs_review expected {SRS_REVIEW_STATE_MODULES} RWKV modules, got {}",
            weights.rwkv_modules.len()
        );
    }
    let module_limit = predict_many_lightning_module_limit()?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.validate_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    let card_rwkv_input =
        features2card_forward_profiled(&weights.features2card, card_features, &mut profile)?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.features2card_ns += start.elapsed_ns();
    }

    let mut encoding = card_rwkv_input;

    if module_limit >= 1 {
        let start = ProfileTimer::start(profile.is_some());
        let next = rwkv_rnn_predict_forward_profiled(
            &weights.rwkv_modules[0],
            &encoding,
            module_state(time_x_shift_b1c_by_module, CARD_STATE_INDEX),
            module_state(time_state_b1hkk_by_module, CARD_STATE_INDEX),
            module_state(channel_state_b1c_by_module, CARD_STATE_INDEX),
            profile.as_deref_mut(),
        )?;
        encoding = next;
        if let Some(profile) = profile.as_deref_mut() {
            profile.card_module_ns += start.elapsed_ns();
        }
    }

    if module_limit >= 2 {
        let start = ProfileTimer::start(profile.is_some());
        let next = rwkv_rnn_predict_forward_profiled(
            &weights.rwkv_modules[1],
            &encoding,
            module_state(time_x_shift_b1c_by_module, DECK_STATE_INDEX),
            module_state(time_state_b1hkk_by_module, DECK_STATE_INDEX),
            module_state(channel_state_b1c_by_module, DECK_STATE_INDEX),
            profile.as_deref_mut(),
        )?;
        encoding = next;
        if let Some(profile) = profile.as_deref_mut() {
            profile.deck_module_ns += start.elapsed_ns();
        }
    }

    if module_limit >= 3 {
        let start = ProfileTimer::start(profile.is_some());
        let next = rwkv_rnn_predict_forward_profiled(
            &weights.rwkv_modules[2],
            &encoding,
            module_state(time_x_shift_b1c_by_module, NOTE_STATE_INDEX),
            module_state(time_state_b1hkk_by_module, NOTE_STATE_INDEX),
            module_state(channel_state_b1c_by_module, NOTE_STATE_INDEX),
            profile.as_deref_mut(),
        )?;
        encoding = next;
        if let Some(profile) = profile.as_deref_mut() {
            profile.note_module_ns += start.elapsed_ns();
        }
    }

    if module_limit >= 4 {
        let start = ProfileTimer::start(profile.is_some());
        let next = rwkv_rnn_predict_forward_profiled(
            &weights.rwkv_modules[3],
            &encoding,
            module_state(time_x_shift_b1c_by_module, PRESET_STATE_INDEX),
            module_state(time_state_b1hkk_by_module, PRESET_STATE_INDEX),
            module_state(channel_state_b1c_by_module, PRESET_STATE_INDEX),
            profile.as_deref_mut(),
        )?;
        encoding = next;
        if let Some(profile) = profile.as_deref_mut() {
            profile.preset_module_ns += start.elapsed_ns();
        }
    }

    if module_limit >= 5 {
        let start = ProfileTimer::start(profile.is_some());
        let next = rwkv_rnn_predict_forward_profiled(
            &weights.rwkv_modules[4],
            &encoding,
            module_state(time_x_shift_b1c_by_module, GLOBAL_STATE_INDEX),
            module_state(time_state_b1hkk_by_module, GLOBAL_STATE_INDEX),
            module_state(channel_state_b1c_by_module, GLOBAL_STATE_INDEX),
            profile.as_deref_mut(),
        )?;
        encoding = next;
        if let Some(profile) = profile.as_deref_mut() {
            profile.global_module_ns += start.elapsed_ns();
        }
    }

    let start = ProfileTimer::start(profile.is_some());
    let x = layer_norm(&encoding, &weights.prehead_norm)?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.prehead_norm_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    let x_p = linear_profiled(&x, &weights.head_p, &mut profile)?.relu()?;
    let out_p_logits = linear_profiled(&x_p, &weights.p_linear, &mut profile)?;
    if let Some(profile) = profile {
        profile.p_head_ns += start.elapsed_ns();
        profile.total_ns += total_start.elapsed_ns();
    }
    Ok(out_p_logits)
}

pub fn srs_review_forward(
    weights: &SrsRwkvRnnWeights,
    card_features: &Tensor,
    time_x_shift_b1c_by_module: Option<&[Vec<Tensor>]>,
    time_state_b1hkk_by_module: Option<&[Vec<Tensor>]>,
    channel_state_b1c_by_module: Option<&[Vec<Tensor>]>,
    return_curve: bool,
) -> Result<SrsReviewOutput> {
    srs_review_forward_profiled(
        weights,
        card_features,
        time_x_shift_b1c_by_module,
        time_state_b1hkk_by_module,
        channel_state_b1c_by_module,
        return_curve,
        None,
    )
}

pub(super) fn srs_review_forward_options(
    weights: &SrsRwkvRnnWeights,
    card_features: &Tensor,
    time_x_shift_b1c_by_module: Option<&[Vec<Tensor>]>,
    time_state_b1hkk_by_module: Option<&[Vec<Tensor>]>,
    channel_state_b1c_by_module: Option<&[Vec<Tensor>]>,
    return_curve: bool,
    return_prediction: bool,
) -> Result<SrsReviewOptionalPredictionOutput> {
    srs_review_forward_profiled_options(
        weights,
        card_features,
        time_x_shift_b1c_by_module,
        time_state_b1hkk_by_module,
        channel_state_b1c_by_module,
        return_curve,
        return_prediction,
        None,
    )
}

pub(super) fn srs_review_forward_profiled(
    weights: &SrsRwkvRnnWeights,
    card_features: &Tensor,
    time_x_shift_b1c_by_module: Option<&[Vec<Tensor>]>,
    time_state_b1hkk_by_module: Option<&[Vec<Tensor>]>,
    channel_state_b1c_by_module: Option<&[Vec<Tensor>]>,
    return_curve: bool,
    mut profile: Option<&mut ForwardProfile>,
) -> Result<SrsReviewOutput> {
    let (
        out_ahead_logits,
        out_w,
        out_p_logits,
        next_time_x_shift_b1c_by_module,
        next_time_state_b1hkk_by_module,
        next_channel_state_b1c_by_module,
    ) = srs_review_forward_profiled_options(
        weights,
        card_features,
        time_x_shift_b1c_by_module,
        time_state_b1hkk_by_module,
        channel_state_b1c_by_module,
        return_curve,
        true,
        profile.take(),
    )?;
    let out_p_logits =
        out_p_logits.expect("prediction logits are present when return_prediction=true");
    Ok((
        out_ahead_logits,
        out_w,
        out_p_logits,
        next_time_x_shift_b1c_by_module,
        next_time_state_b1hkk_by_module,
        next_channel_state_b1c_by_module,
    ))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn srs_review_forward_profiled_options(
    weights: &SrsRwkvRnnWeights,
    card_features: &Tensor,
    time_x_shift_b1c_by_module: Option<&[Vec<Tensor>]>,
    time_state_b1hkk_by_module: Option<&[Vec<Tensor>]>,
    channel_state_b1c_by_module: Option<&[Vec<Tensor>]>,
    return_curve: bool,
    return_prediction: bool,
    mut profile: Option<&mut ForwardProfile>,
) -> Result<SrsReviewOptionalPredictionOutput> {
    let total_start = ProfileTimer::start(profile.is_some());
    if let Some(profile) = profile.as_deref_mut() {
        profile.calls += 1;
    }

    let start = ProfileTimer::start(profile.is_some());
    validate_srs_review_state_modules(
        weights.rwkv_modules.len(),
        time_x_shift_b1c_by_module,
        time_state_b1hkk_by_module,
        channel_state_b1c_by_module,
    )?;
    if weights.rwkv_modules.len() != SRS_REVIEW_STATE_MODULES {
        bail!(
            "srs_review expected {SRS_REVIEW_STATE_MODULES} RWKV modules, got {}",
            weights.rwkv_modules.len()
        );
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.validate_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    let card_rwkv_input =
        features2card_forward_profiled(&weights.features2card, card_features, &mut profile)?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.features2card_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    let (card_encoding, card_time_x, card_time_state, card_channel_state) =
        rwkv_rnn_forward_profiled(
            &weights.rwkv_modules[0],
            &card_rwkv_input,
            module_state(time_x_shift_b1c_by_module, CARD_STATE_INDEX),
            module_state(time_state_b1hkk_by_module, CARD_STATE_INDEX),
            module_state(channel_state_b1c_by_module, CARD_STATE_INDEX),
            profile.as_deref_mut(),
        )?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.card_module_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    let (deck_encoding, deck_time_x, deck_time_state, deck_channel_state) =
        rwkv_rnn_forward_profiled(
            &weights.rwkv_modules[1],
            &card_encoding,
            module_state(time_x_shift_b1c_by_module, DECK_STATE_INDEX),
            module_state(time_state_b1hkk_by_module, DECK_STATE_INDEX),
            module_state(channel_state_b1c_by_module, DECK_STATE_INDEX),
            profile.as_deref_mut(),
        )?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.deck_module_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    let (note_encoding, note_time_x, note_time_state, note_channel_state) =
        rwkv_rnn_forward_profiled(
            &weights.rwkv_modules[2],
            &deck_encoding,
            module_state(time_x_shift_b1c_by_module, NOTE_STATE_INDEX),
            module_state(time_state_b1hkk_by_module, NOTE_STATE_INDEX),
            module_state(channel_state_b1c_by_module, NOTE_STATE_INDEX),
            profile.as_deref_mut(),
        )?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.note_module_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    let (preset_encoding, preset_time_x, preset_time_state, preset_channel_state) =
        rwkv_rnn_forward_profiled(
            &weights.rwkv_modules[3],
            &note_encoding,
            module_state(time_x_shift_b1c_by_module, PRESET_STATE_INDEX),
            module_state(time_state_b1hkk_by_module, PRESET_STATE_INDEX),
            module_state(channel_state_b1c_by_module, PRESET_STATE_INDEX),
            profile.as_deref_mut(),
        )?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.preset_module_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    let (global_encoding, global_time_x, global_time_state, global_channel_state) =
        rwkv_rnn_forward_profiled(
            &weights.rwkv_modules[4],
            &preset_encoding,
            module_state(time_x_shift_b1c_by_module, GLOBAL_STATE_INDEX),
            module_state(time_state_b1hkk_by_module, GLOBAL_STATE_INDEX),
            module_state(channel_state_b1c_by_module, GLOBAL_STATE_INDEX),
            profile.as_deref_mut(),
        )?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.global_module_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    let x = layer_norm(&global_encoding, &weights.prehead_norm)?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.prehead_norm_ns += start.elapsed_ns();
    }

    let (out_ahead_logits, out_w) = if return_curve {
        let start = ProfileTimer::start(profile.is_some());
        let x_w = w_head_forward_profiled(&weights.head_w, &x, &mut profile)?;
        let out_w_logits = linear_profiled(&x_w, &weights.w_linear, &mut profile)?;
        let out_w = nn_ops::softmax(&out_w_logits, D::Minus1)?;

        let ahead_hidden = linear_profiled(&x, &weights.head_ahead_logits, &mut profile)?;
        let x_ahead = linear_profiled(&ahead_hidden.relu()?, &weights.ahead_linear, &mut profile)?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.curve_heads_ns += start.elapsed_ns();
        }
        (Some(x_ahead), Some(out_w))
    } else {
        (None, None)
    };

    let out_p_logits = if return_prediction {
        let start = ProfileTimer::start(profile.is_some());
        let x_p = linear_profiled(&x, &weights.head_p, &mut profile)?.relu()?;
        let out_p_logits = linear_profiled(&x_p, &weights.p_linear, &mut profile)?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.p_head_ns += start.elapsed_ns();
        }
        Some(out_p_logits)
    } else {
        None
    };

    let start = ProfileTimer::start(profile.is_some());
    let output = (
        out_ahead_logits,
        out_w,
        out_p_logits,
        vec![
            card_time_x,
            note_time_x,
            deck_time_x,
            preset_time_x,
            global_time_x,
        ],
        vec![
            card_time_state,
            note_time_state,
            deck_time_state,
            preset_time_state,
            global_time_state,
        ],
        vec![
            card_channel_state,
            note_channel_state,
            deck_channel_state,
            preset_channel_state,
            global_channel_state,
        ],
    );
    if let Some(profile) = profile {
        profile.pack_state_ns += start.elapsed_ns();
        profile.total_ns += total_start.elapsed_ns();
    }

    Ok(output)
}

pub(super) fn w_head_forward_profiled(
    weights: &WHeadWeights,
    xs: &Tensor,
    profile: &mut Option<&mut ForwardProfile>,
) -> Result<Tensor> {
    let x = linear_profiled(xs, &weights.input_linear, profile)?.relu()?;
    let x = layer_norm(&x, &weights.norm)?;
    linear_profiled(&x, &weights.output_linear, profile)
}

pub(super) fn module_state(
    states: Option<&[Vec<Tensor>]>,
    review_state_index: usize,
) -> Option<&[Tensor]> {
    states.and_then(|states| {
        let module_state = &states[review_state_index];
        if module_state.is_empty() {
            None
        } else {
            Some(module_state.as_slice())
        }
    })
}
