#![allow(clippy::useless_conversion)]

use std::{
    cell::RefCell,
    collections::{BTreeMap, VecDeque},
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        OnceLock,
    },
};

use candle_core::{bail, DType, Device, Result, Shape, Tensor, D};
use candle_nn::ops as nn_ops;
use candle_nn::{Linear, Module};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) use self::bindings::{
    channel_mixer_forward_py, prediction_probability_py, rwkv_layer_forward_py,
    rwkv_rnn_forward_py, srs_review_py, time_mixer_forward_py,
};
use self::live_session::LivePredictionSessionState;
pub(crate) use self::prediction_batch::NativePredictionBatch;
pub(crate) use self::review_batch::NativeReviewBatch;
use self::state::{
    append_state_input, batched_module_state, existing_state,
    native_module_states_from_review_output, push_batched_module_state, FlatNativeRnnModuleState,
    FlatNativeRnnState, NativeRnnModuleState, ReviewIds, SrsReviewStateInput,
};
use self::undo::RuntimeUndoFrame;
use self::validation::{
    validate_channel_mixer_shapes, validate_rnn_state_layers, validate_srs_review_state_modules,
    validate_time_mixer_shapes, validate_time_mixer_state_shapes,
};
use crate::cpu_config::*;
use crate::model_weights::{
    load_srs_rwkv_rnn_weights, ChannelMixerNativeF32Weights, Features2CardWeights,
    LayerNormWeights, LinearBlockedInput8Dot12Weights, LinearF32Weights, LinearWeights,
    ModelWeightsError, Rwkv7RnnChannelMixerWeights, Rwkv7RnnLayerWeights, Rwkv7RnnTimeMixerWeights,
    Rwkv7RnnWeights, SrsRwkvRnnWeights, TimeMixerNativeF32Weights, WHeadWeights,
};
use crate::ops::{
    f32_tensor_data, group_norm_2d, l2_normalize_scaled_b1hk, layer_norm_last_dim,
    recycle_time_mixer_middle_scratch_output, single_timestep_b1c_profiled,
    single_timestep_output_b1c_profiled, single_timestep_output_profiled, single_timestep_profiled,
    time_decay_w_b1c, time_decay_w_scalar, time_lerp_parts_b1c,
    time_mixer_middle_scratch_all_values_flat_state_profiled,
    time_mixer_middle_scratch_all_values_profiled, time_mixer_middle_scratch_profiled,
    time_mixer_middle_scratch_values_profiled, time_mixer_middle_scratch_wav_values_profiled,
    with_lightning_recurrence_approximations, TimeMixerMiddleScratchOutput,
};
use crate::profile::{ForwardProfile, ProfileTimer, StateInputProfile};
use crate::state::FeatureState;
use crate::tensor_io::Tensor2List;

#[cfg(target_arch = "x86")]
use std::arch::x86 as x86_arch;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64 as x86_arch;

mod bindings;
mod bulk;
mod channel_mixer;
mod checkpoint_bin;
mod gpu;
mod gpu_process_scan;
mod linear;
mod live_session;
mod output_heads;
mod pipeline;
mod prediction_batch;
mod process_payload;
mod review_batch;
mod runtime;
mod state;
mod state_builder;
mod time_mixer;
mod undo;
mod validation;

use self::channel_mixer::*;
use self::linear::*;
use self::output_heads::*;
use self::time_mixer::*;

type RwkvRnnOutput = (Tensor, Vec<Tensor>, Vec<Tensor>, Vec<Tensor>);
type SrsReviewTensorState = Vec<Vec<Tensor>>;
type SrsReviewOutput = (
    Option<Tensor>,
    Option<Tensor>,
    Tensor,
    SrsReviewTensorState,
    SrsReviewTensorState,
    SrsReviewTensorState,
);
type SrsReviewOptionalPredictionOutput = (
    Option<Tensor>,
    Option<Tensor>,
    Option<Tensor>,
    SrsReviewTensorState,
    SrsReviewTensorState,
    SrsReviewTensorState,
);
type NativeProcessManyPyOutput = (Vec<f64>, Option<Vec<Tensor2List>>, Option<Vec<Tensor2List>>);

const LAYER_NORM_EPS: f32 = 1e-5;
const SRS_REVIEW_STATE_MODULES: usize = 5;
// Avoid repeated state Tensor materialization for interactive predict_many bursts.
const SHORT_BURST_BROADCAST_MAX_BATCH: usize = 192;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PredictManyForwardMode {
    Oracle,
    Lightning,
}

impl PredictManyForwardMode {
    pub(super) fn from_lightning(lightning: bool) -> Self {
        if lightning {
            Self::Lightning
        } else {
            Self::Oracle
        }
    }

    pub(super) fn is_lightning(self) -> bool {
        matches!(self, Self::Lightning)
    }

    pub(super) fn requires_state(self) -> Result<bool> {
        match self {
            Self::Oracle => Ok(true),
            Self::Lightning => Ok(predict_many_lightning_module_limit()? > 0),
        }
    }
}

#[pyclass(unsendable)]
pub struct NativeRnn {
    weights: SrsRwkvRnnWeights,
    card_states: BTreeMap<i64, NativeRnnModuleState>,
    note_states: BTreeMap<i64, NativeRnnModuleState>,
    deck_states: BTreeMap<i64, NativeRnnModuleState>,
    preset_states: BTreeMap<i64, NativeRnnModuleState>,
    global_state: Option<NativeRnnModuleState>,
    flat_cpu_state: FlatNativeRnnState,
    gpu: Option<gpu::GpuPredictor>,
    gpu_process_scan: Option<gpu_process_scan::GpuProcessScan>,
}

#[pyclass(unsendable)]
pub struct NativeRuntime {
    deterministic: FeatureState,
    rnn: NativeRnn,
    undo_stack: VecDeque<RuntimeUndoFrame>,
    undo_limit: usize,
    loaded_scope: Option<checkpoint_bin::CheckpointScope>,
    gpu_process_committed_rows: usize,
    gpu_process_output: gpu_process_scan::GpuScanProcessOutput,
    live_session: Option<LivePredictionSessionState>,
    pending_live_session: Option<LivePredictionSessionState>,
    next_live_session_token: u64,
}

impl NativeRnn {
    fn from_checkpoint(checkpoint_path: PathBuf) -> std::result::Result<Self, ModelWeightsError> {
        let weights = load_srs_rwkv_rnn_weights(checkpoint_path)?;
        Ok(Self::from_weights(weights))
    }

    fn from_weights(weights: SrsRwkvRnnWeights) -> Self {
        Self {
            weights,
            card_states: BTreeMap::new(),
            note_states: BTreeMap::new(),
            deck_states: BTreeMap::new(),
            preset_states: BTreeMap::new(),
            global_state: None,
            flat_cpu_state: FlatNativeRnnState::default(),
            gpu: None,
            gpu_process_scan: None,
        }
    }

    fn materialize_flat_cpu_state(&mut self) -> Result<()> {
        fn materialize_map(
            flat: &mut BTreeMap<i64, FlatNativeRnnModuleState>,
            canonical: &mut BTreeMap<i64, NativeRnnModuleState>,
            name: &str,
        ) -> Result<()> {
            for identity in flat.keys() {
                if canonical.contains_key(identity) {
                    bail!("{name} identity {identity} exists in flat and canonical CPU state");
                }
            }
            let mut identities = flat
                .iter()
                .map(|(identity, state)| (state.storage_id(), *identity))
                .collect::<Vec<_>>();
            identities.sort_unstable();
            for (_, identity) in identities {
                let state = flat.remove(&identity).expect("selected flat state exists");
                match state.materialize() {
                    Ok(state) => {
                        canonical.insert(identity, state);
                    }
                    Err(error) => {
                        flat.insert(identity, state);
                        return Err(error);
                    }
                }
            }
            Ok(())
        }

        materialize_map(
            &mut self.flat_cpu_state.card_states,
            &mut self.card_states,
            "card_states",
        )?;
        materialize_map(
            &mut self.flat_cpu_state.note_states,
            &mut self.note_states,
            "note_states",
        )?;
        materialize_map(
            &mut self.flat_cpu_state.deck_states,
            &mut self.deck_states,
            "deck_states",
        )?;
        materialize_map(
            &mut self.flat_cpu_state.preset_states,
            &mut self.preset_states,
            "preset_states",
        )?;
        if let Some(flat) = self.flat_cpu_state.global_state.as_ref() {
            if self.global_state.is_some() {
                bail!("global state exists in flat and canonical CPU state");
            }
            self.global_state = Some(flat.materialize()?);
            self.flat_cpu_state.global_state = None;
        }
        Ok(())
    }

    fn state_inputs(
        &self,
        card_id: i64,
        note_id: i64,
        deck_id: i64,
        preset_id: i64,
    ) -> SrsReviewStateInput {
        let states = [
            self.card_states.get(&card_id),
            self.note_states.get(&note_id),
            self.deck_states.get(&deck_id),
            self.preset_states.get(&preset_id),
            self.global_state.as_ref(),
        ];
        let mut time_x_shift_b1c_by_module = Vec::with_capacity(SRS_REVIEW_STATE_MODULES);
        let mut time_state_b1hkk_by_module = Vec::with_capacity(SRS_REVIEW_STATE_MODULES);
        let mut channel_state_b1c_by_module = Vec::with_capacity(SRS_REVIEW_STATE_MODULES);
        for state in states {
            append_state_input(
                state,
                &mut time_x_shift_b1c_by_module,
                &mut time_state_b1hkk_by_module,
                &mut channel_state_b1c_by_module,
            );
        }
        (
            time_x_shift_b1c_by_module,
            time_state_b1hkk_by_module,
            channel_state_b1c_by_module,
        )
    }

    fn batch_state_inputs(
        &self,
        ids: &[ReviewIds],
        allow_short_burst_broadcast: bool,
        mut profile: Option<&mut StateInputProfile>,
    ) -> Result<SrsReviewStateInput> {
        if ids.is_empty() {
            bail!("batch_state_inputs requires at least one review id tuple");
        }
        if let Some(profile) = profile.as_deref_mut() {
            profile.calls += 1;
            profile.rows += ids.len();
        }

        let mut card_states = Vec::with_capacity(ids.len());
        let mut note_states = Vec::with_capacity(ids.len());
        let mut deck_states = Vec::with_capacity(ids.len());
        let mut preset_states = Vec::with_capacity(ids.len());
        let lookup_start = ProfileTimer::start(profile.is_some());
        let global_state = self
            .global_state
            .as_ref()
            .ok_or_else(|| candle_core::Error::msg("missing global recurrent state"))?;
        let mut global_states = Vec::with_capacity(ids.len());

        for (card_id, note_id, deck_id, preset_id) in ids {
            card_states.push(existing_state(&self.card_states, *card_id, "card_states")?);
            note_states.push(existing_state(&self.note_states, *note_id, "note_states")?);
            deck_states.push(existing_state(&self.deck_states, *deck_id, "deck_states")?);
            preset_states.push(existing_state(
                &self.preset_states,
                *preset_id,
                "preset_states",
            )?);
            global_states.push(global_state);
        }
        if let Some(profile) = profile.as_deref_mut() {
            profile.lookup_ns += lookup_start.elapsed_ns();
        }

        let mut time_x_shift_b1c_by_module = Vec::with_capacity(SRS_REVIEW_STATE_MODULES);
        let mut time_state_b1hkk_by_module = Vec::with_capacity(SRS_REVIEW_STATE_MODULES);
        let mut channel_state_b1c_by_module = Vec::with_capacity(SRS_REVIEW_STATE_MODULES);
        push_batched_module_state(
            &card_states,
            "card_states",
            &mut time_x_shift_b1c_by_module,
            &mut time_state_b1hkk_by_module,
            &mut channel_state_b1c_by_module,
            allow_short_burst_broadcast,
            profile.as_deref_mut(),
        )?;
        push_batched_module_state(
            &note_states,
            "note_states",
            &mut time_x_shift_b1c_by_module,
            &mut time_state_b1hkk_by_module,
            &mut channel_state_b1c_by_module,
            allow_short_burst_broadcast,
            profile.as_deref_mut(),
        )?;
        push_batched_module_state(
            &deck_states,
            "deck_states",
            &mut time_x_shift_b1c_by_module,
            &mut time_state_b1hkk_by_module,
            &mut channel_state_b1c_by_module,
            allow_short_burst_broadcast,
            profile.as_deref_mut(),
        )?;
        push_batched_module_state(
            &preset_states,
            "preset_states",
            &mut time_x_shift_b1c_by_module,
            &mut time_state_b1hkk_by_module,
            &mut channel_state_b1c_by_module,
            allow_short_burst_broadcast,
            profile.as_deref_mut(),
        )?;
        push_batched_module_state(
            &global_states,
            "global_state",
            &mut time_x_shift_b1c_by_module,
            &mut time_state_b1hkk_by_module,
            &mut channel_state_b1c_by_module,
            allow_short_burst_broadcast,
            profile,
        )?;

        Ok((
            time_x_shift_b1c_by_module,
            time_state_b1hkk_by_module,
            channel_state_b1c_by_module,
        ))
    }

    fn batch_module_state_inputs(
        &self,
        ids: &[ReviewIds],
        module_index: usize,
        allow_short_burst_broadcast: bool,
    ) -> Result<(Vec<Tensor>, Vec<Tensor>, Vec<Tensor>)> {
        if ids.is_empty() {
            bail!("batch_module_state_inputs requires at least one review id tuple");
        }
        let mut states = Vec::with_capacity(ids.len());
        let name = match module_index {
            0 => {
                for (card_id, _, _, _) in ids {
                    states.push(existing_state(&self.card_states, *card_id, "card_states")?);
                }
                "card_states"
            }
            1 => {
                for (_, _, deck_id, _) in ids {
                    states.push(existing_state(&self.deck_states, *deck_id, "deck_states")?);
                }
                "deck_states"
            }
            2 => {
                for (_, note_id, _, _) in ids {
                    states.push(existing_state(&self.note_states, *note_id, "note_states")?);
                }
                "note_states"
            }
            3 => {
                for (_, _, _, preset_id) in ids {
                    states.push(existing_state(
                        &self.preset_states,
                        *preset_id,
                        "preset_states",
                    )?);
                }
                "preset_states"
            }
            4 => {
                let global_state = self
                    .global_state
                    .as_ref()
                    .ok_or_else(|| candle_core::Error::msg("missing global recurrent state"))?;
                states.resize(ids.len(), global_state);
                "global_state"
            }
            _ => bail!("invalid RWKV module index {module_index}"),
        };
        batched_module_state(&states, name, allow_short_burst_broadcast, None)
    }

    fn warm_predict_path(&mut self) -> Result<()> {
        self.materialize_flat_cpu_state()?;
        let (_, feature_dim) = self.weights.features2card.input_linear.weight.dims2()?;
        let features = Tensor::zeros((1usize, feature_dim), DType::F32, &Device::Cpu)?;
        let card_id = self.card_states.keys().next().copied().unwrap_or(i64::MIN);
        let note_id = self.note_states.keys().next().copied().unwrap_or(i64::MIN);
        let deck_id = self.deck_states.keys().next().copied().unwrap_or(i64::MIN);
        let preset_id = self
            .preset_states
            .keys()
            .next()
            .copied()
            .unwrap_or(i64::MIN);
        let (time_x_shift_b1c_by_module, time_state_b1hkk_by_module, channel_state_b1c_by_module) =
            self.state_inputs(card_id, note_id, deck_id, preset_id);
        let _ = srs_review_forward(
            &self.weights,
            &features,
            Some(&time_x_shift_b1c_by_module),
            Some(&time_state_b1hkk_by_module),
            Some(&channel_state_b1c_by_module),
            false,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn store_review_states(
        &mut self,
        card_id: i64,
        note_id: i64,
        deck_id: i64,
        preset_id: i64,
        time_x_shift_b1c_by_module: Vec<Vec<Tensor>>,
        time_state_b1hkk_by_module: Vec<Vec<Tensor>>,
        channel_state_b1c_by_module: Vec<Vec<Tensor>>,
    ) -> Result<()> {
        let mut states = native_module_states_from_review_output(
            time_x_shift_b1c_by_module,
            time_state_b1hkk_by_module,
            channel_state_b1c_by_module,
        )?;
        let global_state = states.pop().expect("review output has five states");
        let preset_state = states.pop().expect("review output has five states");
        let deck_state = states.pop().expect("review output has five states");
        let note_state = states.pop().expect("review output has five states");
        let card_state = states.pop().expect("review output has five states");

        self.flat_cpu_state.card_states.remove(&card_id);
        self.flat_cpu_state.note_states.remove(&note_id);
        self.flat_cpu_state.deck_states.remove(&deck_id);
        self.flat_cpu_state.preset_states.remove(&preset_id);
        self.flat_cpu_state.global_state = None;
        self.card_states.insert(card_id, card_state);
        self.note_states.insert(note_id, note_state);
        self.deck_states.insert(deck_id, deck_state);
        self.preset_states.insert(preset_id, preset_state);
        self.global_state = Some(global_state);
        self.sync_gpu_review_states(card_id, note_id, deck_id, preset_id)
            .map_err(|error| candle_core::Error::msg(error.to_string()))?;
        Ok(())
    }
}

pub fn rwkv_layer_forward(
    weights: &Rwkv7RnnLayerWeights,
    in_bc: &Tensor,
    v0_bc: &Tensor,
    time_state: Option<(&Tensor, &Tensor)>,
    channel_state_b1c: Option<&Tensor>,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    rwkv_layer_forward_profiled(weights, in_bc, v0_bc, time_state, channel_state_b1c, None)
}

fn rwkv_layer_forward_profiled(
    weights: &Rwkv7RnnLayerWeights,
    in_bc: &Tensor,
    v0_bc: &Tensor,
    time_state: Option<(&Tensor, &Tensor)>,
    channel_state_b1c: Option<&Tensor>,
    mut profile: Option<&mut ForwardProfile>,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let total_start = ProfileTimer::start(profile.is_some());
    profile_inc!(profile, layer.calls);

    let start = ProfileTimer::start(profile.is_some());
    let (x_bc, next_v0_bc, next_time_x_b1c, next_time_state_b1hkk) = time_mixer_forward_profiled(
        &weights.time_mixer,
        in_bc,
        v0_bc,
        time_state,
        profile.as_deref_mut(),
    )?;
    profile_add!(profile, layer.time_mixer_ns, start.elapsed_ns());

    let start = ProfileTimer::start(profile.is_some());
    let (out_bc, next_channel_state_b1c) = channel_mixer_forward_profiled(
        &weights.channel_mixer,
        &x_bc,
        channel_state_b1c,
        profile.as_deref_mut(),
    )?;
    profile_add!(profile, layer.channel_mixer_ns, start.elapsed_ns());
    profile_add!(profile, layer.total_ns, total_start.elapsed_ns());

    Ok((
        out_bc,
        next_v0_bc,
        next_time_x_b1c,
        next_time_state_b1hkk,
        next_channel_state_b1c,
    ))
}

fn rwkv_layer_predict_forward_profiled(
    weights: &Rwkv7RnnLayerWeights,
    in_bc: &Tensor,
    v0_bc: &Tensor,
    time_state: Option<(&Tensor, &Tensor)>,
    channel_state_b1c: Option<&Tensor>,
    mut profile: Option<&mut ForwardProfile>,
) -> Result<(Tensor, Tensor)> {
    let total_start = ProfileTimer::start(profile.is_some());
    profile_inc!(profile, layer.calls);

    let start = ProfileTimer::start(profile.is_some());
    let (x_bc, next_v0_bc, _, _) = time_mixer_forward_profiled_options(
        &weights.time_mixer,
        in_bc,
        v0_bc,
        time_state,
        false,
        profile.as_deref_mut(),
    )?;
    profile_add!(profile, layer.time_mixer_ns, start.elapsed_ns());

    let start = ProfileTimer::start(profile.is_some());
    let (out_bc, _) = channel_mixer_forward_profiled(
        &weights.channel_mixer,
        &x_bc,
        channel_state_b1c,
        profile.as_deref_mut(),
    )?;
    profile_add!(profile, layer.channel_mixer_ns, start.elapsed_ns());
    profile_add!(profile, layer.total_ns, total_start.elapsed_ns());

    Ok((out_bc, next_v0_bc))
}

fn rwkv_layer_forward_flat_time_state_profiled(
    weights: &Rwkv7RnnLayerWeights,
    in_bc: &Tensor,
    v0_bc: &Tensor,
    time_state: Option<&TimeMixerFlatLayerState>,
    channel_state_b1c: Option<&Tensor>,
    mut profile: Option<&mut ForwardProfile>,
) -> Result<Option<(Tensor, Tensor, TimeMixerFlatLayerState, Tensor)>> {
    let total_start = ProfileTimer::start(profile.is_some());
    profile_inc!(profile, layer.calls);

    let start = ProfileTimer::start(profile.is_some());
    let Some((x_bc, next_v0_bc, next_time_state)) = time_mixer_forward_flat_state_profiled_options(
        &weights.time_mixer,
        in_bc,
        v0_bc,
        time_state,
        true,
        profile.as_deref_mut(),
    )?
    else {
        return Ok(None);
    };
    let next_time_state =
        next_time_state.expect("state-producing flat time mixer returns flat state");
    profile_add!(profile, layer.time_mixer_ns, start.elapsed_ns());

    let start = ProfileTimer::start(profile.is_some());
    let (out_bc, next_channel_state_b1c) = channel_mixer_forward_profiled(
        &weights.channel_mixer,
        &x_bc,
        channel_state_b1c,
        profile.as_deref_mut(),
    )?;
    profile_add!(profile, layer.channel_mixer_ns, start.elapsed_ns());
    profile_add!(profile, layer.total_ns, total_start.elapsed_ns());

    Ok(Some((
        out_bc,
        next_v0_bc,
        next_time_state,
        next_channel_state_b1c,
    )))
}

fn rwkv_layer_predict_forward_flat_time_state_profiled(
    weights: &Rwkv7RnnLayerWeights,
    in_bc: &Tensor,
    v0_bc: &Tensor,
    time_state: Option<&TimeMixerFlatLayerState>,
    channel_state_b1c: Option<&Tensor>,
    mut profile: Option<&mut ForwardProfile>,
) -> Result<Option<(Tensor, Tensor)>> {
    let total_start = ProfileTimer::start(profile.is_some());
    profile_inc!(profile, layer.calls);

    let start = ProfileTimer::start(profile.is_some());
    let Some((x_bc, next_v0_bc, _next_time_state)) =
        time_mixer_forward_flat_state_profiled_options(
            &weights.time_mixer,
            in_bc,
            v0_bc,
            time_state,
            false,
            profile.as_deref_mut(),
        )?
    else {
        return Ok(None);
    };
    profile_add!(profile, layer.time_mixer_ns, start.elapsed_ns());

    let start = ProfileTimer::start(profile.is_some());
    let (out_bc, _) = channel_mixer_forward_profiled(
        &weights.channel_mixer,
        &x_bc,
        channel_state_b1c,
        profile.as_deref_mut(),
    )?;
    profile_add!(profile, layer.channel_mixer_ns, start.elapsed_ns());
    profile_add!(profile, layer.total_ns, total_start.elapsed_ns());

    Ok(Some((out_bc, next_v0_bc)))
}

fn rwkv_layer_forward_flat_working_state_profiled(
    weights: &Rwkv7RnnLayerWeights,
    in_bc: &Tensor,
    v0_bc: &Tensor,
    time_state: Option<&TimeMixerFlatLayerState>,
    channel_state: Option<&ChannelMixerFlatLayerState>,
    mut profile: Option<&mut ForwardProfile>,
) -> Result<
    Option<(
        Tensor,
        Tensor,
        TimeMixerFlatLayerState,
        ChannelMixerFlatLayerState,
    )>,
> {
    let total_start = ProfileTimer::start(profile.is_some());
    profile_inc!(profile, layer.calls);

    if native_layer_time_channel_values_handoff_enabled() {
        let Some(dot_kernel) = native_linear_dot_kernel() else {
            return Ok(None);
        };
        let (batch_size, channels) = in_bc.dims2()?;
        let start = ProfileTimer::start(profile.is_some());
        let Some((x_values, next_v0_bc, next_time_state)) =
            time_mixer_forward_flat_state_values_raw_output_profiled_options(
                &weights.time_mixer,
                in_bc,
                v0_bc,
                time_state,
                true,
                dot_kernel,
                profile.as_deref_mut(),
            )?
        else {
            return Ok(None);
        };
        let next_time_state =
            next_time_state.expect("state-producing flat time mixer returns flat state");
        profile_add!(profile, layer.time_mixer_ns, start.elapsed_ns());

        let start = ProfileTimer::start(profile.is_some());
        let Some((out_bc, next_channel_state)) =
            channel_mixer_forward_flat_state_from_input_values_profiled(
                &weights.channel_mixer,
                &x_values,
                batch_size,
                channels,
                in_bc.device(),
                channel_state,
                true,
                profile.as_deref_mut(),
            )?
        else {
            return Ok(None);
        };
        let next_channel_state =
            next_channel_state.expect("state-producing flat channel mixer returns flat state");
        profile_add!(profile, layer.channel_mixer_ns, start.elapsed_ns());
        profile_add!(profile, layer.total_ns, total_start.elapsed_ns());

        return Ok(Some((
            out_bc,
            next_v0_bc,
            next_time_state,
            next_channel_state,
        )));
    }

    let start = ProfileTimer::start(profile.is_some());
    let Some((x_bc, next_v0_bc, next_time_state)) = time_mixer_forward_flat_state_profiled_options(
        &weights.time_mixer,
        in_bc,
        v0_bc,
        time_state,
        true,
        profile.as_deref_mut(),
    )?
    else {
        return Ok(None);
    };
    let next_time_state =
        next_time_state.expect("state-producing flat time mixer returns flat state");
    profile_add!(profile, layer.time_mixer_ns, start.elapsed_ns());

    let start = ProfileTimer::start(profile.is_some());
    let Some((out_bc, next_channel_state)) = channel_mixer_forward_flat_state_profiled(
        &weights.channel_mixer,
        &x_bc,
        channel_state,
        true,
        profile.as_deref_mut(),
    )?
    else {
        return Ok(None);
    };
    let next_channel_state =
        next_channel_state.expect("state-producing flat channel mixer returns flat state");
    profile_add!(profile, layer.channel_mixer_ns, start.elapsed_ns());
    profile_add!(profile, layer.total_ns, total_start.elapsed_ns());

    Ok(Some((
        out_bc,
        next_v0_bc,
        next_time_state,
        next_channel_state,
    )))
}

fn rwkv_layer_predict_forward_flat_working_state_profiled(
    weights: &Rwkv7RnnLayerWeights,
    in_bc: &Tensor,
    v0_bc: &Tensor,
    time_state: Option<&TimeMixerFlatLayerState>,
    channel_state: Option<&ChannelMixerFlatLayerState>,
    mut profile: Option<&mut ForwardProfile>,
) -> Result<Option<(Tensor, Tensor)>> {
    let total_start = ProfileTimer::start(profile.is_some());
    profile_inc!(profile, layer.calls);

    if native_layer_time_channel_values_handoff_enabled() {
        let Some(dot_kernel) = native_linear_dot_kernel() else {
            return Ok(None);
        };
        let (batch_size, channels) = in_bc.dims2()?;
        let start = ProfileTimer::start(profile.is_some());
        let Some((x_values, next_v0_bc, _next_time_state)) =
            time_mixer_forward_flat_state_values_raw_output_profiled_options(
                &weights.time_mixer,
                in_bc,
                v0_bc,
                time_state,
                false,
                dot_kernel,
                profile.as_deref_mut(),
            )?
        else {
            return Ok(None);
        };
        profile_add!(profile, layer.time_mixer_ns, start.elapsed_ns());

        let start = ProfileTimer::start(profile.is_some());
        let Some((out_bc, _next_channel_state)) =
            channel_mixer_forward_flat_state_from_input_values_profiled(
                &weights.channel_mixer,
                &x_values,
                batch_size,
                channels,
                in_bc.device(),
                channel_state,
                false,
                profile.as_deref_mut(),
            )?
        else {
            return Ok(None);
        };
        profile_add!(profile, layer.channel_mixer_ns, start.elapsed_ns());
        profile_add!(profile, layer.total_ns, total_start.elapsed_ns());

        return Ok(Some((out_bc, next_v0_bc)));
    }

    let start = ProfileTimer::start(profile.is_some());
    let Some((x_bc, next_v0_bc, _next_time_state)) =
        time_mixer_forward_flat_state_profiled_options(
            &weights.time_mixer,
            in_bc,
            v0_bc,
            time_state,
            false,
            profile.as_deref_mut(),
        )?
    else {
        return Ok(None);
    };
    profile_add!(profile, layer.time_mixer_ns, start.elapsed_ns());

    let start = ProfileTimer::start(profile.is_some());
    let Some((out_bc, _next_channel_state)) = channel_mixer_forward_flat_state_profiled(
        &weights.channel_mixer,
        &x_bc,
        channel_state,
        false,
        profile.as_deref_mut(),
    )?
    else {
        return Ok(None);
    };
    profile_add!(profile, layer.channel_mixer_ns, start.elapsed_ns());
    profile_add!(profile, layer.total_ns, total_start.elapsed_ns());

    Ok(Some((out_bc, next_v0_bc)))
}

#[allow(clippy::too_many_arguments)]
fn rwkv_layer_forward_flat_working_state_values_profiled(
    weights: &Rwkv7RnnLayerWeights,
    input_values: &[f32],
    batch_size: usize,
    channels: usize,
    device: &Device,
    v0_values: &[f32],
    time_state: Option<&TimeMixerFlatLayerState>,
    channel_state: Option<&ChannelMixerFlatLayerState>,
    mut profile: Option<&mut ForwardProfile>,
) -> Result<
    Option<(
        Vec<f32>,
        Option<Vec<f32>>,
        TimeMixerFlatLayerState,
        ChannelMixerFlatLayerState,
    )>,
> {
    let total_start = ProfileTimer::start(profile.is_some());
    profile_inc!(profile, layer.calls);

    let Some(dot_kernel) = native_linear_dot_kernel() else {
        return Ok(None);
    };
    let start = ProfileTimer::start(profile.is_some());
    let Some((x_values, next_v0_values, next_time_state)) =
        time_mixer_forward_flat_state_input_values_raw_output_profiled_options(
            &weights.time_mixer,
            input_values,
            batch_size,
            channels,
            device,
            v0_values,
            time_state,
            true,
            dot_kernel,
            profile.as_deref_mut(),
        )?
    else {
        return Ok(None);
    };
    let next_time_state =
        next_time_state.expect("state-producing flat time mixer returns flat state");
    profile_add!(profile, layer.time_mixer_ns, start.elapsed_ns());

    let start = ProfileTimer::start(profile.is_some());
    let channel_mixer_output = if native_channel_mixer_inplace_output_values_enabled() {
        channel_mixer_forward_flat_state_from_owned_input_values_raw_output_profiled(
            &weights.channel_mixer,
            x_values,
            batch_size,
            channels,
            device,
            channel_state,
            true,
            profile.as_deref_mut(),
        )?
    } else {
        channel_mixer_forward_flat_state_from_input_values_raw_output_profiled(
            &weights.channel_mixer,
            &x_values,
            batch_size,
            channels,
            device,
            channel_state,
            true,
            profile.as_deref_mut(),
        )?
    };
    let Some((output_values, next_channel_state)) = channel_mixer_output else {
        return Ok(None);
    };
    let next_channel_state =
        next_channel_state.expect("state-producing flat channel mixer returns flat state");
    profile_add!(profile, layer.channel_mixer_ns, start.elapsed_ns());
    profile_add!(profile, layer.total_ns, total_start.elapsed_ns());

    Ok(Some((
        output_values,
        next_v0_values,
        next_time_state,
        next_channel_state,
    )))
}

#[allow(clippy::too_many_arguments)]
fn rwkv_layer_predict_forward_flat_working_state_values_profiled(
    weights: &Rwkv7RnnLayerWeights,
    input_values: &[f32],
    batch_size: usize,
    channels: usize,
    device: &Device,
    v0_values: &[f32],
    time_state: Option<&TimeMixerFlatLayerState>,
    channel_state: Option<&ChannelMixerFlatLayerState>,
    mut profile: Option<&mut ForwardProfile>,
) -> Result<Option<(Vec<f32>, Option<Vec<f32>>)>> {
    let total_start = ProfileTimer::start(profile.is_some());
    profile_inc!(profile, layer.calls);

    let Some(dot_kernel) = native_linear_dot_kernel() else {
        return Ok(None);
    };
    let start = ProfileTimer::start(profile.is_some());
    let Some((x_values, next_v0_values, _next_time_state)) =
        time_mixer_forward_flat_state_input_values_raw_output_profiled_options(
            &weights.time_mixer,
            input_values,
            batch_size,
            channels,
            device,
            v0_values,
            time_state,
            false,
            dot_kernel,
            profile.as_deref_mut(),
        )?
    else {
        return Ok(None);
    };
    profile_add!(profile, layer.time_mixer_ns, start.elapsed_ns());

    let start = ProfileTimer::start(profile.is_some());
    let channel_mixer_output = if native_channel_mixer_inplace_output_values_enabled() {
        channel_mixer_forward_flat_state_from_owned_input_values_raw_output_profiled(
            &weights.channel_mixer,
            x_values,
            batch_size,
            channels,
            device,
            channel_state,
            false,
            profile.as_deref_mut(),
        )?
    } else {
        channel_mixer_forward_flat_state_from_input_values_raw_output_profiled(
            &weights.channel_mixer,
            &x_values,
            batch_size,
            channels,
            device,
            channel_state,
            false,
            profile.as_deref_mut(),
        )?
    };
    let Some((output_values, _next_channel_state)) = channel_mixer_output else {
        return Ok(None);
    };
    profile_add!(profile, layer.channel_mixer_ns, start.elapsed_ns());
    profile_add!(profile, layer.total_ns, total_start.elapsed_ns());

    Ok(Some((output_values, next_v0_values)))
}

pub fn rwkv_rnn_forward(
    weights: &Rwkv7RnnWeights,
    in_bc: &Tensor,
    time_x_shift_b1c_by_layer: Option<&[Tensor]>,
    time_state_b1hkk_by_layer: Option<&[Tensor]>,
    channel_state_b1c_by_layer: Option<&[Tensor]>,
) -> Result<RwkvRnnOutput> {
    rwkv_rnn_forward_profiled(
        weights,
        in_bc,
        time_x_shift_b1c_by_layer,
        time_state_b1hkk_by_layer,
        channel_state_b1c_by_layer,
        None,
    )
}

fn rwkv_rnn_forward_profiled(
    weights: &Rwkv7RnnWeights,
    in_bc: &Tensor,
    time_x_shift_b1c_by_layer: Option<&[Tensor]>,
    time_state_b1hkk_by_layer: Option<&[Tensor]>,
    channel_state_b1c_by_layer: Option<&[Tensor]>,
    mut profile: Option<&mut ForwardProfile>,
) -> Result<RwkvRnnOutput> {
    let total_start = ProfileTimer::start(profile.is_some());
    profile_inc!(profile, rnn.calls);

    let start = ProfileTimer::start(profile.is_some());
    validate_rnn_state_layers(
        weights.blocks.len(),
        time_x_shift_b1c_by_layer,
        time_state_b1hkk_by_layer,
        channel_state_b1c_by_layer,
    )?;
    profile_add!(profile, rnn.validate_ns, start.elapsed_ns());

    let start = ProfileTimer::start(profile.is_some());
    let (batch_size, channels) = in_bc.dims2()?;
    let mut x_bc = in_bc.clone();
    let mut v0_bc = Tensor::zeros((batch_size, channels), DType::F32, in_bc.device())?;
    let mut next_time_x_shift_b1c_by_layer = Vec::with_capacity(weights.blocks.len());
    let mut next_time_state_b1hkk_by_layer = Vec::with_capacity(weights.blocks.len());
    let mut next_channel_state_b1c_by_layer = Vec::with_capacity(weights.blocks.len());
    profile_add!(profile, rnn.init_ns, start.elapsed_ns());

    for (layer_index, block) in weights.blocks.iter().enumerate() {
        let time_state = match (time_x_shift_b1c_by_layer, time_state_b1hkk_by_layer) {
            (Some(time_x_shift_b1c_by_layer), Some(time_state_b1hkk_by_layer)) => Some((
                &time_x_shift_b1c_by_layer[layer_index],
                &time_state_b1hkk_by_layer[layer_index],
            )),
            (None, None) => None,
            _ => unreachable!("state layer validation checks all-or-none state groups"),
        };
        let channel_state_b1c = channel_state_b1c_by_layer.map(|states| &states[layer_index]);

        let (
            next_x_bc,
            next_v0_bc,
            next_time_x_shift_b1c,
            next_time_state_b1hkk,
            next_channel_state_b1c,
        ) = {
            let start = ProfileTimer::start(profile.is_some());
            let output = rwkv_layer_forward_profiled(
                block,
                &x_bc,
                &v0_bc,
                time_state,
                channel_state_b1c,
                profile.as_deref_mut(),
            )?;
            profile_add!(profile, rnn.layer_total_ns, start.elapsed_ns());
            output
        };

        x_bc = next_x_bc;
        v0_bc = next_v0_bc;
        next_time_x_shift_b1c_by_layer.push(next_time_x_shift_b1c);
        next_time_state_b1hkk_by_layer.push(next_time_state_b1hkk);
        next_channel_state_b1c_by_layer.push(next_channel_state_b1c);
    }

    profile_add!(profile, rnn.total_ns, total_start.elapsed_ns());

    Ok((
        x_bc,
        next_time_x_shift_b1c_by_layer,
        next_time_state_b1hkk_by_layer,
        next_channel_state_b1c_by_layer,
    ))
}

pub(super) fn rwkv_rnn_forward_flat_time_state_profiled(
    weights: &Rwkv7RnnWeights,
    in_bc: &Tensor,
    time_state_by_layer: Option<&[TimeMixerFlatLayerState]>,
    channel_state_b1c_by_layer: Option<&[Tensor]>,
    mut profile: Option<&mut ForwardProfile>,
) -> Result<Option<(Tensor, Vec<TimeMixerFlatLayerState>, Vec<Tensor>)>> {
    let total_start = ProfileTimer::start(profile.is_some());
    profile_inc!(profile, rnn.calls);

    let start = ProfileTimer::start(profile.is_some());
    if let Some(time_state_by_layer) = time_state_by_layer {
        if time_state_by_layer.len() != weights.blocks.len() {
            bail!(
                "flat time state expected {} layers, got {}",
                weights.blocks.len(),
                time_state_by_layer.len()
            );
        }
    }
    if let Some(channel_state_b1c_by_layer) = channel_state_b1c_by_layer {
        if channel_state_b1c_by_layer.len() != weights.blocks.len() {
            bail!(
                "channel state expected {} layers, got {}",
                weights.blocks.len(),
                channel_state_b1c_by_layer.len()
            );
        }
    }
    profile_add!(profile, rnn.validate_ns, start.elapsed_ns());

    let start = ProfileTimer::start(profile.is_some());
    let (batch_size, channels) = in_bc.dims2()?;
    let mut x_bc = in_bc.clone();
    let mut v0_bc = Tensor::zeros((batch_size, channels), DType::F32, in_bc.device())?;
    let mut next_time_state_by_layer = Vec::with_capacity(weights.blocks.len());
    let mut next_channel_state_b1c_by_layer = Vec::with_capacity(weights.blocks.len());
    profile_add!(profile, rnn.init_ns, start.elapsed_ns());

    for (layer_index, block) in weights.blocks.iter().enumerate() {
        let time_state = time_state_by_layer.map(|states| &states[layer_index]);
        let channel_state_b1c = channel_state_b1c_by_layer.map(|states| &states[layer_index]);
        let start = ProfileTimer::start(profile.is_some());
        let Some((next_x_bc, next_v0_bc, next_time_state, next_channel_state_b1c)) =
            rwkv_layer_forward_flat_time_state_profiled(
                block,
                &x_bc,
                &v0_bc,
                time_state,
                channel_state_b1c,
                profile.as_deref_mut(),
            )?
        else {
            return Ok(None);
        };
        profile_add!(profile, rnn.layer_total_ns, start.elapsed_ns());
        x_bc = next_x_bc;
        v0_bc = next_v0_bc;
        next_time_state_by_layer.push(next_time_state);
        next_channel_state_b1c_by_layer.push(next_channel_state_b1c);
    }

    profile_add!(profile, rnn.total_ns, total_start.elapsed_ns());

    Ok(Some((
        x_bc,
        next_time_state_by_layer,
        next_channel_state_b1c_by_layer,
    )))
}

fn rwkv_rnn_predict_forward_profiled(
    weights: &Rwkv7RnnWeights,
    in_bc: &Tensor,
    time_x_shift_b1c_by_layer: Option<&[Tensor]>,
    time_state_b1hkk_by_layer: Option<&[Tensor]>,
    channel_state_b1c_by_layer: Option<&[Tensor]>,
    mut profile: Option<&mut ForwardProfile>,
) -> Result<Tensor> {
    let total_start = ProfileTimer::start(profile.is_some());
    profile_inc!(profile, rnn.calls);

    let start = ProfileTimer::start(profile.is_some());
    validate_rnn_state_layers(
        weights.blocks.len(),
        time_x_shift_b1c_by_layer,
        time_state_b1hkk_by_layer,
        channel_state_b1c_by_layer,
    )?;
    profile_add!(profile, rnn.validate_ns, start.elapsed_ns());

    let start = ProfileTimer::start(profile.is_some());
    let (batch_size, channels) = in_bc.dims2()?;
    let mut x_bc = in_bc.clone();
    let mut v0_bc = Tensor::zeros((batch_size, channels), DType::F32, in_bc.device())?;
    profile_add!(profile, rnn.init_ns, start.elapsed_ns());

    for (layer_index, block) in weights.blocks.iter().enumerate() {
        let time_state = match (time_x_shift_b1c_by_layer, time_state_b1hkk_by_layer) {
            (Some(time_x_shift_b1c_by_layer), Some(time_state_b1hkk_by_layer)) => Some((
                &time_x_shift_b1c_by_layer[layer_index],
                &time_state_b1hkk_by_layer[layer_index],
            )),
            (None, None) => None,
            _ => unreachable!("state layer validation checks all-or-none state groups"),
        };
        let channel_state_b1c = channel_state_b1c_by_layer.map(|states| &states[layer_index]);

        let start = ProfileTimer::start(profile.is_some());
        let (next_x_bc, next_v0_bc) = rwkv_layer_predict_forward_profiled(
            block,
            &x_bc,
            &v0_bc,
            time_state,
            channel_state_b1c,
            profile.as_deref_mut(),
        )?;
        profile_add!(profile, rnn.layer_total_ns, start.elapsed_ns());
        x_bc = next_x_bc;
        v0_bc = next_v0_bc;
    }

    profile_add!(profile, rnn.total_ns, total_start.elapsed_ns());
    Ok(x_bc)
}

pub(super) fn rwkv_rnn_predict_forward_flat_time_state_profiled(
    weights: &Rwkv7RnnWeights,
    in_bc: &Tensor,
    time_state_by_layer: Option<&[TimeMixerFlatLayerState]>,
    channel_state_b1c_by_layer: Option<&[Tensor]>,
    mut profile: Option<&mut ForwardProfile>,
) -> Result<Option<Tensor>> {
    let total_start = ProfileTimer::start(profile.is_some());
    profile_inc!(profile, rnn.calls);

    let start = ProfileTimer::start(profile.is_some());
    if let Some(time_state_by_layer) = time_state_by_layer {
        if time_state_by_layer.len() != weights.blocks.len() {
            bail!(
                "flat time state expected {} layers, got {}",
                weights.blocks.len(),
                time_state_by_layer.len()
            );
        }
    }
    if let Some(channel_state_b1c_by_layer) = channel_state_b1c_by_layer {
        if channel_state_b1c_by_layer.len() != weights.blocks.len() {
            bail!(
                "channel state expected {} layers, got {}",
                weights.blocks.len(),
                channel_state_b1c_by_layer.len()
            );
        }
    }
    profile_add!(profile, rnn.validate_ns, start.elapsed_ns());

    let start = ProfileTimer::start(profile.is_some());
    let (batch_size, channels) = in_bc.dims2()?;
    let mut x_bc = in_bc.clone();
    let mut v0_bc = Tensor::zeros((batch_size, channels), DType::F32, in_bc.device())?;
    profile_add!(profile, rnn.init_ns, start.elapsed_ns());

    for (layer_index, block) in weights.blocks.iter().enumerate() {
        let time_state = time_state_by_layer.map(|states| &states[layer_index]);
        let channel_state_b1c = channel_state_b1c_by_layer.map(|states| &states[layer_index]);
        let start = ProfileTimer::start(profile.is_some());
        let Some((next_x_bc, next_v0_bc)) = rwkv_layer_predict_forward_flat_time_state_profiled(
            block,
            &x_bc,
            &v0_bc,
            time_state,
            channel_state_b1c,
            profile.as_deref_mut(),
        )?
        else {
            return Ok(None);
        };
        profile_add!(profile, rnn.layer_total_ns, start.elapsed_ns());
        x_bc = next_x_bc;
        v0_bc = next_v0_bc;
    }

    profile_add!(profile, rnn.total_ns, total_start.elapsed_ns());
    Ok(Some(x_bc))
}

pub(super) fn rwkv_rnn_forward_flat_working_state_profiled(
    weights: &Rwkv7RnnWeights,
    in_bc: &Tensor,
    time_state_by_layer: Option<&[TimeMixerFlatLayerState]>,
    channel_state_by_layer: Option<&[ChannelMixerFlatLayerState]>,
    mut profile: Option<&mut ForwardProfile>,
) -> Result<
    Option<(
        Tensor,
        Vec<TimeMixerFlatLayerState>,
        Vec<ChannelMixerFlatLayerState>,
    )>,
> {
    let total_start = ProfileTimer::start(profile.is_some());
    profile_inc!(profile, rnn.calls);

    let start = ProfileTimer::start(profile.is_some());
    if let Some(time_state_by_layer) = time_state_by_layer {
        if time_state_by_layer.len() != weights.blocks.len() {
            bail!(
                "flat time state expected {} layers, got {}",
                weights.blocks.len(),
                time_state_by_layer.len()
            );
        }
    }
    if let Some(channel_state_by_layer) = channel_state_by_layer {
        if channel_state_by_layer.len() != weights.blocks.len() {
            bail!(
                "flat channel state expected {} layers, got {}",
                weights.blocks.len(),
                channel_state_by_layer.len()
            );
        }
    }
    profile_add!(profile, rnn.validate_ns, start.elapsed_ns());

    let start = ProfileTimer::start(profile.is_some());
    let (batch_size, channels) = in_bc.dims2()?;
    let mut x_bc = in_bc.clone();
    let mut v0_bc = Tensor::zeros((batch_size, channels), DType::F32, in_bc.device())?;
    let mut next_time_state_by_layer = Vec::with_capacity(weights.blocks.len());
    let mut next_channel_state_by_layer = Vec::with_capacity(weights.blocks.len());
    profile_add!(profile, rnn.init_ns, start.elapsed_ns());

    if native_layer_values_carrier_enabled() && native_layer_time_channel_values_handoff_enabled() {
        let input_data = f32_tensor_data(in_bc)?;
        let input_values = input_data.as_slice()?;
        if input_values.len() != batch_size * channels {
            return Ok(None);
        }
        let mut x_values = input_values.to_vec();
        let mut v0_values = vec![0.0f32; batch_size * channels];
        for (layer_index, block) in weights.blocks.iter().enumerate() {
            let time_state = time_state_by_layer.map(|states| &states[layer_index]);
            let channel_state = channel_state_by_layer.map(|states| &states[layer_index]);
            let start = ProfileTimer::start(profile.is_some());
            let Some((next_x_values, next_v0_values, next_time_state, next_channel_state)) =
                rwkv_layer_forward_flat_working_state_values_profiled(
                    block,
                    &x_values,
                    batch_size,
                    channels,
                    in_bc.device(),
                    &v0_values,
                    time_state,
                    channel_state,
                    profile.as_deref_mut(),
                )?
            else {
                return Ok(None);
            };
            profile_add!(profile, rnn.layer_total_ns, start.elapsed_ns());
            x_values = next_x_values;
            if let Some(next_v0_values) = next_v0_values {
                v0_values = next_v0_values;
            }
            next_time_state_by_layer.push(next_time_state);
            next_channel_state_by_layer.push(next_channel_state);
        }

        profile_add!(profile, rnn.total_ns, total_start.elapsed_ns());
        let x_bc = Tensor::from_vec(x_values, (batch_size, channels), in_bc.device())?;
        return Ok(Some((
            x_bc,
            next_time_state_by_layer,
            next_channel_state_by_layer,
        )));
    }

    for (layer_index, block) in weights.blocks.iter().enumerate() {
        let time_state = time_state_by_layer.map(|states| &states[layer_index]);
        let channel_state = channel_state_by_layer.map(|states| &states[layer_index]);
        let start = ProfileTimer::start(profile.is_some());
        let Some((next_x_bc, next_v0_bc, next_time_state, next_channel_state)) =
            rwkv_layer_forward_flat_working_state_profiled(
                block,
                &x_bc,
                &v0_bc,
                time_state,
                channel_state,
                profile.as_deref_mut(),
            )?
        else {
            return Ok(None);
        };
        profile_add!(profile, rnn.layer_total_ns, start.elapsed_ns());
        x_bc = next_x_bc;
        v0_bc = next_v0_bc;
        next_time_state_by_layer.push(next_time_state);
        next_channel_state_by_layer.push(next_channel_state);
    }

    profile_add!(profile, rnn.total_ns, total_start.elapsed_ns());

    Ok(Some((
        x_bc,
        next_time_state_by_layer,
        next_channel_state_by_layer,
    )))
}

pub(super) fn rwkv_rnn_predict_forward_flat_working_state_profiled(
    weights: &Rwkv7RnnWeights,
    in_bc: &Tensor,
    time_state_by_layer: Option<&[TimeMixerFlatLayerState]>,
    channel_state_by_layer: Option<&[ChannelMixerFlatLayerState]>,
    mut profile: Option<&mut ForwardProfile>,
) -> Result<Option<Tensor>> {
    let total_start = ProfileTimer::start(profile.is_some());
    profile_inc!(profile, rnn.calls);

    let start = ProfileTimer::start(profile.is_some());
    if let Some(time_state_by_layer) = time_state_by_layer {
        if time_state_by_layer.len() != weights.blocks.len() {
            bail!(
                "flat time state expected {} layers, got {}",
                weights.blocks.len(),
                time_state_by_layer.len()
            );
        }
    }
    if let Some(channel_state_by_layer) = channel_state_by_layer {
        if channel_state_by_layer.len() != weights.blocks.len() {
            bail!(
                "flat channel state expected {} layers, got {}",
                weights.blocks.len(),
                channel_state_by_layer.len()
            );
        }
    }
    profile_add!(profile, rnn.validate_ns, start.elapsed_ns());

    let start = ProfileTimer::start(profile.is_some());
    let (batch_size, channels) = in_bc.dims2()?;
    let mut x_bc = in_bc.clone();
    let mut v0_bc = Tensor::zeros((batch_size, channels), DType::F32, in_bc.device())?;
    profile_add!(profile, rnn.init_ns, start.elapsed_ns());

    if native_layer_values_carrier_enabled() && native_layer_time_channel_values_handoff_enabled() {
        let input_data = f32_tensor_data(in_bc)?;
        let input_values = input_data.as_slice()?;
        if input_values.len() != batch_size * channels {
            return Ok(None);
        }
        let mut x_values = input_values.to_vec();
        let mut v0_values = vec![0.0f32; batch_size * channels];
        for (layer_index, block) in weights.blocks.iter().enumerate() {
            let time_state = time_state_by_layer.map(|states| &states[layer_index]);
            let channel_state = channel_state_by_layer.map(|states| &states[layer_index]);
            let start = ProfileTimer::start(profile.is_some());
            let Some((next_x_values, next_v0_values)) =
                rwkv_layer_predict_forward_flat_working_state_values_profiled(
                    block,
                    &x_values,
                    batch_size,
                    channels,
                    in_bc.device(),
                    &v0_values,
                    time_state,
                    channel_state,
                    profile.as_deref_mut(),
                )?
            else {
                return Ok(None);
            };
            profile_add!(profile, rnn.layer_total_ns, start.elapsed_ns());
            x_values = next_x_values;
            if let Some(next_v0_values) = next_v0_values {
                v0_values = next_v0_values;
            }
        }

        profile_add!(profile, rnn.total_ns, total_start.elapsed_ns());
        let x_bc = Tensor::from_vec(x_values, (batch_size, channels), in_bc.device())?;
        return Ok(Some(x_bc));
    }

    for (layer_index, block) in weights.blocks.iter().enumerate() {
        let time_state = time_state_by_layer.map(|states| &states[layer_index]);
        let channel_state = channel_state_by_layer.map(|states| &states[layer_index]);
        let start = ProfileTimer::start(profile.is_some());
        let Some((next_x_bc, next_v0_bc)) = rwkv_layer_predict_forward_flat_working_state_profiled(
            block,
            &x_bc,
            &v0_bc,
            time_state,
            channel_state,
            profile.as_deref_mut(),
        )?
        else {
            return Ok(None);
        };
        profile_add!(profile, rnn.layer_total_ns, start.elapsed_ns());
        x_bc = next_x_bc;
        v0_bc = next_v0_bc;
    }

    profile_add!(profile, rnn.total_ns, total_start.elapsed_ns());
    Ok(Some(x_bc))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn rwkv_rnn_forward_flat_working_state_values_profiled(
    weights: &Rwkv7RnnWeights,
    input_values: &[f32],
    batch_size: usize,
    channels: usize,
    device: &Device,
    time_state_by_layer: Option<&[TimeMixerFlatLayerState]>,
    channel_state_by_layer: Option<&[ChannelMixerFlatLayerState]>,
    mut profile: Option<&mut ForwardProfile>,
) -> Result<
    Option<(
        Vec<f32>,
        Vec<TimeMixerFlatLayerState>,
        Vec<ChannelMixerFlatLayerState>,
    )>,
> {
    if !(native_layer_values_carrier_enabled()
        && native_layer_time_channel_values_handoff_enabled())
    {
        return Ok(None);
    }

    let total_start = ProfileTimer::start(profile.is_some());
    profile_inc!(profile, rnn.calls);

    let start = ProfileTimer::start(profile.is_some());
    if input_values.len() != batch_size * channels {
        return Ok(None);
    }
    if let Some(time_state_by_layer) = time_state_by_layer {
        if time_state_by_layer.len() != weights.blocks.len() {
            bail!(
                "flat time state expected {} layers, got {}",
                weights.blocks.len(),
                time_state_by_layer.len()
            );
        }
    }
    if let Some(channel_state_by_layer) = channel_state_by_layer {
        if channel_state_by_layer.len() != weights.blocks.len() {
            bail!(
                "flat channel state expected {} layers, got {}",
                weights.blocks.len(),
                channel_state_by_layer.len()
            );
        }
    }
    profile_add!(profile, rnn.validate_ns, start.elapsed_ns());

    let start = ProfileTimer::start(profile.is_some());
    let borrow_first_input = native_rnn_borrow_first_input_values_enabled();
    let mut x_values = if borrow_first_input {
        None
    } else {
        Some(input_values.to_vec())
    };
    let mut v0_values = vec![0.0f32; batch_size * channels];
    let mut next_time_state_by_layer = Vec::with_capacity(weights.blocks.len());
    let mut next_channel_state_by_layer = Vec::with_capacity(weights.blocks.len());
    profile_add!(profile, rnn.init_ns, start.elapsed_ns());

    for (layer_index, block) in weights.blocks.iter().enumerate() {
        let time_state = time_state_by_layer.map(|states| &states[layer_index]);
        let channel_state = channel_state_by_layer.map(|states| &states[layer_index]);
        let current_values = x_values.as_deref().unwrap_or(input_values);
        let start = ProfileTimer::start(profile.is_some());
        let Some((next_x_values, next_v0_values, next_time_state, next_channel_state)) =
            rwkv_layer_forward_flat_working_state_values_profiled(
                block,
                current_values,
                batch_size,
                channels,
                device,
                &v0_values,
                time_state,
                channel_state,
                profile.as_deref_mut(),
            )?
        else {
            return Ok(None);
        };
        profile_add!(profile, rnn.layer_total_ns, start.elapsed_ns());
        x_values = Some(next_x_values);
        if let Some(next_v0_values) = next_v0_values {
            v0_values = next_v0_values;
        }
        next_time_state_by_layer.push(next_time_state);
        next_channel_state_by_layer.push(next_channel_state);
    }

    profile_add!(profile, rnn.total_ns, total_start.elapsed_ns());
    let x_values = x_values.unwrap_or_else(|| input_values.to_vec());
    Ok(Some((
        x_values,
        next_time_state_by_layer,
        next_channel_state_by_layer,
    )))
}

pub(super) fn rwkv_rnn_predict_forward_flat_working_state_values_profiled(
    weights: &Rwkv7RnnWeights,
    input_values: &[f32],
    batch_size: usize,
    channels: usize,
    device: &Device,
    time_state_by_layer: Option<&[TimeMixerFlatLayerState]>,
    channel_state_by_layer: Option<&[ChannelMixerFlatLayerState]>,
    mut profile: Option<&mut ForwardProfile>,
) -> Result<Option<Vec<f32>>> {
    if !(native_layer_values_carrier_enabled()
        && native_layer_time_channel_values_handoff_enabled())
    {
        return Ok(None);
    }

    let total_start = ProfileTimer::start(profile.is_some());
    profile_inc!(profile, rnn.calls);

    let start = ProfileTimer::start(profile.is_some());
    if input_values.len() != batch_size * channels {
        return Ok(None);
    }
    if let Some(time_state_by_layer) = time_state_by_layer {
        if time_state_by_layer.len() != weights.blocks.len() {
            bail!(
                "flat time state expected {} layers, got {}",
                weights.blocks.len(),
                time_state_by_layer.len()
            );
        }
    }
    if let Some(channel_state_by_layer) = channel_state_by_layer {
        if channel_state_by_layer.len() != weights.blocks.len() {
            bail!(
                "flat channel state expected {} layers, got {}",
                weights.blocks.len(),
                channel_state_by_layer.len()
            );
        }
    }
    profile_add!(profile, rnn.validate_ns, start.elapsed_ns());

    let start = ProfileTimer::start(profile.is_some());
    let borrow_first_input = native_rnn_borrow_first_input_values_enabled();
    let mut x_values = if borrow_first_input {
        None
    } else {
        Some(input_values.to_vec())
    };
    let mut v0_values = vec![0.0f32; batch_size * channels];
    profile_add!(profile, rnn.init_ns, start.elapsed_ns());

    for (layer_index, block) in weights.blocks.iter().enumerate() {
        let time_state = time_state_by_layer.map(|states| &states[layer_index]);
        let channel_state = channel_state_by_layer.map(|states| &states[layer_index]);
        let current_values = x_values.as_deref().unwrap_or(input_values);
        let start = ProfileTimer::start(profile.is_some());
        let Some((next_x_values, next_v0_values)) =
            rwkv_layer_predict_forward_flat_working_state_values_profiled(
                block,
                current_values,
                batch_size,
                channels,
                device,
                &v0_values,
                time_state,
                channel_state,
                profile.as_deref_mut(),
            )?
        else {
            return Ok(None);
        };
        profile_add!(profile, rnn.layer_total_ns, start.elapsed_ns());
        x_values = Some(next_x_values);
        if let Some(next_v0_values) = next_v0_values {
            v0_values = next_v0_values;
        }
    }

    profile_add!(profile, rnn.total_ns, total_start.elapsed_ns());
    let x_values = x_values.unwrap_or_else(|| input_values.to_vec());
    Ok(Some(x_values))
}

fn py_value_error(err: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_weights::{
        Features2CardWeights, GateLoraWeights, GroupNormWeights, LoraSimpleWeights,
        Rwkv7RnnChannelMixerWeights, Rwkv7RnnLayerWeights, Rwkv7RnnWeights, SrsRwkvRnnWeights,
        WHeadWeights,
    };
    use crate::weights::{NUM_CURVES, NUM_POINTS};
    use std::sync::OnceLock;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn blocked_dot12_fixture(out_dim: usize, in_dim: usize) -> (Vec<f32>, Vec<f32>) {
        let input = (0..in_dim)
            .map(|index| ((index * 17 + 3) % 101) as f32 / 53.0 - 0.9)
            .collect::<Vec<_>>();
        let weights = (0..out_dim * in_dim)
            .map(|index| ((index * 29 + 11) % 211) as f32 / 97.0 - 1.05)
            .collect::<Vec<_>>();
        (input, weights)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn blocked_dot12_baseline(
        input: &[f32],
        weights: &[f32],
        out_dim: usize,
        output: &mut [f32],
    ) {
        match input.len() {
            128 => linear_project_row_same_x_avx2_fma_dot12_fixed::<128>(
                input, weights, out_dim, output,
            ),
            in_dim => {
                linear_project_row_same_x_avx2_fma_dot12(input, weights, out_dim, in_dim, output)
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn blocked_input8_dot12_matches_hot_row_major_shapes_bit_for_bit() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
            return;
        }
        for (out_dim, in_dim) in [
            (128usize, 128usize),
            (192, 128),
            (256, 128),
            (128, 192),
            (128, 256),
        ] {
            let (input, weights) = blocked_dot12_fixture(out_dim, in_dim);
            let blocked = linear_pack_input8_dot12_weights(&weights, out_dim, in_dim).unwrap();
            let mut expected = vec![0.0f32; out_dim];
            let mut actual = vec![0.0f32; out_dim];
            unsafe {
                blocked_dot12_baseline(&input, &weights, out_dim, &mut expected);
                linear_project_row_same_x_avx2_fma_blocked_input8_dot12(
                    &input,
                    &blocked,
                    &weights,
                    out_dim,
                    &mut actual,
                );
            }
            assert_eq!(
                actual
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "blocked dot12 differs for {out_dim}x{in_dim}"
            );
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn blocked_input8_dot12_batch_router_keeps_multi_row_outputs_exact() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
            return;
        }
        let (base_input, weight_values) = blocked_dot12_fixture(128, 128);
        let weights = LinearF32Weights {
            out_dim: 128,
            in_dim: 128,
            blocked_input8_dot12: linear_pack_input8_dot12_weights(&weight_values, 128, 128),
            weight: weight_values,
            bias: None,
        };
        for row_count in [1usize, 2, 4, 16, 512] {
            let mut input_values = Vec::with_capacity(row_count * 128);
            for row in 0..row_count {
                input_values.extend(base_input.iter().enumerate().map(|(index, value)| {
                    *value + ((row * 7 + index * 3) % 13) as f32 * 0.000_125
                }));
            }
            let mut expected = vec![0.0f32; row_count * 128];
            for row in 0..row_count {
                let input_base = row * 128;
                let output_base = row * 128;
                unsafe {
                    blocked_dot12_baseline(
                        &input_values[input_base..input_base + 128],
                        &weights.weight,
                        128,
                        &mut expected[output_base..output_base + 128],
                    );
                }
            }
            let mut actual = vec![0.0f32; row_count * 128];
            linear_project_f32_batch_same_x(
                NativeLinearDotKernel::Avx2Fma,
                &input_values,
                row_count,
                &weights,
                &mut actual,
            );
            assert_eq!(
                actual
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "blocked batch router differs for {row_count} rows"
            );
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    #[ignore = "manual exact-shape projection microbenchmark"]
    fn benchmark_blocked_input8_dot12_hot_shapes() {
        use std::hint::black_box;
        use std::time::Instant;

        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
            return;
        }
        const WARMUP: usize = 256;
        const ITERATIONS: usize = 4_096;
        for (out_dim, in_dim) in [
            (128usize, 128usize),
            (192, 128),
            (256, 128),
            (128, 192),
            (128, 256),
        ] {
            let (input, weights) = blocked_dot12_fixture(out_dim, in_dim);
            let blocked = linear_pack_input8_dot12_weights(&weights, out_dim, in_dim).unwrap();
            let mut output = vec![0.0f32; out_dim];
            unsafe {
                for _ in 0..WARMUP {
                    blocked_dot12_baseline(&input, &weights, out_dim, &mut output);
                    linear_project_row_same_x_avx2_fma_blocked_input8_dot12(
                        &input,
                        &blocked,
                        &weights,
                        out_dim,
                        &mut output,
                    );
                }
            }

            let baseline_start = Instant::now();
            unsafe {
                for _ in 0..ITERATIONS {
                    blocked_dot12_baseline(
                        black_box(&input),
                        black_box(&weights),
                        out_dim,
                        black_box(&mut output),
                    );
                }
            }
            let baseline_ns = baseline_start.elapsed().as_nanos() as f64 / ITERATIONS as f64;
            let blocked_start = Instant::now();
            unsafe {
                for _ in 0..ITERATIONS {
                    linear_project_row_same_x_avx2_fma_blocked_input8_dot12(
                        black_box(&input),
                        black_box(&blocked),
                        black_box(&weights),
                        out_dim,
                        black_box(&mut output),
                    );
                }
            }
            let blocked_ns = blocked_start.elapsed().as_nanos() as f64 / ITERATIONS as f64;
            println!(
                "blocked-dot12 {out_dim}x{in_dim}: baseline={baseline_ns:.3}ns blocked={blocked_ns:.3}ns speedup={:.3}%",
                (baseline_ns / blocked_ns - 1.0) * 100.0
            );
            black_box(output);
        }
    }

    #[test]
    fn channel_mixer_returns_residual_when_projection_weights_are_zero() {
        let weights = zero_projection_channel_mixer();
        let input = Tensor::new(
            &[[1.0f32, 2.0, 3.0, 4.0], [-4.0, 1.0, 0.5, 2.5]],
            &Device::Cpu,
        )
        .unwrap();

        let (output, state) = channel_mixer_forward(&weights, &input, None).unwrap();

        assert_eq!(output.dims(), &[2, 4]);
        assert_eq!(state.dims(), &[2, 1, 4]);
        assert_eq!(
            output.to_vec2::<f32>().unwrap(),
            input.to_vec2::<f32>().unwrap()
        );
    }

    #[test]
    fn channel_mixer_rejects_wrong_state_shape() {
        let weights = zero_projection_channel_mixer();
        let input = Tensor::zeros((2usize, 4usize), DType::F32, &Device::Cpu).unwrap();
        let state = Tensor::zeros((1usize, 1usize, 4usize), DType::F32, &Device::Cpu).unwrap();

        let err = channel_mixer_forward(&weights, &input, Some(&state)).unwrap_err();

        assert!(err
            .to_string()
            .contains("channel_mixer expected state shape [2, 1, 4]"));
    }

    fn zero_projection_channel_mixer() -> Rwkv7RnnChannelMixerWeights {
        Rwkv7RnnChannelMixerWeights {
            layer_id: 0,
            channel_dim: 3,
            layer_norm: LayerNormWeights {
                weight: Tensor::new(&[1.0f32, 1.0, 1.0, 1.0], &Device::Cpu).unwrap(),
                bias: Tensor::new(&[0.0f32, 0.0, 0.0, 0.0], &Device::Cpu).unwrap(),
            },
            lerp_k: Tensor::zeros((1usize, 1usize, 4usize), DType::F32, &Device::Cpu).unwrap(),
            w_k: LinearWeights {
                weight: Tensor::zeros((3usize, 4usize), DType::F32, &Device::Cpu).unwrap(),
                bias: None,
            },
            w_v: LinearWeights {
                weight: Tensor::zeros((4usize, 3usize), DType::F32, &Device::Cpu).unwrap(),
                bias: None,
            },
            native_f32_cache: OnceLock::new(),
        }
    }

    #[test]
    fn native_linear_kernel_matches_candle_rank2_with_bias() {
        let input = Tensor::new(&[[0.5f32, -1.0, 2.0], [1.5, 0.25, -0.75]], &Device::Cpu).unwrap();
        let weights = LinearWeights {
            weight: Tensor::new(
                &[
                    [1.0f32, 0.5, -0.25],
                    [0.0, -1.0, 2.0],
                    [0.25, 0.25, 0.25],
                    [-0.5, 1.0, 0.0],
                ],
                &Device::Cpu,
            )
            .unwrap(),
            bias: Some(Tensor::new(&[0.125f32, -0.5, 1.0, 0.25], &Device::Cpu).unwrap()),
        };

        let expected = linear(&input, &weights).unwrap();
        let actual = linear_native_f32_with_kernel(&input, &weights, NativeLinearDotKernel::Scalar)
            .unwrap()
            .unwrap();

        assert_eq!(actual.dims(), expected.dims());
        assert_tensor_close(&actual, &expected, 1e-6);
    }

    #[test]
    fn native_linear_kernel_matches_candle_rank3_without_bias() {
        let input = Tensor::from_vec(
            vec![0.5f32, -1.0, 2.0, 1.5, 0.25, -0.75],
            (2usize, 1usize, 3usize),
            &Device::Cpu,
        )
        .unwrap();
        let weights = LinearWeights {
            weight: Tensor::new(
                &[
                    [1.0f32, 0.5, -0.25],
                    [0.0, -1.0, 2.0],
                    [0.25, 0.25, 0.25],
                    [-0.5, 1.0, 0.0],
                ],
                &Device::Cpu,
            )
            .unwrap(),
            bias: None,
        };

        let expected = linear(&input, &weights).unwrap();
        let actual = linear_native_f32_with_kernel(&input, &weights, NativeLinearDotKernel::Scalar)
            .unwrap()
            .unwrap();

        assert_eq!(actual.dims(), expected.dims());
        assert_tensor_close(&actual, &expected, 1e-6);
    }

    #[test]
    fn native_channel_mixer_projection_matches_candle_sequence() {
        let mut weights = Rwkv7RnnChannelMixerWeights {
            layer_id: 0,
            channel_dim: 3,
            layer_norm: unit_layer_norm(4),
            lerp_k: Tensor::zeros((1usize, 1usize, 4usize), DType::F32, &Device::Cpu).unwrap(),
            w_k: LinearWeights {
                weight: Tensor::new(
                    &[
                        [1.0f32, 0.5, -0.25, 0.0],
                        [0.0, -1.0, 2.0, 0.25],
                        [0.25, 0.25, 0.25, -0.5],
                    ],
                    &Device::Cpu,
                )
                .unwrap(),
                bias: None,
            },
            w_v: LinearWeights {
                weight: Tensor::new(
                    &[
                        [1.0f32, -0.25, 0.5],
                        [0.0, 2.0, -1.0],
                        [0.25, 0.5, 0.25],
                        [-0.5, 1.0, 0.0],
                    ],
                    &Device::Cpu,
                )
                .unwrap(),
                bias: None,
            },
            native_f32_cache: OnceLock::new(),
        };
        let input = Tensor::from_vec(
            vec![0.5f32, -1.0, 2.0, 0.25, 1.5, 0.25, -0.75, 2.0],
            (2usize, 1usize, 4usize),
            &Device::Cpu,
        )
        .unwrap();

        let expected_hidden = linear(&input, &weights.w_k).unwrap();
        let expected = linear(
            &expected_hidden.relu().unwrap().sqr().unwrap(),
            &weights.w_v,
        )
        .unwrap();
        let input_bc = input.squeeze(1).unwrap();
        let expected_residual = (&input + &expected).unwrap().squeeze(1).unwrap();
        let actual = channel_mixer_projection_native_with_kernel(
            &input,
            &weights,
            NativeLinearDotKernel::Scalar,
        )
        .unwrap()
        .unwrap();
        let actual_residual = channel_mixer_projection_residual_native_with_kernel(
            &input_bc,
            &input,
            &weights,
            NativeLinearDotKernel::Scalar,
        )
        .unwrap()
        .unwrap();

        assert_eq!(actual.dims(), expected.dims());
        assert_tensor_close(&actual, &expected, 1e-6);
        assert_eq!(actual_residual.dims(), expected_residual.dims());
        assert_tensor_close(&actual_residual, &expected_residual, 1e-6);

        weights.layer_norm = LayerNormWeights {
            weight: Tensor::new(&[1.2f32, -0.7, 0.5, 1.4], &Device::Cpu).unwrap(),
            bias: Tensor::new(&[0.1f32, -0.2, 0.3, -0.4], &Device::Cpu).unwrap(),
        };
        weights.lerp_k = Tensor::from_vec(
            vec![0.0f32, 0.25, 0.75, 1.0],
            (1usize, 1usize, 4usize),
            &Device::Cpu,
        )
        .unwrap();
        let forward_input = Tensor::new(
            &[[0.5f32, -1.0, 2.0, 0.25], [1.5, 0.25, -0.75, 2.0]],
            &Device::Cpu,
        )
        .unwrap();
        let previous_state = Tensor::from_vec(
            vec![-0.2f32, 0.4, 0.8, -1.1, 1.25, -0.5, 0.0, 0.75],
            (2usize, 1usize, 4usize),
            &Device::Cpu,
        )
        .unwrap();
        let (expected_full, expected_state) =
            channel_mixer_forward(&weights, &forward_input, Some(&previous_state)).unwrap();
        let actual_full = channel_mixer_prelude_projection_residual_native_with_kernel(
            &forward_input,
            Some(&previous_state),
            &weights,
            NativeLinearDotKernel::Scalar,
            false,
        )
        .unwrap()
        .unwrap();

        assert_eq!(actual_full.out_bc.dims(), expected_full.dims());
        assert_eq!(actual_full.next_state_b1c.dims(), expected_state.dims());
        assert_tensor_close(&actual_full.out_bc, &expected_full, 1e-6);
        assert_tensor_close(&actual_full.next_state_b1c, &expected_state, 1e-6);
    }

    #[test]
    fn native_time_mixer_projection_group_matches_candle_sequence() {
        let mut weights = zero_output_time_mixer();
        weights.w_r = LinearWeights {
            weight: Tensor::new(
                &[
                    [1.0f32, -0.25, 0.5, 0.0],
                    [0.0, 0.5, -1.0, 0.25],
                    [0.25, 1.0, 0.0, -0.5],
                    [-0.75, 0.0, 0.5, 1.0],
                ],
                &Device::Cpu,
            )
            .unwrap(),
            bias: None,
        };
        weights.w_k = LinearWeights {
            weight: Tensor::new(
                &[
                    [0.2f32, -0.6, 0.1, 0.4],
                    [0.5, 0.25, -0.3, 0.7],
                    [-0.7, 0.1, 0.8, -0.2],
                    [0.3, -0.4, 0.6, 0.9],
                ],
                &Device::Cpu,
            )
            .unwrap(),
            bias: None,
        };
        weights.k_scale_linear = LinearWeights {
            weight: Tensor::new(
                &[[0.25f32, -0.5, 0.75, 0.1], [-0.2, 0.4, 0.3, -0.6]],
                &Device::Cpu,
            )
            .unwrap(),
            bias: Some(Tensor::new(&[0.1f32, -0.2], &Device::Cpu).unwrap()),
        };
        weights.v_scale_linear = LinearWeights {
            weight: Tensor::new(
                &[[0.1f32, 0.3, -0.4, 0.8], [-0.5, 0.2, 0.6, -0.1]],
                &Device::Cpu,
            )
            .unwrap(),
            bias: Some(Tensor::new(&[-0.1f32, 0.2], &Device::Cpu).unwrap()),
        };
        let mut rkvdag_parts = Vec::new();
        for part_index in 0..8 {
            let values = match part_index {
                0 => vec![0.5f32, -1.0, 2.0, 0.25, 1.5, 0.25, -0.75, 2.0],
                1 => vec![-0.25f32, 0.75, 1.0, -1.5, 0.2, -0.4, 0.6, 1.1],
                6 => vec![1.25f32, -0.5, 0.3, 0.9, -1.0, 0.8, -0.2, 0.4],
                7 => vec![0.9f32, -0.8, 0.7, 0.6, 0.5, -0.4, 0.3, -0.2],
                _ => vec![0.0f32; 8],
            };
            rkvdag_parts
                .push(Tensor::from_vec(values, (2usize, 1usize, 4usize), &Device::Cpu).unwrap());
        }

        let (expected_r, expected_k, expected_k_scale, expected_v_scale) =
            time_mixer_projection_group_candle(&weights, &rkvdag_parts, &mut None).unwrap();
        let (actual_r, actual_k, actual_k_scale, actual_v_scale) =
            time_mixer_projection_group_native_with_kernel(
                &rkvdag_parts,
                &weights,
                NativeLinearDotKernel::Scalar,
            )
            .unwrap()
            .unwrap();

        assert_tensor_close(&actual_r, &expected_r, 1e-6);
        assert_tensor_close(&actual_k, &expected_k, 1e-6);
        assert_tensor_close(&actual_k_scale, &expected_k_scale, 1e-6);
        assert_tensor_close(&actual_v_scale, &expected_v_scale, 1e-6);
    }

    #[test]
    fn native_time_mixer_lerp_projection_scratch_matches_candle_sequence() {
        let mut weights = zero_output_time_mixer();
        weights.rkvdag_lerp = Tensor::from_vec(
            (0..32)
                .map(|index| (index as f32 - 15.0) * 0.03125)
                .collect::<Vec<_>>(),
            (8usize, 1usize, 1usize, 4usize),
            &Device::Cpu,
        )
        .unwrap();
        weights.w_r = LinearWeights {
            weight: Tensor::new(
                &[
                    [1.0f32, -0.25, 0.5, 0.0],
                    [0.0, 0.5, -1.0, 0.25],
                    [0.25, 1.0, 0.0, -0.5],
                    [-0.75, 0.0, 0.5, 1.0],
                ],
                &Device::Cpu,
            )
            .unwrap(),
            bias: None,
        };
        weights.w_k = LinearWeights {
            weight: Tensor::new(
                &[
                    [0.2f32, -0.6, 0.1, 0.4],
                    [0.5, 0.25, -0.3, 0.7],
                    [-0.7, 0.1, 0.8, -0.2],
                    [0.3, -0.4, 0.6, 0.9],
                ],
                &Device::Cpu,
            )
            .unwrap(),
            bias: None,
        };
        weights.k_scale_linear = LinearWeights {
            weight: Tensor::new(
                &[[0.25f32, -0.5, 0.75, 0.1], [-0.2, 0.4, 0.3, -0.6]],
                &Device::Cpu,
            )
            .unwrap(),
            bias: Some(Tensor::new(&[0.1f32, -0.2], &Device::Cpu).unwrap()),
        };
        weights.v_scale_linear = LinearWeights {
            weight: Tensor::new(
                &[[0.1f32, 0.3, -0.4, 0.8], [-0.5, 0.2, 0.6, -0.1]],
                &Device::Cpu,
            )
            .unwrap(),
            bias: Some(Tensor::new(&[-0.1f32, 0.2], &Device::Cpu).unwrap()),
        };
        let x_b1c = Tensor::from_vec(
            vec![0.5f32, -1.0, 2.0, 0.25, 1.5, 0.25, -0.75, 2.0],
            (2usize, 1usize, 4usize),
            &Device::Cpu,
        )
        .unwrap();
        let x_shift_b1c = Tensor::from_vec(
            vec![-0.25f32, 0.75, 1.0, -1.5, 0.2, -0.4, 0.6, 1.1],
            (2usize, 1usize, 4usize),
            &Device::Cpu,
        )
        .unwrap();

        let rkvdag_parts = time_lerp_parts_b1c(&x_b1c, &x_shift_b1c, &weights.rkvdag_lerp).unwrap();
        let (expected_r, expected_k, expected_k_scale, expected_v_scale) =
            time_mixer_projection_group_candle(&weights, &rkvdag_parts, &mut None).unwrap();
        let (
            actual_r,
            actual_k,
            actual_k_scale,
            actual_v_scale,
            actual_v,
            actual_d,
            actual_a,
            actual_g,
        ) = time_mixer_lerp_projection_scratch_native_with_kernel(
            &weights,
            &x_b1c,
            &x_shift_b1c,
            NativeLinearDotKernel::Scalar,
        )
        .unwrap()
        .unwrap();
        let actual_values = time_mixer_lerp_projection_scratch_values_native_with_kernel(
            &weights,
            &x_b1c,
            &x_shift_b1c,
            NativeLinearDotKernel::Scalar,
        )
        .unwrap()
        .unwrap();
        let actual_r_from_values = Tensor::from_vec(
            actual_values.r_values,
            (actual_values.batch_size, 1usize, actual_values.channels),
            &Device::Cpu,
        )
        .unwrap();
        let actual_k_from_values = Tensor::from_vec(
            actual_values.k_values,
            (actual_values.batch_size, 1usize, actual_values.channels),
            &Device::Cpu,
        )
        .unwrap();
        let actual_k_scale_from_values = Tensor::from_vec(
            actual_values.k_scale_values,
            (actual_values.batch_size, 1usize, actual_values.heads),
            &Device::Cpu,
        )
        .unwrap();
        let actual_v_scale_from_values = Tensor::from_vec(
            actual_values.v_scale_values,
            (actual_values.batch_size, 1usize, actual_values.heads),
            &Device::Cpu,
        )
        .unwrap();
        let actual_v_from_values = Tensor::from_vec(
            actual_values.v_values,
            (actual_values.batch_size, 1usize, actual_values.channels),
            &Device::Cpu,
        )
        .unwrap();
        let actual_d_from_values = Tensor::from_vec(
            actual_values.d_values,
            (actual_values.batch_size, 1usize, actual_values.channels),
            &Device::Cpu,
        )
        .unwrap();
        let actual_a_from_values = Tensor::from_vec(
            actual_values.a_values,
            (actual_values.batch_size, 1usize, actual_values.channels),
            &Device::Cpu,
        )
        .unwrap();
        let actual_g_from_values = Tensor::from_vec(
            actual_values.g_values,
            (actual_values.batch_size, 1usize, actual_values.channels),
            &Device::Cpu,
        )
        .unwrap();

        assert_tensor_close(&actual_r, &expected_r, 1e-6);
        assert_tensor_close(&actual_k, &expected_k, 1e-6);
        assert_tensor_close(&actual_k_scale, &expected_k_scale, 1e-6);
        assert_tensor_close(&actual_v_scale, &expected_v_scale, 1e-6);
        assert_tensor_close(&actual_v, &rkvdag_parts[2], 1e-6);
        assert_tensor_close(&actual_d, &rkvdag_parts[3], 1e-6);
        assert_tensor_close(&actual_a, &rkvdag_parts[4], 1e-6);
        assert_tensor_close(&actual_g, &rkvdag_parts[5], 1e-6);
        assert_tensor_close(&actual_r_from_values, &expected_r, 1e-6);
        assert_tensor_close(&actual_k_from_values, &expected_k, 1e-6);
        assert_tensor_close(&actual_k_scale_from_values, &expected_k_scale, 1e-6);
        assert_tensor_close(&actual_v_scale_from_values, &expected_v_scale, 1e-6);
        assert_tensor_close(&actual_v_from_values, &rkvdag_parts[2], 1e-6);
        assert_tensor_close(&actual_d_from_values, &rkvdag_parts[3], 1e-6);
        assert_tensor_close(&actual_a_from_values, &rkvdag_parts[4], 1e-6);
        assert_tensor_close(&actual_g_from_values, &rkvdag_parts[5], 1e-6);
    }

    #[test]
    fn native_time_mixer_output_matches_candle_sequence() {
        let in_b1c = Tensor::from_vec(
            vec![0.5f32, -1.0, 2.0, 0.25, 1.5, 0.25, -0.75, 2.0],
            (2usize, 1usize, 4usize),
            &Device::Cpu,
        )
        .unwrap();
        let r_b1hk = Tensor::from_vec(
            vec![0.1f32, 0.2, -0.3, 0.4, 0.7, -0.6, 0.5, -0.2],
            (2usize, 1usize, 2usize, 2usize),
            &Device::Cpu,
        )
        .unwrap();
        let k_b1hk = Tensor::from_vec(
            vec![0.3f32, -0.5, 0.25, 0.75, -0.4, 0.9, 0.6, -0.1],
            (2usize, 1usize, 2usize, 2usize),
            &Device::Cpu,
        )
        .unwrap();
        let v_b1hk = Tensor::from_vec(
            vec![1.0f32, -0.25, 0.5, 0.75, -1.0, 0.3, 0.2, -0.6],
            (2usize, 1usize, 2usize, 2usize),
            &Device::Cpu,
        )
        .unwrap();
        let bonus =
            Tensor::from_vec(vec![0.2f32, -0.1, 0.05, 0.4], (1, 1, 2, 2), &Device::Cpu).unwrap();
        let out_b1c = Tensor::from_vec(
            vec![0.25f32, -0.5, 1.25, 0.75, -0.2, 0.4, -1.1, 0.8],
            (2usize, 1usize, 4usize),
            &Device::Cpu,
        )
        .unwrap();
        let g_b1c = Tensor::from_vec(
            vec![0.9f32, -0.8, 0.7, 0.6, 0.5, -0.4, 0.3, -0.2],
            (2usize, 1usize, 4usize),
            &Device::Cpu,
        )
        .unwrap();
        let mut weights = zero_output_time_mixer();
        weights.w_o = LinearWeights {
            weight: Tensor::new(
                &[
                    [1.0f32, -0.25, 0.5, 0.0],
                    [0.0, 0.5, -1.0, 0.25],
                    [0.25, 1.0, 0.0, -0.5],
                    [-0.75, 0.0, 0.5, 1.0],
                ],
                &Device::Cpu,
            )
            .unwrap(),
            bias: None,
        };

        let expected = time_mixer_output_candle(
            &in_b1c,
            &r_b1hk,
            &k_b1hk,
            &v_b1hk,
            &bonus,
            &out_b1c,
            &g_b1c,
            &weights.w_o,
            &mut None,
        )
        .unwrap();
        let actual = time_mixer_output_native_with_kernel(
            &in_b1c,
            &r_b1hk,
            &k_b1hk,
            &v_b1hk,
            &bonus,
            &out_b1c,
            &g_b1c,
            &weights,
            NativeLinearDotKernel::Scalar,
        )
        .unwrap()
        .unwrap();

        assert_eq!(actual.dims(), expected.dims());
        assert_tensor_close(&actual, &expected, 1e-6);

        let scratch = TimeMixerMiddleOutputScratch {
            middle: TimeMixerMiddleScratchOutput {
                out_bhk: Some(
                    Tensor::zeros((2usize, 2usize, 2usize), DType::F32, &Device::Cpu).unwrap(),
                ),
                out_values: Vec::new(),
                out_sum_values: None,
                out_variance_values: None,
                next_state_bhkk: None,
                next_state_values: None,
                k_values: k_b1hk.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                v_values: v_b1hk.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            },
            dot_kernel: NativeLinearDotKernel::Scalar,
        };
        let actual_from_scratch = time_mixer_output_native_from_middle_scratch(
            &in_b1c,
            &r_b1hk.reshape((2usize, 1usize, 4usize)).unwrap(),
            &bonus,
            &out_b1c,
            &g_b1c,
            &weights,
            &scratch,
        )
        .unwrap()
        .unwrap();
        assert_tensor_close(&actual_from_scratch, &expected, 1e-6);

        weights.out_group_norm = GroupNormWeights {
            weight: Tensor::new(&[1.1f32, -0.7, 0.6, 1.3], &Device::Cpu).unwrap(),
            bias: Tensor::new(&[0.2f32, -0.1, 0.05, -0.3], &Device::Cpu).unwrap(),
        };
        let raw_out_bhk = Tensor::from_vec(
            vec![0.2f32, -0.4, 1.1, 0.7, -0.5, 0.3, -1.2, 0.9],
            (2usize, 2usize, 2usize),
            &Device::Cpu,
        )
        .unwrap();
        let grouped_bc = group_norm_2d(
            &raw_out_bhk.reshape((2usize, 4usize)).unwrap(),
            weights.n_heads,
            &weights.out_group_norm.weight,
            &weights.out_group_norm.bias,
            TIME_MIXER_GROUP_NORM_EPS,
        )
        .unwrap();
        let grouped_b1c = grouped_bc.reshape((2usize, 1usize, 4usize)).unwrap();
        let expected_group_output = time_mixer_output_candle(
            &in_b1c,
            &r_b1hk,
            &k_b1hk,
            &v_b1hk,
            &bonus,
            &grouped_b1c,
            &g_b1c,
            &weights.w_o,
            &mut None,
        )
        .unwrap();
        let group_scratch = TimeMixerMiddleOutputScratch {
            middle: TimeMixerMiddleScratchOutput {
                out_bhk: None,
                out_values: raw_out_bhk.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                out_sum_values: None,
                out_variance_values: None,
                next_state_bhkk: None,
                next_state_values: None,
                k_values: k_b1hk.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                v_values: v_b1hk.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            },
            dot_kernel: NativeLinearDotKernel::Scalar,
        };
        let actual_group_output = time_mixer_group_output_scratch_native(
            &in_b1c,
            &r_b1hk.reshape((2usize, 1usize, 4usize)).unwrap(),
            &bonus,
            &g_b1c,
            &weights,
            &group_scratch,
        )
        .unwrap()
        .unwrap();
        assert_tensor_close(&actual_group_output, &expected_group_output, 1e-6);

        let mut cached_aux_weights = zero_output_time_mixer();
        cached_aux_weights.bonus = bonus.clone();
        cached_aux_weights.w_o = LinearWeights {
            weight: Tensor::new(
                &[
                    [1.0f32, -0.25, 0.5, 0.0],
                    [0.0, 0.5, -1.0, 0.25],
                    [0.25, 1.0, 0.0, -0.5],
                    [-0.75, 0.0, 0.5, 1.0],
                ],
                &Device::Cpu,
            )
            .unwrap(),
            bias: None,
        };
        cached_aux_weights.out_group_norm = GroupNormWeights {
            weight: Tensor::new(&[1.1f32, -0.7, 0.6, 1.3], &Device::Cpu).unwrap(),
            bias: Tensor::new(&[0.2f32, -0.1, 0.05, -0.3], &Device::Cpu).unwrap(),
        };
        let lora_values = TimeMixerLoraDecayScratchValues {
            batch_size: 2,
            channels: 4,
            a_values: vec![0.0f32; 8],
            g_values: g_b1c.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            w_values: vec![0.0f32; 8],
        };
        let r_values = r_b1hk
            .reshape((2usize, 4usize))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let actual_cached_aux_output =
            time_mixer_group_output_scratch_native_from_cached_aux_values(
                &in_b1c.reshape((2usize, 4usize)).unwrap(),
                &r_values,
                &lora_values,
                &cached_aux_weights,
                &group_scratch,
            )
            .unwrap()
            .unwrap();
        assert_tensor_close(&actual_cached_aux_output, &expected_group_output, 1e-6);
    }

    #[test]
    fn native_time_mixer_v_mix_matches_candle_sequence() {
        let mut weights = zero_output_time_mixer_with_layer_id(1);
        weights.v_lora_simple = LoraSimpleWeights {
            a: LinearWeights {
                weight: Tensor::new(
                    &[[0.25f32, -0.5, 0.75, 0.1], [-0.2, 0.4, 0.3, -0.6]],
                    &Device::Cpu,
                )
                .unwrap(),
                bias: None,
            },
            b_and_lamb: LinearWeights {
                weight: Tensor::new(
                    &[[0.5f32, -0.25], [-0.75, 0.2], [0.1, 0.9], [-0.3, -0.4]],
                    &Device::Cpu,
                )
                .unwrap(),
                bias: Some(Tensor::new(&[0.1f32, -0.2, 0.3, -0.4], &Device::Cpu).unwrap()),
            },
        };
        weights.w_v = LinearWeights {
            weight: Tensor::new(
                &[
                    [1.0f32, -0.25, 0.5, 0.0],
                    [0.0, 0.5, -1.0, 0.25],
                    [0.25, 1.0, 0.0, -0.5],
                    [-0.75, 0.0, 0.5, 1.0],
                ],
                &Device::Cpu,
            )
            .unwrap(),
            bias: None,
        };
        let v_b1c = Tensor::from_vec(
            vec![0.5f32, -1.0, 2.0, 0.25, 1.5, 0.25, -0.75, 2.0],
            (2usize, 1usize, 4usize),
            &Device::Cpu,
        )
        .unwrap();
        let v0_bc = Tensor::new(
            &[[0.2f32, 0.4, -0.8, 1.2], [1.0, -1.5, 0.3, 0.6]],
            &Device::Cpu,
        )
        .unwrap();

        let (expected, _) = time_mixer_v_mix_candle(&weights, &v_b1c, &v0_bc, &mut None).unwrap();
        let actual = time_mixer_v_mix_native_with_kernel(
            &v_b1c,
            &v0_bc,
            &weights,
            NativeLinearDotKernel::Scalar,
        )
        .unwrap()
        .unwrap();
        let actual_values = time_mixer_v_mix_values_native_with_kernel(
            &v_b1c,
            &v0_bc,
            &weights,
            NativeLinearDotKernel::Scalar,
        )
        .unwrap()
        .unwrap();
        let actual_from_values = Tensor::from_vec(
            actual_values.values,
            (actual_values.batch_size, 1usize, actual_values.channels),
            &Device::Cpu,
        )
        .unwrap();
        let v_values = v_b1c.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let actual_from_input_values =
            time_mixer_v_mix_values_native_from_input_values_with_kernel(
                &v_values,
                2,
                4,
                &v0_bc,
                &weights,
                NativeLinearDotKernel::Scalar,
            )
            .unwrap()
            .unwrap();
        let actual_from_input_values = Tensor::from_vec(
            actual_from_input_values.values,
            (
                actual_from_input_values.batch_size,
                1usize,
                actual_from_input_values.channels,
            ),
            &Device::Cpu,
        )
        .unwrap();

        assert_eq!(actual.dims(), expected.dims());
        assert_tensor_close(&actual, &expected, 1e-6);
        assert_tensor_close(&actual_from_values, &expected, 1e-6);
        assert_tensor_close(&actual_from_input_values, &expected, 1e-6);

        weights.layer_id = 0;
        let (expected_layer0, expected_layer0_v0) =
            time_mixer_v_mix(&weights, &v_b1c, &v0_bc, &mut None).unwrap();
        let actual_layer0_values = time_mixer_layer0_v_values_native_from_input_values_with_kernel(
            &v_values,
            2,
            4,
            &weights,
            NativeLinearDotKernel::Scalar,
        )
        .unwrap()
        .unwrap();
        let actual_layer0 = Tensor::from_vec(
            actual_layer0_values.values,
            (
                actual_layer0_values.batch_size,
                1usize,
                actual_layer0_values.channels,
            ),
            &Device::Cpu,
        )
        .unwrap();

        assert_tensor_close(&actual_layer0, &expected_layer0, 1e-6);
        assert_tensor_close(
            &actual_layer0.squeeze(1).unwrap(),
            &expected_layer0_v0,
            1e-6,
        );
    }

    #[test]
    fn native_time_mixer_lora_decay_scratch_matches_candle_sequence() {
        let mut weights = zero_output_time_mixer();
        weights.a_lora_simple = LoraSimpleWeights {
            a: LinearWeights {
                weight: Tensor::new(
                    &[[0.25f32, -0.5, 0.75, 0.1], [-0.2, 0.4, 0.3, -0.6]],
                    &Device::Cpu,
                )
                .unwrap(),
                bias: None,
            },
            b_and_lamb: LinearWeights {
                weight: Tensor::new(
                    &[[0.5f32, -0.25], [-0.75, 0.2], [0.1, 0.9], [-0.3, -0.4]],
                    &Device::Cpu,
                )
                .unwrap(),
                bias: Some(Tensor::new(&[0.1f32, -0.2, 0.3, -0.4], &Device::Cpu).unwrap()),
            },
        };
        weights.d_lora_mlp = LoraSimpleWeights {
            a: LinearWeights {
                weight: Tensor::new(
                    &[[-0.15f32, 0.45, -0.7, 0.2], [0.35, -0.1, 0.55, -0.25]],
                    &Device::Cpu,
                )
                .unwrap(),
                bias: None,
            },
            b_and_lamb: LinearWeights {
                weight: Tensor::new(
                    &[[0.2f32, -0.3], [0.6, 0.1], [-0.4, 0.25], [0.15, -0.5]],
                    &Device::Cpu,
                )
                .unwrap(),
                bias: Some(Tensor::new(&[-0.05f32, 0.15, -0.25, 0.35], &Device::Cpu).unwrap()),
            },
        };
        weights.gate_lora = GateLoraWeights {
            a: LinearWeights {
                weight: Tensor::new(
                    &[[0.4f32, -0.2, 0.1, 0.6], [-0.5, 0.3, 0.7, -0.1]],
                    &Device::Cpu,
                )
                .unwrap(),
                bias: None,
            },
            b: LinearWeights {
                weight: Tensor::new(
                    &[[0.3f32, -0.4], [0.1, 0.8], [-0.6, 0.2], [0.5, -0.7]],
                    &Device::Cpu,
                )
                .unwrap(),
                bias: None,
            },
        };
        let a_input_b1c = Tensor::from_vec(
            vec![0.5f32, -1.0, 2.0, 0.25, 1.5, 0.25, -0.75, 2.0],
            (2usize, 1usize, 4usize),
            &Device::Cpu,
        )
        .unwrap();
        let gate_input_b1c = Tensor::from_vec(
            vec![-0.25f32, 0.75, 1.0, -1.5, 0.2, -0.4, 0.6, 1.1],
            (2usize, 1usize, 4usize),
            &Device::Cpu,
        )
        .unwrap();
        let d_input_b1c = Tensor::from_vec(
            vec![1.25f32, -0.5, 0.3, 0.9, -1.0, 0.8, -0.2, 0.4],
            (2usize, 1usize, 4usize),
            &Device::Cpu,
        )
        .unwrap();

        let (expected_a, expected_g, expected_w) = time_mixer_lora_decay_candle(
            &weights,
            &a_input_b1c,
            &gate_input_b1c,
            &d_input_b1c,
            &mut None,
        )
        .unwrap();
        let (actual_a, actual_g, actual_w) = time_mixer_lora_decay_scratch_native_with_kernel(
            &weights,
            &a_input_b1c,
            &gate_input_b1c,
            &d_input_b1c,
            NativeLinearDotKernel::Scalar,
            &mut None,
        )
        .unwrap()
        .unwrap();
        let a_input_values = a_input_b1c.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let gate_input_values = gate_input_b1c
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let d_input_values = d_input_b1c.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let actual_values =
            time_mixer_lora_decay_scratch_values_native_from_input_values_with_kernel(
                &weights,
                &a_input_values,
                &gate_input_values,
                &d_input_values,
                2,
                4,
                NativeLinearDotKernel::Scalar,
                &mut None,
            )
            .unwrap()
            .unwrap();
        let actual_a_from_values = Tensor::from_vec(
            actual_values.a_values,
            (actual_values.batch_size, 1usize, actual_values.channels),
            &Device::Cpu,
        )
        .unwrap();
        let actual_g_from_values = Tensor::from_vec(
            actual_values.g_values,
            (actual_values.batch_size, 1usize, actual_values.channels),
            &Device::Cpu,
        )
        .unwrap();
        let actual_w_from_values = Tensor::from_vec(
            actual_values.w_values,
            (actual_values.batch_size, 1usize, actual_values.channels),
            &Device::Cpu,
        )
        .unwrap();

        assert_tensor_close(&actual_a, &expected_a, 1e-6);
        assert_tensor_close(&actual_g, &expected_g, 1e-6);
        assert_tensor_close(&actual_w, &expected_w, 1e-6);
        assert_tensor_close(&actual_a_from_values, &expected_a, 1e-6);
        assert_tensor_close(&actual_g_from_values, &expected_g, 1e-6);
        assert_tensor_close(&actual_w_from_values, &expected_w, 1e-6);
    }

    #[test]
    fn time_mixer_returns_residual_when_output_projection_is_zero() {
        let weights = zero_output_time_mixer();
        let input = Tensor::new(
            &[[1.0f32, 2.0, 3.0, 4.0], [-4.0, 1.0, 0.5, 2.5]],
            &Device::Cpu,
        )
        .unwrap();
        let v0 = Tensor::zeros((2usize, 4usize), DType::F32, &Device::Cpu).unwrap();

        let (output, next_v0, x_state, recurrent_state) =
            time_mixer_forward(&weights, &input, &v0, None).unwrap();

        assert_eq!(output.dims(), &[2, 4]);
        assert_eq!(next_v0.dims(), &[2, 4]);
        assert_eq!(x_state.dims(), &[2, 1, 4]);
        assert_eq!(recurrent_state.dims(), &[2, 1, 2, 2, 2]);
        assert_eq!(
            output.to_vec2::<f32>().unwrap(),
            input.to_vec2::<f32>().unwrap()
        );
    }

    #[test]
    fn time_mixer_rejects_wrong_recurrent_state_shape() {
        let weights = zero_output_time_mixer();
        let input = Tensor::zeros((2usize, 4usize), DType::F32, &Device::Cpu).unwrap();
        let v0 = Tensor::zeros((2usize, 4usize), DType::F32, &Device::Cpu).unwrap();
        let x_shift = Tensor::zeros((2usize, 1usize, 4usize), DType::F32, &Device::Cpu).unwrap();
        let recurrent_state = Tensor::zeros(
            (2usize, 1usize, 2usize, 2usize, 1usize),
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();

        let err = time_mixer_forward(&weights, &input, &v0, Some((&x_shift, &recurrent_state)))
            .unwrap_err();

        assert!(err
            .to_string()
            .contains("time_mixer expected state_B1HKK shape [2, 1, 2, 2, 2]"));
    }

    #[test]
    fn rwkv_layer_returns_residual_when_projections_are_zero() {
        let weights = Rwkv7RnnLayerWeights {
            layer_id: 0,
            time_mixer: zero_output_time_mixer(),
            channel_mixer: zero_projection_channel_mixer(),
        };
        let input = Tensor::new(
            &[[1.0f32, 2.0, 3.0, 4.0], [-4.0, 1.0, 0.5, 2.5]],
            &Device::Cpu,
        )
        .unwrap();
        let v0 = Tensor::zeros((2usize, 4usize), DType::F32, &Device::Cpu).unwrap();

        let (output, next_v0, time_x_state, time_recurrent_state, channel_state) =
            rwkv_layer_forward(&weights, &input, &v0, None, None).unwrap();

        assert_eq!(output.dims(), &[2, 4]);
        assert_eq!(next_v0.dims(), &[2, 4]);
        assert_eq!(time_x_state.dims(), &[2, 1, 4]);
        assert_eq!(time_recurrent_state.dims(), &[2, 1, 2, 2, 2]);
        assert_eq!(channel_state.dims(), &[2, 1, 4]);
        assert_eq!(
            output.to_vec2::<f32>().unwrap(),
            input.to_vec2::<f32>().unwrap()
        );
    }

    #[test]
    fn rwkv_rnn_returns_residual_when_projections_are_zero() {
        let weights = Rwkv7RnnWeights {
            module_index: 0,
            blocks: vec![zero_output_layer(0), zero_output_layer(1)],
        };
        let input = Tensor::new(
            &[[1.0f32, 2.0, 3.0, 4.0], [-4.0, 1.0, 0.5, 2.5]],
            &Device::Cpu,
        )
        .unwrap();

        let (output, time_x_states, recurrent_states, channel_states) =
            rwkv_rnn_forward(&weights, &input, None, None, None).unwrap();

        assert_eq!(output.dims(), &[2, 4]);
        assert_eq!(time_x_states.len(), 2);
        assert_eq!(recurrent_states.len(), 2);
        assert_eq!(channel_states.len(), 2);
        assert_eq!(time_x_states[0].dims(), &[2, 1, 4]);
        assert_eq!(recurrent_states[0].dims(), &[2, 1, 2, 2, 2]);
        assert_eq!(channel_states[0].dims(), &[2, 1, 4]);
        assert_eq!(
            output.to_vec2::<f32>().unwrap(),
            input.to_vec2::<f32>().unwrap()
        );
    }

    #[test]
    fn rwkv_rnn_rejects_incomplete_state_groups() {
        let weights = Rwkv7RnnWeights {
            module_index: 0,
            blocks: vec![zero_output_layer(0)],
        };
        let input = Tensor::zeros((2usize, 4usize), DType::F32, &Device::Cpu).unwrap();
        let state = Tensor::zeros((2usize, 1usize, 4usize), DType::F32, &Device::Cpu).unwrap();

        let err = rwkv_rnn_forward(&weights, &input, Some(&[state]), None, None).unwrap_err();

        assert!(err
            .to_string()
            .contains("rwkv_rnn_forward requires time_x_shift_b1c_by_layer"));
    }

    #[test]
    fn srs_review_preserves_review_state_order_and_head_shapes() {
        let weights = zero_srs_review_model();
        let input = Tensor::new(&[[1.0f32, 2.0, 3.0], [-4.0, 1.0, 0.5]], &Device::Cpu).unwrap();

        let (ahead, w, p, time_x_states, recurrent_states, channel_states) =
            srs_review_forward(&weights, &input, None, None, None, true).unwrap();

        assert_eq!(ahead.unwrap().dims(), &[2, NUM_POINTS]);
        assert_eq!(w.unwrap().dims(), &[2, NUM_CURVES]);
        assert_eq!(p.dims(), &[2, 4]);
        assert_eq!(
            time_x_states.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![1, 1, 1, 1, 1]
        );
        assert_eq!(
            recurrent_states.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![1, 1, 1, 1, 1]
        );
        assert_eq!(
            channel_states.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![1, 1, 1, 1, 1]
        );
        assert_eq!(time_x_states[CARD_STATE_INDEX][0].dims(), &[2, 1, 4]);
        assert_eq!(time_x_states[NOTE_STATE_INDEX][0].dims(), &[2, 1, 4]);
        assert_eq!(time_x_states[DECK_STATE_INDEX][0].dims(), &[2, 1, 4]);
        assert_eq!(time_x_states[PRESET_STATE_INDEX][0].dims(), &[2, 1, 4]);
        assert_eq!(time_x_states[GLOBAL_STATE_INDEX][0].dims(), &[2, 1, 4]);
    }

    #[test]
    fn srs_review_can_skip_curve_heads() {
        let weights = zero_srs_review_model();
        let input = Tensor::zeros((2usize, 3usize), DType::F32, &Device::Cpu).unwrap();

        let (ahead, w, p, _, _, _) =
            srs_review_forward(&weights, &input, None, None, None, false).unwrap();

        assert!(ahead.is_none());
        assert!(w.is_none());
        assert_eq!(p.dims(), &[2, 4]);
    }

    fn zero_srs_review_model() -> SrsRwkvRnnWeights {
        let d_model = 4usize;
        let feature_dim = 3usize;
        let feature_hidden = 8usize;
        let head_hidden = 8usize;
        SrsRwkvRnnWeights {
            features2card: Features2CardWeights {
                input_linear: zero_linear(feature_hidden, feature_dim, true),
                norm: unit_layer_norm(feature_hidden),
                output_linear: zero_linear(d_model, feature_hidden, true),
            },
            rwkv_modules: (0..SRS_REVIEW_STATE_MODULES)
                .map(|module_index| Rwkv7RnnWeights {
                    module_index,
                    blocks: vec![zero_output_layer(0)],
                })
                .collect(),
            prehead_norm: unit_layer_norm(d_model),
            head_ahead_logits: zero_linear(head_hidden, d_model, true),
            head_w: WHeadWeights {
                input_linear: zero_linear(d_model, d_model, true),
                norm: unit_layer_norm(d_model),
                output_linear: zero_linear(head_hidden, d_model, true),
            },
            head_p: zero_linear(head_hidden, d_model, true),
            ahead_linear: zero_linear(NUM_POINTS, head_hidden, true),
            w_linear: zero_linear(NUM_CURVES, head_hidden, true),
            p_linear: zero_linear(4, head_hidden, true),
        }
    }

    fn unit_layer_norm(width: usize) -> LayerNormWeights {
        LayerNormWeights {
            weight: Tensor::ones((width,), DType::F32, &Device::Cpu).unwrap(),
            bias: Tensor::zeros((width,), DType::F32, &Device::Cpu).unwrap(),
        }
    }

    fn assert_tensor_close(actual: &Tensor, expected: &Tensor, tolerance: f32) {
        assert_eq!(actual.dims(), expected.dims());
        let actual = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            let delta = (actual - expected).abs();
            assert!(
                delta <= tolerance,
                "value {index} differs by {delta}: actual={actual}, expected={expected}"
            );
        }
    }

    fn zero_output_layer(layer_id: usize) -> Rwkv7RnnLayerWeights {
        Rwkv7RnnLayerWeights {
            layer_id,
            time_mixer: zero_output_time_mixer_with_layer_id(layer_id),
            channel_mixer: zero_projection_channel_mixer(),
        }
    }

    fn zero_output_time_mixer() -> Rwkv7RnnTimeMixerWeights {
        zero_output_time_mixer_with_layer_id(0)
    }

    fn zero_output_time_mixer_with_layer_id(layer_id: usize) -> Rwkv7RnnTimeMixerWeights {
        let channels = 4usize;
        let heads = 2usize;
        let head_size = 2usize;
        Rwkv7RnnTimeMixerWeights {
            layer_id,
            n_heads: heads,
            head_size,
            layer_norm: LayerNormWeights {
                weight: Tensor::new(&[1.0f32, 1.0, 1.0, 1.0], &Device::Cpu).unwrap(),
                bias: Tensor::new(&[0.0f32, 0.0, 0.0, 0.0], &Device::Cpu).unwrap(),
            },
            rkvdag_lerp: Tensor::zeros(
                (8usize, 1usize, 1usize, channels),
                DType::F32,
                &Device::Cpu,
            )
            .unwrap(),
            bonus: Tensor::zeros((1usize, 1usize, heads, head_size), DType::F32, &Device::Cpu)
                .unwrap(),
            w_r: zero_linear(channels, channels, false),
            w_k: zero_linear(channels, channels, false),
            w_v: zero_linear(channels, channels, false),
            w_o: zero_linear(channels, channels, false),
            k_scale_linear: zero_linear(heads, channels, true),
            v_scale_linear: zero_linear(heads, channels, true),
            v_lora_simple: zero_lora_simple(channels, 2),
            a_lora_simple: zero_lora_simple(channels, 2),
            d_lora_mlp: zero_lora_simple(channels, 2),
            gate_lora: GateLoraWeights {
                a: zero_linear(2, channels, false),
                b: zero_linear(channels, 2, false),
            },
            out_group_norm: GroupNormWeights {
                weight: Tensor::new(&[1.0f32, 1.0, 1.0, 1.0], &Device::Cpu).unwrap(),
                bias: Tensor::new(&[0.0f32, 0.0, 0.0, 0.0], &Device::Cpu).unwrap(),
            },
            native_f32_cache: OnceLock::new(),
        }
    }

    fn zero_lora_simple(channels: usize, lora_dim: usize) -> LoraSimpleWeights {
        LoraSimpleWeights {
            a: zero_linear(lora_dim, channels, false),
            b_and_lamb: zero_linear(channels, lora_dim, true),
        }
    }

    fn zero_linear(out_dim: usize, in_dim: usize, bias: bool) -> LinearWeights {
        LinearWeights {
            weight: Tensor::zeros((out_dim, in_dim), DType::F32, &Device::Cpu).unwrap(),
            bias: bias.then(|| Tensor::zeros((out_dim,), DType::F32, &Device::Cpu).unwrap()),
        }
    }
}
