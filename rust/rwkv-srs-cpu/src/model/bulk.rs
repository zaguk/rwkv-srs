use std::collections::{BTreeMap, BTreeSet};

use candle_core::{Device, Result as CandleResult, Tensor, D};
use candle_nn::ops as nn_ops;
use pyo3::prelude::*;

use super::process_payload::parse_process_review_payload;
use super::runtime::{
    probabilities_from_logits, probability_from_logits, process_review_inputs_with_state,
    validate_num_threads, with_rayon_threads,
};
use super::state::{review_ids_from_prepared, NativeRnnModuleState, ReviewIds};
use super::undo::BatchUndoFrame;
use super::{
    features2card_forward_profiled, layer_norm, linear_profiled,
    linear_profiled_rowwise_native_exact, py_value_error,
    rwkv_rnn_forward_flat_time_state_profiled, rwkv_rnn_forward_flat_working_state_profiled,
    rwkv_rnn_forward_flat_working_state_values_profiled, rwkv_rnn_forward_profiled,
    rwkv_rnn_predict_forward_flat_time_state_profiled,
    rwkv_rnn_predict_forward_flat_working_state_profiled,
    rwkv_rnn_predict_forward_flat_working_state_values_profiled, rwkv_rnn_predict_forward_profiled,
    w_head_forward_profiled, ChannelMixerFlatLayerState, NativeProcessManyPyOutput, NativeRnn,
    TimeMixerFlatLayerState,
};
use crate::cpu_config::*;
use crate::model_weights::Rwkv7RnnWeights;
use crate::ops::f32_tensor_data;
use crate::profile::{ProfileTimer, RuntimeProfile};
use crate::state::{FeatureState, ReviewInput};
use crate::tensor_io::{tensor_from_2d, Tensor2List};

const BULK_EXECUTION_ORDER: [(BulkStreamModuleKind, usize); 5] = [
    (BulkStreamModuleKind::Card, 0),
    (BulkStreamModuleKind::Deck, 1),
    (BulkStreamModuleKind::Note, 2),
    (BulkStreamModuleKind::Preset, 3),
    (BulkStreamModuleKind::Global, 4),
];

struct BulkReplayChunk {
    inputs: Vec<ReviewInput>,
}

pub(super) struct BulkFeatureStep {
    pub(super) ids: ReviewIds,
    pub(super) predict_features: Vec<f32>,
    pub(super) process_features: Vec<f32>,
}

pub(super) struct BulkFeaturePrepass {
    pub(super) ids: Vec<ReviewIds>,
    pub(super) predict_features: Vec<Vec<f32>>,
    pub(super) process_features: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BulkStreamModuleKind {
    Card,
    Note,
    Deck,
    Preset,
    Global,
}

pub(super) struct BulkStreamPlan {
    pub(super) modules: Vec<BulkModuleStreamPlan>,
    pub(super) final_recurrent_state_keys: BulkStreamStateKeyPlan,
}

pub(super) struct BulkModuleStreamPlan {
    pub(super) kind: BulkStreamModuleKind,
    pub(super) streams: Vec<BulkStream>,
    pub(super) row_to_stream: Vec<usize>,
}

pub(super) struct BulkStream {
    pub(super) key: Option<i64>,
    pub(super) rows: Vec<usize>,
    pub(super) starts_existing: bool,
}

pub(super) struct BulkStreamStateKeyPlan {
    pub(super) card_states: Vec<i64>,
    pub(super) note_states: Vec<i64>,
    pub(super) deck_states: Vec<i64>,
    pub(super) preset_states: Vec<i64>,
    pub(super) global_state: bool,
}

impl BulkStreamModuleKind {
    const ALL: [Self; 5] = [
        Self::Card,
        Self::Note,
        Self::Deck,
        Self::Preset,
        Self::Global,
    ];

    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Card => "card",
            Self::Note => "note",
            Self::Deck => "deck",
            Self::Preset => "preset",
            Self::Global => "global",
        }
    }

    fn stream_key(self, ids: ReviewIds) -> Option<i64> {
        match self {
            Self::Card => Some(ids.0),
            Self::Note => Some(ids.1),
            Self::Deck => Some(ids.2),
            Self::Preset => Some(ids.3),
            Self::Global => None,
        }
    }

    fn state_exists(self, rnn: &NativeRnn, key: Option<i64>) -> bool {
        match self {
            Self::Card => key.is_some_and(|key| rnn.card_states.contains_key(&key)),
            Self::Note => key.is_some_and(|key| rnn.note_states.contains_key(&key)),
            Self::Deck => key.is_some_and(|key| rnn.deck_states.contains_key(&key)),
            Self::Preset => key.is_some_and(|key| rnn.preset_states.contains_key(&key)),
            Self::Global => rnn.global_state.is_some(),
        }
    }
}

impl BulkStreamPlan {
    fn from_ids(rnn: &NativeRnn, ids: &[ReviewIds]) -> Self {
        let modules = BulkStreamModuleKind::ALL
            .into_iter()
            .map(|kind| BulkModuleStreamPlan::from_ids(kind, rnn, ids))
            .collect();

        Self {
            modules,
            final_recurrent_state_keys: BulkStreamStateKeyPlan::from_ids(rnn, ids),
        }
    }

    fn module(&self, kind: BulkStreamModuleKind) -> Option<&BulkModuleStreamPlan> {
        self.modules.iter().find(|module| module.kind == kind)
    }
}

impl BulkModuleStreamPlan {
    fn from_ids(kind: BulkStreamModuleKind, rnn: &NativeRnn, ids: &[ReviewIds]) -> Self {
        let mut stream_indices = BTreeMap::new();
        let mut streams: Vec<BulkStream> = Vec::new();
        let mut row_to_stream = Vec::with_capacity(ids.len());

        for (row_index, ids) in ids.iter().copied().enumerate() {
            let key = kind.stream_key(ids);
            let stream_index = if let Some(index) = stream_indices.get(&key) {
                *index
            } else {
                let index = streams.len();
                stream_indices.insert(key, index);
                streams.push(BulkStream {
                    key,
                    rows: Vec::new(),
                    starts_existing: kind.state_exists(rnn, key),
                });
                index
            };
            streams[stream_index].rows.push(row_index);
            row_to_stream.push(stream_index);
        }

        Self {
            kind,
            streams,
            row_to_stream,
        }
    }
}

impl BulkStreamStateKeyPlan {
    fn from_ids(rnn: &NativeRnn, ids: &[ReviewIds]) -> Self {
        let mut card_states = rnn.card_states.keys().copied().collect::<BTreeSet<_>>();
        let mut note_states = rnn.note_states.keys().copied().collect::<BTreeSet<_>>();
        let mut deck_states = rnn.deck_states.keys().copied().collect::<BTreeSet<_>>();
        let mut preset_states = rnn.preset_states.keys().copied().collect::<BTreeSet<_>>();

        for (card_id, note_id, deck_id, preset_id) in ids.iter().copied() {
            card_states.insert(card_id);
            note_states.insert(note_id);
            deck_states.insert(deck_id);
            preset_states.insert(preset_id);
        }

        Self {
            card_states: card_states.into_iter().collect(),
            note_states: note_states.into_iter().collect(),
            deck_states: deck_states.into_iter().collect(),
            preset_states: preset_states.into_iter().collect(),
            global_state: rnn.global_state.is_some() || !ids.is_empty(),
        }
    }
}

impl BulkReplayChunk {
    fn from_payload(payload: &[u8], profile: &mut Option<&mut RuntimeProfile>) -> PyResult<Self> {
        let start = ProfileTimer::start(profile.is_some());
        let inputs = parse_process_review_payload(payload).map_err(py_value_error)?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.parse_review_ns += start.elapsed_ns();
        }
        Ok(Self { inputs })
    }

    fn process_reference(
        self,
        rnn: &mut NativeRnn,
        deterministic: &mut FeatureState,
        return_curves: bool,
        num_threads: Option<usize>,
        profile: &mut Option<&mut RuntimeProfile>,
    ) -> PyResult<NativeProcessManyPyOutput> {
        process_review_inputs_with_state(
            rnn,
            deterministic,
            &self.inputs,
            return_curves,
            num_threads,
            profile,
        )
    }

    fn feature_prepass(
        &self,
        deterministic: &mut FeatureState,
        profile: &mut Option<&mut RuntimeProfile>,
    ) -> PyResult<BulkFeaturePrepass> {
        let mut ids = Vec::with_capacity(self.inputs.len());
        let mut predict_features = Vec::with_capacity(self.inputs.len());
        let mut process_features = Vec::with_capacity(self.inputs.len());

        for input in &self.inputs {
            let step = feature_prepass_step(deterministic, input, profile)?;
            ids.push(step.ids);
            predict_features.push(step.predict_features);
            process_features.push(step.process_features);
        }

        Ok(BulkFeaturePrepass {
            ids,
            predict_features,
            process_features,
        })
    }

    fn process_layered(
        self,
        rnn: &mut NativeRnn,
        deterministic: &mut FeatureState,
        return_curves: bool,
        num_threads: Option<usize>,
        profile: &mut Option<&mut RuntimeProfile>,
    ) -> PyResult<NativeProcessManyPyOutput> {
        if self.inputs.is_empty() {
            return Ok((
                Vec::new(),
                return_curves.then(Vec::new),
                return_curves.then(Vec::new),
            ));
        }

        let ids = self
            .inputs
            .iter()
            .map(FeatureState::normalized_review_ids)
            .collect::<Vec<_>>();
        let transaction_start = ProfileTimer::start(profile.is_some());
        let undo = BatchUndoFrame::capture(rnn, deterministic, &ids);
        if let Some(profile) = profile.as_deref_mut() {
            profile.transaction_capture_ns += transaction_start.elapsed_ns();
        }
        let result = (|| {
            let prepass = self.feature_prepass(deterministic, profile)?;
            let stream_plan = BulkStreamPlan::from_ids(rnn, &prepass.ids);
            process_bulk_layered_prepass(
                rnn,
                prepass,
                stream_plan,
                return_curves,
                num_threads,
                profile,
            )
        })();
        match result {
            Ok(output) => Ok(output),
            Err(error) => {
                undo.restore(rnn, deterministic);
                Err(error)
            }
        }
    }
}

pub(super) fn feature_prepass_step(
    deterministic: &mut FeatureState,
    input: &ReviewInput,
    profile: &mut Option<&mut RuntimeProfile>,
) -> PyResult<BulkFeatureStep> {
    if let Some(profile) = profile.as_deref_mut() {
        profile.review_count += 1;
    }

    let start = ProfileTimer::start(profile.is_some());
    let predict_row = deterministic.prepare_predict_row(input);
    let ids = review_ids_from_prepared(&predict_row).map_err(py_value_error)?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.prepare_predict_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    let predict_features = deterministic
        .predict_feature_vector(&predict_row)
        .map_err(py_value_error)?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.predict_feature_ns += start.elapsed_ns();
        profile.materialization.predict_feature_vecs += 1;
        profile.materialization.predict_feature_vec_values += predict_features.len();
    }

    let start = ProfileTimer::start(profile.is_some());
    let process_row = deterministic
        .prepare_process_row(input)
        .map_err(py_value_error)?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.prepare_process_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    let process_features = deterministic
        .process_feature_vector(&process_row)
        .map_err(py_value_error)?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.process_feature_ns += start.elapsed_ns();
        profile.materialization.process_feature_vecs += 1;
        profile.materialization.process_feature_vec_values += process_features.len();
    }

    let start = ProfileTimer::start(profile.is_some());
    deterministic
        .record_recurrent_state_update(&process_row)
        .map_err(py_value_error)?;
    deterministic
        .record_processed_row(&process_row)
        .map_err(py_value_error)?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.state_update_ns += start.elapsed_ns();
    }

    Ok(BulkFeatureStep {
        ids,
        predict_features,
        process_features,
    })
}

pub(super) fn feature_prepass_step_into(
    deterministic: &mut FeatureState,
    input: &ReviewInput,
    features: &mut Vec<f32>,
) -> PyResult<ReviewIds> {
    deterministic
        .append_process_feature_pair(input, features)
        .map_err(py_value_error)
}

pub(super) fn process_reviews_bulk_reference_with_state(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    payload: &[u8],
    return_curves: bool,
    num_threads: Option<usize>,
    mut profile: Option<&mut RuntimeProfile>,
) -> PyResult<NativeProcessManyPyOutput> {
    validate_num_threads(num_threads)?;
    rnn.materialize_flat_cpu_state().map_err(py_value_error)?;
    let chunk = BulkReplayChunk::from_payload(payload, &mut profile)?;
    chunk.process_reference(rnn, deterministic, return_curves, num_threads, &mut profile)
}

pub(super) fn process_reviews_bulk_layered_with_state(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    payload: &[u8],
    return_curves: bool,
    num_threads: Option<usize>,
    mut profile: Option<&mut RuntimeProfile>,
) -> PyResult<NativeProcessManyPyOutput> {
    validate_num_threads(num_threads)?;
    rnn.materialize_flat_cpu_state().map_err(py_value_error)?;
    let chunk = BulkReplayChunk::from_payload(payload, &mut profile)?;
    chunk.process_layered(rnn, deterministic, return_curves, num_threads, &mut profile)
}

fn process_bulk_layered_prepass(
    rnn: &mut NativeRnn,
    prepass: BulkFeaturePrepass,
    stream_plan: BulkStreamPlan,
    return_curves: bool,
    num_threads: Option<usize>,
    profile: &mut Option<&mut RuntimeProfile>,
) -> PyResult<NativeProcessManyPyOutput> {
    let row_count = prepass.ids.len();
    let mut query_rows = BulkModuleRows::Tensor(features_to_card_rows(
        rnn,
        prepass.predict_features,
        BulkLane::Predict,
        "predict_features",
        profile,
    )?);
    let mut process_rows = BulkModuleRows::Tensor(features_to_card_rows(
        rnn,
        prepass.process_features,
        BulkLane::Process,
        "process_features",
        profile,
    )?);

    for (kind, weights_index) in BULK_EXECUTION_ORDER {
        let module_plan = stream_plan
            .module(kind)
            .expect("stream plan contains every bulk module kind");
        let weights = &rnn.weights.rwkv_modules[weights_index];
        let query_input = BulkModuleInput::from_rows(&query_rows)?;
        let process_input = BulkModuleInput::from_rows(&process_rows)?;
        let output = execute_bulk_module_pair(
            rnn,
            kind,
            weights,
            module_plan,
            query_input,
            process_input,
            num_threads,
            profile,
        )?;
        apply_bulk_module_final_states(rnn, kind, output.final_states);
        query_rows = output.query_outputs;
        process_rows = output.process_outputs;
    }

    let query_x = query_rows.into_tensor("final", "query_outputs")?;
    let prediction_probabilities =
        bulk_prediction_probabilities(rnn, &query_x, profile).map_err(py_value_error)?;
    if prediction_probabilities.len() != row_count {
        return Err(py_value_error(format!(
            "bulk layered predict returned {} probabilities for {row_count} rows",
            prediction_probabilities.len()
        )));
    }

    let (curve_ahead_logits, curve_w) = if return_curves {
        let process_x = process_rows.into_tensor("final", "process_outputs")?;
        let (ahead, w) = bulk_curve_outputs(rnn, &process_x, profile).map_err(py_value_error)?;
        (Some(ahead), Some(w))
    } else {
        (None, None)
    };

    Ok((prediction_probabilities, curve_ahead_logits, curve_w))
}

struct BulkModulePairOutput {
    query_outputs: BulkModuleRows,
    process_outputs: BulkModuleRows,
    final_states: BulkModuleFinalStates,
}

enum BulkModuleRows {
    Tensor(Tensor),
    Values {
        values: Vec<f32>,
        row_count: usize,
        channels: usize,
        device: Device,
    },
}

impl BulkModuleRows {
    fn into_tensor(self, module_name: &str, field_name: &str) -> PyResult<Tensor> {
        match self {
            Self::Tensor(tensor) => Ok(tensor),
            Self::Values {
                values,
                row_count,
                channels,
                device,
            } => Tensor::from_vec(values, (row_count, channels), &device).map_err(|error| {
                py_value_error(format!(
                    "{module_name} bulk {field_name} raw-value materialization failed: {error}"
                ))
            }),
        }
    }
}

#[derive(Clone, Copy)]
struct BulkModuleInput<'a> {
    tensor: Option<&'a Tensor>,
    values: Option<&'a [f32]>,
    row_count: usize,
    channels: usize,
    device: &'a Device,
}

impl<'a> BulkModuleInput<'a> {
    fn from_rows(rows: &'a BulkModuleRows) -> PyResult<Self> {
        match rows {
            BulkModuleRows::Tensor(tensor) => {
                let (row_count, channels) = tensor.dims2().map_err(py_value_error)?;
                Ok(Self {
                    tensor: Some(tensor),
                    values: None,
                    row_count,
                    channels,
                    device: tensor.device(),
                })
            }
            BulkModuleRows::Values {
                values,
                row_count,
                channels,
                device,
            } => Ok(Self {
                tensor: None,
                values: Some(values),
                row_count: *row_count,
                channels: *channels,
                device,
            }),
        }
    }

    fn narrow_row(
        &self,
        row_index: usize,
        module_name: &str,
        field_name: &str,
    ) -> PyResult<Tensor> {
        let tensor = self.tensor.ok_or_else(|| {
            py_value_error(format!(
                "{module_name} bulk {field_name} requested a tensor row from raw-value input"
            ))
        })?;
        tensor.narrow(0, row_index, 1).map_err(py_value_error)
    }
}

#[derive(Clone)]
pub(super) struct BulkWorkingModuleState {
    pub(super) time_state_by_layer: Vec<TimeMixerFlatLayerState>,
    pub(super) channel_state_b1c_by_layer: Option<Vec<Tensor>>,
    pub(super) channel_flat_state_by_layer: Option<Vec<ChannelMixerFlatLayerState>>,
}

struct BulkPersistentFrontierCandidate {
    stream_index: usize,
    key: i64,
    rows: Vec<usize>,
}

const BULK_PERSISTENT_FRONTIER_MIN_ROWS: usize = 4;

enum BulkModuleFinalStates {
    Keyed(Vec<(i64, NativeRnnModuleState)>),
    Global(Option<NativeRnnModuleState>),
}

#[derive(Clone, Copy)]
enum BulkLane {
    Predict,
    Process,
}

fn features_to_card_rows(
    rnn: &NativeRnn,
    features: Vec<Vec<f32>>,
    lane: BulkLane,
    tensor_name: &str,
    profile: &mut Option<&mut RuntimeProfile>,
) -> PyResult<Tensor> {
    let mut outputs = Vec::with_capacity(features.len());
    for feature_row in features {
        let feature_values = feature_row.len();
        let start = ProfileTimer::start(profile.is_some());
        let feature_tensor =
            tensor_from_2d(vec![feature_row], tensor_name).map_err(py_value_error)?;
        if let Some(profile) = profile.as_deref_mut() {
            match lane {
                BulkLane::Predict => profile.predict_tensor_ns += start.elapsed_ns(),
                BulkLane::Process => profile.process_tensor_ns += start.elapsed_ns(),
            }
            profile.materialization.feature_tensors += 1;
            profile.materialization.feature_tensor_values += feature_values;
        }

        let start = ProfileTimer::start(profile.is_some());
        let output = match lane {
            BulkLane::Predict => features2card_forward_profiled(
                &rnn.weights.features2card,
                &feature_tensor,
                &mut profile
                    .as_deref_mut()
                    .map(|profile| &mut profile.predict_forward),
            ),
            BulkLane::Process => features2card_forward_profiled(
                &rnn.weights.features2card,
                &feature_tensor,
                &mut profile
                    .as_deref_mut()
                    .map(|profile| &mut profile.process_forward),
            ),
        }
        .map_err(py_value_error)?;
        if let Some(profile) = profile.as_deref_mut() {
            match lane {
                BulkLane::Predict => profile.predict_forward_ns += start.elapsed_ns(),
                BulkLane::Process => profile.process_forward_ns += start.elapsed_ns(),
            }
        }
        outputs.push(output);
    }
    let output_refs = outputs.iter().collect::<Vec<_>>();
    Tensor::cat(&output_refs, 0).map_err(py_value_error)
}

#[allow(clippy::too_many_arguments)]
fn execute_bulk_module_pair(
    rnn: &NativeRnn,
    kind: BulkStreamModuleKind,
    weights: &Rwkv7RnnWeights,
    plan: &BulkModuleStreamPlan,
    query_inputs: BulkModuleInput<'_>,
    process_inputs: BulkModuleInput<'_>,
    num_threads: Option<usize>,
    profile: &mut Option<&mut RuntimeProfile>,
) -> PyResult<BulkModulePairOutput> {
    let query_rows = query_inputs.row_count;
    let process_rows = process_inputs.row_count;
    if query_rows != process_rows || query_rows != plan.row_to_stream.len() {
        return Err(py_value_error(format!(
            "{} bulk module expected {} rows, got query={}, process={}",
            kind.name(),
            plan.row_to_stream.len(),
            query_rows,
            process_rows
        )));
    }
    if query_inputs.channels != process_inputs.channels {
        return Err(py_value_error(format!(
            "{} bulk module expected matching query/process channels, got query={}, process={}",
            kind.name(),
            query_inputs.channels,
            process_inputs.channels
        )));
    }

    let channels = query_inputs.channels;
    let use_output_buffer = bulk_module_output_buffer_enabled();
    let mut query_outputs =
        OrderedModuleRows::new(query_rows, channels, query_inputs.device, use_output_buffer);
    let mut process_outputs = OrderedModuleRows::new(
        process_rows,
        channels,
        process_inputs.device,
        use_output_buffer,
    );
    let mut keyed_final_states = Vec::new();
    let mut global_final_state = None;
    let lane_parallel = bulk_lane_parallel_enabled() && profile.is_none();
    let borrow_lane_state = lane_parallel && bulk_lane_parallel_borrow_state_enabled();
    let stream_lane_parallel = lane_parallel && bulk_lane_parallel_stream_install_enabled();
    let flat_time_state = time_mixer_flat_working_state_enabled();
    let flat_channel_state = channel_mixer_flat_working_state_enabled();
    let profile_fast_paths = bulk_profile_fast_paths_enabled();
    let module_values_carrier = bulk_module_values_carrier_enabled()
        && flat_time_state
        && flat_channel_state
        && use_output_buffer
        && layer_values_carrier_enabled()
        && layer_time_channel_values_handoff_enabled();
    let raw_values_handoff = bulk_module_raw_values_handoff_enabled() && module_values_carrier;
    let new_singleton_batch_values = bulk_new_singleton_batch_values_enabled()
        && module_values_carrier
        && (profile.is_none() || profile_fast_paths)
        && kind != BulkStreamModuleKind::Global;
    let deck_persistent_frontier_values = bulk_deck_persistent_frontier_values_enabled()
        && module_values_carrier
        && (profile.is_none() || profile_fast_paths)
        && kind == BulkStreamModuleKind::Deck;
    let card_note_persistent_frontier_values = bulk_card_note_persistent_frontier_values_enabled()
        && module_values_carrier
        && (profile.is_none() || profile_fast_paths)
        && matches!(
            kind,
            BulkStreamModuleKind::Card | BulkStreamModuleKind::Note
        );
    let query_input_data = if module_values_carrier && query_inputs.values.is_none() {
        let tensor = query_inputs.tensor.ok_or_else(|| {
            py_value_error(format!(
                "{} bulk query input has raw values disabled but no tensor",
                kind.name()
            ))
        })?;
        Some(f32_tensor_data(tensor).map_err(py_value_error)?)
    } else {
        None
    };
    let process_input_data = if module_values_carrier && process_inputs.values.is_none() {
        let tensor = process_inputs.tensor.ok_or_else(|| {
            py_value_error(format!(
                "{} bulk process input has raw values disabled but no tensor",
                kind.name()
            ))
        })?;
        Some(f32_tensor_data(tensor).map_err(py_value_error)?)
    } else {
        None
    };
    let query_input_values = if module_values_carrier {
        Some(match query_inputs.values {
            Some(values) => values,
            None => query_input_data
                .as_ref()
                .expect("module values carrier captured query tensor data")
                .as_slice()
                .map_err(py_value_error)?,
        })
    } else {
        None
    };
    let process_input_values = if module_values_carrier {
        Some(match process_inputs.values {
            Some(values) => values,
            None => process_input_data
                .as_ref()
                .expect("module values carrier captured process tensor data")
                .as_slice()
                .map_err(py_value_error)?,
        })
    } else {
        None
    };

    let mut skip_streams = if new_singleton_batch_values
        || deck_persistent_frontier_values
        || card_note_persistent_frontier_values
    {
        vec![false; plan.streams.len()]
    } else {
        Vec::new()
    };
    if new_singleton_batch_values {
        let query_input_values =
            query_input_values.expect("module values carrier has query values");
        let process_input_values =
            process_input_values.expect("module values carrier has process values");
        let start = ProfileTimer::start(profile_fast_paths && profile.is_some());
        execute_new_singleton_stream_batch_values(
            kind,
            weights,
            plan,
            query_input_values,
            process_input_values,
            channels,
            query_inputs.device,
            num_threads,
            &mut query_outputs,
            &mut process_outputs,
            &mut keyed_final_states,
            &mut skip_streams,
        )?;
        record_bulk_fast_path_time(profile, start.elapsed_ns());
    }
    if deck_persistent_frontier_values {
        let query_input_values =
            query_input_values.expect("module values carrier has query values");
        let process_input_values =
            process_input_values.expect("module values carrier has process values");
        let start = ProfileTimer::start(profile_fast_paths && profile.is_some());
        execute_persistent_frontier_values(
            rnn,
            weights,
            plan,
            query_input_values,
            process_input_values,
            channels,
            query_inputs.device,
            num_threads,
            BULK_PERSISTENT_FRONTIER_MIN_ROWS,
            None,
            &mut query_outputs,
            &mut process_outputs,
            &mut keyed_final_states,
            &mut skip_streams,
        )?;
        record_bulk_fast_path_time(profile, start.elapsed_ns());
    }
    if card_note_persistent_frontier_values {
        let query_input_values =
            query_input_values.expect("module values carrier has query values");
        let process_input_values =
            process_input_values.expect("module values carrier has process values");
        let start = ProfileTimer::start(profile_fast_paths && profile.is_some());
        execute_persistent_frontier_values(
            rnn,
            weights,
            plan,
            query_input_values,
            process_input_values,
            channels,
            query_inputs.device,
            num_threads,
            BULK_PERSISTENT_FRONTIER_MIN_ROWS,
            Some(bulk_card_note_persistent_frontier_batch_size()),
            &mut query_outputs,
            &mut process_outputs,
            &mut keyed_final_states,
            &mut skip_streams,
        )?;
        record_bulk_fast_path_time(profile, start.elapsed_ns());
    }

    for (stream_index, stream) in plan.streams.iter().enumerate() {
        if !skip_streams.is_empty() && skip_streams[stream_index] {
            continue;
        }
        let mut current_state = initial_module_state(rnn, kind, stream.key);
        let mut current_flat_state = if flat_time_state {
            if let Some(state) = current_state.as_ref() {
                let profiled_values = profile
                    .as_ref()
                    .map(|_| native_module_state_value_count(state));
                let start = ProfileTimer::start(profile.is_some());
                let flat_state = flat_working_module_state_from_native(state, flat_channel_state)
                    .map_err(py_value_error)?;
                if let Some(profile) = profile.as_deref_mut() {
                    profile.materialization.bulk_flat_state_input_snapshots += 1;
                    profile.materialization.bulk_flat_state_input_values +=
                        profiled_values.expect("bulk flat state input count captured");
                    profile.materialization.bulk_flat_state_input_ns += start.elapsed_ns();
                }
                Some(flat_state)
            } else {
                None
            }
        } else {
            None
        };
        if flat_time_state && stream_lane_parallel {
            let parallel_threads = bulk_lane_parallel_threads(num_threads);
            with_rayon_threads(Some(parallel_threads), || {
                for &row_index in &stream.rows {
                    if module_values_carrier {
                        let query_input_values =
                            query_input_values.expect("module values carrier has query values");
                        let process_input_values =
                            process_input_values.expect("module values carrier has process values");
                        let row_start = row_index * channels;
                        let query_row_values = &query_input_values[row_start..row_start + channels];
                        let process_row_values =
                            &process_input_values[row_start..row_start + channels];
                        let time_state = current_flat_state
                            .as_ref()
                            .map(|state| state.time_state_by_layer.as_slice());
                        let (query_result, process_result) = if borrow_lane_state {
                            let channel_state = current_flat_state
                                .as_ref()
                                .and_then(|state| state.channel_flat_state_by_layer.as_deref());
                            rayon::join(
                                || {
                                    rwkv_rnn_predict_forward_flat_working_state_values_profiled(
                                        weights,
                                        query_row_values,
                                        1,
                                        channels,
                                        query_inputs.device,
                                        time_state,
                                        channel_state,
                                        None,
                                    )
                                    .and_then(|output| {
                                        output.ok_or_else(|| {
                                            candle_core::Error::msg(
                                                "flat working-state values predict path declined",
                                            )
                                        })
                                    })
                                    .map_err(py_value_error)
                                },
                                || {
                                    rwkv_rnn_forward_flat_working_state_values_profiled(
                                        weights,
                                        process_row_values,
                                        1,
                                        channels,
                                        process_inputs.device,
                                        time_state,
                                        channel_state,
                                        None,
                                    )
                                    .and_then(|output| {
                                        output.ok_or_else(|| {
                                            candle_core::Error::msg(
                                                "flat working-state values process path declined",
                                            )
                                        })
                                    })
                                    .map_err(py_value_error)
                                },
                            )
                        } else {
                            let query_channel_state = current_flat_state
                                .as_ref()
                                .and_then(|state| state.channel_flat_state_by_layer.clone());
                            let process_channel_state = current_flat_state
                                .as_ref()
                                .and_then(|state| state.channel_flat_state_by_layer.clone());
                            rayon::join(
                                || {
                                    rwkv_rnn_predict_forward_flat_working_state_values_profiled(
                                        weights,
                                        query_row_values,
                                        1,
                                        channels,
                                        query_inputs.device,
                                        time_state,
                                        query_channel_state.as_deref(),
                                        None,
                                    )
                                    .and_then(|output| {
                                        output.ok_or_else(|| {
                                            candle_core::Error::msg(
                                                "flat working-state values predict path declined",
                                            )
                                        })
                                    })
                                    .map_err(py_value_error)
                                },
                                || {
                                    rwkv_rnn_forward_flat_working_state_values_profiled(
                                        weights,
                                        process_row_values,
                                        1,
                                        channels,
                                        process_inputs.device,
                                        time_state,
                                        process_channel_state.as_deref(),
                                        None,
                                    )
                                    .and_then(|output| {
                                        output.ok_or_else(|| {
                                            candle_core::Error::msg(
                                                "flat working-state values process path declined",
                                            )
                                        })
                                    })
                                    .map_err(py_value_error)
                                },
                            )
                        };

                        let query_output = query_result?;
                        let (process_output, next_time_state_by_layer, next_channel_state) =
                            process_result?;
                        query_outputs.store_values(
                            row_index,
                            &query_output,
                            kind.name(),
                            "query_outputs",
                        )?;
                        process_outputs.store_values(
                            row_index,
                            &process_output,
                            kind.name(),
                            "process_outputs",
                        )?;
                        current_flat_state = Some(BulkWorkingModuleState {
                            time_state_by_layer: next_time_state_by_layer,
                            channel_state_b1c_by_layer: None,
                            channel_flat_state_by_layer: Some(next_channel_state),
                        });
                        continue;
                    }

                    let query_input =
                        query_inputs.narrow_row(row_index, kind.name(), "query_inputs")?;
                    let process_input =
                        process_inputs.narrow_row(row_index, kind.name(), "process_inputs")?;
                    let time_state = current_flat_state
                        .as_ref()
                        .map(|state| state.time_state_by_layer.as_slice());

                    let (query_result, process_result) = if flat_channel_state {
                        let query_channel_state = current_flat_state
                            .as_ref()
                            .and_then(|state| state.channel_flat_state_by_layer.clone());
                        let process_channel_state = current_flat_state
                            .as_ref()
                            .and_then(|state| state.channel_flat_state_by_layer.clone());
                        rayon::join(
                            || {
                                rwkv_rnn_predict_forward_flat_working_state_profiled(
                                    weights,
                                    &query_input,
                                    time_state,
                                    query_channel_state.as_deref(),
                                    None,
                                )
                                .and_then(|output| {
                                    output.ok_or_else(|| {
                                        candle_core::Error::msg(
                                            "flat working-state predict path declined",
                                        )
                                    })
                                })
                                .map_err(py_value_error)
                            },
                            || {
                                rwkv_rnn_forward_flat_working_state_profiled(
                                    weights,
                                    &process_input,
                                    time_state,
                                    process_channel_state.as_deref(),
                                    None,
                                )
                                .and_then(|output| {
                                    output.ok_or_else(|| {
                                        candle_core::Error::msg(
                                            "flat working-state process path declined",
                                        )
                                    })
                                })
                                .map(|(output, time_state, channel_state)| {
                                    (output, time_state, None, Some(channel_state))
                                })
                                .map_err(py_value_error)
                            },
                        )
                    } else {
                        let query_channel_state = current_flat_state
                            .as_ref()
                            .and_then(|state| state.channel_state_b1c_by_layer.clone());
                        let process_channel_state = current_flat_state
                            .as_ref()
                            .and_then(|state| state.channel_state_b1c_by_layer.clone());
                        rayon::join(
                            || {
                                rwkv_rnn_predict_forward_flat_time_state_profiled(
                                    weights,
                                    &query_input,
                                    time_state,
                                    query_channel_state.as_deref(),
                                    None,
                                )
                                .and_then(|output| {
                                    output.ok_or_else(|| {
                                        candle_core::Error::msg(
                                            "flat TimeMixer predict path declined",
                                        )
                                    })
                                })
                                .map_err(py_value_error)
                            },
                            || {
                                rwkv_rnn_forward_flat_time_state_profiled(
                                    weights,
                                    &process_input,
                                    time_state,
                                    process_channel_state.as_deref(),
                                    None,
                                )
                                .and_then(|output| {
                                    output.ok_or_else(|| {
                                        candle_core::Error::msg(
                                            "flat TimeMixer process path declined",
                                        )
                                    })
                                })
                                .map(|(output, time_state, channel_state)| {
                                    (output, time_state, Some(channel_state), None)
                                })
                                .map_err(py_value_error)
                            },
                        )
                    };

                    let query_output = query_result?;
                    let (
                        process_output,
                        next_time_state_by_layer,
                        next_channel_state_b1c_by_layer,
                        next_channel_flat_state_by_layer,
                    ) = process_result?;
                    query_outputs.store(row_index, query_output, kind.name(), "query_outputs")?;
                    process_outputs.store(
                        row_index,
                        process_output,
                        kind.name(),
                        "process_outputs",
                    )?;
                    current_flat_state = Some(BulkWorkingModuleState {
                        time_state_by_layer: next_time_state_by_layer,
                        channel_state_b1c_by_layer: next_channel_state_b1c_by_layer,
                        channel_flat_state_by_layer: next_channel_flat_state_by_layer,
                    });
                }
                Ok(())
            })?;
        } else {
            for &row_index in &stream.rows {
                if module_values_carrier {
                    let query_input_values =
                        query_input_values.expect("module values carrier has query values");
                    let process_input_values =
                        process_input_values.expect("module values carrier has process values");
                    let row_start = row_index * channels;
                    let query_row_values = &query_input_values[row_start..row_start + channels];
                    let process_row_values = &process_input_values[row_start..row_start + channels];
                    let time_state = current_flat_state
                        .as_ref()
                        .map(|state| state.time_state_by_layer.as_slice());

                    if lane_parallel {
                        let parallel_threads = bulk_lane_parallel_threads(num_threads);
                        let query_channel_state = current_flat_state
                            .as_ref()
                            .and_then(|state| state.channel_flat_state_by_layer.clone());
                        let process_channel_state = current_flat_state
                            .as_ref()
                            .and_then(|state| state.channel_flat_state_by_layer.clone());
                        let (query_result, process_result) = with_rayon_threads(
                            Some(parallel_threads),
                            || {
                                Ok(rayon::join(
                                    || {
                                        rwkv_rnn_predict_forward_flat_working_state_values_profiled(
                                            weights,
                                            query_row_values,
                                            1,
                                            channels,
                                            query_inputs.device,
                                            time_state,
                                            query_channel_state.as_deref(),
                                            None,
                                        )
                                        .and_then(|output| {
                                            output.ok_or_else(|| {
                                                candle_core::Error::msg(
                                                    "flat working-state values predict path declined",
                                                )
                                            })
                                        })
                                        .map_err(py_value_error)
                                    },
                                    || {
                                        rwkv_rnn_forward_flat_working_state_values_profiled(
                                            weights,
                                            process_row_values,
                                            1,
                                            channels,
                                            process_inputs.device,
                                            time_state,
                                            process_channel_state.as_deref(),
                                            None,
                                        )
                                        .and_then(|output| {
                                            output.ok_or_else(|| {
                                                candle_core::Error::msg(
                                                    "flat working-state values process path declined",
                                                )
                                            })
                                        })
                                        .map_err(py_value_error)
                                    },
                                ))
                            },
                        )?;

                        let query_output = query_result?;
                        let (process_output, next_time_state_by_layer, next_channel_state) =
                            process_result?;
                        query_outputs.store_values(
                            row_index,
                            &query_output,
                            kind.name(),
                            "query_outputs",
                        )?;
                        process_outputs.store_values(
                            row_index,
                            &process_output,
                            kind.name(),
                            "process_outputs",
                        )?;
                        current_flat_state = Some(BulkWorkingModuleState {
                            time_state_by_layer: next_time_state_by_layer,
                            channel_state_b1c_by_layer: None,
                            channel_flat_state_by_layer: Some(next_channel_state),
                        });
                        continue;
                    }

                    let start = ProfileTimer::start(profile.is_some());
                    let query_channel_state = current_flat_state
                        .as_ref()
                        .and_then(|state| state.channel_flat_state_by_layer.as_deref());
                    let query_output = with_rayon_threads(num_threads, || {
                        rwkv_rnn_predict_forward_flat_working_state_values_profiled(
                            weights,
                            query_row_values,
                            1,
                            channels,
                            query_inputs.device,
                            time_state,
                            query_channel_state,
                            profile
                                .as_deref_mut()
                                .map(|profile| &mut profile.predict_forward),
                        )
                        .and_then(|output| {
                            output.ok_or_else(|| {
                                candle_core::Error::msg(
                                    "flat working-state values predict path declined",
                                )
                            })
                        })
                        .map_err(py_value_error)
                    })?;
                    if let Some(profile) = profile.as_deref_mut() {
                        let elapsed = start.elapsed_ns();
                        profile.predict_forward_ns += elapsed;
                        add_bulk_module_time(profile, BulkLane::Predict, kind, elapsed);
                    }
                    query_outputs.store_values(
                        row_index,
                        &query_output,
                        kind.name(),
                        "query_outputs",
                    )?;

                    let time_state = current_flat_state
                        .as_ref()
                        .map(|state| state.time_state_by_layer.as_slice());
                    let start = ProfileTimer::start(profile.is_some());
                    let process_channel_state = current_flat_state
                        .as_ref()
                        .and_then(|state| state.channel_flat_state_by_layer.as_deref());
                    let (process_output, next_time_state_by_layer, next_channel_state) =
                        with_rayon_threads(num_threads, || {
                            rwkv_rnn_forward_flat_working_state_values_profiled(
                                weights,
                                process_row_values,
                                1,
                                channels,
                                process_inputs.device,
                                time_state,
                                process_channel_state,
                                profile
                                    .as_deref_mut()
                                    .map(|profile| &mut profile.process_forward),
                            )
                            .and_then(|output| {
                                output.ok_or_else(|| {
                                    candle_core::Error::msg(
                                        "flat working-state values process path declined",
                                    )
                                })
                            })
                            .map_err(py_value_error)
                        })?;
                    if let Some(profile) = profile.as_deref_mut() {
                        let elapsed = start.elapsed_ns();
                        profile.process_forward_ns += elapsed;
                        add_bulk_module_time(profile, BulkLane::Process, kind, elapsed);
                    }
                    process_outputs.store_values(
                        row_index,
                        &process_output,
                        kind.name(),
                        "process_outputs",
                    )?;
                    current_flat_state = Some(BulkWorkingModuleState {
                        time_state_by_layer: next_time_state_by_layer,
                        channel_state_b1c_by_layer: None,
                        channel_flat_state_by_layer: Some(next_channel_state),
                    });
                    continue;
                }

                let query_input =
                    query_inputs.narrow_row(row_index, kind.name(), "query_inputs")?;
                let process_input =
                    process_inputs.narrow_row(row_index, kind.name(), "process_inputs")?;

                if flat_time_state {
                    if lane_parallel {
                        let time_state = current_flat_state
                            .as_ref()
                            .map(|state| state.time_state_by_layer.as_slice());
                        let parallel_threads = bulk_lane_parallel_threads(num_threads);
                        let (query_result, process_result) = if flat_channel_state {
                            let query_channel_state = current_flat_state
                                .as_ref()
                                .and_then(|state| state.channel_flat_state_by_layer.clone());
                            let process_channel_state = current_flat_state
                                .as_ref()
                                .and_then(|state| state.channel_flat_state_by_layer.clone());
                            with_rayon_threads(Some(parallel_threads), || {
                                Ok(rayon::join(
                                    || {
                                        rwkv_rnn_predict_forward_flat_working_state_profiled(
                                            weights,
                                            &query_input,
                                            time_state,
                                            query_channel_state.as_deref(),
                                            None,
                                        )
                                        .and_then(|output| {
                                            output.ok_or_else(|| {
                                                candle_core::Error::msg(
                                                    "flat working-state predict path declined",
                                                )
                                            })
                                        })
                                        .map_err(py_value_error)
                                    },
                                    || {
                                        rwkv_rnn_forward_flat_working_state_profiled(
                                            weights,
                                            &process_input,
                                            time_state,
                                            process_channel_state.as_deref(),
                                            None,
                                        )
                                        .and_then(|output| {
                                            output.ok_or_else(|| {
                                                candle_core::Error::msg(
                                                    "flat working-state process path declined",
                                                )
                                            })
                                        })
                                        .map(|(output, time_state, channel_state)| {
                                            (output, time_state, None, Some(channel_state))
                                        })
                                        .map_err(py_value_error)
                                    },
                                ))
                            })?
                        } else {
                            let query_channel_state = current_flat_state
                                .as_ref()
                                .and_then(|state| state.channel_state_b1c_by_layer.clone());
                            let process_channel_state = current_flat_state
                                .as_ref()
                                .and_then(|state| state.channel_state_b1c_by_layer.clone());
                            with_rayon_threads(Some(parallel_threads), || {
                                Ok(rayon::join(
                                    || {
                                        rwkv_rnn_predict_forward_flat_time_state_profiled(
                                            weights,
                                            &query_input,
                                            time_state,
                                            query_channel_state.as_deref(),
                                            None,
                                        )
                                        .and_then(|output| {
                                            output.ok_or_else(|| {
                                                candle_core::Error::msg(
                                                    "flat TimeMixer predict path declined",
                                                )
                                            })
                                        })
                                        .map_err(py_value_error)
                                    },
                                    || {
                                        rwkv_rnn_forward_flat_time_state_profiled(
                                            weights,
                                            &process_input,
                                            time_state,
                                            process_channel_state.as_deref(),
                                            None,
                                        )
                                        .and_then(|output| {
                                            output.ok_or_else(|| {
                                                candle_core::Error::msg(
                                                    "flat TimeMixer process path declined",
                                                )
                                            })
                                        })
                                        .map(|(output, time_state, channel_state)| {
                                            (output, time_state, Some(channel_state), None)
                                        })
                                        .map_err(py_value_error)
                                    },
                                ))
                            })?
                        };

                        let query_output = query_result?;
                        let (
                            process_output,
                            next_time_state_by_layer,
                            next_channel_state_b1c_by_layer,
                            next_channel_flat_state_by_layer,
                        ) = process_result?;
                        query_outputs.store(
                            row_index,
                            query_output,
                            kind.name(),
                            "query_outputs",
                        )?;
                        process_outputs.store(
                            row_index,
                            process_output,
                            kind.name(),
                            "process_outputs",
                        )?;
                        current_flat_state = Some(BulkWorkingModuleState {
                            time_state_by_layer: next_time_state_by_layer,
                            channel_state_b1c_by_layer: next_channel_state_b1c_by_layer,
                            channel_flat_state_by_layer: next_channel_flat_state_by_layer,
                        });
                        continue;
                    }

                    let time_state = current_flat_state
                        .as_ref()
                        .map(|state| state.time_state_by_layer.as_slice());
                    let start = ProfileTimer::start(profile.is_some());
                    let query_output = if flat_channel_state {
                        let channel_state = current_flat_state
                            .as_ref()
                            .and_then(|state| state.channel_flat_state_by_layer.as_deref());
                        with_rayon_threads(num_threads, || {
                            rwkv_rnn_predict_forward_flat_working_state_profiled(
                                weights,
                                &query_input,
                                time_state,
                                channel_state,
                                profile
                                    .as_deref_mut()
                                    .map(|profile| &mut profile.predict_forward),
                            )
                            .and_then(|output| {
                                output.ok_or_else(|| {
                                    candle_core::Error::msg(
                                        "flat working-state predict path declined",
                                    )
                                })
                            })
                            .map_err(py_value_error)
                        })?
                    } else {
                        let channel_state = current_flat_state
                            .as_ref()
                            .and_then(|state| state.channel_state_b1c_by_layer.as_deref());
                        with_rayon_threads(num_threads, || {
                            rwkv_rnn_predict_forward_flat_time_state_profiled(
                                weights,
                                &query_input,
                                time_state,
                                channel_state,
                                profile
                                    .as_deref_mut()
                                    .map(|profile| &mut profile.predict_forward),
                            )
                            .and_then(|output| {
                                output.ok_or_else(|| {
                                    candle_core::Error::msg("flat TimeMixer predict path declined")
                                })
                            })
                            .map_err(py_value_error)
                        })?
                    };
                    if let Some(profile) = profile.as_deref_mut() {
                        let elapsed = start.elapsed_ns();
                        profile.predict_forward_ns += elapsed;
                        add_bulk_module_time(profile, BulkLane::Predict, kind, elapsed);
                    }
                    query_outputs.store(row_index, query_output, kind.name(), "query_outputs")?;

                    let time_state = current_flat_state
                        .as_ref()
                        .map(|state| state.time_state_by_layer.as_slice());
                    let start = ProfileTimer::start(profile.is_some());
                    let (
                        process_output,
                        next_time_state_by_layer,
                        next_channel_state_b1c_by_layer,
                        next_channel_flat_state_by_layer,
                    ) = if flat_channel_state {
                        let channel_state = current_flat_state
                            .as_ref()
                            .and_then(|state| state.channel_flat_state_by_layer.as_deref());
                        let (process_output, next_time_state, next_channel_state) =
                            with_rayon_threads(num_threads, || {
                                rwkv_rnn_forward_flat_working_state_profiled(
                                    weights,
                                    &process_input,
                                    time_state,
                                    channel_state,
                                    profile
                                        .as_deref_mut()
                                        .map(|profile| &mut profile.process_forward),
                                )
                                .and_then(|output| {
                                    output.ok_or_else(|| {
                                        candle_core::Error::msg(
                                            "flat working-state process path declined",
                                        )
                                    })
                                })
                                .map_err(py_value_error)
                            })?;
                        (
                            process_output,
                            next_time_state,
                            None,
                            Some(next_channel_state),
                        )
                    } else {
                        let channel_state = current_flat_state
                            .as_ref()
                            .and_then(|state| state.channel_state_b1c_by_layer.as_deref());
                        let (process_output, next_time_state, next_channel_state) =
                            with_rayon_threads(num_threads, || {
                                rwkv_rnn_forward_flat_time_state_profiled(
                                    weights,
                                    &process_input,
                                    time_state,
                                    channel_state,
                                    profile
                                        .as_deref_mut()
                                        .map(|profile| &mut profile.process_forward),
                                )
                                .and_then(|output| {
                                    output.ok_or_else(|| {
                                        candle_core::Error::msg(
                                            "flat TimeMixer process path declined",
                                        )
                                    })
                                })
                                .map_err(py_value_error)
                            })?;
                        (
                            process_output,
                            next_time_state,
                            Some(next_channel_state),
                            None,
                        )
                    };
                    if let Some(profile) = profile.as_deref_mut() {
                        let elapsed = start.elapsed_ns();
                        profile.process_forward_ns += elapsed;
                        add_bulk_module_time(profile, BulkLane::Process, kind, elapsed);
                    }
                    process_outputs.store(
                        row_index,
                        process_output,
                        kind.name(),
                        "process_outputs",
                    )?;
                    current_flat_state = Some(BulkWorkingModuleState {
                        time_state_by_layer: next_time_state_by_layer,
                        channel_state_b1c_by_layer: next_channel_state_b1c_by_layer,
                        channel_flat_state_by_layer: next_channel_flat_state_by_layer,
                    });
                    continue;
                }

                if lane_parallel {
                    let parallel_threads = bulk_lane_parallel_threads(num_threads);
                    let (query_result, process_result) =
                        with_rayon_threads(Some(parallel_threads), || {
                            if borrow_lane_state {
                                let (time_x, time_state, channel_state) =
                                    module_state_slices(current_state.as_ref());
                                Ok(rayon::join(
                                    || {
                                        rwkv_rnn_predict_forward_profiled(
                                            weights,
                                            &query_input,
                                            time_x,
                                            time_state,
                                            channel_state,
                                            None,
                                        )
                                        .map_err(py_value_error)
                                    },
                                    || {
                                        rwkv_rnn_forward_profiled(
                                            weights,
                                            &process_input,
                                            time_x,
                                            time_state,
                                            channel_state,
                                            None,
                                        )
                                        .map_err(py_value_error)
                                    },
                                ))
                            } else {
                                let query_state = current_state.clone();
                                let process_state = current_state.clone();
                                Ok(rayon::join(
                                    || {
                                        let (time_x, time_state, channel_state) =
                                            module_state_slices(query_state.as_ref());
                                        rwkv_rnn_predict_forward_profiled(
                                            weights,
                                            &query_input,
                                            time_x,
                                            time_state,
                                            channel_state,
                                            None,
                                        )
                                        .map_err(py_value_error)
                                    },
                                    || {
                                        let (time_x, time_state, channel_state) =
                                            module_state_slices(process_state.as_ref());
                                        rwkv_rnn_forward_profiled(
                                            weights,
                                            &process_input,
                                            time_x,
                                            time_state,
                                            channel_state,
                                            None,
                                        )
                                        .map_err(py_value_error)
                                    },
                                ))
                            }
                        })?;

                    let query_output = query_result?;
                    let (
                        process_output,
                        next_time_x_shift_b1c_by_layer,
                        next_time_state_b1hkk_by_layer,
                        next_channel_state_b1c_by_layer,
                    ) = process_result?;

                    query_outputs.store(row_index, query_output, kind.name(), "query_outputs")?;
                    process_outputs.store(
                        row_index,
                        process_output,
                        kind.name(),
                        "process_outputs",
                    )?;
                    current_state = Some(NativeRnnModuleState {
                        time_x_shift_b1c_by_layer: next_time_x_shift_b1c_by_layer,
                        time_state_b1hkk_by_layer: next_time_state_b1hkk_by_layer,
                        channel_state_b1c_by_layer: next_channel_state_b1c_by_layer,
                    });
                    continue;
                }

                let (time_x, time_state, channel_state) =
                    module_state_slices(current_state.as_ref());
                let start = ProfileTimer::start(profile.is_some());
                let query_output = with_rayon_threads(num_threads, || {
                    rwkv_rnn_predict_forward_profiled(
                        weights,
                        &query_input,
                        time_x,
                        time_state,
                        channel_state,
                        profile
                            .as_deref_mut()
                            .map(|profile| &mut profile.predict_forward),
                    )
                    .map_err(py_value_error)
                })?;
                if let Some(profile) = profile.as_deref_mut() {
                    let elapsed = start.elapsed_ns();
                    profile.predict_forward_ns += elapsed;
                    add_bulk_module_time(profile, BulkLane::Predict, kind, elapsed);
                }
                query_outputs.store(row_index, query_output, kind.name(), "query_outputs")?;

                let (time_x, time_state, channel_state) =
                    module_state_slices(current_state.as_ref());
                let start = ProfileTimer::start(profile.is_some());
                let (
                    process_output,
                    next_time_x_shift_b1c_by_layer,
                    next_time_state_b1hkk_by_layer,
                    next_channel_state_b1c_by_layer,
                ) = with_rayon_threads(num_threads, || {
                    rwkv_rnn_forward_profiled(
                        weights,
                        &process_input,
                        time_x,
                        time_state,
                        channel_state,
                        profile
                            .as_deref_mut()
                            .map(|profile| &mut profile.process_forward),
                    )
                    .map_err(py_value_error)
                })?;
                if let Some(profile) = profile.as_deref_mut() {
                    let elapsed = start.elapsed_ns();
                    profile.process_forward_ns += elapsed;
                    add_bulk_module_time(profile, BulkLane::Process, kind, elapsed);
                }
                process_outputs.store(row_index, process_output, kind.name(), "process_outputs")?;
                current_state = Some(NativeRnnModuleState {
                    time_x_shift_b1c_by_layer: next_time_x_shift_b1c_by_layer,
                    time_state_b1hkk_by_layer: next_time_state_b1hkk_by_layer,
                    channel_state_b1c_by_layer: next_channel_state_b1c_by_layer,
                });
            }
        }

        if flat_time_state {
            current_state = if let Some(state) = current_flat_state {
                let profiled_values = profile
                    .as_ref()
                    .map(|_| bulk_working_module_state_value_count(&state));
                let profiled_tensors = profile
                    .as_ref()
                    .map(|_| bulk_working_module_state_tensor_count(&state));
                let start = ProfileTimer::start(profile.is_some());
                let native_state =
                    native_module_state_from_flat_working(state, query_inputs.device)
                        .map_err(py_value_error)?;
                if let Some(profile) = profile.as_deref_mut() {
                    profile.materialization.bulk_flat_state_store_snapshots += 1;
                    profile.materialization.bulk_flat_state_store_tensors +=
                        profiled_tensors.expect("bulk flat state store tensor count captured");
                    profile.materialization.bulk_flat_state_store_values +=
                        profiled_values.expect("bulk flat state store value count captured");
                    profile.materialization.bulk_flat_state_store_ns += start.elapsed_ns();
                }
                Some(native_state)
            } else {
                None
            };
        }

        match kind {
            BulkStreamModuleKind::Global => {
                global_final_state = current_state;
            }
            _ => {
                let key = stream.key.ok_or_else(|| {
                    py_value_error(format!("{} stream is missing a state key", kind.name()))
                })?;
                let state = current_state.ok_or_else(|| {
                    py_value_error(format!(
                        "{} stream {key} produced no final state",
                        kind.name()
                    ))
                })?;
                keyed_final_states.push((key, state));
            }
        }
    }

    Ok(BulkModulePairOutput {
        query_outputs: query_outputs.finish_output(
            kind.name(),
            "query_outputs",
            raw_values_handoff,
        )?,
        process_outputs: process_outputs.finish_output(
            kind.name(),
            "process_outputs",
            raw_values_handoff,
        )?,
        final_states: match kind {
            BulkStreamModuleKind::Global => BulkModuleFinalStates::Global(global_final_state),
            _ => BulkModuleFinalStates::Keyed(keyed_final_states),
        },
    })
}

#[allow(clippy::too_many_arguments)]
fn execute_persistent_frontier_values(
    rnn: &NativeRnn,
    weights: &Rwkv7RnnWeights,
    plan: &BulkModuleStreamPlan,
    query_input_values: &[f32],
    process_input_values: &[f32],
    channels: usize,
    device: &Device,
    num_threads: Option<usize>,
    min_rows: usize,
    max_batch_size: Option<usize>,
    query_outputs: &mut OrderedModuleRows,
    process_outputs: &mut OrderedModuleRows,
    keyed_final_states: &mut Vec<(i64, NativeRnnModuleState)>,
    skip_streams: &mut [bool],
) -> PyResult<()> {
    if plan.kind == BulkStreamModuleKind::Global {
        return Ok(());
    }
    execute_persistent_frontier_group(
        rnn,
        weights,
        plan,
        false,
        query_input_values,
        process_input_values,
        channels,
        device,
        num_threads,
        min_rows,
        max_batch_size,
        query_outputs,
        process_outputs,
        keyed_final_states,
        skip_streams,
    )?;
    execute_persistent_frontier_group(
        rnn,
        weights,
        plan,
        true,
        query_input_values,
        process_input_values,
        channels,
        device,
        num_threads,
        min_rows,
        max_batch_size,
        query_outputs,
        process_outputs,
        keyed_final_states,
        skip_streams,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_persistent_frontier_group(
    rnn: &NativeRnn,
    weights: &Rwkv7RnnWeights,
    plan: &BulkModuleStreamPlan,
    starts_existing: bool,
    query_input_values: &[f32],
    process_input_values: &[f32],
    channels: usize,
    device: &Device,
    num_threads: Option<usize>,
    min_rows: usize,
    max_batch_size: Option<usize>,
    query_outputs: &mut OrderedModuleRows,
    process_outputs: &mut OrderedModuleRows,
    keyed_final_states: &mut Vec<(i64, NativeRnnModuleState)>,
    skip_streams: &mut [bool],
) -> PyResult<()> {
    let mut candidates = Vec::new();
    let mut existing_states = Vec::new();
    for (stream_index, stream) in plan.streams.iter().enumerate() {
        if skip_streams[stream_index] || stream.starts_existing != starts_existing {
            continue;
        }
        if stream.rows.len() < min_rows {
            continue;
        }
        let key = stream.key.ok_or_else(|| {
            py_value_error(format!(
                "{} stream is missing a state key",
                plan.kind.name()
            ))
        })?;
        if starts_existing {
            let state = initial_module_state(rnn, plan.kind, stream.key).ok_or_else(|| {
                py_value_error(format!(
                    "{} stream {key} is missing initial state",
                    plan.kind.name()
                ))
            })?;
            existing_states.push(state);
        }
        candidates.push(BulkPersistentFrontierCandidate {
            stream_index,
            key,
            rows: stream.rows.clone(),
        });
    }
    if candidates.len() < 2 {
        return Ok(());
    }

    let max_batch_size = max_batch_size.unwrap_or(candidates.len()).max(2);
    let mut start = 0usize;
    while start < candidates.len() {
        let end = (start + max_batch_size).min(candidates.len());
        if end - start >= 2 {
            let group_states = if starts_existing {
                &existing_states[start..end]
            } else {
                &[]
            };
            execute_persistent_frontier_candidate_group(
                weights,
                plan.kind,
                &candidates[start..end],
                starts_existing,
                group_states,
                query_input_values,
                process_input_values,
                channels,
                device,
                num_threads,
                query_outputs,
                process_outputs,
                keyed_final_states,
                skip_streams,
            )?;
        }
        start = end;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_persistent_frontier_candidate_group(
    weights: &Rwkv7RnnWeights,
    kind: BulkStreamModuleKind,
    candidates: &[BulkPersistentFrontierCandidate],
    starts_existing: bool,
    existing_states: &[NativeRnnModuleState],
    query_input_values: &[f32],
    process_input_values: &[f32],
    channels: usize,
    device: &Device,
    num_threads: Option<usize>,
    query_outputs: &mut OrderedModuleRows,
    process_outputs: &mut OrderedModuleRows,
    keyed_final_states: &mut Vec<(i64, NativeRnnModuleState)>,
    skip_streams: &mut [bool],
) -> PyResult<()> {
    if candidates.len() < 2 {
        return Ok(());
    }
    if starts_existing && candidates.len() != existing_states.len() {
        return Err(py_value_error(format!(
            "{} persistent frontier expected {} existing states, got {}",
            kind.name(),
            candidates.len(),
            existing_states.len()
        )));
    }

    let mut current_state = if starts_existing {
        Some(
            batched_flat_working_module_state_from_native(existing_states)
                .map_err(py_value_error)?,
        )
    } else {
        None
    };
    let mut active = (0..candidates.len()).collect::<Vec<_>>();
    let parallel_threads = bulk_lane_parallel_threads(num_threads);
    let mut depth = 0usize;
    let reuse_input_buffers = bulk_persistent_frontier_input_buffer_reuse_enabled();
    let mut query_batch_values = Vec::new();
    let mut process_batch_values = Vec::new();

    while !active.is_empty() {
        let batch_size = active.len();
        let expected_values = batch_size * channels;
        if reuse_input_buffers {
            query_batch_values.clear();
            process_batch_values.clear();
            query_batch_values.reserve(expected_values);
            process_batch_values.reserve(expected_values);
        } else {
            query_batch_values = Vec::with_capacity(expected_values);
            process_batch_values = Vec::with_capacity(expected_values);
        }
        for &candidate_index in &active {
            let row_index = candidates[candidate_index].rows[depth];
            let row_start = row_index * channels;
            query_batch_values
                .extend_from_slice(&query_input_values[row_start..row_start + channels]);
            process_batch_values
                .extend_from_slice(&process_input_values[row_start..row_start + channels]);
        }

        let time_state = current_state
            .as_ref()
            .map(|state| state.time_state_by_layer.as_slice());
        let channel_state = current_state
            .as_ref()
            .and_then(|state| state.channel_flat_state_by_layer.as_deref());
        let (query_result, process_result) = with_rayon_threads(Some(parallel_threads), || {
            Ok(rayon::join(
                || {
                    rwkv_rnn_predict_forward_flat_working_state_values_profiled(
                        weights,
                        &query_batch_values,
                        batch_size,
                        channels,
                        device,
                        time_state,
                        channel_state,
                        None,
                    )
                    .and_then(|output| {
                        output.ok_or_else(|| {
                            candle_core::Error::msg("persistent frontier predict path declined")
                        })
                    })
                    .map_err(py_value_error)
                },
                || {
                    rwkv_rnn_forward_flat_working_state_values_profiled(
                        weights,
                        &process_batch_values,
                        batch_size,
                        channels,
                        device,
                        time_state,
                        channel_state,
                        None,
                    )
                    .and_then(|output| {
                        output.ok_or_else(|| {
                            candle_core::Error::msg("persistent frontier process path declined")
                        })
                    })
                    .map_err(py_value_error)
                },
            ))
        })?;

        let query_output_values = query_result?;
        let (process_output_values, next_time_state_by_layer, next_channel_state_by_layer) =
            process_result?;
        if query_output_values.len() != expected_values {
            return Err(py_value_error(format!(
                "{} persistent frontier query batch expected {expected_values} values, got {}",
                kind.name(),
                query_output_values.len()
            )));
        }
        if process_output_values.len() != expected_values {
            return Err(py_value_error(format!(
                "{} persistent frontier process batch expected {expected_values} values, got {}",
                kind.name(),
                process_output_values.len()
            )));
        }

        let mut continuing = Vec::new();
        let mut continuing_batch_indices = Vec::new();
        for (batch_index, &candidate_index) in active.iter().enumerate() {
            let candidate = &candidates[candidate_index];
            let row_index = candidate.rows[depth];
            let row_start = batch_index * channels;
            let row_end = row_start + channels;
            query_outputs.store_values(
                row_index,
                &query_output_values[row_start..row_end],
                kind.name(),
                "query_outputs",
            )?;
            process_outputs.store_values(
                row_index,
                &process_output_values[row_start..row_end],
                kind.name(),
                "process_outputs",
            )?;

            if depth + 1 == candidate.rows.len() {
                let flat_state = flat_working_module_state_batch_row(
                    &next_time_state_by_layer,
                    &next_channel_state_by_layer,
                    batch_index,
                    batch_size,
                )
                .map_err(py_value_error)?;
                let native_state = native_module_state_from_flat_working(flat_state, device)
                    .map_err(py_value_error)?;
                keyed_final_states.push((candidate.key, native_state));
                skip_streams[candidate.stream_index] = true;
            } else {
                continuing.push(candidate_index);
                continuing_batch_indices.push(batch_index);
            }
        }

        if continuing.is_empty() {
            break;
        }
        current_state = Some(
            select_batched_flat_working_state_rows(
                next_time_state_by_layer,
                next_channel_state_by_layer,
                batch_size,
                &continuing_batch_indices,
            )
            .map_err(py_value_error)?,
        );
        active = continuing;
        depth += 1;
    }

    Ok(())
}

fn batched_flat_working_module_state_from_native(
    states: &[NativeRnnModuleState],
) -> CandleResult<BulkWorkingModuleState> {
    let first = states
        .first()
        .ok_or_else(|| candle_core::Error::msg("deck persistent frontier state batch is empty"))?;
    let layer_count = first.time_x_shift_b1c_by_layer.len();
    if first.time_state_b1hkk_by_layer.len() != layer_count
        || first.channel_state_b1c_by_layer.len() != layer_count
    {
        return Err(candle_core::Error::msg(
            "deck persistent frontier initial state layer counts do not match",
        ));
    }

    let flat_states = states
        .iter()
        .map(|state| flat_working_module_state_from_native(state, true))
        .collect::<CandleResult<Vec<_>>>()?;
    let mut time_state_by_layer = Vec::with_capacity(layer_count);
    let mut channel_state_by_layer = Vec::with_capacity(layer_count);

    for layer_index in 0..layer_count {
        let first_layer = &flat_states[0].time_state_by_layer[layer_index];
        let channels = first_layer.channels;
        let heads = first_layer.heads;
        let head_size = first_layer.head_size;
        let recurrent_len = heads * head_size * head_size;
        let mut x_shift_values = Vec::with_capacity(states.len() * channels);
        let mut state_values = Vec::with_capacity(states.len() * recurrent_len);
        for (state_index, state) in flat_states.iter().enumerate() {
            let layer = state.time_state_by_layer.get(layer_index).ok_or_else(|| {
                candle_core::Error::msg(format!(
                    "deck persistent frontier state {state_index} is missing TimeMixer layer {layer_index}"
                ))
            })?;
            if layer.channels != channels || layer.heads != heads || layer.head_size != head_size {
                return Err(candle_core::Error::msg(format!(
                    "deck persistent frontier TimeMixer layer {layer_index} shape mismatch"
                )));
            }
            x_shift_values.extend_from_slice(&layer.x_shift_values);
            state_values.extend_from_slice(&layer.state_values);
        }
        time_state_by_layer.push(TimeMixerFlatLayerState {
            x_shift_values,
            state_values,
            channels,
            heads,
            head_size,
        });

        let first_channel_layers = flat_states[0]
            .channel_flat_state_by_layer
            .as_ref()
            .ok_or_else(|| {
                candle_core::Error::msg(
                    "deck persistent frontier initial state is missing flat ChannelMixer state",
                )
            })?;
        let first_channel = &first_channel_layers[layer_index];
        let mut channel_values = Vec::with_capacity(states.len() * first_channel.channels);
        for (state_index, state) in flat_states.iter().enumerate() {
            let channel_layers = state.channel_flat_state_by_layer.as_ref().ok_or_else(|| {
                candle_core::Error::msg(format!(
                    "deck persistent frontier state {state_index} is missing flat ChannelMixer state"
                ))
            })?;
            let channel_layer = channel_layers.get(layer_index).ok_or_else(|| {
                candle_core::Error::msg(format!(
                    "deck persistent frontier state {state_index} is missing ChannelMixer layer {layer_index}"
                ))
            })?;
            if channel_layer.channels != first_channel.channels {
                return Err(candle_core::Error::msg(format!(
                    "deck persistent frontier ChannelMixer layer {layer_index} shape mismatch"
                )));
            }
            channel_values.extend_from_slice(&channel_layer.state_values);
        }
        channel_state_by_layer.push(ChannelMixerFlatLayerState {
            state_values: channel_values,
            channels: first_channel.channels,
        });
    }

    Ok(BulkWorkingModuleState {
        time_state_by_layer,
        channel_state_b1c_by_layer: None,
        channel_flat_state_by_layer: Some(channel_state_by_layer),
    })
}

fn select_batched_flat_working_state_rows(
    time_state_by_layer: Vec<TimeMixerFlatLayerState>,
    channel_state_by_layer: Vec<ChannelMixerFlatLayerState>,
    batch_size: usize,
    selected_rows: &[usize],
) -> CandleResult<BulkWorkingModuleState> {
    if selected_rows.len() == batch_size
        && selected_rows
            .iter()
            .copied()
            .enumerate()
            .all(|(expected, actual)| expected == actual)
    {
        return Ok(BulkWorkingModuleState {
            time_state_by_layer,
            channel_state_b1c_by_layer: None,
            channel_flat_state_by_layer: Some(channel_state_by_layer),
        });
    }

    if bulk_persistent_frontier_in_place_state_compaction_enabled()
        && selected_rows_are_front_compactable(batch_size, selected_rows)
    {
        return compact_batched_flat_working_state_rows_in_place(
            time_state_by_layer,
            channel_state_by_layer,
            batch_size,
            selected_rows,
        );
    }

    let mut selected_time_state_by_layer = Vec::with_capacity(time_state_by_layer.len());
    for (layer_index, layer) in time_state_by_layer.into_iter().enumerate() {
        let x_stride = layer.channels;
        let state_stride = layer.heads * layer.head_size * layer.head_size;
        let expected_x = batch_size * x_stride;
        let expected_state = batch_size * state_stride;
        if layer.x_shift_values.len() != expected_x || layer.state_values.len() != expected_state {
            return Err(candle_core::Error::msg(format!(
                "deck persistent frontier TimeMixer layer {layer_index} expected {expected_x}/{expected_state} values, got {}/{}",
                layer.x_shift_values.len(),
                layer.state_values.len()
            )));
        }
        let mut x_shift_values = Vec::with_capacity(selected_rows.len() * x_stride);
        let mut state_values = Vec::with_capacity(selected_rows.len() * state_stride);
        for &row in selected_rows {
            if row >= batch_size {
                return Err(candle_core::Error::msg(format!(
                    "deck persistent frontier selected row {row} is outside batch size {batch_size}"
                )));
            }
            let x_start = row * x_stride;
            let state_start = row * state_stride;
            x_shift_values.extend_from_slice(&layer.x_shift_values[x_start..x_start + x_stride]);
            state_values
                .extend_from_slice(&layer.state_values[state_start..state_start + state_stride]);
        }
        selected_time_state_by_layer.push(TimeMixerFlatLayerState {
            x_shift_values,
            state_values,
            channels: layer.channels,
            heads: layer.heads,
            head_size: layer.head_size,
        });
    }

    let mut selected_channel_state_by_layer = Vec::with_capacity(channel_state_by_layer.len());
    for (layer_index, layer) in channel_state_by_layer.into_iter().enumerate() {
        let stride = layer.channels;
        let expected = batch_size * stride;
        if layer.state_values.len() != expected {
            return Err(candle_core::Error::msg(format!(
                "deck persistent frontier ChannelMixer layer {layer_index} expected {expected} values, got {}",
                layer.state_values.len()
            )));
        }
        let mut state_values = Vec::with_capacity(selected_rows.len() * stride);
        for &row in selected_rows {
            if row >= batch_size {
                return Err(candle_core::Error::msg(format!(
                    "deck persistent frontier selected row {row} is outside batch size {batch_size}"
                )));
            }
            let start = row * stride;
            state_values.extend_from_slice(&layer.state_values[start..start + stride]);
        }
        selected_channel_state_by_layer.push(ChannelMixerFlatLayerState {
            state_values,
            channels: layer.channels,
        });
    }

    Ok(BulkWorkingModuleState {
        time_state_by_layer: selected_time_state_by_layer,
        channel_state_b1c_by_layer: None,
        channel_flat_state_by_layer: Some(selected_channel_state_by_layer),
    })
}

fn selected_rows_are_front_compactable(batch_size: usize, selected_rows: &[usize]) -> bool {
    let mut previous = None;
    for &row in selected_rows {
        if row >= batch_size || previous.is_some_and(|previous| row <= previous) {
            return false;
        }
        previous = Some(row);
    }
    true
}

fn compact_batched_flat_working_state_rows_in_place(
    mut time_state_by_layer: Vec<TimeMixerFlatLayerState>,
    mut channel_state_by_layer: Vec<ChannelMixerFlatLayerState>,
    batch_size: usize,
    selected_rows: &[usize],
) -> CandleResult<BulkWorkingModuleState> {
    for (layer_index, layer) in time_state_by_layer.iter_mut().enumerate() {
        let x_stride = layer.channels;
        let state_stride = layer.heads * layer.head_size * layer.head_size;
        compact_flat_state_values_in_place(
            &mut layer.x_shift_values,
            batch_size,
            x_stride,
            selected_rows,
            "TimeMixer",
            layer_index,
            "x-shift",
        )?;
        compact_flat_state_values_in_place(
            &mut layer.state_values,
            batch_size,
            state_stride,
            selected_rows,
            "TimeMixer",
            layer_index,
            "recurrent",
        )?;
    }

    for (layer_index, layer) in channel_state_by_layer.iter_mut().enumerate() {
        compact_flat_state_values_in_place(
            &mut layer.state_values,
            batch_size,
            layer.channels,
            selected_rows,
            "ChannelMixer",
            layer_index,
            "state",
        )?;
    }

    Ok(BulkWorkingModuleState {
        time_state_by_layer,
        channel_state_b1c_by_layer: None,
        channel_flat_state_by_layer: Some(channel_state_by_layer),
    })
}

fn compact_flat_state_values_in_place(
    values: &mut Vec<f32>,
    batch_size: usize,
    stride: usize,
    selected_rows: &[usize],
    layer_kind: &str,
    layer_index: usize,
    value_kind: &str,
) -> CandleResult<()> {
    let expected = batch_size * stride;
    if values.len() != expected {
        return Err(candle_core::Error::msg(format!(
            "deck persistent frontier {layer_kind} layer {layer_index} {value_kind} expected {expected} values, got {}",
            values.len()
        )));
    }
    for (target_row, &source_row) in selected_rows.iter().enumerate() {
        if source_row >= batch_size {
            return Err(candle_core::Error::msg(format!(
                "deck persistent frontier selected row {source_row} is outside batch size {batch_size}"
            )));
        }
        if source_row != target_row {
            let source_start = source_row * stride;
            let target_start = target_row * stride;
            values.copy_within(source_start..source_start + stride, target_start);
        }
    }
    values.truncate(selected_rows.len() * stride);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_new_singleton_stream_batch_values(
    kind: BulkStreamModuleKind,
    weights: &Rwkv7RnnWeights,
    plan: &BulkModuleStreamPlan,
    query_input_values: &[f32],
    process_input_values: &[f32],
    channels: usize,
    device: &Device,
    num_threads: Option<usize>,
    query_outputs: &mut OrderedModuleRows,
    process_outputs: &mut OrderedModuleRows,
    keyed_final_states: &mut Vec<(i64, NativeRnnModuleState)>,
    skip_streams: &mut [bool],
) -> PyResult<()> {
    let mut candidates = Vec::new();
    for (stream_index, stream) in plan.streams.iter().enumerate() {
        if stream.starts_existing || stream.rows.len() != 1 {
            continue;
        }
        let key = stream.key.ok_or_else(|| {
            py_value_error(format!("{} stream is missing a state key", kind.name()))
        })?;
        candidates.push((stream_index, key, stream.rows[0]));
    }
    if candidates.len() < 2 {
        return Ok(());
    }

    let batch_size = candidates.len();
    let mut query_batch_values = Vec::with_capacity(batch_size * channels);
    let mut process_batch_values = Vec::with_capacity(batch_size * channels);
    for &(_, _, row_index) in &candidates {
        let row_start = row_index * channels;
        query_batch_values.extend_from_slice(&query_input_values[row_start..row_start + channels]);
        process_batch_values
            .extend_from_slice(&process_input_values[row_start..row_start + channels]);
    }

    let parallel_threads = bulk_lane_parallel_threads(num_threads);
    let (query_result, process_result) = with_rayon_threads(Some(parallel_threads), || {
        Ok(rayon::join(
            || {
                rwkv_rnn_predict_forward_flat_working_state_values_profiled(
                    weights,
                    &query_batch_values,
                    batch_size,
                    channels,
                    device,
                    None,
                    None,
                    None,
                )
                .and_then(|output| {
                    output.ok_or_else(|| {
                        candle_core::Error::msg(
                            "flat working-state values singleton predict path declined",
                        )
                    })
                })
                .map_err(py_value_error)
            },
            || {
                rwkv_rnn_forward_flat_working_state_values_profiled(
                    weights,
                    &process_batch_values,
                    batch_size,
                    channels,
                    device,
                    None,
                    None,
                    None,
                )
                .and_then(|output| {
                    output.ok_or_else(|| {
                        candle_core::Error::msg(
                            "flat working-state values singleton process path declined",
                        )
                    })
                })
                .map_err(py_value_error)
            },
        ))
    })?;

    let query_output_values = query_result?;
    let (process_output_values, next_time_state_by_layer, next_channel_state_by_layer) =
        process_result?;
    let expected_values = batch_size * channels;
    if query_output_values.len() != expected_values {
        return Err(py_value_error(format!(
            "{} singleton query batch expected {expected_values} values, got {}",
            kind.name(),
            query_output_values.len()
        )));
    }
    if process_output_values.len() != expected_values {
        return Err(py_value_error(format!(
            "{} singleton process batch expected {expected_values} values, got {}",
            kind.name(),
            process_output_values.len()
        )));
    }

    if bulk_new_singleton_state_views_enabled() {
        let mut native_states = native_module_states_from_batched_flat_working(
            next_time_state_by_layer,
            next_channel_state_by_layer,
            batch_size,
            device,
        )
        .map_err(py_value_error)?
        .into_iter();
        for (batch_index, (stream_index, key, row_index)) in candidates.into_iter().enumerate() {
            let row_start = batch_index * channels;
            let row_end = row_start + channels;
            query_outputs.store_values(
                row_index,
                &query_output_values[row_start..row_end],
                kind.name(),
                "query_outputs",
            )?;
            process_outputs.store_values(
                row_index,
                &process_output_values[row_start..row_end],
                kind.name(),
                "process_outputs",
            )?;
            let native_state = native_states.next().ok_or_else(|| {
                py_value_error(format!(
                    "{} singleton state-view batch produced too few states",
                    kind.name()
                ))
            })?;
            keyed_final_states.push((key, native_state));
            skip_streams[stream_index] = true;
        }
        if native_states.next().is_some() {
            return Err(py_value_error(format!(
                "{} singleton state-view batch produced too many states",
                kind.name()
            )));
        }
        return Ok(());
    }

    for (batch_index, (stream_index, key, row_index)) in candidates.into_iter().enumerate() {
        let row_start = batch_index * channels;
        let row_end = row_start + channels;
        query_outputs.store_values(
            row_index,
            &query_output_values[row_start..row_end],
            kind.name(),
            "query_outputs",
        )?;
        process_outputs.store_values(
            row_index,
            &process_output_values[row_start..row_end],
            kind.name(),
            "process_outputs",
        )?;
        let flat_state = flat_working_module_state_batch_row(
            &next_time_state_by_layer,
            &next_channel_state_by_layer,
            batch_index,
            batch_size,
        )
        .map_err(py_value_error)?;
        let native_state =
            native_module_state_from_flat_working(flat_state, device).map_err(py_value_error)?;
        keyed_final_states.push((key, native_state));
        skip_streams[stream_index] = true;
    }

    Ok(())
}

fn native_module_states_from_batched_flat_working(
    time_state_by_layer: Vec<TimeMixerFlatLayerState>,
    channel_state_by_layer: Vec<ChannelMixerFlatLayerState>,
    batch_size: usize,
    device: &Device,
) -> CandleResult<Vec<NativeRnnModuleState>> {
    let mut time_x_shift_batch_by_layer = Vec::with_capacity(time_state_by_layer.len());
    let mut time_state_batch_by_layer = Vec::with_capacity(time_state_by_layer.len());
    for (layer_index, layer) in time_state_by_layer.into_iter().enumerate() {
        let expected_x = batch_size * layer.channels;
        if layer.x_shift_values.len() != expected_x {
            return Err(candle_core::Error::msg(format!(
                "singleton state-view batch TimeMixer layer {layer_index} expected {expected_x} x-shift values, got {}",
                layer.x_shift_values.len()
            )));
        }
        let recurrent_len = layer.heads * layer.head_size * layer.head_size;
        let expected_state = batch_size * recurrent_len;
        if layer.state_values.len() != expected_state {
            return Err(candle_core::Error::msg(format!(
                "singleton state-view batch TimeMixer layer {layer_index} expected {expected_state} recurrent values, got {}",
                layer.state_values.len()
            )));
        }
        time_x_shift_batch_by_layer.push(Tensor::from_vec(
            layer.x_shift_values,
            (batch_size, 1usize, layer.channels),
            device,
        )?);
        time_state_batch_by_layer.push(Tensor::from_vec(
            layer.state_values,
            (
                batch_size,
                1usize,
                layer.heads,
                layer.head_size,
                layer.head_size,
            ),
            device,
        )?);
    }

    let mut channel_state_batch_by_layer = Vec::with_capacity(channel_state_by_layer.len());
    for (layer_index, layer) in channel_state_by_layer.into_iter().enumerate() {
        let expected = batch_size * layer.channels;
        if layer.state_values.len() != expected {
            return Err(candle_core::Error::msg(format!(
                "singleton state-view batch ChannelMixer layer {layer_index} expected {expected} state values, got {}",
                layer.state_values.len()
            )));
        }
        channel_state_batch_by_layer.push(Tensor::from_vec(
            layer.state_values,
            (batch_size, 1usize, layer.channels),
            device,
        )?);
    }

    let layer_count = time_x_shift_batch_by_layer.len();
    let mut states = Vec::with_capacity(batch_size);
    for batch_index in 0..batch_size {
        let mut time_x_shift_b1c_by_layer = Vec::with_capacity(layer_count);
        let mut time_state_b1hkk_by_layer = Vec::with_capacity(layer_count);
        let mut channel_state_b1c_by_layer = Vec::with_capacity(channel_state_batch_by_layer.len());
        for tensor in &time_x_shift_batch_by_layer {
            time_x_shift_b1c_by_layer.push(tensor.narrow(0, batch_index, 1)?);
        }
        for tensor in &time_state_batch_by_layer {
            time_state_b1hkk_by_layer.push(tensor.narrow(0, batch_index, 1)?);
        }
        for tensor in &channel_state_batch_by_layer {
            channel_state_b1c_by_layer.push(tensor.narrow(0, batch_index, 1)?);
        }
        states.push(NativeRnnModuleState {
            time_x_shift_b1c_by_layer,
            time_state_b1hkk_by_layer,
            channel_state_b1c_by_layer,
        });
    }
    Ok(states)
}

fn flat_working_module_state_batch_row(
    time_state_by_layer: &[TimeMixerFlatLayerState],
    channel_state_by_layer: &[ChannelMixerFlatLayerState],
    batch_index: usize,
    batch_size: usize,
) -> CandleResult<BulkWorkingModuleState> {
    let mut time_layers = Vec::with_capacity(time_state_by_layer.len());
    for (layer_index, layer) in time_state_by_layer.iter().enumerate() {
        let x_stride = layer.channels;
        let state_stride = layer.heads * layer.head_size * layer.head_size;
        let expected_x = batch_size * x_stride;
        if layer.x_shift_values.len() != expected_x {
            return Err(candle_core::Error::msg(format!(
                "singleton batch TimeMixer layer {layer_index} expected {expected_x} x-shift values, got {}",
                layer.x_shift_values.len()
            )));
        }
        let expected_state = batch_size * state_stride;
        if layer.state_values.len() != expected_state {
            return Err(candle_core::Error::msg(format!(
                "singleton batch TimeMixer layer {layer_index} expected {expected_state} recurrent values, got {}",
                layer.state_values.len()
            )));
        }
        let x_start = batch_index * x_stride;
        let state_start = batch_index * state_stride;
        time_layers.push(TimeMixerFlatLayerState {
            x_shift_values: layer.x_shift_values[x_start..x_start + x_stride].to_vec(),
            state_values: layer.state_values[state_start..state_start + state_stride].to_vec(),
            channels: layer.channels,
            heads: layer.heads,
            head_size: layer.head_size,
        });
    }

    let mut channel_layers = Vec::with_capacity(channel_state_by_layer.len());
    for (layer_index, layer) in channel_state_by_layer.iter().enumerate() {
        let stride = layer.channels;
        let expected = batch_size * stride;
        if layer.state_values.len() != expected {
            return Err(candle_core::Error::msg(format!(
                "singleton batch ChannelMixer layer {layer_index} expected {expected} state values, got {}",
                layer.state_values.len()
            )));
        }
        let start = batch_index * stride;
        channel_layers.push(ChannelMixerFlatLayerState {
            state_values: layer.state_values[start..start + stride].to_vec(),
            channels: layer.channels,
        });
    }

    Ok(BulkWorkingModuleState {
        time_state_by_layer: time_layers,
        channel_state_b1c_by_layer: None,
        channel_flat_state_by_layer: Some(channel_layers),
    })
}

enum OrderedModuleRows {
    Tensors {
        rows: Vec<Option<Tensor>>,
    },
    Flat {
        values: Vec<f32>,
        seen: Vec<bool>,
        row_count: usize,
        channels: usize,
        device: Device,
    },
}

impl OrderedModuleRows {
    fn new(row_count: usize, channels: usize, device: &Device, flat: bool) -> Self {
        if flat {
            Self::Flat {
                values: vec![0.0f32; row_count * channels],
                seen: vec![false; row_count],
                row_count,
                channels,
                device: device.clone(),
            }
        } else {
            Self::Tensors {
                rows: (0..row_count).map(|_| None).collect(),
            }
        }
    }

    fn store(
        &mut self,
        row_index: usize,
        tensor: Tensor,
        module_name: &str,
        field_name: &str,
    ) -> PyResult<()> {
        match self {
            Self::Tensors { rows } => {
                if row_index >= rows.len() {
                    return Err(py_value_error(format!(
                        "{module_name} bulk {field_name} row {row_index} is out of bounds"
                    )));
                }
                rows[row_index] = Some(tensor);
                Ok(())
            }
            Self::Flat {
                values,
                seen,
                row_count,
                channels,
                ..
            } => {
                if row_index >= *row_count {
                    return Err(py_value_error(format!(
                        "{module_name} bulk {field_name} row {row_index} is out of bounds"
                    )));
                }
                if tensor.dims() != [1usize, *channels] {
                    return Err(py_value_error(format!(
                        "{module_name} bulk {field_name} expected row shape [1, {}], got {:?}",
                        *channels,
                        tensor.dims()
                    )));
                }
                let data = f32_tensor_data(&tensor).map_err(py_value_error)?;
                let row = data.as_slice().map_err(py_value_error)?;
                if row.len() != *channels {
                    return Err(py_value_error(format!(
                        "{module_name} bulk {field_name} expected {} row values, got {}",
                        *channels,
                        row.len()
                    )));
                }
                let start = row_index * *channels;
                values[start..start + *channels].copy_from_slice(row);
                seen[row_index] = true;
                Ok(())
            }
        }
    }

    fn store_values(
        &mut self,
        row_index: usize,
        row: &[f32],
        module_name: &str,
        field_name: &str,
    ) -> PyResult<()> {
        match self {
            Self::Flat {
                values,
                seen,
                row_count,
                channels,
                ..
            } => {
                if row_index >= *row_count {
                    return Err(py_value_error(format!(
                        "{module_name} bulk {field_name} row {row_index} is out of bounds"
                    )));
                }
                if row.len() != *channels {
                    return Err(py_value_error(format!(
                        "{module_name} bulk {field_name} expected {} row values, got {}",
                        *channels,
                        row.len()
                    )));
                }
                let start = row_index * *channels;
                values[start..start + *channels].copy_from_slice(row);
                seen[row_index] = true;
                Ok(())
            }
            Self::Tensors { .. } => Err(py_value_error(format!(
                "{module_name} bulk {field_name} raw-value storage requires flat output buffer"
            ))),
        }
    }

    fn finish_output(
        self,
        module_name: &str,
        field_name: &str,
        raw_values: bool,
    ) -> PyResult<BulkModuleRows> {
        match self {
            Self::Tensors { rows } => Ok(BulkModuleRows::Tensor(cat_ordered_rows(
                rows,
                module_name,
                field_name,
            )?)),
            Self::Flat {
                values,
                seen,
                row_count,
                channels,
                device,
            } => {
                for (row_index, seen) in seen.iter().copied().enumerate() {
                    if !seen {
                        return Err(py_value_error(format!(
                            "{module_name} bulk {field_name} missing row {row_index}"
                        )));
                    }
                }
                if raw_values {
                    Ok(BulkModuleRows::Values {
                        values,
                        row_count,
                        channels,
                        device,
                    })
                } else {
                    Tensor::from_vec(values, (row_count, channels), &device)
                        .map(BulkModuleRows::Tensor)
                        .map_err(py_value_error)
                }
            }
        }
    }
}

fn initial_module_state(
    rnn: &NativeRnn,
    kind: BulkStreamModuleKind,
    key: Option<i64>,
) -> Option<NativeRnnModuleState> {
    match kind {
        BulkStreamModuleKind::Card => key.and_then(|key| rnn.card_states.get(&key).cloned()),
        BulkStreamModuleKind::Note => key.and_then(|key| rnn.note_states.get(&key).cloned()),
        BulkStreamModuleKind::Deck => key.and_then(|key| rnn.deck_states.get(&key).cloned()),
        BulkStreamModuleKind::Preset => key.and_then(|key| rnn.preset_states.get(&key).cloned()),
        BulkStreamModuleKind::Global => rnn.global_state.clone(),
    }
}

fn native_module_state_value_count(state: &NativeRnnModuleState) -> usize {
    state
        .time_x_shift_b1c_by_layer
        .iter()
        .chain(&state.time_state_b1hkk_by_layer)
        .chain(&state.channel_state_b1c_by_layer)
        .map(Tensor::elem_count)
        .sum()
}

fn bulk_working_module_state_value_count(state: &BulkWorkingModuleState) -> usize {
    let time_values = state
        .time_state_by_layer
        .iter()
        .map(|layer| layer.x_shift_values.len() + layer.state_values.len())
        .sum::<usize>();
    let channel_tensor_values = state
        .channel_state_b1c_by_layer
        .as_ref()
        .map(|layers| layers.iter().map(Tensor::elem_count).sum::<usize>())
        .unwrap_or(0);
    let channel_flat_values = state
        .channel_flat_state_by_layer
        .as_ref()
        .map(|layers| {
            layers
                .iter()
                .map(|layer| layer.state_values.len())
                .sum::<usize>()
        })
        .unwrap_or(0);
    time_values + channel_tensor_values + channel_flat_values
}

fn bulk_working_module_state_tensor_count(state: &BulkWorkingModuleState) -> usize {
    let time_tensors = state.time_state_by_layer.len() * 2;
    let channel_tensors = state
        .channel_state_b1c_by_layer
        .as_ref()
        .map(Vec::len)
        .or_else(|| state.channel_flat_state_by_layer.as_ref().map(Vec::len))
        .unwrap_or(0);
    time_tensors + channel_tensors
}

pub(super) fn flat_working_module_state_from_native(
    state: &NativeRnnModuleState,
    flat_channel_state: bool,
) -> CandleResult<BulkWorkingModuleState> {
    let mut time_state_by_layer = Vec::with_capacity(state.time_x_shift_b1c_by_layer.len());
    for (layer_index, (x_shift, recurrent)) in state
        .time_x_shift_b1c_by_layer
        .iter()
        .zip(&state.time_state_b1hkk_by_layer)
        .enumerate()
    {
        let (x_batch, x_one, channels) = x_shift.dims3()?;
        if x_batch != 1 || x_one != 1 {
            return Err(candle_core::Error::msg(format!(
                "flat TimeMixer state expected x-shift layer {layer_index} shape [1, 1, C], got {:?}",
                x_shift.dims()
            )));
        }
        let (state_batch, state_one, heads, head_size, head_size_2) = recurrent.dims5()?;
        if state_batch != 1
            || state_one != 1
            || head_size != head_size_2
            || channels != heads * head_size
        {
            return Err(candle_core::Error::msg(format!(
                "flat TimeMixer state expected recurrent layer {layer_index} shape [1, 1, H, K, K] compatible with C={channels}, got {:?}",
                recurrent.dims()
            )));
        }
        let x_data = f32_tensor_data(x_shift)?;
        let recurrent_data = f32_tensor_data(recurrent)?;
        time_state_by_layer.push(TimeMixerFlatLayerState {
            x_shift_values: x_data.as_slice()?.to_vec(),
            state_values: recurrent_data.as_slice()?.to_vec(),
            channels,
            heads,
            head_size,
        });
    }
    let mut channel_flat_state_by_layer = None;
    let mut channel_state_b1c_by_layer = None;
    if flat_channel_state {
        let mut flat_layers = Vec::with_capacity(state.channel_state_b1c_by_layer.len());
        for (layer_index, channel_state) in state.channel_state_b1c_by_layer.iter().enumerate() {
            let (batch, one, channels) = channel_state.dims3()?;
            if batch != 1 || one != 1 {
                return Err(candle_core::Error::msg(format!(
                    "flat ChannelMixer state expected layer {layer_index} shape [1, 1, C], got {:?}",
                    channel_state.dims()
                )));
            }
            let state_data = f32_tensor_data(channel_state)?;
            flat_layers.push(ChannelMixerFlatLayerState {
                state_values: state_data.as_slice()?.to_vec(),
                channels,
            });
        }
        channel_flat_state_by_layer = Some(flat_layers);
    } else {
        channel_state_b1c_by_layer = Some(state.channel_state_b1c_by_layer.clone());
    }

    Ok(BulkWorkingModuleState {
        time_state_by_layer,
        channel_state_b1c_by_layer,
        channel_flat_state_by_layer,
    })
}

pub(super) fn native_module_state_from_flat_working(
    state: BulkWorkingModuleState,
    device: &Device,
) -> CandleResult<NativeRnnModuleState> {
    let mut time_x_shift_b1c_by_layer = Vec::with_capacity(state.time_state_by_layer.len());
    let mut time_state_b1hkk_by_layer = Vec::with_capacity(state.time_state_by_layer.len());
    for (layer_index, layer) in state.time_state_by_layer.into_iter().enumerate() {
        if layer.x_shift_values.len() != layer.channels {
            return Err(candle_core::Error::msg(format!(
                "flat TimeMixer state layer {layer_index} expected {} x-shift values, got {}",
                layer.channels,
                layer.x_shift_values.len()
            )));
        }
        let recurrent_len = layer.heads * layer.head_size * layer.head_size;
        if layer.state_values.len() != recurrent_len {
            return Err(candle_core::Error::msg(format!(
                "flat TimeMixer state layer {layer_index} expected {recurrent_len} recurrent values, got {}",
                layer.state_values.len()
            )));
        }
        time_x_shift_b1c_by_layer.push(Tensor::from_vec(
            layer.x_shift_values,
            (1usize, 1usize, layer.channels),
            device,
        )?);
        time_state_b1hkk_by_layer.push(Tensor::from_vec(
            layer.state_values,
            (
                1usize,
                1usize,
                layer.heads,
                layer.head_size,
                layer.head_size,
            ),
            device,
        )?);
    }
    let channel_state_b1c_by_layer = if let Some(channel_state_b1c_by_layer) =
        state.channel_state_b1c_by_layer
    {
        channel_state_b1c_by_layer
    } else {
        let channel_flat_state_by_layer = state.channel_flat_state_by_layer.ok_or_else(|| {
            candle_core::Error::msg("flat working state is missing ChannelMixer state")
        })?;
        let mut channel_state_b1c_by_layer = Vec::with_capacity(channel_flat_state_by_layer.len());
        for (layer_index, layer) in channel_flat_state_by_layer.into_iter().enumerate() {
            if layer.state_values.len() != layer.channels {
                return Err(candle_core::Error::msg(format!(
                    "flat ChannelMixer state layer {layer_index} expected {} values, got {}",
                    layer.channels,
                    layer.state_values.len()
                )));
            }
            channel_state_b1c_by_layer.push(Tensor::from_vec(
                layer.state_values,
                (1usize, 1usize, layer.channels),
                device,
            )?);
        }
        channel_state_b1c_by_layer
    };

    Ok(NativeRnnModuleState {
        time_x_shift_b1c_by_layer,
        time_state_b1hkk_by_layer,
        channel_state_b1c_by_layer,
    })
}

fn module_state_slices(
    state: Option<&NativeRnnModuleState>,
) -> (Option<&[Tensor]>, Option<&[Tensor]>, Option<&[Tensor]>) {
    match state {
        Some(state) => (
            Some(state.time_x_shift_b1c_by_layer.as_slice()),
            Some(state.time_state_b1hkk_by_layer.as_slice()),
            Some(state.channel_state_b1c_by_layer.as_slice()),
        ),
        None => (None, None, None),
    }
}

fn cat_ordered_rows(
    rows: Vec<Option<Tensor>>,
    module_name: &str,
    field_name: &str,
) -> PyResult<Tensor> {
    let tensors = rows
        .iter()
        .enumerate()
        .map(|(row_index, tensor)| {
            tensor.as_ref().ok_or_else(|| {
                py_value_error(format!(
                    "{module_name} bulk {field_name} missing row {row_index}"
                ))
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    Tensor::cat(&tensors, 0).map_err(py_value_error)
}

fn apply_bulk_module_final_states(
    rnn: &mut NativeRnn,
    kind: BulkStreamModuleKind,
    final_states: BulkModuleFinalStates,
) {
    rnn.invalidate_gpu();
    match (kind, final_states) {
        (BulkStreamModuleKind::Card, BulkModuleFinalStates::Keyed(states)) => {
            rnn.card_states.extend(states);
        }
        (BulkStreamModuleKind::Note, BulkModuleFinalStates::Keyed(states)) => {
            rnn.note_states.extend(states);
        }
        (BulkStreamModuleKind::Deck, BulkModuleFinalStates::Keyed(states)) => {
            rnn.deck_states.extend(states);
        }
        (BulkStreamModuleKind::Preset, BulkModuleFinalStates::Keyed(states)) => {
            rnn.preset_states.extend(states);
        }
        (BulkStreamModuleKind::Global, BulkModuleFinalStates::Global(state)) => {
            rnn.global_state = state;
        }
        _ => unreachable!("bulk module final-state kind mismatch"),
    }
}

fn add_bulk_module_time(
    profile: &mut RuntimeProfile,
    lane: BulkLane,
    kind: BulkStreamModuleKind,
    elapsed_ns: u128,
) {
    let forward_profile = match lane {
        BulkLane::Predict => &mut profile.predict_forward,
        BulkLane::Process => &mut profile.process_forward,
    };
    match kind {
        BulkStreamModuleKind::Card => forward_profile.card_module_ns += elapsed_ns,
        BulkStreamModuleKind::Deck => forward_profile.deck_module_ns += elapsed_ns,
        BulkStreamModuleKind::Note => forward_profile.note_module_ns += elapsed_ns,
        BulkStreamModuleKind::Preset => forward_profile.preset_module_ns += elapsed_ns,
        BulkStreamModuleKind::Global => forward_profile.global_module_ns += elapsed_ns,
    }
}

fn record_bulk_fast_path_time(profile: &mut Option<&mut RuntimeProfile>, elapsed_ns: u128) {
    if elapsed_ns == 0 {
        return;
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.bulk_fast_path_calls += 1;
        profile.bulk_fast_path_ns += elapsed_ns;
    }
}

pub(super) fn bulk_prediction_probabilities(
    rnn: &NativeRnn,
    query_x: &Tensor,
    profile: &mut Option<&mut RuntimeProfile>,
) -> CandleResult<Vec<f64>> {
    if bulk_prediction_head_batch_enabled() {
        if let Some(probabilities) =
            bulk_prediction_probabilities_batched_row_exact(rnn, query_x, profile)?
        {
            return Ok(probabilities);
        }
    }

    let (rows, _) = query_x.dims2()?;
    let mut probabilities = Vec::with_capacity(rows);
    for row_index in 0..rows {
        let row = query_x.narrow(0, row_index, 1)?;
        let start = ProfileTimer::start(profile.is_some());
        let x = layer_norm(&row, &rnn.weights.prehead_norm)?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.predict_forward.prehead_norm_ns += start.elapsed_ns();
        }

        let start = ProfileTimer::start(profile.is_some());
        let x_p = linear_profiled(
            &x,
            &rnn.weights.head_p,
            &mut profile
                .as_deref_mut()
                .map(|profile| &mut profile.predict_forward),
        )?
        .relu()?;
        let logits = linear_profiled(
            &x_p,
            &rnn.weights.p_linear,
            &mut profile
                .as_deref_mut()
                .map(|profile| &mut profile.predict_forward),
        )?;
        if let Some(profile) = profile.as_deref_mut() {
            let elapsed = start.elapsed_ns();
            profile.predict_forward.p_head_ns += elapsed;
            profile.predict_forward.total_ns += elapsed;
            profile.predict_forward.calls += 1;
        }

        let start = ProfileTimer::start(profile.is_some());
        probabilities.push(probability_from_logits(&logits)?);
        if let Some(profile) = profile.as_deref_mut() {
            profile.predict_output_ns += start.elapsed_ns();
            profile.materialization.prediction_probability_values += 1;
        }
    }
    Ok(probabilities)
}

fn bulk_prediction_probabilities_batched_row_exact(
    rnn: &NativeRnn,
    query_x: &Tensor,
    profile: &mut Option<&mut RuntimeProfile>,
) -> CandleResult<Option<Vec<f64>>> {
    let (rows, _) = query_x.dims2()?;
    if rows == 0 {
        return Ok(Some(Vec::new()));
    }

    let start = ProfileTimer::start(profile.is_some());
    let x = layer_norm(query_x, &rnn.weights.prehead_norm)?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.predict_forward.prehead_norm_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    let mut forward_profile = profile
        .as_deref_mut()
        .map(|profile| &mut profile.predict_forward);
    let Some(x_p) =
        linear_profiled_rowwise_native_exact(&x, &rnn.weights.head_p, &mut forward_profile)?
    else {
        return Ok(None);
    };
    let x_p = x_p.relu()?;
    let Some(logits) =
        linear_profiled_rowwise_native_exact(&x_p, &rnn.weights.p_linear, &mut forward_profile)?
    else {
        return Ok(None);
    };
    if let Some(profile) = profile.as_deref_mut() {
        let elapsed = start.elapsed_ns();
        profile.predict_forward.p_head_ns += elapsed;
        profile.predict_forward.total_ns += elapsed;
        profile.predict_forward.calls += rows;
    }

    let start = ProfileTimer::start(profile.is_some());
    let probabilities = probabilities_from_logits(&logits)?;
    if probabilities.len() != rows {
        return Ok(None);
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.predict_output_ns += start.elapsed_ns();
        profile.materialization.prediction_probability_values += probabilities.len();
    }
    Ok(Some(probabilities))
}

pub(super) fn bulk_curve_outputs(
    rnn: &NativeRnn,
    process_x: &Tensor,
    profile: &mut Option<&mut RuntimeProfile>,
) -> CandleResult<(Vec<Tensor2List>, Vec<Tensor2List>)> {
    if bulk_curve_heads_batch_enabled() {
        if let Some(outputs) = bulk_curve_outputs_batched_row_exact(rnn, process_x, profile)? {
            return Ok(outputs);
        }
    }

    let (rows, _) = process_x.dims2()?;
    let mut ahead_outputs = Vec::with_capacity(rows);
    let mut w_outputs = Vec::with_capacity(rows);
    for row_index in 0..rows {
        let row = process_x.narrow(0, row_index, 1)?;
        let start = ProfileTimer::start(profile.is_some());
        let x = layer_norm(&row, &rnn.weights.prehead_norm)?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.process_forward.prehead_norm_ns += start.elapsed_ns();
        }

        let start = ProfileTimer::start(profile.is_some());
        let x_w = w_head_forward_profiled(
            &rnn.weights.head_w,
            &x,
            &mut profile
                .as_deref_mut()
                .map(|profile| &mut profile.process_forward),
        )?;
        let out_w_logits = linear_profiled(
            &x_w,
            &rnn.weights.w_linear,
            &mut profile
                .as_deref_mut()
                .map(|profile| &mut profile.process_forward),
        )?;
        let out_w = nn_ops::softmax(&out_w_logits, D::Minus1)?;
        let ahead_hidden = linear_profiled(
            &x,
            &rnn.weights.head_ahead_logits,
            &mut profile
                .as_deref_mut()
                .map(|profile| &mut profile.process_forward),
        )?;
        let ahead = linear_profiled(
            &ahead_hidden.relu()?,
            &rnn.weights.ahead_linear,
            &mut profile
                .as_deref_mut()
                .map(|profile| &mut profile.process_forward),
        )?;
        if let Some(profile) = profile.as_deref_mut() {
            let elapsed = start.elapsed_ns();
            profile.process_forward.curve_heads_ns += elapsed;
            profile.process_forward.total_ns += elapsed;
            profile.process_forward.calls += 1;
        }

        let start = ProfileTimer::start(profile.is_some());
        let ahead = ahead.to_vec2::<f32>()?;
        let out_w = out_w.to_vec2::<f32>()?;
        if let Some(profile) = profile.as_deref_mut() {
            let values = tensor2_list_value_count(&ahead) + tensor2_list_value_count(&out_w);
            profile.curve_output_ns += start.elapsed_ns();
            profile.materialization.curve_output_vecs += 2;
            profile.materialization.curve_output_values += values;
        }
        ahead_outputs.push(ahead);
        w_outputs.push(out_w);
    }
    Ok((ahead_outputs, w_outputs))
}

fn bulk_curve_outputs_batched_row_exact(
    rnn: &NativeRnn,
    process_x: &Tensor,
    profile: &mut Option<&mut RuntimeProfile>,
) -> CandleResult<Option<(Vec<Tensor2List>, Vec<Tensor2List>)>> {
    let (rows, _) = process_x.dims2()?;
    if rows == 0 {
        return Ok(Some((Vec::new(), Vec::new())));
    }

    let start = ProfileTimer::start(profile.is_some());
    let x = layer_norm(process_x, &rnn.weights.prehead_norm)?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.process_forward.prehead_norm_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    let mut forward_profile = profile
        .as_deref_mut()
        .map(|profile| &mut profile.process_forward);
    let Some(x_w_hidden) = linear_profiled_rowwise_native_exact(
        &x,
        &rnn.weights.head_w.input_linear,
        &mut forward_profile,
    )?
    else {
        return Ok(None);
    };
    let x_w_hidden = x_w_hidden.relu()?;
    let x_w = layer_norm(&x_w_hidden, &rnn.weights.head_w.norm)?;
    let Some(x_w) = linear_profiled_rowwise_native_exact(
        &x_w,
        &rnn.weights.head_w.output_linear,
        &mut forward_profile,
    )?
    else {
        return Ok(None);
    };
    let Some(out_w_logits) =
        linear_profiled_rowwise_native_exact(&x_w, &rnn.weights.w_linear, &mut forward_profile)?
    else {
        return Ok(None);
    };
    let out_w = nn_ops::softmax(&out_w_logits, D::Minus1)?;
    let Some(ahead_hidden) = linear_profiled_rowwise_native_exact(
        &x,
        &rnn.weights.head_ahead_logits,
        &mut forward_profile,
    )?
    else {
        return Ok(None);
    };
    let ahead_hidden = ahead_hidden.relu()?;
    let Some(ahead) = linear_profiled_rowwise_native_exact(
        &ahead_hidden,
        &rnn.weights.ahead_linear,
        &mut forward_profile,
    )?
    else {
        return Ok(None);
    };
    if let Some(profile) = profile.as_deref_mut() {
        let elapsed = start.elapsed_ns();
        profile.process_forward.curve_heads_ns += elapsed;
        profile.process_forward.total_ns += elapsed;
        profile.process_forward.calls += rows;
    }

    let start = ProfileTimer::start(profile.is_some());
    let ahead_rows = ahead.to_vec2::<f32>()?;
    let out_w_rows = out_w.to_vec2::<f32>()?;
    if ahead_rows.len() != rows || out_w_rows.len() != rows {
        return Ok(None);
    }
    let values = ahead_rows.iter().map(Vec::len).sum::<usize>()
        + out_w_rows.iter().map(Vec::len).sum::<usize>();
    if let Some(profile) = profile.as_deref_mut() {
        profile.curve_output_ns += start.elapsed_ns();
        profile.materialization.curve_output_vecs += rows * 2;
        profile.materialization.curve_output_values += values;
    }

    let mut ahead_outputs = Vec::with_capacity(rows);
    let mut w_outputs = Vec::with_capacity(rows);
    for (ahead_row, out_w_row) in ahead_rows.into_iter().zip(out_w_rows) {
        ahead_outputs.push(vec![ahead_row]);
        w_outputs.push(vec![out_w_row]);
    }
    Ok(Some((ahead_outputs, w_outputs)))
}

fn tensor2_list_value_count(values: &Tensor2List) -> usize {
    values.iter().map(Vec::len).sum()
}

pub(super) fn process_reviews_bulk_feature_prepass_debug(
    deterministic: &FeatureState,
    payload: &[u8],
) -> PyResult<(BulkFeaturePrepass, FeatureState)> {
    let mut profile = None;
    let chunk = BulkReplayChunk::from_payload(payload, &mut profile)?;
    let mut deterministic = deterministic.clone();
    let prepass = chunk.feature_prepass(&mut deterministic, &mut profile)?;
    Ok((prepass, deterministic))
}

pub(super) fn process_reviews_bulk_stream_plan_debug(
    rnn: &NativeRnn,
    deterministic: &FeatureState,
    payload: &[u8],
) -> PyResult<(BulkFeaturePrepass, BulkStreamPlan, FeatureState)> {
    let mut profile = None;
    let chunk = BulkReplayChunk::from_payload(payload, &mut profile)?;
    let mut deterministic = deterministic.clone();
    let prepass = chunk.feature_prepass(&mut deterministic, &mut profile)?;
    let stream_plan = BulkStreamPlan::from_ids(rnn, &prepass.ids);
    Ok((prepass, stream_plan, deterministic))
}
