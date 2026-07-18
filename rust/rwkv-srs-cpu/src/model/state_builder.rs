//! State-only history builder.
//!
//! This is deliberately separate from the prediction-preserving
//! `process_many` pipeline. It advances the same deterministic and recurrent
//! state, but it never constructs query rows or executes prediction/curve
//! heads.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TryRecvError};
use std::sync::Arc;
use std::thread;

use candle_core::{Device, Tensor};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use rayon::ThreadPool;

use super::bulk::{
    flat_working_module_state_from_native, native_module_state_from_flat_working,
    BulkWorkingModuleState,
};
use super::process_payload::parse_process_review_payload;
use super::runtime::{rayon_pool, validate_num_threads, with_rayon_threads};
use super::state::{
    native_module_state_from_parts, review_ids_from_prepared, NativeRnnModuleState, ReviewIds,
};
use super::undo::BatchDeterministicUndoFrame;
use super::{
    features2card_forward_profiled, py_value_error,
    rwkv_rnn_forward_flat_working_state_values_profiled, rwkv_rnn_forward_profiled, NativeRnn,
};
use crate::cpu_config::*;
use crate::model_weights::{Features2CardWeights, Rwkv7RnnWeights};
use crate::ops::f32_tensor_data;
use crate::profile::ProfileTimer;
use crate::state::{FeatureState, ReviewInput};

const STATE_BUILD_STAGE_COUNT: usize = 5;

#[derive(Debug, Default)]
pub(super) struct StateBuildProfile {
    pub(super) total_ns: u128,
    pub(super) parse_review_ns: u128,
    pub(super) transaction_capture_ns: u128,
    pub(super) prepare_process_ns: u128,
    pub(super) process_feature_ns: u128,
    pub(super) deterministic_update_ns: u128,
    pub(super) features2card_ns: u128,
    pub(super) pipeline_ns: u128,
    pub(super) process_forward_worker_ns: u128,
    pub(super) state_conversion_worker_ns: u128,
    pub(super) state_commit_ns: u128,
    pub(super) process_rows: usize,
    pub(super) completed_rows: usize,
    pub(super) stage_compute_ns: [u128; STATE_BUILD_STAGE_COUNT],
    pub(super) stage_state_conversion_ns: [u128; STATE_BUILD_STAGE_COUNT],
}

impl StateBuildProfile {
    pub(super) fn to_pydict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("total_ns", self.total_ns)?;
        dict.set_item("parse_review_ns", self.parse_review_ns)?;
        dict.set_item("transaction_capture_ns", self.transaction_capture_ns)?;
        dict.set_item("prepare_process_ns", self.prepare_process_ns)?;
        dict.set_item("process_feature_ns", self.process_feature_ns)?;
        dict.set_item("deterministic_update_ns", self.deterministic_update_ns)?;
        dict.set_item("features2card_ns", self.features2card_ns)?;
        dict.set_item("pipeline_ns", self.pipeline_ns)?;
        dict.set_item("process_forward_worker_ns", self.process_forward_worker_ns)?;
        dict.set_item(
            "state_conversion_worker_ns",
            self.state_conversion_worker_ns,
        )?;
        dict.set_item("state_commit_ns", self.state_commit_ns)?;
        dict.set_item("query_rows", 0usize)?;
        dict.set_item("process_rows", self.process_rows)?;
        dict.set_item("prediction_head_rows", 0usize)?;
        dict.set_item("curve_head_rows", 0usize)?;
        dict.set_item("completed_rows", self.completed_rows)?;
        dict.set_item("stage_compute_ns", self.stage_compute_ns.to_vec())?;
        dict.set_item(
            "stage_state_conversion_ns",
            self.stage_state_conversion_ns.to_vec(),
        )?;
        Ok(dict.unbind())
    }
}

#[derive(Clone, Copy)]
enum StateBuildStageKind {
    Card,
    Deck,
    Note,
    Preset,
}

impl StateBuildStageKind {
    fn name(self) -> &'static str {
        match self {
            Self::Card => "card",
            Self::Deck => "deck",
            Self::Note => "note",
            Self::Preset => "preset",
        }
    }

    fn profile_index(self) -> usize {
        match self {
            Self::Card => 0,
            Self::Deck => 1,
            Self::Note => 2,
            Self::Preset => 3,
        }
    }

    fn key(self, ids: ReviewIds) -> i64 {
        match self {
            Self::Card => ids.0,
            Self::Deck => ids.2,
            Self::Note => ids.1,
            Self::Preset => ids.3,
        }
    }
}

struct StateBuildRow {
    ids: ReviewIds,
    process_values: Vec<f32>,
}

type StateBuildMessage = Result<StateBuildRow, String>;

struct StateBuildStateOutput {
    card_states: BTreeMap<i64, NativeRnnModuleState>,
    deck_states: BTreeMap<i64, NativeRnnModuleState>,
    note_states: BTreeMap<i64, NativeRnnModuleState>,
    preset_states: BTreeMap<i64, NativeRnnModuleState>,
    global_state: Option<NativeRnnModuleState>,
}

struct KeyedStageOutput {
    states: BTreeMap<i64, NativeRnnModuleState>,
    compute_ns: u128,
    state_conversion_ns: u128,
}

struct GlobalStageOutput {
    state: Option<NativeRnnModuleState>,
    completed_rows: usize,
    compute_ns: u128,
    state_conversion_ns: u128,
}

struct PipelineExecutionOutput {
    completed_rows: usize,
    states: StateBuildStateOutput,
    stage_compute_ns: [u128; STATE_BUILD_STAGE_COUNT],
    stage_state_conversion_ns: [u128; STATE_BUILD_STAGE_COUNT],
}

struct DeterministicStateBuildTransaction<'a> {
    deterministic: &'a mut FeatureState,
    undo: Option<BatchDeterministicUndoFrame>,
    committed: bool,
}

impl<'a> DeterministicStateBuildTransaction<'a> {
    fn new(deterministic: &'a mut FeatureState, ids: &[ReviewIds]) -> Self {
        let undo = BatchDeterministicUndoFrame::capture(deterministic, ids);
        Self {
            deterministic,
            undo: Some(undo),
            committed: false,
        }
    }

    fn commit(mut self) {
        self.committed = true;
        self.undo = None;
    }
}

impl Drop for DeterministicStateBuildTransaction<'_> {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        if let Some(frame) = self.undo.take() {
            frame.restore(self.deterministic);
        }
    }
}

pub(super) fn build_state_only_pipeline_with_state(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    payload: &[u8],
    num_threads: Option<usize>,
    mut profile: Option<&mut StateBuildProfile>,
) -> PyResult<usize> {
    let total_start = ProfileTimer::start(profile.is_some());
    validate_num_threads(num_threads)?;

    let parse_start = ProfileTimer::start(profile.is_some());
    let inputs = parse_process_review_payload(payload).map_err(py_value_error)?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.parse_review_ns += parse_start.elapsed_ns();
        profile.process_rows += inputs.len();
    }
    build_state_only_inputs_pipeline_with_state_impl(
        rnn,
        deterministic,
        &inputs,
        num_threads,
        profile,
        total_start,
    )
}

pub(super) fn build_state_only_inputs_pipeline_with_state(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    inputs: &[ReviewInput],
    num_threads: Option<usize>,
    mut profile: Option<&mut StateBuildProfile>,
) -> PyResult<usize> {
    let total_start = ProfileTimer::start(profile.is_some());
    validate_num_threads(num_threads)?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.process_rows += inputs.len();
    }
    build_state_only_inputs_pipeline_with_state_impl(
        rnn,
        deterministic,
        inputs,
        num_threads,
        profile,
        total_start,
    )
}

fn build_state_only_inputs_pipeline_with_state_impl(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    inputs: &[ReviewInput],
    num_threads: Option<usize>,
    mut profile: Option<&mut StateBuildProfile>,
    total_start: ProfileTimer,
) -> PyResult<usize> {
    if inputs.is_empty() {
        if let Some(profile) = profile.as_deref_mut() {
            profile.total_ns = total_start.elapsed_ns();
        }
        return Ok(0);
    }
    rnn.materialize_flat_cpu_state().map_err(py_value_error)?;
    if rnn.weights.rwkv_modules.len() != STATE_BUILD_STAGE_COUNT {
        return Err(py_value_error(format!(
            "fast state builder expected {STATE_BUILD_STAGE_COUNT} RWKV modules, got {}",
            rnn.weights.rwkv_modules.len()
        )));
    }

    let ids = inputs
        .iter()
        .map(FeatureState::normalized_review_ids)
        .collect::<Vec<_>>();
    let transaction_start = ProfileTimer::start(profile.is_some());
    let mut transaction = DeterministicStateBuildTransaction::new(deterministic, &ids);
    if let Some(profile) = profile.as_deref_mut() {
        profile.transaction_capture_ns += transaction_start.elapsed_ns();
    }

    let worker_threads = state_builder_threads(num_threads);
    let rows = build_state_rows(
        &rnn.weights.features2card,
        &mut transaction,
        inputs,
        worker_threads,
        &mut profile,
    )?;

    let pipeline_start = ProfileTimer::start(profile.is_some());
    let execution = execute_state_pipeline(rnn, rows, num_threads, profile.is_some())?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.pipeline_ns += pipeline_start.elapsed_ns();
        profile.completed_rows = execution.completed_rows;
        profile.stage_compute_ns = execution.stage_compute_ns;
        profile.stage_state_conversion_ns = execution.stage_state_conversion_ns;
        profile.process_forward_worker_ns = execution.stage_compute_ns.iter().sum();
        profile.state_conversion_worker_ns = execution.stage_state_conversion_ns.iter().sum();
    }
    if execution.completed_rows != inputs.len() {
        return Err(py_value_error(format!(
            "fast state builder completed {} rows for {} inputs",
            execution.completed_rows,
            inputs.len()
        )));
    }

    let commit_start = ProfileTimer::start(profile.is_some());
    rnn.invalidate_gpu();
    rnn.card_states.extend(execution.states.card_states);
    rnn.deck_states.extend(execution.states.deck_states);
    rnn.note_states.extend(execution.states.note_states);
    rnn.preset_states.extend(execution.states.preset_states);
    rnn.global_state = execution.states.global_state;
    transaction.commit();
    if let Some(profile) = profile {
        profile.state_commit_ns += commit_start.elapsed_ns();
        profile.total_ns = total_start.elapsed_ns();
    }
    Ok(inputs.len())
}

fn build_state_rows(
    weights: &Features2CardWeights,
    transaction: &mut DeterministicStateBuildTransaction<'_>,
    inputs: &[ReviewInput],
    worker_threads: usize,
    profile: &mut Option<&mut StateBuildProfile>,
) -> PyResult<Vec<StateBuildRow>> {
    let mut steps = Vec::with_capacity(inputs.len());
    for input in inputs {
        let prepare_start = ProfileTimer::start(profile.is_some());
        let process_row = transaction
            .deterministic
            .prepare_process_row(input)
            .map_err(py_value_error)?;
        let ids = review_ids_from_prepared(&process_row).map_err(py_value_error)?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.prepare_process_ns += prepare_start.elapsed_ns();
        }

        let feature_start = ProfileTimer::start(profile.is_some());
        let process_features = transaction
            .deterministic
            .process_feature_vector(&process_row)
            .map_err(py_value_error)?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.process_feature_ns += feature_start.elapsed_ns();
        }

        let update_start = ProfileTimer::start(profile.is_some());
        transaction
            .deterministic
            .record_recurrent_state_update(&process_row)
            .map_err(py_value_error)?;
        transaction
            .deterministic
            .record_processed_row(&process_row)
            .map_err(py_value_error)?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.deterministic_update_ns += update_start.elapsed_ns();
        }
        steps.push((ids, process_features));
    }

    let features2card_start = ProfileTimer::start(profile.is_some());
    let rows = with_rayon_threads(Some(worker_threads), move || {
        steps
            .into_par_iter()
            .map(|(ids, process_features)| {
                Ok(StateBuildRow {
                    ids,
                    process_values: features_to_card_values(
                        weights,
                        process_features,
                        "state_builder_process_features",
                    )?,
                })
            })
            .collect()
    })?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.features2card_ns += features2card_start.elapsed_ns();
    }
    Ok(rows)
}

fn features_to_card_values(
    weights: &Features2CardWeights,
    features: Vec<f32>,
    tensor_name: &str,
) -> PyResult<Vec<f32>> {
    let feature_count = features.len();
    let feature_tensor = Tensor::from_vec(features, (1usize, feature_count), &Device::Cpu)
        .map_err(py_value_error)?;
    let mut profile = None;
    let output = features2card_forward_profiled(weights, &feature_tensor, &mut profile)
        .map_err(|error| py_value_error(format!("{tensor_name} forward failed: {error}")))?;
    let data = f32_tensor_data(&output).map_err(py_value_error)?;
    data.as_slice()
        .map(|values| values.to_vec())
        .map_err(py_value_error)
}

fn execute_state_pipeline(
    rnn: &NativeRnn,
    rows: Vec<StateBuildRow>,
    num_threads: Option<usize>,
    profile_enabled: bool,
) -> PyResult<PipelineExecutionOutput> {
    let row_count = rows.len();
    let worker_threads = state_builder_threads(num_threads);
    let channel_capacity = pipeline_capacity();
    let pool = rayon_pool(worker_threads)?;

    let card_states = clone_states_for_keys(&rnn.card_states, rows.iter().map(|row| row.ids.0));
    let note_states = clone_states_for_keys(&rnn.note_states, rows.iter().map(|row| row.ids.1));
    let deck_states = clone_states_for_keys(&rnn.deck_states, rows.iter().map(|row| row.ids.2));
    let preset_states = clone_states_for_keys(&rnn.preset_states, rows.iter().map(|row| row.ids.3));
    let global_state = rnn.global_state.clone();
    let weights = &rnn.weights.rwkv_modules;

    let outcome: std::result::Result<PipelineExecutionOutput, String> = thread::scope(|scope| {
        let (card_tx, card_rx) = sync_channel::<StateBuildMessage>(channel_capacity);
        let (deck_tx, deck_rx) = sync_channel::<StateBuildMessage>(channel_capacity);
        let (note_tx, note_rx) = sync_channel::<StateBuildMessage>(channel_capacity);
        let (preset_tx, preset_rx) = sync_channel::<StateBuildMessage>(channel_capacity);
        let (global_tx, global_rx) = sync_channel::<StateBuildMessage>(channel_capacity);

        let producer = scope.spawn(move || produce_rows(rows, card_tx));
        let card_pool = Arc::clone(&pool);
        let card = scope.spawn(move || {
            run_keyed_stage(
                StateBuildStageKind::Card,
                &weights[0],
                card_states,
                card_rx,
                deck_tx,
                row_count,
                card_pool,
                profile_enabled,
            )
        });
        let deck_pool = Arc::clone(&pool);
        let deck = scope.spawn(move || {
            run_keyed_stage(
                StateBuildStageKind::Deck,
                &weights[1],
                deck_states,
                deck_rx,
                note_tx,
                row_count,
                deck_pool,
                profile_enabled,
            )
        });
        let note_pool = Arc::clone(&pool);
        let note = scope.spawn(move || {
            run_keyed_stage(
                StateBuildStageKind::Note,
                &weights[2],
                note_states,
                note_rx,
                preset_tx,
                row_count,
                note_pool,
                profile_enabled,
            )
        });
        let preset_pool = Arc::clone(&pool);
        let preset = scope.spawn(move || {
            run_keyed_stage(
                StateBuildStageKind::Preset,
                &weights[3],
                preset_states,
                preset_rx,
                global_tx,
                row_count,
                preset_pool,
                profile_enabled,
            )
        });
        let global_pool = Arc::clone(&pool);
        let global = scope.spawn(move || {
            run_global_stage(
                &weights[4],
                global_state,
                global_rx,
                global_pool,
                profile_enabled,
            )
        });

        let global = join_stage("global", global.join())?;
        let preset = join_stage("preset", preset.join())?;
        let note = join_stage("note", note.join())?;
        let deck = join_stage("deck", deck.join())?;
        let card = join_stage("card", card.join())?;
        join_stage("producer", producer.join())?;

        let mut stage_compute_ns = [0; STATE_BUILD_STAGE_COUNT];
        let mut stage_state_conversion_ns = [0; STATE_BUILD_STAGE_COUNT];
        for (kind, output) in [
            (StateBuildStageKind::Card, &card),
            (StateBuildStageKind::Deck, &deck),
            (StateBuildStageKind::Note, &note),
            (StateBuildStageKind::Preset, &preset),
        ] {
            stage_compute_ns[kind.profile_index()] = output.compute_ns;
            stage_state_conversion_ns[kind.profile_index()] = output.state_conversion_ns;
        }
        stage_compute_ns[4] = global.compute_ns;
        stage_state_conversion_ns[4] = global.state_conversion_ns;

        Ok(PipelineExecutionOutput {
            completed_rows: global.completed_rows,
            states: StateBuildStateOutput {
                card_states: card.states,
                deck_states: deck.states,
                note_states: note.states,
                preset_states: preset.states,
                global_state: global.state,
            },
            stage_compute_ns,
            stage_state_conversion_ns,
        })
    });

    outcome.map_err(py_value_error)
}

fn clone_states_for_keys<V: Clone>(
    states: &BTreeMap<i64, V>,
    keys: impl Iterator<Item = i64>,
) -> BTreeMap<i64, V> {
    keys.collect::<BTreeSet<_>>()
        .into_iter()
        .filter_map(|key| states.get(&key).cloned().map(|state| (key, state)))
        .collect()
}

fn produce_rows(
    rows: Vec<StateBuildRow>,
    output: SyncSender<StateBuildMessage>,
) -> Result<(), String> {
    for row in rows {
        output
            .send(Ok(row))
            .map_err(|_| "state builder card stage closed early".to_string())?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_keyed_stage(
    kind: StateBuildStageKind,
    weights: &Rwkv7RnnWeights,
    mut native_states: BTreeMap<i64, NativeRnnModuleState>,
    input: Receiver<StateBuildMessage>,
    output: SyncSender<StateBuildMessage>,
    row_count: usize,
    pool: Arc<ThreadPool>,
    profile_enabled: bool,
) -> Result<KeyedStageOutput, String> {
    let mut compute_ns = 0u128;
    let mut state_conversion_ns = 0u128;
    let mut working_states: HashMap<i64, Option<BulkWorkingModuleState>> =
        HashMap::with_capacity(row_count.min(16_384));
    loop {
        let batch = receive_batch(&input);
        if batch.rows.is_empty() {
            if let Some(error) = batch.terminal_error {
                let _ = output.send(Err(error.clone()));
                return Err(error);
            }
            break;
        }

        let compute_start = ProfileTimer::start(profile_enabled);
        let processed = pool.install(|| {
            let mut processed = Vec::with_capacity(batch.rows.len());
            for mut row in batch.rows {
                let key = kind.key(row.ids);
                let state = match working_states.entry(key) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        let conversion_start = ProfileTimer::start(profile_enabled);
                        let working = native_states
                            .remove(&key)
                            .map(|state| flat_working_module_state_from_native(&state, true))
                            .transpose()
                            .map_err(|error| {
                                format!(
                                    "state builder {} state conversion failed: {error}",
                                    kind.name()
                                )
                            })?;
                        state_conversion_ns += conversion_start.elapsed_ns();
                        entry.insert(working)
                    }
                };
                let next_state = process_stage_row(weights, state.as_ref(), &mut row)?;
                *state = Some(next_state);
                processed.push(row);
            }
            Ok::<_, String>(processed)
        });
        compute_ns += compute_start.elapsed_ns();
        let processed = match processed {
            Ok(processed) => processed,
            Err(error) => {
                let _ = output.send(Err(error.clone()));
                return Err(error);
            }
        };
        for row in processed {
            if output.send(Ok(row)).is_err() {
                return Err(format!(
                    "state builder {} stage output closed early",
                    kind.name()
                ));
            }
        }
        if let Some(error) = batch.terminal_error {
            let _ = output.send(Err(error.clone()));
            return Err(error);
        }
        if batch.disconnected {
            break;
        }
    }

    let conversion_start = ProfileTimer::start(profile_enabled);
    for (key, state) in working_states {
        let state = state.ok_or_else(|| {
            format!(
                "state builder {} state {key} was not initialized",
                kind.name()
            )
        })?;
        let native =
            native_module_state_from_flat_working(state, &Device::Cpu).map_err(|error| {
                format!(
                    "state builder {} final-state conversion failed: {error}",
                    kind.name()
                )
            })?;
        native_states.insert(key, native);
    }
    state_conversion_ns += conversion_start.elapsed_ns();
    Ok(KeyedStageOutput {
        states: native_states,
        compute_ns,
        state_conversion_ns,
    })
}

fn run_global_stage(
    weights: &Rwkv7RnnWeights,
    native_state: Option<NativeRnnModuleState>,
    input: Receiver<StateBuildMessage>,
    pool: Arc<ThreadPool>,
    profile_enabled: bool,
) -> Result<GlobalStageOutput, String> {
    let conversion_start = ProfileTimer::start(profile_enabled);
    let mut working_state = native_state
        .map(|state| flat_working_module_state_from_native(&state, true))
        .transpose()
        .map_err(|error| format!("state builder global state conversion failed: {error}"))?;
    let mut state_conversion_ns = conversion_start.elapsed_ns();
    let mut compute_ns = 0u128;
    let mut completed_rows = 0usize;
    loop {
        let batch = receive_batch(&input);
        if batch.rows.is_empty() {
            if let Some(error) = batch.terminal_error {
                return Err(error);
            }
            break;
        }

        let compute_start = ProfileTimer::start(profile_enabled);
        pool.install(|| {
            for mut row in batch.rows {
                working_state = Some(process_stage_row(
                    weights,
                    working_state.as_ref(),
                    &mut row,
                )?);
                completed_rows += 1;
            }
            Ok::<_, String>(())
        })?;
        compute_ns += compute_start.elapsed_ns();
        if let Some(error) = batch.terminal_error {
            return Err(error);
        }
        if batch.disconnected {
            break;
        }
    }

    let conversion_start = ProfileTimer::start(profile_enabled);
    let state = working_state
        .map(|state| native_module_state_from_flat_working(state, &Device::Cpu))
        .transpose()
        .map_err(|error| format!("state builder global final-state conversion failed: {error}"))?;
    state_conversion_ns += conversion_start.elapsed_ns();
    Ok(GlobalStageOutput {
        state,
        completed_rows,
        compute_ns,
        state_conversion_ns,
    })
}

fn process_stage_row(
    weights: &Rwkv7RnnWeights,
    state: Option<&BulkWorkingModuleState>,
    row: &mut StateBuildRow,
) -> Result<BulkWorkingModuleState, String> {
    let channels = row.process_values.len();
    let time_state = state.map(|state| state.time_state_by_layer.as_slice());
    let channel_state = state.and_then(|state| state.channel_flat_state_by_layer.as_deref());
    match rwkv_rnn_forward_flat_working_state_values_profiled(
        weights,
        &row.process_values,
        1,
        channels,
        &Device::Cpu,
        time_state,
        channel_state,
        None,
    )
    .map_err(|error| format!("state builder process forward failed: {error}"))?
    {
        Some((process_values, time_state_by_layer, channel_state_by_layer)) => {
            row.process_values = process_values;
            Ok(BulkWorkingModuleState {
                time_state_by_layer,
                channel_state_b1c_by_layer: None,
                channel_flat_state_by_layer: Some(channel_state_by_layer),
            })
        }
        None => process_stage_row_portable(weights, state, row),
    }
}

fn process_stage_row_portable(
    weights: &Rwkv7RnnWeights,
    state: Option<&BulkWorkingModuleState>,
    row: &mut StateBuildRow,
) -> Result<BulkWorkingModuleState, String> {
    let channels = row.process_values.len();
    let native_state = state
        .cloned()
        .map(|state| native_module_state_from_flat_working(state, &Device::Cpu))
        .transpose()
        .map_err(|error| format!("portable state builder state conversion failed: {error}"))?;
    let (time_x_shift, time_state, channel_state) = match native_state.as_ref() {
        Some(state) => (
            Some(state.time_x_shift_b1c_by_layer.as_slice()),
            Some(state.time_state_b1hkk_by_layer.as_slice()),
            Some(state.channel_state_b1c_by_layer.as_slice()),
        ),
        None => (None, None, None),
    };
    let process_x = Tensor::from_vec(row.process_values.clone(), (1usize, channels), &Device::Cpu)
        .map_err(|error| format!("portable state builder process input failed: {error}"))?;
    let (process_output, next_time_x, next_time_state, next_channel_state) =
        rwkv_rnn_forward_profiled(
            weights,
            &process_x,
            time_x_shift,
            time_state,
            channel_state,
            None,
        )
        .map_err(|error| format!("portable state builder process forward failed: {error}"))?;
    row.process_values = f32_tensor_data(&process_output)
        .and_then(|data| data.as_slice().map(|values| values.to_vec()))
        .map_err(|error| format!("portable state builder process output failed: {error}"))?;
    let native_next = native_module_state_from_parts(
        next_time_x,
        next_time_state,
        next_channel_state,
        "portable state builder next state",
    )
    .map_err(|error| format!("portable state builder next-state conversion failed: {error}"))?;
    flat_working_module_state_from_native(&native_next, true)
        .map_err(|error| format!("portable state builder flat-state conversion failed: {error}"))
}

struct StateBuildBatch {
    rows: Vec<StateBuildRow>,
    terminal_error: Option<String>,
    disconnected: bool,
}

fn receive_batch(input: &Receiver<StateBuildMessage>) -> StateBuildBatch {
    let mut batch = StateBuildBatch {
        rows: Vec::with_capacity(pipeline_compute_batch()),
        terminal_error: None,
        disconnected: false,
    };
    match input.recv() {
        Ok(Ok(row)) => batch.rows.push(row),
        Ok(Err(error)) => {
            batch.terminal_error = Some(error);
            return batch;
        }
        Err(_) => {
            batch.disconnected = true;
            return batch;
        }
    }
    while batch.rows.len() < pipeline_compute_batch() {
        match input.try_recv() {
            Ok(Ok(row)) => batch.rows.push(row),
            Ok(Err(error)) => {
                batch.terminal_error = Some(error);
                break;
            }
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                batch.disconnected = true;
                break;
            }
        }
    }
    batch
}

fn join_stage<T>(name: &str, result: thread::Result<Result<T, String>>) -> Result<T, String> {
    result.map_err(|_| format!("state builder {name} stage panicked"))?
}

fn state_builder_threads(num_threads: Option<usize>) -> usize {
    if let Some(threads) = pipeline_threads_override() {
        return threads.max(STATE_BUILD_STAGE_COUNT + 1);
    }
    num_threads
        .unwrap_or_else(|| num_cpus::get_physical().max(1))
        .clamp(STATE_BUILD_STAGE_COUNT + 1, STATE_BUILD_STAGE_COUNT * 2)
}

#[cfg(test)]
mod tests {
    use super::clone_states_for_keys;
    use std::collections::BTreeMap;

    #[test]
    fn state_builder_clones_each_repeated_identity_once() {
        let states = BTreeMap::from([(1, vec![1]), (2, vec![2]), (3, vec![3])]);
        let detached = clone_states_for_keys(&states, [2, 2, 3, 2].into_iter());
        assert_eq!(detached, BTreeMap::from([(2, vec![2]), (3, vec![3])]));
    }
}
