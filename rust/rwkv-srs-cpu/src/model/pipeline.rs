//! High-throughput module-pipeline `process_many` executor.
//!
//! The strict packed/original and legacy bulk-layered executors do not call
//! into this path.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TryRecvError};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use candle_core::{Device, Tensor};
use pyo3::prelude::*;
use rayon::prelude::*;
use rayon::ThreadPool;

use super::bulk::{
    bulk_curve_outputs, bulk_prediction_probabilities, feature_prepass_step,
    flat_working_module_state_from_native, native_module_state_from_flat_working,
    BulkWorkingModuleState,
};
use super::process_payload::parse_process_review_payload;
use super::runtime::{rayon_pool, validate_num_threads, with_rayon_threads};
use super::state::{native_module_state_from_parts, NativeRnnModuleState, ReviewIds};
use super::undo::BatchDeterministicUndoFrame;
use super::{
    features2card_forward_profiled, py_value_error,
    rwkv_rnn_forward_flat_working_state_values_profiled, rwkv_rnn_forward_profiled,
    rwkv_rnn_predict_forward_flat_working_state_values_profiled, rwkv_rnn_predict_forward_profiled,
    NativeProcessManyPyOutput, NativeRnn,
};
use crate::cpu_config::*;
use crate::model_weights::{Features2CardWeights, Rwkv7RnnWeights};
use crate::ops::f32_tensor_data;
use crate::state::{FeatureState, ReviewInput};

const PIPELINE_STAGE_COUNT: usize = 5;

#[derive(Clone, Copy)]
enum PipelineStageKind {
    Card,
    Deck,
    Note,
    Preset,
}

impl PipelineStageKind {
    fn name(self) -> &'static str {
        match self {
            Self::Card => "card",
            Self::Deck => "deck",
            Self::Note => "note",
            Self::Preset => "preset",
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

struct PipelineRow {
    index: usize,
    ids: ReviewIds,
    query_values: Vec<f32>,
    process_values: Vec<f32>,
}

type PipelineMessage = Result<PipelineRow, String>;

struct PipelineStateOutput {
    card_states: BTreeMap<i64, NativeRnnModuleState>,
    deck_states: BTreeMap<i64, NativeRnnModuleState>,
    note_states: BTreeMap<i64, NativeRnnModuleState>,
    preset_states: BTreeMap<i64, NativeRnnModuleState>,
    global_state: Option<NativeRnnModuleState>,
}

struct DeterministicPipelineTransaction<'a> {
    deterministic: &'a mut FeatureState,
    undo: Option<BatchDeterministicUndoFrame>,
    committed: bool,
}

impl<'a> DeterministicPipelineTransaction<'a> {
    fn new(deterministic: &'a mut FeatureState, ids: &[ReviewIds]) -> Self {
        let undo = BatchDeterministicUndoFrame::capture(deterministic, ids);
        Self {
            deterministic,
            undo: Some(undo),
            committed: false,
        }
    }

    fn feature_step(&mut self, input: &ReviewInput) -> PyResult<super::bulk::BulkFeatureStep> {
        let mut profile = None;
        feature_prepass_step(self.deterministic, input, &mut profile)
    }

    fn commit(mut self) {
        self.committed = true;
        self.undo = None;
    }
}

impl Drop for DeterministicPipelineTransaction<'_> {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        if let Some(frame) = self.undo.take() {
            frame.restore(self.deterministic);
        }
    }
}

pub(super) fn process_reviews_pipeline_with_state(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    payload: &[u8],
    return_curves: bool,
    num_threads: Option<usize>,
) -> PyResult<NativeProcessManyPyOutput> {
    let total_start = Instant::now();
    validate_num_threads(num_threads)?;
    let parse_start = Instant::now();
    let inputs = parse_process_review_payload(payload).map_err(py_value_error)?;
    let parse_ns = parse_start.elapsed().as_nanos();
    process_review_inputs_pipeline_with_state_impl(
        rnn,
        deterministic,
        &inputs,
        return_curves,
        num_threads,
        total_start,
        parse_ns,
    )
}

pub(super) fn process_review_inputs_pipeline_with_state(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    inputs: &[ReviewInput],
    return_curves: bool,
    num_threads: Option<usize>,
) -> PyResult<NativeProcessManyPyOutput> {
    let total_start = Instant::now();
    validate_num_threads(num_threads)?;
    process_review_inputs_pipeline_with_state_impl(
        rnn,
        deterministic,
        inputs,
        return_curves,
        num_threads,
        total_start,
        0,
    )
}

fn process_review_inputs_pipeline_with_state_impl(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    inputs: &[ReviewInput],
    return_curves: bool,
    num_threads: Option<usize>,
    total_start: Instant,
    parse_ns: u128,
) -> PyResult<NativeProcessManyPyOutput> {
    if inputs.is_empty() {
        return Ok((
            Vec::new(),
            return_curves.then(Vec::new),
            return_curves.then(Vec::new),
        ));
    }
    rnn.materialize_flat_cpu_state().map_err(py_value_error)?;
    if rnn.weights.rwkv_modules.len() != PIPELINE_STAGE_COUNT {
        return Err(py_value_error(format!(
            "fast process pipeline expected {PIPELINE_STAGE_COUNT} RWKV modules, got {}",
            rnn.weights.rwkv_modules.len()
        )));
    }

    let ids = inputs
        .iter()
        .map(FeatureState::normalized_review_ids)
        .collect::<Vec<_>>();
    let transaction_start = Instant::now();
    let mut deterministic_transaction = DeterministicPipelineTransaction::new(deterministic, &ids);
    let transaction_ns = transaction_start.elapsed().as_nanos();
    let feature_start = Instant::now();
    let worker_threads = pipeline_threads(num_threads);
    let rows = build_pipeline_rows(
        &rnn.weights.features2card,
        &mut deterministic_transaction,
        inputs,
        worker_threads,
    )?;
    let feature_ns = feature_start.elapsed().as_nanos();
    let pipeline_start = Instant::now();
    let (rows, states) = execute_pipeline(rnn, rows, num_threads)?;
    let pipeline_ns = pipeline_start.elapsed().as_nanos();

    let row_count = rows.len();
    let channels = rows
        .first()
        .map(|row| row.query_values.len())
        .ok_or_else(|| py_value_error("fast process pipeline produced no rows"))?;
    let mut query_values = Vec::with_capacity(row_count * channels);
    let mut process_values = return_curves.then(|| Vec::with_capacity(row_count * channels));
    for (expected_index, row) in rows.into_iter().enumerate() {
        if row.index != expected_index {
            return Err(py_value_error(format!(
                "fast process pipeline returned row {} at position {expected_index}",
                row.index
            )));
        }
        if row.query_values.len() != channels || row.process_values.len() != channels {
            return Err(py_value_error(format!(
                "fast process pipeline row {expected_index} has inconsistent channels"
            )));
        }
        query_values.extend(row.query_values);
        if let Some(process_values) = process_values.as_mut() {
            process_values.extend(row.process_values);
        }
    }

    let head_start = Instant::now();
    let query_x = Tensor::from_vec(query_values, (row_count, channels), &Device::Cpu)
        .map_err(py_value_error)?;
    let mut profile = None;
    let probabilities =
        bulk_prediction_probabilities(rnn, &query_x, &mut profile).map_err(py_value_error)?;
    let (curve_ahead_logits, curve_w) = if let Some(process_values) = process_values {
        let process_x = Tensor::from_vec(process_values, (row_count, channels), &Device::Cpu)
            .map_err(py_value_error)?;
        let (ahead, w) =
            bulk_curve_outputs(rnn, &process_x, &mut profile).map_err(py_value_error)?;
        (Some(ahead), Some(w))
    } else {
        (None, None)
    };

    if pipeline_profile_enabled() {
        eprintln!(
            "process_pipeline rows={row_count} parse_ns={parse_ns} transaction_ns={transaction_ns} feature_ns={feature_ns} pipeline_ns={pipeline_ns} head_ns={} total_ns={}",
            head_start.elapsed().as_nanos(),
            total_start.elapsed().as_nanos(),
        );
    }

    rnn.invalidate_gpu();
    rnn.card_states.extend(states.card_states);
    rnn.deck_states.extend(states.deck_states);
    rnn.note_states.extend(states.note_states);
    rnn.preset_states.extend(states.preset_states);
    rnn.global_state = states.global_state;
    deterministic_transaction.commit();
    Ok((probabilities, curve_ahead_logits, curve_w))
}

fn build_pipeline_rows(
    weights: &Features2CardWeights,
    deterministic: &mut DeterministicPipelineTransaction<'_>,
    inputs: &[ReviewInput],
    worker_threads: usize,
) -> PyResult<Vec<PipelineRow>> {
    let mut steps = Vec::with_capacity(inputs.len());
    for input in inputs {
        steps.push(deterministic.feature_step(input)?);
    }

    with_rayon_threads(Some(worker_threads), move || {
        steps
            .into_par_iter()
            .enumerate()
            .map(|(index, step)| {
                let query_values = features_to_card_values(
                    weights,
                    step.predict_features,
                    "pipeline_predict_features",
                )?;
                let process_values = features_to_card_values(
                    weights,
                    step.process_features,
                    "pipeline_process_features",
                )?;
                if query_values.len() != process_values.len() {
                    return Err(py_value_error(format!(
                        "fast process pipeline row {index} has mismatched feature outputs"
                    )));
                }
                Ok(PipelineRow {
                    index,
                    ids: step.ids,
                    query_values,
                    process_values,
                })
            })
            .collect()
    })
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

fn execute_pipeline(
    rnn: &NativeRnn,
    rows: Vec<PipelineRow>,
    num_threads: Option<usize>,
) -> PyResult<(Vec<PipelineRow>, PipelineStateOutput)> {
    let row_count = rows.len();
    let worker_threads = pipeline_threads(num_threads);
    let channel_capacity = pipeline_capacity();

    let pool = rayon_pool(worker_threads)?;

    // Stages only receive shallow clones of the recurrent states they can touch.
    // The authoritative maps stay unchanged until every stage and output head
    // succeeds, so an error cannot empty or partially advance the runtime.
    let card_states = clone_states_for_keys(&rnn.card_states, rows.iter().map(|row| row.ids.0));
    let note_states = clone_states_for_keys(&rnn.note_states, rows.iter().map(|row| row.ids.1));
    let deck_states = clone_states_for_keys(&rnn.deck_states, rows.iter().map(|row| row.ids.2));
    let preset_states = clone_states_for_keys(&rnn.preset_states, rows.iter().map(|row| row.ids.3));
    let global_state = rnn.global_state.clone();

    let weights = &rnn.weights.rwkv_modules;
    let outcome = thread::scope(|scope| {
        let (card_tx, card_rx) = sync_channel::<PipelineMessage>(channel_capacity);
        let (deck_tx, deck_rx) = sync_channel::<PipelineMessage>(channel_capacity);
        let (note_tx, note_rx) = sync_channel::<PipelineMessage>(channel_capacity);
        let (preset_tx, preset_rx) = sync_channel::<PipelineMessage>(channel_capacity);
        let (global_tx, global_rx) = sync_channel::<PipelineMessage>(channel_capacity);
        let (output_tx, output_rx) = sync_channel::<PipelineMessage>(channel_capacity);

        let producer = scope.spawn(move || produce_rows(rows, card_tx));
        let card_pool = Arc::clone(&pool);
        let card = scope.spawn(move || {
            run_keyed_stage(
                PipelineStageKind::Card,
                &weights[0],
                card_states,
                card_rx,
                deck_tx,
                row_count,
                card_pool,
            )
        });
        let deck_pool = Arc::clone(&pool);
        let deck = scope.spawn(move || {
            run_keyed_stage(
                PipelineStageKind::Deck,
                &weights[1],
                deck_states,
                deck_rx,
                note_tx,
                row_count,
                deck_pool,
            )
        });
        let note_pool = Arc::clone(&pool);
        let note = scope.spawn(move || {
            run_keyed_stage(
                PipelineStageKind::Note,
                &weights[2],
                note_states,
                note_rx,
                preset_tx,
                row_count,
                note_pool,
            )
        });
        let preset_pool = Arc::clone(&pool);
        let preset = scope.spawn(move || {
            run_keyed_stage(
                PipelineStageKind::Preset,
                &weights[3],
                preset_states,
                preset_rx,
                global_tx,
                row_count,
                preset_pool,
            )
        });
        let global_pool = Arc::clone(&pool);
        let global = scope.spawn(move || {
            run_global_stage(&weights[4], global_state, global_rx, output_tx, global_pool)
        });

        let mut output_rows = Vec::with_capacity(row_count);
        let mut receive_error = None;
        while let Ok(message) = output_rx.recv() {
            match message {
                Ok(row) => output_rows.push(row),
                Err(error) => {
                    receive_error = Some(error);
                    break;
                }
            }
        }

        let producer_result = join_stage("producer", producer.join());
        let card_result = join_stage("card", card.join());
        let deck_result = join_stage("deck", deck.join());
        let note_result = join_stage("note", note.join());
        let preset_result = join_stage("preset", preset.join());
        let global_result = join_stage("global", global.join());

        if let Some(error) = receive_error {
            return Err(error);
        }
        producer_result?;
        let card_states = card_result?;
        let deck_states = deck_result?;
        let note_states = note_result?;
        let preset_states = preset_result?;
        let global_state = global_result?;
        if output_rows.len() != row_count {
            return Err(format!(
                "fast process pipeline returned {} rows for {row_count} inputs",
                output_rows.len()
            ));
        }

        Ok((
            output_rows,
            PipelineStateOutput {
                card_states,
                deck_states,
                note_states,
                preset_states,
                global_state,
            },
        ))
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

fn produce_rows(rows: Vec<PipelineRow>, output: SyncSender<PipelineMessage>) -> Result<(), String> {
    for row in rows {
        output
            .send(Ok(row))
            .map_err(|_| "fast process pipeline card stage closed early".to_string())?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_keyed_stage(
    kind: PipelineStageKind,
    weights: &Rwkv7RnnWeights,
    mut native_states: BTreeMap<i64, NativeRnnModuleState>,
    input: Receiver<PipelineMessage>,
    output: SyncSender<PipelineMessage>,
    row_count: usize,
    pool: Arc<ThreadPool>,
) -> Result<BTreeMap<i64, NativeRnnModuleState>, String> {
    let mut compute_ns = 0u128;
    let mut working_states: HashMap<i64, Option<BulkWorkingModuleState>> =
        HashMap::with_capacity(row_count.min(16_384));
    loop {
        let batch = receive_pipeline_batch(&input);
        if batch.rows.is_empty() {
            if let Some(error) = batch.terminal_error {
                let _ = output.send(Err(error.clone()));
                return Err(error);
            }
            break;
        }
        let compute_start = Instant::now();
        let processed = pool.install(|| {
            let mut processed = Vec::with_capacity(batch.rows.len());
            for mut row in batch.rows {
                let key = kind.key(row.ids);
                let state = match working_states.entry(key) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        let working = native_states
                            .remove(&key)
                            .map(|state| flat_working_module_state_from_native(&state, true))
                            .transpose()
                            .map_err(|error| {
                                format!(
                                    "fast pipeline {} state conversion failed: {error}",
                                    kind.name()
                                )
                            })?;
                        entry.insert(working)
                    }
                };
                let next_state = process_stage_row_in_pool(weights, state.as_ref(), &mut row)?;
                *state = Some(next_state);
                processed.push(row);
            }
            Ok::<_, String>(processed)
        });
        let processed = match processed {
            Ok(processed) => processed,
            Err(error) => {
                let _ = output.send(Err(error.clone()));
                return Err(error);
            }
        };
        compute_ns += compute_start.elapsed().as_nanos();
        for row in processed {
            if output.send(Ok(row)).is_err() {
                return Err(format!(
                    "fast pipeline {} stage output closed early",
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

    for (key, state) in working_states {
        let state = state.ok_or_else(|| {
            format!(
                "fast pipeline {} state {key} was not initialized",
                kind.name()
            )
        })?;
        let native =
            native_module_state_from_flat_working(state, &Device::Cpu).map_err(|error| {
                format!(
                    "fast pipeline {} final-state conversion failed: {error}",
                    kind.name()
                )
            })?;
        native_states.insert(key, native);
    }
    if pipeline_profile_enabled() {
        eprintln!(
            "process_pipeline_stage name={} compute_ns={compute_ns}",
            kind.name()
        );
    }
    Ok(native_states)
}

fn run_global_stage(
    weights: &Rwkv7RnnWeights,
    native_state: Option<NativeRnnModuleState>,
    input: Receiver<PipelineMessage>,
    output: SyncSender<PipelineMessage>,
    pool: Arc<ThreadPool>,
) -> Result<Option<NativeRnnModuleState>, String> {
    let mut compute_ns = 0u128;
    let mut working_state = native_state
        .map(|state| flat_working_module_state_from_native(&state, true))
        .transpose()
        .map_err(|error| format!("fast pipeline global state conversion failed: {error}"))?;
    loop {
        let batch = receive_pipeline_batch(&input);
        if batch.rows.is_empty() {
            if let Some(error) = batch.terminal_error {
                let _ = output.send(Err(error.clone()));
                return Err(error);
            }
            break;
        }
        let compute_start = Instant::now();
        let processed = pool.install(|| {
            let mut processed = Vec::with_capacity(batch.rows.len());
            for mut row in batch.rows {
                working_state = Some(process_stage_row_in_pool(
                    weights,
                    working_state.as_ref(),
                    &mut row,
                )?);
                processed.push(row);
            }
            Ok::<_, String>(processed)
        });
        let processed = match processed {
            Ok(processed) => processed,
            Err(error) => {
                let _ = output.send(Err(error.clone()));
                return Err(error);
            }
        };
        compute_ns += compute_start.elapsed().as_nanos();
        for row in processed {
            if output.send(Ok(row)).is_err() {
                return Err("fast pipeline global stage output closed early".to_string());
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

    let native_state = working_state
        .map(|state| native_module_state_from_flat_working(state, &Device::Cpu))
        .transpose()
        .map_err(|error| format!("fast pipeline global final-state conversion failed: {error}"))?;
    if pipeline_profile_enabled() {
        eprintln!("process_pipeline_stage name=global compute_ns={compute_ns}");
    }
    Ok(native_state)
}

fn process_stage_row_in_pool(
    weights: &Rwkv7RnnWeights,
    state: Option<&BulkWorkingModuleState>,
    row: &mut PipelineRow,
) -> Result<BulkWorkingModuleState, String> {
    let channels = row.query_values.len();
    if row.process_values.len() != channels {
        return Err(format!(
            "fast pipeline row {} has mismatched lane widths",
            row.index
        ));
    }
    let time_state = state.map(|state| state.time_state_by_layer.as_slice());
    let channel_state = state.and_then(|state| state.channel_flat_state_by_layer.as_deref());
    let (query_result, process_result) = rayon::join(
        || {
            rwkv_rnn_predict_forward_flat_working_state_values_profiled(
                weights,
                &row.query_values,
                1,
                channels,
                &Device::Cpu,
                time_state,
                channel_state,
                None,
            )
            .map_err(|error| format!("fast pipeline query forward failed: {error}"))
        },
        || {
            rwkv_rnn_forward_flat_working_state_values_profiled(
                weights,
                &row.process_values,
                1,
                channels,
                &Device::Cpu,
                time_state,
                channel_state,
                None,
            )
            .map_err(|error| format!("fast pipeline process forward failed: {error}"))
        },
    );
    match (query_result?, process_result?) {
        (
            Some(query_values),
            Some((process_values, time_state_by_layer, channel_state_by_layer)),
        ) => {
            row.query_values = query_values;
            row.process_values = process_values;
            Ok(BulkWorkingModuleState {
                time_state_by_layer,
                channel_state_b1c_by_layer: None,
                channel_flat_state_by_layer: Some(channel_state_by_layer),
            })
        }
        _ => process_stage_row_portable(weights, state, row),
    }
}

fn process_stage_row_portable(
    weights: &Rwkv7RnnWeights,
    state: Option<&BulkWorkingModuleState>,
    row: &mut PipelineRow,
) -> Result<BulkWorkingModuleState, String> {
    let channels = row.query_values.len();
    let native_state = state
        .cloned()
        .map(|state| native_module_state_from_flat_working(state, &Device::Cpu))
        .transpose()
        .map_err(|error| format!("portable pipeline state conversion failed: {error}"))?;
    let (time_x_shift, time_state, channel_state) = match native_state.as_ref() {
        Some(state) => (
            Some(state.time_x_shift_b1c_by_layer.as_slice()),
            Some(state.time_state_b1hkk_by_layer.as_slice()),
            Some(state.channel_state_b1c_by_layer.as_slice()),
        ),
        None => (None, None, None),
    };
    let query_x = Tensor::from_vec(row.query_values.clone(), (1usize, channels), &Device::Cpu)
        .map_err(|error| format!("portable pipeline query input failed: {error}"))?;
    let process_x = Tensor::from_vec(row.process_values.clone(), (1usize, channels), &Device::Cpu)
        .map_err(|error| format!("portable pipeline process input failed: {error}"))?;

    let (query_result, process_result) = rayon::join(
        || {
            rwkv_rnn_predict_forward_profiled(
                weights,
                &query_x,
                time_x_shift,
                time_state,
                channel_state,
                None,
            )
            .map_err(|error| format!("portable pipeline query forward failed: {error}"))
        },
        || {
            rwkv_rnn_forward_profiled(
                weights,
                &process_x,
                time_x_shift,
                time_state,
                channel_state,
                None,
            )
            .map_err(|error| format!("portable pipeline process forward failed: {error}"))
        },
    );
    let query_output = query_result?;
    let (process_output, next_time_x, next_time_state, next_channel_state) = process_result?;
    let query_values = f32_tensor_data(&query_output)
        .and_then(|data| data.as_slice().map(|values| values.to_vec()))
        .map_err(|error| format!("portable pipeline query output failed: {error}"))?;
    let process_values = f32_tensor_data(&process_output)
        .and_then(|data| data.as_slice().map(|values| values.to_vec()))
        .map_err(|error| format!("portable pipeline process output failed: {error}"))?;
    let native_next = native_module_state_from_parts(
        next_time_x,
        next_time_state,
        next_channel_state,
        "portable pipeline next state",
    )
    .map_err(|error| format!("portable pipeline next-state conversion failed: {error}"))?;
    let next_state = flat_working_module_state_from_native(&native_next, true)
        .map_err(|error| format!("portable pipeline flat-state conversion failed: {error}"))?;

    row.query_values = query_values;
    row.process_values = process_values;
    Ok(next_state)
}

struct PipelineBatch {
    rows: Vec<PipelineRow>,
    terminal_error: Option<String>,
    disconnected: bool,
}

fn receive_pipeline_batch(input: &Receiver<PipelineMessage>) -> PipelineBatch {
    let mut batch = PipelineBatch {
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
    result.map_err(|_| format!("fast process pipeline {name} stage panicked"))?
}

fn pipeline_threads(num_threads: Option<usize>) -> usize {
    if let Some(threads) = pipeline_threads_override() {
        return threads.max(PIPELINE_STAGE_COUNT + 1);
    }
    num_threads
        .unwrap_or_else(|| num_cpus::get_physical().max(1))
        .max(PIPELINE_STAGE_COUNT + 1)
        .min(PIPELINE_STAGE_COUNT * 2)
}

#[cfg(test)]
mod tests {
    use super::clone_states_for_keys;
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[derive(Debug)]
    struct CloneCounter(Arc<AtomicUsize>);

    impl Clone for CloneCounter {
        fn clone(&self) -> Self {
            self.0.fetch_add(1, Ordering::Relaxed);
            Self(Arc::clone(&self.0))
        }
    }

    #[test]
    fn pipeline_state_detachment_clones_each_repeated_identity_once() {
        let clone_count = Arc::new(AtomicUsize::new(0));
        let states = BTreeMap::from([
            (1, CloneCounter(Arc::clone(&clone_count))),
            (2, CloneCounter(Arc::clone(&clone_count))),
        ]);

        let detached = clone_states_for_keys(&states, [1, 1, 2, 1, 3, 2].into_iter());

        assert_eq!(detached.keys().copied().collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(clone_count.load(Ordering::Relaxed), 2);
    }
}
