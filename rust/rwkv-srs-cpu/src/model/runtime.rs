use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use candle_core::{bail, Device, Result, Tensor, D};
use candle_nn::ops as nn_ops;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyMapping};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

use super::process_payload::parse_process_review_payload;
use super::state::{
    prepared_i64, review_ids_from_prepared, state_input_tensor_count, state_output_tensor_count,
    ReviewIds,
};
use super::undo::{BatchUndoFrame, RuntimeUndoFrame, UndoFrame};
use super::{
    py_value_error, srs_review_forward_profiled_options,
    srs_review_predict_forward_lightning_modulewise, srs_review_predict_forward_lightning_profiled,
    srs_review_predict_forward_profiled, NativeProcessManyPyOutput, NativeRnn,
    PredictManyForwardMode,
};
use crate::cpu_config::*;
use crate::gpu::{is_gpu_out_of_memory, py_gpu_error};
use crate::id_encoding::ID_SUBMODULES;
use crate::ops::with_fast_layer_norm;
use crate::profile::{ProfileTimer, RuntimeProfile};
use crate::py_state::{predict_review_from_dict, predict_review_from_mapping, review_from_mapping};
use crate::state::{FeatureState, PreparedRow, ReviewInput};
use crate::tensor_io::{tensor_from_2d, Tensor2List};

const PREDICT_MANY_LIGHTNING_PARALLEL_MIN_ROWS: usize = 192;

#[derive(Debug, Default)]
struct FastPredictBatchStageProfile {
    feature_preparation_ns: u128,
    state_lookup_packing_ns: u128,
    model_forward_inclusive_ns: u128,
    model_forward_ns: u128,
    result_construction_ns: u128,
}

#[derive(Debug, Default)]
pub(super) struct FastPredictCallerProfile {
    review_count: usize,
    total_ns: u128,
    input_parse_ns: u128,
    scope_validation_ns: u128,
    flat_state_materialization_ns: u128,
    prepare_predict_ns: u128,
    prediction_wall_ns: u128,
    execution_path: &'static str,
    batch_count: usize,
    rows_per_batch: Vec<usize>,
    feature_preparation_worker_ns: u128,
    feature_preparation_max_batch_ns: u128,
    state_lookup_packing_worker_ns: u128,
    state_lookup_packing_max_batch_ns: u128,
    model_forward_inclusive_worker_ns: u128,
    model_forward_inclusive_max_batch_ns: u128,
    model_forward_worker_ns: u128,
    model_forward_max_batch_ns: u128,
    result_construction_worker_ns: u128,
    result_construction_max_batch_ns: u128,
}

impl FastPredictCallerProfile {
    fn record_batch(&mut self, rows: usize, batch: &FastPredictBatchStageProfile) {
        self.batch_count += 1;
        self.rows_per_batch.push(rows);
        self.feature_preparation_worker_ns += batch.feature_preparation_ns;
        self.feature_preparation_max_batch_ns = self
            .feature_preparation_max_batch_ns
            .max(batch.feature_preparation_ns);
        self.state_lookup_packing_worker_ns += batch.state_lookup_packing_ns;
        self.state_lookup_packing_max_batch_ns = self
            .state_lookup_packing_max_batch_ns
            .max(batch.state_lookup_packing_ns);
        self.model_forward_inclusive_worker_ns += batch.model_forward_inclusive_ns;
        self.model_forward_inclusive_max_batch_ns = self
            .model_forward_inclusive_max_batch_ns
            .max(batch.model_forward_inclusive_ns);
        self.model_forward_worker_ns += batch.model_forward_ns;
        self.model_forward_max_batch_ns =
            self.model_forward_max_batch_ns.max(batch.model_forward_ns);
        self.result_construction_worker_ns += batch.result_construction_ns;
        self.result_construction_max_batch_ns = self
            .result_construction_max_batch_ns
            .max(batch.result_construction_ns);
    }

    pub(super) fn to_pydict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("review_count", self.review_count)?;
        dict.set_item("total_ns", self.total_ns)?;
        dict.set_item("input_parse_ns", self.input_parse_ns)?;
        dict.set_item("scope_validation_ns", self.scope_validation_ns)?;
        dict.set_item(
            "flat_state_materialization_ns",
            self.flat_state_materialization_ns,
        )?;
        dict.set_item("prepare_predict_ns", self.prepare_predict_ns)?;
        dict.set_item("prediction_wall_ns", self.prediction_wall_ns)?;
        dict.set_item("execution_path", self.execution_path)?;
        dict.set_item("batch_count", self.batch_count)?;
        dict.set_item("rows_per_batch", &self.rows_per_batch)?;
        dict.set_item(
            "feature_preparation_worker_ns",
            self.feature_preparation_worker_ns,
        )?;
        dict.set_item(
            "feature_preparation_max_batch_ns",
            self.feature_preparation_max_batch_ns,
        )?;
        dict.set_item(
            "state_lookup_packing_worker_ns",
            self.state_lookup_packing_worker_ns,
        )?;
        dict.set_item(
            "state_lookup_packing_max_batch_ns",
            self.state_lookup_packing_max_batch_ns,
        )?;
        dict.set_item(
            "model_forward_inclusive_worker_ns",
            self.model_forward_inclusive_worker_ns,
        )?;
        dict.set_item(
            "model_forward_inclusive_max_batch_ns",
            self.model_forward_inclusive_max_batch_ns,
        )?;
        dict.set_item("model_forward_worker_ns", self.model_forward_worker_ns)?;
        dict.set_item(
            "model_forward_max_batch_ns",
            self.model_forward_max_batch_ns,
        )?;
        dict.set_item(
            "result_construction_worker_ns",
            self.result_construction_worker_ns,
        )?;
        dict.set_item(
            "result_construction_max_batch_ns",
            self.result_construction_max_batch_ns,
        )?;
        Ok(dict.unbind())
    }
}

pub(super) fn process_reviews_with_state(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    reviews: &Bound<'_, PyAny>,
    return_curves: bool,
    num_threads: Option<usize>,
    mut profile: Option<&mut RuntimeProfile>,
) -> PyResult<NativeProcessManyPyOutput> {
    validate_num_threads(num_threads)?;
    let mut inputs = Vec::new();

    for review in reviews.iter()? {
        let start = ProfileTimer::start(profile.is_some());
        let review = review?;
        let review = review.downcast::<PyMapping>()?;
        let input = review_from_mapping(review, true)?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.parse_review_ns += start.elapsed_ns();
        }
        inputs.push(input);
    }

    process_review_inputs_with_state(
        rnn,
        deterministic,
        &inputs,
        return_curves,
        num_threads,
        &mut profile,
    )
}

pub(super) fn process_packed_reviews_with_state(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    payload: &[u8],
    return_curves: bool,
    num_threads: Option<usize>,
    mut profile: Option<&mut RuntimeProfile>,
) -> PyResult<NativeProcessManyPyOutput> {
    validate_num_threads(num_threads)?;
    let start = ProfileTimer::start(profile.is_some());
    let inputs = parse_process_review_payload(payload).map_err(py_value_error)?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.parse_review_ns += start.elapsed_ns();
    }

    process_review_inputs_with_state(
        rnn,
        deterministic,
        &inputs,
        return_curves,
        num_threads,
        &mut profile,
    )
}

pub(super) fn build_state_only_packed_with_state(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    payload: &[u8],
    num_threads: Option<usize>,
    mut profile: Option<&mut RuntimeProfile>,
) -> PyResult<usize> {
    validate_num_threads(num_threads)?;
    let start = ProfileTimer::start(profile.is_some());
    let inputs = parse_process_review_payload(payload).map_err(py_value_error)?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.parse_review_ns += start.elapsed_ns();
    }

    build_state_only_inputs_with_state(rnn, deterministic, &inputs, num_threads, &mut profile)
}

pub(super) fn build_state_only_inputs_with_state(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    inputs: &[ReviewInput],
    num_threads: Option<usize>,
    profile: &mut Option<&mut RuntimeProfile>,
) -> PyResult<usize> {
    validate_num_threads(num_threads)?;
    rnn.materialize_flat_cpu_state().map_err(py_value_error)?;
    let ids = inputs
        .iter()
        .map(FeatureState::normalized_review_ids)
        .collect::<Vec<_>>();
    let transaction_start = ProfileTimer::start(profile.is_some());
    let undo = BatchUndoFrame::capture(rnn, deterministic, &ids);
    if let Some(profile) = profile.as_deref_mut() {
        profile.transaction_capture_ns += transaction_start.elapsed_ns();
    }

    let result = (|| {
        for input in inputs {
            if let Some(profile) = profile.as_deref_mut() {
                profile.review_count += 1;
            }

            let start = ProfileTimer::start(profile.is_some());
            let process_row = deterministic
                .prepare_process_row(input)
                .map_err(py_value_error)?;
            let ids = review_ids_from_prepared(&process_row).map_err(py_value_error)?;
            if let Some(profile) = profile.as_deref_mut() {
                profile.prepare_process_ns += start.elapsed_ns();
            }

            let (ahead, weights) = process_prepared_row_with_state(
                rnn,
                deterministic,
                &process_row,
                ids,
                false,
                num_threads,
                profile,
            )?;
            debug_assert!(ahead.is_none());
            debug_assert!(weights.is_none());
        }
        Ok(inputs.len())
    })();

    match result {
        Ok(processed_count) => Ok(processed_count),
        Err(error) => {
            undo.restore(rnn, deterministic);
            Err(error)
        }
    }
}

pub(super) fn process_review_inputs_with_state(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    inputs: &[ReviewInput],
    return_curves: bool,
    num_threads: Option<usize>,
    profile: &mut Option<&mut RuntimeProfile>,
) -> PyResult<NativeProcessManyPyOutput> {
    validate_num_threads(num_threads)?;
    rnn.materialize_flat_cpu_state().map_err(py_value_error)?;
    let ids = inputs
        .iter()
        .map(FeatureState::normalized_review_ids)
        .collect::<Vec<_>>();
    let transaction_start = ProfileTimer::start(profile.is_some());
    let undo = BatchUndoFrame::capture(rnn, deterministic, &ids);
    if let Some(profile) = profile.as_deref_mut() {
        profile.transaction_capture_ns += transaction_start.elapsed_ns();
    }
    let result = (|| {
        let mut prediction_probabilities = Vec::with_capacity(inputs.len());
        let mut curve_ahead_logits = return_curves.then(|| Vec::with_capacity(inputs.len()));
        let mut curve_w = return_curves.then(|| Vec::with_capacity(inputs.len()));

        for input in inputs {
            process_review_input_with_state(
                rnn,
                deterministic,
                input,
                return_curves,
                num_threads,
                profile,
                &mut prediction_probabilities,
                &mut curve_ahead_logits,
                &mut curve_w,
            )?;
        }

        Ok((prediction_probabilities, curve_ahead_logits, curve_w))
    })();

    match result {
        Ok(output) => Ok(output),
        Err(error) => {
            undo.restore(rnn, deterministic);
            Err(error)
        }
    }
}

pub(super) fn process_review_input_with_state(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    input: &ReviewInput,
    return_curves: bool,
    num_threads: Option<usize>,
    profile: &mut Option<&mut RuntimeProfile>,
    prediction_probabilities: &mut Vec<f64>,
    curve_ahead_logits: &mut Option<Vec<Tensor2List>>,
    curve_w: &mut Option<Vec<Tensor2List>>,
) -> PyResult<()> {
    if let Some(profile) = profile.as_deref_mut() {
        profile.review_count += 1;
    }

    let start = ProfileTimer::start(profile.is_some());
    let predict_row = deterministic.prepare_predict_row(input);
    let ids = review_ids_from_prepared(&predict_row).map_err(py_value_error)?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.prepare_predict_ns += start.elapsed_ns();
    }

    let probability = predict_prepared_row_probability(
        rnn,
        deterministic,
        &predict_row,
        ids,
        PredictManyForwardMode::Oracle,
        num_threads,
        profile,
    )?;
    prediction_probabilities.push(probability);

    let start = ProfileTimer::start(profile.is_some());
    let process_row = deterministic
        .prepare_process_row(input)
        .map_err(py_value_error)?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.prepare_process_ns += start.elapsed_ns();
    }

    let (ahead_logits, w_logits) = process_prepared_row_with_state(
        rnn,
        deterministic,
        &process_row,
        ids,
        return_curves,
        num_threads,
        profile,
    )?;

    if return_curves {
        curve_ahead_logits
            .as_mut()
            .expect("curve outputs requested")
            .push(ahead_logits.expect("curve logits are present when requested"));
        curve_w
            .as_mut()
            .expect("curve outputs requested")
            .push(w_logits.expect("curve weights are present when requested"));
    }
    Ok(())
}

pub(super) fn undoable_process_review_with_state(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    undo_stack: &mut VecDeque<RuntimeUndoFrame>,
    undo_limit: usize,
    review: &Bound<'_, PyMapping>,
    return_curves: bool,
    num_threads: Option<usize>,
) -> PyResult<(f64, Option<Tensor2List>, Option<Tensor2List>)> {
    validate_num_threads(num_threads)?;
    if undo_limit == 0 {
        return Err(py_value_error(
            "undoable_process is disabled by undo_limit=0",
        ));
    }

    let input = review_from_mapping(review, true)?;
    let (output, undo_frame) =
        undoable_process_input_with_state(rnn, deterministic, &input, return_curves, num_threads)?;
    undo_stack.push_back(RuntimeUndoFrame::model_only(undo_frame));
    if undo_stack.len() > undo_limit {
        undo_stack.pop_front();
    }
    Ok(output)
}

pub(super) fn undoable_process_input_with_state(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    input: &ReviewInput,
    return_curves: bool,
    num_threads: Option<usize>,
) -> PyResult<((f64, Option<Tensor2List>, Option<Tensor2List>), UndoFrame)> {
    validate_num_threads(num_threads)?;
    rnn.materialize_flat_cpu_state().map_err(py_value_error)?;
    let predict_row = deterministic.prepare_predict_row(&input);
    let ids = review_ids_from_prepared(&predict_row).map_err(py_value_error)?;
    let mut profile = None;
    let probability = predict_prepared_row_probability(
        rnn,
        deterministic,
        &predict_row,
        ids,
        PredictManyForwardMode::Oracle,
        num_threads,
        &mut profile,
    )?;

    let process_row = deterministic
        .prepare_process_row(&input)
        .map_err(py_value_error)?;
    let undo_frame =
        UndoFrame::capture(rnn, deterministic, &process_row, ids).map_err(py_value_error)?;

    let mut profile = None;
    let result = process_prepared_row_with_state(
        rnn,
        deterministic,
        &process_row,
        ids,
        return_curves,
        num_threads,
        &mut profile,
    );

    match result {
        Ok((ahead_logits, w_logits)) => Ok(((probability, ahead_logits, w_logits), undo_frame)),
        Err(err) => {
            undo_frame.restore(rnn, deterministic);
            Err(err)
        }
    }
}

fn process_prepared_row_with_state(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    process_row: &PreparedRow,
    ids: ReviewIds,
    return_curves: bool,
    num_threads: Option<usize>,
    profile: &mut Option<&mut RuntimeProfile>,
) -> PyResult<(Option<Tensor2List>, Option<Tensor2List>)> {
    let start = ProfileTimer::start(profile.is_some());
    let process_features = deterministic
        .process_feature_vector(process_row)
        .map_err(py_value_error)?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.process_feature_ns += start.elapsed_ns();
        profile.materialization.process_feature_vecs += 1;
        profile.materialization.process_feature_vec_values += process_features.len();
    }

    let start = ProfileTimer::start(profile.is_some());
    let process_feature_values = process_features.len();
    let process_features =
        tensor_from_feature_vector(process_features, "process_features").map_err(py_value_error)?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.process_tensor_ns += start.elapsed_ns();
        profile.materialization.feature_tensors += 1;
        profile.materialization.feature_tensor_values += process_feature_values;
    }

    let (card_id, note_id, deck_id, preset_id) = ids;
    let start = ProfileTimer::start(profile.is_some());
    let state_inputs = rnn.state_inputs(card_id, note_id, deck_id, preset_id);
    if let Some(profile) = profile.as_deref_mut() {
        profile.materialization.state_input_ns += start.elapsed_ns();
        profile.materialization.state_input_snapshots += 1;
        profile.materialization.state_input_tensor_handles +=
            state_input_tensor_count(&state_inputs);
    }
    let (time_x_shift_b1c_by_module, time_state_b1hkk_by_module, channel_state_b1c_by_module) =
        state_inputs;

    let start = ProfileTimer::start(profile.is_some());
    let (
        out_ahead_logits,
        out_w,
        _out_p_logits,
        next_time_x_shift_b1c_by_module,
        next_time_state_b1hkk_by_module,
        next_channel_state_b1c_by_module,
    ) = with_rayon_threads(num_threads, || {
        srs_review_forward_profiled_options(
            &rnn.weights,
            &process_features,
            Some(&time_x_shift_b1c_by_module),
            Some(&time_state_b1hkk_by_module),
            Some(&channel_state_b1c_by_module),
            return_curves,
            false,
            profile
                .as_deref_mut()
                .map(|profile| &mut profile.process_forward),
        )
        .map_err(py_value_error)
    })?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.process_forward_ns += start.elapsed_ns();
    }

    let state_store_tensor_handles = profile.as_ref().map(|_| {
        state_output_tensor_count(
            &next_time_x_shift_b1c_by_module,
            &next_time_state_b1hkk_by_module,
            &next_channel_state_b1c_by_module,
        )
    });
    let start = ProfileTimer::start(profile.is_some());
    rnn.store_review_states(
        card_id,
        note_id,
        deck_id,
        preset_id,
        next_time_x_shift_b1c_by_module,
        next_time_state_b1hkk_by_module,
        next_channel_state_b1c_by_module,
    )
    .map_err(py_value_error)?;
    let state_store_ns = start.elapsed_ns();
    if let Some(profile) = profile.as_deref_mut() {
        profile.materialization.state_store_ns += state_store_ns;
        profile.materialization.state_store_snapshots += 1;
        profile.materialization.state_store_tensor_handles +=
            state_store_tensor_handles.expect("state store count captured for profiling");
    }

    let start = ProfileTimer::start(profile.is_some());
    deterministic
        .record_recurrent_state_update(process_row)
        .map_err(py_value_error)?;
    deterministic
        .record_processed_row(process_row)
        .map_err(py_value_error)?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.state_update_ns += state_store_ns + start.elapsed_ns();
    }

    if !return_curves {
        return Ok((None, None));
    }

    let start = ProfileTimer::start(profile.is_some());
    let ahead_logits = out_ahead_logits
        .expect("curve logits are present when requested")
        .to_vec2::<f32>()
        .map_err(py_value_error)?;
    let w_logits = out_w
        .expect("curve weights are present when requested")
        .to_vec2::<f32>()
        .map_err(py_value_error)?;
    if let Some(profile) = profile.as_deref_mut() {
        let curve_output_values =
            tensor2_list_value_count(&ahead_logits) + tensor2_list_value_count(&w_logits);
        profile.curve_output_ns += start.elapsed_ns();
        profile.materialization.curve_output_vecs += 2;
        profile.materialization.curve_output_values += curve_output_values;
    }
    Ok((Some(ahead_logits), Some(w_logits)))
}

pub(super) fn predict_reviews_with_state(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    reviews: &Bound<'_, PyAny>,
    batch_size: usize,
    num_threads: Option<usize>,
    lightning: bool,
    mut profile: Option<&mut RuntimeProfile>,
) -> PyResult<Vec<f64>> {
    let mut inputs = Vec::new();
    for review in reviews.iter()? {
        let start = ProfileTimer::start(profile.is_some());
        let review = review?;
        let review = review.downcast::<PyMapping>()?;
        inputs.push(review_from_mapping(review, false)?);
        if let Some(profile) = profile.as_deref_mut() {
            profile.parse_review_ns += start.elapsed_ns();
        }
    }
    predict_review_inputs_cpu(
        rnn,
        deterministic,
        &inputs,
        batch_size,
        num_threads,
        lightning,
        profile,
    )
}

/// Predict native review inputs through the same CPU implementation used by
/// the public mapping-based `predict_many()` binding.
pub(super) fn predict_review_inputs_cpu(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    inputs: &[ReviewInput],
    batch_size: usize,
    num_threads: Option<usize>,
    lightning: bool,
    mut profile: Option<&mut RuntimeProfile>,
) -> PyResult<Vec<f64>> {
    if batch_size < 1 {
        return Err(py_value_error("batch_size must be at least 1"));
    }
    validate_num_threads(num_threads)?;
    rnn.materialize_flat_cpu_state().map_err(py_value_error)?;
    if lightning && profile.is_none() && predict_many_lightning_parallel_batches_enabled() {
        return predict_reviews_with_state_parallel_batches(
            rnn,
            deterministic,
            inputs,
            batch_size,
            num_threads,
        );
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.batch.requested_batch_size = batch_size;
    }
    let forward_mode = PredictManyForwardMode::from_lightning(lightning);
    // The state packer limits broadcasting to short flushes internally. Keep the
    // decision per flush so long predict_many() scans can still broadcast
    // duplicate states in their normal batch-sized chunks.
    let allow_short_burst_broadcast = true;

    let mut prediction_probabilities = Vec::new();
    let mut batch_rows = Vec::with_capacity(batch_size);
    let mut batch_ids = Vec::with_capacity(batch_size);

    for input in inputs {
        if let Some(profile) = profile.as_deref_mut() {
            profile.review_count += 1;
        }

        let start = ProfileTimer::start(profile.is_some());
        let predict_row = deterministic.prepare_predict_row(input);
        let ids = review_ids_from_prepared(&predict_row).map_err(py_value_error)?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.prepare_predict_ns += start.elapsed_ns();
        }

        if deterministic
            .can_batch_predict(&predict_row)
            .map_err(py_value_error)?
        {
            batch_rows.push(predict_row);
            batch_ids.push(ids);
            if batch_rows.len() >= batch_size {
                flush_predict_batch(
                    rnn,
                    deterministic,
                    &mut batch_rows,
                    &mut batch_ids,
                    &mut prediction_probabilities,
                    allow_short_burst_broadcast,
                    forward_mode,
                    num_threads,
                    &mut profile,
                )?;
            }
        } else {
            if let Some(profile) = profile.as_deref_mut() {
                profile.batch.scalar_fallback_count += 1;
                match batch_predict_rejection_reason(deterministic, &predict_row)
                    .map_err(py_value_error)?
                {
                    BatchPredictRejectionReason::MissingEncoding => {
                        profile.batch.missing_encoding_fallback_count += 1;
                    }
                    BatchPredictRejectionReason::MissingState => {
                        profile.batch.missing_state_fallback_count += 1;
                    }
                }
            }
            flush_predict_batch(
                rnn,
                deterministic,
                &mut batch_rows,
                &mut batch_ids,
                &mut prediction_probabilities,
                allow_short_burst_broadcast,
                forward_mode,
                num_threads,
                &mut profile,
            )?;
            let probability = predict_prepared_row_probability(
                rnn,
                deterministic,
                &predict_row,
                ids,
                forward_mode,
                num_threads,
                &mut profile,
            )?;
            prediction_probabilities.push(probability);
        }
    }

    flush_predict_batch(
        rnn,
        deterministic,
        &mut batch_rows,
        &mut batch_ids,
        &mut prediction_probabilities,
        allow_short_burst_broadcast,
        forward_mode,
        num_threads,
        &mut profile,
    )?;
    Ok(prediction_probabilities)
}

/// Profile the current public CPU Fast executor without routing timed public
/// calls through the older sequential diagnostic forward. Worker stage times
/// are accumulated CPU time and overlap across Rayon batches; `prediction_wall_ns`
/// is the corresponding caller-visible native wall interval.
pub(super) fn predict_review_inputs_cpu_fast_caller_profiled(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    inputs: &[ReviewInput],
    batch_size: usize,
    num_threads: Option<usize>,
    input_parse_ns: u128,
    scope_validation_ns: u128,
) -> PyResult<(Vec<f64>, FastPredictCallerProfile)> {
    if batch_size < 1 {
        return Err(py_value_error("batch_size must be at least 1"));
    }
    validate_num_threads(num_threads)?;

    let total_start = Instant::now();
    let mut profile = FastPredictCallerProfile {
        review_count: inputs.len(),
        input_parse_ns,
        scope_validation_ns,
        ..FastPredictCallerProfile::default()
    };

    let materialization_start = Instant::now();
    rnn.materialize_flat_cpu_state().map_err(py_value_error)?;
    profile.flat_state_materialization_ns = materialization_start.elapsed().as_nanos();

    let prepare_start = Instant::now();
    let mut prepared_rows = Vec::with_capacity(inputs.len());
    let mut prepared_ids = Vec::with_capacity(inputs.len());
    let mut all_batchable = true;
    for input in inputs {
        let row = deterministic.prepare_predict_row(input);
        let ids = review_ids_from_prepared(&row).map_err(py_value_error)?;
        all_batchable &= deterministic
            .can_batch_predict(&row)
            .map_err(py_value_error)?;
        prepared_rows.push(row);
        prepared_ids.push(ids);
    }
    profile.prepare_predict_ns = prepare_start.elapsed().as_nanos();

    if prepared_rows.is_empty() {
        profile.execution_path = "empty";
        profile.total_ns = total_start.elapsed().as_nanos() + input_parse_ns + scope_validation_ns;
        return Ok((Vec::new(), profile));
    }

    let uses_parallel_batches = all_batchable
        && prepared_rows.len() >= PREDICT_MANY_LIGHTNING_PARALLEL_MIN_ROWS
        && prepared_rows.len() > batch_size
        && num_threads != Some(1)
        && predict_many_lightning_parallel_batches_enabled();
    if !uses_parallel_batches {
        profile.execution_path = "sequential_fallback";
        let prediction_start = Instant::now();
        let probabilities = predict_prepared_reviews_with_state_sequential(
            rnn,
            deterministic,
            prepared_rows,
            prepared_ids,
            batch_size,
            num_threads,
        )?;
        profile.prediction_wall_ns = prediction_start.elapsed().as_nanos();
        profile.total_ns = total_start.elapsed().as_nanos() + input_parse_ns + scope_validation_ns;
        return Ok((probabilities, profile));
    }

    profile.execution_path = "parallel_batches";
    let ranges = (0..prepared_rows.len())
        .step_by(batch_size)
        .map(|start| (start, (start + batch_size).min(prepared_rows.len())))
        .collect::<Vec<_>>();
    let rnn_ref: &NativeRnn = rnn;
    let deterministic_ref: &FeatureState = deterministic;
    let prediction_start = Instant::now();
    let batches = with_rayon_threads(num_threads, || {
        ranges
            .par_iter()
            .map(|&(start, end)| {
                let mut batch_profile = FastPredictBatchStageProfile::default();
                let probabilities = predict_prepared_batch_lightning_staged(
                    rnn_ref,
                    deterministic_ref,
                    &prepared_rows[start..end],
                    &prepared_ids[start..end],
                    &mut batch_profile,
                )?;
                Ok((end - start, probabilities, batch_profile))
            })
            .collect::<PyResult<Vec<_>>>()
    })?;
    profile.prediction_wall_ns = prediction_start.elapsed().as_nanos();

    let mut probabilities = Vec::with_capacity(prepared_rows.len());
    for (rows, batch_probabilities, batch_profile) in batches {
        profile.record_batch(rows, &batch_profile);
        probabilities.extend(batch_probabilities);
    }
    profile.total_ns = total_start.elapsed().as_nanos() + input_parse_ns + scope_validation_ns;
    Ok((probabilities, profile))
}

pub(super) fn predict_reviews_gpu_with_state(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    loaded_scope: Option<&super::checkpoint_bin::CheckpointScope>,
    reviews: &Bound<'_, PyAny>,
    batch_size: usize,
    num_threads: Option<usize>,
) -> PyResult<Vec<f64>> {
    let mut inputs = Vec::new();
    let mut input_parse_ns = 0u128;
    for review in reviews.iter()? {
        let start = Instant::now();
        let review = review?;
        let input = if let Ok(review) = review.downcast::<PyDict>() {
            predict_review_from_dict(review)?
        } else {
            predict_review_from_mapping(review.downcast::<PyMapping>()?)?
        };
        input_parse_ns += start.elapsed().as_nanos();
        inputs.push(input);
    }
    predict_review_inputs_gpu(
        rnn,
        deterministic,
        loaded_scope,
        &inputs,
        batch_size,
        num_threads,
        input_parse_ns,
    )
}

/// Predict native review inputs through the same GPU implementation used by
/// the public mapping-based `predict_many()` binding.
#[allow(clippy::too_many_arguments)]
pub(super) fn predict_review_inputs_gpu(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    loaded_scope: Option<&super::checkpoint_bin::CheckpointScope>,
    inputs: &[ReviewInput],
    batch_size: usize,
    num_threads: Option<usize>,
    input_parse_ns: u128,
) -> PyResult<Vec<f64>> {
    if batch_size < 1 {
        return Err(py_value_error("batch_size must be at least 1"));
    }
    validate_num_threads(num_threads)?;
    rnn.reset_gpu_prediction_pipeline_last();
    if std::env::var("RWKV_SRS_GPU_PARALLEL_FEATURES").as_deref() != Ok("0") {
        return predict_review_inputs_gpu_parallel_features(
            rnn,
            deterministic,
            loaded_scope,
            inputs,
            batch_size,
            num_threads,
            input_parse_ns,
        );
    }
    let mut probabilities = Vec::new();
    let mut batch_features = Vec::with_capacity(batch_size);
    let mut batch_ids = Vec::with_capacity(batch_size);
    let mut feature_build_ns = 0u128;
    let mut fallback_rows = 0u64;

    let scope_validation_start = Instant::now();
    if let Some(scope) = loaded_scope {
        for input in inputs {
            require_loaded_scope(scope, input)?;
        }
    }
    let scope_validation_ns = scope_validation_start.elapsed().as_nanos();

    for input in inputs {
        let start = Instant::now();
        let direct = deterministic
            .direct_predict_features(input)
            .map_err(py_value_error)?;
        feature_build_ns += start.elapsed().as_nanos();
        if let Some((features, ids)) = direct {
            batch_features.push(features);
            batch_ids.push(ids);
            if batch_features.len() == batch_size {
                flush_gpu_batch(rnn, &mut batch_features, &mut batch_ids, &mut probabilities)?;
            }
        } else {
            fallback_rows += 1;
            flush_gpu_batch(rnn, &mut batch_features, &mut batch_ids, &mut probabilities)?;
            let row = deterministic.prepare_predict_row(input);
            let ids = review_ids_from_prepared(&row).map_err(py_value_error)?;
            let mut profile = None;
            probabilities.push(predict_prepared_row_probability(
                rnn,
                deterministic,
                &row,
                ids,
                PredictManyForwardMode::Oracle,
                num_threads,
                &mut profile,
            )?);
        }
    }
    flush_gpu_batch(rnn, &mut batch_features, &mut batch_ids, &mut probabilities)?;
    rnn.record_gpu_host_preparation(
        input_parse_ns,
        scope_validation_ns,
        feature_build_ns,
        fallback_rows,
    );
    Ok(probabilities)
}

fn predict_review_inputs_gpu_parallel_features(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    loaded_scope: Option<&super::checkpoint_bin::CheckpointScope>,
    review_inputs: &[ReviewInput],
    batch_size: usize,
    num_threads: Option<usize>,
    input_parse_ns: u128,
) -> PyResult<Vec<f64>> {
    if review_inputs.len() > batch_size
        && super::gpu::gpu_prediction_pipeline_enabled().map_err(py_value_error)?
    {
        return predict_review_inputs_gpu_parallel_features_pipelined(
            rnn,
            deterministic,
            loaded_scope,
            review_inputs,
            batch_size,
            num_threads,
            input_parse_ns,
        );
    }
    let mut inputs = Vec::with_capacity(batch_size);
    let mut probabilities = Vec::new();
    let mut batch_features = Vec::with_capacity(batch_size);
    let mut batch_ids = Vec::with_capacity(batch_size);
    let mut feature_build_ns = 0u128;
    let mut fallback_rows = 0u64;

    let scope_validation_start = Instant::now();
    if let Some(scope) = loaded_scope {
        for input in review_inputs {
            require_loaded_scope(scope, input)?;
        }
    }
    let scope_validation_ns = scope_validation_start.elapsed().as_nanos();

    for input in review_inputs {
        inputs.push(input.clone());
        if inputs.len() == batch_size {
            consume_gpu_input_chunk(
                rnn,
                deterministic,
                &mut inputs,
                &mut batch_features,
                &mut batch_ids,
                &mut probabilities,
                batch_size,
                num_threads,
                &mut feature_build_ns,
                &mut fallback_rows,
            )?;
        }
    }
    consume_gpu_input_chunk(
        rnn,
        deterministic,
        &mut inputs,
        &mut batch_features,
        &mut batch_ids,
        &mut probabilities,
        batch_size,
        num_threads,
        &mut feature_build_ns,
        &mut fallback_rows,
    )?;
    flush_gpu_batch(rnn, &mut batch_features, &mut batch_ids, &mut probabilities)?;
    rnn.record_gpu_host_preparation(
        input_parse_ns,
        scope_validation_ns,
        feature_build_ns,
        fallback_rows,
    );
    Ok(probabilities)
}

#[allow(clippy::too_many_arguments)]
fn predict_review_inputs_gpu_parallel_features_pipelined(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    loaded_scope: Option<&super::checkpoint_bin::CheckpointScope>,
    review_inputs: &[ReviewInput],
    batch_size: usize,
    num_threads: Option<usize>,
    input_parse_ns: u128,
) -> PyResult<Vec<f64>> {
    let scope_validation_start = Instant::now();
    if let Some(scope) = loaded_scope {
        for input in review_inputs {
            require_loaded_scope(scope, input)?;
        }
    }
    let scope_validation_ns = scope_validation_start.elapsed().as_nanos();

    rnn.begin_gpu_prediction_pipeline().map_err(py_gpu_error)?;
    let pipeline_start = Instant::now();
    let mut feature_build_ns = 0u128;
    let mut fallback_rows = 0u64;
    let result = (|| {
        let mut inputs = Vec::with_capacity(batch_size);
        let mut probabilities = Vec::with_capacity(review_inputs.len());
        let mut batch_features = Vec::with_capacity(batch_size);
        let mut batch_ids = Vec::with_capacity(batch_size);
        let mut pending = VecDeque::with_capacity(2);
        let mut pending_bytes = 0u64;

        for input in review_inputs {
            inputs.push(input.clone());
            if inputs.len() == batch_size {
                consume_gpu_input_chunk_pipelined(
                    rnn,
                    deterministic,
                    &mut inputs,
                    &mut batch_features,
                    &mut batch_ids,
                    &mut pending,
                    &mut pending_bytes,
                    &mut probabilities,
                    batch_size,
                    num_threads,
                    &mut feature_build_ns,
                    &mut fallback_rows,
                )?;
            }
        }
        consume_gpu_input_chunk_pipelined(
            rnn,
            deterministic,
            &mut inputs,
            &mut batch_features,
            &mut batch_ids,
            &mut pending,
            &mut pending_bytes,
            &mut probabilities,
            batch_size,
            num_threads,
            &mut feature_build_ns,
            &mut fallback_rows,
        )?;
        submit_gpu_prediction_pipeline_batch(
            rnn,
            &mut batch_features,
            &mut batch_ids,
            &mut pending,
            &mut pending_bytes,
            &mut probabilities,
        )?;
        drain_gpu_prediction_pipeline(rnn, &mut pending, &mut pending_bytes, &mut probabilities)?;
        Ok(probabilities)
    })();
    rnn.finish_gpu_prediction_pipeline(pipeline_start.elapsed().as_nanos(), result.is_ok());
    if result.is_ok() {
        rnn.record_gpu_host_preparation(
            input_parse_ns,
            scope_validation_ns,
            feature_build_ns,
            fallback_rows,
        );
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn consume_gpu_input_chunk_pipelined(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    inputs: &mut Vec<ReviewInput>,
    batch_features: &mut Vec<Vec<f32>>,
    batch_ids: &mut Vec<ReviewIds>,
    pending: &mut VecDeque<super::gpu::PendingGpuPrediction>,
    pending_bytes: &mut u64,
    probabilities: &mut Vec<f64>,
    batch_size: usize,
    num_threads: Option<usize>,
    feature_build_ns: &mut u128,
    fallback_rows: &mut u64,
) -> PyResult<()> {
    if inputs.is_empty() {
        return Ok(());
    }
    let feature_build_overlaps_gpu = !pending.is_empty();
    let start = Instant::now();
    let direct_results = if inputs.len() >= 256 && num_threads != Some(1) {
        with_rayon_threads(num_threads, || {
            inputs
                .par_iter()
                .map(|input| deterministic.direct_predict_features(input))
                .collect::<std::result::Result<Vec<_>, String>>()
                .map_err(py_value_error)
        })
    } else {
        inputs
            .iter()
            .map(|input| deterministic.direct_predict_features(input))
            .collect::<std::result::Result<Vec<_>, String>>()
            .map_err(py_value_error)
    };
    let direct_results = match direct_results {
        Ok(results) => results,
        Err(error) => {
            if let Some(cleanup_error) =
                discard_gpu_prediction_pipeline(rnn, pending, pending_bytes)
            {
                return Err(py_gpu_error(cleanup_error.context(format!(
                    "while draining queued work after feature construction failed: {error}"
                ))));
            }
            return Err(error);
        }
    };
    let elapsed_ns = start.elapsed().as_nanos();
    *feature_build_ns += elapsed_ns;
    if feature_build_overlaps_gpu {
        rnn.record_gpu_prediction_pipeline_overlap(elapsed_ns);
    }
    for (input, direct) in inputs.drain(..).zip(direct_results) {
        if let Some((features, ids)) = direct {
            batch_features.push(features);
            batch_ids.push(ids);
            if batch_features.len() == batch_size {
                submit_gpu_prediction_pipeline_batch(
                    rnn,
                    batch_features,
                    batch_ids,
                    pending,
                    pending_bytes,
                    probabilities,
                )?;
            }
        } else {
            *fallback_rows += 1;
            submit_gpu_prediction_pipeline_batch(
                rnn,
                batch_features,
                batch_ids,
                pending,
                pending_bytes,
                probabilities,
            )?;
            drain_gpu_prediction_pipeline(rnn, pending, pending_bytes, probabilities)?;
            rnn.record_gpu_prediction_pipeline_fallback();
            let row = deterministic.prepare_predict_row(&input);
            let ids = review_ids_from_prepared(&row).map_err(py_value_error)?;
            let mut profile = None;
            probabilities.push(predict_prepared_row_probability(
                rnn,
                deterministic,
                &row,
                ids,
                PredictManyForwardMode::Oracle,
                num_threads,
                &mut profile,
            )?);
        }
    }
    Ok(())
}

fn submit_gpu_prediction_pipeline_batch(
    rnn: &mut NativeRnn,
    batch_features: &mut Vec<Vec<f32>>,
    batch_ids: &mut Vec<ReviewIds>,
    pending: &mut VecDeque<super::gpu::PendingGpuPrediction>,
    pending_bytes: &mut u64,
    probabilities: &mut Vec<f64>,
) -> PyResult<()> {
    if batch_features.is_empty() {
        return Ok(());
    }
    let submission_overlaps_gpu = !pending.is_empty();
    let submission_start = Instant::now();
    let submission = rnn.submit_gpu_prediction(batch_features, batch_ids);
    let submission_ns = submission_start.elapsed().as_nanos();
    if submission_overlaps_gpu {
        rnn.record_gpu_prediction_pipeline_overlap(submission_ns);
    }
    match submission {
        Ok(submitted) => {
            *pending_bytes += submitted.transient_bytes();
            rnn.record_gpu_prediction_pipeline_high_water(*pending_bytes);
            pending.push_back(submitted);
            batch_features.clear();
            batch_ids.clear();
            if pending.len() == 2 {
                collect_oldest_gpu_prediction(rnn, pending, pending_bytes, probabilities)?;
            }
            Ok(())
        }
        Err(error) if is_gpu_out_of_memory(&error) && batch_features.len() > 1 => {
            // The failed chunk was not submitted: resource-allocation errors
            // are captured before queue submission. Drain older work before
            // invoking the existing recursive splitter so only one adaptive
            // branch owns transient prediction resources at a time.
            drain_gpu_prediction_pipeline(rnn, pending, pending_bytes, probabilities)?;
            rnn.record_gpu_prediction_pipeline_fallback();
            rnn.recover_gpu_prediction_oom().map_err(py_gpu_error)?;
            predict_gpu_adaptive(rnn, batch_features, batch_ids, probabilities)?;
            batch_features.clear();
            batch_ids.clear();
            Ok(())
        }
        Err(error) => {
            let cleanup = discard_gpu_prediction_pipeline(rnn, pending, pending_bytes);
            let error = if let Some(cleanup_error) = cleanup {
                error.context(format!(
                    "also failed to drain a queued GPU prediction: {cleanup_error:#}"
                ))
            } else {
                error
            };
            Err(py_gpu_error(error))
        }
    }
}

fn collect_oldest_gpu_prediction(
    rnn: &mut NativeRnn,
    pending: &mut VecDeque<super::gpu::PendingGpuPrediction>,
    pending_bytes: &mut u64,
    probabilities: &mut Vec<f64>,
) -> PyResult<()> {
    let submitted = pending
        .pop_front()
        .expect("prediction pipeline collection requires a pending chunk");
    *pending_bytes -= submitted.transient_bytes();
    match rnn.collect_gpu_prediction(submitted) {
        Ok(values) => {
            probabilities.extend(values);
            Ok(())
        }
        Err(error) => {
            let cleanup = discard_gpu_prediction_pipeline(rnn, pending, pending_bytes);
            let error = if let Some(cleanup_error) = cleanup {
                error.context(format!(
                    "also failed to drain a later GPU prediction: {cleanup_error:#}"
                ))
            } else {
                error
            };
            Err(py_gpu_error(error))
        }
    }
}

fn drain_gpu_prediction_pipeline(
    rnn: &mut NativeRnn,
    pending: &mut VecDeque<super::gpu::PendingGpuPrediction>,
    pending_bytes: &mut u64,
    probabilities: &mut Vec<f64>,
) -> PyResult<()> {
    while !pending.is_empty() {
        collect_oldest_gpu_prediction(rnn, pending, pending_bytes, probabilities)?;
    }
    Ok(())
}

fn discard_gpu_prediction_pipeline(
    rnn: &mut NativeRnn,
    pending: &mut VecDeque<super::gpu::PendingGpuPrediction>,
    pending_bytes: &mut u64,
) -> Option<anyhow::Error> {
    let mut first_error = None;
    while let Some(submitted) = pending.pop_front() {
        *pending_bytes -= submitted.transient_bytes();
        if let Err(error) = rnn.collect_gpu_prediction(submitted) {
            first_error.get_or_insert(error);
        }
    }
    first_error
}

#[allow(clippy::too_many_arguments)]
fn consume_gpu_input_chunk(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    inputs: &mut Vec<ReviewInput>,
    batch_features: &mut Vec<Vec<f32>>,
    batch_ids: &mut Vec<ReviewIds>,
    probabilities: &mut Vec<f64>,
    batch_size: usize,
    num_threads: Option<usize>,
    feature_build_ns: &mut u128,
    fallback_rows: &mut u64,
) -> PyResult<()> {
    if inputs.is_empty() {
        return Ok(());
    }
    let start = Instant::now();
    let direct_results = if inputs.len() >= 256 && num_threads != Some(1) {
        with_rayon_threads(num_threads, || {
            inputs
                .par_iter()
                .map(|input| deterministic.direct_predict_features(input))
                .collect::<std::result::Result<Vec<_>, String>>()
                .map_err(py_value_error)
        })?
    } else {
        inputs
            .iter()
            .map(|input| deterministic.direct_predict_features(input))
            .collect::<std::result::Result<Vec<_>, String>>()
            .map_err(py_value_error)?
    };
    *feature_build_ns += start.elapsed().as_nanos();
    for (input, direct) in inputs.drain(..).zip(direct_results) {
        if let Some((features, ids)) = direct {
            batch_features.push(features);
            batch_ids.push(ids);
            if batch_features.len() == batch_size {
                flush_gpu_batch(rnn, batch_features, batch_ids, probabilities)?;
            }
        } else {
            *fallback_rows += 1;
            flush_gpu_batch(rnn, batch_features, batch_ids, probabilities)?;
            let row = deterministic.prepare_predict_row(&input);
            let ids = review_ids_from_prepared(&row).map_err(py_value_error)?;
            let mut profile = None;
            probabilities.push(predict_prepared_row_probability(
                rnn,
                deterministic,
                &row,
                ids,
                PredictManyForwardMode::Oracle,
                num_threads,
                &mut profile,
            )?);
        }
    }
    Ok(())
}

pub(super) fn require_loaded_scope(
    scope: &super::checkpoint_bin::CheckpointScope,
    input: &ReviewInput,
) -> PyResult<()> {
    let (card_id, note_id, deck_id, preset_id) = FeatureState::normalized_review_ids(input);
    let mut unavailable = Vec::new();
    for (field, identity, available) in [
        ("card_id", card_id, &scope.card_ids),
        ("note_id", note_id, &scope.note_ids),
        ("deck_id", deck_id, &scope.deck_ids),
        ("preset_id", preset_id, &scope.preset_ids),
    ] {
        if !available.contains(&identity) {
            unavailable.push(format!("{field}={identity}"));
        }
    }
    if unavailable.is_empty() {
        return Ok(());
    }
    Err(py_value_error(format!(
        "Review identity is outside the selectively loaded checkpoint scope: {}. Reload the checkpoint with this card included or omit cards= to load the full state.",
        unavailable.join(", ")
    )))
}

fn flush_gpu_batch(
    rnn: &mut NativeRnn,
    batch_features: &mut Vec<Vec<f32>>,
    batch_ids: &mut Vec<ReviewIds>,
    probabilities: &mut Vec<f64>,
) -> PyResult<()> {
    if batch_features.is_empty() {
        return Ok(());
    }
    predict_gpu_adaptive(rnn, batch_features, batch_ids, probabilities)?;
    batch_features.clear();
    batch_ids.clear();
    Ok(())
}

fn predict_gpu_adaptive(
    rnn: &mut NativeRnn,
    features: &[Vec<f32>],
    ids: &[ReviewIds],
    probabilities: &mut Vec<f64>,
) -> PyResult<()> {
    match rnn.predict_gpu(features, ids) {
        Ok(values) => {
            if values.len() != features.len() {
                return Err(py_value_error(format!(
                    "GPU predict returned {} probabilities for {} rows",
                    values.len(),
                    features.len()
                )));
            }
            probabilities.extend(values);
            Ok(())
        }
        Err(error) if is_gpu_out_of_memory(&error) && features.len() > 1 => {
            rnn.recover_gpu_prediction_oom().map_err(py_gpu_error)?;
            let midpoint = features.len() / 2;
            predict_gpu_adaptive(rnn, &features[..midpoint], &ids[..midpoint], probabilities)?;
            predict_gpu_adaptive(rnn, &features[midpoint..], &ids[midpoint..], probabilities)
        }
        Err(error) => Err(py_gpu_error(error)),
    }
}

fn predict_reviews_with_state_parallel_batches(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    inputs: &[ReviewInput],
    batch_size: usize,
    num_threads: Option<usize>,
) -> PyResult<Vec<f64>> {
    let mut prepared_rows = Vec::new();
    let mut prepared_ids = Vec::new();
    let mut all_batchable = true;
    for input in inputs {
        let row = deterministic.prepare_predict_row(input);
        let ids = review_ids_from_prepared(&row).map_err(py_value_error)?;
        all_batchable &= deterministic
            .can_batch_predict(&row)
            .map_err(py_value_error)?;
        prepared_rows.push(row);
        prepared_ids.push(ids);
    }

    if prepared_rows.is_empty() {
        return Ok(Vec::new());
    }
    if !all_batchable
        || prepared_rows.len() < PREDICT_MANY_LIGHTNING_PARALLEL_MIN_ROWS
        || prepared_rows.len() <= batch_size
        || num_threads == Some(1)
    {
        return predict_prepared_reviews_with_state_sequential(
            rnn,
            deterministic,
            prepared_rows,
            prepared_ids,
            batch_size,
            num_threads,
        );
    }

    let ranges = (0..prepared_rows.len())
        .step_by(batch_size)
        .map(|start| (start, (start + batch_size).min(prepared_rows.len())))
        .collect::<Vec<_>>();
    let rnn_ref: &NativeRnn = rnn;
    let deterministic_ref: &FeatureState = deterministic;
    let batches = with_rayon_threads(num_threads, || {
        ranges
            .par_iter()
            .map(|&(start, end)| {
                predict_prepared_batch_lightning(
                    rnn_ref,
                    deterministic_ref,
                    &prepared_rows[start..end],
                    &prepared_ids[start..end],
                )
            })
            .collect::<PyResult<Vec<_>>>()
    })?;

    Ok(batches.into_iter().flatten().collect())
}

fn predict_prepared_batch_lightning(
    rnn: &NativeRnn,
    deterministic: &FeatureState,
    batch_rows: &[PreparedRow],
    batch_ids: &[ReviewIds],
) -> PyResult<Vec<f64>> {
    let predict_features = deterministic
        .feature_vectors(batch_rows)
        .map_err(py_value_error)?;
    let predict_features =
        tensor_from_2d(predict_features, "predict_features").map_err(py_value_error)?;
    let predict_p_logits = if predict_many_lightning_modulewise_state_pack_enabled() {
        with_fast_layer_norm(batch_rows.len() > 1, || {
            srs_review_predict_forward_lightning_modulewise(
                &rnn.weights,
                &predict_features,
                |module_index| rnn.batch_module_state_inputs(batch_ids, module_index, true),
            )
        })
        .map_err(py_value_error)?
    } else {
        let state_inputs = rnn
            .batch_state_inputs(batch_ids, true, None)
            .map_err(py_value_error)?;
        with_fast_layer_norm(batch_rows.len() > 1, || {
            srs_review_predict_forward_lightning_profiled(
                &rnn.weights,
                &predict_features,
                Some(state_inputs.0.as_slice()),
                Some(state_inputs.1.as_slice()),
                Some(state_inputs.2.as_slice()),
                None,
            )
        })
        .map_err(py_value_error)?
    };
    let probabilities = probabilities_from_logits(&predict_p_logits).map_err(py_value_error)?;
    if probabilities.len() != batch_rows.len() {
        return Err(py_value_error(format!(
            "batched predict returned {} probabilities for {} rows",
            probabilities.len(),
            batch_rows.len()
        )));
    }
    Ok(probabilities)
}

fn predict_prepared_batch_lightning_staged(
    rnn: &NativeRnn,
    deterministic: &FeatureState,
    batch_rows: &[PreparedRow],
    batch_ids: &[ReviewIds],
    profile: &mut FastPredictBatchStageProfile,
) -> PyResult<Vec<f64>> {
    // This diagnostic mirrors `predict_prepared_batch_lightning` while adding
    // timers. Keeping it separate leaves the production hot loop free of
    // profiling branches; the Python parity test requires bit-exact output
    // from both implementations so future Fast-path changes cannot silently
    // leave this benchmark on different arithmetic.
    let feature_start = Instant::now();
    let predict_features = deterministic
        .feature_vectors(batch_rows)
        .map_err(py_value_error)?;
    let predict_features =
        tensor_from_2d(predict_features, "predict_features").map_err(py_value_error)?;
    profile.feature_preparation_ns = feature_start.elapsed().as_nanos();

    let predict_p_logits = if predict_many_lightning_modulewise_state_pack_enabled() {
        let forward_start = Instant::now();
        let mut state_lookup_packing_ns = 0u128;
        let logits = with_fast_layer_norm(batch_rows.len() > 1, || {
            srs_review_predict_forward_lightning_modulewise(
                &rnn.weights,
                &predict_features,
                |module_index| {
                    let state_start = Instant::now();
                    let state = rnn.batch_module_state_inputs(batch_ids, module_index, true);
                    state_lookup_packing_ns += state_start.elapsed().as_nanos();
                    state
                },
            )
        })
        .map_err(py_value_error)?;
        profile.model_forward_inclusive_ns = forward_start.elapsed().as_nanos();
        profile.state_lookup_packing_ns = state_lookup_packing_ns;
        profile.model_forward_ns = profile
            .model_forward_inclusive_ns
            .saturating_sub(state_lookup_packing_ns);
        logits
    } else {
        let state_start = Instant::now();
        let state_inputs = rnn
            .batch_state_inputs(batch_ids, true, None)
            .map_err(py_value_error)?;
        profile.state_lookup_packing_ns = state_start.elapsed().as_nanos();

        let forward_start = Instant::now();
        let logits = with_fast_layer_norm(batch_rows.len() > 1, || {
            srs_review_predict_forward_lightning_profiled(
                &rnn.weights,
                &predict_features,
                Some(state_inputs.0.as_slice()),
                Some(state_inputs.1.as_slice()),
                Some(state_inputs.2.as_slice()),
                None,
            )
        })
        .map_err(py_value_error)?;
        profile.model_forward_inclusive_ns = forward_start.elapsed().as_nanos();
        profile.model_forward_ns = profile.model_forward_inclusive_ns;
        logits
    };

    let result_start = Instant::now();
    let probabilities = probabilities_from_logits(&predict_p_logits).map_err(py_value_error)?;
    profile.result_construction_ns = result_start.elapsed().as_nanos();
    if probabilities.len() != batch_rows.len() {
        return Err(py_value_error(format!(
            "batched predict returned {} probabilities for {} rows",
            probabilities.len(),
            batch_rows.len()
        )));
    }
    Ok(probabilities)
}

fn predict_prepared_reviews_with_state_sequential(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    prepared_rows: Vec<PreparedRow>,
    prepared_ids: Vec<ReviewIds>,
    batch_size: usize,
    num_threads: Option<usize>,
) -> PyResult<Vec<f64>> {
    let mut prediction_probabilities = Vec::with_capacity(prepared_rows.len());
    let mut batch_rows = Vec::with_capacity(batch_size);
    let mut batch_ids = Vec::with_capacity(batch_size);
    let mut profile = None;
    for (predict_row, ids) in prepared_rows.into_iter().zip(prepared_ids) {
        if deterministic
            .can_batch_predict(&predict_row)
            .map_err(py_value_error)?
        {
            batch_rows.push(predict_row);
            batch_ids.push(ids);
            if batch_rows.len() >= batch_size {
                flush_predict_batch(
                    rnn,
                    deterministic,
                    &mut batch_rows,
                    &mut batch_ids,
                    &mut prediction_probabilities,
                    true,
                    PredictManyForwardMode::Lightning,
                    num_threads,
                    &mut profile,
                )?;
            }
        } else {
            flush_predict_batch(
                rnn,
                deterministic,
                &mut batch_rows,
                &mut batch_ids,
                &mut prediction_probabilities,
                true,
                PredictManyForwardMode::Lightning,
                num_threads,
                &mut profile,
            )?;
            prediction_probabilities.push(predict_prepared_row_probability(
                rnn,
                deterministic,
                &predict_row,
                ids,
                PredictManyForwardMode::Lightning,
                num_threads,
                &mut profile,
            )?);
        }
    }
    flush_predict_batch(
        rnn,
        deterministic,
        &mut batch_rows,
        &mut batch_ids,
        &mut prediction_probabilities,
        true,
        PredictManyForwardMode::Lightning,
        num_threads,
        &mut profile,
    )?;
    Ok(prediction_probabilities)
}

pub(super) fn predict_probability_with_state(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    review: &Bound<'_, PyMapping>,
    num_threads: Option<usize>,
) -> PyResult<f64> {
    validate_num_threads(num_threads)?;
    rnn.materialize_flat_cpu_state().map_err(py_value_error)?;
    let input = review_from_mapping(review, false)?;
    let predict_row = deterministic.prepare_predict_row(&input);
    let ids = review_ids_from_prepared(&predict_row).map_err(py_value_error)?;
    let mut profile = None;
    predict_prepared_row_probability(
        rnn,
        deterministic,
        &predict_row,
        ids,
        PredictManyForwardMode::Oracle,
        num_threads,
        &mut profile,
    )
}

pub(super) fn warm_rayon_pool(num_threads: usize) -> PyResult<()> {
    validate_num_threads(Some(num_threads))?;
    let _pool = rayon_pool(num_threads)?;
    Ok(())
}

pub(super) fn probability_from_logits(logits: &Tensor) -> Result<f64> {
    let probabilities = probabilities_from_logits(logits)?;
    if probabilities.is_empty() {
        bail!("prediction logits have no rows");
    }
    Ok(probabilities[0])
}

pub(super) fn probabilities_from_logits(logits: &Tensor) -> Result<Vec<f64>> {
    let probabilities = nn_ops::softmax(logits, D::Minus1)?;
    let probabilities = probabilities.to_vec2::<f32>()?;
    probabilities
        .into_iter()
        .map(|row| {
            let failure_probability = row.first().ok_or_else(|| {
                candle_core::Error::msg("prediction logits have no class dimension")
            })?;
            Ok(f64::from(1.0 - *failure_probability))
        })
        .collect()
}

fn flush_predict_batch(
    rnn: &mut NativeRnn,
    deterministic: &FeatureState,
    batch_rows: &mut Vec<PreparedRow>,
    batch_ids: &mut Vec<ReviewIds>,
    prediction_probabilities: &mut Vec<f64>,
    allow_short_burst_broadcast: bool,
    forward_mode: PredictManyForwardMode,
    num_threads: Option<usize>,
    profile: &mut Option<&mut RuntimeProfile>,
) -> PyResult<()> {
    if batch_rows.is_empty() {
        return Ok(());
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.batch.record_flush(batch_ids);
    }

    let start = ProfileTimer::start(profile.is_some());
    let predict_features = deterministic
        .feature_vectors(batch_rows)
        .map_err(py_value_error)?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.predict_feature_ns += start.elapsed_ns();
        profile.materialization.predict_feature_vecs += predict_features.len();
        profile.materialization.predict_feature_vec_values +=
            predict_features.iter().map(Vec::len).sum::<usize>();
    }

    let start = ProfileTimer::start(profile.is_some());
    let predict_feature_values = tensor2_list_value_count(&predict_features);
    let predict_features =
        tensor_from_2d(predict_features, "predict_features").map_err(py_value_error)?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.predict_tensor_ns += start.elapsed_ns();
        profile.materialization.feature_tensors += 1;
        profile.materialization.feature_tensor_values += predict_feature_values;
    }

    let state_inputs = if forward_mode.requires_state().map_err(py_value_error)? {
        let start = ProfileTimer::start(profile.is_some());
        let state_inputs = rnn
            .batch_state_inputs(
                batch_ids,
                allow_short_burst_broadcast,
                profile
                    .as_deref_mut()
                    .map(|profile| &mut profile.state_input),
            )
            .map_err(py_value_error)?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.materialization.state_input_ns += start.elapsed_ns();
            profile.materialization.state_input_snapshots += 1;
            profile.materialization.state_input_tensor_handles +=
                state_input_tensor_count(&state_inputs);
        }
        Some(state_inputs)
    } else {
        None
    };
    let (time_x_shift_b1c_by_module, time_state_b1hkk_by_module, channel_state_b1c_by_module) =
        match state_inputs.as_ref() {
            Some((time_x_shift, time_state, channel_state)) => (
                Some(time_x_shift.as_slice()),
                Some(time_state.as_slice()),
                Some(channel_state.as_slice()),
            ),
            None => (None, None, None),
        };

    let start = ProfileTimer::start(profile.is_some());
    let predict_p_logits = with_rayon_threads(num_threads, || {
        with_fast_layer_norm(batch_rows.len() > 1, || {
            if forward_mode.is_lightning() {
                srs_review_predict_forward_lightning_profiled(
                    &rnn.weights,
                    &predict_features,
                    time_x_shift_b1c_by_module,
                    time_state_b1hkk_by_module,
                    channel_state_b1c_by_module,
                    profile
                        .as_deref_mut()
                        .map(|profile| &mut profile.predict_forward),
                )
            } else {
                srs_review_predict_forward_profiled(
                    &rnn.weights,
                    &predict_features,
                    time_x_shift_b1c_by_module,
                    time_state_b1hkk_by_module,
                    channel_state_b1c_by_module,
                    profile
                        .as_deref_mut()
                        .map(|profile| &mut profile.predict_forward),
                )
            }
        })
        .map_err(py_value_error)
    })?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.predict_forward_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    let probabilities = probabilities_from_logits(&predict_p_logits).map_err(py_value_error)?;
    if probabilities.len() != batch_rows.len() {
        return Err(py_value_error(format!(
            "batched predict returned {} probabilities for {} rows",
            probabilities.len(),
            batch_rows.len()
        )));
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.predict_output_ns += start.elapsed_ns();
        profile.materialization.prediction_probability_values += probabilities.len();
    }
    prediction_probabilities.extend(probabilities);
    batch_rows.clear();
    batch_ids.clear();
    Ok(())
}

fn predict_prepared_row_probability(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    predict_row: &PreparedRow,
    ids: ReviewIds,
    forward_mode: PredictManyForwardMode,
    num_threads: Option<usize>,
    profile: &mut Option<&mut RuntimeProfile>,
) -> PyResult<f64> {
    let start = ProfileTimer::start(profile.is_some());
    let predict_features = deterministic
        .predict_feature_vector(predict_row)
        .map_err(py_value_error)?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.predict_feature_ns += start.elapsed_ns();
        profile.materialization.predict_feature_vecs += 1;
        profile.materialization.predict_feature_vec_values += predict_features.len();
    }

    let start = ProfileTimer::start(profile.is_some());
    let predict_feature_values = predict_features.len();
    let predict_features =
        tensor_from_feature_vector(predict_features, "predict_features").map_err(py_value_error)?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.predict_tensor_ns += start.elapsed_ns();
        profile.materialization.feature_tensors += 1;
        profile.materialization.feature_tensor_values += predict_feature_values;
    }

    let state_inputs = if forward_mode.requires_state().map_err(py_value_error)? {
        let (card_id, note_id, deck_id, preset_id) = ids;
        let start = ProfileTimer::start(profile.is_some());
        let state_inputs = rnn.state_inputs(card_id, note_id, deck_id, preset_id);
        if let Some(profile) = profile.as_deref_mut() {
            profile.materialization.state_input_ns += start.elapsed_ns();
            profile.materialization.state_input_snapshots += 1;
            profile.materialization.state_input_tensor_handles +=
                state_input_tensor_count(&state_inputs);
        }
        Some(state_inputs)
    } else {
        None
    };
    let (time_x_shift_b1c_by_module, time_state_b1hkk_by_module, channel_state_b1c_by_module) =
        match state_inputs.as_ref() {
            Some((time_x_shift, time_state, channel_state)) => (
                Some(time_x_shift.as_slice()),
                Some(time_state.as_slice()),
                Some(channel_state.as_slice()),
            ),
            None => (None, None, None),
        };

    let start = ProfileTimer::start(profile.is_some());
    let predict_p_logits = with_rayon_threads(num_threads, || {
        if forward_mode.is_lightning() {
            srs_review_predict_forward_lightning_profiled(
                &rnn.weights,
                &predict_features,
                time_x_shift_b1c_by_module,
                time_state_b1hkk_by_module,
                channel_state_b1c_by_module,
                profile
                    .as_deref_mut()
                    .map(|profile| &mut profile.predict_forward),
            )
        } else {
            srs_review_predict_forward_profiled(
                &rnn.weights,
                &predict_features,
                time_x_shift_b1c_by_module,
                time_state_b1hkk_by_module,
                channel_state_b1c_by_module,
                profile
                    .as_deref_mut()
                    .map(|profile| &mut profile.predict_forward),
            )
        }
        .map_err(py_value_error)
    })?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.predict_forward_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    let probability = probability_from_logits(&predict_p_logits).map_err(py_value_error)?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.predict_output_ns += start.elapsed_ns();
        profile.materialization.prediction_probability_values += 1;
    }
    Ok(probability)
}

enum BatchPredictRejectionReason {
    MissingEncoding,
    MissingState,
}

fn batch_predict_rejection_reason(
    deterministic: &FeatureState,
    row: &PreparedRow,
) -> Result<BatchPredictRejectionReason> {
    for (submodule, _) in ID_SUBMODULES {
        let id = prepared_i64(row, submodule)?;
        if !deterministic
            .id_encodings
            .get(submodule)
            .expect("id encoding map initialized for every submodule")
            .contains_key(&id)
        {
            return Ok(BatchPredictRejectionReason::MissingEncoding);
        }
    }

    let card_id = prepared_i64(row, "card_id")?;
    let note_id = prepared_i64(row, "note_id")?;
    let deck_id = prepared_i64(row, "deck_id")?;
    let preset_id = prepared_i64(row, "preset_id")?;
    if !deterministic
        .recurrent_state_keys
        .card_states
        .contains(&card_id)
        || !deterministic
            .recurrent_state_keys
            .note_states
            .contains(&note_id)
        || !deterministic
            .recurrent_state_keys
            .deck_states
            .contains(&deck_id)
        || !deterministic
            .recurrent_state_keys
            .preset_states
            .contains(&preset_id)
    {
        return Ok(BatchPredictRejectionReason::MissingState);
    }

    Ok(BatchPredictRejectionReason::MissingState)
}

fn tensor_from_feature_vector(values: Vec<f32>, name: &str) -> Result<Tensor> {
    let cols = values.len();
    if cols == 0 {
        bail!("{name} must not be empty");
    }
    Tensor::from_vec(values, (1usize, cols), &Device::Cpu)
}

fn tensor2_list_value_count(values: &Tensor2List) -> usize {
    values.iter().map(Vec::len).sum()
}

pub(super) fn validate_num_threads(num_threads: Option<usize>) -> PyResult<()> {
    if matches!(num_threads, Some(0)) {
        return Err(py_value_error("num_threads must be at least 1"));
    }
    Ok(())
}

pub(super) fn with_rayon_threads<R, F>(num_threads: Option<usize>, f: F) -> PyResult<R>
where
    R: Send,
    F: FnOnce() -> PyResult<R> + Send,
{
    let Some(num_threads) = num_threads else {
        return f();
    };
    validate_num_threads(Some(num_threads))?;
    rayon_pool(num_threads)?.install(f)
}

pub(super) fn rayon_pool(num_threads: usize) -> PyResult<Arc<ThreadPool>> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<ThreadPool>>>> = OnceLock::new();
    let pools = POOLS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut pools = pools
        .lock()
        .map_err(|_| py_value_error("Rayon thread pool cache is poisoned"))?;
    if let Some(pool) = pools.get(&num_threads) {
        return Ok(Arc::clone(pool));
    }
    let pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .thread_name(move |index| format!("rwkv-p-{num_threads}-{index}"))
        .build()
        .map_err(|error| py_value_error(format!("failed to build Rayon thread pool: {error}")))?;
    let pool = Arc::new(pool);
    pools.insert(num_threads, Arc::clone(&pool));
    Ok(pool)
}
