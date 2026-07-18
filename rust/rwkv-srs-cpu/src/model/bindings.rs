use std::collections::BTreeSet;
use std::path::PathBuf;
use std::time::Instant;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyMapping};

use super::bulk::{
    feature_prepass_step, process_reviews_bulk_feature_prepass_debug,
    process_reviews_bulk_layered_with_state, process_reviews_bulk_reference_with_state,
    process_reviews_bulk_stream_plan_debug, BulkModuleStreamPlan, BulkStreamPlan,
    BulkStreamStateKeyPlan,
};
use super::gpu_process_scan::{
    build_state_only_review_inputs_gpu_scan_with_state, process_review_inputs_gpu_scan_with_state,
    process_reviews_gpu_scan_with_state, synchronize_gpu_process_state_with_state,
};
use super::live_session::{
    initialize_live_session_predictions, reconcile_live_session_candidates,
    reconcile_live_session_membership, refresh_live_session, replace_live_session_candidates,
    LiveCandidateSeedNative, LiveCandidateSnapshot, LiveOrder, LivePredictionMode,
    LivePredictionSessionState, LiveRefreshOutput, LiveRefreshProfile,
};
use super::pipeline::{
    process_review_inputs_pipeline_with_state, process_reviews_pipeline_with_state,
};
use super::runtime::{
    build_state_only_inputs_with_state, build_state_only_packed_with_state,
    predict_probability_with_state, predict_review_inputs_cpu,
    predict_review_inputs_cpu_fast_caller_profiled, predict_review_inputs_gpu,
    predict_reviews_gpu_with_state, predict_reviews_with_state, probability_from_logits,
    process_packed_reviews_with_state, process_review_inputs_with_state,
    process_reviews_with_state, require_loaded_scope, undoable_process_input_with_state,
    warm_rayon_pool, FastPredictCallerProfile,
};
use super::state::{
    native_module_state_to_py, native_state_map_to_pydict, nested_tensor_vec_from_3d,
    nested_tensor_vec_from_5d, nested_tensor_vec_to_vec3, nested_tensor_vec_to_vec5,
    py_optional_entity_state_from_snapshot, py_state_map_from_snapshot, tensor_vec_from_3d,
    tensor_vec_from_5d, tensor_vec_to_vec3, tensor_vec_to_vec5, SrsReviewPyState3,
    SrsReviewPyState5,
};
use super::state_builder::{
    build_state_only_inputs_pipeline_with_state, build_state_only_pipeline_with_state,
    StateBuildProfile,
};
use super::undo::{BatchRecurrentUndoFrame, RuntimeUndoFrame};
use super::{
    channel_mixer_forward, checkpoint_bin, load_srs_rwkv_rnn_weights, py_value_error,
    rwkv_layer_forward, rwkv_rnn_forward, srs_review_forward, srs_review_forward_options,
    time_mixer_forward, NativePredictionBatch, NativeProcessManyPyOutput, NativeReviewBatch,
    NativeRnn, NativeRuntime,
};
use crate::gpu::{py_gpu_error, py_gpu_unavailable_error};
use crate::id_encoding::TorchMt19937;
use crate::profile::{ProfileTimer, RuntimeProfile};
use crate::py_state::{
    feature_state_id_encoding_snapshot, feature_state_recurrent_state_snapshot,
    feature_state_snapshot, predict_review_from_dict_with_card_id,
    predict_review_from_mapping_with_card_id, prepared_feature_fields_from_mapping,
    prepared_record_fields_from_mapping, restore_feature_state_id_encodings,
    restore_feature_state_recurrent_keys, restore_feature_state_snapshot, review_from_mapping,
    row_to_pydict, torch_rng_state_bytes, DeterministicState,
};
use crate::state::{FeatureState, MaybeId};
use crate::tensor_io::{
    tensor_from_2d, tensor_from_3d, tensor_from_5d, tensor_to_vec5, Tensor2List, Tensor3List,
    Tensor5List,
};

type TimeMixerPyOutput = (Tensor2List, Tensor2List, Tensor3List, Tensor5List);
type RwkvLayerPyOutput = (
    Tensor2List,
    Tensor2List,
    Tensor3List,
    Tensor5List,
    Tensor3List,
);
type RwkvRnnPyOutput = (
    Tensor2List,
    Vec<Tensor3List>,
    Vec<Tensor5List>,
    Vec<Tensor3List>,
);
type NativeReviewPyOutput = (Option<Tensor2List>, Option<Tensor2List>, Tensor2List);
type NativeProcessManyProfiledPyOutput = (
    Vec<f64>,
    Option<Vec<Tensor2List>>,
    Option<Vec<Tensor2List>>,
    Py<PyDict>,
);
type NativePredictManyProfiledPyOutput = (Vec<f64>, Py<PyDict>);
type NativePredictManyListTransportProfiledPyOutput = (Py<PyList>, Py<PyDict>);
type NativePredictManyF32TransportProfiledPyOutput = (Py<PyBytes>, Py<PyDict>);
type NativeStateOnlyProfiledPyOutput = (usize, Py<PyDict>);
type NativeGpuScanPyOutput = (Vec<f64>, Option<Py<PyBytes>>, Option<Py<PyBytes>>);
type NativeGpuProcessProgressPyOutput = (usize, Vec<f64>, Option<Py<PyBytes>>, Option<Py<PyBytes>>);
type SrsReviewPyOutput = (
    Option<Tensor2List>,
    Option<Tensor2List>,
    Tensor2List,
    SrsReviewPyState3,
    SrsReviewPyState5,
    SrsReviewPyState3,
);

fn require_loaded_inputs_scope(
    scope: Option<&checkpoint_bin::CheckpointScope>,
    inputs: &[crate::state::ReviewInput],
) -> PyResult<()> {
    if let Some(scope) = scope {
        for input in inputs {
            require_loaded_scope(scope, input)?;
        }
    }
    Ok(())
}

fn require_loaded_batch_scope(
    scope: Option<&checkpoint_bin::CheckpointScope>,
    batch: &NativeReviewBatch,
) -> PyResult<()> {
    require_loaded_inputs_scope(scope, batch.inputs())
}

fn predict_prediction_batch_cpu(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    scope: Option<&checkpoint_bin::CheckpointScope>,
    batch: &NativePredictionBatch,
    batch_size: usize,
    num_threads: Option<usize>,
    lightning: bool,
) -> PyResult<Vec<f64>> {
    require_loaded_inputs_scope(scope, batch.inputs())?;
    predict_review_inputs_cpu(
        rnn,
        deterministic,
        batch.inputs(),
        batch_size,
        num_threads,
        lightning,
        None,
    )
}

fn profile_prediction_batch_fast(
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    scope: Option<&checkpoint_bin::CheckpointScope>,
    batch: &NativePredictionBatch,
    batch_size: usize,
    num_threads: Option<usize>,
) -> PyResult<(Vec<f64>, FastPredictCallerProfile)> {
    let scope_start = Instant::now();
    require_loaded_inputs_scope(scope, batch.inputs())?;
    let scope_validation_ns = scope_start.elapsed().as_nanos();
    predict_review_inputs_cpu_fast_caller_profiled(
        rnn,
        deterministic,
        batch.inputs(),
        batch_size,
        num_threads,
        0,
        scope_validation_ns,
    )
}

fn probabilities_to_f32_bytes<'py>(
    py: Python<'py>,
    probabilities: Vec<f64>,
) -> Bound<'py, PyBytes> {
    // The shared CPU and GPU predictors return f64 because the established
    // list API exposes Python floats. Their probabilities originate in f32;
    // narrowing here reconstructs those bits while keeping the numerical
    // implementation shared with the ordinary list-returning path.
    let probabilities = probabilities
        .into_iter()
        .map(|probability| probability as f32)
        .collect::<Vec<_>>();
    PyBytes::new_bound(py, bytemuck::cast_slice(&probabilities))
}

#[pymethods]
impl NativeRnn {
    #[new]
    fn new(checkpoint_path: PathBuf) -> PyResult<Self> {
        let mut rnn = Self::from_checkpoint(checkpoint_path).map_err(py_value_error)?;
        rnn.warm_predict_path().map_err(py_value_error)?;
        Ok(rnn)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (card_features, card_id, note_id, deck_id, preset_id, skip, return_curve=true))]
    fn review(
        &mut self,
        card_features: Tensor2List,
        card_id: i64,
        note_id: i64,
        deck_id: i64,
        preset_id: i64,
        skip: bool,
        return_curve: bool,
    ) -> PyResult<NativeReviewPyOutput> {
        self.materialize_flat_cpu_state().map_err(py_value_error)?;
        let card_features =
            tensor_from_2d(card_features, "card_features").map_err(py_value_error)?;
        let (time_x_shift_b1c_by_module, time_state_b1hkk_by_module, channel_state_b1c_by_module) =
            self.state_inputs(card_id, note_id, deck_id, preset_id);

        let (
            out_ahead_logits,
            out_w,
            out_p_logits,
            next_time_x_shift_b1c_by_module,
            next_time_state_b1hkk_by_module,
            next_channel_state_b1c_by_module,
        ) = srs_review_forward(
            &self.weights,
            &card_features,
            Some(&time_x_shift_b1c_by_module),
            Some(&time_state_b1hkk_by_module),
            Some(&channel_state_b1c_by_module),
            return_curve,
        )
        .map_err(py_value_error)?;

        if !skip {
            self.store_review_states(
                card_id,
                note_id,
                deck_id,
                preset_id,
                next_time_x_shift_b1c_by_module,
                next_time_state_b1hkk_by_module,
                next_channel_state_b1c_by_module,
            )
            .map_err(py_value_error)?;
        }

        Ok((
            out_ahead_logits
                .map(|tensor| tensor.to_vec2::<f32>())
                .transpose()
                .map_err(py_value_error)?,
            out_w
                .map(|tensor| tensor.to_vec2::<f32>())
                .transpose()
                .map_err(py_value_error)?,
            out_p_logits.to_vec2::<f32>().map_err(py_value_error)?,
        ))
    }

    #[pyo3(signature = (predict_features, process_features, ids, return_curves=false))]
    fn process_many(
        &mut self,
        predict_features: Tensor2List,
        process_features: Tensor2List,
        ids: Vec<(i64, i64, i64, i64)>,
        return_curves: bool,
    ) -> PyResult<NativeProcessManyPyOutput> {
        self.materialize_flat_cpu_state().map_err(py_value_error)?;
        let predict_features =
            tensor_from_2d(predict_features, "predict_features").map_err(py_value_error)?;
        let process_features =
            tensor_from_2d(process_features, "process_features").map_err(py_value_error)?;
        let (predict_rows, predict_cols) = predict_features.dims2().map_err(py_value_error)?;
        let (process_rows, process_cols) = process_features.dims2().map_err(py_value_error)?;
        if predict_rows != process_rows || predict_cols != process_cols {
            return Err(py_value_error(format!(
                "predict_features and process_features must have matching shapes, got {:?} and {:?}",
                predict_features.dims(),
                process_features.dims()
            )));
        }
        if ids.len() != predict_rows {
            return Err(py_value_error(format!(
                "ids length must match feature rows, got {} ids for {predict_rows} rows",
                ids.len()
            )));
        }

        let undo = BatchRecurrentUndoFrame::capture(self, &ids);
        let result = (|| {
            let mut prediction_probabilities = Vec::with_capacity(ids.len());
            let mut curve_ahead_logits = return_curves.then(|| Vec::with_capacity(ids.len()));
            let mut curve_w = return_curves.then(|| Vec::with_capacity(ids.len()));

            for (index, (card_id, note_id, deck_id, preset_id)) in ids.iter().copied().enumerate() {
                let predict_feature = predict_features
                    .narrow(0, index, 1)
                    .map_err(py_value_error)?;
                let (
                    time_x_shift_b1c_by_module,
                    time_state_b1hkk_by_module,
                    channel_state_b1c_by_module,
                ) = self.state_inputs(card_id, note_id, deck_id, preset_id);
                let (_, _, predict_p_logits, _, _, _) = srs_review_forward(
                    &self.weights,
                    &predict_feature,
                    Some(&time_x_shift_b1c_by_module),
                    Some(&time_state_b1hkk_by_module),
                    Some(&channel_state_b1c_by_module),
                    false,
                )
                .map_err(py_value_error)?;
                prediction_probabilities
                    .push(probability_from_logits(&predict_p_logits).map_err(py_value_error)?);

                let process_feature = process_features
                    .narrow(0, index, 1)
                    .map_err(py_value_error)?;
                let (
                    time_x_shift_b1c_by_module,
                    time_state_b1hkk_by_module,
                    channel_state_b1c_by_module,
                ) = self.state_inputs(card_id, note_id, deck_id, preset_id);
                let (
                    out_ahead_logits,
                    out_w,
                    _out_p_logits,
                    next_time_x_shift_b1c_by_module,
                    next_time_state_b1hkk_by_module,
                    next_channel_state_b1c_by_module,
                ) = srs_review_forward_options(
                    &self.weights,
                    &process_feature,
                    Some(&time_x_shift_b1c_by_module),
                    Some(&time_state_b1hkk_by_module),
                    Some(&channel_state_b1c_by_module),
                    return_curves,
                    false,
                )
                .map_err(py_value_error)?;

                self.store_review_states(
                    card_id,
                    note_id,
                    deck_id,
                    preset_id,
                    next_time_x_shift_b1c_by_module,
                    next_time_state_b1hkk_by_module,
                    next_channel_state_b1c_by_module,
                )
                .map_err(py_value_error)?;

                if return_curves {
                    curve_ahead_logits
                        .as_mut()
                        .expect("curve outputs requested")
                        .push(
                            out_ahead_logits
                                .expect("curve logits are present when requested")
                                .to_vec2::<f32>()
                                .map_err(py_value_error)?,
                        );
                    curve_w.as_mut().expect("curve outputs requested").push(
                        out_w
                            .expect("curve weights are present when requested")
                            .to_vec2::<f32>()
                            .map_err(py_value_error)?,
                    );
                }
            }

            Ok((prediction_probabilities, curve_ahead_logits, curve_w))
        })();
        match result {
            Ok(output) => Ok(output),
            Err(error) => {
                undo.restore(self);
                Err(error)
            }
        }
    }

    #[pyo3(signature = (deterministic, reviews, return_curves=false, num_threads=None))]
    fn process_reviews(
        &mut self,
        mut deterministic: PyRefMut<'_, DeterministicState>,
        reviews: &Bound<'_, PyAny>,
        return_curves: bool,
        num_threads: Option<usize>,
    ) -> PyResult<NativeProcessManyPyOutput> {
        process_reviews_with_state(
            self,
            &mut deterministic.inner,
            reviews,
            return_curves,
            num_threads,
            None,
        )
    }

    fn recurrent_state_lists<'py>(&mut self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        self.materialize_flat_cpu_state().map_err(py_value_error)?;
        let dict = PyDict::new_bound(py);
        dict.set_item(
            "card_states",
            native_state_map_to_pydict(py, &self.card_states)?,
        )?;
        dict.set_item(
            "note_states",
            native_state_map_to_pydict(py, &self.note_states)?,
        )?;
        dict.set_item(
            "deck_states",
            native_state_map_to_pydict(py, &self.deck_states)?,
        )?;
        dict.set_item(
            "preset_states",
            native_state_map_to_pydict(py, &self.preset_states)?,
        )?;
        dict.set_item(
            "global_state",
            self.global_state
                .as_ref()
                .map(native_module_state_to_py)
                .transpose()
                .map_err(py_value_error)?,
        )?;
        Ok(dict.unbind())
    }

    fn recurrent_state_key_snapshot<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item(
            "card_states",
            self.card_states
                .keys()
                .chain(self.flat_cpu_state.card_states.keys())
                .copied()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "note_states",
            self.note_states
                .keys()
                .chain(self.flat_cpu_state.note_states.keys())
                .copied()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "deck_states",
            self.deck_states
                .keys()
                .chain(self.flat_cpu_state.deck_states.keys())
                .copied()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "preset_states",
            self.preset_states
                .keys()
                .chain(self.flat_cpu_state.preset_states.keys())
                .copied()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "global_state",
            self.global_state.is_some() || self.flat_cpu_state.global_state.is_some(),
        )?;
        Ok(dict.unbind())
    }

    fn restore_recurrent_state_lists(&mut self, snapshot: &Bound<'_, PyMapping>) -> PyResult<()> {
        self.card_states = py_state_map_from_snapshot(snapshot, "card_states")?;
        self.note_states = py_state_map_from_snapshot(snapshot, "note_states")?;
        self.deck_states = py_state_map_from_snapshot(snapshot, "deck_states")?;
        self.preset_states = py_state_map_from_snapshot(snapshot, "preset_states")?;
        self.global_state = py_optional_entity_state_from_snapshot(snapshot, "global_state")?;
        self.flat_cpu_state.clear();
        self.invalidate_gpu();
        Ok(())
    }
}

impl NativeRuntime {
    fn require_no_live_session(&self, operation: &str) -> PyResult<()> {
        if self.live_session.is_none() && self.pending_live_session.is_none() {
            return Ok(());
        }
        Err(py_value_error(format!(
            "{operation} cannot mutate the runtime while a live prediction session is active; \
             use the corresponding live-session operation or close the session first"
        )))
    }

    fn live_session_ref(&self, token: u64) -> PyResult<&LivePredictionSessionState> {
        if self
            .pending_live_session
            .as_ref()
            .is_some_and(|state| state.token == token)
        {
            return Err(py_value_error(
                "live prediction session construction is not finalized",
            ));
        }
        let state = self
            .live_session
            .as_ref()
            .ok_or_else(|| py_value_error("live prediction session is closed"))?;
        if state.token != token {
            return Err(py_value_error("live prediction session token is stale"));
        }
        Ok(state)
    }

    fn live_session_mut(&mut self, token: u64) -> PyResult<&mut LivePredictionSessionState> {
        let state = self
            .live_session
            .as_mut()
            .ok_or_else(|| py_value_error("live prediction session is closed"))?;
        if state.token != token {
            return Err(py_value_error("live prediction session token is stale"));
        }
        Ok(state)
    }

    fn record_failed_live_reconcile(
        &mut self,
        token: u64,
        card_id_parse_ns: u128,
        seed_parse_ns: u128,
        scope_validation_ns: u128,
        native_total_ns: u128,
    ) -> PyResult<()> {
        self.live_session_mut(token)?
            .record_failed_reconcile(LiveRefreshProfile {
                card_id_parse_ns,
                seed_parse_ns,
                scope_validation_ns,
                native_total_ns,
                ..LiveRefreshProfile::default()
            });
        Ok(())
    }
}

fn checkpoint_scope_from_optional_identity_lists(
    card_ids: Option<Vec<i64>>,
    note_ids: Option<Vec<i64>>,
    deck_ids: Option<Vec<i64>>,
    preset_ids: Option<Vec<i64>>,
) -> PyResult<Option<checkpoint_bin::CheckpointScope>> {
    let provided = [
        card_ids.is_some(),
        note_ids.is_some(),
        deck_ids.is_some(),
        preset_ids.is_some(),
    ];
    if provided.iter().any(|value| *value) && !provided.iter().all(|value| *value) {
        return Err(py_value_error(
            "selective checkpoint restore requires all four identity lists",
        ));
    }
    Ok(card_ids.map(|card_ids| checkpoint_bin::CheckpointScope {
        card_ids: card_ids.into_iter().collect(),
        note_ids: note_ids
            .expect("all selective identity lists checked above")
            .into_iter()
            .collect(),
        deck_ids: deck_ids
            .expect("all selective identity lists checked above")
            .into_iter()
            .collect(),
        preset_ids: preset_ids
            .expect("all selective identity lists checked above")
            .into_iter()
            .collect(),
    }))
}

#[pymethods]
impl NativeRuntime {
    #[new]
    #[pyo3(signature = (
        checkpoint_path,
        torch_seed=5489,
        undo_limit=30,
        restore_path=None,
        card_ids=None,
        note_ids=None,
        deck_ids=None,
        preset_ids=None,
    ))]
    fn new(
        checkpoint_path: PathBuf,
        torch_seed: u64,
        undo_limit: usize,
        restore_path: Option<PathBuf>,
        card_ids: Option<Vec<i64>>,
        note_ids: Option<Vec<i64>>,
        deck_ids: Option<Vec<i64>>,
        preset_ids: Option<Vec<i64>>,
    ) -> PyResult<Self> {
        let scope = checkpoint_scope_from_optional_identity_lists(
            card_ids, note_ids, deck_ids, preset_ids,
        )?;
        if restore_path.is_none() && scope.is_some() {
            return Err(py_value_error(
                "selective checkpoint restore identities require restore_path",
            ));
        }
        let rnn = NativeRnn::from_checkpoint(checkpoint_path).map_err(py_value_error)?;
        let mut runtime = Self {
            deterministic: FeatureState::with_torch_seed(torch_seed),
            rnn,
            undo_stack: Default::default(),
            undo_limit,
            loaded_scope: None,
            gpu_process_committed_rows: 0,
            gpu_process_output: (Vec::new(), None, None),
            live_session: None,
            pending_live_session: None,
            next_live_session_token: 1,
        };
        // Restore before warming the CPU execution path. The warm-up is the
        // first operation that reads the process-wide CPU profile, so malformed
        // model/checkpoint input can fail without permanently claiming it.
        if let Some(path) = restore_path {
            checkpoint_bin::restore_checkpoint_bin_path(&path, &mut runtime, scope.as_ref())
                .map_err(py_value_error)?;
        }
        runtime.loaded_scope = scope;
        runtime.rnn.warm_predict_path().map_err(py_value_error)?;
        Ok(runtime)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (candidates, initial_target_timestamp_seconds, initial_target_day_offset, order, mode, batch_size, refresh_limit, num_threads=None, profiling=false, initial_select_limit=2))]
    fn create_live_prediction_session(
        &mut self,
        py: Python<'_>,
        candidates: &Bound<'_, PyAny>,
        initial_target_timestamp_seconds: f64,
        initial_target_day_offset: f64,
        order: &str,
        mode: &str,
        batch_size: usize,
        refresh_limit: usize,
        num_threads: Option<usize>,
        profiling: bool,
        initial_select_limit: usize,
    ) -> PyResult<(u64, Py<PyDict>)> {
        if self.live_session.is_some() || self.pending_live_session.is_some() {
            return Err(py_value_error(
                "only one live prediction session may be active per runtime",
            ));
        }
        let seeds = live_candidate_seeds_from_py(candidates)?;
        if let Some(scope) = self.loaded_scope.as_ref() {
            for seed in &seeds {
                require_loaded_scope(scope, &seed.row)?;
            }
        }
        let token = self.next_live_session_token.max(1);
        let mut state = LivePredictionSessionState::new_unranked(
            token,
            &seeds,
            initial_target_timestamp_seconds,
            initial_target_day_offset,
            LiveOrder::parse(order)?,
            LivePredictionMode::parse(mode)?,
            batch_size,
            refresh_limit,
            num_threads,
            profiling,
        )?;
        let _initial_profile = py.allow_threads(|| {
            initialize_live_session_predictions(&mut state, &mut self.rnn, &mut self.deterministic)
        })?;
        // Construct the fallible Python result before committing lifecycle or
        // undo changes. Even an allocation failure must leave the runtime as
        // though session construction was never attempted.
        let initial_result =
            live_refresh_output_to_pydict(py, state.initial_output(initial_select_limit))?;
        // Keep construction staged until the adapter has converted the compact
        // result and allocated its facade. Finalization is the sole commit point
        // for lifecycle and undo changes; abort simply drops this derived state.
        self.pending_live_session = Some(state);
        Ok((token, initial_result))
    }

    fn finalize_live_prediction_session(&mut self, token: u64) -> PyResult<()> {
        if self.live_session.is_some() {
            return Err(py_value_error(
                "only one live prediction session may be active per runtime",
            ));
        }
        let pending = self
            .pending_live_session
            .as_ref()
            .ok_or_else(|| py_value_error("live prediction session construction is not pending"))?;
        if pending.token != token {
            return Err(py_value_error("live prediction session token is stale"));
        }

        let state = self
            .pending_live_session
            .take()
            .expect("pending live session checked above");
        self.undo_stack.clear();
        self.next_live_session_token = token.wrapping_add(1).max(1);
        self.live_session = Some(state);
        Ok(())
    }

    fn abort_live_prediction_session(&mut self, token: u64) -> PyResult<bool> {
        let pending = self
            .pending_live_session
            .as_ref()
            .ok_or_else(|| py_value_error("live prediction session construction is not pending"))?;
        if pending.token != token {
            return Err(py_value_error("live prediction session token is stale"));
        }
        self.pending_live_session = None;
        Ok(true)
    }

    #[pyo3(signature = (token, select_limit=2, exclude_card_ids=Vec::new()))]
    fn live_current_selection<'py>(
        &mut self,
        py: Python<'py>,
        token: u64,
        select_limit: usize,
        exclude_card_ids: Vec<i64>,
    ) -> PyResult<Py<PyDict>> {
        let state = self.live_session_mut(token)?;
        live_refresh_output_to_pydict(
            py,
            state.current_selection_output(select_limit, &exclude_card_ids),
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (token, target_timestamp_seconds, target_day_offset, select_limit=2, exclude_card_ids=Vec::new(), exclude_refresh_card_ids=Vec::new(), retention_extra=0.0))]
    fn live_refresh<'py>(
        &mut self,
        py: Python<'py>,
        token: u64,
        target_timestamp_seconds: f64,
        target_day_offset: f64,
        select_limit: usize,
        exclude_card_ids: Vec<i64>,
        exclude_refresh_card_ids: Vec<i64>,
        retention_extra: f64,
    ) -> PyResult<Py<PyDict>> {
        self.live_session_ref(token)?;
        let Self {
            deterministic,
            rnn,
            undo_stack,
            live_session,
            ..
        } = self;
        let state = live_session
            .as_mut()
            .expect("live session checked before field split");
        let live_undo = undo_stack
            .back_mut()
            .and_then(|frame| frame.live.as_mut())
            .filter(|frame| frame.session_token == token);
        let output = py.allow_threads(|| {
            refresh_live_session(
                state,
                rnn,
                deterministic,
                target_timestamp_seconds,
                target_day_offset,
                select_limit,
                &exclude_card_ids,
                &exclude_refresh_card_ids,
                retention_extra,
                live_undo,
            )
        })?;
        let result_start = std::time::Instant::now();
        let result = live_refresh_output_to_pydict(py, output)?;
        state.record_python_result_ns(result_start.elapsed().as_nanos());
        Ok(result)
    }

    #[pyo3(signature = (token, review, requeue_after_prediction=false, num_threads=None, return_curves=true))]
    fn live_process_answer(
        &mut self,
        py: Python<'_>,
        token: u64,
        review: &Bound<'_, PyMapping>,
        requeue_after_prediction: bool,
        num_threads: Option<usize>,
        return_curves: bool,
    ) -> PyResult<(f64, Option<Tensor2List>, Option<Tensor2List>, u64)> {
        if self.undo_limit == 0 {
            return Err(py_value_error(
                "live process_answer is disabled by undo_limit=0",
            ));
        }
        let input = review_from_mapping(review, true)?;
        if let Some(scope) = self.loaded_scope.as_ref() {
            require_loaded_scope(scope, &input)?;
        }
        let slot = self
            .live_session_ref(token)?
            .candidate_slot(input.card_id)?;
        self.live_session_ref(token)?
            .validate_answer_update(slot, &input)?;
        py.allow_threads(|| {
            synchronize_gpu_process_state_with_state(&mut self.rnn).map_err(py_gpu_error)
        })?;
        let live_undo = self.live_session_ref(token)?.begin_answer_undo(slot);
        let (output, model_undo) = py.allow_threads(|| {
            undoable_process_input_with_state(
                &mut self.rnn,
                &mut self.deterministic,
                &input,
                return_curves,
                num_threads,
            )
        })?;
        let state = self.live_session_mut(token)?;
        state.apply_answer(slot, &input, requeue_after_prediction);
        let generation = state.generation();
        self.undo_stack.push_back(RuntimeUndoFrame {
            model: model_undo,
            live: Some(live_undo),
        });
        if self.undo_stack.len() > self.undo_limit {
            self.undo_stack.pop_front();
        }
        Ok((output.0, output.1, output.2, generation))
    }

    fn live_undo_last_process(&mut self, py: Python<'_>, token: u64) -> PyResult<(usize, u64)> {
        self.live_session_ref(token)?;
        let live_token = self
            .undo_stack
            .back()
            .and_then(|frame| frame.live.as_ref())
            .map(|frame| frame.session_token)
            .ok_or_else(|| py_value_error("no live-session process is available to undo"))?;
        if live_token != token {
            return Err(py_value_error(
                "latest undo frame does not belong to this live prediction session",
            ));
        }
        py.allow_threads(|| {
            synchronize_gpu_process_state_with_state(&mut self.rnn).map_err(py_gpu_error)
        })?;
        let frame = self
            .undo_stack
            .pop_back()
            .expect("live undo frame checked above");
        let live_frame = frame
            .live
            .expect("live undo frame presence checked before model restoration");
        let state = self
            .live_session
            .as_mut()
            .expect("live session checked before undo");
        py.allow_threads(|| {
            frame.model.restore(&mut self.rnn, &mut self.deterministic);
            state.restore_undo(live_frame)
        })?;
        Ok((
            self.undo_stack.len(),
            self.live_session_ref(token)?.generation(),
        ))
    }

    fn live_exclude_card(&mut self, token: u64, card_id: i64) -> PyResult<u64> {
        self.live_session_ref(token)?;
        let (live_session, undo_stack) = (&mut self.live_session, &mut self.undo_stack);
        let state = live_session.as_mut().expect("live session checked above");
        let undo = undo_stack
            .back_mut()
            .and_then(|frame| frame.live.as_mut())
            .filter(|frame| frame.session_token == token);
        state.exclude_card(card_id, undo)
    }

    fn live_include_card(&mut self, token: u64, card_id: i64) -> PyResult<u64> {
        self.live_session_ref(token)?;
        let (live_session, undo_stack) = (&mut self.live_session, &mut self.undo_stack);
        let state = live_session.as_mut().expect("live session checked above");
        let undo = undo_stack
            .back_mut()
            .and_then(|frame| frame.live.as_mut())
            .filter(|frame| frame.session_token == token);
        state.include_card(card_id, undo)
    }

    fn live_remove_candidate(&mut self, token: u64, card_id: i64) -> PyResult<u64> {
        self.live_session_ref(token)?;
        let (live_session, undo_stack) = (&mut self.live_session, &mut self.undo_stack);
        let state = live_session.as_mut().expect("live session checked above");
        let undo = undo_stack
            .back_mut()
            .and_then(|frame| frame.live.as_mut())
            .filter(|frame| frame.session_token == token);
        state.remove_card(card_id, undo)
    }

    fn live_upsert_candidates(
        &mut self,
        token: u64,
        candidates: &Bound<'_, PyAny>,
    ) -> PyResult<u64> {
        self.live_session_ref(token)?;
        let seeds = live_candidate_seeds_from_py(candidates)?;
        if let Some(scope) = self.loaded_scope.as_ref() {
            for seed in &seeds {
                require_loaded_scope(scope, &seed.row)?;
            }
        }
        let (live_session, undo_stack) = (&mut self.live_session, &mut self.undo_stack);
        let state = live_session.as_mut().expect("live session checked above");
        let undo = undo_stack
            .back_mut()
            .and_then(|frame| frame.live.as_mut())
            .filter(|frame| frame.session_token == token);
        state.upsert_candidates(&seeds, undo)
    }

    fn live_replace_candidates(
        &mut self,
        py: Python<'_>,
        token: u64,
        candidates: &Bound<'_, PyAny>,
    ) -> PyResult<u64> {
        self.live_session_ref(token)?;
        let seeds = live_candidate_seeds_from_py(candidates)?;
        if let Some(scope) = self.loaded_scope.as_ref() {
            for seed in &seeds {
                require_loaded_scope(scope, &seed.row)?;
            }
        }
        let Self {
            live_session,
            rnn,
            deterministic,
            undo_stack,
            ..
        } = self;
        let state = live_session.as_mut().expect("live session checked above");
        let generation = py
            .allow_threads(|| replace_live_session_candidates(state, &seeds, rnn, deterministic))?;
        undo_stack.clear();
        Ok(generation)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (token, candidates, target_timestamp_seconds, target_day_offset, select_limit=2, exclude_card_ids=Vec::new(), retention_extra=0.0))]
    fn live_reconcile_candidates<'py>(
        &mut self,
        py: Python<'py>,
        token: u64,
        candidates: &Bound<'_, PyAny>,
        target_timestamp_seconds: f64,
        target_day_offset: f64,
        select_limit: usize,
        exclude_card_ids: Vec<i64>,
        retention_extra: f64,
    ) -> PyResult<Py<PyDict>> {
        self.live_session_ref(token)?;
        let native_total_start = Instant::now();
        let seed_parse_start = Instant::now();
        let seeds_result = live_candidate_seeds_from_py(candidates);
        let seed_parse_ns = seed_parse_start.elapsed().as_nanos();
        let seeds = match seeds_result {
            Ok(seeds) => seeds,
            Err(error) => {
                self.record_failed_live_reconcile(
                    token,
                    0,
                    seed_parse_ns,
                    0,
                    native_total_start.elapsed().as_nanos(),
                )?;
                return Err(error);
            }
        };
        let scope_validation_start = Instant::now();
        let scope_result = self.loaded_scope.as_ref().map_or(Ok(()), |scope| {
            py.allow_threads(|| validate_live_seed_scope(scope, &seeds))
        });
        let scope_validation_ns = scope_validation_start.elapsed().as_nanos();
        if let Err(error) = scope_result {
            self.record_failed_live_reconcile(
                token,
                0,
                seed_parse_ns,
                scope_validation_ns,
                native_total_start.elapsed().as_nanos(),
            )?;
            return Err(error);
        }
        let Self {
            live_session,
            rnn,
            deterministic,
            undo_stack,
            ..
        } = self;
        let state = live_session.as_mut().expect("live session checked above");
        let live_undo = undo_stack
            .back_mut()
            .and_then(|frame| frame.live.as_mut())
            .filter(|frame| frame.session_token == token);
        let output_result = py.allow_threads(|| {
            reconcile_live_session_candidates(
                state,
                &seeds,
                rnn,
                deterministic,
                target_timestamp_seconds,
                target_day_offset,
                select_limit,
                &exclude_card_ids,
                retention_extra,
                live_undo,
            )
        });
        let output = match output_result {
            Ok(output) => output,
            Err(error) => {
                state.record_failed_reconcile(LiveRefreshProfile {
                    seed_parse_ns,
                    scope_validation_ns,
                    native_total_ns: native_total_start.elapsed().as_nanos(),
                    ..LiveRefreshProfile::default()
                });
                return Err(error);
            }
        };
        state.record_reconcile_boundary_ns(
            0,
            seed_parse_ns,
            scope_validation_ns,
            native_total_start.elapsed().as_nanos(),
        );
        let result_start = std::time::Instant::now();
        let result = live_refresh_output_to_pydict(py, output)?;
        state.record_reconcile_python_result_ns(result_start.elapsed().as_nanos());
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (token, card_ids, changed_candidates, target_timestamp_seconds, target_day_offset, select_limit=2, exclude_card_ids=Vec::new(), retention_extra=0.0))]
    fn live_reconcile_membership<'py>(
        &mut self,
        py: Python<'py>,
        token: u64,
        card_ids: &Bound<'_, PyAny>,
        changed_candidates: &Bound<'_, PyAny>,
        target_timestamp_seconds: f64,
        target_day_offset: f64,
        select_limit: usize,
        exclude_card_ids: Vec<i64>,
        retention_extra: f64,
    ) -> PyResult<Py<PyDict>> {
        self.live_session_ref(token)?;
        let native_total_start = Instant::now();
        let card_id_parse_start = Instant::now();
        let card_ids_result = live_card_ids_from_py(card_ids);
        let card_id_parse_ns = card_id_parse_start.elapsed().as_nanos();
        let card_ids = match card_ids_result {
            Ok(card_ids) => card_ids,
            Err(error) => {
                self.record_failed_live_reconcile(
                    token,
                    card_id_parse_ns,
                    0,
                    0,
                    native_total_start.elapsed().as_nanos(),
                )?;
                return Err(error);
            }
        };
        let seed_parse_start = Instant::now();
        let changed_candidates_result = live_candidate_seeds_from_py(changed_candidates);
        let seed_parse_ns = seed_parse_start.elapsed().as_nanos();
        let changed_candidates = match changed_candidates_result {
            Ok(changed_candidates) => changed_candidates,
            Err(error) => {
                self.record_failed_live_reconcile(
                    token,
                    card_id_parse_ns,
                    seed_parse_ns,
                    0,
                    native_total_start.elapsed().as_nanos(),
                )?;
                return Err(error);
            }
        };
        let scope_validation_start = Instant::now();
        let scope_result = self.loaded_scope.as_ref().map_or(Ok(()), |scope| {
            py.allow_threads(|| validate_live_seed_scope(scope, &changed_candidates))
        });
        let scope_validation_ns = scope_validation_start.elapsed().as_nanos();
        if let Err(error) = scope_result {
            self.record_failed_live_reconcile(
                token,
                card_id_parse_ns,
                seed_parse_ns,
                scope_validation_ns,
                native_total_start.elapsed().as_nanos(),
            )?;
            return Err(error);
        }
        let Self {
            live_session,
            rnn,
            deterministic,
            undo_stack,
            ..
        } = self;
        let state = live_session.as_mut().expect("live session checked above");
        let live_undo = undo_stack
            .back_mut()
            .and_then(|frame| frame.live.as_mut())
            .filter(|frame| frame.session_token == token);
        let output_result = py.allow_threads(|| {
            reconcile_live_session_membership(
                state,
                &card_ids,
                &changed_candidates,
                rnn,
                deterministic,
                target_timestamp_seconds,
                target_day_offset,
                select_limit,
                &exclude_card_ids,
                retention_extra,
                live_undo,
            )
        });
        let output = match output_result {
            Ok(output) => output,
            Err(error) => {
                state.record_failed_reconcile(LiveRefreshProfile {
                    card_id_parse_ns,
                    seed_parse_ns,
                    scope_validation_ns,
                    native_total_ns: native_total_start.elapsed().as_nanos(),
                    ..LiveRefreshProfile::default()
                });
                return Err(error);
            }
        };
        state.record_reconcile_boundary_ns(
            card_id_parse_ns,
            seed_parse_ns,
            scope_validation_ns,
            native_total_start.elapsed().as_nanos(),
        );
        let result_start = Instant::now();
        let result = live_refresh_output_to_pydict(py, output)?;
        state.record_reconcile_python_result_ns(result_start.elapsed().as_nanos());
        Ok(result)
    }

    fn live_set_retention_extra(&mut self, token: u64, value: f64) -> PyResult<u64> {
        self.live_session_ref(token)?;
        let (live_session, undo_stack) = (&mut self.live_session, &mut self.undo_stack);
        let state = live_session.as_mut().expect("live session checked above");
        let undo = undo_stack
            .back_mut()
            .and_then(|frame| frame.live.as_mut())
            .filter(|frame| frame.session_token == token);
        state.set_retention_extra(value, undo)
    }

    fn live_set_mode(&mut self, token: u64, mode: &str) -> PyResult<u64> {
        let mode = LivePredictionMode::parse(mode)?;
        let generation = self.live_session_mut(token)?.set_mode(mode);
        if mode != LivePredictionMode::Gpu {
            self.rnn.invalidate_gpu();
        }
        Ok(generation)
    }

    fn live_candidate<'py>(
        &self,
        py: Python<'py>,
        token: u64,
        card_id: i64,
    ) -> PyResult<Option<Py<PyDict>>> {
        self.live_session_ref(token)?
            .candidate(card_id)
            .map(|candidate| live_candidate_snapshot_to_pydict(py, &candidate))
            .transpose()
    }

    fn live_snapshot<'py>(&self, py: Python<'py>, token: u64) -> PyResult<Py<PyList>> {
        let values = PyList::empty_bound(py);
        for candidate in self.live_session_ref(token)?.snapshot() {
            values.append(live_candidate_snapshot_to_pydict(py, &candidate)?)?;
        }
        Ok(values.unbind())
    }

    fn live_info<'py>(&self, py: Python<'py>, token: u64) -> PyResult<Py<PyDict>> {
        let state = self.live_session_ref(token)?;
        let dict = PyDict::new_bound(py);
        dict.set_item("token", token)?;
        dict.set_item("generation", state.generation())?;
        dict.set_item("order", state.order().as_str())?;
        dict.set_item("mode", state.mode().as_str())?;
        dict.set_item("batch_size", state.batch_size())?;
        dict.set_item("refresh_limit", state.refresh_limit())?;
        dict.set_item("num_threads", state.num_threads())?;
        dict.set_item("retention_extra", state.retention_extra())?;
        Ok(dict.unbind())
    }

    fn live_profile<'py>(&self, py: Python<'py>, token: u64) -> PyResult<Py<PyDict>> {
        let profile = &self.live_session_ref(token)?.profile;
        let dict = PyDict::new_bound(py);
        dict.set_item("enabled", profile.enabled)?;
        dict.set_item("refresh_calls", profile.refresh_calls)?;
        dict.set_item("failed_refresh_calls", profile.failed_refresh_calls)?;
        dict.set_item("last", live_refresh_profile_to_pydict(py, profile.last)?)?;
        dict.set_item(
            "cumulative",
            live_refresh_profile_to_pydict(py, profile.cumulative)?,
        )?;
        dict.set_item("reconcile_calls", profile.reconcile_calls)?;
        dict.set_item("failed_reconcile_calls", profile.failed_reconcile_calls)?;
        dict.set_item(
            "reconcile_last",
            live_refresh_profile_to_pydict(py, profile.reconcile_last)?,
        )?;
        dict.set_item(
            "reconcile_cumulative",
            live_refresh_profile_to_pydict(py, profile.reconcile_cumulative)?,
        )?;
        dict.set_item(
            "failed_reconcile_last",
            live_refresh_profile_to_pydict(py, profile.failed_reconcile_last)?,
        )?;
        dict.set_item(
            "failed_reconcile_cumulative",
            live_refresh_profile_to_pydict(py, profile.failed_reconcile_cumulative)?,
        )?;
        Ok(dict.unbind())
    }

    fn live_allocation_profile<'py>(&self, py: Python<'py>, token: u64) -> PyResult<Py<PyDict>> {
        let state = self.live_session_ref(token)?;
        let mut live_undo_frame_count = 0usize;
        let mut reconciliation_snapshot_count = 0usize;
        let mut reconciliation_snapshot_candidate_count = 0usize;
        let mut reconciliation_snapshot_tracked_capacity_bytes = 0usize;
        for frame in &self.undo_stack {
            let Some(live) = frame
                .live
                .as_ref()
                .filter(|live| live.session_token == token)
            else {
                continue;
            };
            live_undo_frame_count += 1;
            if !live.has_reconciliation_snapshot() {
                continue;
            }
            reconciliation_snapshot_count += 1;
            let candidate_count = live.reconciliation_snapshot_candidate_count();
            reconciliation_snapshot_candidate_count =
                reconciliation_snapshot_candidate_count.saturating_add(candidate_count);
            reconciliation_snapshot_tracked_capacity_bytes =
                reconciliation_snapshot_tracked_capacity_bytes
                    .saturating_add(live.reconciliation_snapshot_tracked_capacity_bytes());
        }

        let dict = PyDict::new_bound(py);
        dict.set_item("active_candidate_count", state.candidate_count())?;
        dict.set_item(
            "active_candidate_tracked_capacity_bytes",
            state.tracked_candidate_capacity_bytes(),
        )?;
        dict.set_item("live_undo_frame_count", live_undo_frame_count)?;
        dict.set_item(
            "reconciliation_snapshot_count",
            reconciliation_snapshot_count,
        )?;
        dict.set_item(
            "reconciliation_snapshot_candidate_count",
            reconciliation_snapshot_candidate_count,
        )?;
        dict.set_item(
            "reconciliation_snapshot_tracked_capacity_bytes",
            reconciliation_snapshot_tracked_capacity_bytes,
        )?;
        Ok(dict.unbind())
    }

    fn live_last_refresh_debug<'py>(&self, py: Python<'py>, token: u64) -> PyResult<Py<PyDict>> {
        let (membership, transport) = self.live_session_ref(token)?.last_refresh_debug();
        let dict = PyDict::new_bound(py);
        dict.set_item("membership_card_ids", membership)?;
        dict.set_item("transport_card_ids", transport)?;
        Ok(dict.unbind())
    }

    fn close_live_prediction_session(&mut self, token: u64) -> PyResult<bool> {
        self.live_session_ref(token)?;
        self.live_session_mut(token)?.advance_generation();
        self.live_session = None;
        self.undo_stack.clear();
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (card_features, card_id, note_id, deck_id, preset_id, skip, return_curve=true))]
    fn review(
        &mut self,
        card_features: Tensor2List,
        card_id: i64,
        note_id: i64,
        deck_id: i64,
        preset_id: i64,
        skip: bool,
        return_curve: bool,
    ) -> PyResult<NativeReviewPyOutput> {
        if !skip {
            self.require_no_live_session("review")?;
            self.undo_stack.clear();
        }
        self.rnn.review(
            card_features,
            card_id,
            note_id,
            deck_id,
            preset_id,
            skip,
            return_curve,
        )
    }

    #[pyo3(signature = (reviews, return_curves=false, num_threads=None))]
    fn process_reviews(
        &mut self,
        reviews: &Bound<'_, PyAny>,
        return_curves: bool,
        num_threads: Option<usize>,
    ) -> PyResult<NativeProcessManyPyOutput> {
        self.require_no_live_session("process_reviews")?;
        self.undo_stack.clear();
        process_reviews_with_state(
            &mut self.rnn,
            &mut self.deterministic,
            reviews,
            return_curves,
            num_threads,
            None,
        )
    }

    #[pyo3(signature = (batch, return_curves=false, num_threads=None))]
    fn process_review_batch(
        &mut self,
        batch: PyRef<'_, NativeReviewBatch>,
        return_curves: bool,
        num_threads: Option<usize>,
    ) -> PyResult<NativeProcessManyPyOutput> {
        self.require_no_live_session("process_review_batch")?;
        require_loaded_batch_scope(self.loaded_scope.as_ref(), &batch)?;
        self.undo_stack.clear();
        let mut profile = None;
        process_review_inputs_with_state(
            &mut self.rnn,
            &mut self.deterministic,
            batch.inputs(),
            return_curves,
            num_threads,
            &mut profile,
        )
    }

    #[pyo3(signature = (payload, return_curves=false, num_threads=None))]
    fn process_reviews_packed(
        &mut self,
        payload: Vec<u8>,
        return_curves: bool,
        num_threads: Option<usize>,
    ) -> PyResult<NativeProcessManyPyOutput> {
        self.require_no_live_session("process_reviews_packed")?;
        self.undo_stack.clear();
        process_packed_reviews_with_state(
            &mut self.rnn,
            &mut self.deterministic,
            &payload,
            return_curves,
            num_threads,
            None,
        )
    }

    #[pyo3(signature = (payload, num_threads=None))]
    fn build_state_only_packed(
        &mut self,
        payload: Vec<u8>,
        num_threads: Option<usize>,
    ) -> PyResult<usize> {
        self.require_no_live_session("build_state_only")?;
        self.undo_stack.clear();
        build_state_only_packed_with_state(
            &mut self.rnn,
            &mut self.deterministic,
            &payload,
            num_threads,
            None,
        )
    }

    #[pyo3(signature = (batch, num_threads=None))]
    fn build_state_only_batch(
        &mut self,
        batch: PyRef<'_, NativeReviewBatch>,
        num_threads: Option<usize>,
    ) -> PyResult<usize> {
        self.require_no_live_session("build_state_only_batch")?;
        require_loaded_batch_scope(self.loaded_scope.as_ref(), &batch)?;
        self.undo_stack.clear();
        let mut profile = None;
        build_state_only_inputs_with_state(
            &mut self.rnn,
            &mut self.deterministic,
            batch.inputs(),
            num_threads,
            &mut profile,
        )
    }

    #[pyo3(signature = (payload, num_threads=None))]
    fn build_state_only_pipeline(
        &mut self,
        payload: Vec<u8>,
        num_threads: Option<usize>,
    ) -> PyResult<usize> {
        self.require_no_live_session("build_state_only")?;
        self.undo_stack.clear();
        build_state_only_pipeline_with_state(
            &mut self.rnn,
            &mut self.deterministic,
            &payload,
            num_threads,
            None,
        )
    }

    #[pyo3(signature = (batch, num_threads=None))]
    fn build_state_only_batch_pipeline(
        &mut self,
        batch: PyRef<'_, NativeReviewBatch>,
        num_threads: Option<usize>,
    ) -> PyResult<usize> {
        self.require_no_live_session("build_state_only_batch_pipeline")?;
        require_loaded_batch_scope(self.loaded_scope.as_ref(), &batch)?;
        self.undo_stack.clear();
        build_state_only_inputs_pipeline_with_state(
            &mut self.rnn,
            &mut self.deterministic,
            batch.inputs(),
            num_threads,
            None,
        )
    }

    #[pyo3(signature = (payload, return_curves=false, num_threads=None))]
    fn process_reviews_bulk_reference(
        &mut self,
        payload: Vec<u8>,
        return_curves: bool,
        num_threads: Option<usize>,
    ) -> PyResult<NativeProcessManyPyOutput> {
        self.require_no_live_session("process_reviews_bulk_reference")?;
        self.undo_stack.clear();
        process_reviews_bulk_reference_with_state(
            &mut self.rnn,
            &mut self.deterministic,
            &payload,
            return_curves,
            num_threads,
            None,
        )
    }

    #[pyo3(signature = (batch, return_curves=false, num_threads=None))]
    fn process_review_batch_bulk_reference(
        &mut self,
        batch: PyRef<'_, NativeReviewBatch>,
        return_curves: bool,
        num_threads: Option<usize>,
    ) -> PyResult<NativeProcessManyPyOutput> {
        self.require_no_live_session("process_review_batch_bulk_reference")?;
        require_loaded_batch_scope(self.loaded_scope.as_ref(), &batch)?;
        self.undo_stack.clear();
        let payload = batch.to_packed_payload();
        process_reviews_bulk_reference_with_state(
            &mut self.rnn,
            &mut self.deterministic,
            &payload,
            return_curves,
            num_threads,
            None,
        )
    }

    #[pyo3(signature = (payload, return_curves=false, num_threads=None))]
    fn process_reviews_bulk_layered(
        &mut self,
        payload: Vec<u8>,
        return_curves: bool,
        num_threads: Option<usize>,
    ) -> PyResult<NativeProcessManyPyOutput> {
        self.require_no_live_session("process_reviews_bulk_layered")?;
        self.undo_stack.clear();
        process_reviews_bulk_layered_with_state(
            &mut self.rnn,
            &mut self.deterministic,
            &payload,
            return_curves,
            num_threads,
            None,
        )
    }

    #[pyo3(signature = (batch, return_curves=false, num_threads=None))]
    fn process_review_batch_bulk_layered(
        &mut self,
        batch: PyRef<'_, NativeReviewBatch>,
        return_curves: bool,
        num_threads: Option<usize>,
    ) -> PyResult<NativeProcessManyPyOutput> {
        self.require_no_live_session("process_review_batch_bulk_layered")?;
        require_loaded_batch_scope(self.loaded_scope.as_ref(), &batch)?;
        self.undo_stack.clear();
        let payload = batch.to_packed_payload();
        process_reviews_bulk_layered_with_state(
            &mut self.rnn,
            &mut self.deterministic,
            &payload,
            return_curves,
            num_threads,
            None,
        )
    }

    #[pyo3(signature = (payload, return_curves=false, num_threads=None))]
    fn process_reviews_pipeline(
        &mut self,
        payload: Vec<u8>,
        return_curves: bool,
        num_threads: Option<usize>,
    ) -> PyResult<NativeProcessManyPyOutput> {
        self.require_no_live_session("process_reviews_pipeline")?;
        self.undo_stack.clear();
        process_reviews_pipeline_with_state(
            &mut self.rnn,
            &mut self.deterministic,
            &payload,
            return_curves,
            num_threads,
        )
    }

    #[pyo3(signature = (batch, return_curves=false, num_threads=None))]
    fn process_review_batch_pipeline(
        &mut self,
        batch: PyRef<'_, NativeReviewBatch>,
        return_curves: bool,
        num_threads: Option<usize>,
    ) -> PyResult<NativeProcessManyPyOutput> {
        self.require_no_live_session("process_review_batch_pipeline")?;
        require_loaded_batch_scope(self.loaded_scope.as_ref(), &batch)?;
        self.undo_stack.clear();
        process_review_inputs_pipeline_with_state(
            &mut self.rnn,
            &mut self.deterministic,
            batch.inputs(),
            return_curves,
            num_threads,
        )
    }

    #[pyo3(signature = (payload, return_curves=false, num_threads=None, defer_cpu_state=false))]
    fn process_reviews_gpu_scan<'py>(
        &mut self,
        py: Python<'py>,
        payload: Vec<u8>,
        return_curves: bool,
        num_threads: Option<usize>,
        defer_cpu_state: bool,
    ) -> PyResult<NativeGpuScanPyOutput> {
        self.require_no_live_session("process_reviews_gpu_scan")?;
        self.undo_stack.clear();
        self.gpu_process_committed_rows = 0;
        process_reviews_gpu_scan_with_state(
            &mut self.rnn,
            &mut self.deterministic,
            &payload,
            return_curves,
            num_threads,
            defer_cpu_state,
            &mut self.gpu_process_committed_rows,
            &mut self.gpu_process_output,
        )?;
        let (probabilities, ahead, weights) =
            std::mem::replace(&mut self.gpu_process_output, (Vec::new(), None, None));
        let ahead =
            ahead.map(|values| PyBytes::new_bound(py, bytemuck::cast_slice(&values)).unbind());
        let weights =
            weights.map(|values| PyBytes::new_bound(py, bytemuck::cast_slice(&values)).unbind());
        Ok((probabilities, ahead, weights))
    }

    #[pyo3(signature = (batch, return_curves=false, num_threads=None, defer_cpu_state=false, fully_resident_state=false))]
    fn process_review_batch_gpu_scan<'py>(
        &mut self,
        py: Python<'py>,
        batch: PyRef<'_, NativeReviewBatch>,
        return_curves: bool,
        num_threads: Option<usize>,
        defer_cpu_state: bool,
        fully_resident_state: bool,
    ) -> PyResult<NativeGpuScanPyOutput> {
        self.require_no_live_session("process_review_batch_gpu_scan")?;
        self.undo_stack.clear();
        self.gpu_process_committed_rows = 0;
        self.gpu_process_output = (
            Vec::new(),
            return_curves.then(Vec::new),
            return_curves.then(Vec::new),
        );
        require_loaded_batch_scope(self.loaded_scope.as_ref(), &batch)?;
        process_review_inputs_gpu_scan_with_state(
            &mut self.rnn,
            &mut self.deterministic,
            batch.inputs(),
            return_curves,
            num_threads,
            defer_cpu_state,
            fully_resident_state,
            &mut self.gpu_process_committed_rows,
            &mut self.gpu_process_output,
        )?;
        let (probabilities, ahead, weights) =
            std::mem::replace(&mut self.gpu_process_output, (Vec::new(), None, None));
        let ahead =
            ahead.map(|values| PyBytes::new_bound(py, bytemuck::cast_slice(&values)).unbind());
        let weights =
            weights.map(|values| PyBytes::new_bound(py, bytemuck::cast_slice(&values)).unbind());
        Ok((probabilities, ahead, weights))
    }

    #[pyo3(signature = (batch, num_threads=None, defer_cpu_state=false, fully_resident_state=false))]
    fn build_state_only_batch_gpu_scan(
        &mut self,
        batch: PyRef<'_, NativeReviewBatch>,
        num_threads: Option<usize>,
        defer_cpu_state: bool,
        fully_resident_state: bool,
    ) -> PyResult<usize> {
        self.require_no_live_session("build_state_only_batch_gpu_scan")?;
        self.undo_stack.clear();
        self.gpu_process_committed_rows = 0;
        self.gpu_process_output = (Vec::new(), None, None);
        require_loaded_batch_scope(self.loaded_scope.as_ref(), &batch)?;
        build_state_only_review_inputs_gpu_scan_with_state(
            &mut self.rnn,
            &mut self.deterministic,
            batch.inputs(),
            num_threads,
            defer_cpu_state,
            fully_resident_state,
            &mut self.gpu_process_committed_rows,
        )?;
        Ok(self.gpu_process_committed_rows)
    }

    fn take_gpu_process_progress<'py>(
        &mut self,
        py: Python<'py>,
    ) -> NativeGpuProcessProgressPyOutput {
        let committed_rows = std::mem::take(&mut self.gpu_process_committed_rows);
        let (probabilities, ahead, weights) =
            std::mem::replace(&mut self.gpu_process_output, (Vec::new(), None, None));
        let ahead =
            ahead.map(|values| PyBytes::new_bound(py, bytemuck::cast_slice(&values)).unbind());
        let weights =
            weights.map(|values| PyBytes::new_bound(py, bytemuck::cast_slice(&values)).unbind());
        (committed_rows, probabilities, ahead, weights)
    }

    fn take_gpu_process_committed_rows(&mut self) -> usize {
        std::mem::take(&mut self.gpu_process_committed_rows)
    }

    fn synchronize_gpu_process_state(&mut self) -> PyResult<u128> {
        synchronize_gpu_process_state_with_state(&mut self.rnn).map_err(py_gpu_error)
    }

    #[pyo3(signature = (reviews, return_curves=false, num_threads=None))]
    fn process_reviews_profiled<'py>(
        &mut self,
        py: Python<'py>,
        reviews: &Bound<'_, PyAny>,
        return_curves: bool,
        num_threads: Option<usize>,
    ) -> PyResult<NativeProcessManyProfiledPyOutput> {
        self.require_no_live_session("process_reviews_profiled")?;
        self.undo_stack.clear();
        let total_start = ProfileTimer::start(true);
        let mut profile = RuntimeProfile::default();
        let (prediction_probabilities, curve_ahead_logits, curve_w) = process_reviews_with_state(
            &mut self.rnn,
            &mut self.deterministic,
            reviews,
            return_curves,
            num_threads,
            Some(&mut profile),
        )?;
        profile.total_ns = total_start.elapsed_ns();
        Ok((
            prediction_probabilities,
            curve_ahead_logits,
            curve_w,
            profile.to_pydict(py)?,
        ))
    }

    #[pyo3(signature = (payload, return_curves=false, num_threads=None))]
    fn process_reviews_packed_profiled<'py>(
        &mut self,
        py: Python<'py>,
        payload: Vec<u8>,
        return_curves: bool,
        num_threads: Option<usize>,
    ) -> PyResult<NativeProcessManyProfiledPyOutput> {
        self.require_no_live_session("process_reviews_packed_profiled")?;
        self.undo_stack.clear();
        let total_start = ProfileTimer::start(true);
        let mut profile = RuntimeProfile::default();
        let (prediction_probabilities, curve_ahead_logits, curve_w) =
            process_packed_reviews_with_state(
                &mut self.rnn,
                &mut self.deterministic,
                &payload,
                return_curves,
                num_threads,
                Some(&mut profile),
            )?;
        profile.total_ns = total_start.elapsed_ns();
        Ok((
            prediction_probabilities,
            curve_ahead_logits,
            curve_w,
            profile.to_pydict(py)?,
        ))
    }

    #[pyo3(signature = (payload, num_threads=None))]
    fn build_state_only_packed_profiled<'py>(
        &mut self,
        py: Python<'py>,
        payload: Vec<u8>,
        num_threads: Option<usize>,
    ) -> PyResult<NativeStateOnlyProfiledPyOutput> {
        self.require_no_live_session("build_state_only_packed_profiled")?;
        self.undo_stack.clear();
        let total_start = ProfileTimer::start(true);
        let mut profile = RuntimeProfile::default();
        let processed_count = build_state_only_packed_with_state(
            &mut self.rnn,
            &mut self.deterministic,
            &payload,
            num_threads,
            Some(&mut profile),
        )?;
        profile.total_ns = total_start.elapsed_ns();
        let profile_dict = profile.to_pydict(py)?;
        {
            let dict = profile_dict.bind(py);
            dict.set_item("query_rows", 0usize)?;
            dict.set_item("process_rows", processed_count)?;
            dict.set_item("prediction_head_rows", 0usize)?;
            dict.set_item("curve_head_rows", 0usize)?;
            dict.set_item("completed_rows", processed_count)?;
        }
        Ok((processed_count, profile_dict))
    }

    #[pyo3(signature = (payload, num_threads=None))]
    fn build_state_only_pipeline_profiled<'py>(
        &mut self,
        py: Python<'py>,
        payload: Vec<u8>,
        num_threads: Option<usize>,
    ) -> PyResult<NativeStateOnlyProfiledPyOutput> {
        self.require_no_live_session("build_state_only_pipeline_profiled")?;
        self.undo_stack.clear();
        let mut profile = StateBuildProfile::default();
        let processed_count = build_state_only_pipeline_with_state(
            &mut self.rnn,
            &mut self.deterministic,
            &payload,
            num_threads,
            Some(&mut profile),
        )?;
        Ok((processed_count, profile.to_pydict(py)?))
    }

    #[pyo3(signature = (payload, return_curves=false, num_threads=None))]
    fn process_reviews_bulk_reference_profiled<'py>(
        &mut self,
        py: Python<'py>,
        payload: Vec<u8>,
        return_curves: bool,
        num_threads: Option<usize>,
    ) -> PyResult<NativeProcessManyProfiledPyOutput> {
        self.require_no_live_session("process_reviews_bulk_reference_profiled")?;
        self.undo_stack.clear();
        let total_start = ProfileTimer::start(true);
        let mut profile = RuntimeProfile::default();
        let (prediction_probabilities, curve_ahead_logits, curve_w) =
            process_reviews_bulk_reference_with_state(
                &mut self.rnn,
                &mut self.deterministic,
                &payload,
                return_curves,
                num_threads,
                Some(&mut profile),
            )?;
        profile.total_ns = total_start.elapsed_ns();
        Ok((
            prediction_probabilities,
            curve_ahead_logits,
            curve_w,
            profile.to_pydict(py)?,
        ))
    }

    #[pyo3(signature = (payload, return_curves=false, num_threads=None))]
    fn process_reviews_bulk_layered_profiled<'py>(
        &mut self,
        py: Python<'py>,
        payload: Vec<u8>,
        return_curves: bool,
        num_threads: Option<usize>,
    ) -> PyResult<NativeProcessManyProfiledPyOutput> {
        self.require_no_live_session("process_reviews_bulk_layered_profiled")?;
        self.undo_stack.clear();
        let total_start = ProfileTimer::start(true);
        let mut profile = RuntimeProfile::default();
        let (prediction_probabilities, curve_ahead_logits, curve_w) =
            process_reviews_bulk_layered_with_state(
                &mut self.rnn,
                &mut self.deterministic,
                &payload,
                return_curves,
                num_threads,
                Some(&mut profile),
            )?;
        profile.total_ns = total_start.elapsed_ns();
        Ok((
            prediction_probabilities,
            curve_ahead_logits,
            curve_w,
            profile.to_pydict(py)?,
        ))
    }

    fn process_reviews_bulk_feature_prepass_debug<'py>(
        &self,
        py: Python<'py>,
        payload: Vec<u8>,
    ) -> PyResult<Py<PyDict>> {
        let (prepass, deterministic) =
            process_reviews_bulk_feature_prepass_debug(&self.deterministic, &payload)?;
        let dict = PyDict::new_bound(py);
        dict.set_item("ids", prepass.ids)?;
        dict.set_item("predict_features", prepass.predict_features)?;
        dict.set_item("process_features", prepass.process_features)?;
        dict.set_item("snapshot", feature_state_snapshot(py, &deterministic)?)?;
        dict.set_item(
            "recurrent_state_keys",
            feature_state_recurrent_state_snapshot(py, &deterministic)?,
        )?;
        dict.set_item(
            "id_encodings",
            feature_state_id_encoding_snapshot(py, &deterministic)?,
        )?;
        dict.set_item(
            "torch_rng_state",
            deterministic.id_rng.to_torch_rng_state_bytes(),
        )?;
        Ok(dict.unbind())
    }

    fn process_reviews_bulk_stream_plan_debug<'py>(
        &self,
        py: Python<'py>,
        payload: Vec<u8>,
    ) -> PyResult<Py<PyDict>> {
        let (prepass, stream_plan, deterministic) =
            process_reviews_bulk_stream_plan_debug(&self.rnn, &self.deterministic, &payload)?;
        let dict = PyDict::new_bound(py);
        dict.set_item("ids", prepass.ids)?;
        dict.set_item("modules", bulk_stream_modules_to_pylist(py, &stream_plan)?)?;
        dict.set_item(
            "final_recurrent_state_keys",
            bulk_stream_state_key_plan_to_pydict(py, &stream_plan.final_recurrent_state_keys)?,
        )?;
        dict.set_item("snapshot", feature_state_snapshot(py, &deterministic)?)?;
        dict.set_item(
            "recurrent_state_keys",
            feature_state_recurrent_state_snapshot(py, &deterministic)?,
        )?;
        Ok(dict.unbind())
    }

    fn process_review_feature_step_debug<'py>(
        &mut self,
        py: Python<'py>,
        review: &Bound<'_, PyMapping>,
    ) -> PyResult<Py<PyDict>> {
        self.require_no_live_session("process_review_feature_step_debug")?;
        let input = review_from_mapping(review, true)?;
        let mut profile = None;
        let step = feature_prepass_step(&mut self.deterministic, &input, &mut profile)?;

        let dict = PyDict::new_bound(py);
        dict.set_item("ids", step.ids)?;
        dict.set_item("predict_features", step.predict_features)?;
        dict.set_item("process_features", step.process_features)?;
        Ok(dict.unbind())
    }

    #[pyo3(signature = (reviews, batch_size=80, num_threads=None, lightning=false))]
    fn predict_reviews_profiled<'py>(
        &mut self,
        py: Python<'py>,
        reviews: &Bound<'_, PyAny>,
        batch_size: usize,
        num_threads: Option<usize>,
        lightning: bool,
    ) -> PyResult<NativePredictManyProfiledPyOutput> {
        let total_start = ProfileTimer::start(true);
        let mut profile = RuntimeProfile::default();
        let prediction_probabilities = predict_reviews_with_state(
            &mut self.rnn,
            &mut self.deterministic,
            reviews,
            batch_size,
            num_threads,
            lightning,
            Some(&mut profile),
        )?;
        profile.total_ns = total_start.elapsed_ns();
        Ok((prediction_probabilities, profile.to_pydict(py)?))
    }

    #[pyo3(signature = (reviews, batch_size=96, num_threads=None))]
    fn predict_reviews_fast_caller_profiled<'py>(
        &mut self,
        py: Python<'py>,
        reviews: &Bound<'_, PyAny>,
        batch_size: usize,
        num_threads: Option<usize>,
    ) -> PyResult<NativePredictManyProfiledPyOutput> {
        let mut inputs = Vec::new();
        let parse_start = Instant::now();
        for review in reviews.iter()? {
            let review = review?;
            let review = review.downcast::<PyMapping>()?;
            inputs.push(review_from_mapping(review, false)?);
        }
        let input_parse_ns = parse_start.elapsed().as_nanos();
        let scope_start = Instant::now();
        require_loaded_inputs_scope(self.loaded_scope.as_ref(), &inputs)?;
        let scope_validation_ns = scope_start.elapsed().as_nanos();
        let (prediction_probabilities, profile) = predict_review_inputs_cpu_fast_caller_profiled(
            &mut self.rnn,
            &mut self.deterministic,
            &inputs,
            batch_size,
            num_threads,
            input_parse_ns,
            scope_validation_ns,
        )?;
        Ok((prediction_probabilities, profile.to_pydict(py)?))
    }

    #[pyo3(signature = (batch, batch_size=96, num_threads=None))]
    fn predict_review_batch_fast_caller_profiled<'py>(
        &mut self,
        py: Python<'py>,
        batch: PyRef<'_, NativePredictionBatch>,
        batch_size: usize,
        num_threads: Option<usize>,
    ) -> PyResult<NativePredictManyProfiledPyOutput> {
        let batch = batch.clone();
        let (prediction_probabilities, profile) = py.allow_threads(|| {
            profile_prediction_batch_fast(
                &mut self.rnn,
                &mut self.deterministic,
                self.loaded_scope.as_ref(),
                &batch,
                batch_size,
                num_threads,
            )
        })?;
        Ok((prediction_probabilities, profile.to_pydict(py)?))
    }

    #[pyo3(signature = (batch, batch_size=96, num_threads=None))]
    fn predict_review_batch_fast_list_transport_profiled(
        &mut self,
        py: Python<'_>,
        batch: PyRef<'_, NativePredictionBatch>,
        batch_size: usize,
        num_threads: Option<usize>,
    ) -> PyResult<NativePredictManyListTransportProfiledPyOutput> {
        let batch = batch.clone();
        let (prediction_probabilities, profile) = py.allow_threads(|| {
            profile_prediction_batch_fast(
                &mut self.rnn,
                &mut self.deterministic,
                self.loaded_scope.as_ref(),
                &batch,
                batch_size,
                num_threads,
            )
        })?;
        let result_start = Instant::now();
        let result = PyList::new_bound(py, prediction_probabilities).unbind();
        let binding_result_construction_ns = result_start.elapsed().as_nanos();
        let profile = profile.to_pydict(py)?;
        profile.bind(py).set_item(
            "binding_result_construction_ns",
            binding_result_construction_ns,
        )?;
        Ok((result, profile))
    }

    #[pyo3(signature = (batch, batch_size=96, num_threads=None))]
    fn predict_review_batch_fast_f32_transport_profiled(
        &mut self,
        py: Python<'_>,
        batch: PyRef<'_, NativePredictionBatch>,
        batch_size: usize,
        num_threads: Option<usize>,
    ) -> PyResult<NativePredictManyF32TransportProfiledPyOutput> {
        let batch = batch.clone();
        let (prediction_probabilities, profile) = py.allow_threads(|| {
            profile_prediction_batch_fast(
                &mut self.rnn,
                &mut self.deterministic,
                self.loaded_scope.as_ref(),
                &batch,
                batch_size,
                num_threads,
            )
        })?;
        let result_start = Instant::now();
        let result = probabilities_to_f32_bytes(py, prediction_probabilities).unbind();
        let binding_result_construction_ns = result_start.elapsed().as_nanos();
        let profile = profile.to_pydict(py)?;
        profile.bind(py).set_item(
            "binding_result_construction_ns",
            binding_result_construction_ns,
        )?;
        Ok((result, profile))
    }

    #[pyo3(signature = (reviews, batch_size=80, num_threads=None, lightning=false))]
    fn predict_reviews(
        &mut self,
        reviews: &Bound<'_, PyAny>,
        batch_size: usize,
        num_threads: Option<usize>,
        lightning: bool,
    ) -> PyResult<Vec<f64>> {
        predict_reviews_with_state(
            &mut self.rnn,
            &mut self.deterministic,
            reviews,
            batch_size,
            num_threads,
            lightning,
            None,
        )
    }

    #[pyo3(signature = (batch, batch_size=80, num_threads=None, lightning=false))]
    fn predict_review_batch(
        &mut self,
        py: Python<'_>,
        batch: PyRef<'_, NativePredictionBatch>,
        batch_size: usize,
        num_threads: Option<usize>,
        lightning: bool,
    ) -> PyResult<Vec<f64>> {
        let batch = batch.clone();
        py.allow_threads(|| {
            predict_prediction_batch_cpu(
                &mut self.rnn,
                &mut self.deterministic,
                self.loaded_scope.as_ref(),
                &batch,
                batch_size,
                num_threads,
                lightning,
            )
        })
    }

    #[pyo3(signature = (reviews, batch_size=96, num_threads=None))]
    fn predict_reviews_fast_f32<'py>(
        &mut self,
        py: Python<'py>,
        reviews: &Bound<'_, PyAny>,
        batch_size: usize,
        num_threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let predictions = predict_reviews_with_state(
            &mut self.rnn,
            &mut self.deterministic,
            reviews,
            batch_size,
            num_threads,
            true,
            None,
        )?;
        Ok(probabilities_to_f32_bytes(py, predictions))
    }

    #[pyo3(signature = (batch, batch_size=96, num_threads=None))]
    fn predict_review_batch_fast_f32<'py>(
        &mut self,
        py: Python<'py>,
        batch: PyRef<'_, NativePredictionBatch>,
        batch_size: usize,
        num_threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let batch = batch.clone();
        let predictions = py.allow_threads(|| {
            predict_prediction_batch_cpu(
                &mut self.rnn,
                &mut self.deterministic,
                self.loaded_scope.as_ref(),
                &batch,
                batch_size,
                num_threads,
                true,
            )
        })?;
        Ok(probabilities_to_f32_bytes(py, predictions))
    }

    #[pyo3(signature = (reviews, batch_size=4096, num_threads=None))]
    fn predict_reviews_gpu<'py>(
        &mut self,
        py: Python<'py>,
        reviews: &Bound<'_, PyAny>,
        batch_size: usize,
        num_threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyList>> {
        let binding_start = Instant::now();
        synchronize_gpu_process_state_with_state(&mut self.rnn).map_err(py_gpu_error)?;
        self.rnn.release_gpu_process_cache();
        let predictions = predict_reviews_gpu_with_state(
            &mut self.rnn,
            &mut self.deterministic,
            self.loaded_scope.as_ref(),
            reviews,
            batch_size,
            num_threads,
        )?;
        let result_start = Instant::now();
        let result = PyList::new_bound(py, predictions);
        let python_result_ns = result_start.elapsed().as_nanos();
        self.rnn
            .record_gpu_python_result(python_result_ns, binding_start.elapsed().as_nanos());
        Ok(result)
    }

    #[pyo3(signature = (reviews, batch_size=4096, num_threads=None))]
    fn predict_reviews_gpu_f32<'py>(
        &mut self,
        py: Python<'py>,
        reviews: &Bound<'_, PyAny>,
        batch_size: usize,
        num_threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let binding_start = Instant::now();
        synchronize_gpu_process_state_with_state(&mut self.rnn).map_err(py_gpu_error)?;
        self.rnn.release_gpu_process_cache();
        let predictions = predict_reviews_gpu_with_state(
            &mut self.rnn,
            &mut self.deterministic,
            self.loaded_scope.as_ref(),
            reviews,
            batch_size,
            num_threads,
        )?;
        let result_start = Instant::now();
        let result = probabilities_to_f32_bytes(py, predictions);
        let python_result_ns = result_start.elapsed().as_nanos();
        self.rnn
            .record_gpu_python_result(python_result_ns, binding_start.elapsed().as_nanos());
        Ok(result)
    }

    #[pyo3(signature = (batch, batch_size=4096, num_threads=None))]
    fn predict_review_batch_gpu<'py>(
        &mut self,
        py: Python<'py>,
        batch: PyRef<'_, NativePredictionBatch>,
        batch_size: usize,
        num_threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyList>> {
        let binding_start = Instant::now();
        synchronize_gpu_process_state_with_state(&mut self.rnn).map_err(py_gpu_error)?;
        self.rnn.release_gpu_process_cache();
        let batch = batch.clone();
        let predictions = py.allow_threads(|| {
            predict_review_inputs_gpu(
                &mut self.rnn,
                &mut self.deterministic,
                self.loaded_scope.as_ref(),
                batch.inputs(),
                batch_size,
                num_threads,
                0,
            )
        })?;
        let result_start = Instant::now();
        let result = PyList::new_bound(py, predictions);
        let python_result_ns = result_start.elapsed().as_nanos();
        self.rnn
            .record_gpu_python_result(python_result_ns, binding_start.elapsed().as_nanos());
        Ok(result)
    }

    #[pyo3(signature = (batch, batch_size=4096, num_threads=None))]
    fn predict_review_batch_gpu_f32<'py>(
        &mut self,
        py: Python<'py>,
        batch: PyRef<'_, NativePredictionBatch>,
        batch_size: usize,
        num_threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let binding_start = Instant::now();
        synchronize_gpu_process_state_with_state(&mut self.rnn).map_err(py_gpu_error)?;
        self.rnn.release_gpu_process_cache();
        let batch = batch.clone();
        let predictions = py.allow_threads(|| {
            predict_review_inputs_gpu(
                &mut self.rnn,
                &mut self.deterministic,
                self.loaded_scope.as_ref(),
                batch.inputs(),
                batch_size,
                num_threads,
                0,
            )
        })?;
        let result_start = Instant::now();
        let result = probabilities_to_f32_bytes(py, predictions);
        let python_result_ns = result_start.elapsed().as_nanos();
        self.rnn
            .record_gpu_python_result(python_result_ns, binding_start.elapsed().as_nanos());
        Ok(result)
    }

    #[pyo3(signature = (review, num_threads=None))]
    fn predict_probability(
        &mut self,
        review: &Bound<'_, PyMapping>,
        num_threads: Option<usize>,
    ) -> PyResult<f64> {
        predict_probability_with_state(&mut self.rnn, &mut self.deterministic, review, num_threads)
    }

    #[pyo3(signature = (review, return_curves=true, num_threads=None))]
    fn undoable_process_review(
        &mut self,
        review: &Bound<'_, PyMapping>,
        return_curves: bool,
        num_threads: Option<usize>,
    ) -> PyResult<(f64, Option<Tensor2List>, Option<Tensor2List>)> {
        self.require_no_live_session("undoable_process_review")?;
        super::runtime::undoable_process_review_with_state(
            &mut self.rnn,
            &mut self.deterministic,
            &mut self.undo_stack,
            self.undo_limit,
            review,
            return_curves,
            num_threads,
        )
    }

    fn undo_last_process(&mut self) -> PyResult<usize> {
        self.require_no_live_session("undo_last_process")?;
        let undo_frame = self
            .undo_stack
            .pop_back()
            .ok_or_else(|| py_value_error("no undoable process is available"))?;
        undo_frame
            .model
            .restore(&mut self.rnn, &mut self.deterministic);
        Ok(self.undo_stack.len())
    }

    fn clear_undo_history(&mut self) -> PyResult<usize> {
        self.require_no_live_session("clear_undo_history")?;
        self.undo_stack.clear();
        Ok(0)
    }

    fn undo_depth(&self) -> usize {
        self.undo_stack.len()
    }

    fn warm_predict_path(&mut self) -> PyResult<()> {
        self.rnn.warm_predict_path().map_err(py_value_error)
    }

    fn warm_thread_pool(&self, num_threads: usize) -> PyResult<()> {
        warm_rayon_pool(num_threads)
    }

    fn initialize_gpu<'py>(&mut self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        synchronize_gpu_process_state_with_state(&mut self.rnn).map_err(py_gpu_error)?;
        self.rnn.release_gpu_process_cache();
        self.rnn.ensure_gpu().map_err(py_gpu_unavailable_error)?;
        self.rnn
            .gpu_profile_pydict(py)?
            .ok_or_else(|| py_value_error("GPU predictor did not initialize"))
    }

    fn gpu_profile<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyDict>>> {
        self.rnn.gpu_profile_pydict(py)
    }

    fn initialize_gpu_process<'py>(&mut self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        self.rnn
            .ensure_gpu_process_scan()
            .map_err(py_gpu_unavailable_error)?;
        self.rnn
            .gpu_process_profile_pydict(py)?
            .ok_or_else(|| py_value_error("GPU process executor did not initialize"))
    }

    fn gpu_process_profile<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyDict>>> {
        self.rnn.gpu_process_profile_pydict(py)
    }

    fn synchronize_gpu(&mut self) -> PyResult<u128> {
        self.rnn.synchronize_gpu().map_err(py_gpu_error)
    }

    fn release_gpu(&mut self) -> bool {
        self.rnn.release_gpu()
    }

    fn prepare_predict<'py>(
        &self,
        py: Python<'py>,
        review: &Bound<'py, PyMapping>,
    ) -> PyResult<Py<PyDict>> {
        let input = review_from_mapping(review, false)?;
        row_to_pydict(py, &self.deterministic.prepare_predict_row(&input))
    }

    fn prepare_process<'py>(
        &self,
        py: Python<'py>,
        review: &Bound<'py, PyMapping>,
    ) -> PyResult<Py<PyDict>> {
        let input = review_from_mapping(review, true)?;
        let row = self
            .deterministic
            .prepare_process_row(&input)
            .map_err(py_value_error)?;
        row_to_pydict(py, &row)
    }

    fn record_processed(&mut self, row: &Bound<'_, PyMapping>) -> PyResult<()> {
        self.require_no_live_session("record_processed")?;
        self.undo_stack.clear();
        let prepared = prepared_record_fields_from_mapping(row)?;
        self.deterministic
            .record_processed_row(&prepared)
            .map_err(py_value_error)
    }

    #[pyo3(signature = (row, mutate_id_encodings))]
    fn feature_vector(
        &mut self,
        row: &Bound<'_, PyMapping>,
        mutate_id_encodings: bool,
    ) -> PyResult<Vec<f32>> {
        if mutate_id_encodings {
            self.require_no_live_session("feature_vector(mutate_id_encodings=True)")?;
        }
        let prepared = prepared_feature_fields_from_mapping(row)?;
        self.deterministic
            .feature_vector(&prepared, mutate_id_encodings)
            .map_err(py_value_error)
    }

    fn process_feature_vector(&mut self, row: &Bound<'_, PyMapping>) -> PyResult<Vec<f32>> {
        self.require_no_live_session("process_feature_vector")?;
        let prepared = prepared_feature_fields_from_mapping(row)?;
        self.deterministic
            .process_feature_vector(&prepared)
            .map_err(py_value_error)
    }

    fn predict_feature_vector(&mut self, row: &Bound<'_, PyMapping>) -> PyResult<Vec<f32>> {
        let prepared = prepared_feature_fields_from_mapping(row)?;
        self.deterministic
            .predict_feature_vector(&prepared)
            .map_err(py_value_error)
    }

    #[pyo3(signature = (row, skip))]
    fn skip_needs_rng_restore(&self, row: &Bound<'_, PyMapping>, skip: bool) -> PyResult<bool> {
        let prepared = prepared_feature_fields_from_mapping(row)?;
        self.deterministic
            .skip_needs_rng_restore(&prepared, skip)
            .map_err(py_value_error)
    }

    fn can_batch_predict(&self, row: &Bound<'_, PyMapping>) -> PyResult<bool> {
        let prepared = prepared_feature_fields_from_mapping(row)?;
        self.deterministic
            .can_batch_predict(&prepared)
            .map_err(py_value_error)
    }

    fn feature_vectors(&self, rows: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f32>>> {
        let mut prepared_rows = Vec::new();
        for row in rows.iter()? {
            let row = row?;
            let row = row.downcast::<PyMapping>()?;
            prepared_rows.push(prepared_feature_fields_from_mapping(row)?);
        }
        self.deterministic
            .feature_vectors(&prepared_rows)
            .map_err(py_value_error)
    }

    fn record_recurrent_state_update(&mut self, row: &Bound<'_, PyMapping>) -> PyResult<()> {
        self.require_no_live_session("record_recurrent_state_update")?;
        let prepared = prepared_feature_fields_from_mapping(row)?;
        self.deterministic
            .record_recurrent_state_update(&prepared)
            .map_err(py_value_error)
    }

    fn recurrent_state_snapshot<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        feature_state_recurrent_state_snapshot(py, &self.deterministic)
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        feature_state_snapshot(py, &self.deterministic)
    }

    fn restore_snapshot(&mut self, snapshot: &Bound<'_, PyMapping>) -> PyResult<()> {
        self.require_no_live_session("restore_snapshot")?;
        self.undo_stack.clear();
        restore_feature_state_snapshot(&mut self.deterministic, snapshot)
    }

    fn restore_recurrent_state_keys(&mut self, snapshot: &Bound<'_, PyMapping>) -> PyResult<()> {
        self.require_no_live_session("restore_recurrent_state_keys")?;
        self.undo_stack.clear();
        restore_feature_state_recurrent_keys(&mut self.deterministic, snapshot)
    }

    fn restore_id_encoding_snapshot(&mut self, snapshot: &Bound<'_, PyMapping>) -> PyResult<()> {
        self.require_no_live_session("restore_id_encoding_snapshot")?;
        self.undo_stack.clear();
        restore_feature_state_id_encodings(&mut self.deterministic, snapshot)
    }

    fn restore_torch_rng_state(&mut self, rng_state: &Bound<'_, PyAny>) -> PyResult<()> {
        self.require_no_live_session("restore_torch_rng_state")?;
        let bytes = torch_rng_state_bytes(rng_state)?;
        self.deterministic.id_rng =
            TorchMt19937::from_torch_rng_state(&bytes).map_err(py_value_error)?;
        Ok(())
    }

    fn live_restore_torch_rng_state(
        &mut self,
        token: u64,
        rng_state: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.live_session_ref(token)?;
        let bytes = torch_rng_state_bytes(rng_state)?;
        self.deterministic.id_rng =
            TorchMt19937::from_torch_rng_state(&bytes).map_err(py_value_error)?;
        Ok(())
    }

    fn torch_rng_state(&self) -> Vec<u8> {
        self.deterministic.id_rng.to_torch_rng_state_bytes()
    }

    fn id_encoding_snapshot<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        feature_state_id_encoding_snapshot(py, &self.deterministic)
    }

    fn recurrent_state_lists<'py>(&mut self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        self.rnn.recurrent_state_lists(py)
    }

    fn recurrent_state_key_snapshot<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        self.rnn.recurrent_state_key_snapshot(py)
    }

    fn restore_recurrent_state_lists(&mut self, snapshot: &Bound<'_, PyMapping>) -> PyResult<()> {
        self.require_no_live_session("restore_recurrent_state_lists")?;
        self.undo_stack.clear();
        self.rnn.restore_recurrent_state_lists(snapshot)
    }

    fn write_checkpoint_bin(&self, path: PathBuf, metadata_json: Vec<u8>) -> PyResult<()> {
        checkpoint_bin::write_checkpoint_bin_path(
            &path,
            metadata_json,
            &self.deterministic,
            &self.rnn,
        )
        .map_err(py_value_error)
    }

    fn expected_checkpoint_bin_size(
        &self,
        metadata_len: usize,
        card_count: usize,
        note_count: usize,
        deck_count: usize,
        preset_count: usize,
    ) -> PyResult<usize> {
        checkpoint_bin::expected_checkpoint_bin_size(
            &self.rnn,
            metadata_len,
            card_count,
            note_count,
            deck_count,
            preset_count,
        )
        .map_err(py_value_error)
    }

    fn write_merged_checkpoint_bin(
        &self,
        backing_path: PathBuf,
        path: PathBuf,
        metadata_json: Vec<u8>,
        card_ids: Vec<i64>,
        note_ids: Vec<i64>,
        deck_ids: Vec<i64>,
        preset_ids: Vec<i64>,
    ) -> PyResult<()> {
        let scope = checkpoint_bin::CheckpointScope {
            card_ids: card_ids.into_iter().collect(),
            note_ids: note_ids.into_iter().collect(),
            deck_ids: deck_ids.into_iter().collect(),
            preset_ids: preset_ids.into_iter().collect(),
        };
        checkpoint_bin::write_merged_checkpoint_bin_path(
            &backing_path,
            &path,
            metadata_json,
            self,
            &scope,
        )
        .map_err(py_value_error)
    }

    #[pyo3(signature = (path, card_ids=None, note_ids=None, deck_ids=None, preset_ids=None))]
    fn restore_checkpoint_bin(
        &mut self,
        path: PathBuf,
        card_ids: Option<Vec<i64>>,
        note_ids: Option<Vec<i64>>,
        deck_ids: Option<Vec<i64>>,
        preset_ids: Option<Vec<i64>>,
    ) -> PyResult<()> {
        self.require_no_live_session("restore_checkpoint_bin")?;
        let scope = checkpoint_scope_from_optional_identity_lists(
            card_ids, note_ids, deck_ids, preset_ids,
        )?;
        checkpoint_bin::restore_checkpoint_bin_path(&path, self, scope.as_ref())
            .map_err(py_value_error)?;
        self.loaded_scope = scope;
        self.rnn.invalidate_gpu();
        self.undo_stack.clear();
        Ok(())
    }
}

fn live_candidate_seeds_from_py(
    candidates: &Bound<'_, PyAny>,
) -> PyResult<Vec<LiveCandidateSeedNative>> {
    let py = candidates.py();
    let numbers = py.import_bound("numbers")?;
    let real_type = numbers.getattr("Real")?;
    let integral_type = numbers.getattr("Integral")?;
    let capacity = candidates
        .downcast::<PyList>()
        .map_or(0, |candidates| candidates.len());
    let mut seeds = Vec::with_capacity(capacity);
    for (index, candidate) in candidates.iter()?.enumerate() {
        let candidate = candidate?;
        let row_value = candidate.getattr(pyo3::intern!(py, "row"))?;
        let row = live_candidate_seed_review_from_py(&row_value, &integral_type, index)?;
        seeds.push(LiveCandidateSeedNative {
            row,
            target_retrievability: live_candidate_seed_real(
                &candidate.getattr(pyo3::intern!(py, "target_retrievability"))?,
                &real_type,
                index,
                "target_retrievability",
            )?,
            intraday_target_retrievability: live_candidate_seed_real(
                &candidate.getattr(pyo3::intern!(py, "intraday_target_retrievability"))?,
                &real_type,
                index,
                "intraday_target_retrievability",
            )?,
            tie_breaker: live_candidate_seed_tie_breaker(
                &candidate.getattr(pyo3::intern!(py, "tie_breaker"))?,
                &integral_type,
                index,
            )?,
            random_key: live_candidate_seed_u64(
                &candidate.getattr(pyo3::intern!(py, "random_key"))?,
                &integral_type,
                index,
                "random_key",
            )?,
        });
    }
    Ok(seeds)
}

fn live_card_ids_from_py(card_ids: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    let py = card_ids.py();
    let integral_type = py.import_bound("numbers")?.getattr("Integral")?;
    let capacity = card_ids
        .downcast::<PyList>()
        .map_or(0, |card_ids| card_ids.len());
    let mut values = Vec::with_capacity(capacity);
    for (index, value) in card_ids.iter()?.enumerate() {
        let card_id = live_strict_card_id(
            &value?,
            &integral_type,
            LiveCardIdLocation::Membership(index),
        )?;
        values.push(card_id);
    }
    Ok(values)
}

fn validate_live_seed_scope(
    scope: &checkpoint_bin::CheckpointScope,
    seeds: &[LiveCandidateSeedNative],
) -> PyResult<()> {
    for seed in seeds {
        require_loaded_scope(scope, &seed.row)?;
    }
    Ok(())
}

fn live_candidate_seed_review_from_py(
    row_value: &Bound<'_, PyAny>,
    integral_type: &Bound<'_, PyAny>,
    index: usize,
) -> PyResult<crate::state::ReviewInput> {
    if let Ok(row) = row_value.downcast::<PyDict>() {
        return live_candidate_seed_review_from_mapping(
            row_value,
            row.as_mapping(),
            integral_type,
            index,
        );
    }
    if let Ok(mapping) = row_value.downcast::<PyMapping>() {
        return live_candidate_seed_review_from_mapping(row_value, mapping, integral_type, index);
    }
    if row_value.hasattr("to_dict")? {
        let converted = row_value.call_method0("to_dict")?;
        if let Ok(row) = converted.downcast::<PyDict>() {
            return live_candidate_seed_review_from_mapping(
                &converted,
                row.as_mapping(),
                integral_type,
                index,
            );
        }
        if let Ok(mapping) = converted.downcast::<PyMapping>() {
            return live_candidate_seed_review_from_mapping(
                &converted,
                mapping,
                integral_type,
                index,
            );
        }
    }
    Err(PyTypeError::new_err(format!(
        "candidates[{index}].row must be a mapping or expose to_dict()."
    )))
}

fn live_candidate_seed_review_from_mapping(
    row_value: &Bound<'_, PyAny>,
    mapping: &Bound<'_, PyMapping>,
    integral_type: &Bound<'_, PyAny>,
    index: usize,
) -> PyResult<crate::state::ReviewInput> {
    let card_value = if let Ok(row) = row_value.downcast::<PyDict>() {
        row.get_item(pyo3::intern!(row.py(), "card_id"))?
            .ok_or_else(|| py_value_error("Review input is missing required fields: ['card_id']"))?
    } else {
        mapping.get_item("card_id")?
    };
    let card_id = live_strict_card_id(
        &card_value,
        integral_type,
        LiveCardIdLocation::Candidate(index),
    )?;
    if let Ok(row) = row_value.downcast::<PyDict>() {
        predict_review_from_dict_with_card_id(row, card_id)
    } else {
        predict_review_from_mapping_with_card_id(mapping, card_id)
    }
}

#[derive(Clone, Copy)]
enum LiveCardIdLocation {
    Candidate(usize),
    Membership(usize),
}

fn live_strict_card_id(
    value: &Bound<'_, PyAny>,
    integral_type: &Bound<'_, PyAny>,
    location: LiveCardIdLocation,
) -> PyResult<i64> {
    if value.is_instance_of::<PyBool>()
        || !(value.is_instance_of::<PyInt>() || value.is_instance(integral_type)?)
    {
        let message = match location {
            LiveCardIdLocation::Candidate(index) => {
                format!("candidates[{index}].row.card_id must be an integer.")
            }
            LiveCardIdLocation::Membership(index) => {
                format!("card_ids[{index}] must be an integer.")
            }
        };
        return Err(PyTypeError::new_err(message));
    }
    let parsed = value.extract::<i128>();
    let parsed = match parsed {
        Ok(parsed) => parsed,
        Err(_) => return Err(live_card_id_range_error(location)),
    };
    i64::try_from(parsed).map_err(|_| live_card_id_range_error(location))
}

fn live_card_id_range_error(location: LiveCardIdLocation) -> PyErr {
    let message = match location {
        LiveCardIdLocation::Candidate(index) => {
            format!("candidates[{index}].row.card_id must fit in a signed 64-bit integer.")
        }
        LiveCardIdLocation::Membership(index) => {
            format!("card_ids[{index}] must fit in a signed 64-bit integer.")
        }
    };
    py_value_error(message)
}

fn live_candidate_seed_real(
    value: &Bound<'_, PyAny>,
    real_type: &Bound<'_, PyAny>,
    index: usize,
    field: &str,
) -> PyResult<f64> {
    if value.is_instance_of::<PyBool>()
        || !(value.is_instance_of::<PyFloat>()
            || value.is_instance_of::<PyInt>()
            || value.is_instance(real_type)?)
    {
        return Err(PyTypeError::new_err(format!(
            "candidates[{index}].{field} must be a real number."
        )));
    }
    value.extract::<f64>().map_err(|_| {
        PyTypeError::new_err(format!(
            "candidates[{index}].{field} must be a real number."
        ))
    })
}

fn live_candidate_seed_require_integral(
    value: &Bound<'_, PyAny>,
    integral_type: &Bound<'_, PyAny>,
    index: usize,
    field: &str,
) -> PyResult<()> {
    if value.is_instance_of::<PyBool>()
        || !(value.is_instance_of::<PyInt>() || value.is_instance(integral_type)?)
    {
        return Err(PyTypeError::new_err(format!(
            "candidates[{index}].{field} must be an integer."
        )));
    }
    Ok(())
}

fn live_candidate_seed_tie_breaker(
    value: &Bound<'_, PyAny>,
    integral_type: &Bound<'_, PyAny>,
    index: usize,
) -> PyResult<u64> {
    live_candidate_seed_u64(value, integral_type, index, "tie_breaker")
}

fn live_candidate_seed_u64(
    value: &Bound<'_, PyAny>,
    integral_type: &Bound<'_, PyAny>,
    index: usize,
    field: &str,
) -> PyResult<u64> {
    live_candidate_seed_require_integral(value, integral_type, index, field)?;
    let value = value.extract::<i128>().map_err(|_| {
        py_value_error(format!(
            "candidates[{index}].{field} must fit in an unsigned 64-bit integer."
        ))
    })?;
    if value < 0 {
        return Err(py_value_error(format!(
            "candidates[{index}].{field} must be non-negative."
        )));
    }
    u64::try_from(value).map_err(|_| {
        py_value_error(format!(
            "candidates[{index}].{field} must fit in an unsigned 64-bit integer."
        ))
    })
}

fn live_refresh_output_to_pydict<'py>(
    py: Python<'py>,
    output: LiveRefreshOutput,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("generation", output.generation)?;
    dict.set_item("refreshed_count", output.refreshed_count)?;
    dict.set_item("eligible_count", output.eligible_count)?;
    dict.set_item("active_count", output.active_count)?;
    let selected = PyList::empty_bound(py);
    for value in output.selected {
        let item = PyDict::new_bound(py);
        item.set_item("card_id", value.card_id)?;
        item.set_item("retrievability", value.retrievability)?;
        item.set_item("target_retrievability", value.target_retrievability)?;
        selected.append(item)?;
    }
    dict.set_item("selected", selected)?;
    dict.set_item("next_retention_extra", output.next_retention_extra)?;
    Ok(dict.unbind())
}

fn maybe_id_to_option(value: MaybeId) -> Option<i64> {
    match value {
        MaybeId::Present(value) => Some(value),
        MaybeId::Missing => None,
    }
}

fn live_candidate_snapshot_to_pydict<'py>(
    py: Python<'py>,
    candidate: &LiveCandidateSnapshot,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("card_id", candidate.card_id)?;
    dict.set_item("review_id", candidate.review_id)?;
    dict.set_item("note_id", maybe_id_to_option(candidate.note_id))?;
    dict.set_item("deck_id", maybe_id_to_option(candidate.deck_id))?;
    dict.set_item("preset_id", maybe_id_to_option(candidate.preset_id))?;
    dict.set_item("retrievability", candidate.retrievability)?;
    dict.set_item("target_retrievability", candidate.target_retrievability)?;
    dict.set_item(
        "intraday_target_retrievability",
        candidate.intraday_target_retrievability,
    )?;
    dict.set_item(
        "applicable_target_retrievability",
        candidate.applicable_target_retrievability,
    )?;
    dict.set_item("tie_breaker", candidate.tie_breaker)?;
    dict.set_item("random_key", candidate.random_key)?;
    dict.set_item("status", candidate.status.as_str())?;
    dict.set_item("eligible", candidate.eligible)?;
    dict.set_item("has_prior_review", candidate.has_prior_review)?;
    dict.set_item(
        "last_review_timestamp_seconds",
        candidate.last_review_timestamp_seconds,
    )?;
    dict.set_item("last_review_day_offset", candidate.last_review_day_offset)?;
    dict.set_item("query_timestamp_seconds", candidate.query_timestamp_seconds)?;
    dict.set_item("query_day_offset", candidate.query_day_offset)?;
    dict.set_item("elapsed_seconds", candidate.elapsed_seconds)?;
    dict.set_item("elapsed_days", candidate.elapsed_days)?;
    Ok(dict.unbind())
}

fn live_refresh_profile_to_pydict<'py>(
    py: Python<'py>,
    profile: LiveRefreshProfile,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("card_id_parse_ns", profile.card_id_parse_ns)?;
    dict.set_item("seed_parse_ns", profile.seed_parse_ns)?;
    dict.set_item("scope_validation_ns", profile.scope_validation_ns)?;
    dict.set_item("native_total_ns", profile.native_total_ns)?;
    dict.set_item("membership_selection_ns", profile.membership_selection_ns)?;
    dict.set_item("input_construction_ns", profile.input_construction_ns)?;
    dict.set_item("prediction_ns", profile.prediction_ns)?;
    dict.set_item("application_ns", profile.application_ns)?;
    dict.set_item("ordering_ns", profile.ordering_ns)?;
    dict.set_item("commit_snapshot_ns", profile.commit_snapshot_ns)?;
    dict.set_item("final_selection_ns", profile.final_selection_ns)?;
    dict.set_item("python_result_ns", profile.python_result_ns)?;
    dict.set_item("refreshed_count", profile.refreshed_count)?;
    dict.set_item("selected_count", profile.selected_count)?;
    dict.set_item("partial_merge_count", profile.partial_merge_count)?;
    dict.set_item("full_rebuild_count", profile.full_rebuild_count)?;
    Ok(dict.unbind())
}

fn bulk_stream_modules_to_pylist<'py>(
    py: Python<'py>,
    plan: &BulkStreamPlan,
) -> PyResult<Bound<'py, PyList>> {
    let modules = PyList::empty_bound(py);
    for module in &plan.modules {
        modules.append(bulk_module_stream_plan_to_pydict(py, module)?)?;
    }
    Ok(modules)
}

fn bulk_module_stream_plan_to_pydict<'py>(
    py: Python<'py>,
    module: &BulkModuleStreamPlan,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("module", module.kind.name())?;
    dict.set_item("row_to_stream", module.row_to_stream.clone())?;

    let streams = PyList::empty_bound(py);
    for stream in &module.streams {
        let stream_dict = PyDict::new_bound(py);
        stream_dict.set_item("key", stream.key)?;
        stream_dict.set_item("rows", stream.rows.clone())?;
        stream_dict.set_item("starts_existing", stream.starts_existing)?;
        streams.append(stream_dict)?;
    }
    dict.set_item("streams", streams)?;

    Ok(dict.unbind())
}

fn bulk_stream_state_key_plan_to_pydict<'py>(
    py: Python<'py>,
    plan: &BulkStreamStateKeyPlan,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("card_states", plan.card_states.clone())?;
    dict.set_item("note_states", plan.note_states.clone())?;
    dict.set_item("deck_states", plan.deck_states.clone())?;
    dict.set_item("preset_states", plan.preset_states.clone())?;
    dict.set_item("global_state", plan.global_state)?;
    Ok(dict.unbind())
}

#[allow(clippy::useless_conversion)]
#[pyfunction(name = "channel_mixer_forward")]
#[pyo3(signature = (checkpoint_path, module_index, layer_index, in_bc, state_b1c=None))]
pub(crate) fn channel_mixer_forward_py(
    checkpoint_path: PathBuf,
    module_index: usize,
    layer_index: usize,
    in_bc: Tensor2List,
    state_b1c: Option<Tensor3List>,
) -> PyResult<(Tensor2List, Tensor3List)> {
    let weights = load_srs_rwkv_rnn_weights(checkpoint_path).map_err(py_value_error)?;
    let module = weights
        .rwkv_modules
        .get(module_index)
        .ok_or_else(|| py_value_error(format!("unknown RWKV module index {module_index}")))?;
    let block = module.blocks.get(layer_index).ok_or_else(|| {
        py_value_error(format!(
            "unknown layer index {layer_index} for RWKV module {module_index}"
        ))
    })?;

    let in_bc = tensor_from_2d(in_bc, "in_BC").map_err(py_value_error)?;
    let state_b1c = state_b1c
        .map(|state| tensor_from_3d(state, "state_B1C"))
        .transpose()
        .map_err(py_value_error)?;
    let (out_bc, next_state_b1c) =
        channel_mixer_forward(&block.channel_mixer, &in_bc, state_b1c.as_ref())
            .map_err(py_value_error)?;

    Ok((
        out_bc.to_vec2::<f32>().map_err(py_value_error)?,
        next_state_b1c.to_vec3::<f32>().map_err(py_value_error)?,
    ))
}

#[allow(clippy::useless_conversion)]
#[pyfunction(name = "time_mixer_forward")]
#[pyo3(signature = (checkpoint_path, module_index, layer_index, in_bc, v0_bc, x_shift_b1c=None, state_b1hkk=None))]
pub(crate) fn time_mixer_forward_py(
    checkpoint_path: PathBuf,
    module_index: usize,
    layer_index: usize,
    in_bc: Tensor2List,
    v0_bc: Tensor2List,
    x_shift_b1c: Option<Tensor3List>,
    state_b1hkk: Option<Tensor5List>,
) -> PyResult<TimeMixerPyOutput> {
    let weights = load_srs_rwkv_rnn_weights(checkpoint_path).map_err(py_value_error)?;
    let module = weights
        .rwkv_modules
        .get(module_index)
        .ok_or_else(|| py_value_error(format!("unknown RWKV module index {module_index}")))?;
    let block = module.blocks.get(layer_index).ok_or_else(|| {
        py_value_error(format!(
            "unknown layer index {layer_index} for RWKV module {module_index}"
        ))
    })?;

    let in_bc = tensor_from_2d(in_bc, "in_BC").map_err(py_value_error)?;
    let v0_bc = tensor_from_2d(v0_bc, "v0_BC").map_err(py_value_error)?;
    let x_shift_b1c = x_shift_b1c
        .map(|state| tensor_from_3d(state, "x_shift_B1C"))
        .transpose()
        .map_err(py_value_error)?;
    let state_b1hkk = state_b1hkk
        .map(|state| tensor_from_5d(state, "state_B1HKK"))
        .transpose()
        .map_err(py_value_error)?;
    let state = match (x_shift_b1c.as_ref(), state_b1hkk.as_ref()) {
        (Some(x_shift_b1c), Some(state_b1hkk)) => Some((x_shift_b1c, state_b1hkk)),
        (None, None) => None,
        _ => {
            return Err(py_value_error(
                "time_mixer_forward requires both x_shift_b1c and state_b1hkk, or neither",
            ))
        }
    };

    let (out_bc, next_v0_bc, next_x_shift_b1c, next_state_b1hkk) =
        time_mixer_forward(&block.time_mixer, &in_bc, &v0_bc, state).map_err(py_value_error)?;

    Ok((
        out_bc.to_vec2::<f32>().map_err(py_value_error)?,
        next_v0_bc.to_vec2::<f32>().map_err(py_value_error)?,
        next_x_shift_b1c.to_vec3::<f32>().map_err(py_value_error)?,
        tensor_to_vec5(&next_state_b1hkk).map_err(py_value_error)?,
    ))
}

#[allow(clippy::too_many_arguments, clippy::useless_conversion)]
#[pyfunction(name = "rwkv_layer_forward")]
#[pyo3(signature = (checkpoint_path, module_index, layer_index, in_bc, v0_bc, time_x_shift_b1c=None, time_state_b1hkk=None, channel_state_b1c=None))]
pub(crate) fn rwkv_layer_forward_py(
    checkpoint_path: PathBuf,
    module_index: usize,
    layer_index: usize,
    in_bc: Tensor2List,
    v0_bc: Tensor2List,
    time_x_shift_b1c: Option<Tensor3List>,
    time_state_b1hkk: Option<Tensor5List>,
    channel_state_b1c: Option<Tensor3List>,
) -> PyResult<RwkvLayerPyOutput> {
    let weights = load_srs_rwkv_rnn_weights(checkpoint_path).map_err(py_value_error)?;
    let module = weights
        .rwkv_modules
        .get(module_index)
        .ok_or_else(|| py_value_error(format!("unknown RWKV module index {module_index}")))?;
    let block = module.blocks.get(layer_index).ok_or_else(|| {
        py_value_error(format!(
            "unknown layer index {layer_index} for RWKV module {module_index}"
        ))
    })?;

    let in_bc = tensor_from_2d(in_bc, "in_BC").map_err(py_value_error)?;
    let v0_bc = tensor_from_2d(v0_bc, "v0_BC").map_err(py_value_error)?;
    let time_x_shift_b1c = time_x_shift_b1c
        .map(|state| tensor_from_3d(state, "time_x_shift_B1C"))
        .transpose()
        .map_err(py_value_error)?;
    let time_state_b1hkk = time_state_b1hkk
        .map(|state| tensor_from_5d(state, "time_state_B1HKK"))
        .transpose()
        .map_err(py_value_error)?;
    let time_state = match (time_x_shift_b1c.as_ref(), time_state_b1hkk.as_ref()) {
        (Some(time_x_shift_b1c), Some(time_state_b1hkk)) => {
            Some((time_x_shift_b1c, time_state_b1hkk))
        }
        (None, None) => None,
        _ => return Err(py_value_error(
            "rwkv_layer_forward requires both time_x_shift_b1c and time_state_b1hkk, or neither",
        )),
    };
    let channel_state_b1c = channel_state_b1c
        .map(|state| tensor_from_3d(state, "channel_state_B1C"))
        .transpose()
        .map_err(py_value_error)?;

    let (out_bc, next_v0_bc, next_time_x_shift_b1c, next_time_state_b1hkk, next_channel_state_b1c) =
        rwkv_layer_forward(
            block,
            &in_bc,
            &v0_bc,
            time_state,
            channel_state_b1c.as_ref(),
        )
        .map_err(py_value_error)?;

    Ok((
        out_bc.to_vec2::<f32>().map_err(py_value_error)?,
        next_v0_bc.to_vec2::<f32>().map_err(py_value_error)?,
        next_time_x_shift_b1c
            .to_vec3::<f32>()
            .map_err(py_value_error)?,
        tensor_to_vec5(&next_time_state_b1hkk).map_err(py_value_error)?,
        next_channel_state_b1c
            .to_vec3::<f32>()
            .map_err(py_value_error)?,
    ))
}

#[allow(clippy::too_many_arguments, clippy::useless_conversion)]
#[pyfunction(name = "rwkv_rnn_forward")]
#[pyo3(signature = (checkpoint_path, module_index, in_bc, time_x_shift_b1c_by_layer=None, time_state_b1hkk_by_layer=None, channel_state_b1c_by_layer=None))]
pub(crate) fn rwkv_rnn_forward_py(
    checkpoint_path: PathBuf,
    module_index: usize,
    in_bc: Tensor2List,
    time_x_shift_b1c_by_layer: Option<Vec<Tensor3List>>,
    time_state_b1hkk_by_layer: Option<Vec<Tensor5List>>,
    channel_state_b1c_by_layer: Option<Vec<Tensor3List>>,
) -> PyResult<RwkvRnnPyOutput> {
    let weights = load_srs_rwkv_rnn_weights(checkpoint_path).map_err(py_value_error)?;
    let module = weights
        .rwkv_modules
        .get(module_index)
        .ok_or_else(|| py_value_error(format!("unknown RWKV module index {module_index}")))?;

    let in_bc = tensor_from_2d(in_bc, "in_BC").map_err(py_value_error)?;
    let time_x_shift_b1c_by_layer = time_x_shift_b1c_by_layer
        .map(|states| tensor_vec_from_3d(states, "time_x_shift_B1C_by_layer"))
        .transpose()
        .map_err(py_value_error)?;
    let time_state_b1hkk_by_layer = time_state_b1hkk_by_layer
        .map(|states| tensor_vec_from_5d(states, "time_state_B1HKK_by_layer"))
        .transpose()
        .map_err(py_value_error)?;
    let channel_state_b1c_by_layer = channel_state_b1c_by_layer
        .map(|states| tensor_vec_from_3d(states, "channel_state_B1C_by_layer"))
        .transpose()
        .map_err(py_value_error)?;

    let (
        out_bc,
        next_time_x_shift_b1c_by_layer,
        next_time_state_b1hkk_by_layer,
        next_channel_state_b1c_by_layer,
    ) = rwkv_rnn_forward(
        module,
        &in_bc,
        time_x_shift_b1c_by_layer.as_deref(),
        time_state_b1hkk_by_layer.as_deref(),
        channel_state_b1c_by_layer.as_deref(),
    )
    .map_err(py_value_error)?;

    Ok((
        out_bc.to_vec2::<f32>().map_err(py_value_error)?,
        tensor_vec_to_vec3(next_time_x_shift_b1c_by_layer).map_err(py_value_error)?,
        tensor_vec_to_vec5(next_time_state_b1hkk_by_layer).map_err(py_value_error)?,
        tensor_vec_to_vec3(next_channel_state_b1c_by_layer).map_err(py_value_error)?,
    ))
}

#[allow(clippy::too_many_arguments, clippy::useless_conversion)]
#[pyfunction(name = "srs_review")]
#[pyo3(signature = (checkpoint_path, card_features, time_x_shift_b1c_by_module=None, time_state_b1hkk_by_module=None, channel_state_b1c_by_module=None, return_curve=true))]
pub(crate) fn srs_review_py(
    checkpoint_path: PathBuf,
    card_features: Tensor2List,
    time_x_shift_b1c_by_module: Option<SrsReviewPyState3>,
    time_state_b1hkk_by_module: Option<SrsReviewPyState5>,
    channel_state_b1c_by_module: Option<SrsReviewPyState3>,
    return_curve: bool,
) -> PyResult<SrsReviewPyOutput> {
    let weights = load_srs_rwkv_rnn_weights(checkpoint_path).map_err(py_value_error)?;
    let card_features = tensor_from_2d(card_features, "card_features").map_err(py_value_error)?;
    let time_x_shift_b1c_by_module = time_x_shift_b1c_by_module
        .map(|states| nested_tensor_vec_from_3d(states, "time_x_shift_B1C_by_module"))
        .transpose()
        .map_err(py_value_error)?;
    let time_state_b1hkk_by_module = time_state_b1hkk_by_module
        .map(|states| nested_tensor_vec_from_5d(states, "time_state_B1HKK_by_module"))
        .transpose()
        .map_err(py_value_error)?;
    let channel_state_b1c_by_module = channel_state_b1c_by_module
        .map(|states| nested_tensor_vec_from_3d(states, "channel_state_B1C_by_module"))
        .transpose()
        .map_err(py_value_error)?;

    let (
        out_ahead_logits,
        out_w,
        out_p_logits,
        next_time_x_shift_b1c_by_module,
        next_time_state_b1hkk_by_module,
        next_channel_state_b1c_by_module,
    ) = srs_review_forward(
        &weights,
        &card_features,
        time_x_shift_b1c_by_module.as_deref(),
        time_state_b1hkk_by_module.as_deref(),
        channel_state_b1c_by_module.as_deref(),
        return_curve,
    )
    .map_err(py_value_error)?;

    Ok((
        out_ahead_logits
            .map(|tensor| tensor.to_vec2::<f32>())
            .transpose()
            .map_err(py_value_error)?,
        out_w
            .map(|tensor| tensor.to_vec2::<f32>())
            .transpose()
            .map_err(py_value_error)?,
        out_p_logits.to_vec2::<f32>().map_err(py_value_error)?,
        nested_tensor_vec_to_vec3(next_time_x_shift_b1c_by_module).map_err(py_value_error)?,
        nested_tensor_vec_to_vec5(next_time_state_b1hkk_by_module).map_err(py_value_error)?,
        nested_tensor_vec_to_vec3(next_channel_state_b1c_by_module).map_err(py_value_error)?,
    ))
}

#[pyfunction(name = "prediction_probability")]
pub(crate) fn prediction_probability_py(logits: Tensor2List) -> PyResult<f64> {
    let logits = tensor_from_2d(logits, "prediction_logits").map_err(py_value_error)?;
    probability_from_logits(&logits).map_err(py_value_error)
}
