use std::{collections::BTreeSet, time::Instant};

use pyo3::prelude::*;
use pyo3::types::PyDict;

macro_rules! profile_inc {
    ($profile:expr, $section:ident.$field:ident) => {
        if let Some(profile) = $profile.as_deref_mut() {
            profile.$section.$field += 1;
        }
    };
}

macro_rules! profile_add {
    ($profile:expr, $section:ident.$field:ident, $elapsed_ns:expr) => {
        if let Some(profile) = $profile.as_deref_mut() {
            profile.$section.$field += $elapsed_ns;
        }
    };
}

#[derive(Debug, Default)]
pub(crate) struct RuntimeProfile {
    pub(crate) review_count: usize,
    pub(crate) total_ns: u128,
    pub(crate) parse_review_ns: u128,
    pub(crate) prepare_predict_ns: u128,
    pub(crate) predict_feature_ns: u128,
    pub(crate) predict_tensor_ns: u128,
    pub(crate) predict_forward_ns: u128,
    pub(crate) predict_output_ns: u128,
    pub(crate) prepare_process_ns: u128,
    pub(crate) process_feature_ns: u128,
    pub(crate) process_tensor_ns: u128,
    pub(crate) process_forward_ns: u128,
    pub(crate) bulk_fast_path_calls: usize,
    pub(crate) bulk_fast_path_ns: u128,
    pub(crate) state_update_ns: u128,
    pub(crate) curve_output_ns: u128,
    pub(crate) transaction_capture_ns: u128,
    pub(crate) batch: BatchProfile,
    pub(crate) materialization: MaterializationProfile,
    pub(crate) state_input: StateInputProfile,
    pub(crate) predict_forward: ForwardProfile,
    pub(crate) process_forward: ForwardProfile,
}

#[derive(Debug, Default)]
pub(crate) struct BatchProfile {
    pub(crate) requested_batch_size: usize,
    pub(crate) flush_count: usize,
    pub(crate) rows_per_flush: Vec<usize>,
    pub(crate) scalar_fallback_count: usize,
    pub(crate) missing_encoding_fallback_count: usize,
    pub(crate) missing_state_fallback_count: usize,
    pub(crate) duplicate_card_ids_by_flush: Vec<usize>,
    pub(crate) duplicate_note_ids_by_flush: Vec<usize>,
    pub(crate) duplicate_deck_ids_by_flush: Vec<usize>,
    pub(crate) duplicate_preset_ids_by_flush: Vec<usize>,
    pub(crate) duplicate_global_ids_by_flush: Vec<usize>,
}

#[derive(Debug, Default)]
pub(crate) struct MaterializationProfile {
    pub(crate) predict_feature_vecs: usize,
    pub(crate) predict_feature_vec_values: usize,
    pub(crate) process_feature_vecs: usize,
    pub(crate) process_feature_vec_values: usize,
    pub(crate) feature_tensors: usize,
    pub(crate) feature_tensor_values: usize,
    pub(crate) state_input_snapshots: usize,
    pub(crate) state_input_tensor_handles: usize,
    pub(crate) state_input_ns: u128,
    pub(crate) state_store_snapshots: usize,
    pub(crate) state_store_tensor_handles: usize,
    pub(crate) state_store_ns: u128,
    pub(crate) bulk_flat_state_input_snapshots: usize,
    pub(crate) bulk_flat_state_input_values: usize,
    pub(crate) bulk_flat_state_input_ns: u128,
    pub(crate) bulk_flat_state_store_snapshots: usize,
    pub(crate) bulk_flat_state_store_tensors: usize,
    pub(crate) bulk_flat_state_store_values: usize,
    pub(crate) bulk_flat_state_store_ns: u128,
    pub(crate) prediction_probability_values: usize,
    pub(crate) curve_output_vecs: usize,
    pub(crate) curve_output_values: usize,
}

#[derive(Debug, Default)]
pub(crate) struct StateInputProfile {
    pub(crate) calls: usize,
    pub(crate) rows: usize,
    pub(crate) lookup_ns: u128,
    pub(crate) cat_ns: u128,
    pub(crate) cat_calls: usize,
    pub(crate) cat_input_tensors: usize,
    pub(crate) broadcast_ns: u128,
    pub(crate) broadcast_calls: usize,
    pub(crate) broadcast_input_tensors: usize,
    pub(crate) card_cat_ns: u128,
    pub(crate) note_cat_ns: u128,
    pub(crate) deck_cat_ns: u128,
    pub(crate) preset_cat_ns: u128,
    pub(crate) global_cat_ns: u128,
    pub(crate) time_x_cat_ns: u128,
    pub(crate) time_state_cat_ns: u128,
    pub(crate) channel_cat_ns: u128,
}

#[derive(Debug, Default)]
pub(crate) struct ForwardProfile {
    pub(crate) calls: usize,
    pub(crate) total_ns: u128,
    pub(crate) validate_ns: u128,
    pub(crate) features2card_ns: u128,
    pub(crate) card_module_ns: u128,
    pub(crate) deck_module_ns: u128,
    pub(crate) note_module_ns: u128,
    pub(crate) preset_module_ns: u128,
    pub(crate) global_module_ns: u128,
    pub(crate) prehead_norm_ns: u128,
    pub(crate) curve_heads_ns: u128,
    pub(crate) p_head_ns: u128,
    pub(crate) pack_state_ns: u128,
    pub(crate) rnn: RnnProfile,
    pub(crate) layer: LayerProfile,
    pub(crate) time_mixer: TimeMixerProfile,
    pub(crate) channel_mixer: ChannelMixerProfile,
    pub(crate) layout: LayoutProfile,
}

#[derive(Debug, Default)]
pub(crate) struct LayoutProfile {
    pub(crate) linear_calls: usize,
    pub(crate) linear_non_contiguous_inputs: usize,
    pub(crate) linear_rank3_calls: usize,
    pub(crate) linear_rank3_non_contiguous_inputs: usize,
    pub(crate) linear_rank4_calls: usize,
    pub(crate) linear_rank4_non_contiguous_inputs: usize,
    pub(crate) linear_native_calls: usize,
    pub(crate) linear_native_rank2_calls: usize,
    pub(crate) linear_native_rank3_calls: usize,
    pub(crate) linear_native_other_rank_calls: usize,
    pub(crate) linear_native_scalar_calls: usize,
    pub(crate) linear_native_avx2_fma_calls: usize,
    pub(crate) linear_native_pulp_calls: usize,
    pub(crate) linear_native_fallback_calls: usize,
    pub(crate) linear_native_output_values: usize,
    pub(crate) reshape_calls: usize,
    pub(crate) reshape_non_contiguous_inputs: usize,
}

#[derive(Debug, Default)]
pub(crate) struct RnnProfile {
    pub(crate) calls: usize,
    pub(crate) total_ns: u128,
    pub(crate) validate_ns: u128,
    pub(crate) init_ns: u128,
    pub(crate) layer_total_ns: u128,
}

#[derive(Debug, Default)]
pub(crate) struct LayerProfile {
    pub(crate) calls: usize,
    pub(crate) total_ns: u128,
    pub(crate) time_mixer_ns: u128,
    pub(crate) channel_mixer_ns: u128,
}

#[derive(Debug, Default)]
pub(crate) struct TimeMixerProfile {
    pub(crate) calls: usize,
    pub(crate) total_ns: u128,
    pub(crate) validate_ns: u128,
    pub(crate) layer_norm_ns: u128,
    pub(crate) state_ns: u128,
    pub(crate) lerp_split_ns: u128,
    pub(crate) projection_ns: u128,
    pub(crate) native_projection_group_calls: usize,
    pub(crate) native_projection_group_rows: usize,
    pub(crate) v_mix_ns: u128,
    pub(crate) native_v_mix_calls: usize,
    pub(crate) native_v_mix_rows: usize,
    pub(crate) lora_decay_ns: u128,
    pub(crate) native_lora_decay_scratch_calls: usize,
    pub(crate) native_lora_decay_scratch_rows: usize,
    pub(crate) lora_first_projection_ns: u128,
    pub(crate) lora_second_projection_ns: u128,
    pub(crate) lora_a_ns: u128,
    pub(crate) a_lora_a_projection_ns: u128,
    pub(crate) a_lora_b_projection_ns: u128,
    pub(crate) lora_a_activation_ns: u128,
    pub(crate) lora_gate_ns: u128,
    pub(crate) gate_lora_a_projection_ns: u128,
    pub(crate) gate_lora_b_projection_ns: u128,
    pub(crate) lora_gate_activation_ns: u128,
    pub(crate) lora_d_ns: u128,
    pub(crate) d_lora_a_projection_ns: u128,
    pub(crate) d_lora_b_projection_ns: u128,
    pub(crate) lora_d_activation_ns: u128,
    pub(crate) lora_decay_elementwise_ns: u128,
    pub(crate) lora_decay_temporary_tensors: usize,
    pub(crate) lora_decay_temporary_values: usize,
    pub(crate) reshape_norm_ns: u128,
    pub(crate) recurrence_squeeze_ns: u128,
    pub(crate) recurrence_ns: u128,
    pub(crate) group_norm_ns: u128,
    pub(crate) output_ns: u128,
    pub(crate) native_output_calls: usize,
    pub(crate) native_output_rows: usize,
    pub(crate) single_timestep: SingleTimestepProfile,
}

#[derive(Debug, Default)]
pub(crate) struct ChannelMixerProfile {
    pub(crate) calls: usize,
    pub(crate) total_ns: u128,
    pub(crate) validate_ns: u128,
    pub(crate) layer_norm_ns: u128,
    pub(crate) state_ns: u128,
    pub(crate) lerp_ns: u128,
    pub(crate) projection_ns: u128,
    pub(crate) native_projection_calls: usize,
    pub(crate) native_projection_rows: usize,
    pub(crate) output_ns: u128,
}

#[derive(Debug, Default)]
pub(crate) struct SingleTimestepProfile {
    pub(crate) calls: usize,
    pub(crate) total_ns: u128,
    pub(crate) validate_ns: u128,
    pub(crate) tensor_read_ns: u128,
    pub(crate) unsqueeze_ns: u128,
    pub(crate) decay_ns: u128,
    pub(crate) deformation_prepare_ns: u128,
    pub(crate) deformation_matmul_ns: u128,
    pub(crate) state_decay_ns: u128,
    pub(crate) value_outer_ns: u128,
    pub(crate) state_update_ns: u128,
    pub(crate) output_matmul_ns: u128,
    pub(crate) output_squeeze_ns: u128,
    pub(crate) fused_state_output_ns: u128,
    pub(crate) state_output_scalar_rows: usize,
    pub(crate) state_output_avx2_fma_rows: usize,
    pub(crate) state_output_pulp_rows: usize,
    pub(crate) output_scalar_rows: usize,
    pub(crate) output_avx2_fma_rows: usize,
    pub(crate) output_pulp_rows: usize,
    pub(crate) tensor_write_ns: u128,
}

pub(crate) struct ProfileTimer(Option<Instant>);

impl ProfileTimer {
    pub(crate) fn start(enabled: bool) -> Self {
        Self(enabled.then(Instant::now))
    }

    pub(crate) fn elapsed_ns(&self) -> u128 {
        self.0
            .as_ref()
            .map_or(0, |start| start.elapsed().as_nanos())
    }
}

impl RuntimeProfile {
    pub(crate) fn to_pydict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("review_count", self.review_count)?;
        dict.set_item("total_ns", self.total_ns)?;
        dict.set_item("parse_review_ns", self.parse_review_ns)?;
        dict.set_item("prepare_predict_ns", self.prepare_predict_ns)?;
        dict.set_item("predict_feature_ns", self.predict_feature_ns)?;
        dict.set_item("predict_tensor_ns", self.predict_tensor_ns)?;
        dict.set_item("predict_forward_ns", self.predict_forward_ns)?;
        dict.set_item("predict_output_ns", self.predict_output_ns)?;
        dict.set_item("prepare_process_ns", self.prepare_process_ns)?;
        dict.set_item("process_feature_ns", self.process_feature_ns)?;
        dict.set_item("process_tensor_ns", self.process_tensor_ns)?;
        dict.set_item("process_forward_ns", self.process_forward_ns)?;
        dict.set_item("bulk_fast_path_calls", self.bulk_fast_path_calls)?;
        dict.set_item("bulk_fast_path_ns", self.bulk_fast_path_ns)?;
        dict.set_item("state_update_ns", self.state_update_ns)?;
        dict.set_item("curve_output_ns", self.curve_output_ns)?;
        dict.set_item("transaction_capture_ns", self.transaction_capture_ns)?;
        dict.set_item("batch", self.batch.to_pydict(py)?)?;
        dict.set_item("materialization", self.materialization.to_pydict(py)?)?;
        dict.set_item("state_input", self.state_input.to_pydict(py)?)?;
        dict.set_item("predict_forward", self.predict_forward.to_pydict(py)?)?;
        dict.set_item("process_forward", self.process_forward.to_pydict(py)?)?;
        Ok(dict.unbind())
    }
}

impl BatchProfile {
    pub(crate) fn record_flush(&mut self, ids: &[(i64, i64, i64, i64)]) {
        self.flush_count += 1;
        self.rows_per_flush.push(ids.len());
        self.duplicate_card_ids_by_flush.push(duplicate_count(
            ids.iter().map(|(card_id, _, _, _)| *card_id),
        ));
        self.duplicate_note_ids_by_flush.push(duplicate_count(
            ids.iter().map(|(_, note_id, _, _)| *note_id),
        ));
        self.duplicate_deck_ids_by_flush.push(duplicate_count(
            ids.iter().map(|(_, _, deck_id, _)| *deck_id),
        ));
        self.duplicate_preset_ids_by_flush.push(duplicate_count(
            ids.iter().map(|(_, _, _, preset_id)| *preset_id),
        ));
        self.duplicate_global_ids_by_flush
            .push(ids.len().saturating_sub(1));
    }

    fn to_pydict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("requested_batch_size", self.requested_batch_size)?;
        dict.set_item("flush_count", self.flush_count)?;
        dict.set_item("rows_per_flush", &self.rows_per_flush)?;
        dict.set_item("scalar_fallback_count", self.scalar_fallback_count)?;
        dict.set_item(
            "missing_encoding_fallback_count",
            self.missing_encoding_fallback_count,
        )?;
        dict.set_item(
            "missing_state_fallback_count",
            self.missing_state_fallback_count,
        )?;
        dict.set_item(
            "duplicate_card_ids_by_flush",
            &self.duplicate_card_ids_by_flush,
        )?;
        dict.set_item(
            "duplicate_note_ids_by_flush",
            &self.duplicate_note_ids_by_flush,
        )?;
        dict.set_item(
            "duplicate_deck_ids_by_flush",
            &self.duplicate_deck_ids_by_flush,
        )?;
        dict.set_item(
            "duplicate_preset_ids_by_flush",
            &self.duplicate_preset_ids_by_flush,
        )?;
        dict.set_item(
            "duplicate_global_ids_by_flush",
            &self.duplicate_global_ids_by_flush,
        )?;
        Ok(dict.unbind())
    }
}

fn duplicate_count(ids: impl Iterator<Item = i64>) -> usize {
    let mut count = 0usize;
    let mut seen = BTreeSet::new();
    for id in ids {
        if !seen.insert(id) {
            count += 1;
        }
    }
    count
}

impl MaterializationProfile {
    fn to_pydict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("predict_feature_vecs", self.predict_feature_vecs)?;
        dict.set_item(
            "predict_feature_vec_values",
            self.predict_feature_vec_values,
        )?;
        dict.set_item("process_feature_vecs", self.process_feature_vecs)?;
        dict.set_item(
            "process_feature_vec_values",
            self.process_feature_vec_values,
        )?;
        dict.set_item("feature_tensors", self.feature_tensors)?;
        dict.set_item("feature_tensor_values", self.feature_tensor_values)?;
        dict.set_item("state_input_snapshots", self.state_input_snapshots)?;
        dict.set_item(
            "state_input_tensor_handles",
            self.state_input_tensor_handles,
        )?;
        dict.set_item("state_input_ns", self.state_input_ns)?;
        dict.set_item("state_store_snapshots", self.state_store_snapshots)?;
        dict.set_item(
            "state_store_tensor_handles",
            self.state_store_tensor_handles,
        )?;
        dict.set_item("state_store_ns", self.state_store_ns)?;
        dict.set_item(
            "bulk_flat_state_input_snapshots",
            self.bulk_flat_state_input_snapshots,
        )?;
        dict.set_item(
            "bulk_flat_state_input_values",
            self.bulk_flat_state_input_values,
        )?;
        dict.set_item("bulk_flat_state_input_ns", self.bulk_flat_state_input_ns)?;
        dict.set_item(
            "bulk_flat_state_store_snapshots",
            self.bulk_flat_state_store_snapshots,
        )?;
        dict.set_item(
            "bulk_flat_state_store_tensors",
            self.bulk_flat_state_store_tensors,
        )?;
        dict.set_item(
            "bulk_flat_state_store_values",
            self.bulk_flat_state_store_values,
        )?;
        dict.set_item("bulk_flat_state_store_ns", self.bulk_flat_state_store_ns)?;
        dict.set_item(
            "prediction_probability_values",
            self.prediction_probability_values,
        )?;
        dict.set_item("curve_output_vecs", self.curve_output_vecs)?;
        dict.set_item("curve_output_values", self.curve_output_values)?;
        Ok(dict.unbind())
    }
}

impl StateInputProfile {
    pub(crate) fn record_cat(
        &mut self,
        module_name: &str,
        field_name: &str,
        input_tensors: usize,
        ns: u128,
    ) {
        self.cat_ns += ns;
        self.cat_calls += 1;
        self.cat_input_tensors += input_tensors;
        match module_name {
            "card_states" => self.card_cat_ns += ns,
            "note_states" => self.note_cat_ns += ns,
            "deck_states" => self.deck_cat_ns += ns,
            "preset_states" => self.preset_cat_ns += ns,
            "global_state" => self.global_cat_ns += ns,
            _ => {}
        }
        match field_name {
            "time_x_shift_b1c_by_layer" => self.time_x_cat_ns += ns,
            "time_state_b1hkk_by_layer" => self.time_state_cat_ns += ns,
            "channel_state_b1c_by_layer" => self.channel_cat_ns += ns,
            _ => {}
        }
    }

    pub(crate) fn record_broadcast(
        &mut self,
        module_name: &str,
        field_name: &str,
        input_tensors: usize,
        ns: u128,
    ) {
        self.broadcast_ns += ns;
        self.broadcast_calls += 1;
        self.broadcast_input_tensors += input_tensors;
        match module_name {
            "card_states" => self.card_cat_ns += ns,
            "note_states" => self.note_cat_ns += ns,
            "deck_states" => self.deck_cat_ns += ns,
            "preset_states" => self.preset_cat_ns += ns,
            "global_state" => self.global_cat_ns += ns,
            _ => {}
        }
        match field_name {
            "time_x_shift_b1c_by_layer" => self.time_x_cat_ns += ns,
            "time_state_b1hkk_by_layer" => self.time_state_cat_ns += ns,
            "channel_state_b1c_by_layer" => self.channel_cat_ns += ns,
            _ => {}
        }
    }

    fn to_pydict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("calls", self.calls)?;
        dict.set_item("rows", self.rows)?;
        dict.set_item("lookup_ns", self.lookup_ns)?;
        dict.set_item("cat_ns", self.cat_ns)?;
        dict.set_item("cat_calls", self.cat_calls)?;
        dict.set_item("cat_input_tensors", self.cat_input_tensors)?;
        dict.set_item("broadcast_ns", self.broadcast_ns)?;
        dict.set_item("broadcast_calls", self.broadcast_calls)?;
        dict.set_item("broadcast_input_tensors", self.broadcast_input_tensors)?;
        dict.set_item("card_cat_ns", self.card_cat_ns)?;
        dict.set_item("note_cat_ns", self.note_cat_ns)?;
        dict.set_item("deck_cat_ns", self.deck_cat_ns)?;
        dict.set_item("preset_cat_ns", self.preset_cat_ns)?;
        dict.set_item("global_cat_ns", self.global_cat_ns)?;
        dict.set_item("time_x_cat_ns", self.time_x_cat_ns)?;
        dict.set_item("time_state_cat_ns", self.time_state_cat_ns)?;
        dict.set_item("channel_cat_ns", self.channel_cat_ns)?;
        Ok(dict.unbind())
    }
}

impl ForwardProfile {
    fn to_pydict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("calls", self.calls)?;
        dict.set_item("total_ns", self.total_ns)?;
        dict.set_item("validate_ns", self.validate_ns)?;
        dict.set_item("features2card_ns", self.features2card_ns)?;
        dict.set_item("card_module_ns", self.card_module_ns)?;
        dict.set_item("deck_module_ns", self.deck_module_ns)?;
        dict.set_item("note_module_ns", self.note_module_ns)?;
        dict.set_item("preset_module_ns", self.preset_module_ns)?;
        dict.set_item("global_module_ns", self.global_module_ns)?;
        dict.set_item("prehead_norm_ns", self.prehead_norm_ns)?;
        dict.set_item("curve_heads_ns", self.curve_heads_ns)?;
        dict.set_item("p_head_ns", self.p_head_ns)?;
        dict.set_item("pack_state_ns", self.pack_state_ns)?;
        dict.set_item("rnn", self.rnn.to_pydict(py)?)?;
        dict.set_item("layer", self.layer.to_pydict(py)?)?;
        dict.set_item("time_mixer", self.time_mixer.to_pydict(py)?)?;
        dict.set_item("channel_mixer", self.channel_mixer.to_pydict(py)?)?;
        dict.set_item("layout", self.layout.to_pydict(py)?)?;
        Ok(dict.unbind())
    }
}

impl LayoutProfile {
    fn to_pydict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("linear_calls", self.linear_calls)?;
        dict.set_item(
            "linear_non_contiguous_inputs",
            self.linear_non_contiguous_inputs,
        )?;
        dict.set_item("linear_rank3_calls", self.linear_rank3_calls)?;
        dict.set_item(
            "linear_rank3_non_contiguous_inputs",
            self.linear_rank3_non_contiguous_inputs,
        )?;
        dict.set_item("linear_rank4_calls", self.linear_rank4_calls)?;
        dict.set_item(
            "linear_rank4_non_contiguous_inputs",
            self.linear_rank4_non_contiguous_inputs,
        )?;
        dict.set_item("linear_native_calls", self.linear_native_calls)?;
        dict.set_item("linear_native_rank2_calls", self.linear_native_rank2_calls)?;
        dict.set_item("linear_native_rank3_calls", self.linear_native_rank3_calls)?;
        dict.set_item(
            "linear_native_other_rank_calls",
            self.linear_native_other_rank_calls,
        )?;
        dict.set_item(
            "linear_native_scalar_calls",
            self.linear_native_scalar_calls,
        )?;
        dict.set_item(
            "linear_native_avx2_fma_calls",
            self.linear_native_avx2_fma_calls,
        )?;
        dict.set_item("linear_native_pulp_calls", self.linear_native_pulp_calls)?;
        dict.set_item(
            "linear_native_fallback_calls",
            self.linear_native_fallback_calls,
        )?;
        dict.set_item(
            "linear_native_output_values",
            self.linear_native_output_values,
        )?;
        dict.set_item("reshape_calls", self.reshape_calls)?;
        dict.set_item(
            "reshape_non_contiguous_inputs",
            self.reshape_non_contiguous_inputs,
        )?;
        Ok(dict.unbind())
    }
}

impl RnnProfile {
    fn to_pydict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("calls", self.calls)?;
        dict.set_item("total_ns", self.total_ns)?;
        dict.set_item("validate_ns", self.validate_ns)?;
        dict.set_item("init_ns", self.init_ns)?;
        dict.set_item("layer_total_ns", self.layer_total_ns)?;
        Ok(dict.unbind())
    }
}

impl LayerProfile {
    fn to_pydict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("calls", self.calls)?;
        dict.set_item("total_ns", self.total_ns)?;
        dict.set_item("time_mixer_ns", self.time_mixer_ns)?;
        dict.set_item("channel_mixer_ns", self.channel_mixer_ns)?;
        Ok(dict.unbind())
    }
}

impl TimeMixerProfile {
    fn to_pydict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("calls", self.calls)?;
        dict.set_item("total_ns", self.total_ns)?;
        dict.set_item("validate_ns", self.validate_ns)?;
        dict.set_item("layer_norm_ns", self.layer_norm_ns)?;
        dict.set_item("state_ns", self.state_ns)?;
        dict.set_item("lerp_split_ns", self.lerp_split_ns)?;
        dict.set_item("projection_ns", self.projection_ns)?;
        dict.set_item(
            "native_projection_group_calls",
            self.native_projection_group_calls,
        )?;
        dict.set_item(
            "native_projection_group_rows",
            self.native_projection_group_rows,
        )?;
        dict.set_item("v_mix_ns", self.v_mix_ns)?;
        dict.set_item("native_v_mix_calls", self.native_v_mix_calls)?;
        dict.set_item("native_v_mix_rows", self.native_v_mix_rows)?;
        dict.set_item("lora_decay_ns", self.lora_decay_ns)?;
        dict.set_item(
            "native_lora_decay_scratch_calls",
            self.native_lora_decay_scratch_calls,
        )?;
        dict.set_item(
            "native_lora_decay_scratch_rows",
            self.native_lora_decay_scratch_rows,
        )?;
        dict.set_item("lora_first_projection_ns", self.lora_first_projection_ns)?;
        dict.set_item("lora_second_projection_ns", self.lora_second_projection_ns)?;
        dict.set_item("lora_a_ns", self.lora_a_ns)?;
        dict.set_item("a_lora_a_projection_ns", self.a_lora_a_projection_ns)?;
        dict.set_item("a_lora_b_projection_ns", self.a_lora_b_projection_ns)?;
        dict.set_item("lora_a_activation_ns", self.lora_a_activation_ns)?;
        dict.set_item("lora_gate_ns", self.lora_gate_ns)?;
        dict.set_item("gate_lora_a_projection_ns", self.gate_lora_a_projection_ns)?;
        dict.set_item("gate_lora_b_projection_ns", self.gate_lora_b_projection_ns)?;
        dict.set_item("lora_gate_activation_ns", self.lora_gate_activation_ns)?;
        dict.set_item("lora_d_ns", self.lora_d_ns)?;
        dict.set_item("d_lora_a_projection_ns", self.d_lora_a_projection_ns)?;
        dict.set_item("d_lora_b_projection_ns", self.d_lora_b_projection_ns)?;
        dict.set_item("lora_d_activation_ns", self.lora_d_activation_ns)?;
        dict.set_item("lora_decay_elementwise_ns", self.lora_decay_elementwise_ns)?;
        dict.set_item(
            "lora_decay_temporary_tensors",
            self.lora_decay_temporary_tensors,
        )?;
        dict.set_item(
            "lora_decay_temporary_values",
            self.lora_decay_temporary_values,
        )?;
        dict.set_item("reshape_norm_ns", self.reshape_norm_ns)?;
        dict.set_item("recurrence_squeeze_ns", self.recurrence_squeeze_ns)?;
        dict.set_item("recurrence_ns", self.recurrence_ns)?;
        dict.set_item("group_norm_ns", self.group_norm_ns)?;
        dict.set_item("output_ns", self.output_ns)?;
        dict.set_item("native_output_calls", self.native_output_calls)?;
        dict.set_item("native_output_rows", self.native_output_rows)?;
        dict.set_item("single_timestep", self.single_timestep.to_pydict(py)?)?;
        Ok(dict.unbind())
    }
}

impl SingleTimestepProfile {
    fn to_pydict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("calls", self.calls)?;
        dict.set_item("total_ns", self.total_ns)?;
        dict.set_item("validate_ns", self.validate_ns)?;
        dict.set_item("tensor_read_ns", self.tensor_read_ns)?;
        dict.set_item("unsqueeze_ns", self.unsqueeze_ns)?;
        dict.set_item("decay_ns", self.decay_ns)?;
        dict.set_item("deformation_prepare_ns", self.deformation_prepare_ns)?;
        dict.set_item("deformation_matmul_ns", self.deformation_matmul_ns)?;
        dict.set_item("state_decay_ns", self.state_decay_ns)?;
        dict.set_item("value_outer_ns", self.value_outer_ns)?;
        dict.set_item("state_update_ns", self.state_update_ns)?;
        dict.set_item("output_matmul_ns", self.output_matmul_ns)?;
        dict.set_item("output_squeeze_ns", self.output_squeeze_ns)?;
        dict.set_item("fused_state_output_ns", self.fused_state_output_ns)?;
        dict.set_item("state_output_scalar_rows", self.state_output_scalar_rows)?;
        dict.set_item(
            "state_output_avx2_fma_rows",
            self.state_output_avx2_fma_rows,
        )?;
        dict.set_item("state_output_pulp_rows", self.state_output_pulp_rows)?;
        dict.set_item("output_scalar_rows", self.output_scalar_rows)?;
        dict.set_item("output_avx2_fma_rows", self.output_avx2_fma_rows)?;
        dict.set_item("output_pulp_rows", self.output_pulp_rows)?;
        dict.set_item("tensor_write_ns", self.tensor_write_ns)?;
        Ok(dict.unbind())
    }
}

impl ChannelMixerProfile {
    fn to_pydict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("calls", self.calls)?;
        dict.set_item("total_ns", self.total_ns)?;
        dict.set_item("validate_ns", self.validate_ns)?;
        dict.set_item("layer_norm_ns", self.layer_norm_ns)?;
        dict.set_item("state_ns", self.state_ns)?;
        dict.set_item("lerp_ns", self.lerp_ns)?;
        dict.set_item("projection_ns", self.projection_ns)?;
        dict.set_item("native_projection_calls", self.native_projection_calls)?;
        dict.set_item("native_projection_rows", self.native_projection_rows)?;
        dict.set_item("output_ns", self.output_ns)?;
        Ok(dict.unbind())
    }
}
