use std::collections::BTreeMap;
use std::sync::Arc;

use candle_core::{bail, Result, Tensor};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyMapping, PyTuple};

use crate::profile::{ProfileTimer, StateInputProfile};
use crate::state::PreparedRow;
use crate::tensor_io::{tensor_from_3d, tensor_from_5d, tensor_to_vec5, Tensor3List, Tensor5List};

use super::{py_value_error, SHORT_BURST_BROADCAST_MAX_BATCH, SRS_REVIEW_STATE_MODULES};

pub(crate) type ReviewIds = (i64, i64, i64, i64);
pub(crate) type SrsReviewStateInput = (Vec<Vec<Tensor>>, Vec<Vec<Tensor>>, Vec<Vec<Tensor>>);
pub(crate) type SrsReviewPyState3 = Vec<Vec<Tensor3List>>;
pub(crate) type SrsReviewPyState5 = Vec<Vec<Tensor5List>>;
type EntityStatePy = (Vec<Tensor3List>, Vec<Tensor5List>, Vec<Tensor3List>);

#[derive(Debug, Clone)]
pub(crate) struct NativeRnnModuleState {
    pub(crate) time_x_shift_b1c_by_layer: Vec<Tensor>,
    pub(crate) time_state_b1hkk_by_layer: Vec<Tensor>,
    pub(crate) channel_state_b1c_by_layer: Vec<Tensor>,
}

pub(crate) const FLAT_STATE_CHANNELS: usize = 128;
pub(crate) const FLAT_STATE_HEADS: usize = 4;
pub(crate) const FLAT_STATE_HEAD_SIZE: usize = 32;
pub(crate) const FLAT_STATE_MATRIX_ELEMENTS: usize =
    FLAT_STATE_HEADS * FLAT_STATE_HEAD_SIZE * FLAT_STATE_HEAD_SIZE;
pub(crate) const FLAT_STATE_LAYER_ELEMENTS: usize =
    FLAT_STATE_CHANNELS + FLAT_STATE_MATRIX_ELEMENTS + FLAT_STATE_CHANNELS;

/// One immutable entity state retained in its native GPU arena layout.
///
/// Multiple entries normally share the same bounded readback allocation. The
/// slice is authoritative until a CPU numerical operation requires Candle
/// tensors; checkpoint writing and a subsequent GPU replay can consume it
/// directly without constructing per-layer tensor metadata.
#[derive(Debug, Clone)]
pub(crate) struct FlatNativeRnnModuleState {
    values: Arc<Vec<f32>>,
    offset: usize,
    layers: usize,
}

impl FlatNativeRnnModuleState {
    pub(crate) fn from_shared_values(
        values: Arc<Vec<f32>>,
        entity: usize,
        layers: usize,
    ) -> Result<Self> {
        let entity_values = layers
            .checked_mul(FLAT_STATE_LAYER_ELEMENTS)
            .ok_or_else(|| candle_core::Error::msg("flat recurrent-state size overflow"))?;
        let offset = entity
            .checked_mul(entity_values)
            .ok_or_else(|| candle_core::Error::msg("flat recurrent-state offset overflow"))?;
        let end = offset
            .checked_add(entity_values)
            .ok_or_else(|| candle_core::Error::msg("flat recurrent-state range overflow"))?;
        if end > values.len() {
            bail!(
                "flat recurrent-state entity {entity} ends at {end}, beyond {} values",
                values.len()
            );
        }
        Ok(Self {
            values,
            offset,
            layers,
        })
    }

    pub(crate) fn layers(&self) -> usize {
        self.layers
    }

    pub(crate) fn values(&self) -> &[f32] {
        let len = self.layers * FLAT_STATE_LAYER_ELEMENTS;
        &self.values[self.offset..self.offset + len]
    }

    pub(crate) fn storage_id(&self) -> usize {
        Arc::as_ptr(&self.values) as usize
    }

    pub(crate) fn materialize(&self) -> Result<NativeRnnModuleState> {
        let values = self.values();
        let mut time_x_shift_b1c_by_layer = Vec::with_capacity(self.layers);
        let mut time_state_b1hkk_by_layer = Vec::with_capacity(self.layers);
        let mut channel_state_b1c_by_layer = Vec::with_capacity(self.layers);
        for layer in 0..self.layers {
            let base = layer * FLAT_STATE_LAYER_ELEMENTS;
            time_x_shift_b1c_by_layer.push(Tensor::from_vec(
                values[base..base + FLAT_STATE_CHANNELS].to_vec(),
                (1usize, 1usize, FLAT_STATE_CHANNELS),
                &candle_core::Device::Cpu,
            )?);
            let state_base = base + FLAT_STATE_CHANNELS;
            time_state_b1hkk_by_layer.push(Tensor::from_vec(
                values[state_base..state_base + FLAT_STATE_MATRIX_ELEMENTS].to_vec(),
                (
                    1usize,
                    1usize,
                    FLAT_STATE_HEADS,
                    FLAT_STATE_HEAD_SIZE,
                    FLAT_STATE_HEAD_SIZE,
                ),
                &candle_core::Device::Cpu,
            )?);
            let channel_base = state_base + FLAT_STATE_MATRIX_ELEMENTS;
            channel_state_b1c_by_layer.push(Tensor::from_vec(
                values[channel_base..channel_base + FLAT_STATE_CHANNELS].to_vec(),
                (1usize, 1usize, FLAT_STATE_CHANNELS),
                &candle_core::Device::Cpu,
            )?);
        }
        Ok(NativeRnnModuleState {
            time_x_shift_b1c_by_layer,
            time_state_b1hkk_by_layer,
            channel_state_b1c_by_layer,
        })
    }
}

#[derive(Debug, Default)]
pub(crate) struct FlatNativeRnnState {
    pub(crate) card_states: BTreeMap<i64, FlatNativeRnnModuleState>,
    pub(crate) note_states: BTreeMap<i64, FlatNativeRnnModuleState>,
    pub(crate) deck_states: BTreeMap<i64, FlatNativeRnnModuleState>,
    pub(crate) preset_states: BTreeMap<i64, FlatNativeRnnModuleState>,
    pub(crate) global_state: Option<FlatNativeRnnModuleState>,
}

impl FlatNativeRnnState {
    pub(crate) fn is_empty(&self) -> bool {
        self.card_states.is_empty()
            && self.note_states.is_empty()
            && self.deck_states.is_empty()
            && self.preset_states.is_empty()
            && self.global_state.is_none()
    }

    pub(crate) fn clear(&mut self) {
        self.card_states.clear();
        self.note_states.clear();
        self.deck_states.clear();
        self.preset_states.clear();
        self.global_state = None;
    }
}

pub(crate) fn prepared_i64(row: &PreparedRow, field: &str) -> Result<i64> {
    let value = row
        .get(field)
        .ok_or_else(|| candle_core::Error::msg(format!("prepared row is missing '{field}'")))?;
    value.as_i64(field).map_err(candle_core::Error::msg)
}

pub(crate) fn review_ids_from_prepared(row: &PreparedRow) -> Result<ReviewIds> {
    Ok((
        prepared_i64(row, "card_id")?,
        prepared_i64(row, "note_id")?,
        prepared_i64(row, "deck_id")?,
        prepared_i64(row, "preset_id")?,
    ))
}

pub(crate) fn state_input_tensor_count(state: &SrsReviewStateInput) -> usize {
    nested_tensor_handle_count(&state.0)
        + nested_tensor_handle_count(&state.1)
        + nested_tensor_handle_count(&state.2)
}

pub(crate) fn state_output_tensor_count(
    time_x_shift_b1c_by_module: &[Vec<Tensor>],
    time_state_b1hkk_by_module: &[Vec<Tensor>],
    channel_state_b1c_by_module: &[Vec<Tensor>],
) -> usize {
    nested_tensor_handle_count(time_x_shift_b1c_by_module)
        + nested_tensor_handle_count(time_state_b1hkk_by_module)
        + nested_tensor_handle_count(channel_state_b1c_by_module)
}

fn nested_tensor_handle_count(values: &[Vec<Tensor>]) -> usize {
    values.iter().map(Vec::len).sum()
}

pub(crate) fn tensor_vec_from_3d(values: Vec<Tensor3List>, name: &str) -> Result<Vec<Tensor>> {
    values
        .into_iter()
        .enumerate()
        .map(|(layer_index, state)| tensor_from_3d(state, &format!("{name}[{layer_index}]")))
        .collect()
}

pub(crate) fn tensor_vec_from_5d(values: Vec<Tensor5List>, name: &str) -> Result<Vec<Tensor>> {
    values
        .into_iter()
        .enumerate()
        .map(|(layer_index, state)| tensor_from_5d(state, &format!("{name}[{layer_index}]")))
        .collect()
}

pub(crate) fn nested_tensor_vec_from_3d(
    values: SrsReviewPyState3,
    name: &str,
) -> Result<Vec<Vec<Tensor>>> {
    values
        .into_iter()
        .enumerate()
        .map(|(module_index, states)| {
            tensor_vec_from_3d(states, &format!("{name}[{module_index}]"))
        })
        .collect()
}

pub(crate) fn nested_tensor_vec_from_5d(
    values: SrsReviewPyState5,
    name: &str,
) -> Result<Vec<Vec<Tensor>>> {
    values
        .into_iter()
        .enumerate()
        .map(|(module_index, states)| {
            tensor_vec_from_5d(states, &format!("{name}[{module_index}]"))
        })
        .collect()
}

pub(crate) fn tensor_vec_to_vec3(tensors: Vec<Tensor>) -> Result<Vec<Tensor3List>> {
    tensors
        .iter()
        .map(|tensor| tensor.to_vec3::<f32>())
        .collect()
}

pub(crate) fn tensor_vec_to_vec5(tensors: Vec<Tensor>) -> Result<Vec<Tensor5List>> {
    tensors.iter().map(tensor_to_vec5).collect()
}

pub(crate) fn nested_tensor_vec_to_vec3(tensors: Vec<Vec<Tensor>>) -> Result<SrsReviewPyState3> {
    tensors.into_iter().map(tensor_vec_to_vec3).collect()
}

pub(crate) fn nested_tensor_vec_to_vec5(tensors: Vec<Vec<Tensor>>) -> Result<SrsReviewPyState5> {
    tensors.into_iter().map(tensor_vec_to_vec5).collect()
}

pub(crate) fn append_state_input(
    state: Option<&NativeRnnModuleState>,
    time_x_shift_b1c_by_module: &mut Vec<Vec<Tensor>>,
    time_state_b1hkk_by_module: &mut Vec<Vec<Tensor>>,
    channel_state_b1c_by_module: &mut Vec<Vec<Tensor>>,
) {
    if let Some(state) = state {
        time_x_shift_b1c_by_module.push(state.time_x_shift_b1c_by_layer.clone());
        time_state_b1hkk_by_module.push(state.time_state_b1hkk_by_layer.clone());
        channel_state_b1c_by_module.push(state.channel_state_b1c_by_layer.clone());
    } else {
        time_x_shift_b1c_by_module.push(Vec::new());
        time_state_b1hkk_by_module.push(Vec::new());
        channel_state_b1c_by_module.push(Vec::new());
    }
}

pub(crate) fn existing_state<'a>(
    states: &'a BTreeMap<i64, NativeRnnModuleState>,
    id: i64,
    name: &str,
) -> Result<&'a NativeRnnModuleState> {
    states.get(&id).ok_or_else(|| {
        candle_core::Error::msg(format!("missing {name} recurrent state for id {id}"))
    })
}

pub(crate) fn push_batched_module_state(
    states: &[&NativeRnnModuleState],
    name: &str,
    time_x_shift_b1c_by_module: &mut Vec<Vec<Tensor>>,
    time_state_b1hkk_by_module: &mut Vec<Vec<Tensor>>,
    channel_state_b1c_by_module: &mut Vec<Vec<Tensor>>,
    allow_short_burst_broadcast: bool,
    profile: Option<&mut StateInputProfile>,
) -> Result<()> {
    let (time_x_shift, time_state, channel_state) =
        batched_module_state(states, name, allow_short_burst_broadcast, profile)?;
    time_x_shift_b1c_by_module.push(time_x_shift);
    time_state_b1hkk_by_module.push(time_state);
    channel_state_b1c_by_module.push(channel_state);
    Ok(())
}

pub(super) fn batched_module_state(
    states: &[&NativeRnnModuleState],
    name: &str,
    allow_short_burst_broadcast: bool,
    mut profile: Option<&mut StateInputProfile>,
) -> Result<(Vec<Tensor>, Vec<Tensor>, Vec<Tensor>)> {
    let first = states
        .first()
        .ok_or_else(|| candle_core::Error::msg(format!("{name} batch state is empty")))?;
    let layer_count = first.time_x_shift_b1c_by_layer.len();
    if first.time_state_b1hkk_by_layer.len() != layer_count
        || first.channel_state_b1c_by_layer.len() != layer_count
    {
        bail!("{name} recurrent state layer counts do not match");
    }

    for (index, state) in states.iter().enumerate() {
        if state.time_x_shift_b1c_by_layer.len() != layer_count
            || state.time_state_b1hkk_by_layer.len() != layer_count
            || state.channel_state_b1c_by_layer.len() != layer_count
        {
            bail!(
                "{name}[{index}] expected {layer_count} recurrent state layers, got {}, {}, and {}",
                state.time_x_shift_b1c_by_layer.len(),
                state.time_state_b1hkk_by_layer.len(),
                state.channel_state_b1c_by_layer.len()
            );
        }
    }

    let mut time_x_shift = Vec::with_capacity(layer_count);
    let mut time_state = Vec::with_capacity(layer_count);
    let mut channel_state = Vec::with_capacity(layer_count);
    let allow_short_burst_broadcast =
        allow_short_burst_broadcast && states.len() <= SHORT_BURST_BROADCAST_MAX_BATCH;
    for layer_index in 0..layer_count {
        let start = ProfileTimer::start(profile.is_some());
        let (packed_state, broadcasted) = pack_state_layer(
            states,
            |state| &state.time_x_shift_b1c_by_layer[layer_index],
            name,
            "time_x_shift_b1c_by_layer",
            layer_index,
            allow_short_burst_broadcast,
        )?;
        time_x_shift.push(packed_state);
        if let Some(profile) = profile.as_deref_mut() {
            record_state_pack(
                profile,
                broadcasted,
                name,
                "time_x_shift_b1c_by_layer",
                states.len(),
                start.elapsed_ns(),
            );
        }

        let start = ProfileTimer::start(profile.is_some());
        let (packed_state, broadcasted) = pack_state_layer(
            states,
            |state| &state.time_state_b1hkk_by_layer[layer_index],
            name,
            "time_state_b1hkk_by_layer",
            layer_index,
            allow_short_burst_broadcast,
        )?;
        time_state.push(packed_state);
        if let Some(profile) = profile.as_deref_mut() {
            record_state_pack(
                profile,
                broadcasted,
                name,
                "time_state_b1hkk_by_layer",
                states.len(),
                start.elapsed_ns(),
            );
        }

        let start = ProfileTimer::start(profile.is_some());
        let (packed_state, broadcasted) = pack_state_layer(
            states,
            |state| &state.channel_state_b1c_by_layer[layer_index],
            name,
            "channel_state_b1c_by_layer",
            layer_index,
            allow_short_burst_broadcast,
        )?;
        channel_state.push(packed_state);
        if let Some(profile) = profile.as_deref_mut() {
            record_state_pack(
                profile,
                broadcasted,
                name,
                "channel_state_b1c_by_layer",
                states.len(),
                start.elapsed_ns(),
            );
        }
    }

    Ok((time_x_shift, time_state, channel_state))
}

fn record_state_pack(
    profile: &mut StateInputProfile,
    broadcasted: bool,
    module_name: &str,
    field_name: &str,
    input_tensors: usize,
    ns: u128,
) {
    if broadcasted {
        profile.record_broadcast(module_name, field_name, input_tensors, ns);
    } else {
        profile.record_cat(module_name, field_name, input_tensors, ns);
    }
}

fn pack_state_layer<F>(
    states: &[&NativeRnnModuleState],
    tensor: F,
    state_name: &str,
    field_name: &str,
    layer_index: usize,
    allow_short_burst_broadcast: bool,
) -> Result<(Tensor, bool)>
where
    F: Fn(&NativeRnnModuleState) -> &Tensor,
{
    let first = tensor(states[0]);
    if first.dims().first().copied() != Some(1) {
        bail!(
            "{state_name}.{field_name}[{layer_index}] expected first dimension 1, got {:?}",
            first.dims()
        );
    }
    let first_tail = &first.dims()[1..];
    let mut tensors = Vec::with_capacity(states.len());
    let mut all_same_tensor = allow_short_burst_broadcast;
    for (index, state) in states.iter().enumerate() {
        let value = tensor(state);
        if value.dims().first().copied() != Some(1) || &value.dims()[1..] != first_tail {
            bail!(
                "{state_name}[{index}].{field_name}[{layer_index}] expected shape [1, {:?}], got {:?}",
                first_tail,
                value.dims()
            );
        }
        if allow_short_burst_broadcast {
            all_same_tensor &= value.id() == first.id();
        }
        tensors.push(value);
    }
    if all_same_tensor {
        let mut dims = first.dims().to_vec();
        dims[0] = states.len();
        return Ok((first.broadcast_as(dims)?, true));
    }
    Ok((Tensor::cat(&tensors, 0)?, false))
}

pub(crate) fn native_module_states_from_review_output(
    time_x_shift_b1c_by_module: Vec<Vec<Tensor>>,
    time_state_b1hkk_by_module: Vec<Vec<Tensor>>,
    channel_state_b1c_by_module: Vec<Vec<Tensor>>,
) -> Result<Vec<NativeRnnModuleState>> {
    if time_x_shift_b1c_by_module.len() != SRS_REVIEW_STATE_MODULES
        || time_state_b1hkk_by_module.len() != SRS_REVIEW_STATE_MODULES
        || channel_state_b1c_by_module.len() != SRS_REVIEW_STATE_MODULES
    {
        bail!(
            "native recurrent state expected {SRS_REVIEW_STATE_MODULES} module states, got {}, {}, and {}",
            time_x_shift_b1c_by_module.len(),
            time_state_b1hkk_by_module.len(),
            channel_state_b1c_by_module.len()
        );
    }

    time_x_shift_b1c_by_module
        .into_iter()
        .zip(time_state_b1hkk_by_module)
        .zip(channel_state_b1c_by_module)
        .enumerate()
        .map(
            |(module_index, ((time_x_shift, time_state), channel_state))| {
                native_module_state_from_parts(
                    time_x_shift,
                    time_state,
                    channel_state,
                    &format!("review_state[{module_index}]"),
                )
            },
        )
        .collect()
}

pub(crate) fn native_module_state_from_parts(
    time_x_shift_b1c_by_layer: Vec<Tensor>,
    time_state_b1hkk_by_layer: Vec<Tensor>,
    channel_state_b1c_by_layer: Vec<Tensor>,
    name: &str,
) -> Result<NativeRnnModuleState> {
    if time_x_shift_b1c_by_layer.len() != time_state_b1hkk_by_layer.len()
        || time_x_shift_b1c_by_layer.len() != channel_state_b1c_by_layer.len()
    {
        bail!(
            "{name} expected time x-shift, time recurrent, and channel state layer counts to match, got {}, {}, and {}",
            time_x_shift_b1c_by_layer.len(),
            time_state_b1hkk_by_layer.len(),
            channel_state_b1c_by_layer.len()
        );
    }
    Ok(NativeRnnModuleState {
        time_x_shift_b1c_by_layer,
        time_state_b1hkk_by_layer,
        channel_state_b1c_by_layer,
    })
}

pub(crate) fn native_module_state_to_py(state: &NativeRnnModuleState) -> Result<EntityStatePy> {
    Ok((
        tensor_vec_to_vec3(state.time_x_shift_b1c_by_layer.clone())?,
        tensor_vec_to_vec5(state.time_state_b1hkk_by_layer.clone())?,
        tensor_vec_to_vec3(state.channel_state_b1c_by_layer.clone())?,
    ))
}

pub(crate) fn native_state_map_to_pydict<'py>(
    py: Python<'py>,
    states: &BTreeMap<i64, NativeRnnModuleState>,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new_bound(py);
    for (id, state) in states {
        dict.set_item(
            *id,
            native_module_state_to_py(state).map_err(py_value_error)?,
        )?;
    }
    Ok(dict.unbind())
}

pub(crate) fn py_state_map_from_snapshot(
    snapshot: &Bound<'_, PyMapping>,
    field: &str,
) -> PyResult<BTreeMap<i64, NativeRnnModuleState>> {
    if !snapshot.contains(field)? {
        return Err(py_value_error(format!(
            "recurrent state snapshot is missing '{field}'"
        )));
    }
    let value = snapshot.get_item(field)?;
    let mapping = value.downcast::<PyMapping>()?;
    let mut states = BTreeMap::new();
    for item in mapping.items()?.iter()? {
        let item = item?;
        let item = item.downcast::<PyTuple>()?;
        let id = parse_i64_py(&item.get_item(0)?, field)?;
        let state = py_entity_state_from_any(&item.get_item(1)?, field)?;
        states.insert(id, state);
    }
    Ok(states)
}

pub(crate) fn py_optional_entity_state_from_snapshot(
    snapshot: &Bound<'_, PyMapping>,
    field: &str,
) -> PyResult<Option<NativeRnnModuleState>> {
    if !snapshot.contains(field)? {
        return Err(py_value_error(format!(
            "recurrent state snapshot is missing '{field}'"
        )));
    }
    let value = snapshot.get_item(field)?;
    if value.is_none() {
        return Ok(None);
    }
    py_entity_state_from_any(&value, field).map(Some)
}

fn py_entity_state_from_any(
    value: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<NativeRnnModuleState> {
    let (time_x, time_recurrent, channel): EntityStatePy = value.extract()?;
    let time_x = tensor_vec_from_3d(time_x, &format!("{name}.time_x_shift_b1c_by_layer"))
        .map_err(py_value_error)?;
    let time_recurrent =
        tensor_vec_from_5d(time_recurrent, &format!("{name}.time_state_b1hkk_by_layer"))
            .map_err(py_value_error)?;
    let channel = tensor_vec_from_3d(channel, &format!("{name}.channel_state_b1c_by_layer"))
        .map_err(py_value_error)?;
    native_module_state_from_parts(time_x, time_recurrent, channel, name).map_err(py_value_error)
}

fn parse_i64_py(value: &Bound<'_, PyAny>, field: &str) -> PyResult<i64> {
    value
        .extract::<i64>()
        .map_err(|_| py_value_error(format!("recurrent state '{field}' keys must be integers")))
}
