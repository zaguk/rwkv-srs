#![allow(clippy::useless_conversion)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyMapping, PyTuple};

use crate::features::CARD_FEATURE_COLUMNS;
use crate::id_encoding::{empty_id_encodings, TorchMt19937, ID_SUBMODULES};
use crate::state::{FeatureState, MaybeId, PreparedRow, ReviewInput, RowValue};

#[pyclass]
pub struct DeterministicState {
    pub(crate) inner: FeatureState,
}

#[pymethods]
impl DeterministicState {
    #[new]
    #[pyo3(signature = (torch_seed=5489))]
    fn new(torch_seed: u64) -> Self {
        Self {
            inner: FeatureState::with_torch_seed(torch_seed),
        }
    }

    fn prepare_predict<'py>(
        &self,
        py: Python<'py>,
        review: &Bound<'py, PyMapping>,
    ) -> PyResult<Py<PyDict>> {
        let input = review_from_mapping(review, false)?;
        row_to_pydict(py, &self.inner.prepare_predict_row(&input))
    }

    fn prepare_process<'py>(
        &self,
        py: Python<'py>,
        review: &Bound<'py, PyMapping>,
    ) -> PyResult<Py<PyDict>> {
        let input = review_from_mapping(review, true)?;
        let row = self
            .inner
            .prepare_process_row(&input)
            .map_err(PyValueError::new_err)?;
        row_to_pydict(py, &row)
    }

    fn record_processed(&mut self, row: &Bound<'_, PyMapping>) -> PyResult<()> {
        let prepared = prepared_record_fields_from_mapping(row)?;
        self.inner
            .record_processed_row(&prepared)
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (row, mutate_id_encodings))]
    fn feature_vector(
        &mut self,
        row: &Bound<'_, PyMapping>,
        mutate_id_encodings: bool,
    ) -> PyResult<Vec<f32>> {
        let prepared = prepared_feature_fields_from_mapping(row)?;
        self.inner
            .feature_vector(&prepared, mutate_id_encodings)
            .map_err(PyValueError::new_err)
    }

    fn process_feature_vector(&mut self, row: &Bound<'_, PyMapping>) -> PyResult<Vec<f32>> {
        let prepared = prepared_feature_fields_from_mapping(row)?;
        self.inner
            .process_feature_vector(&prepared)
            .map_err(PyValueError::new_err)
    }

    fn predict_feature_vector(&mut self, row: &Bound<'_, PyMapping>) -> PyResult<Vec<f32>> {
        let prepared = prepared_feature_fields_from_mapping(row)?;
        self.inner
            .predict_feature_vector(&prepared)
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (row, skip))]
    fn skip_needs_rng_restore(&self, row: &Bound<'_, PyMapping>, skip: bool) -> PyResult<bool> {
        let prepared = prepared_feature_fields_from_mapping(row)?;
        self.inner
            .skip_needs_rng_restore(&prepared, skip)
            .map_err(PyValueError::new_err)
    }

    fn can_batch_predict(&self, row: &Bound<'_, PyMapping>) -> PyResult<bool> {
        let prepared = prepared_feature_fields_from_mapping(row)?;
        self.inner
            .can_batch_predict(&prepared)
            .map_err(PyValueError::new_err)
    }

    fn feature_vectors(&self, rows: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f32>>> {
        let mut prepared_rows = Vec::new();
        for row in rows.iter()? {
            let row = row?;
            let row = row.downcast::<PyMapping>()?;
            prepared_rows.push(prepared_feature_fields_from_mapping(row)?);
        }
        self.inner
            .feature_vectors(&prepared_rows)
            .map_err(PyValueError::new_err)
    }

    fn process_deterministic<'py>(
        &mut self,
        py: Python<'py>,
        review: &Bound<'py, PyMapping>,
    ) -> PyResult<Py<PyDict>> {
        let input = review_from_mapping(review, true)?;
        let row = self
            .inner
            .prepare_process_row(&input)
            .map_err(PyValueError::new_err)?;
        self.inner
            .record_processed_row(&row)
            .map_err(PyValueError::new_err)?;
        row_to_pydict(py, &row)
    }

    fn record_recurrent_state_update(&mut self, row: &Bound<'_, PyMapping>) -> PyResult<()> {
        let prepared = prepared_feature_fields_from_mapping(row)?;
        self.inner
            .record_recurrent_state_update(&prepared)
            .map_err(PyValueError::new_err)
    }

    fn recurrent_state_snapshot<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        feature_state_recurrent_state_snapshot(py, &self.inner)
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        feature_state_snapshot(py, &self.inner)
    }

    fn restore_snapshot(&mut self, snapshot: &Bound<'_, PyMapping>) -> PyResult<()> {
        restore_feature_state_snapshot(&mut self.inner, snapshot)
    }

    fn restore_recurrent_state_keys(&mut self, snapshot: &Bound<'_, PyMapping>) -> PyResult<()> {
        restore_feature_state_recurrent_keys(&mut self.inner, snapshot)
    }

    fn restore_id_encoding_snapshot(&mut self, snapshot: &Bound<'_, PyMapping>) -> PyResult<()> {
        restore_feature_state_id_encodings(&mut self.inner, snapshot)
    }

    fn restore_torch_rng_state(&mut self, rng_state: &Bound<'_, PyAny>) -> PyResult<()> {
        let bytes = torch_rng_state_bytes(rng_state)?;
        self.inner.id_rng =
            TorchMt19937::from_torch_rng_state(&bytes).map_err(PyValueError::new_err)?;
        Ok(())
    }

    fn torch_rng_state(&self) -> Vec<u8> {
        self.inner.id_rng.to_torch_rng_state_bytes()
    }

    fn id_encoding_snapshot<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        feature_state_id_encoding_snapshot(py, &self.inner)
    }
}

pub(crate) fn feature_state_recurrent_state_snapshot(
    py: Python<'_>,
    state: &FeatureState,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item(
        "card_states",
        state
            .recurrent_state_keys
            .card_states
            .iter()
            .copied()
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "note_states",
        state
            .recurrent_state_keys
            .note_states
            .iter()
            .copied()
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "deck_states",
        state
            .recurrent_state_keys
            .deck_states
            .iter()
            .copied()
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "preset_states",
        state
            .recurrent_state_keys
            .preset_states
            .iter()
            .copied()
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("global_state", state.recurrent_state_keys.global_state)?;
    Ok(dict.unbind())
}

pub(crate) fn feature_state_snapshot(py: Python<'_>, state: &FeatureState) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("first_day_offset", state.first_day_offset)?;
    dict.set_item("prev_day_offset", state.prev_day_offset)?;
    dict.set_item(
        "card_set",
        state.card_set.iter().copied().collect::<Vec<_>>(),
    )?;
    dict.set_item("card_count", state.card_count)?;
    dict.set_item("last_new_cards", state.last_new_cards.clone())?;
    dict.set_item("i", state.i)?;
    dict.set_item("last_i", state.last_i.clone())?;
    dict.set_item("today", state.today)?;
    dict.set_item("today_reviews", state.today_reviews)?;
    dict.set_item("today_new_cards", state.today_new_cards)?;
    dict.set_item("card2first_day_offset", state.card2first_day_offset.clone())?;
    dict.set_item(
        "card2elapsed_days_cumulative",
        state.card2elapsed_days_cumulative.clone(),
    )?;
    dict.set_item(
        "card2elapsed_seconds_cumulative",
        state.card2elapsed_seconds_cumulative.clone(),
    )?;
    Ok(dict.unbind())
}

pub(crate) fn restore_feature_state_snapshot(
    state: &mut FeatureState,
    snapshot: &Bound<'_, PyMapping>,
) -> PyResult<()> {
    state.first_day_offset = optional_f64_field(snapshot, "first_day_offset")?;
    state.prev_day_offset = optional_f64_field(snapshot, "prev_day_offset")?;
    state.card_set = required_i64_vec(snapshot, "card_set")?
        .into_iter()
        .collect();
    state.card_count = if snapshot.contains("card_count")? {
        let value = required_i64(snapshot, "card_count")?;
        usize::try_from(value)
            .map_err(|_| PyValueError::new_err("state field 'card_count' must be non-negative"))?
    } else {
        state.card_set.len()
    };
    state.last_new_cards = required_usize_map(snapshot, "last_new_cards")?;
    state.i = required_i64(snapshot, "i")?;
    state.last_i = required_i64_map(snapshot, "last_i")?;
    state.today = required_f64(snapshot, "today")?;
    state.today_reviews = required_i64(snapshot, "today_reviews")?;
    state.today_new_cards = required_i64(snapshot, "today_new_cards")?;
    state.card2first_day_offset = required_f64_map(snapshot, "card2first_day_offset")?;
    state.card2elapsed_days_cumulative =
        required_f64_map(snapshot, "card2elapsed_days_cumulative")?;
    state.card2elapsed_seconds_cumulative =
        required_f64_map(snapshot, "card2elapsed_seconds_cumulative")?;
    Ok(())
}

pub(crate) fn restore_feature_state_recurrent_keys(
    state: &mut FeatureState,
    snapshot: &Bound<'_, PyMapping>,
) -> PyResult<()> {
    state.recurrent_state_keys.card_states = required_i64_vec(snapshot, "card_states")?
        .into_iter()
        .collect();
    state.recurrent_state_keys.note_states = required_i64_vec(snapshot, "note_states")?
        .into_iter()
        .collect();
    state.recurrent_state_keys.deck_states = required_i64_vec(snapshot, "deck_states")?
        .into_iter()
        .collect();
    state.recurrent_state_keys.preset_states = required_i64_vec(snapshot, "preset_states")?
        .into_iter()
        .collect();
    state.recurrent_state_keys.global_state = required_bool(snapshot, "global_state")?;
    Ok(())
}

pub(crate) fn restore_feature_state_id_encodings(
    state: &mut FeatureState,
    snapshot: &Bound<'_, PyMapping>,
) -> PyResult<()> {
    let mut id_encodings = empty_id_encodings();
    for (submodule, dim) in ID_SUBMODULES {
        if !snapshot.contains(submodule)? {
            continue;
        }
        let encodings_value = snapshot.get_item(submodule)?;
        if encodings_value.is_none() {
            continue;
        }
        let encodings = encodings_value.downcast::<PyMapping>()?;
        let target = id_encodings
            .get_mut(submodule)
            .expect("id encoding map initialized for every submodule");
        for item in encodings.items()?.iter()? {
            let item = item?;
            let item = item.downcast::<PyTuple>()?;
            let id = parse_i64(&item.get_item(0)?, submodule)?;
            let encoding = parse_f32_vec(&item.get_item(1)?, submodule, dim)?;
            target.insert(id, encoding);
        }
    }
    state.id_encodings = id_encodings;
    Ok(())
}

pub(crate) fn feature_state_id_encoding_snapshot(
    py: Python<'_>,
    state: &FeatureState,
) -> PyResult<Py<PyDict>> {
    let root = PyDict::new_bound(py);
    for (submodule, encodings) in &state.id_encodings {
        let submodule_dict = PyDict::new_bound(py);
        for (id, encoding) in encodings {
            submodule_dict.set_item(id, encoding)?;
        }
        root.set_item(submodule, submodule_dict)?;
    }
    Ok(root.unbind())
}

pub(crate) fn review_from_mapping(
    review: &Bound<'_, PyMapping>,
    require_process: bool,
) -> PyResult<ReviewInput> {
    Ok(ReviewInput {
        review_id: required_i64(review, "review_id")?,
        card_id: required_i64(review, "card_id")?,
        note_id: required_id(review, "note_id")?,
        deck_id: required_id(review, "deck_id")?,
        preset_id: required_id(review, "preset_id")?,
        day_offset: required_f64(review, "day_offset")?,
        elapsed_days: required_f64(review, "elapsed_days")?,
        elapsed_seconds: required_f64(review, "elapsed_seconds")?,
        rating: if require_process {
            Some(required_i64(review, "rating")?)
        } else {
            optional_i64(review, "rating")?
        },
        duration: if require_process {
            Some(required_f64(review, "duration")?)
        } else {
            optional_f64(review, "duration")?
        },
        state: if require_process {
            Some(required_f64(review, "state")?)
        } else {
            optional_f64(review, "state")?
        },
    })
}

/// Fast process-row parser for the plain dictionaries emitted by the public
/// Python API. Fetch every field with interned keys before converting values;
/// this avoids the generic mapping protocol and repeated temporary key
/// allocation on large histories.
pub(crate) fn process_review_from_dict(review: &Bound<'_, PyDict>) -> PyResult<ReviewInput> {
    let py = review.py();
    let review_id = review.get_item(pyo3::intern!(py, "review_id"))?;
    let card_id = review.get_item(pyo3::intern!(py, "card_id"))?;
    let note_id = review.get_item(pyo3::intern!(py, "note_id"))?;
    let deck_id = review.get_item(pyo3::intern!(py, "deck_id"))?;
    let preset_id = review.get_item(pyo3::intern!(py, "preset_id"))?;
    let day_offset = review.get_item(pyo3::intern!(py, "day_offset"))?;
    let elapsed_days = review.get_item(pyo3::intern!(py, "elapsed_days"))?;
    let elapsed_seconds = review.get_item(pyo3::intern!(py, "elapsed_seconds"))?;
    let rating = review.get_item(pyo3::intern!(py, "rating"))?;
    let duration = review.get_item(pyo3::intern!(py, "duration"))?;
    let state = review.get_item(pyo3::intern!(py, "state"))?;
    let missing = [
        ("review_id", review_id.is_none()),
        ("card_id", card_id.is_none()),
        ("note_id", note_id.is_none()),
        ("deck_id", deck_id.is_none()),
        ("preset_id", preset_id.is_none()),
        ("day_offset", day_offset.is_none()),
        ("elapsed_days", elapsed_days.is_none()),
        ("elapsed_seconds", elapsed_seconds.is_none()),
        ("rating", rating.is_none()),
        ("duration", duration.is_none()),
        ("state", state.is_none()),
    ]
    .into_iter()
    .filter_map(|(field, is_missing)| is_missing.then_some(field))
    .collect::<Vec<_>>();
    if !missing.is_empty() {
        let missing = missing
            .into_iter()
            .map(|field| format!("'{field}'"))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(PyValueError::new_err(format!(
            "Review input is missing required fields: [{missing}]"
        )));
    }

    Ok(ReviewInput {
        review_id: parse_i64(
            review_id.as_ref().expect("required field validated"),
            "review_id",
        )?,
        card_id: parse_i64(
            card_id.as_ref().expect("required field validated"),
            "card_id",
        )?,
        note_id: parse_id(
            note_id.as_ref().expect("required field validated"),
            "note_id",
        )?,
        deck_id: parse_id(
            deck_id.as_ref().expect("required field validated"),
            "deck_id",
        )?,
        preset_id: parse_id(
            preset_id.as_ref().expect("required field validated"),
            "preset_id",
        )?,
        day_offset: parse_f64(
            day_offset.as_ref().expect("required field validated"),
            "day_offset",
        )?,
        elapsed_days: parse_f64(
            elapsed_days.as_ref().expect("required field validated"),
            "elapsed_days",
        )?,
        elapsed_seconds: parse_f64(
            elapsed_seconds.as_ref().expect("required field validated"),
            "elapsed_seconds",
        )?,
        rating: Some(parse_i64(
            rating.as_ref().expect("required field validated"),
            "rating",
        )?),
        duration: Some(parse_f64(
            duration.as_ref().expect("required field validated"),
            "duration",
        )?),
        state: Some(parse_f64(
            state.as_ref().expect("required field validated"),
            "state",
        )?),
    })
}

/// Parse only fields observed by immutable prediction. This avoids probing
/// process-only optional fields for every row in a collection scan.
pub(crate) fn predict_review_from_mapping(review: &Bound<'_, PyMapping>) -> PyResult<ReviewInput> {
    predict_review_from_mapping_with_optional_card_id(review, None)
}

pub(crate) fn predict_review_from_mapping_with_card_id(
    review: &Bound<'_, PyMapping>,
    card_id: i64,
) -> PyResult<ReviewInput> {
    predict_review_from_mapping_with_optional_card_id(review, Some(card_id))
}

fn predict_review_from_mapping_with_optional_card_id(
    review: &Bound<'_, PyMapping>,
    card_id: Option<i64>,
) -> PyResult<ReviewInput> {
    Ok(ReviewInput {
        review_id: required_i64(review, "review_id")?,
        card_id: match card_id {
            Some(card_id) => card_id,
            None => required_i64(review, "card_id")?,
        },
        note_id: required_id(review, "note_id")?,
        deck_id: required_id(review, "deck_id")?,
        preset_id: required_id(review, "preset_id")?,
        day_offset: required_f64(review, "day_offset")?,
        elapsed_days: required_f64(review, "elapsed_days")?,
        elapsed_seconds: required_f64(review, "elapsed_seconds")?,
        rating: None,
        duration: None,
        state: None,
    })
}

/// Fast path for the plain dictionaries emitted by the public Python API.
pub(crate) fn predict_review_from_dict(review: &Bound<'_, PyDict>) -> PyResult<ReviewInput> {
    predict_review_from_dict_with_optional_card_id(review, None)
}

pub(crate) fn predict_review_from_dict_with_card_id(
    review: &Bound<'_, PyDict>,
    card_id: i64,
) -> PyResult<ReviewInput> {
    predict_review_from_dict_with_optional_card_id(review, Some(card_id))
}

fn predict_review_from_dict_with_optional_card_id(
    review: &Bound<'_, PyDict>,
    parsed_card_id: Option<i64>,
) -> PyResult<ReviewInput> {
    let py = review.py();
    let review_id = review.get_item(pyo3::intern!(py, "review_id"))?;
    let card_id = if parsed_card_id.is_none() {
        review.get_item(pyo3::intern!(py, "card_id"))?
    } else {
        None
    };
    let note_id = review.get_item(pyo3::intern!(py, "note_id"))?;
    let deck_id = review.get_item(pyo3::intern!(py, "deck_id"))?;
    let preset_id = review.get_item(pyo3::intern!(py, "preset_id"))?;
    let day_offset = review.get_item(pyo3::intern!(py, "day_offset"))?;
    let elapsed_days = review.get_item(pyo3::intern!(py, "elapsed_days"))?;
    let elapsed_seconds = review.get_item(pyo3::intern!(py, "elapsed_seconds"))?;
    let missing = [
        ("review_id", review_id.is_none()),
        ("card_id", parsed_card_id.is_none() && card_id.is_none()),
        ("note_id", note_id.is_none()),
        ("deck_id", deck_id.is_none()),
        ("preset_id", preset_id.is_none()),
        ("day_offset", day_offset.is_none()),
        ("elapsed_days", elapsed_days.is_none()),
        ("elapsed_seconds", elapsed_seconds.is_none()),
    ]
    .into_iter()
    .filter_map(|(field, is_missing)| is_missing.then_some(field))
    .collect::<Vec<_>>();
    if !missing.is_empty() {
        let missing = missing
            .into_iter()
            .map(|field| format!("'{field}'"))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(PyValueError::new_err(format!(
            "Review input is missing required fields: [{missing}]"
        )));
    }

    Ok(ReviewInput {
        review_id: parse_i64(
            review_id.as_ref().expect("required field validated"),
            "review_id",
        )?,
        card_id: match parsed_card_id {
            Some(card_id) => card_id,
            None => parse_i64(
                card_id.as_ref().expect("required field validated"),
                "card_id",
            )?,
        },
        note_id: parse_id(
            note_id.as_ref().expect("required field validated"),
            "note_id",
        )?,
        deck_id: parse_id(
            deck_id.as_ref().expect("required field validated"),
            "deck_id",
        )?,
        preset_id: parse_id(
            preset_id.as_ref().expect("required field validated"),
            "preset_id",
        )?,
        day_offset: parse_f64(
            day_offset.as_ref().expect("required field validated"),
            "day_offset",
        )?,
        elapsed_days: parse_f64(
            elapsed_days.as_ref().expect("required field validated"),
            "elapsed_days",
        )?,
        elapsed_seconds: parse_f64(
            elapsed_seconds.as_ref().expect("required field validated"),
            "elapsed_seconds",
        )?,
        rating: None,
        duration: None,
        state: None,
    })
}

pub(crate) fn prepared_record_fields_from_mapping(
    row: &Bound<'_, PyMapping>,
) -> PyResult<PreparedRow> {
    let mut prepared = PreparedRow::new();
    prepared.insert(
        "card_id".to_string(),
        RowValue::Int(required_i64(row, "card_id")?),
    );
    prepared.insert(
        "elapsed_days".to_string(),
        RowValue::Float(required_f64(row, "elapsed_days")?),
    );
    prepared.insert(
        "elapsed_seconds".to_string(),
        RowValue::Float(required_f64(row, "elapsed_seconds")?),
    );
    prepared.insert(
        "day_offset".to_string(),
        RowValue::Float(required_f64(row, "day_offset")?),
    );
    Ok(prepared)
}

pub(crate) fn prepared_feature_fields_from_mapping(
    row: &Bound<'_, PyMapping>,
) -> PyResult<PreparedRow> {
    let mut prepared = PreparedRow::new();
    for column in CARD_FEATURE_COLUMNS {
        prepared.insert(
            column.to_string(),
            RowValue::Float(required_f64(row, column)?),
        );
    }
    for (submodule, _) in ID_SUBMODULES {
        prepared.insert(
            submodule.to_string(),
            RowValue::Int(required_i64(row, submodule)?),
        );
    }
    prepared.insert(
        "day_offset".to_string(),
        RowValue::Float(required_f64(row, "day_offset")?),
    );
    prepared.insert(
        "day_offset_first".to_string(),
        RowValue::Float(required_f64(row, "day_offset_first")?),
    );
    Ok(prepared)
}

pub(crate) fn row_to_pydict(py: Python<'_>, row: &PreparedRow) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new_bound(py);
    for (key, value) in row {
        match value {
            RowValue::Bool(value) => dict.set_item(key, value)?,
            RowValue::Float(value) => dict.set_item(key, value)?,
            RowValue::Int(value) => dict.set_item(key, value)?,
        }
    }
    Ok(dict.unbind())
}

fn required_id(review: &Bound<'_, PyMapping>, field: &str) -> PyResult<MaybeId> {
    let value = review.get_item(field)?;
    parse_id(&value, field)
}

pub(crate) fn required_i64(review: &Bound<'_, PyMapping>, field: &str) -> PyResult<i64> {
    let value = review.get_item(field)?;
    parse_i64(&value, field)
}

pub(crate) fn required_f64(review: &Bound<'_, PyMapping>, field: &str) -> PyResult<f64> {
    let value = review.get_item(field)?;
    parse_f64(&value, field)
}

pub(crate) fn required_bool(review: &Bound<'_, PyMapping>, field: &str) -> PyResult<bool> {
    let value = review.get_item(field)?;
    value
        .extract::<bool>()
        .map_err(|_| PyValueError::new_err(format!("state field '{field}' must be a bool")))
}

fn optional_i64(review: &Bound<'_, PyMapping>, field: &str) -> PyResult<Option<i64>> {
    if !review.contains(field)? {
        return Ok(None);
    }
    let value = review.get_item(field)?;
    if is_missing(&value)? {
        Ok(None)
    } else {
        Ok(Some(parse_i64(&value, field)?))
    }
}

fn optional_f64(review: &Bound<'_, PyMapping>, field: &str) -> PyResult<Option<f64>> {
    if !review.contains(field)? {
        return Ok(None);
    }
    let value = review.get_item(field)?;
    if is_missing(&value)? {
        Ok(None)
    } else {
        Ok(Some(parse_f64(&value, field)?))
    }
}

pub(crate) fn optional_f64_field(
    review: &Bound<'_, PyMapping>,
    field: &str,
) -> PyResult<Option<f64>> {
    let value = review.get_item(field)?;
    if value.is_none() {
        Ok(None)
    } else {
        Ok(Some(parse_f64(&value, field)?))
    }
}

pub(crate) fn required_i64_vec(review: &Bound<'_, PyMapping>, field: &str) -> PyResult<Vec<i64>> {
    let value = review.get_item(field)?;
    value
        .extract::<Vec<i64>>()
        .map_err(|_| PyValueError::new_err(format!("state field '{field}' must be a list of ints")))
}

pub(crate) fn required_i64_map(
    review: &Bound<'_, PyMapping>,
    field: &str,
) -> PyResult<std::collections::BTreeMap<i64, i64>> {
    let value = review.get_item(field)?;
    value
        .extract::<std::collections::BTreeMap<i64, i64>>()
        .map_err(|_| {
            PyValueError::new_err(format!(
                "state field '{field}' must be an int-to-int mapping"
            ))
        })
}

pub(crate) fn required_usize_map(
    review: &Bound<'_, PyMapping>,
    field: &str,
) -> PyResult<std::collections::BTreeMap<i64, usize>> {
    let int_map = required_i64_map(review, field)?;
    int_map
        .into_iter()
        .map(|(key, value)| {
            usize::try_from(value)
                .map(|value| (key, value))
                .map_err(|_| {
                    PyValueError::new_err(format!(
                        "state field '{field}' contains negative value for key {key}"
                    ))
                })
        })
        .collect()
}

pub(crate) fn required_f64_map(
    review: &Bound<'_, PyMapping>,
    field: &str,
) -> PyResult<std::collections::BTreeMap<i64, f64>> {
    let value = review.get_item(field)?;
    value
        .extract::<std::collections::BTreeMap<i64, f64>>()
        .map_err(|_| {
            PyValueError::new_err(format!(
                "state field '{field}' must be an int-to-float mapping"
            ))
        })
}

fn parse_id(value: &Bound<'_, PyAny>, field: &str) -> PyResult<MaybeId> {
    // Plain Python ints/floats are overwhelmingly common in prediction scans.
    // Avoid the generic pandas/NumPy missing-value machinery (attribute
    // probing plus type-name lookup) when the value can be decoded directly.
    if value.is_none() {
        return Ok(MaybeId::Missing);
    }
    if let Ok(value) = value.extract::<i64>() {
        return Ok(MaybeId::Present(value));
    }
    if let Ok(value) = value.extract::<f64>() {
        if value.is_nan() {
            return Ok(MaybeId::Missing);
        }
        if let Some(value) = integral_f64_to_i64(value) {
            return Ok(MaybeId::Present(value));
        }
        return Err(PyValueError::new_err(format!(
            "review field '{field}' must be an integer"
        )));
    }
    if is_missing(value)? {
        Ok(MaybeId::Missing)
    } else {
        Ok(MaybeId::Present(parse_i64(value, field)?))
    }
}

pub(crate) fn parse_i64(value: &Bound<'_, PyAny>, field: &str) -> PyResult<i64> {
    if let Ok(value) = value.extract::<i64>() {
        return Ok(value);
    }
    if let Ok(value) = value.extract::<f64>() {
        if value.is_nan() {
            return Err(PyValueError::new_err(format!(
                "review field '{field}' must be an integer, got missing value"
            )));
        }
        if let Some(value) = integral_f64_to_i64(value) {
            return Ok(value);
        }
        return Err(PyValueError::new_err(format!(
            "review field '{field}' must be an integer"
        )));
    }
    let value = scalar(value)?;
    if is_missing(&value)? {
        return Err(PyValueError::new_err(format!(
            "review field '{field}' must be an integer, got missing value"
        )));
    }
    if let Ok(value) = value.extract::<i64>() {
        return Ok(value);
    }
    if let Ok(value) = value.extract::<f64>() {
        if let Some(value) = integral_f64_to_i64(value) {
            return Ok(value);
        }
    }
    Err(PyValueError::new_err(format!(
        "review field '{field}' must be an integer"
    )))
}

fn integral_f64_to_i64(value: f64) -> Option<i64> {
    // i64::MAX rounds to 2^63 as f64, so the upper boundary must be strict.
    (value.is_finite()
        && value.fract() == 0.0
        && value >= i64::MIN as f64
        && value < 9_223_372_036_854_775_808.0)
        .then(|| value as i64)
}

fn parse_f64(value: &Bound<'_, PyAny>, field: &str) -> PyResult<f64> {
    if let Ok(value) = value.extract::<f64>() {
        if value.is_finite() {
            return Ok(value);
        }
        let suffix = if value.is_nan() {
            ", got missing value"
        } else {
            ""
        };
        return Err(PyValueError::new_err(format!(
            "review field '{field}' must be a finite number{suffix}"
        )));
    }
    let value = scalar(value)?;
    if is_missing(&value)? {
        return Err(PyValueError::new_err(format!(
            "review field '{field}' must be a finite number, got missing value"
        )));
    }
    let value = value.extract::<f64>().map_err(|_| {
        PyValueError::new_err(format!("review field '{field}' must be a finite number"))
    })?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(PyValueError::new_err(format!(
            "review field '{field}' must be a finite number"
        )))
    }
}

pub(crate) fn parse_f32_vec(
    value: &Bound<'_, PyAny>,
    field: &str,
    dim: usize,
) -> PyResult<Vec<f32>> {
    let value = if value.hasattr("detach")? {
        value.call_method0("detach")?.call_method0("cpu")?
    } else {
        value.clone()
    };
    let values = if let Ok(values) = value.extract::<Vec<f32>>() {
        values
    } else if value.hasattr("tolist")? {
        value.call_method0("tolist")?.extract::<Vec<f32>>()?
    } else {
        return Err(PyValueError::new_err(format!(
            "id encoding '{field}' must be a float vector"
        )));
    };
    if values.len() != dim {
        return Err(PyValueError::new_err(format!(
            "id encoding '{field}' must have length {dim}, got {}",
            values.len()
        )));
    }
    Ok(values)
}

pub(crate) fn torch_rng_state_bytes(value: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    let value = if value.hasattr("detach")? {
        value.call_method0("detach")?.call_method0("cpu")?
    } else {
        value.clone()
    };
    if let Ok(values) = value.extract::<Vec<u8>>() {
        return Ok(values);
    }
    if value.hasattr("tolist")? {
        return value.call_method0("tolist")?.extract::<Vec<u8>>();
    }
    Err(PyValueError::new_err(
        "torch RNG state must be a uint8 tensor or byte sequence",
    ))
}

fn scalar<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if value.hasattr("item")? {
        value.call_method0("item")
    } else {
        Ok(value.clone())
    }
}

fn is_missing(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    if value.is_none() {
        return Ok(true);
    }
    let type_name: String = value.get_type().getattr("__name__")?.extract()?;
    if type_name == "NAType" || type_name == "NaTType" {
        return Ok(true);
    }
    if let Ok(value) = value.extract::<f64>() {
        return Ok(value.is_nan());
    }
    Ok(false)
}
