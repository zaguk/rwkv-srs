use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyMapping};

use crate::py_state::{predict_review_from_dict, predict_review_from_mapping};
use crate::state::{MaybeId, ReviewInput};

const PREDICTION_RECORD_LEN: usize = 88;

/// Immutable normalized prediction rows shared by every native predictor.
///
/// Views retain one `Arc` and a range so callers can reuse or slice a batch
/// without rebuilding Python mappings or cloning the complete input.
#[pyclass(frozen)]
#[derive(Clone)]
pub(crate) struct NativePredictionBatch {
    inputs: Arc<[ReviewInput]>,
    start: usize,
    end: usize,
}

impl NativePredictionBatch {
    pub(super) fn inputs(&self) -> &[ReviewInput] {
        &self.inputs[self.start..self.end]
    }
}

#[pymethods]
impl NativePredictionBatch {
    #[new]
    fn new(reviews: &Bound<'_, PyAny>) -> PyResult<Self> {
        let capacity = reviews.len().unwrap_or(0);
        let mut inputs = Vec::with_capacity(capacity);
        for (index, review) in reviews.iter()?.enumerate() {
            let review = review?;
            let input = if let Ok(review) = review.downcast::<PyDict>() {
                predict_review_from_dict(review)
            } else if let Ok(review) = review.downcast::<PyMapping>() {
                predict_review_from_mapping(review)
            } else {
                return Err(PyValueError::new_err(format!(
                    "Review at index {index} is invalid for predict_many(): expected a mapping"
                )));
            }
            .map_err(|error| {
                PyValueError::new_err(format!(
                    "Review at index {index} is invalid for predict_many(): {error}"
                ))
            })?;
            inputs.push(input);
        }
        Ok(Self::from_inputs(inputs))
    }

    /// Build a batch from contiguous 88-byte little-endian prediction records.
    /// The public Python facade documents the fixed field order.
    #[staticmethod]
    fn from_record_buffer(records: &[u8]) -> PyResult<Self> {
        parse_prediction_record_buffer(records)
            .map(Self::from_inputs)
            .map_err(PyValueError::new_err)
    }

    fn __len__(&self) -> usize {
        self.inputs().len()
    }

    fn slice(&self, start: usize, end: usize) -> PyResult<Self> {
        let len = self.inputs().len();
        if start > end || end > len {
            return Err(PyValueError::new_err(format!(
                "PredictionBatch slice [{start}:{end}] is outside a batch of {len} rows"
            )));
        }
        Ok(Self {
            inputs: Arc::clone(&self.inputs),
            start: self.start + start,
            end: self.start + end,
        })
    }
}

impl NativePredictionBatch {
    fn from_inputs(inputs: Vec<ReviewInput>) -> Self {
        let inputs: Arc<[ReviewInput]> = inputs.into();
        let end = inputs.len();
        Self {
            inputs,
            start: 0,
            end,
        }
    }
}

fn parse_prediction_record_buffer(records: &[u8]) -> Result<Vec<ReviewInput>, String> {
    if records.len() % PREDICTION_RECORD_LEN != 0 {
        return Err(format!(
            "prediction record buffer length must be a multiple of {PREDICTION_RECORD_LEN} bytes; got {} bytes",
            records.len()
        ));
    }

    let row_count = records.len() / PREDICTION_RECORD_LEN;
    let mut rows = Vec::with_capacity(row_count);
    let mut offset = 0;
    for row_index in 0..row_count {
        let review_id = read_i64(records, offset, "review_id")?;
        offset += 8;
        let card_id = read_i64(records, offset, "card_id")?;
        offset += 8;
        let note_id = read_maybe_id(records, &mut offset, row_index, "note_id")?;
        let deck_id = read_maybe_id(records, &mut offset, row_index, "deck_id")?;
        let preset_id = read_maybe_id(records, &mut offset, row_index, "preset_id")?;
        let day_offset = read_finite_f64(records, offset, row_index, "day_offset")?;
        offset += 8;
        let elapsed_days = read_finite_f64(records, offset, row_index, "elapsed_days")?;
        offset += 8;
        let elapsed_seconds = read_finite_f64(records, offset, row_index, "elapsed_seconds")?;
        offset += 8;

        rows.push(ReviewInput {
            review_id,
            card_id,
            note_id,
            deck_id,
            preset_id,
            day_offset,
            elapsed_days,
            elapsed_seconds,
            rating: None,
            duration: None,
            state: None,
        });
    }
    Ok(rows)
}

fn read_maybe_id(
    records: &[u8],
    offset: &mut usize,
    row_index: usize,
    field: &str,
) -> Result<MaybeId, String> {
    let present = read_i64(records, *offset, field)?;
    *offset += 8;
    let id = read_i64(records, *offset, field)?;
    *offset += 8;
    match present {
        0 => Ok(MaybeId::Missing),
        1 => Ok(MaybeId::Present(id)),
        _ => Err(format!(
            "packed prediction row {row_index} field '{field}' has invalid presence flag {present}"
        )),
    }
}

fn read_finite_f64(
    records: &[u8],
    offset: usize,
    row_index: usize,
    field: &str,
) -> Result<f64, String> {
    let value = f64::from_le_bytes(read_array(records, offset, field)?);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!(
            "packed prediction row {row_index} field '{field}' must be finite"
        ))
    }
}

fn read_i64(records: &[u8], offset: usize, field: &str) -> Result<i64, String> {
    Ok(i64::from_le_bytes(read_array(records, offset, field)?))
}

fn read_array(records: &[u8], offset: usize, field: &str) -> Result<[u8; 8], String> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| format!("packed prediction field '{field}' offset overflow"))?;
    records
        .get(offset..end)
        .ok_or_else(|| format!("packed prediction field '{field}' is truncated"))?
        .try_into()
        .map_err(|_| format!("packed prediction field '{field}' has invalid width"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_parser_accepts_empty_input() {
        assert!(parse_prediction_record_buffer(&[]).unwrap().is_empty());
    }

    #[test]
    fn record_parser_rejects_invalid_length_before_reading() {
        let error = parse_prediction_record_buffer(&[0; 87]).unwrap_err();
        assert!(error.contains("multiple of 88"));
    }
}
