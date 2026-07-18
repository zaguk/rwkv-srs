use std::ffi::CStr;
use std::io::Write as _;
use std::os::raw::c_char;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyFloat, PyMapping, PyStringMethods};
use sha2::{Digest, Sha256};

use super::process_payload::{pack_process_review_payload, parse_process_review_record_buffer};
use crate::py_state::{process_review_from_dict, review_from_mapping};
use crate::state::{MaybeId, ReviewInput};

const INITIAL_HISTORY_DIGEST: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

/// Immutable normalized process rows shared by all native process executors.
///
/// Views retain one `Arc` and a range, so Python can request bounded public
/// batches without cloning the complete normalized history.
#[pyclass(frozen)]
#[derive(Clone)]
pub(crate) struct NativeReviewBatch {
    inputs: Arc<[ReviewInput]>,
    start: usize,
    end: usize,
}

impl NativeReviewBatch {
    pub(super) fn inputs(&self) -> &[ReviewInput] {
        &self.inputs[self.start..self.end]
    }

    pub(super) fn to_packed_payload(&self) -> Vec<u8> {
        pack_process_review_payload(self.inputs())
    }
}

#[pymethods]
impl NativeReviewBatch {
    #[new]
    fn new(reviews: &Bound<'_, PyAny>) -> PyResult<Self> {
        let capacity = reviews.len().unwrap_or(0);
        let mut inputs = Vec::with_capacity(capacity);
        for (index, review) in reviews.iter()?.enumerate() {
            let review = review?;
            let input = if let Ok(review) = review.downcast::<PyDict>() {
                process_review_from_dict(review)
            } else {
                review_from_mapping(review.downcast::<PyMapping>()?, true)
            }
            .map_err(|error| {
                PyValueError::new_err(format!(
                    "Review at index {index} is invalid for process_many(): {error}"
                ))
            })?;
            inputs.push(input);
        }
        let inputs: Arc<[ReviewInput]> = inputs.into();
        let end = inputs.len();
        Ok(Self {
            inputs,
            start: 0,
            end,
        })
    }

    /// Build a batch from contiguous 112-byte little-endian process records.
    /// The public Python facade documents the fixed field order.
    #[staticmethod]
    fn from_record_buffer(records: &[u8]) -> PyResult<Self> {
        let inputs = parse_process_review_record_buffer(records).map_err(PyValueError::new_err)?;
        let inputs: Arc<[ReviewInput]> = inputs.into();
        let end = inputs.len();
        Ok(Self {
            inputs,
            start: 0,
            end,
        })
    }

    fn __len__(&self) -> usize {
        self.inputs().len()
    }

    fn slice(&self, start: usize, end: usize) -> PyResult<Self> {
        let len = self.inputs().len();
        if start > end || end > len {
            return Err(PyValueError::new_err(format!(
                "ReviewBatch slice [{start}:{end}] is outside a batch of {len} rows"
            )));
        }
        Ok(Self {
            inputs: Arc::clone(&self.inputs),
            start: self.start + start,
            end: self.start + end,
        })
    }

    /// Return the exact v1 chained digest and last review ID for a prefix.
    /// Canonicalization happens before recurrent mutation, so a malformed
    /// fingerprint value cannot leave model state ahead of Python metadata.
    #[pyo3(signature = (previous_digest, count=None))]
    fn history_advance(
        &self,
        py: Python<'_>,
        previous_digest: Option<&str>,
        count: Option<usize>,
    ) -> PyResult<(Option<String>, Option<i64>)> {
        let count = count.unwrap_or_else(|| self.inputs().len());
        if count > self.inputs().len() {
            return Err(PyValueError::new_err(format!(
                "history prefix count {count} exceeds ReviewBatch length {}",
                self.inputs().len()
            )));
        }
        let inputs = &self.inputs()[..count];
        let last_review_id = inputs.last().map(|input| input.review_id);
        let Some(previous_digest) = previous_digest else {
            return Ok((None, last_review_id));
        };
        let mut digest = decode_sha256_hex(previous_digest)?;
        let mut canonical = Vec::with_capacity(256);
        for input in inputs {
            canonical.clear();
            canonical_review_bytes(py, input, &mut canonical)?;
            let mut hasher = Sha256::new();
            hasher.update(digest);
            hasher.update(&canonical);
            digest = hasher.finalize().into();
        }
        Ok((Some(encode_sha256_hex(&digest)), last_review_id))
    }

    /// Return whether this batch's prefix has an expected v1 history fingerprint.
    ///
    /// A short batch is a normal mismatch rather than an error. This preserves
    /// `check_history_consistency()` semantics while keeping parsing,
    /// canonicalization, and chained SHA-256 work inside Rust.
    #[pyo3(signature = (expected_digest, expected_count, expected_last_review_id=None))]
    fn matches_history_fingerprint(
        &self,
        py: Python<'_>,
        expected_digest: &str,
        expected_count: usize,
        expected_last_review_id: Option<i64>,
    ) -> PyResult<bool> {
        if expected_count > self.inputs().len() {
            return Ok(false);
        }
        let (actual_digest, actual_last_review_id) =
            self.history_advance(py, Some(INITIAL_HISTORY_DIGEST), Some(expected_count))?;
        Ok(actual_digest.as_deref() == Some(expected_digest)
            && actual_last_review_id == expected_last_review_id)
    }
}

fn canonical_review_bytes(
    py: Python<'_>,
    input: &ReviewInput,
    output: &mut Vec<u8>,
) -> PyResult<()> {
    write!(
        output,
        "{{\"review_id\":{},\"card_id\":{},\"note_id\":",
        input.review_id, input.card_id
    )
    .expect("writing to Vec cannot fail");
    append_maybe_id(output, input.note_id);
    output.extend_from_slice(b",\"deck_id\":");
    append_maybe_id(output, input.deck_id);
    output.extend_from_slice(b",\"preset_id\":");
    append_maybe_id(output, input.preset_id);
    output.extend_from_slice(b",\"day_offset\":");
    append_python_float(py, output, input.day_offset)?;
    output.extend_from_slice(b",\"elapsed_days\":");
    append_python_float(py, output, input.elapsed_days)?;
    output.extend_from_slice(b",\"elapsed_seconds\":");
    append_python_float(py, output, input.elapsed_seconds)?;
    write!(
        output,
        ",\"rating\":{}",
        input.rating.expect("process batch rating is present")
    )
    .expect("writing to Vec cannot fail");
    output.extend_from_slice(b",\"duration\":");
    append_python_float(
        py,
        output,
        input.duration.expect("process batch duration is present"),
    )?;
    output.extend_from_slice(b",\"state\":");
    append_python_integer_float(
        py,
        output,
        input.state.expect("process batch state is present"),
    )?;
    output.push(b'}');
    Ok(())
}

fn append_maybe_id(output: &mut Vec<u8>, value: MaybeId) {
    match value {
        MaybeId::Present(value) => {
            write!(output, "{value}").expect("writing to Vec cannot fail");
        }
        MaybeId::Missing => output.extend_from_slice(b"null"),
    }
}

fn append_python_float(py: Python<'_>, output: &mut Vec<u8>, value: f64) -> PyResult<()> {
    // This is the same formatter CPython's JSON encoder reaches through
    // `float.__repr__`, including exponent spelling and the required `.0`.
    let pointer = unsafe {
        pyo3::ffi::PyOS_double_to_string(
            value,
            b'r' as c_char,
            0,
            pyo3::ffi::Py_DTSF_ADD_DOT_0,
            std::ptr::null_mut(),
        )
    };
    if pointer.is_null() {
        return Err(PyErr::fetch(py));
    }
    let value = unsafe { CStr::from_ptr(pointer) };
    output.extend_from_slice(value.to_bytes());
    unsafe { pyo3::ffi::PyMem_Free(pointer.cast()) };
    Ok(())
}

fn append_python_integer_float(py: Python<'_>, output: &mut Vec<u8>, value: f64) -> PyResult<()> {
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(PyValueError::new_err(
            "Review field 'state' must be an integer or missing.",
        ));
    }
    if value >= i64::MIN as f64 && value < 9_223_372_036_854_775_808.0 {
        write!(output, "{}", value as i64).expect("writing to Vec cannot fail");
        return Ok(());
    }
    // Extremely large integral f64 values are uncommon, but Python's v1
    // canonicalizer converts them to an arbitrary-precision integer rather
    // than scientific notation. Use the same Python conversion on that cold
    // edge while keeping ordinary state values allocation-free.
    let integer = PyFloat::new_bound(py, value).call_method0("__int__")?;
    output.extend_from_slice(integer.str()?.to_cow()?.as_bytes());
    Ok(())
}

fn decode_sha256_hex(value: &str) -> PyResult<[u8; 32]> {
    if value.len() != 64 {
        return Err(PyValueError::new_err(
            "history digest must contain exactly 64 hexadecimal characters",
        ));
    }
    let mut decoded = [0u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        decoded[index] = (hex_nibble(pair[0])? << 4) | hex_nibble(pair[1])?;
    }
    Ok(decoded)
}

fn hex_nibble(value: u8) -> PyResult<u8> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        b'A'..=b'F' => Ok(value - b'A' + 10),
        _ => Err(PyValueError::new_err(
            "history digest must contain only hexadecimal characters",
        )),
    }
}

fn encode_sha256_hex(value: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(64);
    for byte in value {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}
