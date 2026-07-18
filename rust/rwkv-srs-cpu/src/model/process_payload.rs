use crate::state::{MaybeId, ReviewInput};

const PROCESS_REVIEW_PAYLOAD_MAGIC: &[u8; 8] = b"RWSRSP01";
const PROCESS_REVIEW_PAYLOAD_HEADER_LEN: usize = 16;
pub(super) const PROCESS_REVIEW_PAYLOAD_RECORD_LEN: usize = 112;

pub(crate) fn parse_process_review_payload(payload: &[u8]) -> Result<Vec<ReviewInput>, String> {
    if payload.len() < PROCESS_REVIEW_PAYLOAD_HEADER_LEN {
        return Err("packed process review payload is shorter than the header".to_string());
    }
    let magic = &payload[..PROCESS_REVIEW_PAYLOAD_MAGIC.len()];
    if magic != PROCESS_REVIEW_PAYLOAD_MAGIC {
        return Err("packed process review payload has unsupported magic".to_string());
    }

    let row_count = read_u64(payload, 8, "row_count")?;
    let row_count = usize::try_from(row_count)
        .map_err(|_| "packed process review row_count does not fit usize".to_string())?;
    let expected_len = PROCESS_REVIEW_PAYLOAD_HEADER_LEN
        .checked_add(
            row_count
                .checked_mul(PROCESS_REVIEW_PAYLOAD_RECORD_LEN)
                .ok_or_else(|| "packed process review payload length overflow".to_string())?,
        )
        .ok_or_else(|| "packed process review payload length overflow".to_string())?;
    if payload.len() != expected_len {
        return Err(format!(
            "packed process review payload length mismatch: expected {expected_len} bytes for {row_count} rows, got {} bytes",
            payload.len()
        ));
    }

    parse_process_review_records(&payload[PROCESS_REVIEW_PAYLOAD_HEADER_LEN..], row_count)
}

pub(super) fn parse_process_review_record_buffer(
    records: &[u8],
) -> Result<Vec<ReviewInput>, String> {
    if records.len() % PROCESS_REVIEW_PAYLOAD_RECORD_LEN != 0 {
        return Err(format!(
            "process review record buffer length must be a multiple of {PROCESS_REVIEW_PAYLOAD_RECORD_LEN} bytes; got {} bytes",
            records.len()
        ));
    }
    parse_process_review_records(records, records.len() / PROCESS_REVIEW_PAYLOAD_RECORD_LEN)
}

fn parse_process_review_records(
    records: &[u8],
    row_count: usize,
) -> Result<Vec<ReviewInput>, String> {
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
        let rating = read_i64(records, offset, "rating")?;
        offset += 8;
        let duration = read_finite_f64(records, offset, row_index, "duration")?;
        offset += 8;
        let state = read_finite_f64(records, offset, row_index, "state")?;
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
            rating: Some(rating),
            duration: Some(duration),
            state: Some(state),
        });
    }
    Ok(rows)
}

pub(super) fn pack_process_review_payload(inputs: &[ReviewInput]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(
        PROCESS_REVIEW_PAYLOAD_HEADER_LEN + PROCESS_REVIEW_PAYLOAD_RECORD_LEN * inputs.len(),
    );
    payload.extend_from_slice(PROCESS_REVIEW_PAYLOAD_MAGIC);
    payload.extend_from_slice(&(inputs.len() as u64).to_le_bytes());
    for input in inputs {
        payload.extend_from_slice(&input.review_id.to_le_bytes());
        payload.extend_from_slice(&input.card_id.to_le_bytes());
        append_maybe_id(&mut payload, input.note_id);
        append_maybe_id(&mut payload, input.deck_id);
        append_maybe_id(&mut payload, input.preset_id);
        payload.extend_from_slice(&input.day_offset.to_le_bytes());
        payload.extend_from_slice(&input.elapsed_days.to_le_bytes());
        payload.extend_from_slice(&input.elapsed_seconds.to_le_bytes());
        payload.extend_from_slice(
            &input
                .rating
                .expect("process review payload requires rating")
                .to_le_bytes(),
        );
        payload.extend_from_slice(
            &input
                .duration
                .expect("process review payload requires duration")
                .to_le_bytes(),
        );
        payload.extend_from_slice(
            &input
                .state
                .expect("process review payload requires state")
                .to_le_bytes(),
        );
    }
    payload
}

fn append_maybe_id(payload: &mut Vec<u8>, value: MaybeId) {
    match value {
        MaybeId::Present(value) => {
            payload.extend_from_slice(&1i64.to_le_bytes());
            payload.extend_from_slice(&value.to_le_bytes());
        }
        MaybeId::Missing => {
            payload.extend_from_slice(&0i64.to_le_bytes());
            payload.extend_from_slice(&0i64.to_le_bytes());
        }
    }
}

fn read_maybe_id(
    payload: &[u8],
    offset: &mut usize,
    row_index: usize,
    field: &str,
) -> Result<MaybeId, String> {
    let present = read_i64(payload, *offset, field)?;
    *offset += 8;
    let id = read_i64(payload, *offset, field)?;
    *offset += 8;
    match present {
        0 => Ok(MaybeId::Missing),
        1 => Ok(MaybeId::Present(id)),
        _ => Err(format!(
            "packed process review row {row_index} field '{field}' has invalid presence flag {present}"
        )),
    }
}

fn read_finite_f64(
    payload: &[u8],
    offset: usize,
    row_index: usize,
    field: &str,
) -> Result<f64, String> {
    let value = read_f64(payload, offset, field)?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!(
            "packed process review row {row_index} field '{field}' must be finite"
        ))
    }
}

fn read_u64(payload: &[u8], offset: usize, field: &str) -> Result<u64, String> {
    let bytes = read_array::<8>(payload, offset, field)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_i64(payload: &[u8], offset: usize, field: &str) -> Result<i64, String> {
    let bytes = read_array::<8>(payload, offset, field)?;
    Ok(i64::from_le_bytes(bytes))
}

fn read_f64(payload: &[u8], offset: usize, field: &str) -> Result<f64, String> {
    let bytes = read_array::<8>(payload, offset, field)?;
    Ok(f64::from_le_bytes(bytes))
}

fn read_array<const N: usize>(
    payload: &[u8],
    offset: usize,
    field: &str,
) -> Result<[u8; N], String> {
    let end = offset
        .checked_add(N)
        .ok_or_else(|| format!("packed process review field '{field}' offset overflow"))?;
    let bytes = payload
        .get(offset..end)
        .ok_or_else(|| format!("packed process review field '{field}' is truncated"))?;
    bytes
        .try_into()
        .map_err(|_| format!("packed process review field '{field}' has invalid width"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packed_one_row() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(PROCESS_REVIEW_PAYLOAD_MAGIC);
        bytes.extend_from_slice(&1u64.to_le_bytes());
        for value in [1i64, 2, 1, 3, 0, 0, 1, 4] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in [5.0f64, 6.0, 7.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&3i64.to_le_bytes());
        for value in [8.0f64, 9.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn parse_process_review_payload_decodes_fixed_records() {
        let rows = parse_process_review_payload(&packed_one_row()).unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].review_id, 1);
        assert_eq!(rows[0].card_id, 2);
        assert_eq!(rows[0].note_id, MaybeId::Present(3));
        assert_eq!(rows[0].deck_id, MaybeId::Missing);
        assert_eq!(rows[0].preset_id, MaybeId::Present(4));
        assert_eq!(rows[0].day_offset, 5.0);
        assert_eq!(rows[0].elapsed_days, 6.0);
        assert_eq!(rows[0].elapsed_seconds, 7.0);
        assert_eq!(rows[0].rating, Some(3));
        assert_eq!(rows[0].duration, Some(8.0));
        assert_eq!(rows[0].state, Some(9.0));
    }

    #[test]
    fn parse_process_review_payload_rejects_wrong_length() {
        let mut bytes = packed_one_row();
        bytes.pop();

        assert!(parse_process_review_payload(&bytes)
            .unwrap_err()
            .contains("length mismatch"));
    }
}
