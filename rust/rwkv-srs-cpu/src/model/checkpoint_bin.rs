use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{copy, BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    mem,
    path::Path,
    slice,
};

use candle_core::{bail, Device, Result, Shape, Tensor};

use crate::id_encoding::{
    empty_id_encodings, TorchMt19937, ID_SUBMODULES, TORCH_RNG_STATE_BYTE_LEN,
};
use crate::model_weights::Rwkv7RnnWeights;
use crate::state::FeatureState;

use super::state::{
    native_module_state_from_parts, FlatNativeRnnModuleState, NativeRnnModuleState,
    FLAT_STATE_CHANNELS, FLAT_STATE_HEADS, FLAT_STATE_HEAD_SIZE, FLAT_STATE_LAYER_ELEMENTS,
    FLAT_STATE_MATRIX_ELEMENTS,
};
use super::{NativeRnn, NativeRuntime};

#[cfg(test)]
thread_local! {
    static ENTITY_STATE_READ_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

const CHECKPOINT_BIN_MAGIC: &[u8] = b"RWKVPCPUBINCHK1";
const CHECKPOINT_BIN_VERSION: u32 = 2;
const LEGACY_CHECKPOINT_BIN_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub(super) struct CheckpointScope {
    pub(super) card_ids: BTreeSet<i64>,
    pub(super) note_ids: BTreeSet<i64>,
    pub(super) deck_ids: BTreeSet<i64>,
    pub(super) preset_ids: BTreeSet<i64>,
}

impl CheckpointScope {
    fn ids_for_submodule(&self, submodule: &str) -> Result<&BTreeSet<i64>> {
        match submodule {
            "card_id" => Ok(&self.card_ids),
            "note_id" => Ok(&self.note_ids),
            "deck_id" => Ok(&self.deck_ids),
            "preset_id" => Ok(&self.preset_ids),
            _ => bail!("unsupported checkpoint identity submodule {submodule:?}"),
        }
    }
}

#[derive(Clone, Copy)]
enum TensorReadMode {
    Bulk,
    #[cfg(test)]
    Scalar,
}

pub(super) fn write_checkpoint_bin_path(
    path: &Path,
    metadata_json: Vec<u8>,
    deterministic: &FeatureState,
    rnn: &NativeRnn,
) -> Result<()> {
    let file = File::create(path).map_err(|err| {
        candle_core::Error::msg(format!(
            "failed to create Rust binary checkpoint {}: {err}",
            path.display()
        ))
    })?;
    let mut writer = BufWriter::with_capacity(1024 * 1024, file);
    write_runtime_checkpoint_bin(&mut writer, &metadata_json, deterministic, rnn)?;
    writer.flush().map_err(|err| {
        candle_core::Error::msg(format!("failed to flush Rust binary checkpoint: {err}"))
    })?;
    Ok(())
}

/// Return the exact v2 byte count for canonical processed state with these
/// identity cardinalities and this runtime's model/metadata representation.
///
/// Every processed card contributes all deterministic per-card records, and
/// every normalized identity contributes one encoding and one recurrent state.
/// The caller supplies metadata separately because its JSON length belongs to
/// the Python checkpoint contract rather than the native tensor layout.
pub(super) fn expected_checkpoint_bin_size(
    rnn: &NativeRnn,
    metadata_len: usize,
    card_count: usize,
    note_count: usize,
    deck_count: usize,
    preset_count: usize,
) -> Result<usize> {
    if metadata_len == 0 {
        bail!("binary checkpoint metadata must not be empty");
    }
    let populated = card_count != 0;
    if !populated && (note_count != 0 || deck_count != 0 || preset_count != 0) {
        bail!("note_count, deck_count, and preset_count must be zero when card_count is zero");
    }
    if populated && (note_count == 0 || deck_count == 0 || preset_count == 0) {
        bail!(
            "note_count, deck_count, and preset_count must be nonzero when card_count is nonzero; missing identities are stored as normalized placeholders"
        );
    }

    let modules = &rnn.weights.rwkv_modules;
    if modules.len() != super::SRS_REVIEW_STATE_MODULES {
        bail!(
            "checkpoint size calculation expected {} RWKV state modules, got {}",
            super::SRS_REVIEW_STATE_MODULES,
            modules.len()
        );
    }
    // The forward topology is card -> deck -> note -> preset -> global, while
    // the durable state maps are named card/note/deck/preset.
    let card_state_size = expected_module_state_size(&modules[0])?;
    let deck_state_size = expected_module_state_size(&modules[1])?;
    let note_state_size = expected_module_state_size(&modules[2])?;
    let preset_state_size = expected_module_state_size(&modules[3])?;
    let global_state_size = expected_module_state_size(&modules[4])?;

    let mut size = 0usize;
    checked_size_add(&mut size, CHECKPOINT_BIN_MAGIC.len(), "checkpoint magic")?;
    checked_size_add(&mut size, mem::size_of::<u32>(), "checkpoint version")?;
    checked_size_add(&mut size, mem::size_of::<u64>(), "metadata length")?;
    checked_size_add(&mut size, metadata_len, "metadata")?;

    let option_f64_size = mem::size_of::<u8>() + if populated { mem::size_of::<f64>() } else { 0 };
    checked_size_add(&mut size, option_f64_size, "first_day_offset")?;
    checked_size_add(&mut size, option_f64_size, "prev_day_offset")?;
    checked_size_add(&mut size, mem::size_of::<u64>(), "card_count")?;
    checked_counted_size_add(&mut size, card_count, mem::size_of::<i64>(), "card_set")?;
    checked_counted_size_add(
        &mut size,
        card_count,
        mem::size_of::<i64>() + mem::size_of::<u64>(),
        "last_new_cards",
    )?;
    checked_size_add(&mut size, mem::size_of::<i64>(), "review index")?;
    checked_counted_size_add(&mut size, card_count, mem::size_of::<i64>() * 2, "last_i")?;
    checked_size_add(
        &mut size,
        mem::size_of::<f64>() + mem::size_of::<i64>() * 2,
        "today counters",
    )?;
    for field in [
        "card2first_day_offset",
        "card2elapsed_days_cumulative",
        "card2elapsed_seconds_cumulative",
    ] {
        checked_counted_size_add(
            &mut size,
            card_count,
            mem::size_of::<i64>() + mem::size_of::<f64>(),
            field,
        )?;
    }

    checked_size_add(&mut size, mem::size_of::<u64>(), "id_encodings")?;
    for (submodule, dim) in ID_SUBMODULES {
        checked_size_add(
            &mut size,
            mem::size_of::<u64>() + submodule.len(),
            "id encoding submodule name",
        )?;
        let count = match submodule {
            "card_id" => card_count,
            "note_id" => note_count,
            "deck_id" => deck_count,
            "preset_id" => preset_count,
            _ => bail!("unsupported checkpoint identity submodule {submodule:?}"),
        };
        let encoding_bytes = dim
            .checked_mul(mem::size_of::<f32>())
            .ok_or_else(|| candle_core::Error::msg("ID encoding byte size overflow"))?;
        checked_counted_size_add(
            &mut size,
            count,
            mem::size_of::<i64>() + mem::size_of::<u64>() + encoding_bytes,
            submodule,
        )?;
    }

    checked_size_add(
        &mut size,
        mem::size_of::<u64>() + TORCH_RNG_STATE_BYTE_LEN,
        "torch_rng_state",
    )?;
    for (field, count, state_size) in [
        ("card_states", card_count, card_state_size),
        ("note_states", note_count, note_state_size),
        ("deck_states", deck_count, deck_state_size),
        ("preset_states", preset_count, preset_state_size),
    ] {
        checked_counted_size_add(
            &mut size,
            count,
            mem::size_of::<i64>() + mem::size_of::<u64>() + state_size,
            field,
        )?;
    }
    checked_size_add(&mut size, mem::size_of::<u8>(), "global_state tag")?;
    if populated {
        checked_size_add(&mut size, global_state_size, "global_state")?;
    }
    Ok(size)
}

fn expected_module_state_size(module: &Rwkv7RnnWeights) -> Result<usize> {
    let mut size = 3usize
        .checked_mul(mem::size_of::<u64>())
        .ok_or_else(|| candle_core::Error::msg("module state vector size overflow"))?;
    for layer in &module.blocks {
        let heads = layer.time_mixer.n_heads;
        let head_size = layer.time_mixer.head_size;
        let channels = heads
            .checked_mul(head_size)
            .ok_or_else(|| candle_core::Error::msg("module state channel count overflow"))?;
        let recurrent_values = channels
            .checked_mul(head_size)
            .ok_or_else(|| candle_core::Error::msg("module recurrent state size overflow"))?;
        checked_size_add(
            &mut size,
            expected_tensor_size(3, channels)?,
            "time_x_shift state",
        )?;
        checked_size_add(
            &mut size,
            expected_tensor_size(5, recurrent_values)?,
            "time recurrent state",
        )?;
        checked_size_add(
            &mut size,
            expected_tensor_size(3, channels)?,
            "channel state",
        )?;
    }
    Ok(size)
}

fn expected_tensor_size(rank: usize, value_count: usize) -> Result<usize> {
    let shape_bytes = rank
        .checked_mul(mem::size_of::<u64>())
        .ok_or_else(|| candle_core::Error::msg("tensor shape byte size overflow"))?;
    let value_bytes = value_count
        .checked_mul(mem::size_of::<f32>())
        .ok_or_else(|| candle_core::Error::msg("tensor value byte size overflow"))?;
    mem::size_of::<u8>()
        .checked_add(shape_bytes)
        .and_then(|size| size.checked_add(mem::size_of::<u64>()))
        .and_then(|size| size.checked_add(value_bytes))
        .ok_or_else(|| candle_core::Error::msg("tensor byte size overflow"))
}

fn checked_counted_size_add(
    size: &mut usize,
    count: usize,
    entry_size: usize,
    field: &str,
) -> Result<()> {
    checked_size_add(size, mem::size_of::<u64>(), field)?;
    let entries_size = count.checked_mul(entry_size).ok_or_else(|| {
        candle_core::Error::msg(format!("checkpoint field {field} byte size overflow"))
    })?;
    checked_size_add(size, entries_size, field)
}

fn checked_size_add(size: &mut usize, value: usize, field: &str) -> Result<()> {
    *size = size.checked_add(value).ok_or_else(|| {
        candle_core::Error::msg(format!("checkpoint field {field} byte size overflow"))
    })?;
    Ok(())
}

pub(super) fn write_merged_checkpoint_bin_path(
    backing_path: &Path,
    path: &Path,
    metadata_json: Vec<u8>,
    runtime: &NativeRuntime,
    scope: &CheckpointScope,
) -> Result<()> {
    if metadata_json.is_empty() {
        bail!("binary checkpoint metadata must not be empty");
    }
    let source = File::open(backing_path).map_err(|err| {
        candle_core::Error::msg(format!(
            "failed to open Rust binary checkpoint {}: {err}",
            backing_path.display()
        ))
    })?;
    let destination = File::create(path).map_err(|err| {
        candle_core::Error::msg(format!(
            "failed to create Rust binary checkpoint {}: {err}",
            path.display()
        ))
    })?;
    let mut reader = BufReader::with_capacity(1024 * 1024, source);
    let mut writer = BufWriter::with_capacity(1024 * 1024, destination);
    write_merged_runtime_checkpoint_bin(&mut reader, &mut writer, &metadata_json, runtime, scope)?;
    writer.flush().map_err(|err| {
        candle_core::Error::msg(format!("failed to flush Rust binary checkpoint: {err}"))
    })
}

pub(super) fn restore_checkpoint_bin_path(
    path: &Path,
    runtime: &mut NativeRuntime,
    scope: Option<&CheckpointScope>,
) -> Result<()> {
    let file = File::open(path).map_err(|err| {
        candle_core::Error::msg(format!(
            "failed to open Rust binary checkpoint {}: {err}",
            path.display()
        ))
    })?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    restore_runtime_checkpoint_bin(&mut reader, runtime, scope)
}

fn write_runtime_checkpoint_bin<W: Write>(
    writer: &mut W,
    metadata_json: &[u8],
    deterministic: &FeatureState,
    rnn: &NativeRnn,
) -> Result<()> {
    if metadata_json.is_empty() {
        bail!("binary checkpoint metadata must not be empty");
    }
    bin_write_all(writer, CHECKPOINT_BIN_MAGIC)?;
    bin_write_u32(writer, CHECKPOINT_BIN_VERSION)?;
    bin_write_u64(writer, metadata_json.len(), "metadata_json")?;
    bin_write_all(writer, metadata_json)?;
    write_runtime_state_bin_v2(writer, deterministic, rnn)
}

fn write_merged_runtime_checkpoint_bin<R: Read + Seek, W: Write + Seek>(
    reader: &mut R,
    writer: &mut W,
    metadata_json: &[u8],
    runtime: &NativeRuntime,
    scope: &CheckpointScope,
) -> Result<()> {
    let mut magic = vec![0u8; CHECKPOINT_BIN_MAGIC.len()];
    bin_read_exact(reader, &mut magic, "binary checkpoint magic")?;
    if magic != CHECKPOINT_BIN_MAGIC {
        bail!("unsupported Rust binary checkpoint magic");
    }
    let version = bin_read_u32(reader, "binary checkpoint version")?;
    if version != CHECKPOINT_BIN_VERSION {
        bail!(
            "low-memory merge-save requires Rust binary checkpoint version {CHECKPOINT_BIN_VERSION}, got {version}"
        );
    }
    let backing_metadata_len = bin_read_usize(reader, "metadata_json")?;
    reader
        .seek(SeekFrom::Current(
            i64::try_from(backing_metadata_len)
                .map_err(|_| candle_core::Error::msg("checkpoint metadata length exceeds i64"))?,
        ))
        .map_err(|err| {
            candle_core::Error::msg(format!("failed to skip Rust checkpoint metadata: {err}"))
        })?;

    bin_write_all(writer, CHECKPOINT_BIN_MAGIC)?;
    bin_write_u32(writer, CHECKPOINT_BIN_VERSION)?;
    bin_write_u64(writer, metadata_json.len(), "metadata_json")?;
    bin_write_all(writer, metadata_json)?;
    write_merged_runtime_state_bin_v2(reader, writer, runtime, scope)
}

fn write_merged_runtime_state_bin_v2<R: Read + Seek, W: Write + Seek>(
    reader: &mut R,
    writer: &mut W,
    runtime: &NativeRuntime,
    scope: &CheckpointScope,
) -> Result<()> {
    let current = &runtime.deterministic;

    let _ = bin_read_option_f64(reader, "first_day_offset")?;
    bin_write_option_f64(writer, current.first_day_offset)?;
    let _ = bin_read_option_f64(reader, "prev_day_offset")?;
    bin_write_option_f64(writer, current.prev_day_offset)?;
    let _ = bin_read_usize(reader, "card_count")?;
    bin_write_u64(writer, current.card_count, "card_count")?;

    bin_copy_merged_i64_set(
        reader,
        writer,
        "card_set",
        &current.card_set,
        &scope.card_ids,
    )?;
    bin_copy_merged_i64_map(
        reader,
        writer,
        "last_new_cards",
        &current.last_new_cards,
        &scope.card_ids,
        |reader| bin_read_usize(reader, "last_new_cards"),
        |writer, value| bin_write_u64(writer, *value, "last_new_cards"),
    )?;

    let _ = bin_read_i64(reader, "i")?;
    bin_write_i64(writer, current.i)?;
    bin_copy_merged_i64_map(
        reader,
        writer,
        "last_i",
        &current.last_i,
        &scope.card_ids,
        |reader| bin_read_i64(reader, "last_i"),
        |writer, value| bin_write_i64(writer, *value),
    )?;

    let _ = bin_read_f64(reader, "today")?;
    bin_write_f64(writer, current.today)?;
    let _ = bin_read_i64(reader, "today_reviews")?;
    bin_write_i64(writer, current.today_reviews)?;
    let _ = bin_read_i64(reader, "today_new_cards")?;
    bin_write_i64(writer, current.today_new_cards)?;

    bin_copy_merged_i64_map(
        reader,
        writer,
        "card2first_day_offset",
        &current.card2first_day_offset,
        &scope.card_ids,
        |reader| bin_read_f64(reader, "card2first_day_offset"),
        |writer, value| bin_write_f64(writer, *value),
    )?;
    bin_copy_merged_i64_map(
        reader,
        writer,
        "card2elapsed_days_cumulative",
        &current.card2elapsed_days_cumulative,
        &scope.card_ids,
        |reader| bin_read_f64(reader, "card2elapsed_days_cumulative"),
        |writer, value| bin_write_f64(writer, *value),
    )?;
    bin_copy_merged_i64_map(
        reader,
        writer,
        "card2elapsed_seconds_cumulative",
        &current.card2elapsed_seconds_cumulative,
        &scope.card_ids,
        |reader| bin_read_f64(reader, "card2elapsed_seconds_cumulative"),
        |writer, value| bin_write_f64(writer, *value),
    )?;

    bin_copy_merged_id_encodings(reader, writer, current, scope)?;

    let _ = bin_read_bytes(reader, "torch_rng_state")?;
    bin_write_bytes(
        writer,
        &current.id_rng.to_torch_rng_state_bytes(),
        "torch_rng_state",
    )?;

    bin_copy_merged_indexed_state_map(
        reader,
        writer,
        "card_states",
        &runtime.rnn.card_states,
        &runtime.rnn.flat_cpu_state.card_states,
        &scope.card_ids,
    )?;
    bin_copy_merged_indexed_state_map(
        reader,
        writer,
        "note_states",
        &runtime.rnn.note_states,
        &runtime.rnn.flat_cpu_state.note_states,
        &scope.note_ids,
    )?;
    bin_copy_merged_indexed_state_map(
        reader,
        writer,
        "deck_states",
        &runtime.rnn.deck_states,
        &runtime.rnn.flat_cpu_state.deck_states,
        &scope.deck_ids,
    )?;
    bin_copy_merged_indexed_state_map(
        reader,
        writer,
        "preset_states",
        &runtime.rnn.preset_states,
        &runtime.rnn.flat_cpu_state.preset_states,
        &scope.preset_ids,
    )?;
    bin_write_optional_state_combined(
        writer,
        runtime.rnn.global_state.as_ref(),
        runtime.rnn.flat_cpu_state.global_state.as_ref(),
    )
}

fn bin_copy_merged_i64_set<R: Read, W: Write + Seek>(
    reader: &mut R,
    writer: &mut W,
    field: &str,
    current: &BTreeSet<i64>,
    selected: &BTreeSet<i64>,
) -> Result<()> {
    let backing_len = bin_read_usize(reader, field)?;
    let count_position = write_count_placeholder(writer, field)?;
    let mut output_count = 0usize;
    let mut current_values = current.iter().copied().peekable();
    for _ in 0..backing_len {
        let backing_value = bin_read_i64(reader, field)?;
        while current_values
            .peek()
            .is_some_and(|value| *value < backing_value)
        {
            bin_write_i64(writer, current_values.next().expect("peeked current value"))?;
            output_count += 1;
        }
        if current_values.peek().copied() == Some(backing_value) {
            bin_write_i64(
                writer,
                current_values.next().expect("matched current value"),
            )?;
            output_count += 1;
        } else if !selected.contains(&backing_value) {
            bin_write_i64(writer, backing_value)?;
            output_count += 1;
        }
    }
    for value in current_values {
        bin_write_i64(writer, value)?;
        output_count += 1;
    }
    patch_count(writer, count_position, output_count, field)
}

fn bin_copy_merged_i64_map<R, W, V, ReadValue, WriteValue>(
    reader: &mut R,
    writer: &mut W,
    field: &str,
    current: &BTreeMap<i64, V>,
    selected: &BTreeSet<i64>,
    mut read_value: ReadValue,
    mut write_value: WriteValue,
) -> Result<()>
where
    R: Read,
    W: Write + Seek,
    ReadValue: FnMut(&mut R) -> Result<V>,
    WriteValue: FnMut(&mut W, &V) -> Result<()>,
{
    let backing_len = bin_read_usize(reader, field)?;
    let count_position = write_count_placeholder(writer, field)?;
    let mut output_count = 0usize;
    let mut current_values = current.iter().peekable();
    for _ in 0..backing_len {
        let backing_key = bin_read_i64(reader, field)?;
        let backing_value = read_value(reader)?;
        while let Some((current_key, current_value)) = current_values.peek().copied() {
            if *current_key >= backing_key {
                break;
            }
            bin_write_i64(writer, *current_key)?;
            write_value(writer, current_value)?;
            current_values.next();
            output_count += 1;
        }
        if let Some((current_key, current_value)) = current_values.peek().copied() {
            if *current_key == backing_key {
                bin_write_i64(writer, *current_key)?;
                write_value(writer, current_value)?;
                current_values.next();
                output_count += 1;
                continue;
            }
        }
        if !selected.contains(&backing_key) {
            bin_write_i64(writer, backing_key)?;
            write_value(writer, &backing_value)?;
            output_count += 1;
        }
    }
    for (key, value) in current_values {
        bin_write_i64(writer, *key)?;
        write_value(writer, value)?;
        output_count += 1;
    }
    patch_count(writer, count_position, output_count, field)
}

fn bin_copy_merged_id_encodings<R: Read, W: Write + Seek>(
    reader: &mut R,
    writer: &mut W,
    current: &FeatureState,
    scope: &CheckpointScope,
) -> Result<()> {
    let module_count = bin_read_usize(reader, "id_encodings")?;
    bin_write_u64(writer, module_count, "id_encodings")?;
    for _ in 0..module_count {
        let submodule = bin_read_string(reader, "id_encodings.submodule")?;
        let Some((_, dim)) = ID_SUBMODULES
            .iter()
            .copied()
            .find(|(name, _)| *name == submodule)
        else {
            bail!("unsupported id_encodings submodule {submodule:?}");
        };
        bin_write_string(writer, &submodule)?;
        let selected = scope.ids_for_submodule(&submodule)?;
        let current_values = current
            .id_encodings
            .get(submodule.as_str())
            .expect("known identity encoding submodule");
        bin_copy_merged_i64_map(
            reader,
            writer,
            &format!("id_encodings[{submodule}]"),
            current_values,
            selected,
            |reader| {
                let value_count = bin_read_usize(reader, &submodule)?;
                if value_count != dim {
                    bail!("id_encodings[{submodule}] expected length {dim}, got {value_count}");
                }
                (0..value_count)
                    .map(|_| bin_read_f32(reader, &submodule))
                    .collect()
            },
            |writer, values| {
                if values.len() != dim {
                    bail!(
                        "id_encodings[{submodule}] expected length {dim}, got {}",
                        values.len()
                    );
                }
                bin_write_u64(writer, values.len(), &submodule)?;
                for value in values {
                    bin_write_f32(writer, *value)?;
                }
                Ok(())
            },
        )?;
    }
    Ok(())
}

fn write_count_placeholder<W: Write + Seek>(writer: &mut W, field: &str) -> Result<u64> {
    let position = writer.stream_position().map_err(|err| {
        candle_core::Error::msg(format!("failed to locate checkpoint count {field}: {err}"))
    })?;
    bin_write_u64(writer, 0, field)?;
    Ok(position)
}

fn patch_count<W: Write + Seek>(
    writer: &mut W,
    position: u64,
    count: usize,
    field: &str,
) -> Result<()> {
    let end = writer.stream_position().map_err(|err| {
        candle_core::Error::msg(format!("failed to finish checkpoint field {field}: {err}"))
    })?;
    writer.seek(SeekFrom::Start(position)).map_err(|err| {
        candle_core::Error::msg(format!("failed to patch checkpoint count {field}: {err}"))
    })?;
    bin_write_u64(writer, count, field)?;
    writer.seek(SeekFrom::Start(end)).map_err(|err| {
        candle_core::Error::msg(format!("failed to resume checkpoint field {field}: {err}"))
    })?;
    Ok(())
}

enum MergedStateEntry<'a> {
    Backing { offset: u64, byte_len: usize },
    Current(&'a NativeRnnModuleState),
    Flat(&'a FlatNativeRnnModuleState),
}

fn bin_copy_merged_indexed_state_map<'a, R: Read + Seek, W: Write>(
    reader: &mut R,
    writer: &mut W,
    field: &str,
    current: &'a BTreeMap<i64, NativeRnnModuleState>,
    flat: &'a BTreeMap<i64, FlatNativeRnnModuleState>,
    selected: &BTreeSet<i64>,
) -> Result<()> {
    let len = bin_read_usize(reader, field)?;
    let mut backing_index = bin_vec_with_capacity(len, field)?;
    for item_index in 0..len {
        let identity = bin_read_i64(reader, &format!("{field}[{item_index}].id"))?;
        let byte_len = bin_read_usize(reader, &format!("{field}[{identity}].byte_len"))?;
        backing_index.push((identity, byte_len));
    }
    let data_start = reader.stream_position().map_err(|err| {
        candle_core::Error::msg(format!(
            "failed to locate Rust checkpoint field {field}: {err}"
        ))
    })?;
    let mut offset = data_start;
    let mut entries = BTreeMap::new();
    for (identity, byte_len) in backing_index {
        if !selected.contains(&identity) {
            entries.insert(identity, MergedStateEntry::Backing { offset, byte_len });
        }
        offset = offset
            .checked_add(u64::try_from(byte_len).map_err(|_| {
                candle_core::Error::msg(format!("{field}[{identity}] length exceeds u64"))
            })?)
            .ok_or_else(|| candle_core::Error::msg(format!("{field} offset overflow")))?;
    }
    let backing_data_end = offset;
    for (identity, state) in current {
        entries.insert(*identity, MergedStateEntry::Current(state));
    }
    for (identity, state) in flat {
        let previous = entries.insert(*identity, MergedStateEntry::Flat(state));
        if matches!(
            previous,
            Some(MergedStateEntry::Current(_) | MergedStateEntry::Flat(_))
        ) {
            bail!("{field}[{identity}] exists in multiple CPU state representations");
        }
    }

    bin_write_u64(writer, entries.len(), field)?;
    for (identity, entry) in &entries {
        bin_write_i64(writer, *identity)?;
        let byte_len = match entry {
            MergedStateEntry::Backing { byte_len, .. } => *byte_len,
            MergedStateEntry::Current(state) => bin_entity_state_size(state)?,
            MergedStateEntry::Flat(state) => bin_flat_entity_state_size(state)?,
        };
        bin_write_u64(writer, byte_len, field)?;
    }
    let mut reader_position = data_start;
    for (identity, entry) in entries {
        match entry {
            MergedStateEntry::Backing { offset, byte_len } => {
                if reader_position != offset {
                    reader.seek(SeekFrom::Start(offset)).map_err(|err| {
                        candle_core::Error::msg(format!(
                            "failed to seek Rust checkpoint field {field}[{identity}]: {err}"
                        ))
                    })?;
                }
                let mut limited = (&mut *reader).take(byte_len as u64);
                let copied = copy(&mut limited, writer).map_err(|err| {
                    candle_core::Error::msg(format!(
                        "failed to copy Rust checkpoint field {field}[{identity}]: {err}"
                    ))
                })?;
                if copied != byte_len as u64 {
                    bail!("{field}[{identity}] expected {byte_len} backing bytes, copied {copied}");
                }
                reader_position = offset + byte_len as u64;
            }
            MergedStateEntry::Current(state) => bin_write_entity_state(writer, state)?,
            MergedStateEntry::Flat(state) => bin_write_flat_entity_state(writer, state)?,
        }
    }
    if reader_position != backing_data_end {
        reader
            .seek(SeekFrom::Start(backing_data_end))
            .map_err(|err| {
                candle_core::Error::msg(format!(
                    "failed to finish Rust checkpoint field {field}: {err}"
                ))
            })?;
    }
    Ok(())
}

fn restore_runtime_checkpoint_bin<R: Read + Seek>(
    reader: &mut R,
    runtime: &mut NativeRuntime,
    scope: Option<&CheckpointScope>,
) -> Result<()> {
    restore_runtime_checkpoint_bin_with_mode(reader, runtime, scope, TensorReadMode::Bulk)
}

fn restore_runtime_checkpoint_bin_with_mode<R: Read + Seek>(
    reader: &mut R,
    runtime: &mut NativeRuntime,
    scope: Option<&CheckpointScope>,
    tensor_read_mode: TensorReadMode,
) -> Result<()> {
    let mut magic = vec![0u8; CHECKPOINT_BIN_MAGIC.len()];
    bin_read_exact(reader, &mut magic, "binary checkpoint magic")?;
    if magic != CHECKPOINT_BIN_MAGIC {
        bail!("unsupported Rust binary checkpoint magic");
    }
    let version = bin_read_u32(reader, "binary checkpoint version")?;
    if !matches!(
        version,
        LEGACY_CHECKPOINT_BIN_VERSION | CHECKPOINT_BIN_VERSION
    ) {
        bail!(
            "unsupported Rust binary checkpoint version {version}, expected {LEGACY_CHECKPOINT_BIN_VERSION} or {CHECKPOINT_BIN_VERSION}"
        );
    }
    let metadata_len = bin_read_usize(reader, "metadata_json")?;
    let _metadata = bin_read_exact_vec(reader, metadata_len, "binary checkpoint metadata")?;
    if version == LEGACY_CHECKPOINT_BIN_VERSION {
        read_runtime_state_bin_v1(reader, runtime, tensor_read_mode)?;
        if let Some(scope) = scope {
            filter_runtime_state(runtime, scope);
        }
        return Ok(());
    }
    read_runtime_state_bin_v2(reader, runtime, scope, tensor_read_mode)
}

fn write_runtime_state_bin_v2<W: Write>(
    writer: &mut W,
    state: &FeatureState,
    rnn: &NativeRnn,
) -> Result<()> {
    bin_write_option_f64(writer, state.first_day_offset)?;
    bin_write_option_f64(writer, state.prev_day_offset)?;
    bin_write_u64(writer, state.card_count, "card_count")?;
    bin_write_i64_set(writer, &state.card_set)?;
    bin_write_i64_usize_map(writer, &state.last_new_cards, "last_new_cards")?;
    bin_write_i64(writer, state.i)?;
    bin_write_i64_i64_map(writer, &state.last_i)?;
    bin_write_f64(writer, state.today)?;
    bin_write_i64(writer, state.today_reviews)?;
    bin_write_i64(writer, state.today_new_cards)?;
    bin_write_i64_f64_map(writer, &state.card2first_day_offset)?;
    bin_write_i64_f64_map(writer, &state.card2elapsed_days_cumulative)?;
    bin_write_i64_f64_map(writer, &state.card2elapsed_seconds_cumulative)?;
    bin_write_id_encodings(writer, &state.id_encodings)?;
    let rng_state = state.id_rng.to_torch_rng_state_bytes();
    bin_write_bytes(writer, &rng_state, "torch_rng_state")?;
    bin_write_indexed_state_map_combined(
        writer,
        &rnn.card_states,
        &rnn.flat_cpu_state.card_states,
    )?;
    bin_write_indexed_state_map_combined(
        writer,
        &rnn.note_states,
        &rnn.flat_cpu_state.note_states,
    )?;
    bin_write_indexed_state_map_combined(
        writer,
        &rnn.deck_states,
        &rnn.flat_cpu_state.deck_states,
    )?;
    bin_write_indexed_state_map_combined(
        writer,
        &rnn.preset_states,
        &rnn.flat_cpu_state.preset_states,
    )?;
    bin_write_optional_state_combined(
        writer,
        rnn.global_state.as_ref(),
        rnn.flat_cpu_state.global_state.as_ref(),
    )?;
    Ok(())
}

fn read_runtime_state_bin_v1<R: Read>(
    reader: &mut R,
    runtime: &mut NativeRuntime,
    tensor_read_mode: TensorReadMode,
) -> Result<()> {
    let first_day_offset = bin_read_option_f64(reader, "first_day_offset")?;
    let prev_day_offset = bin_read_option_f64(reader, "prev_day_offset")?;
    let card_set = bin_read_i64_set(reader, "card_set")?;
    let last_new_cards = bin_read_i64_usize_map(reader, "last_new_cards")?;
    let i = bin_read_i64(reader, "i")?;
    let last_i = bin_read_i64_i64_map(reader, "last_i")?;
    let today = bin_read_f64(reader, "today")?;
    let today_reviews = bin_read_i64(reader, "today_reviews")?;
    let today_new_cards = bin_read_i64(reader, "today_new_cards")?;
    let card2first_day_offset = bin_read_i64_f64_map(reader, "card2first_day_offset")?;
    let card2elapsed_days_cumulative =
        bin_read_i64_f64_map(reader, "card2elapsed_days_cumulative")?;
    let card2elapsed_seconds_cumulative =
        bin_read_i64_f64_map(reader, "card2elapsed_seconds_cumulative")?;
    let id_encodings = bin_read_id_encodings(reader)?;
    let rng_state = bin_read_bytes(reader, "torch_rng_state")?;
    let card_states = bin_read_state_map(reader, "card_states", tensor_read_mode)?;
    let note_states = bin_read_state_map(reader, "note_states", tensor_read_mode)?;
    let deck_states = bin_read_state_map(reader, "deck_states", tensor_read_mode)?;
    let preset_states = bin_read_state_map(reader, "preset_states", tensor_read_mode)?;
    let global_state = bin_read_optional_state(reader, "global_state", tensor_read_mode)?;

    let mut deterministic = FeatureState::with_torch_seed(5489);
    deterministic.first_day_offset = first_day_offset;
    deterministic.prev_day_offset = prev_day_offset;
    deterministic.card_set = card_set;
    deterministic.card_count = deterministic.card_set.len();
    deterministic.last_new_cards = last_new_cards;
    deterministic.i = i;
    deterministic.last_i = last_i;
    deterministic.today = today;
    deterministic.today_reviews = today_reviews;
    deterministic.today_new_cards = today_new_cards;
    deterministic.card2first_day_offset = card2first_day_offset;
    deterministic.card2elapsed_days_cumulative = card2elapsed_days_cumulative;
    deterministic.card2elapsed_seconds_cumulative = card2elapsed_seconds_cumulative;
    deterministic.id_encodings = id_encodings;
    deterministic.id_rng =
        TorchMt19937::from_torch_rng_state(&rng_state).map_err(candle_core::Error::msg)?;

    runtime.rnn.card_states = card_states;
    runtime.rnn.note_states = note_states;
    runtime.rnn.deck_states = deck_states;
    runtime.rnn.preset_states = preset_states;
    runtime.rnn.global_state = global_state;
    runtime.rnn.flat_cpu_state.clear();
    deterministic.recurrent_state_keys.card_states =
        runtime.rnn.card_states.keys().copied().collect();
    deterministic.recurrent_state_keys.note_states =
        runtime.rnn.note_states.keys().copied().collect();
    deterministic.recurrent_state_keys.deck_states =
        runtime.rnn.deck_states.keys().copied().collect();
    deterministic.recurrent_state_keys.preset_states =
        runtime.rnn.preset_states.keys().copied().collect();
    deterministic.recurrent_state_keys.global_state = runtime.rnn.global_state.is_some();
    runtime.deterministic = deterministic;
    Ok(())
}

fn read_runtime_state_bin_v2<R: Read + Seek>(
    reader: &mut R,
    runtime: &mut NativeRuntime,
    scope: Option<&CheckpointScope>,
    tensor_read_mode: TensorReadMode,
) -> Result<()> {
    let first_day_offset = bin_read_option_f64(reader, "first_day_offset")?;
    let prev_day_offset = bin_read_option_f64(reader, "prev_day_offset")?;
    let card_count = bin_read_usize(reader, "card_count")?;
    let card_set =
        bin_read_i64_set_selected(reader, "card_set", scope.map(|value| &value.card_ids))?;
    let last_new_cards = bin_read_i64_usize_map_selected(
        reader,
        "last_new_cards",
        scope.map(|value| &value.card_ids),
    )?;
    let i = bin_read_i64(reader, "i")?;
    let last_i =
        bin_read_i64_i64_map_selected(reader, "last_i", scope.map(|value| &value.card_ids))?;
    let today = bin_read_f64(reader, "today")?;
    let today_reviews = bin_read_i64(reader, "today_reviews")?;
    let today_new_cards = bin_read_i64(reader, "today_new_cards")?;
    let card2first_day_offset = bin_read_i64_f64_map_selected(
        reader,
        "card2first_day_offset",
        scope.map(|value| &value.card_ids),
    )?;
    let card2elapsed_days_cumulative = bin_read_i64_f64_map_selected(
        reader,
        "card2elapsed_days_cumulative",
        scope.map(|value| &value.card_ids),
    )?;
    let card2elapsed_seconds_cumulative = bin_read_i64_f64_map_selected(
        reader,
        "card2elapsed_seconds_cumulative",
        scope.map(|value| &value.card_ids),
    )?;
    let id_encodings = bin_read_id_encodings_selected(reader, scope)?;
    let rng_state = bin_read_bytes(reader, "torch_rng_state")?;
    let card_states = bin_read_indexed_state_map(
        reader,
        "card_states",
        scope.map(|value| &value.card_ids),
        tensor_read_mode,
    )?;
    let note_states = bin_read_indexed_state_map(
        reader,
        "note_states",
        scope.map(|value| &value.note_ids),
        tensor_read_mode,
    )?;
    let deck_states = bin_read_indexed_state_map(
        reader,
        "deck_states",
        scope.map(|value| &value.deck_ids),
        tensor_read_mode,
    )?;
    let preset_states = bin_read_indexed_state_map(
        reader,
        "preset_states",
        scope.map(|value| &value.preset_ids),
        tensor_read_mode,
    )?;
    let global_state = bin_read_optional_state(reader, "global_state", tensor_read_mode)?;

    let mut deterministic = FeatureState::with_torch_seed(5489);
    deterministic.first_day_offset = first_day_offset;
    deterministic.prev_day_offset = prev_day_offset;
    deterministic.card_set = card_set;
    deterministic.card_count = card_count;
    deterministic.last_new_cards = last_new_cards;
    deterministic.i = i;
    deterministic.last_i = last_i;
    deterministic.today = today;
    deterministic.today_reviews = today_reviews;
    deterministic.today_new_cards = today_new_cards;
    deterministic.card2first_day_offset = card2first_day_offset;
    deterministic.card2elapsed_days_cumulative = card2elapsed_days_cumulative;
    deterministic.card2elapsed_seconds_cumulative = card2elapsed_seconds_cumulative;
    deterministic.id_encodings = id_encodings;
    deterministic.id_rng =
        TorchMt19937::from_torch_rng_state(&rng_state).map_err(candle_core::Error::msg)?;
    runtime.rnn.card_states = card_states;
    runtime.rnn.note_states = note_states;
    runtime.rnn.deck_states = deck_states;
    runtime.rnn.preset_states = preset_states;
    runtime.rnn.global_state = global_state;
    runtime.rnn.flat_cpu_state.clear();
    set_recurrent_state_keys(&mut deterministic, &runtime.rnn);
    runtime.deterministic = deterministic;
    Ok(())
}

fn filter_runtime_state(runtime: &mut NativeRuntime, scope: &CheckpointScope) {
    filter_feature_state(&mut runtime.deterministic, scope);
    runtime
        .rnn
        .card_states
        .retain(|identity, _| scope.card_ids.contains(identity));
    runtime
        .rnn
        .note_states
        .retain(|identity, _| scope.note_ids.contains(identity));
    runtime
        .rnn
        .deck_states
        .retain(|identity, _| scope.deck_ids.contains(identity));
    runtime
        .rnn
        .preset_states
        .retain(|identity, _| scope.preset_ids.contains(identity));
    set_recurrent_state_keys(&mut runtime.deterministic, &runtime.rnn);
}

fn filter_feature_state(state: &mut FeatureState, scope: &CheckpointScope) {
    state
        .card_set
        .retain(|identity| scope.card_ids.contains(identity));
    state
        .last_new_cards
        .retain(|identity, _| scope.card_ids.contains(identity));
    state
        .last_i
        .retain(|identity, _| scope.card_ids.contains(identity));
    state
        .card2first_day_offset
        .retain(|identity, _| scope.card_ids.contains(identity));
    state
        .card2elapsed_days_cumulative
        .retain(|identity, _| scope.card_ids.contains(identity));
    state
        .card2elapsed_seconds_cumulative
        .retain(|identity, _| scope.card_ids.contains(identity));
    for (submodule, _) in ID_SUBMODULES {
        let selected = scope
            .ids_for_submodule(submodule)
            .expect("ID_SUBMODULES only contains supported identity modules");
        state
            .id_encodings
            .get_mut(submodule)
            .expect("id encoding map initialized for every submodule")
            .retain(|identity, _| selected.contains(identity));
    }
}

fn set_recurrent_state_keys(state: &mut FeatureState, rnn: &NativeRnn) {
    state.recurrent_state_keys.card_states = rnn
        .card_states
        .keys()
        .chain(rnn.flat_cpu_state.card_states.keys())
        .copied()
        .collect();
    state.recurrent_state_keys.note_states = rnn
        .note_states
        .keys()
        .chain(rnn.flat_cpu_state.note_states.keys())
        .copied()
        .collect();
    state.recurrent_state_keys.deck_states = rnn
        .deck_states
        .keys()
        .chain(rnn.flat_cpu_state.deck_states.keys())
        .copied()
        .collect();
    state.recurrent_state_keys.preset_states = rnn
        .preset_states
        .keys()
        .chain(rnn.flat_cpu_state.preset_states.keys())
        .copied()
        .collect();
    state.recurrent_state_keys.global_state =
        rnn.global_state.is_some() || rnn.flat_cpu_state.global_state.is_some();
}

#[derive(Clone, Copy)]
enum CpuStateEntry<'a> {
    Canonical(&'a NativeRnnModuleState),
    Flat(&'a FlatNativeRnnModuleState),
}

fn for_each_cpu_state<'a, F>(
    canonical: &'a BTreeMap<i64, NativeRnnModuleState>,
    flat: &'a BTreeMap<i64, FlatNativeRnnModuleState>,
    mut visit: F,
) -> Result<()>
where
    F: FnMut(i64, CpuStateEntry<'a>) -> Result<()>,
{
    let mut canonical = canonical.iter().peekable();
    let mut flat = flat.iter().peekable();
    loop {
        match (canonical.peek(), flat.peek()) {
            (Some((canonical_id, _)), Some((flat_id, _))) => match canonical_id.cmp(flat_id) {
                std::cmp::Ordering::Less => {
                    let (identity, state) = canonical.next().expect("peeked canonical state");
                    visit(*identity, CpuStateEntry::Canonical(state))?;
                }
                std::cmp::Ordering::Greater => {
                    let (identity, state) = flat.next().expect("peeked flat state");
                    visit(*identity, CpuStateEntry::Flat(state))?;
                }
                std::cmp::Ordering::Equal => {
                    bail!(
                            "recurrent-state identity {canonical_id} exists in flat and canonical CPU state"
                        );
                }
            },
            (Some(_), None) => {
                let (identity, state) = canonical.next().expect("peeked canonical state");
                visit(*identity, CpuStateEntry::Canonical(state))?;
            }
            (None, Some(_)) => {
                let (identity, state) = flat.next().expect("peeked flat state");
                visit(*identity, CpuStateEntry::Flat(state))?;
            }
            (None, None) => return Ok(()),
        }
    }
}

fn bin_write_indexed_state_map_combined<W: Write>(
    writer: &mut W,
    canonical: &BTreeMap<i64, NativeRnnModuleState>,
    flat: &BTreeMap<i64, FlatNativeRnnModuleState>,
) -> Result<()> {
    let len = canonical
        .len()
        .checked_add(flat.len())
        .ok_or_else(|| candle_core::Error::msg("indexed state map length overflow"))?;
    bin_write_u64(writer, len, "indexed state map")?;
    for_each_cpu_state(canonical, flat, |identity, state| {
        bin_write_i64(writer, identity)?;
        let byte_len = match state {
            CpuStateEntry::Canonical(state) => bin_entity_state_size(state)?,
            CpuStateEntry::Flat(state) => bin_flat_entity_state_size(state)?,
        };
        bin_write_u64(writer, byte_len, "entity state")
    })?;
    for_each_cpu_state(canonical, flat, |_identity, state| match state {
        CpuStateEntry::Canonical(state) => bin_write_entity_state(writer, state),
        CpuStateEntry::Flat(state) => bin_write_flat_entity_state(writer, state),
    })
}

fn bin_read_indexed_state_map<R: Read + Seek>(
    reader: &mut R,
    field: &str,
    selected_ids: Option<&BTreeSet<i64>>,
    tensor_read_mode: TensorReadMode,
) -> Result<BTreeMap<i64, NativeRnnModuleState>> {
    let len = bin_read_usize(reader, field)?;
    let mut index = bin_vec_with_capacity(len, field)?;
    for item_index in 0..len {
        let identity = bin_read_i64(reader, &format!("{field}[{item_index}].id"))?;
        let byte_len = bin_read_usize(reader, &format!("{field}[{identity}].byte_len"))?;
        index.push((identity, byte_len));
    }

    let data_start = reader.stream_position().map_err(|err| {
        candle_core::Error::msg(format!(
            "failed to locate Rust checkpoint field {field}: {err}"
        ))
    })?;
    let mut offset = data_start;
    let mut reader_position = data_start;
    let mut states = BTreeMap::new();
    for (identity, byte_len) in index {
        let next_offset = offset
            .checked_add(u64::try_from(byte_len).map_err(|_| {
                candle_core::Error::msg(format!("{field}[{identity}] length exceeds u64"))
            })?)
            .ok_or_else(|| candle_core::Error::msg(format!("{field} offset overflow")))?;
        if selected_ids
            .map(|selected| selected.contains(&identity))
            .unwrap_or(true)
        {
            if reader_position != offset {
                reader.seek(SeekFrom::Start(offset)).map_err(|err| {
                    candle_core::Error::msg(format!(
                        "failed to seek Rust checkpoint field {field}[{identity}]: {err}"
                    ))
                })?;
            }
            let mut entry_reader = (&mut *reader).take(byte_len as u64);
            let state = bin_read_entity_state(
                &mut entry_reader,
                &format!("{field}[{identity}]"),
                tensor_read_mode,
            )?;
            let remaining = entry_reader.limit();
            if remaining != 0 {
                bail!(
                    "{field}[{identity}] expected {byte_len} bytes, consumed {}",
                    byte_len as u64 - remaining
                );
            }
            states.insert(identity, state);
            reader_position = next_offset;
        }
        offset = next_offset;
    }
    if reader_position != offset {
        reader.seek(SeekFrom::Start(offset)).map_err(|err| {
            candle_core::Error::msg(format!(
                "failed to seek past Rust checkpoint field {field}: {err}"
            ))
        })?;
    }
    Ok(states)
}

fn bin_entity_state_size(state: &NativeRnnModuleState) -> Result<usize> {
    let time_x = bin_tensor_vec_size(&state.time_x_shift_b1c_by_layer, 3)?;
    let time_state = bin_tensor_vec_size(&state.time_state_b1hkk_by_layer, 5)?;
    let channel = bin_tensor_vec_size(&state.channel_state_b1c_by_layer, 3)?;
    time_x
        .checked_add(time_state)
        .and_then(|size| size.checked_add(channel))
        .ok_or_else(|| candle_core::Error::msg("entity state byte size overflow"))
}

fn bin_flat_entity_state_size(state: &FlatNativeRnnModuleState) -> Result<usize> {
    fn tensor_vec_size(layers: usize, rank: usize, values: usize) -> Result<usize> {
        let tensor_size = mem::size_of::<u8>()
            .checked_add(
                rank.checked_mul(mem::size_of::<u64>())
                    .ok_or_else(|| candle_core::Error::msg("flat tensor shape size overflow"))?,
            )
            .and_then(|size| size.checked_add(mem::size_of::<u64>()))
            .and_then(|size| size.checked_add(values.checked_mul(mem::size_of::<f32>())?))
            .ok_or_else(|| candle_core::Error::msg("flat tensor byte size overflow"))?;
        mem::size_of::<u64>()
            .checked_add(
                layers
                    .checked_mul(tensor_size)
                    .ok_or_else(|| candle_core::Error::msg("flat tensor vector size overflow"))?,
            )
            .ok_or_else(|| candle_core::Error::msg("flat tensor vector size overflow"))
    }

    let layers = state.layers();
    let time_x = tensor_vec_size(layers, 3, FLAT_STATE_CHANNELS)?;
    let recurrent = tensor_vec_size(layers, 5, FLAT_STATE_MATRIX_ELEMENTS)?;
    let channel = tensor_vec_size(layers, 3, FLAT_STATE_CHANNELS)?;
    time_x
        .checked_add(recurrent)
        .and_then(|size| size.checked_add(channel))
        .ok_or_else(|| candle_core::Error::msg("flat entity state byte size overflow"))
}

fn bin_tensor_vec_size(tensors: &[Tensor], expected_rank: usize) -> Result<usize> {
    let mut size = mem::size_of::<u64>();
    for tensor in tensors {
        if tensor.rank() != expected_rank {
            bail!(
                "binary checkpoint tensor expected rank {expected_rank}, got shape {:?}",
                tensor.dims()
            );
        }
        let tensor_size = mem::size_of::<u8>()
            .checked_add(
                expected_rank
                    .checked_mul(mem::size_of::<u64>())
                    .ok_or_else(|| candle_core::Error::msg("tensor shape byte size overflow"))?,
            )
            .and_then(|value| value.checked_add(mem::size_of::<u64>()))
            .and_then(|value| {
                value.checked_add(tensor.elem_count().checked_mul(mem::size_of::<f32>())?)
            })
            .ok_or_else(|| candle_core::Error::msg("tensor byte size overflow"))?;
        size = size
            .checked_add(tensor_size)
            .ok_or_else(|| candle_core::Error::msg("tensor vector byte size overflow"))?;
    }
    Ok(size)
}

fn bin_read_state_map<R: Read>(
    reader: &mut R,
    field: &str,
    tensor_read_mode: TensorReadMode,
) -> Result<BTreeMap<i64, NativeRnnModuleState>> {
    let len = bin_read_usize(reader, field)?;
    let mut states = BTreeMap::new();
    for index in 0..len {
        let id = bin_read_i64(reader, &format!("{field}[{index}].id"))?;
        let state = bin_read_entity_state(reader, &format!("{field}[{id}]"), tensor_read_mode)?;
        states.insert(id, state);
    }
    Ok(states)
}

#[cfg(test)]
fn bin_write_optional_state<W: Write>(
    writer: &mut W,
    state: Option<&NativeRnnModuleState>,
) -> Result<()> {
    match state {
        Some(state) => {
            bin_write_u8(writer, 1)?;
            bin_write_entity_state(writer, state)
        }
        None => bin_write_u8(writer, 0),
    }
}

fn bin_write_optional_state_combined<W: Write>(
    writer: &mut W,
    canonical: Option<&NativeRnnModuleState>,
    flat: Option<&FlatNativeRnnModuleState>,
) -> Result<()> {
    match (canonical, flat) {
        (Some(_), Some(_)) => bail!("global state exists in flat and canonical CPU state"),
        (Some(state), None) => {
            bin_write_u8(writer, 1)?;
            bin_write_entity_state(writer, state)
        }
        (None, Some(state)) => {
            bin_write_u8(writer, 1)?;
            bin_write_flat_entity_state(writer, state)
        }
        (None, None) => bin_write_u8(writer, 0),
    }
}

fn bin_read_optional_state<R: Read>(
    reader: &mut R,
    field: &str,
    tensor_read_mode: TensorReadMode,
) -> Result<Option<NativeRnnModuleState>> {
    match bin_read_u8(reader, field)? {
        0 => Ok(None),
        1 => bin_read_entity_state(reader, field, tensor_read_mode).map(Some),
        tag => {
            bail!("{field} optional state tag must be 0 or 1, got {tag}");
        }
    }
}

fn bin_write_entity_state<W: Write>(writer: &mut W, state: &NativeRnnModuleState) -> Result<()> {
    bin_write_tensor_vec(writer, &state.time_x_shift_b1c_by_layer, 3)?;
    bin_write_tensor_vec(writer, &state.time_state_b1hkk_by_layer, 5)?;
    bin_write_tensor_vec(writer, &state.channel_state_b1c_by_layer, 3)?;
    Ok(())
}

fn bin_write_flat_entity_state<W: Write>(
    writer: &mut W,
    state: &FlatNativeRnnModuleState,
) -> Result<()> {
    let layers = state.layers();
    let values = state.values();
    if values.len() != layers * FLAT_STATE_LAYER_ELEMENTS {
        bail!("flat recurrent-state layout has the wrong size");
    }

    bin_write_u64(writer, layers, "flat time-shift tensor vector")?;
    for layer in 0..layers {
        let base = layer * FLAT_STATE_LAYER_ELEMENTS;
        bin_write_flat_tensor(
            writer,
            &[1, 1, FLAT_STATE_CHANNELS],
            &values[base..base + FLAT_STATE_CHANNELS],
        )?;
    }
    bin_write_u64(writer, layers, "flat recurrent tensor vector")?;
    for layer in 0..layers {
        let base = layer * FLAT_STATE_LAYER_ELEMENTS + FLAT_STATE_CHANNELS;
        bin_write_flat_tensor(
            writer,
            &[
                1,
                1,
                FLAT_STATE_HEADS,
                FLAT_STATE_HEAD_SIZE,
                FLAT_STATE_HEAD_SIZE,
            ],
            &values[base..base + FLAT_STATE_MATRIX_ELEMENTS],
        )?;
    }
    bin_write_u64(writer, layers, "flat channel-state tensor vector")?;
    for layer in 0..layers {
        let base =
            layer * FLAT_STATE_LAYER_ELEMENTS + FLAT_STATE_CHANNELS + FLAT_STATE_MATRIX_ELEMENTS;
        bin_write_flat_tensor(
            writer,
            &[1, 1, FLAT_STATE_CHANNELS],
            &values[base..base + FLAT_STATE_CHANNELS],
        )?;
    }
    Ok(())
}

fn bin_write_flat_tensor<W: Write>(writer: &mut W, dims: &[usize], values: &[f32]) -> Result<()> {
    let expected_values = dims
        .iter()
        .try_fold(1usize, |count, dim| count.checked_mul(*dim))
        .ok_or_else(|| candle_core::Error::msg("flat tensor element count overflow"))?;
    if expected_values != values.len() {
        bail!(
            "flat tensor shape {:?} expects {expected_values} values, got {}",
            dims,
            values.len()
        );
    }
    bin_write_u8(
        writer,
        u8::try_from(dims.len()).map_err(|_| candle_core::Error::msg("tensor rank exceeds u8"))?,
    )?;
    for dim in dims {
        bin_write_u64(writer, *dim, "flat tensor dimension")?;
    }
    bin_write_u64(writer, values.len(), "flat tensor values")?;
    if cfg!(target_endian = "little") && values.iter().all(|value| !value.is_nan()) {
        bin_write_all(writer, bytemuck::cast_slice(values))
    } else {
        for value in values {
            bin_write_f32(writer, *value)?;
        }
        Ok(())
    }
}

fn bin_read_entity_state<R: Read>(
    reader: &mut R,
    field: &str,
    tensor_read_mode: TensorReadMode,
) -> Result<NativeRnnModuleState> {
    #[cfg(test)]
    ENTITY_STATE_READ_COUNT.with(|count| count.set(count.get() + 1));
    let time_x = bin_read_tensor_vec(
        reader,
        &format!("{field}.time_x_shift_b1c_by_layer"),
        3,
        tensor_read_mode,
    )?;
    let time_recurrent = bin_read_tensor_vec(
        reader,
        &format!("{field}.time_state_b1hkk_by_layer"),
        5,
        tensor_read_mode,
    )?;
    let channel = bin_read_tensor_vec(
        reader,
        &format!("{field}.channel_state_b1c_by_layer"),
        3,
        tensor_read_mode,
    )?;
    native_module_state_from_parts(time_x, time_recurrent, channel, field)
}

fn bin_write_tensor_vec<W: Write>(
    writer: &mut W,
    tensors: &[Tensor],
    expected_rank: usize,
) -> Result<()> {
    bin_write_u64(writer, tensors.len(), "tensor vector")?;
    for tensor in tensors {
        bin_write_tensor(writer, tensor, expected_rank)?;
    }
    Ok(())
}

fn bin_read_tensor_vec<R: Read>(
    reader: &mut R,
    field: &str,
    expected_rank: usize,
    tensor_read_mode: TensorReadMode,
) -> Result<Vec<Tensor>> {
    let len = bin_read_usize(reader, field)?;
    let mut tensors = bin_vec_with_capacity(len, field)?;
    for index in 0..len {
        tensors.push(bin_read_tensor(
            reader,
            &format!("{field}[{index}]"),
            expected_rank,
            tensor_read_mode,
        )?);
    }
    Ok(tensors)
}

fn bin_write_tensor<W: Write>(writer: &mut W, tensor: &Tensor, expected_rank: usize) -> Result<()> {
    let dims = tensor.dims();
    if dims.len() != expected_rank {
        bail!(
            "binary checkpoint tensor expected rank {expected_rank}, got shape {:?}",
            dims
        );
    }
    bin_write_u8(
        writer,
        u8::try_from(dims.len()).map_err(|_| candle_core::Error::msg("tensor rank exceeds u8"))?,
    )?;
    for dim in dims {
        bin_write_u64(writer, *dim, "tensor dimension")?;
    }
    let values = tensor.flatten_all()?.to_vec1::<f32>()?;
    bin_write_u64(writer, values.len(), "tensor values")?;
    for value in values {
        bin_write_f32(writer, value)?;
    }
    Ok(())
}

fn bin_read_tensor<R: Read>(
    reader: &mut R,
    field: &str,
    expected_rank: usize,
    tensor_read_mode: TensorReadMode,
) -> Result<Tensor> {
    let rank = usize::from(bin_read_u8(reader, field)?);
    if rank != expected_rank {
        bail!("{field} expected rank {expected_rank}, got {rank}");
    }
    let mut dims = Vec::with_capacity(rank);
    for index in 0..rank {
        dims.push(bin_read_usize(reader, &format!("{field}.shape[{index}]"))?);
    }
    let expected_values = dims
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| candle_core::Error::msg(format!("{field} element count overflow")))?;
    let value_count = bin_read_usize(reader, &format!("{field}.values"))?;
    if value_count != expected_values {
        bail!("{field} expected {expected_values} values from shape, got {value_count}");
    }
    let values = match tensor_read_mode {
        TensorReadMode::Bulk => bin_read_f32_vec_bulk(reader, value_count, field)?,
        #[cfg(test)]
        TensorReadMode::Scalar => bin_read_f32_vec_scalar(reader, value_count, field)?,
    };
    Tensor::from_vec(values, Shape::from_dims(&dims), &Device::Cpu)
}

fn bin_read_f32_vec_bulk<R: Read>(
    reader: &mut R,
    value_count: usize,
    field: &str,
) -> Result<Vec<f32>> {
    let byte_count = value_count
        .checked_mul(mem::size_of::<f32>())
        .ok_or_else(|| candle_core::Error::msg(format!("{field} byte count overflow")))?;
    if byte_count == 0 {
        return Ok(Vec::new());
    }

    if cfg!(target_endian = "little") {
        let mut values: Vec<f32> = bin_vec_with_capacity(value_count, field)?;
        // SAFETY: `values` has capacity for `value_count` contiguous `f32`
        // slots. We view that spare capacity as raw bytes, fill all bytes with
        // `read_exact`, and only set the vector length after a successful full
        // read. Every possible 32-bit pattern is a valid `f32` value.
        unsafe {
            let bytes = slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), byte_count);
            bin_read_exact(reader, bytes, field)?;
            values.set_len(value_count);
        }
        Ok(values)
    } else {
        let bytes = bin_read_exact_vec(reader, byte_count, field)?;
        Ok(bytes
            .chunks_exact(mem::size_of::<f32>())
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }
}

#[cfg(test)]
fn bin_read_f32_vec_scalar<R: Read>(
    reader: &mut R,
    value_count: usize,
    field: &str,
) -> Result<Vec<f32>> {
    let mut values = bin_vec_with_capacity(value_count, field)?;
    for _ in 0..value_count {
        values.push(bin_read_f32(reader, field)?);
    }
    Ok(values)
}

fn bin_write_id_encodings<W: Write>(
    writer: &mut W,
    values: &crate::id_encoding::IdEncodings,
) -> Result<()> {
    bin_write_u64(writer, ID_SUBMODULES.len(), "id_encodings")?;
    for (submodule, dim) in ID_SUBMODULES {
        bin_write_string(writer, submodule)?;
        let encodings = values
            .get(submodule)
            .ok_or_else(|| candle_core::Error::msg(format!("missing id_encodings[{submodule}]")))?;
        bin_write_u64(writer, encodings.len(), submodule)?;
        for (id, encoding) in encodings {
            if encoding.len() != dim {
                bail!(
                    "id_encodings[{submodule}][{id}] expected length {dim}, got {}",
                    encoding.len()
                );
            }
            bin_write_i64(writer, *id)?;
            bin_write_u64(writer, encoding.len(), submodule)?;
            for value in encoding {
                bin_write_f32(writer, *value)?;
            }
        }
    }
    Ok(())
}

fn bin_read_id_encodings<R: Read>(reader: &mut R) -> Result<crate::id_encoding::IdEncodings> {
    bin_read_id_encodings_selected(reader, None)
}

fn bin_read_id_encodings_selected<R: Read>(
    reader: &mut R,
    scope: Option<&CheckpointScope>,
) -> Result<crate::id_encoding::IdEncodings> {
    let module_count = bin_read_usize(reader, "id_encodings")?;
    let mut out = empty_id_encodings();
    for _ in 0..module_count {
        let submodule = bin_read_string(reader, "id_encodings.submodule")?;
        let Some((_, dim)) = ID_SUBMODULES
            .iter()
            .copied()
            .find(|(name, _)| *name == submodule)
        else {
            bail!("unsupported id_encodings submodule {submodule:?}");
        };
        let encoding_count = bin_read_usize(reader, &format!("id_encodings[{submodule}]"))?;
        let target = out
            .get_mut(submodule.as_str())
            .expect("id encoding map initialized for every submodule");
        let selected_ids = scope
            .map(|scope| scope.ids_for_submodule(&submodule))
            .transpose()?;
        for index in 0..encoding_count {
            let id = bin_read_i64(reader, &format!("id_encodings[{submodule}][{index}].id"))?;
            let value_count = bin_read_usize(reader, &format!("id_encodings[{submodule}][{id}]"))?;
            if value_count != dim {
                bail!("id_encodings[{submodule}][{id}] expected length {dim}, got {value_count}");
            }
            let selected = selected_ids
                .map(|selected_ids| selected_ids.contains(&id))
                .unwrap_or(true);
            let mut encoding = bin_vec_with_capacity(
                if selected { value_count } else { 0 },
                &format!("id_encodings[{submodule}][{id}]"),
            )?;
            for _ in 0..value_count {
                let value = bin_read_f32(reader, &format!("id_encodings[{submodule}][{id}]"))?;
                if selected {
                    encoding.push(value);
                }
            }
            if selected {
                target.insert(id, encoding);
            }
        }
    }
    Ok(out)
}

fn bin_write_i64_set<W: Write>(writer: &mut W, values: &BTreeSet<i64>) -> Result<()> {
    bin_write_u64(writer, values.len(), "i64 set")?;
    for value in values {
        bin_write_i64(writer, *value)?;
    }
    Ok(())
}

fn bin_read_i64_set<R: Read>(reader: &mut R, field: &str) -> Result<BTreeSet<i64>> {
    bin_read_i64_set_selected(reader, field, None)
}

fn bin_read_i64_set_selected<R: Read>(
    reader: &mut R,
    field: &str,
    selected_ids: Option<&BTreeSet<i64>>,
) -> Result<BTreeSet<i64>> {
    let len = bin_read_usize(reader, field)?;
    let mut values = BTreeSet::new();
    for _ in 0..len {
        let value = bin_read_i64(reader, field)?;
        if selected_ids
            .map(|selected_ids| selected_ids.contains(&value))
            .unwrap_or(true)
        {
            values.insert(value);
        }
    }
    Ok(values)
}

fn bin_write_i64_usize_map<W: Write>(
    writer: &mut W,
    values: &BTreeMap<i64, usize>,
    field: &str,
) -> Result<()> {
    bin_write_u64(writer, values.len(), field)?;
    for (key, value) in values {
        bin_write_i64(writer, *key)?;
        bin_write_u64(writer, *value, field)?;
    }
    Ok(())
}

fn bin_read_i64_usize_map<R: Read>(reader: &mut R, field: &str) -> Result<BTreeMap<i64, usize>> {
    bin_read_i64_usize_map_selected(reader, field, None)
}

fn bin_read_i64_usize_map_selected<R: Read>(
    reader: &mut R,
    field: &str,
    selected_ids: Option<&BTreeSet<i64>>,
) -> Result<BTreeMap<i64, usize>> {
    let len = bin_read_usize(reader, field)?;
    let mut values = BTreeMap::new();
    for _ in 0..len {
        let key = bin_read_i64(reader, field)?;
        let value = bin_read_usize(reader, field)?;
        if selected_ids
            .map(|selected_ids| selected_ids.contains(&key))
            .unwrap_or(true)
        {
            values.insert(key, value);
        }
    }
    Ok(values)
}

fn bin_write_i64_i64_map<W: Write>(writer: &mut W, values: &BTreeMap<i64, i64>) -> Result<()> {
    bin_write_u64(writer, values.len(), "i64 map")?;
    for (key, value) in values {
        bin_write_i64(writer, *key)?;
        bin_write_i64(writer, *value)?;
    }
    Ok(())
}

fn bin_read_i64_i64_map<R: Read>(reader: &mut R, field: &str) -> Result<BTreeMap<i64, i64>> {
    bin_read_i64_i64_map_selected(reader, field, None)
}

fn bin_read_i64_i64_map_selected<R: Read>(
    reader: &mut R,
    field: &str,
    selected_ids: Option<&BTreeSet<i64>>,
) -> Result<BTreeMap<i64, i64>> {
    let len = bin_read_usize(reader, field)?;
    let mut values = BTreeMap::new();
    for _ in 0..len {
        let key = bin_read_i64(reader, field)?;
        let value = bin_read_i64(reader, field)?;
        if selected_ids
            .map(|selected_ids| selected_ids.contains(&key))
            .unwrap_or(true)
        {
            values.insert(key, value);
        }
    }
    Ok(values)
}

fn bin_write_i64_f64_map<W: Write>(writer: &mut W, values: &BTreeMap<i64, f64>) -> Result<()> {
    bin_write_u64(writer, values.len(), "f64 map")?;
    for (key, value) in values {
        bin_write_i64(writer, *key)?;
        bin_write_f64(writer, *value)?;
    }
    Ok(())
}

fn bin_read_i64_f64_map<R: Read>(reader: &mut R, field: &str) -> Result<BTreeMap<i64, f64>> {
    bin_read_i64_f64_map_selected(reader, field, None)
}

fn bin_read_i64_f64_map_selected<R: Read>(
    reader: &mut R,
    field: &str,
    selected_ids: Option<&BTreeSet<i64>>,
) -> Result<BTreeMap<i64, f64>> {
    let len = bin_read_usize(reader, field)?;
    let mut values = BTreeMap::new();
    for _ in 0..len {
        let key = bin_read_i64(reader, field)?;
        let value = bin_read_f64(reader, field)?;
        if selected_ids
            .map(|selected_ids| selected_ids.contains(&key))
            .unwrap_or(true)
        {
            values.insert(key, value);
        }
    }
    Ok(values)
}

fn bin_write_option_f64<W: Write>(writer: &mut W, value: Option<f64>) -> Result<()> {
    match value {
        Some(value) => {
            bin_write_u8(writer, 1)?;
            bin_write_f64(writer, value)
        }
        None => bin_write_u8(writer, 0),
    }
}

fn bin_read_option_f64<R: Read>(reader: &mut R, field: &str) -> Result<Option<f64>> {
    match bin_read_u8(reader, field)? {
        0 => Ok(None),
        1 => bin_read_f64(reader, field).map(Some),
        tag => {
            bail!("{field} option tag must be 0 or 1, got {tag}");
        }
    }
}

fn bin_write_string<W: Write>(writer: &mut W, value: &str) -> Result<()> {
    bin_write_bytes(writer, value.as_bytes(), "string")
}

fn bin_read_string<R: Read>(reader: &mut R, field: &str) -> Result<String> {
    let bytes = bin_read_bytes(reader, field)?;
    String::from_utf8(bytes)
        .map_err(|err| candle_core::Error::msg(format!("{field} is not UTF-8: {err}")))
}

fn bin_write_bytes<W: Write>(writer: &mut W, bytes: &[u8], field: &str) -> Result<()> {
    bin_write_u64(writer, bytes.len(), field)?;
    bin_write_all(writer, bytes)
}

fn bin_read_bytes<R: Read>(reader: &mut R, field: &str) -> Result<Vec<u8>> {
    let len = bin_read_usize(reader, field)?;
    bin_read_exact_vec(reader, len, field)
}

fn bin_vec_with_capacity<T>(len: usize, field: &str) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|err| {
        candle_core::Error::msg(format!(
            "Rust checkpoint field {field} declares an unsupported length {len}: {err}"
        ))
    })?;
    Ok(values)
}

fn bin_read_exact_vec<R: Read>(reader: &mut R, len: usize, field: &str) -> Result<Vec<u8>> {
    const READ_CHUNK_BYTES: usize = 64 * 1024;

    let mut bytes = Vec::new();
    let mut remaining = len;
    let mut chunk = [0u8; READ_CHUNK_BYTES];
    while remaining != 0 {
        let chunk_len = remaining.min(chunk.len());
        bin_read_exact(reader, &mut chunk[..chunk_len], field)?;
        bytes.try_reserve_exact(chunk_len).map_err(|err| {
            candle_core::Error::msg(format!(
                "Rust checkpoint field {field} declares an unsupported length {len}: {err}"
            ))
        })?;
        bytes.extend_from_slice(&chunk[..chunk_len]);
        remaining -= chunk_len;
    }
    Ok(bytes)
}

fn bin_write_u8<W: Write>(writer: &mut W, value: u8) -> Result<()> {
    bin_write_all(writer, &[value])
}

fn bin_read_u8<R: Read>(reader: &mut R, field: &str) -> Result<u8> {
    let mut bytes = [0u8; 1];
    bin_read_exact(reader, &mut bytes, field)?;
    Ok(bytes[0])
}

fn bin_write_u32<W: Write>(writer: &mut W, value: u32) -> Result<()> {
    bin_write_all(writer, &value.to_le_bytes())
}

fn bin_read_u32<R: Read>(reader: &mut R, field: &str) -> Result<u32> {
    let mut bytes = [0u8; 4];
    bin_read_exact(reader, &mut bytes, field)?;
    Ok(u32::from_le_bytes(bytes))
}

fn bin_write_u64<W: Write>(writer: &mut W, value: usize, field: &str) -> Result<()> {
    let value = u64::try_from(value)
        .map_err(|_| candle_core::Error::msg(format!("{field} length exceeds u64")))?;
    bin_write_all(writer, &value.to_le_bytes())
}

fn bin_read_usize<R: Read>(reader: &mut R, field: &str) -> Result<usize> {
    let mut bytes = [0u8; 8];
    bin_read_exact(reader, &mut bytes, field)?;
    let value = u64::from_le_bytes(bytes);
    usize::try_from(value)
        .map_err(|_| candle_core::Error::msg(format!("{field} length exceeds usize")))
}

fn bin_write_i64<W: Write>(writer: &mut W, value: i64) -> Result<()> {
    bin_write_all(writer, &value.to_le_bytes())
}

fn bin_read_i64<R: Read>(reader: &mut R, field: &str) -> Result<i64> {
    let mut bytes = [0u8; 8];
    bin_read_exact(reader, &mut bytes, field)?;
    Ok(i64::from_le_bytes(bytes))
}

fn bin_write_f64<W: Write>(writer: &mut W, value: f64) -> Result<()> {
    let bytes = if value.is_nan() {
        0x7ff8_0000_0000_0000u64.to_le_bytes()
    } else {
        value.to_le_bytes()
    };
    bin_write_all(writer, &bytes)
}

fn bin_read_f64<R: Read>(reader: &mut R, field: &str) -> Result<f64> {
    let mut bytes = [0u8; 8];
    bin_read_exact(reader, &mut bytes, field)?;
    Ok(f64::from_le_bytes(bytes))
}

fn bin_write_f32<W: Write>(writer: &mut W, value: f32) -> Result<()> {
    let bytes = if value.is_nan() {
        0x7fc0_0000u32.to_le_bytes()
    } else {
        value.to_le_bytes()
    };
    bin_write_all(writer, &bytes)
}

fn bin_read_f32<R: Read>(reader: &mut R, field: &str) -> Result<f32> {
    let mut bytes = [0u8; 4];
    bin_read_exact(reader, &mut bytes, field)?;
    Ok(f32::from_le_bytes(bytes))
}

fn bin_write_all<W: Write>(writer: &mut W, bytes: &[u8]) -> Result<()> {
    writer.write_all(bytes).map_err(|err| {
        candle_core::Error::msg(format!("failed to write Rust binary checkpoint: {err}"))
    })
}

fn bin_read_exact<R: Read>(reader: &mut R, bytes: &mut [u8], field: &str) -> Result<()> {
    reader.read_exact(bytes).map_err(|err| {
        candle_core::Error::msg(format!(
            "failed to read Rust binary checkpoint field {field}: {err}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use std::{io::Cursor, path::Path};

    use super::*;
    use crate::model::srs_review_forward;
    use crate::state::{MaybeId, ReviewInput};

    const TEST_MODEL_ID: &str = "RWKV_trained_on_101_4999";
    const TEST_CARD_ID: i64 = 101;
    const TEST_NOTE_ID: i64 = 1001;
    const TEST_DECK_ID: i64 = 201;
    const TEST_PRESET_ID: i64 = 301;
    const TEST_METADATA_JSON: &[u8] = br#"{"format":"checkpoint-bin-test"}"#;

    #[test]
    fn checkpoint_declared_lengths_fail_without_panicking_or_allocating() {
        assert!(bin_vec_with_capacity::<u8>(usize::MAX, "hostile").is_err());

        let mut empty = Cursor::new(Vec::<u8>::new());
        assert!(bin_read_exact_vec(&mut empty, usize::MAX, "hostile").is_err());

        let mut indexed = Vec::new();
        bin_write_u64(&mut indexed, usize::MAX, "hostile index").unwrap();
        let error = bin_read_indexed_state_map(
            &mut Cursor::new(indexed),
            "hostile index",
            None,
            TensorReadMode::Bulk,
        )
        .unwrap_err();
        assert!(error.to_string().contains("unsupported length"));
    }

    #[test]
    fn checkpoint_writer_canonicalizes_nan_payloads_and_signs() {
        let mut bytes = Vec::new();
        bin_write_f32(&mut bytes, f32::from_bits(0xffc0_1234)).unwrap();
        bin_write_f64(&mut bytes, f64::from_bits(0xfff8_0000_0000_1234)).unwrap();
        assert_eq!(&bytes[..4], &0x7fc0_0000u32.to_le_bytes());
        assert_eq!(&bytes[4..], &0x7ff8_0000_0000_0000u64.to_le_bytes());
    }

    #[test]
    fn expected_checkpoint_size_matches_empty_and_populated_writes() {
        let empty = empty_runtime();
        assert_eq!(
            expected_checkpoint_bin_size(&empty.rnn, TEST_METADATA_JSON.len(), 0, 0, 0, 0).unwrap(),
            checkpoint_bytes(&empty).len()
        );

        let populated = canonical_runtime_with_reused_and_changed_identities();
        assert_eq!(populated.deterministic.card_set.len(), 2);
        assert_eq!(populated.rnn.card_states.len(), 2);
        assert_eq!(populated.rnn.note_states.len(), 2);
        assert_eq!(populated.rnn.deck_states.len(), 2);
        assert_eq!(populated.rnn.preset_states.len(), 2);
        assert_eq!(
            expected_checkpoint_bin_size(&populated.rnn, TEST_METADATA_JSON.len(), 2, 2, 2, 2,)
                .unwrap(),
            checkpoint_bytes(&populated).len()
        );
    }

    #[test]
    fn flat_cpu_state_streams_byte_identical_checkpoint_and_restores() {
        let canonical = canonical_runtime_with_reused_and_changed_identities();
        let expected_prediction = prediction_logits(&canonical);
        let expected = checkpoint_bytes(&canonical);

        let mut flat = canonical_runtime_with_reused_and_changed_identities();
        move_runtime_state_to_flat(&mut flat.rnn);
        assert!(flat.rnn.card_states.is_empty());
        assert_eq!(flat.rnn.flat_cpu_state.card_states.len(), 2);
        assert_eq!(checkpoint_bytes(&flat), expected);

        let mut restored = empty_runtime();
        restore_runtime_checkpoint_bin(&mut Cursor::new(expected.as_slice()), &mut restored, None)
            .unwrap();
        assert_eq!(checkpoint_bytes(&restored), expected);
        assert_eq!(prediction_logits(&restored), expected_prediction);
    }

    #[test]
    fn expected_checkpoint_size_rejects_impossible_counts_and_overflow() {
        let runtime = empty_runtime();
        let error =
            expected_checkpoint_bin_size(&runtime.rnn, TEST_METADATA_JSON.len(), 0, 1, 0, 0)
                .unwrap_err();
        assert!(error.to_string().contains("when card_count is zero"));

        let error =
            expected_checkpoint_bin_size(&runtime.rnn, TEST_METADATA_JSON.len(), 1, 0, 1, 1)
                .unwrap_err();
        assert!(error.to_string().contains("normalized placeholders"));

        assert!(expected_checkpoint_bin_size(
            &runtime.rnn,
            TEST_METADATA_JSON.len(),
            usize::MAX,
            1,
            1,
            1,
        )
        .is_err());
    }

    #[test]
    fn bulk_f32_reader_matches_scalar_reader_bit_for_bit() {
        let expected = [
            0.0f32,
            -0.0,
            1.25,
            -123.5,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(0x7fc0_0001),
        ];
        let mut bytes = Vec::new();
        for value in expected {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        let scalar = bin_read_f32_vec_scalar(
            &mut Cursor::new(bytes.as_slice()),
            expected.len(),
            "test_scalar_values",
        )
        .unwrap();
        let bulk = bin_read_f32_vec_bulk(
            &mut Cursor::new(bytes.as_slice()),
            expected.len(),
            "test_bulk_values",
        )
        .unwrap();

        assert_eq!(
            scalar
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            bulk.iter().map(|value| value.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn bulk_checkpoint_restore_matches_scalar_restore_state_and_prediction() {
        let source_runtime = runtime_with_recurrent_state();
        let checkpoint = checkpoint_bytes(&source_runtime);

        let mut scalar_runtime = empty_runtime();
        restore_runtime_checkpoint_bin_with_mode(
            &mut Cursor::new(checkpoint.as_slice()),
            &mut scalar_runtime,
            None,
            TensorReadMode::Scalar,
        )
        .unwrap();

        let mut bulk_runtime = empty_runtime();
        restore_runtime_checkpoint_bin_with_mode(
            &mut Cursor::new(checkpoint.as_slice()),
            &mut bulk_runtime,
            None,
            TensorReadMode::Bulk,
        )
        .unwrap();

        assert_eq!(
            checkpoint_bytes(&bulk_runtime),
            checkpoint_bytes(&scalar_runtime)
        );
        assert_eq!(
            prediction_logits(&bulk_runtime),
            prediction_logits(&scalar_runtime)
        );
    }

    #[test]
    fn selective_checkpoint_restore_keeps_only_requested_entities() {
        let mut source_runtime = runtime_with_recurrent_state();
        source_runtime
            .rnn
            .card_states
            .insert(TEST_CARD_ID + 1, module_state(3, 0.501));
        source_runtime
            .rnn
            .note_states
            .insert(TEST_NOTE_ID + 1, module_state(2, 0.601));
        source_runtime
            .rnn
            .deck_states
            .insert(TEST_DECK_ID + 1, module_state(4, 0.701));
        source_runtime
            .rnn
            .preset_states
            .insert(TEST_PRESET_ID + 1, module_state(3, 0.801));
        source_runtime.deterministic.card_set =
            [TEST_CARD_ID, TEST_CARD_ID + 1].into_iter().collect();
        source_runtime.deterministic.card_count = 2;
        source_runtime.deterministic.last_i.insert(TEST_CARD_ID, 7);
        source_runtime
            .deterministic
            .last_i
            .insert(TEST_CARD_ID + 1, 8);
        let checkpoint = checkpoint_bytes(&source_runtime);

        let scope = CheckpointScope {
            card_ids: [TEST_CARD_ID].into_iter().collect(),
            note_ids: [TEST_NOTE_ID].into_iter().collect(),
            deck_ids: [TEST_DECK_ID].into_iter().collect(),
            preset_ids: [TEST_PRESET_ID].into_iter().collect(),
        };
        let mut selective_runtime = empty_runtime();
        let mut selective_reader = CountingCursor::new(&checkpoint);
        restore_runtime_checkpoint_bin_with_mode(
            &mut selective_reader,
            &mut selective_runtime,
            Some(&scope),
            TensorReadMode::Bulk,
        )
        .unwrap();
        let mut full_runtime = empty_runtime();
        let mut full_reader = CountingCursor::new(&checkpoint);
        restore_runtime_checkpoint_bin_with_mode(
            &mut full_reader,
            &mut full_runtime,
            None,
            TensorReadMode::Bulk,
        )
        .unwrap();

        assert_eq!(
            selective_runtime
                .rnn
                .card_states
                .keys()
                .copied()
                .collect::<Vec<_>>(),
            vec![TEST_CARD_ID]
        );
        assert_eq!(selective_runtime.rnn.note_states.len(), 1);
        assert_eq!(selective_runtime.rnn.deck_states.len(), 1);
        assert_eq!(selective_runtime.rnn.preset_states.len(), 1);
        assert_eq!(selective_runtime.deterministic.card_count, 2);
        assert_eq!(
            selective_runtime.deterministic.card_set,
            [TEST_CARD_ID].into_iter().collect()
        );
        assert_eq!(
            selective_runtime
                .deterministic
                .last_i
                .keys()
                .copied()
                .collect::<Vec<_>>(),
            vec![TEST_CARD_ID]
        );
        assert_eq!(
            prediction_logits(&selective_runtime),
            prediction_logits(&source_runtime)
        );
        assert!(selective_reader.bytes_read < full_reader.bytes_read);
    }

    #[test]
    fn selective_restore_supports_legacy_v1_checkpoint() {
        let mut source_runtime = runtime_with_recurrent_state();
        source_runtime
            .rnn
            .card_states
            .insert(TEST_CARD_ID + 1, module_state(3, 0.501));
        source_runtime.deterministic.card_set =
            [TEST_CARD_ID, TEST_CARD_ID + 1].into_iter().collect();
        source_runtime.deterministic.card_count = 2;
        let checkpoint = legacy_checkpoint_bytes(&source_runtime);
        let scope = CheckpointScope {
            card_ids: [TEST_CARD_ID].into_iter().collect(),
            note_ids: [TEST_NOTE_ID].into_iter().collect(),
            deck_ids: [TEST_DECK_ID].into_iter().collect(),
            preset_ids: [TEST_PRESET_ID].into_iter().collect(),
        };

        let mut restored = empty_runtime();
        restore_runtime_checkpoint_bin_with_mode(
            &mut Cursor::new(checkpoint.as_slice()),
            &mut restored,
            Some(&scope),
            TensorReadMode::Bulk,
        )
        .unwrap();

        assert_eq!(restored.deterministic.card_count, 2);
        assert_eq!(restored.rnn.card_states.len(), 1);
        assert!(restored.rnn.card_states.contains_key(&TEST_CARD_ID));
        assert_eq!(
            prediction_logits(&restored),
            prediction_logits(&source_runtime)
        );

        let mut output = Cursor::new(Vec::new());
        let error = write_merged_runtime_checkpoint_bin(
            &mut Cursor::new(checkpoint),
            &mut output,
            TEST_METADATA_JSON,
            &restored,
            &scope,
        )
        .unwrap_err();
        assert!(error.to_string().contains("low-memory merge-save requires"));
    }

    #[test]
    fn scoped_merge_streams_untouched_states_into_full_checkpoint() {
        let mut source_runtime = runtime_with_recurrent_state();
        source_runtime
            .rnn
            .card_states
            .insert(TEST_CARD_ID + 1, module_state(3, 0.901));
        source_runtime.deterministic.card_set =
            [TEST_CARD_ID, TEST_CARD_ID + 1].into_iter().collect();
        source_runtime.deterministic.card_count = 2;
        source_runtime.deterministic.last_i.insert(TEST_CARD_ID, 7);
        source_runtime
            .deterministic
            .last_i
            .insert(TEST_CARD_ID + 1, 8);
        let backing = checkpoint_bytes(&source_runtime);
        let scope = CheckpointScope {
            card_ids: [TEST_CARD_ID].into_iter().collect(),
            note_ids: [TEST_NOTE_ID].into_iter().collect(),
            deck_ids: [TEST_DECK_ID].into_iter().collect(),
            preset_ids: [TEST_PRESET_ID].into_iter().collect(),
        };

        let mut partial = empty_runtime();
        restore_runtime_checkpoint_bin(
            &mut Cursor::new(backing.as_slice()),
            &mut partial,
            Some(&scope),
        )
        .unwrap();
        let updated_card_state = module_state(3, 1.234);
        partial
            .rnn
            .card_states
            .insert(TEST_CARD_ID, updated_card_state.clone());
        partial.deterministic.last_i.insert(TEST_CARD_ID, 99);

        source_runtime
            .rnn
            .card_states
            .insert(TEST_CARD_ID, updated_card_state);
        source_runtime.deterministic.last_i.insert(TEST_CARD_ID, 99);
        let expected = checkpoint_bytes(&source_runtime);

        ENTITY_STATE_READ_COUNT.with(|count| count.set(0));
        let mut merged = Cursor::new(Vec::new());
        write_merged_runtime_checkpoint_bin(
            &mut Cursor::new(backing.as_slice()),
            &mut merged,
            TEST_METADATA_JSON,
            &partial,
            &scope,
        )
        .unwrap();

        assert_eq!(merged.into_inner(), expected);
        ENTITY_STATE_READ_COUNT.with(|count| assert_eq!(count.get(), 0));
        assert_eq!(partial.rnn.card_states.len(), 1);
        assert!(partial.rnn.card_states.contains_key(&TEST_CARD_ID));

        move_runtime_state_to_flat(&mut partial.rnn);
        let mut flat_merged = Cursor::new(Vec::new());
        write_merged_runtime_checkpoint_bin(
            &mut Cursor::new(backing),
            &mut flat_merged,
            TEST_METADATA_JSON,
            &partial,
            &scope,
        )
        .unwrap();
        assert_eq!(flat_merged.into_inner(), expected);
        assert!(partial.rnn.card_states.is_empty());
        assert_eq!(partial.rnn.flat_cpu_state.card_states.len(), 1);
    }

    fn runtime_with_recurrent_state() -> NativeRuntime {
        let mut runtime = empty_runtime();
        runtime
            .rnn
            .card_states
            .insert(TEST_CARD_ID, module_state(3, 0.001));
        runtime
            .rnn
            .note_states
            .insert(TEST_NOTE_ID, module_state(2, 0.101));
        runtime
            .rnn
            .deck_states
            .insert(TEST_DECK_ID, module_state(4, 0.201));
        runtime
            .rnn
            .preset_states
            .insert(TEST_PRESET_ID, module_state(3, 0.301));
        runtime.rnn.global_state = Some(module_state(4, 0.401));
        runtime
    }

    fn canonical_runtime_with_reused_and_changed_identities() -> NativeRuntime {
        let mut runtime = empty_runtime();
        let identities = [
            (TEST_CARD_ID, TEST_NOTE_ID, TEST_DECK_ID, TEST_PRESET_ID),
            (
                TEST_CARD_ID + 1,
                TEST_NOTE_ID,
                TEST_DECK_ID + 1,
                TEST_PRESET_ID,
            ),
            (
                TEST_CARD_ID,
                TEST_NOTE_ID + 1,
                TEST_DECK_ID + 1,
                TEST_PRESET_ID + 1,
            ),
        ];
        for (index, (card_id, note_id, deck_id, preset_id)) in identities.into_iter().enumerate() {
            let input = ReviewInput {
                review_id: index as i64 + 1,
                card_id,
                note_id: MaybeId::Present(note_id),
                deck_id: MaybeId::Present(deck_id),
                preset_id: MaybeId::Present(preset_id),
                day_offset: index as f64,
                elapsed_days: 1.0,
                elapsed_seconds: 60.0,
                rating: Some(3),
                duration: Some(5.0),
                state: Some(2.0),
            };
            let mut features = Vec::new();
            runtime
                .deterministic
                .append_process_feature_pair(&input, &mut features)
                .unwrap();
            runtime
                .rnn
                .card_states
                .insert(card_id, module_state(3, 0.001 + index as f32));
            runtime
                .rnn
                .note_states
                .insert(note_id, module_state(2, 0.101 + index as f32));
            runtime
                .rnn
                .deck_states
                .insert(deck_id, module_state(4, 0.201 + index as f32));
            runtime
                .rnn
                .preset_states
                .insert(preset_id, module_state(3, 0.301 + index as f32));
        }
        runtime.rnn.global_state = Some(module_state(4, 0.401));
        runtime
    }

    fn empty_runtime() -> NativeRuntime {
        NativeRuntime {
            deterministic: FeatureState::with_torch_seed(5489),
            rnn: NativeRnn::from_checkpoint(model_path()).unwrap(),
            undo_stack: Default::default(),
            undo_limit: 30,
            loaded_scope: None,
            gpu_process_committed_rows: 0,
            gpu_process_output: (Vec::new(), None, None),
            live_session: None,
            pending_live_session: None,
            next_live_session_token: 1,
        }
    }

    fn checkpoint_bytes(runtime: &NativeRuntime) -> Vec<u8> {
        let mut bytes = Vec::new();
        write_runtime_checkpoint_bin(
            &mut bytes,
            TEST_METADATA_JSON,
            &runtime.deterministic,
            &runtime.rnn,
        )
        .unwrap();
        bytes
    }

    fn move_runtime_state_to_flat(rnn: &mut NativeRnn) {
        fn move_map(
            canonical: &mut BTreeMap<i64, NativeRnnModuleState>,
            flat: &mut BTreeMap<i64, FlatNativeRnnModuleState>,
        ) {
            for (identity, state) in std::mem::take(canonical) {
                flat.insert(identity, flat_state(&state));
            }
        }

        move_map(&mut rnn.card_states, &mut rnn.flat_cpu_state.card_states);
        move_map(&mut rnn.note_states, &mut rnn.flat_cpu_state.note_states);
        move_map(&mut rnn.deck_states, &mut rnn.flat_cpu_state.deck_states);
        move_map(
            &mut rnn.preset_states,
            &mut rnn.flat_cpu_state.preset_states,
        );
        rnn.flat_cpu_state.global_state = rnn.global_state.take().map(|state| flat_state(&state));
    }

    fn flat_state(state: &NativeRnnModuleState) -> FlatNativeRnnModuleState {
        let layers = state.time_x_shift_b1c_by_layer.len();
        let mut values = Vec::with_capacity(layers * FLAT_STATE_LAYER_ELEMENTS);
        for layer in 0..layers {
            values.extend(
                state.time_x_shift_b1c_by_layer[layer]
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
            );
            values.extend(
                state.time_state_b1hkk_by_layer[layer]
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
            );
            values.extend(
                state.channel_state_b1c_by_layer[layer]
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
            );
        }
        FlatNativeRnnModuleState::from_shared_values(values.into(), 0, layers).unwrap()
    }

    fn legacy_checkpoint_bytes(runtime: &NativeRuntime) -> Vec<u8> {
        let mut bytes = Vec::new();
        bin_write_all(&mut bytes, CHECKPOINT_BIN_MAGIC).unwrap();
        bin_write_u32(&mut bytes, LEGACY_CHECKPOINT_BIN_VERSION).unwrap();
        bin_write_u64(&mut bytes, TEST_METADATA_JSON.len(), "metadata_json").unwrap();
        bin_write_all(&mut bytes, TEST_METADATA_JSON).unwrap();
        let state = &runtime.deterministic;
        bin_write_option_f64(&mut bytes, state.first_day_offset).unwrap();
        bin_write_option_f64(&mut bytes, state.prev_day_offset).unwrap();
        bin_write_i64_set(&mut bytes, &state.card_set).unwrap();
        bin_write_i64_usize_map(&mut bytes, &state.last_new_cards, "last_new_cards").unwrap();
        bin_write_i64(&mut bytes, state.i).unwrap();
        bin_write_i64_i64_map(&mut bytes, &state.last_i).unwrap();
        bin_write_f64(&mut bytes, state.today).unwrap();
        bin_write_i64(&mut bytes, state.today_reviews).unwrap();
        bin_write_i64(&mut bytes, state.today_new_cards).unwrap();
        bin_write_i64_f64_map(&mut bytes, &state.card2first_day_offset).unwrap();
        bin_write_i64_f64_map(&mut bytes, &state.card2elapsed_days_cumulative).unwrap();
        bin_write_i64_f64_map(&mut bytes, &state.card2elapsed_seconds_cumulative).unwrap();
        bin_write_id_encodings(&mut bytes, &state.id_encodings).unwrap();
        bin_write_bytes(
            &mut bytes,
            &state.id_rng.to_torch_rng_state_bytes(),
            "torch_rng_state",
        )
        .unwrap();
        legacy_write_state_map(&mut bytes, &runtime.rnn.card_states);
        legacy_write_state_map(&mut bytes, &runtime.rnn.note_states);
        legacy_write_state_map(&mut bytes, &runtime.rnn.deck_states);
        legacy_write_state_map(&mut bytes, &runtime.rnn.preset_states);
        bin_write_optional_state(&mut bytes, runtime.rnn.global_state.as_ref()).unwrap();
        bytes
    }

    fn legacy_write_state_map(writer: &mut Vec<u8>, states: &BTreeMap<i64, NativeRnnModuleState>) {
        bin_write_u64(writer, states.len(), "state map").unwrap();
        for (identity, state) in states {
            bin_write_i64(writer, *identity).unwrap();
            bin_write_entity_state(writer, state).unwrap();
        }
    }

    struct CountingCursor {
        inner: Cursor<Vec<u8>>,
        bytes_read: usize,
    }

    impl CountingCursor {
        fn new(bytes: &[u8]) -> Self {
            Self {
                inner: Cursor::new(bytes.to_vec()),
                bytes_read: 0,
            }
        }
    }

    impl Read for CountingCursor {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            let count = self.inner.read(buffer)?;
            self.bytes_read += count;
            Ok(count)
        }
    }

    impl Seek for CountingCursor {
        fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
            self.inner.seek(position)
        }
    }

    fn prediction_logits(runtime: &NativeRuntime) -> Vec<Vec<f32>> {
        let (_, feature_dim) = runtime
            .rnn
            .weights
            .features2card
            .input_linear
            .weight
            .dims2()
            .unwrap();
        let features = Tensor::from_vec(
            (0..feature_dim)
                .map(|index| ((index % 17) as f32 - 8.0) / 100.0)
                .collect::<Vec<_>>(),
            (1usize, feature_dim),
            &Device::Cpu,
        )
        .unwrap();
        let (time_x_shift, time_state, channel_state) =
            runtime
                .rnn
                .state_inputs(TEST_CARD_ID, TEST_NOTE_ID, TEST_DECK_ID, TEST_PRESET_ID);
        let (_, _, logits, _, _, _) = srs_review_forward(
            &runtime.rnn.weights,
            &features,
            Some(&time_x_shift),
            Some(&time_state),
            Some(&channel_state),
            false,
        )
        .unwrap();
        logits.to_vec2::<f32>().unwrap()
    }

    fn module_state(layer_count: usize, seed: f32) -> NativeRnnModuleState {
        let time_x = (0..layer_count)
            .map(|layer_index| tensor_with_values(&[1, 1, 128], seed, layer_index))
            .collect();
        let time_recurrent = (0..layer_count)
            .map(|layer_index| tensor_with_values(&[1, 1, 4, 32, 32], seed + 0.01, layer_index))
            .collect();
        let channel = (0..layer_count)
            .map(|layer_index| tensor_with_values(&[1, 1, 128], seed + 0.02, layer_index))
            .collect();
        native_module_state_from_parts(time_x, time_recurrent, channel, "test_state").unwrap()
    }

    fn tensor_with_values(shape: &[usize], seed: f32, layer_index: usize) -> Tensor {
        let value_count = shape.iter().product::<usize>();
        Tensor::from_vec(
            (0..value_count)
                .map(|index| seed + layer_index as f32 * 0.001 + index as f32 * 0.000_001)
                .collect::<Vec<_>>(),
            Shape::from_dims(shape),
            &Device::Cpu,
        )
        .unwrap()
    }

    fn model_path() -> std::path::PathBuf {
        repo_root().join(format!("tests/fixtures/models/{TEST_MODEL_ID}.safetensors"))
    }

    fn repo_root() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("crate lives under rust/rwkv-srs-cpu")
            .to_path_buf()
    }
}
