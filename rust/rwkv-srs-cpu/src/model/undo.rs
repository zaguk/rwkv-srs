use std::collections::{BTreeMap, BTreeSet};

use candle_core::Result;

use super::live_session::LiveIndexUndoFrame;
use super::state::{prepared_i64, NativeRnnModuleState, ReviewIds};
use super::NativeRnn;
use crate::id_encoding::{TorchMt19937, ID_SUBMODULES};
use crate::state::{FeatureState, PreparedRow};

#[derive(Debug)]
pub(crate) struct UndoFrame {
    recurrent: RecurrentUndoFrame,
    deterministic: DeterministicUndoFrame,
}

#[derive(Debug)]
pub(crate) struct RuntimeUndoFrame {
    pub(crate) model: UndoFrame,
    pub(crate) live: Option<LiveIndexUndoFrame>,
}

impl RuntimeUndoFrame {
    pub(crate) fn model_only(model: UndoFrame) -> Self {
        Self { model, live: None }
    }
}

#[derive(Debug)]
struct RecurrentUndoFrame {
    card_id: i64,
    note_id: i64,
    deck_id: i64,
    preset_id: i64,
    card_state: Option<NativeRnnModuleState>,
    note_state: Option<NativeRnnModuleState>,
    deck_state: Option<NativeRnnModuleState>,
    preset_state: Option<NativeRnnModuleState>,
    global_state: Option<NativeRnnModuleState>,
}

#[derive(Debug)]
pub(crate) struct DeterministicUndoFrame {
    first_day_offset: Option<f64>,
    prev_day_offset: Option<f64>,
    card_set_contains: bool,
    card_count: usize,
    last_new_cards: Option<usize>,
    i: i64,
    last_i: Option<i64>,
    today: f64,
    today_reviews: i64,
    today_new_cards: i64,
    card2first_day_offset: Option<f64>,
    card2elapsed_days_cumulative: Option<f64>,
    card2elapsed_seconds_cumulative: Option<f64>,
    id_encodings: Vec<IdEncodingUndoEntry>,
    recurrent_keys: RecurrentKeyUndoFrame,
    id_rng: TorchMt19937,
}

#[derive(Debug)]
pub(crate) struct BatchUndoFrame {
    recurrent: BatchRecurrentUndoFrame,
    deterministic: BatchDeterministicUndoFrame,
}

/// Compact first-before-value journal for a complete CPU batch.
///
/// Batch processing only inserts identity encodings/recurrent keys and mutates
/// card-local history entries. Capturing each row as a full scalar undo frame
/// therefore duplicated the 624-word RNG state and repeated the same map
/// values for common identities. This frame snapshots process-wide scalars and
/// the RNG once, then records each touched identity exactly once.
#[derive(Debug)]
pub(crate) struct BatchDeterministicUndoFrame {
    first_day_offset: Option<f64>,
    prev_day_offset: Option<f64>,
    card_count: usize,
    i: i64,
    today: f64,
    today_reviews: i64,
    today_new_cards: i64,
    cards: Vec<BatchCardUndoEntry>,
    id_encodings: Vec<IdEncodingUndoEntry>,
    recurrent_keys: BatchRecurrentKeyUndoFrame,
    id_rng: TorchMt19937,
}

#[derive(Debug)]
struct BatchCardUndoEntry {
    card_id: i64,
    card_set_contains: bool,
    last_new_cards: Option<usize>,
    last_i: Option<i64>,
    card2first_day_offset: Option<f64>,
    card2elapsed_days_cumulative: Option<f64>,
    card2elapsed_seconds_cumulative: Option<f64>,
}

#[derive(Debug)]
struct BatchRecurrentKeyUndoFrame {
    card_states: Vec<(i64, bool)>,
    note_states: Vec<(i64, bool)>,
    deck_states: Vec<(i64, bool)>,
    preset_states: Vec<(i64, bool)>,
    global_state_present: bool,
}

#[derive(Debug)]
pub(crate) struct BatchRecurrentUndoFrame {
    card_states: Vec<(i64, Option<NativeRnnModuleState>)>,
    note_states: Vec<(i64, Option<NativeRnnModuleState>)>,
    deck_states: Vec<(i64, Option<NativeRnnModuleState>)>,
    preset_states: Vec<(i64, Option<NativeRnnModuleState>)>,
    global_state: Option<NativeRnnModuleState>,
}

struct BatchIdentitySets {
    card_ids: BTreeSet<i64>,
    note_ids: BTreeSet<i64>,
    deck_ids: BTreeSet<i64>,
    preset_ids: BTreeSet<i64>,
}

#[derive(Debug)]
struct IdEncodingUndoEntry {
    submodule: &'static str,
    id: i64,
    previous: Option<Vec<f32>>,
}

#[derive(Debug)]
struct RecurrentKeyUndoFrame {
    card_id: i64,
    note_id: i64,
    deck_id: i64,
    preset_id: i64,
    card_state_present: bool,
    note_state_present: bool,
    deck_state_present: bool,
    preset_state_present: bool,
    global_state_present: bool,
}

impl UndoFrame {
    pub(crate) fn capture(
        rnn: &NativeRnn,
        deterministic: &FeatureState,
        process_row: &PreparedRow,
        ids: ReviewIds,
    ) -> Result<Self> {
        let (card_id, note_id, deck_id, preset_id) = ids;

        Ok(Self {
            recurrent: RecurrentUndoFrame {
                card_id,
                note_id,
                deck_id,
                preset_id,
                card_state: rnn.card_states.get(&card_id).cloned(),
                note_state: rnn.note_states.get(&note_id).cloned(),
                deck_state: rnn.deck_states.get(&deck_id).cloned(),
                preset_state: rnn.preset_states.get(&preset_id).cloned(),
                global_state: rnn.global_state.clone(),
            },
            deterministic: DeterministicUndoFrame::capture(deterministic, process_row, ids)?,
        })
    }

    pub(crate) fn restore(self, rnn: &mut NativeRnn, deterministic: &mut FeatureState) {
        let Self {
            recurrent,
            deterministic: deterministic_frame,
        } = self;
        restore_map_entry(
            &mut rnn.card_states,
            recurrent.card_id,
            recurrent.card_state,
        );
        restore_map_entry(
            &mut rnn.note_states,
            recurrent.note_id,
            recurrent.note_state,
        );
        restore_map_entry(
            &mut rnn.deck_states,
            recurrent.deck_id,
            recurrent.deck_state,
        );
        restore_map_entry(
            &mut rnn.preset_states,
            recurrent.preset_id,
            recurrent.preset_state,
        );
        rnn.global_state = recurrent.global_state;
        rnn.restore_gpu_review_states_after_undo(
            recurrent.card_id,
            recurrent.note_id,
            recurrent.deck_id,
            recurrent.preset_id,
        );

        deterministic_frame.restore(deterministic);
    }
}

impl DeterministicUndoFrame {
    pub(crate) fn capture(
        deterministic: &FeatureState,
        process_row: &PreparedRow,
        ids: ReviewIds,
    ) -> Result<Self> {
        let (card_id, note_id, deck_id, preset_id) = ids;
        let id_encodings = ID_SUBMODULES
            .iter()
            .map(|&(submodule, _)| {
                let id = prepared_i64(process_row, submodule)?;
                Ok(IdEncodingUndoEntry {
                    submodule,
                    id,
                    previous: deterministic
                        .id_encodings
                        .get(submodule)
                        .and_then(|encodings| encodings.get(&id))
                        .cloned(),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            first_day_offset: deterministic.first_day_offset,
            prev_day_offset: deterministic.prev_day_offset,
            card_set_contains: deterministic.card_set.contains(&card_id),
            card_count: deterministic.card_count,
            last_new_cards: deterministic.last_new_cards.get(&card_id).copied(),
            i: deterministic.i,
            last_i: deterministic.last_i.get(&card_id).copied(),
            today: deterministic.today,
            today_reviews: deterministic.today_reviews,
            today_new_cards: deterministic.today_new_cards,
            card2first_day_offset: deterministic.card2first_day_offset.get(&card_id).copied(),
            card2elapsed_days_cumulative: deterministic
                .card2elapsed_days_cumulative
                .get(&card_id)
                .copied(),
            card2elapsed_seconds_cumulative: deterministic
                .card2elapsed_seconds_cumulative
                .get(&card_id)
                .copied(),
            id_encodings,
            recurrent_keys: RecurrentKeyUndoFrame {
                card_id,
                note_id,
                deck_id,
                preset_id,
                card_state_present: deterministic
                    .recurrent_state_keys
                    .card_states
                    .contains(&card_id),
                note_state_present: deterministic
                    .recurrent_state_keys
                    .note_states
                    .contains(&note_id),
                deck_state_present: deterministic
                    .recurrent_state_keys
                    .deck_states
                    .contains(&deck_id),
                preset_state_present: deterministic
                    .recurrent_state_keys
                    .preset_states
                    .contains(&preset_id),
                global_state_present: deterministic.recurrent_state_keys.global_state,
            },
            id_rng: deterministic.id_rng.clone(),
        })
    }

    pub(crate) fn restore(self, deterministic: &mut FeatureState) {
        let Self {
            first_day_offset,
            prev_day_offset,
            card_set_contains,
            card_count,
            last_new_cards,
            i,
            last_i,
            today,
            today_reviews,
            today_new_cards,
            card2first_day_offset,
            card2elapsed_days_cumulative,
            card2elapsed_seconds_cumulative,
            id_encodings,
            recurrent_keys,
            id_rng,
        } = self;
        let card_id = recurrent_keys.card_id;

        deterministic.first_day_offset = first_day_offset;
        deterministic.prev_day_offset = prev_day_offset;
        restore_set_entry(&mut deterministic.card_set, card_id, card_set_contains);
        deterministic.card_count = card_count;
        restore_map_entry(&mut deterministic.last_new_cards, card_id, last_new_cards);
        deterministic.i = i;
        restore_map_entry(&mut deterministic.last_i, card_id, last_i);
        deterministic.today = today;
        deterministic.today_reviews = today_reviews;
        deterministic.today_new_cards = today_new_cards;
        restore_map_entry(
            &mut deterministic.card2first_day_offset,
            card_id,
            card2first_day_offset,
        );
        restore_map_entry(
            &mut deterministic.card2elapsed_days_cumulative,
            card_id,
            card2elapsed_days_cumulative,
        );
        restore_map_entry(
            &mut deterministic.card2elapsed_seconds_cumulative,
            card_id,
            card2elapsed_seconds_cumulative,
        );

        for entry in id_encodings {
            let encodings = deterministic
                .id_encodings
                .get_mut(entry.submodule)
                .expect("id encoding map initialized for every submodule");
            restore_map_entry(encodings, entry.id, entry.previous);
        }
        deterministic.id_rng = id_rng;

        restore_set_entry(
            &mut deterministic.recurrent_state_keys.card_states,
            recurrent_keys.card_id,
            recurrent_keys.card_state_present,
        );
        restore_set_entry(
            &mut deterministic.recurrent_state_keys.note_states,
            recurrent_keys.note_id,
            recurrent_keys.note_state_present,
        );
        restore_set_entry(
            &mut deterministic.recurrent_state_keys.deck_states,
            recurrent_keys.deck_id,
            recurrent_keys.deck_state_present,
        );
        restore_set_entry(
            &mut deterministic.recurrent_state_keys.preset_states,
            recurrent_keys.preset_id,
            recurrent_keys.preset_state_present,
        );
        deterministic.recurrent_state_keys.global_state = recurrent_keys.global_state_present;
    }
}

impl BatchUndoFrame {
    pub(crate) fn capture(
        rnn: &NativeRnn,
        deterministic: &FeatureState,
        ids: &[ReviewIds],
    ) -> Self {
        let identities = BatchIdentitySets::from_ids(ids);
        Self {
            recurrent: BatchRecurrentUndoFrame::capture_identities(rnn, &identities),
            deterministic: BatchDeterministicUndoFrame::capture_identities(
                deterministic,
                &identities,
            ),
        }
    }

    pub(crate) fn restore(self, rnn: &mut NativeRnn, deterministic: &mut FeatureState) {
        self.deterministic.restore(deterministic);
        self.recurrent.restore(rnn);
    }
}

impl BatchDeterministicUndoFrame {
    pub(crate) fn capture(deterministic: &FeatureState, ids: &[ReviewIds]) -> Self {
        let identities = BatchIdentitySets::from_ids(ids);
        Self::capture_identities(deterministic, &identities)
    }

    fn capture_identities(deterministic: &FeatureState, identities: &BatchIdentitySets) -> Self {
        // Intentionally exhaustive: adding deterministic state must force this
        // rollback journal to make an explicit capture/restore decision.
        let FeatureState {
            first_day_offset,
            prev_day_offset,
            card_set,
            card_count,
            last_new_cards,
            i,
            last_i,
            today,
            today_reviews,
            today_new_cards,
            card2first_day_offset,
            card2elapsed_days_cumulative,
            card2elapsed_seconds_cumulative,
            id_encodings: deterministic_id_encodings,
            recurrent_state_keys,
            id_rng,
        } = deterministic;
        let BatchIdentitySets {
            card_ids,
            note_ids,
            deck_ids,
            preset_ids,
        } = identities;

        let cards = card_ids
            .iter()
            .copied()
            .map(|card_id| BatchCardUndoEntry {
                card_id,
                card_set_contains: card_set.contains(&card_id),
                last_new_cards: last_new_cards.get(&card_id).copied(),
                last_i: last_i.get(&card_id).copied(),
                card2first_day_offset: card2first_day_offset.get(&card_id).copied(),
                card2elapsed_days_cumulative: card2elapsed_days_cumulative.get(&card_id).copied(),
                card2elapsed_seconds_cumulative: card2elapsed_seconds_cumulative
                    .get(&card_id)
                    .copied(),
            })
            .collect();

        let mut id_encodings =
            Vec::with_capacity(card_ids.len() + note_ids.len() + deck_ids.len() + preset_ids.len());
        for (submodule, identity_ids) in [
            ("card_id", &card_ids),
            ("note_id", &note_ids),
            ("deck_id", &deck_ids),
            ("preset_id", &preset_ids),
        ] {
            let encodings = deterministic_id_encodings
                .get(submodule)
                .expect("id encoding map initialized for every submodule");
            id_encodings.extend(identity_ids.iter().copied().map(|id| IdEncodingUndoEntry {
                submodule,
                id,
                previous: encodings.get(&id).cloned(),
            }));
        }

        Self {
            first_day_offset: *first_day_offset,
            prev_day_offset: *prev_day_offset,
            card_count: *card_count,
            i: *i,
            today: *today,
            today_reviews: *today_reviews,
            today_new_cards: *today_new_cards,
            cards,
            id_encodings,
            recurrent_keys: BatchRecurrentKeyUndoFrame {
                card_states: capture_set_entries(&recurrent_state_keys.card_states, card_ids),
                note_states: capture_set_entries(&recurrent_state_keys.note_states, note_ids),
                deck_states: capture_set_entries(&recurrent_state_keys.deck_states, deck_ids),
                preset_states: capture_set_entries(&recurrent_state_keys.preset_states, preset_ids),
                global_state_present: recurrent_state_keys.global_state,
            },
            id_rng: id_rng.clone(),
        }
    }

    pub(crate) fn restore(self, deterministic: &mut FeatureState) {
        deterministic.first_day_offset = self.first_day_offset;
        deterministic.prev_day_offset = self.prev_day_offset;
        deterministic.card_count = self.card_count;
        deterministic.i = self.i;
        deterministic.today = self.today;
        deterministic.today_reviews = self.today_reviews;
        deterministic.today_new_cards = self.today_new_cards;

        for entry in self.cards {
            restore_set_entry(
                &mut deterministic.card_set,
                entry.card_id,
                entry.card_set_contains,
            );
            restore_map_entry(
                &mut deterministic.last_new_cards,
                entry.card_id,
                entry.last_new_cards,
            );
            restore_map_entry(&mut deterministic.last_i, entry.card_id, entry.last_i);
            restore_map_entry(
                &mut deterministic.card2first_day_offset,
                entry.card_id,
                entry.card2first_day_offset,
            );
            restore_map_entry(
                &mut deterministic.card2elapsed_days_cumulative,
                entry.card_id,
                entry.card2elapsed_days_cumulative,
            );
            restore_map_entry(
                &mut deterministic.card2elapsed_seconds_cumulative,
                entry.card_id,
                entry.card2elapsed_seconds_cumulative,
            );
        }

        for entry in self.id_encodings {
            let encodings = deterministic
                .id_encodings
                .get_mut(entry.submodule)
                .expect("id encoding map initialized for every submodule");
            restore_map_entry(encodings, entry.id, entry.previous);
        }
        deterministic.id_rng = self.id_rng;

        restore_set_entries(
            &mut deterministic.recurrent_state_keys.card_states,
            self.recurrent_keys.card_states,
        );
        restore_set_entries(
            &mut deterministic.recurrent_state_keys.note_states,
            self.recurrent_keys.note_states,
        );
        restore_set_entries(
            &mut deterministic.recurrent_state_keys.deck_states,
            self.recurrent_keys.deck_states,
        );
        restore_set_entries(
            &mut deterministic.recurrent_state_keys.preset_states,
            self.recurrent_keys.preset_states,
        );
        deterministic.recurrent_state_keys.global_state = self.recurrent_keys.global_state_present;
    }

    #[cfg(test)]
    fn tracked_capacity_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.cards.capacity() * std::mem::size_of::<BatchCardUndoEntry>()
            + self.id_encodings.capacity() * std::mem::size_of::<IdEncodingUndoEntry>()
            + self
                .id_encodings
                .iter()
                .filter_map(|entry| entry.previous.as_ref())
                .map(|encoding| encoding.capacity() * std::mem::size_of::<f32>())
                .sum::<usize>()
            + self.recurrent_keys.card_states.capacity() * std::mem::size_of::<(i64, bool)>()
            + self.recurrent_keys.note_states.capacity() * std::mem::size_of::<(i64, bool)>()
            + self.recurrent_keys.deck_states.capacity() * std::mem::size_of::<(i64, bool)>()
            + self.recurrent_keys.preset_states.capacity() * std::mem::size_of::<(i64, bool)>()
    }
}

impl BatchRecurrentUndoFrame {
    pub(crate) fn capture(rnn: &NativeRnn, ids: &[ReviewIds]) -> Self {
        let identities = BatchIdentitySets::from_ids(ids);
        Self::capture_identities(rnn, &identities)
    }

    fn capture_identities(rnn: &NativeRnn, identities: &BatchIdentitySets) -> Self {
        Self {
            card_states: capture_map_entries(&rnn.card_states, &identities.card_ids),
            note_states: capture_map_entries(&rnn.note_states, &identities.note_ids),
            deck_states: capture_map_entries(&rnn.deck_states, &identities.deck_ids),
            preset_states: capture_map_entries(&rnn.preset_states, &identities.preset_ids),
            global_state: rnn.global_state.clone(),
        }
    }

    pub(crate) fn restore(self, rnn: &mut NativeRnn) {
        restore_map_entries(&mut rnn.card_states, self.card_states);
        restore_map_entries(&mut rnn.note_states, self.note_states);
        restore_map_entries(&mut rnn.deck_states, self.deck_states);
        restore_map_entries(&mut rnn.preset_states, self.preset_states);
        rnn.global_state = self.global_state;
        // A failed batch is rare; dropping the optional GPU cache is both
        // cheaper and safer than replaying an arbitrary set of restored keys.
        rnn.invalidate_gpu();
    }
}

impl BatchIdentitySets {
    fn from_ids(ids: &[ReviewIds]) -> Self {
        let mut identities = Self {
            card_ids: BTreeSet::new(),
            note_ids: BTreeSet::new(),
            deck_ids: BTreeSet::new(),
            preset_ids: BTreeSet::new(),
        };
        for &(card_id, note_id, deck_id, preset_id) in ids {
            identities.card_ids.insert(card_id);
            identities.note_ids.insert(note_id);
            identities.deck_ids.insert(deck_id);
            identities.preset_ids.insert(preset_id);
        }
        identities
    }
}

fn capture_map_entries<V: Clone>(
    map: &BTreeMap<i64, V>,
    keys: &BTreeSet<i64>,
) -> Vec<(i64, Option<V>)> {
    keys.iter()
        .copied()
        .map(|key| (key, map.get(&key).cloned()))
        .collect()
}

fn restore_map_entries<V>(map: &mut BTreeMap<i64, V>, entries: Vec<(i64, Option<V>)>) {
    for (key, previous) in entries {
        restore_map_entry(map, key, previous);
    }
}

fn restore_map_entry<V>(map: &mut BTreeMap<i64, V>, key: i64, previous: Option<V>) {
    if let Some(value) = previous {
        map.insert(key, value);
    } else {
        map.remove(&key);
    }
}

fn restore_set_entry(set: &mut BTreeSet<i64>, key: i64, was_present: bool) {
    if was_present {
        set.insert(key);
    } else {
        set.remove(&key);
    }
}

fn capture_set_entries(set: &BTreeSet<i64>, keys: &BTreeSet<i64>) -> Vec<(i64, bool)> {
    keys.iter()
        .copied()
        .map(|key| (key, set.contains(&key)))
        .collect()
}

fn restore_set_entries(set: &mut BTreeSet<i64>, entries: Vec<(i64, bool)>) {
    for (key, was_present) in entries {
        restore_set_entry(set, key, was_present);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{MaybeId, ReviewInput};

    fn review(
        review_id: i64,
        card_id: i64,
        note_id: i64,
        deck_id: i64,
        preset_id: i64,
        day_offset: f64,
    ) -> ReviewInput {
        ReviewInput {
            review_id,
            card_id,
            note_id: MaybeId::Present(note_id),
            deck_id: MaybeId::Present(deck_id),
            preset_id: MaybeId::Present(preset_id),
            day_offset,
            elapsed_days: 1.0,
            elapsed_seconds: 86_400.0,
            rating: Some(3),
            duration: Some(2.0),
            state: Some(2.0),
        }
    }

    fn assert_feature_state_eq(actual: &FeatureState, expected: &FeatureState) {
        assert_eq!(actual.first_day_offset, expected.first_day_offset);
        assert_eq!(actual.prev_day_offset, expected.prev_day_offset);
        assert_eq!(actual.card_set, expected.card_set);
        assert_eq!(actual.card_count, expected.card_count);
        assert_eq!(actual.last_new_cards, expected.last_new_cards);
        assert_eq!(actual.i, expected.i);
        assert_eq!(actual.last_i, expected.last_i);
        assert_eq!(actual.today, expected.today);
        assert_eq!(actual.today_reviews, expected.today_reviews);
        assert_eq!(actual.today_new_cards, expected.today_new_cards);
        assert_eq!(actual.card2first_day_offset, expected.card2first_day_offset);
        assert_eq!(
            actual.card2elapsed_days_cumulative,
            expected.card2elapsed_days_cumulative
        );
        assert_eq!(
            actual.card2elapsed_seconds_cumulative,
            expected.card2elapsed_seconds_cumulative
        );
        assert_eq!(actual.id_encodings, expected.id_encodings);
        assert_eq!(
            actual.recurrent_state_keys.card_states,
            expected.recurrent_state_keys.card_states
        );
        assert_eq!(
            actual.recurrent_state_keys.note_states,
            expected.recurrent_state_keys.note_states
        );
        assert_eq!(
            actual.recurrent_state_keys.deck_states,
            expected.recurrent_state_keys.deck_states
        );
        assert_eq!(
            actual.recurrent_state_keys.preset_states,
            expected.recurrent_state_keys.preset_states
        );
        assert_eq!(
            actual.recurrent_state_keys.global_state,
            expected.recurrent_state_keys.global_state
        );
        assert_eq!(
            actual.id_rng.to_torch_rng_state_bytes(),
            expected.id_rng.to_torch_rng_state_bytes()
        );
    }

    #[test]
    fn compact_batch_deterministic_undo_restores_repeated_and_unseen_identities() {
        let mut state = FeatureState::with_torch_seed(1234);
        let existing = review(1, 1, 11, 21, 31, 100.0);
        state
            .append_process_feature_pair(&existing, &mut Vec::new())
            .unwrap();
        let before = state.clone();

        let inputs = [
            review(2, 1, 11, 21, 31, 101.0),
            review(3, 1, 12, 21, 31, 102.0),
            review(4, 2, 12, 22, 32, 103.0),
        ];
        let ids = inputs
            .iter()
            .map(FeatureState::normalized_review_ids)
            .collect::<Vec<_>>();
        let undo = BatchDeterministicUndoFrame::capture(&state, &ids);

        // Two repeated card slots and two identities in every other module are
        // journaled once, rather than once per input row.
        assert_eq!(undo.cards.len(), 2);
        assert_eq!(undo.id_encodings.len(), 8);
        assert_eq!(undo.recurrent_keys.card_states.len(), 2);
        assert_eq!(undo.recurrent_keys.note_states.len(), 2);

        for input in &inputs {
            state
                .append_process_feature_pair(input, &mut Vec::new())
                .unwrap();
        }
        assert_ne!(state.i, before.i);
        assert!(state.id_encodings["card_id"].contains_key(&2));
        assert!(state.id_encodings["note_id"].contains_key(&12));

        undo.restore(&mut state);
        assert_feature_state_eq(&state, &before);
    }

    #[test]
    fn compact_batch_deterministic_undo_restores_a_committed_prefix() {
        let mut state = FeatureState::with_torch_seed(4321);
        let before = state.clone();
        let inputs = [
            review(1, 7, 17, 27, 37, 10.0),
            review(2, 7, 18, 27, 37, 11.0),
            review(3, 8, 18, 28, 38, 12.0),
        ];
        let ids = inputs
            .iter()
            .map(FeatureState::normalized_review_ids)
            .collect::<Vec<_>>();
        let undo = BatchDeterministicUndoFrame::capture(&state, &ids);

        // Model a later-row failure after an arbitrary prefix has mutated all
        // deterministic state categories.
        for input in &inputs[..2] {
            state
                .append_process_feature_pair(input, &mut Vec::new())
                .unwrap();
        }
        undo.restore(&mut state);

        assert_feature_state_eq(&state, &before);
    }

    #[test]
    fn compact_batch_journal_memory_scales_with_unique_identities_not_rows() {
        const ROWS: usize = 10_000;
        let state = FeatureState::with_torch_seed(1234);
        let old_rng_snapshot_bytes = ROWS * std::mem::size_of::<TorchMt19937>();

        let repeated_ids = vec![(1, 11, 21, 31); ROWS];
        let repeated = BatchDeterministicUndoFrame::capture(&state, &repeated_ids);
        assert!(repeated.tracked_capacity_bytes() < old_rng_snapshot_bytes / 1_000);

        let unique_ids = (0..ROWS as i64)
            .map(|id| (id, id + 20_000, id + 40_000, id + 60_000))
            .collect::<Vec<_>>();
        let unique = BatchDeterministicUndoFrame::capture(&state, &unique_ids);
        // Even an all-unseen batch is substantially below the old journal's
        // RNG snapshots alone, before counting its per-row entry vectors.
        assert!(unique.tracked_capacity_bytes() < old_rng_snapshot_bytes / 4);
    }
}
