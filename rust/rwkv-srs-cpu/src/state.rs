use std::collections::{BTreeMap, BTreeSet};

use crate::features::{
    day_offset_encoding, elapsed_seconds_cos, elapsed_seconds_sin, scale_cum_new_cards_today,
    scale_cum_reviews_today, scale_day_offset_diff, scale_diff_new_cards, scale_diff_reviews,
    scale_duration, scale_elapsed_days, scale_elapsed_days_cumulative, scale_elapsed_seconds,
    scale_elapsed_seconds_cumulative, scale_state, CARD_FEATURE_COLUMNS, ID_PLACEHOLDER,
};
use crate::id_encoding::{
    empty_id_encodings, IdEncodings, TorchMt19937, ID_ENCODING_LEN, ID_SUBMODULES,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MaybeId {
    Present(i64),
    Missing,
}

#[derive(Debug, Clone)]
pub struct ReviewInput {
    pub review_id: i64,
    pub card_id: i64,
    pub note_id: MaybeId,
    pub deck_id: MaybeId,
    pub preset_id: MaybeId,
    pub day_offset: f64,
    pub elapsed_days: f64,
    pub elapsed_seconds: f64,
    pub rating: Option<i64>,
    pub duration: Option<f64>,
    pub state: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RowValue {
    Bool(bool),
    Float(f64),
    Int(i64),
}

impl RowValue {
    pub fn as_f64(&self, field: &str) -> Result<f64, String> {
        match self {
            RowValue::Float(value) => Ok(*value),
            RowValue::Int(value) => Ok(*value as f64),
            RowValue::Bool(_) => Err(format!("field '{field}' must be numeric")),
        }
    }

    pub fn as_i64(&self, field: &str) -> Result<i64, String> {
        match self {
            RowValue::Int(value) => Ok(*value),
            RowValue::Float(value) if value.is_finite() && value.fract() == 0.0 => {
                Ok(*value as i64)
            }
            _ => Err(format!("field '{field}' must be an integer")),
        }
    }
}

pub type PreparedRow = BTreeMap<String, RowValue>;

#[derive(Debug, Clone, Default)]
pub struct RecurrentStateKeys {
    pub card_states: BTreeSet<i64>,
    pub note_states: BTreeSet<i64>,
    pub deck_states: BTreeSet<i64>,
    pub preset_states: BTreeSet<i64>,
    pub global_state: bool,
}

#[derive(Debug, Clone)]
pub struct FeatureState {
    pub first_day_offset: Option<f64>,
    pub prev_day_offset: Option<f64>,
    pub card_set: BTreeSet<i64>,
    pub card_count: usize,
    pub last_new_cards: BTreeMap<i64, usize>,
    pub i: i64,
    pub last_i: BTreeMap<i64, i64>,
    pub today: f64,
    pub today_reviews: i64,
    pub today_new_cards: i64,
    pub card2first_day_offset: BTreeMap<i64, f64>,
    pub card2elapsed_days_cumulative: BTreeMap<i64, f64>,
    pub card2elapsed_seconds_cumulative: BTreeMap<i64, f64>,
    pub id_encodings: IdEncodings,
    pub recurrent_state_keys: RecurrentStateKeys,
    pub(crate) id_rng: TorchMt19937,
}

impl Default for FeatureState {
    fn default() -> Self {
        Self::with_torch_seed(5489)
    }
}

impl FeatureState {
    pub fn normalized_review_ids(input: &ReviewInput) -> (i64, i64, i64, i64) {
        let card_id = input.card_id;
        let (note_id, _) = normalize_id(input.note_id, Some(card_id));
        let (deck_id, _) = normalize_id(input.deck_id, None);
        let (preset_id, _) = normalize_id(input.preset_id, None);
        (card_id, note_id, deck_id, preset_id)
    }

    pub fn with_torch_seed(torch_seed: u64) -> Self {
        Self {
            first_day_offset: None,
            prev_day_offset: None,
            card_set: BTreeSet::new(),
            card_count: 0,
            last_new_cards: BTreeMap::new(),
            i: 0,
            last_i: BTreeMap::new(),
            today: -1.0,
            today_reviews: 0,
            today_new_cards: 0,
            card2first_day_offset: BTreeMap::new(),
            card2elapsed_days_cumulative: BTreeMap::new(),
            card2elapsed_seconds_cumulative: BTreeMap::new(),
            id_encodings: empty_id_encodings(),
            recurrent_state_keys: RecurrentStateKeys::default(),
            id_rng: TorchMt19937::seed_from_u64(torch_seed),
        }
    }

    pub fn prepare_predict_row(&self, input: &ReviewInput) -> PreparedRow {
        let mut row = self.add_same(input);
        row.insert("is_query".to_string(), RowValue::Float(1.0));
        row.insert("skip".to_string(), RowValue::Bool(true));
        row.insert("scaled_duration".to_string(), RowValue::Float(0.0));
        row.insert("scaled_state".to_string(), RowValue::Float(0.0));
        for rating in 1..=4 {
            row.insert(format!("rating_{rating}"), RowValue::Float(0.0));
        }
        row
    }

    pub fn prepare_process_row(&self, input: &ReviewInput) -> Result<PreparedRow, String> {
        let rating = input
            .rating
            .ok_or_else(|| "process review is missing field 'rating'".to_string())?;
        let duration = input
            .duration
            .ok_or_else(|| "process review is missing field 'duration'".to_string())?;
        let state = input
            .state
            .ok_or_else(|| "process review is missing field 'state'".to_string())?;

        let mut row = self.add_same(input);
        row.insert("is_query".to_string(), RowValue::Float(0.0));
        row.insert("skip".to_string(), RowValue::Bool(false));
        row.insert(
            "scaled_duration".to_string(),
            RowValue::Float(scale_duration(duration)),
        );
        row.insert(
            "scaled_state".to_string(),
            RowValue::Float(scale_state(state)),
        );
        for index in 1..=4 {
            row.insert(
                format!("rating_{index}"),
                RowValue::Float(if rating == index { 1.0 } else { 0.0 }),
            );
        }
        Ok(row)
    }

    pub fn feature_vector(
        &mut self,
        row: &PreparedRow,
        mutate_id_encodings: bool,
    ) -> Result<Vec<f32>, String> {
        let mut features = Vec::with_capacity(CARD_FEATURE_COLUMNS.len() + ID_ENCODING_LEN + 28);
        self.append_feature_vector(row, mutate_id_encodings, &mut features)?;
        Ok(features)
    }

    pub fn append_feature_vector(
        &mut self,
        row: &PreparedRow,
        mutate_id_encodings: bool,
        features: &mut Vec<f32>,
    ) -> Result<(), String> {
        features.reserve(CARD_FEATURE_COLUMNS.len() + ID_ENCODING_LEN + 28);
        append_card_features(row, features)?;

        for (submodule, dim) in ID_SUBMODULES {
            let id = get_i64(row, submodule)?;
            if let Some(encoding) = self
                .id_encodings
                .get(submodule)
                .and_then(|encodings| encodings.get(&id))
            {
                features.extend_from_slice(encoding);
                continue;
            }

            let encoding = self.id_rng.id_encoding(dim);
            if mutate_id_encodings {
                self.id_encodings
                    .get_mut(submodule)
                    .expect("id encoding map initialized for every submodule")
                    .insert(id, encoding.clone());
            }
            features.extend_from_slice(&encoding);
        }

        append_day_offset_features(row, features)?;
        Ok(())
    }

    pub fn process_feature_vector(&mut self, row: &PreparedRow) -> Result<Vec<f32>, String> {
        self.feature_vector(row, true)
    }

    pub fn predict_feature_vector(&mut self, row: &PreparedRow) -> Result<Vec<f32>, String> {
        let rng_state = if self.skip_needs_rng_restore(row, true)? {
            Some(self.id_rng.clone())
        } else {
            None
        };

        let result = self.feature_vector(row, false);
        if let Some(rng_state) = rng_state {
            self.id_rng = rng_state;
        }
        result
    }

    pub fn append_process_feature_pair(
        &mut self,
        input: &ReviewInput,
        features: &mut Vec<f32>,
    ) -> Result<(i64, i64, i64, i64), String> {
        self.append_process_feature_rows(input, features, true)
    }

    pub fn append_process_feature_only(
        &mut self,
        input: &ReviewInput,
        features: &mut Vec<f32>,
    ) -> Result<(i64, i64, i64, i64), String> {
        self.append_process_feature_rows(input, features, false)
    }

    fn append_process_feature_rows(
        &mut self,
        input: &ReviewInput,
        features: &mut Vec<f32>,
        include_query: bool,
    ) -> Result<(i64, i64, i64, i64), String> {
        let rating = input
            .rating
            .ok_or_else(|| "process review is missing field 'rating'".to_string())?;
        let duration = input
            .duration
            .ok_or_else(|| "process review is missing field 'duration'".to_string())?;
        let state = input
            .state
            .ok_or_else(|| "process review is missing field 'state'".to_string())?;
        let card_id = input.card_id;
        let (note_id, note_missing) = normalize_id(input.note_id, Some(card_id));
        let (deck_id, deck_missing) = normalize_id(input.deck_id, None);
        let (preset_id, preset_missing) = normalize_id(input.preset_id, None);
        let ids = (card_id, note_id, deck_id, preset_id);
        let elapsed_days_cumulative = self
            .card2elapsed_days_cumulative
            .get(&card_id)
            .copied()
            .unwrap_or(0.0)
            + input.elapsed_days;
        let elapsed_seconds_cumulative = self
            .card2elapsed_seconds_cumulative
            .get(&card_id)
            .copied()
            .unwrap_or(0.0)
            + input.elapsed_seconds;
        let day_offset = self
            .first_day_offset
            .map_or(0.0, |first| input.day_offset - first);
        let day_offset_first = self
            .card2first_day_offset
            .get(&card_id)
            .copied()
            .unwrap_or(day_offset);
        let previous_day_offset = self.prev_day_offset.unwrap_or(0.0);
        let diff_new_cards = self
            .last_new_cards
            .get(&card_id)
            .map(|last| self.card_count as i64 - *last as i64)
            .unwrap_or(0)
            .max(0) as f64;
        let diff_reviews = self
            .last_i
            .get(&card_id)
            .map(|last| (self.i - *last - 1).max(0))
            .unwrap_or(0) as f64;
        let (mut today_reviews, mut today_new_cards) = (self.today_reviews, self.today_new_cards);
        if day_offset != self.today {
            today_new_cards = 0;
            today_reviews = -1;
        }
        today_reviews += 1;
        if !self.card_set.contains(&card_id) {
            today_new_cards += 1;
        }

        let mut card_features = [
            scale_elapsed_days(input.elapsed_days) as f32,
            scale_elapsed_days_cumulative(elapsed_days_cumulative) as f32,
            scale_elapsed_seconds(input.elapsed_seconds) as f32,
            elapsed_seconds_sin(input.elapsed_seconds) as f32,
            elapsed_seconds_cos(input.elapsed_seconds) as f32,
            scale_elapsed_seconds_cumulative(elapsed_seconds_cumulative) as f32,
            elapsed_seconds_sin(elapsed_seconds_cumulative) as f32,
            elapsed_seconds_cos(elapsed_seconds_cumulative) as f32,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            note_missing as f32,
            deck_missing as f32,
            preset_missing as f32,
            scale_day_offset_diff(day_offset - previous_day_offset) as f32,
            ((py_mod(day_offset, 7.0) - 3.0) / 3.0) as f32,
            scale_diff_new_cards(diff_new_cards) as f32,
            scale_diff_reviews(diff_reviews) as f32,
            scale_cum_new_cards_today(today_new_cards as f64) as f32,
            scale_cum_reviews_today(today_reviews as f64) as f32,
            0.0,
            1.0,
        ];
        let start = features.len();
        if include_query {
            features.extend(card_features);
        }
        for ((submodule, dim), id) in ID_SUBMODULES
            .iter()
            .copied()
            .zip([card_id, note_id, deck_id, preset_id])
        {
            let missing = !self
                .id_encodings
                .get(submodule)
                .expect("id encoding map initialized for every submodule")
                .contains_key(&id);
            if missing {
                let encoding = self.id_rng.id_encoding(dim);
                self.id_encodings
                    .get_mut(submodule)
                    .expect("id encoding map initialized for every submodule")
                    .insert(id, encoding);
            }
            if include_query {
                features.extend_from_slice(
                    self.id_encodings
                        .get(submodule)
                        .and_then(|encodings| encodings.get(&id))
                        .expect("identity encoding inserted above"),
                );
            }
        }
        let day_features = day_offset_encoding(day_offset, day_offset_first);
        if include_query {
            features.extend_from_slice(&day_features);
        }

        card_features[8] = scale_duration(duration) as f32;
        for index in 1..=4 {
            card_features[8 + index] = if rating == index as i64 { 1.0 } else { 0.0 };
        }
        card_features[22] = scale_state(state) as f32;
        card_features[23] = 0.0;
        features.extend(card_features);
        for (submodule, id) in ID_SUBMODULES
            .iter()
            .map(|(submodule, _)| *submodule)
            .zip([card_id, note_id, deck_id, preset_id])
        {
            features.extend_from_slice(
                self.id_encodings
                    .get(submodule)
                    .and_then(|encodings| encodings.get(&id))
                    .expect("identity encoding inserted above"),
            );
        }
        features.extend_from_slice(&day_features);
        let expected_values = if include_query { 184 } else { 92 };
        if features.len() != start + expected_values {
            return Err(format!(
                "direct process feature rows have {} values, expected {expected_values}",
                features.len() - start,
            ));
        }

        self.record_recurrent_ids(ids);
        self.record_processed_values(
            card_id,
            input.elapsed_days,
            input.elapsed_seconds,
            day_offset,
        );
        Ok(ids)
    }

    pub fn skip_needs_rng_restore(&self, row: &PreparedRow, skip: bool) -> Result<bool, String> {
        if !skip {
            return Ok(false);
        }

        for (submodule, _) in ID_SUBMODULES {
            let id = get_i64(row, submodule)?;
            if !self
                .id_encodings
                .get(submodule)
                .expect("id encoding map initialized for every submodule")
                .contains_key(&id)
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub fn feature_vectors(&self, rows: &[PreparedRow]) -> Result<Vec<Vec<f32>>, String> {
        rows.iter()
            .map(|row| {
                let mut features =
                    Vec::with_capacity(CARD_FEATURE_COLUMNS.len() + ID_ENCODING_LEN + 28);
                append_card_features(row, &mut features)?;

                for (submodule, _) in ID_SUBMODULES {
                    let id = get_i64(row, submodule)?;
                    let encoding = self
                        .id_encodings
                        .get(submodule)
                        .and_then(|encodings| encodings.get(&id))
                        .ok_or_else(|| {
                            format!("missing id encoding for submodule '{submodule}' id {id}")
                        })?;
                    features.extend_from_slice(encoding);
                }

                append_day_offset_features(row, &mut features)?;
                Ok(features)
            })
            .collect()
    }

    /// Build the immutable 92-value prediction input without materializing a
    /// string-keyed `PreparedRow`.
    ///
    /// Collection scans call this once per card, so avoiding the intermediate
    /// `BTreeMap<String, RowValue>` and its repeated lookups is material to the
    /// GPU path. `None` retains the existing scalar-oracle fallback for unseen
    /// identities whose encodings or recurrent states do not exist yet.
    pub fn direct_predict_features(
        &self,
        input: &ReviewInput,
    ) -> Result<Option<(Vec<f32>, (i64, i64, i64, i64))>, String> {
        let card_id = input.card_id;
        let (note_id, note_missing) = normalize_id(input.note_id, Some(card_id));
        let (deck_id, deck_missing) = normalize_id(input.deck_id, None);
        let (preset_id, preset_missing) = normalize_id(input.preset_id, None);
        let ids = (card_id, note_id, deck_id, preset_id);

        for ((submodule, _), id) in ID_SUBMODULES
            .iter()
            .copied()
            .zip([card_id, note_id, deck_id, preset_id])
        {
            if !self
                .id_encodings
                .get(submodule)
                .expect("id encoding map initialized for every submodule")
                .contains_key(&id)
            {
                return Ok(None);
            }
        }
        if !self.recurrent_state_keys.card_states.contains(&card_id)
            || !self.recurrent_state_keys.note_states.contains(&note_id)
            || !self.recurrent_state_keys.deck_states.contains(&deck_id)
            || !self.recurrent_state_keys.preset_states.contains(&preset_id)
        {
            return Ok(None);
        }

        let elapsed_days_cumulative = self
            .card2elapsed_days_cumulative
            .get(&card_id)
            .copied()
            .unwrap_or(0.0)
            + input.elapsed_days;
        let elapsed_seconds_cumulative = self
            .card2elapsed_seconds_cumulative
            .get(&card_id)
            .copied()
            .unwrap_or(0.0)
            + input.elapsed_seconds;
        let day_offset = self
            .first_day_offset
            .map_or(0.0, |first| input.day_offset - first);
        let day_offset_first = self
            .card2first_day_offset
            .get(&card_id)
            .copied()
            .unwrap_or(day_offset);
        let previous_day_offset = self.prev_day_offset.unwrap_or(0.0);
        let diff_new_cards = self
            .last_new_cards
            .get(&card_id)
            .map(|last| self.card_count as i64 - *last as i64)
            .unwrap_or(0)
            .max(0) as f64;
        let diff_reviews = self
            .last_i
            .get(&card_id)
            .map(|last| (self.i - *last - 1).max(0))
            .unwrap_or(0) as f64;
        let (mut today_reviews, mut today_new_cards) = (self.today_reviews, self.today_new_cards);
        if day_offset != self.today {
            today_new_cards = 0;
            today_reviews = -1;
        }
        today_reviews += 1;
        if !self.card_set.contains(&card_id) {
            today_new_cards += 1;
        }

        let mut features = Vec::with_capacity(CARD_FEATURE_COLUMNS.len() + ID_ENCODING_LEN + 28);
        features.extend([
            scale_elapsed_days(input.elapsed_days) as f32,
            scale_elapsed_days_cumulative(elapsed_days_cumulative) as f32,
            scale_elapsed_seconds(input.elapsed_seconds) as f32,
            elapsed_seconds_sin(input.elapsed_seconds) as f32,
            elapsed_seconds_cos(input.elapsed_seconds) as f32,
            scale_elapsed_seconds_cumulative(elapsed_seconds_cumulative) as f32,
            elapsed_seconds_sin(elapsed_seconds_cumulative) as f32,
            elapsed_seconds_cos(elapsed_seconds_cumulative) as f32,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            note_missing as f32,
            deck_missing as f32,
            preset_missing as f32,
            scale_day_offset_diff(day_offset - previous_day_offset) as f32,
            ((py_mod(day_offset, 7.0) - 3.0) / 3.0) as f32,
            scale_diff_new_cards(diff_new_cards) as f32,
            scale_diff_reviews(diff_reviews) as f32,
            scale_cum_new_cards_today(today_new_cards as f64) as f32,
            scale_cum_reviews_today(today_reviews as f64) as f32,
            0.0,
            1.0,
        ]);
        for ((submodule, _), id) in ID_SUBMODULES
            .iter()
            .copied()
            .zip([card_id, note_id, deck_id, preset_id])
        {
            features.extend_from_slice(
                self.id_encodings
                    .get(submodule)
                    .and_then(|encodings| encodings.get(&id))
                    .expect("identity encoding presence checked above"),
            );
        }
        features.extend(day_offset_encoding(day_offset, day_offset_first));
        if features.len() != 92 {
            return Err(format!(
                "direct prediction feature vector has {} values, expected 92",
                features.len()
            ));
        }
        Ok(Some((features, ids)))
    }

    pub fn can_batch_predict(&self, row: &PreparedRow) -> Result<bool, String> {
        for (submodule, _) in ID_SUBMODULES {
            let id = get_i64(row, submodule)?;
            if !self
                .id_encodings
                .get(submodule)
                .expect("id encoding map initialized for every submodule")
                .contains_key(&id)
            {
                return Ok(false);
            }
        }

        Ok(self
            .recurrent_state_keys
            .card_states
            .contains(&get_i64(row, "card_id")?)
            && self
                .recurrent_state_keys
                .note_states
                .contains(&get_i64(row, "note_id")?)
            && self
                .recurrent_state_keys
                .deck_states
                .contains(&get_i64(row, "deck_id")?)
            && self
                .recurrent_state_keys
                .preset_states
                .contains(&get_i64(row, "preset_id")?))
    }

    pub fn record_recurrent_state_update(&mut self, row: &PreparedRow) -> Result<(), String> {
        self.record_recurrent_ids((
            get_i64(row, "card_id")?,
            get_i64(row, "note_id")?,
            get_i64(row, "deck_id")?,
            get_i64(row, "preset_id")?,
        ));
        Ok(())
    }

    pub fn record_processed_row(&mut self, row: &PreparedRow) -> Result<(), String> {
        let card_id = get_i64(row, "card_id")?;
        let elapsed_days = get_f64(row, "elapsed_days")?;
        let elapsed_seconds = get_f64(row, "elapsed_seconds")?;
        let day_offset = get_f64(row, "day_offset")?;

        self.record_processed_values(card_id, elapsed_days, elapsed_seconds, day_offset);
        Ok(())
    }

    fn record_recurrent_ids(&mut self, ids: (i64, i64, i64, i64)) {
        self.recurrent_state_keys.card_states.insert(ids.0);
        self.recurrent_state_keys.note_states.insert(ids.1);
        self.recurrent_state_keys.deck_states.insert(ids.2);
        self.recurrent_state_keys.preset_states.insert(ids.3);
        self.recurrent_state_keys.global_state = true;
    }

    fn record_processed_values(
        &mut self,
        card_id: i64,
        elapsed_days: f64,
        elapsed_seconds: f64,
        day_offset: f64,
    ) {
        *self
            .card2elapsed_days_cumulative
            .entry(card_id)
            .or_insert(0.0) += elapsed_days;
        *self
            .card2elapsed_seconds_cumulative
            .entry(card_id)
            .or_insert(0.0) += elapsed_seconds;

        if self.first_day_offset.is_none() {
            self.first_day_offset = Some(day_offset);
        }

        if day_offset != self.today {
            self.today = day_offset;
            self.today_new_cards = 0;
            self.today_reviews = -1;
        }
        self.today_reviews += 1;

        if !self.card_set.contains(&card_id) {
            self.today_new_cards += 1;
            self.card_set.insert(card_id);
            self.card_count += 1;
            self.card2first_day_offset.insert(
                card_id,
                day_offset - self.first_day_offset.expect("first_day_offset set above"),
            );
        }

        self.prev_day_offset = Some(day_offset);
        self.last_i.insert(card_id, self.i);
        self.last_new_cards.insert(card_id, self.card_count);
        self.i += 1;
    }

    fn add_same(&self, input: &ReviewInput) -> PreparedRow {
        let card_id = input.card_id;
        let elapsed_days_cumulative = self
            .card2elapsed_days_cumulative
            .get(&card_id)
            .copied()
            .unwrap_or(0.0)
            + input.elapsed_days;
        let elapsed_seconds_cumulative = self
            .card2elapsed_seconds_cumulative
            .get(&card_id)
            .copied()
            .unwrap_or(0.0)
            + input.elapsed_seconds;

        let mut day_offset = input.day_offset;
        if let Some(first_day_offset) = self.first_day_offset {
            day_offset -= first_day_offset;
        } else {
            day_offset = 0.0;
        }

        let day_offset_first = self
            .card2first_day_offset
            .get(&card_id)
            .copied()
            .unwrap_or(day_offset);

        let (note_id, note_missing) = normalize_id(input.note_id, Some(card_id));
        let (deck_id, deck_missing) = normalize_id(input.deck_id, None);
        let (preset_id, preset_missing) = normalize_id(input.preset_id, None);

        let previous_day_offset = self.prev_day_offset.unwrap_or(0.0);
        let unscaled_diff_new_cards = self
            .last_new_cards
            .get(&card_id)
            .map(|last| self.card_count as i64 - *last as i64)
            .unwrap_or(0)
            .max(0) as f64;
        let unscaled_diff_reviews = self
            .last_i
            .get(&card_id)
            .map(|last| (self.i - *last - 1).max(0))
            .unwrap_or(0) as f64;

        let mut row_today_reviews = self.today_reviews;
        let mut row_today_new_cards = self.today_new_cards;
        if day_offset != self.today {
            row_today_new_cards = 0;
            row_today_reviews = -1;
        }
        row_today_reviews += 1;
        if !self.card_set.contains(&card_id) {
            row_today_new_cards += 1;
        }

        let mut row = PreparedRow::new();
        row.insert("review_id".to_string(), RowValue::Int(input.review_id));
        row.insert("card_id".to_string(), RowValue::Int(card_id));
        row.insert("note_id".to_string(), RowValue::Int(note_id));
        row.insert("deck_id".to_string(), RowValue::Int(deck_id));
        row.insert("preset_id".to_string(), RowValue::Int(preset_id));
        row.insert("day_offset".to_string(), RowValue::Float(day_offset));
        row.insert(
            "day_offset_first".to_string(),
            RowValue::Float(day_offset_first),
        );
        row.insert(
            "elapsed_days".to_string(),
            RowValue::Float(input.elapsed_days),
        );
        row.insert(
            "elapsed_days_cumulative".to_string(),
            RowValue::Float(elapsed_days_cumulative),
        );
        row.insert(
            "scaled_elapsed_days_cumulative".to_string(),
            RowValue::Float(scale_elapsed_days_cumulative(elapsed_days_cumulative)),
        );
        row.insert(
            "elapsed_seconds".to_string(),
            RowValue::Float(input.elapsed_seconds),
        );
        row.insert(
            "scaled_elapsed_seconds".to_string(),
            RowValue::Float(scale_elapsed_seconds(input.elapsed_seconds)),
        );
        row.insert(
            "scaled_elapsed_days".to_string(),
            RowValue::Float(scale_elapsed_days(input.elapsed_days)),
        );
        row.insert(
            "elapsed_seconds_cumulative".to_string(),
            RowValue::Float(elapsed_seconds_cumulative),
        );
        row.insert(
            "scaled_elapsed_seconds_cumulative".to_string(),
            RowValue::Float(scale_elapsed_seconds_cumulative(elapsed_seconds_cumulative)),
        );
        row.insert(
            "elapsed_seconds_sin".to_string(),
            RowValue::Float(elapsed_seconds_sin(input.elapsed_seconds)),
        );
        row.insert(
            "elapsed_seconds_cos".to_string(),
            RowValue::Float(elapsed_seconds_cos(input.elapsed_seconds)),
        );
        row.insert(
            "elapsed_seconds_cumulative_sin".to_string(),
            RowValue::Float(elapsed_seconds_sin(elapsed_seconds_cumulative)),
        );
        row.insert(
            "elapsed_seconds_cumulative_cos".to_string(),
            RowValue::Float(elapsed_seconds_cos(elapsed_seconds_cumulative)),
        );
        row.insert("note_id_is_nan".to_string(), RowValue::Float(note_missing));
        row.insert("deck_id_is_nan".to_string(), RowValue::Float(deck_missing));
        row.insert(
            "preset_id_is_nan".to_string(),
            RowValue::Float(preset_missing),
        );
        row.insert(
            "day_offset_diff".to_string(),
            RowValue::Float(scale_day_offset_diff(day_offset - previous_day_offset)),
        );
        row.insert(
            "day_of_week".to_string(),
            RowValue::Float((py_mod(day_offset, 7.0) - 3.0) / 3.0),
        );
        row.insert(
            "diff_new_cards".to_string(),
            RowValue::Float(scale_diff_new_cards(unscaled_diff_new_cards)),
        );
        row.insert(
            "diff_reviews".to_string(),
            RowValue::Float(scale_diff_reviews(unscaled_diff_reviews)),
        );
        row.insert(
            "cum_new_cards_today".to_string(),
            RowValue::Float(scale_cum_new_cards_today(row_today_new_cards as f64)),
        );
        row.insert(
            "cum_reviews_today".to_string(),
            RowValue::Float(scale_cum_reviews_today(row_today_reviews as f64)),
        );
        row
    }
}

fn append_card_features(row: &PreparedRow, features: &mut Vec<f32>) -> Result<(), String> {
    for column in CARD_FEATURE_COLUMNS {
        features.push(get_f64(row, column)? as f32);
    }
    Ok(())
}

fn append_day_offset_features(row: &PreparedRow, features: &mut Vec<f32>) -> Result<(), String> {
    features.extend(day_offset_encoding(
        get_f64(row, "day_offset")?,
        get_f64(row, "day_offset_first")?,
    ));
    Ok(())
}

fn get_f64(row: &PreparedRow, field: &str) -> Result<f64, String> {
    row.get(field)
        .ok_or_else(|| format!("prepared row is missing field '{field}'"))?
        .as_f64(field)
}

fn get_i64(row: &PreparedRow, field: &str) -> Result<i64, String> {
    row.get(field)
        .ok_or_else(|| format!("prepared row is missing field '{field}'"))?
        .as_i64(field)
}

fn normalize_id(value: MaybeId, note_card_id: Option<i64>) -> (i64, f64) {
    match value {
        MaybeId::Present(id) => (id, 0.0),
        MaybeId::Missing => (ID_PLACEHOLDER + note_card_id.unwrap_or(0), 1.0),
    }
}

fn py_mod(value: f64, modulus: f64) -> f64 {
    ((value % modulus) + modulus) % modulus
}

#[cfg(test)]
mod tests {
    use super::*;

    fn review(card_id: i64, day_offset: f64, elapsed_days: f64) -> ReviewInput {
        ReviewInput {
            review_id: card_id,
            card_id,
            note_id: MaybeId::Present(card_id + 100),
            deck_id: MaybeId::Present(7),
            preset_id: MaybeId::Present(8),
            day_offset,
            elapsed_days,
            elapsed_seconds: 120.0,
            rating: Some(3),
            duration: Some(2.0),
            state: Some(1.0),
        }
    }

    #[test]
    fn predict_preparation_does_not_mutate_state() {
        let state = FeatureState::default();
        let prepared = state.prepare_predict_row(&review(1, 10.0, 0.0));
        assert_eq!(prepared["day_offset"], RowValue::Float(0.0));
        assert_eq!(prepared["is_query"], RowValue::Float(1.0));
        assert!(state.card_set.is_empty());
        assert_eq!(state.i, 0);
    }

    #[test]
    fn process_record_updates_same_counters_as_python_oracle() {
        let mut state = FeatureState::default();
        let first = state.prepare_process_row(&review(1, 10.0, 0.0)).unwrap();
        state.record_processed_row(&first).unwrap();

        assert_eq!(state.first_day_offset, Some(0.0));
        assert_eq!(state.prev_day_offset, Some(0.0));
        assert_eq!(state.today, 0.0);
        assert_eq!(state.today_reviews, 0);
        assert_eq!(state.today_new_cards, 1);
        assert_eq!(state.i, 1);

        let second = state.prepare_process_row(&review(1, 11.0, 1.0)).unwrap();
        assert_eq!(second["day_offset"], RowValue::Float(11.0));
        state.record_processed_row(&second).unwrap();

        assert_eq!(state.prev_day_offset, Some(11.0));
        assert_eq!(state.today, 11.0);
        assert_eq!(state.today_reviews, 0);
        assert_eq!(state.today_new_cards, 0);
        assert_eq!(state.i, 2);
        assert_eq!(state.card_set.len(), 1);
    }

    #[test]
    fn feature_vector_uses_torch_seeded_id_encodings_and_mutation_flag() {
        let mut state = FeatureState::with_torch_seed(12_345);
        let prepared = state.prepare_process_row(&review(1, 10.0, 0.0)).unwrap();
        let vector = state.feature_vector(&prepared, true).unwrap();

        assert_eq!(vector.len(), 92);
        assert_eq!(
            state.id_encodings["card_id"][&1],
            vec![0.5, -0.5, -0.5, -0.5, -1.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5]
        );
        assert_eq!(
            &vector[24..36],
            state.id_encodings["card_id"][&1].as_slice()
        );

        let query = state.prepare_predict_row(&review(2, 11.0, 1.0));
        let before_len = state.id_encodings["card_id"].len();
        let _ = state.feature_vector(&query, false).unwrap();
        assert_eq!(state.id_encodings["card_id"].len(), before_len);
    }

    #[test]
    fn batch_feature_vectors_require_existing_id_encodings() {
        let mut state = FeatureState::with_torch_seed(12_345);
        let prepared = state.prepare_process_row(&review(1, 10.0, 0.0)).unwrap();
        state.feature_vector(&prepared, true).unwrap();

        let rows = vec![prepared.clone(), prepared];
        let vectors = state.feature_vectors(&rows).unwrap();
        assert_eq!(vectors.len(), 2);
        assert_eq!(vectors[0], vectors[1]);

        let query = state.prepare_predict_row(&review(2, 11.0, 1.0));
        assert!(state.feature_vectors(&[query]).is_err());
    }

    #[test]
    fn direct_predict_features_are_bit_exact_with_prepared_rows() {
        let mut state = FeatureState::with_torch_seed(12_345);
        let first_input = review(1, 10.0, 0.0);
        let first = state.prepare_process_row(&first_input).unwrap();
        state.process_feature_vector(&first).unwrap();
        state.record_recurrent_state_update(&first).unwrap();
        state.record_processed_row(&first).unwrap();

        let mut query_input = review(1, 12.5, 2.25);
        query_input.elapsed_seconds = 91_234.5;
        let prepared = state.prepare_predict_row(&query_input);
        let expected = state.feature_vectors(&[prepared]).unwrap().pop().unwrap();
        let (direct, ids) = state
            .direct_predict_features(&query_input)
            .unwrap()
            .expect("processed identities are batchable");

        assert_eq!(ids, (1, 101, 7, 8));
        assert_eq!(direct, expected);
    }

    #[test]
    fn direct_predict_features_retain_unseen_fallback() {
        let state = FeatureState::with_torch_seed(12_345);
        assert!(state
            .direct_predict_features(&review(1, 10.0, 0.0))
            .unwrap()
            .is_none());
    }

    #[test]
    fn direct_process_feature_pairs_are_bit_exact_with_prepared_rows() {
        let mut reference = FeatureState::with_torch_seed(12_345);
        let mut direct = FeatureState::with_torch_seed(12_345);
        let mut inputs = vec![
            review(1, 10.0, 0.0),
            review(1, 12.5, 2.25),
            review(2, 13.0, 0.5),
        ];
        inputs[1].elapsed_seconds = 91_234.5;
        inputs[2].note_id = MaybeId::Missing;
        inputs[2].deck_id = MaybeId::Missing;
        inputs[2].preset_id = MaybeId::Missing;

        for input in &inputs {
            let mut expected = Vec::new();
            let predict = reference.prepare_predict_row(input);
            expected.extend(reference.predict_feature_vector(&predict).unwrap());
            let process = reference.prepare_process_row(input).unwrap();
            expected.extend(reference.process_feature_vector(&process).unwrap());
            reference.record_recurrent_state_update(&process).unwrap();
            reference.record_processed_row(&process).unwrap();

            let mut actual = Vec::new();
            let ids = direct
                .append_process_feature_pair(input, &mut actual)
                .unwrap();
            assert_eq!(ids, FeatureState::normalized_review_ids(input));
            assert_eq!(actual, expected);
        }

        assert_eq!(direct.first_day_offset, reference.first_day_offset);
        assert_eq!(direct.prev_day_offset, reference.prev_day_offset);
        assert_eq!(direct.card_set, reference.card_set);
        assert_eq!(direct.card_count, reference.card_count);
        assert_eq!(direct.last_new_cards, reference.last_new_cards);
        assert_eq!(direct.i, reference.i);
        assert_eq!(direct.last_i, reference.last_i);
        assert_eq!(direct.today, reference.today);
        assert_eq!(direct.today_reviews, reference.today_reviews);
        assert_eq!(direct.today_new_cards, reference.today_new_cards);
        assert_eq!(
            direct.card2first_day_offset,
            reference.card2first_day_offset
        );
        assert_eq!(
            direct.card2elapsed_days_cumulative,
            reference.card2elapsed_days_cumulative
        );
        assert_eq!(
            direct.card2elapsed_seconds_cumulative,
            reference.card2elapsed_seconds_cumulative
        );
        assert_eq!(direct.id_encodings, reference.id_encodings);
        assert_eq!(
            direct.recurrent_state_keys.card_states,
            reference.recurrent_state_keys.card_states
        );
        assert_eq!(
            direct.recurrent_state_keys.note_states,
            reference.recurrent_state_keys.note_states
        );
        assert_eq!(
            direct.recurrent_state_keys.deck_states,
            reference.recurrent_state_keys.deck_states
        );
        assert_eq!(
            direct.recurrent_state_keys.preset_states,
            reference.recurrent_state_keys.preset_states
        );
        assert_eq!(
            direct.recurrent_state_keys.global_state,
            reference.recurrent_state_keys.global_state
        );
        assert_eq!(
            direct.id_rng.to_torch_rng_state_bytes(),
            reference.id_rng.to_torch_rng_state_bytes()
        );
    }

    #[test]
    fn direct_process_feature_only_matches_pair_process_rows_and_state() {
        let mut pair_state = FeatureState::with_torch_seed(12_345);
        let mut process_only_state = FeatureState::with_torch_seed(12_345);
        let mut inputs = vec![
            review(1, 10.0, 0.0),
            review(1, 12.5, 2.25),
            review(2, 13.0, 0.5),
        ];
        inputs[1].elapsed_seconds = 91_234.5;
        inputs[2].note_id = MaybeId::Missing;
        inputs[2].deck_id = MaybeId::Missing;
        inputs[2].preset_id = MaybeId::Missing;

        for input in &inputs {
            let mut pair = Vec::new();
            let pair_ids = pair_state
                .append_process_feature_pair(input, &mut pair)
                .unwrap();
            let mut process_only = Vec::new();
            let process_only_ids = process_only_state
                .append_process_feature_only(input, &mut process_only)
                .unwrap();
            assert_eq!(process_only_ids, pair_ids);
            assert_eq!(process_only, pair[92..]);
        }

        assert_eq!(process_only_state.i, pair_state.i);
        assert_eq!(process_only_state.id_encodings, pair_state.id_encodings);
        assert_eq!(
            process_only_state.recurrent_state_keys.card_states,
            pair_state.recurrent_state_keys.card_states
        );
        assert_eq!(
            process_only_state.id_rng.to_torch_rng_state_bytes(),
            pair_state.id_rng.to_torch_rng_state_bytes()
        );
    }

    #[test]
    fn predict_feature_vector_restores_rng_for_unseen_ids() {
        let query = review(2, 11.0, 1.0);

        let mut raw_state = FeatureState::with_torch_seed(12_345);
        let raw_query = raw_state.prepare_predict_row(&query);
        let expected = raw_state.feature_vector(&raw_query, false).unwrap();
        let raw_process_row = raw_state.prepare_process_row(&query).unwrap();
        let raw_next_process = raw_state.process_feature_vector(&raw_process_row).unwrap();

        let mut public_state = FeatureState::with_torch_seed(12_345);
        let public_query = public_state.prepare_predict_row(&query);
        assert!(public_state
            .skip_needs_rng_restore(&public_query, true)
            .unwrap());

        let first = public_state.predict_feature_vector(&public_query).unwrap();
        let second = public_state.predict_feature_vector(&public_query).unwrap();
        assert_eq!(first, expected);
        assert_eq!(second, expected);
        assert!(public_state.id_encodings["card_id"].is_empty());

        let public_process_row = public_state.prepare_process_row(&query).unwrap();
        let public_process = public_state
            .process_feature_vector(&public_process_row)
            .unwrap();
        assert_ne!(public_process, raw_next_process);

        let mut clean_state = FeatureState::with_torch_seed(12_345);
        let clean_process_row = clean_state.prepare_process_row(&query).unwrap();
        let clean_process = clean_state
            .process_feature_vector(&clean_process_row)
            .unwrap();
        assert_eq!(public_process, clean_process);
    }

    #[test]
    fn can_batch_predict_requires_id_encodings_and_recurrent_state_keys() {
        let mut state = FeatureState::with_torch_seed(12_345);
        let processed = state.prepare_process_row(&review(1, 10.0, 0.0)).unwrap();

        assert!(!state.can_batch_predict(&processed).unwrap());
        state.process_feature_vector(&processed).unwrap();
        assert!(!state.can_batch_predict(&processed).unwrap());
        assert_eq!(state.i, 0);

        state.record_recurrent_state_update(&processed).unwrap();
        assert!(state.can_batch_predict(&processed).unwrap());
        assert_eq!(state.i, 0);

        let query = state.prepare_predict_row(&review(2, 11.0, 1.0));
        assert!(!state.can_batch_predict(&query).unwrap());
        state.predict_feature_vector(&query).unwrap();
        assert!(!state.recurrent_state_keys.card_states.contains(&2));
        assert!(!state.can_batch_predict(&query).unwrap());

        state.feature_vector(&query, true).unwrap();
        assert!(!state.can_batch_predict(&query).unwrap());
    }
}
