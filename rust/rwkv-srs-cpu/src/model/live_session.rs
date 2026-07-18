//! Rust-owned candidate index and state machine for compact live prediction.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use pyo3::prelude::*;

use super::gpu_process_scan::synchronize_gpu_process_state_with_state;
use super::runtime::{predict_review_inputs_cpu, predict_review_inputs_gpu};
use super::{py_value_error, NativeRnn};
use crate::gpu::py_gpu_error;
use crate::state::{FeatureState, MaybeId, ReviewInput};

// Optimized 20k-slot measurements put the merge/full-sort crossover at about
// 95% updated membership (see the ignored benchmark at the end of this file).
const PARTIAL_REFRESH_REBUILD_NUMERATOR: usize = 19;
const PARTIAL_REFRESH_REBUILD_DENOMINATOR: usize = 20;
const UNDO_SLOT_UNRECORDED: u8 = 0;
const UNDO_SLOT_REFRESH_DELTA: u8 = 1;
const UNDO_SLOT_FULL_DELTA: u8 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LiveOrder {
    RetrievabilityAscending,
    RetrievabilityDescending,
    RelativeOverdueness,
    Random,
}

impl LiveOrder {
    pub(crate) fn parse(value: &str) -> PyResult<Self> {
        match value {
            "retrievability_ascending" => Ok(Self::RetrievabilityAscending),
            "retrievability_descending" => Ok(Self::RetrievabilityDescending),
            "relative_overdueness" => Ok(Self::RelativeOverdueness),
            "random" => Ok(Self::Random),
            _ => Err(py_value_error(format!(
                "order must be 'retrievability_ascending', \
                 'retrievability_descending', 'relative_overdueness', or 'random'; \
                 got {value:?}"
            ))),
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::RetrievabilityAscending => "retrievability_ascending",
            Self::RetrievabilityDescending => "retrievability_descending",
            Self::RelativeOverdueness => "relative_overdueness",
            Self::Random => "random",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LivePredictionMode {
    Oracle,
    Fast,
    Gpu,
}

impl LivePredictionMode {
    pub(crate) fn parse(value: &str) -> PyResult<Self> {
        match value {
            "oracle" => Ok(Self::Oracle),
            "fast" => Ok(Self::Fast),
            "gpu" => Ok(Self::Gpu),
            _ => Err(py_value_error(format!(
                "mode must be 'oracle', 'fast', or 'gpu'; got {value:?}"
            ))),
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Oracle => "oracle",
            Self::Fast => "fast",
            Self::Gpu => "gpu",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct LiveCandidateSeedNative {
    pub(crate) row: ReviewInput,
    pub(crate) target_retrievability: f64,
    pub(crate) intraday_target_retrievability: f64,
    pub(crate) tie_breaker: u64,
    pub(crate) random_key: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CandidateStatus {
    Active,
    Excluded,
    PendingRefresh,
    ExcludedPendingRefresh,
    Removed,
}

impl CandidateStatus {
    fn is_included(self) -> bool {
        matches!(self, Self::Active | Self::PendingRefresh)
    }

    fn is_excluded(self) -> bool {
        matches!(self, Self::Excluded | Self::ExcludedPendingRefresh)
    }

    fn prediction_is_usable(self) -> bool {
        matches!(self, Self::Active | Self::Excluded)
    }

    fn excluded(self) -> Self {
        match self {
            Self::Active => Self::Excluded,
            Self::PendingRefresh => Self::ExcludedPendingRefresh,
            other => other,
        }
    }

    fn included(self) -> Self {
        match self {
            Self::Excluded => Self::Active,
            Self::ExcludedPendingRefresh => Self::PendingRefresh,
            other => other,
        }
    }

    fn refreshed(self) -> Self {
        match self {
            Self::PendingRefresh => Self::Active,
            Self::ExcludedPendingRefresh => Self::Excluded,
            other => other,
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Excluded => "excluded",
            Self::PendingRefresh => "pending_refresh",
            Self::ExcludedPendingRefresh => "excluded_pending_refresh",
            Self::Removed => "removed",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct LiveCandidate {
    pub(crate) review_id: i64,
    pub(crate) card_id: i64,
    pub(crate) note_id: MaybeId,
    pub(crate) deck_id: MaybeId,
    pub(crate) preset_id: MaybeId,
    pub(crate) has_prior_review: bool,
    pub(crate) new_elapsed_days: f64,
    pub(crate) new_elapsed_seconds: f64,
    pub(crate) last_review_timestamp_seconds: Option<f64>,
    pub(crate) last_review_day_offset: Option<f64>,
    pub(crate) query_timestamp_seconds: f64,
    pub(crate) query_day_offset: f64,
    pub(crate) queried_elapsed_days: f64,
    pub(crate) queried_elapsed_seconds: f64,
}

impl LiveCandidate {
    fn from_seed(
        seed: &LiveCandidateSeedNative,
        target_timestamp_seconds: f64,
        target_day_offset: f64,
    ) -> PyResult<Self> {
        let elapsed_days = seed.row.elapsed_days;
        let elapsed_seconds = seed.row.elapsed_seconds;
        if !elapsed_days.is_finite() || !elapsed_seconds.is_finite() {
            return Err(py_value_error(format!(
                "candidate {} elapsed values must be finite",
                seed.row.card_id
            )));
        }
        let day_is_new = elapsed_days == -1.0;
        let seconds_is_new = elapsed_seconds == -1.0;
        if day_is_new != seconds_is_new {
            return Err(py_value_error(format!(
                "candidate {} must use the new-card sentinel for both elapsed_days and \
                 elapsed_seconds",
                seed.row.card_id
            )));
        }
        if !day_is_new && (elapsed_days < 0.0 || elapsed_seconds < 0.0) {
            return Err(py_value_error(format!(
                "candidate {} elapsed values must be non-negative or exactly -1",
                seed.row.card_id
            )));
        }
        let (last_review_timestamp_seconds, last_review_day_offset) = if day_is_new {
            (None, None)
        } else {
            let timestamp_anchor = target_timestamp_seconds - elapsed_seconds;
            let day_anchor = target_day_offset - elapsed_days;
            if !timestamp_anchor.is_finite() || !day_anchor.is_finite() {
                return Err(py_value_error(format!(
                    "candidate {} cannot derive finite last-review anchors from the initial target",
                    seed.row.card_id
                )));
            }
            (Some(timestamp_anchor), Some(day_anchor))
        };
        Ok(Self {
            review_id: seed.row.review_id,
            card_id: seed.row.card_id,
            note_id: seed.row.note_id,
            deck_id: seed.row.deck_id,
            preset_id: seed.row.preset_id,
            has_prior_review: !day_is_new,
            new_elapsed_days: elapsed_days,
            new_elapsed_seconds: elapsed_seconds,
            last_review_timestamp_seconds,
            last_review_day_offset,
            query_timestamp_seconds: target_timestamp_seconds,
            query_day_offset: target_day_offset,
            queried_elapsed_days: elapsed_days,
            queried_elapsed_seconds: elapsed_seconds,
        })
    }

    fn prediction_input(
        &self,
        target_timestamp_seconds: f64,
        target_day_offset: f64,
    ) -> ReviewInput {
        let (elapsed_days, elapsed_seconds) = if self.has_prior_review {
            (
                target_day_offset
                    - self
                        .last_review_day_offset
                        .expect("reviewed candidate has a day anchor"),
                target_timestamp_seconds
                    - self
                        .last_review_timestamp_seconds
                        .expect("reviewed candidate has a timestamp anchor"),
            )
        } else {
            (self.new_elapsed_days, self.new_elapsed_seconds)
        };
        self.review_input(target_day_offset, elapsed_days, elapsed_seconds)
    }

    fn initial_prediction_input(&self) -> ReviewInput {
        self.review_input(
            self.query_day_offset,
            self.queried_elapsed_days,
            self.queried_elapsed_seconds,
        )
    }

    fn review_input(
        &self,
        target_day_offset: f64,
        elapsed_days: f64,
        elapsed_seconds: f64,
    ) -> ReviewInput {
        ReviewInput {
            review_id: self.review_id,
            card_id: self.card_id,
            note_id: self.note_id,
            deck_id: self.deck_id,
            preset_id: self.preset_id,
            day_offset: target_day_offset,
            elapsed_days,
            elapsed_seconds,
            rating: None,
            duration: None,
            state: None,
        }
    }

    fn answer_timestamp_seconds(&self, target_timestamp_seconds: f64, elapsed_seconds: f64) -> f64 {
        if self.has_prior_review {
            self.last_review_timestamp_seconds
                .expect("reviewed candidate has a timestamp anchor")
                + elapsed_seconds
        } else {
            target_timestamp_seconds
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct LiveCandidateValue {
    candidate: LiveCandidate,
    prediction: f64,
    normal_target: f64,
    intraday_target: f64,
    tie_breaker: u64,
    random_key: u64,
    status: CandidateStatus,
    eligible: bool,
}

#[derive(Clone, Copy, Debug)]
struct LiveRefreshUndoValue {
    query_timestamp_seconds: f64,
    query_day_offset: f64,
    queried_elapsed_days: f64,
    queried_elapsed_seconds: f64,
    prediction: f64,
    status: CandidateStatus,
    eligible: bool,
}

#[derive(Debug)]
pub(crate) struct LiveIndexUndoFrame {
    pub(crate) session_token: u64,
    previous_generation: u64,
    previous_retention_extra: f64,
    previous_target_timestamp_seconds: f64,
    previous_target_day_offset: f64,
    candidate_deltas: Vec<(usize, Option<LiveCandidateValue>)>,
    refresh_deltas: Vec<(usize, LiveRefreshUndoValue)>,
    recorded_slots: Vec<u8>,
    pre_reconciliation_state: Option<Box<LiveReconciliationUndoState>>,
}

/// Logical live-session state retained across a full-universe reconciliation.
///
/// Runtime configuration, profiling, diagnostics, and reusable scratch buffers
/// deliberately remain on the active session. Undo only needs the candidate
/// slot space and its rank/query state; retaining every scratch allocation in
/// each undo frame would multiply memory by the configured undo depth.
#[derive(Debug)]
struct LiveReconciliationUndoState {
    candidates: Vec<LiveCandidate>,
    slot_by_card_id: HashMap<i64, usize>,
    predictions: Vec<f64>,
    normal_targets: Vec<f64>,
    intraday_targets: Vec<f64>,
    tie_breakers: Vec<u64>,
    random_keys: Vec<u64>,
    statuses: Vec<CandidateStatus>,
    eligibilities: Vec<bool>,
    ordered_slots: Vec<usize>,
    retention_extra: f64,
    target_timestamp_seconds: f64,
    target_day_offset: f64,
}

impl LiveReconciliationUndoState {
    fn candidate_count(&self) -> usize {
        self.candidates.len()
    }

    fn tracked_capacity_bytes(&self) -> usize {
        tracked_candidate_capacity_bytes(
            &self.candidates,
            &self.slot_by_card_id,
            &self.predictions,
            &self.normal_targets,
            &self.intraday_targets,
            &self.tie_breakers,
            &self.random_keys,
            &self.statuses,
            &self.eligibilities,
            &self.ordered_slots,
        )
    }
}

impl LiveIndexUndoFrame {
    fn capture(state: &LivePredictionSessionState, answered_slot: usize) -> Self {
        // The normal answer/refresh path needs one full delta for the answered
        // card and compact deltas for refreshed cards. Leave room for a few
        // uncommon include/remove/upsert mutations without reserving thousands
        // of full candidate snapshots on every answer.
        let mut candidate_deltas = Vec::with_capacity(4);
        candidate_deltas.push((answered_slot, Some(state.value(answered_slot))));
        let mut recorded_slots = vec![UNDO_SLOT_UNRECORDED; state.candidates.len()];
        recorded_slots[answered_slot] = UNDO_SLOT_FULL_DELTA;
        Self {
            session_token: state.token,
            previous_generation: state.generation,
            previous_retention_extra: state.retention_extra,
            previous_target_timestamp_seconds: state.target_timestamp_seconds,
            previous_target_day_offset: state.target_day_offset,
            candidate_deltas,
            refresh_deltas: Vec::with_capacity(state.refresh_limit.min(state.candidates.len())),
            recorded_slots,
            pre_reconciliation_state: None,
        }
    }

    pub(crate) fn record_slot(&mut self, state: &LivePredictionSessionState, slot: usize) {
        if self.pre_reconciliation_state.is_some() {
            return;
        }
        self.ensure_recorded_slot(slot);
        match self.recorded_slots[slot] {
            UNDO_SLOT_UNRECORDED => {
                self.candidate_deltas.push((slot, Some(state.value(slot))));
                self.recorded_slots[slot] = UNDO_SLOT_FULL_DELTA;
            }
            UNDO_SLOT_REFRESH_DELTA => {
                let refresh_value = self
                    .refresh_deltas
                    .iter()
                    .find_map(|(recorded_slot, value)| (*recorded_slot == slot).then_some(*value))
                    .expect("refresh-recorded live slot has a refresh delta");
                let mut value = state.value(slot);
                refresh_value.restore_full_value(&mut value);
                self.candidate_deltas.push((slot, Some(value)));
                self.recorded_slots[slot] = UNDO_SLOT_FULL_DELTA;
            }
            _ => {}
        }
    }

    pub(crate) fn record_refresh_slot(&mut self, state: &LivePredictionSessionState, slot: usize) {
        if self.pre_reconciliation_state.is_some() {
            return;
        }
        self.ensure_recorded_slot(slot);
        if self.recorded_slots[slot] == UNDO_SLOT_UNRECORDED {
            self.refresh_deltas
                .push((slot, state.refresh_undo_value(slot)));
            self.recorded_slots[slot] = UNDO_SLOT_REFRESH_DELTA;
        }
    }

    pub(crate) fn record_new_slot(&mut self, slot: usize) {
        if self.pre_reconciliation_state.is_some() {
            return;
        }
        self.ensure_recorded_slot(slot);
        if self.recorded_slots[slot] == UNDO_SLOT_UNRECORDED {
            self.candidate_deltas.push((slot, None));
            self.recorded_slots[slot] = UNDO_SLOT_FULL_DELTA;
        }
    }

    fn ensure_recorded_slot(&mut self, slot: usize) {
        if slot >= self.recorded_slots.len() {
            self.recorded_slots.resize(slot + 1, UNDO_SLOT_UNRECORDED);
        }
    }

    pub(crate) fn record_reconciliation_state(&mut self, state: LivePredictionSessionState) {
        // The first complete replacement after an answer is the only boundary
        // that matters to that answer's undo. Later mutations and replacements
        // are discarded when this snapshot is restored, so recording their
        // slot deltas would both waste memory and refer to the wrong slot space.
        if self.pre_reconciliation_state.is_none() {
            self.pre_reconciliation_state = Some(Box::new(state.into_reconciliation_undo_state()));
        }
    }

    pub(crate) fn reconciliation_snapshot_candidate_count(&self) -> usize {
        self.pre_reconciliation_state
            .as_deref()
            .map_or(0, LiveReconciliationUndoState::candidate_count)
    }

    pub(crate) fn has_reconciliation_snapshot(&self) -> bool {
        self.pre_reconciliation_state.is_some()
    }

    pub(crate) fn reconciliation_snapshot_tracked_capacity_bytes(&self) -> usize {
        self.pre_reconciliation_state
            .as_deref()
            .map_or(0, LiveReconciliationUndoState::tracked_capacity_bytes)
    }
}

impl LiveRefreshUndoValue {
    fn restore_full_value(self, value: &mut LiveCandidateValue) {
        value.candidate.query_timestamp_seconds = self.query_timestamp_seconds;
        value.candidate.query_day_offset = self.query_day_offset;
        value.candidate.queried_elapsed_days = self.queried_elapsed_days;
        value.candidate.queried_elapsed_seconds = self.queried_elapsed_seconds;
        value.prediction = self.prediction;
        value.status = self.status;
        value.eligible = self.eligible;
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LiveRefreshProfile {
    pub(crate) card_id_parse_ns: u128,
    pub(crate) seed_parse_ns: u128,
    pub(crate) scope_validation_ns: u128,
    /// Native reconciliation wall time from boundary parsing through final selection.
    pub(crate) native_total_ns: u128,
    pub(crate) membership_selection_ns: u128,
    pub(crate) input_construction_ns: u128,
    pub(crate) prediction_ns: u128,
    pub(crate) application_ns: u128,
    pub(crate) ordering_ns: u128,
    /// Replacement swap plus optional first reconciliation undo snapshot.
    pub(crate) commit_snapshot_ns: u128,
    pub(crate) final_selection_ns: u128,
    pub(crate) python_result_ns: u128,
    pub(crate) refreshed_count: u64,
    pub(crate) selected_count: u64,
    pub(crate) partial_merge_count: u64,
    pub(crate) full_rebuild_count: u64,
}

impl LiveRefreshProfile {
    fn add_assign(&mut self, other: Self) {
        self.card_id_parse_ns += other.card_id_parse_ns;
        self.seed_parse_ns += other.seed_parse_ns;
        self.scope_validation_ns += other.scope_validation_ns;
        self.native_total_ns += other.native_total_ns;
        self.membership_selection_ns += other.membership_selection_ns;
        self.input_construction_ns += other.input_construction_ns;
        self.prediction_ns += other.prediction_ns;
        self.application_ns += other.application_ns;
        self.ordering_ns += other.ordering_ns;
        self.commit_snapshot_ns += other.commit_snapshot_ns;
        self.final_selection_ns += other.final_selection_ns;
        self.python_result_ns += other.python_result_ns;
        self.refreshed_count += other.refreshed_count;
        self.selected_count += other.selected_count;
        self.partial_merge_count += other.partial_merge_count;
        self.full_rebuild_count += other.full_rebuild_count;
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct LiveSessionProfile {
    pub(crate) enabled: bool,
    pub(crate) refresh_calls: u64,
    pub(crate) failed_refresh_calls: u64,
    pub(crate) cumulative: LiveRefreshProfile,
    pub(crate) last: LiveRefreshProfile,
    pub(crate) reconcile_calls: u64,
    pub(crate) failed_reconcile_calls: u64,
    pub(crate) reconcile_cumulative: LiveRefreshProfile,
    pub(crate) reconcile_last: LiveRefreshProfile,
    pub(crate) failed_reconcile_cumulative: LiveRefreshProfile,
    pub(crate) failed_reconcile_last: LiveRefreshProfile,
}

#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
struct RankKey {
    class: u8,
    order_key: u64,
}

#[derive(Clone, Copy, Debug)]
struct RankEntry {
    key: RankKey,
    slot: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct LiveSelectionValue {
    pub(crate) card_id: i64,
    pub(crate) retrievability: f64,
    pub(crate) target_retrievability: f64,
}

#[derive(Debug)]
pub(crate) struct LiveRefreshOutput {
    pub(crate) generation: u64,
    pub(crate) refreshed_count: usize,
    pub(crate) eligible_count: usize,
    pub(crate) active_count: usize,
    pub(crate) selected: Vec<LiveSelectionValue>,
    pub(crate) next_retention_extra: Option<f64>,
}

#[derive(Clone, Debug)]
pub(crate) struct LiveCandidateSnapshot {
    pub(crate) card_id: i64,
    pub(crate) review_id: i64,
    pub(crate) note_id: MaybeId,
    pub(crate) deck_id: MaybeId,
    pub(crate) preset_id: MaybeId,
    pub(crate) retrievability: f64,
    pub(crate) target_retrievability: f64,
    pub(crate) intraday_target_retrievability: f64,
    pub(crate) applicable_target_retrievability: f64,
    pub(crate) tie_breaker: u64,
    pub(crate) random_key: u64,
    pub(crate) status: CandidateStatus,
    pub(crate) eligible: bool,
    pub(crate) has_prior_review: bool,
    pub(crate) last_review_timestamp_seconds: Option<f64>,
    pub(crate) last_review_day_offset: Option<f64>,
    pub(crate) query_timestamp_seconds: f64,
    pub(crate) query_day_offset: f64,
    pub(crate) elapsed_seconds: f64,
    pub(crate) elapsed_days: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct LivePredictionSessionState {
    pub(crate) token: u64,
    candidates: Vec<LiveCandidate>,
    slot_by_card_id: HashMap<i64, usize>,
    predictions: Vec<f64>,
    normal_targets: Vec<f64>,
    intraday_targets: Vec<f64>,
    tie_breakers: Vec<u64>,
    random_keys: Vec<u64>,
    statuses: Vec<CandidateStatus>,
    eligibilities: Vec<bool>,
    ordered_slots: Vec<usize>,
    transport_slots: Vec<usize>,
    generation: u64,
    retention_extra: f64,
    target_timestamp_seconds: f64,
    target_day_offset: f64,
    order: LiveOrder,
    mode: LivePredictionMode,
    batch_size: usize,
    refresh_limit: usize,
    num_threads: Option<usize>,
    selected_scratch: Vec<usize>,
    updated_entry_scratch: Vec<RankEntry>,
    merge_scratch: Vec<usize>,
    input_scratch: Vec<ReviewInput>,
    commit_slots_scratch: Vec<usize>,
    update_marks: Vec<u64>,
    update_mark_generation: u64,
    last_membership_slots: Vec<usize>,
    last_transport_slots: Vec<usize>,
    pub(crate) profile: LiveSessionProfile,
}

impl LivePredictionSessionState {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_unranked(
        token: u64,
        seeds: &[LiveCandidateSeedNative],
        target_timestamp_seconds: f64,
        target_day_offset: f64,
        order: LiveOrder,
        mode: LivePredictionMode,
        batch_size: usize,
        refresh_limit: usize,
        num_threads: Option<usize>,
        profiling: bool,
    ) -> PyResult<Self> {
        let mut state = Self::new_empty(
            token,
            seeds.len(),
            target_timestamp_seconds,
            target_day_offset,
            order,
            mode,
            batch_size,
            refresh_limit,
            num_threads,
            profiling,
        )?;
        for seed in seeds {
            state.push_seed(seed, target_timestamp_seconds, target_day_offset)?;
        }
        Ok(state)
    }

    #[allow(clippy::too_many_arguments)]
    fn new_empty(
        token: u64,
        candidate_capacity: usize,
        target_timestamp_seconds: f64,
        target_day_offset: f64,
        order: LiveOrder,
        mode: LivePredictionMode,
        batch_size: usize,
        refresh_limit: usize,
        num_threads: Option<usize>,
        profiling: bool,
    ) -> PyResult<Self> {
        validate_target_time(target_timestamp_seconds, target_day_offset)?;
        if batch_size == 0 {
            return Err(py_value_error("batch_size must be at least 1"));
        }
        if refresh_limit == 0 {
            return Err(py_value_error("refresh_limit must be at least 1"));
        }

        let refresh_capacity = refresh_limit.min(candidate_capacity);
        Ok(Self {
            token,
            candidates: Vec::with_capacity(candidate_capacity),
            slot_by_card_id: HashMap::with_capacity(candidate_capacity),
            predictions: Vec::with_capacity(candidate_capacity),
            normal_targets: Vec::with_capacity(candidate_capacity),
            intraday_targets: Vec::with_capacity(candidate_capacity),
            tie_breakers: Vec::with_capacity(candidate_capacity),
            random_keys: Vec::with_capacity(candidate_capacity),
            statuses: Vec::with_capacity(candidate_capacity),
            eligibilities: Vec::with_capacity(candidate_capacity),
            ordered_slots: Vec::with_capacity(candidate_capacity),
            transport_slots: Vec::with_capacity(refresh_capacity),
            generation: 1,
            retention_extra: 0.0,
            target_timestamp_seconds,
            target_day_offset,
            order,
            mode,
            batch_size,
            refresh_limit,
            num_threads,
            selected_scratch: Vec::with_capacity(refresh_capacity),
            updated_entry_scratch: Vec::with_capacity(refresh_capacity),
            merge_scratch: Vec::with_capacity(candidate_capacity),
            input_scratch: Vec::with_capacity(refresh_capacity),
            commit_slots_scratch: Vec::with_capacity(refresh_capacity),
            update_marks: Vec::with_capacity(candidate_capacity),
            update_mark_generation: 0,
            last_membership_slots: Vec::with_capacity(refresh_capacity),
            last_transport_slots: Vec::with_capacity(refresh_capacity),
            profile: LiveSessionProfile {
                enabled: profiling,
                ..LiveSessionProfile::default()
            },
        })
    }

    fn into_reconciliation_undo_state(self) -> LiveReconciliationUndoState {
        LiveReconciliationUndoState {
            candidates: self.candidates,
            slot_by_card_id: self.slot_by_card_id,
            predictions: self.predictions,
            normal_targets: self.normal_targets,
            intraday_targets: self.intraday_targets,
            tie_breakers: self.tie_breakers,
            random_keys: self.random_keys,
            statuses: self.statuses,
            eligibilities: self.eligibilities,
            ordered_slots: self.ordered_slots,
            retention_extra: self.retention_extra,
            target_timestamp_seconds: self.target_timestamp_seconds,
            target_day_offset: self.target_day_offset,
        }
    }

    fn restore_reconciliation_undo_state(&mut self, previous: LiveReconciliationUndoState) {
        self.candidates = previous.candidates;
        self.slot_by_card_id = previous.slot_by_card_id;
        self.predictions = previous.predictions;
        self.normal_targets = previous.normal_targets;
        self.intraday_targets = previous.intraday_targets;
        self.tie_breakers = previous.tie_breakers;
        self.random_keys = previous.random_keys;
        self.statuses = previous.statuses;
        self.eligibilities = previous.eligibilities;
        self.ordered_slots = previous.ordered_slots;
        self.retention_extra = previous.retention_extra;
        self.target_timestamp_seconds = previous.target_timestamp_seconds;
        self.target_day_offset = previous.target_day_offset;

        // Keep the active session's reusable allocations, but discard every
        // slot-indexed scratch value from the replaced universe.
        let candidate_count = self.candidates.len();
        let refresh_capacity = self.refresh_limit.min(candidate_count);
        self.transport_slots.clear();
        self.transport_slots.reserve(refresh_capacity);
        self.selected_scratch.clear();
        self.selected_scratch.reserve(refresh_capacity);
        self.updated_entry_scratch.clear();
        self.updated_entry_scratch.reserve(refresh_capacity);
        self.merge_scratch.clear();
        self.merge_scratch.reserve(candidate_count);
        self.input_scratch.clear();
        self.input_scratch.reserve(refresh_capacity);
        self.commit_slots_scratch.clear();
        self.commit_slots_scratch.reserve(refresh_capacity);
        self.update_marks.clear();
        self.update_marks.resize(candidate_count, 0);
        self.update_mark_generation = 0;
        self.last_membership_slots.clear();
        self.last_membership_slots.reserve(refresh_capacity);
        self.last_transport_slots.clear();
        self.last_transport_slots.reserve(refresh_capacity);
    }

    fn push_seed(
        &mut self,
        seed: &LiveCandidateSeedNative,
        target_timestamp_seconds: f64,
        target_day_offset: f64,
    ) -> PyResult<usize> {
        let candidate =
            LiveCandidate::from_seed(seed, target_timestamp_seconds, target_day_offset)?;
        self.push_value(LiveCandidateValue {
            candidate,
            prediction: f64::NAN,
            normal_target: seed.target_retrievability,
            intraday_target: seed.intraday_target_retrievability,
            tie_breaker: seed.tie_breaker,
            random_key: seed.random_key,
            status: CandidateStatus::PendingRefresh,
            eligible: false,
        })
    }

    fn push_reused_candidate(
        &mut self,
        source: &Self,
        source_slot: usize,
        target_timestamp_seconds: f64,
        target_day_offset: f64,
    ) -> PyResult<usize> {
        let mut value = source.value(source_slot);
        let input = value
            .candidate
            .prediction_input(target_timestamp_seconds, target_day_offset);
        if let Some(error) = live_prediction_input_error(&input) {
            return Err(error);
        }
        value.candidate.query_timestamp_seconds = target_timestamp_seconds;
        value.candidate.query_day_offset = target_day_offset;
        value.candidate.queried_elapsed_days = input.elapsed_days;
        value.candidate.queried_elapsed_seconds = input.elapsed_seconds;
        value.prediction = f64::NAN;
        value.status = CandidateStatus::PendingRefresh;
        value.eligible = false;
        self.push_value(value)
    }

    fn push_value(&mut self, value: LiveCandidateValue) -> PyResult<usize> {
        let slot = self.candidates.len();
        match self.slot_by_card_id.entry(value.candidate.card_id) {
            std::collections::hash_map::Entry::Occupied(_) => {
                return Err(py_value_error(format!(
                    "duplicate live candidate card_id={}",
                    value.candidate.card_id
                )));
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(slot);
            }
        }
        self.candidates.push(value.candidate);
        self.predictions.push(value.prediction);
        self.normal_targets.push(value.normal_target);
        self.intraday_targets.push(value.intraday_target);
        self.tie_breakers.push(value.tie_breaker);
        self.random_keys.push(value.random_key);
        self.statuses.push(value.status);
        self.eligibilities.push(value.eligible);
        self.update_marks.push(0);
        Ok(slot)
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn candidate_count(&self) -> usize {
        self.candidates.len()
    }

    pub(crate) fn tracked_candidate_capacity_bytes(&self) -> usize {
        tracked_candidate_capacity_bytes(
            &self.candidates,
            &self.slot_by_card_id,
            &self.predictions,
            &self.normal_targets,
            &self.intraday_targets,
            &self.tie_breakers,
            &self.random_keys,
            &self.statuses,
            &self.eligibilities,
            &self.ordered_slots,
        )
    }

    pub(crate) fn advance_generation(&mut self) -> u64 {
        self.generation = self.generation.wrapping_add(1);
        self.generation
    }

    pub(crate) fn order(&self) -> LiveOrder {
        self.order
    }

    pub(crate) fn mode(&self) -> LivePredictionMode {
        self.mode
    }

    pub(crate) fn set_mode(&mut self, mode: LivePredictionMode) -> u64 {
        self.mode = mode;
        self.advance_generation()
    }

    pub(crate) fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub(crate) fn refresh_limit(&self) -> usize {
        self.refresh_limit
    }

    pub(crate) fn num_threads(&self) -> Option<usize> {
        self.num_threads
    }

    pub(crate) fn retention_extra(&self) -> f64 {
        self.retention_extra
    }

    pub(crate) fn candidate_slot(&self, card_id: i64) -> PyResult<usize> {
        self.slot_by_card_id.get(&card_id).copied().ok_or_else(|| {
            py_value_error(format!("live candidate card_id={card_id} does not exist"))
        })
    }

    pub(crate) fn begin_answer_undo(&self, slot: usize) -> LiveIndexUndoFrame {
        LiveIndexUndoFrame::capture(self, slot)
    }

    pub(crate) fn apply_answer(
        &mut self,
        slot: usize,
        input: &ReviewInput,
        requeue_after_prediction: bool,
    ) {
        let candidate = &mut self.candidates[slot];
        candidate.review_id = input.review_id;
        candidate.note_id = input.note_id;
        candidate.deck_id = input.deck_id;
        candidate.preset_id = input.preset_id;
        candidate.last_review_timestamp_seconds = Some(
            candidate
                .answer_timestamp_seconds(self.target_timestamp_seconds, input.elapsed_seconds),
        );
        candidate.last_review_day_offset = Some(input.day_offset);
        candidate.has_prior_review = true;

        self.predictions[slot] = f64::NAN;
        self.eligibilities[slot] = false;
        if requeue_after_prediction {
            self.statuses[slot] = if self.statuses[slot].is_excluded() {
                CandidateStatus::ExcludedPendingRefresh
            } else {
                CandidateStatus::PendingRefresh
            };
        } else {
            self.statuses[slot] = CandidateStatus::Removed;
            self.slot_by_card_id.remove(&candidate.card_id);
        }
        self.generation = self.generation.wrapping_add(1);
        self.update_order_after_refresh(&[slot], false);
    }

    pub(crate) fn validate_answer_update(&self, slot: usize, input: &ReviewInput) -> PyResult<()> {
        let timestamp = self.candidates[slot]
            .answer_timestamp_seconds(self.target_timestamp_seconds, input.elapsed_seconds);
        if !timestamp.is_finite() || !input.day_offset.is_finite() {
            return Err(py_value_error(format!(
                "answer for candidate {} cannot derive finite last-review anchors",
                input.card_id
            )));
        }
        Ok(())
    }

    pub(crate) fn restore_undo(&mut self, mut frame: LiveIndexUndoFrame) -> PyResult<()> {
        if frame.session_token != self.token {
            return Err(py_value_error(
                "latest undo frame belongs to a different live prediction session",
            ));
        }
        let next_generation = self.generation.wrapping_add(1);
        if let Some(previous_state) = frame.pre_reconciliation_state.take() {
            self.restore_reconciliation_undo_state(*previous_state);
        }
        let retention_changed =
            self.retention_extra.to_bits() != frame.previous_retention_extra.to_bits();
        self.retention_extra = frame.previous_retention_extra;
        self.target_timestamp_seconds = frame.previous_target_timestamp_seconds;
        self.target_day_offset = frame.previous_target_day_offset;
        let mut restored_slots = std::mem::take(&mut self.commit_slots_scratch);
        restored_slots.clear();
        restored_slots.reserve(frame.candidate_deltas.len() + frame.refresh_deltas.len());
        for (slot, previous) in frame.refresh_deltas {
            if frame.recorded_slots[slot] != UNDO_SLOT_REFRESH_DELTA {
                continue;
            }
            restored_slots.push(slot);
            self.restore_refresh_value(slot, previous);
        }
        for (slot, previous) in frame.candidate_deltas {
            restored_slots.push(slot);
            match previous {
                Some(value) => self.restore_value(slot, value),
                None => {
                    if slot < self.statuses.len() {
                        let card_id = self.candidates[slot].card_id;
                        if self.slot_by_card_id.get(&card_id).copied() == Some(slot) {
                            self.slot_by_card_id.remove(&card_id);
                        }
                        self.statuses[slot] = CandidateStatus::Removed;
                        self.eligibilities[slot] = false;
                    }
                }
            }
        }
        self.generation = next_generation.max(frame.previous_generation.wrapping_add(1));
        if retention_changed {
            self.recompute_all_eligibilities();
            self.rebuild_order();
        } else {
            self.update_order_after_refresh(&restored_slots, false);
        }
        restored_slots.clear();
        self.commit_slots_scratch = restored_slots;
        self.last_membership_slots.clear();
        self.last_transport_slots.clear();
        Ok(())
    }

    pub(crate) fn exclude_card(
        &mut self,
        card_id: i64,
        undo: Option<&mut LiveIndexUndoFrame>,
    ) -> PyResult<u64> {
        let slot = self.candidate_slot(card_id)?;
        if let Some(undo) = undo {
            undo.record_slot(self, slot);
        }
        self.statuses[slot] = self.statuses[slot].excluded();
        self.generation = self.generation.wrapping_add(1);
        Ok(self.generation)
    }

    pub(crate) fn include_card(
        &mut self,
        card_id: i64,
        undo: Option<&mut LiveIndexUndoFrame>,
    ) -> PyResult<u64> {
        let slot = self.candidate_slot(card_id)?;
        if let Some(undo) = undo {
            undo.record_slot(self, slot);
        }
        self.statuses[slot] = self.statuses[slot].included();
        self.generation = self.generation.wrapping_add(1);
        Ok(self.generation)
    }

    pub(crate) fn remove_card(
        &mut self,
        card_id: i64,
        undo: Option<&mut LiveIndexUndoFrame>,
    ) -> PyResult<u64> {
        let slot = self.candidate_slot(card_id)?;
        if let Some(undo) = undo {
            undo.record_slot(self, slot);
        }
        self.statuses[slot] = CandidateStatus::Removed;
        self.eligibilities[slot] = false;
        self.slot_by_card_id.remove(&card_id);
        self.generation = self.generation.wrapping_add(1);
        self.update_order_after_refresh(&[slot], false);
        Ok(self.generation)
    }

    pub(crate) fn upsert_candidates(
        &mut self,
        seeds: &[LiveCandidateSeedNative],
        mut undo: Option<&mut LiveIndexUndoFrame>,
    ) -> PyResult<u64> {
        let mut seen = HashSet::with_capacity(seeds.len());
        for seed in seeds {
            if !seen.insert(seed.row.card_id) {
                return Err(py_value_error(format!(
                    "duplicate live candidate card_id={}",
                    seed.row.card_id
                )));
            }
        }
        let values = seeds
            .iter()
            .map(|seed| {
                candidate_value_from_seed(
                    seed,
                    self.target_timestamp_seconds,
                    self.target_day_offset,
                    self.retention_extra,
                )
            })
            .collect::<PyResult<Vec<_>>>()?;
        for mut value in values {
            let card_id = value.candidate.card_id;
            if let Some(slot) = self.slot_by_card_id.get(&card_id).copied() {
                if let Some(undo) = undo.as_deref_mut() {
                    undo.record_slot(self, slot);
                }
                if self.statuses[slot].is_excluded() {
                    value.status = CandidateStatus::ExcludedPendingRefresh;
                }
                self.restore_value(slot, value);
            } else {
                let slot = self.candidates.len();
                if let Some(undo) = undo.as_deref_mut() {
                    undo.record_new_slot(slot);
                }
                self.slot_by_card_id.insert(card_id, slot);
                self.candidates.push(value.candidate);
                self.predictions.push(value.prediction);
                self.normal_targets.push(value.normal_target);
                self.intraday_targets.push(value.intraday_target);
                self.tie_breakers.push(value.tie_breaker);
                self.random_keys.push(value.random_key);
                self.statuses.push(value.status);
                self.eligibilities.push(value.eligible);
                self.update_marks.push(0);
            }
        }
        self.generation = self.generation.wrapping_add(1);
        self.rebuild_order();
        Ok(self.generation)
    }

    pub(crate) fn set_retention_extra(
        &mut self,
        value: f64,
        _undo: Option<&mut LiveIndexUndoFrame>,
    ) -> PyResult<u64> {
        validate_retention_extra(value)?;
        self.retention_extra = value;
        self.recompute_all_eligibilities();
        self.rebuild_order();
        self.generation = self.generation.wrapping_add(1);
        Ok(self.generation)
    }

    pub(crate) fn candidate(&self, card_id: i64) -> Option<LiveCandidateSnapshot> {
        let slot = self.slot_by_card_id.get(&card_id).copied()?;
        Some(self.snapshot_slot(slot))
    }

    pub(crate) fn snapshot(&self) -> Vec<LiveCandidateSnapshot> {
        self.ordered_slots
            .iter()
            .copied()
            .filter(|&slot| self.statuses[slot] != CandidateStatus::Removed)
            .map(|slot| self.snapshot_slot(slot))
            .collect()
    }

    pub(crate) fn last_refresh_debug(&self) -> (Vec<i64>, Vec<i64>) {
        (
            self.last_membership_slots
                .iter()
                .map(|&slot| self.candidates[slot].card_id)
                .collect(),
            self.last_transport_slots
                .iter()
                .map(|&slot| self.candidates[slot].card_id)
                .collect(),
        )
    }

    pub(crate) fn record_python_result_ns(&mut self, value: u128) {
        if !self.profile.enabled {
            return;
        }
        self.profile.last.python_result_ns += value;
        self.profile.cumulative.python_result_ns += value;
    }

    pub(crate) fn record_reconcile_python_result_ns(&mut self, value: u128) {
        if !self.profile.enabled {
            return;
        }
        self.profile.reconcile_last.python_result_ns += value;
        self.profile.reconcile_cumulative.python_result_ns += value;
    }

    pub(crate) fn record_reconcile_boundary_ns(
        &mut self,
        card_id_parse_ns: u128,
        seed_parse_ns: u128,
        scope_validation_ns: u128,
        native_total_ns: u128,
    ) {
        if !self.profile.enabled {
            return;
        }
        self.profile.reconcile_last.card_id_parse_ns += card_id_parse_ns;
        self.profile.reconcile_last.seed_parse_ns += seed_parse_ns;
        self.profile.reconcile_last.scope_validation_ns += scope_validation_ns;
        self.profile.reconcile_last.native_total_ns += native_total_ns;
        self.profile.reconcile_cumulative.card_id_parse_ns += card_id_parse_ns;
        self.profile.reconcile_cumulative.seed_parse_ns += seed_parse_ns;
        self.profile.reconcile_cumulative.scope_validation_ns += scope_validation_ns;
        self.profile.reconcile_cumulative.native_total_ns += native_total_ns;
    }

    pub(crate) fn record_failed_reconcile(&mut self, profile: LiveRefreshProfile) {
        self.profile.failed_reconcile_calls += 1;
        if !self.profile.enabled {
            return;
        }
        self.profile.failed_reconcile_last = profile;
        self.profile.failed_reconcile_cumulative.add_assign(profile);
    }

    fn value(&self, slot: usize) -> LiveCandidateValue {
        LiveCandidateValue {
            candidate: self.candidates[slot].clone(),
            prediction: self.predictions[slot],
            normal_target: self.normal_targets[slot],
            intraday_target: self.intraday_targets[slot],
            tie_breaker: self.tie_breakers[slot],
            random_key: self.random_keys[slot],
            status: self.statuses[slot],
            eligible: self.eligibilities[slot],
        }
    }

    fn refresh_undo_value(&self, slot: usize) -> LiveRefreshUndoValue {
        let candidate = &self.candidates[slot];
        LiveRefreshUndoValue {
            query_timestamp_seconds: candidate.query_timestamp_seconds,
            query_day_offset: candidate.query_day_offset,
            queried_elapsed_days: candidate.queried_elapsed_days,
            queried_elapsed_seconds: candidate.queried_elapsed_seconds,
            prediction: self.predictions[slot],
            status: self.statuses[slot],
            eligible: self.eligibilities[slot],
        }
    }

    fn restore_refresh_value(&mut self, slot: usize, value: LiveRefreshUndoValue) {
        let candidate = &mut self.candidates[slot];
        candidate.query_timestamp_seconds = value.query_timestamp_seconds;
        candidate.query_day_offset = value.query_day_offset;
        candidate.queried_elapsed_days = value.queried_elapsed_days;
        candidate.queried_elapsed_seconds = value.queried_elapsed_seconds;
        self.predictions[slot] = value.prediction;
        self.statuses[slot] = value.status;
        self.eligibilities[slot] = value.eligible;
    }

    fn restore_value(&mut self, slot: usize, value: LiveCandidateValue) {
        if let Some(current) = self.candidates.get(slot) {
            self.slot_by_card_id.remove(&current.card_id);
        }
        if value.status != CandidateStatus::Removed {
            self.slot_by_card_id.insert(value.candidate.card_id, slot);
        }
        self.candidates[slot] = value.candidate;
        self.predictions[slot] = value.prediction;
        self.normal_targets[slot] = value.normal_target;
        self.intraday_targets[slot] = value.intraday_target;
        self.tie_breakers[slot] = value.tie_breaker;
        self.random_keys[slot] = value.random_key;
        self.statuses[slot] = value.status;
        self.eligibilities[slot] = value.eligible;
    }

    fn snapshot_slot(&self, slot: usize) -> LiveCandidateSnapshot {
        let candidate = &self.candidates[slot];
        LiveCandidateSnapshot {
            card_id: candidate.card_id,
            review_id: candidate.review_id,
            note_id: candidate.note_id,
            deck_id: candidate.deck_id,
            preset_id: candidate.preset_id,
            retrievability: self.predictions[slot],
            target_retrievability: self.normal_targets[slot],
            intraday_target_retrievability: self.intraday_targets[slot],
            applicable_target_retrievability: self.applicable_target(slot),
            tie_breaker: self.tie_breakers[slot],
            random_key: self.random_keys[slot],
            status: self.statuses[slot],
            eligible: self.eligibilities[slot],
            has_prior_review: candidate.has_prior_review,
            last_review_timestamp_seconds: candidate.last_review_timestamp_seconds,
            last_review_day_offset: candidate.last_review_day_offset,
            query_timestamp_seconds: candidate.query_timestamp_seconds,
            query_day_offset: candidate.query_day_offset,
            elapsed_seconds: candidate.queried_elapsed_seconds,
            elapsed_days: candidate.queried_elapsed_days,
        }
    }

    fn select_refresh_membership(&mut self, excluded_card_ids: &[i64]) {
        if excluded_card_ids.is_empty() {
            self.select_refresh_membership_unexcluded();
            return;
        }

        self.selected_scratch.clear();
        self.advance_update_mark();
        let excluded_slots = excluded_card_ids
            .iter()
            .filter_map(|card_id| self.slot_by_card_id.get(card_id).copied())
            .collect::<HashSet<_>>();
        for slot in 0..self.candidates.len() {
            if self.selected_scratch.len() >= self.refresh_limit {
                break;
            }
            if self.statuses[slot] == CandidateStatus::PendingRefresh
                && !excluded_slots.contains(&slot)
            {
                self.selected_scratch.push(slot);
                self.update_marks[slot] = self.update_mark_generation;
            }
        }
        for &slot in &self.ordered_slots {
            if self.selected_scratch.len() >= self.refresh_limit {
                break;
            }
            if !self.statuses[slot].is_included()
                || excluded_slots.contains(&slot)
                || self.update_marks[slot] == self.update_mark_generation
            {
                continue;
            }
            self.selected_scratch.push(slot);
            self.update_marks[slot] = self.update_mark_generation;
        }
        self.transport_slots.clear();
        self.transport_slots.extend(
            (0..self.candidates.len())
                .filter(|&slot| self.update_marks[slot] == self.update_mark_generation),
        );
    }

    fn select_refresh_membership_unexcluded(&mut self) {
        self.selected_scratch.clear();
        self.advance_update_mark();
        for slot in 0..self.candidates.len() {
            if self.selected_scratch.len() >= self.refresh_limit {
                break;
            }
            if self.statuses[slot] == CandidateStatus::PendingRefresh {
                self.selected_scratch.push(slot);
                self.update_marks[slot] = self.update_mark_generation;
            }
        }
        for &slot in &self.ordered_slots {
            if self.selected_scratch.len() >= self.refresh_limit {
                break;
            }
            if !self.statuses[slot].is_included()
                || self.update_marks[slot] == self.update_mark_generation
            {
                continue;
            }
            self.selected_scratch.push(slot);
            self.update_marks[slot] = self.update_mark_generation;
        }
        self.transport_slots.clear();
        self.transport_slots.extend(
            (0..self.candidates.len())
                .filter(|&slot| self.update_marks[slot] == self.update_mark_generation),
        );
    }

    fn commit_refresh_debug(&mut self) {
        std::mem::swap(&mut self.selected_scratch, &mut self.last_membership_slots);
        std::mem::swap(&mut self.transport_slots, &mut self.last_transport_slots);
    }

    fn advance_update_mark(&mut self) {
        self.update_mark_generation = self.update_mark_generation.wrapping_add(1);
        if self.update_mark_generation == 0 {
            self.update_marks.fill(0);
            self.update_mark_generation = 1;
        }
    }

    fn recompute_all_eligibilities(&mut self) {
        for slot in 0..self.candidates.len() {
            self.eligibilities[slot] = self.calculate_eligibility(slot);
        }
    }

    fn calculate_eligibility(&self, slot: usize) -> bool {
        let prediction = self.predictions[slot];
        let target = self.applicable_target(slot);
        self.statuses[slot].prediction_is_usable()
            && prediction.is_finite()
            && target.is_finite()
            && prediction < (target + self.retention_extra).clamp(0.0, 1.0)
    }

    fn applicable_target(&self, slot: usize) -> f64 {
        let elapsed_days = self.candidates[slot].queried_elapsed_days;
        if (0.0..1.0).contains(&elapsed_days) {
            self.intraday_targets[slot]
        } else {
            self.normal_targets[slot]
        }
    }

    fn rebuild_order(&mut self) {
        let mut ordered = std::mem::take(&mut self.ordered_slots);
        ordered.clear();
        ordered.extend(
            self.statuses
                .iter()
                .enumerate()
                .filter_map(|(slot, status)| (*status != CandidateStatus::Removed).then_some(slot)),
        );
        let order = self.order;
        let predictions = &self.predictions;
        let eligibilities = &self.eligibilities;
        let normal_targets = &self.normal_targets;
        let intraday_targets = &self.intraday_targets;
        let tie_breakers = &self.tie_breakers;
        let random_keys = &self.random_keys;
        let candidates = &self.candidates;
        ordered.sort_unstable_by(|&left, &right| {
            compare_slots(
                left,
                right,
                order,
                predictions,
                eligibilities,
                normal_targets,
                intraday_targets,
                tie_breakers,
                random_keys,
                candidates,
            )
        });
        self.ordered_slots = ordered;
    }

    fn update_order_after_refresh(&mut self, updated_slots: &[usize], force_rebuild: bool) -> bool {
        if force_rebuild
            || self.ordered_slots.is_empty()
            || updated_slots.len() * PARTIAL_REFRESH_REBUILD_DENOMINATOR
                >= self.ordered_slots.len() * PARTIAL_REFRESH_REBUILD_NUMERATOR
        {
            self.rebuild_order();
            return true;
        }

        self.merge_updated_order(updated_slots);
        false
    }

    fn merge_updated_order(&mut self, updated_slots: &[usize]) {
        self.advance_update_mark();
        for &slot in updated_slots {
            self.update_marks[slot] = self.update_mark_generation;
        }

        let mut remaining = std::mem::take(&mut self.merge_scratch);
        remaining.clear();
        remaining.extend(
            self.ordered_slots
                .iter()
                .copied()
                .filter(|&slot| self.update_marks[slot] != self.update_mark_generation),
        );
        let order = self.order;
        let predictions = &self.predictions;
        let eligibilities = &self.eligibilities;
        let normal_targets = &self.normal_targets;
        let intraday_targets = &self.intraday_targets;
        let tie_breakers = &self.tie_breakers;
        let random_keys = &self.random_keys;
        let candidates = &self.candidates;
        let mut updated = std::mem::take(&mut self.updated_entry_scratch);
        updated.clear();
        updated.extend(
            updated_slots
                .iter()
                .copied()
                .filter(|&slot| self.statuses[slot] != CandidateStatus::Removed)
                .map(|slot| RankEntry {
                    key: rank_key(
                        slot,
                        order,
                        predictions,
                        eligibilities,
                        normal_targets,
                        intraday_targets,
                        random_keys,
                        candidates,
                    ),
                    slot,
                }),
        );
        updated.sort_unstable_by(|left, right| {
            left.key
                .cmp(&right.key)
                .then_with(|| tie_breakers[left.slot].cmp(&tie_breakers[right.slot]))
                .then_with(|| {
                    candidates[left.slot]
                        .card_id
                        .cmp(&candidates[right.slot].card_id)
                })
        });

        let mut merged = std::mem::take(&mut self.ordered_slots);
        merged.clear();
        let (mut old_index, mut updated_index) = (0, 0);
        while old_index < remaining.len() && updated_index < updated.len() {
            let old_slot = remaining[old_index];
            let entry = &updated[updated_index];
            let updated_slot = entry.slot;
            let comparison = compare_slot_to_rank_entry(
                old_slot,
                entry,
                order,
                predictions,
                eligibilities,
                normal_targets,
                intraday_targets,
                tie_breakers,
                random_keys,
                candidates,
            );
            if comparison != Ordering::Greater {
                merged.push(old_slot);
                old_index += 1;
            } else {
                merged.push(updated_slot);
                updated_index += 1;
            }
        }
        merged.extend_from_slice(&remaining[old_index..]);
        merged.extend(updated[updated_index..].iter().map(|entry| entry.slot));
        self.ordered_slots = merged;
        remaining.clear();
        updated.clear();
        self.merge_scratch = remaining;
        self.updated_entry_scratch = updated;
    }

    #[cfg(test)]
    fn final_selection(
        &self,
        select_limit: usize,
        excluded_card_ids: &HashSet<i64>,
    ) -> (usize, usize, Vec<LiveSelectionValue>, Option<f64>) {
        self.final_selection_by(select_limit, |_, card_id| {
            excluded_card_ids.contains(&card_id)
        })
    }

    #[cfg(test)]
    fn final_selection_small_exclusion(
        &self,
        select_limit: usize,
        excluded_card_ids: &[i64],
    ) -> (usize, usize, Vec<LiveSelectionValue>, Option<f64>) {
        self.final_selection_by(select_limit, |_, card_id| {
            excluded_card_ids.contains(&card_id)
        })
    }

    fn final_selection_marked_exclusion(
        &mut self,
        select_limit: usize,
        excluded_card_ids: &[i64],
    ) -> (usize, usize, Vec<LiveSelectionValue>, Option<f64>) {
        self.advance_update_mark();
        let mark = self.update_mark_generation;
        for card_id in excluded_card_ids {
            if let Some(&slot) = self.slot_by_card_id.get(card_id) {
                self.update_marks[slot] = mark;
            }
        }
        let update_marks = &self.update_marks;
        self.final_selection_by(select_limit, |slot, _| update_marks[slot] == mark)
    }

    fn final_selection_no_exclusion(
        &self,
        select_limit: usize,
    ) -> (usize, usize, Vec<LiveSelectionValue>, Option<f64>) {
        self.final_selection_by(select_limit, |_, _| false)
    }

    pub(crate) fn initial_output(&self, select_limit: usize) -> LiveRefreshOutput {
        let (eligible_count, active_count, selected, next_retention_extra) =
            self.final_selection_no_exclusion(select_limit);
        LiveRefreshOutput {
            generation: self.generation,
            refreshed_count: self.candidates.len(),
            eligible_count,
            active_count,
            selected,
            next_retention_extra,
        }
    }

    pub(crate) fn current_selection_output(
        &mut self,
        select_limit: usize,
        excluded_card_ids: &[i64],
    ) -> LiveRefreshOutput {
        let (eligible_count, active_count, selected, next_retention_extra) =
            if excluded_card_ids.is_empty() {
                self.final_selection_no_exclusion(select_limit)
            } else {
                self.final_selection_marked_exclusion(select_limit, excluded_card_ids)
            };
        LiveRefreshOutput {
            generation: self.generation,
            refreshed_count: 0,
            eligible_count,
            active_count,
            selected,
            next_retention_extra,
        }
    }

    fn final_selection_by(
        &self,
        select_limit: usize,
        mut is_excluded: impl FnMut(usize, i64) -> bool,
    ) -> (usize, usize, Vec<LiveSelectionValue>, Option<f64>) {
        let mut active_count = 0usize;
        let mut eligible_count = 0usize;
        let mut selected = Vec::with_capacity(select_limit.min(16));
        let mut next_retention_extra: Option<f64> = None;
        for &slot in &self.ordered_slots {
            let status = self.statuses[slot];
            let card_id = self.candidates[slot].card_id;
            if !status.is_included() || is_excluded(slot, card_id) {
                continue;
            }
            active_count += 1;
            if self.eligibilities[slot] {
                eligible_count += 1;
                if selected.len() < select_limit {
                    selected.push(LiveSelectionValue {
                        card_id,
                        retrievability: self.predictions[slot],
                        target_retrievability: self.applicable_target(slot),
                    });
                }
            } else if let Some(boundary) = self.next_boundary(slot) {
                next_retention_extra = Some(match next_retention_extra {
                    Some(current) => current.min(boundary),
                    None => boundary,
                });
            }
        }
        (eligible_count, active_count, selected, next_retention_extra)
    }

    fn next_boundary(&self, slot: usize) -> Option<f64> {
        if !self.statuses[slot].prediction_is_usable() {
            return None;
        }
        let prediction = self.predictions[slot];
        let target = self.applicable_target(slot);
        if !prediction.is_finite() || !target.is_finite() || prediction >= 1.0 {
            return None;
        }
        let mut required = next_up(prediction) - target;
        if required <= prediction - target {
            required = next_up(required);
        }
        while prediction >= (target + required).clamp(0.0, 1.0) {
            required = next_up(required);
        }
        (required > self.retention_extra).then_some(required)
    }
}

#[allow(clippy::too_many_arguments)]
fn tracked_candidate_capacity_bytes(
    candidates: &Vec<LiveCandidate>,
    slot_by_card_id: &HashMap<i64, usize>,
    predictions: &Vec<f64>,
    normal_targets: &Vec<f64>,
    intraday_targets: &Vec<f64>,
    tie_breakers: &Vec<u64>,
    random_keys: &Vec<u64>,
    statuses: &Vec<CandidateStatus>,
    eligibilities: &Vec<bool>,
    ordered_slots: &Vec<usize>,
) -> usize {
    let vector_bytes = [
        candidates
            .capacity()
            .saturating_mul(std::mem::size_of::<LiveCandidate>()),
        predictions
            .capacity()
            .saturating_mul(std::mem::size_of::<f64>()),
        normal_targets
            .capacity()
            .saturating_mul(std::mem::size_of::<f64>()),
        intraday_targets
            .capacity()
            .saturating_mul(std::mem::size_of::<f64>()),
        tie_breakers
            .capacity()
            .saturating_mul(std::mem::size_of::<u64>()),
        random_keys
            .capacity()
            .saturating_mul(std::mem::size_of::<u64>()),
        statuses
            .capacity()
            .saturating_mul(std::mem::size_of::<CandidateStatus>()),
        eligibilities
            .capacity()
            .saturating_mul(std::mem::size_of::<bool>()),
        ordered_slots
            .capacity()
            .saturating_mul(std::mem::size_of::<usize>()),
    ]
    .into_iter()
    .fold(0usize, usize::saturating_add);
    let slot_map_entry_bytes = slot_by_card_id
        .capacity()
        .saturating_mul(std::mem::size_of::<(i64, usize)>());
    vector_bytes.saturating_add(slot_map_entry_bytes)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn refresh_live_session(
    state: &mut LivePredictionSessionState,
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    target_timestamp_seconds: f64,
    target_day_offset: f64,
    select_limit: usize,
    excluded_card_ids: &[i64],
    excluded_refresh_card_ids: &[i64],
    retention_extra: f64,
    undo: Option<&mut LiveIndexUndoFrame>,
) -> PyResult<LiveRefreshOutput> {
    validate_target_time(target_timestamp_seconds, target_day_offset)?;
    validate_retention_extra(retention_extra)?;

    let membership_start = Instant::now();
    state.select_refresh_membership(excluded_refresh_card_ids);
    let membership_ns = membership_start.elapsed().as_nanos();

    let input_start = Instant::now();
    let mut inputs = std::mem::take(&mut state.input_scratch);
    inputs.clear();
    inputs.extend(state.transport_slots.iter().map(|&slot| {
        state.candidates[slot].prediction_input(target_timestamp_seconds, target_day_offset)
    }));
    if let Some(error) = inputs.iter().find_map(live_prediction_input_error) {
        state.input_scratch = inputs;
        return Err(error);
    }
    // Identity-changing operations validate the selective checkpoint scope
    // before committing to the session. Refresh only adjusts query time, so
    // repeating those HashSet lookups here would be redundant hot-path work.
    let input_ns = input_start.elapsed().as_nanos();

    let prediction_start = Instant::now();
    let prediction_result = match state.mode {
        LivePredictionMode::Oracle | LivePredictionMode::Fast => predict_review_inputs_cpu(
            rnn,
            deterministic,
            &inputs,
            state.batch_size,
            state.num_threads,
            state.mode == LivePredictionMode::Fast,
            None,
        ),
        LivePredictionMode::Gpu => {
            synchronize_gpu_process_state_with_state(rnn).map_err(py_gpu_error)?;
            rnn.release_gpu_process_cache();
            predict_review_inputs_gpu(
                rnn,
                deterministic,
                None,
                &inputs,
                state.batch_size,
                state.num_threads,
                0,
            )
        }
    };
    let predictions = match prediction_result {
        Ok(predictions) => predictions,
        Err(error) => {
            state.input_scratch = inputs;
            state.profile.failed_refresh_calls += 1;
            return Err(error);
        }
    };
    let prediction_ns = prediction_start.elapsed().as_nanos();
    if predictions.len() != state.transport_slots.len() {
        state.input_scratch = inputs;
        state.profile.failed_refresh_calls += 1;
        return Err(py_value_error(format!(
            "live refresh returned {} predictions for {} candidates",
            predictions.len(),
            state.transport_slots.len()
        )));
    }

    let application_start = Instant::now();
    let retention_changed = state.retention_extra.to_bits() != retention_extra.to_bits();
    if let Some(undo) = undo {
        for &slot in &state.transport_slots {
            undo.record_refresh_slot(state, slot);
        }
    }
    state.retention_extra = retention_extra;
    state.target_timestamp_seconds = target_timestamp_seconds;
    state.target_day_offset = target_day_offset;
    for ((&slot, &prediction), input) in state.transport_slots.iter().zip(&predictions).zip(&inputs)
    {
        let candidate = &mut state.candidates[slot];
        candidate.query_timestamp_seconds = target_timestamp_seconds;
        candidate.query_day_offset = target_day_offset;
        candidate.queried_elapsed_days = input.elapsed_days;
        candidate.queried_elapsed_seconds = input.elapsed_seconds;
        state.predictions[slot] = prediction;
        state.statuses[slot] = state.statuses[slot].refreshed();
        state.eligibilities[slot] = state.calculate_eligibility(slot);
    }
    inputs.clear();
    state.input_scratch = inputs;
    if retention_changed {
        state.recompute_all_eligibilities();
    }
    let mut updated_slots = std::mem::take(&mut state.commit_slots_scratch);
    updated_slots.clear();
    updated_slots.extend_from_slice(&state.transport_slots);
    state.commit_refresh_debug();
    let application_ns = application_start.elapsed().as_nanos();

    let ordering_start = Instant::now();
    let full_rebuild = state.update_order_after_refresh(&updated_slots, retention_changed);
    updated_slots.clear();
    state.commit_slots_scratch = updated_slots;
    let ordering_ns = ordering_start.elapsed().as_nanos();

    let final_start = Instant::now();
    let (eligible_count, active_count, selected, next_retention_extra) =
        if excluded_card_ids.is_empty() {
            state.final_selection_no_exclusion(select_limit)
        } else {
            state.final_selection_marked_exclusion(select_limit, excluded_card_ids)
        };
    let final_selection_ns = final_start.elapsed().as_nanos();
    state.generation = state.generation.wrapping_add(1);

    let last = LiveRefreshProfile {
        membership_selection_ns: membership_ns,
        input_construction_ns: input_ns,
        prediction_ns,
        application_ns,
        ordering_ns,
        final_selection_ns,
        python_result_ns: 0,
        refreshed_count: predictions.len() as u64,
        selected_count: selected.len() as u64,
        partial_merge_count: u64::from(!full_rebuild),
        full_rebuild_count: u64::from(full_rebuild),
        ..LiveRefreshProfile::default()
    };
    state.profile.refresh_calls += 1;
    if state.profile.enabled {
        state.profile.last = last;
        state.profile.cumulative.add_assign(last);
    }

    Ok(LiveRefreshOutput {
        generation: state.generation,
        refreshed_count: predictions.len(),
        eligible_count,
        active_count,
        selected,
        next_retention_extra,
    })
}

pub(crate) fn initialize_live_session_predictions(
    state: &mut LivePredictionSessionState,
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
) -> PyResult<LiveRefreshProfile> {
    let input_start = Instant::now();
    let inputs = state
        .candidates
        .iter()
        .map(LiveCandidate::initial_prediction_input)
        .collect::<Vec<_>>();
    if let Some(error) = inputs.iter().find_map(live_prediction_input_error) {
        return Err(error);
    }
    let input_construction_ns = input_start.elapsed().as_nanos();
    // Seeds have already passed selective-scope validation in the binding.
    let prediction_start = Instant::now();
    let predictions = match state.mode {
        LivePredictionMode::Oracle | LivePredictionMode::Fast => predict_review_inputs_cpu(
            rnn,
            deterministic,
            &inputs,
            state.batch_size,
            state.num_threads,
            state.mode == LivePredictionMode::Fast,
            None,
        )?,
        LivePredictionMode::Gpu => {
            synchronize_gpu_process_state_with_state(rnn).map_err(py_gpu_error)?;
            rnn.release_gpu_process_cache();
            predict_review_inputs_gpu(
                rnn,
                deterministic,
                None,
                &inputs,
                state.batch_size,
                state.num_threads,
                0,
            )?
        }
    };
    let prediction_ns = prediction_start.elapsed().as_nanos();
    if predictions.len() != state.candidates.len() {
        return Err(py_value_error(format!(
            "live session initialization returned {} predictions for {} candidates",
            predictions.len(),
            state.candidates.len()
        )));
    }
    let application_start = Instant::now();
    for (slot, prediction) in predictions.into_iter().enumerate() {
        state.predictions[slot] = prediction;
        state.statuses[slot] = CandidateStatus::Active;
    }
    state.recompute_all_eligibilities();
    let application_ns = application_start.elapsed().as_nanos();
    let ordering_start = Instant::now();
    state.rebuild_order();
    let ordering_ns = ordering_start.elapsed().as_nanos();
    Ok(LiveRefreshProfile {
        input_construction_ns,
        prediction_ns,
        application_ns,
        ordering_ns,
        refreshed_count: state.candidates.len() as u64,
        full_rebuild_count: 1,
        ..LiveRefreshProfile::default()
    })
}

#[allow(clippy::too_many_arguments)]
fn predicted_replacement_from_state(
    state: &LivePredictionSessionState,
    seeds: &[LiveCandidateSeedNative],
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    target_timestamp_seconds: f64,
    target_day_offset: f64,
    retention_extra: f64,
) -> PyResult<(LivePredictionSessionState, LiveRefreshProfile)> {
    let construction_start = Instant::now();
    let mut replacement = LivePredictionSessionState::new_unranked(
        state.token,
        seeds,
        target_timestamp_seconds,
        target_day_offset,
        state.order,
        state.mode,
        state.batch_size,
        state.refresh_limit,
        state.num_threads,
        state.profile.enabled,
    )?;
    replacement.retention_extra = retention_extra;
    let candidate_construction_ns = construction_start.elapsed().as_nanos();
    let mut profile = initialize_live_session_predictions(&mut replacement, rnn, deterministic)?;
    profile.input_construction_ns += candidate_construction_ns;
    Ok((replacement, profile))
}

#[allow(clippy::too_many_arguments)]
fn predicted_membership_replacement_from_state(
    state: &LivePredictionSessionState,
    card_ids: &[i64],
    changed_candidates: &[LiveCandidateSeedNative],
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    target_timestamp_seconds: f64,
    target_day_offset: f64,
    retention_extra: f64,
) -> PyResult<(LivePredictionSessionState, LiveRefreshProfile)> {
    let construction_start = Instant::now();
    let mut replacement = LivePredictionSessionState::new_empty(
        state.token,
        card_ids.len(),
        target_timestamp_seconds,
        target_day_offset,
        state.order,
        state.mode,
        state.batch_size,
        state.refresh_limit,
        state.num_threads,
        state.profile.enabled,
    )?;
    replacement.retention_extra = retention_extra;

    let mut changed_by_card_id = HashMap::with_capacity(changed_candidates.len());
    for (index, seed) in changed_candidates.iter().enumerate() {
        if changed_by_card_id
            .insert(seed.row.card_id, (index, seed))
            .is_some()
        {
            return Err(py_value_error(format!(
                "duplicate changed candidate card_id={}",
                seed.row.card_id
            )));
        }
    }
    for (index, &card_id) in card_ids.iter().enumerate() {
        if let Some((_changed_index, seed)) = changed_by_card_id.remove(&card_id) {
            replacement.push_seed(seed, target_timestamp_seconds, target_day_offset)?;
            continue;
        }
        let source_slot = state
            .slot_by_card_id
            .get(&card_id)
            .copied()
            .ok_or_else(|| {
                if replacement.slot_by_card_id.contains_key(&card_id) {
                    return py_value_error(format!("duplicate live candidate card_id={card_id}"));
                }
                py_value_error(format!(
                    "card_ids[{index}]={card_id} is not an active live candidate; provide a \
                 changed_candidates seed."
                ))
            })?;
        replacement.push_reused_candidate(
            state,
            source_slot,
            target_timestamp_seconds,
            target_day_offset,
        )?;
    }
    if let Some((card_id, (index, _seed))) = changed_by_card_id
        .into_iter()
        .min_by_key(|(_card_id, (index, _seed))| *index)
    {
        return Err(py_value_error(format!(
            "changed_candidates[{index}].row.card_id={card_id} is not present in card_ids."
        )));
    }

    let candidate_construction_ns = construction_start.elapsed().as_nanos();
    let mut profile = initialize_live_session_predictions(&mut replacement, rnn, deterministic)?;
    profile.input_construction_ns += candidate_construction_ns;
    Ok((replacement, profile))
}

pub(crate) fn replace_live_session_candidates(
    state: &mut LivePredictionSessionState,
    seeds: &[LiveCandidateSeedNative],
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
) -> PyResult<u64> {
    let (mut replacement, _profile) = predicted_replacement_from_state(
        state,
        seeds,
        rnn,
        deterministic,
        state.target_timestamp_seconds,
        state.target_day_offset,
        state.retention_extra,
    )?;
    replacement.generation = state.generation.wrapping_add(1);
    replacement.profile = std::mem::take(&mut state.profile);
    let generation = replacement.generation;
    *state = replacement;
    Ok(generation)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn reconcile_live_session_candidates(
    state: &mut LivePredictionSessionState,
    seeds: &[LiveCandidateSeedNative],
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    target_timestamp_seconds: f64,
    target_day_offset: f64,
    select_limit: usize,
    excluded_card_ids: &[i64],
    retention_extra: f64,
    undo: Option<&mut LiveIndexUndoFrame>,
) -> PyResult<LiveRefreshOutput> {
    // Construct and predict the replacement completely before touching the
    // active session. A validation or inference failure therefore leaves the
    // candidate universe, generation, and paired undo frames unchanged.
    let replacement_result = (|| {
        validate_target_time(target_timestamp_seconds, target_day_offset)?;
        validate_retention_extra(retention_extra)?;
        predicted_replacement_from_state(
            state,
            seeds,
            rnn,
            deterministic,
            target_timestamp_seconds,
            target_day_offset,
            retention_extra,
        )
    })();
    commit_reconciled_live_session_replacement(
        state,
        replacement_result,
        select_limit,
        excluded_card_ids,
        undo,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn reconcile_live_session_membership(
    state: &mut LivePredictionSessionState,
    card_ids: &[i64],
    changed_candidates: &[LiveCandidateSeedNative],
    rnn: &mut NativeRnn,
    deterministic: &mut FeatureState,
    target_timestamp_seconds: f64,
    target_day_offset: f64,
    select_limit: usize,
    excluded_card_ids: &[i64],
    retention_extra: f64,
    undo: Option<&mut LiveIndexUndoFrame>,
) -> PyResult<LiveRefreshOutput> {
    // Existing candidates retain their exact last-review anchors. Only
    // explicitly changed/new candidates are rebuilt from Python-supplied rows.
    let replacement_result = (|| {
        validate_target_time(target_timestamp_seconds, target_day_offset)?;
        validate_retention_extra(retention_extra)?;
        predicted_membership_replacement_from_state(
            state,
            card_ids,
            changed_candidates,
            rnn,
            deterministic,
            target_timestamp_seconds,
            target_day_offset,
            retention_extra,
        )
    })();
    commit_reconciled_live_session_replacement(
        state,
        replacement_result,
        select_limit,
        excluded_card_ids,
        undo,
    )
}

fn commit_reconciled_live_session_replacement(
    state: &mut LivePredictionSessionState,
    replacement_result: PyResult<(LivePredictionSessionState, LiveRefreshProfile)>,
    select_limit: usize,
    excluded_card_ids: &[i64],
    undo: Option<&mut LiveIndexUndoFrame>,
) -> PyResult<LiveRefreshOutput> {
    let (mut replacement, mut last) = replacement_result?;
    let commit_start = Instant::now();
    replacement.generation = state.generation.wrapping_add(1);
    replacement.profile = state.profile.clone();
    replacement.last_membership_slots = (0..replacement.candidates.len()).collect();
    replacement.last_transport_slots = (0..replacement.candidates.len()).collect();

    let previous_state = std::mem::replace(state, replacement);
    if let Some(undo) = undo {
        undo.record_reconciliation_state(previous_state);
    }
    last.commit_snapshot_ns = commit_start.elapsed().as_nanos();

    let final_start = Instant::now();
    let mut output = state.current_selection_output(select_limit, excluded_card_ids);
    last.final_selection_ns = final_start.elapsed().as_nanos();
    last.selected_count = output.selected.len() as u64;
    output.refreshed_count = state.candidates.len();
    state.profile.reconcile_calls += 1;
    if state.profile.enabled {
        state.profile.reconcile_last = last;
        state.profile.reconcile_cumulative.add_assign(last);
    }
    Ok(output)
}

fn candidate_value_from_seed(
    seed: &LiveCandidateSeedNative,
    target_timestamp_seconds: f64,
    target_day_offset: f64,
    _retention_extra: f64,
) -> PyResult<LiveCandidateValue> {
    let candidate = LiveCandidate::from_seed(seed, target_timestamp_seconds, target_day_offset)?;
    let prediction = f64::NAN;
    let eligible = false;
    Ok(LiveCandidateValue {
        candidate,
        prediction,
        normal_target: seed.target_retrievability,
        intraday_target: seed.intraday_target_retrievability,
        tie_breaker: seed.tie_breaker,
        random_key: seed.random_key,
        status: CandidateStatus::PendingRefresh,
        eligible,
    })
}

fn compare_slots(
    left: usize,
    right: usize,
    order: LiveOrder,
    predictions: &[f64],
    eligibilities: &[bool],
    normal_targets: &[f64],
    intraday_targets: &[f64],
    tie_breakers: &[u64],
    random_keys: &[u64],
    candidates: &[LiveCandidate],
) -> Ordering {
    rank_key(
        left,
        order,
        predictions,
        eligibilities,
        normal_targets,
        intraday_targets,
        random_keys,
        candidates,
    )
    .cmp(&rank_key(
        right,
        order,
        predictions,
        eligibilities,
        normal_targets,
        intraday_targets,
        random_keys,
        candidates,
    ))
    .then_with(|| tie_breakers[left].cmp(&tie_breakers[right]))
    .then_with(|| candidates[left].card_id.cmp(&candidates[right].card_id))
}

#[allow(clippy::too_many_arguments)]
fn rank_key(
    slot: usize,
    order: LiveOrder,
    predictions: &[f64],
    eligibilities: &[bool],
    normal_targets: &[f64],
    intraday_targets: &[f64],
    random_keys: &[u64],
    candidates: &[LiveCandidate],
) -> RankKey {
    // Random ordering is caller-seeded and intentionally independent of the
    // prediction value after the eligibility partition. The other orders put
    // unusable numeric values last, then encode their finite f64 sort value as
    // one comparable integer. Normalize signed zero because IEEE partial_cmp
    // treats -0.0 and 0.0 as equal.
    if order == LiveOrder::Random {
        return RankKey {
            class: u8::from(!eligibilities[slot]),
            order_key: random_keys[slot],
        };
    }

    let prediction = predictions[slot];
    let value = match order {
        LiveOrder::RetrievabilityAscending | LiveOrder::RetrievabilityDescending => {
            prediction.is_finite().then_some(prediction)
        }
        LiveOrder::RelativeOverdueness => {
            let target =
                applicable_target_from_parts(slot, candidates, normal_targets, intraday_targets);
            if prediction.is_finite() && target.is_finite() {
                let score = prediction.max(0.0001) / target.max(0.0001);
                score.is_finite().then_some(score)
            } else {
                None
            }
        }
        LiveOrder::Random => unreachable!("random order returned above"),
    };
    let class = (u8::from(!eligibilities[slot]) << 1) | u8::from(value.is_none());
    let order_key = if let Some(value) = value {
        let ascending_key = ordered_f64_key(value);
        if order == LiveOrder::RetrievabilityDescending {
            !ascending_key
        } else {
            ascending_key
        }
    } else {
        0
    };
    RankKey { class, order_key }
}

fn applicable_target_from_parts(
    slot: usize,
    candidates: &[LiveCandidate],
    normal_targets: &[f64],
    intraday_targets: &[f64],
) -> f64 {
    if (0.0..1.0).contains(&candidates[slot].queried_elapsed_days) {
        intraday_targets[slot]
    } else {
        normal_targets[slot]
    }
}

fn ordered_f64_key(value: f64) -> u64 {
    debug_assert!(value.is_finite());
    let normalized = if value == 0.0 { 0.0 } else { value };
    let bits = normalized.to_bits();
    if bits >> 63 == 0 {
        bits ^ (1 << 63)
    } else {
        !bits
    }
}

fn compare_slot_to_rank_entry(
    left_slot: usize,
    right: &RankEntry,
    order: LiveOrder,
    predictions: &[f64],
    eligibilities: &[bool],
    normal_targets: &[f64],
    intraday_targets: &[f64],
    tie_breakers: &[u64],
    random_keys: &[u64],
    candidates: &[LiveCandidate],
) -> Ordering {
    compare_key_to_rank_entry(
        rank_key(
            left_slot,
            order,
            predictions,
            eligibilities,
            normal_targets,
            intraday_targets,
            random_keys,
            candidates,
        ),
        left_slot,
        right,
        tie_breakers,
        candidates,
    )
}

fn compare_key_to_rank_entry(
    left_key: RankKey,
    left_slot: usize,
    right: &RankEntry,
    tie_breakers: &[u64],
    candidates: &[LiveCandidate],
) -> Ordering {
    left_key
        .cmp(&right.key)
        .then_with(|| tie_breakers[left_slot].cmp(&tie_breakers[right.slot]))
        .then_with(|| {
            candidates[left_slot]
                .card_id
                .cmp(&candidates[right.slot].card_id)
        })
}

fn validate_target_time(timestamp_seconds: f64, day_offset: f64) -> PyResult<()> {
    if !timestamp_seconds.is_finite() {
        return Err(py_value_error("target_timestamp_seconds must be finite"));
    }
    if !day_offset.is_finite() {
        return Err(py_value_error("target_day_offset must be finite"));
    }
    Ok(())
}

fn validate_retention_extra(value: f64) -> PyResult<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(py_value_error("retention_extra must be finite"))
    }
}

fn live_prediction_input_error(input: &ReviewInput) -> Option<PyErr> {
    if input.day_offset.is_finite()
        && input.elapsed_days.is_finite()
        && input.elapsed_seconds.is_finite()
    {
        return None;
    }
    Some(py_value_error(format!(
        "candidate {} produced non-finite time-adjusted prediction values",
        input.card_id
    )))
}

fn next_up(value: f64) -> f64 {
    if value.is_nan() || value == f64::INFINITY {
        return value;
    }
    if value == -0.0 {
        return f64::from_bits(1);
    }
    let bits = value.to_bits();
    if value >= 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seed(
        card_id: i64,
        elapsed_days: f64,
        normal: f64,
        intraday: f64,
        tie: u64,
    ) -> LiveCandidateSeedNative {
        LiveCandidateSeedNative {
            row: ReviewInput {
                review_id: card_id,
                card_id,
                note_id: MaybeId::Present(card_id + 100),
                deck_id: MaybeId::Present(1),
                preset_id: MaybeId::Present(2),
                day_offset: 10.0,
                elapsed_days,
                elapsed_seconds: if elapsed_days == -1.0 {
                    -1.0
                } else {
                    elapsed_days * 86_400.0
                },
                rating: None,
                duration: None,
                state: None,
            },
            target_retrievability: normal,
            intraday_target_retrievability: intraday,
            tie_breaker: tie,
            random_key: card_id as u64,
        }
    }

    fn state(
        seeds: &[LiveCandidateSeedNative],
        order: LiveOrder,
        refresh_limit: usize,
    ) -> LivePredictionSessionState {
        LivePredictionSessionState::new_unranked(
            1,
            seeds,
            1_000_000.0,
            10.0,
            order,
            LivePredictionMode::Oracle,
            8,
            refresh_limit,
            Some(1),
            false,
        )
        .unwrap()
    }

    fn activate(state: &mut LivePredictionSessionState, predictions: &[f64]) {
        for (slot, &prediction) in predictions.iter().enumerate() {
            state.predictions[slot] = prediction;
            state.statuses[slot] = CandidateStatus::Active;
        }
        state.recompute_all_eligibilities();
        state.rebuild_order();
    }

    fn ordered_cards(state: &LivePredictionSessionState) -> Vec<i64> {
        state
            .ordered_slots
            .iter()
            .map(|&slot| state.candidates[slot].card_id)
            .collect()
    }

    #[test]
    fn exact_ordering_keeps_non_finite_values_last_and_ties_deterministic() {
        let seeds = [
            seed(10, 2.0, 0.9, 0.9, 0),
            seed(11, 2.0, 0.9, 0.9, 5),
            seed(12, 2.0, 0.9, 0.9, 0),
            seed(13, 2.0, 0.9, 0.9, 0),
            seed(9, 2.0, 0.9, 0.9, 5),
        ];
        let predictions = [0.4, 0.2, f64::NAN, f64::INFINITY, 0.2];

        let mut ascending = state(&seeds, LiveOrder::RetrievabilityAscending, 5);
        activate(&mut ascending, &predictions);
        assert_eq!(ordered_cards(&ascending), vec![9, 11, 10, 12, 13]);

        let mut descending = state(&seeds, LiveOrder::RetrievabilityDescending, 5);
        activate(&mut descending, &predictions);
        assert_eq!(ordered_cards(&descending), vec![10, 9, 11, 12, 13]);
    }

    #[test]
    fn relative_overdueness_uses_the_applicable_target() {
        let seeds = [
            seed(1, 2.0, 0.9, 0.5, 30),
            seed(2, 2.0, 0.8, 0.5, 20),
            seed(3, 0.5, 0.9, 0.7, 10),
        ];
        let mut state = state(&seeds, LiveOrder::RelativeOverdueness, 3);
        activate(&mut state, &[0.81, 0.68, 0.63]);

        // 0.68 / 0.80 = 0.85, while the interday and intraday candidates
        // both score 0.90 and therefore fall back to the caller tie breaker.
        assert_eq!(ordered_cards(&state), vec![2, 3, 1]);
    }

    #[test]
    fn random_order_uses_caller_keys_after_eligibility() {
        let mut seeds = [
            seed(1, 2.0, 0.9, 0.9, 3),
            seed(2, 2.0, 0.9, 0.9, 2),
            seed(3, 2.0, 0.9, 0.9, 1),
        ];
        seeds[0].random_key = 30;
        seeds[1].random_key = 10;
        seeds[2].random_key = 0;
        let mut state = state(&seeds, LiveOrder::Random, 3);
        activate(&mut state, &[0.3, 0.2, 0.95]);

        // Card 3 has the smallest random key, but remains after both eligible
        // cards. Prediction changes do not reshuffle the stable caller keys.
        assert_eq!(ordered_cards(&state), vec![2, 1, 3]);
        activate(&mut state, &[0.1, 0.8, 0.95]);
        assert_eq!(ordered_cards(&state), vec![2, 1, 3]);
    }

    #[test]
    fn compact_rank_entry_matches_canonical_comparator_on_edge_values() {
        let predictions = [
            f64::NEG_INFINITY,
            -f64::MAX,
            -1.0,
            -0.0,
            0.0,
            f64::MIN_POSITIVE,
            0.4,
            0.4,
            1.0,
            f64::MAX,
            f64::INFINITY,
            f64::NAN,
        ];
        let seeds = (0..predictions.len())
            .map(|index| seed(100 + index as i64, 2.0, 0.5, 0.5, (index % 3) as u64))
            .collect::<Vec<_>>();

        for order in [
            LiveOrder::RetrievabilityAscending,
            LiveOrder::RetrievabilityDescending,
            LiveOrder::RelativeOverdueness,
            LiveOrder::Random,
        ] {
            let mut state = state(&seeds, order, predictions.len());
            activate(&mut state, &predictions);
            for left in 0..predictions.len() {
                for right in 0..predictions.len() {
                    let canonical = compare_slots(
                        left,
                        right,
                        order,
                        &state.predictions,
                        &state.eligibilities,
                        &state.normal_targets,
                        &state.intraday_targets,
                        &state.tie_breakers,
                        &state.random_keys,
                        &state.candidates,
                    );
                    let right_entry = RankEntry {
                        key: rank_key(
                            right,
                            order,
                            &state.predictions,
                            &state.eligibilities,
                            &state.normal_targets,
                            &state.intraday_targets,
                            &state.random_keys,
                            &state.candidates,
                        ),
                        slot: right,
                    };
                    assert_eq!(
                        compare_slot_to_rank_entry(
                            left,
                            &right_entry,
                            order,
                            &state.predictions,
                            &state.eligibilities,
                            &state.normal_targets,
                            &state.intraday_targets,
                            &state.tie_breakers,
                            &state.random_keys,
                            &state.candidates,
                        ),
                        canonical,
                        "left={left}, right={right}, order={order:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn eligibility_is_strict_clamped_and_uses_intraday_target() {
        let seeds = [
            seed(1, 2.0, 0.9, 0.1, 0),
            seed(2, 2.0, 2.0, 0.1, 0),
            seed(3, 0.5, 0.99, 0.4, 0),
            seed(4, 2.0, f64::NAN, 0.9, 0),
        ];
        let mut state = state(&seeds, LiveOrder::RetrievabilityAscending, 4);
        activate(&mut state, &[0.9, 0.99, 0.5, 0.1]);
        assert_eq!(state.eligibilities, vec![false, true, false, false]);
        state.retention_extra = 0.1;
        state.recompute_all_eligibilities();
        assert_eq!(state.eligibilities, vec![true, true, false, false]);
    }

    #[test]
    fn partial_merge_matches_an_independent_full_sort() {
        let seeds = (0..100)
            .map(|card_id| seed(card_id, 2.0, 0.8, 0.8, (card_id % 7) as u64))
            .collect::<Vec<_>>();
        let mut state = state(&seeds, LiveOrder::RetrievabilityAscending, 16);
        let initial = (0..100)
            .map(|index| ((index * 37) % 101) as f64 / 101.0)
            .collect::<Vec<_>>();
        activate(&mut state, &initial);
        let updated = [3usize, 9, 17, 22, 41, 58, 63, 77, 88, 96];
        for (offset, &slot) in updated.iter().enumerate() {
            state.predictions[slot] = 0.001 * offset as f64;
            state.eligibilities[slot] = state.calculate_eligibility(slot);
        }
        state.update_order_after_refresh(&updated, false);

        let mut expected = (0..100).collect::<Vec<_>>();
        expected.sort_by(|&left, &right| {
            compare_slots(
                left,
                right,
                state.order,
                &state.predictions,
                &state.eligibilities,
                &state.normal_targets,
                &state.intraday_targets,
                &state.tie_breakers,
                &state.random_keys,
                &state.candidates,
            )
        });
        assert_eq!(state.ordered_slots, expected);
    }

    #[test]
    fn membership_uses_rank_but_transport_uses_stable_slot_order() {
        let seeds = (0..6)
            .map(|card_id| seed(card_id, 2.0, 0.9, 0.9, 0))
            .collect::<Vec<_>>();
        let mut state = state(&seeds, LiveOrder::RetrievabilityAscending, 3);
        activate(&mut state, &[0.6, 0.5, 0.4, 0.3, 0.2, 0.1]);
        state.statuses[4] = CandidateStatus::PendingRefresh;
        state.eligibilities[4] = false;
        state.rebuild_order();
        state.select_refresh_membership(&[]);
        state.commit_refresh_debug();
        let (membership, transport) = state.last_refresh_debug();
        assert_eq!(membership, vec![4, 5, 3]);
        assert_eq!(transport, vec![3, 4, 5]);
    }

    #[test]
    fn next_boundary_uses_the_next_representable_strict_value() {
        let seeds = [seed(1, 2.0, 0.9, 0.9, 0)];
        let mut state = state(&seeds, LiveOrder::RetrievabilityAscending, 1);
        activate(&mut state, &[0.91]);
        let boundary = state.next_boundary(0).unwrap();
        assert!(boundary > 0.01);
        state.retention_extra = boundary;
        state.recompute_all_eligibilities();
        assert!(state.eligibilities[0]);
    }

    #[test]
    fn initial_prediction_input_preserves_seed_elapsed_values_exactly() {
        let seed = seed(1, 0.1, 0.9, 0.9, 0);
        let candidate = LiveCandidate::from_seed(&seed, 10_000_000_000_000_000.0, 10.0).unwrap();
        let input = candidate.initial_prediction_input();
        assert_eq!(
            input.elapsed_days.to_bits(),
            seed.row.elapsed_days.to_bits()
        );
        assert_eq!(
            input.elapsed_seconds.to_bits(),
            seed.row.elapsed_seconds.to_bits()
        );
    }

    #[test]
    fn membership_reuse_preserves_exact_anchors_and_slot_map_detects_duplicates() {
        pyo3::prepare_freethreaded_python();
        let source_seeds = [seed(1, 2.0, 0.91, 0.81, 11), seed(2, 3.0, 0.92, 0.82, 12)];
        let source = state(&source_seeds, LiveOrder::RetrievabilityAscending, 2);
        let source_anchor = source.candidates[0].last_review_timestamp_seconds;
        let source_day_anchor = source.candidates[0].last_review_day_offset;
        let target_timestamp = 1_000_123.25;
        let target_day = 10.5;
        let mut replacement = LivePredictionSessionState::new_empty(
            1,
            2,
            target_timestamp,
            target_day,
            LiveOrder::RetrievabilityAscending,
            LivePredictionMode::Oracle,
            8,
            2,
            Some(1),
            false,
        )
        .unwrap();

        replacement
            .push_reused_candidate(&source, 0, target_timestamp, target_day)
            .unwrap();
        assert_eq!(
            replacement.candidates[0].last_review_timestamp_seconds,
            source_anchor
        );
        assert_eq!(
            replacement.candidates[0].last_review_day_offset,
            source_day_anchor
        );
        assert_eq!(replacement.normal_targets, vec![0.91]);
        assert_eq!(replacement.intraday_targets, vec![0.81]);
        assert_eq!(replacement.tie_breakers, vec![11]);

        let error = replacement
            .push_reused_candidate(&source, 0, target_timestamp, target_day)
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("duplicate live candidate card_id=1"));
        assert_eq!(replacement.candidates.len(), 1);
        assert_eq!(replacement.slot_by_card_id.get(&1), Some(&0));
    }

    #[test]
    fn reconciliation_snapshot_restores_only_logical_candidate_state() {
        let old_seeds = (0..6)
            .map(|card_id| seed(card_id, 2.0, 0.9, 0.9, card_id as u64))
            .collect::<Vec<_>>();
        let mut previous = state(&old_seeds, LiveOrder::RetrievabilityAscending, 4);
        activate(&mut previous, &[0.6, 0.5, 0.4, 0.3, 0.2, 0.1]);
        previous.retention_extra = 0.03;
        previous.target_timestamp_seconds = 1_000_030.0;
        previous.target_day_offset = 10.5;
        let previous_order = ordered_cards(&previous);
        let snapshot = previous.into_reconciliation_undo_state();

        let current_seeds = [seed(100, 2.0, 0.9, 0.9, 0)];
        let mut current = state(&current_seeds, LiveOrder::RetrievabilityAscending, 4);
        activate(&mut current, &[0.7]);
        current.generation = 77;
        current.profile.reconcile_calls = 5;
        current.transport_slots.push(0);
        current.selected_scratch.push(0);
        current.merge_scratch.push(0);
        current.commit_slots_scratch.push(0);
        current.last_membership_slots.push(0);
        current.last_transport_slots.push(0);

        current.restore_reconciliation_undo_state(snapshot);

        assert_eq!(ordered_cards(&current), previous_order);
        assert_eq!(current.generation, 77);
        assert_eq!(current.profile.reconcile_calls, 5);
        assert_eq!(current.retention_extra, 0.03);
        assert_eq!(current.target_timestamp_seconds, 1_000_030.0);
        assert_eq!(current.target_day_offset, 10.5);
        assert!(current.transport_slots.is_empty());
        assert!(current.selected_scratch.is_empty());
        assert!(current.updated_entry_scratch.is_empty());
        assert!(current.merge_scratch.is_empty());
        assert!(current.input_scratch.is_empty());
        assert!(current.commit_slots_scratch.is_empty());
        assert_eq!(current.update_marks, vec![0; old_seeds.len()]);
        assert!(current.last_membership_slots.is_empty());
        assert!(current.last_transport_slots.is_empty());
    }

    #[test]
    fn measured_mostly_full_cutoff_selects_full_rebuild() {
        let seeds = (0..20)
            .map(|card_id| seed(card_id, 2.0, 0.9, 0.9, 0))
            .collect::<Vec<_>>();
        let mut state = state(&seeds, LiveOrder::RetrievabilityAscending, 20);
        activate(&mut state, &[0.5; 20]);
        assert!(!state.update_order_after_refresh(&(0..18).collect::<Vec<_>>(), false));
        assert!(state.update_order_after_refresh(&(0..19).collect::<Vec<_>>(), false));
    }

    #[test]
    fn marked_exclusions_match_hash_reference() {
        let seeds = (0..512)
            .map(|card_id| seed(card_id, 2.0, 0.9, 0.9, (card_id % 31) as u64))
            .collect::<Vec<_>>();
        let mut state = state(&seeds, LiveOrder::RetrievabilityAscending, 512);
        let predictions = (0..512)
            .map(|index| ((index * 257) % 521) as f64 / 521.0)
            .collect::<Vec<_>>();
        activate(&mut state, &predictions);

        for excluded_count in [0usize, 1, 2, 32, 33, 256, 512] {
            let mut excluded = state
                .candidates
                .iter()
                .take(excluded_count)
                .map(|candidate| candidate.card_id)
                .collect::<Vec<_>>();
            excluded.extend([i64::MAX, i64::MAX]);
            let excluded_set = excluded.iter().copied().collect::<HashSet<_>>();
            let expected = state.final_selection(2, &excluded_set);
            let actual = state.final_selection_marked_exclusion(2, &excluded);
            assert_eq!(actual.0, expected.0);
            assert_eq!(actual.1, expected.1);
            assert_eq!(actual.3, expected.3);
            assert_eq!(
                actual
                    .2
                    .iter()
                    .map(|selection| {
                        (
                            selection.card_id,
                            selection.retrievability.to_bits(),
                            selection.target_retrievability.to_bits(),
                        )
                    })
                    .collect::<Vec<_>>(),
                expected
                    .2
                    .iter()
                    .map(|selection| {
                        (
                            selection.card_id,
                            selection.retrievability.to_bits(),
                            selection.target_retrievability.to_bits(),
                        )
                    })
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    #[ignore = "manual 20k-slot full-sort versus partial-merge crossover benchmark"]
    fn benchmark_live_index_crossover() {
        let seeds = (0..20_000)
            .map(|card_id| seed(card_id, 2.0, 0.9, 0.9, (card_id % 31) as u64))
            .collect::<Vec<_>>();
        let mut base = state(&seeds, LiveOrder::RetrievabilityAscending, 20_000);
        let predictions = (0..20_000)
            .map(|index| ((index * 7_919) % 20_011) as f64 / 20_011.0)
            .collect::<Vec<_>>();
        activate(&mut base, &predictions);

        eprintln!("updated_slots,partial_merge_ns,full_rebuild_ns");
        for updated_count in [128usize, 512, 2_048, 4_096, 8_192, 12_000, 16_000, 19_000] {
            let updated = (0..updated_count)
                .map(|index| (index * 37) % 20_000)
                .collect::<Vec<_>>();
            let mut partial = base.clone();
            let mut full = base.clone();
            for (offset, &slot) in updated.iter().enumerate() {
                let prediction = ((offset * 104_729) % 20_021) as f64 / 20_021.0;
                partial.predictions[slot] = prediction;
                partial.eligibilities[slot] = partial.calculate_eligibility(slot);
                full.predictions[slot] = prediction;
                full.eligibilities[slot] = full.calculate_eligibility(slot);
            }
            let partial_start = Instant::now();
            partial.merge_updated_order(&updated);
            let partial_ns = partial_start.elapsed().as_nanos();
            let full_start = Instant::now();
            full.rebuild_order();
            let full_ns = full_start.elapsed().as_nanos();
            assert_eq!(partial.ordered_slots, full.ordered_slots);
            eprintln!("{updated_count},{partial_ns},{full_ns}");
        }
    }

    #[test]
    #[ignore = "manual 20k-slot slice, slot-mark, and hash-set exclusion benchmark"]
    fn benchmark_live_selection_exclusion_crossover() {
        let seeds = (0..20_000)
            .map(|card_id| seed(card_id, 2.0, 0.9, 0.9, (card_id % 31) as u64))
            .collect::<Vec<_>>();
        let mut state = state(&seeds, LiveOrder::RetrievabilityAscending, 8_192);
        let predictions = (0..20_000)
            .map(|index| ((index * 7_919) % 20_011) as f64 / 20_011.0)
            .collect::<Vec<_>>();
        activate(&mut state, &predictions);

        eprintln!("excluded,slice_ns_per_call,mark_ns_per_call,hash_ns_per_call");
        for excluded_count in [
            0usize, 1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512, 1_024, 2_048, 4_096, 8_192,
            12_000, 16_000, 20_000,
        ] {
            let excluded = state
                .candidates
                .iter()
                .take(excluded_count)
                .map(|candidate| candidate.card_id)
                .collect::<Vec<_>>();
            let excluded_set = excluded.iter().copied().collect::<HashSet<_>>();
            let slice_result = state.final_selection_small_exclusion(2, &excluded);
            let mark_result = state.final_selection_marked_exclusion(2, &excluded);
            let hash_result = state.final_selection(2, &excluded_set);
            assert_eq!(slice_result.0, hash_result.0);
            assert_eq!(slice_result.1, hash_result.1);
            assert_eq!(slice_result.3, hash_result.3);
            assert_eq!(mark_result.0, hash_result.0);
            assert_eq!(mark_result.1, hash_result.1);
            assert_eq!(mark_result.3, hash_result.3);
            assert_eq!(
                slice_result
                    .2
                    .iter()
                    .map(|selection| selection.card_id)
                    .collect::<Vec<_>>(),
                hash_result
                    .2
                    .iter()
                    .map(|selection| selection.card_id)
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                mark_result
                    .2
                    .iter()
                    .map(|selection| selection.card_id)
                    .collect::<Vec<_>>(),
                hash_result
                    .2
                    .iter()
                    .map(|selection| selection.card_id)
                    .collect::<Vec<_>>()
            );

            const ITERATIONS: u128 = 200;
            let slice_start = Instant::now();
            for _ in 0..ITERATIONS {
                std::hint::black_box(
                    state.final_selection_small_exclusion(2, std::hint::black_box(&excluded)),
                );
            }
            let slice_ns = slice_start.elapsed().as_nanos() / ITERATIONS;
            let mark_start = Instant::now();
            for _ in 0..ITERATIONS {
                std::hint::black_box(
                    state.final_selection_marked_exclusion(2, std::hint::black_box(&excluded)),
                );
            }
            let mark_ns = mark_start.elapsed().as_nanos() / ITERATIONS;
            let hash_start = Instant::now();
            for _ in 0..ITERATIONS {
                std::hint::black_box(state.final_selection(2, std::hint::black_box(&excluded_set)));
            }
            let hash_ns = hash_start.elapsed().as_nanos() / ITERATIONS;
            eprintln!("{excluded_count},{slice_ns},{mark_ns},{hash_ns}");
        }
    }
}
