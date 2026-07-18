"""Generic live-session API types and the Torch/reference state machine."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, replace as dataclass_replace
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

from rwkv_srs._api_core import (
    _PREDICT_REQUIRED_COLUMNS,
    ExecutionMode,
    ReviewInput,
    UndoUnavailableError,
    _coerce_review,
    _require_columns,
    _validate_batch_size,
    _validate_execution_mode,
)

LiveOrder = Literal[
    "retrievability_ascending",
    "retrievability_descending",
    "relative_overdueness",
    "random",
]
LiveCandidateStatus = Literal[
    "active",
    "excluded",
    "pending_refresh",
    "excluded_pending_refresh",
]

LIVE_ORDERS: tuple[LiveOrder, ...] = (
    "retrievability_ascending",
    "retrievability_descending",
    "relative_overdueness",
    "random",
)


@dataclass(frozen=True, slots=True)
class LiveCandidateSeed:
    """Seed one generic candidate from the row used for its initial prediction.

    ``row`` supplies the candidate identities and initial elapsed-time values.
    The session derives immutable last-review anchors from those elapsed values
    and the explicit initial target timestamp/day passed to the runtime.
    ``random_key`` is caller-generated, session-stable ordering state used only
    by ``order="random"``; preserve it when reseeding the same candidate.
    """

    row: ReviewInput
    target_retrievability: float
    intraday_target_retrievability: float
    tie_breaker: int
    random_key: int = 0


@dataclass(frozen=True, slots=True)
class LiveSelection:
    """One compact candidate selected by a live refresh."""

    card_id: int
    retrievability: float
    target_retrievability: float


@dataclass(frozen=True, slots=True)
class LiveRefreshResult:
    """Compact refresh output; refreshed per-card rows remain native-owned."""

    generation: int
    refreshed_count: int
    eligible_count: int
    active_count: int
    selected: tuple[LiveSelection, ...]
    next_retention_extra: float | None


@dataclass(frozen=True, slots=True)
class LiveCandidateSnapshot:
    """Explicit diagnostic snapshot for one live candidate."""

    card_id: int
    review_id: int
    note_id: int | None
    deck_id: int | None
    preset_id: int | None
    retrievability: float
    target_retrievability: float
    intraday_target_retrievability: float
    applicable_target_retrievability: float
    tie_breaker: int
    random_key: int
    status: LiveCandidateStatus
    eligible: bool
    has_prior_review: bool
    last_review_timestamp_seconds: float | None
    last_review_day_offset: float | None
    query_timestamp_seconds: float
    query_day_offset: float
    elapsed_seconds: float
    elapsed_days: float


@runtime_checkable
class LivePredictionSessionProtocol(Protocol):
    """Backend-independent live-session surface."""

    @property
    def initial_result(self) -> LiveRefreshResult: ...

    @property
    def generation(self) -> int: ...

    @property
    def mode(self) -> ExecutionMode: ...

    def current_selection(
        self,
        *,
        select_limit: int = 2,
        exclude_card_ids: Any = (),
    ) -> LiveRefreshResult: ...

    def refresh(
        self,
        *,
        target_timestamp_seconds: float,
        target_day_offset: float,
        select_limit: int = 2,
        exclude_card_ids: Any = (),
        exclude_refresh_card_ids: Any = (),
        retention_extra: float = 0.0,
    ) -> LiveRefreshResult: ...

    def reconcile_candidates(
        self,
        candidates: Any,
        *,
        target_timestamp_seconds: float,
        target_day_offset: float,
        select_limit: int = 2,
        exclude_card_ids: Any = (),
        retention_extra: float = 0.0,
    ) -> LiveRefreshResult: ...

    def reconcile_membership(
        self,
        card_ids: Any,
        changed_candidates: Any = (),
        *,
        target_timestamp_seconds: float,
        target_day_offset: float,
        select_limit: int = 2,
        exclude_card_ids: Any = (),
        retention_extra: float = 0.0,
    ) -> LiveRefreshResult: ...

    def process_answer(
        self,
        review_row: ReviewInput,
        *,
        requeue_after_prediction: bool = False,
        return_curves: bool = True,
        num_threads: int | None = None,
    ) -> float | tuple[float, Any]: ...

    def undo_last_process(self) -> int: ...

    def exclude_card(self, card_id: int) -> int: ...

    def include_card(self, card_id: int) -> int: ...

    def remove_candidate(self, card_id: int) -> int: ...

    def upsert_candidates(self, candidates: Any) -> int: ...

    def replace_candidates(self, candidates: Any) -> int: ...

    def candidate(self, card_id: int) -> LiveCandidateSnapshot | None: ...

    def snapshot(self) -> tuple[LiveCandidateSnapshot, ...]: ...

    def set_retention_extra(self, value: float) -> int: ...

    def set_mode(self, mode: ExecutionMode) -> int: ...

    def profile(self) -> dict[str, Any]: ...

    def allocation_profile(self) -> dict[str, int | None]: ...

    def last_refresh_debug(self) -> dict[str, tuple[int, ...]]: ...

    def close(self) -> None: ...


def _validate_live_order(value: str) -> LiveOrder:
    if not isinstance(value, str):
        raise TypeError("order must be a string.")
    if value not in LIVE_ORDERS:
        choices = ", ".join(repr(order) for order in LIVE_ORDERS)
        raise ValueError(f"order must be one of {choices}; got {value!r}.")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def _positive_int(value: Any, name: str) -> int:
    result = _non_negative_int(value, name)
    if result == 0:
        raise ValueError(f"{name} must be at least 1.")
    return result


def _card_id(value: Any, name: str = "card_id") -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if not -(2**63) <= result < 2**63:
        raise ValueError(f"{name} must fit in a signed 64-bit integer.")
    return result


def _coerce_live_seed(
    seed: LiveCandidateSeed, *, index: int
) -> tuple[dict[str, Any], float, float, int, int]:
    if not isinstance(seed, LiveCandidateSeed):
        raise TypeError(f"candidates[{index}] must be a LiveCandidateSeed.")
    row = _coerce_review(seed.row)
    _require_columns(row, _PREDICT_REQUIRED_COLUMNS)
    card = _card_id(row["card_id"], f"candidates[{index}].row.card_id")
    row["card_id"] = card
    normal_target = _real_float_allow_nonfinite(
        seed.target_retrievability,
        f"candidates[{index}].target_retrievability",
    )
    intraday_target = _real_float_allow_nonfinite(
        seed.intraday_target_retrievability,
        f"candidates[{index}].intraday_target_retrievability",
    )
    tie_breaker = _non_negative_int(
        seed.tie_breaker, f"candidates[{index}].tie_breaker"
    )
    if tie_breaker >= 2**64:
        raise ValueError(
            f"candidates[{index}].tie_breaker must fit in an unsigned 64-bit integer."
        )
    random_key = _non_negative_int(seed.random_key, f"candidates[{index}].random_key")
    if random_key >= 2**64:
        raise ValueError(
            f"candidates[{index}].random_key must fit in an unsigned 64-bit integer."
        )
    return row, normal_target, intraday_target, tie_breaker, random_key


def _coerce_live_seeds(
    candidates: Any,
) -> list[tuple[dict[str, Any], float, float, int, int]]:
    values = [
        _coerce_live_seed(seed, index=index) for index, seed in enumerate(candidates)
    ]
    seen: set[int] = set()
    for index, (row, _normal, _intraday, _tie, _random) in enumerate(values):
        card = int(row["card_id"])
        if card in seen:
            raise ValueError(f"duplicate candidates[{index}].row.card_id={card}.")
        seen.add(card)
    return values


def _materialize_native_live_seeds(candidates: Any) -> list[LiveCandidateSeed]:
    """Materialize native seeds without copying or reparsing their review rows.

    The Rust binding performs field, duplicate, and checkpoint-scope validation
    natively before it mutates session state. Keeping only the
    public-type check here preserves the API's early, index-specific error for
    callers that accidentally mix unrelated objects into the iterable.
    """

    values = list(candidates)
    for index, seed in enumerate(values):
        if not isinstance(seed, LiveCandidateSeed):
            raise TypeError(f"candidates[{index}] must be a LiveCandidateSeed.")
    return values


def _materialize_native_card_ids(card_ids: Any) -> list[Any]:
    """Materialize ordered membership while leaving strict parsing to Rust."""

    return list(card_ids)


def _real_float_allow_nonfinite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number.")
    return float(value)


def _coerce_card_ids(values: Any, name: str = "exclude_card_ids") -> list[int]:
    return [_card_id(value, f"{name}[{index}]") for index, value in enumerate(values)]


def _coerce_unique_card_ids(values: Any, name: str = "card_ids") -> list[int]:
    card_ids = _coerce_card_ids(values, name)
    seen: set[int] = set()
    for index, card_id in enumerate(card_ids):
        if card_id in seen:
            raise ValueError(f"duplicate {name}[{index}]={card_id}.")
        seen.add(card_id)
    return card_ids


def _refresh_result_from_mapping(value: Mapping[str, Any]) -> LiveRefreshResult:
    return LiveRefreshResult(
        generation=int(value["generation"]),
        refreshed_count=int(value["refreshed_count"]),
        eligible_count=int(value["eligible_count"]),
        active_count=int(value["active_count"]),
        selected=tuple(
            LiveSelection(
                card_id=int(item["card_id"]),
                retrievability=float(item["retrievability"]),
                target_retrievability=float(item["target_retrievability"]),
            )
            for item in value["selected"]
        ),
        next_retention_extra=(
            None
            if value["next_retention_extra"] is None
            else float(value["next_retention_extra"])
        ),
    )


def _candidate_snapshot_from_mapping(value: Mapping[str, Any]) -> LiveCandidateSnapshot:
    return LiveCandidateSnapshot(
        card_id=int(value["card_id"]),
        review_id=int(value["review_id"]),
        note_id=None if value["note_id"] is None else int(value["note_id"]),
        deck_id=None if value["deck_id"] is None else int(value["deck_id"]),
        preset_id=None if value["preset_id"] is None else int(value["preset_id"]),
        retrievability=float(value["retrievability"]),
        target_retrievability=float(value["target_retrievability"]),
        intraday_target_retrievability=float(value["intraday_target_retrievability"]),
        applicable_target_retrievability=float(
            value["applicable_target_retrievability"]
        ),
        tie_breaker=int(value["tie_breaker"]),
        random_key=int(value["random_key"]),
        status=value["status"],
        eligible=bool(value["eligible"]),
        has_prior_review=bool(value["has_prior_review"]),
        last_review_timestamp_seconds=(
            None
            if value["last_review_timestamp_seconds"] is None
            else float(value["last_review_timestamp_seconds"])
        ),
        last_review_day_offset=(
            None
            if value["last_review_day_offset"] is None
            else float(value["last_review_day_offset"])
        ),
        query_timestamp_seconds=float(value["query_timestamp_seconds"]),
        query_day_offset=float(value["query_day_offset"]),
        elapsed_seconds=float(value["elapsed_seconds"]),
        elapsed_days=float(value["elapsed_days"]),
    )


@dataclass(slots=True)
class _ReferenceCandidate:
    review_id: int
    card_id: int
    note_id: Any
    deck_id: Any
    preset_id: Any
    has_prior_review: bool
    new_elapsed_days: float
    new_elapsed_seconds: float
    last_review_timestamp_seconds: float | None
    last_review_day_offset: float | None
    query_timestamp_seconds: float
    query_day_offset: float
    elapsed_seconds: float
    elapsed_days: float
    prediction: float
    normal_target: float
    intraday_target: float
    tie_breaker: int
    random_key: int
    status: LiveCandidateStatus
    slot: int


class ReferenceLivePredictionSession:
    """Intentionally simple scan-and-sort reference used by the Torch backend.

    It establishes ordering/refresh semantics for differential tests. Candidate
    storage and row construction remain in Python, so this is not the optimized
    implementation callers should use for the 8k-row hot path.
    """

    def __init__(
        self,
        runtime: Any,
        candidates: Any,
        *,
        initial_target_timestamp_seconds: float,
        initial_target_day_offset: float,
        order: LiveOrder,
        mode: ExecutionMode,
        batch_size: int | None,
        refresh_limit: int,
        num_threads: int | None = None,
        initial_select_limit: int = 2,
    ) -> None:
        self._runtime = runtime
        self._closed = False
        self._order = _validate_live_order(order)
        self._mode = _validate_execution_mode(mode)
        self._batch_size = (
            None if batch_size is None else _validate_batch_size(batch_size)
        )
        self._num_threads = num_threads
        self._refresh_limit = _positive_int(refresh_limit, "refresh_limit")
        initial_select_limit = _non_negative_int(
            initial_select_limit,
            "initial_select_limit",
        )
        self._target_timestamp = _finite_float(
            initial_target_timestamp_seconds,
            "initial_target_timestamp_seconds",
        )
        self._target_day = _finite_float(
            initial_target_day_offset, "initial_target_day_offset"
        )
        self._retention_extra = 0.0
        self._generation = 1
        self._by_id: dict[int, _ReferenceCandidate] = {}
        self._slots: list[_ReferenceCandidate] = []
        self._last_membership: tuple[int, ...] = ()
        self._last_transport: tuple[int, ...] = ()
        for row, normal, intraday, tie, random_key in _coerce_live_seeds(candidates):
            candidate = self._candidate_from_seed(
                row, normal, intraday, tie, random_key
            )
            self._by_id[candidate.card_id] = candidate
            self._slots.append(candidate)
        rows = [self._initial_prediction_row(candidate) for candidate in self._slots]
        predictions = runtime.predict_many(
            rows,
            batch_size=self._batch_size,
            num_threads=self._num_threads,
            mode=self._mode,
        )
        if len(predictions) != len(self._slots):
            raise RuntimeError("reference live predictor returned the wrong row count")
        for candidate, prediction in zip(self._slots, predictions):
            candidate.prediction = float(prediction)
            candidate.status = "active"
        self._initial_result = self._selection_result(
            select_limit=initial_select_limit,
            excluded_card_ids=set(),
            refreshed_count=len(self._slots),
        )

    @property
    def initial_result(self) -> LiveRefreshResult:
        self._require_open()
        return self._initial_result

    @property
    def generation(self) -> int:
        self._require_open()
        return self._generation

    @property
    def mode(self) -> ExecutionMode:
        self._require_open()
        return self._mode

    def refresh(
        self,
        *,
        target_timestamp_seconds: float,
        target_day_offset: float,
        select_limit: int = 2,
        exclude_card_ids: Any = (),
        exclude_refresh_card_ids: Any = (),
        retention_extra: float = 0.0,
    ) -> LiveRefreshResult:
        self._require_open()
        timestamp = _finite_float(target_timestamp_seconds, "target_timestamp_seconds")
        day = _finite_float(target_day_offset, "target_day_offset")
        limit = _non_negative_int(select_limit, "select_limit")
        extra = _finite_float(retention_extra, "retention_extra")
        excluded = set(_coerce_card_ids(exclude_card_ids))
        refresh_excluded = set(_coerce_card_ids(exclude_refresh_card_ids))
        ordered = self._ordered()
        membership: list[_ReferenceCandidate] = []
        chosen: set[int] = set()
        for candidate in self._slots:
            if len(membership) >= self._refresh_limit:
                break
            if (
                self._by_id.get(candidate.card_id) is candidate
                and candidate.status == "pending_refresh"
                and candidate.card_id not in refresh_excluded
            ):
                membership.append(candidate)
                chosen.add(candidate.card_id)
        for candidate in ordered:
            if len(membership) >= self._refresh_limit:
                break
            if (
                candidate.card_id in refresh_excluded
                or candidate.card_id in chosen
                or candidate.status not in {
                "active",
                "pending_refresh",
                }
            ):
                continue
            membership.append(candidate)
            chosen.add(candidate.card_id)
        transport = sorted(membership, key=lambda candidate: candidate.slot)
        rows = [
            self._prediction_row(candidate, timestamp, day) for candidate in transport
        ]
        predictions = self._runtime.predict_many(
            rows,
            batch_size=self._batch_size,
            num_threads=self._num_threads,
            mode=self._mode,
        )
        if len(predictions) != len(transport):
            raise RuntimeError("reference live predictor returned the wrong row count")
        for candidate, row, prediction in zip(transport, rows, predictions):
            candidate.prediction = float(prediction)
            candidate.query_timestamp_seconds = timestamp
            candidate.query_day_offset = day
            candidate.elapsed_days = float(row["elapsed_days"])
            candidate.elapsed_seconds = float(row["elapsed_seconds"])
            if candidate.status == "pending_refresh":
                candidate.status = "active"
            elif candidate.status == "excluded_pending_refresh":
                candidate.status = "excluded"
        self._retention_extra = extra
        self._target_timestamp = timestamp
        self._target_day = day
        self._generation += 1
        self._last_membership = tuple(candidate.card_id for candidate in membership)
        self._last_transport = tuple(candidate.card_id for candidate in transport)

        return self._selection_result(
            select_limit=limit,
            excluded_card_ids=excluded,
            refreshed_count=len(transport),
        )

    def current_selection(
        self,
        *,
        select_limit: int = 2,
        exclude_card_ids: Any = (),
    ) -> LiveRefreshResult:
        """Read the current rank without refreshing any predictions."""

        self._require_open()
        return self._selection_result(
            select_limit=_non_negative_int(select_limit, "select_limit"),
            excluded_card_ids=set(_coerce_card_ids(exclude_card_ids)),
            refreshed_count=0,
        )

    def reconcile_candidates(
        self,
        candidates: Any,
        *,
        target_timestamp_seconds: float,
        target_day_offset: float,
        select_limit: int = 2,
        exclude_card_ids: Any = (),
        retention_extra: float = 0.0,
    ) -> LiveRefreshResult:
        """Atomically predict and install a complete replacement universe."""

        self._require_open()
        timestamp = _finite_float(target_timestamp_seconds, "target_timestamp_seconds")
        day = _finite_float(target_day_offset, "target_day_offset")
        limit = _non_negative_int(select_limit, "select_limit")
        extra = _finite_float(retention_extra, "retention_extra")
        excluded = set(_coerce_card_ids(exclude_card_ids))
        replacements: list[_ReferenceCandidate] = []
        by_id: dict[int, _ReferenceCandidate] = {}
        for row, normal, intraday, tie, random_key in _coerce_live_seeds(candidates):
            candidate = self._candidate_from_seed(
                row,
                normal,
                intraday,
                tie,
                random_key,
                target_timestamp_seconds=timestamp,
                target_day_offset=day,
                slot=len(replacements),
            )
            replacements.append(candidate)
            by_id[candidate.card_id] = candidate
        rows = [self._initial_prediction_row(candidate) for candidate in replacements]
        predictions = self._runtime.predict_many(
            rows,
            batch_size=self._batch_size,
            num_threads=self._num_threads,
            mode=self._mode,
        )
        if len(predictions) != len(replacements):
            raise RuntimeError("reference live predictor returned the wrong row count")
        for candidate, prediction in zip(replacements, predictions):
            candidate.prediction = float(prediction)
            candidate.status = "active"

        self._slots = replacements
        self._by_id = by_id
        self._target_timestamp = timestamp
        self._target_day = day
        self._retention_extra = extra
        self._generation += 1
        self._last_membership = tuple(candidate.card_id for candidate in replacements)
        self._last_transport = self._last_membership
        return self._selection_result(
            select_limit=limit,
            excluded_card_ids=excluded,
            refreshed_count=len(replacements),
        )

    def reconcile_membership(
        self,
        card_ids: Any,
        changed_candidates: Any = (),
        *,
        target_timestamp_seconds: float,
        target_day_offset: float,
        select_limit: int = 2,
        exclude_card_ids: Any = (),
        retention_extra: float = 0.0,
    ) -> LiveRefreshResult:
        """Atomically reconcile ordered membership while reusing unchanged anchors.

        ``card_ids`` is the complete desired universe in stable transport order.
        ``changed_candidates`` must seed every new ID and may replace an
        existing ID whose identities, anchors, targets, or tie-breaker changed.
        """

        self._require_open()
        timestamp = _finite_float(target_timestamp_seconds, "target_timestamp_seconds")
        day = _finite_float(target_day_offset, "target_day_offset")
        limit = _non_negative_int(select_limit, "select_limit")
        extra = _finite_float(retention_extra, "retention_extra")
        excluded = set(_coerce_card_ids(exclude_card_ids))
        desired_card_ids = _coerce_unique_card_ids(card_ids)
        desired = set(desired_card_ids)
        changed = _coerce_live_seeds(changed_candidates)
        changed_by_id = {
            int(row["card_id"]): (row, normal, intraday, tie, random_key)
            for row, normal, intraday, tie, random_key in changed
        }
        for index, (row, _normal, _intraday, _tie, _random) in enumerate(changed):
            card_id = int(row["card_id"])
            if card_id not in desired:
                raise ValueError(
                    f"changed_candidates[{index}].row.card_id={card_id} is not "
                    "present in card_ids."
                )

        replacements: list[_ReferenceCandidate] = []
        by_id: dict[int, _ReferenceCandidate] = {}
        for index, card_id in enumerate(desired_card_ids):
            update = changed_by_id.get(card_id)
            if update is not None:
                row, normal, intraday, tie, random_key = update
                candidate = self._candidate_from_seed(
                    row,
                    normal,
                    intraday,
                    tie,
                    random_key,
                    target_timestamp_seconds=timestamp,
                    target_day_offset=day,
                    slot=index,
                )
            else:
                existing = self._by_id.get(card_id)
                if existing is None:
                    raise ValueError(
                        f"card_ids[{index}]={card_id} is not an active live candidate; "
                        "provide a changed_candidates seed."
                    )
                row = self._prediction_row(existing, timestamp, day)
                candidate = dataclass_replace(
                    existing,
                    query_timestamp_seconds=timestamp,
                    query_day_offset=day,
                    elapsed_seconds=float(row["elapsed_seconds"]),
                    elapsed_days=float(row["elapsed_days"]),
                    prediction=math.nan,
                    status="pending_refresh",
                    slot=index,
                )
            replacements.append(candidate)
            by_id[card_id] = candidate

        rows = [self._initial_prediction_row(candidate) for candidate in replacements]
        predictions = self._runtime.predict_many(
            rows,
            batch_size=self._batch_size,
            num_threads=self._num_threads,
            mode=self._mode,
        )
        if len(predictions) != len(replacements):
            raise RuntimeError("reference live predictor returned the wrong row count")
        for candidate, prediction in zip(replacements, predictions):
            candidate.prediction = float(prediction)
            candidate.status = "active"

        self._slots = replacements
        self._by_id = by_id
        self._target_timestamp = timestamp
        self._target_day = day
        self._retention_extra = extra
        self._generation += 1
        self._last_membership = tuple(desired_card_ids)
        self._last_transport = self._last_membership
        return self._selection_result(
            select_limit=limit,
            excluded_card_ids=excluded,
            refreshed_count=len(replacements),
        )

    def process_answer(
        self,
        review_row: ReviewInput,
        *,
        requeue_after_prediction: bool = False,
        return_curves: bool = True,
        num_threads: int | None = None,
    ) -> float | tuple[float, Any]:
        self._require_open()
        raise UndoUnavailableError(
            "live process_answer is unavailable on the Torch backend because Torch has no native undo."
        )

    def undo_last_process(self) -> int:
        self._require_open()
        raise UndoUnavailableError(
            "live undo_last_process is unavailable on the Torch backend because Torch has no native undo."
        )

    def exclude_card(self, card_id: int) -> int:
        candidate = self._candidate_required(card_id)
        if candidate.status == "active":
            candidate.status = "excluded"
        elif candidate.status == "pending_refresh":
            candidate.status = "excluded_pending_refresh"
        self._generation += 1
        return self._generation

    def include_card(self, card_id: int) -> int:
        candidate = self._candidate_required(card_id)
        if candidate.status == "excluded":
            candidate.status = "active"
        elif candidate.status == "excluded_pending_refresh":
            candidate.status = "pending_refresh"
        self._generation += 1
        return self._generation

    def remove_candidate(self, card_id: int) -> int:
        self._require_open()
        card = _card_id(card_id)
        if self._by_id.pop(card, None) is None:
            raise ValueError(f"live candidate card_id={card} does not exist")
        self._generation += 1
        return self._generation

    def upsert_candidates(self, candidates: Any) -> int:
        self._require_open()
        replacements = [
            self._candidate_from_seed(row, normal, intraday, tie, random_key)
            for row, normal, intraday, tie, random_key in _coerce_live_seeds(candidates)
        ]
        for replacement in replacements:
            card = replacement.card_id
            existing = self._by_id.get(card)
            replacement.status = (
                "excluded_pending_refresh"
                if existing is not None
                and existing.status in {"excluded", "excluded_pending_refresh"}
                else "pending_refresh"
            )
            if existing is None:
                replacement.slot = len(self._slots)
                self._slots.append(replacement)
            else:
                replacement.slot = existing.slot
                self._slots[existing.slot] = replacement
            self._by_id[card] = replacement
        self._generation += 1
        return self._generation

    def replace_candidates(self, candidates: Any) -> int:
        self._require_open()
        replacements: list[_ReferenceCandidate] = []
        by_id: dict[int, _ReferenceCandidate] = {}
        for row, normal, intraday, tie, random_key in _coerce_live_seeds(candidates):
            candidate = self._candidate_from_seed(
                row, normal, intraday, tie, random_key
            )
            candidate.slot = len(replacements)
            replacements.append(candidate)
            by_id[candidate.card_id] = candidate
        rows = [
            self._prediction_row(candidate, self._target_timestamp, self._target_day)
            for candidate in replacements
        ]
        predictions = self._runtime.predict_many(
            rows,
            batch_size=self._batch_size,
            num_threads=self._num_threads,
            mode=self._mode,
        )
        if len(predictions) != len(replacements):
            raise RuntimeError("reference live predictor returned the wrong row count")
        for candidate, prediction in zip(replacements, predictions):
            candidate.prediction = float(prediction)
            candidate.status = "active"
        self._slots = replacements
        self._by_id = by_id
        self._generation += 1
        return self._generation

    def candidate(self, card_id: int) -> LiveCandidateSnapshot | None:
        self._require_open()
        candidate = self._by_id.get(_card_id(card_id))
        return None if candidate is None else self._snapshot(candidate)

    def snapshot(self) -> tuple[LiveCandidateSnapshot, ...]:
        self._require_open()
        return tuple(self._snapshot(candidate) for candidate in self._ordered())

    def set_retention_extra(self, value: float) -> int:
        self._require_open()
        self._retention_extra = _finite_float(value, "retention_extra")
        self._generation += 1
        return self._generation

    def set_mode(self, mode: ExecutionMode) -> int:
        """Retain the reference session's fixed Oracle executor."""

        self._require_open()
        mode = _validate_execution_mode(mode)
        # The Python reference exists only to check Rust ordering semantics;
        # it deliberately does not grow Fast/GPU execution or fallback support.
        if mode != "oracle":
            raise ValueError(
                f"live set_mode({mode!r}) is only supported by the Rust backend."
            )
        self._mode = mode
        self._generation += 1
        return self._generation

    def profile(self) -> dict[str, Any]:
        self._require_open()
        return {"enabled": False, "backend": "python-reference"}

    def allocation_profile(self) -> dict[str, int | None]:
        """Return logical counts; native capacity bytes are Rust-specific."""

        self._require_open()
        return {
            "active_candidate_count": len(self._slots),
            "active_candidate_tracked_capacity_bytes": None,
            "live_undo_frame_count": 0,
            "reconciliation_snapshot_count": 0,
            "reconciliation_snapshot_candidate_count": 0,
            "reconciliation_snapshot_tracked_capacity_bytes": 0,
        }

    def last_refresh_debug(self) -> dict[str, tuple[int, ...]]:
        self._require_open()
        return {
            "membership_card_ids": self._last_membership,
            "transport_card_ids": self._last_transport,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._generation += 1
        self._closed = True
        callback = getattr(self._runtime, "_reference_live_session_closed", None)
        if callback is not None:
            callback(self)

    def __enter__(self) -> ReferenceLivePredictionSession:
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _candidate_from_seed(
        self,
        row: dict[str, Any],
        normal: float,
        intraday: float,
        tie: int,
        random_key: int,
        *,
        target_timestamp_seconds: float | None = None,
        target_day_offset: float | None = None,
        slot: int | None = None,
    ) -> _ReferenceCandidate:
        target_timestamp = (
            self._target_timestamp
            if target_timestamp_seconds is None
            else float(target_timestamp_seconds)
        )
        target_day = (
            self._target_day if target_day_offset is None else float(target_day_offset)
        )
        elapsed_days = float(row["elapsed_days"])
        elapsed_seconds = float(row["elapsed_seconds"])
        if not math.isfinite(elapsed_days) or not math.isfinite(elapsed_seconds):
            raise ValueError("live seed elapsed values must be finite")
        is_new = elapsed_days == -1.0
        if is_new != (elapsed_seconds == -1.0):
            raise ValueError("new candidates must use -1 for both elapsed fields")
        if not is_new and (elapsed_days < 0.0 or elapsed_seconds < 0.0):
            raise ValueError("elapsed values must be non-negative or exactly -1")
        if is_new:
            timestamp_anchor = None
            day_anchor = None
        else:
            timestamp_anchor = target_timestamp - elapsed_seconds
            day_anchor = target_day - elapsed_days
            if not math.isfinite(timestamp_anchor) or not math.isfinite(day_anchor):
                raise ValueError("live seed cannot derive finite last-review anchors")
        return _ReferenceCandidate(
            review_id=int(row["review_id"]),
            card_id=int(row["card_id"]),
            note_id=row["note_id"],
            deck_id=row["deck_id"],
            preset_id=row["preset_id"],
            has_prior_review=not is_new,
            new_elapsed_days=elapsed_days,
            new_elapsed_seconds=elapsed_seconds,
            last_review_timestamp_seconds=timestamp_anchor,
            last_review_day_offset=day_anchor,
            query_timestamp_seconds=target_timestamp,
            query_day_offset=target_day,
            elapsed_seconds=elapsed_seconds,
            elapsed_days=elapsed_days,
            prediction=math.nan,
            normal_target=normal,
            intraday_target=intraday,
            tie_breaker=tie,
            random_key=random_key,
            status="pending_refresh",
            slot=len(self._slots) if slot is None else int(slot),
        )

    def _prediction_row(
        self, candidate: _ReferenceCandidate, timestamp: float, day: float
    ) -> dict[str, Any]:
        if candidate.has_prior_review:
            assert candidate.last_review_timestamp_seconds is not None
            assert candidate.last_review_day_offset is not None
            elapsed_seconds = timestamp - candidate.last_review_timestamp_seconds
            elapsed_days = day - candidate.last_review_day_offset
        else:
            elapsed_seconds = candidate.new_elapsed_seconds
            elapsed_days = candidate.new_elapsed_days
        if not math.isfinite(elapsed_seconds) or not math.isfinite(elapsed_days):
            raise ValueError(
                f"candidate {candidate.card_id} produced non-finite time-adjusted prediction values"
            )
        return self._row(candidate, day, elapsed_days, elapsed_seconds)

    def _initial_prediction_row(self, candidate: _ReferenceCandidate) -> dict[str, Any]:
        return self._row(
            candidate,
            candidate.query_day_offset,
            candidate.elapsed_days,
            candidate.elapsed_seconds,
        )

    @staticmethod
    def _row(
        candidate: _ReferenceCandidate,
        day: float,
        elapsed_days: float,
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        return {
            "review_id": candidate.review_id,
            "card_id": candidate.card_id,
            "note_id": candidate.note_id,
            "deck_id": candidate.deck_id,
            "preset_id": candidate.preset_id,
            "day_offset": day,
            "elapsed_days": elapsed_days,
            "elapsed_seconds": elapsed_seconds,
        }

    def _ordered(self) -> list[_ReferenceCandidate]:
        candidates = list(self._by_id.values())

        def key(candidate: _ReferenceCandidate):
            prediction = candidate.prediction
            if self._order == "random":
                return (
                    not self._eligible(candidate),
                    False,
                    candidate.random_key,
                    candidate.tie_breaker,
                    candidate.card_id,
                )
            target = self._target(candidate)
            finite = math.isfinite(prediction) and (
                self._order != "relative_overdueness" or math.isfinite(target)
            )
            if self._order == "retrievability_ascending":
                directed = prediction
            elif self._order == "retrievability_descending":
                directed = -prediction
            else:
                directed = max(prediction, 0.0001) / max(target, 0.0001)
                finite = finite and math.isfinite(directed)
            return (
                not self._eligible(candidate),
                not finite,
                directed if finite else 0.0,
                candidate.tie_breaker,
                candidate.card_id,
            )

        return sorted(candidates, key=key)

    def _selection_result(
        self,
        *,
        select_limit: int,
        excluded_card_ids: set[int],
        refreshed_count: int,
    ) -> LiveRefreshResult:
        active = [
            candidate
            for candidate in self._ordered()
            if candidate.status in {"active", "pending_refresh"}
            and candidate.card_id not in excluded_card_ids
        ]
        eligible = [candidate for candidate in active if self._eligible(candidate)]
        selected = tuple(
            LiveSelection(
                card_id=candidate.card_id,
                retrievability=candidate.prediction,
                target_retrievability=self._target(candidate),
            )
            for candidate in eligible[:select_limit]
        )
        boundaries = [
            boundary
            for candidate in active
            if (boundary := self._next_boundary(candidate)) is not None
        ]
        return LiveRefreshResult(
            generation=self._generation,
            refreshed_count=refreshed_count,
            eligible_count=len(eligible),
            active_count=len(active),
            selected=selected,
            next_retention_extra=min(boundaries) if boundaries else None,
        )

    def _target(self, candidate: _ReferenceCandidate) -> float:
        return (
            candidate.intraday_target
            if 0.0 <= candidate.elapsed_days < 1.0
            else candidate.normal_target
        )

    def _eligible(self, candidate: _ReferenceCandidate) -> bool:
        if candidate.status not in {"active", "excluded"}:
            return False
        target = self._target(candidate)
        return (
            math.isfinite(candidate.prediction)
            and math.isfinite(target)
            and candidate.prediction
            < min(1.0, max(0.0, target + self._retention_extra))
        )

    def _next_boundary(self, candidate: _ReferenceCandidate) -> float | None:
        if candidate.status != "active":
            return None
        target = self._target(candidate)
        prediction = candidate.prediction
        if (
            not math.isfinite(prediction)
            or not math.isfinite(target)
            or prediction >= 1.0
        ):
            return None
        required = math.nextafter(prediction, math.inf) - target
        if required <= prediction - target:
            required = math.nextafter(required, math.inf)
        while prediction >= min(1.0, max(0.0, target + required)):
            required = math.nextafter(required, math.inf)
        return required if required > self._retention_extra else None

    def _snapshot(self, candidate: _ReferenceCandidate) -> LiveCandidateSnapshot:
        return LiveCandidateSnapshot(
            card_id=candidate.card_id,
            review_id=candidate.review_id,
            note_id=None
            if _is_missing_id(candidate.note_id)
            else int(candidate.note_id),
            deck_id=None
            if _is_missing_id(candidate.deck_id)
            else int(candidate.deck_id),
            preset_id=None
            if _is_missing_id(candidate.preset_id)
            else int(candidate.preset_id),
            retrievability=candidate.prediction,
            target_retrievability=candidate.normal_target,
            intraday_target_retrievability=candidate.intraday_target,
            applicable_target_retrievability=self._target(candidate),
            tie_breaker=candidate.tie_breaker,
            random_key=candidate.random_key,
            status=candidate.status,
            eligible=self._eligible(candidate),
            has_prior_review=candidate.has_prior_review,
            last_review_timestamp_seconds=candidate.last_review_timestamp_seconds,
            last_review_day_offset=candidate.last_review_day_offset,
            query_timestamp_seconds=candidate.query_timestamp_seconds,
            query_day_offset=candidate.query_day_offset,
            elapsed_seconds=candidate.elapsed_seconds,
            elapsed_days=candidate.elapsed_days,
        )

    def _candidate_required(self, card_id: int) -> _ReferenceCandidate:
        self._require_open()
        card = _card_id(card_id)
        try:
            return self._by_id[card]
        except KeyError as exc:
            raise ValueError(f"live candidate card_id={card} does not exist") from exc

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("live prediction session is closed")


def _is_missing_id(value: Any) -> bool:
    return value is None or (
        isinstance(value, numbers.Real) and math.isnan(float(value))
    )


__all__ = [
    "LiveCandidateSeed",
    "LiveCandidateSnapshot",
    "LiveCandidateStatus",
    "LiveOrder",
    "LivePredictionSessionProtocol",
    "LiveRefreshResult",
    "LiveSelection",
]
