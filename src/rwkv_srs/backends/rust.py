from __future__ import annotations

import gc
import json
import math
import numbers
import os
import queue
import struct
import threading
import time
import weakref
from array import array
from collections import deque
from collections.abc import Iterable, Sized
from pathlib import Path
from typing import Any, Literal, cast

from rwkv_srs._api_core import (
    _CHECKPOINT_VERSION,
    _DEFAULT_PROCESS_MANY_BATCH_SIZE,
    _DEFAULT_RUST_PREDICT_MANY_BATCH_SIZE,
    _DEFAULT_RUST_PROCESS_MANY_THREADS,
    _DEFAULT_RUST_SCALAR_THREADS,
    _HISTORY_FINGERPRINT_ALGORITHM,
    _HISTORY_FINGERPRINT_CANONICALIZATION,
    _HISTORY_FINGERPRINT_FIELDS,
    _HISTORY_FINGERPRINT_VERSION,
    _INITIAL_HISTORY_DIGEST,
    _PREDICT_REQUIRED_COLUMNS,
    _PROCESS_REQUIRED_COLUMNS,
    _CheckpointCardScope,
    CpuMode,
    ExecutionMode,
    GpuError,
    GpuExecutionError,
    GpuOperation,
    GpuOutOfMemoryError,
    GpuProcessError,
    GpuUnavailableError,
    PRETRAINED_MODEL_DIR,
    ReviewInput,
    TEST_MODEL_FIXTURE_DIR,
    UndoUnavailableError,
    _atomic_output_path,
    _chain_digest,
    _checkpoint_scope_from_metadata,
    _coerce_probability_many_inputs,
    _coerce_review,
    _fingerprint_reviews,
    _normalize_review_id,
    _require_columns,
    _validate_elapsed_seconds,
    _validate_cpu_mode,
    _validate_execution_mode,
    _validate_gpu_operation,
    _validate_fingerprint_metadata,
    _validate_batch_size,
    _validate_num_threads,
    _validate_retention_probability,
)
from rwkv_srs.live import (
    LiveCandidateSeed,
    LiveCandidateSnapshot,
    LiveOrder,
    LiveRefreshResult,
    _candidate_snapshot_from_mapping,
    _coerce_card_ids,
    _finite_float,
    _materialize_native_card_ids,
    _materialize_native_live_seeds,
    _non_negative_int,
    _positive_int,
    _refresh_result_from_mapping,
    _validate_live_order,
)
from rwkv_srs.prediction_batch import PredictionBatch
from rwkv_srs.review_batch import ReviewBatch

DEFAULT_PREDICT_THREADS = _DEFAULT_RUST_SCALAR_THREADS
DEFAULT_PREDICT_MANY_BATCH_SIZE = _DEFAULT_RUST_PREDICT_MANY_BATCH_SIZE
DEFAULT_FAST_PREDICT_MANY_BATCH_SIZE = 96
DEFAULT_GPU_PREDICT_MANY_BATCH_SIZE = 8192
DEFAULT_GPU_PROCESS_MANY_BATCH_SIZE = 32_767
DEFAULT_GPU_BUILD_STATE_ONLY_BATCH_SIZE = 32_767
DEFAULT_GPU_FULLY_RESIDENT_MIN_REVIEWS = 200_000
DEFAULT_RUST_SEED = 5489
GPU_PROCESS_CURVE_SIZE = 128
DEFAULT_PROCESS_THREADS = _DEFAULT_RUST_SCALAR_THREADS
DEFAULT_PROCESS_MANY_BATCH_SIZE = _DEFAULT_PROCESS_MANY_BATCH_SIZE
DEFAULT_PROCESS_MANY_THREADS = _DEFAULT_RUST_PROCESS_MANY_THREADS
DEFAULT_UNDO_LIMIT = 30
PROCESS_MANY_BULK_ENV_VAR = "RWKV_SRS_PROCESS_MANY_BULK"
CPU_PROFILE_ENV_VAR = "RWKV_SRS_CPU_PROFILE"
FAST_RUST_PROCESS_PROFILE_ENV = {
    CPU_PROFILE_ENV_VAR: "fast",
}
_NATIVE_USIZE_MAX = (1 << (struct.calcsize("P") * 8)) - 1

_CPU_MODE_LOCK = threading.Lock()
_CPU_MODE_CLAIMED: CpuMode | None = None

os.environ.setdefault("RAYON_NUM_THREADS", str(DEFAULT_PREDICT_THREADS))
_GLOBAL_RAYON_NUM_THREADS: int | None
try:
    _GLOBAL_RAYON_NUM_THREADS = int(os.environ["RAYON_NUM_THREADS"])
except ValueError:
    _GLOBAL_RAYON_NUM_THREADS = None

try:
    # The compiled PyO3 extension does not publish a Python stub surface yet.
    import rwkv_srs._native as _native  # type: ignore[import-untyped]
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Rust backend is not built. Run "
        "`scripts/build_rust_release_extension.sh` for benchmarking, or "
        "`.venv/bin/python -m maturin develop --skip-install --manifest-path "
        "rust/rwkv-srs-cpu/Cargo.toml` for local debug builds."
    ) from exc

_EXPECTED_NATIVE_API_VERSION = 33

try:
    _LOADED_NATIVE_API_VERSION = int(_native.native_api_version())
except AttributeError as exc:  # pragma: no cover
    raise RuntimeError(
        "Rust backend extension is stale. Rebuild it with "
        "`scripts/build_rust_release_extension.sh` so the adapter can verify "
        "the native binding API version."
    ) from exc


def gpu_device_info(operation: GpuOperation = "predict") -> dict[str, Any]:
    """Return adapter capabilities for GPU prediction or processing."""
    operation = _validate_gpu_operation(operation)
    try:
        return dict(_native.gpu_device_info(operation))
    except BaseException as exc:
        translated = _translate_native_gpu_error(
            exc,
            operation=operation,
            phase="probe",
        )
        if translated is None:
            raise
        raise translated from exc


def gpu_available(operation: GpuOperation = "predict") -> bool:
    """Return whether an adapter satisfies one GPU operation's requirements."""
    try:
        gpu_device_info(operation)
    except GpuError:
        return False
    return True


def check_checkpoint_history_consistency(
    checkpoint: str | Path,
    reviews: Iterable[ReviewInput] | ReviewBatch,
) -> bool:
    """Return whether reviews match the history represented by a checkpoint.

    This reads only the Rust checkpoint's bounded JSON metadata. It does not
    construct a model runtime or load recurrent state. Review parsing,
    canonicalization, and chained SHA-256 comparison remain in the native
    :class:`~rwkv_srs.ReviewBatch` implementation.
    """
    metadata = _load_checkpoint_dict(checkpoint)
    fingerprint = metadata.get("history_fingerprint")
    if fingerprint is None:
        raise ValueError("This checkpoint does not contain a history fingerprint.")
    _validate_fingerprint_metadata(fingerprint)
    batch = reviews if isinstance(reviews, ReviewBatch) else ReviewBatch(reviews)
    return batch._matches_history_fingerprint(
        digest=str(fingerprint["digest"]),
        processed_review_count=int(fingerprint["processed_review_count"]),
        last_review_id=fingerprint["last_review_id"],
    )


if _LOADED_NATIVE_API_VERSION != _EXPECTED_NATIVE_API_VERSION:  # pragma: no cover
    raise RuntimeError(
        "Rust backend extension API version mismatch: expected "
        f"{_EXPECTED_NATIVE_API_VERSION}, got {_LOADED_NATIVE_API_VERSION}. "
        "Rebuild the native extension with `scripts/build_rust_release_extension.sh`."
    )


def _translate_native_gpu_error(
    exc: BaseException,
    *,
    operation: GpuOperation,
    phase: str,
    committed_rows: int = 0,
    partial_results: Iterable[Any] = (),
    state_recoverable: bool | None = None,
) -> GpuError | None:
    """Convert owner-thread-safe native markers into the public hierarchy."""

    if isinstance(exc, GpuError):
        return exc
    if isinstance(exc, _native.NativeGpuOutOfMemoryError):
        recoverable = True if state_recoverable is None else state_recoverable
        return GpuOutOfMemoryError(
            str(exc),
            operation=operation,
            phase=phase,
            committed_rows=committed_rows,
            partial_results=partial_results,
            state_recoverable=recoverable,
            retryable_on_cpu=recoverable,
        )
    if isinstance(exc, _native.NativeGpuUnavailableError):
        recoverable = True if state_recoverable is None else state_recoverable
        return GpuUnavailableError(
            str(exc),
            operation=operation,
            phase=phase,
            committed_rows=committed_rows,
            partial_results=partial_results,
            state_recoverable=recoverable,
            retryable_on_cpu=recoverable,
        )
    if not isinstance(exc, _native.NativeGpuError):
        return None
    error_type: type[GpuError] = (
        GpuUnavailableError if phase in {"probe", "initialize"} else GpuExecutionError
    )
    recoverable = (
        operation == "predict" if state_recoverable is None else state_recoverable
    )
    return error_type(
        str(exc),
        operation=operation,
        phase=phase,
        committed_rows=committed_rows,
        partial_results=partial_results,
        state_recoverable=recoverable,
        retryable_on_cpu=recoverable,
    )


try:
    _PHYSICAL_CPU_COUNT = max(1, int(_native.physical_cpu_count()))
except AttributeError as exc:  # pragma: no cover
    raise RuntimeError(
        "Rust backend extension is stale. Rebuild it with "
        "`scripts/build_rust_release_extension.sh` so the adapter can detect "
        "the physical CPU core count."
    ) from exc

# Use physical cores as the default for true batched Rust inference. Scalar
# calls still use the one-thread global Rayon pool to avoid per-call dispatch
# overhead and small-kernel overparallelization.
DEFAULT_PREDICT_MANY_THREADS = _PHYSICAL_CPU_COUNT


ReviewCurve = tuple[array, array]
_ProcessResult = float | tuple[float, ReviewCurve]
_ProcessResults = list[float] | list[tuple[float, ReviewCurve]]
_NativeEntityState = tuple[list[Any], list[Any], list[Any]]

_RUST_CHECKPOINT_FORMAT = "rwkv-p-rust-checkpoint-v1"
_RUST_CHECKPOINT_BIN_MAGIC = b"RWKVPCPUBINCHK1"
_RUST_CHECKPOINT_BIN_VERSION = 2
_PROCESS_REVIEW_PAYLOAD_MAGIC = b"RWSRSP01"
_PROCESS_REVIEW_PAYLOAD_HEADER = struct.Struct("<8sQ")
_PROCESS_REVIEW_PAYLOAD_RECORD = struct.Struct("<qqqqqqqqdddqdd")
def _prefer_fully_resident_gpu_process_state(
    reviews: Iterable[ReviewInput] | ReviewBatch,
) -> bool:
    """Use the sharded arena only when the complete input size is known and large."""
    if not isinstance(reviews, Sized):
        return False
    try:
        return len(reviews) >= DEFAULT_GPU_FULLY_RESIDENT_MIN_REVIEWS
    except Exception:
        # A custom iterable's advisory __len__ must not make an otherwise valid
        # streaming process call fail. Unknown inputs retain the working set.
        return False


def _native_review_batches(
    reviews: Iterable[ReviewInput] | ReviewBatch,
    *,
    batch_size: int,
) -> Iterable[ReviewBatch]:
    """Materialize bounded immutable native batches without copying dict rows."""
    batch_size = _validate_batch_size(batch_size)
    if isinstance(reviews, ReviewBatch):
        for start in range(0, len(reviews), batch_size):
            yield reviews._slice(start, min(start + batch_size, len(reviews)))
        return

    pending: list[ReviewInput] = []
    for review in reviews:
        if isinstance(review, dict):
            # Preserve streaming validation: an invalid row must fail before a
            # later generator error can mask it. Values remain unconverted and
            # are normalized exactly once by the native batch parser.
            _require_columns(review, _PROCESS_REQUIRED_COLUMNS)
            row: ReviewInput = review
        else:
            materialized = _coerce_review(review)
            _require_columns(materialized, _PROCESS_REQUIRED_COLUMNS)
            row = materialized
        pending.append(row)
        if len(pending) == batch_size:
            native_batch = ReviewBatch(pending)
            pending = []
            yield native_batch
    if pending:
        yield ReviewBatch(pending)


def _predict_many_batch_size(
    batch_size: int | None,
    *,
    lightning: bool,
    row_count: int,
    gpu: bool = False,
) -> int:
    if batch_size is not None:
        return _validate_batch_size(batch_size)
    if gpu:
        return DEFAULT_GPU_PREDICT_MANY_BATCH_SIZE
    lightning_modules = os.environ.get("RWKV_SRS_PREDICT_MANY_LIGHTNING_MODULES", "5")
    if lightning and lightning_modules != "0":
        return DEFAULT_FAST_PREDICT_MANY_BATCH_SIZE
    return DEFAULT_PREDICT_MANY_BATCH_SIZE


def _native_usize(value: Any, name: str) -> int:
    result = _non_negative_int(value, name)
    if result > _NATIVE_USIZE_MAX:
        raise OverflowError(f"{name} is too large for the native runtime.")
    return result


_DETERMINISTIC_RNNPROCESS_ATTRS = {
    "first_day_offset",
    "prev_row",
    "card_set",
    "card_count",
    "last_new_cards",
    "i",
    "last_i",
    "today",
    "today_reviews",
    "today_new_cards",
    "card2first_day_offset",
    "card2elapsed_days_cumulative",
    "card2elapsed_seconds_cumulative",
    "id_encodings",
    "_day_offset_encoding_cache",
}

_RECURRENT_RNNPROCESS_ATTRS = {
    "card_states",
    "note_states",
    "deck_states",
    "preset_states",
    "global_state",
}


def backend_name() -> str:
    return _native.backend_name()


def _process_many_bulk_enabled() -> bool:
    value = os.environ.get(PROCESS_MANY_BULK_ENV_VAR, "").strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on", "bulk", "bulk_layered"}:
        return True
    raise ValueError(
        f"{PROCESS_MANY_BULK_ENV_VAR} must be one of 1/true/yes/on/bulk/bulk_layered or 0/false/no/off."
    )


def _create_rnn_and_claim_cpu_mode(
    *,
    mode: CpuMode,
    path: str | Path,
    device: Any,
    dtype: Any,
    seed: int | None,
    undo_limit: int,
    runtime_owner_thread: bool,
    restore_path: str | Path | None = None,
    restore_scope: _CheckpointCardScope | None = None,
) -> RustRNNProcess:
    """Construct fallible model/checkpoint state before committing CPU mode.

    Native runtime construction loads the model and optional checkpoint before
    its final CPU warm-up. Serializing the environment override with the claim
    prevents concurrent constructors from observing a different profile and
    lets malformed inputs fail before native's process-wide profile is read.
    """

    global _CPU_MODE_CLAIMED
    with _CPU_MODE_LOCK:
        if _CPU_MODE_CLAIMED is not None and _CPU_MODE_CLAIMED != mode:
            raise RuntimeError(
                "Rust CPU modes cannot be mixed in one Python process. "
                f"This process already initialized {_CPU_MODE_CLAIMED!r}; "
                f"start a fresh interpreter to use {mode!r}."
            )

        previous = os.environ.get(CPU_PROFILE_ENV_VAR)
        os.environ[CPU_PROFILE_ENV_VAR] = mode
        rnn: RustRNNProcess | None = None
        try:
            rnn = RustRNNProcess(
                path=path,
                device=device,
                dtype=dtype,
                seed=seed,
                undo_limit=undo_limit,
                runtime_owner_thread=runtime_owner_thread,
                restore_path=restore_path,
                restore_scope=restore_scope,
            )
            _native.claim_cpu_profile(mode)
            _CPU_MODE_CLAIMED = mode
        except BaseException:
            try:
                if rnn is not None:
                    rnn.close()
            except BaseException:
                # Preserve the constructor/claim failure; this object never
                # became caller-visible, so close is strictly best effort.
                pass
            finally:
                if previous is None:
                    os.environ.pop(CPU_PROFILE_ENV_VAR, None)
                else:
                    os.environ[CPU_PROFILE_ENV_VAR] = previous
            raise

        return rnn


def get_interval(
    curve: ReviewCurve,
    retention_probability: float,
) -> float | None:
    """Return when a curve first falls below `retention_probability`.

    The returned interval is measured in seconds. `None` means the modeled curve
    did not cross the threshold within the supported ahead-curve horizon.
    """
    retention_probability = _validate_retention_probability(retention_probability)
    return _native.curve_interval(
        _curve_values_for_native(curve[0], expected_len=128, name="curve[0]"),
        _curve_values_for_native(curve[1], expected_len=128, name="curve[1]"),
        retention_probability,
    )


def get_probability(
    curve: ReviewCurve,
    elapsed_seconds: float,
) -> float:
    """Return the retention probability from a processed-review curve.

    `elapsed_seconds` is measured from the processed review that produced the
    curve to the future review being predicted. Values below one second follow
    the trained model behavior and clamp to one second internally. Values past
    the trained curve horizon are rejected instead of extrapolated.
    """
    elapsed_seconds = _validate_elapsed_seconds(elapsed_seconds)
    probabilities = _native.predict_curve(
        _curve_matrix_for_native(curve[0]),
        _curve_matrix_for_native(curve[1]),
        [elapsed_seconds],
    )
    if len(probabilities) != 1:
        raise RuntimeError(
            f"Rust curve probability returned {len(probabilities)} values for one input."
        )
    return float(probabilities[0])


def get_probability_many(
    curves: Iterable[ReviewCurve],
    elapsed_seconds: Iterable[float],
) -> list[float]:
    """Return pairwise retention probabilities for processed-review curves.

    Each curve is evaluated at the elapsed-second value in the same position.
    The two iterables must therefore have equal lengths. Values below one
    second follow the trained model behavior and clamp to one second inside the
    native curve evaluator. Values past the trained curve horizon are rejected
    instead of extrapolated.
    """
    curve_values, elapsed_values = _coerce_probability_many_inputs(
        curves,
        elapsed_seconds,
    )
    if not curve_values:
        return []

    ahead_logits: list[Any] = []
    weights: list[Any] = []
    for index, curve in enumerate(curve_values):
        try:
            ahead_component, weight_component = curve
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"curves[{index}] must contain exactly two curve components."
            ) from exc
        ahead_logits.append(
            _curve_row_for_native(
                ahead_component,
                expected_len=128,
                name=f"curves[{index}][0]",
            )
        )
        weights.append(
            _curve_row_for_native(
                weight_component,
                expected_len=128,
                name=f"curves[{index}][1]",
            )
        )

    probabilities = _native.predict_curve(
        ahead_logits,
        weights,
        elapsed_values,
    )
    if len(probabilities) != len(curve_values):
        raise RuntimeError(
            "Rust curve probability returned "
            f"{len(probabilities)} values for {len(curve_values)} inputs."
        )
    return [float(probability) for probability in probabilities]


class RustLivePredictionSession:
    """Thin token facade over Rust-owned state in the parent runtime.

    All candidate rows, predictions, rank storage, refresh scratch buffers, and
    undo deltas remain inside ``NativeRuntime``. This object only validates the
    compact call surface and forwards operations to that runtime's owner thread.
    """

    def __init__(
        self,
        runtime: RWKV_SRS,
        token: int,
        initial_result: LiveRefreshResult,
        *,
        mode: ExecutionMode,
        fallback_mode: CpuMode | None,
    ) -> None:
        self._runtime = runtime
        self._token = int(token)
        self._initial_result = initial_result
        self._generation = int(initial_result.generation)
        self._mode = mode
        self._fallback_mode = fallback_mode
        self._closed = False

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
        """Return the executor currently used by prediction refreshes."""

        self._require_open()
        return self._mode

    @property
    def current_undo_depth(self) -> int:
        self._require_open()
        return self._runtime.current_undo_depth

    def current_selection(
        self,
        *,
        select_limit: int = 2,
        exclude_card_ids: Iterable[int] = (),
    ) -> LiveRefreshResult:
        """Return compact selections from the current rank without predicting."""

        self._require_open()
        value = self._runtime._rnn.live_current_selection(
            self._token,
            select_limit=_non_negative_int(select_limit, "select_limit"),
            exclude_card_ids=_coerce_card_ids(exclude_card_ids),
        )
        result = _refresh_result_from_mapping(value)
        self._generation = result.generation
        return result

    def refresh(
        self,
        *,
        target_timestamp_seconds: float,
        target_day_offset: float,
        select_limit: int = 2,
        exclude_card_ids: Iterable[int] = (),
        exclude_refresh_card_ids: Iterable[int] = (),
        retention_extra: float = 0.0,
    ) -> LiveRefreshResult:
        """Refresh native-selected membership and return only compact selections."""
        self._require_open()
        timestamp = _finite_float(target_timestamp_seconds, "target_timestamp_seconds")
        day = _finite_float(target_day_offset, "target_day_offset")
        select_limit = _non_negative_int(select_limit, "select_limit")
        extra = _finite_float(retention_extra, "retention_extra")
        excluded = _coerce_card_ids(exclude_card_ids)
        refresh_excluded = _coerce_card_ids(exclude_refresh_card_ids)

        def run() -> dict[str, Any]:
            return self._runtime._rnn.live_refresh(
                self._token,
                target_timestamp_seconds=timestamp,
                target_day_offset=day,
                select_limit=select_limit,
                exclude_card_ids=excluded,
                exclude_refresh_card_ids=refresh_excluded,
                retention_extra=extra,
            )

        value = self._run_prediction_with_fallback(run)
        result = _refresh_result_from_mapping(value)
        self._generation = result.generation
        return result

    def reconcile_candidates(
        self,
        candidates: Iterable[LiveCandidateSeed],
        *,
        target_timestamp_seconds: float,
        target_day_offset: float,
        select_limit: int = 2,
        exclude_card_ids: Iterable[int] = (),
        retention_extra: float = 0.0,
    ) -> LiveRefreshResult:
        """Atomically replace, fully predict, and rank the candidate universe.

        Unlike :meth:`replace_candidates`, this operation preserves paired
        answer undo. Undoing across the boundary restores the exact universe
        and rank that existed before the corresponding answer.
        """

        self._require_open()
        seeds = _materialize_native_live_seeds(candidates)
        timestamp = _finite_float(
            target_timestamp_seconds,
            "target_timestamp_seconds",
        )
        day = _finite_float(target_day_offset, "target_day_offset")
        limit = _non_negative_int(select_limit, "select_limit")
        excluded = _coerce_card_ids(exclude_card_ids)
        extra = _finite_float(retention_extra, "retention_extra")

        def run() -> dict[str, Any]:
            return self._runtime._rnn.live_reconcile_candidates(
                self._token,
                seeds,
                target_timestamp_seconds=timestamp,
                target_day_offset=day,
                select_limit=limit,
                exclude_card_ids=excluded,
                retention_extra=extra,
            )

        value = self._run_prediction_with_fallback(run)
        result = _refresh_result_from_mapping(value)
        self._generation = result.generation
        return result

    def reconcile_membership(
        self,
        card_ids: Iterable[int],
        changed_candidates: Iterable[LiveCandidateSeed] = (),
        *,
        target_timestamp_seconds: float,
        target_day_offset: float,
        select_limit: int = 2,
        exclude_card_ids: Iterable[int] = (),
        retention_extra: float = 0.0,
    ) -> LiveRefreshResult:
        """Atomically reconcile membership while reusing unchanged native anchors.

        ``card_ids`` is the complete desired universe in stable transport
        order. Supply seeds only for new candidates or existing candidates
        whose identities, anchors, targets, or tie-breaker changed.
        """

        self._require_open()
        desired_card_ids = _materialize_native_card_ids(card_ids)
        changed = _materialize_native_live_seeds(changed_candidates)
        timestamp = _finite_float(
            target_timestamp_seconds,
            "target_timestamp_seconds",
        )
        day = _finite_float(target_day_offset, "target_day_offset")
        limit = _non_negative_int(select_limit, "select_limit")
        excluded = _coerce_card_ids(exclude_card_ids)
        extra = _finite_float(retention_extra, "retention_extra")

        def run() -> dict[str, Any]:
            return self._runtime._rnn.live_reconcile_membership(
                self._token,
                desired_card_ids,
                changed,
                target_timestamp_seconds=timestamp,
                target_day_offset=day,
                select_limit=limit,
                exclude_card_ids=excluded,
                retention_extra=extra,
            )

        value = self._run_prediction_with_fallback(run)
        result = _refresh_result_from_mapping(value)
        self._generation = result.generation
        return result

    def process_answer(
        self,
        review_row: ReviewInput,
        *,
        requeue_after_prediction: bool = False,
        return_curves: bool = True,
        num_threads: int | None = None,
    ) -> _ProcessResult:
        """Undoably process one answer and update/remove its native candidate.

        Pass ``return_curves=False`` to return only the immediate prediction and
        skip calculation and materialization of the post-answer curve heads.
        """
        self._require_open()
        if not isinstance(requeue_after_prediction, bool):
            raise TypeError("requeue_after_prediction must be a bool.")
        if not isinstance(return_curves, bool):
            raise TypeError("return_curves must be a bool.")
        result, generation = self._runtime._live_process_answer(
            self._token,
            review_row,
            requeue_after_prediction=requeue_after_prediction,
            return_curves=return_curves,
            num_threads=num_threads,
        )
        self._generation = generation
        return result

    def undo_last_process(self) -> int:
        """Undo the latest live answer and all native index changes attached to it."""
        self._require_open()
        remaining, generation = self._runtime._live_undo_last_process(self._token)
        self._generation = generation
        return remaining

    def exclude_card(self, card_id: int) -> int:
        self._require_open()
        self._generation = self._runtime._rnn.live_exclude_card(
            self._token,
            _coerce_card_ids([card_id], "card_id")[0],
        )
        return self._generation

    def include_card(self, card_id: int) -> int:
        self._require_open()
        self._generation = self._runtime._rnn.live_include_card(
            self._token,
            _coerce_card_ids([card_id], "card_id")[0],
        )
        return self._generation

    def remove_candidate(self, card_id: int) -> int:
        """Remove one candidate from the transient live universe."""
        self._require_open()
        self._generation = self._runtime._rnn.live_remove_candidate(
            self._token,
            _coerce_card_ids([card_id], "card_id")[0],
        )
        return self._generation

    def upsert_candidates(self, candidates: Iterable[LiveCandidateSeed]) -> int:
        """Insert or replace seeds; changed candidates become refresh-priority."""
        self._require_open()
        seeds = _materialize_native_live_seeds(candidates)
        self._generation = self._runtime._rnn.live_upsert_candidates(self._token, seeds)
        return self._generation

    def replace_candidates(self, candidates: Iterable[LiveCandidateSeed]) -> int:
        """Atomically replace the transient universe and clear answer undo history."""
        self._require_open()
        seeds = _materialize_native_live_seeds(candidates)
        self._generation = self._runtime._rnn.live_replace_candidates(
            self._token, seeds
        )
        self._runtime._undo_metadata_stack.clear()
        return self._generation

    def candidate(self, card_id: int) -> LiveCandidateSnapshot | None:
        """Return one explicit diagnostic snapshot, or ``None`` if absent."""
        self._require_open()
        value = self._runtime._rnn.live_candidate(
            self._token,
            _coerce_card_ids([card_id], "card_id")[0],
        )
        return None if value is None else _candidate_snapshot_from_mapping(value)

    def snapshot(self) -> tuple[LiveCandidateSnapshot, ...]:
        """Return an explicit full diagnostic snapshot (not a hot-path API)."""
        self._require_open()
        return tuple(
            _candidate_snapshot_from_mapping(value)
            for value in self._runtime._rnn.live_snapshot(self._token)
        )

    def set_retention_extra(self, value: float) -> int:
        self._require_open()
        self._generation = self._runtime._rnn.live_set_retention_extra(
            self._token,
            _finite_float(value, "retention_extra"),
        )
        return self._generation

    def set_mode(self, mode: ExecutionMode) -> int:
        """Change the live prediction executor without replacing session state."""

        self._require_open()
        mode = _validate_execution_mode(mode)
        if mode != "gpu":
            self._runtime.release_gpu()
        self._generation = self._runtime._rnn.live_set_mode(self._token, mode)
        self._mode = mode
        return self._generation

    def profile(self) -> dict[str, Any]:
        """Return optional cumulative and last-refresh native stage timings."""
        self._require_open()
        return dict(self._runtime._rnn.live_profile(self._token))

    def allocation_profile(self) -> dict[str, int]:
        """Return native-owned candidate and reconciliation capacity counters."""

        self._require_open()
        return {
            str(key): int(value)
            for key, value in self._runtime._rnn.live_allocation_profile(
                self._token
            ).items()
        }

    def last_refresh_debug(self) -> dict[str, tuple[int, ...]]:
        """Return explicit membership/transport IDs for diagnostics and tests."""
        self._require_open()
        value = self._runtime._rnn.live_last_refresh_debug(self._token)
        return {
            "membership_card_ids": tuple(
                int(card_id) for card_id in value["membership_card_ids"]
            ),
            "transport_card_ids": tuple(
                int(card_id) for card_id in value["transport_card_ids"]
            ),
        }

    def close(self) -> None:
        """Idempotently drop transient native session state on the runtime owner thread."""
        if self._closed:
            return
        self._runtime._close_live_session(self._token, facade=self)

    def __enter__(self) -> RustLivePredictionSession:
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort finalizer.
        try:
            self.close()
        except Exception:
            pass

    def _mark_closed(self) -> None:
        if not self._closed:
            self._generation += 1
            self._closed = True

    def _run_prediction_with_fallback(self, operation: Any) -> dict[str, Any]:
        try:
            return operation()
        except GpuError as exc:
            if (
                self._mode != "gpu"
                or self._fallback_mode is None
                or not exc.retryable_on_cpu
            ):
                raise
            self.set_mode(self._fallback_mode)
            return operation()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("live prediction session is closed")


class RWKV_SRS:
    """Rust-backed RWKV-SRS inference state.

    The Rust adapter owns the public API directly instead of subclassing the
    internal Torch oracle. Its RNG, scalar outputs, array.array curve outputs,
    and Rust-native binary checkpoints behave independently of whether Torch is
    installed. When ``seed`` is omitted, native deterministic state uses seed
    5489. The generic runtime wheel does not bundle model weights. A downstream
    distribution may install repository-owned weights under
    ``rwkv_srs/pretrained`` for named lookup; direct wheel users pass an
    explicit ``.safetensors`` path through ``model=``.
    """

    supports_native_review_batch_consistency = True

    def __init__(
        self,
        *,
        model: str | Path | None = None,
        checkpoint: str | Path | None = None,
        cards: Iterable[ReviewInput] | None = None,
        device: Any = "cpu",
        dtype: Any = "float32",
        seed: int | None = None,
        undo_limit: int = DEFAULT_UNDO_LIMIT,
        runtime_owner_thread: bool = False,
        cpu_mode: CpuMode = "oracle",
    ) -> None:
        if (model is None) == (checkpoint is None):
            raise ValueError("Pass exactly one of model=... or checkpoint=....")
        if cards is not None and checkpoint is None:
            raise ValueError("cards= is only valid when loading checkpoint=....")

        self.undo_limit = _validate_undo_limit(undo_limit)
        self.runtime_owner_thread = _validate_runtime_owner_thread(runtime_owner_thread)
        self.cpu_mode = _validate_cpu_mode(cpu_mode)
        self.device = _normalize_device(device)
        self.dtype = _normalize_dtype(dtype)
        self.model_id: str | None = None
        self.model_path: Path | None = None
        self.last_review_id: Any = None
        self.processed_review_count: int = 0
        self._history_digest: str | None = _INITIAL_HISTORY_DIGEST
        self._seed = seed
        self._undo_metadata_stack: deque[dict[str, Any]] = deque()
        self._state_scope: _CheckpointCardScope | None = None
        self._backing_checkpoint_path: Path | None = None
        self._live_session_ref: (
            weakref.ReferenceType[RustLivePredictionSession] | None
        ) = None

        if checkpoint is not None:
            requested_scope = (
                None if cards is None else _CheckpointCardScope.from_cards(cards)
            )
            self._load_checkpoint(checkpoint, requested_scope=requested_scope)
            return

        self.model_id, self.model_path = _resolve_rust_model(model)
        self._rnn = _create_rnn_and_claim_cpu_mode(
            mode=self.cpu_mode,
            path=self.model_path,
            device=self.device,
            dtype=self.dtype,
            seed=seed,
            undo_limit=self.undo_limit,
            runtime_owner_thread=self.runtime_owner_thread,
        )

    def close(self) -> None:
        """Release native Rust runtime resources held by this object."""
        active = self._active_live_session()
        if active is not None:
            try:
                self._close_live_session(active._token, facade=active)
            except Exception:
                active._mark_closed()
        rnn = getattr(self, "_rnn", None)
        if rnn is not None and hasattr(rnn, "close"):
            rnn.close()

    def _active_live_session(self) -> RustLivePredictionSession | None:
        reference = getattr(self, "_live_session_ref", None)
        if reference is None:
            return None
        live = reference()
        if live is None or live._closed:
            self._live_session_ref = None
            return None
        return live

    def _require_no_active_live_session(self, operation: str) -> None:
        if self._active_live_session() is not None:
            raise RuntimeError(
                f"{operation} cannot mutate the runtime while a live prediction session is "
                "active; use the live-session operation or close it first."
            )

    def _close_live_session(
        self,
        token: int,
        *,
        facade: RustLivePredictionSession,
    ) -> None:
        if facade._closed:
            return
        try:
            self._rnn.close_live_prediction_session(token)
        finally:
            self._undo_metadata_stack.clear()
            facade._mark_closed()
            current = self._active_live_session()
            if current is facade:
                self._live_session_ref = None

    def initialize_gpu(self, operation: GpuOperation = "predict") -> dict[str, Any]:
        """Initialize and retain the selected prediction or processing GPU cache."""
        return self._rnn.initialize_gpu(_validate_gpu_operation(operation))

    def gpu_available(self, operation: GpuOperation = "predict") -> bool:
        """Return whether this model can initialize for the selected GPU operation.

        A successful check retains the selected operation's GPU cache for later calls.
        """
        operation = _validate_gpu_operation(operation)
        try:
            self.initialize_gpu(operation)
        except GpuError:
            return False
        return True

    def gpu_profile(self, operation: GpuOperation = "predict") -> dict[str, Any] | None:
        """Return cumulative measurements for one GPU operation cache."""
        return self._rnn.gpu_profile(_validate_gpu_operation(operation))

    def synchronize_gpu(self) -> int:
        """Materialize deferred process state, wait for GPU work, and return nanoseconds."""
        return self._rnn.synchronize_gpu()

    def release_gpu(self) -> bool:
        """Release model/state GPU buffers while retaining canonical CPU state."""
        return self._rnn.release_gpu()

    def __enter__(self) -> RWKV_SRS:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort finalizer.
        try:
            self.close()
        except Exception:
            pass

    def predict(
        self,
        review: ReviewInput,
        *,
        num_threads: int | None = None,
    ) -> float:
        """Return the immediate success probability without modifying state."""
        row = _coerce_review(review)
        _require_columns(row, _PREDICT_REQUIRED_COLUMNS)
        self._require_loaded_state(row)
        return self._predict_row(
            row,
            num_threads=_native_num_threads(
                num_threads,
                default=DEFAULT_PREDICT_THREADS,
            ),
        )

    def predict_many(
        self,
        reviews: Iterable[ReviewInput] | PredictionBatch,
        *,
        batch_size: int | None = None,
        num_threads: int | None = None,
        mode: ExecutionMode | None = None,
        fallback_mode: CpuMode | None = None,
    ) -> list[float]:
        """Return immediate success probabilities without modifying state.

        ``mode="oracle"`` uses the normal CPU executor, ``mode="fast"`` uses
        the prediction-only approximate executor formerly named lightning, and
        ``mode="gpu"`` uses the immutable GPU prediction backend. Omitting the
        mode preserves the normal executor; shared low-level CPU kernels still
        follow the constructor's process-wide ``cpu_mode``. GPU errors remain
        strict by default. Supplying ``fallback_mode="oracle"`` or ``"fast"``
        releases the failed GPU cache and retries the immutable call on CPU.
        A prebuilt :class:`~rwkv_srs.PredictionBatch` bypasses repeated Python
        mapping materialization and native field parsing on every call.
        """
        mode = "oracle" if mode is None else _validate_execution_mode(mode)
        fallback_mode = (
            None if fallback_mode is None else _validate_cpu_mode(fallback_mode)
        )
        if fallback_mode is not None and mode != "gpu":
            raise ValueError("fallback_mode is only valid when mode='gpu'.")
        lightning = mode == "fast"
        gpu = mode == "gpu"
        requested_batch_size = batch_size
        rows: list[dict[str, Any]] | PredictionBatch
        if isinstance(reviews, PredictionBatch):
            rows = reviews
        else:
            rows = [
                review if gpu and type(review) is dict else _coerce_review(review)
                for review in reviews
            ]
        batch_size = _predict_many_batch_size(
            batch_size,
            lightning=lightning,
            gpu=gpu,
            row_count=len(rows),
        )
        if not gpu and not isinstance(rows, PredictionBatch):
            for row in rows:
                _require_columns(row, _PREDICT_REQUIRED_COLUMNS)
                self._require_loaded_state(row)
        resolved_threads = _native_num_threads(
            num_threads,
            default=DEFAULT_PREDICT_MANY_THREADS,
        )
        if not gpu:
            return self._rnn.imm_predict_many(
                rows,
                batch_size=batch_size,
                num_threads=resolved_threads,
                lightning=lightning,
                gpu=False,
            )
        try:
            return self._rnn.imm_predict_many(
                rows,
                batch_size=batch_size,
                num_threads=resolved_threads,
                lightning=False,
                gpu=True,
            )
        except GpuError as exc:
            if fallback_mode is None or not exc.retryable_on_cpu:
                raise
            self.release_gpu()
            # The GPU path validates selective scope natively and may stop at
            # the first failed batch. Validate every row before retrying the
            # complete immutable call on CPU so fallback cannot manufacture
            # state omitted from a selectively loaded checkpoint.
            if not isinstance(rows, PredictionBatch):
                for row in rows:
                    _require_columns(row, _PREDICT_REQUIRED_COLUMNS)
                    self._require_loaded_state(row)
            fallback_batch_size = _predict_many_batch_size(
                requested_batch_size,
                lightning=fallback_mode == "fast",
                gpu=False,
                row_count=len(rows),
            )
            return self._rnn.imm_predict_many(
                rows,
                batch_size=fallback_batch_size,
                num_threads=resolved_threads,
                lightning=fallback_mode == "fast",
                gpu=False,
            )

    def predict_many_f32(
        self,
        reviews: Iterable[ReviewInput] | PredictionBatch,
        *,
        batch_size: int | None = None,
        num_threads: int | None = None,
        mode: Literal["fast", "gpu"] = "gpu",
    ) -> array:
        """Return GPU or CPU Fast probabilities in a compact ``array('f')``.

        This Rust-only transport runs the same immutable prediction
        implementation as ``predict_many(..., mode=mode)`` but avoids
        allocating one Python float object per row. ``mode`` supports
        ``"gpu"`` (the backward-compatible default) and ``"fast"``. Both
        executors produce FP32 probabilities before widening them for the
        ordinary Python list contract, so the compact result preserves those
        exact FP32 values. ``"oracle"`` is intentionally unsupported.

        Callers that need a normal ``list[float]`` should continue using
        :meth:`predict_many`; converting this array back into a list merely
        moves the boxing cost into caller code. The array directly supports the
        Python buffer protocol for consumers such as NumPy. GPU failures use
        the same :class:`GpuError` hierarchy as ``predict_many``; callers may
        catch one and explicitly retry the immutable rows with ``mode="fast"``.

        The Torch backend intentionally does not implement this transport: it
        remains the frozen correctness oracle rather than a production
        performance surface.
        """
        mode = _validate_execution_mode(mode)
        if mode == "oracle":
            raise ValueError(
                "predict_many_f32 mode must be 'fast' or 'gpu'; "
                "the Oracle result is available through predict_many()."
            )
        fast = mode == "fast"
        gpu = mode == "gpu"
        rows: list[dict[str, Any]] | PredictionBatch
        if isinstance(reviews, PredictionBatch):
            rows = reviews
        else:
            rows = [
                review if gpu and type(review) is dict else _coerce_review(review)
                for review in reviews
            ]
        batch_size = _predict_many_batch_size(
            batch_size,
            lightning=fast,
            gpu=gpu,
            row_count=len(rows),
        )
        if fast and not isinstance(rows, PredictionBatch):
            for row in rows:
                _require_columns(row, _PREDICT_REQUIRED_COLUMNS)
                self._require_loaded_state(row)
        return self._rnn.imm_predict_many_f32(
            rows,
            batch_size=batch_size,
            num_threads=_native_num_threads(
                num_threads,
                default=DEFAULT_PREDICT_MANY_THREADS,
            ),
            fast=fast,
        )

    def predict_many_live_session(
        self,
        candidates: Iterable[LiveCandidateSeed],
        *,
        initial_target_timestamp_seconds: float,
        initial_target_day_offset: float,
        order: LiveOrder = "retrievability_ascending",
        mode: ExecutionMode = "gpu",
        batch_size: int | None = None,
        refresh_limit: int = DEFAULT_GPU_PREDICT_MANY_BATCH_SIZE,
        num_threads: int | None = None,
        profiling: bool = False,
        initial_select_limit: int = 2,
        fallback_mode: CpuMode | None = None,
    ) -> RustLivePredictionSession:
        """Create or return this runtime's single Rust-owned live session.

        Initial predictions are computed natively once so the first refresh can
        choose membership from a real retrievability rank. Subsequent refreshes
        construct and apply up to ``refresh_limit`` rows without materializing
        those rows or probabilities in Python. With ``mode="gpu"``, an explicit
        CPU ``fallback_mode`` retries recoverable prediction failures and leaves
        the session in that CPU mode; strict GPU behavior remains the default.
        """
        active = self._active_live_session()
        if active is not None:
            return active
        order = _validate_live_order(order)
        mode = _validate_execution_mode(mode)
        fallback_mode = (
            None if fallback_mode is None else _validate_cpu_mode(fallback_mode)
        )
        if fallback_mode is not None and mode != "gpu":
            raise ValueError("fallback_mode is only valid when mode='gpu'.")
        refresh_limit = _positive_int(refresh_limit, "refresh_limit")
        if not isinstance(profiling, bool):
            raise TypeError("profiling must be a bool.")
        timestamp = _finite_float(
            initial_target_timestamp_seconds,
            "initial_target_timestamp_seconds",
        )
        day = _finite_float(initial_target_day_offset, "initial_target_day_offset")
        seeds = _materialize_native_live_seeds(candidates)
        resolved_batch_size = _predict_many_batch_size(
            batch_size,
            lightning=mode == "fast",
            gpu=mode == "gpu",
            row_count=len(seeds),
        )
        resolved_threads = _native_num_threads(
            num_threads,
            default=DEFAULT_PREDICT_MANY_THREADS,
        )
        initial_select_limit = _native_usize(
            initial_select_limit,
            "initial_select_limit",
        )
        effective_mode = mode
        try:
            token, initial_value = self._rnn.create_live_prediction_session(
                seeds,
                initial_target_timestamp_seconds=timestamp,
                initial_target_day_offset=day,
                order=order,
                mode=effective_mode,
                batch_size=resolved_batch_size,
                refresh_limit=refresh_limit,
                num_threads=resolved_threads,
                profiling=profiling,
                initial_select_limit=initial_select_limit,
            )
        except GpuError as exc:
            if fallback_mode is None or not exc.retryable_on_cpu:
                raise
            self.release_gpu()
            effective_mode = fallback_mode
            resolved_batch_size = _predict_many_batch_size(
                batch_size,
                lightning=effective_mode == "fast",
                gpu=False,
                row_count=len(seeds),
            )
            token, initial_value = self._rnn.create_live_prediction_session(
                seeds,
                initial_target_timestamp_seconds=timestamp,
                initial_target_day_offset=day,
                order=order,
                mode=effective_mode,
                batch_size=resolved_batch_size,
                refresh_limit=refresh_limit,
                num_threads=resolved_threads,
                profiling=profiling,
                initial_select_limit=initial_select_limit,
            )
        try:
            initial_result = _refresh_result_from_mapping(initial_value)
            live = RustLivePredictionSession(
                self,
                token,
                initial_result,
                mode=effective_mode,
                fallback_mode=fallback_mode,
            )
            live_reference = weakref.ref(live)
            self._rnn.finalize_live_prediction_session(token)
        except BaseException:
            try:
                self._rnn.abort_live_prediction_session(token)
            except BaseException:
                # Preserve the conversion/finalization error. Native finalize
                # has no fallible work after its commit point, so an abort error
                # here means either that finalization already committed or that
                # the original failure made the owner runtime unavailable.
                pass
            raise

        # Finalization atomically installs native candidate state and clears its
        # undo frames. These remaining Python operations do not allocate.
        self._undo_metadata_stack.clear()
        self._live_session_ref = live_reference
        return live

    def process(
        self,
        review: ReviewInput,
        *,
        return_curves: bool = True,
        num_threads: int | None = None,
    ) -> _ProcessResult:
        """Return the immediate probability, then update inference state.

        Pass ``return_curves=False`` to return only the probability and skip
        calculation and materialization of the post-review curve heads.
        """
        self._require_no_active_live_session("process")
        if not isinstance(return_curves, bool):
            raise TypeError("return_curves must be a bool.")
        row = _coerce_review(review)
        _require_columns(row, _PROCESS_REQUIRED_COLUMNS)
        self._require_loaded_state(row)
        next_history_digest = None
        if self._history_digest is not None:
            next_history_digest = _chain_digest(self._history_digest, row)

        self._clear_undo_history()
        result = self._rnn.process_row(
            row,
            return_curves=return_curves,
            num_threads=_native_num_threads(
                num_threads,
                default=DEFAULT_PROCESS_THREADS,
            ),
        )
        self.last_review_id = _normalize_review_id(row["review_id"])
        if next_history_digest is not None:
            self._history_digest = next_history_digest
            self.processed_review_count += 1
        return result

    def undoable_process(
        self,
        review: ReviewInput,
        *,
        return_curves: bool = True,
        num_threads: int | None = None,
    ) -> _ProcessResult:
        """Process one review and record enough Rust state to undo it.

        Pass ``return_curves=False`` to return only the probability and skip
        calculation and materialization of the post-review curve heads.
        """
        self._require_no_active_live_session("undoable_process")
        if not isinstance(return_curves, bool):
            raise TypeError("return_curves must be a bool.")
        if self.undo_limit == 0:
            raise UndoUnavailableError(
                "undoable_process is disabled because undo_limit=0."
            )
        row = _coerce_review(review)
        _require_columns(row, _PROCESS_REQUIRED_COLUMNS)
        self._require_loaded_state(row)
        previous_metadata = self._undo_metadata_snapshot()
        next_history_digest = None
        if self._history_digest is not None:
            next_history_digest = _chain_digest(self._history_digest, row)

        native_depth_before = self._rnn.undo_depth()
        try:
            result = self._rnn.undoable_process_row(
                row,
                return_curves=return_curves,
                num_threads=_native_num_threads(
                    num_threads,
                    default=DEFAULT_PROCESS_THREADS,
                ),
            )
            self.last_review_id = _normalize_review_id(row["review_id"])
            if next_history_digest is not None:
                self._history_digest = next_history_digest
                self.processed_review_count += 1
            self._undo_metadata_stack.append(previous_metadata)
            self._trim_undo_metadata_stack()
            return result
        except Exception:
            if self._rnn.undo_depth() > native_depth_before:
                self._rnn.undo_last_process()
            raise

    def undo_last_process(self) -> int:
        """Undo the latest `undoable_process()` call and return remaining depth."""
        self._require_no_active_live_session("undo_last_process")
        if not self._undo_metadata_stack:
            raise UndoUnavailableError("No undoable process is available.")
        try:
            remaining = self._rnn.undo_last_process()
        except ValueError as exc:
            self._undo_metadata_stack.clear()
            raise UndoUnavailableError("No undoable process is available.") from exc

        previous_metadata = self._undo_metadata_stack.pop()
        self._restore_undo_metadata(previous_metadata)
        if remaining != len(self._undo_metadata_stack):
            self._undo_metadata_stack.clear()
            self._rnn.clear_undo_history()
            raise RuntimeError(
                "Rust undo history and Python metadata history diverged."
            )
        return remaining

    @property
    def current_undo_depth(self) -> int:
        """Return how many `undoable_process()` calls can currently be undone."""
        depth = self._rnn.undo_depth()
        if depth != len(self._undo_metadata_stack):
            raise RuntimeError(
                "Rust undo history and Python metadata history diverged."
            )
        return depth

    def undo_depth(self) -> int:
        """Compatibility alias for `current_undo_depth`."""
        return self.current_undo_depth

    def process_many(
        self,
        reviews: Iterable[ReviewInput] | ReviewBatch,
        *,
        batch_size: int | None = None,
        return_curves: bool = False,
        num_threads: int | None = None,
        mode: ExecutionMode | None = None,
        fallback_mode: CpuMode | None = None,
    ) -> list[float] | list[tuple[float, ReviewCurve]]:
        """Sequentially process review rows and update inference state.

        Inputs are consumed in ordered batches so long histories do not need to
        be materialized before native processing begins. This is a normal
        mutating process path, so calling it clears undo history immediately
        even if there are no rows or later validation rejects the input.

        With ``mode="gpu"``, this uses the FP32 associative-scan
        executor and lazily materializes its flat FP32 state before later CPU
        operations. For a sized input of at least 200,000 rows, the executor
        keeps all process state in a sharded GPU arena for the duration of the
        replay; shorter inputs and streaming iterables retain the lower-memory
        working set. An omitted mode uses the constructor's ``cpu_mode``. The
        CPU mode is process-wide because native configuration is snapshotted;
        an explicit CPU mode must therefore match the constructor. The legacy
        RWKV_SRS_PROCESS_MANY_BULK environment flag still routes ``oracle``
        runtimes through the bulk path for validation.

        GPU errors remain strict by default. An explicit ``fallback_mode``
        synchronizes any safely committed GPU prefix, processes the exact
        remaining suffix on CPU, and keeps later input batches on CPU. Without
        fallback, ``GpuError.committed_rows`` and ``partial_results`` expose the
        complete successful prefix of this public call. A process fallback mode
        must match the runtime's constructor ``cpu_mode``.
        """
        self._require_no_active_live_session("process_many")
        mode = self.cpu_mode if mode is None else _validate_execution_mode(mode)
        fallback_mode = (
            None if fallback_mode is None else _validate_cpu_mode(fallback_mode)
        )
        if fallback_mode is not None and mode != "gpu":
            raise ValueError("fallback_mode is only valid when mode='gpu'.")
        if fallback_mode is not None and fallback_mode != self.cpu_mode:
            raise RuntimeError(
                f"process_many(fallback_mode={fallback_mode!r}) requires a runtime "
                f"constructed with cpu_mode={fallback_mode!r}; this runtime uses "
                f"cpu_mode={self.cpu_mode!r}."
            )
        if mode != "gpu" and mode != self.cpu_mode:
            raise RuntimeError(
                f"process_many(mode={mode!r}) cannot run on a runtime constructed "
                f"with cpu_mode={self.cpu_mode!r}; Rust CPU modes are process-wide. "
                f"Start a fresh interpreter and construct RWKV_SRS(cpu_mode={mode!r})."
            )
        if mode == "gpu":
            return self._process_many_gpu_scan_impl(
                reviews,
                batch_size=batch_size,
                return_curves=return_curves,
                num_threads=num_threads,
                fallback_mode=fallback_mode,
            )
        if mode == "fast":
            return self._process_many_pipeline_impl(
                reviews,
                batch_size=batch_size,
                return_curves=return_curves,
                num_threads=num_threads,
            )
        bulk_layered = _process_many_bulk_enabled()
        return self._process_many_impl(
            reviews,
            batch_size=batch_size,
            return_curves=return_curves,
            num_threads=num_threads,
            bulk_layered=bulk_layered,
        )

    def build_state_only(
        self,
        reviews: Iterable[ReviewInput] | ReviewBatch,
        *,
        batch_size: int | None = None,
        num_threads: int | None = None,
        mode: ExecutionMode | None = None,
    ) -> int:
        """Build canonical state from answers without calculating predictions.

        This consumes the same ordered review rows and updates the same history
        metadata as :meth:`process_many`, but returns only the number of
        committed reviews. It supports the runtime's configured ``oracle`` or
        ``fast`` CPU mode and a dedicated ``gpu`` path that omits query rows,
        prediction heads, curves, and prediction readback.

        A sized GPU input of at least 200,000 rows automatically uses sharded,
        fully resident process state. This reduces per-batch host transfer and
        materialization work at the cost of higher transient GPU memory. It is
        synchronized before checkpointing or CPU work and is never serialized
        as a separate checkpoint representation. Streaming iterables keep the
        existing bounded working-set path because their complete size is not
        known in advance.

        Native work is transactional within each public batch. If a later
        input batch fails, earlier successful batches remain committed and the
        runtime metadata continues to describe that committed prefix.
        """
        self._require_no_active_live_session("build_state_only")
        mode = self.cpu_mode if mode is None else _validate_execution_mode(mode)
        if mode == "gpu":
            return self._build_state_only_gpu_scan_impl(
                reviews,
                batch_size=batch_size,
                num_threads=num_threads,
            )
        if mode != self.cpu_mode:
            raise RuntimeError(
                f"build_state_only(mode={mode!r}) cannot run on a runtime constructed "
                f"with cpu_mode={self.cpu_mode!r}; Rust CPU modes are process-wide. "
                f"Start a fresh interpreter and construct RWKV_SRS(cpu_mode={mode!r})."
            )
        if batch_size is None:
            batch_size = DEFAULT_PROCESS_MANY_BATCH_SIZE
        resolved_threads = _native_num_threads(
            num_threads,
            default=DEFAULT_PROCESS_MANY_THREADS,
        )

        self._clear_undo_history()
        processed_count = 0
        for rows in _native_review_batches(
            reviews,
            batch_size=batch_size,
        ):
            committed_count = self._build_state_only_rows(
                rows,
                fast=mode == "fast",
                num_threads=resolved_threads,
            )
            if committed_count != len(rows):
                self._record_processed_batch(rows, count=committed_count)
                processed_count += committed_count
                raise RuntimeError(
                    "Native build_state_only committed "
                    f"{committed_count} rows for a {len(rows)}-row batch; "
                    f"{processed_count} rows are committed in this call."
                )
            processed_count += committed_count
        return processed_count

    def _build_state_only_gpu_scan_impl(
        self,
        reviews: Iterable[ReviewInput] | ReviewBatch,
        *,
        batch_size: int | None,
        num_threads: int | None,
    ) -> int:
        fully_resident_state = _prefer_fully_resident_gpu_process_state(reviews)
        if batch_size is None:
            batch_size = DEFAULT_GPU_BUILD_STATE_ONLY_BATCH_SIZE
        resolved_threads = _native_num_threads(
            num_threads,
            default=DEFAULT_PROCESS_MANY_THREADS,
        )
        self._clear_undo_history()
        processed_count = 0
        batches = iter(_native_review_batches(reviews, batch_size=batch_size))
        while True:
            try:
                rows = next(batches)
            except StopIteration:
                break
            except Exception as exc:
                if processed_count == 0:
                    raise
                raise GpuProcessError(
                    "build_state_only could not materialize the next input batch "
                    f"after committing {processed_count} rows: {exc}",
                    operation="process",
                    phase="input",
                    committed_rows=processed_count,
                    state_recoverable=True,
                    retryable_on_cpu=False,
                ) from exc
            try:
                committed_count = self._build_state_only_gpu_scan_rows(
                    rows,
                    num_threads=resolved_threads,
                    defer_cpu_state=True,
                    fully_resident_state=fully_resident_state,
                )
            except GpuError as exc:
                exc._prepend_process_progress(processed_count, ())
                raise
            if committed_count != len(rows):
                processed_count += committed_count
                raise GpuProcessError(
                    "Native GPU build_state_only committed "
                    f"{committed_count} rows for a {len(rows)}-row batch; "
                    f"{processed_count} rows are committed in this call.",
                    operation="process",
                    phase="process",
                    committed_rows=processed_count,
                    state_recoverable=True,
                    retryable_on_cpu=False,
                )
            processed_count += committed_count
        return processed_count

    def _process_many_gpu_scan_impl(
        self,
        reviews: Iterable[ReviewInput] | ReviewBatch,
        *,
        batch_size: int | None,
        return_curves: bool,
        num_threads: int | None,
        fallback_mode: CpuMode | None,
    ) -> list[float] | list[tuple[float, ReviewCurve]]:
        fully_resident_state = _prefer_fully_resident_gpu_process_state(reviews)
        requested_batch_size = batch_size
        if batch_size is None:
            batch_size = DEFAULT_GPU_PROCESS_MANY_BATCH_SIZE
        cpu_batch_size = (
            DEFAULT_PROCESS_MANY_BATCH_SIZE
            if requested_batch_size is None
            else batch_size
        )
        resolved_threads = _native_num_threads(
            num_threads,
            default=DEFAULT_PROCESS_MANY_THREADS,
        )
        fallback_bulk_layered = (
            _process_many_bulk_enabled() if fallback_mode == "oracle" else False
        )
        self._clear_undo_history()
        results: list[_ProcessResult] = []
        using_cpu_fallback = False
        batches = iter(
            _native_review_batches(
                reviews,
                batch_size=batch_size,
            )
        )
        while True:
            try:
                rows = next(batches)
            except StopIteration:
                break
            except Exception as exc:
                if not results:
                    raise
                phase = "cpu_fallback" if using_cpu_fallback else "input"
                raise GpuProcessError(
                    "process_many could not materialize the next input batch "
                    f"after committing a prefix: {exc}",
                    operation="process",
                    phase=phase,
                    committed_rows=len(results),
                    partial_results=results,
                    state_recoverable=True,
                    retryable_on_cpu=False,
                ) from exc
            if using_cpu_fallback:
                self._append_cpu_fallback_rows(
                    results,
                    rows,
                    batch_size=cpu_batch_size,
                    mode=fallback_mode,
                    return_curves=return_curves,
                    num_threads=resolved_threads,
                    bulk_layered=fallback_bulk_layered,
                )
                continue
            try:
                results.extend(
                    self._process_many_gpu_scan_rows(
                        rows,
                        return_curves=return_curves,
                        num_threads=resolved_threads,
                        defer_cpu_state=True,
                        fully_resident_state=fully_resident_state,
                    )
                )
            except GpuError as exc:
                completed_before_batch = len(results)
                if (
                    fallback_mode is None
                    or not exc.retryable_on_cpu
                    or not exc.state_recoverable
                ):
                    exc._prepend_process_progress(completed_before_batch, results)
                    raise
                results.extend(cast(tuple[_ProcessResult, ...], exc.partial_results))
                try:
                    self.release_gpu()
                except GpuError as synchronization_error:
                    synchronization_error.state_recoverable = False
                    synchronization_error.retryable_on_cpu = False
                    synchronization_error._prepend_process_progress(
                        len(results), results
                    )
                    raise synchronization_error from exc
                self._append_cpu_fallback_rows(
                    results,
                    rows._slice(exc.committed_rows, len(rows)),
                    batch_size=cpu_batch_size,
                    mode=fallback_mode,
                    return_curves=return_curves,
                    num_threads=resolved_threads,
                    bulk_layered=fallback_bulk_layered,
                )
                using_cpu_fallback = True
        return cast(_ProcessResults, results)

    def _append_cpu_fallback_rows(
        self,
        results: list[_ProcessResult],
        rows: ReviewBatch,
        *,
        batch_size: int,
        mode: CpuMode | None,
        return_curves: bool,
        num_threads: int | None,
        bulk_layered: bool,
    ) -> None:
        if mode is None:
            raise RuntimeError("CPU fallback mode is missing.")
        for start in range(0, len(rows), batch_size):
            chunk = rows._slice(start, min(start + batch_size, len(rows)))
            try:
                if mode == "fast":
                    chunk_results = self._process_many_pipeline_rows(
                        chunk,
                        return_curves=return_curves,
                        num_threads=num_threads,
                    )
                else:
                    chunk_results = self._process_many_rows(
                        chunk,
                        return_curves=return_curves,
                        num_threads=num_threads,
                        bulk_layered=bulk_layered,
                    )
            except Exception as exc:
                if not results:
                    raise
                raise GpuProcessError(
                    f"CPU continuation after a GPU failure did not complete: {exc}",
                    operation="process",
                    phase="cpu_fallback",
                    committed_rows=len(results),
                    partial_results=results,
                    state_recoverable=True,
                    retryable_on_cpu=False,
                ) from exc
            results.extend(cast(list[_ProcessResult], chunk_results))

    def _process_many_bulk_layered(
        self,
        reviews: Iterable[ReviewInput] | ReviewBatch,
        *,
        batch_size: int | None = None,
        return_curves: bool = False,
        num_threads: int | None = None,
    ) -> list[float] | list[tuple[float, ReviewCurve]]:
        """Experimental Phase 7 process_many() path used by benchmarks/tests."""
        return self._process_many_impl(
            reviews,
            batch_size=batch_size,
            return_curves=return_curves,
            num_threads=num_threads,
            bulk_layered=True,
        )

    def _process_many_impl(
        self,
        reviews: Iterable[ReviewInput] | ReviewBatch,
        *,
        batch_size: int | None,
        return_curves: bool,
        num_threads: int | None,
        bulk_layered: bool,
    ) -> list[float] | list[tuple[float, ReviewCurve]]:
        if batch_size is None:
            batch_size = DEFAULT_PROCESS_MANY_BATCH_SIZE
        resolved_threads = _native_num_threads(
            num_threads,
            default=DEFAULT_PROCESS_MANY_THREADS,
        )
        self._clear_undo_history()
        results: list[_ProcessResult] = []
        for rows in _native_review_batches(
            reviews,
            batch_size=batch_size,
        ):
            results.extend(
                self._process_many_rows(
                    rows,
                    return_curves=return_curves,
                    num_threads=resolved_threads,
                    bulk_layered=bulk_layered,
                )
            )
        return cast(_ProcessResults, results)

    def _process_many_pipeline_impl(
        self,
        reviews: Iterable[ReviewInput] | ReviewBatch,
        *,
        batch_size: int | None,
        return_curves: bool,
        num_threads: int | None,
    ) -> list[float] | list[tuple[float, ReviewCurve]]:
        if batch_size is None:
            batch_size = DEFAULT_PROCESS_MANY_BATCH_SIZE
        resolved_threads = _native_num_threads(
            num_threads,
            default=DEFAULT_PROCESS_MANY_THREADS,
        )
        self._clear_undo_history()
        results: list[_ProcessResult] = []
        for rows in _native_review_batches(
            reviews,
            batch_size=batch_size,
        ):
            results.extend(
                self._process_many_pipeline_rows(
                    rows,
                    return_curves=return_curves,
                    num_threads=resolved_threads,
                )
            )
        return cast(_ProcessResults, results)

    def get_interval(
        self,
        curve: ReviewCurve,
        retention_probability: float,
    ) -> float | None:
        """Return when a processed-review curve first falls below a threshold."""
        return get_interval(curve, retention_probability)

    def get_probability(
        self,
        curve: ReviewCurve,
        elapsed_seconds: float,
    ) -> float:
        """Return the curve probability at an elapsed-second interval."""
        return get_probability(curve, elapsed_seconds)

    def get_probability_many(
        self,
        curves: Iterable[ReviewCurve],
        elapsed_seconds: Iterable[float],
    ) -> list[float]:
        """Return pairwise probabilities for curves and elapsed intervals."""
        return get_probability_many(curves, elapsed_seconds)

    def _predict_row(self, row: dict[str, Any], *, num_threads: int | None) -> float:
        return self._rnn.imm_predict_probability(row, num_threads=num_threads)

    def _live_process_answer(
        self,
        token: int,
        review: ReviewInput,
        *,
        requeue_after_prediction: bool,
        return_curves: bool,
        num_threads: int | None,
    ) -> tuple[_ProcessResult, int]:
        if self.undo_limit == 0:
            raise UndoUnavailableError(
                "live process_answer is disabled because undo_limit=0."
            )
        row = _coerce_review(review)
        _require_columns(row, _PROCESS_REQUIRED_COLUMNS)
        self._require_loaded_state(row)
        previous_metadata = self._undo_metadata_snapshot()
        next_history_digest = None
        if self._history_digest is not None:
            next_history_digest = _chain_digest(self._history_digest, row)
        native_depth_before = self._rnn.undo_depth()
        try:
            result, generation = self._rnn.live_process_answer(
                token,
                row,
                requeue_after_prediction=requeue_after_prediction,
                return_curves=return_curves,
                num_threads=_native_num_threads(
                    num_threads,
                    default=DEFAULT_PROCESS_THREADS,
                ),
            )
            self.last_review_id = _normalize_review_id(row["review_id"])
            if next_history_digest is not None:
                self._history_digest = next_history_digest
                self.processed_review_count += 1
            self._undo_metadata_stack.append(previous_metadata)
            self._trim_undo_metadata_stack()
            return result, generation
        except Exception:
            if self._rnn.undo_depth() > native_depth_before:
                self._rnn.live_undo_last_process(token)
            raise

    def _live_undo_last_process(self, token: int) -> tuple[int, int]:
        if not self._undo_metadata_stack:
            raise UndoUnavailableError("No live-session process is available to undo.")
        try:
            remaining, generation = self._rnn.live_undo_last_process(token)
        except ValueError as exc:
            self._undo_metadata_stack.clear()
            raise UndoUnavailableError(
                "No live-session process is available to undo."
            ) from exc
        previous_metadata = self._undo_metadata_stack.pop()
        self._restore_undo_metadata(previous_metadata)
        if remaining != len(self._undo_metadata_stack):
            self._undo_metadata_stack.clear()
            raise RuntimeError(
                "Rust live undo history and Python metadata history diverged."
            )
        return remaining, generation

    def _process_many_rows(
        self,
        rows: ReviewBatch,
        *,
        return_curves: bool,
        num_threads: int | None,
        bulk_layered: bool = False,
    ) -> list[float] | list[tuple[float, ReviewCurve]]:
        prepared_metadata = self._prepare_processed_batch(rows)
        results = self._rnn.process_batch(
            rows,
            return_curves=return_curves,
            num_threads=num_threads,
            packed=not bulk_layered,
            bulk_layered=bulk_layered,
        )
        self._commit_processed_batch(rows, prepared_metadata=prepared_metadata)
        return results

    def _process_many_pipeline_rows(
        self,
        rows: ReviewBatch,
        *,
        return_curves: bool,
        num_threads: int | None,
    ) -> list[float] | list[tuple[float, ReviewCurve]]:
        prepared_metadata = self._prepare_processed_batch(rows)
        results = self._rnn.process_batch_pipeline(
            rows,
            return_curves=return_curves,
            num_threads=num_threads,
        )
        self._commit_processed_batch(rows, prepared_metadata=prepared_metadata)
        return results

    def _build_state_only_rows(
        self,
        rows: ReviewBatch,
        *,
        fast: bool,
        num_threads: int | None,
    ) -> int:
        prepared_metadata = self._prepare_processed_batch(rows)
        committed_count = self._rnn.build_state_only_batch(
            rows,
            fast=fast,
            num_threads=num_threads,
        )
        if committed_count == len(rows):
            self._commit_processed_batch(rows, prepared_metadata=prepared_metadata)
        return committed_count

    def _build_state_only_gpu_scan_rows(
        self,
        rows: ReviewBatch,
        *,
        num_threads: int | None,
        defer_cpu_state: bool,
        fully_resident_state: bool,
    ) -> int:
        prepared_metadata = self._prepare_processed_batch(rows)
        try:
            committed_count = self._rnn.build_state_only_batch_gpu_scan(
                rows,
                num_threads=num_threads,
                defer_cpu_state=defer_cpu_state,
                fully_resident_state=fully_resident_state,
            )
        except GpuError as exc:
            self._record_processed_batch(rows, count=exc.committed_rows)
            raise
        except BaseException:
            committed_rows = self._rnn.take_gpu_process_committed_rows()
            self._record_processed_batch(rows, count=committed_rows)
            raise
        if committed_count == len(rows):
            self._commit_processed_batch(rows, prepared_metadata=prepared_metadata)
        else:
            self._record_processed_batch(rows, count=committed_count)
        return committed_count

    def _process_many_gpu_scan_rows(
        self,
        rows: ReviewBatch,
        *,
        return_curves: bool,
        num_threads: int | None,
        defer_cpu_state: bool,
        fully_resident_state: bool,
    ) -> list[float] | list[tuple[float, ReviewCurve]]:
        prepared_metadata = self._prepare_processed_batch(rows)
        try:
            results = self._rnn.process_batch_gpu_scan(
                rows,
                return_curves=return_curves,
                num_threads=num_threads,
                defer_cpu_state=defer_cpu_state,
                fully_resident_state=fully_resident_state,
            )
        except GpuError as exc:
            self._record_processed_batch(rows, count=exc.committed_rows)
            raise
        except BaseException:
            committed_rows = self._rnn.take_gpu_process_committed_rows()
            self._record_processed_batch(rows, count=committed_rows)
            raise
        self._commit_processed_batch(rows, prepared_metadata=prepared_metadata)
        return results

    def _prepare_processed_batch(
        self,
        rows: ReviewBatch,
        *,
        count: int | None = None,
    ) -> tuple[str | None, int | None]:
        return rows._history_advance(self._history_digest, count)

    def _commit_processed_batch(
        self,
        rows: ReviewBatch,
        *,
        count: int | None = None,
        prepared_metadata: tuple[str | None, int | None] | None = None,
    ) -> None:
        committed_count = len(rows) if count is None else int(count)
        if committed_count < 0 or committed_count > len(rows):
            raise RuntimeError(
                f"Native process reported {committed_count} committed rows for a "
                f"{len(rows)}-row ReviewBatch."
            )
        if committed_count == 0:
            return
        if prepared_metadata is None:
            prepared_metadata = self._prepare_processed_batch(
                rows,
                count=committed_count,
            )
        digest, last_review_id = prepared_metadata
        if self._history_digest is not None:
            if digest is None:
                raise RuntimeError("Native ReviewBatch omitted a tracked history digest.")
            self._history_digest = digest
            self.processed_review_count += committed_count
        self.last_review_id = last_review_id

    def _record_processed_batch(
        self,
        rows: ReviewBatch,
        *,
        count: int | None = None,
    ) -> None:
        self._commit_processed_batch(rows, count=count)

    def _record_processed_rows(
        self,
        rows: list[dict[str, Any]] | ReviewBatch,
    ) -> None:
        if isinstance(rows, ReviewBatch):
            self._record_processed_batch(rows)
            return
        track_history = self._history_digest is not None
        for row in rows:
            if track_history:
                assert self._history_digest is not None
                self._history_digest = _chain_digest(self._history_digest, row)
                self.processed_review_count += 1
            self.last_review_id = _normalize_review_id(row["review_id"])

    def check_history_consistency(
        self,
        reviews: Iterable[ReviewInput] | ReviewBatch,
    ) -> bool:
        """Return whether reviews match the prefix represented by this state.

        A :class:`~rwkv_srs.ReviewBatch` keeps parsing, canonicalization, and
        chained SHA-256 work in Rust. Ordinary iterables retain the established
        backend-neutral Python path.
        """
        if self._history_digest is None:
            raise ValueError("This checkpoint does not contain a history fingerprint.")
        expected = self._history_fingerprint_dict()
        assert expected is not None
        if isinstance(reviews, ReviewBatch):
            return reviews._matches_history_fingerprint(
                digest=str(expected["digest"]),
                processed_review_count=int(expected["processed_review_count"]),
                last_review_id=expected["last_review_id"],
            )
        actual = _fingerprint_reviews(reviews, limit=self.processed_review_count)
        return (
            actual["digest"] == expected["digest"]
            and actual["processed_review_count"] == expected["processed_review_count"]
            and actual["last_review_id"] == expected["last_review_id"]
        )

    def save_checkpoint(self, path: str | Path) -> None:
        """Write a Rust-native checkpoint containing resumable state."""
        path = Path(path)
        if path.suffix != ".bin":
            raise ValueError("Rust backend checkpoints must use the .bin extension.")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_checkpoint_bin(path)

    def expected_checkpoint_size(
        self,
        *,
        card_count: int,
        note_count: int,
        deck_count: int,
        preset_count: int,
    ) -> int:
        """Return the expected size in bytes of the next ``.bin`` save.

        Counts are distinct normalized identities that will have recurrent
        state, not review-row counts. The calculation includes the current
        binary-v2 metadata, deterministic per-card records, ID encodings, RNG
        state, all four recurrent maps, the global recurrent state, and format
        indexes. It excludes transient undo, live-session, and GPU state.

        A nonempty canonical history stores at least one normalized note, deck,
        and preset identity. Missing note IDs normalize per card; missing deck
        and preset IDs share one placeholder apiece.
        """
        counts = {
            "card_count": _native_usize(card_count, "card_count"),
            "note_count": _native_usize(note_count, "note_count"),
            "deck_count": _native_usize(deck_count, "deck_count"),
            "preset_count": _native_usize(preset_count, "preset_count"),
        }
        metadata_json = self._checkpoint_metadata_json()
        return self._rnn.expected_checkpoint_bin_size(
            len(metadata_json),
            **counts,
        )

    def _write_checkpoint_bin(self, path: str | Path) -> None:
        metadata_json = self._checkpoint_metadata_json()
        if self._backing_checkpoint_path is None:
            with _atomic_output_path(Path(path)) as temporary_path:
                self._rnn.write_checkpoint_bin(temporary_path, metadata_json)
        else:
            if self._state_scope is None:
                raise RuntimeError(
                    "A backed selective runtime is missing its state scope."
                )
            with _atomic_output_path(Path(path)) as temporary_path:
                self._rnn.write_merged_checkpoint_bin(
                    self._backing_checkpoint_path,
                    temporary_path,
                    metadata_json,
                    self._state_scope,
                )
        if self._backing_checkpoint_path is not None:
            self._backing_checkpoint_path = Path(path).resolve()

    def _checkpoint_metadata_json(self) -> bytes:
        metadata = self._checkpoint_metadata_dict(
            storage_format="rwkv-p-rust-checkpoint-bin-v2",
        )
        if self._backing_checkpoint_path is not None:
            metadata["state_scope"] = None
        return json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def _checkpoint_metadata_dict(
        self,
        *,
        storage_format: str | None = None,
    ) -> dict[str, Any]:
        metadata = {
            "format": _RUST_CHECKPOINT_FORMAT,
            "version": _CHECKPOINT_VERSION,
            "model_id": self.model_id,
            "model_path": str(self.model_path) if self.model_id is None else None,
            "last_review_id": self.last_review_id,
            "processed_review_count": self.processed_review_count,
            "history_fingerprint": self._history_fingerprint_dict(),
            "state_scope": (
                None if self._state_scope is None else self._state_scope.to_metadata()
            ),
        }
        if storage_format is not None:
            metadata["storage_format"] = storage_format
        return metadata

    def _history_fingerprint_dict(self) -> dict[str, Any] | None:
        if self._history_digest is None:
            return None
        return {
            "version": _HISTORY_FINGERPRINT_VERSION,
            "algorithm": _HISTORY_FINGERPRINT_ALGORITHM,
            "canonicalization": _HISTORY_FINGERPRINT_CANONICALIZATION,
            "fields": list(_HISTORY_FINGERPRINT_FIELDS),
            "last_review_id": self.last_review_id,
            "processed_review_count": self.processed_review_count,
            "digest": self._history_digest,
        }

    def _undo_metadata_snapshot(self) -> dict[str, Any]:
        return {
            "last_review_id": self.last_review_id,
            "processed_review_count": self.processed_review_count,
            "history_digest": self._history_digest,
        }

    def _restore_undo_metadata(self, metadata: dict[str, Any]) -> None:
        self.last_review_id = metadata["last_review_id"]
        self.processed_review_count = int(metadata["processed_review_count"])
        self._history_digest = metadata["history_digest"]

    def _clear_undo_history(self) -> None:
        had_undo_history = bool(self._undo_metadata_stack)
        self._undo_metadata_stack.clear()
        # Avoid an unconditional Python->Rust boundary call on hot process()
        # paths. Public undo APIs verify Python/native depth agreement, so this
        # only clears native history when Python knows there is history to drop.
        if had_undo_history and hasattr(self, "_rnn"):
            self._rnn.clear_undo_history()

    def _trim_undo_metadata_stack(self) -> None:
        if self.undo_limit > 0 and len(self._undo_metadata_stack) > self.undo_limit:
            while len(self._undo_metadata_stack) > self.undo_limit:
                self._undo_metadata_stack.popleft()

    def _require_loaded_state(self, row: ReviewInput) -> None:
        if self._state_scope is not None:
            self._state_scope.require_review(row)

    def _load_checkpoint(
        self,
        path: str | Path,
        *,
        requested_scope: _CheckpointCardScope | None,
    ) -> None:
        path = Path(path)
        checkpoint = _load_checkpoint_dict(path)
        stored_scope = _checkpoint_scope_from_metadata(checkpoint.get("state_scope"))
        if (
            requested_scope is not None
            and stored_scope is not None
            and not requested_scope.is_subset_of(stored_scope)
        ):
            raise ValueError(
                "Requested cards are outside this checkpoint's saved state_scope."
            )
        self._state_scope = (
            requested_scope if requested_scope is not None else stored_scope
        )
        if requested_scope is not None and stored_scope is None:
            self._backing_checkpoint_path = path.resolve()

        self.model_id = checkpoint.get("model_id")
        model_path_value = checkpoint.get("model_path")
        if self.model_id is not None:
            self.model_path = _get_safetensors_model_path(self.model_id)
        elif model_path_value is not None:
            self.model_path = _require_safetensors_model_path(model_path_value)
        else:
            raise ValueError("Checkpoint is missing model_id/model_path metadata.")

        last_review_id = checkpoint.get(
            "last_review_id",
            checkpoint.get("last_review_th", None),
        )
        processed_review_count = int(checkpoint.get("processed_review_count", 0))
        fingerprint = checkpoint.get("history_fingerprint")
        if fingerprint is None:
            history_digest = None
        else:
            _validate_fingerprint_metadata(fingerprint)
            history_digest = fingerprint["digest"]
            processed_review_count = int(fingerprint["processed_review_count"])
            last_review_id = fingerprint["last_review_id"]

        self._rnn = _create_rnn_and_claim_cpu_mode(
            mode=self.cpu_mode,
            path=self.model_path,
            device=self.device,
            dtype=self.dtype,
            seed=self._seed,
            undo_limit=self.undo_limit,
            runtime_owner_thread=self.runtime_owner_thread,
            restore_path=path,
            restore_scope=self._state_scope,
        )

        self.last_review_id = last_review_id
        self.processed_review_count = processed_review_count
        self._history_digest = history_digest

        self._undo_metadata_stack.clear()


class _OwnerThreadError:
    __slots__ = ("exc_type", "message")

    def __init__(self, exc_type: type[BaseException], message: str) -> None:
        self.exc_type = exc_type
        self.message = message

    @classmethod
    def from_exception(cls, exc: BaseException) -> _OwnerThreadError:
        return cls(exc.__class__, str(exc))

    def raise_on_caller(self) -> None:
        try:
            exc = self.exc_type(self.message)
        except Exception:
            exc = RuntimeError(self.message)
        raise exc from None


class _OwnerThreadResult:
    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value


class _OwnerThreadNativeRuntime:
    """Own `_native.NativeRuntime` on one thread and proxy calls synchronously."""

    def __init__(
        self,
        checkpoint_path: str,
        torch_seed: int,
        undo_limit: int,
        restore_path: str | None,
        card_ids: list[int] | None,
        note_ids: list[int] | None,
        deck_ids: list[int] | None,
        preset_ids: list[int] | None,
    ) -> None:
        self._calls: queue.Queue = queue.Queue()
        self._closed = False
        self._lock = threading.Lock()
        ready: queue.Queue = queue.Queue(maxsize=1)
        self._thread = threading.Thread(
            target=_owner_thread_runtime_main,
            args=(
                self._calls,
                checkpoint_path,
                torch_seed,
                undo_limit,
                restore_path,
                card_ids,
                note_ids,
                deck_ids,
                preset_ids,
                ready,
            ),
            name="rwkv-srs-native-runtime",
            daemon=True,
        )
        # Drain cycles containing direct, thread-bound native runtimes on the
        # caller before this worker can trigger Python's process-wide cyclic GC.
        # Collecting them on the worker would violate PyO3's unsendable drop
        # contract even though they are unrelated to this proxy.
        gc.collect()
        self._thread.start()
        outcome = ready.get()
        if isinstance(outcome, _OwnerThreadError):
            with self._lock:
                self._closed = True
            outcome.raise_on_caller()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        def call(*args, **kwargs):
            return self._call(name, args, kwargs)

        return call

    def close(self, timeout: float = 5.0) -> None:
        response: queue.Queue = queue.Queue(maxsize=1)
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._calls.put((None, (), {}, response))
        try:
            outcome = response.get(timeout=timeout)
        except queue.Empty:
            return
        if isinstance(outcome, _OwnerThreadError):
            outcome.raise_on_caller()
        self._thread.join(timeout=timeout)

    def __del__(self) -> None:  # pragma: no cover - best-effort finalizer.
        try:
            self.close(timeout=1.0)
        except Exception:
            pass

    def _call(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        response: queue.Queue = queue.Queue(maxsize=1)
        with self._lock:
            if self._closed:
                raise RuntimeError("Native runtime owner thread is closed.")
            self._calls.put((name, args, kwargs, response))
        outcome = response.get()
        if isinstance(outcome, _OwnerThreadError):
            outcome.raise_on_caller()
        return outcome.value


def _owner_thread_runtime_main(
    calls: queue.Queue,
    checkpoint_path: str,
    torch_seed: int,
    undo_limit: int,
    restore_path: str | None,
    card_ids: list[int] | None,
    note_ids: list[int] | None,
    deck_ids: list[int] | None,
    preset_ids: list[int] | None,
    ready: queue.Queue,
) -> None:
    runtime = None
    try:
        runtime = _native.NativeRuntime(
            checkpoint_path,
            torch_seed,
            undo_limit,
            restore_path,
            card_ids,
            note_ids,
            deck_ids,
            preset_ids,
        )
    except BaseException as exc:
        ready.put(_OwnerThreadError.from_exception(exc))
        return

    ready.put(_OwnerThreadResult(None))
    try:
        while True:
            request = calls.get()
            name = None
            args = ()
            kwargs = {}
            response = None
            method = None
            try:
                name, args, kwargs, response = request
                if name is None:
                    runtime = None
                    response.put(_OwnerThreadResult(None))
                    return
                try:
                    assert runtime is not None
                    method = getattr(runtime, name)
                    response.put(_OwnerThreadResult(method(*args, **kwargs)))
                except BaseException as exc:
                    response.put(_OwnerThreadError.from_exception(exc))
            except BaseException as exc:
                if response is None:
                    raise
                response.put(_OwnerThreadError.from_exception(exc))
            finally:
                request = None
                name = None
                args = ()
                kwargs = {}
                response = None
                method = None
    finally:
        runtime = None


class RustRNNProcess:
    """Low-level native bridge; public callers should construct :class:`RWKV_SRS`.

    Direct construction follows the already-established native CPU profile (or
    initializes it from ``RWKV_SRS_CPU_PROFILE`` during warm-up). It deliberately
    does not participate in the adapter's Python-side mode claim; mixing this
    internal bridge with public runtimes is therefore unsupported.
    """

    def __init__(
        self,
        path: str | Path,
        device: Any = "cpu",
        dtype: Any = "float32",
        seed: int | None = None,
        undo_limit: int = DEFAULT_UNDO_LIMIT,
        runtime_owner_thread: bool = False,
        restore_path: str | Path | None = None,
        restore_scope: _CheckpointCardScope | None = None,
    ) -> None:
        undo_limit = _validate_undo_limit(undo_limit)
        object.__setattr__(self, "device", _normalize_device(device))
        object.__setattr__(self, "dtype", _normalize_dtype(dtype))
        object.__setattr__(self, "undo_limit", undo_limit)
        object.__setattr__(
            self,
            "runtime_owner_thread",
            _validate_runtime_owner_thread(runtime_owner_thread),
        )
        object.__setattr__(
            self,
            "safetensors_path",
            _require_safetensors_model_path(path),
        )
        runtime_cls = (
            _OwnerThreadNativeRuntime
            if self.runtime_owner_thread
            else _native.NativeRuntime
        )
        restore_args: tuple[
            str | None,
            list[int] | None,
            list[int] | None,
            list[int] | None,
            list[int] | None,
        ] = (
            (None, None, None, None, None)
            if restore_path is None
            else (
                str(Path(restore_path)),
                None if restore_scope is None else sorted(restore_scope.card_ids),
                None if restore_scope is None else sorted(restore_scope.note_ids),
                None if restore_scope is None else sorted(restore_scope.deck_ids),
                None if restore_scope is None else sorted(restore_scope.preset_ids),
            )
        )
        runtime = runtime_cls(
            str(self.safetensors_path),
            DEFAULT_RUST_SEED if seed is None else int(seed),
            undo_limit,
            *restore_args,
        )
        object.__setattr__(self, "_native_runtime", runtime)
        default_predict_many_threads = _native_num_threads(
            None,
            default=DEFAULT_PREDICT_MANY_THREADS,
        )
        if default_predict_many_threads is not None:
            runtime.warm_thread_pool(default_predict_many_threads)

    def close(self) -> None:
        runtime = self.__dict__.get("_native_runtime")
        if runtime is None:
            return
        if hasattr(runtime, "close"):
            runtime.close()
            return
        object.__setattr__(self, "_native_runtime", None)

    def __del__(self) -> None:  # pragma: no cover - best-effort finalizer.
        try:
            self.close()
        except Exception:
            pass

    def __getattr__(self, name: str) -> Any:
        if name in _DETERMINISTIC_RNNPROCESS_ATTRS:
            return self._deterministic_attr(name)
        if name in _RECURRENT_RNNPROCESS_ATTRS:
            return self._recurrent_attr(name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if (
            name in _DETERMINISTIC_RNNPROCESS_ATTRS
            and "_native_runtime" in self.__dict__
        ):
            self._set_deterministic_attr(name, value)
            return
        if name in _RECURRENT_RNNPROCESS_ATTRS and "_native_runtime" in self.__dict__:
            self._set_recurrent_attr(name, value)
            return
        object.__setattr__(self, name, value)

    def imm_predict(self, row: dict[str, Any]) -> Any:
        return _ScalarFloat(self.imm_predict_probability(row))

    def imm_predict_probability(
        self,
        row: dict[str, Any],
        *,
        num_threads: int | None = None,
    ) -> float:
        self.synchronize_gpu_process_state()
        return float(self._native_runtime.predict_probability(row, num_threads))

    def warm_predict_path(self) -> None:
        self._native_runtime.warm_predict_path()

    def initialize_gpu(self, operation: GpuOperation = "predict") -> dict[str, Any]:
        operation = _validate_gpu_operation(operation)
        try:
            if operation == "process":
                return dict(self._native_runtime.initialize_gpu_process())
            self.synchronize_gpu_process_state()
            return dict(self._native_runtime.initialize_gpu())
        except BaseException as exc:
            translated = _translate_native_gpu_error(
                exc,
                operation=operation,
                phase="initialize",
            )
            if translated is None:
                raise
            raise translated from exc

    def gpu_available(self, operation: GpuOperation = "predict") -> bool:
        operation = _validate_gpu_operation(operation)
        try:
            self.initialize_gpu(operation)
        except GpuError:
            return False
        return True

    def gpu_profile(self, operation: GpuOperation = "predict") -> dict[str, Any] | None:
        operation = _validate_gpu_operation(operation)
        profile = (
            self._native_runtime.gpu_process_profile()
            if operation == "process"
            else self._native_runtime.gpu_profile()
        )
        return None if profile is None else dict(profile)

    def synchronize_gpu(self) -> int:
        process_ns = self.synchronize_gpu_process_state()
        try:
            return process_ns + int(self._native_runtime.synchronize_gpu())
        except BaseException as exc:
            translated = _translate_native_gpu_error(
                exc,
                operation="predict",
                phase="synchronize",
                state_recoverable=True,
            )
            if translated is None:
                raise
            raise translated from exc

    def release_gpu(self) -> bool:
        self.synchronize_gpu_process_state()
        return bool(self._native_runtime.release_gpu())

    def imm_predict_many(
        self,
        rows: list[dict[str, Any]] | PredictionBatch,
        *,
        batch_size: int = DEFAULT_PREDICT_MANY_BATCH_SIZE,
        num_threads: int | None = DEFAULT_PREDICT_MANY_THREADS,
        lightning: bool = False,
        gpu: bool = False,
    ) -> list[float]:
        if not isinstance(lightning, bool):
            raise TypeError("lightning must be a bool.")
        if not isinstance(gpu, bool):
            raise TypeError("gpu must be a bool.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.synchronize_gpu_process_state()
        native_batch = (
            rows._native_batch if isinstance(rows, PredictionBatch) else None
        )
        if gpu:
            try:
                if native_batch is None:
                    predictions = self._native_runtime.predict_reviews_gpu(
                        rows,
                        batch_size,
                        num_threads,
                    )
                else:
                    predictions = self._native_runtime.predict_review_batch_gpu(
                        native_batch,
                        batch_size,
                        num_threads,
                    )
            except BaseException as exc:
                translated = _translate_native_gpu_error(
                    exc,
                    operation="predict",
                    phase="predict",
                    state_recoverable=True,
                )
                if translated is None:
                    raise
                raise translated from exc
        else:
            if native_batch is None:
                predictions = self._native_runtime.predict_reviews(
                    rows,
                    batch_size,
                    num_threads,
                    lightning,
                )
            else:
                predictions = self._native_runtime.predict_review_batch(
                    native_batch,
                    batch_size,
                    num_threads,
                    lightning,
                )
        return [float(prediction) for prediction in predictions]

    def imm_predict_many_f32(
        self,
        rows: list[dict[str, Any]] | PredictionBatch,
        *,
        batch_size: int = DEFAULT_GPU_PREDICT_MANY_BATCH_SIZE,
        num_threads: int | None = DEFAULT_PREDICT_MANY_THREADS,
        fast: bool = False,
    ) -> array:
        if not isinstance(fast, bool):
            raise TypeError("fast must be a bool.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.synchronize_gpu_process_state()
        native_batch = (
            rows._native_batch if isinstance(rows, PredictionBatch) else None
        )
        if fast:
            if native_batch is None:
                encoded = self._native_runtime.predict_reviews_fast_f32(
                    rows,
                    batch_size,
                    num_threads,
                )
            else:
                encoded = self._native_runtime.predict_review_batch_fast_f32(
                    native_batch,
                    batch_size,
                    num_threads,
                )
        else:
            try:
                if native_batch is None:
                    encoded = self._native_runtime.predict_reviews_gpu_f32(
                        rows,
                        batch_size,
                        num_threads,
                    )
                else:
                    encoded = self._native_runtime.predict_review_batch_gpu_f32(
                        native_batch,
                        batch_size,
                        num_threads,
                    )
            except BaseException as exc:
                translated = _translate_native_gpu_error(
                    exc,
                    operation="predict",
                    phase="predict",
                    state_recoverable=True,
                )
                if translated is None:
                    raise
                raise translated from exc
        predictions = array("f")
        if predictions.itemsize != 4:  # pragma: no cover - exotic C ABI.
            raise RuntimeError("predict_many_f32 requires 32-bit C float storage.")
        predictions.frombytes(encoded)
        if len(predictions) != len(rows):
            raise RuntimeError(
                "Rust compact predict returned "
                f"{len(predictions)} probabilities for {len(rows)} rows."
            )
        return predictions

    def predict_batch_fast_transport_profiled(
        self,
        rows: PredictionBatch,
        *,
        batch_size: int = DEFAULT_FAST_PREDICT_MANY_BATCH_SIZE,
        num_threads: int | None = DEFAULT_PREDICT_MANY_THREADS,
        compact: bool,
    ) -> tuple[list[float] | array, dict[str, Any]]:
        """Profile list versus compact transport around the same Fast body."""
        if not isinstance(rows, PredictionBatch):
            raise TypeError("rows must be a PredictionBatch.")
        if not isinstance(compact, bool):
            raise TypeError("compact must be a bool.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        total_start = time.perf_counter_ns()
        self.synchronize_gpu_process_state()
        native_start = time.perf_counter_ns()
        if compact:
            encoded, profile = (
                self._native_runtime.predict_review_batch_fast_f32_transport_profiled(
                    rows._native_batch,
                    batch_size,
                    num_threads,
                )
            )
        else:
            encoded, profile = (
                self._native_runtime.predict_review_batch_fast_list_transport_profiled(
                    rows._native_batch,
                    batch_size,
                    num_threads,
                )
            )
        native_call_ns = time.perf_counter_ns() - native_start

        adapter_result_start = time.perf_counter_ns()
        if compact:
            predictions: list[float] | array = array("f")
            if predictions.itemsize != 4:  # pragma: no cover - exotic C ABI.
                raise RuntimeError("predict_many_f32 requires 32-bit C float storage.")
            predictions.frombytes(encoded)
        else:
            predictions = encoded
        adapter_result_ns = time.perf_counter_ns() - adapter_result_start
        if len(predictions) != len(rows):
            raise RuntimeError(
                "Rust profiled Fast predict returned "
                f"{len(predictions)} probabilities for {len(rows)} rows."
            )

        native_profile_total_ns = int(profile["total_ns"])
        profile["python_adapter"] = {
            "total_ns": time.perf_counter_ns() - total_start,
            "native_call_ns": native_call_ns,
            "native_result_boundary_residual_ns": max(
                0,
                native_call_ns - native_profile_total_ns,
            ),
            "binding_result_construction_ns": int(
                profile["binding_result_construction_ns"]
            ),
            "adapter_result_construction_ns": adapter_result_ns,
            "prediction_rows": len(predictions),
            "transport": "f32" if compact else "list",
        }
        return predictions, profile

    def create_live_prediction_session(
        self,
        candidates: list[LiveCandidateSeed],
        *,
        initial_target_timestamp_seconds: float,
        initial_target_day_offset: float,
        order: LiveOrder,
        mode: ExecutionMode,
        batch_size: int,
        refresh_limit: int,
        num_threads: int | None,
        profiling: bool,
        initial_select_limit: int,
    ) -> tuple[int, dict[str, Any]]:
        self.synchronize_gpu_process_state()
        try:
            token, initial_result = self._native_runtime.create_live_prediction_session(
                candidates,
                initial_target_timestamp_seconds,
                initial_target_day_offset,
                order,
                mode,
                batch_size,
                refresh_limit,
                num_threads,
                profiling,
                initial_select_limit,
            )
        except BaseException as exc:
            translated = _translate_native_gpu_error(
                exc,
                operation="predict",
                phase="live_initialize",
                state_recoverable=True,
            )
            if translated is None:
                raise
            raise translated from exc
        try:
            materialized_result = dict(initial_result)
        except BaseException:
            try:
                self._native_runtime.abort_live_prediction_session(token)
            except BaseException:
                pass
            raise
        return int(token), materialized_result

    def finalize_live_prediction_session(self, token: int) -> None:
        self._native_runtime.finalize_live_prediction_session(token)

    def abort_live_prediction_session(self, token: int) -> bool:
        return bool(self._native_runtime.abort_live_prediction_session(token))

    def live_current_selection(
        self,
        token: int,
        *,
        select_limit: int,
        exclude_card_ids: list[int],
    ) -> dict[str, Any]:
        return dict(
            self._native_runtime.live_current_selection(
                token,
                select_limit,
                exclude_card_ids,
            )
        )

    def live_refresh(
        self,
        token: int,
        *,
        target_timestamp_seconds: float,
        target_day_offset: float,
        select_limit: int,
        exclude_card_ids: list[int],
        exclude_refresh_card_ids: list[int],
        retention_extra: float,
    ) -> dict[str, Any]:
        try:
            return dict(
                self._native_runtime.live_refresh(
                    token,
                    target_timestamp_seconds,
                    target_day_offset,
                    select_limit,
                    exclude_card_ids,
                    exclude_refresh_card_ids,
                    retention_extra,
                )
            )
        except BaseException as exc:
            translated = _translate_native_gpu_error(
                exc,
                operation="predict",
                phase="live_refresh",
                state_recoverable=True,
            )
            if translated is None:
                raise
            raise translated from exc

    def live_process_answer(
        self,
        token: int,
        row: dict[str, Any],
        *,
        requeue_after_prediction: bool,
        return_curves: bool,
        num_threads: int | None,
    ) -> tuple[_ProcessResult, int]:
        self.synchronize_gpu_process_state()
        depth_before = self.undo_depth()
        try:
            prediction, ahead, w, generation = self._native_runtime.live_process_answer(
                token,
                row,
                requeue_after_prediction,
                num_threads,
                return_curves,
            )
            if not return_curves:
                assert ahead is None
                assert w is None
                return float(prediction), int(generation)
            assert ahead is not None
            assert w is not None
            result = float(prediction), (_curve_array(ahead), _curve_array(w))
            return result, int(generation)
        except Exception:
            if self.undo_depth() > depth_before:
                self._native_runtime.live_undo_last_process(token)
            raise

    def live_undo_last_process(self, token: int) -> tuple[int, int]:
        self.synchronize_gpu_process_state()
        remaining, generation = self._native_runtime.live_undo_last_process(token)
        return int(remaining), int(generation)

    def live_exclude_card(self, token: int, card_id: int) -> int:
        return int(self._native_runtime.live_exclude_card(token, card_id))

    def live_include_card(self, token: int, card_id: int) -> int:
        return int(self._native_runtime.live_include_card(token, card_id))

    def live_remove_candidate(self, token: int, card_id: int) -> int:
        return int(self._native_runtime.live_remove_candidate(token, card_id))

    def live_upsert_candidates(
        self,
        token: int,
        candidates: list[LiveCandidateSeed],
    ) -> int:
        return int(self._native_runtime.live_upsert_candidates(token, candidates))

    def live_replace_candidates(
        self,
        token: int,
        candidates: list[LiveCandidateSeed],
    ) -> int:
        return int(self._native_runtime.live_replace_candidates(token, candidates))

    def live_reconcile_candidates(
        self,
        token: int,
        candidates: list[LiveCandidateSeed],
        *,
        target_timestamp_seconds: float,
        target_day_offset: float,
        select_limit: int,
        exclude_card_ids: list[int],
        retention_extra: float,
    ) -> dict[str, Any]:
        try:
            return dict(
                self._native_runtime.live_reconcile_candidates(
                    token,
                    candidates,
                    target_timestamp_seconds,
                    target_day_offset,
                    select_limit,
                    exclude_card_ids,
                    retention_extra,
                )
            )
        except BaseException as exc:
            translated = _translate_native_gpu_error(
                exc,
                operation="predict",
                phase="live_reconcile",
                state_recoverable=True,
            )
            if translated is None:
                raise
            raise translated from exc

    def live_reconcile_membership(
        self,
        token: int,
        card_ids: list[Any],
        changed_candidates: list[LiveCandidateSeed],
        *,
        target_timestamp_seconds: float,
        target_day_offset: float,
        select_limit: int,
        exclude_card_ids: list[int],
        retention_extra: float,
    ) -> dict[str, Any]:
        try:
            return dict(
                self._native_runtime.live_reconcile_membership(
                    token,
                    card_ids,
                    changed_candidates,
                    target_timestamp_seconds,
                    target_day_offset,
                    select_limit,
                    exclude_card_ids,
                    retention_extra,
                )
            )
        except BaseException as exc:
            translated = _translate_native_gpu_error(
                exc,
                operation="predict",
                phase="live_reconcile",
                state_recoverable=True,
            )
            if translated is None:
                raise
            raise translated from exc

    def live_candidate(self, token: int, card_id: int) -> dict[str, Any] | None:
        value = self._native_runtime.live_candidate(token, card_id)
        return None if value is None else dict(value)

    def live_snapshot(self, token: int) -> list[dict[str, Any]]:
        return [dict(value) for value in self._native_runtime.live_snapshot(token)]

    def live_set_retention_extra(self, token: int, value: float) -> int:
        return int(self._native_runtime.live_set_retention_extra(token, value))

    def live_set_mode(self, token: int, mode: ExecutionMode) -> int:
        return int(self._native_runtime.live_set_mode(token, mode))

    def live_profile(self, token: int) -> dict[str, Any]:
        value = dict(self._native_runtime.live_profile(token))
        value["last"] = dict(value["last"])
        value["cumulative"] = dict(value["cumulative"])
        return value

    def live_allocation_profile(self, token: int) -> dict[str, int]:
        return {
            str(key): int(value)
            for key, value in dict(
                self._native_runtime.live_allocation_profile(token)
            ).items()
        }

    def live_last_refresh_debug(self, token: int) -> dict[str, Any]:
        return dict(self._native_runtime.live_last_refresh_debug(token))

    def close_live_prediction_session(self, token: int) -> bool:
        return bool(self._native_runtime.close_live_prediction_session(token))

    def process_row(
        self,
        row: dict[str, Any],
        *,
        return_curves: bool = True,
        num_threads: int | None = None,
    ) -> _ProcessResult:
        result = self.process_rows(
            [row],
            return_curves=return_curves,
            num_threads=num_threads,
        )
        assert result
        return result[0]

    def undoable_process_row(
        self,
        row: dict[str, Any],
        *,
        return_curves: bool = True,
        num_threads: int | None = None,
    ) -> _ProcessResult:
        if self.undo_limit == 0:
            raise UndoUnavailableError(
                "undoable_process is disabled because undo_limit=0."
            )
        self.synchronize_gpu_process_state()
        undo_depth_before = self.undo_depth()
        try:
            prediction, ahead, w = self._native_runtime.undoable_process_review(
                row,
                return_curves,
                num_threads,
            )
            if not return_curves:
                assert ahead is None
                assert w is None
                return float(prediction)
            assert ahead is not None
            assert w is not None
            return float(prediction), (_curve_array(ahead), _curve_array(w))
        except Exception:
            if self.undo_depth() > undo_depth_before:
                self._native_runtime.undo_last_process()
            raise

    def process_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        return_curves: bool,
        num_threads: int | None = None,
        packed: bool = False,
        bulk_reference: bool = False,
        bulk_layered: bool = False,
    ):
        if not rows:
            return []
        mode_count = sum(
            bool(value) for value in (packed, bulk_reference, bulk_layered)
        )
        if mode_count > 1:
            raise ValueError(
                "packed, bulk_reference, and bulk_layered are mutually exclusive"
            )

        self.synchronize_gpu_process_state()
        if bulk_layered:
            payload = _pack_process_reviews(rows)
            (
                prediction_probabilities,
                curve_aheads,
                curve_ws,
            ) = self._native_runtime.process_reviews_bulk_layered(
                payload,
                return_curves,
                num_threads,
            )
        elif bulk_reference:
            payload = _pack_process_reviews(rows)
            (
                prediction_probabilities,
                curve_aheads,
                curve_ws,
            ) = self._native_runtime.process_reviews_bulk_reference(
                payload,
                return_curves,
                num_threads,
            )
        elif packed:
            payload = _pack_process_reviews(rows)
            (
                prediction_probabilities,
                curve_aheads,
                curve_ws,
            ) = self._native_runtime.process_reviews_packed(
                payload,
                return_curves,
                num_threads,
            )
        else:
            (
                prediction_probabilities,
                curve_aheads,
                curve_ws,
            ) = self._native_runtime.process_reviews(
                rows,
                return_curves,
                num_threads,
            )
        predictions = [float(probability) for probability in prediction_probabilities]

        if not return_curves:
            return predictions
        assert curve_aheads is not None
        assert curve_ws is not None
        return [
            (prediction, (_curve_array(ahead), _curve_array(w)))
            for prediction, ahead, w in zip(predictions, curve_aheads, curve_ws)
        ]

    def process_batch(
        self,
        batch: ReviewBatch,
        *,
        return_curves: bool,
        num_threads: int | None = None,
        packed: bool = False,
        bulk_reference: bool = False,
        bulk_layered: bool = False,
    ):
        if not batch:
            return []
        mode_count = sum(
            bool(value) for value in (bulk_reference, bulk_layered)
        )
        if mode_count > 1:
            raise ValueError("bulk_reference and bulk_layered are mutually exclusive")

        self.synchronize_gpu_process_state()
        if bulk_layered:
            output = self._native_runtime.process_review_batch_bulk_layered(
                batch._native_batch,
                return_curves,
                num_threads,
            )
        elif bulk_reference:
            output = self._native_runtime.process_review_batch_bulk_reference(
                batch._native_batch,
                return_curves,
                num_threads,
            )
        else:
            # ``packed`` is retained in this internal signature for callers
            # selecting the old strict path. A native batch is already parsed
            # and immutable, so both strict variants share this entrypoint.
            del packed
            output = self._native_runtime.process_review_batch(
                batch._native_batch,
                return_curves,
                num_threads,
            )
        prediction_probabilities, curve_aheads, curve_ws = output
        predictions = [float(probability) for probability in prediction_probabilities]
        if not return_curves:
            return predictions
        assert curve_aheads is not None
        assert curve_ws is not None
        return [
            (prediction, (_curve_array(ahead), _curve_array(w)))
            for prediction, ahead, w in zip(predictions, curve_aheads, curve_ws)
        ]

    def process_rows_pipeline(
        self,
        rows: list[dict[str, Any]],
        *,
        return_curves: bool,
        num_threads: int | None = None,
    ):
        if not rows:
            return []
        self.synchronize_gpu_process_state()
        payload = _pack_process_reviews(rows)
        (
            prediction_probabilities,
            curve_aheads,
            curve_ws,
        ) = self._native_runtime.process_reviews_pipeline(
            payload,
            return_curves,
            num_threads,
        )
        predictions = [float(probability) for probability in prediction_probabilities]

        if not return_curves:
            return predictions
        assert curve_aheads is not None
        assert curve_ws is not None
        return [
            (prediction, (_curve_array(ahead), _curve_array(w)))
            for prediction, ahead, w in zip(predictions, curve_aheads, curve_ws)
        ]

    def process_batch_pipeline(
        self,
        batch: ReviewBatch,
        *,
        return_curves: bool,
        num_threads: int | None = None,
    ):
        if not batch:
            return []
        self.synchronize_gpu_process_state()
        prediction_probabilities, curve_aheads, curve_ws = (
            self._native_runtime.process_review_batch_pipeline(
                batch._native_batch,
                return_curves,
                num_threads,
            )
        )
        predictions = [float(probability) for probability in prediction_probabilities]
        if not return_curves:
            return predictions
        assert curve_aheads is not None
        assert curve_ws is not None
        return [
            (prediction, (_curve_array(ahead), _curve_array(w)))
            for prediction, ahead, w in zip(predictions, curve_aheads, curve_ws)
        ]

    def build_state_only_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        fast: bool,
        num_threads: int | None = None,
    ) -> int:
        if not rows:
            return 0
        self.synchronize_gpu_process_state()
        payload = _pack_process_reviews(rows)
        if fast:
            processed_count = self._native_runtime.build_state_only_pipeline(
                payload,
                num_threads,
            )
        else:
            processed_count = self._native_runtime.build_state_only_packed(
                payload,
                num_threads,
            )
        return int(processed_count)

    def build_state_only_batch(
        self,
        batch: ReviewBatch,
        *,
        fast: bool,
        num_threads: int | None = None,
    ) -> int:
        if not batch:
            return 0
        self.synchronize_gpu_process_state()
        if fast:
            processed_count = self._native_runtime.build_state_only_batch_pipeline(
                batch._native_batch,
                num_threads,
            )
        else:
            processed_count = self._native_runtime.build_state_only_batch(
                batch._native_batch,
                num_threads,
            )
        return int(processed_count)

    def build_state_only_batch_gpu_scan(
        self,
        batch: ReviewBatch,
        *,
        num_threads: int | None = None,
        defer_cpu_state: bool = False,
        fully_resident_state: bool = False,
    ) -> int:
        if not batch:
            return 0
        try:
            processed_count = self._native_runtime.build_state_only_batch_gpu_scan(
                batch._native_batch,
                num_threads,
                defer_cpu_state,
                fully_resident_state,
            )
        except BaseException as exc:
            committed_rows = int(
                self._native_runtime.take_gpu_process_committed_rows()
            )
            translated = _translate_native_gpu_error(
                exc,
                operation="process",
                phase="process",
                committed_rows=committed_rows,
                state_recoverable=isinstance(
                    exc,
                    (
                        _native.NativeGpuOutOfMemoryError,
                        _native.NativeGpuUnavailableError,
                    ),
                ),
            )
            if translated is not None:
                raise translated from exc
            if committed_rows:
                raise GpuProcessError(
                    str(exc),
                    operation="process",
                    phase="process",
                    committed_rows=committed_rows,
                    state_recoverable=True,
                    retryable_on_cpu=False,
                ) from exc
            raise
        committed_rows = self.take_gpu_process_committed_rows()
        if committed_rows != int(processed_count):
            raise GpuProcessError(
                "GPU state-only progress reported "
                f"{committed_rows} committed rows for a {processed_count}-row result.",
                operation="process",
                phase="process",
                committed_rows=committed_rows,
                state_recoverable=True,
                retryable_on_cpu=False,
            )
        return int(processed_count)

    def build_state_only_rows_profiled(
        self,
        rows: list[dict[str, Any]],
        *,
        fast: bool,
        num_threads: int | None = None,
    ) -> tuple[int, dict[str, Any]]:
        if not rows:
            return 0, {
                "query_rows": 0,
                "process_rows": 0,
                "prediction_head_rows": 0,
                "curve_head_rows": 0,
                "completed_rows": 0,
            }

        total_start = time.perf_counter_ns()
        self.synchronize_gpu_process_state()
        payload_start = time.perf_counter_ns()
        payload = _pack_process_reviews(rows)
        payload_build_ns = time.perf_counter_ns() - payload_start

        native_start = time.perf_counter_ns()
        if fast:
            processed_count, profile = (
                self._native_runtime.build_state_only_pipeline_profiled(
                    payload,
                    num_threads,
                )
            )
        else:
            processed_count, profile = self._native_runtime.build_state_only_packed_profiled(
                payload,
                num_threads,
            )
        native_call_ns = time.perf_counter_ns() - native_start

        sync_state_ns = 0
        profile = dict(profile)
        profile["python_adapter"] = {
            "total_ns": time.perf_counter_ns() - total_start,
            "native_call_ns": native_call_ns,
            "payload_build_ns": payload_build_ns,
            "sync_state_ns": sync_state_ns,
            "returned_scalar_count": 1,
            "returned_prediction_rows": 0,
            "returned_curve_rows": 0,
            "fast": fast,
        }
        return int(processed_count), profile

    def process_rows_gpu_scan(
        self,
        rows: list[dict[str, Any]],
        *,
        return_curves: bool,
        num_threads: int | None = None,
        defer_cpu_state: bool = False,
    ):
        return self.process_batch_gpu_scan(
            ReviewBatch(rows),
            return_curves=return_curves,
            num_threads=num_threads,
            defer_cpu_state=defer_cpu_state,
        )

    def process_batch_gpu_scan(
        self,
        batch: ReviewBatch,
        *,
        return_curves: bool,
        num_threads: int | None = None,
        defer_cpu_state: bool = False,
        fully_resident_state: bool = False,
    ):
        if not batch:
            return []
        try:
            (
                prediction_probabilities,
                curve_aheads,
                curve_ws,
            ) = self._native_runtime.process_review_batch_gpu_scan(
                batch._native_batch,
                return_curves,
                num_threads,
                defer_cpu_state,
                fully_resident_state,
            )
        except BaseException as exc:
            (
                committed_rows,
                partial_probabilities,
                partial_aheads,
                partial_weights,
            ) = self._native_runtime.take_gpu_process_progress()
            partial_results = _gpu_scan_results(
                partial_probabilities,
                partial_aheads,
                partial_weights,
                return_curves=return_curves,
            )
            if len(partial_results) != int(committed_rows):
                raise RuntimeError(
                    "GPU process failure progress returned "
                    f"{len(partial_results)} results for {committed_rows} committed rows."
                ) from exc
            translated = _translate_native_gpu_error(
                exc,
                operation="process",
                phase="process",
                committed_rows=int(committed_rows),
                partial_results=partial_results,
                # Initialization failures have not started a process batch;
                # the OOM path preserves a synchronizable committed prefix.
                # Other execution failures remain conservative because a
                # device/dispatch failure may make resident state unusable.
                state_recoverable=isinstance(
                    exc,
                    (
                        _native.NativeGpuOutOfMemoryError,
                        _native.NativeGpuUnavailableError,
                    ),
                ),
            )
            if translated is not None:
                raise translated from exc
            if committed_rows:
                raise GpuProcessError(
                    str(exc),
                    operation="process",
                    phase="process",
                    committed_rows=int(committed_rows),
                    partial_results=partial_results,
                    state_recoverable=True,
                    retryable_on_cpu=False,
                ) from exc
            raise
        results = _gpu_scan_results(
            prediction_probabilities,
            curve_aheads,
            curve_ws,
            return_curves=return_curves,
        )
        self.take_gpu_process_committed_rows()
        return results

    def take_gpu_process_committed_rows(self) -> int:
        return int(self._native_runtime.take_gpu_process_committed_rows())

    def synchronize_gpu_process_state(self) -> int:
        try:
            return int(self._native_runtime.synchronize_gpu_process_state())
        except BaseException as exc:
            translated = _translate_native_gpu_error(
                exc,
                operation="process",
                phase="synchronize",
                state_recoverable=False,
            )
            if translated is None:
                raise
            raise translated from exc

    def undo_last_process(self) -> int:
        self.synchronize_gpu_process_state()
        remaining = int(self._native_runtime.undo_last_process())
        return remaining

    def clear_undo_history(self) -> int:
        return int(self._native_runtime.clear_undo_history())

    def undo_depth(self) -> int:
        return int(self._native_runtime.undo_depth())

    def bulk_feature_prepass_debug(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return self._native_runtime.process_reviews_bulk_feature_prepass_debug(
            _pack_process_reviews(rows)
        )

    def bulk_stream_plan_debug(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        self.synchronize_gpu_process_state()
        return self._native_runtime.process_reviews_bulk_stream_plan_debug(
            _pack_process_reviews(rows)
        )

    def process_rows_profiled(
        self,
        rows: list[dict[str, Any]],
        *,
        return_curves: bool,
        num_threads: int | None = None,
        packed: bool = False,
        bulk_reference: bool = False,
        bulk_layered: bool = False,
    ):
        if not rows:
            return [], {"review_count": 0}
        mode_count = sum(
            bool(value) for value in (packed, bulk_reference, bulk_layered)
        )
        if mode_count > 1:
            raise ValueError(
                "packed, bulk_reference, and bulk_layered are mutually exclusive"
            )

        total_start = time.perf_counter_ns()
        self.synchronize_gpu_process_state()

        payload = None
        payload_build_ns = 0
        if packed or bulk_reference or bulk_layered:
            payload_start = time.perf_counter_ns()
            payload = _pack_process_reviews(rows)
            payload_build_ns = time.perf_counter_ns() - payload_start

        native_start = time.perf_counter_ns()
        if bulk_layered:
            assert payload is not None
            (
                prediction_probabilities,
                curve_aheads,
                curve_ws,
                profile,
            ) = self._native_runtime.process_reviews_bulk_layered_profiled(
                payload,
                return_curves,
                num_threads,
            )
        elif bulk_reference:
            assert payload is not None
            (
                prediction_probabilities,
                curve_aheads,
                curve_ws,
                profile,
            ) = self._native_runtime.process_reviews_bulk_reference_profiled(
                payload,
                return_curves,
                num_threads,
            )
        elif packed:
            assert payload is not None
            (
                prediction_probabilities,
                curve_aheads,
                curve_ws,
                profile,
            ) = self._native_runtime.process_reviews_packed_profiled(
                payload,
                return_curves,
                num_threads,
            )
        else:
            (
                prediction_probabilities,
                curve_aheads,
                curve_ws,
                profile,
            ) = self._native_runtime.process_reviews_profiled(
                rows,
                return_curves,
                num_threads,
            )
        native_call_ns = time.perf_counter_ns() - native_start

        sync_state_ns = 0

        prediction_start = time.perf_counter_ns()
        predictions = [float(probability) for probability in prediction_probabilities]
        prediction_materialization_ns = time.perf_counter_ns() - prediction_start

        if not return_curves:
            profile["python_adapter"] = {
                "total_ns": time.perf_counter_ns() - total_start,
                "native_call_ns": native_call_ns,
                "payload_build_ns": payload_build_ns,
                "packed": packed,
                "bulk_reference": bulk_reference,
                "bulk_layered": bulk_layered,
                "sync_state_ns": sync_state_ns,
                "prediction_materialization_ns": prediction_materialization_ns,
                "curve_materialization_ns": 0,
                "prediction_rows": len(prediction_probabilities),
                "curve_rows": 0,
            }
            return predictions, profile
        assert curve_aheads is not None
        assert curve_ws is not None
        curve_start = time.perf_counter_ns()
        results = [
            (prediction, (_curve_array(ahead), _curve_array(w)))
            for prediction, ahead, w in zip(predictions, curve_aheads, curve_ws)
        ]
        curve_materialization_ns = time.perf_counter_ns() - curve_start
        profile["python_adapter"] = {
            "total_ns": time.perf_counter_ns() - total_start,
            "native_call_ns": native_call_ns,
            "payload_build_ns": payload_build_ns,
            "packed": packed,
            "bulk_reference": bulk_reference,
            "bulk_layered": bulk_layered,
            "sync_state_ns": sync_state_ns,
            "prediction_materialization_ns": prediction_materialization_ns,
            "curve_materialization_ns": curve_materialization_ns,
            "prediction_rows": len(prediction_probabilities),
            "curve_rows": len(curve_aheads),
        }
        return results, profile

    def predict_rows_profiled(
        self,
        rows: list[dict[str, Any]],
        *,
        batch_size: int = DEFAULT_PREDICT_MANY_BATCH_SIZE,
        num_threads: int | None = DEFAULT_PREDICT_MANY_THREADS,
        lightning: bool = False,
    ):
        if not rows:
            return [], {"review_count": 0}
        if not isinstance(lightning, bool):
            raise TypeError("lightning must be a bool.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        total_start = time.perf_counter_ns()
        self.synchronize_gpu_process_state()

        native_start = time.perf_counter_ns()
        prediction_probabilities, profile = (
            self._native_runtime.predict_reviews_profiled(
                rows,
                batch_size,
                num_threads,
                lightning,
            )
        )
        native_call_ns = time.perf_counter_ns() - native_start

        prediction_start = time.perf_counter_ns()
        predictions = [float(probability) for probability in prediction_probabilities]
        prediction_materialization_ns = time.perf_counter_ns() - prediction_start

        profile["python_adapter"] = {
            "total_ns": time.perf_counter_ns() - total_start,
            "native_call_ns": native_call_ns,
            "sync_state_ns": 0,
            "prediction_materialization_ns": prediction_materialization_ns,
            "curve_materialization_ns": 0,
            "prediction_rows": len(prediction_probabilities),
            "curve_rows": 0,
        }
        return predictions, profile

    def predict_rows_fast_caller_profiled(
        self,
        rows: list[dict[str, Any]] | PredictionBatch,
        *,
        batch_size: int = DEFAULT_FAST_PREDICT_MANY_BATCH_SIZE,
        num_threads: int | None = DEFAULT_PREDICT_MANY_THREADS,
    ):
        """Profile the current parallel CPU Fast public executor.

        This diagnostic is intentionally separate from ``predict_rows_profiled``:
        the older profiler retains its detailed sequential-forward counters,
        while this one reports caller-relevant stages from the same parallel
        batch organization used by ``predict_many(mode="fast")``.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        total_start = time.perf_counter_ns()
        self.synchronize_gpu_process_state()

        validation_start = time.perf_counter_ns()
        if not isinstance(rows, PredictionBatch):
            for row in rows:
                _require_columns(row, _PREDICT_REQUIRED_COLUMNS)
        adapter_validation_ns = time.perf_counter_ns() - validation_start

        native_start = time.perf_counter_ns()
        if isinstance(rows, PredictionBatch):
            prediction_probabilities, profile = (
                self._native_runtime.predict_review_batch_fast_caller_profiled(
                    rows._native_batch,
                    batch_size,
                    num_threads,
                )
            )
        else:
            prediction_probabilities, profile = (
                self._native_runtime.predict_reviews_fast_caller_profiled(
                    rows,
                    batch_size,
                    num_threads,
                )
            )
        native_call_ns = time.perf_counter_ns() - native_start

        prediction_start = time.perf_counter_ns()
        predictions = [float(probability) for probability in prediction_probabilities]
        prediction_materialization_ns = time.perf_counter_ns() - prediction_start
        native_profile_total_ns = int(profile["total_ns"])
        profile["python_adapter"] = {
            "total_ns": time.perf_counter_ns() - total_start,
            "validation_ns": adapter_validation_ns,
            "native_call_ns": native_call_ns,
            "native_result_boundary_residual_ns": max(
                0,
                native_call_ns - native_profile_total_ns,
            ),
            "prediction_materialization_ns": prediction_materialization_ns,
            "prediction_rows": len(prediction_probabilities),
            "input_surface": (
                "prediction_batch"
                if isinstance(rows, PredictionBatch)
                else "mappings"
            ),
        }
        return predictions, profile

    def predict_rows_public_path_profiled(self, rows: list[dict[str, Any]]):
        if not rows:
            return [], {"review_count": 0}

        self.synchronize_gpu_process_state()

        profile = {
            "review_count": 0,
            "total_ns": 0,
            "prepare_predict_ns": 0,
            "rng_check_ns": 0,
            "feature_tensor_ns": 0,
            "recurrent_sync_ns": 0,
            "feature_to_list_ns": 0,
            "native_review_ns": 0,
            "prediction_materialization_ns": 0,
            "rng_restore_ns": 0,
        }
        predictions = []
        total_start = time.perf_counter_ns()

        for row in rows:
            profile["review_count"] += 1

            start = time.perf_counter_ns()
            row = self._prepare_predict_row(row)
            profile["prepare_predict_ns"] += time.perf_counter_ns() - start

            start = time.perf_counter_ns()
            rng_state = (
                self.torch_rng_state()
                if self._skip_needs_rng_restore(row, True)
                else None
            )
            profile["rng_check_ns"] += time.perf_counter_ns() - start

            try:
                start = time.perf_counter_ns()
                features = self._feature_array_for_run(row, skip=True)
                profile["feature_tensor_ns"] += time.perf_counter_ns() - start

                start = time.perf_counter_ns()
                profile["recurrent_sync_ns"] += time.perf_counter_ns() - start

                start = time.perf_counter_ns()
                feature_values = _to_list(features)
                profile["feature_to_list_ns"] += time.perf_counter_ns() - start

                start = time.perf_counter_ns()
                (
                    _out_ahead_logits,
                    _out_w,
                    out_p_logits,
                ) = self._native_runtime.review(
                    feature_values,
                    int(row["card_id"]),
                    int(row["note_id"]),
                    int(row["deck_id"]),
                    int(row["preset_id"]),
                    True,
                    False,
                )
                profile["native_review_ns"] += time.perf_counter_ns() - start

                start = time.perf_counter_ns()
                predictions.append(float(_native.prediction_probability(out_p_logits)))
                profile["prediction_materialization_ns"] += (
                    time.perf_counter_ns() - start
                )
            finally:
                if rng_state is not None:
                    start = time.perf_counter_ns()
                    self.restore_torch_rng_state(rng_state)
                    profile["rng_restore_ns"] += time.perf_counter_ns() - start

        profile["total_ns"] = time.perf_counter_ns() - total_start
        return predictions, profile

    def run(
        self,
        row: dict[str, Any],
        skip: bool,
        *,
        return_curve: bool = True,
    ) -> tuple[ReviewCurve | None, array]:
        self.synchronize_gpu_process_state()
        rng_state = (
            self.torch_rng_state() if self._skip_needs_rng_restore(row, skip) else None
        )
        try:
            features = self._feature_array_for_run(row, skip=skip)
            (
                out_ahead_logits,
                out_w,
                out_p_logits,
            ) = self._native_runtime.review(
                features,
                int(row["card_id"]),
                int(row["note_id"]),
                int(row["deck_id"]),
                int(row["preset_id"]),
                bool(skip),
                return_curve,
            )

            if not skip:
                self._native_runtime.record_recurrent_state_update(row)

            imm_probs = array(
                "f", [float(_native.prediction_probability(out_p_logits))]
            )
            curve = None
            if return_curve:
                assert out_ahead_logits is not None
                assert out_w is not None
                curve = (_curve_array(out_ahead_logits), _curve_array(out_w))
            return curve, imm_probs
        finally:
            if rng_state is not None:
                self.restore_torch_rng_state(rng_state)

    def predict_func(self, curve, elapsed_seconds):
        out_ahead_logits, out_w = curve
        probs = _native.predict_curve(
            _curve_matrix_for_native(out_ahead_logits),
            _curve_matrix_for_native(out_w),
            _float_list_1d(elapsed_seconds),
        )
        return array("f", (float(probability) for probability in probs))

    def get_tensor(self, row, *, mutate_id_encodings: bool):
        vector = self._native_runtime.feature_vector(
            row,
            mutate_id_encodings=mutate_id_encodings,
        )
        return _feature_array(vector)

    def get_tensors(self, rows):
        return _feature_batch_array(self._native_runtime.feature_vectors(rows))

    def _prepare_predict_row(self, row):
        return self._native_runtime.prepare_predict(row)

    def _prepare_process_row(self, row):
        return self._native_runtime.prepare_process(row)

    def _record_processed_row(self, row):
        self._native_runtime.record_processed(row)

    def _skip_needs_rng_restore(self, row, skip):
        return self._native_runtime.skip_needs_rng_restore(row, skip)

    def _can_batch_predict(self, row):
        self.synchronize_gpu_process_state()
        return self._native_runtime.can_batch_predict(row)

    def _ensure_native_runtime_current(self) -> None:
        return None

    def _ensure_native_recurrent_current(self) -> None:
        self.synchronize_gpu_process_state()

    def _feature_array_for_run(self, row, *, skip: bool) -> list[list[float]]:
        if skip:
            vector = self._native_runtime.predict_feature_vector(row)
        else:
            vector = self._native_runtime.process_feature_vector(row)
        return _feature_array(vector)

    def export_state(self) -> dict[str, Any]:
        self.synchronize_gpu_process_state()
        snapshot = self._native_runtime.snapshot()
        recurrent = self._native_runtime.recurrent_state_lists()
        return {
            "card_states": _normalize_int_key_map(recurrent["card_states"]),
            "note_states": _normalize_int_key_map(recurrent["note_states"]),
            "deck_states": _normalize_int_key_map(recurrent["deck_states"]),
            "preset_states": _normalize_int_key_map(recurrent["preset_states"]),
            "global_state": recurrent["global_state"],
            "first_day_offset": snapshot["first_day_offset"],
            "prev_day_offset": snapshot["prev_day_offset"],
            "card_set": sorted(int(card_id) for card_id in snapshot["card_set"]),
            "card_count": int(snapshot["card_count"]),
            "last_new_cards": _int_key_int_value_map(snapshot["last_new_cards"]),
            "i": int(snapshot["i"]),
            "last_i": _int_key_int_value_map(snapshot["last_i"]),
            "today": float(snapshot["today"]),
            "today_reviews": int(snapshot["today_reviews"]),
            "today_new_cards": int(snapshot["today_new_cards"]),
            "card2first_day_offset": _int_key_float_value_map(
                snapshot["card2first_day_offset"]
            ),
            "card2elapsed_days_cumulative": _int_key_float_value_map(
                snapshot["card2elapsed_days_cumulative"]
            ),
            "card2elapsed_seconds_cumulative": _int_key_float_value_map(
                snapshot["card2elapsed_seconds_cumulative"]
            ),
            "id_encodings": _normalize_id_encoding_snapshot(
                self._native_runtime.id_encoding_snapshot()
            ),
        }

    def restore_exported_state(self, state: dict[str, Any]) -> None:
        self.synchronize_gpu_process_state()
        state = _native_ready(state)
        snapshot = {
            "first_day_offset": state["first_day_offset"],
            "prev_day_offset": state.get("prev_day_offset"),
            "card_set": sorted(int(card_id) for card_id in state["card_set"]),
            "card_count": int(state.get("card_count", len(state["card_set"]))),
            "last_new_cards": _int_key_int_value_map(state["last_new_cards"]),
            "i": int(state["i"]),
            "last_i": _int_key_int_value_map(state["last_i"]),
            "today": float(state["today"]),
            "today_reviews": int(state["today_reviews"]),
            "today_new_cards": int(state["today_new_cards"]),
            "card2first_day_offset": _int_key_float_value_map(
                state["card2first_day_offset"]
            ),
            "card2elapsed_days_cumulative": _int_key_float_value_map(
                state["card2elapsed_days_cumulative"]
            ),
            "card2elapsed_seconds_cumulative": _int_key_float_value_map(
                state["card2elapsed_seconds_cumulative"]
            ),
        }
        card_states = _state_map_to_native(state["card_states"])
        note_states = _state_map_to_native(state["note_states"])
        deck_states = _state_map_to_native(state["deck_states"])
        preset_states = _state_map_to_native(state["preset_states"])
        global_state = _entity_state_to_native(state["global_state"])
        recurrent = {
            "card_states": card_states,
            "note_states": note_states,
            "deck_states": deck_states,
            "preset_states": preset_states,
            "global_state": global_state,
        }
        recurrent_keys = {
            "card_states": sorted(card_states),
            "note_states": sorted(note_states),
            "deck_states": sorted(deck_states),
            "preset_states": sorted(preset_states),
            "global_state": global_state is not None,
        }
        self._native_runtime.restore_snapshot(snapshot)
        self._native_runtime.restore_id_encoding_snapshot(state["id_encodings"])
        self._native_runtime.restore_recurrent_state_lists(recurrent)
        self._native_runtime.restore_recurrent_state_keys(recurrent_keys)

    # These historical names describe the serialized Torch-compatible RNG
    # byte format used by native deterministic features. They do not inspect,
    # import, or synchronize with the Torch Python package.
    def torch_rng_state(self) -> list[int]:
        return list(self._native_runtime.torch_rng_state())

    def write_checkpoint_bin(self, path: str | Path, metadata_json: bytes) -> None:
        self.synchronize_gpu_process_state()
        self._native_runtime.write_checkpoint_bin(Path(path), metadata_json)

    def expected_checkpoint_bin_size(
        self,
        metadata_len: int,
        *,
        card_count: int,
        note_count: int,
        deck_count: int,
        preset_count: int,
    ) -> int:
        return int(
            self._native_runtime.expected_checkpoint_bin_size(
                metadata_len,
                card_count,
                note_count,
                deck_count,
                preset_count,
            )
        )

    def write_merged_checkpoint_bin(
        self,
        backing_path: str | Path,
        path: str | Path,
        metadata_json: bytes,
        scope: _CheckpointCardScope,
    ) -> None:
        self.synchronize_gpu_process_state()
        self._native_runtime.write_merged_checkpoint_bin(
            Path(backing_path),
            Path(path),
            metadata_json,
            sorted(scope.card_ids),
            sorted(scope.note_ids),
            sorted(scope.deck_ids),
            sorted(scope.preset_ids),
        )

    def restore_checkpoint_bin(
        self,
        path: str | Path,
        scope: _CheckpointCardScope | None = None,
    ) -> None:
        self.synchronize_gpu_process_state()
        if scope is None:
            self._native_runtime.restore_checkpoint_bin(
                Path(path), None, None, None, None
            )
            return
        self._native_runtime.restore_checkpoint_bin(
            Path(path),
            sorted(scope.card_ids),
            sorted(scope.note_ids),
            sorted(scope.deck_ids),
            sorted(scope.preset_ids),
        )

    def restore_torch_rng_state(self, rng_state: Any) -> None:
        self._native_runtime.restore_torch_rng_state(_native_ready(rng_state))

    def _deterministic_attr(self, name: str) -> Any:
        if name == "_day_offset_encoding_cache":
            return {}
        if name == "id_encodings":
            return _normalize_id_encoding_snapshot(
                self._native_runtime.id_encoding_snapshot()
            )
        snapshot = self._native_runtime.snapshot()
        if name == "prev_row":
            prev_day_offset = snapshot["prev_day_offset"]
            return None if prev_day_offset is None else {"day_offset": prev_day_offset}
        if name == "card_set":
            return set(snapshot["card_set"])
        if name in {"last_new_cards", "last_i"}:
            return _int_key_int_value_map(snapshot[name])
        if name in {
            "card2first_day_offset",
            "card2elapsed_days_cumulative",
            "card2elapsed_seconds_cumulative",
        }:
            return _int_key_float_value_map(snapshot[name])
        return snapshot[name]

    def _recurrent_attr(self, name: str) -> Any:
        self.synchronize_gpu_process_state()
        recurrent = self._native_runtime.recurrent_state_lists()
        if name == "global_state":
            return recurrent[name]
        return _normalize_int_key_map(recurrent[name])

    def _set_deterministic_attr(self, name: str, value: Any) -> None:
        if name == "_day_offset_encoding_cache":
            return
        if name == "id_encodings":
            self._native_runtime.restore_id_encoding_snapshot(_native_ready(value))
            return
        snapshot = self._native_runtime.snapshot()
        if name == "prev_row":
            snapshot["prev_day_offset"] = (
                None if value is None else float(value["day_offset"])
            )
        elif name == "card_set":
            snapshot["card_set"] = sorted(int(card_id) for card_id in value)
            snapshot["card_count"] = len(snapshot["card_set"])
        elif name in {"last_new_cards", "last_i"}:
            snapshot[name] = _int_key_int_value_map(value)
        elif name in {
            "card2first_day_offset",
            "card2elapsed_days_cumulative",
            "card2elapsed_seconds_cumulative",
        }:
            snapshot[name] = _int_key_float_value_map(value)
        else:
            snapshot[name] = value
        self._native_runtime.restore_snapshot(snapshot)

    def _set_recurrent_attr(self, name: str, value: Any) -> None:
        self.synchronize_gpu_process_state()
        recurrent = self._native_runtime.recurrent_state_lists()
        recurrent[name] = (
            _entity_state_to_native(value)
            if name == "global_state"
            else _state_map_to_native(value)
        )
        self._native_runtime.restore_recurrent_state_lists(recurrent)
        recurrent_keys = self._native_runtime.recurrent_state_key_snapshot()
        if name == "global_state":
            recurrent_keys[name] = value is not None
        else:
            recurrent_keys[name] = sorted(int(key) for key in value)
        self._native_runtime.restore_recurrent_state_keys(recurrent_keys)


def _resolve_rust_model(model: str | Path | None) -> tuple[str | None, Path]:
    if model is None:
        raise ValueError("model must not be None")

    if isinstance(model, str) and not any(sep in model for sep in ("/", "\\")):
        if Path(model).suffix == "":
            return model, _get_safetensors_model_path(model)

    path = _require_safetensors_model_path(model)
    return None, path


def _get_safetensors_model_path(model_id: str) -> Path:
    candidates = (
        TEST_MODEL_FIXTURE_DIR / f"{model_id}.safetensors",
        PRETRAINED_MODEL_DIR / f"{model_id}.safetensors",
    )
    for path in candidates:
        if path.exists():
            return path
    available = ", ".join(_available_safetensors_model_ids()) or "none"
    expected = " or ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Unknown Rust RWKV-SRS model {model_id!r}. The generic runtime wheel "
        f"does not bundle model weights. Package a repository-owned model under "
        f"rwkv_srs/pretrained or pass an explicit .safetensors path. Lookup "
        f"expected {expected}; available models: {available}."
    )


def _available_safetensors_model_ids() -> tuple[str, ...]:
    model_ids: set[str] = set()
    for root in (TEST_MODEL_FIXTURE_DIR, PRETRAINED_MODEL_DIR):
        if root.exists():
            model_ids.update(path.stem for path in root.glob("*.safetensors"))
    return tuple(sorted(model_ids))


def _require_safetensors_model_path(path: str | Path) -> Path:
    path = Path(path)
    if path.suffix != ".safetensors":
        raise ValueError(
            "Rust backend model files must be .safetensors; pass an explicit "
            "path to Rust-compatible model weights."
        )
    if not path.exists():
        raise FileNotFoundError(
            f"Rust backend safetensors model file does not exist: {path}"
        )
    return path


def _load_checkpoint_dict(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix != ".bin":
        raise ValueError(
            "Rust backend checkpoints must be Rust-native .bin files. "
            "Torch .pt/.pth checkpoints and legacy JSON checkpoints are not "
            "supported by the Rust backend."
        )
    return _load_checkpoint_bin_metadata(path)


def _load_checkpoint_bin_metadata(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        magic = f.read(len(_RUST_CHECKPOINT_BIN_MAGIC))
        if magic != _RUST_CHECKPOINT_BIN_MAGIC:
            raise ValueError("Unsupported Rust binary checkpoint magic.")
        version = int.from_bytes(f.read(4), "little")
        if version not in {1, _RUST_CHECKPOINT_BIN_VERSION}:
            raise ValueError(
                f"Unsupported Rust binary checkpoint version {version}; expected 1 or {_RUST_CHECKPOINT_BIN_VERSION}."
            )
        metadata_len = int.from_bytes(f.read(8), "little")
        metadata_remaining = path.stat().st_size - f.tell()
        if metadata_len > metadata_remaining:
            raise ValueError("Rust binary checkpoint metadata is truncated.")
        metadata = f.read(metadata_len)
        if len(metadata) != metadata_len:
            raise ValueError("Rust binary checkpoint metadata is truncated.")
    checkpoint = json.loads(metadata.decode("utf-8"))
    if checkpoint.get("format") not in {None, _RUST_CHECKPOINT_FORMAT}:
        raise ValueError(
            f"Unsupported Rust checkpoint format: {checkpoint.get('format')!r}"
        )
    expected_storage_format = f"rwkv-p-rust-checkpoint-bin-v{version}"
    if checkpoint.get("storage_format") != expected_storage_format:
        raise ValueError(
            f"Unsupported Rust checkpoint storage format: {checkpoint.get('storage_format')!r}"
        )
    return checkpoint


def _pack_process_reviews(rows: list[dict[str, Any]]) -> bytes:
    payload = bytearray(
        _PROCESS_REVIEW_PAYLOAD_HEADER.size
        + _PROCESS_REVIEW_PAYLOAD_RECORD.size * len(rows)
    )
    _PROCESS_REVIEW_PAYLOAD_HEADER.pack_into(
        payload,
        0,
        _PROCESS_REVIEW_PAYLOAD_MAGIC,
        len(rows),
    )
    offset = _PROCESS_REVIEW_PAYLOAD_HEADER.size
    for index, row in enumerate(rows):
        note_present, note_id = _packed_optional_id(row["note_id"], "note_id")
        deck_present, deck_id = _packed_optional_id(row["deck_id"], "deck_id")
        preset_present, preset_id = _packed_optional_id(row["preset_id"], "preset_id")
        try:
            _PROCESS_REVIEW_PAYLOAD_RECORD.pack_into(
                payload,
                offset,
                _packed_required_int(row["review_id"], "review_id"),
                _packed_required_int(row["card_id"], "card_id"),
                note_present,
                note_id,
                deck_present,
                deck_id,
                preset_present,
                preset_id,
                _packed_required_float(row["day_offset"], "day_offset"),
                _packed_required_float(row["elapsed_days"], "elapsed_days"),
                _packed_required_float(row["elapsed_seconds"], "elapsed_seconds"),
                _packed_required_int(row["rating"], "rating"),
                _packed_required_float(row["duration"], "duration"),
                _packed_required_float(row["state"], "state"),
            )
        except struct.error as exc:
            raise ValueError(
                f"Review at index {index} cannot be encoded for Rust process_many()."
            ) from exc
        offset += _PROCESS_REVIEW_PAYLOAD_RECORD.size
    return bytes(payload)


def _packed_optional_id(value: Any, field: str) -> tuple[int, int]:
    value = _unwrap_tensor_like(value)
    if _packed_is_missing(value):
        return 0, 0
    return 1, _packed_required_int(value, field)


def _packed_required_int(value: Any, field: str) -> int:
    value = _unwrap_tensor_like(value)
    if _packed_is_missing(value):
        raise ValueError(
            f"Review field {field!r} must be an integer, got missing value."
        )
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        value = float(value)
        if math.isfinite(value) and value.is_integer():
            return int(value)
    raise ValueError(f"Review field {field!r} must be an integer.")


def _packed_required_float(value: Any, field: str) -> float:
    value = _unwrap_tensor_like(value)
    if _packed_is_missing(value):
        raise ValueError(
            f"Review field {field!r} must be a finite number, got missing value."
        )
    if isinstance(value, numbers.Real):
        value = float(value)
        if math.isfinite(value):
            return value
    raise ValueError(f"Review field {field!r} must be a finite number.")


def _packed_is_missing(value: Any) -> bool:
    if value is None:
        return True
    if type(value).__name__ in {"NAType", "NaTType"}:
        return True
    return isinstance(value, numbers.Real) and math.isnan(float(value))


def _normalize_device(device: Any) -> Any:
    device_type = getattr(device, "type", None)
    device_name = device_type if device_type is not None else str(device)
    if device_name != "cpu":
        raise ValueError("Rust backend is CPU-only; pass device='cpu'.")
    return device


def _normalize_dtype(dtype: Any) -> Any:
    dtype_name = str(dtype)
    if dtype_name not in {"float32", "torch.float32"}:
        raise ValueError("Rust backend currently supports float32 only.")
    return dtype


def _native_num_threads(num_threads: int | None, *, default: int) -> int | None:
    resolved = _validate_num_threads(num_threads)
    if resolved is None:
        resolved = default
    # Using Rayon/Candle's global pool avoids custom-pool dispatch overhead.
    if resolved == _GLOBAL_RAYON_NUM_THREADS:
        return None
    return resolved


def _validate_undo_limit(undo_limit: int) -> int:
    if isinstance(undo_limit, bool) or not isinstance(undo_limit, numbers.Integral):
        raise TypeError("undo_limit must be an integer.")
    undo_limit = int(undo_limit)
    if undo_limit < 0:
        raise ValueError("undo_limit must be non-negative.")
    return undo_limit


def _validate_runtime_owner_thread(runtime_owner_thread: bool) -> bool:
    if not isinstance(runtime_owner_thread, bool):
        raise TypeError("runtime_owner_thread must be a bool.")
    return runtime_owner_thread


class _ScalarFloat(float):
    def item(self) -> float:
        return float(self)


def _curve_array(values: Any) -> array:
    values = _to_list(values)
    if values and isinstance(values[0], (array, list, tuple)):
        if len(values) != 1:
            raise ValueError(f"Expected a single curve row, got {len(values)} rows.")
        values = _to_list(values[0])
    return array("f", (float(value) for value in values))


def _gpu_scan_curve_rows(values: bytes, rows: int) -> list[array]:
    """Split a native flat-FP32 curve buffer without boxing every value."""
    flat = array("f")
    flat.frombytes(values)
    expected = rows * GPU_PROCESS_CURVE_SIZE
    if len(flat) != expected:
        raise ValueError(f"Expected {expected} flat GPU curve values, got {len(flat)}.")
    return [
        flat[start : start + GPU_PROCESS_CURVE_SIZE]
        for start in range(0, expected, GPU_PROCESS_CURVE_SIZE)
    ]


def _gpu_scan_results(
    probabilities: Iterable[float],
    curve_aheads: bytes | None,
    curve_weights: bytes | None,
    *,
    return_curves: bool,
) -> list[_ProcessResult]:
    predictions = [float(probability) for probability in probabilities]
    if not return_curves:
        return cast(list[_ProcessResult], predictions)
    if curve_aheads is None or curve_weights is None:
        raise RuntimeError("GPU process omitted requested partial curve results.")
    ahead_rows = _gpu_scan_curve_rows(curve_aheads, len(predictions))
    weight_rows = _gpu_scan_curve_rows(curve_weights, len(predictions))
    return [
        (prediction, (ahead, weight))
        for prediction, ahead, weight in zip(predictions, ahead_rows, weight_rows)
    ]


def _curve_matrix_for_native(values: Any) -> list[list[float]]:
    values = _to_list(values)
    if not values:
        return [[]]
    if isinstance(values[0], (array, list, tuple)):
        return [[float(value) for value in _to_list(row)] for row in values]
    return [[float(value) for value in values]]


def _curve_row_for_native(
    values: Any,
    *,
    expected_len: int,
    name: str,
) -> Any:
    # process() and process_many(return_curves=True) return array('f') rows.
    # Preserve those buffer-backed rows here instead of first boxing every
    # value into a Python float; PyO3 can extract them directly as one matrix
    # row. Less common tensor/list-like inputs retain the scalar helper's
    # normalization behavior.
    if isinstance(values, array):
        if len(values) != expected_len:
            raise ValueError(
                f"{name} must contain {expected_len} float values, got {len(values)}."
            )
        return values
    return _curve_values_for_native(
        values,
        expected_len=expected_len,
        name=name,
    )


def _curve_values_for_native(
    values: Any,
    *,
    expected_len: int,
    name: str,
) -> list[float]:
    values = _to_list(values)
    if values and isinstance(values[0], (array, list, tuple)):
        if len(values) != 1:
            raise ValueError(f"{name} must contain one curve row, got {len(values)}.")
        values = _to_list(values[0])
    out = [float(value) for value in values]
    if len(out) != expected_len:
        raise ValueError(
            f"{name} must contain {expected_len} float values, got {len(out)}."
        )
    return out


def _float_list_1d(value: Any) -> list[float]:
    out: list[float] = []

    def visit(item: Any) -> None:
        item = _unwrap_tensor_like(item)
        if isinstance(item, (array, list, tuple)):
            for child in item:
                visit(child)
            return
        out.append(float(item))

    visit(value)
    return out


def _feature_array(values: list[float]) -> list[list[float]]:
    return [[float(value) for value in values]]


def _feature_batch_array(values: list[list[float]]) -> list[list[float]]:
    return [[float(value) for value in row] for row in values]


def _native_ready(value: Any) -> Any:
    value = _unwrap_tensor_like(value)
    if isinstance(value, dict):
        return {_native_key(key): _native_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native_ready(item) for item in value]
    if isinstance(value, set):
        return sorted(_native_ready(item) for item in value)
    return value


def _unwrap_tensor_like(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except TypeError:
            return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def _native_key(key: Any) -> Any:
    key = _unwrap_tensor_like(key)
    if isinstance(key, str):
        try:
            return int(key)
        except ValueError:
            return key
    if isinstance(key, numbers.Integral):
        return int(key)
    return key


def _to_list(value: Any) -> list:
    value = _unwrap_tensor_like(value)
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, array):
        return value.tolist()
    if isinstance(value, (str, bytes)):
        raise TypeError(f"Expected a sequence of numeric values, got {type(value)!r}.")
    if isinstance(value, numbers.Real):
        return [value]
    return list(value)


def _normalize_int_key_map(values: dict[Any, Any]) -> dict[int, Any]:
    return {int(key): _native_ready(value) for key, value in values.items()}


def _state_map_to_native(states: dict[Any, Any]) -> dict[int, _NativeEntityState]:
    converted: dict[int, _NativeEntityState] = {}
    for key, value in states.items():
        native_key = int(key)
        entity_state = _entity_state_to_native(value)
        if entity_state is None:
            raise TypeError(
                f"Rust recurrent state map entry {native_key} must not be None."
            )
        converted[native_key] = entity_state
    return converted


def _entity_state_to_native(value: Any) -> _NativeEntityState | None:
    value = _native_ready(value)
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return value[0], value[1], value[2]
    if not isinstance(value, dict):
        raise TypeError(f"Unsupported Rust recurrent state type: {type(value)!r}")

    time_x_states = []
    time_recurrent_states = []
    channel_states = []
    for layer_index in sorted(value):
        time_state, channel_state = value[layer_index]
        time_x, time_recurrent = time_state
        time_x_states.append(_native_ready(time_x))
        time_recurrent_states.append(_native_ready(time_recurrent))
        channel_states.append(_native_ready(channel_state))
    return time_x_states, time_recurrent_states, channel_states


def _normalize_id_encoding_snapshot(
    values: dict[Any, Any],
) -> dict[str, dict[int, list[float]]]:
    return {
        str(submodule): {
            int(id_value): [float(item) for item in _to_list(encoding)]
            for id_value, encoding in encodings.items()
        }
        for submodule, encodings in values.items()
    }


def _int_key_int_value_map(values: dict[Any, Any]) -> dict[int, int]:
    return {int(key): int(value) for key, value in values.items()}


def _int_key_float_value_map(values: dict[Any, Any]) -> dict[int, float]:
    return {int(key): float(value) for key, value in values.items()}
