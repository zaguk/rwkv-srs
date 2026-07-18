from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

PACKAGE_ROOT = Path(__file__).resolve().parent
PRETRAINED_MODEL_DIR = PACKAGE_ROOT / "pretrained"
REPO_ROOT = PACKAGE_ROOT.parents[1]
TEST_MODEL_FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "models"

ReviewInput = Mapping[str, Any]
CpuMode = Literal["oracle", "fast"]
ExecutionMode = Literal["oracle", "fast", "gpu"]
GpuOperation = Literal["predict", "process"]

CPU_MODES = ("oracle", "fast")
EXECUTION_MODES = (*CPU_MODES, "gpu")
GPU_OPERATIONS = ("predict", "process")


class UndoUnavailableError(RuntimeError):
    """Raised when no recorded Rust process step can be undone."""


class GpuError(RuntimeError):
    """Base class for operational GPU failures.

    Invalid inputs and API arguments continue to raise ``ValueError``.  This
    hierarchy is reserved for failures where a valid operation could not be
    completed by the selected GPU executor. ``committed_rows`` and
    ``partial_results`` describe the exact successful prefix of a mutating
    process call. ``state_recoverable`` says whether the current runtime can
    still materialize that prefix, while ``retryable_on_cpu`` says whether
    RWKV-SRS can safely continue the uncommitted suffix on the CPU.
    """

    def __init__(
        self,
        message: str,
        *,
        operation: GpuOperation,
        phase: str,
        committed_rows: int = 0,
        partial_results: Iterable[Any] = (),
        state_recoverable: bool,
        retryable_on_cpu: bool,
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.phase = str(phase)
        self.committed_rows = int(committed_rows)
        self.partial_results = tuple(partial_results)
        self.state_recoverable = bool(state_recoverable)
        self.retryable_on_cpu = bool(retryable_on_cpu)

    def _prepend_process_progress(
        self,
        committed_rows: int,
        partial_results: Iterable[Any],
    ) -> None:
        """Add already completed public batches to native failure progress."""

        prefix = tuple(partial_results)
        self.committed_rows += int(committed_rows)
        self.partial_results = prefix + self.partial_results


class GpuUnavailableError(GpuError):
    """Raised when the adapter cannot initialize or fit the requested operation."""


class GpuOutOfMemoryError(GpuError):
    """Raised when a recoverable GPU allocation ultimately cannot fit."""


class GpuExecutionError(GpuError):
    """Raised for non-allocation GPU initialization or execution failures."""


class GpuProcessError(GpuError):
    """Raised when another native error follows a committed GPU process prefix."""


@contextmanager
def _atomic_output_path(destination: Path) -> Iterator[Path]:
    """Yield a sibling temporary and atomically publish it on success."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        yield temporary_path
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_cpu_mode(value: str) -> CpuMode:
    if not isinstance(value, str):
        raise TypeError("cpu_mode must be a string.")
    if value not in CPU_MODES:
        choices = ", ".join(repr(mode) for mode in CPU_MODES)
        raise ValueError(f"cpu_mode must be one of {choices}; got {value!r}.")
    return value  # type: ignore[return-value]


def _validate_execution_mode(value: str) -> ExecutionMode:
    if not isinstance(value, str):
        raise TypeError("mode must be a string.")
    if value not in EXECUTION_MODES:
        choices = ", ".join(repr(mode) for mode in EXECUTION_MODES)
        raise ValueError(f"mode must be one of {choices}; got {value!r}.")
    return value  # type: ignore[return-value]


def _validate_gpu_operation(value: str) -> GpuOperation:
    if not isinstance(value, str):
        raise TypeError("operation must be a string.")
    if value not in GPU_OPERATIONS:
        choices = ", ".join(repr(operation) for operation in GPU_OPERATIONS)
        raise ValueError(f"operation must be one of {choices}; got {value!r}.")
    return value  # type: ignore[return-value]


_CHECKPOINT_VERSION = 5
_HISTORY_FINGERPRINT_VERSION = 1
_HISTORY_FINGERPRINT_ALGORITHM = "sha256-chain"
_HISTORY_FINGERPRINT_CANONICALIZATION = "rwkv-p-review-v1"
_INITIAL_HISTORY_DIGEST = "0" * 64
_ID_PLACEHOLDER = 314159265358979323

_CHECKPOINT_CARD_COLUMNS = (
    "card_id",
    "note_id",
    "deck_id",
    "preset_id",
)

_PREDICT_REQUIRED_COLUMNS = (
    "review_id",
    "card_id",
    "note_id",
    "deck_id",
    "preset_id",
    "day_offset",
    "elapsed_days",
    "elapsed_seconds",
)

_PROCESS_REQUIRED_COLUMNS = _PREDICT_REQUIRED_COLUMNS + (
    "rating",
    "duration",
    "state",
)

_HISTORY_FINGERPRINT_FIELDS = _PROCESS_REQUIRED_COLUMNS
_HISTORY_INTEGER_FIELDS = {
    "review_id",
    "card_id",
    "note_id",
    "deck_id",
    "preset_id",
    "rating",
    "state",
}
_HISTORY_FLOAT_FIELDS = {
    "day_offset",
    "elapsed_days",
    "elapsed_seconds",
    "duration",
}

# Tuned on the local CPU benchmark suite. Scalar paths perform many small
# forwards and prefer one thread; batched predict_many() uses backend-specific
# runtime defaults because the useful wider parallelism depends on the machine.
# Keep these in the backend layer so normal consumers get measured defaults
# without benchmark-only env vars.
_DEFAULT_TORCH_SCALAR_THREADS = 1
_DEFAULT_TORCH_PROCESS_MANY_THREADS = 1
_DEFAULT_RUST_SCALAR_THREADS = 1
_DEFAULT_RUST_PROCESS_MANY_THREADS = 1
_DEFAULT_TORCH_PREDICT_MANY_BATCH_SIZE = 104
_DEFAULT_RUST_PREDICT_MANY_BATCH_SIZE = 192
_DEFAULT_PROCESS_MANY_BATCH_SIZE = 10_000


@dataclass(frozen=True)
class _CheckpointCardScope:
    """Identity sets whose checkpoint state is available in a runtime."""

    card_ids: frozenset[int]
    note_ids: frozenset[int]
    deck_ids: frozenset[int]
    preset_ids: frozenset[int]

    @classmethod
    def from_cards(cls, cards: Iterable[ReviewInput]) -> _CheckpointCardScope:
        ids: dict[str, set[int]] = {field: set() for field in _CHECKPOINT_CARD_COLUMNS}
        for card in cards:
            row = _coerce_review(card)
            _require_columns(row, _CHECKPOINT_CARD_COLUMNS)
            card_id = _checkpoint_required_id(row["card_id"], "card_id")
            ids["card_id"].add(card_id)
            ids["note_id"].add(
                _checkpoint_optional_id(row["note_id"], "note_id", card_id=card_id)
            )
            ids["deck_id"].add(_checkpoint_optional_id(row["deck_id"], "deck_id"))
            ids["preset_id"].add(_checkpoint_optional_id(row["preset_id"], "preset_id"))
        return cls(
            card_ids=frozenset(ids["card_id"]),
            note_ids=frozenset(ids["note_id"]),
            deck_ids=frozenset(ids["deck_id"]),
            preset_ids=frozenset(ids["preset_id"]),
        )

    @classmethod
    def from_metadata(cls, value: Mapping[str, Any]) -> _CheckpointCardScope:
        expected = {
            "card_ids": "card_id",
            "note_ids": "note_id",
            "deck_ids": "deck_id",
            "preset_ids": "preset_id",
        }
        unknown = set(value) - set(expected)
        if unknown:
            raise ValueError(
                f"Checkpoint state_scope contains unsupported fields: {sorted(unknown)}"
            )
        missing = set(expected) - set(value)
        if missing:
            raise ValueError(
                f"Checkpoint state_scope is missing fields: {sorted(missing)}"
            )
        normalized = {
            source: frozenset(
                _checkpoint_required_id(item, field) for item in value[source]
            )
            for source, field in expected.items()
        }
        return cls(**normalized)

    def to_metadata(self) -> dict[str, list[int]]:
        return {
            "card_ids": sorted(self.card_ids),
            "note_ids": sorted(self.note_ids),
            "deck_ids": sorted(self.deck_ids),
            "preset_ids": sorted(self.preset_ids),
        }

    def is_subset_of(self, other: _CheckpointCardScope) -> bool:
        return (
            self.card_ids <= other.card_ids
            and self.note_ids <= other.note_ids
            and self.deck_ids <= other.deck_ids
            and self.preset_ids <= other.preset_ids
        )

    def require_review(self, row: Mapping[str, Any]) -> None:
        _require_columns(row, _CHECKPOINT_CARD_COLUMNS)
        card_id = _checkpoint_required_id(row["card_id"], "card_id")
        identities = {
            "card_id": (card_id, self.card_ids),
            "note_id": (
                _checkpoint_optional_id(row["note_id"], "note_id", card_id=card_id),
                self.note_ids,
            ),
            "deck_id": (
                _checkpoint_optional_id(row["deck_id"], "deck_id"),
                self.deck_ids,
            ),
            "preset_id": (
                _checkpoint_optional_id(row["preset_id"], "preset_id"),
                self.preset_ids,
            ),
        }
        unavailable = [
            f"{field}={identity}"
            for field, (identity, available) in identities.items()
            if identity not in available
        ]
        if unavailable:
            raise ValueError(
                "Review identity is outside the selectively loaded checkpoint scope: "
                + ", ".join(unavailable)
                + ". Reload the checkpoint with this card included or omit cards= to "
                "load the full state."
            )


def _checkpoint_scope_from_metadata(value: Any) -> _CheckpointCardScope | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("Checkpoint state_scope must be a mapping or null.")
    return _CheckpointCardScope.from_metadata(value)


def _checkpoint_required_id(value: Any, field: str) -> int:
    value = _unwrap_scalar(value)
    if _is_missing(value):
        raise ValueError(f"Checkpoint card field {field!r} must not be missing.")
    try:
        return _canonical_int(value, field)
    except ValueError as exc:
        raise ValueError(
            f"Checkpoint card field {field!r} must be an integer."
        ) from exc


def _checkpoint_optional_id(
    value: Any,
    field: str,
    *,
    card_id: int | None = None,
) -> int:
    value = _unwrap_scalar(value)
    if _is_missing(value):
        if field == "note_id":
            if card_id is None:
                raise ValueError(
                    "Checkpoint card note_id normalization requires card_id."
                )
            return _ID_PLACEHOLDER + card_id
        return _ID_PLACEHOLDER
    return _checkpoint_required_id(value, field)


def _available_model_ids(model_dir: str | Path | None = None) -> tuple[str, ...]:
    roots = (
        (PRETRAINED_MODEL_DIR, TEST_MODEL_FIXTURE_DIR)
        if model_dir is None
        else (Path(model_dir),)
    )
    model_ids: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        model_ids.update(path.stem for path in root.glob("*.pth"))
        model_ids.update(path.stem for path in root.glob("*.pt"))
    return tuple(sorted(model_ids))


def _resolve_model(model: str | Path | None) -> tuple[str | None, Path]:
    if model is None:
        raise ValueError("model must not be None")

    if isinstance(model, str) and not any(sep in model for sep in ("/", "\\")):
        return model, _get_model_path(model)

    path = Path(model)
    if not path.exists():
        raise FileNotFoundError(f"RWKV-SRS model file does not exist: {path}")
    return None, path


def _get_model_path(model_id: str) -> Path:
    candidates = (
        TEST_MODEL_FIXTURE_DIR / f"{model_id}.pth",
        PRETRAINED_MODEL_DIR / f"{model_id}.pt",
        PRETRAINED_MODEL_DIR / f"{model_id}.pth",
    )
    for path in candidates:
        if path.exists():
            return path
    available = ", ".join(_available_model_ids()) or "none"
    expected = " or ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Unknown RWKV-SRS model {model_id!r}. "
        f"Expected {expected}; available models: {available}."
    )


def _coerce_review(review: ReviewInput) -> dict[str, Any]:
    if hasattr(review, "to_dict"):
        return review.to_dict()
    return dict(review)


def _require_columns(row: Mapping[str, Any], columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in row]
    if missing:
        raise ValueError(f"Review input is missing required fields: {missing}")


def _validate_batch_size(batch_size: int, *, name: str = "batch_size") -> int:
    if isinstance(batch_size, bool) or not isinstance(batch_size, numbers.Integral):
        raise TypeError(f"{name} must be an integer.")
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError(f"{name} must be at least 1.")
    return batch_size


def _validate_num_threads(
    num_threads: int | None,
    *,
    name: str = "num_threads",
) -> int | None:
    if num_threads is None:
        return None
    if isinstance(num_threads, bool) or not isinstance(num_threads, numbers.Integral):
        raise TypeError(f"{name} must be an integer.")
    num_threads = int(num_threads)
    if num_threads < 1:
        raise ValueError(f"{name} must be at least 1.")
    return num_threads


def _validate_retention_probability(
    retention_probability: float,
    *,
    name: str = "retention_probability",
) -> float:
    if isinstance(retention_probability, bool) or not isinstance(
        retention_probability,
        numbers.Real,
    ):
        raise TypeError(f"{name} must be a real number.")
    retention_probability = float(retention_probability)
    if not math.isfinite(retention_probability):
        raise ValueError(f"{name} must be finite.")
    if not 0.0 < retention_probability < 1.0:
        raise ValueError(f"{name} must be greater than 0 and less than 1.")
    return retention_probability


def _validate_elapsed_seconds(
    elapsed_seconds: float,
    *,
    name: str = "elapsed_seconds",
) -> float:
    if isinstance(elapsed_seconds, bool) or not isinstance(
        elapsed_seconds,
        numbers.Real,
    ):
        raise TypeError(f"{name} must be a real number.")
    elapsed_seconds = float(elapsed_seconds)
    if not math.isfinite(elapsed_seconds):
        raise ValueError(f"{name} must be finite.")
    if elapsed_seconds < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return elapsed_seconds


def _validate_elapsed_seconds_many(
    elapsed_seconds: Iterable[float],
) -> list[float]:
    if isinstance(elapsed_seconds, (str, bytes)):
        raise TypeError("elapsed_seconds must be an iterable of real numbers.")
    try:
        values = iter(elapsed_seconds)
    except TypeError as exc:
        raise TypeError(
            "elapsed_seconds must be an iterable of real numbers."
        ) from exc
    return [
        _validate_elapsed_seconds(value, name=f"elapsed_seconds[{index}]")
        for index, value in enumerate(values)
    ]


def _coerce_probability_many_inputs(
    curves: Iterable[Any],
    elapsed_seconds: Iterable[float],
) -> tuple[list[Any], list[float]]:
    if isinstance(curves, (str, bytes)):
        raise TypeError("curves must be an iterable of two-component curves.")
    try:
        curve_values = list(curves)
    except TypeError as exc:
        raise TypeError(
            "curves must be an iterable of two-component curves."
        ) from exc
    elapsed_values = _validate_elapsed_seconds_many(elapsed_seconds)
    if len(curve_values) != len(elapsed_values):
        raise ValueError(
            "curves and elapsed_seconds must have equal lengths; "
            f"got {len(curve_values)} curves and {len(elapsed_values)} elapsed values."
        )
    return curve_values, elapsed_values


def _coerce_review_batches(
    reviews: Iterable[ReviewInput],
    *,
    required_columns: tuple[str, ...],
    batch_size: int,
) -> Iterable[list[dict[str, Any]]]:
    batch_size = _validate_batch_size(batch_size)
    batch: list[dict[str, Any]] = []
    for review in reviews:
        row = _coerce_review(review)
        _require_columns(row, required_columns)
        batch.append(row)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _fingerprint_reviews(
    reviews: Iterable[ReviewInput],
    *,
    limit: int,
) -> dict[str, Any]:
    digest = _INITIAL_HISTORY_DIGEST
    count = 0
    last_review_id = None

    for review in reviews:
        if count >= limit:
            break
        row = _coerce_review(review)
        _require_columns(row, _HISTORY_FINGERPRINT_FIELDS)
        digest = _chain_digest(digest, row)
        count += 1
        last_review_id = _normalize_review_id(row["review_id"])

    return {
        "version": _HISTORY_FINGERPRINT_VERSION,
        "algorithm": _HISTORY_FINGERPRINT_ALGORITHM,
        "canonicalization": _HISTORY_FINGERPRINT_CANONICALIZATION,
        "fields": list(_HISTORY_FINGERPRINT_FIELDS),
        "last_review_id": last_review_id,
        "processed_review_count": count,
        "digest": digest,
    }


def _chain_digest(previous_digest: str, row: ReviewInput) -> str:
    h = hashlib.sha256()
    h.update(bytes.fromhex(previous_digest))
    h.update(_canonical_review_bytes(row))
    return h.hexdigest()


def _canonical_review_bytes(row: ReviewInput) -> bytes:
    record = {
        field: _canonical_review_value(field, row[field])
        for field in _HISTORY_FINGERPRINT_FIELDS
    }
    return json.dumps(
        record,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("ascii")


def _canonical_review_value(field: str, value: Any) -> Any:
    value = _unwrap_scalar(value)
    if _is_missing(value):
        return None
    if field in _HISTORY_INTEGER_FIELDS:
        return _canonical_int(value, field)
    if field in _HISTORY_FLOAT_FIELDS:
        return _canonical_float(value, field)
    return value


def _unwrap_scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if type(value).__name__ in {"NAType", "NaTType"}:
        return True
    return isinstance(value, numbers.Real) and math.isnan(float(value))


def _canonical_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        value = float(value)
        if math.isfinite(value) and value.is_integer():
            return int(value)
    raise ValueError(f"Review field {field!r} must be an integer or missing.")


def _canonical_float(value: Any, field: str) -> float:
    if isinstance(value, numbers.Real):
        value = float(value)
        if math.isfinite(value):
            return value
    raise ValueError(f"Review field {field!r} must be a finite number or missing.")


def _validate_fingerprint_metadata(fingerprint: Mapping[str, Any]) -> None:
    expected = {
        "version": _HISTORY_FINGERPRINT_VERSION,
        "algorithm": _HISTORY_FINGERPRINT_ALGORITHM,
        "canonicalization": _HISTORY_FINGERPRINT_CANONICALIZATION,
        "fields": list(_HISTORY_FINGERPRINT_FIELDS),
    }
    for key, value in expected.items():
        if fingerprint.get(key) != value:
            raise ValueError(
                f"Unsupported history fingerprint {key}: "
                f"expected {value!r}, got {fingerprint.get(key)!r}"
            )


def _normalize_review_id(review_id: Any) -> Any:
    review_id = _unwrap_scalar(review_id)
    if isinstance(review_id, numbers.Integral):
        return int(review_id)
    if isinstance(review_id, numbers.Real):
        review_id = float(review_id)
        return int(review_id) if review_id.is_integer() else review_id
    return review_id
