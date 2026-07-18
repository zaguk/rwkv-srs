from __future__ import annotations

from typing import Any

from rwkv_srs._api_core import (
    GpuError,
    GpuExecutionError,
    GpuOperation,
    GpuOutOfMemoryError,
    GpuProcessError,
    GpuUnavailableError,
    UndoUnavailableError,
    _validate_gpu_operation,
)
from rwkv_srs._backend import selected_backend
from rwkv_srs.live import (
    LiveCandidateSeed,
    LiveCandidateSnapshot,
    LiveCandidateStatus,
    LiveOrder,
    LivePredictionSessionProtocol,
    LiveRefreshResult,
    LiveSelection,
)
from rwkv_srs.prediction_batch import PredictionBatch
from rwkv_srs.review_batch import ReviewBatch


def _load_backend_attr(name: str) -> Any:
    selected = selected_backend()
    if selected == "rust":
        from rwkv_srs.backends import rust

        return getattr(rust, name)

    try:
        from importlib import import_module

        torch = import_module("rwkv_srs.backends.torch")
    except ModuleNotFoundError as exc:  # pragma: no cover
        if exc.name == "rwkv_srs.backends.torch":
            raise RuntimeError(
                "This RWKV-SRS distribution is Rust-only and does not include "
                "the internal Torch oracle. Use the default Rust backend."
            ) from exc
        if exc.name in {"numpy", "torch"}:
            raise RuntimeError(
                "The internal Torch oracle requires the repository's full test "
                "environment. Install `requirements.txt`, or use the default "
                "Rust backend."
            ) from exc
        raise
    return getattr(torch, name)


def _load_rwkv_srs() -> type[Any]:
    return _load_backend_attr("RWKV_SRS")


def gpu_device_info(operation: GpuOperation = "predict") -> dict[str, Any]:
    """Return Rust GPU adapter details without importing Rust at module import time."""
    operation = _validate_gpu_operation(operation)
    # Preserve the backend import's detailed diagnostic. In particular, an API
    # version mismatch must not be relabeled as a merely missing extension.
    from rwkv_srs.backends.rust import gpu_device_info as rust_gpu_device_info

    return rust_gpu_device_info(operation)


def check_checkpoint_history_consistency(checkpoint: Any, reviews: Any) -> bool:
    """Compare review history with a Rust checkpoint without loading its state."""
    from rwkv_srs.backends.rust import (
        check_checkpoint_history_consistency as rust_check_checkpoint_history,
    )

    return bool(rust_check_checkpoint_history(checkpoint, reviews))


def __getattr__(name: str) -> Any:
    if name == "RWKV_SRS":
        return _load_rwkv_srs()
    if name == "get_interval":
        return _load_backend_attr("get_interval")
    if name == "get_probability":
        return _load_backend_attr("get_probability")
    if name == "get_probability_many":
        return _load_backend_attr("get_probability_many")
    if name == "gpu_available":
        return _load_backend_attr("gpu_available")
    raise AttributeError(name)


def backend_name() -> str:
    return selected_backend()


__all__ = [
    "RWKV_SRS",
    "PredictionBatch",
    "ReviewBatch",
    "LiveCandidateSeed",
    "LiveCandidateSnapshot",
    "LiveCandidateStatus",
    "LiveOrder",
    "LivePredictionSessionProtocol",
    "LiveRefreshResult",
    "LiveSelection",
    "GpuError",
    "GpuExecutionError",
    "GpuOutOfMemoryError",
    "GpuProcessError",
    "GpuUnavailableError",
    "UndoUnavailableError",
    "backend_name",
    "check_checkpoint_history_consistency",
    "get_interval",
    "get_probability",
    "get_probability_many",
    "gpu_available",
    "gpu_device_info",
]
