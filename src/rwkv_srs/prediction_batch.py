from __future__ import annotations

import importlib
import struct
from collections.abc import Iterable
from typing import Any

from rwkv_srs._api_core import ReviewInput


_PREDICTION_RECORD = struct.Struct("<qqqqqqqqddd")


def _native_prediction_batch_type() -> Any:
    try:
        native = importlib.import_module("rwkv_srs._native")
        return native.NativePredictionBatch
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "PredictionBatch requires the current Rust backend extension. Build it "
            "with `scripts/build_rust_release_extension.sh`."
        ) from exc


class PredictionBatch:
    """Immutable Rust-owned inputs for repeated high-throughput prediction.

    ``PredictionBatch(reviews)`` normalizes ordinary mappings once. A caller
    that already owns columnar data can use :meth:`from_buffer` with this
    little-endian fixed-record layout::

        <qqqqqqqqddd

    The fields are ``review_id``, ``card_id``, then presence/value pairs for
    ``note_id``, ``deck_id``, and ``preset_id``, followed by ``day_offset``,
    ``elapsed_days``, and ``elapsed_seconds``. Presence is 0 or 1 and each
    record is exactly 88 bytes. The input is copied into normalized Rust-owned
    state, so later caller mutation cannot affect predictions.

    This input type is intentionally Rust-only. The Torch backend is a frozen
    correctness oracle and does not implement the native batch entrypoint.
    """

    __slots__ = ("_native_batch",)
    _native_batch: Any
    RECORD_FORMAT = _PREDICTION_RECORD.format
    RECORD_SIZE = _PREDICTION_RECORD.size

    def __init__(self, reviews: Iterable[ReviewInput]) -> None:
        native_type = _native_prediction_batch_type()
        object.__setattr__(self, "_native_batch", native_type(reviews))

    @classmethod
    def from_buffer(cls, records: Any) -> PredictionBatch:
        """Build from contiguous fixed-schema records without Python row objects."""
        try:
            view = memoryview(records)
        except TypeError as exc:
            raise TypeError("records must support the Python buffer protocol.") from exc
        if not view.c_contiguous:
            raise ValueError("records must be C-contiguous.")
        try:
            encoded = records if isinstance(records, bytes) else view.cast("B").tobytes()
        except TypeError as exc:
            raise ValueError(
                "records must be castable to a contiguous byte buffer."
            ) from exc
        native_type = _native_prediction_batch_type()
        return cls._from_native(native_type.from_record_buffer(encoded))

    @classmethod
    def _from_native(cls, native_batch: Any) -> PredictionBatch:
        batch = object.__new__(cls)
        object.__setattr__(batch, "_native_batch", native_batch)
        return batch

    def __len__(self) -> int:
        return len(self._native_batch)

    def __getitem__(self, index: slice) -> PredictionBatch:
        if not isinstance(index, slice):
            raise TypeError("PredictionBatch supports slices, not individual row access.")
        start, end, step = index.indices(len(self))
        if step != 1:
            raise ValueError("PredictionBatch slices must use a step of 1.")
        return self._from_native(self._native_batch.slice(start, end))

    def __setattr__(self, name: str, value: Any) -> None:
        del name, value
        raise AttributeError("PredictionBatch is immutable.")
