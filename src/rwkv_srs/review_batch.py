from __future__ import annotations

import importlib
import struct
from collections.abc import Iterable
from typing import Any

from rwkv_srs._api_core import ReviewInput


_PROCESS_REVIEW_RECORD = struct.Struct("<qqqqqqqqdddqdd")


def _native_review_batch_type() -> Any:
    try:
        native = importlib.import_module("rwkv_srs._native")
        return native.NativeReviewBatch
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "ReviewBatch requires the current Rust backend extension. Build it with "
            "`scripts/build_rust_release_extension.sh`."
        ) from exc


class ReviewBatch:
    """Immutable native process rows for high-throughput history ingestion.

    ``ReviewBatch(reviews)`` parses ordinary mapping rows once in Rust. For a
    columnar producer, :meth:`from_buffer` accepts contiguous fixed records in
    this little-endian layout::

        <qqqqqqqqdddqdd

    The fields are ``review_id``, ``card_id``, then presence/value pairs for
    ``note_id``, ``deck_id``, and ``preset_id``, followed by ``day_offset``,
    ``elapsed_days``, ``elapsed_seconds``, ``rating``, ``duration``, and
    ``state``. Presence is 0 or 1 and each record is exactly 112 bytes. The
    buffer is copied into normalized Rust-owned state, so later caller
    mutation cannot affect processing or history bookkeeping.
    """

    __slots__ = ("_native_batch",)
    _native_batch: Any
    RECORD_FORMAT = _PROCESS_REVIEW_RECORD.format
    RECORD_SIZE = _PROCESS_REVIEW_RECORD.size

    def __init__(self, reviews: Iterable[ReviewInput]) -> None:
        native_type = _native_review_batch_type()
        object.__setattr__(self, "_native_batch", native_type(reviews))

    @classmethod
    def from_buffer(cls, records: Any) -> ReviewBatch:
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
            raise ValueError("records must be castable to a contiguous byte buffer.") from exc
        native_type = _native_review_batch_type()
        return cls._from_native(native_type.from_record_buffer(encoded))

    @classmethod
    def _from_native(cls, native_batch: Any) -> ReviewBatch:
        batch = object.__new__(cls)
        object.__setattr__(batch, "_native_batch", native_batch)
        return batch

    def __len__(self) -> int:
        return len(self._native_batch)

    def __getitem__(self, index: slice) -> ReviewBatch:
        if not isinstance(index, slice):
            raise TypeError("ReviewBatch supports slices, not individual row access.")
        start, end, step = index.indices(len(self))
        if step != 1:
            raise ValueError("ReviewBatch slices must use a step of 1.")
        return self._slice(start, end)

    def __setattr__(self, name: str, value: Any) -> None:
        del name, value
        raise AttributeError("ReviewBatch is immutable.")

    def _slice(self, start: int, end: int) -> ReviewBatch:
        return self._from_native(self._native_batch.slice(start, end))

    def _history_advance(
        self,
        previous_digest: str | None,
        count: int | None = None,
    ) -> tuple[str | None, int | None]:
        digest, last_review_id = self._native_batch.history_advance(
            previous_digest,
            count,
        )
        return digest, last_review_id

    def _matches_history_fingerprint(
        self,
        *,
        digest: str,
        processed_review_count: int,
        last_review_id: int | None,
    ) -> bool:
        return bool(
            self._native_batch.matches_history_fingerprint(
                digest,
                processed_review_count,
                last_review_id,
            )
        )
