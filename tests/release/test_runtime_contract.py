from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rwkv_srs import GpuUnavailableError, LiveCandidateSeed, RWKV_SRS


def _runtime(model: Path, data: dict[str, Any]) -> RWKV_SRS:
    return RWKV_SRS(
        model=model,
        seed=int(data["seed"]),
        cpu_mode="fast",
        undo_limit=4,
    )


def test_checkpoint_roundtrip_preserves_predictions_and_metadata(
    synthetic_model: Path,
    release_data: dict[str, Any],
    tmp_path: Path,
) -> None:
    runtime = _runtime(synthetic_model, release_data)
    runtime.process_many(
        release_data["process_rows"],
        batch_size=3,
        mode="fast",
    )
    expected = runtime.predict_many(
        release_data["prediction_rows"],
        batch_size=3,
        mode="fast",
    )
    checkpoint = tmp_path / "roundtrip.bin"
    runtime.save_checkpoint(checkpoint)

    restored = RWKV_SRS(checkpoint=checkpoint, cpu_mode="fast")
    assert restored.processed_review_count == runtime.processed_review_count
    assert restored.last_review_id == runtime.last_review_id
    assert restored.predict_many(
        release_data["prediction_rows"],
        batch_size=2,
        mode="fast",
    ) == pytest.approx(expected, abs=1e-7)


def test_undo_and_live_session_restore_exact_candidate_index(
    synthetic_model: Path,
    release_data: dict[str, Any],
) -> None:
    runtime = _runtime(synthetic_model, release_data)
    runtime.process_many(release_data["process_rows"], batch_size=3, mode="fast")
    seeds = [
        LiveCandidateSeed(
            row=row,
            target_retrievability=0.9,
            intraday_target_retrievability=0.85,
            tie_breaker=index % 2,
        )
        for index, row in enumerate(release_data["prediction_rows"])
    ]
    live = runtime.predict_many_live_session(
        seeds,
        initial_target_timestamp_seconds=release_data[
            "initial_target_timestamp_seconds"
        ],
        initial_target_day_offset=release_data["initial_target_day_offset"],
        order="retrievability_ascending",
        mode="fast",
        batch_size=3,
        refresh_limit=3,
    )
    before = live.snapshot()
    assert live.initial_result.active_count == len(seeds)
    prediction = live.process_answer(
        release_data["live_answer"],
        requeue_after_prediction=True,
        return_curves=False,
    )
    assert isinstance(prediction, float)
    live.refresh(
        target_timestamp_seconds=release_data["initial_target_timestamp_seconds"]
        + 60.0,
        target_day_offset=release_data["initial_target_day_offset"],
        select_limit=3,
    )
    assert live.current_undo_depth == 1
    assert live.undo_last_process() == 0
    assert live.snapshot() == before
    live.close()
    live.close()
    with pytest.raises(RuntimeError, match="closed"):
        live.current_selection(select_limit=1)


def test_cpu_batch_failure_leaves_runtime_state_unchanged(
    synthetic_model: Path,
    release_data: dict[str, Any],
    tmp_path: Path,
) -> None:
    runtime = _runtime(synthetic_model, release_data)
    runtime.process_many(release_data["process_rows"][:2], mode="fast")
    prediction_before = runtime.predict_many(
        release_data["prediction_rows"],
        mode="fast",
    )
    metadata_before = (
        runtime.processed_review_count,
        runtime.last_review_id,
    )
    before = tmp_path / "before.bin"
    after = tmp_path / "after.bin"
    runtime.save_checkpoint(before)

    invalid = dict(release_data["process_rows"][3])
    del invalid["rating"]
    with pytest.raises(ValueError):
        runtime.process_many(
            [release_data["process_rows"][2], invalid],
            batch_size=2,
            mode="fast",
        )

    runtime.save_checkpoint(after)
    assert after.read_bytes() == before.read_bytes()
    assert (
        runtime.processed_review_count,
        runtime.last_review_id,
    ) == metadata_before
    assert runtime.predict_many(
        release_data["prediction_rows"],
        mode="fast",
    ) == pytest.approx(prediction_before, abs=1e-7)


def test_gpu_unavailable_can_fall_back_without_gpu_hardware(
    synthetic_model: Path,
    release_data: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RWKV_SRS_GPU_ADAPTER", "rwkv-release-test-missing-adapter")
    baseline_rows = release_data["process_rows"][:4]
    prediction_rows = release_data["prediction_rows"]

    runtime = _runtime(synthetic_model, release_data)
    control = _runtime(synthetic_model, release_data)
    runtime.process_many(baseline_rows, mode="fast")
    control.process_many(baseline_rows, mode="fast")
    expected = control.predict_many(prediction_rows, mode="fast")

    # Force the initialization failure through the public preflight API. This
    # does not depend on a GPU being present and exposes the structured error a
    # caller can use to choose CPU Fast safely before submitting mutable work.
    strict = _runtime(synthetic_model, release_data)
    with pytest.raises(GpuUnavailableError) as raised:
        strict.initialize_gpu("predict")
    assert raised.value.committed_rows == 0
    assert raised.value.state_recoverable is True
    assert raised.value.retryable_on_cpu is True

    assert runtime.predict_many(
        prediction_rows,
        mode="gpu",
        fallback_mode="fast",
    ) == pytest.approx(expected, abs=1e-7)

    gpu_process = _runtime(synthetic_model, release_data)
    cpu_process = _runtime(synthetic_model, release_data)
    actual_process = gpu_process.process_many(
        baseline_rows,
        mode="gpu",
        fallback_mode="fast",
    )
    expected_process = cpu_process.process_many(baseline_rows, mode="fast")
    assert actual_process == pytest.approx(expected_process, abs=1e-7)
    assert gpu_process.predict_many(
        prediction_rows,
        mode="fast",
    ) == pytest.approx(
        cpu_process.predict_many(prediction_rows, mode="fast"),
        abs=1e-7,
    )
