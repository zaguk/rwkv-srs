#!/usr/bin/env python3
"""Emit stable public-runtime outputs for one Fast SIMD configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rwkv_srs import RWKV_SRS
import rwkv_srs._native as native


def run_probe(model: Path, fixture: Path) -> dict[str, Any]:
    data = json.loads(fixture.read_text(encoding="utf-8"))
    process_rows = data["process_rows"]
    prediction_rows = data["prediction_rows"]
    seed = int(data["seed"])

    scalar = RWKV_SRS(model=model, seed=seed, cpu_mode="fast")
    scalar_process = [scalar.process(row, return_curves=False) for row in process_rows]

    batch = RWKV_SRS(model=model, seed=seed, cpu_mode="fast")
    batch_process = batch.process_many(
        process_rows,
        batch_size=3,
        return_curves=False,
        mode="fast",
    )
    scalar_predictions = [batch.predict(row) for row in prediction_rows]
    oracle_predictions = batch.predict_many(
        prediction_rows,
        batch_size=3,
        mode="oracle",
    )
    fast_predictions = batch.predict_many(
        prediction_rows,
        batch_size=3,
        mode="fast",
    )
    return {
        "native_api_version": native.native_api_version(),
        "cpu_mode": batch.cpu_mode,
        "simd_status": dict(native.simd_status()),
        "scalar_process": scalar_process,
        "batch_process": batch_process,
        "scalar_predictions": scalar_predictions,
        "oracle_predictions": oracle_predictions,
        "fast_predictions": fast_predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_probe(args.model, args.fixture), sort_keys=True))


if __name__ == "__main__":
    main()
