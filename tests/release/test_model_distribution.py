from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Any

import pytest


def test_downstream_packaged_repository_model_resolves(
    release_data: dict[str, Any],
) -> None:
    model_id = os.environ.get("RWKV_SRS_RELEASE_PACKAGED_MODEL_ID")
    if model_id is None:
        pytest.skip("downstream model-overlay test is not active")

    from rwkv_srs import RWKV_SRS

    runtime = RWKV_SRS(model=model_id, seed=int(release_data["seed"]))
    expected_sha256 = os.environ["RWKV_SRS_RELEASE_PACKAGED_MODEL_SHA256"]
    model_path = Path(runtime.model_path)

    assert runtime.model_id == model_id
    assert model_path.name == f"{model_id}.safetensors"
    assert model_path.parent.name == "pretrained"
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == expected_sha256

    prediction = runtime.predict(release_data["prediction_rows"][0])
    assert math.isfinite(prediction)
    assert 0.0 <= prediction <= 1.0
