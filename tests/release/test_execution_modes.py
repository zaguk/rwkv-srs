from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import pytest


FIXTURE = Path(__file__).with_name("fixtures") / "reviews.json"
PROBE = Path(__file__).with_name("runtime_probe.py")
OUTPUT_FIELDS = (
    "scalar_process",
    "batch_process",
    "scalar_predictions",
    "oracle_predictions",
    "fast_predictions",
)


@pytest.mark.parametrize(
    ("expected_key", "disable_simd"),
    (("default_simd", False), ("forced_scalar", True)),
)
def test_fast_execution_modes_match_frozen_outputs(
    synthetic_model: Path,
    expected_outputs: dict[str, Any],
    expected_key: str,
    disable_simd: bool,
) -> None:
    env = os.environ.copy()
    env["RWKV_SRS_BACKEND"] = "rust"
    if disable_simd:
        env["RWKV_SRS_DISABLE_SIMD"] = "1"
    else:
        env.pop("RWKV_SRS_DISABLE_SIMD", None)
    result = subprocess.run(
        [
            sys.executable,
            str(PROBE),
            "--model",
            str(synthetic_model),
            "--fixture",
            str(FIXTURE),
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    actual = json.loads(result.stdout)
    assert actual["native_api_version"] == 33
    assert actual["cpu_mode"] == "fast"
    assert actual["simd_status"]["disabled_by_env"] is disable_simd
    if disable_simd:
        assert actual["simd_status"]["linear_kernel"] == "candle_fallback"
    elif platform.machine().lower() in {"aarch64", "arm64"}:
        assert actual["simd_status"]["linear_kernel"] == "pulp"
        assert actual["simd_status"]["pulp_arch"] == "neon"
    elif platform.machine().lower() in {"amd64", "x86_64"}:
        # Modern x86 runners normally select the tuned AVX2/FMA kernel. Pulp is
        # the valid portable SIMD dispatch when those exact features are not
        # available. An older x86 host may legitimately reach the scalar
        # fallback, but the selected kernel must agree with native detection.
        if actual["simd_status"]["custom_avx2_fma_available"]:
            assert actual["simd_status"]["linear_kernel"] == "avx2_fma"
        elif actual["simd_status"]["pulp_available"]:
            assert actual["simd_status"]["linear_kernel"] == "pulp"
        else:
            assert actual["simd_status"]["linear_kernel"] == "candle_fallback"

    expected = expected_outputs[expected_key]
    for field in OUTPUT_FIELDS:
        assert actual[field] == pytest.approx(expected[field], abs=5e-5)
    assert actual["scalar_process"] == pytest.approx(
        actual["batch_process"],
        abs=5e-5,
    )
    assert actual["scalar_predictions"] == pytest.approx(
        actual["oracle_predictions"],
        abs=5e-5,
    )
