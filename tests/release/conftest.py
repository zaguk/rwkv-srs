from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.release.model_fixture import write_synthetic_model


FIXTURE_DIR = Path(__file__).with_name("fixtures")
SYNTHETIC_MODEL_SHA256 = (
    "06af8329c7193896b212d63f7409587cc85d70e94679f64099196e1ffbec4dfc"
)


@pytest.fixture(scope="session")
def release_data() -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / "reviews.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def expected_outputs() -> dict[str, Any]:
    return json.loads(
        (FIXTURE_DIR / "expected_outputs.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="session")
def synthetic_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("model") / "public-synthetic.safetensors"
    digest = write_synthetic_model(path)
    assert digest == SYNTHETIC_MODEL_SHA256
    return path
