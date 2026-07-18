from __future__ import annotations

from importlib import metadata
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


EXPECTED_NATIVE_API_VERSION = 33
FIXTURE_DIR = Path(__file__).with_name("fixtures")


def test_release_fixture_provenance_matches_contents() -> None:
    provenance = json.loads(
        (FIXTURE_DIR / "PROVENANCE.json").read_text(encoding="utf-8")
    )
    assert provenance["contains_external_data"] is False
    assert provenance["contains_trained_parameters"] is False
    for relative, record in provenance["files"].items():
        path = FIXTURE_DIR / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_package_import_is_lazy_and_native_api_matches() -> None:
    code = """
import json
import sys

import rwkv_srs
lazy = "rwkv_srs._native" not in sys.modules
import rwkv_srs._native as native
from rwkv_srs import LiveCandidateSeed, RWKV_SRS

print(json.dumps({
    "backend": native.backend_name(),
    "lazy": lazy,
    "native_api_version": native.native_api_version(),
    "runtime_name": RWKV_SRS.__name__,
    "seed_name": LiveCandidateSeed.__name__,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env=os.environ.copy(),
        text=True,
    )
    value = json.loads(result.stdout)
    assert value == {
        "backend": "rust",
        "lazy": True,
        "native_api_version": EXPECTED_NATIVE_API_VERSION,
        "runtime_name": "RWKV_SRS",
        "seed_name": "LiveCandidateSeed",
    }


def test_artifact_run_uses_installed_rust_only_package() -> None:
    if os.environ.get("RWKV_SRS_RELEASE_ARTIFACT_TEST") != "1":
        return

    import rwkv_srs

    package_path = Path(rwkv_srs.__file__).resolve()
    distribution_root = Path(
        metadata.distribution("rwkv-srs").locate_file("")
    ).resolve()
    assert distribution_root in package_path.parents
    source_root_value = os.environ.get("RWKV_SRS_RELEASE_SOURCE_ROOT")
    if source_root_value:
        for value in source_root_value.split(os.pathsep):
            source_root = Path(value).resolve()
            source_package = source_root / "src" / "rwkv_srs"
            assert source_package not in package_path.parents
    assert importlib.util.find_spec("rwkv_srs.backends.torch") is None


def test_artifact_contains_third_party_notices_without_project_license() -> None:
    if os.environ.get("RWKV_SRS_RELEASE_ARTIFACT_TEST") != "1":
        return

    distribution = metadata.distribution("rwkv-srs")
    assert distribution.metadata.get("License") is None
    assert distribution.metadata.get("License-Expression") is None
    assert "Repository, https://github.com/zaguk/rwkv-srs" in (
        distribution.metadata.get_all("Project-URL") or ()
    )
    expected = {
        "THIRD_PARTY_LICENSES.txt",
        "THIRD_PARTY_NOTICES.md",
    }
    assert set(distribution.metadata.get_all("License-File", ())) == expected

    files = distribution.files or ()
    license_files = {
        Path(entry.name): distribution.locate_file(entry)
        for entry in files
        if ".dist-info/licenses/" in str(entry)
    }
    assert set(license_files) == {Path(name) for name in expected}
    assert (
        license_files[Path("THIRD_PARTY_NOTICES.md")]
        .read_text(encoding="utf-8")
        .startswith("# Third-party notices\n")
    )
    assert (
        license_files[Path("THIRD_PARTY_LICENSES.txt")]
        .read_text(encoding="utf-8")
        .startswith("RWKV-SRS THIRD-PARTY LICENSE TEXT ARCHIVE\n")
    )
