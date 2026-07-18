#!/usr/bin/env python3
"""Create and verify RWKV-SRS wheel checksums and release provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tomllib
from typing import Any

try:
    from scripts.rust_wheel_contract import (
        PUBLIC_MANYLINUX_POLICY,
        WheelContractError,
        WheelIdentity,
        verify_rust_wheel,
    )
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from rust_wheel_contract import (  # type: ignore[import-not-found,no-redef]
        PUBLIC_MANYLINUX_POLICY,
        WheelContractError,
        WheelIdentity,
        verify_rust_wheel,
    )


TARGETS: dict[str, tuple[str, str]] = {
    "linux-x86_64": ("linux", "x86_64"),
    "linux-aarch64": ("linux", "aarch64"),
    "macos-x86_64": ("macos", "x86_64"),
    "macos-aarch64": ("macos", "aarch64"),
    "windows-x86_64": ("windows", "x86_64"),
    "windows-aarch64": ("windows", "aarch64"),
}
PROVENANCE_SUFFIX = ".provenance.json"
CHECKSUM_SUFFIX = ".sha256"
RELEASE_PROVENANCE_NAME = "RELEASE_PROVENANCE.json"
RELEASE_CHECKSUMS_NAME = "SHA256SUMS"
SOURCE_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ReleaseManifestError(ValueError):
    """Raised when release metadata does not match its wheel or source."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_version(root: Path) -> str:
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    cargo = tomllib.loads(
        (root / "rust" / "rwkv-srs-cpu" / "Cargo.toml").read_text(encoding="utf-8")
    )
    python_version = str(pyproject["project"]["version"])
    rust_version = str(cargo["package"]["version"])
    if python_version != rust_version:
        raise ReleaseManifestError(
            "Python and Rust package versions differ: "
            f"{python_version!r} != {rust_version!r}"
        )
    return python_version


def validate_release_tag(tag: str, version: str) -> None:
    expected = f"v{version}"
    if tag != expected:
        raise ReleaseManifestError(
            f"release tag {tag!r} does not match package version {version!r}; "
            f"expected {expected!r}"
        )


def _is_git_root(root: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return Path(result.stdout.strip()).resolve() == root.resolve()


def resolve_source_commit(root: Path, supplied: str | None = None) -> str:
    repository_commit: str | None = None
    if _is_git_root(root):
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        repository_commit = result.stdout.strip().lower()
    if supplied:
        supplied = supplied.lower()
        if SOURCE_COMMIT_PATTERN.fullmatch(supplied) is None:
            raise ReleaseManifestError(f"invalid supplied source commit: {supplied!r}")
        if repository_commit is not None and supplied != repository_commit:
            raise ReleaseManifestError(
                f"supplied source commit {supplied} does not match HEAD "
                f"{repository_commit}"
            )
        return supplied
    github_commit = os.environ.get("GITHUB_SHA", "").lower()
    if github_commit:
        if SOURCE_COMMIT_PATTERN.fullmatch(github_commit) is None:
            raise ReleaseManifestError(
                f"invalid GITHUB_SHA source commit: {github_commit!r}"
            )
        if repository_commit is not None and github_commit != repository_commit:
            raise ReleaseManifestError(
                f"GITHUB_SHA {github_commit} does not match HEAD {repository_commit}"
            )
        return github_commit
    if repository_commit is not None:
        return repository_commit

    candidates: list[str] = []
    export_provenance = root / "PUBLIC_EXPORT_PROVENANCE.json"
    if export_provenance.is_file():
        try:
            value = json.loads(export_provenance.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                candidates.append(str(value.get("source_commit", "")))
        except (OSError, json.JSONDecodeError):
            pass
    for candidate in candidates:
        candidate = candidate.lower()
        if SOURCE_COMMIT_PATTERN.fullmatch(candidate):
            return candidate
    raise ReleaseManifestError("could not determine a full 40-character source commit")


def source_tree_state(root: Path) -> str:
    if _is_git_root(root):
        result = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return "dirty" if result.stdout else "clean"
    if (root / "PUBLIC_EXPORT_PROVENANCE.json").is_file():
        return "exported"
    return "unknown"


def _command_version(command: list[str], *, cwd: Path) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return (result.stdout or result.stderr).strip().splitlines()[0]


def _python_version(python: Path, *, cwd: Path) -> str:
    return _command_version(
        [str(python), "-c", "import platform; print(platform.python_version())"],
        cwd=cwd,
    )


def _target(target_id: str) -> tuple[str, str]:
    try:
        return TARGETS[target_id]
    except KeyError as error:
        raise ReleaseManifestError(
            f"unsupported release target: {target_id!r}"
        ) from error


def _verify_target_wheel(
    wheel: Path,
    *,
    target_id: str,
    expected_version: str,
) -> WheelIdentity:
    operating_system, architecture = _target(target_id)
    return verify_rust_wheel(
        wheel,
        manylinux_policy=(
            PUBLIC_MANYLINUX_POLICY if operating_system == "linux" else None
        ),
        expected_os=operating_system,
        expected_arch=architecture,
        expected_version=expected_version,
    )


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_wheel_sidecars(
    wheel: Path,
    *,
    root: Path,
    build_python: Path,
    tested_pythons: tuple[Path, ...],
    target_id: str,
    source_commit: str | None = None,
    source_ref: str | None = None,
    release_tag: str | None = None,
) -> tuple[Path, Path]:
    """Validate a tested wheel and write checksum/provenance sidecars."""

    root = root.resolve()
    wheel = wheel.resolve()
    version = source_version(root)
    if release_tag:
        validate_release_tag(release_tag, version)
    identity = _verify_target_wheel(
        wheel,
        target_id=target_id,
        expected_version=version,
    )
    operating_system, architecture = _target(target_id)
    commit = resolve_source_commit(root, source_commit)
    export_provenance = root / "PUBLIC_EXPORT_PROVENANCE.json"
    public_export: dict[str, str] | None = None
    if export_provenance.is_file():
        value = json.loads(export_provenance.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ReleaseManifestError("PUBLIC_EXPORT_PROVENANCE.json is not an object")
        private_source_commit = str(value.get("source_commit", "")).lower()
        if SOURCE_COMMIT_PATTERN.fullmatch(private_source_commit) is None:
            raise ReleaseManifestError(
                "PUBLIC_EXPORT_PROVENANCE.json has an invalid source commit"
            )
        public_export = {
            "manifest_sha256": sha256(export_provenance),
            "private_source_commit": private_source_commit,
        }

    tested_versions = sorted(
        {_python_version(path, cwd=root) for path in tested_pythons}
    )
    record: dict[str, Any] = {
        "schema_version": 1,
        "artifact": {
            "filename": wheel.name,
            "sha256": sha256(wheel),
            "size": wheel.stat().st_size,
            "type": "wheel",
        },
        "package": {
            "distribution": identity.distribution,
            "version": identity.version,
        },
        "target": {
            "id": target_id,
            "operating_system": operating_system,
            "architecture": architecture,
        },
        "wheel": {
            "python_tag": identity.python_tag,
            "abi_tag": identity.abi_tag,
            "platform_tags": list(identity.platform_tags),
        },
        "build": {
            "cargo_profile": "release-ci",
            "cpu_tuning": "portable",
            "pgo": False,
            "manylinux_policy": (
                PUBLIC_MANYLINUX_POLICY if operating_system == "linux" else None
            ),
            "builder_python": _python_version(build_python, cwd=root),
            "tested_python_versions": tested_versions,
            "rustc": _command_version(["rustc", "--version"], cwd=root),
            "cargo": _command_version(["cargo", "--version"], cwd=root),
            "maturin": _command_version(
                [str(build_python), "-m", "maturin", "--version"], cwd=root
            ),
        },
        "source": {
            "commit": commit,
            "ref": source_ref or os.environ.get("GITHUB_REF"),
            "tree_state": source_tree_state(root),
            "cargo_lock_sha256": sha256(root / "rust" / "rwkv-srs-cpu" / "Cargo.lock"),
            "pyproject_sha256": sha256(root / "pyproject.toml"),
            "public_export": public_export,
        },
        "validation": {
            "installed_wheel_tests": "tests/release",
            "tested_wheel_sha256": sha256(wheel),
        },
        "github_actions": {
            key.lower(): os.environ[key]
            for key in (
                "GITHUB_REPOSITORY",
                "GITHUB_RUN_ID",
                "GITHUB_RUN_ATTEMPT",
                "GITHUB_WORKFLOW_REF",
            )
            if os.environ.get(key)
        }
        or None,
    }
    provenance = wheel.with_name(wheel.name + PROVENANCE_SUFFIX)
    checksum = wheel.with_name(wheel.name + CHECKSUM_SUFFIX)
    _atomic_write(
        provenance,
        (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    _atomic_write(
        checksum,
        f"{record['artifact']['sha256']}  {wheel.name}\n".encode("ascii"),
    )
    return provenance, checksum


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseManifestError(
            f"invalid JSON manifest {path.name}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ReleaseManifestError(f"manifest is not an object: {path.name}")
    return value


def _object_field(
    record: dict[str, Any], field: str, *, manifest: Path
) -> dict[str, Any]:
    value = record.get(field)
    if not isinstance(value, dict):
        raise ReleaseManifestError(
            f"manifest field {field!r} is not an object: {manifest.name}"
        )
    return value


def assemble_release_assets(
    artifact_dir: Path,
    *,
    release_tag: str,
    expected_source_commit: str | None = None,
) -> tuple[Path, Path]:
    """Verify all six tested wheels and create aggregate release manifests."""

    artifact_dir = artifact_dir.resolve()
    prohibited = sorted(
        path.name
        for path in artifact_dir.iterdir()
        if path.suffix.lower() in {".pyd", ".so", ".dll", ".dylib"}
        or path.name.endswith(".tar.gz")
    )
    if prohibited:
        raise ReleaseManifestError(
            f"release directory contains prohibited standalone artifacts: {prohibited}"
        )
    wheels = sorted(artifact_dir.glob("*.whl"))
    if len(wheels) != len(TARGETS):
        raise ReleaseManifestError(
            f"expected {len(TARGETS)} release wheels, found {len(wheels)}"
        )

    expected_files = {
        path.name
        for wheel in wheels
        for path in (
            wheel,
            wheel.with_name(wheel.name + PROVENANCE_SUFFIX),
            wheel.with_name(wheel.name + CHECKSUM_SUFFIX),
        )
    }
    expected_files.update({RELEASE_PROVENANCE_NAME, RELEASE_CHECKSUMS_NAME})
    unexpected_files = sorted(
        path.name
        for path in artifact_dir.iterdir()
        if not path.is_file() or path.name not in expected_files
    )
    if unexpected_files:
        raise ReleaseManifestError(
            f"release directory contains unexpected files: {unexpected_files}"
        )

    records: list[dict[str, Any]] = []
    targets: set[str] = set()
    versions: set[str] = set()
    commits: set[str] = set()
    cargo_lock_hashes: set[str] = set()
    pyproject_hashes: set[str] = set()
    checksum_lines: list[str] = []
    for wheel in wheels:
        provenance_path = wheel.with_name(wheel.name + PROVENANCE_SUFFIX)
        checksum_path = wheel.with_name(wheel.name + CHECKSUM_SUFFIX)
        if not provenance_path.is_file() or not checksum_path.is_file():
            raise ReleaseManifestError(f"wheel sidecars are missing for {wheel.name}")
        record = _read_json(provenance_path)
        if record.get("schema_version") != 1:
            raise ReleaseManifestError(
                f"unsupported provenance schema for {wheel.name}"
            )
        artifact = _object_field(record, "artifact", manifest=provenance_path)
        target = _object_field(record, "target", manifest=provenance_path)
        package = _object_field(record, "package", manifest=provenance_path)
        source = _object_field(record, "source", manifest=provenance_path)
        build = _object_field(record, "build", manifest=provenance_path)
        wheel_record = _object_field(record, "wheel", manifest=provenance_path)
        validation = _object_field(record, "validation", manifest=provenance_path)
        target_id = str(target.get("id", ""))
        version = str(package.get("version", ""))
        commit = str(source.get("commit", "")).lower()
        tree_state = str(source.get("tree_state", ""))
        digest = sha256(wheel)
        if artifact.get("filename") != wheel.name:
            raise ReleaseManifestError(
                f"provenance filename does not match {wheel.name}"
            )
        if (
            artifact.get("type") != "wheel"
            or artifact.get("sha256") != digest
            or artifact.get("size") != wheel.stat().st_size
        ):
            raise ReleaseManifestError(
                f"provenance digest or size does not match {wheel.name}"
            )
        if package.get("distribution") != "rwkv-srs":
            raise ReleaseManifestError(
                f"unexpected provenance distribution for {wheel.name}"
            )
        expected_checksum = f"{digest}  {wheel.name}\n"
        if checksum_path.read_text(encoding="ascii") != expected_checksum:
            raise ReleaseManifestError(f"checksum sidecar does not match {wheel.name}")
        identity = _verify_target_wheel(
            wheel,
            target_id=target_id,
            expected_version=version,
        )
        operating_system, architecture = _target(target_id)
        if (
            target.get("operating_system") != operating_system
            or target.get("architecture") != architecture
        ):
            raise ReleaseManifestError(
                f"provenance target fields do not match {target_id}"
            )
        if wheel_record != {
            "python_tag": identity.python_tag,
            "abi_tag": identity.abi_tag,
            "platform_tags": list(identity.platform_tags),
        }:
            raise ReleaseManifestError(
                f"provenance wheel tags do not match {wheel.name}"
            )
        tested_versions = build.get("tested_python_versions", [])
        if not isinstance(tested_versions, list) or not all(
            isinstance(value, str) for value in tested_versions
        ):
            raise ReleaseManifestError(
                f"invalid tested Python versions for {wheel.name}"
            )
        for required in ("3.11", "3.14"):
            if not any(
                value == required or value.startswith(f"{required}.")
                for value in tested_versions
            ):
                raise ReleaseManifestError(
                    f"{wheel.name} was not tested with Python {required}"
                )
        if (
            build.get("cargo_profile") != "release-ci"
            or build.get("cpu_tuning") != "portable"
            or build.get("pgo") is not False
        ):
            raise ReleaseManifestError(
                f"non-portable release build provenance for {wheel.name}"
            )
        expected_policy = (
            PUBLIC_MANYLINUX_POLICY if operating_system == "linux" else None
        )
        if build.get("manylinux_policy") != expected_policy:
            raise ReleaseManifestError(
                f"incorrect manylinux provenance for {wheel.name}"
            )
        if validation.get("tested_wheel_sha256") != digest:
            raise ReleaseManifestError(
                f"tested-wheel digest does not match {wheel.name}"
            )
        if target_id in targets:
            raise ReleaseManifestError(f"duplicate release target: {target_id}")
        if SOURCE_COMMIT_PATTERN.fullmatch(commit) is None:
            raise ReleaseManifestError(
                f"invalid source commit for {wheel.name}: {commit!r}"
            )
        if tree_state not in {"clean", "exported"}:
            raise ReleaseManifestError(
                f"non-publishable source tree state for {wheel.name}: {tree_state!r}"
            )
        targets.add(target_id)
        versions.add(version)
        commits.add(commit)
        cargo_lock_hashes.add(str(source.get("cargo_lock_sha256", "")))
        pyproject_hashes.add(str(source.get("pyproject_sha256", "")))
        checksum_lines.append(expected_checksum.rstrip("\n"))
        records.append(record)

    if targets != set(TARGETS):
        raise ReleaseManifestError(
            f"release targets differ from the required matrix: {sorted(targets)}"
        )
    if len(versions) != 1:
        raise ReleaseManifestError(f"release wheel versions differ: {sorted(versions)}")
    version = next(iter(versions))
    validate_release_tag(release_tag, version)
    if len(commits) != 1:
        raise ReleaseManifestError(f"release source commits differ: {sorted(commits)}")
    if (
        len(cargo_lock_hashes) != 1
        or SHA256_PATTERN.fullmatch(next(iter(cargo_lock_hashes))) is None
    ):
        raise ReleaseManifestError("release Cargo.lock hashes differ or are missing")
    if (
        len(pyproject_hashes) != 1
        or SHA256_PATTERN.fullmatch(next(iter(pyproject_hashes))) is None
    ):
        raise ReleaseManifestError("release pyproject hashes differ or are missing")
    commit = next(iter(commits))
    if expected_source_commit is not None and commit != expected_source_commit.lower():
        raise ReleaseManifestError(
            f"release source commit {commit} does not match {expected_source_commit}"
        )

    aggregate = {
        "schema_version": 1,
        "release_tag": release_tag,
        "package_version": version,
        "source_commit": commit,
        "published_artifact_policy": "wheels-and-manifests-only",
        "wheels": records,
    }
    provenance = artifact_dir / RELEASE_PROVENANCE_NAME
    checksums = artifact_dir / RELEASE_CHECKSUMS_NAME
    _atomic_write(
        provenance,
        (json.dumps(aggregate, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    _atomic_write(
        checksums,
        ("\n".join(sorted(checksum_lines)) + "\n").encode("ascii"),
    )
    return provenance, checksums


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="write one wheel's sidecars")
    create.add_argument("--wheel", type=Path, required=True)
    create.add_argument("--root", type=Path, default=Path.cwd())
    create.add_argument("--build-python", type=Path, default=Path(sys.executable))
    create.add_argument("--test-python", type=Path, action="append", default=[])
    create.add_argument("--target", choices=tuple(TARGETS), required=True)
    create.add_argument("--source-commit")
    create.add_argument("--source-ref")
    create.add_argument("--release-tag")

    assemble = subparsers.add_parser(
        "assemble", help="verify six target wheels and create release manifests"
    )
    assemble.add_argument("--artifact-dir", type=Path, required=True)
    assemble.add_argument("--release-tag", required=True)
    assemble.add_argument("--expected-source-commit")
    args = parser.parse_args()

    try:
        if args.command == "create":
            tested = tuple(args.test_python) or (args.build_python,)
            paths = write_wheel_sidecars(
                args.wheel,
                root=args.root,
                build_python=args.build_python,
                tested_pythons=tested,
                target_id=args.target,
                source_commit=args.source_commit,
                source_ref=args.source_ref,
                release_tag=args.release_tag,
            )
        else:
            paths = assemble_release_assets(
                args.artifact_dir,
                release_tag=args.release_tag,
                expected_source_commit=args.expected_source_commit,
            )
    except (
        OSError,
        subprocess.CalledProcessError,
        WheelContractError,
        ReleaseManifestError,
    ) as error:
        raise SystemExit(f"release manifest failed: {error}") from error
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
