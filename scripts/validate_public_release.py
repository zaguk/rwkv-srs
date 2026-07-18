#!/usr/bin/env python3
"""Build and test public RWKV-SRS artifacts outside the source checkout."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile

try:
    from scripts.release_manifest import (
        TARGETS,
        source_version,
        validate_release_tag,
        write_wheel_sidecars,
    )
    from scripts.rust_wheel_contract import (
        PUBLIC_MANYLINUX_POLICY,
        verify_rust_wheel,
    )
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from release_manifest import (  # type: ignore[import-not-found,no-redef]
        TARGETS,
        source_version,
        validate_release_tag,
        write_wheel_sidecars,
    )
    from rust_wheel_contract import (  # type: ignore[import-not-found,no-redef]
        PUBLIC_MANYLINUX_POLICY,
        verify_rust_wheel,
    )


FORBIDDEN_SDIST_SUFFIXES = (
    "src/rwkv_srs/_state.py",
    "src/rwkv_srs/_torch_checkpoint.py",
    "src/rwkv_srs/api.py",
    "src/rwkv_srs/architecture.py",
    "src/rwkv_srs/config.py",
    "src/rwkv_srs/cpu_inference.py",
    "src/rwkv_srs/data_processing.py",
    "src/rwkv_srs/backends/torch.py",
)
REQUIRED_SDIST_SUFFIXES = (
    "THIRD_PARTY_LICENSES.txt",
    "THIRD_PARTY_NOTICES.md",
    "rust-toolchain.toml",
    "scripts/build_local_artifact.py",
    "scripts/generate_third_party_notices.py",
    "scripts/release_manifest.py",
    "tests/release/test_import_contract.py",
    "tests/release/test_execution_modes.py",
    "tests/release/test_model_distribution.py",
    "tests/release/test_runtime_contract.py",
    "tests/release/fixtures/PROVENANCE.json",
)
CARGO_TEST_MODEL_NAMES = (
    "RWKV_trained_on_101_4999.safetensors",
    "RWKV_trained_on_5000_10000.safetensors",
)
DOWNSTREAM_MODEL_ID = "RWKV_trained_on_101_4999"
DOWNSTREAM_MODEL_RELATIVE = Path(
    "tests/fixtures/models/RWKV_trained_on_101_4999.safetensors"
)


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _create_test_environment(
    base_python: Path,
    destination: Path,
    *,
    root: Path,
) -> Path:
    _run([str(base_python), "-m", "venv", str(destination)], cwd=root)
    python = _venv_python(destination)
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "pytest>=8,<10",
        ],
        cwd=root,
    )
    return python


def _artifact_test_environment(source_roots: tuple[Path, ...]) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["RWKV_SRS_BACKEND"] = "rust"
    env["RWKV_SRS_RELEASE_ARTIFACT_TEST"] = "1"
    env["RWKV_SRS_RELEASE_SOURCE_ROOT"] = os.pathsep.join(
        str(root) for root in source_roots
    )
    return env


def _run_release_tests(
    python: Path,
    test_root: Path,
    *,
    source_roots: tuple[Path, ...],
) -> None:
    _run(
        [str(python), "-m", "pytest", "-q", "tests/release"],
        cwd=test_root,
        env=_artifact_test_environment(source_roots),
    )


def _installed_pretrained_model_dir(python: Path, *, cwd: Path) -> Path:
    result = subprocess.run(
        [
            str(python),
            "-c",
            (
                "from rwkv_srs._api_core import PRETRAINED_MODEL_DIR; "
                "print(PRETRAINED_MODEL_DIR)"
            ),
        ],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def _test_downstream_model_packaging(
    python: Path,
    test_root: Path,
    *,
    root: Path,
    source_roots: tuple[Path, ...],
) -> None:
    source_model = root / DOWNSTREAM_MODEL_RELATIVE
    if not source_model.is_file():
        raise RuntimeError(f"repository model is missing: {source_model}")

    model_dir = _installed_pretrained_model_dir(python, cwd=test_root)
    model_dir.mkdir(parents=True, exist_ok=True)
    packaged_model = model_dir / f"{DOWNSTREAM_MODEL_ID}.safetensors"
    shutil.copy2(source_model, packaged_model)

    env = _artifact_test_environment(source_roots)
    env["RWKV_SRS_RELEASE_PACKAGED_MODEL_ID"] = DOWNSTREAM_MODEL_ID
    env["RWKV_SRS_RELEASE_PACKAGED_MODEL_SHA256"] = _sha256(source_model)
    _run(
        [
            str(python),
            "-m",
            "pytest",
            "-q",
            (
                "tests/release/test_model_distribution.py::"
                "test_downstream_packaged_repository_model_resolves"
            ),
        ],
        cwd=test_root,
        env=env,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _python_runtime_directories(python: Path, *, cwd: Path) -> tuple[Path, ...]:
    result = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import os, sys, sysconfig; "
                "values = (sys.base_prefix, sys.base_exec_prefix, sys.prefix, "
                "sys.exec_prefix, os.path.dirname(sys.executable), "
                "sysconfig.get_config_var('BINDIR'), "
                "sysconfig.get_config_var('LIBDIR')); "
                "print('\\n'.join(str(value) for value in values if value))"
            ),
        ],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    directories: list[Path] = []
    seen: set[str] = set()
    for line in result.stdout.splitlines():
        directory = Path(line).resolve()
        for candidate in (directory, directory / "DLLs"):
            key = os.path.normcase(str(candidate))
            if candidate.is_dir() and key not in seen:
                seen.add(key)
                directories.append(candidate)
    return tuple(directories)


def _native_test_environment(root: Path, python: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYO3_PYTHON"] = str(python)
    if os.name == "nt":
        runtime_directories = _python_runtime_directories(python, cwd=root)
        if not runtime_directories:
            raise RuntimeError(
                f"could not locate Windows runtime DLL directories for {python}"
            )
        existing_path = env.get("PATH")
        path_entries = [str(path) for path in runtime_directories]
        if existing_path:
            path_entries.append(existing_path)
        env["PATH"] = os.pathsep.join(path_entries)
    return env


def _test_native_crate(root: Path, python: Path) -> None:
    model_dir = root / "tests" / "fixtures" / "models"
    missing = [
        model_dir / name
        for name in CARGO_TEST_MODEL_NAMES
        if not (model_dir / name).is_file()
    ]
    if missing:
        command = [
            str(python),
            str(root / "tests" / "release" / "model_fixture.py"),
        ]
        for path in missing:
            command.extend(("--output", str(path)))
        _run(command, cwd=root)
    _run(
        [
            "cargo",
            "test",
            "--locked",
            "--manifest-path",
            str(root / "rust" / "rwkv-srs-cpu" / "Cargo.toml"),
        ],
        cwd=root,
        env=_native_test_environment(root, python),
    )


def _verify_third_party_notices(root: Path, python: Path) -> None:
    _run(
        [
            str(python),
            str(root / "scripts" / "generate_third_party_notices.py"),
            "--check",
        ],
        cwd=root,
    )


def _build_wheel(
    root: Path,
    python: Path,
    destination: Path,
    *,
    target_id: str,
    expected_version: str,
) -> Path:
    destination.mkdir(parents=True)
    if platform.system() == "Linux":
        env = os.environ.copy()
        env["PYTHON"] = str(python)
        env["OUT_DIR"] = str(destination)
        _run(
            ["bash", str(root / "scripts" / "build_rust_ci_wheel.sh")],
            cwd=root,
            env=env,
        )
    else:
        _run(
            [
                str(python),
                "-m",
                "maturin",
                "build",
                "--locked",
                "--profile",
                "release-ci",
                "--out",
                str(destination),
            ],
            cwd=root,
        )
        wheels = sorted(destination.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"expected one wheel, found {wheels}")
    wheels = sorted(destination.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one wheel, found {wheels}")
    operating_system, architecture = TARGETS[target_id]
    verify_rust_wheel(
        wheels[0],
        manylinux_policy=(
            PUBLIC_MANYLINUX_POLICY if operating_system == "linux" else None
        ),
        expected_os=operating_system,
        expected_arch=architecture,
        expected_version=expected_version,
    )
    return wheels[0]


def _build_sdist(root: Path, python: Path, destination: Path) -> Path:
    destination.mkdir(parents=True)
    _run(
        [str(python), "-m", "maturin", "sdist", "--out", str(destination)],
        cwd=root,
    )
    archives = sorted(destination.glob("*.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected one source distribution, found {archives}")
    return archives[0]


def _sdist_members(sdist: Path) -> list[str]:
    with tarfile.open(sdist, "r:gz") as archive:
        members = archive.getmembers()
    unsafe = [
        member.name
        for member in members
        if member.issym()
        or member.islnk()
        or Path(member.name).is_absolute()
        or ".." in Path(member.name).parts
    ]
    if unsafe:
        raise RuntimeError(f"unsafe source-distribution members: {unsafe}")
    return [member.name for member in members if member.isfile()]


def _verify_sdist(sdist: Path) -> None:
    names = _sdist_members(sdist)
    missing = [
        suffix
        for suffix in REQUIRED_SDIST_SUFFIXES
        if not any(name.endswith(suffix) for name in names)
    ]
    forbidden = [
        name
        for name in names
        if any(name.endswith(suffix) for suffix in FORBIDDEN_SDIST_SUFFIXES)
        or "/src/rwkv_srs/model/" in name
        or "/src/rwkv_srs/pretrained/" in name
        or "/tests/fixtures/models/" in name
    ]
    if missing:
        raise RuntimeError(f"source distribution is missing release files: {missing}")
    if forbidden:
        raise RuntimeError(f"source distribution contains forbidden files: {forbidden}")


def _extract_sdist(sdist: Path, destination: Path) -> Path:
    destination.mkdir(parents=True)
    _sdist_members(sdist)
    with tarfile.open(sdist, "r:gz") as archive:
        for member in archive.getmembers():
            target = destination / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise RuntimeError(f"unsupported archive member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"could not read archive member: {member.name}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            target.chmod(member.mode & 0o777)
    roots = [path for path in destination.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise RuntimeError(f"expected one extracted source root, found {roots}")
    return roots[0]


def _copy_release_tests(root: Path, destination: Path, name: str) -> Path:
    test_root = destination / name
    (test_root / "tests").mkdir(parents=True)
    shutil.copy2(root / "tests" / "__init__.py", test_root / "tests" / "__init__.py")
    shutil.copytree(root / "tests" / "release", test_root / "tests" / "release")
    return test_root


def _test_wheel(
    wheel: Path,
    *,
    root: Path,
    base_python: Path,
    work: Path,
) -> None:
    python = _create_test_environment(base_python, work / "wheel-venv", root=root)
    _run(
        [str(python), "-m", "pip", "install", "--no-deps", str(wheel)],
        cwd=work,
    )
    test_root = _copy_release_tests(root, work, "wheel-test-root")
    source_roots = (root,)
    _run_release_tests(
        python,
        test_root,
        source_roots=source_roots,
    )
    _test_downstream_model_packaging(
        python,
        test_root,
        root=root,
        source_roots=source_roots,
    )


def _test_sdist(
    sdist: Path,
    *,
    root: Path,
    base_python: Path,
    work: Path,
) -> None:
    _verify_sdist(sdist)
    source = _extract_sdist(sdist, work / "sdist-source")
    python = _create_test_environment(base_python, work / "sdist-venv", root=root)
    _run(
        [str(python), "-m", "pip", "install", "--no-deps", str(sdist)],
        cwd=work,
    )
    _run_release_tests(
        python,
        _copy_release_tests(source, work, "sdist-test-root"),
        source_roots=(root, source),
    )


def _host_target_id() -> str:
    system = platform.system().lower()
    operating_system = {
        "linux": "linux",
        "darwin": "macos",
        "windows": "windows",
    }.get(system)
    machine = platform.machine().lower()
    architecture = {
        "amd64": "x86_64",
        "x86_64": "x86_64",
        "arm64": "aarch64",
        "aarch64": "aarch64",
    }.get(machine)
    target_id = f"{operating_system}-{architecture}"
    if operating_system is None or architecture is None or target_id not in TARGETS:
        raise RuntimeError(
            f"unsupported release host: system={platform.system()!r}, "
            f"machine={platform.machine()!r}"
        )
    return target_id


def _resolve_python(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    # Do not resolve the final symlink: POSIX virtualenv interpreters commonly
    # point at the base executable, and replacing the venv path with that base
    # path silently loses the environment's installed build dependencies.
    return Path(os.path.abspath(path))


def _unique_pythons(paths: list[Path]) -> tuple[Path, ...]:
    result: list[Path] = []
    seen: set[str] = set()
    for value in paths:
        path = _resolve_python(value)
        key = os.path.normcase(str(path))
        if key not in seen:
            seen.add(key)
            result.append(path)
    return tuple(result)


def _prepare_empty_directory(path: Path, *, description: str) -> Path:
    path = path.resolve()
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"{description} must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _verify_directory_layout(work: Path, output: Path) -> None:
    if output == work or work.is_relative_to(output):
        raise SystemExit(
            "output directory must not equal or contain the validation work directory"
        )


def _publish_validated_wheel(wheel: Path, output_dir: Path) -> Path:
    destination = output_dir / wheel.name
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{wheel.name}.", suffix=".tmp", dir=output_dir
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(wheel, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    if _sha256(destination) != _sha256(wheel):
        destination.unlink(missing_ok=True)
        raise RuntimeError("published wheel bytes differ from the tested wheel")
    return destination


def _verify_portable_build_environment() -> None:
    if os.environ.get("RWKV_SRS_PGO", "0") not in {"", "0"}:
        raise RuntimeError("public release validation does not permit PGO builds")
    flags = " ".join(
        os.environ.get(name, "") for name in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS")
    ).lower()
    forbidden = (
        "target-cpu",
        "target-feature",
        "profile-use",
        "profile-generate",
    )
    present = [value for value in forbidden if value in flags]
    if present:
        raise RuntimeError(
            "public release validation owns portable CPU and profile settings; "
            f"remove {present} from Rust flags"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-kind",
        choices=("wheel", "sdist", "both"),
        default="both",
    )
    parser.add_argument(
        "--build-python",
        type=Path,
        default=Path(sys.executable),
        help="Python containing Maturin (and Zig support on Linux).",
    )
    parser.add_argument(
        "--test-python",
        type=Path,
        action="append",
        default=[],
        help=(
            "Python used to install and test the exact wheel; repeat to test "
            "the same abi3 wheel with multiple interpreter versions."
        ),
    )
    parser.add_argument(
        "--sdist-test-python",
        type=Path,
        help="Python used for validation-only source-distribution testing.",
    )
    parser.add_argument(
        "--expected-target",
        choices=tuple(TARGETS),
        help="Require the exact release OS/architecture wheel tag.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Empty persistent directory that receives only the exact tested "
            "wheel and its checksum/provenance sidecars."
        ),
    )
    parser.add_argument("--source-commit")
    parser.add_argument("--source-ref")
    parser.add_argument(
        "--release-tag",
        help="When non-empty, require an exact v<package-version> tag.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Keep build/test work under this empty directory.",
    )
    parser.add_argument(
        "--skip-cargo-tests",
        action="store_true",
        help="Skip the native crate suite during focused artifact development.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    build_python = _resolve_python(args.build_python)
    test_pythons = _unique_pythons(args.test_python or [build_python])
    sdist_test_python = _resolve_python(args.sdist_test_python or build_python)
    target_id = args.expected_target or _host_target_id()
    release_tag = args.release_tag or None
    if args.work_dir is None:
        with tempfile.TemporaryDirectory(
            prefix="rwkv-srs-public-release-"
        ) as temporary:
            work = Path(temporary)
            output_dir = _prepare_empty_directory(
                args.output_dir or work / "artifacts",
                description="output directory",
            )
            _verify_directory_layout(work, output_dir)
            _validate(
                root,
                build_python,
                work,
                args.artifact_kind,
                run_cargo_tests=not args.skip_cargo_tests,
                test_pythons=test_pythons,
                sdist_test_python=sdist_test_python,
                target_id=target_id,
                output_dir=output_dir,
                source_commit=args.source_commit,
                source_ref=args.source_ref,
                release_tag=release_tag,
            )
    else:
        work = _prepare_empty_directory(
            args.work_dir,
            description="work directory",
        )
        output_dir = _prepare_empty_directory(
            args.output_dir or work / "artifacts",
            description="output directory",
        )
        _verify_directory_layout(work, output_dir)
        _validate(
            root,
            build_python,
            work,
            args.artifact_kind,
            run_cargo_tests=not args.skip_cargo_tests,
            test_pythons=test_pythons,
            sdist_test_python=sdist_test_python,
            target_id=target_id,
            output_dir=output_dir,
            source_commit=args.source_commit,
            source_ref=args.source_ref,
            release_tag=release_tag,
        )


def _validate(
    root: Path,
    python: Path,
    work: Path,
    artifact_kind: str,
    *,
    run_cargo_tests: bool,
    test_pythons: tuple[Path, ...],
    sdist_test_python: Path,
    target_id: str,
    output_dir: Path,
    source_commit: str | None,
    source_ref: str | None,
    release_tag: str | None,
) -> None:
    _verify_portable_build_environment()
    version = source_version(root)
    if release_tag:
        validate_release_tag(release_tag, version)
    _verify_third_party_notices(root, python)
    if run_cargo_tests:
        _test_native_crate(root, python)
    build_artifacts = work / "build-artifacts"
    if artifact_kind in {"wheel", "both"}:
        wheel = _build_wheel(
            root,
            python,
            build_artifacts / "wheel",
            target_id=target_id,
            expected_version=version,
        )
        for index, test_python in enumerate(test_pythons, start=1):
            _test_wheel(
                wheel,
                root=root,
                base_python=test_python,
                work=work / f"wheel-test-{index}",
            )
        published_wheel = _publish_validated_wheel(wheel, output_dir)
        write_wheel_sidecars(
            published_wheel,
            root=root,
            build_python=python,
            tested_pythons=test_pythons,
            target_id=target_id,
            source_commit=source_commit,
            source_ref=source_ref,
            release_tag=release_tag,
        )
        print(f"verified installed wheel: {published_wheel}")
    if artifact_kind in {"sdist", "both"}:
        sdist = _build_sdist(root, python, build_artifacts / "sdist")
        _test_sdist(
            sdist,
            root=root,
            base_python=sdist_test_python,
            work=work / "sdist-test",
        )
        print(f"verified validation-only source distribution: {sdist}")


if __name__ == "__main__":
    main()
