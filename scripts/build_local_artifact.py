#!/usr/bin/env python3
"""Build a portable release or host-native RWKV-SRS Python artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile

try:
    from scripts.rust_wheel_contract import (
        PUBLIC_MANYLINUX_POLICY,
        WheelContractError,
        verify_rust_wheel,
    )
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from rust_wheel_contract import (  # type: ignore[import-not-found,no-redef]
        PUBLIC_MANYLINUX_POLICY,
        WheelContractError,
        verify_rust_wheel,
    )


ROOT = Path(__file__).resolve().parents[1]
NATIVE_SUFFIXES = (".so", ".pyd", ".dll", ".dylib")
BUILD_PROFILES = {
    "release": "release-ci",
    "native": "release-local",
}


class BuildError(ValueError):
    """Raised when a requested local artifact would be invalid or ambiguous."""


def _run(command: list[str], *, env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _build_environment(build: str) -> dict[str, str]:
    env = os.environ.copy()
    rustflags = env.get("RUSTFLAGS", "")
    encoded = env.get("CARGO_ENCODED_RUSTFLAGS", "")
    feature_flags = f"{rustflags} {encoded}".lower()
    if "target-cpu" in feature_flags or "target-feature" in feature_flags:
        raise BuildError(
            "remove target-cpu/target-feature from RUSTFLAGS and "
            "CARGO_ENCODED_RUSTFLAGS; the selected build mode owns CPU tuning"
        )
    if build == "native":
        if encoded:
            raise BuildError(
                "native builds require CARGO_ENCODED_RUSTFLAGS to be unset so "
                "target-cpu=native can be applied unambiguously"
            )
        env["RUSTFLAGS"] = f"{rustflags} -C target-cpu=native".strip()
    return env


def _zig_path(python: Path, *, env: dict[str, str]) -> Path | None:
    configured = env.get("CARGO_ZIGBUILD_ZIG_PATH")
    if configured:
        path = Path(configured)
        return path if path.is_file() else None
    executable = shutil.which("zig", path=env.get("PATH"))
    if executable:
        return Path(executable)
    result = subprocess.run(
        [
            str(python),
            "-c",
            (
                "from pathlib import Path; import ziglang; "
                "print(Path(ziglang.__file__).with_name('zig'))"
            ),
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    path = Path(result.stdout.strip())
    return path if path.is_file() else None


def _maturin_command(
    python: Path,
    staging: Path,
    *,
    build: str,
    system: str,
) -> list[str]:
    command = [
        str(python),
        "-m",
        "maturin",
        "build",
        "--locked",
        "--profile",
        BUILD_PROFILES[build],
        "--out",
        str(staging),
    ]
    if system == "Linux":
        if build == "release":
            command.extend(
                (
                    "--zig",
                    "--compatibility",
                    PUBLIC_MANYLINUX_POLICY,
                    "--auditwheel",
                    "check",
                )
            )
        else:
            command.extend(("--compatibility", "linux", "--auditwheel", "skip"))
    return command


def _single_wheel(directory: Path) -> Path:
    wheels = sorted(directory.glob("*.whl"))
    if len(wheels) != 1:
        formatted = ", ".join(path.name for path in wheels) or "none"
        raise BuildError(
            f"Maturin must produce exactly one wheel in {directory}; found {formatted}"
        )
    return wheels[0]


def _native_member(wheel: Path) -> str:
    with zipfile.ZipFile(wheel) as archive:
        members = []
        for name in archive.namelist():
            path = PurePosixPath(name)
            if (
                len(path.parts) == 2
                and path.parts[0] == "rwkv_srs"
                and path.name.startswith("_native")
                and path.name.endswith(NATIVE_SUFFIXES)
            ):
                members.append(name)
    if len(members) != 1:
        raise BuildError(
            f"expected one native extension in {wheel}, found {sorted(members)}"
        )
    return members[0]


def _publish_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_native(wheel: Path, member: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(wheel) as archive, archive.open(member) as source:
            with temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_build_info(
    destination: Path,
    *,
    build: str,
    artifact: str,
    python: Path,
    wheel: Path | None,
    native: Path | None,
) -> None:
    data = {
        "schema_version": 1,
        "build": build,
        "cargo_profile": BUILD_PROFILES[build],
        "cpu_tuning": "portable" if build == "release" else "target-cpu=native",
        "requested_artifact": artifact,
        "platform": platform.platform(),
        "python_executable": str(python),
        "builder_python": sys.version,
        "wheel": None
        if wheel is None
        else {"file": wheel.name, "sha256": _sha256(wheel)},
        "native_extension": None
        if native is None
        else {"file": native.name, "sha256": _sha256(native)},
    }
    payload = (json.dumps(data, indent=2, sort_keys=True) + "\n").encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def build_artifacts(
    *,
    build: str,
    artifact: str,
    python: Path,
    out_dir: Path,
) -> tuple[Path | None, Path | None]:
    env = _build_environment(build)
    system = platform.system()
    if build == "release" and system == "Linux":
        zig = _zig_path(python, env=env)
        if zig is None:
            raise BuildError(
                "portable Linux release builds require Zig; install "
                "maturin[zig] or set CARGO_ZIGBUILD_ZIG_PATH"
            )
        env["CARGO_ZIGBUILD_ZIG_PATH"] = str(zig)

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rwkv-srs-local-build-") as temporary:
        staging = Path(temporary)
        _run(
            _maturin_command(
                python,
                staging,
                build=build,
                system=system,
            ),
            env=env,
        )
        staged_wheel = _single_wheel(staging)
        verify_rust_wheel(
            staged_wheel,
            manylinux_policy=(
                PUBLIC_MANYLINUX_POLICY
                if build == "release" and system == "Linux"
                else None
            ),
        )
        member = _native_member(staged_wheel)

        wheel_output: Path | None = None
        native_output: Path | None = None
        if artifact in {"wheel", "both"}:
            wheel_output = out_dir / staged_wheel.name
            _publish_file(staged_wheel, wheel_output)
        if artifact in {"extension", "both"}:
            native_output = out_dir / PurePosixPath(member).name
            _publish_native(staged_wheel, member, native_output)

    _write_build_info(
        out_dir / "build-info.json",
        build=build,
        artifact=artifact,
        python=python,
        wheel=wheel_output,
        native=native_output,
    )
    return wheel_output, native_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "build",
        choices=tuple(BUILD_PROFILES),
        help=(
            "release builds a portable distributable; native optimizes for "
            "this machine and must not be redistributed"
        ),
    )
    parser.add_argument(
        "--artifact",
        choices=("wheel", "extension", "both"),
        default="both",
        help="Output to retain after the verified build. Default: both.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory. Default: dist/local/<build>.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter containing Maturin. Default: this interpreter.",
    )
    args = parser.parse_args()
    out_dir = args.out_dir or ROOT / "dist" / "local" / args.build
    python = args.python.expanduser()
    if not python.is_absolute():
        python = Path.cwd() / python
    python = Path(os.path.abspath(python))
    if not python.is_file():
        raise SystemExit(f"Python interpreter does not exist: {python}")

    try:
        wheel, native = build_artifacts(
            build=args.build,
            artifact=args.artifact,
            python=python,
            out_dir=out_dir,
        )
    except (
        BuildError,
        OSError,
        subprocess.CalledProcessError,
        WheelContractError,
        zipfile.BadZipFile,
    ) as error:
        raise SystemExit(f"artifact build failed: {error}") from error

    print(f"build mode: {args.build}")
    print(f"Cargo profile: {BUILD_PROFILES[args.build]}")
    if wheel is not None:
        print(f"wheel: {wheel}")
    if native is not None:
        print(f"native extension: {native}")
    print(f"build metadata: {(out_dir / 'build-info.json').resolve()}")
    if args.build == "native":
        print("warning: this native build is only for CPUs compatible with this host")


if __name__ == "__main__":
    main()
