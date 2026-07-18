#!/usr/bin/env python3
"""Validate the public Rust-only RWKV-SRS wheel boundary."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import re
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path


RUNTIME_PACKAGE_FILES = frozenset(
    {
        "rwkv_srs/__init__.py",
        "rwkv_srs/_api_core.py",
        "rwkv_srs/_backend.py",
        "rwkv_srs/_rust.py",
        "rwkv_srs/live.py",
        "rwkv_srs/prediction_batch.py",
        "rwkv_srs/review_batch.py",
        "rwkv_srs/backends/__init__.py",
        "rwkv_srs/backends/rust.py",
    }
)
THIRD_PARTY_LICENSE_FILES = frozenset(
    {
        "THIRD_PARTY_LICENSES.txt",
        "THIRD_PARTY_NOTICES.md",
    }
)
_NATIVE_SUFFIXES = (".so", ".pyd", ".dll", ".dylib")
PUBLIC_MANYLINUX_POLICY = "manylinux_2_38"
PUBLIC_PYTHON_TAG = "cp39"
PUBLIC_ABI_TAG = "abi3"
PUBLIC_DISTRIBUTION = "rwkv-srs"


@dataclass(frozen=True)
class WheelIdentity:
    """Validated identity and compatibility tags for one release wheel."""

    distribution: str
    version: str
    python_tag: str
    abi_tag: str
    platform_tags: tuple[str, ...]


class WheelContractError(ValueError):
    """Raised when a wheel crosses the Rust-only package boundary."""


def verify_rust_wheel(
    wheel: str | Path,
    *,
    manylinux_policy: str | None = None,
    expected_os: str | None = None,
    expected_arch: str | None = None,
    expected_version: str | None = None,
) -> WheelIdentity:
    wheel = Path(wheel)
    filename_identity = _filename_identity(wheel)

    with zipfile.ZipFile(wheel) as archive:
        names = {name for name in archive.namelist() if not name.endswith("/")}
        package_files = {name for name in names if name.startswith("rwkv_srs/")}
        native_files = {
            name
            for name in package_files
            if name.startswith("rwkv_srs/_native") and name.endswith(_NATIVE_SUFFIXES)
        }
        if len(native_files) != 1:
            raise WheelContractError(
                f"expected one native extension, found {sorted(native_files)}"
            )

        missing = RUNTIME_PACKAGE_FILES - package_files
        unexpected = package_files - RUNTIME_PACKAGE_FILES - native_files
        if missing:
            raise WheelContractError(
                f"Rust wheel is missing runtime files: {sorted(missing)}"
            )
        if unexpected:
            raise WheelContractError(
                f"Rust wheel contains non-runtime files: {sorted(unexpected)}"
            )

        metadata_files = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise WheelContractError(
                f"expected one METADATA file, found {metadata_files}"
            )
        metadata = BytesParser(policy=policy.default).parsebytes(
            archive.read(metadata_files[0])
        )

        license_files = {
            Path(name).name for name in names if ".dist-info/licenses/" in name
        }
        if license_files != THIRD_PARTY_LICENSE_FILES:
            raise WheelContractError(
                "wheel third-party license files differ from the required set: "
                f"{sorted(license_files)}"
            )

        wheel_metadata_files = [
            name for name in names if name.endswith(".dist-info/WHEEL")
        ]
        if len(wheel_metadata_files) != 1:
            raise WheelContractError(
                f"expected one WHEEL metadata file, found {wheel_metadata_files}"
            )
        wheel_metadata = BytesParser(policy=policy.default).parsebytes(
            archive.read(wheel_metadata_files[0])
        )

    distribution = str(metadata.get("Name", ""))
    version = str(metadata.get("Version", ""))
    if distribution != PUBLIC_DISTRIBUTION:
        raise WheelContractError(
            f"expected distribution {PUBLIC_DISTRIBUTION!r}, found {distribution!r}"
        )
    if not version:
        raise WheelContractError("wheel metadata does not declare a version")
    if expected_version is not None and version != expected_version:
        raise WheelContractError(
            f"expected version {expected_version!r}, found {version!r}"
        )

    expected_prefix = f"{distribution.replace('-', '_')}-{version}"
    prefix, python_tag, abi_tag, filename_platforms = filename_identity
    if prefix != expected_prefix:
        raise WheelContractError(
            f"wheel filename prefix {prefix!r} does not match {expected_prefix!r}"
        )
    if python_tag != PUBLIC_PYTHON_TAG or abi_tag != PUBLIC_ABI_TAG:
        raise WheelContractError(
            "expected the stable ABI tags "
            f"{PUBLIC_PYTHON_TAG}-{PUBLIC_ABI_TAG}, found {python_tag}-{abi_tag}"
        )

    wheel_tags = wheel_metadata.get_all("Tag", [])
    metadata_platforms = _metadata_platforms(
        wheel_tags,
        expected_python_tag=python_tag,
        expected_abi_tag=abi_tag,
    )
    if set(filename_platforms) != set(metadata_platforms):
        raise WheelContractError(
            "wheel filename and WHEEL metadata platform tags differ: "
            f"{sorted(filename_platforms)} != {sorted(metadata_platforms)}"
        )

    if manylinux_policy is not None:
        _verify_manylinux_policy(
            wheel,
            filename_platforms,
            metadata_platforms,
            manylinux_policy,
        )
    if expected_os is not None or expected_arch is not None:
        if expected_os is None or expected_arch is None:
            raise WheelContractError(
                "expected_os and expected_arch must be supplied together"
            )
        _verify_target(filename_platforms, expected_os, expected_arch)

    extras = {value.lower() for value in metadata.get_all("Provides-Extra", [])}
    if "torch" in extras:
        raise WheelContractError("Rust wheel must not advertise a Torch extra")
    for requirement in metadata.get_all("Requires-Dist", []):
        package = re.split(r"[\s\[\](<>=;]", requirement, maxsplit=1)[0]
        if package.lower().replace("_", "-") == "torch":
            raise WheelContractError("Rust wheel must not depend on Torch")

    declared_license_files = set(metadata.get_all("License-File", []))
    if declared_license_files != THIRD_PARTY_LICENSE_FILES:
        raise WheelContractError(
            "wheel License-File metadata differs from the required third-party "
            f"set: {sorted(declared_license_files)}"
        )
    if metadata.get("License") or metadata.get("License-Expression"):
        raise WheelContractError(
            "RWKV-SRS intentionally has no project-level license metadata"
        )
    return WheelIdentity(
        distribution=distribution,
        version=version,
        python_tag=python_tag,
        abi_tag=abi_tag,
        platform_tags=tuple(sorted(filename_platforms)),
    )


def _filename_identity(wheel: Path) -> tuple[str, str, str, tuple[str, ...]]:
    if wheel.suffix != ".whl":
        raise WheelContractError(f"not a wheel filename: {wheel.name}")
    try:
        prefix, python_tag, abi_tag, platforms = wheel.stem.rsplit("-", 3)
    except ValueError as error:
        raise WheelContractError(f"invalid wheel filename: {wheel.name}") from error
    platform_tags = tuple(value for value in platforms.split(".") if value)
    if not prefix or not python_tag or not abi_tag or not platform_tags:
        raise WheelContractError(f"invalid wheel filename: {wheel.name}")
    return prefix, python_tag, abi_tag, platform_tags


def _metadata_platforms(
    wheel_tags: list[str],
    *,
    expected_python_tag: str,
    expected_abi_tag: str,
) -> tuple[str, ...]:
    if not wheel_tags:
        raise WheelContractError("WHEEL metadata does not declare any tags")
    platforms: list[str] = []
    for tag in wheel_tags:
        try:
            python_tag, abi_tag, platform_values = tag.split("-", 2)
        except ValueError as error:
            raise WheelContractError(f"invalid WHEEL Tag: {tag!r}") from error
        if python_tag != expected_python_tag or abi_tag != expected_abi_tag:
            raise WheelContractError(
                f"WHEEL metadata tag does not match the filename ABI: {tag!r}"
            )
        platforms.extend(value for value in platform_values.split(".") if value)
    return tuple(platforms)


def _verify_manylinux_policy(
    wheel: Path,
    filename_platforms: tuple[str, ...],
    metadata_platforms: tuple[str, ...],
    expected_policy: str,
) -> None:
    if re.fullmatch(r"manylinux_[0-9]+_[0-9]+", expected_policy) is None:
        raise WheelContractError(
            f"invalid expected manylinux policy: {expected_policy!r}"
        )
    filename_linux_platforms = [
        platform for platform in filename_platforms if "linux" in platform
    ]
    metadata_linux_platforms = [
        platform for platform in metadata_platforms if "linux" in platform
    ]
    if not filename_linux_platforms or not metadata_linux_platforms:
        raise WheelContractError(
            f"wheel does not contain Linux platform tags: {wheel.name}"
        )
    if len(filename_linux_platforms) != len(filename_platforms) or len(
        metadata_linux_platforms
    ) != len(metadata_platforms):
        raise WheelContractError(
            f"wheel mixes Linux and non-Linux platform tags: {wheel.name}"
        )

    expected_prefix = f"{expected_policy}_"
    unexpected = sorted(
        {
            *(
                platform
                for platform in filename_linux_platforms
                if not platform.startswith(expected_prefix)
            ),
            *(
                platform
                for platform in metadata_linux_platforms
                if not platform.startswith(expected_prefix)
            ),
        }
    )
    if unexpected:
        raise WheelContractError(
            f"expected {expected_policy} Linux tags, found {unexpected}"
        )


def _verify_target(
    platform_tags: tuple[str, ...],
    expected_os: str,
    expected_arch: str,
) -> None:
    expected_os = expected_os.lower()
    expected_arch = expected_arch.lower()
    if expected_os not in {"linux", "macos", "windows"}:
        raise WheelContractError(f"unsupported expected OS: {expected_os!r}")
    if expected_arch not in {"x86_64", "aarch64"}:
        raise WheelContractError(
            f"unsupported expected architecture: {expected_arch!r}"
        )

    if expected_os == "linux":
        suffix = "x86_64" if expected_arch == "x86_64" else "aarch64"
        pattern = re.compile(rf"(?:manylinux_[0-9]+_[0-9]+|linux)_{suffix}$")
    elif expected_os == "macos":
        suffix = "x86_64" if expected_arch == "x86_64" else "arm64"
        pattern = re.compile(rf"macosx_[0-9]+_[0-9]+_{suffix}$")
    else:
        suffix = "amd64" if expected_arch == "x86_64" else "arm64"
        pattern = re.compile(rf"win_{suffix}$")

    unexpected = sorted(tag for tag in platform_tags if pattern.fullmatch(tag) is None)
    if unexpected:
        raise WheelContractError(
            f"wheel is not for {expected_os}/{expected_arch}: {unexpected}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manylinux-policy",
        help="Require this exact manylinux policy in filename and WHEEL tags.",
    )
    parser.add_argument("--expected-os", choices=("linux", "macos", "windows"))
    parser.add_argument("--expected-arch", choices=("x86_64", "aarch64"))
    parser.add_argument("--expected-version")
    parser.add_argument("wheels", nargs="+", type=Path)
    args = parser.parse_args()
    for wheel in args.wheels:
        try:
            verify_rust_wheel(
                wheel,
                manylinux_policy=args.manylinux_policy,
                expected_os=args.expected_os,
                expected_arch=args.expected_arch,
                expected_version=args.expected_version,
            )
        except (OSError, zipfile.BadZipFile, WheelContractError) as error:
            raise SystemExit(f"invalid Rust wheel {wheel}: {error}") from error
        print(f"verified Rust-only wheel: {wheel}")


if __name__ == "__main__":
    main()
