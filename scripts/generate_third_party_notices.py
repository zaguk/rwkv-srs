#!/usr/bin/env python3
"""Generate the locked Rust dependency inventory and bundled license archive."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import tarfile
import tomllib
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "rust" / "rwkv-srs-cpu" / "Cargo.toml"
LOCKFILE = ROOT / "rust" / "rwkv-srs-cpu" / "Cargo.lock"
NOTICES = ROOT / "THIRD_PARTY_NOTICES.md"
LICENSE_ARCHIVE = ROOT / "THIRD_PARTY_LICENSES.txt"
LICENSE_PREFIXES = ("LICENSE", "LICENCE", "COPYING", "NOTICE")


class NoticeError(ValueError):
    """Raised when the locked dependency graph cannot be inventoried safely."""


def _fetch_locked_archives() -> None:
    result = subprocess.run(
        [
            "cargo",
            "fetch",
            "--locked",
            "--manifest-path",
            str(MANIFEST),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise NoticeError(f"cargo fetch failed: {detail}")


def _canonical_lockfile_bytes() -> bytes:
    """Return Cargo.lock bytes with checkout-dependent line endings removed."""

    return LOCKFILE.read_bytes().replace(b"\r\n", b"\n")


def _locked_registry_packages() -> list[dict[str, str]]:
    lock_data = tomllib.loads(LOCKFILE.read_text(encoding="utf-8"))
    manifest_data = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    workspace = manifest_data.get("package", {})
    workspace_identity = (str(workspace.get("name")), str(workspace.get("version")))
    packages: list[dict[str, str]] = []
    for value in lock_data.get("package", ()):
        name = str(value.get("name"))
        version = str(value.get("version"))
        source = value.get("source")
        if source is None:
            if (name, version) != workspace_identity:
                raise NoticeError(
                    "unsupported non-registry locked package: " f"{name} {version}"
                )
            continue
        if not str(source).startswith("registry+"):
            raise NoticeError(
                f"unsupported non-registry package source for {name} {version}: "
                f"{source}"
            )
        checksum = value.get("checksum")
        if not checksum:
            raise NoticeError(f"locked registry package lacks checksum: {name} {version}")
        packages.append(
            {
                "name": name,
                "version": version,
                "source": str(source),
                "checksum": str(checksum),
            }
        )
    packages.sort(
        key=lambda package: (
            package["name"],
            package["version"],
            package["source"],
        )
    )
    return packages


def _cargo_home() -> Path:
    configured = os.environ.get("CARGO_HOME")
    return Path(configured).expanduser() if configured else Path.home() / ".cargo"


def _crate_archive(package: dict[str, str]) -> Path:
    filename = f"{package['name']}-{package['version']}.crate"
    cache_root = _cargo_home() / "registry" / "cache"
    candidates = sorted(cache_root.glob(f"*/{filename}"))
    mismatched: list[str] = []
    for candidate in candidates:
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if digest == package["checksum"]:
            return candidate
        mismatched.append(f"{candidate} ({digest})")
    package_label = f"{package['name']} {package['version']}"
    if mismatched:
        raise NoticeError(
            f"cached crate checksum mismatch for {package_label}; expected "
            f"{package['checksum']}, found " + ", ".join(mismatched)
        )
    raise NoticeError(
        f"missing checksum-verified crate archive for {package_label} under "
        f"{cache_root}"
    )


def _archive_file(archive: tarfile.TarFile, member_name: str) -> bytes:
    try:
        member = archive.getmember(member_name)
    except KeyError as error:
        raise NoticeError(f"crate archive lacks {member_name}") from error
    extracted = archive.extractfile(member)
    if not member.isfile() or extracted is None:
        raise NoticeError(f"crate archive member is not a regular file: {member_name}")
    return extracted.read()


def _package_archive_contents(
    locked: dict[str, str],
) -> tuple[dict[str, Any], list[tuple[str, bytes]]]:
    archive_path = _crate_archive(locked)
    root = f"{locked['name']}-{locked['version']}"
    with tarfile.open(archive_path, "r:*") as archive:
        cargo_toml = tomllib.loads(
            _archive_file(archive, f"{root}/Cargo.toml").decode("utf-8")
        )
        package = cargo_toml.get("package")
        if not isinstance(package, dict):
            raise NoticeError(f"{archive_path} has no [package] table")
        if (
            str(package.get("name")) != locked["name"]
            or str(package.get("version")) != locked["version"]
        ):
            raise NoticeError(
                f"crate manifest identity differs from Cargo.lock: {archive_path}"
            )

        license_files: list[tuple[str, bytes]] = []
        prefix = f"{root}/"
        seen_names: set[str] = set()
        for member in archive.getmembers():
            if not member.isfile() or not member.name.startswith(prefix):
                continue
            relative = member.name[len(prefix) :]
            if "/" in relative or not relative.upper().startswith(LICENSE_PREFIXES):
                continue
            if relative in seen_names:
                raise NoticeError(
                    f"duplicate root license member in {archive_path}: {relative}"
                )
            seen_names.add(relative)
            license_files.append((relative, _archive_file(archive, member.name)))

    normalized = {
        "name": locked["name"],
        "version": locked["version"],
        "license": package.get("license"),
        "authors": package.get("authors") or (),
        "repository": package.get("repository"),
        "homepage": package.get("homepage"),
        "source": locked["source"],
    }
    if not normalized["license"]:
        raise NoticeError(
            "dependency without declared license metadata: "
            f"{locked['name']} {locked['version']}"
        )
    license_files.sort(key=lambda item: item[0])
    return normalized, license_files


def _escape_table(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _source_label(package: dict[str, Any]) -> str:
    return str(
        package.get("repository")
        or package.get("homepage")
        or package.get("source")
        or "unknown"
    )


def _notice_document(
    packages: list[dict[str, Any]],
    license_records: dict[str, tuple[bytes, list[str]]],
    missing_license_files: list[str],
) -> bytes:
    rows = []
    for package in packages:
        authors = ", ".join(package.get("authors") or ()) or "not declared"
        rows.append(
            "| "
            + " | ".join(
                _escape_table(value)
                for value in (
                    package["name"],
                    package["version"],
                    package["license"],
                    authors,
                    _source_label(package),
                )
            )
            + " |"
        )

    missing_rows = "\n".join(f"- `{value}`" for value in missing_license_files)
    if not missing_rows:
        missing_rows = "- None"

    document = f"""# Third-party notices

RWKV-SRS is source-visible, all-rights-reserved software. This document records
third-party provenance and license terms; it does not grant a license to
RWKV-SRS itself.

## RWKV and model provenance

- The original spaced-repetition adaptation was published in
  `open-spaced-repetition/srs-benchmark`. The release owner has privately
  retained permission to publish, modify, and redistribute that implementation,
  this Rust derivative, and the converted model weights.
- The benchmark implementation cites `BlinkDL/RWKV-LM` and
  `SmerkyG/RWKV_Explained`, both distributed under Apache-2.0. Apache license
  texts from the locked dependency sources are retained in
  `THIRD_PARTY_LICENSES.txt`.
- Model origins, source hashes, and converted-file hashes are recorded in
  `tests/fixtures/models/PROVENANCE.md`.

## Locked Rust dependency inventory

This inventory was generated from `rust/rwkv-srs-cpu/Cargo.lock`. It covers the
complete locked graph, including target-specific dependencies for supported
Linux, macOS, and Windows builds; an individual wheel can contain a subset.

- Cargo.lock SHA-256: `{hashlib.sha256(_canonical_lockfile_bytes()).hexdigest()}`
- Third-party packages: {len(packages)}
- Unique bundled license/notice texts: {len(license_records)}

| Package | Version | Declared license | Declared authors | Source/repository |
|---|---:|---|---|---|
{chr(10).join(rows)}

## Published crates without a root license file

The following crate archives declare an SPDX license in Cargo metadata but do
not contain a root-level `LICENSE`, `LICENCE`, `COPYING`, or `NOTICE` file. Their
declared expressions, authors, and repositories remain recorded in the table
above. Standard texts supplied by other locked packages are retained in the
license archive where applicable.

{missing_rows}

## License text archive

`THIRD_PARTY_LICENSES.txt` contains every unique root-level license and notice
file shipped in the locked crates.io source archives. Exact duplicate texts are
stored once and mapped back to every package archive member that supplied them.
"""
    return document.encode("utf-8")


def _license_archive(
    license_records: dict[str, tuple[bytes, list[str]]],
) -> bytes:
    chunks = [
        b"RWKV-SRS THIRD-PARTY LICENSE TEXT ARCHIVE\n",
        b"\nThis archive does not license RWKV-SRS itself.\n",
    ]
    for digest, (content, members) in sorted(license_records.items()):
        chunks.extend(
            (
                b"\n" + b"=" * 79 + b"\n",
                f"SHA-256: {digest}\n".encode(),
                b"Source archive members:\n",
                "".join(f"- {member}\n" for member in sorted(members)).encode(),
                b"=" * 79 + b"\n\n",
                content,
                b"" if content.endswith(b"\n") else b"\n",
            )
        )
    return b"".join(chunks)


def generate() -> tuple[bytes, bytes]:
    _fetch_locked_archives()
    packages: list[dict[str, Any]] = []
    license_records: dict[str, tuple[bytes, list[str]]] = {}
    missing_license_files: list[str] = []
    for locked in _locked_registry_packages():
        package, files = _package_archive_contents(locked)
        packages.append(package)
        package_label = f"{package['name']} {package['version']}"
        if not files:
            missing_license_files.append(package_label)
            continue
        for filename, content in files:
            digest = hashlib.sha256(content).hexdigest()
            member = f"{package_label}/{filename}"
            if digest in license_records:
                existing_content, members = license_records[digest]
                if existing_content != content:
                    raise NoticeError(f"SHA-256 collision for {member}")
                members.append(member)
            else:
                license_records[digest] = (content, [member])

    notices = _notice_document(packages, license_records, missing_license_files)
    licenses = _license_archive(license_records)
    return notices, licenses


def _check(path: Path, expected: bytes) -> bool:
    if not path.is_file():
        print(f"missing generated notice file: {path.relative_to(ROOT)}")
        return False
    actual = path.read_bytes()
    if actual == expected:
        return True
    print(
        f"stale generated notice file: {path.relative_to(ROOT)} "
        f"(expected SHA-256 {hashlib.sha256(expected).hexdigest()}, "
        f"found {hashlib.sha256(actual).hexdigest()})"
    )
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when committed notice outputs differ from the locked graph.",
    )
    args = parser.parse_args()
    try:
        notices, licenses = generate()
    except (
        NoticeError,
        OSError,
        tarfile.TarError,
        tomllib.TOMLDecodeError,
        UnicodeDecodeError,
    ) as error:
        raise SystemExit(f"could not generate third-party notices: {error}") from error

    if args.check:
        if not (_check(NOTICES, notices) and _check(LICENSE_ARCHIVE, licenses)):
            raise SystemExit(1)
        print("third-party notices match the locked dependency graph")
        return

    NOTICES.write_bytes(notices)
    LICENSE_ARCHIVE.write_bytes(licenses)
    print(f"wrote {NOTICES.relative_to(ROOT)}")
    print(f"wrote {LICENSE_ARCHIVE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
