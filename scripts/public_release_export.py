#!/usr/bin/env python3
"""Create a classified, reproducible public RWKV-SRS source export."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any


DEFAULT_POLICY = "release/public_export_policy.json"
PROVENANCE_NAME = "PUBLIC_EXPORT_PROVENANCE.json"
ALLOWED_FILE_MODES = {"100644", "100755"}


class ExportError(ValueError):
    """Raised when a public export violates its policy."""


def _git(root: Path, *arguments: str, text: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=text,
    )
    return result.stdout


def _repository_root(start: Path) -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=start,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def _require_clean(root: Path) -> None:
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ExportError("public exports require a clean source tree")


def _commit(root: Path, treeish: str) -> str:
    return str(_git(root, "rev-parse", "--verify", f"{treeish}^{{commit}}")).strip()


def _load_policy(root: Path, commit: str, path: str) -> tuple[dict[str, Any], bytes]:
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts:
        raise ExportError(f"unsafe policy path: {path!r}")
    raw = _git(root, "show", f"{commit}:{pure.as_posix()}", text=False)
    assert isinstance(raw, bytes)
    try:
        policy = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExportError(f"invalid export policy: {error}") from error
    if not isinstance(policy, dict):
        raise ExportError("export policy must be a JSON object")
    return policy, raw


def _tree_entries(root: Path, commit: str) -> dict[str, str]:
    raw = _git(root, "ls-tree", "-r", "-z", "--full-tree", commit, text=False)
    assert isinstance(raw, bytes)
    entries: dict[str, str] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, kind, _object = metadata.decode("ascii").split(" ", 2)
        path = raw_path.decode("utf-8")
        if kind != "blob" or mode not in ALLOWED_FILE_MODES:
            raise ExportError(f"unsupported tracked entry {mode} {kind} {path}")
        entries[path] = mode
    return entries


def _path_set_digest(paths: list[str]) -> str:
    value = "".join(f"{path}\n" for path in sorted(paths)).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _matches(path: str, rule: dict[str, Any]) -> bool:
    exact = rule.get("paths", [])
    prefixes = rule.get("prefixes", [])
    if not isinstance(exact, list) or not all(
        isinstance(value, str) for value in exact
    ):
        raise ExportError(f"rule {rule.get('id')!r} has invalid paths")
    if not isinstance(prefixes, list) or not all(
        isinstance(value, str) and value.endswith("/") for value in prefixes
    ):
        raise ExportError(f"rule {rule.get('id')!r} has invalid prefixes")
    return path in exact or any(path.startswith(prefix) for prefix in prefixes)


def _classify(
    policy: dict[str, Any], paths: list[str]
) -> tuple[list[str], dict[str, str]]:
    rules = policy.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ExportError("export policy must contain non-empty rules")
    seen_ids: set[str] = set()
    for rule in rules:
        if not isinstance(rule, dict):
            raise ExportError("each export rule must be an object")
        rule_id = rule.get("id")
        action = rule.get("action")
        if not isinstance(rule_id, str) or not rule_id or rule_id in seen_ids:
            raise ExportError(f"invalid or duplicate rule id: {rule_id!r}")
        if action not in {"include", "exclude"}:
            raise ExportError(f"rule {rule_id!r} has invalid action: {action!r}")
        seen_ids.add(rule_id)

    included: list[str] = []
    classifications: dict[str, str] = {}
    for path in paths:
        matches = [rule for rule in rules if _matches(path, rule)]
        if len(matches) != 1:
            match_ids = [rule.get("id") for rule in matches]
            raise ExportError(
                f"tracked path must match exactly one rule: {path!r} matched {match_ids}"
            )
        rule = matches[0]
        classifications[path] = str(rule["id"])
        if rule["action"] == "include":
            included.append(path)
    if not included:
        raise ExportError("export policy selected no files")
    return included, classifications


def _validate_policy(
    policy: dict[str, Any],
    entries: dict[str, str],
) -> tuple[list[str], dict[str, str]]:
    if policy.get("schema_version") != 1:
        raise ExportError("unsupported export-policy schema version")
    expected = policy.get("tracked_paths_sha256")
    actual = _path_set_digest(list(entries))
    if expected != actual:
        raise ExportError(
            "tracked path set does not match policy: "
            f"expected {expected!r}, calculated {actual}"
        )
    return _classify(policy, sorted(entries))


def _safe_destination(root: Path, destination: Path) -> Path:
    destination = destination.resolve()
    if destination.exists():
        raise ExportError(f"export destination already exists: {destination}")
    if destination == root or destination.is_relative_to(root):
        raise ExportError("export destination must be outside the source repository")
    if root.is_relative_to(destination):
        raise ExportError("export destination must not contain the source repository")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _archive(
    root: Path,
    commit: str,
    included: list[str],
    destination: Path,
    modes: dict[str, str],
) -> None:
    with tempfile.NamedTemporaryFile(
        prefix="rwkv-srs-public-export-",
        suffix=".tar",
        dir=destination.parent,
        delete=False,
    ) as temporary:
        archive_path = Path(temporary.name)
    try:
        subprocess.run(
            [
                "git",
                "archive",
                "--format=tar",
                f"--output={archive_path}",
                commit,
                "--",
                *included,
            ],
            cwd=root,
            check=True,
        )
        expected = set(included)
        extracted: set[str] = set()
        with tarfile.open(archive_path, "r:") as archive:
            for member in archive.getmembers():
                pure = PurePosixPath(member.name)
                if pure.is_absolute() or ".." in pure.parts:
                    raise ExportError(f"unsafe archive path: {member.name!r}")
                if member.isdir():
                    continue
                if not member.isfile() or member.name not in expected:
                    raise ExportError(f"unexpected archive member: {member.name!r}")
                target = destination / pure.as_posix()
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ExportError(f"could not read archive member: {member.name!r}")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(int(modes[member.name][-3:], 8))
                extracted.add(member.name)
        if extracted != expected:
            missing = sorted(expected - extracted)
            raise ExportError(f"archive omitted allowed files: {missing}")
    finally:
        archive_path.unlink(missing_ok=True)


def _file_record(root: Path, relative: str, mode: str) -> dict[str, Any]:
    path = root / relative
    content = path.read_bytes()
    return {
        "path": relative,
        "mode": mode,
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _write_provenance(
    destination: Path,
    *,
    commit: str,
    policy: dict[str, Any],
    policy_raw: bytes,
    included: list[str],
    modes: dict[str, str],
    classifications: dict[str, str],
) -> None:
    files = [_file_record(destination, path, modes[path]) for path in sorted(included)]
    file_set = "".join(
        f"{record['path']}\0{record['mode']}\0{record['size']}\0{record['sha256']}\n"
        for record in files
    ).encode("utf-8")
    manifest = {
        "schema_version": 1,
        "source_repository": policy.get("source_repository"),
        "source_commit": commit,
        "export_policy": {
            "id": policy.get("policy_id"),
            "version": policy.get("policy_version"),
            "sha256": hashlib.sha256(policy_raw).hexdigest(),
        },
        "source_file_count": len(files),
        "source_files_sha256": hashlib.sha256(file_set).hexdigest(),
        "files": files,
        "excluded_classification_counts": {
            rule_id: sum(1 for value in classifications.values() if value == rule_id)
            for rule_id in sorted(set(classifications.values()))
            if all(
                not (rule.get("id") == rule_id and rule.get("action") == "include")
                for rule in policy["rules"]
            )
        },
        "manifest_note": (
            "File records cover every source-derived export file. "
            f"This generated {PROVENANCE_NAME} cannot self-hash."
        ),
    }
    (destination / PROVENANCE_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def export_public_tree(
    root: Path,
    *,
    treeish: str,
    policy_path: str,
    destination: Path,
    check_only: bool,
) -> tuple[str, int]:
    _require_clean(root)
    commit = _commit(root, treeish)
    policy, policy_raw = _load_policy(root, commit, policy_path)
    entries = _tree_entries(root, commit)
    included, classifications = _validate_policy(policy, entries)
    if check_only:
        return commit, len(included)

    destination = _safe_destination(root, destination)
    staging = Path(
        tempfile.mkdtemp(prefix=".rwkv-srs-public-export-", dir=destination.parent)
    )
    try:
        _archive(root, commit, included, staging, entries)
        _write_provenance(
            staging,
            commit=commit,
            policy=policy,
            policy_raw=policy_raw,
            included=included,
            modes=entries,
            classifications=classifications,
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return commit, len(included)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--treeish", default="HEAD")
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.check_only and args.out_dir is None:
        parser.error("--out-dir is required unless --check-only is used")

    root = _repository_root(Path.cwd())
    try:
        commit, count = export_public_tree(
            root,
            treeish=args.treeish,
            policy_path=args.policy,
            destination=args.out_dir or Path("unused"),
            check_only=args.check_only,
        )
    except (ExportError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"public export failed: {error}") from error
    if args.check_only:
        print(f"public export policy accepts {count} files at {commit}")
    else:
        print(
            f"exported {count} source files from {commit} to {args.out_dir.resolve()}"
        )


if __name__ == "__main__":
    main()
