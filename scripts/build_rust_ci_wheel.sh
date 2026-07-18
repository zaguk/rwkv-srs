#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
OUT_DIR="${OUT_DIR:-$ROOT/dist/wheels}"
MANYLINUX_POLICY="manylinux_2_38"

if [ ! -x "$PYTHON" ]; then
  echo "missing project Python: $PYTHON" >&2
  echo "Create or repair .venv before building the Rust wheel." >&2
  exit 1
fi

MATURIN_ARGS=(--locked)
if [ "$(uname -s)" = "Linux" ]; then
  for arg in "$@"; do
    case "$arg" in
      --compatibility|--compatibility=*|--zig|--auditwheel|--auditwheel=*)
        echo "Linux release builds own --compatibility, --zig, and --auditwheel." >&2
        echo "The enforced policy is $MANYLINUX_POLICY through Zig." >&2
        exit 2
        ;;
    esac
  done

  if [ -z "${CARGO_ZIGBUILD_ZIG_PATH:-}" ] && ! command -v zig >/dev/null 2>&1; then
    ZIG_PATH="$($PYTHON -c '
from pathlib import Path
import ziglang

print(Path(ziglang.__file__).with_name("zig"))
' 2>/dev/null || true)"
    if [ ! -x "$ZIG_PATH" ]; then
      echo "missing Zig compiler for the $MANYLINUX_POLICY release build" >&2
      echo "Install the Rust build extra: $PYTHON -m pip install \".[rust-build]\"" >&2
      exit 1
    fi
    export CARGO_ZIGBUILD_ZIG_PATH="$ZIG_PATH"
  fi

  export RWKV_SRS_EXPECTED_MANYLINUX_POLICY="$MANYLINUX_POLICY"
  MATURIN_ARGS+=(
    --zig
    --compatibility "$MANYLINUX_POLICY"
    --auditwheel check
  )
fi
MATURIN_ARGS+=("$@")

if [ "${RWKV_SRS_PGO:-0}" = "1" ]; then
  exec "$PYTHON" "$ROOT/scripts/build_rust_pgo_wheel.py" \
    --out-dir "$OUT_DIR" -- "${MATURIN_ARGS[@]}"
fi

STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/rwkv-srs-wheel.XXXXXX")"
cleanup() {
  rm -rf "$STAGING_DIR"
}
trap cleanup EXIT

"$PYTHON" -m maturin build \
  --profile release-ci \
  --out "$STAGING_DIR" \
  "${MATURIN_ARGS[@]}"

shopt -s nullglob
wheels=("$STAGING_DIR"/*.whl)
if [ "${#wheels[@]}" -eq 0 ]; then
  echo "maturin did not produce a wheel in $STAGING_DIR" >&2
  exit 1
fi

contract_args=()
if [ "$(uname -s)" = "Linux" ]; then
  case "$(uname -m)" in
    x86_64|amd64)
      release_arch="x86_64"
      ;;
    aarch64|arm64)
      release_arch="aarch64"
      ;;
    *)
      echo "unsupported Linux release architecture: $(uname -m)" >&2
      exit 1
      ;;
  esac
  package_version="$($PYTHON -c '
import pathlib, tomllib

print(tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"]["version"])
')"
  contract_args+=(
    --manylinux-policy "$MANYLINUX_POLICY"
    --expected-os linux
    --expected-arch "$release_arch"
    --expected-version "$package_version"
  )
fi
"$PYTHON" "$ROOT/scripts/rust_wheel_contract.py" \
  "${contract_args[@]}" "${wheels[@]}"
mkdir -p "$OUT_DIR"
for wheel in "${wheels[@]}"; do
  cp "$wheel" "$OUT_DIR/$(basename "$wheel")"
  echo "$OUT_DIR/$(basename "$wheel")"
done
