# RWKV-SRS

RWKV-SRS is a Rust-backed Python runtime for recurrent spaced-repetition
inference. Its native backend supports scalar and batched review processing,
CPU Oracle/Fast prediction, optional wgpu prediction and processing, atomic
checkpoints, LIFO undo, and Rust-owned live prediction sessions.

The public Python package is a thin facade over the Rust/PyO3 implementation.
It has no Torch dependency and lazily loads its native extension.

## Attribution and AI authorship

RWKV-SRS originated from the
[RWKV implementation](https://github.com/open-spaced-repetition/srs-benchmark/tree/main/rwkv)
in the Open Spaced Repetition `srs-benchmark` project, authored by
[1DWalker](https://github.com/1DWalker). We gratefully acknowledge that work as
the foundation of this project.

**AI-authorship disclaimer:** This entire RWKV-SRS project—including its
adaptation, Rust implementation, Python bindings, tests, tooling,
documentation, and subsequent development—was coded by OpenAI's Codex under
human direction. This statement does not supersede the attribution above for
the original RWKV source on which the project was based.

Third-party attributions and exact locked dependency license texts are recorded
in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt).

> The API and checkpoint format are still pre-1.0. Pin an exact package version
> and retain checkpoint backups while integrating it.

## Models

The generic wheel is intentionally model-free. Construct a new runtime with an
explicit path to a compatible Safetensors model:

```python
from rwkv_srs import RWKV_SRS

runtime = RWKV_SRS(model="/path/to/model.safetensors")
```

The repository retains two redistribution-cleared Safetensors models under
`tests/fixtures/models/`, with hashes and origin recorded beside them. They are
excluded from generic wheels and source distributions. Downstream applications
may deliberately copy a selected model under `rwkv_srs/pretrained/`; named
lookup then works without requiring their users to locate a model manually.

Omitting `seed` selects the native deterministic seed `5489`.

## Build and test

RWKV-SRS requires Python 3.11 or newer and a Rust toolchain. For development:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
RWKV_SRS_BACKEND=rust python -m pytest -q tests/release
```

Raw Cargo tests do not require linking a Python extension. The public
repository already contains the two models used by the native tests:

```bash
cargo test --manifest-path rust/rwkv-srs-cpu/Cargo.toml
```

The declared Rust MSRV is 1.87. Release and development tooling is pinned by
`rust-toolchain.toml`; public CI checks both the MSRV and pinned toolchain.

The independent Python release suite still generates its own small,
schema-conformant synthetic model and does not derive expected outputs from the
repository-trained weights.

Build and exercise an installed wheel plus a rebuilt source distribution in
fresh environments outside the checkout:

```bash
python scripts/validate_public_release.py
```

### Manual wheel and native-extension builds

Use any Python 3.11-or-newer installation with the Rust build tools installed,
then select either a portable `release` build or a host-tuned `native` build.
The build always runs directly on the current machine. A virtual environment
is optional and only isolates Python build-tool dependencies; it does not
change CPU detection or native code generation.

On Linux or macOS:

```bash
python3 -m pip install ".[rust-build]"

# Portable release-ci output for distribution on this OS/architecture.
python3 scripts/build_local_artifact.py release --artifact both

# Host-tuned release-local output for this machine only.
python3 scripts/build_local_artifact.py native --artifact both
```

On Windows PowerShell:

```powershell
py -3.11 -m pip install ".[rust-build]"

# Portable release-ci wheel and extracted .pyd.
py -3.11 .\scripts\build_local_artifact.py `
  release --artifact both

# Host-tuned release-local wheel and extracted .pyd.
py -3.11 .\scripts\build_local_artifact.py `
  native --artifact both
```

`--artifact` accepts `wheel`, `extension`, or `both`:

- `wheel` retains only the complete installable wheel.
- `extension` retains only `_native.pyd` on Windows or `_native*.so` on
  Linux/macOS.
- `both`, the default, retains both outputs.

Outputs are written to `dist/local/release/` or `dist/local/native/` by
default. Pass `--out-dir <path>` to choose another directory. Every build also
writes `build-info.json` containing the selected profile and output hashes.

Native builds use `target-cpu=native` and must not be redistributed to CPUs
that may lack the same instruction-set features. An extracted native extension
still requires the Python files in the `rwkv_srs` package; retain the wheel
when a complete installable artifact is needed.

Linux release wheels are built with Zig and are required to carry the exact
`manylinux_2_38` compatibility tag. The release test suite also runs the Fast
path with SIMD forced off and verifies typed GPU-unavailable recovery without
requiring GPU hardware.

## Public source export

The public tree is generated from an exact clean commit using a classified
positive allowlist:

```bash
python scripts/public_release_export.py --check-only
python scripts/public_release_export.py --out-dir ../rwkv-srs-public
```

Every tracked file must be classified. The export records its source commit and
the mode, size, and SHA-256 of every copied source file in
`PUBLIC_EXPORT_PROVENANCE.json`.

See [docs/PUBLIC_RELEASE.md](docs/PUBLIC_RELEASE.md) for the artifact and export
contract, and [rust/rwkv-srs-cpu/README.md](rust/rwkv-srs-cpu/README.md) for the
native crate.

## Internal development checkout

The private development checkout may contain a frozen Torch oracle, extended
corpora, benchmarks, PGO tooling, and experiment records. Those are used for
differential qualification and performance work; they are not part of the
generated public repository or Rust-only artifacts. The public repository is a
generated distribution mirror; the private repository remains canonical.
