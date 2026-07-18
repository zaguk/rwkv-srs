# Public Rust package release checks

The generated public repository contains the Rust/PyO3 runtime, the thin
Python facade, a compact test suite, and two redistribution-cleared Safetensors
models. It deliberately omits the internal Torch oracle, private/reference
corpora, `.pth` source checkpoints, experimental benchmarks, and PGO training
machinery. It is a generated distribution mirror of the canonical private
repository and does not operate a public contribution workflow.

Package metadata intentionally contains no project-level license declaration.
Required third-party notices remain separate from that choice.
`THIRD_PARTY_NOTICES.md` inventories the complete locked cross-platform Cargo
graph, while `THIRD_PARTY_LICENSES.txt` retains the exact unique license and
notice texts shipped in dependency source archives. Both files are embedded in
wheel metadata and included in sdists.

The generic wheel and source distribution are model-free. Applications pass an
explicit Safetensors path. Downstream products may deliberately copy one of the
repository models from `tests/fixtures/models/` into
`rwkv_srs/pretrained/`; the artifact validator exercises this overlay and
named-model lookup separately from the generic wheel contract.

The native crate declares Rust 1.87 as its MSRV. Public CI checks that version
explicitly while ordinary release jobs use the exact compiler pinned in
`rust-toolchain.toml`.

## Local artifact validation

Install the build tools and run the artifact validator:

```bash
python -m pip install "maturin[zig]>=1.9,<2"
python scripts/validate_public_release.py
```

On Linux this builds and validates an `abi3` wheel tagged exactly
`manylinux_2_38`. The validator then:

1. verifies that generated third-party notices match the locked Cargo graph;
2. generates schema-conformant synthetic models if the private Cargo fixtures
   are absent and runs the complete native crate suite;
3. creates a fresh environment outside the checkout;
4. installs the wheel without source-tree imports;
5. runs `tests/release/` against the installed wheel;
6. builds a source distribution;
7. verifies that its release tests, notices, and notice generator are present
   while Torch-only source and model weights are absent;
8. installs that source distribution into a second fresh environment;
9. runs the release suite from the extracted source artifact; and
10. overlays one repository model into an installed generic wheel environment
   and verifies named lookup plus native inference.

Use `--artifact-kind wheel` on a non-Linux target when only the native wheel
lane is needed. No test requires GPU hardware: GPU recovery is verified through
an intentionally nonexistent adapter and the typed public preflight error.

To retain the exact verified wheel and its release metadata, provide empty,
separate work and output directories:

```bash
python scripts/validate_public_release.py \
  --artifact-kind wheel \
  --expected-target linux-x86_64 \
  --work-dir /tmp/rwkv-srs-release-work \
  --output-dir /tmp/rwkv-srs-release-assets
```

Only the installed-and-tested wheel is copied to `--output-dir`. Its adjacent
`.sha256` and `.provenance.json` files bind the checksum, byte size, package and
wheel tags, source commit, target, pinned build tools, portable Cargo profile,
and tested Python versions. A validation-only sdist remains under
`--work-dir`; it is never copied into the publishable output directory.

## Target-native release CI

Public CI builds one portable `cp39-abi3` wheel on each target and installs the
same wheel under both the minimum supported Python (3.11) and newest supported
Python (3.14):

| Target ID | Hosted runner | Required native behavior |
|---|---|---|
| `linux-x86_64` | `ubuntu-24.04` | exact `manylinux_2_38`, default SIMD, forced scalar |
| `linux-aarch64` | `ubuntu-24.04-arm` | exact `manylinux_2_38`, Pulp/NEON, forced scalar |
| `macos-x86_64` | `macos-15-intel` | x86 default SIMD, forced scalar |
| `macos-aarch64` | `macos-15` | Pulp/NEON, forced scalar |
| `windows-x86_64` | `windows-2022` | x86 default SIMD, forced scalar |
| `windows-aarch64` | `windows-11-arm` | Pulp/NEON, forced scalar |

GitHub currently labels `windows-11-arm` as public preview. It remains a
blocking release lane: no Windows ARM64 wheel is published unless that native
runner builds, installs, and executes it successfully.

Literal 32-bit x86/i686 is not supported. Hosted CI proves CPU execution and
typed GPU-unavailable recovery; it does not qualify Vulkan, Metal, or DX12
computation without separate target-native GPU hardware.

Fast SIMD reductions may differ in low floating-point bits across AVX2/FMA,
Pulp/NEON, and scalar execution. The frozen public suite currently accepts an
absolute probability tolerance of `5e-5` while requiring exact restoration for
checkpoint/undo state where the API promises it. The package and native
checkpoint format are pre-1.0: callers should pin an exact RWKV-SRS version,
retain checkpoint backups during upgrades, and use only Rust-native `.bin`
runtime checkpoints with this backend. Model inputs are Rust-readable
Safetensors files; Torch `.pth`/`.pt` checkpoints are not supported by the
distributed backend.

Every successful matrix lane uploads only its exact tested wheel and sidecars.
For a tag such as `v0.1.0`, the release job requires the Python package, Rust
crate, wheel, and tag versions all to equal `0.1.0`; downloads all six job
artifacts; rechecks their ABI, platform, architecture, hashes, sizes, targets,
and common source commit; and writes aggregate `SHA256SUMS` and
`RELEASE_PROVENANCE.json` files. It then creates a GitHub Release whose uploaded
project assets contain only wheels and manifests. Standalone `.pyd`, `.so`,
`.dll`, or `.dylib` files and the validation-only sdist are rejected from the
release directory. GitHub may additionally display its automatic source-code
archives; those are platform-generated links, not uploaded package artifacts.

## Manual local builds

The cross-platform local builder exposes the two supported optimization modes
without requiring callers to assemble Maturin or Rust flags themselves:

```bash
python scripts/build_local_artifact.py release --artifact both
python scripts/build_local_artifact.py native --artifact both
```

`release` uses the portable `release-ci` profile and enforces
`manylinux_2_38` through Zig on Linux. `native` uses `release-local` plus
`target-cpu=native`, tags Linux wheels as host-local `linux` wheels, and must
not be redistributed to incompatible CPUs. `--artifact` accepts `wheel`,
`extension`, or `both`; outputs and `build-info.json` are written under
`dist/local/<build>/` unless `--out-dir` is supplied.

## Reproducible public export

Public exports are positive allowlists over an exact clean commit:

```bash
python scripts/public_release_export.py --check-only
python scripts/public_release_export.py --out-dir ../rwkv-srs-public
```

Every tracked path must match exactly one include or exclusion rule in
`release/public_export_policy.json`. The policy also freezes the hash of the
complete tracked path set, so a newly added file fails closed even if a broad
classification might otherwise match it.

The exporter uses `git archive`, preserves executable modes, rejects links and
unsafe destinations, and writes `PUBLIC_EXPORT_PROVENANCE.json` with the source
commit and SHA-256, size, and mode of every source-derived exported file.

Run `scripts/validate_public_release.py` from the generated tree before
publishing or mirroring it. The exporter does not scan for secrets, publish
artifacts, or initialize a remote repository; those remain explicit
release-owner steps. Initial binary distribution uses GitHub Releases only and
ordinary portable `release-ci` builds; public PGO artifacts remain deferred
until a training corpus is available.
