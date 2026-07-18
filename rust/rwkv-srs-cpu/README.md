# rwkv-srs-cpu Rust Backend

This crate is the Rust/native CPU backend for RWKV-P.

Most users should access it through the Python wrapper in `src/rwkv_srs`. The
crate is kept in this monorepo so the Python API, Torch reference backend, Rust
backend, and parity tests can evolve together.

The future Anki addon release process will build this backend for each target
platform and bundle the resulting native extension into platform-specific
`.ankiaddon` assets.

## Commands

```bash
cd rust/rwkv-srs-cpu && cargo test
cd rust/rwkv-srs-cpu && cargo build --profile release-development
```

From the repository root, build the Python extension:

```bash
scripts/build_rust_release_extension.sh
```

That script defaults to the `release-development` profile for iteration. To
build an optimized local extension for the current CPU:

```bash
scripts/build_rust_local_extension.sh
```

Build the distributable PyO3 `abi3` wheel:

```bash
scripts/build_rust_ci_wheel.sh
```

The wheel should be tagged `cp39-abi3`. The chosen native limited API minimum
is `abi3-py39`; see `docs/RUST_ABI3_BUILD.md` and
`docs/RUST_BUILD_PROFILES.md` for details.

The generated `target/` directory and copied native extension are build outputs
and should not be committed.
