# Public release contract tests

This directory is the self-contained acceptance suite for an installed
Rust-only RWKV-SRS artifact. It deliberately depends only on Python's standard
library, `pytest`, and the installed `rwkv_srs` package. It does not import
Torch, pandas, PyArrow, internal test helpers, trained weights, or private
datasets.

`model_fixture.py` generates a small deterministic Safetensors model from the
native tensor schema. The values are name-derived repeating patterns; they are
not learned parameters and do not reproduce any external model. The generated
model hash and numerical results are frozen so that schema or execution
changes require an explicit fixture update.

Run the suite from a source checkout with an installed native extension:

```bash
RWKV_SRS_BACKEND=rust python -m pytest -q tests/release
```

`scripts/validate_public_release.py` builds artifacts, creates environments
outside the checkout, installs the wheel and rebuilt source distribution, and
runs this same suite against each installed package.

Release CI passes both Python 3.11 and 3.14 to the validator. It builds one
`cp39-abi3` wheel per OS/architecture and installs those exact same wheel bytes
under both interpreters; it does not rebuild once per Python version.

The normal artifact suite skips `test_model_distribution.py` and proves that
generic artifacts are model-free. The validator then deliberately copies one
redistribution-cleared repository model into the installed package's
`rwkv_srs/pretrained/` directory and reruns that test. This separately proves
the downstream overlay layout, model hash, named lookup, native loading, and a
prediction without weakening the generic wheel-member contract.

Installed-artifact tests also require `THIRD_PARTY_NOTICES.md` and
`THIRD_PARTY_LICENSES.txt` under standardized distribution metadata and reject
any project-level `License` or `License-Expression`. The source distribution
retains those files, their deterministic generator, and the pinned Rust
toolchain declaration.

The default-SIMD values use a small absolute tolerance because AVX2/FMA, Pulp,
NEON, and scalar reductions need not be bit-identical. Checkpoint and undo
state restoration checks remain exact where the public contract requires it.
