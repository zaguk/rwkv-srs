# RWKV-SRS AArch64 portability patch

This directory contains `gemm-common` 0.19.0 from crates.io, whose source
archive has SHA-256
`88027625910cc9b1085aaaa1c4bc46bb3a36aad323452b33c25b5e4e7c8e2a3e`.
It remains under the upstream MIT license in `LICENSE`.

The only source changes are four `#[target_feature(enable = "fp16,neon")]`
annotations in `src/simd.rs`. The functions contain optional FP16 AArch64
assembly and are reached through the crate's existing runtime FP16 feature
detection, but upstream 0.19.0 did not annotate the helper functions
themselves. Baseline Linux and Windows ARM64 assemblers consequently rejected
the crate with `instruction requires: fullfp16`.

The annotations let the optional instructions compile in isolated functions;
they do not enable FP16 globally or raise RWKV-SRS's ARM64 CPU baseline. Remove
this patch when a locked upstream release contains an equivalent fix and the
Linux and Windows ARM64 release lanes pass without it.
