# Third-party notices

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

- Cargo.lock SHA-256: `a83481c02cc3c8d6518955edb2d6f67bf6e31a7e7a83de3b8aa2bbde88842996`
- Third-party packages: 183
- Unique bundled license/notice texts: 109

| Package | Version | Declared license | Declared authors | Source/repository |
|---|---:|---|---|---|
| allocator-api2 | 0.2.21 | MIT OR Apache-2.0 | Zakarum <zaq.dev@icloud.com> | https://github.com/zakarumych/allocator-api2 |
| android_system_properties | 0.1.5 | MIT/Apache-2.0 | Nicolas Silva <nical@fastmail.com> | https://github.com/nical/android_system_properties |
| anyhow | 1.0.102 | MIT OR Apache-2.0 | David Tolnay <dtolnay@gmail.com> | https://github.com/dtolnay/anyhow |
| arrayvec | 0.7.8 | MIT OR Apache-2.0 | bluss | https://github.com/bluss/arrayvec |
| ash | 0.38.0+1.3.281 | MIT OR Apache-2.0 | Maik Klein <maikklein@googlemail.com>, Benjamin Saunders <ben.e.saunders@gmail.com>, Marijn Suijten <marijn@traverseresearch.nl> | https://github.com/ash-rs/ash |
| autocfg | 1.5.0 | Apache-2.0 OR MIT | Josh Stone <cuviper@gmail.com> | https://github.com/cuviper/autocfg |
| bit-set | 0.10.0 | Apache-2.0 OR MIT | Alexis Beingessner <a.beingessner@gmail.com> | https://github.com/contain-rs/bit-set |
| bit-vec | 0.9.1 | Apache-2.0 OR MIT | Alexis Beingessner <a.beingessner@gmail.com> | https://github.com/contain-rs/bit-vec |
| bitflags | 2.11.1 | MIT OR Apache-2.0 | The Rust Project Developers | https://github.com/bitflags/bitflags |
| block-buffer | 0.10.4 | MIT OR Apache-2.0 | RustCrypto Developers | https://github.com/RustCrypto/utils |
| block2 | 0.6.2 | MIT | Mads Marquart <mads@marquart.dk> | https://github.com/madsmtm/objc2 |
| bumpalo | 3.20.3 | MIT OR Apache-2.0 | Nick Fitzgerald <fitzgen@gmail.com> | https://github.com/fitzgen/bumpalo |
| bytemuck | 1.25.0 | Zlib OR Apache-2.0 OR MIT | Lokathor <zefria@gmail.com> | https://github.com/Lokathor/bytemuck |
| bytemuck_derive | 1.10.2 | Zlib OR Apache-2.0 OR MIT | Lokathor <zefria@gmail.com> | https://github.com/Lokathor/bytemuck |
| byteorder | 1.5.0 | Unlicense OR MIT | Andrew Gallant <jamslam@gmail.com> | https://github.com/BurntSushi/byteorder |
| candle-core | 0.9.2 | MIT OR Apache-2.0 | not declared | https://github.com/huggingface/candle |
| candle-nn | 0.9.2 | MIT OR Apache-2.0 | not declared | https://github.com/huggingface/candle |
| cfg-if | 1.0.4 | MIT OR Apache-2.0 | Alex Crichton <alex@alexcrichton.com> | https://github.com/rust-lang/cfg-if |
| cfg_aliases | 0.2.1 | MIT | Zicklag <zicklag@katharostech.com> | https://github.com/katharostech/cfg_aliases |
| codespan-reporting | 0.13.1 | Apache-2.0 | Brendan Zabarauskas <bjzaba@yahoo.com.au> | https://github.com/brendanzab/codespan |
| cpufeatures | 0.2.17 | MIT OR Apache-2.0 | RustCrypto Developers | https://github.com/RustCrypto/utils |
| crc32fast | 1.5.0 | MIT OR Apache-2.0 | Sam Rijs <srijs@airpost.net>, Alex Crichton <alex@alexcrichton.com> | https://github.com/srijs/rust-crc32fast |
| crossbeam-deque | 0.8.6 | MIT OR Apache-2.0 | not declared | https://github.com/crossbeam-rs/crossbeam |
| crossbeam-epoch | 0.9.18 | MIT OR Apache-2.0 | not declared | https://github.com/crossbeam-rs/crossbeam |
| crossbeam-utils | 0.8.21 | MIT OR Apache-2.0 | not declared | https://github.com/crossbeam-rs/crossbeam |
| crunchy | 0.2.4 | MIT | Eira Fransham <jackefransham@gmail.com> | https://github.com/eira-fransham/crunchy |
| crypto-common | 0.1.7 | MIT OR Apache-2.0 | RustCrypto Developers | https://github.com/RustCrypto/traits |
| digest | 0.10.7 | MIT OR Apache-2.0 | RustCrypto Developers | https://github.com/RustCrypto/traits |
| dispatch2 | 0.3.1 | Zlib OR Apache-2.0 OR MIT | Mads Marquart <mads@marquart.dk>, Mary <mary@mary.zone> | https://github.com/madsmtm/objc2 |
| document-features | 0.2.12 | MIT OR Apache-2.0 | Slint Developers <info@slint.dev> | https://github.com/slint-ui/document-features |
| dyn-stack | 0.13.2 | MIT | sarah <> | https://codeberg.org/sarah-quinones/dyn-stack |
| dyn-stack-macros | 0.1.3 | MIT | sarah quiñones<sarah@veganb.tw> | https://github.com/kitegi/dynstack/ |
| either | 1.16.0 | MIT OR Apache-2.0 | not declared | https://github.com/rayon-rs/either |
| enum-as-inner | 0.6.1 | MIT/Apache-2.0 | Benjamin Fry <benjaminfry@me.com> | https://github.com/bluejekyll/enum-as-inner |
| equivalent | 1.0.2 | Apache-2.0 OR MIT | not declared | https://github.com/indexmap-rs/equivalent |
| float8 | 0.6.1 | MIT | not declared | https://github.com/EricLBuehler/float8 |
| foldhash | 0.2.0 | Zlib | Orson Peters <orsonpeters@gmail.com> | https://github.com/orlp/foldhash |
| futures-core | 0.3.32 | MIT OR Apache-2.0 | not declared | https://github.com/rust-lang/futures-rs |
| futures-task | 0.3.32 | MIT OR Apache-2.0 | not declared | https://github.com/rust-lang/futures-rs |
| futures-util | 0.3.32 | MIT OR Apache-2.0 | not declared | https://github.com/rust-lang/futures-rs |
| gemm | 0.19.0 | MIT | sarah <> | https://github.com/sarah-ek/gemm/ |
| gemm-c32 | 0.19.0 | MIT | sarah <> | https://github.com/sarah-ek/gemm/ |
| gemm-c64 | 0.19.0 | MIT | sarah <> | https://github.com/sarah-ek/gemm/ |
| gemm-common | 0.19.0 | MIT | sarah <> | vendored+rust/rwkv-srs-cpu/vendor/gemm-common-0.19.0 (upstream: https://github.com/sarah-ek/gemm/) |
| gemm-f16 | 0.19.0 | MIT | sarah <> | https://github.com/sarah-ek/gemm/ |
| gemm-f32 | 0.19.0 | MIT | sarah <> | https://github.com/sarah-ek/gemm/ |
| gemm-f64 | 0.19.0 | MIT | sarah <> | https://github.com/sarah-ek/gemm/ |
| generic-array | 0.14.7 | MIT | Bartłomiej Kamiński <fizyk20@gmail.com>, Aaron Trent <novacrazy@gmail.com> | https://github.com/fizyk20/generic-array.git |
| getrandom | 0.3.4 | MIT OR Apache-2.0 | The Rand Project Developers | https://github.com/rust-random/getrandom |
| gpu-allocator | 0.28.0 | MIT OR Apache-2.0 | Traverse Research <opensource@traverseresearch.nl> | https://github.com/Traverse-Research/gpu-allocator |
| half | 2.7.1 | MIT OR Apache-2.0 | Kathryn Long <squeeself@gmail.com> | https://github.com/VoidStarKat/half-rs |
| hashbrown | 0.16.1 | MIT OR Apache-2.0 | Amanieu d'Antras <amanieu@gmail.com> | https://github.com/rust-lang/hashbrown |
| hashbrown | 0.17.1 | MIT OR Apache-2.0 | not declared | https://github.com/rust-lang/hashbrown |
| heck | 0.5.0 | MIT OR Apache-2.0 | not declared | https://github.com/withoutboats/heck |
| hermit-abi | 0.5.2 | MIT OR Apache-2.0 | Stefan Lankes | https://github.com/hermit-os/hermit-rs |
| indexmap | 2.14.0 | Apache-2.0 OR MIT | not declared | https://github.com/indexmap-rs/indexmap |
| indoc | 2.0.7 | MIT OR Apache-2.0 | David Tolnay <dtolnay@gmail.com> | https://github.com/dtolnay/indoc |
| itoa | 1.0.18 | MIT OR Apache-2.0 | David Tolnay <dtolnay@gmail.com> | https://github.com/dtolnay/itoa |
| js-sys | 0.3.103 | MIT OR Apache-2.0 | The wasm-bindgen Developers | https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/js-sys |
| libc | 0.2.186 | MIT OR Apache-2.0 | The Rust Project Developers | https://github.com/rust-lang/libc |
| libloading | 0.8.9 | ISC | Simonas Kazlauskas <libloading@kazlauskas.me> | https://github.com/nagisa/rust_libloading/ |
| libm | 0.2.16 | MIT | Alex Crichton <alex@alexcrichton.com>, Amanieu d'Antras <amanieu@gmail.com>, Jorge Aparicio <japaricious@gmail.com>, Trevor Gross <tg@trevorgross.com> | https://github.com/rust-lang/compiler-builtins |
| litrs | 1.0.0 | MIT OR Apache-2.0 | Lukas Kalbertodt <lukas.kalbertodt@gmail.com> | https://github.com/LukasKalbertodt/litrs |
| lock_api | 0.4.14 | MIT OR Apache-2.0 | Amanieu d'Antras <amanieu@gmail.com> | https://github.com/Amanieu/parking_lot |
| log | 0.4.33 | MIT OR Apache-2.0 | The Rust Project Developers | https://github.com/rust-lang/log |
| memchr | 2.8.0 | Unlicense OR MIT | Andrew Gallant <jamslam@gmail.com>, bluss | https://github.com/BurntSushi/memchr |
| memmap2 | 0.9.10 | MIT OR Apache-2.0 | Dan Burkert <dan@danburkert.com>, Yevhenii Reizner <razrfalcon@gmail.com>, The Contributors | https://github.com/RazrFalcon/memmap2-rs |
| memoffset | 0.9.1 | MIT | Gilad Naaman <gilad.naaman@gmail.com> | https://github.com/Gilnaa/memoffset |
| naga | 30.0.0 | MIT OR Apache-2.0 | gfx-rs developers | https://github.com/gfx-rs/wgpu |
| naga-types | 30.0.0 | MIT OR Apache-2.0 | gfx-rs developers | https://github.com/gfx-rs/wgpu |
| num-complex | 0.4.6 | MIT OR Apache-2.0 | The Rust Project Developers | https://github.com/rust-num/num-complex |
| num-traits | 0.2.19 | MIT OR Apache-2.0 | The Rust Project Developers | https://github.com/rust-num/num-traits |
| num_cpus | 1.17.0 | MIT OR Apache-2.0 | Sean McArthur <sean@seanmonstar.com> | https://github.com/seanmonstar/num_cpus |
| objc2 | 0.6.4 | MIT | Mads Marquart <mads@marquart.dk> | https://github.com/madsmtm/objc2 |
| objc2-core-foundation | 0.3.2 | Zlib OR Apache-2.0 OR MIT | not declared | https://github.com/madsmtm/objc2 |
| objc2-core-graphics | 0.3.2 | Zlib OR Apache-2.0 OR MIT | not declared | https://github.com/madsmtm/objc2 |
| objc2-encode | 4.1.0 | MIT | Mads Marquart <mads@marquart.dk> | https://github.com/madsmtm/objc2 |
| objc2-foundation | 0.3.2 | MIT | not declared | https://github.com/madsmtm/objc2 |
| objc2-io-surface | 0.3.2 | Zlib OR Apache-2.0 OR MIT | not declared | https://github.com/madsmtm/objc2 |
| objc2-metal | 0.3.2 | Zlib OR Apache-2.0 OR MIT | not declared | https://github.com/madsmtm/objc2 |
| objc2-quartz-core | 0.3.2 | Zlib OR Apache-2.0 OR MIT | not declared | https://github.com/madsmtm/objc2 |
| once_cell | 1.21.4 | MIT OR Apache-2.0 | Aleksey Kladov <aleksey.kladov@gmail.com> | https://github.com/matklad/once_cell |
| ordered-float | 5.3.0 | MIT | Jonathan Reem <jonathan.reem@gmail.com>, Matt Brubeck <mbrubeck@limpet.net> | https://github.com/reem/rust-ordered-float |
| parking_lot | 0.12.5 | MIT OR Apache-2.0 | Amanieu d'Antras <amanieu@gmail.com> | https://github.com/Amanieu/parking_lot |
| parking_lot_core | 0.9.12 | MIT OR Apache-2.0 | Amanieu d'Antras <amanieu@gmail.com> | https://github.com/Amanieu/parking_lot |
| paste | 1.0.15 | MIT OR Apache-2.0 | David Tolnay <dtolnay@gmail.com> | https://github.com/dtolnay/paste |
| pin-project-lite | 0.2.17 | Apache-2.0 OR MIT | not declared | https://github.com/taiki-e/pin-project-lite |
| pollster | 1.0.1 | Apache-2.0/MIT | Joshua Barretto <joshua@jsbarretto.com> | https://github.com/zesterer/pollster |
| portable-atomic | 1.13.1 | Apache-2.0 OR MIT | not declared | https://github.com/taiki-e/portable-atomic |
| portable-atomic-util | 0.2.7 | Apache-2.0 OR MIT | not declared | https://github.com/taiki-e/portable-atomic-util |
| ppv-lite86 | 0.2.21 | MIT OR Apache-2.0 | The CryptoCorrosion Contributors | https://github.com/cryptocorrosion/cryptocorrosion |
| presser | 0.3.1 | MIT OR Apache-2.0 | Embark <opensource@embark-studios.com>, Gray Olson <gray@grayolson.com | https://github.com/EmbarkStudios/presser |
| proc-macro2 | 1.0.106 | MIT OR Apache-2.0 | David Tolnay <dtolnay@gmail.com>, Alex Crichton <alex@alexcrichton.com> | https://github.com/dtolnay/proc-macro2 |
| profiling | 1.0.18 | MIT OR Apache-2.0 | Philip Degarmo <aclysma@gmail.com> | https://github.com/aclysma/profiling |
| pulp | 0.22.2 | MIT | sarah quiñones <sarah@veganb.tw> | https://github.com/sarah-quinones/pulp/ |
| pulp-wasm-simd-flag | 0.1.0 | MIT | sarah quiñones <sarah@veganb.tw> | https://github.com/sarah-quinones/pulp/ |
| pyo3 | 0.22.6 | MIT OR Apache-2.0 | PyO3 Project and Contributors <https://github.com/PyO3> | https://github.com/pyo3/pyo3 |
| pyo3-build-config | 0.22.6 | MIT OR Apache-2.0 | PyO3 Project and Contributors <https://github.com/PyO3> | https://github.com/pyo3/pyo3 |
| pyo3-ffi | 0.22.6 | MIT OR Apache-2.0 | PyO3 Project and Contributors <https://github.com/PyO3> | https://github.com/pyo3/pyo3 |
| pyo3-macros | 0.22.6 | MIT OR Apache-2.0 | PyO3 Project and Contributors <https://github.com/PyO3> | https://github.com/pyo3/pyo3 |
| pyo3-macros-backend | 0.22.6 | MIT OR Apache-2.0 | PyO3 Project and Contributors <https://github.com/PyO3> | https://github.com/pyo3/pyo3 |
| quote | 1.0.45 | MIT OR Apache-2.0 | David Tolnay <dtolnay@gmail.com> | https://github.com/dtolnay/quote |
| r-efi | 5.3.0 | MIT OR Apache-2.0 OR LGPL-2.1-or-later | not declared | https://github.com/r-efi/r-efi |
| rand | 0.9.4 | MIT OR Apache-2.0 | The Rand Project Developers, The Rust Project Developers | https://github.com/rust-random/rand |
| rand_chacha | 0.9.0 | MIT OR Apache-2.0 | The Rand Project Developers, The Rust Project Developers, The CryptoCorrosion Contributors | https://github.com/rust-random/rand |
| rand_core | 0.9.5 | MIT OR Apache-2.0 | The Rand Project Developers, The Rust Project Developers | https://github.com/rust-random/rand |
| rand_distr | 0.5.1 | MIT OR Apache-2.0 | The Rand Project Developers | https://github.com/rust-random/rand_distr |
| range-alloc | 0.1.5 | MIT OR Apache-2.0 | the gfx-rs Developers | https://github.com/gfx-rs/range-alloc |
| raw-cpuid | 11.6.0 | MIT | Gerd Zellweger <mail@gerdzellweger.com> | https://github.com/gz/rust-cpuid |
| raw-window-handle | 0.6.2 | MIT OR Apache-2.0 OR Zlib | Osspial <osspial@gmail.com> | https://github.com/rust-windowing/raw-window-handle |
| raw-window-metal | 1.1.0 | MIT OR Apache-2.0 | not declared | https://github.com/rust-windowing/raw-window-metal |
| rayon | 1.12.0 | MIT OR Apache-2.0 | not declared | https://github.com/rayon-rs/rayon |
| rayon-core | 1.13.0 | MIT OR Apache-2.0 | not declared | https://github.com/rayon-rs/rayon |
| reborrow | 0.5.5 | MIT | sarah <> | https://github.com/sarah-ek/reborrow/ |
| redox_syscall | 0.5.18 | MIT | Jeremy Soller <jackpot51@gmail.com> | https://gitlab.redox-os.org/redox-os/syscall |
| renderdoc-sys | 1.1.0 | MIT OR Apache-2.0 | Eyal Kalderon <ebkalderon@gmail.com> | https://github.com/ebkalderon/renderdoc-rs |
| rustc-hash | 1.1.0 | Apache-2.0/MIT | The Rust Project Developers | https://github.com/rust-lang-nursery/rustc-hash |
| rustversion | 1.0.22 | MIT OR Apache-2.0 | David Tolnay <dtolnay@gmail.com> | https://github.com/dtolnay/rustversion |
| safetensors | 0.7.0 | Apache-2.0 | not declared | https://github.com/huggingface/safetensors |
| same-file | 1.0.6 | Unlicense/MIT | Andrew Gallant <jamslam@gmail.com> | https://github.com/BurntSushi/same-file |
| scopeguard | 1.2.0 | MIT OR Apache-2.0 | bluss | https://github.com/bluss/scopeguard |
| seq-macro | 0.3.6 | MIT OR Apache-2.0 | David Tolnay <dtolnay@gmail.com> | https://github.com/dtolnay/seq-macro |
| serde | 1.0.228 | MIT OR Apache-2.0 | Erick Tryzelaar <erick.tryzelaar@gmail.com>, David Tolnay <dtolnay@gmail.com> | https://github.com/serde-rs/serde |
| serde_core | 1.0.228 | MIT OR Apache-2.0 | Erick Tryzelaar <erick.tryzelaar@gmail.com>, David Tolnay <dtolnay@gmail.com> | https://github.com/serde-rs/serde |
| serde_derive | 1.0.228 | MIT OR Apache-2.0 | Erick Tryzelaar <erick.tryzelaar@gmail.com>, David Tolnay <dtolnay@gmail.com> | https://github.com/serde-rs/serde |
| serde_json | 1.0.149 | MIT OR Apache-2.0 | Erick Tryzelaar <erick.tryzelaar@gmail.com>, David Tolnay <dtolnay@gmail.com> | https://github.com/serde-rs/json |
| sha2 | 0.10.9 | MIT OR Apache-2.0 | RustCrypto Developers | https://github.com/RustCrypto/hashes |
| slab | 0.4.12 | MIT | Carl Lerche <me@carllerche.com> | https://github.com/tokio-rs/slab |
| smallvec | 1.15.2 | MIT OR Apache-2.0 | The Servo Project Developers | https://github.com/servo/rust-smallvec |
| spirv | 0.4.0+sdk-1.4.341.0 | Apache-2.0 | Lei Zhang <antiagainst@gmail.com> | https://github.com/gfx-rs/rspirv |
| stable_deref_trait | 1.2.1 | MIT OR Apache-2.0 | Robert Grosse <n210241048576@gmail.com> | https://github.com/storyyeller/stable_deref_trait |
| static_assertions | 1.1.0 | MIT OR Apache-2.0 | Nikolai Vazquez | https://github.com/nvzqz/static-assertions-rs |
| syn | 2.0.117 | MIT OR Apache-2.0 | David Tolnay <dtolnay@gmail.com> | https://github.com/dtolnay/syn |
| synstructure | 0.13.2 | MIT | Nika Layzell <nika@thelayzells.com> | https://github.com/mystor/synstructure |
| sysctl | 0.6.0 | MIT | Johannes Lundberg <johalun0@gmail.com>, Ivan Temchenko <ivan.temchenko@yandex.ua>, Fabian Freyer <fabian.freyer@physik.tu-berlin.de> | https://github.com/johalun/sysctl-rs |
| target-lexicon | 0.12.16 | Apache-2.0 WITH LLVM-exception | Dan Gohman <sunfish@mozilla.com> | https://github.com/bytecodealliance/target-lexicon |
| thiserror | 1.0.69 | MIT OR Apache-2.0 | David Tolnay <dtolnay@gmail.com> | https://github.com/dtolnay/thiserror |
| thiserror | 2.0.18 | MIT OR Apache-2.0 | David Tolnay <dtolnay@gmail.com> | https://github.com/dtolnay/thiserror |
| thiserror-impl | 1.0.69 | MIT OR Apache-2.0 | David Tolnay <dtolnay@gmail.com> | https://github.com/dtolnay/thiserror |
| thiserror-impl | 2.0.18 | MIT OR Apache-2.0 | David Tolnay <dtolnay@gmail.com> | https://github.com/dtolnay/thiserror |
| typed-path | 0.12.3 | MIT OR Apache-2.0 | Chip Senkbeil <chip@senkbeil.org> | https://github.com/chipsenkbeil/typed-path |
| typenum | 1.20.1 | MIT OR Apache-2.0 | not declared | https://github.com/paholg/typenum |
| unicode-ident | 1.0.24 | (MIT OR Apache-2.0) AND Unicode-3.0 | David Tolnay <dtolnay@gmail.com> | https://github.com/dtolnay/unicode-ident |
| unicode-width | 0.2.2 | MIT OR Apache-2.0 | kwantam <kwantam@gmail.com>, Manish Goregaokar <manishsmail@gmail.com> | https://github.com/unicode-rs/unicode-width |
| unindent | 0.2.4 | MIT OR Apache-2.0 | David Tolnay <dtolnay@gmail.com> | https://github.com/dtolnay/indoc |
| version_check | 0.9.5 | MIT/Apache-2.0 | Sergio Benitez <sb@sergio.bz> | https://github.com/SergioBenitez/version_check |
| walkdir | 2.5.0 | Unlicense/MIT | Andrew Gallant <jamslam@gmail.com> | https://github.com/BurntSushi/walkdir |
| wasip2 | 1.0.3+wasi-0.2.9 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | not declared | https://github.com/bytecodealliance/wasi-rs |
| wasm-bindgen | 0.2.126 | MIT OR Apache-2.0 | The wasm-bindgen Developers | https://github.com/wasm-bindgen/wasm-bindgen |
| wasm-bindgen-futures | 0.4.76 | MIT OR Apache-2.0 | The wasm-bindgen Developers | https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/futures |
| wasm-bindgen-macro | 0.2.126 | MIT OR Apache-2.0 | The wasm-bindgen Developers | https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/macro |
| wasm-bindgen-macro-support | 0.2.126 | MIT OR Apache-2.0 | The wasm-bindgen Developers | https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/macro-support |
| wasm-bindgen-shared | 0.2.126 | MIT OR Apache-2.0 | The wasm-bindgen Developers | https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/shared |
| web-sys | 0.3.103 | MIT OR Apache-2.0 | The wasm-bindgen Developers | https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/web-sys |
| wgpu | 30.0.0 | MIT OR Apache-2.0 | gfx-rs developers | https://github.com/gfx-rs/wgpu |
| wgpu-core | 30.0.0 | MIT OR Apache-2.0 | gfx-rs developers | https://github.com/gfx-rs/wgpu |
| wgpu-core-deps-apple | 30.0.0 | MIT OR Apache-2.0 | gfx-rs developers | https://github.com/gfx-rs/wgpu |
| wgpu-core-deps-windows-linux-android | 30.0.0 | MIT OR Apache-2.0 | gfx-rs developers | https://github.com/gfx-rs/wgpu |
| wgpu-hal | 30.0.0 | MIT OR Apache-2.0 | gfx-rs developers | https://github.com/gfx-rs/wgpu |
| wgpu-naga-bridge | 30.0.0 | MIT OR Apache-2.0 | gfx-rs developers | https://github.com/gfx-rs/wgpu |
| wgpu-types | 30.0.0 | MIT OR Apache-2.0 | gfx-rs developers | https://github.com/gfx-rs/wgpu |
| winapi-util | 0.1.11 | Unlicense OR MIT | Andrew Gallant <jamslam@gmail.com> | https://github.com/BurntSushi/winapi-util |
| windows | 0.62.2 | MIT OR Apache-2.0 | not declared | https://github.com/microsoft/windows-rs |
| windows-collections | 0.3.2 | MIT OR Apache-2.0 | not declared | https://github.com/microsoft/windows-rs |
| windows-core | 0.62.2 | MIT OR Apache-2.0 | not declared | https://github.com/microsoft/windows-rs |
| windows-future | 0.3.2 | MIT OR Apache-2.0 | not declared | https://github.com/microsoft/windows-rs |
| windows-implement | 0.60.2 | MIT OR Apache-2.0 | not declared | https://github.com/microsoft/windows-rs |
| windows-interface | 0.59.3 | MIT OR Apache-2.0 | not declared | https://github.com/microsoft/windows-rs |
| windows-link | 0.2.1 | MIT OR Apache-2.0 | not declared | https://github.com/microsoft/windows-rs |
| windows-numerics | 0.3.1 | MIT OR Apache-2.0 | not declared | https://github.com/microsoft/windows-rs |
| windows-result | 0.4.1 | MIT OR Apache-2.0 | not declared | https://github.com/microsoft/windows-rs |
| windows-strings | 0.5.1 | MIT OR Apache-2.0 | not declared | https://github.com/microsoft/windows-rs |
| windows-sys | 0.61.2 | MIT OR Apache-2.0 | not declared | https://github.com/microsoft/windows-rs |
| windows-threading | 0.2.1 | MIT OR Apache-2.0 | not declared | https://github.com/microsoft/windows-rs |
| wit-bindgen | 0.57.1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | Alex Crichton <alex@alexcrichton.com> | https://github.com/bytecodealliance/wit-bindgen |
| yoke | 0.8.2 | Unicode-3.0 | Manish Goregaokar <manishsmail@gmail.com> | https://github.com/unicode-org/icu4x |
| yoke-derive | 0.8.2 | Unicode-3.0 | Manish Goregaokar <manishsmail@gmail.com> | https://github.com/unicode-org/icu4x |
| zerocopy | 0.8.48 | BSD-2-Clause OR Apache-2.0 OR MIT | Joshua Liebow-Feeser <joshlf@google.com>, Jack Wrenn <jswrenn@amazon.com> | https://github.com/google/zerocopy |
| zerocopy-derive | 0.8.48 | BSD-2-Clause OR Apache-2.0 OR MIT | Joshua Liebow-Feeser <joshlf@google.com>, Jack Wrenn <jswrenn@amazon.com> | https://github.com/google/zerocopy |
| zerofrom | 0.1.8 | Unicode-3.0 | The ICU4X Project Developers | https://github.com/unicode-org/icu4x |
| zerofrom-derive | 0.1.7 | Unicode-3.0 | Manish Goregaokar <manishsmail@gmail.com> | https://github.com/unicode-org/icu4x |
| zip | 7.2.0 | MIT | Mathijs van de Nes <git@mathijs.vd-nes.nl>, Marli Frost <marli@frost.red>, Ryan Levick <ryan.levick@gmail.com>, Chris Hennick <hennickc@amazon.com> | https://github.com/zip-rs/zip2.git |
| zmij | 1.0.21 | MIT | David Tolnay <dtolnay@gmail.com> | https://github.com/dtolnay/zmij |

## Published crates without a root license file

The following crate archives or explicitly vendored sources declare an SPDX
license in Cargo metadata but do not contain a root-level `LICENSE`, `LICENCE`,
`COPYING`, or `NOTICE` file. Their declared expressions, authors, and
repositories remain recorded in the table above. Standard texts supplied by
other locked packages are retained in the license archive where applicable.

- `block2 0.6.2`
- `candle-nn 0.9.2`
- `dispatch2 0.3.1`
- `objc2 0.6.4`
- `objc2-core-foundation 0.3.2`
- `objc2-core-graphics 0.3.2`
- `objc2-encode 4.1.0`
- `objc2-foundation 0.3.2`
- `objc2-io-surface 0.3.2`
- `objc2-metal 0.3.2`
- `objc2-quartz-core 0.3.2`
- `profiling 1.0.18`
- `pulp-wasm-simd-flag 0.1.0`
- `r-efi 5.3.0`
- `spirv 0.4.0+sdk-1.4.341.0`

## License text archive

`THIRD_PARTY_LICENSES.txt` contains every unique root-level license and notice
file shipped in the locked crates.io source archives and explicitly patched
vendored sources. Exact duplicate texts are stored once and mapped back to
every package member that supplied them.
