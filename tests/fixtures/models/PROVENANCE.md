# Repository model provenance

The two Safetensors files in this directory are deterministic tensor-for-tensor
conversions of the pretrained RWKV-SRS checkpoints published by
`open-spaced-repetition/srs-benchmark`. Conversion changed the container format
only; it did not train or alter the tensor values.

The release owner confirmed on 2026-07-17 that permission was obtained to
publish and redistribute the original checkpoints and these converted model
files. Evidence of that permission is retained privately. This provenance
record documents the release decision; it does not grant a public license.

| Repository file | SHA-256 | Source checkpoint | Source SHA-256 |
|---|---|---|---|
| `RWKV_trained_on_101_4999.safetensors` | `fa35cd87ee3589ad7457d54f844c90cb94d19d56e5104df04aa5fb0dcf28c4e9` | `RWKV_trained_on_101_4999.pth` | `82d60a4de2e26b4064b86359874e458d1546c0e9651c5dbdf9b8f449a9b97114` |
| `RWKV_trained_on_5000_10000.safetensors` | `0b1566d8159464fc512c8cb49525aefd3f84c9baaacdf0ea2e8d7a398729d91a` | `RWKV_trained_on_5000_10000.pth` | `8fb7fea081b38623ffc0d84ed18303cf5d62b42b9105e91112264a34286beeeb` |

The generic RWKV-SRS wheel and source distribution intentionally exclude these
files. A downstream distribution may deliberately copy a selected repository
model into `rwkv_srs/pretrained/` to enable named-model lookup.
