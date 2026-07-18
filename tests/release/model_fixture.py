"""Generate the deterministic synthetic model used by public release tests.

The fixture contains no trained weights or external model data. Its schema is
an independent transcription of the native public checkpoint contract, and
its values are short repeating patterns derived only from tensor names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import BinaryIO


D_MODEL = 128
N_HEADS = 4
HEAD_SIZE = D_MODEL // N_HEADS
CARD_FEATURES_DIM = 92
FEATURES_FC_DIM = 4 * D_MODEL
HEAD_DIM = 4 * D_MODEL
NUM_CURVES = 128
NUM_POINTS = 128
EXPECTED_TENSOR_COUNT = 504
EXPECTED_PARAMETER_COUNT = 2_762_884
MODULE_CONFIGS = (
    (3, 3, 2),
    (4, 2, 1),
    (2, 3, 2),
    (3, 2, 1),
    (4, 2, 1),
)


def write_synthetic_model(path: Path) -> str:
    """Write the model and return its SHA-256 digest."""

    specs = expected_tensor_specs()
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, shape in specs.items():
        size = math.prod(shape) * 4
        offsets[name] = (cursor, cursor + size)
        cursor += size

    header: dict[str, object] = {
        "__metadata__": {
            "fixture": "rwkv-srs-public-synthetic-v1",
            "origin": "deterministically generated; no trained parameters",
        }
    }
    for name, shape in specs.items():
        start, end = offsets[name]
        header[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [start, end],
        }
    header_bytes = json.dumps(
        header,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    header_bytes += b" " * (-len(header_bytes) % 8)

    digest = hashlib.sha256()
    with path.open("wb") as output:
        _write(output, digest, struct.pack("<Q", len(header_bytes)))
        _write(output, digest, header_bytes)
        for name, shape in specs.items():
            _write_pattern(output, digest, _tensor_pattern(name), math.prod(shape))
    return digest.hexdigest()


def expected_tensor_specs() -> dict[str, tuple[int, ...]]:
    specs: dict[str, tuple[int, ...]] = {}

    _add_linear(specs, "features2card.0", FEATURES_FC_DIM, CARD_FEATURES_DIM, True)
    _add_layer_norm(specs, "features2card.2", FEATURES_FC_DIM)
    _add_linear(specs, "features2card.3", D_MODEL, FEATURES_FC_DIM, True)

    for module_index, (layers, factor_num, factor_den) in enumerate(MODULE_CONFIGS):
        channel_dim = D_MODEL * factor_num // factor_den
        for layer_index in range(layers):
            _add_rwkv_block(
                specs,
                module_index=module_index,
                layer_index=layer_index,
                channel_dim=channel_dim,
            )

    _add_layer_norm(specs, "prehead_norm", D_MODEL)
    _add_linear(specs, "head_ahead_logits.0", HEAD_DIM, D_MODEL, True)
    _add_linear(specs, "head_w.0", D_MODEL, D_MODEL, True)
    _add_layer_norm(specs, "head_w.2", D_MODEL)
    _add_linear(specs, "head_w.4", HEAD_DIM, D_MODEL, True)
    _add_linear(specs, "head_p.0", HEAD_DIM, D_MODEL, True)
    _add_linear(specs, "ahead_linear", NUM_POINTS, HEAD_DIM, True)
    _add_linear(specs, "w_linear", NUM_CURVES, HEAD_DIM, True)
    _add_linear(specs, "p_linear", 4, HEAD_DIM, True)

    specs = dict(sorted(specs.items()))
    assert len(specs) == EXPECTED_TENSOR_COUNT
    assert sum(math.prod(shape) for shape in specs.values()) == EXPECTED_PARAMETER_COUNT
    return specs


def _add_rwkv_block(
    specs: dict[str, tuple[int, ...]],
    *,
    module_index: int,
    layer_index: int,
    channel_dim: int,
) -> None:
    prefix = f"rwkv_modules.{module_index}.blocks.{layer_index}"
    time = f"{prefix}.time_mixer"
    _add_layer_norm(specs, f"{time}.layer_norm", D_MODEL)
    _add_param(specs, f"{time}.rkvdag_lerp", (8, 1, 1, D_MODEL))
    _add_param(specs, f"{time}.bonus", (1, 1, N_HEADS, HEAD_SIZE))
    for name in ("W_r", "W_k", "W_v", "W_o"):
        _add_linear(specs, f"{time}.{name}", D_MODEL, D_MODEL, False)
    _add_linear(specs, f"{time}.k_scale_linear", N_HEADS, D_MODEL, True)
    _add_linear(specs, f"{time}.v_scale_linear", N_HEADS, D_MODEL, True)
    _add_lora_simple(specs, f"{time}.v_lora_simple", 8)
    _add_lora_simple(specs, f"{time}.a_lora_simple", 16)
    _add_lora_simple(specs, f"{time}.d_lora_mlp", 16)
    _add_linear(specs, f"{time}.lora_A_g", 16, D_MODEL, False)
    _add_linear(specs, f"{time}.lora_B_g", D_MODEL, 16, False)
    _add_layer_norm(specs, f"{time}.out_group_norm", D_MODEL)

    channel = f"{prefix}.channel_mixer"
    _add_layer_norm(specs, f"{channel}.layer_norm", D_MODEL)
    _add_param(specs, f"{channel}.lerp_k", (1, 1, D_MODEL))
    _add_linear(specs, f"{channel}.W_k", channel_dim, D_MODEL, False)
    _add_linear(specs, f"{channel}.W_v", D_MODEL, channel_dim, False)


def _add_lora_simple(
    specs: dict[str, tuple[int, ...]],
    prefix: str,
    d_lora: int,
) -> None:
    _add_linear(specs, f"{prefix}.A", d_lora, D_MODEL, False)
    _add_linear(specs, f"{prefix}.B_and_lamb", D_MODEL, d_lora, True)


def _add_layer_norm(
    specs: dict[str, tuple[int, ...]],
    prefix: str,
    dim: int,
) -> None:
    _add_param(specs, f"{prefix}.weight", (dim,))
    _add_param(specs, f"{prefix}.bias", (dim,))


def _add_linear(
    specs: dict[str, tuple[int, ...]],
    prefix: str,
    out_dim: int,
    in_dim: int,
    bias: bool,
) -> None:
    _add_param(specs, f"{prefix}.weight", (out_dim, in_dim))
    if bias:
        _add_param(specs, f"{prefix}.bias", (out_dim,))


def _add_param(
    specs: dict[str, tuple[int, ...]],
    name: str,
    shape: tuple[int, ...],
) -> None:
    assert name not in specs
    specs[name] = shape


def _tensor_pattern(name: str) -> tuple[float, ...]:
    seed = hashlib.sha256(name.encode("utf-8")).digest()[0]
    pattern_length = 17
    centered = tuple(
        (((index * 7 + seed) % pattern_length) - 8) / 8.0
        for index in range(pattern_length)
    )
    if name.endswith(".weight") and (
        "layer_norm" in name or "group_norm" in name or name == "prehead_norm.weight"
    ):
        return tuple(1.0 + 0.015 * value for value in centered)
    elif name.endswith(".weight"):
        scale = 0.005 * (1 + seed % 3)
        return tuple(scale * value for value in centered)
    elif name.endswith(".bias"):
        scale = 0.0005 * (1 + seed % 5)
        return tuple(scale * value for value in centered)
    elif name.endswith(".bonus"):
        return tuple(0.002 * value for value in centered)
    elif name.endswith("rkvdag_lerp") or name.endswith("lerp_k"):
        return tuple(0.01 * value for value in centered)
    else:
        scale = 0.0002 * (1 + seed % 4)
        return tuple(scale * value for value in centered)


def _write_pattern(
    output: BinaryIO,
    digest: "hashlib._Hash",
    pattern: tuple[float, ...],
    count: int,
) -> None:
    pattern_bytes = struct.pack(f"<{len(pattern)}f", *pattern)
    complete, remainder = divmod(count, len(pattern))
    block_patterns = max(1, 65_536 // len(pattern_bytes))
    block = pattern_bytes * block_patterns
    while complete >= block_patterns:
        _write(output, digest, block)
        complete -= block_patterns
    if complete:
        _write(output, digest, pattern_bytes * complete)
    if remainder:
        _write(output, digest, pattern_bytes[: remainder * 4])


def _write(output: BinaryIO, digest: "hashlib._Hash", value: bytes) -> None:
    output.write(value)
    digest.update(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        action="append",
        type=Path,
        required=True,
        help="Write one generated model; may be supplied more than once.",
    )
    args = parser.parse_args()
    for output in args.output:
        digest = write_synthetic_model(output)
        print(f"{digest}  {output}")


if __name__ == "__main__":
    main()
