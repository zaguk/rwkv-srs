from __future__ import annotations

# Compatibility shim for older imports. New code should import from
# rwkv_srs.backends.rust.
from rwkv_srs.backends.rust import *  # noqa: F401,F403
from rwkv_srs.backends.rust import (  # noqa: F401
    _RUST_CHECKPOINT_BIN_MAGIC,
    _load_checkpoint_dict,
)
