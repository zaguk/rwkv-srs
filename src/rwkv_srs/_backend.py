from __future__ import annotations

import os

_BACKEND_ENV_VAR = "RWKV_SRS_BACKEND"
_DEFAULT_BACKEND = "rust"
_BACKEND_ALIASES = {
    "py": "python",
    "python": "python",
    "torch": "python",
    "rs": "rust",
    "rust": "rust",
}


def selected_backend() -> str:
    env_var = _BACKEND_ENV_VAR
    value = os.environ.get(env_var, _DEFAULT_BACKEND).strip().lower()
    if value == "":
        value = _DEFAULT_BACKEND
    try:
        return _BACKEND_ALIASES[value]
    except KeyError as exc:
        supported = ", ".join(sorted(_BACKEND_ALIASES))
        raise ValueError(
            f"Unsupported {env_var}={value!r}; expected one of: {supported}."
        ) from exc
