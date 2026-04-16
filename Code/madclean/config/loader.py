from __future__ import annotations

import copy
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from madclean.config.settings import CleaningConfig


def _default_config_path() -> Path:
    # loader.py -> config -> madclean -> Code
    return Path(__file__).resolve().parents[2] / "configurations.json"


def _coerce_value(name: str, value: Any, default: Any) -> Any:
    if name == "validator_failure_strategy":
        allowed = {"accept_cleaned", "leave_uncleaned", "ask_user"}
        s = str(value).strip()
        return s if s in allowed else default
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int):
        try:
            return int(value)
        except Exception:
            return default
    if isinstance(default, float) or default is None:
        # Keep optional floats permissive.
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return default
    if isinstance(default, list):
        return value if isinstance(value, list) else default
    if isinstance(default, dict):
        return value if isinstance(value, dict) else default
    return value


def load_default_cleaning_config(config_path: str | Path | None = None) -> CleaningConfig:
    """
    Load optional JSON overrides for CleaningConfig defaults.

    Unknown keys are ignored. Invalid values fall back to dataclass defaults.
    """
    defaults = CleaningConfig()
    path = Path(config_path) if config_path else _default_config_path()
    if not path.exists():
        return defaults

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return defaults
    if not isinstance(raw, dict):
        return defaults

    out: dict[str, Any] = {}
    for f in fields(CleaningConfig):
        name = f.name
        default_value = copy.deepcopy(getattr(defaults, name))
        if name not in raw:
            out[name] = default_value
            continue

        incoming = raw.get(name)
        if name == "sample_sizes" and isinstance(incoming, dict):
            merged = copy.deepcopy(default_value)
            for k, sub in incoming.items():
                if isinstance(sub, dict) and isinstance(merged.get(k), dict):
                    merged[k].update(sub)
                else:
                    merged[k] = sub
            out[name] = merged
            continue

        out[name] = _coerce_value(name, incoming, default_value)

    return CleaningConfig(**out)

