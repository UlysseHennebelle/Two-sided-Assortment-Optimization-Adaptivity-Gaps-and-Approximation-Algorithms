"""TOML configuration helpers shared by scripts and experiments."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a TOML configuration without applying hidden defaults."""

    with Path(path).open("rb") as stream:
        return tomllib.load(stream)


def config_hash(config: dict[str, Any]) -> str:
    """Return a stable SHA-256 hash for a parsed configuration."""

    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
