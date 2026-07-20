"""TOML configuration helpers shared by scripts and experiments."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a TOML configuration without applying hidden defaults."""

    with Path(path).open("rb") as stream:
        return tomllib.load(stream)
