"""Paper policy identifiers and common algorithm results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class Side(str, Enum):
    CUSTOMERS = "customers"
    SUPPLIERS = "suppliers"


class PolicyClass(str, Enum):
    FULLY_STATIC = "FS"
    ONE_SIDED_STATIC = "OS"
    ONE_SIDED_ADAPTIVE = "OA"
    FULLY_ADAPTIVE = "FA"


@dataclass(frozen=True, slots=True)
class AlgorithmResult:
    name: str
    value: float
    initiating_side: Side | None = None
    replications: tuple[float, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
