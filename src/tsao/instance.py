"""Market instances for the model in Section 2 and its MNL specialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _positive_vector(value: float | Sequence[float], size: int, name: str) -> NDArray[np.float64]:
    array = np.full(size, float(value), dtype=np.float64) if np.isscalar(value) else np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError(f"{name} must contain finite, strictly positive values")
    result = np.array(array, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _capacities(
    value: int | Sequence[int | None] | None,
    size: int,
    name: str,
) -> tuple[int | None, ...]:
    if value is None:
        return (None,) * size
    values = (
        (int(value),) * size
        if np.isscalar(value)
        else tuple(None if item is None else int(item) for item in value)
    )
    if len(values) != size or any(item is not None and item <= 0 for item in values):
        raise ValueError(f"{name} must contain {size} positive integers or None")
    return values


@dataclass(frozen=True, slots=True)
class MarketInstance:
    """Immutable MNL instance with customer matrix ``v`` and supplier matrix ``w``.

    ``v[i, j]`` is customer ``i``'s weight for supplier ``j`` and ``w[j, i]``
    is supplier ``j``'s weight for customer ``i``.
    """

    v: ArrayLike
    w: ArrayLike
    customer_outside: float | Sequence[float] = 1.0
    supplier_outside: float | Sequence[float] = 1.0
    customer_capacities: int | Sequence[int] | None = None
    supplier_capacities: int | Sequence[int] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        v = np.asarray(self.v, dtype=np.float64)
        w = np.asarray(self.w, dtype=np.float64)
        if v.ndim != 2 or w.ndim != 2 or w.shape != (v.shape[1], v.shape[0]):
            raise ValueError(f"Expected v=(n,m) and w=(m,n); got v={v.shape}, w={w.shape}")
        if not np.all(np.isfinite(v)) or not np.all(np.isfinite(w)) or np.any(v < 0) or np.any(w < 0):
            raise ValueError("Preference weights must be finite and nonnegative")
        v_copy = np.array(v, dtype=np.float64, copy=True)
        w_copy = np.array(w, dtype=np.float64, copy=True)
        v_copy.setflags(write=False)
        w_copy.setflags(write=False)
        object.__setattr__(self, "v", v_copy)
        object.__setattr__(self, "w", w_copy)
        object.__setattr__(self, "customer_outside", _positive_vector(self.customer_outside, v.shape[0], "customer_outside"))
        object.__setattr__(self, "supplier_outside", _positive_vector(self.supplier_outside, v.shape[1], "supplier_outside"))
        object.__setattr__(self, "customer_capacities", _capacities(self.customer_capacities, v.shape[0], "customer_capacities"))
        object.__setattr__(self, "supplier_capacities", _capacities(self.supplier_capacities, v.shape[1], "supplier_capacities"))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def num_customers(self) -> int:
        return int(self.v.shape[0])

    @property
    def num_suppliers(self) -> int:
        return int(self.v.shape[1])

    def transposed(self) -> "MarketInstance":
        """Return a new instance with customer and supplier roles exchanged."""

        return MarketInstance(
            v=self.w,
            w=self.v,
            customer_outside=self.supplier_outside,
            supplier_outside=self.customer_outside,
            customer_capacities=self.supplier_capacities,
            supplier_capacities=self.customer_capacities,
            metadata={**self.metadata, "transposed": not bool(self.metadata.get("transposed", False))},
        )

    def with_outside_option(self, value: float) -> "MarketInstance":
        """Return the same preferences with a common outside-option weight."""

        return MarketInstance(
            self.v,
            self.w,
            customer_outside=value,
            supplier_outside=value,
            customer_capacities=self.customer_capacities,
            supplier_capacities=self.supplier_capacities,
            metadata=self.metadata,
        )

    def normalized_outside(self) -> "MarketInstance":
        """Return an equivalent instance whose outside weights are all one.

        Dividing every agent's item weights by that agent's outside weight does
        not change MNL probabilities.
        """

        return MarketInstance(
            self.v / self.customer_outside[:, None],
            self.w / self.supplier_outside[:, None],
            customer_outside=1.0,
            supplier_outside=1.0,
            customer_capacities=self.customer_capacities,
            supplier_capacities=self.supplier_capacities,
            metadata={**self.metadata, "outside_normalized": True},
        )
