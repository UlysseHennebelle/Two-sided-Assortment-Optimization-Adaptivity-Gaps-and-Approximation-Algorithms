"""Single-agent MNL assortment oracles used by Algorithms 1 and 2.

The unconstrained oracle retains the notebook's key optimization: sort items by
continuation revenue and add them while the next revenue is at least the current
MNL expected value. It therefore avoids enumerating all assortments.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Hashable, Iterable, Sequence

from .choice import mnl_expected_value


@dataclass(frozen=True, slots=True)
class MnlOption:
    item_id: Hashable
    revenue: float
    weight: float


@dataclass(frozen=True, slots=True)
class AssortmentSolution:
    item_ids: tuple[Hashable, ...]
    value: float


def optimize_mnl(options: Iterable[MnlOption], outside_weight: float = 1.0) -> AssortmentSolution:
    """Solve the unconstrained revenue-ordered MNL assortment problem exactly."""

    if outside_weight <= 0:
        raise ValueError("outside_weight must be positive")
    ordered = sorted((option for option in options if option.weight > 0), key=lambda option: option.revenue, reverse=True)
    chosen: list[Hashable] = []
    numerator = 0.0
    total_weight = 0.0
    value = 0.0
    for option in ordered:
        if option.revenue + 1e-15 < value:
            break
        chosen.append(option.item_id)
        numerator += option.revenue * option.weight
        total_weight += option.weight
        value = numerator / (outside_weight + total_weight)
    return AssortmentSolution(tuple(chosen), float(value))


def optimize_mnl_cardinality(
    options: Sequence[MnlOption],
    capacity: int,
    outside_weight: float = 1.0,
) -> AssortmentSolution:
    """Exact constrained oracle by enumeration, intended for small validation cases.

    Section 6 assumes access to a polynomial-time constrained oracle. The paper's
    current numerical experiments are unconstrained, so this explicit routine is
    deliberately limited to small tests instead of hiding an exponential method
    inside a large experiment.
    """

    if capacity <= 0:
        return AssortmentSolution((), 0.0)
    positive = [option for option in options if option.weight > 0]
    best = AssortmentSolution((), 0.0)
    for size in range(1, min(capacity, len(positive)) + 1):
        for subset in combinations(positive, size):
            value = mnl_expected_value(
                [option.revenue for option in subset],
                [option.weight for option in subset],
                outside_weight,
            )
            if value > best.value:
                best = AssortmentSolution(tuple(option.item_id for option in subset), value)
    return best
