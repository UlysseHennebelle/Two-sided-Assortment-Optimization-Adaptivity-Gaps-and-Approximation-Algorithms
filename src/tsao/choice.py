"""MNL probabilities, demand, values, and seeded sampling."""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def mnl_probabilities(weights: ArrayLike, outside_weight: float = 1.0) -> tuple[float, NDArray[np.float64]]:
    """Return outside and item probabilities for an offered weight vector."""

    item_weights = np.asarray(weights, dtype=np.float64)
    if item_weights.ndim != 1 or np.any(item_weights < 0) or outside_weight <= 0:
        raise ValueError("MNL weights must be one-dimensional and nonnegative; outside weight must be positive")
    denominator = float(outside_weight + item_weights.sum())
    return float(outside_weight / denominator), item_weights / denominator


def mnl_demand(weights: ArrayLike, outside_weight: float = 1.0) -> float:
    """Return the probability that any offered item is selected."""

    outside_probability, _ = mnl_probabilities(weights, outside_weight)
    return 1.0 - outside_probability


def mnl_expected_value(revenues: ArrayLike, weights: ArrayLike, outside_weight: float = 1.0) -> float:
    """Return ``sum(revenue * weight) / (outside + sum(weight))``."""

    revenue_array = np.asarray(revenues, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    if revenue_array.shape != weight_array.shape:
        raise ValueError("revenues and weights must have identical shapes")
    if np.any(weight_array < 0) or outside_weight <= 0:
        raise ValueError("weights must be nonnegative and outside_weight positive")
    return float(np.dot(revenue_array, weight_array) / (outside_weight + weight_array.sum()))


def sample_mnl(
    item_ids: Sequence[Hashable],
    weights: ArrayLike,
    rng: np.random.Generator,
    outside_weight: float = 1.0,
) -> Hashable | None:
    """Sample one item, returning ``None`` for the outside option."""

    weight_array = np.asarray(weights, dtype=np.float64)
    if len(item_ids) != len(weight_array):
        raise ValueError("item_ids and weights must have the same length")
    threshold = float(rng.random() * (outside_weight + weight_array.sum()))
    if threshold < outside_weight:
        return None
    threshold -= outside_weight
    cumulative = 0.0
    for item_id, weight in zip(item_ids, weight_array, strict=True):
        cumulative += float(weight)
        if threshold < cumulative:
            return item_id
    return item_ids[-1] if item_ids else None
