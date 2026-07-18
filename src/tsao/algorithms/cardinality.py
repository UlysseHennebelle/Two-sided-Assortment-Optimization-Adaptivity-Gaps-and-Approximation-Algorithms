"""Small-instance cardinality-constrained extensions from Section 6."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from ..instance import MarketInstance
from .fully_static import fully_static_value


def edge_set_respects_capacities(instance: MarketInstance, edges: NDArray[np.bool_]) -> bool:
    """Check the two-way cardinality constraints for a reciprocal edge set."""

    selected = np.asarray(edges, dtype=bool)
    for i, capacity in enumerate(instance.customer_capacities):
        if capacity is not None and int(selected[i].sum()) > capacity:
            return False
    for j, capacity in enumerate(instance.supplier_capacities):
        if capacity is not None and int(selected[:, j].sum()) > capacity:
            return False
    return True


def enumerate_cardinality_fully_static(instance: MarketInstance) -> float:
    """Exact validation helper for Section 6 on tiny instances only."""

    edge_ids = [(i, j) for i in range(instance.num_customers) for j in range(instance.num_suppliers)]
    best = 0.0
    for size in range(len(edge_ids) + 1):
        for subset in combinations(edge_ids, size):
            edges = np.zeros(instance.v.shape, dtype=bool)
            for i, j in subset:
                edges[i, j] = True
            if edge_set_respects_capacities(instance, edges):
                best = max(best, fully_static_value(instance, edges))
    return best
