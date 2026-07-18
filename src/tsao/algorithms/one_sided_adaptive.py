"""Empirical ``ALG(OA)`` based on Algorithm 1 and side comparison.

The legacy experiments run the adaptive greedy algorithm independently from
both initiating sides and report the larger sample mean. The confirmed protocol
uses 50 replications per side; tests pass smaller values explicitly.
"""

from __future__ import annotations

import numpy as np

from ..instance import MarketInstance
from ..policy import AlgorithmResult, Side
from .arbitrary_order_greedy import arbitrary_order_greedy_customers


def one_sided_adaptive_algorithm(
    instance: MarketInstance,
    replications_per_side: int = 50,
    seed: int = 0,
) -> AlgorithmResult:
    """Estimate Algorithm 1 from both sides and select the larger mean."""

    if replications_per_side <= 0:
        raise ValueError("replications_per_side must be positive")
    children = np.random.SeedSequence(seed).spawn(2 * replications_per_side)
    replication_seeds = tuple(int(child.generate_state(1, dtype=np.uint64)[0]) for child in children)
    customer_values = tuple(
        arbitrary_order_greedy_customers(instance, np.random.default_rng(children[k])).value
        for k in range(replications_per_side)
    )
    transposed = instance.transposed()
    supplier_values = tuple(
        arbitrary_order_greedy_customers(
            transposed,
            np.random.default_rng(children[replications_per_side + k]),
        ).value
        for k in range(replications_per_side)
    )
    customer_mean = float(np.mean(customer_values))
    supplier_mean = float(np.mean(supplier_values))
    side = Side.CUSTOMERS if customer_mean >= supplier_mean else Side.SUPPLIERS
    return AlgorithmResult(
        name="ALG_OA",
        value=max(customer_mean, supplier_mean),
        initiating_side=side,
        replications=customer_values + supplier_values,
        metadata={
            "replications_per_side": replications_per_side,
            "customer_mean": customer_mean,
            "supplier_mean": supplier_mean,
            "side_selection": "larger_sample_mean",
            "replication_seeds": replication_seeds,
        },
    )
