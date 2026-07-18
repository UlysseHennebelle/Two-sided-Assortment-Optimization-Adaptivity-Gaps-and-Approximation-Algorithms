"""Algorithm 1: Arbitrary Order Greedy from Section 4.1.

For each initiating agent, the continuation revenue of a responding agent is
the marginal increase in its MNL demand after adding the initiator to its
backlog. The revenue-ordered MNL oracle then selects the assortment without
enumerating subsets. This is the core legacy optimization retained by ALG(OS),
ALG(OA), ALG(FA), and the optimized exact dynamic programs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..choice import mnl_demand, sample_mnl
from ..instance import MarketInstance
from ..oracle import MnlOption, optimize_mnl


@dataclass(frozen=True, slots=True)
class GreedyTrace:
    value: float
    order: tuple[int, ...]
    assortments: tuple[tuple[int, ...], ...]
    choices: tuple[int | None, ...]


def marginal_responding_demand(instance: MarketInstance, supplier: int, customer: int, backlog_weight: float) -> float:
    """Return ``f_j(C union {i}) - f_j(C)`` under supplier ``j``'s MNL."""

    outside = instance.supplier_outside[supplier]
    before = backlog_weight / (outside + backlog_weight)
    after_weight = backlog_weight + instance.w[supplier, customer]
    after = after_weight / (outside + after_weight)
    return float(after - before)


def arbitrary_order_greedy_customers(
    instance: MarketInstance,
    rng: np.random.Generator,
    order: tuple[int, ...] | None = None,
) -> GreedyTrace:
    """Run Algorithm 1 with customers initiating and observed choices."""

    if order is None:
        order = tuple(int(item) for item in rng.permutation(instance.num_customers))
    if tuple(sorted(order)) != tuple(range(instance.num_customers)):
        raise ValueError("order must be a permutation of all customers")
    backlog_weights = np.zeros(instance.num_suppliers, dtype=np.float64)
    assortments: list[tuple[int, ...]] = [()] * instance.num_customers
    choices: list[int | None] = [None] * instance.num_customers
    for customer in order:
        options = [
            MnlOption(
                supplier,
                marginal_responding_demand(instance, supplier, customer, backlog_weights[supplier]),
                instance.v[customer, supplier],
            )
            for supplier in range(instance.num_suppliers)
        ]
        solution = optimize_mnl(options, instance.customer_outside[customer])
        assortment = tuple(int(supplier) for supplier in solution.item_ids)
        choice = sample_mnl(
            assortment,
            [instance.v[customer, supplier] for supplier in assortment],
            rng,
            instance.customer_outside[customer],
        )
        assortments[customer] = assortment
        choices[customer] = None if choice is None else int(choice)
        if choice is not None:
            supplier = int(choice)
            backlog_weights[supplier] += instance.w[supplier, customer]
    value = sum(
        backlog_weights[supplier] / (instance.supplier_outside[supplier] + backlog_weights[supplier])
        for supplier in range(instance.num_suppliers)
    )
    return GreedyTrace(float(value), order, tuple(assortments), tuple(choices))
