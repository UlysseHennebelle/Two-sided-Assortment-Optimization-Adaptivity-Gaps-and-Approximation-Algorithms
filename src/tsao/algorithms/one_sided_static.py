"""Section 7's simulated-greedy ``ALG(OS)`` benchmark.

Each replication simulates Algorithm 1 to construct a complete static menu
family and then evaluates that fixed policy once. Choices used during
construction update only the marginal-demand weights; evaluation never adapts
the menus. This preserves the paper's simulate-versus-observe distinction.
"""

from __future__ import annotations

import numpy as np

from ..choice import sample_mnl
from ..instance import MarketInstance
from ..policy import AlgorithmResult, Side
from .arbitrary_order_greedy import revenue_ordered_customer_assortment


def construct_static_menus_customers(
    instance: MarketInstance,
    rng: np.random.Generator,
) -> tuple[tuple[int, ...], ...]:
    """Simulate Algorithm 1 once and retain only its customer assortments."""

    order = tuple(int(item) for item in rng.permutation(instance.num_customers))
    backlog_weights = np.zeros(instance.num_suppliers, dtype=np.float64)
    menus: list[tuple[int, ...]] = [()] * instance.num_customers
    for customer in order:
        menu = revenue_ordered_customer_assortment(instance, customer, backlog_weights)
        menus[customer] = menu
        choice = sample_mnl(
            menu,
            [instance.v[customer, supplier] for supplier in menu],
            rng,
            instance.customer_outside[customer],
        )
        if choice is not None:
            supplier = int(choice)
            backlog_weights[supplier] += instance.w[supplier, customer]
    return tuple(menus)


def simulate_static_menus_customers(
    instance: MarketInstance,
    menus: tuple[tuple[int, ...], ...],
    rng: np.random.Generator,
) -> float:
    """Simulate initiating choices and return exact conditional response value."""

    if len(menus) != instance.num_customers:
        raise ValueError("one menu is required per customer")
    backlog_weights = np.zeros(instance.num_suppliers, dtype=np.float64)
    for customer, menu in enumerate(menus):
        choice = sample_mnl(
            menu,
            [instance.v[customer, supplier] for supplier in menu],
            rng,
            instance.customer_outside[customer],
        )
        if choice is not None:
            supplier = int(choice)
            backlog_weights[supplier] += instance.w[supplier, customer]
    return float(
        sum(
            backlog_weights[supplier] / (instance.supplier_outside[supplier] + backlog_weights[supplier])
            for supplier in range(instance.num_suppliers)
        )
    )


def _evaluate_side(
    instance: MarketInstance,
    replications: int,
    seed_sequence: np.random.SeedSequence,
) -> tuple[float, tuple[float, ...], tuple[int, ...]]:
    replication_seeds = seed_sequence.spawn(replications)
    values: list[float] = []
    seed_values: list[int] = []
    for replication_seed in replication_seeds:
        seed_values.append(int(replication_seed.generate_state(1, dtype=np.uint64)[0]))
        rng = np.random.default_rng(replication_seed)
        menus = construct_static_menus_customers(instance, rng)
        values.append(simulate_static_menus_customers(instance, menus, rng))
    return float(np.mean(values)), tuple(values), tuple(seed_values)


def one_sided_static_algorithm(
    instance: MarketInstance,
    replications_per_side: int = 50,
    seed: int = 0,
) -> AlgorithmResult:
    """Run independent end-to-end static-greedy replications from both sides."""

    if replications_per_side <= 0:
        raise ValueError("replications_per_side must be positive")
    customer_seed, supplier_seed = np.random.SeedSequence(seed).spawn(2)
    customer_mean, customer_values, customer_replication_seeds = _evaluate_side(
        instance, replications_per_side, customer_seed
    )
    supplier_mean, supplier_values, supplier_replication_seeds = _evaluate_side(
        instance.transposed(), replications_per_side, supplier_seed
    )
    side = Side.CUSTOMERS if customer_mean >= supplier_mean else Side.SUPPLIERS
    return AlgorithmResult(
        name="ALG_OS",
        value=max(customer_mean, supplier_mean),
        initiating_side=side,
        replications=customer_values + supplier_values,
        metadata={
            "end_to_end_replications_per_side": replications_per_side,
            "evaluations_per_constructed_policy": 1,
            "customer_mean": customer_mean,
            "supplier_mean": supplier_mean,
            "replication_seeds": customer_replication_seeds + supplier_replication_seeds,
        },
    )
