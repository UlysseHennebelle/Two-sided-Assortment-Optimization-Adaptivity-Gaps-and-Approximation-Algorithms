"""Section 7's simulated-greedy ``ALG(OS)`` benchmark.

Algorithm 1 is simulated to construct a complete static menu family. Choices
used during construction update only the marginal-demand weights; the final
menus are then evaluated in independent simulations without adapting them.
This preserves the notebook's important simulate-versus-observe distinction.
"""

from __future__ import annotations

import numpy as np

from ..choice import sample_mnl
from ..instance import MarketInstance
from ..oracle import MnlOption, optimize_mnl
from ..policy import AlgorithmResult, Side
from .arbitrary_order_greedy import marginal_responding_demand


def construct_static_menus_customers(
    instance: MarketInstance,
    rng: np.random.Generator,
) -> tuple[tuple[int, ...], ...]:
    """Simulate Algorithm 1 once and retain only its customer assortments."""

    order = tuple(int(item) for item in rng.permutation(instance.num_customers))
    backlog_weights = np.zeros(instance.num_suppliers, dtype=np.float64)
    menus: list[tuple[int, ...]] = [()] * instance.num_customers
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
        menu = tuple(int(supplier) for supplier in solution.item_ids)
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
    construction_replications: int,
    evaluation_replications: int,
    seed_sequence: np.random.SeedSequence,
) -> tuple[float, tuple[float, ...], tuple[int, ...], tuple[int, ...]]:
    construction_seeds = seed_sequence.spawn(construction_replications)
    values: list[float] = []
    construction_seed_values: list[int] = []
    evaluation_seed_values: list[int] = []
    for construction_seed in construction_seeds:
        construction_seed_values.append(int(construction_seed.generate_state(1, dtype=np.uint64)[0]))
        construction_rng = np.random.default_rng(construction_seed)
        menus = construct_static_menus_customers(instance, construction_rng)
        for evaluation_seed in construction_seed.spawn(evaluation_replications):
            evaluation_seed_values.append(int(evaluation_seed.generate_state(1, dtype=np.uint64)[0]))
            values.append(simulate_static_menus_customers(instance, menus, np.random.default_rng(evaluation_seed)))
    return float(np.mean(values)), tuple(values), tuple(construction_seed_values), tuple(evaluation_seed_values)


def one_sided_static_algorithm(
    instance: MarketInstance,
    construction_replications: int = 50,
    evaluation_replications: int = 50,
    seed: int = 0,
) -> AlgorithmResult:
    """Construct and evaluate simulated-greedy static policies from both sides."""

    if construction_replications <= 0 or evaluation_replications <= 0:
        raise ValueError("replication counts must be positive")
    customer_seed, supplier_seed = np.random.SeedSequence(seed).spawn(2)
    customer_mean, customer_values, customer_construction_seeds, customer_evaluation_seeds = _evaluate_side(
        instance, construction_replications, evaluation_replications, customer_seed
    )
    supplier_mean, supplier_values, supplier_construction_seeds, supplier_evaluation_seeds = _evaluate_side(
        instance.transposed(), construction_replications, evaluation_replications, supplier_seed
    )
    side = Side.CUSTOMERS if customer_mean >= supplier_mean else Side.SUPPLIERS
    return AlgorithmResult(
        name="ALG_OS",
        value=max(customer_mean, supplier_mean),
        initiating_side=side,
        replications=customer_values + supplier_values,
        metadata={
            "construction_replications_per_side": construction_replications,
            "evaluation_replications_per_construction": evaluation_replications,
            "customer_mean": customer_mean,
            "supplier_mean": supplier_mean,
            "construction_seeds": customer_construction_seeds + supplier_construction_seeds,
            "replication_seeds": customer_evaluation_seeds + supplier_evaluation_seeds,
        },
    )
