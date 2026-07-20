"""Exact ``OPT(OS)`` for the small instances used in Table 2."""

from __future__ import annotations

from itertools import chain, combinations, product

import numpy as np

from ..choice import mnl_demand
from ..instance import MarketInstance
from ..policy import Side


def _powerset(items: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(chain.from_iterable(combinations(items, size) for size in range(len(items) + 1)))


def value_customer_static_menus(
    instance: MarketInstance,
    menus: tuple[tuple[int, ...], ...],
) -> float:
    """Evaluate a customer-initiated static menu family exactly."""

    outcomes = [((None,) + menu) for menu in menus]
    total = 0.0
    for choices in product(*outcomes):
        probability = 1.0
        backlog_weights = np.zeros(instance.num_suppliers, dtype=np.float64)
        for customer, choice in enumerate(choices):
            menu = menus[customer]
            denominator = instance.customer_outside[customer] + sum(instance.v[customer, j] for j in menu)
            if choice is None:
                probability *= instance.customer_outside[customer] / denominator
            else:
                supplier = int(choice)
                probability *= instance.v[customer, supplier] / denominator
                backlog_weights[supplier] += instance.w[supplier, customer]
        response_value = sum(
            mnl_demand([backlog_weights[j]], instance.supplier_outside[j])
            for j in range(instance.num_suppliers)
        )
        total += probability * response_value
    return float(total)


def optimize_customer_one_sided_static(instance: MarketInstance) -> float:
    """Enumerate all static menu families using responder-wise expectations.

    For a fixed menu family, initiating choices are independent across agents.
    Linearity of expectation lets each responder's match probability be
    evaluated from only the Bernoulli indicators that it was selected. This is
    exactly equivalent to enumerating every joint initiating-choice outcome,
    but makes the paper's size-four endpoint practical.
    """

    n = instance.num_customers
    m = instance.num_suppliers
    menu_count = 1 << m
    menu_masks = np.arange(menu_count, dtype=np.uint64)
    menu_edges = (
        (menu_masks[:, None] >> np.arange(m, dtype=np.uint64)[None, :]) & 1
    ).astype(np.float64)
    choice_probabilities = np.empty((n, menu_count, m), dtype=np.float64)
    for customer in range(n):
        denominators = (
            instance.customer_outside[customer]
            + menu_edges @ instance.v[customer]
        )
        choice_probabilities[customer] = (
            menu_edges * instance.v[customer][None, :] / denominators[:, None]
        )

    menu_families = np.indices((menu_count,) * n, dtype=np.int32).reshape(n, -1).T
    customer_indices = np.broadcast_to(np.arange(n), menu_families.shape)
    subset_masks = np.arange(1 << n, dtype=np.uint64)
    subset_edges = (
        (subset_masks[:, None] >> np.arange(n, dtype=np.uint64)[None, :]) & 1
    ).astype(bool)
    family_values = np.zeros(len(menu_families), dtype=np.float64)
    for supplier in range(m):
        probabilities = choice_probabilities[
            customer_indices,
            menu_families,
            supplier,
        ]
        backlog_weights = subset_edges @ instance.w[supplier]
        demands = backlog_weights / (
            instance.supplier_outside[supplier] + backlog_weights
        )
        for selected, demand in zip(subset_edges, demands, strict=True):
            outcome_probabilities = np.prod(
                np.where(selected[None, :], probabilities, 1.0 - probabilities),
                axis=1,
            )
            family_values += demand * outcome_probabilities
    return float(np.max(family_values))


def _optimize_customer_one_sided_static_scalar(instance: MarketInstance) -> float:
    """Scalar joint-outcome enumerator retained as an exact test oracle."""

    all_menus = _powerset(tuple(range(instance.num_suppliers)))
    return max(
        value_customer_static_menus(instance, tuple(menus))
        for menus in product(all_menus, repeat=instance.num_customers)
    )


def optimize_one_sided_static(instance: MarketInstance) -> tuple[float, Side, tuple[float, float]]:
    """Return the maximum of the customer- and supplier-initiated optima."""

    customer_value = optimize_customer_one_sided_static(instance)
    supplier_value = optimize_customer_one_sided_static(instance.transposed())
    if customer_value >= supplier_value:
        return customer_value, Side.CUSTOMERS, (customer_value, supplier_value)
    return supplier_value, Side.SUPPLIERS, (customer_value, supplier_value)
