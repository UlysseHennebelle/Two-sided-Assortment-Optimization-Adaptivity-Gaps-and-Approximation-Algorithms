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
    """Enumerate all initiating-side menu families; intended for ``n,m <= 3``."""

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
