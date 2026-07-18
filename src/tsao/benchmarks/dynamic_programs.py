"""Optimized exact dynamic programs for ``OPT(OA)`` and ``OPT(FA)``.

These routines replicate and clarify the important optimizations from legacy
cell 4:

* canonical memoization keeps only unprocessed agents and pending choices;
* already completed matches are rewards on transitions, not part of the key;
* when one side is exhausted, all remaining decisions have a closed-form MNL
  demand value;
* for a candidate next agent, the outside-option continuation is evaluated
  once, each possible choice continuation is evaluated once, and the optimal
  assortment is recovered by the revenue-ordered MNL oracle;
* all backlog alternatives share continuation increment 1, so they are
  aggregated into one pseudo-option whose weight is their total MNL weight.

This removes explicit powerset enumeration while preserving the mathematics of
the optimized notebook DP.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from ..instance import MarketInstance
from ..oracle import MnlOption, optimize_mnl
from ..policy import Side
from ..state import AdaptiveState


@dataclass(frozen=True, slots=True)
class DynamicProgramResult:
    name: str
    value: float
    states: int
    cache_hits: int
    side: Side | None = None
    side_values: tuple[float, float] | None = None


def _terminal_value(instance: MarketInstance, state: AdaptiveState) -> float:
    """Evaluate remaining agents when the opposite side is exhausted."""

    if not state.customers():
        total = 0.0
        for supplier in state.suppliers():
            weight = sum(instance.w[supplier, i] for i in state.supplier_backlog(supplier))
            total += weight / (instance.supplier_outside[supplier] + weight)
        return float(total)
    if not state.suppliers():
        total = 0.0
        for customer in state.customers():
            weight = sum(instance.v[customer, j] for j in state.customer_backlog(customer))
            total += weight / (instance.customer_outside[customer] + weight)
        return float(total)
    raise ValueError("terminal shortcut requires one exhausted side")


def _one_sided_customer_value(instance: MarketInstance) -> tuple[float, int, int]:
    initial = AdaptiveState.initial(instance)

    @lru_cache(maxsize=None)
    def value(state: AdaptiveState) -> float:
        customers = state.customers()
        if not customers:
            return _terminal_value(instance, state)
        best = 0.0
        for customer in customers:
            outside_state, _ = state.process_customer(customer, None)
            outside_value = value(outside_state)
            options: list[MnlOption] = []
            for supplier in state.suppliers():
                choice_state, reward = state.process_customer(customer, supplier)
                continuation = reward + value(choice_state)
                options.append(
                    MnlOption(supplier, continuation - outside_value, instance.v[customer, supplier])
                )
            assortment = optimize_mnl(options, instance.customer_outside[customer])
            best = max(best, outside_value + assortment.value)
        return best

    optimum = value(initial)
    info = value.cache_info()
    return float(optimum), int(info.currsize), int(info.hits)


def optimize_one_sided_adaptive(instance: MarketInstance) -> DynamicProgramResult:
    """Compute both initiating orientations and return ``OPT(OA)``."""

    customer_value, customer_states, customer_hits = _one_sided_customer_value(instance)
    supplier_value, supplier_states, supplier_hits = _one_sided_customer_value(instance.transposed())
    side = Side.CUSTOMERS if customer_value >= supplier_value else Side.SUPPLIERS
    return DynamicProgramResult(
        "OPT_OA",
        max(customer_value, supplier_value),
        customer_states + supplier_states,
        customer_hits + supplier_hits,
        side,
        (customer_value, supplier_value),
    )


def optimize_fully_adaptive(instance: MarketInstance) -> DynamicProgramResult:
    """Compute Appendix A's Bellman optimum using the MNL state compression."""

    initial = AdaptiveState.initial(instance)

    @lru_cache(maxsize=None)
    def value(state: AdaptiveState) -> float:
        customers = state.customers()
        suppliers = state.suppliers()
        if not customers or not suppliers:
            return _terminal_value(instance, state)
        best = 0.0
        for customer in customers:
            outside_state, _ = state.process_customer(customer, None)
            outside_value = value(outside_state)
            backlog_weight = sum(instance.v[customer, j] for j in state.customer_backlog(customer))
            options = [MnlOption(("backlog", customer), 1.0, backlog_weight)]
            for supplier in suppliers:
                choice_state, reward = state.process_customer(customer, supplier)
                continuation = reward + value(choice_state)
                options.append(
                    MnlOption(("supplier", supplier), continuation - outside_value, instance.v[customer, supplier])
                )
            assortment = optimize_mnl(options, instance.customer_outside[customer])
            best = max(best, outside_value + assortment.value)
        for supplier in suppliers:
            outside_state, _ = state.process_supplier(supplier, None)
            outside_value = value(outside_state)
            backlog_weight = sum(instance.w[supplier, i] for i in state.supplier_backlog(supplier))
            options = [MnlOption(("backlog", supplier), 1.0, backlog_weight)]
            for customer in customers:
                choice_state, reward = state.process_supplier(supplier, customer)
                continuation = reward + value(choice_state)
                options.append(
                    MnlOption(("customer", customer), continuation - outside_value, instance.w[supplier, customer])
                )
            assortment = optimize_mnl(options, instance.supplier_outside[supplier])
            best = max(best, outside_value + assortment.value)
        return best

    optimum = value(initial)
    info = value.cache_info()
    return DynamicProgramResult("OPT_FA", float(optimum), int(info.currsize), int(info.hits))
