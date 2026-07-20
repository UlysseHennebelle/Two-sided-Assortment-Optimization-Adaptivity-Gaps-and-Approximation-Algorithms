"""Exact dynamic programs for ``OPT(OA)`` and ``OPT(FA)``.

* canonical memoization keeps only unprocessed agents and pending choices;
* already completed matches are rewards on transitions, not part of the key;
* when one side is exhausted, all remaining decisions have a closed-form MNL
  demand value;
* for a candidate next agent, the outside-option continuation is evaluated
  once, each possible choice continuation is evaluated once, and the optimal
  assortment is recovered by the revenue-ordered MNL oracle;
* all backlog alternatives share continuation increment 1, so they are
  aggregated into one pseudo-option whose weight is their total MNL weight.

This state representation avoids explicit powerset enumeration.
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
    raise ValueError("terminal evaluation requires one exhausted side")


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


def _optimize_fully_adaptive_state(instance: MarketInstance) -> DynamicProgramResult:
    """State-object reference implementation of the fully adaptive DP."""

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


def optimize_fully_adaptive(instance: MarketInstance) -> DynamicProgramResult:
    """Compute Appendix A's Bellman optimum with a packed canonical key.

    A customer digit is zero while that customer is unprocessed, one after an
    outside/dead choice, and ``2 + j`` while its pending choice targets an
    unprocessed supplier ``j``.  Supplier digits use the symmetric encoding.
    The two mixed-radix digit vectors are packed into one Python integer before
    memoization.  This represents exactly the same canonical states as
    :class:`AdaptiveState`, but avoids retaining two tuples and a dataclass
    instance for every cached state.  That memory reduction is material at the
    Section 7 endpoint: a balanced size-six instance has 19,886,511 noninitial
    cached states.
    """

    n, m = instance.num_customers, instance.num_suppliers
    customer_base = m + 2
    supplier_base = n + 2
    customer_places = tuple(customer_base**i for i in range(n))
    supplier_places = tuple(supplier_base**j for j in range(m))
    customer_space = customer_base**n

    cache: dict[int, float] = {}
    cache_hits = 0

    def value(code: int) -> float:
        nonlocal cache_hits
        cached = cache.get(code)
        if cached is not None:
            cache_hits += 1
            return cached

        customer_code = code % customer_space
        supplier_code = code // customer_space
        customer_digits = tuple(
            (customer_code // customer_places[i]) % customer_base for i in range(n)
        )
        supplier_digits = tuple(
            (supplier_code // supplier_places[j]) % supplier_base for j in range(m)
        )
        customers = tuple(i for i, digit in enumerate(customer_digits) if digit == 0)
        suppliers = tuple(j for j, digit in enumerate(supplier_digits) if digit == 0)

        if not customers:
            total = 0.0
            for supplier in suppliers:
                pending_digit = supplier + 2
                weight = sum(
                    instance.w[supplier, i]
                    for i, digit in enumerate(customer_digits)
                    if digit == pending_digit
                )
                total += weight / (instance.supplier_outside[supplier] + weight)
            result = float(total)
            cache[code] = result
            return result
        if not suppliers:
            total = 0.0
            for customer in customers:
                pending_digit = customer + 2
                weight = sum(
                    instance.v[customer, j]
                    for j, digit in enumerate(supplier_digits)
                    if digit == pending_digit
                )
                total += weight / (instance.customer_outside[customer] + weight)
            result = float(total)
            cache[code] = result
            return result

        best = 0.0
        for customer in customers:
            pending_digit = customer + 2
            cleared_supplier_code = supplier_code
            backlog_weight = 0.0
            for supplier, digit in enumerate(supplier_digits):
                if digit == pending_digit:
                    cleared_supplier_code += (1 - digit) * supplier_places[supplier]
                    backlog_weight += instance.v[customer, supplier]
            outside_code = (
                customer_code
                + customer_places[customer]
                + customer_space * cleared_supplier_code
            )
            outside_value = value(outside_code)
            options = [MnlOption(-1, 1.0, backlog_weight)]
            for supplier in suppliers:
                choice_code = outside_code + (supplier + 1) * customer_places[customer]
                continuation = value(choice_code)
                options.append(
                    MnlOption(supplier, continuation - outside_value, instance.v[customer, supplier])
                )
            assortment = optimize_mnl(options, instance.customer_outside[customer])
            best = max(best, outside_value + assortment.value)

        for supplier in suppliers:
            pending_digit = supplier + 2
            cleared_customer_code = customer_code
            backlog_weight = 0.0
            for customer, digit in enumerate(customer_digits):
                if digit == pending_digit:
                    cleared_customer_code += (1 - digit) * customer_places[customer]
                    backlog_weight += instance.w[supplier, customer]
            outside_code = (
                cleared_customer_code
                + customer_space * (supplier_code + supplier_places[supplier])
            )
            outside_value = value(outside_code)
            options = [MnlOption(-1, 1.0, backlog_weight)]
            for customer in customers:
                choice_code = (
                    outside_code
                    + customer_space * (customer + 1) * supplier_places[supplier]
                )
                continuation = value(choice_code)
                options.append(
                    MnlOption(customer, continuation - outside_value, instance.w[supplier, customer])
                )
            assortment = optimize_mnl(options, instance.supplier_outside[supplier])
            best = max(best, outside_value + assortment.value)
        cache[code] = best
        return best

    optimum = value(0)
    states = len(cache)
    hits = cache_hits
    cache.clear()
    return DynamicProgramResult("OPT_FA", float(optimum), states, hits)
