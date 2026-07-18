"""Concave Problem (39), the ``UB(OA)`` benchmark from Appendix F.1.2."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp

from ..instance import MarketInstance
from ..policy import Side


@dataclass(frozen=True, slots=True)
class OaUpperBoundResult:
    value: float
    side: Side
    customer_value: float
    supplier_value: float
    customer_status: str
    supplier_status: str
    customer_solver: str | None
    supplier_solver: str | None


def _customer_upper_bound(instance: MarketInstance, solver: str | None = None) -> tuple[float, str, str | None]:
    normalized = instance.normalized_outside()
    n, m = normalized.num_customers, normalized.num_suppliers
    y = cp.Variable((n, m), nonneg=True)
    constraints = []
    for i in range(n):
        weighted_sum = cp.sum(cp.multiply(normalized.v[i, :], y[i, :]))
        for j in range(m):
            constraints.append(y[i, j] + weighted_sum <= 1.0)
    z = [cp.sum(cp.multiply(normalized.v[:, j] * normalized.w[j, :], y[:, j])) for j in range(m)]
    objective = cp.Maximize(cp.sum([1.0 - cp.inv_pos(1.0 + item) for item in z]))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver)
    if problem.value is None:
        raise RuntimeError(f"Problem (39) failed with status {problem.status}")
    solver_name = problem.solver_stats.solver_name if problem.solver_stats is not None else None
    return float(problem.value), str(problem.status), solver_name


def upper_bound_one_sided_adaptive(
    instance: MarketInstance,
    solver: str | None = None,
) -> OaUpperBoundResult:
    """Solve both orientations without mutating the input instance."""

    customer_value, customer_status, customer_solver = _customer_upper_bound(instance, solver)
    supplier_value, supplier_status, supplier_solver = _customer_upper_bound(instance.transposed(), solver)
    side = Side.CUSTOMERS if customer_value >= supplier_value else Side.SUPPLIERS
    return OaUpperBoundResult(
        max(customer_value, supplier_value),
        side,
        customer_value,
        supplier_value,
        customer_status,
        supplier_status,
        customer_solver,
        supplier_solver,
    )
