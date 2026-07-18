"""Linear Problem (40), the ``UB(FA)`` benchmark from Appendix F.1.3."""

from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB

from ..instance import MarketInstance
from .solver import SolverResult, result_from_gurobi


def upper_bound_fully_adaptive(
    instance: MarketInstance,
    threads: int = 1,
    output: bool = False,
    seed: int = 0,
) -> SolverResult:
    """Solve Problem (40) directly with one matrix of match probabilities."""

    normalized = instance.normalized_outside()
    n, m = normalized.num_customers, normalized.num_suppliers
    model = gp.Model("ub_fully_adaptive_problem_40")
    model.Params.OutputFlag = int(output)
    model.Params.Threads = int(threads)
    solver_seed = int(seed % 2_000_000_000)
    model.Params.Seed = solver_seed
    x = model.addVars(n, m, lb=0.0, name="x")
    for i in range(n):
        row_sum = gp.quicksum(x[i, ell] for ell in range(m))
        for j in range(m):
            model.addConstr(x[i, j] <= normalized.v[i, j] * (1.0 - row_sum))
    for j in range(m):
        column_sum = gp.quicksum(x[k, j] for k in range(n))
        for i in range(n):
            model.addConstr(x[i, j] <= normalized.w[j, i] * (1.0 - column_sum))
    model.setObjective(gp.quicksum(x[i, j] for i in range(n) for j in range(m)), GRB.MAXIMIZE)
    model.optimize()
    return result_from_gurobi("UB_FA", model, {"formulation": 40, "solver_seed": solver_seed})
