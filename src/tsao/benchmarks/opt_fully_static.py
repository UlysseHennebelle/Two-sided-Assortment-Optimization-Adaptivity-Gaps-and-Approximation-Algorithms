"""Bilinear Problem (38), the Section 7 ``OPT(FS)`` benchmark."""

from __future__ import annotations

from itertools import combinations

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from ..algorithms.fully_static import fully_static_value
from ..instance import MarketInstance
from .solver import SolverResult, result_from_gurobi


def optimize_fully_static(
    instance: MarketInstance,
    mip_gap: float = 0.01,
    time_limit_seconds: float = 7200.0,
    threads: int = 1,
    output: bool = False,
    seed: int = 0,
) -> SolverResult:
    """Solve the continuous bilinear reformulation of Problem (11).

    The legacy function returned only Gurobi's objective bound. The corrected
    API retains both incumbent and bound so reporting can distinguish a feasible
    solution from an upper bound when the time or gap limit is reached.
    """

    normalized = instance.normalized_outside()
    n, m = normalized.num_customers, normalized.num_suppliers
    model = gp.Model("opt_fully_static_problem_38")
    model.Params.OutputFlag = int(output)
    model.Params.NonConvex = 2
    model.Params.MIPGap = float(mip_gap)
    model.Params.TimeLimit = float(time_limit_seconds)
    model.Params.Threads = int(threads)
    solver_seed = int(seed % 2_000_000_000)
    model.Params.Seed = solver_seed
    y = model.addVars(n, m, lb=0.0, name="y")
    z = model.addVars(m, n, lb=0.0, name="z")
    for i in range(n):
        row_sum = gp.quicksum(y[i, ell] for ell in range(m))
        for j in range(m):
            model.addConstr(y[i, j] <= normalized.v[i, j] * (1.0 - row_sum))
    for j in range(m):
        column_sum = gp.quicksum(z[j, k] for k in range(n))
        for i in range(n):
            model.addConstr(z[j, i] <= normalized.w[j, i] * (1.0 - column_sum))
    model.setObjective(gp.quicksum(y[i, j] * z[j, i] for i in range(n) for j in range(m)), GRB.MAXIMIZE)
    model.optimize()
    return result_from_gurobi(
        "OPT_FS",
        model,
        {
            "formulation": 38,
            "mip_gap_target": mip_gap,
            "time_limit_seconds": time_limit_seconds,
            "solver_seed": solver_seed,
            "legacy_reported_field": "best_bound",
        },
    )


def enumerate_fully_static(instance: MarketInstance) -> float:
    """Enumerate reciprocal edge sets for tiny exact validation cases."""

    edges = [(i, j) for i in range(instance.num_customers) for j in range(instance.num_suppliers)]
    best = 0.0
    for size in range(len(edges) + 1):
        for subset in combinations(edges, size):
            selected = np.zeros(instance.v.shape, dtype=bool)
            for i, j in subset:
                selected[i, j] = True
            best = max(best, fully_static_value(instance, selected))
    return best
