"""Shared solver result and status handling for Appendix F formulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import gurobipy as gp


@dataclass(frozen=True, slots=True)
class SolverResult:
    name: str
    status: str
    runtime_seconds: float
    incumbent: float | None
    best_bound: float | None
    relative_gap: float | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def value(self) -> float | None:
        """Return the incumbent; callers must request ``best_bound`` explicitly."""

        return self.incumbent


def gurobi_status_name(status: int) -> str:
    """Translate a Gurobi status code into a stable lowercase name."""

    names = {
        gp.GRB.LOADED: "loaded",
        gp.GRB.OPTIMAL: "optimal",
        gp.GRB.INFEASIBLE: "infeasible",
        gp.GRB.INF_OR_UNBD: "infeasible_or_unbounded",
        gp.GRB.UNBOUNDED: "unbounded",
        gp.GRB.CUTOFF: "cutoff",
        gp.GRB.ITERATION_LIMIT: "iteration_limit",
        gp.GRB.NODE_LIMIT: "node_limit",
        gp.GRB.TIME_LIMIT: "time_limit",
        gp.GRB.SOLUTION_LIMIT: "solution_limit",
        gp.GRB.INTERRUPTED: "interrupted",
        gp.GRB.NUMERIC: "numeric",
        gp.GRB.SUBOPTIMAL: "suboptimal",
        gp.GRB.USER_OBJ_LIMIT: "user_objective_limit",
    }
    return names.get(status, f"status_{status}")


def result_from_gurobi(name: str, model: gp.Model, metadata: Mapping[str, Any] | None = None) -> SolverResult:
    """Extract incumbent, bound, and gap without assuming a solution exists."""

    has_solution = model.SolCount > 0
    incumbent = float(model.ObjVal) if has_solution else None
    best_bound = float(model.ObjBound) if hasattr(model, "ObjBound") else None
    gap = float(model.MIPGap) if has_solution and hasattr(model, "MIPGap") else None
    return SolverResult(
        name=name,
        status=gurobi_status_name(model.Status),
        runtime_seconds=float(model.Runtime),
        incumbent=incumbent,
        best_bound=best_bound,
        relative_gap=gap,
        metadata=dict(metadata or {}),
    )
