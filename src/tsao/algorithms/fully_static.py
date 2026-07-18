"""Corrected empirical implementation of ``ALG(FS)`` from Section 5.

The implementation follows legacy cell 8 because that is the algorithm used to
produce the paper data:

1. split edges using the confirmed threshold ``alpha=(sqrt(5)-1)/2``;
2. solve the low/low LP and independently round its edge variables;
3. run the legacy marginal-demand greedy routine on customer-high edges;
4. transpose supplier-high edges once and run the same greedy routine;
5. return the best of the three values.

This is intentionally the empirical high-value greedy routine rather than the
continuous-greedy method in the proof of Lemma 5.4. The revised paper must name
that computational choice explicitly. The legacy code transposed the
supplier-high subinstance twice before evaluating it; this module corrects that
error by using one immutable transposition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from numpy.typing import NDArray

from ..choice import mnl_demand
from ..instance import MarketInstance
from ..policy import AlgorithmResult

EMPIRICAL_ALPHA = (math.sqrt(5.0) - 1.0) / 2.0


@dataclass(frozen=True, slots=True)
class StaticCandidate:
    name: str
    value: float
    edges: NDArray[np.bool_] | None


def fully_static_value(instance: MarketInstance, edges: NDArray[np.bool_]) -> float:
    """Evaluate Problem (11) for a reciprocal binary edge set."""

    selected = np.asarray(edges, dtype=bool)
    if selected.shape != instance.v.shape:
        raise ValueError(f"edges must have shape {instance.v.shape}")
    customer_denominators = instance.customer_outside + np.sum(instance.v * selected, axis=1)
    supplier_denominators = instance.supplier_outside + np.sum(instance.w * selected.T, axis=1)
    terms = (instance.v / customer_denominators[:, None]) * (instance.w.T / supplier_denominators[None, :])
    return float(np.sum(terms * selected))


def _masked_instance(instance: MarketInstance, mask: NDArray[np.bool_]) -> MarketInstance:
    return MarketInstance(
        np.where(mask, instance.v, 0.0),
        np.where(mask.T, instance.w, 0.0),
        customer_outside=instance.customer_outside,
        supplier_outside=instance.supplier_outside,
        metadata=instance.metadata,
    )


def low_value_candidate(
    instance: MarketInstance,
    replications: int,
    rng: np.random.Generator,
    output: bool = False,
) -> StaticCandidate:
    """Solve the legacy low/low relaxation and estimate independent rounding.

    The LP uses explicit row and column slack variables. This is the factored
    equivalent of the notebook's extra last row/column in ``y``.
    """

    if replications <= 0:
        raise ValueError("replications must be positive")
    n, m = instance.num_customers, instance.num_suppliers
    model = gp.Model("alg_fs_low_value")
    model.Params.OutputFlag = int(output)
    y = model.addVars(n, m, lb=0.0, name="y")
    row_slack = model.addVars(n, lb=0.0, name="row_slack")
    column_slack = model.addVars(m, lb=0.0, name="column_slack")
    for i in range(n):
        model.addConstr(
            row_slack[i] + gp.quicksum(instance.w[j, i] * y[i, j] for j in range(m)) <= 1.0
        )
    for j in range(m):
        model.addConstr(
            column_slack[j] + gp.quicksum(instance.v[i, j] * y[i, j] for i in range(n)) <= 1.0
        )
    for i in range(n):
        for j in range(m):
            model.addConstr(y[i, j] <= row_slack[i])
            model.addConstr(y[i, j] <= column_slack[j])
    model.setObjective(
        gp.quicksum(instance.v[i, j] * instance.w[j, i] * y[i, j] for i in range(n) for j in range(m)),
        GRB.MAXIMIZE,
    )
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Low-value relaxation failed with Gurobi status {model.Status}")
    probabilities = np.array([[y[i, j].X for j in range(m)] for i in range(n)], dtype=np.float64)
    probabilities = np.clip(probabilities, 0.0, 1.0)
    values = []
    best_edges: NDArray[np.bool_] | None = None
    best_value = -math.inf
    for _ in range(replications):
        edges = rng.random((n, m)) <= probabilities
        value = fully_static_value(instance, edges)
        values.append(value)
        if value > best_value:
            best_value = value
            best_edges = edges
    # The notebook reports the Monte Carlo mean, not the best realized rounding.
    return StaticCandidate("low_low", float(np.mean(values)), best_edges)


def high_value_greedy_candidate(instance: MarketInstance, name: str = "high_value") -> StaticCandidate:
    """Run the confirmed empirical marginal-demand greedy routine.

    Customers are processed in index order and assigned to the supplier with
    largest marginal demand increase. As in the notebook, customer weights only
    determine whether an edge is available; supplier weights determine the
    greedy marginal. A legacy bug selected supplier zero when a customer had no
    available edge. The corrected routine skips that customer instead.
    """

    n, m = instance.num_customers, instance.num_suppliers
    supplier_weights = np.zeros(m, dtype=np.float64)
    edges = np.zeros((n, m), dtype=bool)
    for i in range(n):
        best_supplier: int | None = None
        best_marginal = 0.0
        for j in range(m):
            if instance.v[i, j] <= 0.0:
                continue
            before = mnl_demand([supplier_weights[j]], instance.supplier_outside[j])
            after = mnl_demand([supplier_weights[j] + instance.w[j, i]], instance.supplier_outside[j])
            marginal = after - before
            if marginal > best_marginal:
                best_marginal = marginal
                best_supplier = j
        if best_supplier is None:
            continue
        edges[i, best_supplier] = True
        supplier_weights[best_supplier] += instance.w[best_supplier, i]
    return StaticCandidate(name, fully_static_value(instance, edges), edges)


def fully_static_algorithm(
    instance: MarketInstance,
    replications: int = 50,
    seed: int = 0,
    alpha: float = EMPIRICAL_ALPHA,
    output: bool = False,
) -> AlgorithmResult:
    """Evaluate the three legacy Section 5 candidates and return their maximum."""

    if alpha <= 0:
        raise ValueError("alpha must be positive")
    working_instance = instance.normalized_outside()
    low_low_mask = (working_instance.v < alpha) & (working_instance.w.T < alpha)
    customer_high_mask = working_instance.v >= alpha
    supplier_high_mask = (working_instance.v < alpha) & (working_instance.w.T >= alpha)
    rng = np.random.default_rng(seed)
    low_low = low_value_candidate(_masked_instance(working_instance, low_low_mask), replications, rng, output)
    customer_high = high_value_greedy_candidate(
        _masked_instance(working_instance, customer_high_mask), "customer_high"
    )
    supplier_high_instance = _masked_instance(working_instance, supplier_high_mask).transposed()
    supplier_high = high_value_greedy_candidate(supplier_high_instance, "supplier_high")
    candidates = (low_low, customer_high, supplier_high)
    best = max(candidates, key=lambda candidate: candidate.value)
    return AlgorithmResult(
        name="ALG_FS",
        value=best.value,
        metadata={
            "alpha": alpha,
            "candidate_values": {candidate.name: candidate.value for candidate in candidates},
            "selected_candidate": best.name,
            "rounding_replications": replications,
            "corrected_single_supplier_high_transposition": True,
        },
    )
