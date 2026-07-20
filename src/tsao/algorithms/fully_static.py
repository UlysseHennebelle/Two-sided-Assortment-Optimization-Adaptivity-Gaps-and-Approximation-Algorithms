"""Fully static approximation algorithm from Section 5.

Edges are partitioned into low/low, customer-high, and supplier-high regions.
The low/low region is evaluated by independently rounding its relaxation. The
two high-value regions use the marginal-demand construction. The algorithm
returns the largest expected matching value among the resulting candidates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from numpy.typing import NDArray
from scipy import sparse

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


def _low_value_relaxation_probabilities(
    instance: MarketInstance,
    output: bool = False,
    solver_method: int = 2,
) -> NDArray[np.float64]:
    """Solve the factored low/low LP through Gurobi's matrix API."""

    n, m = instance.num_customers, instance.num_suppliers
    active = (instance.v > 0.0) & (instance.w.T > 0.0)
    rows, columns = np.nonzero(active)
    probabilities = np.zeros((n, m), dtype=np.float64)
    if len(rows) == 0:
        return probabilities

    edge_indices = np.arange(len(rows))
    row_weights = sparse.csr_matrix(
        (instance.w[columns, rows], (rows, edge_indices)),
        shape=(n, len(rows)),
    )
    column_weights = sparse.csr_matrix(
        (instance.v[rows, columns], (columns, edge_indices)),
        shape=(m, len(rows)),
    )
    model = gp.Model("alg_fs_low_value")
    model.Params.OutputFlag = int(output)
    model.Params.Threads = 1
    model.Params.Method = solver_method
    y = model.addMVar(len(rows), lb=0.0, name="y")
    row_slack = model.addMVar(n, lb=0.0, name="row_slack")
    column_slack = model.addMVar(m, lb=0.0, name="column_slack")
    model.addConstr(row_slack + row_weights @ y <= 1.0)
    model.addConstr(column_slack + column_weights @ y <= 1.0)
    model.addConstr(y <= row_slack[rows])
    model.addConstr(y <= column_slack[columns])
    objective = instance.v[rows, columns] * instance.w[columns, rows]
    model.setObjective(objective @ y, GRB.MAXIMIZE)
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Low-value relaxation failed with Gurobi status {model.Status}")
    probabilities[rows, columns] = np.asarray(y.X, dtype=np.float64)
    return np.clip(probabilities, 0.0, 1.0)


def low_value_candidate(
    instance: MarketInstance,
    replications: int,
    rng: np.random.Generator,
    output: bool = False,
) -> StaticCandidate:
    """Solve the low/low relaxation and estimate independent rounding."""

    if replications <= 0:
        raise ValueError("replications must be positive")
    n, m = instance.num_customers, instance.num_suppliers
    probabilities = _low_value_relaxation_probabilities(instance, output)
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
    return StaticCandidate("low_low", float(np.mean(values)), best_edges)


def high_value_greedy_candidate(instance: MarketInstance, name: str = "high_value") -> StaticCandidate:
    """Construct a high-value edge set by marginal responding-side demand.

    Initiators are processed in index order and assigned to the available
    responder with the largest positive marginal demand increase.
    """

    n, m = instance.num_customers, instance.num_suppliers
    supplier_weights = np.zeros(m, dtype=np.float64)
    edges = np.zeros((n, m), dtype=bool)
    for i in range(n):
        available = instance.v[i] > 0.0
        if not np.any(available):
            continue
        outside = instance.supplier_outside
        before = 1.0 - outside / (outside + supplier_weights)
        after_weights = supplier_weights + instance.w[:, i]
        after = 1.0 - outside / (outside + after_weights)
        marginals = np.where(available, after - before, -np.inf)
        best_supplier = int(np.argmax(marginals))
        if marginals[best_supplier] <= 0.0:
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
    """Evaluate the three Section 5 candidates and return their maximum."""

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
            "low_value_solver_threads": 1,
            "low_value_solver_method": "barrier",
        },
    )
