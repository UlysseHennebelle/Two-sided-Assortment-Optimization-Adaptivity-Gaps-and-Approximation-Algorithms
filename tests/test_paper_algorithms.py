import gurobipy as gp
import numpy as np
from gurobipy import GRB

from tsao.algorithms.fully_adaptive import (
    fully_adaptive_experiment_algorithm,
    fully_adaptive_theorem_algorithm,
)
from tsao.algorithms.fully_static import (
    EMPIRICAL_ALPHA,
    _low_value_relaxation_probabilities,
    fully_static_algorithm,
    high_value_greedy_candidate,
)
from tsao.algorithms.arbitrary_order_greedy import (
    marginal_responding_demand,
    revenue_ordered_customer_assortment,
)
from tsao.algorithms.one_sided_adaptive import one_sided_adaptive_algorithm
from tsao.algorithms.one_sided_static import one_sided_static_algorithm
from tsao.instance import MarketInstance
from tsao.oracle import MnlOption, optimize_mnl


def test_high_value_greedy_skips_customer_with_no_available_edge() -> None:
    instance = MarketInstance([[0.0], [1.0]], [[10.0, 1.0]])
    candidate = high_value_greedy_candidate(instance)
    assert candidate.edges is not None
    assert not bool(candidate.edges[0, 0])
    assert bool(candidate.edges[1, 0])


def test_fully_static_supplier_high_case_uses_single_pure_transposition() -> None:
    instance = MarketInstance([[0.1, 0.2], [0.15, 0.1]], [[1.2, 0.9], [1.0, 1.1]])
    checksum = instance.checksum()
    result = fully_static_algorithm(instance, replications=2, seed=4)
    assert instance.checksum() == checksum
    assert result.metadata["alpha"] == EMPIRICAL_ALPHA
    assert set(result.metadata["candidate_values"]) == {"low_low", "customer_high", "supplier_high"}


def test_low_value_matrix_relaxation_matches_scalar_formulation() -> None:
    instance = MarketInstance(
        [[0.11, 0.23, 0.37], [0.19, 0.31, 0.43]],
        [[0.29, 0.17], [0.41, 0.13], [0.07, 0.47]],
    )
    matrix_probabilities = _low_value_relaxation_probabilities(instance)

    n, m = instance.num_customers, instance.num_suppliers
    model = gp.Model("scalar_low_value_reference")
    model.Params.OutputFlag = 0
    y = model.addVars(n, m, lb=0.0)
    row_slack = model.addVars(n, lb=0.0)
    column_slack = model.addVars(m, lb=0.0)
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
        gp.quicksum(
            instance.v[i, j] * instance.w[j, i] * y[i, j]
            for i in range(n)
            for j in range(m)
        ),
        GRB.MAXIMIZE,
    )
    model.optimize()
    scalar_probabilities = np.array([[y[i, j].X for j in range(m)] for i in range(n)])

    np.testing.assert_allclose(matrix_probabilities, scalar_probabilities, atol=1e-9)


def test_array_greedy_assortment_matches_generic_mnl_oracle() -> None:
    instance = MarketInstance(
        [[0.0, 0.4, 0.7], [0.2, 0.5, 0.3]],
        [[0.6, 0.2], [0.8, 0.4], [0.1, 0.9]],
        customer_outside=[1.2, 0.9],
        supplier_outside=[0.8, 1.1, 1.3],
    )
    backlog = np.array([0.5, 0.0, 0.7])
    options = [
        MnlOption(
            supplier,
            marginal_responding_demand(instance, supplier, 1, backlog[supplier]),
            instance.v[1, supplier],
        )
        for supplier in range(instance.num_suppliers)
    ]
    expected = optimize_mnl(options, instance.customer_outside[1]).item_ids
    assert revenue_ordered_customer_assortment(instance, 1, backlog) == expected


def test_stochastic_paper_algorithms_use_bounded_replications() -> None:
    instance = MarketInstance([[0.4, 0.8], [0.7, 0.2]], [[0.5, 0.9], [0.3, 0.6]])
    os_result = one_sided_static_algorithm(instance, replications_per_side=2, seed=7)
    oa_result = one_sided_adaptive_algorithm(instance, replications_per_side=2, seed=8)
    fa_theorem = fully_adaptive_theorem_algorithm(instance, replications=4, seed=9)
    fa_experiment = fully_adaptive_experiment_algorithm(instance, 2, 8, oa_result)
    assert len(os_result.replications) == 4
    assert os_result.metadata["end_to_end_replications_per_side"] == 2
    assert os_result.metadata["evaluations_per_constructed_policy"] == 1
    assert len(oa_result.replications) == 4
    assert len(fa_theorem.replications) == 4
    assert fa_experiment.value == oa_result.value
    assert fa_experiment.metadata["reuses"] == "ALG_OA"
