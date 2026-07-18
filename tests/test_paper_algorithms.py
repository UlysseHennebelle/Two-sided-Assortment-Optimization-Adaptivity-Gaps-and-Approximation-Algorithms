import numpy as np

from tsao.algorithms.fully_adaptive import (
    fully_adaptive_experiment_algorithm,
    fully_adaptive_theorem_algorithm,
)
from tsao.algorithms.fully_static import (
    EMPIRICAL_ALPHA,
    fully_static_algorithm,
    high_value_greedy_candidate,
)
from tsao.algorithms.one_sided_adaptive import one_sided_adaptive_algorithm
from tsao.algorithms.one_sided_static import one_sided_static_algorithm
from tsao.instance import MarketInstance


def test_high_value_greedy_skips_customer_with_no_available_edge() -> None:
    instance = MarketInstance([[0.0], [1.0]], [[10.0, 1.0]])
    candidate = high_value_greedy_candidate(instance)
    assert candidate.edges is not None
    assert not bool(candidate.edges[0, 0])
    assert bool(candidate.edges[1, 0])


def test_fully_static_supplier_high_case_uses_single_pure_transposition() -> None:
    instance = MarketInstance([[0.1, 0.2], [0.15, 0.1]], [[1.2, 0.9], [1.0, 1.1]])
    result = fully_static_algorithm(instance, replications=2, seed=4)
    assert result.metadata["corrected_single_supplier_high_transposition"] is True
    assert result.metadata["alpha"] == EMPIRICAL_ALPHA
    assert set(result.metadata["candidate_values"]) == {"low_low", "customer_high", "supplier_high"}


def test_stochastic_paper_algorithms_use_bounded_replications() -> None:
    instance = MarketInstance([[0.4, 0.8], [0.7, 0.2]], [[0.5, 0.9], [0.3, 0.6]])
    os_result = one_sided_static_algorithm(instance, construction_replications=2, evaluation_replications=2, seed=7)
    oa_result = one_sided_adaptive_algorithm(instance, replications_per_side=2, seed=8)
    fa_theorem = fully_adaptive_theorem_algorithm(instance, replications=4, seed=9)
    fa_experiment = fully_adaptive_experiment_algorithm(instance, 2, 8, oa_result)
    assert len(os_result.replications) == 8
    assert len(oa_result.replications) == 4
    assert len(fa_theorem.replications) == 4
    assert fa_experiment.value == oa_result.value
    assert fa_experiment.metadata["reuses"] == "ALG_OA"
