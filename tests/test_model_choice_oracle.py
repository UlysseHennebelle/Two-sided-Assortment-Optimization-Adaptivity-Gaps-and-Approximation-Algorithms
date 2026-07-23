from itertools import chain, combinations

import numpy as np

from tsao.choice import mnl_expected_value, mnl_probabilities
from tsao.instance import MarketInstance
from tsao.oracle import MnlOption, optimize_mnl


def test_instance_transposition_is_an_involution() -> None:
    instance = MarketInstance(
        [[0.2, 0.8], [0.4, 0.6]],
        [[0.3, 0.9], [0.5, 0.7]],
        customer_outside=[1.0, 2.0],
        supplier_outside=[3.0, 4.0],
    )
    restored = instance.transposed().transposed()
    np.testing.assert_array_equal(restored.v, instance.v)
    np.testing.assert_array_equal(restored.w, instance.w)
    np.testing.assert_array_equal(restored.customer_outside, instance.customer_outside)
    np.testing.assert_array_equal(restored.supplier_outside, instance.supplier_outside)
    assert restored.customer_capacities == instance.customer_capacities
    assert restored.supplier_capacities == instance.supplier_capacities


def test_mnl_probabilities_sum_to_one() -> None:
    outside, items = mnl_probabilities([0.2, 0.5, 0.3], outside_weight=2.0)
    assert abs(outside + float(items.sum()) - 1.0) < 1e-12


def test_revenue_ordered_oracle_matches_subset_enumeration() -> None:
    options = [
        MnlOption("a", 0.9, 0.2),
        MnlOption("b", 0.4, 1.3),
        MnlOption("c", 0.1, 2.0),
        MnlOption("d", -0.2, 4.0),
    ]
    solution = optimize_mnl(options, outside_weight=1.7)
    subsets = chain.from_iterable(combinations(options, size) for size in range(len(options) + 1))
    exact = max(
        mnl_expected_value(
            [option.revenue for option in subset],
            [option.weight for option in subset],
            outside_weight=1.7,
        )
        for subset in subsets
    )
    assert abs(solution.value - exact) < 1e-12
