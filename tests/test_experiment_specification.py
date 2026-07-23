import inspect
import math

import numpy as np

from tsao.algorithms.fully_static import EMPIRICAL_ALPHA
from tsao.algorithms.one_sided_static import one_sided_static_algorithm
from tsao.benchmarks.ub_one_sided_adaptive import upper_bound_one_sided_adaptive
from tsao.config import load_config
from tsao.generation.appendix_g import generate_appendix_g_instance
from tsao.generation.section7 import generate_section7_campaign
from tsao.instance import MarketInstance


def test_section7_seed_assignment_is_stable_when_grid_expands() -> None:
    def coordinates(campaign):
        return {
            (item.parameters["size"], item.replicate): (
                item.instance_id,
                item.generation_seed,
            )
            for item in campaign
        }

    initial = coordinates(generate_section7_campaign("campaign", [2, 3], 2, 1234))
    expanded = coordinates(generate_section7_campaign("campaign", [2, 3, 4], 3, 1234))
    assert initial.items() <= expanded.items()


def test_ub_oa_is_pure_and_compares_both_sides() -> None:
    instance = MarketInstance([[0.3, 0.6], [0.2, 0.9]], [[0.4, 0.8], [0.5, 0.7]])
    original_v = instance.v.copy()
    original_w = instance.w.copy()
    result = upper_bound_one_sided_adaptive(instance)
    np.testing.assert_array_equal(instance.v, original_v)
    np.testing.assert_array_equal(instance.w, original_w)
    assert result.value == max(result.customer_value, result.supplier_value)


def test_numerical_protocol_matches_the_paper_configuration() -> None:
    config = load_config("config.toml")
    section = config["section7"]
    assert EMPIRICAL_ALPHA == (math.sqrt(5.0) - 1.0) / 2.0
    assert section["fully_static_sizes"] == [2, 3, 4, 5, 6, 8, 9]
    assert inspect.signature(one_sided_static_algorithm).parameters["replications_per_side"].default == 50
    assert section["algorithms"]["one_sided_static_reps_per_side"] == 50
    assert section["benchmarks"] == {
        "max_exact_os_size": 4,
        "max_opt_oa_size": 7,
        "max_opt_fa_size": 6,
    }
    for name in ("figure3", "figure4"):
        figure = config[name]
        assert figure["sizes"] == [50, 100, 200, 500]
        assert figure["fully_static_rounding_reps"] == 10
        assert figure["one_sided_static_reps_per_side"] == 10
        assert figure["one_sided_adaptive_reps_per_side"] == 10


def test_grouped_instance_distribution_parameters() -> None:
    instance = generate_appendix_g_instance(3, 3, q=2, seed=12)
    assert instance.metadata["customer_distribution"] == "exponential_rate_1"
    assert instance.metadata["supplier_distribution"] == "uniform_0_1"
    assert abs(instance.metadata["nonzero_probability"] - math.sqrt(2.0) / 2.0) < 1e-15
    assert np.all(instance.v >= 0) and np.all(instance.w >= 0)
