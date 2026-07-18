import math

import numpy as np

from tsao.algorithms.fully_static import EMPIRICAL_ALPHA
from tsao.benchmarks.ub_one_sided_adaptive import upper_bound_one_sided_adaptive
from tsao.generation.appendix_g import generate_appendix_g_instance
from tsao.instance import MarketInstance


def test_ub_oa_does_not_mutate_or_leave_instance_transposed() -> None:
    instance = MarketInstance([[0.3, 0.6], [0.2, 0.9]], [[0.4, 0.8], [0.5, 0.7]])
    checksum = instance.checksum()
    result = upper_bound_one_sided_adaptive(instance)
    assert instance.checksum() == checksum
    assert result.value == max(result.customer_value, result.supplier_value)


def test_confirmed_alpha_is_unchanged() -> None:
    assert EMPIRICAL_ALPHA == (math.sqrt(5.0) - 1.0) / 2.0


def test_appendix_g_uses_confirmed_legacy_distribution_orientation() -> None:
    instance = generate_appendix_g_instance(3, 3, q=2, seed=12)
    assert instance.metadata["customer_distribution"] == "exponential_rate_1"
    assert instance.metadata["supplier_distribution"] == "uniform_0_1"
    assert abs(instance.metadata["nonzero_probability"] - math.sqrt(2.0) / 2.0) < 1e-15
    assert np.all(instance.v >= 0) and np.all(instance.w >= 0)
