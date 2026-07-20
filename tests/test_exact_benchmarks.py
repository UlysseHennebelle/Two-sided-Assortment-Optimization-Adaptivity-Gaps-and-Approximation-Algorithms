from tsao.benchmarks.dynamic_programs import (
    _optimize_fully_adaptive_state,
    optimize_fully_adaptive,
    optimize_one_sided_adaptive,
)
from tsao.benchmarks.opt_fully_static import enumerate_fully_static, optimize_fully_static
from tsao.benchmarks.opt_one_sided_static import (
    _optimize_customer_one_sided_static_scalar,
    optimize_customer_one_sided_static,
    optimize_one_sided_static,
)
from tsao.benchmarks.ub_fully_adaptive import upper_bound_fully_adaptive
from tsao.benchmarks.ub_one_sided_adaptive import upper_bound_one_sided_adaptive
from tsao.instance import MarketInstance


def _instance() -> MarketInstance:
    return MarketInstance([[0.4, 0.8], [0.7, 0.2]], [[0.5, 0.9], [0.3, 0.6]])


def test_optimized_dynamic_programs_match_enumerated_reference_values() -> None:
    instance = _instance()
    oa = optimize_one_sided_adaptive(instance)
    fa = optimize_fully_adaptive(instance)
    assert abs(oa.value - 0.3384715469203004) < 1e-12
    assert abs(fa.value - 0.3703856162304916) < 1e-12
    assert oa.cache_hits > 0
    assert fa.cache_hits > 0


def test_packed_fully_adaptive_dp_matches_state_object_reference() -> None:
    instances = [
        _instance(),
        MarketInstance(
            [[0.2, 0.7, 0.4], [0.9, 0.3, 0.6]],
            [[0.4, 0.6], [0.7, 0.3], [0.5, 0.8]],
        ),
    ]
    for instance in instances:
        packed = optimize_fully_adaptive(instance)
        reference = _optimize_fully_adaptive_state(instance)
        assert abs(packed.value - reference.value) < 1e-12
        assert packed.states == reference.states


def test_exact_policy_values_are_nested_with_tolerance() -> None:
    instance = _instance()
    fs = enumerate_fully_static(instance)
    os = optimize_one_sided_static(instance)[0]
    oa = optimize_one_sided_adaptive(instance).value
    fa = optimize_fully_adaptive(instance).value
    tolerance = 1e-12
    assert fs <= os + tolerance
    assert os <= oa + tolerance
    assert oa <= fa + tolerance


def test_responder_wise_opt_os_matches_joint_outcome_enumerator() -> None:
    instance = MarketInstance(
        [[0.2, 0.7, 0.4], [0.9, 0.3, 0.6], [0.5, 0.8, 0.1]],
        [[0.4, 0.6, 0.2], [0.7, 0.3, 0.9], [0.5, 0.8, 0.4]],
        customer_outside=[0.8, 1.2, 0.9],
        supplier_outside=[1.1, 0.7, 1.3],
    )
    expected = _optimize_customer_one_sided_static_scalar(instance)
    actual = optimize_customer_one_sided_static(instance)
    assert abs(actual - expected) < 1e-12


def test_appendix_f_formulations_bound_tiny_exact_values() -> None:
    instance = _instance()
    fs = enumerate_fully_static(instance)
    oa = optimize_one_sided_adaptive(instance).value
    fa = optimize_fully_adaptive(instance).value
    opt_fs = optimize_fully_static(instance, mip_gap=0.05, time_limit_seconds=20.0, output=False, seed=3)
    ub_oa = upper_bound_one_sided_adaptive(instance)
    ub_fa = upper_bound_fully_adaptive(instance, output=False, seed=4)
    assert opt_fs.incumbent is not None and opt_fs.best_bound is not None
    assert opt_fs.incumbent <= fs + 1e-6
    assert opt_fs.best_bound + 1e-8 >= fs
    assert opt_fs.relative_gap is not None and opt_fs.relative_gap <= 0.05 + 1e-9
    assert ub_oa.value + 1e-7 >= oa
    assert ub_fa.incumbent is not None and ub_fa.incumbent + 1e-7 >= fa
