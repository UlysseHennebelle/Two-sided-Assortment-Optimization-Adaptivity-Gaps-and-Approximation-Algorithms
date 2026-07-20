"""Evaluate Section 7 algorithms and benchmarks on one generated instance."""

from __future__ import annotations

from typing import Any

from ..algorithms.fully_static import fully_static_algorithm
from ..algorithms.one_sided_adaptive import one_sided_adaptive_algorithm
from ..algorithms.one_sided_static import one_sided_static_algorithm
from ..benchmarks.dynamic_programs import optimize_fully_adaptive, optimize_one_sided_adaptive
from ..benchmarks.opt_fully_static import optimize_fully_static
from ..benchmarks.opt_one_sided_static import optimize_one_sided_static
from ..benchmarks.ub_fully_adaptive import upper_bound_fully_adaptive
from ..benchmarks.ub_one_sided_adaptive import upper_bound_one_sided_adaptive
from ..generation.section7 import GeneratedInstance
from ..policy import AlgorithmResult
from .common import algorithm_record, simple_value_record, solver_record, timed


def run_alg_fs(generated: GeneratedInstance, config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate ``ALG(FS)``."""

    seed = generated.generation_seed ^ 0xF501
    result, runtime = timed(
        lambda: fully_static_algorithm(
            generated.instance,
            int(config["algorithms"]["fully_static_rounding_reps"]),
            seed,
            output=False,
        )
    )
    return algorithm_record(
        generated.campaign_id, generated.instance_id, generated.instance_id, result, runtime, seed
    )


def run_opt_fs(generated: GeneratedInstance, config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate Problem (38), ``OPT(FS)``."""

    solver = config["solver"]
    seed = (generated.generation_seed ^ 0xF502) % 2_000_000_000
    result = optimize_fully_static(
        generated.instance,
        float(solver["mip_gap"]),
        float(solver["time_limit_seconds"]),
        int(solver["threads"]),
        bool(solver["output"]),
        seed,
    )
    return solver_record(
        generated.campaign_id, generated.instance_id, generated.instance_id, result, seed
    )


def run_alg_os(generated: GeneratedInstance, config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate ``ALG(OS)``."""

    seed = generated.generation_seed ^ 0x0501
    result, runtime = timed(
        lambda: one_sided_static_algorithm(
            generated.instance,
            int(config["algorithms"]["one_sided_static_reps_per_side"]),
            seed,
        )
    )
    return algorithm_record(
        generated.campaign_id, generated.instance_id, generated.instance_id, result, runtime, seed
    )


def run_alg_oa(generated: GeneratedInstance, config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate ``ALG(OA)``."""

    seed = generated.generation_seed ^ 0x0A01
    result, runtime = timed(
        lambda: one_sided_adaptive_algorithm(
            generated.instance,
            int(config["algorithms"]["one_sided_adaptive_reps_per_side"]),
            seed,
        )
    )
    return algorithm_record(
        generated.campaign_id, generated.instance_id, generated.instance_id, result, runtime, seed
    )


def reuse_alg_fa(generated: GeneratedInstance, alg_oa_record: dict[str, Any]) -> dict[str, Any]:
    """Store the Section 7 ``ALG(FA)`` value defined by ``ALG(OA)``."""

    if alg_oa_record["instance_id"] != generated.instance_id:
        raise ValueError("ALG(FA) requires the matching ALG(OA) result")
    seed = generated.generation_seed ^ 0x0A01
    result = AlgorithmResult(name="ALG_FA", value=float(alg_oa_record["value"]))
    return algorithm_record(
        generated.campaign_id, generated.instance_id, generated.instance_id, result, 0.0, seed
    )


def run_opt_os(generated: GeneratedInstance, config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate exact ``OPT(OS)``."""

    del config
    (value, _, _), runtime = timed(lambda: optimize_one_sided_static(generated.instance))
    return simple_value_record(
        generated.campaign_id,
        generated.instance_id,
        generated.instance_id,
        "OPT_OS",
        value,
        runtime,
    )


def run_opt_oa(generated: GeneratedInstance, config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate exact ``OPT(OA)``."""

    del config
    result, runtime = timed(lambda: optimize_one_sided_adaptive(generated.instance))
    return simple_value_record(
        generated.campaign_id,
        generated.instance_id,
        generated.instance_id,
        "OPT_OA",
        result.value,
        runtime,
    )


def run_ub_oa(generated: GeneratedInstance, config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate Problem (39), ``UB(OA)``."""

    del config
    result, runtime = timed(lambda: upper_bound_one_sided_adaptive(generated.instance))
    record = simple_value_record(
        generated.campaign_id,
        generated.instance_id,
        generated.instance_id,
        "UB_OA",
        result.value,
        runtime,
    )
    record["status"] = f"customers:{result.customer_status};suppliers:{result.supplier_status}"
    return record


def run_opt_fa(generated: GeneratedInstance, config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate exact ``OPT(FA)``."""

    del config
    result, runtime = timed(lambda: optimize_fully_adaptive(generated.instance))
    return simple_value_record(
        generated.campaign_id,
        generated.instance_id,
        generated.instance_id,
        "OPT_FA",
        result.value,
        runtime,
    )


def run_ub_fa(generated: GeneratedInstance, config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate Problem (40), ``UB(FA)``."""

    seed = (generated.generation_seed ^ 0xFA02) % 2_000_000_000
    result = upper_bound_fully_adaptive(
        generated.instance,
        int(config["solver"]["threads"]),
        bool(config["solver"]["output"]),
        seed,
    )
    return solver_record(
        generated.campaign_id, generated.instance_id, generated.instance_id, result, seed
    )


SECTION7_RUNNERS = {
    "ALG_FS": run_alg_fs,
    "OPT_FS": run_opt_fs,
    "ALG_OS": run_alg_os,
    "ALG_OA": run_alg_oa,
    "OPT_OS": run_opt_os,
    "OPT_OA": run_opt_oa,
    "UB_OA": run_ub_oa,
    "OPT_FA": run_opt_fa,
    "UB_FA": run_ub_fa,
}
