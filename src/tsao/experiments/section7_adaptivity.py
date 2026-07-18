"""Section 7 policy comparison for Tables 2-4."""

from __future__ import annotations

from typing import Any

import cvxpy as cp

from ..algorithms.fully_static import fully_static_algorithm
from ..algorithms.fully_adaptive import fully_adaptive_experiment_algorithm
from ..algorithms.one_sided_adaptive import one_sided_adaptive_algorithm
from ..algorithms.one_sided_static import one_sided_static_algorithm
from ..benchmarks.dynamic_programs import optimize_fully_adaptive, optimize_one_sided_adaptive
from ..benchmarks.opt_one_sided_static import optimize_one_sided_static
from ..benchmarks.ub_fully_adaptive import upper_bound_fully_adaptive
from ..benchmarks.ub_one_sided_adaptive import upper_bound_one_sided_adaptive
from ..generation.section7 import GeneratedInstance
from .common import algorithm_record, replication_records, simple_value_record, solver_record, timed


def run_section7_adaptivity(
    generated: GeneratedInstance,
    config: dict[str, Any],
    config_digest: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run all adaptive algorithms, exact values, and upper bounds for one instance."""

    instance = generated.instance
    scenario_id = generated.instance_id
    algorithms = config["algorithms"]
    benchmarks = config["benchmarks"]
    runs: list[dict[str, Any]] = []
    replications: list[dict[str, Any]] = []

    size = max(instance.num_customers, instance.num_suppliers)
    if size not in set(config["fully_static_sizes"]):
        fs_seed = generated.generation_seed ^ 0xF501
        fs_result, fs_runtime = timed(
            lambda: fully_static_algorithm(
                instance,
                int(algorithms["fully_static_rounding_reps"]),
                fs_seed,
                output=False,
            )
        )
        runs.append(
            algorithm_record(
                generated.campaign_id,
                generated.instance_id,
                scenario_id,
                fs_result,
                fs_runtime,
                fs_seed,
                config_digest,
            )
        )

    os_seed = generated.generation_seed ^ 0x0501
    os_result, os_runtime = timed(
        lambda: one_sided_static_algorithm(
            instance,
            int(algorithms["one_sided_static_construction_reps"]),
            int(algorithms["one_sided_static_evaluation_reps"]),
            os_seed,
        )
    )
    os_run = algorithm_record(generated.campaign_id, generated.instance_id, scenario_id, os_result, os_runtime, os_seed, config_digest)
    runs.append(os_run)
    replications.extend(replication_records(os_run, os_result))

    oa_seed = generated.generation_seed ^ 0x0A01
    oa_result, oa_runtime = timed(
        lambda: one_sided_adaptive_algorithm(
            instance, int(algorithms["one_sided_adaptive_reps_per_side"]), oa_seed
        )
    )
    oa_run = algorithm_record(generated.campaign_id, generated.instance_id, scenario_id, oa_result, oa_runtime, oa_seed, config_digest)
    runs.append(oa_run)
    replications.extend(replication_records(oa_run, oa_result))

    fa_result = fully_adaptive_experiment_algorithm(
        instance,
        int(algorithms["one_sided_adaptive_reps_per_side"]),
        oa_seed,
        oa_result,
    )
    fa_run = algorithm_record(generated.campaign_id, generated.instance_id, scenario_id, fa_result, 0.0, oa_seed, config_digest)
    runs.append(fa_run)

    if size <= int(benchmarks["max_exact_os_size"]):
        (opt_os_value, side, side_values), runtime = timed(lambda: optimize_one_sided_static(instance))
        runs.append(
            simple_value_record(
                generated.campaign_id,
                generated.instance_id,
                scenario_id,
                "OPT_OS",
                opt_os_value,
                runtime,
                config_digest,
                {"initiating_side": side.value, "side_values": side_values},
            )
        )
    if size <= int(benchmarks["max_opt_oa_size"]):
        result, runtime = timed(lambda: optimize_one_sided_adaptive(instance))
        runs.append(
            simple_value_record(
                generated.campaign_id,
                generated.instance_id,
                scenario_id,
                "OPT_OA",
                result.value,
                runtime,
                config_digest,
                {"states": result.states, "cache_hits": result.cache_hits, "side_values": result.side_values},
            )
        )
    ub_oa, ub_oa_runtime = timed(lambda: upper_bound_one_sided_adaptive(instance))
    ub_oa_record = simple_value_record(
            generated.campaign_id,
            generated.instance_id,
            scenario_id,
            "UB_OA",
            ub_oa.value,
            ub_oa_runtime,
            config_digest,
            {"side": ub_oa.side.value, "side_values": (ub_oa.customer_value, ub_oa.supplier_value)},
        )
    ub_oa_record["status"] = f"customers:{ub_oa.customer_status};suppliers:{ub_oa.supplier_status}"
    ub_oa_record["solver_name"] = (
        ub_oa.customer_solver
        if ub_oa.customer_solver == ub_oa.supplier_solver
        else f"{ub_oa.customer_solver}/{ub_oa.supplier_solver}"
    )
    ub_oa_record["solver_version"] = cp.__version__
    runs.append(ub_oa_record)
    if size <= int(benchmarks["max_opt_fa_size"]):
        result, runtime = timed(lambda: optimize_fully_adaptive(instance))
        runs.append(
            simple_value_record(
                generated.campaign_id,
                generated.instance_id,
                scenario_id,
                "OPT_FA",
                result.value,
                runtime,
                config_digest,
                {"states": result.states, "cache_hits": result.cache_hits},
            )
        )
    ub_fa_seed = (generated.generation_seed ^ 0xFA02) % 2_000_000_000
    ub_fa = upper_bound_fully_adaptive(
        instance,
        int(config["solver"]["threads"]),
        bool(config["solver"]["output"]),
        ub_fa_seed,
    )
    runs.append(
        solver_record(
            generated.campaign_id,
            generated.instance_id,
            scenario_id,
            ub_fa,
            ub_fa_seed,
            config_digest,
        )
    )
    return runs, replications
