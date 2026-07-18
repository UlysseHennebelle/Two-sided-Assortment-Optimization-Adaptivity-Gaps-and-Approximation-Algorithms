"""Section 7 Table 1 experiment job."""

from __future__ import annotations

from typing import Any

from ..algorithms.fully_static import fully_static_algorithm
from ..benchmarks.opt_fully_static import optimize_fully_static
from ..generation.section7 import GeneratedInstance
from .common import algorithm_record, replication_records, solver_record, timed


def run_section7_fully_static(
    generated: GeneratedInstance,
    config: dict[str, Any],
    config_digest: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run ALG(FS) and Problem (38) on one generated instance."""

    algorithm_seed = generated.generation_seed ^ 0xF501
    algorithm, algorithm_runtime = timed(
        lambda: fully_static_algorithm(
            generated.instance,
            int(config["algorithms"]["fully_static_rounding_reps"]),
            algorithm_seed,
            output=False,
        )
    )
    scenario_id = generated.instance_id
    algorithm_run = algorithm_record(
        generated.campaign_id,
        generated.instance_id,
        scenario_id,
        algorithm,
        algorithm_runtime,
        algorithm_seed,
        config_digest,
    )
    solver_config = config["solver"]
    solver_seed = (generated.generation_seed ^ 0xF502) % 2_000_000_000
    solver_result = optimize_fully_static(
        generated.instance,
        float(solver_config["mip_gap"]),
        float(solver_config["time_limit_seconds"]),
        int(solver_config["threads"]),
        bool(solver_config["output"]),
        solver_seed,
    )
    solver_run = solver_record(
        generated.campaign_id,
        generated.instance_id,
        scenario_id,
        solver_result,
        solver_seed,
        config_digest,
    )
    return [algorithm_run, solver_run], replication_records(algorithm_run, algorithm)
