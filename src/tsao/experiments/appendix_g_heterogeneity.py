"""Appendix G heterogeneity campaign underlying Figure 3."""

from __future__ import annotations

from typing import Any

from ..algorithms.fully_static import fully_static_algorithm
from ..algorithms.one_sided_adaptive import one_sided_adaptive_algorithm
from ..algorithms.one_sided_static import one_sided_static_algorithm
from ..generation.section7 import GeneratedInstance
from .common import algorithm_record, replication_records, timed


def run_appendix_g_heterogeneity(
    generated: GeneratedInstance,
    config: dict[str, Any],
    config_digest: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run ALG(FS), ALG(OS), and ALG(OA) on one structured instance."""

    algorithms = config["algorithms"]
    scenario_id = generated.instance_id
    specifications = [
        (
            lambda seed: fully_static_algorithm(
                generated.instance, int(algorithms["fully_static_rounding_reps"]), seed
            ),
            generated.generation_seed ^ 0xF501,
        ),
        (
            lambda seed: one_sided_static_algorithm(
                generated.instance,
                int(algorithms["one_sided_static_construction_reps"]),
                int(algorithms["one_sided_static_evaluation_reps"]),
                seed,
            ),
            generated.generation_seed ^ 0x0501,
        ),
        (
            lambda seed: one_sided_adaptive_algorithm(
                generated.instance, int(algorithms["one_sided_adaptive_reps_per_side"]), seed
            ),
            generated.generation_seed ^ 0x0A01,
        ),
    ]
    runs: list[dict[str, Any]] = []
    replications: list[dict[str, Any]] = []
    for call, seed in specifications:
        result, runtime = timed(lambda call=call, seed=seed: call(seed))
        run = algorithm_record(
            generated.campaign_id,
            generated.instance_id,
            scenario_id,
            result,
            runtime,
            seed,
            config_digest,
        )
        runs.append(run)
        replications.extend(replication_records(run, result))
    return runs, replications
