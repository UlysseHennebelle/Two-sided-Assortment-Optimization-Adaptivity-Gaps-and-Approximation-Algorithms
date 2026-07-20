"""Evaluate the algorithms used in Appendix G."""

from __future__ import annotations

import hashlib
from typing import Any

from ..algorithms.fully_static import fully_static_algorithm
from ..algorithms.one_sided_adaptive import one_sided_adaptive_algorithm
from ..algorithms.one_sided_static import one_sided_static_algorithm
from ..generation.section7 import GeneratedInstance
from .common import algorithm_record, timed

APPENDIX_G_ALGORITHMS = ("ALG_FS", "ALG_OS", "ALG_OA")


def outside_scenario_id(instance_id: str, outside_option: float) -> str:
    """Return the identifier for one outside-option evaluation."""

    payload = f"{instance_id}|outside={outside_option:.17g}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def evaluate(
    generated: GeneratedInstance,
    config: dict[str, Any],
    algorithms: tuple[str, ...] = APPENDIX_G_ALGORITHMS,
    *,
    scenario_id: str | None = None,
    outside_option: float | None = None,
) -> list[dict[str, Any]]:
    """Return the selected Appendix G values for one instance or scenario."""

    scenario = (
        generated.instance
        if outside_option is None
        else generated.instance.with_outside_option(float(outside_option))
    )
    scenario_key = scenario_id or generated.instance_id
    specifications = {
        "ALG_FS": (
            lambda seed: fully_static_algorithm(
                scenario, int(config["fully_static_rounding_reps"]), seed
            ),
            generated.generation_seed ^ 0xF501,
        ),
        "ALG_OS": (
            lambda seed: one_sided_static_algorithm(
                scenario, int(config["one_sided_static_reps_per_side"]), seed
            ),
            generated.generation_seed ^ 0x0501,
        ),
        "ALG_OA": (
            lambda seed: one_sided_adaptive_algorithm(
                scenario, int(config["one_sided_adaptive_reps_per_side"]), seed
            ),
            generated.generation_seed ^ 0x0A01,
        ),
    }
    unknown = set(algorithms) - set(specifications)
    if unknown:
        raise ValueError(f"Unknown Appendix G algorithms: {sorted(unknown)}")

    rows = []
    for name in algorithms:
        call, seed = specifications[name]
        result, runtime = timed(lambda call=call, seed=seed: call(seed))
        row = algorithm_record(
            generated.campaign_id,
            generated.instance_id,
            scenario_key,
            result,
            runtime,
            seed,
        )
        row["market_size"] = generated.instance.num_customers
        row["q"] = int(generated.parameters["q"])
        row["outside_option"] = outside_option
        rows.append(row)
    return rows
