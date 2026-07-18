"""Appendix G outside-option campaign underlying Figure 4."""

from __future__ import annotations

import hashlib
from typing import Any

from ..generation.section7 import GeneratedInstance
from .appendix_g_heterogeneity import run_appendix_g_heterogeneity


def outside_scenario_id(instance_id: str, outside_option: float) -> str:
    return hashlib.sha256(f"{instance_id}|outside={outside_option:.17g}".encode("utf-8")).hexdigest()[:24]


def run_appendix_g_outside_option(
    generated: GeneratedInstance,
    outside_option: float,
    config: dict[str, Any],
    config_digest: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate one base instance at one outside-option value."""

    scenario = generated.instance.with_outside_option(outside_option)
    scenario_id = outside_scenario_id(generated.instance_id, outside_option)
    scenario_generated = GeneratedInstance(
        generated.instance_id,
        generated.campaign_id,
        generated.experiment,
        generated.replicate,
        generated.generation_seed,
        scenario,
        {**generated.parameters, "outside_option": outside_option, "scenario_id": scenario_id},
    )
    runs, replications = run_appendix_g_heterogeneity(scenario_generated, config, config_digest)
    old_to_new: dict[str, str] = {}
    for run in runs:
        old_run_id = run["run_id"]
        run["scenario_id"] = scenario_id
        run["run_id"] = hashlib.sha256(
            f"{generated.instance_id}|{scenario_id}|{run['algorithm']}|{run['algorithm_seed']}".encode("utf-8")
        ).hexdigest()[:24]
        old_to_new[old_run_id] = run["run_id"]
    for record in replications:
        record["run_id"] = old_to_new[record["run_id"]]
    return runs, replications
