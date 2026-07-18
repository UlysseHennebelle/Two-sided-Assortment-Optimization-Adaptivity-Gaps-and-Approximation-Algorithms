"""Shared run-record construction for all paper experiments."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any, Callable

import gurobipy as gp

from ..benchmarks.solver import SolverResult
from ..policy import AlgorithmResult
from ..storage.schemas import SCHEMA_VERSION


def stable_run_id(instance_id: str, scenario_id: str, algorithm: str, seed: int) -> str:
    payload = f"{instance_id}|{scenario_id}|{algorithm}|{seed}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def timed(call: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    result = call()
    return result, float(time.perf_counter() - start)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def algorithm_record(
    campaign_id: str,
    instance_id: str,
    scenario_id: str,
    result: AlgorithmResult,
    runtime_seconds: float,
    seed: int,
    config_digest: str,
) -> dict[str, Any]:
    run_id = stable_run_id(instance_id, scenario_id, result.name, seed)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "campaign_id": campaign_id,
        "instance_id": instance_id,
        "scenario_id": scenario_id,
        "algorithm": result.name,
        "initiating_side": result.initiating_side.value if result.initiating_side else None,
        "status": "completed",
        "value": result.value,
        "incumbent": None,
        "best_bound": None,
        "relative_gap": None,
        "runtime_seconds": runtime_seconds,
        "algorithm_seed": seed,
        "solver_seed": None,
        "solver_name": None,
        "solver_version": None,
        "config_hash": config_digest,
        "metadata_json": _json(result.metadata),
        "created_at_utc": datetime.now(timezone.utc),
    }


def solver_record(
    campaign_id: str,
    instance_id: str,
    scenario_id: str,
    result: SolverResult,
    seed: int,
    config_digest: str,
) -> dict[str, Any]:
    run_id = stable_run_id(instance_id, scenario_id, result.name, seed)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "campaign_id": campaign_id,
        "instance_id": instance_id,
        "scenario_id": scenario_id,
        "algorithm": result.name,
        "initiating_side": None,
        "status": result.status,
        "value": result.incumbent,
        "incumbent": result.incumbent,
        "best_bound": result.best_bound,
        "relative_gap": result.relative_gap,
        "runtime_seconds": result.runtime_seconds,
        "algorithm_seed": None,
        "solver_seed": seed,
        "solver_name": "gurobi",
        "solver_version": ".".join(str(item) for item in gp.gurobi.version()),
        "config_hash": config_digest,
        "metadata_json": _json(result.metadata),
        "created_at_utc": datetime.now(timezone.utc),
    }


def simple_value_record(
    campaign_id: str,
    instance_id: str,
    scenario_id: str,
    name: str,
    value: float,
    runtime_seconds: float,
    config_digest: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = AlgorithmResult(name=name, value=float(value), metadata=metadata or {})
    return algorithm_record(campaign_id, instance_id, scenario_id, result, runtime_seconds, 0, config_digest)


def replication_records(run_record: dict[str, Any], result: AlgorithmResult) -> list[dict[str, Any]]:
    """Expand retained stochastic values and their exact child seeds."""

    seeds = tuple(result.metadata.get("replication_seeds", ()))
    values = result.replications
    if not values:
        return []
    if seeds and len(seeds) != len(values):
        raise ValueError(f"Seed/value count mismatch for {result.name}")
    split = len(values) // 2 if result.name in {"ALG_OS", "ALG_OA", "ALG_FA"} else len(values)
    records = []
    for index, value in enumerate(values):
        side = "customers" if index < split else "suppliers"
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_record["run_id"],
                "replication": index,
                "initiating_side": side,
                "simulation_seed": seeds[index] if seeds else None,
                "matches": float(value),
            }
        )
    return records
