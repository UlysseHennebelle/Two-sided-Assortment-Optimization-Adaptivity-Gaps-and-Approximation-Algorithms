"""Construct final result rows for the paper experiments."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from typing import Any

from ..benchmarks.solver import SolverResult
from ..policy import AlgorithmResult


def stable_run_id(instance_id: str, scenario_id: str, algorithm: str, seed: int) -> str:
    """Return the deterministic identifier for one algorithm evaluation."""

    payload = f"{instance_id}|{scenario_id}|{algorithm}|{seed}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def timed(call: Callable[[], Any]) -> tuple[Any, float]:
    """Return a computation result and its elapsed wall-clock time."""

    start = time.perf_counter()
    result = call()
    return result, float(time.perf_counter() - start)


def _row(
    campaign_id: str,
    instance_id: str,
    scenario_id: str,
    algorithm: str,
    status: str,
    value: float | None,
    runtime_seconds: float,
    seed: int,
    *,
    incumbent: float | None = None,
    best_bound: float | None = None,
    relative_gap: float | None = None,
    solver: bool = False,
) -> dict[str, Any]:
    return {
        "run_id": stable_run_id(instance_id, scenario_id, algorithm, seed),
        "campaign_id": campaign_id,
        "instance_id": instance_id,
        "scenario_id": scenario_id,
        "market_size": None,
        "q": None,
        "outside_option": None,
        "algorithm": algorithm,
        "status": status,
        "value": value,
        "incumbent": incumbent,
        "best_bound": best_bound,
        "relative_gap": relative_gap,
        "runtime_seconds": runtime_seconds,
        "algorithm_seed": None if solver else seed,
        "solver_seed": seed if solver else None,
    }


def algorithm_record(
    campaign_id: str,
    instance_id: str,
    scenario_id: str,
    result: AlgorithmResult,
    runtime_seconds: float,
    seed: int,
    config_digest: str | None = None,
) -> dict[str, Any]:
    """Create a result row for a sampled or deterministic algorithm."""

    del config_digest
    return _row(
        campaign_id,
        instance_id,
        scenario_id,
        result.name,
        "completed",
        float(result.value),
        runtime_seconds,
        seed,
    )


def solver_record(
    campaign_id: str,
    instance_id: str,
    scenario_id: str,
    result: SolverResult,
    seed: int,
    config_digest: str | None = None,
) -> dict[str, Any]:
    """Create a result row for an optimization model."""

    del config_digest
    return _row(
        campaign_id,
        instance_id,
        scenario_id,
        result.name,
        result.status,
        result.incumbent,
        result.runtime_seconds,
        seed,
        incumbent=result.incumbent,
        best_bound=result.best_bound,
        relative_gap=result.relative_gap,
        solver=True,
    )


def simple_value_record(
    campaign_id: str,
    instance_id: str,
    scenario_id: str,
    name: str,
    value: float,
    runtime_seconds: float,
    config_digest: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a result row for a directly computed benchmark value."""

    del config_digest, metadata
    return _row(
        campaign_id,
        instance_id,
        scenario_id,
        name,
        "completed",
        float(value),
        runtime_seconds,
        0,
    )


def replication_records(run_record: dict[str, Any], result: AlgorithmResult) -> list[dict[str, Any]]:
    """Return no rows because published output stores final values only."""

    del run_record, result
    return []
