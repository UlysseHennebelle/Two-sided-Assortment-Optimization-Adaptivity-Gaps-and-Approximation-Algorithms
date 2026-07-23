"""Run selected experiments against the compact Parquet artifacts."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from ..generation.section7 import GeneratedInstance
from ..storage.parquet import append_results, iter_generated_instances, read_results
from .appendix_g import APPENDIX_G_ALGORITHMS, evaluate, outside_scenario_id
from .section7 import SECTION7_RUNNERS, reuse_alg_fa


def _selected(
    source: Iterable[GeneratedInstance],
    sizes: set[int] | None,
    shard_count: int,
    shard_index: int,
    maximum: int | None,
) -> Iterable[GeneratedInstance]:
    count = 0
    for generated in source:
        size = generated.instance.num_customers
        if sizes is not None and size not in sizes:
            continue
        if int(generated.instance_id, 16) % shard_count != shard_index:
            continue
        if maximum is not None and count >= maximum:
            break
        count += 1
        yield generated


def _existing(
    path: Path,
) -> tuple[
    set[tuple[str, str]],
    dict[str, dict[str, Any]],
    dict[str, set[tuple[str, str]]],
]:
    table = read_results(path, ["instance_id", "scenario_id", "algorithm", "value", "run_id"])
    rows = table.to_pylist()
    keys = {(str(row["scenario_id"]), str(row["algorithm"])) for row in rows}
    oa = {
        str(row["instance_id"]): row
        for row in rows
        if row["algorithm"] == "ALG_OA" and row["scenario_id"] == row["instance_id"]
    }
    by_instance: dict[str, set[tuple[str, str]]] = {}
    for row in rows:
        by_instance.setdefault(str(row["instance_id"]), set()).add(
            (str(row["scenario_id"]), str(row["algorithm"]))
        )
    return keys, oa, by_instance


def _flush(path: Path, pending: list[dict[str, Any]], keys: set[tuple[str, str]]) -> None:
    if not pending:
        return
    append_results(path, pending)
    keys.update((str(row["scenario_id"]), str(row["algorithm"])) for row in pending)
    pending.clear()


def run_section7(
    instances_path: Path,
    results_path: Path,
    config: dict[str, Any],
    algorithms: tuple[str, ...],
    sizes: set[int] | None,
    shard_count: int,
    shard_index: int,
    maximum: int | None,
    progress: Callable[[str], None] = print,
) -> int:
    """Fill missing Section 7 values for the requested algorithms."""

    unknown = set(algorithms) - (set(SECTION7_RUNNERS) | {"ALG_FA"})
    if unknown:
        raise ValueError(f"Unknown Section 7 algorithms: {sorted(unknown)}")
    keys, oa_records, by_instance = _existing(results_path)
    pending: list[dict[str, Any]] = []
    completed = 0
    adaptivity_algorithms = {
        "ALG_OS", "OPT_OS", "ALG_OA", "OPT_OA", "UB_OA",
        "ALG_FA", "OPT_FA", "UB_FA",
    }
    adaptivity_sizes = set(config["adaptivity_sizes"])
    fully_static_sizes = set(config["fully_static_sizes"])
    complete_ids = {
        instance_id
        for instance_id, completed in by_instance.items()
        if all((instance_id, algorithm) in completed for algorithm in algorithms)
    }
    source = iter_generated_instances(
        instances_path,
        campaign_id=str(config["campaign_id"]),
        exclude_instance_ids=complete_ids,
    )
    for generated in _selected(source, sizes, shard_count, shard_index, maximum):
        market_size = generated.instance.num_customers
        for algorithm in algorithms:
            key = (generated.instance_id, algorithm)
            if key in keys:
                continue
            if (
                algorithm in adaptivity_algorithms
                and market_size not in adaptivity_sizes
            ):
                continue
            if algorithm == "OPT_OS" and market_size > int(config["benchmarks"]["max_exact_os_size"]):
                continue
            if algorithm == "OPT_OA" and market_size > int(config["benchmarks"]["max_opt_oa_size"]):
                continue
            if algorithm == "OPT_FA" and market_size > int(config["benchmarks"]["max_opt_fa_size"]):
                continue
            if algorithm == "OPT_FS" and market_size not in fully_static_sizes:
                continue
            if algorithm == "ALG_FA":
                source_row = oa_records.get(generated.instance_id)
                if source_row is None:
                    raise ValueError("Run ALG_OA before ALG_FA for each Section 7 instance")
                row = reuse_alg_fa(generated, source_row)
            else:
                row = SECTION7_RUNNERS[algorithm](generated, config)
            row["market_size"] = market_size
            pending.append(row)
            completed += 1
            if algorithm == "ALG_OA":
                oa_records[generated.instance_id] = row
            if len(pending) >= 25:
                _flush(results_path, pending, keys)
                progress(f"completed={completed}")
    _flush(results_path, pending, keys)
    return completed


def run_figure3(
    instances_path: Path,
    results_path: Path,
    config: dict[str, Any],
    algorithms: tuple[str, ...] = APPENDIX_G_ALGORITHMS,
    sizes: set[int] | None = None,
    shard_count: int = 1,
    shard_index: int = 0,
    maximum: int | None = None,
    progress: Callable[[str], None] = print,
) -> int:
    """Fill missing Figure 3 values."""

    keys, _, by_instance = _existing(results_path)
    pending: list[dict[str, Any]] = []
    completed = 0
    complete_ids = {
        instance_id
        for instance_id, completed in by_instance.items()
        if all((instance_id, algorithm) in completed for algorithm in algorithms)
    }
    source = iter_generated_instances(
        instances_path,
        campaign_id=str(config["campaign_id"]),
        exclude_instance_ids=complete_ids,
    )
    for generated in _selected(source, sizes, shard_count, shard_index, maximum):
        missing = tuple(
            algorithm for algorithm in algorithms if (generated.instance_id, algorithm) not in keys
        )
        if not missing:
            continue
        pending.extend(evaluate(generated, config, missing))
        completed += len(missing)
        if len(pending) >= 25:
            _flush(results_path, pending, keys)
            progress(f"completed={completed}")
    _flush(results_path, pending, keys)
    return completed


def run_figure4(
    instances_path: Path,
    results_path: Path,
    config: dict[str, Any],
    algorithms: tuple[str, ...] = APPENDIX_G_ALGORITHMS,
    sizes: set[int] | None = None,
    shard_count: int = 1,
    shard_index: int = 0,
    maximum: int | None = None,
    progress: Callable[[str], None] = print,
) -> int:
    """Fill missing Figure 4 values."""

    keys, _, by_instance = _existing(results_path)
    pending: list[dict[str, Any]] = []
    completed = 0
    complete_ids = {
        instance_id
        for instance_id, completed in by_instance.items()
        if all(
            (outside_scenario_id(instance_id, float(outside_option)), algorithm) in completed
            for outside_option in config["outside_option_values"]
            for algorithm in algorithms
        )
    }
    source = iter_generated_instances(
        instances_path,
        campaign_id=str(config["campaign_id"]),
        exclude_instance_ids=complete_ids,
    )
    for generated in _selected(source, sizes, shard_count, shard_index, maximum):
        for outside_option in config["outside_option_values"]:
            scenario_id = outside_scenario_id(generated.instance_id, float(outside_option))
            missing = tuple(
                algorithm for algorithm in algorithms if (scenario_id, algorithm) not in keys
            )
            if not missing:
                continue
            pending.extend(
                evaluate(
                    generated,
                    config,
                    missing,
                    scenario_id=scenario_id,
                    outside_option=float(outside_option),
                )
            )
            completed += len(missing)
            if len(pending) >= 25:
                _flush(results_path, pending, keys)
                progress(f"completed={completed}")
    _flush(results_path, pending, keys)
    return completed
