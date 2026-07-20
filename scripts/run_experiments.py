"""Generate paper instances and evaluate selected algorithms."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsao.config import load_config  # noqa: E402
from tsao.experiments.appendix_g import APPENDIX_G_ALGORITHMS  # noqa: E402
from tsao.experiments.runner import run_figure3, run_figure4, run_section7  # noqa: E402
from tsao.experiments.section7 import SECTION7_RUNNERS  # noqa: E402
from tsao.generation.appendix_g import iter_appendix_g_campaign  # noqa: E402
from tsao.generation.section7 import GeneratedInstance, generate_section7_campaign  # noqa: E402
from tsao.storage.parquet import iter_generated_instances, write_instances  # noqa: E402


def _cell_seed(master: int, size: int, q: int) -> int:
    return int(np.random.SeedSequence([master, size, q]).generate_state(1, dtype=np.uint64)[0])


def _section7_instances(config: dict[str, Any]) -> Iterable[GeneratedInstance]:
    return generate_section7_campaign(
        str(config["campaign_id"]),
        [int(value) for value in config["all_sizes"]],
        int(config["instances_per_size"]),
        int(config["seed"]),
        int(config["round_digits"]),
    )


def _figure3_instances(config: dict[str, Any]) -> Iterable[GeneratedInstance]:
    for size in config["sizes"]:
        for q in config["q_values"]:
            samples = 100 + int(100 // (int(q) * int(q)))
            yield from iter_appendix_g_campaign(
                str(config["campaign_id"]),
                [int(size)],
                [int(q)],
                lambda _q, count=samples: count,
                _cell_seed(int(config["seed"]), int(size), int(q)),
                config["group_scales"],
                float(config["nonzero_probability"]),
                int(config["round_digits"]),
            )


def _figure4_instances(config: dict[str, Any]) -> Iterable[GeneratedInstance]:
    for size in config["sizes"]:
        q = int(config["q"])
        count = int(config["instances_per_size"])
        yield from iter_appendix_g_campaign(
            str(config["campaign_id"]),
            [int(size)],
            [q],
            lambda _q, samples=count: samples,
            _cell_seed(int(config["seed"]), int(size), q),
            config["group_scales"],
            float(config["nonzero_probability"]),
            int(config["round_digits"]),
        )


def generate(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Write the configured generated-instance artifact."""

    generators = {
        "section7": lambda: _section7_instances(config["section7"]),
        "figure3": lambda: _figure3_instances(config["figure3"]),
        "figure4": lambda: _figure4_instances(config["figure4"]),
    }
    names = tuple(generators) if args.experiment == "all" else (args.experiment,)
    selected_campaigns = {str(config[name]["campaign_id"]) for name in names}
    existing = (
        (
            item
            for item in iter_generated_instances(args.instances)
            if item.campaign_id not in selected_campaigns
        )
        if args.instances.exists()
        else ()
    )
    generated = chain(existing, *(generators[name]() for name in names))
    write_instances(args.instances, generated)
    print(f"instances={args.instances}")


def run(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Evaluate missing results for one paper experiment."""

    sizes = set(args.sizes) if args.sizes else None
    if args.experiment == "section7":
        algorithms = tuple(args.algorithms or (*SECTION7_RUNNERS, "ALG_FA"))
        count = run_section7(
            args.instances,
            args.results,
            config["section7"],
            algorithms,
            sizes,
            args.shard_count,
            args.shard_index,
            args.max_instances,
        )
    elif args.experiment == "figure3":
        algorithms = tuple(args.algorithms or APPENDIX_G_ALGORITHMS)
        count = run_figure3(
            args.instances,
            args.results,
            config["figure3"],
            algorithms,
            sizes,
            args.shard_count,
            args.shard_index,
            args.max_instances,
        )
    else:
        algorithms = tuple(args.algorithms or APPENDIX_G_ALGORITHMS)
        count = run_figure4(
            args.instances,
            args.results,
            config["figure4"],
            algorithms,
            sizes,
            args.shard_count,
            args.shard_index,
            args.max_instances,
        )
    print(f"completed={count} results={args.results}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.toml")
    parser.add_argument("--instances", type=Path, default=ROOT / "data" / "instances.parquet")
    parser.add_argument("--results", type=Path, default=ROOT / "data" / "results.parquet")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument(
        "--experiment", choices=["section7", "figure3", "figure4", "all"], default="all"
    )

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("experiment", choices=["section7", "figure3", "figure4"])
    run_parser.add_argument("--algorithms", nargs="+")
    run_parser.add_argument("--sizes", nargs="+", type=int)
    run_parser.add_argument("--max-instances", type=int)
    run_parser.add_argument("--shard-count", type=int, default=1)
    run_parser.add_argument("--shard-index", type=int, default=0)

    args = parser.parse_args()
    if getattr(args, "shard_count", 1) <= 0:
        raise ValueError("shard-count must be positive")
    if not 0 <= getattr(args, "shard_index", 0) < getattr(args, "shard_count", 1):
        raise ValueError("shard-index must be smaller than shard-count")
    config = load_config(args.config)
    if args.command == "generate":
        generate(args, config)
    else:
        run(args, config)


if __name__ == "__main__":
    main()
