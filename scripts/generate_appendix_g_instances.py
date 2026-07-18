"""Generate Appendix G heterogeneity and outside-option base instances."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import ROOT
from tsao.config import load_config
from tsao.generation.appendix_g import generate_appendix_g_campaign
from tsao.storage.parquet import dataset_part_path, write_generated_instances


def _seed(master: int, *parts: int) -> int:
    return int(np.random.SeedSequence([master, *parts]).generate_state(1, dtype=np.uint64)[0])


def _heterogeneity(args, config) -> int:
    count = 0
    sizes = [size for size in config["generated_sizes"] if args.max_size is None or size <= args.max_size]
    for size in sizes:
        for q in config["q_values"]:
            part_id = f"{config['campaign_id']}-size-{size}-q-{q}"
            target = dataset_part_path(args.output_root, "instances", part_id)
            if target.exists() and not args.no_resume:
                print(f"resume_skip={target}")
                continue
            samples = args.samples_per_cell or (100 + int(100 // (q * q)))
            generated = generate_appendix_g_campaign(
                config["campaign_id"],
                [size],
                [q],
                lambda _q, samples=samples: samples,
                _seed(int(config["seed"]), size, q),
                config["group_scales"],
                float(config["nonzero_probability"]),
                int(config["round_digits"]),
            )
            write_generated_instances(
                args.output_root,
                generated,
                part_id,
                not args.no_resume,
            )
            count += len(generated)
            print(f"heterogeneity size={size} q={q} total={count}")
    return count


def _outside(args, config) -> int:
    count = 0
    sizes = [size for size in config["generated_sizes"] if args.max_size is None or size <= args.max_size]
    samples = args.samples_per_cell or int(config["instances_per_size"])
    for size in sizes:
        q = int(config["q"])
        part_id = f"{config['campaign_id']}-size-{size}-q-{q}"
        target = dataset_part_path(args.output_root, "instances", part_id)
        if target.exists() and not args.no_resume:
            print(f"resume_skip={target}")
            continue
        generated = generate_appendix_g_campaign(
            config["campaign_id"],
            [size],
            [q],
            lambda _q, samples=samples: samples,
            _seed(int(config["seed"]), size, q),
            config["group_scales"],
            float(config["nonzero_probability"]),
            int(config["round_digits"]),
        )
        write_generated_instances(
            args.output_root,
            generated,
            part_id,
            not args.no_resume,
        )
        count += len(generated)
        print(f"outside-base size={size} total={count}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heterogeneity-config", type=Path, default=ROOT / "configs" / "appendix_g_heterogeneity.toml")
    parser.add_argument("--outside-config", type=Path, default=ROOT / "configs" / "appendix_g_outside_option.toml")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "parquet")
    parser.add_argument("--experiment", choices=["heterogeneity", "outside", "both"], default="both")
    parser.add_argument("--max-size", type=int)
    parser.add_argument("--samples-per-cell", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    total = 0
    if args.experiment in {"heterogeneity", "both"}:
        total += _heterogeneity(args, load_config(args.heterogeneity_config))
    if args.experiment in {"outside", "both"}:
        total += _outside(args, load_config(args.outside_config))
    print(f"instances={total}")


if __name__ == "__main__":
    main()
