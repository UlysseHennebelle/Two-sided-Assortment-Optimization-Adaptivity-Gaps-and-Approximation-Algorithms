"""Generate Section 7 matrices and store them as Parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import ROOT
from tsao.config import load_config
from tsao.generation.section7 import generate_section7_campaign
from tsao.storage.parquet import dataset_part_path, write_generated_instances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "section7.toml")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "parquet")
    parser.add_argument("--max-size", type=int)
    parser.add_argument("--instances-per-size", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    sizes = [size for size in config["all_sizes"] if args.max_size is None or size <= args.max_size]
    count = args.instances_per_size or int(config["instances_per_size"])
    part_id = f"{config['campaign_id']}-sizes-{'-'.join(str(size) for size in sizes)}"
    target = dataset_part_path(args.output_root, "instances", part_id)
    if target.exists() and not args.no_resume:
        print(f"resume_skip={target}")
        return
    generated = generate_section7_campaign(
        config["campaign_id"], sizes, count, int(config["seed"]), int(config["round_digits"])
    )
    path = write_generated_instances(
        args.output_root,
        generated,
        part_id,
        not args.no_resume,
    )
    print(f"instances={len(generated)} path={path}")


if __name__ == "__main__":
    main()
