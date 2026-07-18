"""Run Figure 3 algorithms on generated Appendix G instances."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import ROOT, bounded_instances
from tsao.config import config_hash, load_config
from tsao.experiments.appendix_g_heterogeneity import run_appendix_g_heterogeneity
from tsao.storage.parquet import dataset_part_path, iter_generated_instances, write_dataset_part
from tsao.storage.schemas import ALGORITHM_RUN_SCHEMA, REPLICATION_SCHEMA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "appendix_g_heterogeneity.toml")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "parquet")
    parser.add_argument("--max-size", type=int)
    parser.add_argument("--max-instances", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    digest = config_hash(config)
    source = iter_generated_instances(args.output_root, config["campaign_id"], "appendix_g")
    count = 0
    for generated in bounded_instances(source, args.max_size, args.max_instances):
        part_id = f"{generated.instance_id}-appendix-g-heterogeneity"
        if dataset_part_path(args.output_root, "algorithm_runs", part_id).exists() and not args.no_resume:
            print(f"resume_skip={generated.instance_id}")
            continue
        runs, replications = run_appendix_g_heterogeneity(generated, config, digest)
        write_dataset_part(args.output_root, "algorithm_runs", runs, ALGORITHM_RUN_SCHEMA, part_id, not args.no_resume)
        if replications:
            write_dataset_part(args.output_root, "simulation_replications", replications, REPLICATION_SCHEMA, part_id, not args.no_resume)
        count += 1
        print(f"completed={count} instance={generated.instance_id}")


if __name__ == "__main__":
    main()
