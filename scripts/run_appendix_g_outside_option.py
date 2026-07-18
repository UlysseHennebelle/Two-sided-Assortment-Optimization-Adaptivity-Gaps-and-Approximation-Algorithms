"""Run Figure 4 algorithms on each outside-option scenario."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import ROOT, bounded_instances
from tsao.config import config_hash, load_config
from tsao.experiments.appendix_g_outside_option import outside_scenario_id, run_appendix_g_outside_option
from tsao.storage.parquet import dataset_part_path, iter_generated_instances, write_dataset_part
from tsao.storage.schemas import ALGORITHM_RUN_SCHEMA, REPLICATION_SCHEMA, SCENARIO_SCHEMA, SCHEMA_VERSION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "appendix_g_outside_option.toml")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "parquet")
    parser.add_argument("--max-size", type=int)
    parser.add_argument("--max-instances", type=int)
    parser.add_argument("--max-scenarios", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    digest = config_hash(config)
    source = iter_generated_instances(args.output_root, config["campaign_id"], "appendix_g")
    count = 0
    values = config["outside_option_values"][: args.max_scenarios]
    for generated in bounded_instances(source, args.max_size, args.max_instances):
        scenario_records = []
        for outside_option in values:
            scenario_id = outside_scenario_id(generated.instance_id, float(outside_option))
            part_id = f"{generated.instance_id}-{scenario_id}-appendix-g-outside"
            if dataset_part_path(args.output_root, "algorithm_runs", part_id).exists() and not args.no_resume:
                print(f"resume_skip={generated.instance_id} scenario={scenario_id}")
                continue
            scenario_records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "scenario_id": scenario_id,
                    "instance_id": generated.instance_id,
                    "campaign_id": generated.campaign_id,
                    "outside_option": float(outside_option),
                    "q": int(config["q"]),
                    "parameters_json": json.dumps({"outside_option": outside_option, "q": config["q"]}, sort_keys=True),
                }
            )
            runs, replications = run_appendix_g_outside_option(
                generated, float(outside_option), config, digest
            )
            write_dataset_part(args.output_root, "algorithm_runs", runs, ALGORITHM_RUN_SCHEMA, part_id, not args.no_resume)
            if replications:
                write_dataset_part(args.output_root, "simulation_replications", replications, REPLICATION_SCHEMA, part_id, not args.no_resume)
        if scenario_records:
            write_dataset_part(
                args.output_root,
                "scenarios",
                scenario_records,
                SCENARIO_SCHEMA,
                f"{generated.instance_id}-outside-scenarios",
                not args.no_resume,
            )
        count += 1
        print(f"completed={count} instance={generated.instance_id} scenarios={len(values)}")


if __name__ == "__main__":
    main()
