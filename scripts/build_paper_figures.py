"""Build Figures 3-4 from run, instance, and scenario Parquet data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import ROOT
from tsao.config import load_config
from tsao.reporting.figures import heterogeneity_figure, outside_option_figure
from tsao.storage.parquet import read_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "figures")
    args = parser.parse_args()
    runs = read_dataset(args.data_root, "algorithm_runs").to_pandas()
    instances = read_dataset(args.data_root, "instances").select(
        ["instance_id", "campaign_id", "num_customers", "parameters_json"]
    ).to_pandas()
    instances["q"] = instances["parameters_json"].map(lambda value: json.loads(value).get("q"))
    merged = runs.merge(instances, on=["instance_id", "campaign_id"], how="left").rename(
        columns={"num_customers": "market_size"}
    )
    merged = merged[merged["algorithm"].isin(["ALG_FS", "ALG_OS", "ALG_OA"])]
    args.output.mkdir(parents=True, exist_ok=True)

    heterogeneity_config = load_config(ROOT / "configs" / "appendix_g_heterogeneity.toml")
    heterogeneity = merged[merged["campaign_id"] == heterogeneity_config["campaign_id"]]
    heterogeneity_figure(
        heterogeneity,
        heterogeneity_config["figure_sizes"],
        args.output / "figure3_heterogeneity.pdf",
    )

    outside_config = load_config(ROOT / "configs" / "appendix_g_outside_option.toml")
    outside = merged[merged["campaign_id"] == outside_config["campaign_id"]]
    scenarios = read_dataset(args.data_root, "scenarios").select(["scenario_id", "outside_option"]).to_pandas()
    outside = outside.merge(scenarios, on="scenario_id", how="left")
    outside_option_figure(outside, outside_config["figure_sizes"], args.output / "figure4_outside_option.pdf")
    print(f"figures={args.output}")


if __name__ == "__main__":
    main()
