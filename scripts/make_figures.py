"""Build the paper's Figures 3 and 4 from final result values."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsao.config import load_config  # noqa: E402
from tsao.reporting.figures import heterogeneity_figure, outside_option_figure  # noqa: E402
from tsao.storage.parquet import read_results  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.toml")
    parser.add_argument("--results", type=Path, default=ROOT / "data" / "results.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "figures")
    args = parser.parse_args()

    config = load_config(args.config)
    runs = read_results(args.results).to_pandas()
    algorithms = ["ALG_FS", "ALG_OS", "ALG_OA"]

    figure3_config = config["figure3"]
    figure3 = runs[
        (runs["campaign_id"] == figure3_config["campaign_id"])
        & runs["algorithm"].isin(algorithms)
    ]
    heterogeneity_figure(
        figure3,
        figure3_config["sizes"],
        args.output / "figure3_heterogeneity.pdf",
    )

    figure4_config = config["figure4"]
    figure4 = runs[
        (runs["campaign_id"] == figure4_config["campaign_id"])
        & runs["algorithm"].isin(algorithms)
    ]
    outside_option_figure(
        figure4,
        figure4_config["sizes"],
        args.output / "figure4_outside_option.pdf",
    )
    print(f"figures={args.output}")


if __name__ == "__main__":
    main()
