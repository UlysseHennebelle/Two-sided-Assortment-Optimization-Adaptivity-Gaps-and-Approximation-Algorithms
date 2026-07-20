"""Build the paper's four numerical tables from final result values."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsao.config import load_config  # noqa: E402
from tsao.reporting.tables import paper_tabular, table1, table2, table3, table4  # noqa: E402
from tsao.storage.parquet import read_results  # noqa: E402


TABLE_NAMES = ("table1", "table2", "table3", "table4")


def _flatten(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.reset_index()
    result.columns = [
        "__".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in result.columns
    ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.toml")
    parser.add_argument("--results", type=Path, default=ROOT / "data" / "results.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "tables")
    parser.add_argument("--tables", nargs="+", choices=TABLE_NAMES, default=list(TABLE_NAMES))
    args = parser.parse_args()

    section = load_config(args.config)["section7"]
    runs = read_results(args.results).to_pandas()
    runs = runs[runs["campaign_id"] == section["campaign_id"]]
    fully_static_sizes = [int(size) for size in section["fully_static_sizes"]]
    adaptivity_sizes = [int(size) for size in section["adaptivity_sizes"]]
    expected_counts = {size: int(section["instances_per_size"]) for size in fully_static_sizes}
    builders: dict[str, Callable[[], pd.DataFrame]] = {
        "table1": lambda: table1(runs, fully_static_sizes, expected_counts),
        "table2": lambda: table2(runs, adaptivity_sizes),
        "table3": lambda: table3(runs, adaptivity_sizes),
        "table4": lambda: table4(runs, adaptivity_sizes),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    for name in args.tables:
        frame = builders[name]()
        pq.write_table(
            pa.Table.from_pandas(_flatten(frame), preserve_index=False),
            args.output / f"{name}.parquet",
            compression="zstd",
        )
        (args.output / f"{name}.tex").write_text(
            paper_tabular(name, frame),
            encoding="utf-8",
        )
        print(f"table={args.output / (name + '.parquet')}")


if __name__ == "__main__":
    main()
