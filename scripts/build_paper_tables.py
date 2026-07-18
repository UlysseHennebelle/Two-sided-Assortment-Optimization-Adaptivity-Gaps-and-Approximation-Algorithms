"""Build Tables 1-4 from raw Parquet runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from _common import ROOT
from tsao.config import load_config
from tsao.reporting.tables import latex_table, table1, table2, table3, table4
from tsao.storage.parquet import read_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "parquet")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "section7.toml")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "tables")
    args = parser.parse_args()
    runs = read_dataset(args.data_root, "algorithm_runs").to_pandas()
    campaign_id = load_config(args.config)["campaign_id"]
    runs = runs[runs["campaign_id"] == campaign_id]
    instances = read_dataset(args.data_root, "instances").select(["instance_id", "num_customers"]).to_pandas()
    runs = runs.merge(instances, on="instance_id", how="left").rename(columns={"num_customers": "market_size"})
    tables = {
        "table1": table1(runs),
        "table2": table2(runs),
        "table3": table3(runs),
        "table4": table4(runs),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        storage_frame = frame.reset_index()
        storage_frame.columns = [
            "__".join(str(part) for part in column if str(part)) if isinstance(column, tuple) else str(column)
            for column in storage_frame.columns
        ]
        pq.write_table(
            pa.Table.from_pandas(storage_frame, preserve_index=False),
            args.output / f"{name}.parquet",
            compression="zstd",
        )
        (args.output / f"{name}.tex").write_text(latex_table(frame, name.title(), f"tab:{name}"), encoding="utf-8")
        print(f"wrote={args.output / (name + '.parquet')}")


if __name__ == "__main__":
    main()
