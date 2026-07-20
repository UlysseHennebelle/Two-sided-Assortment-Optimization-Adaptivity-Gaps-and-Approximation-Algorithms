"""Read and write the two authoritative Parquet artifacts."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from ..generation.section7 import GeneratedInstance
from ..instance import MarketInstance
from .schemas import INSTANCE_SCHEMA, RESULT_SCHEMA


def instance_record(generated: GeneratedInstance) -> dict[str, Any]:
    """Convert a generated instance to its Parquet row."""

    instance = generated.instance
    return {
        "instance_id": generated.instance_id,
        "campaign_id": generated.campaign_id,
        "experiment": generated.experiment,
        "replicate": generated.replicate,
        "master_seed": generated.master_seed,
        "generation_seed": generated.generation_seed,
        "num_customers": instance.num_customers,
        "num_suppliers": instance.num_suppliers,
        "q": generated.parameters.get("q"),
        "v_flat": instance.v.ravel(order="C").tolist(),
        "w_flat": instance.w.ravel(order="C").tolist(),
        "customer_outside": instance.customer_outside.tolist(),
        "supplier_outside": instance.supplier_outside.tolist(),
        "customer_capacities": list(instance.customer_capacities),
        "supplier_capacities": list(instance.supplier_capacities),
        "checksum": instance.checksum(),
    }


def instance_from_record(record: dict[str, Any]) -> MarketInstance:
    """Reconstruct a market instance from one Parquet row."""

    n = int(record["num_customers"])
    m = int(record["num_suppliers"])
    instance = MarketInstance(
        np.asarray(record["v_flat"], dtype=np.float64).reshape((n, m)),
        np.asarray(record["w_flat"], dtype=np.float64).reshape((m, n)),
        np.asarray(record["customer_outside"], dtype=np.float64),
        np.asarray(record["supplier_outside"], dtype=np.float64),
        tuple(record["customer_capacities"]),
        tuple(record["supplier_capacities"]),
    )
    if instance.checksum() != record["checksum"]:
        raise ValueError(f"Checksum mismatch for instance {record['instance_id']}")
    return instance


def generated_instance_from_record(record: dict[str, Any]) -> GeneratedInstance:
    """Reconstruct a generated instance and its campaign coordinates."""

    parameters = {"size": int(record["num_customers"])}
    if record.get("q") is not None:
        parameters["q"] = int(record["q"])
    return GeneratedInstance(
        instance_id=str(record["instance_id"]),
        campaign_id=str(record["campaign_id"]),
        experiment=str(record["experiment"]),
        replicate=int(record["replicate"]),
        generation_seed=int(record["generation_seed"]),
        instance=instance_from_record(record),
        parameters=parameters,
        master_seed=int(record["master_seed"]) if record.get("master_seed") is not None else None,
    )


def _atomic_target(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def write_instances(path: str | Path, generated: Iterable[GeneratedInstance]) -> Path:
    """Write generated instances to one compressed Parquet file."""

    target = Path(path)
    temporary = _atomic_target(target)
    writer = pq.ParquetWriter(temporary, INSTANCE_SCHEMA, compression="zstd")
    try:
        count = 0
        for item in generated:
            writer.write_table(pa.Table.from_pylist([instance_record(item)], schema=INSTANCE_SCHEMA))
            count += 1
        if count == 0:
            writer.write_table(pa.Table.from_pylist([], schema=INSTANCE_SCHEMA))
    finally:
        writer.close()
    os.replace(temporary, target)
    return target


def iter_generated_instances(
    path: str | Path,
    campaign_id: str | None = None,
    experiment: str | None = None,
    min_size: int | None = None,
    max_size: int | None = None,
    exclude_instance_ids: set[str] | frozenset[str] | None = None,
) -> Iterable[GeneratedInstance]:
    """Stream instances with optional campaign and size filters."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    dataset = pads.dataset(source, format="parquet", schema=INSTANCE_SCHEMA)
    expression = None

    def add(term: Any) -> None:
        nonlocal expression
        expression = term if expression is None else expression & term

    if campaign_id is not None:
        add(pads.field("campaign_id") == campaign_id)
    if experiment is not None:
        add(pads.field("experiment") == experiment)
    if min_size is not None:
        add((pads.field("num_customers") >= min_size) & (pads.field("num_suppliers") >= min_size))
    if max_size is not None:
        add((pads.field("num_customers") <= max_size) & (pads.field("num_suppliers") <= max_size))
    if exclude_instance_ids:
        add(~pads.field("instance_id").isin(sorted(exclude_instance_ids)))

    for batch in dataset.to_batches(filter=expression, batch_size=1, use_threads=False):
        for record in batch.to_pylist():
            yield generated_instance_from_record(record)


def read_instances(path: str | Path, columns: Sequence[str] | None = None) -> pa.Table:
    """Read selected columns from the generated-instance artifact."""

    return pq.read_table(path, columns=None if columns is None else list(columns))


def read_results(path: str | Path, columns: Sequence[str] | None = None) -> pa.Table:
    """Read selected columns from the final-result artifact."""

    source = Path(path)
    if not source.exists():
        return pa.Table.from_pylist([], schema=RESULT_SCHEMA).select(columns or RESULT_SCHEMA.names)
    return pq.read_table(source, columns=None if columns is None else list(columns))


def write_results(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    """Write final numerical results in stable run-ID order."""

    target = Path(path)
    rows = list(records)
    rows.sort(key=lambda row: str(row["run_id"]))
    table = pa.Table.from_pylist(rows, schema=RESULT_SCHEMA)
    temporary = _atomic_target(target)
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, target)
    return target


def append_results(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    """Add new results and atomically replace the compact result file."""

    target = Path(path)
    additions = pa.Table.from_pylist(list(records), schema=RESULT_SCHEMA)
    if additions.num_rows == 0:
        return target
    current = read_results(target)
    combined = pa.concat_tables([current, additions], promote_options="default")
    run_ids = combined.column("run_id")
    if pc.count_distinct(run_ids).as_py() != combined.num_rows:
        raise ValueError("Duplicate run_id while appending results")
    return write_results(target, combined.to_pylist())
