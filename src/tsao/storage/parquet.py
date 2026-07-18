"""Atomic, resumable Parquet storage for generated instances and results."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from ..generation.section7 import GeneratedInstance
from ..instance import MarketInstance
from .schemas import INSTANCE_SCHEMA, SCHEMA_VERSION


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def instance_record(generated: GeneratedInstance) -> dict[str, Any]:
    """Convert a generated instance to the canonical Arrow record."""

    instance = generated.instance
    return {
        "schema_version": SCHEMA_VERSION,
        "instance_id": generated.instance_id,
        "campaign_id": generated.campaign_id,
        "experiment": generated.experiment,
        "replicate": generated.replicate,
        "generation_seed": generated.generation_seed,
        "num_customers": instance.num_customers,
        "num_suppliers": instance.num_suppliers,
        "v_flat": instance.v.ravel(order="C").tolist(),
        "w_flat": instance.w.ravel(order="C").tolist(),
        "customer_outside": instance.customer_outside.tolist(),
        "supplier_outside": instance.supplier_outside.tolist(),
        "customer_capacities": list(instance.customer_capacities),
        "supplier_capacities": list(instance.supplier_capacities),
        "checksum": instance.checksum(),
        "parameters_json": _json(generated.parameters),
        "metadata_json": _json(instance.metadata),
        "created_at_utc": datetime.now(timezone.utc),
    }


def instance_from_record(record: dict[str, Any]) -> tuple[str, MarketInstance]:
    """Reconstruct an instance and verify its checksum."""

    n = int(record["num_customers"])
    m = int(record["num_suppliers"])
    instance = MarketInstance(
        np.asarray(record["v_flat"], dtype=np.float64).reshape((n, m)),
        np.asarray(record["w_flat"], dtype=np.float64).reshape((m, n)),
        customer_outside=record["customer_outside"],
        supplier_outside=record["supplier_outside"],
        customer_capacities=record["customer_capacities"],
        supplier_capacities=record["supplier_capacities"],
        metadata=json.loads(record["metadata_json"]),
    )
    if instance.checksum() != record["checksum"]:
        raise ValueError(f"Checksum mismatch for instance {record['instance_id']}")
    return str(record["instance_id"]), instance


def generated_instance_from_record(record: dict[str, Any]) -> GeneratedInstance:
    """Reconstruct the generation wrapper used by experiment jobs."""

    instance_id, instance = instance_from_record(record)
    return GeneratedInstance(
        instance_id=instance_id,
        campaign_id=str(record["campaign_id"]),
        experiment=str(record["experiment"]),
        replicate=int(record["replicate"]),
        generation_seed=int(record["generation_seed"]),
        instance=instance,
        parameters=json.loads(record["parameters_json"]),
    )


def write_parquet_atomic(path: str | Path, records: Iterable[dict[str, Any]], schema: pa.Schema) -> Path:
    """Write one Parquet file through a same-directory atomic replacement."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(records), schema=schema)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp.parquet")
    try:
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def write_dataset_part(
    root: str | Path,
    dataset: str,
    records: Iterable[dict[str, Any]],
    schema: pa.Schema,
    part_id: str,
    resume: bool = True,
) -> Path:
    """Write an idempotent named part into a Parquet dataset directory."""

    target = Path(root) / dataset / f"part-{part_id}.parquet"
    if resume and target.exists():
        return target
    return write_parquet_atomic(target, records, schema)


def dataset_part_path(root: str | Path, dataset: str, part_id: str) -> Path:
    """Return the exact path used by a named dataset part."""

    return Path(root) / dataset / f"part-{part_id}.parquet"


def read_dataset(root: str | Path, dataset: str) -> pa.Table:
    """Read all Parquet parts in a dataset directory."""

    directory = Path(root) / dataset
    parts = sorted(directory.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No Parquet parts found in {directory}")
    return pa.concat_tables([pq.read_table(part) for part in parts], promote_options="default")


def write_generated_instances(
    root: str | Path,
    generated: Iterable[GeneratedInstance],
    part_id: str,
    resume: bool = True,
) -> Path:
    """Write generated instance matrices and metadata as one dataset part."""

    return write_dataset_part(
        root,
        "instances",
        (instance_record(item) for item in generated),
        INSTANCE_SCHEMA,
        part_id,
        resume,
    )


def iter_generated_instances(
    root: str | Path,
    campaign_id: str | None = None,
    experiment: str | None = None,
) -> Iterable[GeneratedInstance]:
    """Stream generated instances with optional Arrow predicate filters."""

    directory = Path(root) / "instances"
    if not directory.exists():
        raise FileNotFoundError(directory)
    dataset = pads.dataset(directory, format="parquet")
    expression = None
    if campaign_id is not None:
        expression = pads.field("campaign_id") == campaign_id
    if experiment is not None:
        experiment_expression = pads.field("experiment") == experiment
        expression = experiment_expression if expression is None else expression & experiment_expression
    for batch in dataset.to_batches(filter=expression):
        for record in batch.to_pylist():
            yield generated_instance_from_record(record)
