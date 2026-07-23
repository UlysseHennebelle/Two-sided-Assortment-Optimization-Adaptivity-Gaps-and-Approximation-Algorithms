"""Parquet schemas for generated instances and final numerical results."""

from __future__ import annotations

import pyarrow as pa

INSTANCE_SCHEMA = pa.schema(
    [
        ("instance_id", pa.string()),
        ("campaign_id", pa.string()),
        ("experiment", pa.string()),
        ("replicate", pa.int32()),
        ("master_seed", pa.uint64()),
        ("generation_seed", pa.uint64()),
        ("num_customers", pa.int32()),
        ("num_suppliers", pa.int32()),
        ("q", pa.int32()),
        ("v_flat", pa.large_list(pa.float64())),
        ("w_flat", pa.large_list(pa.float64())),
        ("customer_outside", pa.large_list(pa.float64())),
        ("supplier_outside", pa.large_list(pa.float64())),
        ("customer_capacities", pa.large_list(pa.int32())),
        ("supplier_capacities", pa.large_list(pa.int32())),
    ]
)

RESULT_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("campaign_id", pa.string()),
        ("instance_id", pa.string()),
        ("scenario_id", pa.string()),
        ("market_size", pa.int32()),
        ("q", pa.int32()),
        ("outside_option", pa.float64()),
        ("algorithm", pa.string()),
        ("status", pa.string()),
        ("value", pa.float64()),
        ("incumbent", pa.float64()),
        ("best_bound", pa.float64()),
        ("relative_gap", pa.float64()),
        ("runtime_seconds", pa.float64()),
        ("algorithm_seed", pa.uint64()),
        ("solver_seed", pa.uint64()),
    ]
)
