"""Arrow schemas for all authoritative numerical datasets."""

from __future__ import annotations

import pyarrow as pa

SCHEMA_VERSION = 1

INSTANCE_SCHEMA = pa.schema(
    [
        ("schema_version", pa.int16()),
        ("instance_id", pa.string()),
        ("campaign_id", pa.string()),
        ("experiment", pa.string()),
        ("replicate", pa.int32()),
        ("generation_seed", pa.uint64()),
        ("num_customers", pa.int32()),
        ("num_suppliers", pa.int32()),
        ("v_flat", pa.large_list(pa.float64())),
        ("w_flat", pa.large_list(pa.float64())),
        ("customer_outside", pa.large_list(pa.float64())),
        ("supplier_outside", pa.large_list(pa.float64())),
        ("customer_capacities", pa.large_list(pa.int32())),
        ("supplier_capacities", pa.large_list(pa.int32())),
        ("checksum", pa.string()),
        ("parameters_json", pa.string()),
        ("metadata_json", pa.string()),
        ("created_at_utc", pa.timestamp("us", tz="UTC")),
    ]
)

SCENARIO_SCHEMA = pa.schema(
    [
        ("schema_version", pa.int16()),
        ("scenario_id", pa.string()),
        ("instance_id", pa.string()),
        ("campaign_id", pa.string()),
        ("outside_option", pa.float64()),
        ("q", pa.int32()),
        ("parameters_json", pa.string()),
    ]
)

ALGORITHM_RUN_SCHEMA = pa.schema(
    [
        ("schema_version", pa.int16()),
        ("run_id", pa.string()),
        ("campaign_id", pa.string()),
        ("instance_id", pa.string()),
        ("scenario_id", pa.string()),
        ("algorithm", pa.string()),
        ("initiating_side", pa.string()),
        ("status", pa.string()),
        ("value", pa.float64()),
        ("incumbent", pa.float64()),
        ("best_bound", pa.float64()),
        ("relative_gap", pa.float64()),
        ("runtime_seconds", pa.float64()),
        ("algorithm_seed", pa.uint64()),
        ("solver_seed", pa.uint64()),
        ("solver_name", pa.string()),
        ("solver_version", pa.string()),
        ("config_hash", pa.string()),
        ("metadata_json", pa.string()),
        ("created_at_utc", pa.timestamp("us", tz="UTC")),
    ]
)

REPLICATION_SCHEMA = pa.schema(
    [
        ("schema_version", pa.int16()),
        ("run_id", pa.string()),
        ("replication", pa.int32()),
        ("initiating_side", pa.string()),
        ("simulation_seed", pa.uint64()),
        ("matches", pa.float64()),
    ]
)

SOLUTION_SCHEMA = pa.schema(
    [
        ("schema_version", pa.int16()),
        ("run_id", pa.string()),
        ("solution_type", pa.string()),
        ("selected_edges_flat", pa.large_list(pa.bool_())),
        ("num_customers", pa.int32()),
        ("num_suppliers", pa.int32()),
        ("metadata_json", pa.string()),
    ]
)

SUMMARY_SCHEMA = pa.schema(
    [
        ("schema_version", pa.int16()),
        ("campaign_id", pa.string()),
        ("artifact", pa.string()),
        ("market_size", pa.int32()),
        ("metric", pa.string()),
        ("minimum", pa.float64()),
        ("mean", pa.float64()),
        ("maximum", pa.float64()),
        ("sample_size", pa.int32()),
        ("parameters_json", pa.string()),
    ]
)
