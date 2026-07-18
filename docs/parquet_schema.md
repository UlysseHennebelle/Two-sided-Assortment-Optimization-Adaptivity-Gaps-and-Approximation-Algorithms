# Parquet data dictionary

Schema version is currently 1. All files use Zstandard compression.

## `instances`

One row per generated base instance. Matrices are native Arrow large-list
columns in row-major order and are reconstructed using `num_customers` and
`num_suppliers`.

Important fields: `instance_id`, `campaign_id`, `experiment`, `replicate`,
`generation_seed`, `v_flat`, `w_flat`, outside vectors, capacity vectors,
matrix checksum, generation parameters, metadata, and UTC creation time.

## `scenarios`

One row per reuse of a base instance under scenario parameters. Figure 4 stores
`scenario_id`, `instance_id`, `outside_option`, and `q`, without copying the
matrices.

## `algorithm_runs`

One row per algorithm or benchmark execution. It stores `value`, `incumbent`,
`best_bound`, `relative_gap`, status, runtime, initiating side, algorithm and
solver seeds, solver name/version, configuration hash, and method metadata.

For Problem (38), `value` and `incumbent` are the best feasible objective;
`best_bound` is stored separately. Reporting must not silently substitute one
for the other.

## `simulation_replications`

One row per stochastic value retained by ALG(OS), ALG(OA), or another sampled
policy. It contains the parent `run_id`, replication number, initiating side,
child seed, and realized/conditional expected match value.

## `solutions`

Optional detailed solution records. Reciprocal edge matrices use a flattened
Boolean list plus their shape. Policy-specific traces are described by the
metadata field.

## `summaries`

Derived min/mean/max/count records for paper artifacts. Summaries are always
rebuildable from raw run and replication datasets.
