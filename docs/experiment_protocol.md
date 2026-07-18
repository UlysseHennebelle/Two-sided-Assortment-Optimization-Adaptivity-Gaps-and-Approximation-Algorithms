# Experiment protocol

## General rules

- All instances are generated from stored 64-bit seeds.
- Base preference draws are rounded to two decimals.
- Every algorithm and simulation receives a recorded child seed.
- All raw matrices, scenarios, runs, replications, and summaries are Parquet.
- A run record contains status, runtime, incumbent, best bound, gap, solver
  version, configuration hash, and algorithm metadata where applicable.
- Full jobs are resumable at an instance/scenario part-file boundary.

## Section 7

- Balanced unconstrained markets.
- Customer weights: `U[0,1]`.
- Supplier weights: `Exp(1)`.
- Common outside option: 1.
- Twenty instances per size.
- Two-hour solver limit.
- Problem (38) target gap: 1%.
- `ALG(FS)`: 50 low-value roundings.
- `ALG(OS)`: 50 constructed policies per side and 50 evaluations per policy.
- `ALG(OA)`: 50 Algorithm 1 replications per side.
- Experimental `ALG(FA)`: the already computed `ALG(OA)` value.

The master instance campaign uses the union of all sizes appearing in Tables
1-4 so overlapping comparisons share the same instance IDs.

## Appendix G heterogeneity

- Generated sizes: 20, 50, 100, 200, 500.
- Displayed sizes by default: 50, 100, 200, 500.
- `q=1,...,10` prototypes per side.
- Samples: `100 + floor(100/q^2)` per size and q.
- Prototype scales: 0.5, 1, 2.
- Customer prototypes: sparse `Exp(1)`.
- Supplier prototypes: sparse `U[0,1]`.
- Each entry is nonzero with probability `sqrt(2)/2`.
- Outside option: 1.

## Appendix G outside option

- Base instances use `q=10` and the same grouped generator.
- Fifty base instances per size.
- Each base matrix is stored once.
- Outside values are the ten log-spaced values from `1/32` to `8` used by the
  notebook's inverse scale grid.
- The same base instance and algorithm seed are used across outside scenarios,
  so changes along a curve are not caused by newly sampled preferences.

## Development limits

The repository's smoke configuration uses size 2, two replications, a 20-second
Gurobi limit, and a 5% target gap. Tests never run an instance larger than 3 or
request more than 10 replications from an algorithmic sampling step.
