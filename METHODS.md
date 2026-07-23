# Implementation and experiment methods

This document identifies the code that produces the paper's numerical results.
Mathematical definitions and theorem statements are given in *Two-sided
Assortment Optimization: Adaptivity Gaps and Approximation Algorithms*.

## Model

`MarketInstance` stores the customer preference matrix
\(v\in\mathbb{R}_+^{n\times m}\), supplier preference matrix
\(w\in\mathbb{R}_+^{m\times n}\),
outside-option weights, and optional assortment capacities. Instances are
immutable. Exchanging the two sides creates a new instance and never changes
the original object.

MNL choice probabilities and expected demand are implemented in
`src/tsao/choice.py`. The revenue-ordered MNL assortment oracle in
`src/tsao/oracle.py` is shared by the greedy algorithms and dynamic programs.

## Approximation algorithms

- `fully_static.py` implements the Section 5 fully static algorithm. It
  partitions edges into the three weight regions, solves the low-weight
  relaxation, performs independent rounding, constructs both high-weight
  candidates, and returns the largest candidate value.
- `one_sided_static.py` constructs a complete static menu family by simulating
  Algorithm 1 and evaluates each constructed policy once. It compares the two
  possible initiating sides.
- `one_sided_adaptive.py` applies Algorithm 1 with observed choices and compares
  the two possible initiating sides.
- `fully_adaptive.py` implements the policy in Theorem 4.3 and the Section 7
  comparison value.

All stochastic algorithms derive their streams from explicit NumPy seed
sequences. Side changes use `MarketInstance.transposed()`, which returns a new
object.

## Numerical benchmarks

- `opt_fully_static.py` solves the bilinear fully static formulation.
- `opt_one_sided_static.py` computes the one-sided static optimum.
- `dynamic_programs.py` computes the exact one-sided adaptive and fully
  adaptive values with canonical memoized states.
- `ub_one_sided_adaptive.py` and `ub_fully_adaptive.py` solve the upper-bound
  formulations.

Optimization records distinguish feasible values, best bounds, relative gaps,
statuses, and runtimes.

## Experiments

Section 7 uses balanced markets with customer weights sampled uniformly from
\([0,1]\), supplier weights sampled from an exponential distribution with
rate one, and two-decimal base draws. It contains 20 instances at each size
\(2,3,4,5,6,7,8,9,10,11,15,20\). Table 1 uses the configured fully static
sizes \(2,3,4,5,6,8,9\); size 9 is the largest worked size. The adaptivity
tables use sizes \(2,3,4,5,6,7,8,10,15,20\). Exact computations stop at size
4 for `OPT_OS`, size 7 for `OPT_OA`, and size 6 for `OPT_FA`.

Figures 3 and 4 use grouped, sparse preference matrices. Customer prototypes
use exponential draws, supplier prototypes use uniform draws, and agents select
one of \(q\) prototypes on their side. Both figures use market sizes 50, 100,
200, and 500. Figure 3 varies \(q=1,\ldots,10\) and generates
\(100+\lfloor100/q^2\rfloor\) instances for each size and value of \(q\).
Figure 4 generates 50 base instances per size at \(q=10\), then reuses each
base matrix pair across the 10 configured outside-option values.

`scripts/run_experiments.py` is the sole generation and evaluation entry point.
It filters already-computed `(scenario_id, algorithm)` pairs and appends only
missing final values.

## Data artifacts

`data/instances.parquet` contains one row per generated instance. Matrix values
are stored in row-major lists with their dimensions. Campaign coordinates,
generation seeds, and the grouped-instance parameter \(q\) accompany each
instance. The official campaign identifiers are
`section7`, `figure3`, and `figure4`. The file contains 5,052 rows: 240 for
Section 7, 4,612 for Figure 3, and 200 base instances for Figure 4.

`data/results.parquet` contains one row per final algorithm or benchmark value.
It includes the campaign, instance, scenario, algorithm, status, value,
optimization fields, runtime, and relevant seed. Run identifiers are stable
hashes of the evaluation coordinates. The file contains 21,496 final values:
1,660 for Section 7, 13,836 for Figure 3, and 6,000 for Figure 4.

`scripts/make_tables.py` derives Tables 1–4 directly from final results.
`scripts/make_figures.py` derives Figures 3–4 directly from final results.
