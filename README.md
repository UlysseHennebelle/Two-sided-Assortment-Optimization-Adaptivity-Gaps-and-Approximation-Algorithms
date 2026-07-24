# Two-sided Assortment Optimization

Official computational companion to *Two-sided Assortment Optimization:
Adaptivity Gaps and Approximation Algorithms*.

The repository provides the paper algorithms, the generated experimental
instances, their final numerical values, and the code that builds Tables 1–4
and Figures 3–4.

## Published artifacts

- `data/instances.parquet` — all 5,052 generated market instances, including
  seeds, dimensions, grouped-instance parameter `q`, and matrices.
- `data/results.parquet` — all 21,496 final algorithm and benchmark values,
  including statuses, bounds, gaps, runtimes, and seeds where applicable.
- `tables/` — Tables 1–4 as numerical Parquet files and LaTeX `tabular`
  blocks. Table 1 reports the worked sizes through \(m=n=9\).
- `figures/` — Figures 3 and 4 as PDF files.

## Repository layout

- `src/tsao/algorithms/` implements the approximation algorithms.
- `src/tsao/benchmarks/` implements the exact values and upper bounds.
- `src/tsao/generation/` implements the experiment instance distributions.
- `src/tsao/experiments/` evaluates algorithms and records final values.
- `src/tsao/reporting/` derives the paper figures and tables.
- `scripts/` contains the three public commands.
- `config.toml` is the complete numerical experiment configuration.
- `METHODS.md` maps the implementation to the paper.

The public campaign identifiers are `section7`, `figure3`, and `figure4`.
They are used consistently by the configuration, generated instances, final
results, and resume logic.

## Environment

The project uses Python 3.11 and Gurobi. The supplied environment installs
Gurobi 13, which requires macOS 13 Ventura or newer. Before creating the
environment, verify that your operating system satisfies the
[Gurobi platform requirements](https://docs.gurobi.com/projects/optimizer/en/current/reference/releasenotes/platforms.html)
and the requirements of the other dependencies.

On a supported system, create the supplied Conda environment and install the
package:

```powershell
conda env create -f environment.yml
conda run -n gurobi-env python -m pip install -e .
```

On an older system such as macOS Monterey, create or activate an environment
containing compatible package versions, then install this repository into that
environment:

```bash
python -m pip install -e .
```

The editable installation is required after creating or switching
environments. It registers the `tsao` package and installs the dependencies
declared in `pyproject.toml`. Numerical experiment parameters are loaded from
`config.toml`.

A valid Gurobi license is required for optimization-based algorithms and
benchmarks.

## Reproduce the outputs

The repository already includes the complete instances and results. The runner
skips values present in `data/results.parquet`, so the following commands are
safe to resume after interruption.

```powershell
conda run -n gurobi-env python scripts/run_experiments.py run section7
conda run -n gurobi-env python scripts/run_experiments.py run figure3
conda run -n gurobi-env python scripts/run_experiments.py run figure4
conda run -n gurobi-env python scripts/make_tables.py
conda run -n gurobi-env python scripts/make_figures.py
```

To regenerate configured instances, use:

```powershell
conda run -n gurobi-env python scripts/run_experiments.py generate --experiment all
```

Generation replaces the selected configured campaigns in
`data/instances.parquet` while retaining campaigns not selected by the command.
Algorithm execution can be restricted with `--algorithms`, `--sizes`,
`--max-instances`, and deterministic `--shard-count`/`--shard-index` options.

## Tests

```powershell
conda run -n gurobi-env python -m pytest
```

## Run partial experiments

The runner fills only missing results, so any of these commands can be repeated
to resume an interrupted run. Options such as `--results` must appear before
the `run` subcommand.

Run only `ALG_OA` and `ALG_FA` for selected Section 7 sizes:

```powershell
conda run -n gurobi-env python scripts/run_experiments.py run section7 --algorithms ALG_OA ALG_FA --sizes 2 3 4
```

Run at most five selected Section 7 instances and store the results separately:

```powershell
conda run -n gurobi-env python scripts/run_experiments.py --results data/results_partial.parquet run section7 --algorithms ALG_OA ALG_FA --sizes 2 3 --max-instances 5
```

Run only `ALG_OA` for two Figure 3 sizes:

```powershell
conda run -n gurobi-env python scripts/run_experiments.py run figure3 --algorithms ALG_OA --sizes 50 100
```

Run one deterministic shard of a campaign; use shard indices `0`, `1`, `2`,
and `3` to cover all four shards:

```powershell
conda run -n gurobi-env python scripts/run_experiments.py run section7 --algorithms ALG_OA --shard-count 4 --shard-index 0
```

Build only Tables 2 and 3 from a partial result file:

```powershell
conda run -n gurobi-env python scripts/make_tables.py --results data/results_partial.parquet --output tables/partial --tables table2 table3
```

Existing rows are not overwritten. To recompute an existing algorithm result
with changed parameters, first remove the corresponding rows from the result
artifact or write the new run to a separate result file.
