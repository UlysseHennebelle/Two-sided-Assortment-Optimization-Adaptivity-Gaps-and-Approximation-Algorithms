# Two-sided Assortment Optimization

Official computational companion to *Two-sided Assortment Optimization:
Adaptivity Gaps and Approximation Algorithms*.

The repository provides the paper algorithms, the generated experimental
instances, their final numerical values, and the code that builds Tables 1–4
and Figures 3–4.

## Published artifacts

- `data/instances.parquet` — all 5,052 generated market instances, including
  seeds, dimensions, grouped-instance parameter `q`, and matrices.
- `data/results.parquet` — all 21,506 final algorithm and benchmark values,
  including statuses, bounds, gaps, runtimes, and seeds where applicable.
- `tables/` — Tables 1–4 as numerical Parquet files and LaTeX `tabular`
  blocks. Table 1 reports the worked sizes through \(m=n=10\).
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

The project requires Python 3.10 or newer. The supplied environment uses
Python 3.11 and Gurobi 13, which requires macOS 13 Ventura or newer.
Collaborators may instead install the package in their own compatible Python
environment. Before creating the supplied environment, verify that your
operating system satisfies the
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

## Generate instances and run experiments

An instance artifact must exist before any algorithm can run. The repository
already includes the published `data/instances.parquet`, so users working with
that artifact can skip generation. When using a new or custom instance path,
generate the instances first:

```powershell
conda run -n gurobi-env python scripts/run_experiments.py generate --experiment all
```

Generation is deterministic. With the same configuration and fixed seeds, it
produces identical instances; it does not draw a new independent sample.
`--experiment all` generates all three campaigns. When the selected instance
artifact already exists, the command replaces those campaigns. Selecting
`section7`, `figure3`, or `figure4` replaces only that campaign and retains the
others.

The command does not modify the result artifact. If generation settings that
affect existing matrices are changed, existing results may no longer describe
the newly generated instances. In that case, use a new result file or remove the
affected result rows before running algorithms.

After the instance artifact exists, run the algorithms and build the reporting
artifacts:

```powershell
conda run -n gurobi-env python scripts/run_experiments.py run section7
conda run -n gurobi-env python scripts/run_experiments.py run figure3
conda run -n gurobi-env python scripts/run_experiments.py run figure4
conda run -n gurobi-env python scripts/make_tables.py
conda run -n gurobi-env python scripts/make_figures.py
```

The runner skips values already present in `data/results.parquet`, so these
commands are safe to resume after interruption.

## Tests

```powershell
conda run -n gurobi-env python -m pytest
```

## Run partial experiments

### Basic behavior

This section assumes that the selected instance artifact already exists, either
because it was generated first or because the published artifact is being used.
The runner reads generated instances from `data/instances.parquet` and writes
final values to `data/results.parquet`. It computes only missing
`(scenario_id, algorithm)` pairs: existing results are never overwritten.
Repeating the same command therefore resumes an interrupted run.

The simplest command runs every configured Section 7 algorithm on every
eligible generated instance:

```powershell
conda run -n gurobi-env python scripts/run_experiments.py run section7
```

The positional experiment name is one of `section7`, `figure3`, or `figure4`.

The repository already contains `data/instances.parquet`, so the default
`run` commands can be used without generating first. The runner never creates
instances automatically:

- if the selected instance path does not exist, `run` raises
  `FileNotFoundError`;
- if the file exists but contains no matching campaign, size, or eligible
  instance, the run completes zero results;
- if the result path does not exist, it is treated as empty and is created when
  the first result batch is written.

### Configuration, instance, and result paths

The three global path options must appear before the `run` or `generate`
subcommand:

- `--config PATH` selects the TOML configuration; the default is
  `config.toml`. It defines campaign identifiers, fixed seeds, generated sizes
  and replication counts, algorithm sampling counts, solver settings,
  reporting grids, and exact-benchmark limits. There are no hidden
  configuration defaults.
- `--instances PATH` selects the instance Parquet artifact; the default is
  `data/instances.parquet`. `generate` writes this artifact, replacing the
  selected campaign and retaining other campaigns. `run` reads it without
  modifying it.
- `--results PATH` selects the result Parquet artifact; the default is
  `data/results.parquet`. `run` reads existing rows, skips completed
  `(scenario_id, algorithm)` pairs, and appends missing rows. For `run`, this
  must be a file path, not a directory.

For example, this uses a custom configuration and keeps both generated
instances and results separate from the published artifacts:

```powershell
conda run -n gurobi-env python scripts/run_experiments.py --config config_custom.toml --instances data/instances_custom.parquet generate --experiment section7
conda run -n gurobi-env python scripts/run_experiments.py --config config_custom.toml --instances data/instances_custom.parquet --results data/results_custom.parquet run section7
```

The same configuration and instance path must be used for generation and the
corresponding run.

### Select algorithms

Use `--algorithms` followed by one or more algorithm names. Section 7 accepts
`ALG_FS`, `OPT_FS`, `ALG_OS`, `OPT_OS`, `ALG_OA`, `OPT_OA`, `UB_OA`, `ALG_FA`,
`OPT_FA`, and `UB_FA`. Figures 3 and 4 accept `ALG_FS`, `ALG_OS`, and `ALG_OA`.

```powershell
conda run -n gurobi-env python scripts/run_experiments.py run section7 --algorithms ALG_OS ALG_OA
```

The runner checks each algorithm separately for each selected instance. If
`ALG_OS` already exists but `ALG_OA` is missing, only `ALG_OA` is run.
`ALG_FA` reuses the matching `ALG_OA` value; when writing a new partial result
file, request `ALG_OA` before `ALG_FA`.

### Select sizes

Use `--sizes` to keep only the listed market sizes:

```powershell
conda run -n gurobi-env python scripts/run_experiments.py run section7 --algorithms ALG_OS ALG_OA --sizes 2 5
```

This option filters instances already present in the instance artifact; it
does not generate new sizes. Section 7 also enforces the configured
`fully_static_sizes`, `adaptivity_sizes`, and exact-benchmark size limits. A
requested size that is absent or ineligible for the requested algorithm
produces no result for that combination.

To add a new size or more generated instances per size, edit `config.toml`,
then run `generate --experiment CAMPAIGN` before running algorithms.

### Limit the number of instances per size

`--max-instances N` selects generated replicate indices `0` through `N-1` for
every selected size. It is a per-size limit, not a total campaign limit.
Repeating the command resumes missing algorithm results within the same fixed
subset; it does not advance to later replicate indices.

```powershell
conda run -n gurobi-env python scripts/run_experiments.py run section7 --algorithms ALG_OS ALG_OA --sizes 2 5 --max-instances 5
```

This selects five size-2 instances and five size-5 instances. With two
algorithms, it can create at most
`2 sizes x 5 instances x 2 algorithms = 20` new result rows. Existing pairs
are skipped and gaps are filled independently: if `ALG_OS` exists for one
instance but `ALG_OA` does not, only the latter is computed.

For Figure 3, replicate indices restart for each `(size, q)` cell, so the limit
is applied separately to every selected `(size, q)` combination.

### Write a partial run to a separate result file

This preserves the official results and writes the partial run to one separate
Parquet file:

```powershell
conda run -n gurobi-env python scripts/run_experiments.py --results data/results_partial.parquet run section7 --algorithms ALG_OS ALG_OA --sizes 2 5 --max-instances 5
```

A new result path starts empty. It is not automatically combined with
`data/results.parquet`. Changing a configuration value also does not
invalidate existing rows: to recompute them, remove the relevant rows first or
write the new run to a different result artifact.

### Build tables from partial results

The table builder reads available Section 7 rows without running algorithms.
For each reported ratio, it uses only instances having both the numerator and
denominator, then computes the per-size minimum, mean, maximum, and count.
Missing comparisons are rendered as `-`.

```powershell
conda run -n gurobi-env python scripts/make_tables.py --results data/results_partial.parquet --output tables/partial --tables table2 table3
```

The result file must contain the required pairs. For example, `ALG_OA` alone
cannot produce `ALG_OA/OPT_OA`; the matching `OPT_OA` rows must also be
present. The output sizes still come from the Section 7 lists in the selected
configuration.

### Run shards in parallel

Sharding is an advanced option for distributing a large run across independent
processes. `--shard-count K` divides the selected instances into `K`
deterministic, disjoint groups, and `--shard-index I` runs group `I`, where
indices range from `0` to `K-1`. Running every index covers the selected
instances exactly once without changing instance or algorithm seeds.

Parallel processes must not write to the same result file. Give every shard a
different Parquet file in one otherwise empty directory. For a simple
two-process run, start these commands at the same time in two terminals:

```powershell
# Terminal 1
conda run -n gurobi-env python scripts/run_experiments.py --results output/section7-shards/shard-0.parquet run section7 --algorithms ALG_OS ALG_OA --sizes 2 5 --max-instances 5 --shard-count 2 --shard-index 0
```

```powershell
# Terminal 2
conda run -n gurobi-env python scripts/run_experiments.py --results output/section7-shards/shard-1.parquet run section7 --algorithms ALG_OS ALG_OA --sizes 2 5 --max-instances 5 --shard-count 2 --shard-index 1
```

Each command can be repeated independently to resume its shard. The
`--max-instances 5` subset is selected before sharding, so both shards together
still cover at most five instances per size, not five per shard.

After all shards finish, pass their directory to the table builder; the
Parquet files are read together as one result dataset:

```powershell
conda run -n gurobi-env python scripts/make_tables.py --results output/section7-shards --output tables/section7-shards --tables table2 table3
```

### Complete public option reference

`scripts/run_experiments.py` has three global options, all described above:
`--config`, `--instances`, and `--results`.

Its `generate` subcommand accepts:

- `--experiment {section7,figure3,figure4,all}`: campaign to replace in the
  instance artifact; the default is `all`.

Its `run` subcommand accepts:

- positional `experiment`: required campaign, one of `section7`, `figure3`, or
  `figure4`;
- `--algorithms NAME ...`: algorithms to run; omitting it selects every
  algorithm supported by the campaign;
- `--sizes N ...`: existing market sizes to select; omitting it selects every
  eligible size;
- `--max-instances N`: fixed number of generated replicates per selected size,
  and per `(size, q)` cell for Figure 3; omitting it selects all generated
  replicates;
- `--shard-count K`: number of deterministic partitions; the default is `1`;
- `--shard-index I`: partition to run, from `0` through `K-1`; the default is
  `0`.

The seeds, number of generated instances, algorithm simulation counts, solver
limits, Figure 3 `q` values, and Figure 4 outside-option values are
configuration fields rather than command-line parameters.

`scripts/make_tables.py` accepts `--config`, `--results`, `--output`, and
`--tables {table1,table2,table3,table4} ...`. By default it reads the published
configuration and results, writes to `tables/`, and builds all four tables.
For reporting only, `--results` may also be a directory containing compatible
shard Parquet files.

`scripts/make_figures.py` accepts `--config`, `--results`, and `--output`. By
default it reads the published configuration and results, writes to
`figures/`, and builds both figures.

Use `python SCRIPT --help` to display the accepted options and defaults for any
command.
