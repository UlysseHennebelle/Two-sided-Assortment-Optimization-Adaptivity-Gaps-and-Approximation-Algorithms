# Paper-to-code map

The table lists every scientific source file and its relationship to the paper.
Package-marker `__init__.py` files contain no logic and are omitted.

## Project and documentation files

| File | Purpose |
|---|---|
| `README.md` | Installation, bounded validation, full job commands, and documentation entry point. |
| `AGENTS.md` | Paper-first implementation rules, corrected-code authority, runtime limits, and immutable legacy boundary. |
| `pyproject.toml` | Installable `src` package and pytest configuration. |
| `environment.yml` | Reproducible Python 3.11/Gurobi/CVXPY/PyArrow environment. |
| `.gitignore` | Excludes caches, temporary files, rendered outputs, and generated Parquet data. |
| `docs/algorithm_implementation.md` | Detailed algorithm, benchmark, and DP implementation record. |
| `docs/corrections.md` | Open correction/choice ledger for the revised paper. |
| `docs/experiment_protocol.md` | Section 7 and Appendix G generation/evaluation contract. |
| `docs/parquet_schema.md` | Dataset and field dictionary. |
| `docs/paper_code_map.md` | This file-level paper mapping. |

| File | Purpose and content | Paper experiment relation |
|---|---|---|
| `src/tsao/config.py` | Load and hash immutable TOML experiment definitions. | All campaigns and artifact provenance. |
| `src/tsao/instance.py` | Immutable `V`, `W`, outside options, capacities, transposition, normalization, checksums. | Section 2; all tables and figures. |
| `src/tsao/choice.py` | MNL probability, demand, expected value, and sampling. | Sections 2 and 5; all experiments. |
| `src/tsao/oracle.py` | Revenue-ordered unconstrained MNL oracle and tiny constrained validator. | Assumptions 2.1/6.1; Algorithm 1. |
| `src/tsao/state.py` | Canonical pending-choice/backlog state for adaptive decisions. | Section 2 policies; Appendix A. |
| `src/tsao/policy.py` | FS/OS/OA/FA and initiating-side identifiers; algorithm results. | Tables 2-3 labels. |
| `src/tsao/simulation.py` | Independent seeded replications and uncertainty summaries. | Algorithm 2; Section 7; Appendix G. |
| `src/tsao/algorithms/fully_static.py` | Empirical Section 5 FS partition, LP rounding, high-value greedy, corrections. | Table 1; Tables 2-3; Figures 3-4. |
| `src/tsao/algorithms/arbitrary_order_greedy.py` | Algorithm 1 with backlog marginal demand. | Section 4; OS/OA/FA computations. |
| `src/tsao/algorithms/one_sided_static.py` | Simulated Algorithm 1 menu construction and fixed-menu evaluation. | ALG(OS), Tables 2-3, Figures 3-4. |
| `src/tsao/algorithms/one_sided_adaptive.py` | Algorithm 1 replications from both sides and side selection. | Algorithm 2 / ALG(OA), Tables 2-3, Figures 3-4. |
| `src/tsao/algorithms/fully_adaptive.py` | Theorem 4.3 policy plus explicitly separate empirical OA-reuse convention. | ALG(FA), Table 3. |
| `src/tsao/algorithms/cardinality.py` | Tiny exact capacity validation. | Section 6; no current numerical artifact. |
| `src/tsao/benchmarks/solver.py` | Normalized Gurobi status/incumbent/bound/gap extraction. | Appendix F and Table 4. |
| `src/tsao/benchmarks/opt_fully_static.py` | Problem (38) and tiny reciprocal-edge enumeration. | OPT(FS), Tables 1-2. |
| `src/tsao/benchmarks/opt_one_sided_static.py` | Exact small static menu enumeration from both sides. | OPT(OS), Table 2. |
| `src/tsao/benchmarks/dynamic_programs.py` | Optimized memoized OPT(OA) and OPT(FA) Bellman recursions. | Appendix A; Tables 2-4. |
| `src/tsao/benchmarks/ub_one_sided_adaptive.py` | Problem (39) in both orientations. | UB(OA), Tables 2-4. |
| `src/tsao/benchmarks/ub_fully_adaptive.py` | Direct Problem (40). | UB(FA), Tables 2-4. |
| `src/tsao/generation/section7.py` | Seeded uniform/exponential balanced markets. | Section 7. |
| `src/tsao/generation/appendix_g.py` | Seeded sparse grouped prototypes and outside scenarios. | Appendix G, Figures 3-4. |
| `src/tsao/storage/schemas.py` | Arrow schemas for every numerical dataset. | Reproducibility of Tables 1-4/Figures 3-4. |
| `src/tsao/storage/parquet.py` | Atomic part writes, streaming reads, checksummed reconstruction. | Entire new data campaign. |
| `src/tsao/experiments/common.py` | Stable run IDs, timers, run/replication records. | All experiment metadata. |
| `src/tsao/experiments/section7_fully_static.py` | Run ALG(FS) and OPT(FS) per instance. | Table 1. |
| `src/tsao/experiments/section7_adaptivity.py` | Run OS/OA/FA algorithms, optima, and bounds. | Tables 2-4. |
| `src/tsao/experiments/appendix_g_heterogeneity.py` | Run FS/OS/OA on one q-structured instance. | Figure 3. |
| `src/tsao/experiments/appendix_g_outside_option.py` | Reuse one base instance for one `v0` scenario. | Figure 4. |
| `src/tsao/reporting/tables.py` | Exact published ratios, timeout-aware summaries, LaTeX rendering. | Tables 1-4. |
| `src/tsao/reporting/figures.py` | Four-panel q and outside-option plots. | Figures 3-4. |

## Configuration and scripts

| File | Purpose | Paper relation |
|---|---|---|
| `configs/section7.toml` | Full main campaign, algorithm counts, solver limits, benchmark cutoffs. | Section 7, Tables 1-4. |
| `configs/appendix_g_heterogeneity.toml` | q grid, sizes, scales, probability, sampling rule. | Figure 3. |
| `configs/appendix_g_outside_option.toml` | q=10 bases and outside grid. | Figure 4. |
| `configs/smoke.toml` | Size-2, two-rep, 5%-gap development campaign. | Validation only. |
| `scripts/check_environment.py` | Gurobi license and dependency smoke solve. | Computational environment. |
| `scripts/generate_section7_instances.py` | Write Section 7 instances to Parquet. | Section 7 inputs. |
| `scripts/run_section7_fully_static.py` | Resumable Table 1 job. | Table 1. |
| `scripts/run_section7_adaptivity.py` | Resumable Tables 2-4 job. | Tables 2-4. |
| `scripts/generate_appendix_g_instances.py` | Batch q/base generation without retaining the full campaign in memory. | Figures 3-4 inputs. |
| `scripts/run_appendix_g_heterogeneity.py` | Resumable Figure 3 runs. | Figure 3. |
| `scripts/run_appendix_g_outside_option.py` | Resumable per-instance/per-v0 runs and scenario rows. | Figure 4. |
| `scripts/build_paper_tables.py` | Write table data as Parquet and presentation as LaTeX. | Tables 1-4. |
| `scripts/build_paper_figures.py` | Write PDF figures from Parquet joins. | Figures 3-4. |
| `scripts/validate_corrected_project.py` | Execute bounded pytest suite. | Scientific validation. |
| `scripts/_common.py` | Source bootstrap and max-size/max-instance guards. | All jobs. |

## Tests

| File | Purpose |
|---|---|
| `tests/test_model_choice_oracle.py` | Instance/MNL/oracle mathematical invariants. |
| `tests/test_policy_state.py` | Pending-choice, backlog, matching, and canonical-state transitions. |
| `tests/test_paper_algorithms.py` | Corrected FS/OS/OA/FA behavior with bounded replications. |
| `tests/test_exact_benchmarks.py` | DP legacy parity, policy nesting, Problems (38)-(40). |
| `tests/test_corrections.py` | Named regression checks for confirmed corrections and choices. |
| `tests/test_parquet_storage.py` | Exact matrix/checksum/schema round-trip. |
| `tests/test_experiment_smoke.py` | Size-2 end-to-end run and Parquet write. |

The root annotated notebook and all files under `legacy/` are support artifacts;
no source module imports them.
