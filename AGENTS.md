# Project Context

This repository is the corrected computational companion to the paper
`paper hennebelle.pdf`, *Two-sided Assortment Optimization: Adaptivity Gaps and
Approximation Algorithms*.

The paper is the primary specification for the model, policy classes,
algorithms, optimization formulations, and experiments. The legacy notebook is
implementation support, not production code.

# Authoritative Sources

- Mathematical definitions and experiment labels: `paper hennebelle.pdf`.
- Open correction decisions and implementation details: `docs/corrections.md`
  and `docs/paper_code_map.md`.
- Immutable historical code and data: `legacy/`.
- Corrected executable implementation: `src/tsao/`.
- New authoritative numerical data: `data/parquet/`.

# Environment

Use the Conda environment `gurobi-env` for every project command. Prefer:

```powershell
conda run -n gurobi-env python <command>
```

The environment must provide Python 3.11, Gurobi, CVXPY, NumPy, pandas,
PyArrow, Matplotlib, and pytest.

# Implementation Rules

- Implement the corrected algorithms only. Do not add a production legacy
  compatibility mode.
- Use the legacy notebook to understand optimized implementations, especially
  the one-sided and fully adaptive dynamic programs, but express the result as
  documented, tested package code.
- Preserve the confirmed empirical choices that are not errors: the notebook's
  alpha, the greedy high-value Fully Static routine, 50 OA simulations per
  initiating side, reuse of ALG(OA) for the experimental ALG(FA) comparison,
  the Appendix G distribution orientation, and two-decimal base draws.
- Correct the double-transposition error in ALG(FS) and the missing final
  transposition in UB(OA). Record additional corrections in
  `docs/corrections.md` and add a named regression test.
- Side changes must be pure transformations. Never mutate an instance into its
  transpose and rely on a later call to restore it.
- Store all newly generated numerical data in Parquet. Do not generate Excel,
  CSV, JSON, or pickle data files. TOML configuration and PDF/PNG/LaTeX report
  artifacts are allowed.
- Store random seeds, solver versions, statuses, incumbents, bounds, gaps,
  configuration hashes, and schema versions.
- Never edit files under `legacy/`.

# Runtime Limits During Development

- Run only instances with `m, n <= 3`.
- Use at most 10 stochastic replications in tests or development checks.
- Use a 5% Gurobi gap for development checks unless an exact tiny enumeration
  is faster.
- Do not run medium or large paper campaigns during implementation.

# Documentation Rules

Each algorithm module and `docs/algorithm_implementation.md` must explain:

1. the corresponding paper algorithm or benchmark;
2. the legacy implementation idea that was retained;
3. state compression, memoization, and other optimizations;
4. corrections and intentional deviations;
5. the Section 7 or Appendix G outputs that consume it.

When a choice is unclear, make the narrowest mathematically coherent choice,
record it in `docs/corrections.md`, and report it to Ulysse.
