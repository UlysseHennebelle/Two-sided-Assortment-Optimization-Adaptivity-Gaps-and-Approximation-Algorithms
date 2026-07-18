# Two-sided Assortment Optimization

Corrected, reproducible implementation of the algorithms and numerical
experiments in *Two-sided Assortment Optimization: Adaptivity Gaps and
Approximation Algorithms*.

The package follows the paper rather than the historical notebook layout.
Historical artifacts remain immutable under `legacy/`; new instances and
results are written only as Parquet datasets under `data/parquet/`.

## Environment

```powershell
conda env update -n gurobi-env -f environment.yml
conda run -n gurobi-env python scripts/check_environment.py
conda run -n gurobi-env python -m pytest
```

## Small smoke run

Development checks deliberately use only markets of size at most 3 and at most
10 stochastic replications:

```powershell
conda run -n gurobi-env python scripts/validate_corrected_project.py
```

## Paper jobs

```powershell
conda run -n gurobi-env python scripts/generate_section7_instances.py
conda run -n gurobi-env python scripts/run_section7_fully_static.py
conda run -n gurobi-env python scripts/run_section7_adaptivity.py
conda run -n gurobi-env python scripts/generate_appendix_g_instances.py
conda run -n gurobi-env python scripts/run_appendix_g_heterogeneity.py
conda run -n gurobi-env python scripts/run_appendix_g_outside_option.py
conda run -n gurobi-env python scripts/build_paper_tables.py
conda run -n gurobi-env python scripts/build_paper_figures.py
```

The default configurations describe the complete paper campaigns. They are not
executed as part of tests. See `docs/experiment_protocol.md` before launching a
full campaign.

## Documentation

- `docs/paper_code_map.md`: paper section to source/script/output mapping.
- `docs/algorithm_implementation.md`: detailed algorithm and DP implementation notes.
- `docs/corrections.md`: open correction ledger for the revised paper.
- `docs/experiment_protocol.md`: exact generation and evaluation protocol.
- `docs/parquet_schema.md`: data dictionary.
