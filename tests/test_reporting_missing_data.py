import math

import pandas as pd

from tsao.reporting.tables import RUN_VALUE_COLUMNS, paper_tabular, table1, table2, table3, table4


def _run(instance_id: str, algorithm: str, value: float | None, size: int) -> dict:
    return {
        "instance_id": instance_id,
        "market_size": size,
        "algorithm": algorithm,
        "value": value,
        "incumbent": value if algorithm == "OPT_FS" else None,
        "best_bound": value if algorithm == "OPT_FS" else None,
        "runtime_seconds": 1.0,
    }


def test_all_tables_accept_completely_missing_runs() -> None:
    runs = pd.DataFrame(columns=sorted(RUN_VALUE_COLUMNS))
    first = table1(runs, [2, 3], {2: 2, 3: 1})
    assert list(first.index) == [2, 3]
    assert first.loc[2, "solved_instances"] == 0
    assert first.loc[2, "expected_instances"] == 2
    assert math.isnan(first.loc[2, "ratio_mean"])
    rendered = paper_tabular("table1", first)
    assert "N/A" not in rendered
    assert "-" in rendered
    assert rendered.startswith(r"\begin{tabular}{cccc|cc}")
    assert r"\end{table}" not in rendered
    second = table2(runs, [2, 3])
    third = table3(runs, [2, 3])
    fourth = table4(runs, [2, 3])
    assert list(second.index) == [2, 3]
    assert list(third.index) == [2, 3]
    assert list(fourth.index) == [2, 3]
    assert paper_tabular("table2", second).startswith(
        r"\begin{tabular}{cccc|ccc|ccc|ccc|ccc|ccc}"
    )
    assert paper_tabular("table3", third).startswith(
        r"\begin{tabular}{cccc|ccc|ccc|ccc}"
    )
    assert paper_tabular("table4", fourth).startswith(r"\begin{tabular}{cc|c|c|c}")
    assert "count" not in paper_tabular("table2", second)


def test_table1_reports_available_pairs_and_expected_coverage() -> None:
    runs = pd.DataFrame(
        [
            _run("complete", "ALG_FS", 0.8, 2),
            _run("complete", "OPT_FS", 1.0, 2),
            _run("missing-opt", "ALG_FS", 0.9, 2),
        ]
    )
    result = table1(runs, [2, 3], {2: 2, 3: 0})
    assert result.loc[2, "ratio_mean"] == 0.8
    assert result.loc[2, "solved_instances"] == 1
    assert result.loc[2, "expected_instances"] == 2
    assert math.isnan(result.loc[3, "ratio_mean"])


def test_ratio_tables_accept_an_fs_only_dataset() -> None:
    runs = pd.DataFrame([_run("only", "ALG_FS", 0.8, 2)])
    assert table2(runs, [2]).isna().any().any()
    assert table3(runs, [2]).isna().any().any()
