"""Derive Tables 1-4 from raw Parquet run records."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


RUN_VALUE_COLUMNS = {
    "instance_id",
    "market_size",
    "algorithm",
    "value",
    "incumbent",
    "best_bound",
    "runtime_seconds",
}


def _size_index(runs: pd.DataFrame, market_sizes: list[int] | None) -> pd.Index:
    if market_sizes is None:
        values = sorted(int(value) for value in runs.get("market_size", pd.Series(dtype=int)).dropna().unique())
    else:
        values = sorted(set(int(value) for value in market_sizes))
    return pd.Index(values, name="market_size")


def _wide_values(runs: pd.DataFrame) -> pd.DataFrame:
    missing = RUN_VALUE_COLUMNS - set(runs.columns)
    if missing:
        raise ValueError(f"Missing run columns: {sorted(missing)}")
    if runs.empty:
        return pd.DataFrame(
            index=pd.MultiIndex.from_arrays([[], []], names=["instance_id", "market_size"])
        )
    frame = runs.copy()
    frame["reported_value"] = pd.to_numeric(frame["value"], errors="coerce")
    opt_fs = frame["algorithm"] == "OPT_FS"
    if opt_fs.any():
        opt_values = frame.loc[opt_fs, "best_bound"].fillna(frame.loc[opt_fs, "incumbent"])
        frame.loc[opt_fs, "reported_value"] = pd.to_numeric(opt_values, errors="coerce")
    return frame.pivot_table(
        index=["instance_id", "market_size"],
        columns="algorithm",
        values="reported_value",
        aggfunc="first",
    )


def _ratio_summary(wide: pd.DataFrame, numerator: str, denominator: str) -> pd.DataFrame:
    columns = ["min", "mean", "max", "count"]
    if numerator not in wide.columns or denominator not in wide.columns:
        return pd.DataFrame({column: pd.Series(dtype=float) for column in columns}).rename_axis("market_size")
    subset = wide[[numerator, denominator]].dropna()
    ratios = (subset[numerator] / subset[denominator]).replace([np.inf, -np.inf], np.nan).dropna()
    if ratios.empty:
        return pd.DataFrame({column: pd.Series(dtype=float) for column in columns}).rename_axis("market_size")
    grouped = ratios.groupby(level="market_size")
    return grouped.agg(["min", "mean", "max", "count"])


def table1(
    runs: pd.DataFrame,
    market_sizes: list[int] | None = None,
    expected_counts: Mapping[int, int] | None = None,
) -> pd.DataFrame:
    """Return ALG(FS)/OPT(FS) and OPT(FS) timing statistics."""

    sizes = _size_index(runs, market_sizes)
    wide = _wide_values(runs)
    ratios = _ratio_summary(wide, "ALG_FS", "OPT_FS").reindex(sizes)
    times = (
        runs[runs["algorithm"] == "OPT_FS"]
        .groupby("market_size")["runtime_seconds"]
        .agg(["mean", "max"])
        .reindex(sizes)
    )
    ratios.columns = ["ratio_min", "ratio_mean", "ratio_max", "solved_instances"]
    ratios["solved_instances"] = ratios["solved_instances"].fillna(0).astype("int64")
    if expected_counts is None:
        counts = (
            runs[["instance_id", "market_size"]]
            .drop_duplicates()
            .groupby("market_size")["instance_id"]
            .count()
        )
    else:
        counts = pd.Series(
            {int(size): int(count) for size, count in expected_counts.items()},
            dtype="int64",
        )
    ratios["expected_instances"] = counts.reindex(sizes).fillna(0).astype("int64")
    return ratios.join(times.rename(columns={"mean": "time_mean", "max": "time_max"}))


TABLE2_RATIOS: Mapping[str, tuple[str, str]] = {
    "OPT_OS/OPT_FS": ("OPT_OS", "OPT_FS"),
    "UB_OA/ALG_FS": ("UB_OA", "ALG_FS"),
    "OPT_OA/ALG_OS": ("OPT_OA", "ALG_OS"),
    "UB_OA/ALG_OS": ("UB_OA", "ALG_OS"),
    "OPT_FA/OPT_OA": ("OPT_FA", "OPT_OA"),
    "UB_FA/ALG_OA": ("UB_FA", "ALG_OA"),
}


TABLE3_RATIOS: Mapping[str, tuple[str, str]] = {
    "ALG_OA/OPT_OA": ("ALG_OA", "OPT_OA"),
    "ALG_OA/UB_OA": ("ALG_OA", "UB_OA"),
    "ALG_FA/OPT_FA": ("ALG_FA", "OPT_FA"),
    "ALG_FA/UB_FA": ("ALG_FA", "UB_FA"),
}


def ratio_table(
    runs: pd.DataFrame,
    definitions: Mapping[str, tuple[str, str]],
    market_sizes: list[int] | None = None,
) -> pd.DataFrame:
    """Build a min/mean/max table for named numerator/denominator pairs."""

    wide = _wide_values(runs)
    sizes = _size_index(runs, market_sizes)
    pieces = []
    for label, (numerator, denominator) in definitions.items():
        summary = _ratio_summary(wide, numerator, denominator).reindex(sizes)
        summary["count"] = summary["count"].fillna(0).astype("int64")
        summary.columns = pd.MultiIndex.from_product([[label], summary.columns])
        pieces.append(summary)
    return pd.concat(pieces, axis=1).sort_index()


def table2(runs: pd.DataFrame, market_sizes: list[int] | None = None) -> pd.DataFrame:
    return ratio_table(runs, TABLE2_RATIOS, market_sizes)


def table3(runs: pd.DataFrame, market_sizes: list[int] | None = None) -> pd.DataFrame:
    return ratio_table(runs, TABLE3_RATIOS, market_sizes)


def table4(runs: pd.DataFrame, market_sizes: list[int] | None = None) -> pd.DataFrame:
    """Return average runtimes for OPT(OA), UB(OA), OPT(FA), and UB(FA)."""

    selected = runs[runs["algorithm"].isin(["OPT_OA", "UB_OA", "OPT_FA", "UB_FA"])]
    algorithms = ["OPT_OA", "UB_OA", "OPT_FA", "UB_FA"]
    sizes = _size_index(runs, market_sizes)
    if selected.empty:
        return pd.DataFrame(index=sizes, columns=algorithms, dtype=float)
    return (
        selected.pivot_table(
            index="market_size",
            columns="algorithm",
            values="runtime_seconds",
            aggfunc="mean",
        )
        .reindex(index=sizes, columns=algorithms)
        .sort_index()
    )


def _number(value: object, decimals: int, integer_one: bool = False) -> str:
    """Format one paper table entry, using a dash for unavailable values."""

    if pd.isna(value):
        return "-"
    numeric = float(value)
    if integer_one and abs(numeric - 1.0) < 5e-12:
        return "1"
    return f"{numeric:.{decimals}f}"


def _time(value: object) -> str:
    """Round runtimes to milliseconds and suppress insignificant zeroes."""

    if pd.isna(value):
        return "-"
    rendered = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return rendered if rendered and rendered != "-0" else "0"


def _rows(frame: pd.DataFrame, values: list[list[str]]) -> list[str]:
    return [
        f"        {int(size)} & " + " & ".join(row) + r" \\"
        for size, row in zip(frame.index, values, strict=True)
    ]


def _table1_tabular(frame: pd.DataFrame) -> str:
    values = [
        [
            _number(row["ratio_min"], 2, True),
            _number(row["ratio_mean"], 2, True),
            _number(row["ratio_max"], 2, True),
            _time(row["time_mean"]),
            _time(row["time_max"]),
        ]
        for _, row in frame.iterrows()
    ]
    lines = [
        r"\begin{tabular}{cccc|cc}",
        r"        \toprule",
        r"        \multicolumn{1}{c}{} & \multicolumn{3}{c}{$\alg_\fullyS/\OPT_\fullyS$} & \multicolumn{2}{c}{time $\OPT_\fullyS$} \\ \cmidrule{2-4} \cmidrule{5-6}",
        r"        $m=n$ & min & mean & max & mean & max \\ ",
        r"        \midrule",
        *_rows(frame, values),
        r"        \bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


TABLE2_HEADERS = {
    "OPT_OS/OPT_FS": r"$\OPT_{\onesidedS}/\OPT_{\fullyS}$",
    "UB_OA/ALG_FS": r"$\UB_{\onesidedA}/\alg_{\fullyS}$",
    "OPT_OA/ALG_OS": r"$\OPT_{\onesidedA}/\alg_{\onesidedS}$",
    "UB_OA/ALG_OS": r"$\UB_{\onesidedA}/\alg_{\onesidedS}$",
    "OPT_FA/OPT_OA": r"$\OPT_{\fullyA}/\OPT_{\onesidedA}$",
    "UB_FA/ALG_OA": r"$\UB_{\fullyA}/\alg_{\onesidedA}$",
}


TABLE3_HEADERS = {
    "ALG_OA/OPT_OA": r"$\alg_{\onesidedA}/\OPT_{\onesidedA}$",
    "ALG_OA/UB_OA": r"$\alg_{\onesidedA}/\UB_{\onesidedA}$",
    "ALG_FA/OPT_FA": r"$\alg_{\fullyA}/\OPT_{\fullyA}$",
    "ALG_FA/UB_FA": r"$\alg_{\fullyA}/\UB_{\fullyA}$",
}


def _ratio_tabular(
    frame: pd.DataFrame,
    headers: Mapping[str, str],
    column_specification: str,
) -> str:
    labels = list(headers)
    values = [
        [
            _number(frame.loc[size, (label, statistic)], 2, True)
            for label in labels
            for statistic in ("min", "mean", "max")
        ]
        for size in frame.index
    ]
    group_header = " & ".join(
        [r"\multicolumn{1}{c}{}"]
        + [rf"\multicolumn{{3}}{{c}}{{{headers[label]}}}" for label in labels]
    )
    statistic_header = " & ".join([r"$m=n$"] + ["min", "mean", "max"] * len(labels))
    last_column = 1 + 3 * len(labels)
    lines = [
        rf"\begin{{tabular}}{{{column_specification}}}",
        r"        \toprule",
        f"        {group_header} " + r"\\ " + rf"\cmidrule{{2-{last_column}}}",
        f"        {statistic_header} " + r"\\",
        r"        \midrule",
        *_rows(frame, values),
        r"        \bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def _table4_tabular(frame: pd.DataFrame) -> str:
    values = [
        [_time(row[algorithm]) for algorithm in ("OPT_OA", "UB_OA", "OPT_FA", "UB_FA")]
        for _, row in frame.iterrows()
    ]
    lines = [
        r"\begin{tabular}{cc|c|c|c}",
        r"        \toprule",
        r"        $m=n$ & $\OPT_\onesidedA$ & $\UB_\onesidedA$ & $\OPT_\fullyA$ & $\UB_\fullyA$ \\ ",
        r"        \midrule",
        *_rows(frame, values),
        r"        \bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def paper_tabular(name: str, frame: pd.DataFrame) -> str:
    """Render one tabular block using the notation and layout of the paper."""

    if name == "table1":
        return _table1_tabular(frame)
    if name == "table2":
        return _ratio_tabular(frame, TABLE2_HEADERS, "cccc|ccc|ccc|ccc|ccc|ccc")
    if name == "table3":
        return _ratio_tabular(frame, TABLE3_HEADERS, "cccc|ccc|ccc|ccc")
    if name == "table4":
        return _table4_tabular(frame)
    raise ValueError(f"Unknown paper table: {name}")
