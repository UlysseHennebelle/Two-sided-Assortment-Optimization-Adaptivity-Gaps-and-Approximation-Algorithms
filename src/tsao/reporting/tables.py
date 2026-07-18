"""Derive Tables 1-4 from raw Parquet run records."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def _wide_values(runs: pd.DataFrame) -> pd.DataFrame:
    required = {"instance_id", "market_size", "algorithm", "value", "incumbent", "best_bound", "runtime_seconds"}
    missing = required - set(runs.columns)
    if missing:
        raise ValueError(f"Missing run columns: {sorted(missing)}")
    frame = runs.copy()
    frame["reported_value"] = frame["value"]
    opt_fs = frame["algorithm"] == "OPT_FS"
    frame.loc[opt_fs, "reported_value"] = frame.loc[opt_fs, "best_bound"].fillna(frame.loc[opt_fs, "incumbent"])
    return frame.pivot_table(
        index=["instance_id", "market_size"],
        columns="algorithm",
        values="reported_value",
        aggfunc="first",
    )


def _ratio_summary(wide: pd.DataFrame, numerator: str, denominator: str) -> pd.DataFrame:
    subset = wide[[numerator, denominator]].dropna()
    ratios = (subset[numerator] / subset[denominator]).replace([np.inf, -np.inf], np.nan).dropna()
    grouped = ratios.groupby(level="market_size")
    return grouped.agg(["min", "mean", "max", "count"])


def table1(runs: pd.DataFrame) -> pd.DataFrame:
    """Return ALG(FS)/OPT(FS) and OPT(FS) timing statistics."""

    wide = _wide_values(runs)
    ratios = _ratio_summary(wide, "ALG_FS", "OPT_FS")
    times = runs[runs["algorithm"] == "OPT_FS"].groupby("market_size")["runtime_seconds"].agg(["mean", "max"])
    ratios.columns = ["ratio_min", "ratio_mean", "ratio_max", "solved_instances"]
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


def ratio_table(runs: pd.DataFrame, definitions: Mapping[str, tuple[str, str]]) -> pd.DataFrame:
    """Build a min/mean/max table for named numerator/denominator pairs."""

    wide = _wide_values(runs)
    pieces = []
    for label, (numerator, denominator) in definitions.items():
        summary = _ratio_summary(wide, numerator, denominator)
        summary.columns = pd.MultiIndex.from_product([[label], summary.columns])
        pieces.append(summary)
    return pd.concat(pieces, axis=1).sort_index()


def table2(runs: pd.DataFrame) -> pd.DataFrame:
    return ratio_table(runs, TABLE2_RATIOS)


def table3(runs: pd.DataFrame) -> pd.DataFrame:
    return ratio_table(runs, TABLE3_RATIOS)


def table4(runs: pd.DataFrame) -> pd.DataFrame:
    """Return average runtimes for OPT(OA), UB(OA), OPT(FA), and UB(FA)."""

    selected = runs[runs["algorithm"].isin(["OPT_OA", "UB_OA", "OPT_FA", "UB_FA"])]
    return selected.pivot_table(
        index="market_size",
        columns="algorithm",
        values="runtime_seconds",
        aggfunc="mean",
    ).sort_index()


def latex_table(frame: pd.DataFrame, caption: str, label: str) -> str:
    """Render a presentation artifact; numeric source data remain Parquet."""

    return frame.to_latex(float_format=lambda value: f"{value:.3f}", caption=caption, label=label)
