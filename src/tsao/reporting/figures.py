"""Build Figures 3 and 4 from final algorithm values."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ALGORITHM_COLORS = {"ALG_FS": "tab:blue", "ALG_OS": "tab:orange", "ALG_OA": "tab:green"}


def _panel_axes(count: int):
    rows = 2
    columns = (count + rows - 1) // rows
    figure, axes = plt.subplots(rows, columns, figsize=(5.0 * columns, 4.0 * rows), squeeze=False)
    return figure, list(axes.ravel())


def heterogeneity_figure(data: pd.DataFrame, sizes: Sequence[int], output: str | Path) -> Path:
    """Plot mean matches by ``q`` for each requested market size."""

    figure, axes = _panel_axes(len(sizes))
    for axis, size in zip(axes, sizes, strict=False):
        subset = data[data["market_size"] == size]
        grouped = subset.groupby(["q", "algorithm"], as_index=False)["value"].mean()
        for algorithm, color in ALGORITHM_COLORS.items():
            series = grouped[grouped["algorithm"] == algorithm].sort_values("q")
            axis.plot(series["q"], series["value"], label=algorithm, color=color)
        axis.set(title=f"m = {size}, n = {size}", xlabel="q", ylabel="matches")
        axis.legend()
    for axis in axes[len(sizes) :]:
        axis.set_visible(False)
    figure.tight_layout()
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return target


def outside_option_figure(data: pd.DataFrame, sizes: Sequence[int], output: str | Path) -> Path:
    """Plot mean matches by outside-option value on logarithmic axes."""

    figure, axes = _panel_axes(len(sizes))
    for axis, size in zip(axes, sizes, strict=False):
        subset = data[data["market_size"] == size]
        grouped = subset.groupby(["outside_option", "algorithm"], as_index=False)["value"].mean()
        for algorithm, color in ALGORITHM_COLORS.items():
            series = grouped[grouped["algorithm"] == algorithm].sort_values("outside_option")
            axis.plot(series["outside_option"], series["value"], label=algorithm, color=color)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=2)
        axis.set(title=f"m = {size}, n = {size}", xlabel=r"$v_0$", ylabel="matches")
        axis.legend()
    for axis in axes[len(sizes) :]:
        axis.set_visible(False)
    figure.tight_layout()
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return target
