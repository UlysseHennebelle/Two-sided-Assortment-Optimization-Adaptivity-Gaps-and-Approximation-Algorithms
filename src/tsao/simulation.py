"""Reproducible replication helpers for stochastic policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True, slots=True)
class SimulationSummary:
    values: tuple[float, ...]
    mean: float
    standard_error: float


def simulate_replications(
    simulation: Callable[[np.random.Generator], float],
    replications: int,
    seed: int,
) -> SimulationSummary:
    """Run independent child RNG streams and retain every realized value."""

    if replications <= 0:
        raise ValueError("replications must be positive")
    seed_sequence = np.random.SeedSequence(seed)
    values = tuple(float(simulation(np.random.default_rng(child))) for child in seed_sequence.spawn(replications))
    standard_error = float(np.std(values, ddof=1) / np.sqrt(replications)) if replications > 1 else 0.0
    return SimulationSummary(values, float(np.mean(values)), standard_error)
