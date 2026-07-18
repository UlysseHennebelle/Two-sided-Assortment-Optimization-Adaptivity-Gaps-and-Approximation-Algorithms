"""Structured generators for Appendix G sensitivity experiments.

This module follows the confirmed data-generating code rather than the reversed
distribution sentence in the current PDF:

* customer-side ``v`` prototypes use sparse ``Exp(1)`` draws;
* supplier-side ``w`` prototypes use sparse ``U[0,1]`` draws;
* a draw is nonzero with probability ``sqrt(2)/2``;
* each prototype receives one scale from ``{0.5,1,2}``;
* agents independently select one of ``q`` prototypes on their side.

Base draws are rounded before the group scale is applied, as in
``BiGraph.roll_instance_groups``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from ..instance import MarketInstance
from .section7 import GeneratedInstance, stable_instance_id


def _sparse_rounded(
    rng: np.random.Generator,
    distribution: str,
    shape: tuple[int, int],
    nonzero_probability: float,
    round_digits: int,
) -> np.ndarray:
    if distribution == "exponential":
        draws = rng.exponential(1.0, size=shape)
    elif distribution == "uniform":
        draws = rng.uniform(0.0, 1.0, size=shape)
    else:
        raise ValueError(f"Unsupported Appendix G distribution: {distribution}")
    mask = rng.binomial(1, nonzero_probability, size=shape)
    return np.round(mask * draws, round_digits)


def generate_appendix_g_instance(
    num_customers: int,
    num_suppliers: int,
    q: int,
    seed: int,
    group_scales: Sequence[float] = (0.5, 1.0, 2.0),
    nonzero_probability: float = math.sqrt(2.0) / 2.0,
    round_digits: int = 2,
) -> MarketInstance:
    """Generate one grouped Appendix G instance."""

    if q <= 0 or num_customers <= 0 or num_suppliers <= 0:
        raise ValueError("q and market dimensions must be positive")
    if not 0.0 <= nonzero_probability <= 1.0:
        raise ValueError("nonzero_probability must lie in [0,1]")
    scales = np.asarray(group_scales, dtype=np.float64)
    if scales.ndim != 1 or len(scales) == 0 or np.any(scales <= 0):
        raise ValueError("group_scales must be a nonempty positive sequence")
    rng = np.random.default_rng(seed)
    customer_prototypes = _sparse_rounded(
        rng, "exponential", (q, num_suppliers), nonzero_probability, round_digits
    )
    supplier_prototypes = _sparse_rounded(
        rng, "uniform", (q, num_customers), nonzero_probability, round_digits
    )
    customer_prototypes *= rng.choice(scales, size=q)[:, None]
    supplier_prototypes *= rng.choice(scales, size=q)[:, None]
    customer_groups = rng.integers(0, q, size=num_customers)
    supplier_groups = rng.integers(0, q, size=num_suppliers)
    v = customer_prototypes[customer_groups].copy()
    w = supplier_prototypes[supplier_groups].copy()
    return MarketInstance(
        v,
        w,
        metadata={
            "experiment": "appendix_g",
            "q": q,
            "group_scales": tuple(float(item) for item in scales),
            "nonzero_probability": nonzero_probability,
            "customer_distribution": "exponential_rate_1",
            "supplier_distribution": "uniform_0_1",
            "round_digits_before_scale": round_digits,
            "generation_seed": seed,
        },
    )


def generate_appendix_g_campaign(
    campaign_id: str,
    sizes: Sequence[int],
    q_values: Sequence[int],
    samples_for_q: Callable[[int], int],
    seed: int,
    group_scales: Sequence[float] = (0.5, 1.0, 2.0),
    nonzero_probability: float = math.sqrt(2.0) / 2.0,
    round_digits: int = 2,
) -> list[GeneratedInstance]:
    """Generate a grouped campaign with a configurable sample-count rule."""

    jobs = [(size, q, replicate) for size in sizes for q in q_values for replicate in range(int(samples_for_q(q)))]
    children = np.random.SeedSequence(seed).spawn(len(jobs))
    generated: list[GeneratedInstance] = []
    for job, child in zip(jobs, children, strict=True):
        size, q, replicate = job
        child_seed = int(child.generate_state(1, dtype=np.uint64)[0])
        instance = generate_appendix_g_instance(
            size,
            size,
            q,
            child_seed,
            group_scales,
            nonzero_probability,
            round_digits,
        )
        generated.append(
            GeneratedInstance(
                stable_instance_id(campaign_id, replicate, child_seed, instance),
                campaign_id,
                "appendix_g",
                replicate,
                child_seed,
                instance,
                {"size": size, "q": q},
            )
        )
    return generated


def outside_option_scenarios(instance: MarketInstance, values: Sequence[float]) -> tuple[MarketInstance, ...]:
    """Reuse one base matrix pair under several common outside-option values."""

    return tuple(instance.with_outside_option(float(value)) for value in values)
