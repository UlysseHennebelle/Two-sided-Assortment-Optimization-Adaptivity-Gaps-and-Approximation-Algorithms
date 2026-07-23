"""Synthetic instance generation for Section 7.

Section 7 uses balanced MNL markets with customer weights sampled uniformly on
``[0,1]`` and supplier weights sampled from ``Exp(1)``. Base draws are rounded
to two decimal places.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..instance import MarketInstance


@dataclass(frozen=True, slots=True)
class GeneratedInstance:
    instance_id: str
    campaign_id: str
    experiment: str
    replicate: int
    generation_seed: int
    instance: MarketInstance
    parameters: Mapping[str, Any]
    master_seed: int | None = None


def stable_instance_id(
    campaign_id: str,
    replicate: int,
    seed: int,
    num_customers: int,
    num_suppliers: int,
    q: int | None = None,
) -> str:
    """Return an identifier derived from deterministic campaign coordinates."""

    payload = (
        f"{campaign_id}|n={num_customers}|m={num_suppliers}|q={q}|"
        f"replicate={replicate}|seed={seed}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def section7_generation_seed(master_seed: int, size: int, replicate: int) -> int:
    """Derive a stable instance seed from its campaign coordinates."""

    if master_seed < 0:
        raise ValueError("master_seed must be nonnegative")
    if size <= 0:
        raise ValueError("size must be positive")
    if replicate < 0:
        raise ValueError("replicate must be nonnegative")
    return int(
        np.random.SeedSequence([master_seed, size, replicate]).generate_state(1, dtype=np.uint64)[0]
    )


def generate_section7_instance(
    num_customers: int,
    num_suppliers: int,
    seed: int,
    round_digits: int = 2,
) -> MarketInstance:
    """Generate one Section 7 instance with an explicit seed."""

    if num_customers <= 0 or num_suppliers <= 0:
        raise ValueError("market dimensions must be positive")
    rng = np.random.default_rng(seed)
    v = np.round(rng.uniform(0.0, 1.0, size=(num_customers, num_suppliers)), round_digits)
    w = np.round(rng.exponential(1.0, size=(num_suppliers, num_customers)), round_digits)
    return MarketInstance(
        v,
        w,
        metadata={
            "experiment": "section7",
            "customer_distribution": "uniform_0_1",
            "supplier_distribution": "exponential_rate_1",
            "round_digits": round_digits,
            "generation_seed": seed,
        },
    )


def generate_section7_campaign(
    campaign_id: str,
    sizes: list[int],
    instances_per_size: int,
    seed: int,
    round_digits: int = 2,
) -> list[GeneratedInstance]:
    """Generate the configured balanced Section 7 campaign."""

    if instances_per_size <= 0:
        raise ValueError("instances_per_size must be positive")
    if len(set(sizes)) != len(sizes):
        raise ValueError("sizes must not contain duplicates")
    generated: list[GeneratedInstance] = []
    for size in sizes:
        for replicate in range(instances_per_size):
            child_seed = section7_generation_seed(seed, size, replicate)
            instance = generate_section7_instance(size, size, child_seed, round_digits)
            generated.append(
                GeneratedInstance(
                    stable_instance_id(campaign_id, replicate, child_seed, size, size),
                    campaign_id,
                    "section7",
                    replicate,
                    child_seed,
                    instance,
                    {"size": size, "round_digits": round_digits, "master_seed": seed},
                    seed,
                )
            )
    return generated
