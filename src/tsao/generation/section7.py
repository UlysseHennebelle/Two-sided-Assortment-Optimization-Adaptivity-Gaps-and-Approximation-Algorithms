"""Synthetic instance generation for Section 7.

Section 7 uses balanced MNL markets with customer weights sampled uniformly on
``[0,1]`` and supplier weights sampled from ``Exp(1)``. Base draws are rounded
to two decimals, matching the confirmed notebook protocol.
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


def stable_instance_id(campaign_id: str, replicate: int, seed: int, instance: MarketInstance) -> str:
    payload = f"{campaign_id}|{replicate}|{seed}|{instance.checksum()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


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

    children = np.random.SeedSequence(seed).spawn(len(sizes) * instances_per_size)
    generated: list[GeneratedInstance] = []
    position = 0
    for size in sizes:
        for replicate in range(instances_per_size):
            child_seed = int(children[position].generate_state(1, dtype=np.uint64)[0])
            position += 1
            instance = generate_section7_instance(size, size, child_seed, round_digits)
            generated.append(
                GeneratedInstance(
                    stable_instance_id(campaign_id, replicate, child_seed, instance),
                    campaign_id,
                    "section7",
                    replicate,
                    child_seed,
                    instance,
                    {"size": size, "round_digits": round_digits},
                )
            )
    return generated
