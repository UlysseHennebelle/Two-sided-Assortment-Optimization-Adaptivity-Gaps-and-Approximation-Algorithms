"""Theorem 4.3 policy and the confirmed Section 7 FA comparison convention."""

from __future__ import annotations

import numpy as np

from ..instance import MarketInstance
from ..policy import AlgorithmResult, Side
from .arbitrary_order_greedy import arbitrary_order_greedy_customers
from .one_sided_adaptive import one_sided_adaptive_algorithm


def fully_adaptive_theorem_algorithm(
    instance: MarketInstance,
    replications: int = 50,
    seed: int = 0,
) -> AlgorithmResult:
    """Evaluate Theorem 4.3's fair random initiating-side policy."""

    children = np.random.SeedSequence(seed).spawn(replications)
    replication_seeds = tuple(int(child.generate_state(1, dtype=np.uint64)[0]) for child in children)
    values: list[float] = []
    sides: list[str] = []
    transposed = instance.transposed()
    for child in children:
        rng = np.random.default_rng(child)
        customers_first = bool(rng.integers(0, 2) == 0)
        values.append(arbitrary_order_greedy_customers(instance if customers_first else transposed, rng).value)
        sides.append(Side.CUSTOMERS.value if customers_first else Side.SUPPLIERS.value)
    return AlgorithmResult(
        name="ALG_FA_THEOREM",
        value=float(np.mean(values)),
        replications=tuple(values),
        metadata={
            "side_selection": "fair_coin",
            "sampled_sides": tuple(sides),
            "replication_seeds": replication_seeds,
        },
    )


def fully_adaptive_experiment_algorithm(
    instance: MarketInstance,
    replications_per_side: int = 50,
    seed: int = 0,
    oa_result: AlgorithmResult | None = None,
) -> AlgorithmResult:
    """Return the confirmed experimental ALG(FA), which reuses ALG(OA).

    This convention is retained because the author explicitly confirmed it and
    Appendix G states that ALG(OA) and ALG(FA) are the same empirical method.
    The distinct Theorem 4.3 policy remains available above so the revised paper
    can describe the distinction precisely.
    """

    result = oa_result or one_sided_adaptive_algorithm(instance, replications_per_side, seed)
    return AlgorithmResult(
        name="ALG_FA",
        value=result.value,
        initiating_side=result.initiating_side,
        replications=result.replications,
        metadata={**result.metadata, "reuses": "ALG_OA"},
    )
