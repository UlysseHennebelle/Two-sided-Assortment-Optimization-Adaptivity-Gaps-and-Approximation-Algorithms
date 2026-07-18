"""Canonical adaptive state corresponding to Appendix A's backlogs.

The legacy DP mutated a ``BiGraph``, undid transitions, hashed a lossy string,
and subtracted already realized matches from cached values. This module keeps
the same state compression explicitly:

- only unprocessed agents and pending choices toward them are retained;
- completed matches and dead choices are discarded;
- DP values are future matches, so realized matches never enter the cache key.
"""

from __future__ import annotations

from dataclasses import dataclass

from .instance import MarketInstance

NONE = -1


def _members(mask: int, size: int) -> tuple[int, ...]:
    return tuple(index for index in range(size) if mask & (1 << index))


@dataclass(frozen=True, slots=True)
class AdaptiveState:
    remaining_customers: int
    remaining_suppliers: int
    customer_pending: tuple[int, ...]
    supplier_pending: tuple[int, ...]

    @classmethod
    def initial(cls, instance: MarketInstance) -> "AdaptiveState":
        return cls(
            remaining_customers=(1 << instance.num_customers) - 1,
            remaining_suppliers=(1 << instance.num_suppliers) - 1,
            customer_pending=(NONE,) * instance.num_customers,
            supplier_pending=(NONE,) * instance.num_suppliers,
        )

    def customers(self) -> tuple[int, ...]:
        return _members(self.remaining_customers, len(self.customer_pending))

    def suppliers(self) -> tuple[int, ...]:
        return _members(self.remaining_suppliers, len(self.supplier_pending))

    def customer_is_remaining(self, customer: int) -> bool:
        return bool(self.remaining_customers & (1 << customer))

    def supplier_is_remaining(self, supplier: int) -> bool:
        return bool(self.remaining_suppliers & (1 << supplier))

    def customer_backlog(self, customer: int) -> tuple[int, ...]:
        return tuple(supplier for supplier, target in enumerate(self.supplier_pending) if target == customer)

    def supplier_backlog(self, supplier: int) -> tuple[int, ...]:
        return tuple(customer for customer, target in enumerate(self.customer_pending) if target == supplier)

    def process_customer(self, customer: int, chosen_supplier: int | None) -> tuple["AdaptiveState", int]:
        """Remove a customer, returning the canonical next state and match reward."""

        if not self.customer_is_remaining(customer):
            raise ValueError(f"customer {customer} is already processed")
        reward = 0
        customer_pending = list(self.customer_pending)
        supplier_pending = list(self.supplier_pending)
        customer_pending[customer] = NONE
        if chosen_supplier is not None:
            if self.supplier_is_remaining(chosen_supplier):
                customer_pending[customer] = chosen_supplier
            elif supplier_pending[chosen_supplier] == customer:
                reward = 1
            else:
                raise ValueError("a processed supplier can be chosen only from the customer's backlog")
        supplier_pending = [NONE if target == customer else target for target in supplier_pending]
        return (
            AdaptiveState(
                self.remaining_customers & ~(1 << customer),
                self.remaining_suppliers,
                tuple(customer_pending),
                tuple(supplier_pending),
            ),
            reward,
        )

    def process_supplier(self, supplier: int, chosen_customer: int | None) -> tuple["AdaptiveState", int]:
        """Remove a supplier, returning the canonical next state and match reward."""

        if not self.supplier_is_remaining(supplier):
            raise ValueError(f"supplier {supplier} is already processed")
        reward = 0
        customer_pending = list(self.customer_pending)
        supplier_pending = list(self.supplier_pending)
        supplier_pending[supplier] = NONE
        if chosen_customer is not None:
            if self.customer_is_remaining(chosen_customer):
                supplier_pending[supplier] = chosen_customer
            elif customer_pending[chosen_customer] == supplier:
                reward = 1
            else:
                raise ValueError("a processed customer can be chosen only from the supplier's backlog")
        customer_pending = [NONE if target == supplier else target for target in customer_pending]
        return (
            AdaptiveState(
                self.remaining_customers,
                self.remaining_suppliers & ~(1 << supplier),
                tuple(customer_pending),
                tuple(supplier_pending),
            ),
            reward,
        )
