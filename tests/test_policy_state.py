from tsao.instance import MarketInstance
from tsao.state import AdaptiveState, NONE


def test_pending_choice_becomes_match_and_dead_backlogs_are_cleared() -> None:
    instance = MarketInstance([[0.5], [0.8]], [[0.4, 0.7]])
    state = AdaptiveState.initial(instance)
    state, reward = state.process_supplier(0, 1)
    assert reward == 0
    assert state.customer_backlog(1) == (0,)
    state, reward = state.process_customer(1, 0)
    assert reward == 1
    assert state.supplier_pending == (NONE,)


def test_outside_choice_discards_all_backlogs_to_processed_agent() -> None:
    instance = MarketInstance([[0.5], [0.8]], [[0.4, 0.7]])
    state = AdaptiveState.initial(instance)
    state, _ = state.process_supplier(0, 1)
    state, reward = state.process_customer(1, None)
    assert reward == 0
    assert state.supplier_pending == (NONE,)
