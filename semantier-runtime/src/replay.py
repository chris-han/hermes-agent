from events import ExecutionEventStore, ExecutionEventType


def replay():
    store = ExecutionEventStore()
    events = store.list_events()

    state = {
        "committed_keys": set()
    }

    for e in events:
        if e["event_type"] == ExecutionEventType.EXECUTION_COMMITTED.value:
            key = e["payload"]["idempotency_key"]
            state["committed_keys"].add(key)

    return state