from gateway.continuity import select_continuity_history, select_persisted_transcript


def test_select_continuity_history_prefers_stored_history():
    stored = [{"role": "user", "content": "stored"}]
    fallback = [{"role": "user", "content": "fallback"}]

    assert select_continuity_history(stored, fallback) == stored


def test_select_continuity_history_falls_back_when_stored_history_empty():
    fallback = [
        {"role": "user", "content": "The job is currently in an error state."},
        {
            "role": "assistant",
            "content": "Would you like me to investigate what went wrong or help you fix/remove it?",
        },
    ]

    assert select_continuity_history([], fallback) == fallback


def test_select_continuity_history_returns_empty_when_no_history_available():
    assert select_continuity_history([], None) == []


def test_select_persisted_transcript_prefers_longer_fallback_store():
    sqlite_history = [{"role": "user", "content": "new only"}]
    jsonl_history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "prior answer"},
        {"role": "user", "content": "new only"},
    ]

    assert select_persisted_transcript(sqlite_history, jsonl_history) == jsonl_history


def test_select_persisted_transcript_prefers_primary_on_tie():
    sqlite_history = [{"role": "user", "content": "sqlite"}]
    jsonl_history = [{"role": "user", "content": "jsonl"}]

    assert select_persisted_transcript(sqlite_history, jsonl_history) == sqlite_history
