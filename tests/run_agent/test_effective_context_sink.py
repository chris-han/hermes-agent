from run_agent import AIAgent
from hermes_state import SessionDB


def test_flush_messages_emits_stable_identity_to_optional_ecc_sink(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="test")
    received = []
    agent = AIAgent.__new__(AIAgent)
    agent._session_db = db
    agent._session_db_created = True
    agent.session_id = "s1"
    agent._last_flushed_db_idx = 0
    agent._effective_context_sink = received.append
    agent._apply_persist_user_message_override = lambda messages: None

    AIAgent._flush_messages_to_session_db(
        agent,
        [{"role": "assistant", "content": "persisted", "finish_reason": "stop"}],
        [],
    )

    assert len(received) == 1
    assert received[0]["session_id"] == "s1"
    assert received[0]["role"] == "assistant"
    assert received[0]["content"] == "persisted"
    assert received[0]["message_sequence"] == 1
    assert received[0]["message_identity_status"] == "stable"
