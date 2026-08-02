import re
import sqlite3

from hermes_state import SessionDB


UUIDV7_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def test_append_message_assigns_stable_uuid7_and_sequence(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="cli")

    db.append_message("s1", role="user", content="one")
    db.append_message("s1", role="assistant", content="two")

    messages = db.get_messages("s1")
    assert [m["message_sequence"] for m in messages] == [1, 2]
    assert [m["message_identity_status"] for m in messages] == ["stable", "stable"]
    assert all(UUIDV7_RE.match(m["message_id"]) for m in messages)


def test_legacy_backfill_is_deterministic_and_marks_timestamp_ties(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.create_session("s1", source="cli")
    first = db.append_message("s1", role="user", content="one", timestamp=1000.0)
    second = db.append_message("s1", role="assistant", content="two", timestamp=1000.0)
    third = db.append_message("s1", role="user", content="three", timestamp=1001.0)
    db.close()

    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE messages SET message_id=NULL, message_sequence=NULL, message_identity_status=NULL"
        )
        conn.execute("UPDATE schema_version SET version=12")

    db = SessionDB(path)
    messages = db.get_messages("s1")

    assert [m["id"] for m in messages] == [first, second, third]
    assert [m["message_sequence"] for m in messages] == [1, 2, 3]
    assert [m["message_identity_status"] for m in messages] == [
        "legacy_timestamp_tie",
        "legacy_timestamp_tie",
        "legacy_backfilled",
    ]
    assert all(UUIDV7_RE.match(m["message_id"]) for m in messages)


def test_replace_messages_preserves_existing_identity_when_available(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="cli")
    db.append_message("s1", role="user", content="one")
    original = db.get_messages("s1")[0]

    db.replace_messages("s1", [original, {"role": "assistant", "content": "two"}])
    messages = db.get_messages("s1")

    assert messages[0]["message_id"] == original["message_id"]
    assert messages[0]["message_sequence"] == original["message_sequence"]
    assert [m["message_sequence"] for m in messages] == [1, 2]
    assert all(UUIDV7_RE.match(m["message_id"]) for m in messages)


def test_archive_and_compact_allocates_new_monotonic_identity(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="cli")
    db.append_message("s1", role="user", content="one")
    db.append_message("s1", role="assistant", content="two")

    db.archive_and_compact(
        "s1",
        [
            {"role": "user", "content": "summary"},
            {"role": "assistant", "content": "continued"},
        ],
    )

    active = db.get_messages("s1")
    all_messages = db.get_messages("s1", include_inactive=True)
    assert [m["message_sequence"] for m in active] == [3, 4]
    assert [m["message_sequence"] for m in all_messages] == [1, 2, 3, 4]
    assert all(UUIDV7_RE.match(m["message_id"]) for m in all_messages)


def test_branch_session_copies_inclusive_stable_transcript_idempotently(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("source", source="cli")
    db.append_message("source", role="user", content="one")
    db.append_message("source", role="assistant", content="two")
    db.append_message("source", role="user", content="later")
    boundary = db.get_messages("source")[1]

    result = db.branch_session_at_message(
        "source",
        "child",
        title="Branch",
        branch_point_message_id=boundary["message_id"],
        branch_point_sequence=boundary["message_sequence"],
    )
    assert result["message_count"] == 2
    child = db.get_messages("child")
    assert [message["content"] for message in child] == ["one", "two"]
    assert [message["message_id"] for message in child] == [
        message["message_id"] for message in db.get_messages("source")[:2]
    ]

    replay = db.branch_session_at_message(
        "source",
        "child",
        title="Branch",
        branch_point_message_id=boundary["message_id"],
        branch_point_sequence=boundary["message_sequence"],
    )
    assert replay["idempotent"] is True
