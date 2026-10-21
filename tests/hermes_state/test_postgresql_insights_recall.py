"""Real hosted insights and recall metadata compared with native SQLite history."""

import json
from uuid import uuid4

import pytest

from tests.hermes_state.test_postgresql_sessions import (
    pg_settings as shared_pg_settings,
)

pg_settings = shared_pg_settings


def _source(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    import agent.insights as insights

    import time

    # The suite pins TZ through env; synchronize libc before either oracle.
    if hasattr(time, "tzset"):
        time.tzset()
    now = 1800000000.0
    monkeypatch.setattr(insights.time, "time", lambda: now)
    db = SessionDB(tmp_path / "source" / "state.db")
    for sid, source, model in [
        ("same", "cli", "model-a"),
        ("included", "cli", "gpt-5.4-mini"),
        ("other-platform", "telegram", "custom-unpriced"),
        ("old", "cli", "custom-unpriced"),
    ]:
        db.create_session(sid, source, model=model)
    db.update_token_counts(
        "same",
        input_tokens=100,
        output_tokens=5,
        model="model-a",
        billing_provider="custom",
        estimated_cost_usd=1.5,
        actual_cost_usd=1.0,
        cost_status="estimated",
        cost_source="provider",
        api_call_count=1,
    )
    db.update_token_counts(
        "included",
        input_tokens=200,
        model="gpt-5.4-mini",
        billing_provider="openai-codex",
        estimated_cost_usd=0.0,
        actual_cost_usd=0.0,
        cost_status="included",
        cost_source="none",
        api_call_count=1,
    )
    db.update_token_counts(
        "other-platform",
        input_tokens=50,
        model="custom-unpriced",
        cost_status="unknown",
        api_call_count=1,
    )
    first = db.append_message(
        "same",
        "assistant",
        "Archived skill call",
        tool_calls=[
            {"function": {"name": "terminal", "arguments": "{}"}},
            {
                "function": {
                    "name": "skill_view",
                    "arguments": json.dumps({"name": "技能"}),
                }
            },
        ],
    )
    second = db.append_message("same", "tool", "terminal result", tool_name="terminal")
    db.append_message(
        "same",
        "assistant",
        "Skill edit",
        tool_calls=[
            {
                "function": {
                    "name": "skill_manage",
                    "arguments": json.dumps({"name": "技能"}),
                }
            }
        ],
    )
    db.append_message("included", "user", "Included subscription request")
    db.append_message("other-platform", "user", "Unknown price request")
    db.append_message("old", "user", "Outside time window")
    for index, sid in enumerate(("same", "included", "other-platform")):
        db._conn.execute(
            "UPDATE sessions SET started_at=?,ended_at=? WHERE id=?",
            (now - index * 100, now - index * 100 + 60, sid),
        )
    db._conn.execute(
        "UPDATE sessions SET started_at=?,ended_at=? WHERE id='old'",
        (now - 45 * 86400, now - 45 * 86400 + 60),
    )
    db._conn.execute("UPDATE messages SET active=0,compacted=1 WHERE id=?", (first,))
    db._conn.execute("UPDATE messages SET active=0,compacted=0 WHERE id=?", (second,))
    db._conn.commit()
    return db, first, second


def _report(report):
    # Tied tools have a set-based source order; the public counts/ranking agree.
    result = dict(report)
    result.pop("generated_at", None)
    result["tools"] = sorted(result["tools"], key=lambda row: row["tool"])
    return result


def test_insights_native_pg_matches_sqlite_source_filters_costs_and_usage(
    tmp_path, pg_settings, monkeypatch, capsys
):
    from agent.insights import InsightsEngine
    from hermes_state_postgresql import (
        PostgreSQLSessionDB,
        SessionScope,
        provision_scope,
    )
    import sqlite3

    source, _, _ = _source(tmp_path, monkeypatch)
    expected = {
        platform: _report(InsightsEngine(source).generate(days=30, source=platform))
        for platform in (None, "cli", "telegram", "missing", "cli' OR 1=1 --")
    }
    breakdown = InsightsEngine(source).get_usage_breakdown()
    scope = SessionScope("insights", "workspace", str(uuid4()))
    provision_scope(pg_settings["semantier_migrate"], scope, app_role="semantier_app")
    with PostgreSQLSessionDB(
        pg_settings["semantier_migrate"], scope, home=tmp_path
    ) as importer:
        importer.import_sqlite_snapshot(source.db_path)
    source.close()
    monkeypatch.setattr(
        sqlite3, "connect", lambda *a, **k: pytest.fail("PG insights opened SQLite")
    )
    with PostgreSQLSessionDB(
        pg_settings["semantier_app"], scope, home=tmp_path, read_only=True
    ) as db:
        assert not hasattr(db, "_conn")
        engine = InsightsEngine(db)
        for platform, oracle in expected.items():
            assert _report(engine.generate(days=30, source=platform)) == oracle
        actual = engine.get_usage_breakdown()
        actual["tools"] = sorted(actual["tools"], key=lambda row: row["tool"])
        breakdown["tools"] = sorted(breakdown["tools"], key=lambda row: row["tool"])
        assert actual == breakdown
        text = engine.format_terminal(engine.generate())
        assert (
            "~$1.50" in text
            and "included" in text.lower()
            and "unknown" in text.lower()
        )
    from types import SimpleNamespace
    from hermes_cli import config
    from hermes_cli.main_agent_cmds import cmd_insights

    home = tmp_path / "hosted-cli"
    monkeypatch.setenv("HERMES_STATE_BACKEND", "postgresql")
    monkeypatch.setenv("HERMES_DATABASE_URL", pg_settings["semantier_app"])
    monkeypatch.setenv(
        "HERMES_STATE_SCOPE_MAP",
        json.dumps(
            {
                str(home): dict(
                    zip(
                        ("tenant_id", "workspace_id", "source_store_id"), scope.identity
                    )
                )
            }
        ),
    )
    monkeypatch.setattr(config, "get_hermes_home", lambda: home)
    assert not (home / "state.db").exists()
    cmd_insights(SimpleNamespace(days=30, source="cli"))
    assert "~$1.50" in capsys.readouterr().out
    monkeypatch.setattr(config, "get_hermes_home", lambda: tmp_path / "unmapped")
    cmd_insights(SimpleNamespace(days=30, source="cli"))
    refusal = capsys.readouterr().out
    assert "~$1.50" not in refusal and (
        "scope" in refusal.lower() or "mapping" in refusal.lower()
    )


def test_recall_metadata_keeps_compacted_hidden_and_colliding_scopes_separate(
    tmp_path, pg_settings, monkeypatch
):
    from tools.session_search_tool import (
        _get_message_storage_state,
        _is_compacted_message,
    )
    from hermes_state_postgresql import (
        PostgreSQLSessionDB,
        SessionScope,
        provision_scope,
    )
    import sqlite3
    from psycopg import sql

    source, compacted, hidden = _source(tmp_path, monkeypatch)
    from agent.insights import InsightsEngine

    expected_report = _report(InsightsEngine(source).generate())
    expected = {
        message: _get_message_storage_state(source, message)
        for message in (compacted, hidden, 999999)
    }
    assert expected[compacted] == {"session_id": "same", "active": 0, "compacted": 1}
    assert expected[hidden] == {"session_id": "same", "active": 0, "compacted": 0}
    scopes = [
        SessionScope("recall-" + str(i), "workspace", str(uuid4())) for i in range(2)
    ]
    for scope in scopes:
        provision_scope(
            pg_settings["semantier_migrate"], scope, app_role="semantier_app"
        )
        with PostgreSQLSessionDB(
            pg_settings["semantier_migrate"], scope, home=tmp_path
        ) as importer:
            importer.import_sqlite_snapshot(source.db_path)
    source.close()
    with PostgreSQLSessionDB(
        pg_settings["semantier_migrate"], scopes[1], home=tmp_path
    ) as writer:
        writer.create_session("foreign-only", "cli", model="foreign-private-model")
        with writer._transaction() as conn:
            conn.execute(
                sql.SQL("UPDATE {} SET active=1,compacted=0 WHERE id=%s").format(
                    writer.table("messages")
                ),
                (compacted,),
            )
    monkeypatch.setattr(
        sqlite3, "connect", lambda *a, **k: pytest.fail("PG recall opened SQLite")
    )
    with PostgreSQLSessionDB(
        pg_settings["semantier_app"], scopes[0], home=tmp_path, read_only=True
    ) as first:
        for message, value in expected.items():
            assert _get_message_storage_state(first, message) == value
        assert _report(InsightsEngine(first).generate()) == expected_report
        assert _is_compacted_message(first, compacted)
        assert not _is_compacted_message(first, hidden)
    with PostgreSQLSessionDB(
        pg_settings["semantier_app"], scopes[1], home=tmp_path, read_only=True
    ) as second:
        assert InsightsEngine(second).generate()["overview"]["total_sessions"] == 4
        assert not _is_compacted_message(second, compacted)
        assert _get_message_storage_state(second, compacted)["active"] == 1
    assert _get_message_storage_state(first, compacted) is None
