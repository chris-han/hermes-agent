# Review-only sessions maintenance admission proposal

Status: NOT APPLIED. This proposal adds no new database selector and has no executed test result.

Source: hermes_cli/sessions_cmd_journal_mode.py, SHA-256 44b9b1e4e13081294562c2805859081b95283811c5de36e1773b12025f40b20c
Tests: tests/hermes_cli/test_sessions_set_journal_mode.py, SHA-256 e45819cdb2c1bb1b5c4debf0fd43a28aa6058f7323a6f56ce0567a8b6c17ca66
Parser/holder contract test: tests/hermes_cli/test_sessions_held_store_gate.py, SHA-256 1d9c3dfe0381171895554089b051f957c74145cd4b7bc0f2fcd0c3f7369c24c4

## Contract and scope

Standalone offline operators may use the documented set-journal-mode --db path (website/docs/user-guide/sessions.md:706–732). An authenticated-session execution boundary is not authority to convert shared database journal mode. The runtime optional-boundary contract preserves no-provider standalone behavior; authenticated IO must prove appropriate scope or fail closed. This proposal refuses governed invocation until a dedicated governed operator/store admission exists.

The production entry point uses _PRE_DB_HANDLERS before opening SessionDB. Place the guard at the very start of cmd_set_journal_mode, before importing the store resolver and before path resolution, exists/header probes, holder scans, or SQLite connect. Reuse current_execution_boundary() and get_execution_boundary_provider(); no new helper or policy knob. Both active-boundary-only and registered-provider-without-session refuse, even with --force. Existing standalone quiescence, platform, integrity, mode-transition, and --force behavior remain untouched.

## Exact production insertion

Replace only the current function header plus its first import with:

```python
def cmd_set_journal_mode(args) -> int:
    from gateway.execution_boundary import current_execution_boundary, get_execution_boundary_provider

    if current_execution_boundary() is not None or get_execution_boundary_provider() is not None:
        print("✗ Refusing journal-mode maintenance in a governed runtime: an offline operator context is required.")
        return 1

    from hermes_state import _default_db_path
```

The remaining body is unchanged. Rejection prints the shown explanation and returns status 1; no path values or store contents are read.

## Test changes

1. Parameterize the existing real-store conversion test over explicit_db=[False, True]. Keep every existing content/header/sidecar assertion. Pass db=str(db) only in the explicit case. This proves both documented standalone entry paths remain functional.
2. Add this negative invariant in the same test file. It uses a real temporary database, preserves its bytes, and makes any resolver/header/open access fail visibly:

```python
@pytest.mark.parametrize("scope_state", ["active", "provider", "both"])
@pytest.mark.parametrize("explicit_db", [False, True])
def test_set_journal_mode_refuses_governed_invocations_before_store_io(
    scope_state, explicit_db, tmp_path, monkeypatch, capsys
):
    from contextlib import nullcontext
    from pathlib import Path
    from gateway import execution_boundary as boundary
    import hermes_cli.sessions_cmd_journal_mode as journal

    db = tmp_path / "state.db"
    _wal_store(db)
    before = db.read_bytes()
    original = boundary.get_execution_boundary_provider()
    scope = boundary.ExecutionBoundary(source="test")
    context = boundary.bind_execution_boundary(scope) if scope_state != "provider" else nullcontext()

    def unexpected_io(*args, **kwargs):
        pytest.fail("Governed maintenance reached store IO")

    try:
        if scope_state != "active":
            boundary.replace_execution_boundary_provider(object())
        else:
            boundary.clear_execution_boundary_provider()
        with context, monkeypatch.context() as guarded:
            guarded.setattr("hermes_state._default_db_path", unexpected_io)
            guarded.setattr(Path, "exists", unexpected_io)
            guarded.setattr(journal, "_header_mode", unexpected_io)
            guarded.setattr(journal.sqlite3, "connect", unexpected_io)
            assert cmd_sessions(_args("delete", db=str(db) if explicit_db else None, force=True)) == 1
    finally:
        boundary.replace_execution_boundary_provider(original)

    assert db.read_bytes() == before
    assert "offline operator context" in capsys.readouterr().out
```

3. In test_sessions_held_store_gate.py, replace the broad comment claiming no sessions command ever selects another database. Explain that profile-session commands use the default store, while standalone journal-mode maintenance uses a distinct pre-DB handler. Keep all holder, exact path, force, preview, and quiet-store assertions. Check every parser except exactly set-journal-mode for absence of --db/--db-path/--database; positively assert that set-journal-mode accepts --db PATH and is mapped to the pre-DB maintenance handler. Make this parser distinction only together with the governed guard and its tests.

## Verification after approval/application

Run the full held-store and journal-mode files, existing sessions command tests, and the execution-boundary tests through scripts/run_tests.sh. Review the final three-file diff, use Ruff, and checkpoint locally. Do not publish while the separately blocked transcription repair or cancelled CI setup edit remains pending.

No source or test file was changed to prepare this proposal.
