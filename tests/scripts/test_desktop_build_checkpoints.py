"""Build retries reuse only verified successful products."""
import pytest
import json

from scripts.bundles.desktop import BuildCheckpoints


def test_resume_reuses_success_and_invalidates_changed_products(tmp_path):
    log = tmp_path / "checkpoints.jsonl"
    first, second = tmp_path / "first", tmp_path / "second"
    calls = []

    def build(path, value):
        calls.append(path.name)
        path.write_text(value)

    def execute():
        checkpoints = BuildCheckpoints(log, "source-version")
        checkpoints.execute("first", [first], lambda: build(first, "one"))
        checkpoints.execute("second", [second], lambda: build(second, first.read_text()))

    execute()
    execute()
    assert calls == ["first", "second"]
    first.write_text("corrupt")
    execute()
    assert calls == ["first", "second", "first"]
    second.unlink()
    execute()
    assert calls[-1] == "second"
    changed = BuildCheckpoints(log, "different-source-version")
    changed.execute("first", [first], lambda: build(first, "new"))
    assert first.read_text() == "new"


def test_failure_and_partial_log_never_claim_success(tmp_path):
    log, output = tmp_path / "checkpoints.jsonl", tmp_path / "product"
    calls = []

    def fail():
        output.write_text("incomplete")
        raise RuntimeError("packaging failed")

    with pytest.raises(RuntimeError, match="packaging failed"):
        BuildCheckpoints(log, "source").execute("package", [output], fail)
    log.write_text('{"interrupted":')

    def succeed():
        calls.append("rebuilt")
        output.write_text("complete")

    BuildCheckpoints(log, "source").execute("package", [output], succeed)
    BuildCheckpoints(log, "source").execute("package", [output], succeed)
    assert calls == ["rebuilt"]


def test_beta_reuses_outputs_without_hashing_and_records_timings(tmp_path, monkeypatch):
    def forbidden(_path):
        raise AssertionError("Beta must not hash checkpoint outputs")

    monkeypatch.setattr("scripts.bundles.desktop_prepare.fingerprint", forbidden)
    log, output = tmp_path / "checkpoints.jsonl", tmp_path / "product"
    calls = []

    def build():
        calls.append("built")
        output.write_text("beta")

    BuildCheckpoints(log, "beta", verify=False).execute("desktop", [output], build)
    BuildCheckpoints(log, "beta", verify=False).execute("desktop", [output], build)
    assert calls == ["built"]
    events = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    timings = [event for event in events if event["type"] == "milestone-timing"]
    assert [event["status"] for event in timings] == ["SUCCESS", "HIT"]
    assert all(event["seconds"] >= 0 and event["mode"] == "Beta" for event in timings)
