"""Local-model jobs use PM transfers for every entry point."""

from __future__ import annotations

import time
from collections import deque

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.pm._range_server import RangeHandler, dl_server, url


def test_rate_and_eta_report_only_an_honest_number():
    """Rate is the slope across the window; anything unmeasurable is unknown.

    A fabricated rate would render as a confident ETA that keeps being wrong,
    so a single sample, a window too short to divide by, and a stalled
    transfer must all report (None, None) rather than a guess.
    """
    from hermes_cli.web_routers.local_models import _rate_and_eta

    mib = 1 << 20
    assert _rate_and_eta(deque([(0.0, 0), (1.0, mib)]), mib, 3 * mib) == (float(mib), 2)
    assert _rate_and_eta(deque([(0.0, 0)]), 0, 100) == (None, None)
    assert _rate_and_eta(deque([(0.0, 0), (0.1, 50)]), 50, 100) == (None, None)
    assert _rate_and_eta(deque([(0.0, 40), (1.0, 40)]), 40, 100) == (None, None)
    # A rate with no denominator: ETA stays unknown, speed still useful.
    assert _rate_and_eta(deque([(0.0, 0), (1.0, mib)]), mib, None) == (float(mib), None)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from hermes_cli.web_routers import local_models as lm

    monkeypatch.setattr(lm, "_JOBS", {})
    monkeypatch.setattr(lm, "_RUNNING", {})
    monkeypatch.setattr(lm.bootstrap, "refresh_local_runtime", lambda: False)
    app = FastAPI()
    from tests.hermes_cli._local_runtime_handlers import router as handler_router
    app.include_router(handler_router())
    with TestClient(app) as client:
        yield client


def poll(client, job_id, until=("done", "error", "paused")):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        response = client.get(f"/api/local-models/jobs/{job_id}")
        assert response.status_code == 200, response.text
        job = response.json()
        if job["status"] in until:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job did not settle: {job}")




def test_quickstart_plan_progress_uses_actual_whole_plan_bytes(client, monkeypatch, dl_server, tmp_path):
    from hermes_cli.web_routers import local_models as lm

    bodies = {"first": b"first", "second": b"second"}
    RangeHandler.payloads = {"/" + name: body for name, body in bodies.items()}
    plan = [(url(dl_server, "/" + name), tmp_path / (name + ".gguf"), 1000) for name in bodies]
    job = lm._job("quickstart", "test")
    lm._run_download_plan(job, plan, "test")
    expected = sum(map(len, bodies.values()))
    assert job["done_bytes"] == job["total_bytes"] == expected
    assert sum(end - start for rows in job["ranges"].values() for start, end in rows) == expected


def test_existing_weights_do_not_skip_missing_companion_download(client, monkeypatch, dl_server, tmp_path):
    from dataclasses import replace
    from hermes_cli.local_runtime.catalog import AssetFile
    from hermes_cli.web_routers import local_models as lm

    entry = replace(lm.catalog.CATALOG[0], mmproj=AssetFile("projector.gguf", 1000), draft=None)
    variant = entry.variants[0]
    monkeypatch.setattr(lm, "_quickstart_target", lambda *args: (entry, variant))
    monkeypatch.setattr(lm, "_runtime_target", lambda *args: ("test-engine", "cpu"))
    monkeypatch.setattr(lm.binaries, "installed_engine", lambda *args: object())
    monkeypatch.setattr(lm, "_ensure_server", lambda *args, **kwargs: None)
    monkeypatch.setattr(lm, "_assign_default", lambda *args: None)
    monkeypatch.setattr(lm.bootstrap, "staged_model_ids", lambda: [variant.model_id])
    weights = tmp_path / "weights.gguf"
    weights.write_bytes(b"already downloaded")
    companion = tmp_path / "assets" / "projector.gguf"
    RangeHandler.payloads["/projector"] = b"projector bytes"
    plan = [(url(dl_server, "/weights"), weights, 1000),
            (url(dl_server, "/projector"), companion, 1000)]
    monkeypatch.setattr(lm, "_download_plan", lambda *args: plan)
    response = client.post("/api/local-models/quickstart", json={"model_id": entry.id})
    assert response.json().get("job_id"), "weights-only staging incorrectly skipped the projector"
    job = poll(client, response.json()["job_id"])
    assert job["status"] == "done", job
    assert companion.read_bytes() == RangeHandler.payloads["/projector"]
    assert job["done_bytes"] == weights.stat().st_size + companion.stat().st_size
