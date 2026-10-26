"""Paused work remains discoverable, and terminal phase changes honor pause intent."""

from __future__ import annotations

import threading

import pytest

from pm.downloader import DownloadPaused




def test_pause_request_prevents_transition_into_activation(monkeypatch):
    from hermes_cli.web_routers import local_models as lm

    monkeypatch.setattr(lm, "_JOBS", {})
    monkeypatch.setattr(lm, "_RUNNING", {})
    job = lm._job("quickstart", "model")
    job["phase"] = "downloading"
    pause = threading.Event()
    lm._RUNNING[job["job_id"]] = {"pause": pause}
    # Same explicit pause intent as the retained terminal setup flow.
    pause.set()
    with pytest.raises(DownloadPaused):
        lm._step(job, "starting-server", "Starting")
    assert job["phase"] == "downloading"
