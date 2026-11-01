"""Read-only WSL path conversion can recover from an interop timeout."""

import subprocess

import pytest

from scripts.bundles.git_command import linux_path


def test_transient_wslpath_timeout_retries_without_weakening_path_validation(monkeypatch):
    calls = []

    def convert(command, **kwargs):
        calls.append((command, kwargs))
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(command, 30)
        return "/mnt/d/repo/hermes-channel1-build\n"

    monkeypatch.setattr(subprocess, "check_output", convert)
    assert linux_path("Ubuntu-22.04", r"D:\repo\hermes-channel1-build") == "/mnt/d/repo/hermes-channel1-build"
    assert len(calls) == 2 and all(call[1]["timeout"] == 30 for call in calls)


def test_wslpath_timeout_is_bounded_and_invalid_output_is_never_accepted(monkeypatch):
    calls = []

    def timeout(command, **_kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, 30)

    monkeypatch.setattr(subprocess, "check_output", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        linux_path("Ubuntu-22.04", r"D:\repo\hermes-channel1-build")
    assert len(calls) == 3
    monkeypatch.setattr(subprocess, "check_output", lambda *_a, **_k: "relative\n")
    with pytest.raises(ValueError):
        linux_path("Ubuntu-22.04", r"D:\repo\hermes-channel1-build")


def test_wslpath_non_timeout_errors_are_not_retried(monkeypatch):
    calls = []

    def fail(command, **_kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "check_output", fail)
    with pytest.raises(subprocess.CalledProcessError):
        linux_path("Ubuntu-22.04", r"D:\repo\hermes-channel1-build")
    assert len(calls) == 1
