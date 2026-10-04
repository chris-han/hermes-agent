"""Live callers must allocate temporary files only inside the active session.

Allocator wrappers create real filesystem objects, then stop before external
processing. The production call paths and execution boundary remain real.
"""
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

import pytest

from gateway.execution_boundary import (
    BoundaryPaths, BoundaryPolicy, ExecutionBoundary,
    GovernedExecutionBoundaryRequired, bind_execution_boundary,
    clear_execution_boundary_provider, get_execution_boundary_provider,
    replace_execution_boundary_provider,
)


class AllocationObserved(BaseException):
    """Stop after allocation, before a codec or user command runs."""


CASES = ("silk", "trim", "cloud", "command_stt", "local_stt",
         "command_tts", "wav_delivery", "skill_batch")


@pytest.fixture
def staging(tmp_path, monkeypatch):
    previous = get_execution_boundary_provider()
    clear_execution_boundary_provider()
    host = tmp_path / "host-temp"
    host.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(host))
    runs = tmp_path / "session" / "runs"
    runs.mkdir(parents=True)
    boundary = ExecutionBoundary(
        source="gateway_runner", session_id="session",
        paths=BoundaryPaths(runs_root=runs, artifacts_root=runs),
        policy=BoundaryPolicy(require_boundary=True, allowed_write_roots=(runs,)),
    )
    try:
        yield host, runs, boundary
    finally:
        clear_execution_boundary_provider()
        if previous is not None:
            replace_execution_boundary_provider(previous)


def _invocation(case, monkeypatch, runs):
    source = runs / "audio.wav"
    source.write_bytes(b"fixture audio")
    def silk():
        from tools import transcription_audio as audio, transcription_tools as facade
        monkeypatch.setattr(facade, "_HAS_PILK", True)
        return lambda: audio._prepare_audio_for_transcription(str(source.with_suffix(".silk")))
    def trim():
        from tools import transcription_audio as audio
        monkeypatch.setattr(audio, "_find_ffmpeg_binary", lambda: "unused-ffmpeg")
        monkeypatch.setattr(audio, "_probe_audio_duration", lambda _: 120.0)
        return lambda: audio._trim_silence_for_cloud_stt(str(source), {})
    def cloud():
        from tools import transcription_cloud as cloud, transcription_tools as facade
        monkeypatch.setattr(facade, "_HAS_OPENAI", True)
        monkeypatch.setattr(cloud, "_resolve_openai_audio_client_config", lambda: (None, None))
        monkeypatch.setattr(cloud, "_with_openai_client", lambda *args: args[-1](None))
        return lambda: cloud._transcribe_openai(str(source), "whisper-1", language="en")
    def command_stt():
        from tools.transcription_command import _transcribe_command_stt
        return lambda: _transcribe_command_stt(
            str(source), "fixture", {"command": "unused {output_path}"}, {}, language_override="en")
    def local_stt():
        from tools import transcription_local as local
        monkeypatch.setattr(local, "_get_local_command_template", lambda: "unused {output_dir}")
        return lambda: local._transcribe_local_command(str(source), "base", language="en")
    def command_tts():
        from tools.tts_command_provider import _generate_command_tts
        return lambda: _generate_command_tts(
            "private utterance", str(runs / "out.mp3"), "fixture", {"command": "unused {input_path}"}, {})
    def wav_delivery():
        from tools.tts_tool_delivery import _write_wav_bytes_as
        return lambda: _write_wav_bytes_as(b"private audio", str(runs / "out.mp3"))
    def skill_batch():
        from tools import skill_manager_tool as skills
        from tools.registry import registry
        monkeypatch.setattr(skills, "SKILLS_DIR", runs / "skills")
        monkeypatch.setattr(skills, "_skill_gate_bypass", SimpleNamespace(get=lambda: True))
        return lambda: registry.dispatch("skill_manage", {
            "action": "", "name": "", "operations": [{
                "action": "create", "name": "fixture",
                "content": "---\nname: fixture\ndescription: Test bounded staging.\n---\n# Fixture\n",
            }],
        })

    factories = {
        "silk": silk, "trim": trim, "cloud": cloud,
        "command_stt": command_stt, "local_stt": local_stt,
        "command_tts": command_tts, "wav_delivery": wav_delivery,
        "skill_batch": skill_batch,
    }
    return factories[case]()


def _observe_allocation(monkeypatch, allocations):
    real_mkdtemp = tempfile.mkdtemp
    real_named = tempfile.NamedTemporaryFile

    def directory(*args, **kwargs):
        path = Path(real_mkdtemp(*args, **kwargs))
        allocations.append(path.resolve())
        shutil.rmtree(path)
        raise AllocationObserved()

    def named(*args, **kwargs):
        handle = real_named(*args, **kwargs)
        path = Path(handle.name)
        allocations.append(path.resolve())
        handle.close()
        path.unlink(missing_ok=True)
        raise AllocationObserved()

    monkeypatch.setattr(tempfile, "mkdtemp", directory)
    monkeypatch.setattr(tempfile, "NamedTemporaryFile", named)


@pytest.mark.parametrize("case", CASES)
def test_live_staging_uses_session_roots(case, staging, monkeypatch):
    host, runs, boundary = staging
    invoke = _invocation(case, monkeypatch, runs)
    allocations = []
    _observe_allocation(monkeypatch, allocations)
    with bind_execution_boundary(boundary), pytest.raises(AllocationObserved):
        invoke()
    assert allocations and all(path.is_relative_to(runs) for path in allocations)
    assert list(host.iterdir()) == []


@pytest.mark.parametrize("case", CASES)
def test_live_staging_fails_before_allocating_without_governed_boundary(case, staging, monkeypatch):
    host, runs, _ = staging
    invoke = _invocation(case, monkeypatch, runs)
    allocations = []
    _observe_allocation(monkeypatch, allocations)
    replace_execution_boundary_provider(object())
    try:
        result = invoke()
    except GovernedExecutionBoundaryRequired:
        pass
    except AllocationObserved:
        pytest.fail(f"{case} allocated with a missing governed boundary: {allocations}")
    else:
        assert "BOUNDARY_REQUIRED" in str(result)
    assert allocations == []
    assert list(host.iterdir()) == []
