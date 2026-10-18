"""Real process contention for hub records, provenance, removal and JSON readers."""
import json
import multiprocessing
import os
from multiprocessing.connection import wait
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.skills_hub import HubLockFile, TapsManager

WORKERS = 4
WRITES = 40


def _record(lock, name):
    lock.record_install(
        name=name, source="github", identifier=f"owner/repo/{name}",
        trust_level="community", scan_verdict="safe", skill_hash=f"sha256:{name}",
        install_path=name, files=["SKILL.md", "data.json"],
        metadata={"name": name, "label": "来源"},
        scan_provenance={"scanner": "test", "digest": name},
    )


def _worker(root, tag, start):
    root = Path(root)
    lock, taps = HubLockFile(root / "lock.json"), TapsManager(root / "taps.json")
    import tools.skills_hub as hub
    from tools.skills_hub_install import uninstall_skill

    os.environ["HERMES_HOME"] = str(root / "home")
    hub._skills_dir = lambda: root / "skills"
    HubLockFile.DEFAULT_PATH = staticmethod(lambda: root / "lock.json")
    start.wait(timeout=30)
    for i in range(WRITES):
        name, old = f"worker-{tag}-{i}", f"old-{tag}-{i}"
        _record(lock, name)
        ok, message = uninstall_skill(old)
        assert ok, message
        assert not (root / "skills" / old).exists()
        assert taps.add(f"owner/{name}")
        assert taps.remove(f"owner/{old}")


def test_processes_preserve_records_provenance_and_uninstalls(tmp_path):
    lock, taps = HubLockFile(tmp_path / "lock.json"), TapsManager(tmp_path / "taps.json")
    old = [f"old-{tag}-{i}" for tag in range(WORKERS) for i in range(WRITES)]
    for name in old:
        directory = tmp_path / "skills" / name
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text("test fixture", encoding="utf-8")
    lock.save({"version": 1, "installed": {
        name: {"identifier": name, "source": "github", "trust_level": "community", "install_path": name}
        for name in old
    }})
    taps.save([{"repo": f"owner/{name}", "path": "skills/"} for name in old])
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Barrier(WORKERS + 1)
    processes = [ctx.Process(target=_worker, args=(str(tmp_path), tag, start)) for tag in range(WORKERS)]
    reads = 0
    try:
        for process in processes:
            process.start()
        start.wait(timeout=30)
        while any(process.is_alive() for process in processes):
            # Read raw bytes: load() would hide torn JSON by returning EMPTY.
            for path in (lock.path, taps.path):
                data = json.loads(path.read_text(encoding="utf-8-sig"))
                assert isinstance(data, dict)
                reads += 1
            wait([process.sentinel for process in processes if process.is_alive()], timeout=0.001)
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=10)
    assert reads > 0
    expected = {f"worker-{tag}-{i}" for tag in range(WORKERS) for i in range(WRITES)}
    installed = lock.load()["installed"]
    assert set(installed) == expected
    assert not any((tmp_path / "skills" / name).exists() for name in old)
    for name, entry in installed.items():
        assert {key: entry[key] for key in (
            "source", "identifier", "trust_level", "scan_verdict", "content_hash",
            "install_path", "files", "metadata", "scan_provenance",
        )} == {
            "source": "github", "identifier": f"owner/repo/{name}",
            "trust_level": "community", "scan_verdict": "safe", "content_hash": f"sha256:{name}",
            "install_path": name, "files": ["SKILL.md", "data.json"],
            "metadata": {"name": name, "label": "来源"},
            "scan_provenance": {"scanner": "test", "digest": name},
        }
        assert entry["installed_at"] == entry["updated_at"]
        assert entry["installed_at"].endswith("+00:00")
    assert {tap["repo"] for tap in taps.load()} == {f"owner/{name}" for name in expected}
    assert all(tap["path"] == "skills/" for tap in taps.load())


@pytest.mark.parametrize("error", [OSError("lock unavailable"), TimeoutError("lock busy")])
@pytest.mark.parametrize("operation", ["install", "uninstall", "tap_add", "tap_remove"])
def test_lock_failure_never_mutates_state(tmp_path, error, operation):
    lock, taps = HubLockFile(tmp_path / "lock.json"), TapsManager(tmp_path / "taps.json")
    _record(lock, "keep")
    taps.add("owner/keep")
    before = [path.read_bytes() for path in (lock.path, taps.path)]
    actions = {
        "install": lambda: _record(lock, "new"),
        "uninstall": lambda: lock.record_uninstall("keep"),
        "tap_add": lambda: taps.add("owner/new"),
        "tap_remove": lambda: taps.remove("owner/keep"),
    }
    with patch("hermes_cli.auth._file_lock", side_effect=error):
        with pytest.raises(type(error)):
            actions[operation]()
    assert [path.read_bytes() for path in (lock.path, taps.path)] == before
