"""Regression tests for the shared terminal runtime handlers.

Test-only transport against the real in-process handlers; the runtime pieces
underneath are exercised against temp HERMES_HOME (autouse fixture). Network
downloads are stubbed at the urllib boundary — never live."""

from __future__ import annotations

from pathlib import Path

import pytest

# The real loopback range server (shared with tests/pm): importing the
# fixture name at module scope registers it for these tests too.


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from tests.hermes_cli._local_runtime_handlers import client as handler_client
    return handler_client()

def _write_fake_gguf(path: Path, size: int = 1024) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + b"\x00" * size)


# ── status ───────────────────────────────────────────────────


def test_status_shape_and_defaults(client):
    r = client.get("/api/local-models/status")
    assert r.status_code == 200
    data = r.json()
    # Contract: every key the pane's first paint needs, present and typed.
    assert isinstance(data["enabled"], bool)
    assert isinstance(data["tag"], str) and data["tag"].startswith("b")
    assert isinstance(data["runtime_installed"], bool)
    assert isinstance(data["server_running"], bool)
    assert isinstance(data["models"], list)


def test_status_renders_degraded_when_config_cannot_be_read(client, monkeypatch):
    """The status pane is garnish: an unreadable/uninitialized config renders defaults, never a 500."""
    from hermes_cli import config as config_mod

    def _boom():
        raise FileNotFoundError("profile home is gone")

    monkeypatch.setattr(config_mod, "load_config_readonly", _boom)
    r = client.get("/api/local-models/status")
    assert r.status_code == 200 and r.json()["enabled"] is False


def test_status_lists_staged_models_with_labels(client, tmp_path):
    from hermes_cli.local_runtime.bootstrap import models_dir

    _write_fake_gguf(models_dir() / "Some-Model.gguf", size=2048)
    data = client.get("/api/local-models/status").json()
    ids = [m["id"] for m in data["models"]]
    assert "Some-Model" in ids
    row = data["models"][ids.index("Some-Model")]
    assert row["size_bytes"] > 0
    assert row["size_label"].endswith("GB")


def test_status_sizes_a_split_model_by_all_its_parts(client):
    """A split's first file can be a metadata stub of a few MB. The row reports the model the user
    has on disk, so a side-loaded split outside the catalog is every part, not part 1."""
    from hermes_cli.local_runtime.bootstrap import models_dir

    _write_fake_gguf(models_dir() / "Side-Split-00001-of-00002.gguf", size=1024)
    _write_fake_gguf(models_dir() / "Side-Split-00002-of-00002.gguf", size=8192)
    rows = {m["id"]: m for m in client.get("/api/local-models/status").json()["models"]}

    assert rows["Side-Split"]["size_bytes"] == (4 + 1024) + (4 + 8192)


def test_status_tracks_preset_spill_and_restored_window(client, tmp_path, monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace

    from hermes_cli.local_runtime import bootstrap, presets
    from hermes_cli.local_runtime.binaries import runtimes_root
    from hermes_cli.local_runtime.context_policy import FLOOR, RUNTIME_OVERHEAD_BYTES, ub_logits_bytes
    from hermes_cli.local_runtime.estimator import HardwareBudget, LayerKind, ModelProfile, ctx_bytes
    from hermes_cli.local_runtime.growth import save_window_override
    from hermes_cli.web_routers import local_models

    # Dense spill has no override-tensor flag: status must use the recorded decision.
    profile = ModelProfile("status-mtp", 16 << 30, 0, 262144,
                           [(LayerKind.FULL, 4096)] * 32, n_vocab=151936)
    model_id = profile.name
    _write_fake_gguf(bootstrap.models_dir() / f"{model_id}.gguf")
    monkeypatch.setattr(presets, "read_gguf_header", lambda p: SimpleNamespace(sampling_defaults={}))
    monkeypatch.setattr(presets, "profile_from_gguf", lambda h: profile)
    monkeypatch.setattr(local_models, "_state_endpoint", lambda: {"base_url": "http://127.0.0.1:1/v1"})

    server_window = FLOOR

    def router_response(running, route, **kwargs):
        if route == "/models":
            return {"data": [{"id": model_id, "status": {"value": "loaded"}}]}
        assert route == f"/props?model={model_id}&autoload=false"
        return {"default_generation_settings": {"n_ctx": server_window}}

    monkeypatch.setattr(local_models, "_router_request", router_response)
    floor_need = profile.weights_bytes + ctx_bytes(replace(profile, kv_scale=1.2), FLOOR)
    lean = RUNTIME_OVERHEAD_BYTES + ub_logits_bytes(profile.n_vocab, mtp_capable=True)
    stacked = RUNTIME_OVERHEAD_BYTES + ub_logits_bytes(profile.n_vocab, mtp_capable=True, mtp_prefill=True)
    ini = runtimes_root() / "presets.ini"
    grown = 73728
    for device, override, spilled in ((floor_need + lean - 1, FLOOR, True),
                                      (floor_need + stacked, grown, False)):
        save_window_override(model_id, override)
        preset = presets.generate_presets(bootstrap.models_dir(),
                                         HardwareBudget(device, device, 8 << 30), ini, {model_id})[0]
        assert preset.window == override and preset.spilled is spilled
        assert preset.keys["spec-type"] == "draft-mtp"
        assert "ubatch-size" not in preset.keys and "override-tensor" not in preset.keys
        # Deliberately differ from the plan to prove the server remains the grant authority.
        server_window = preset.window - 1024
        response = client.get("/api/local-models/status")
        assert response.status_code == 200
        data = response.json()
        assert data["loaded_models"][model_id] == "loaded"
        placement = data["placement"][model_id]
        assert placement["spilled"] is spilled
        assert placement["window"] == preset.window
        assert placement["granted_window"] == server_window
        assert placement["granted_window_label"] == local_models._k_label(server_window)


# ── hardware ─────────────────────────────────────────────────


def test_hardware_plain_facts(client):
    data = client.get("/api/local-models/hardware").json()
    assert isinstance(data["uma"], bool)
    assert data["ram_total_bytes"] > 0
    assert data["vram_total_bytes"] >= 0
    # GPU fields are None-able (non-NVIDIA machines) but must exist.
    assert "gpu_name" in data and "gpu_util_percent" in data and "vram_used_bytes" in data


# ── catalog ──────────────────────────────────────────────────


def test_catalog_prices_every_entry_for_this_machine(client):
    data = client.get("/api/local-models/catalog").json()
    assert len(data["models"]) >= 3
    for row in data["models"]:
        # The three user questions, answered on every row:
        assert row["size_label"].endswith("GB")            # how big
        assert isinstance(row["fits"], bool)               # will it fit
        assert row["fit_summary"]                          # what shape
        if row["fits"]:
            assert row["start_window"] >= 1
            assert row["start_window_label"].endswith("K")
        else:
            assert "memory" in row["fit_summary"].lower()
        assert isinstance(row["downloaded"], bool)


@pytest.mark.parametrize("backend, expected", [
    ("auto", "qwen3.8-27b"), ("cuda", "qwen3.8-27b"),
    ("vulkan", "qwen3.6-35b-a3b"), ("cpu", "qwen3.6-35b-a3b"),
])
def test_calibrated_catalog_and_quickstart_respect_backend(client, monkeypatch, backend, expected):
    from hermes_cli.local_runtime import binaries, bootstrap, catalog, hardware
    from hermes_cli.local_runtime.estimator import HardwareBudget
    from hermes_cli.web_routers import local_models

    budget = HardwareBudget(int(48 * (1 << 30) * .8), 48 << 30, 0, uma=True,
                            gpu_name="NVIDIA RTX Spark N1X (5120-core Blackwell RTX GPU)", platform="win32")
    monkeypatch.setattr(hardware, "probe_budget", lambda **kw: budget)
    monkeypatch.setattr(catalog, "refresh_catalog_soon", lambda: None)
    monkeypatch.setattr(bootstrap, "staged_model_ids", lambda: set())
    monkeypatch.setattr(binaries, "installed_engine",
                        lambda *args, **kwargs: binaries.Engine("cuda", "b10964", Path("unused")))
    monkeypatch.setattr(local_models, "_load_config", lambda: {"local_runtime": {"backend": backend}})

    def no_probe(*args, **kwargs):
        raise AssertionError("recommendation must not launch a model or run another hardware probe")

    monkeypatch.setattr(bootstrap, "_detect_gpu_vendor", no_probe)
    monkeypatch.setattr(bootstrap, "ensure_local_runtime", no_probe)
    response = client.get("/api/local-models/catalog")
    assert response.status_code == 200
    chosen = [row["id"] for row in response.json()["models"] if row["recommended"]]
    assert chosen == [expected]
    assert local_models._quickstart_target(local_models.QuickstartBody(), budget)[0].id == expected
    explicit = local_models.QuickstartBody(model_id="qwen3.8-27b")
    assert local_models._quickstart_target(explicit, budget)[0].id == "qwen3.8-27b"


def test_catalog_never_hides_unaffordable_models(client, monkeypatch):
    """Unaffordable entries stay visible with a plain reason — hiding them
    is how users conclude the feature is broken."""
    from hermes_cli.local_runtime.estimator import HardwareBudget

    tiny = HardwareBudget(usable_vram_bytes=1 << 30, total_device_bytes=1 << 30,
                          ram_available_bytes=1 << 30)
    monkeypatch.setattr("hermes_cli.local_runtime.hardware.probe_budget",
                        lambda **kw: tiny)
    data = client.get("/api/local-models/catalog").json()
    from hermes_cli.local_runtime.catalog import CATALOG

    assert len(data["models"]) == len(CATALOG)
    refused = [m for m in data["models"] if not m["fits"]]
    assert refused, "a 1 GiB machine must refuse the 20 GB models"
    for row in refused:
        assert row["fit_detail"] or row["fit_summary"]


# ── downloads ────────────────────────────────────────────────


# ── runtime install ──────────────────────────────────────────


def test_runtime_install_rejects_impossible_combo(client, monkeypatch):
    """Impossible platform/backend combos fail the POST itself with the
    resolver's honest message — not a background job that dies silently.
    (win-arm64-vulkan; the old cuda case became real upstream at ~b1036x.)"""
    from pm import paths
    from pm.lock import Lockfile

    lock_path = Path(paths.partials_root()).parent / "unavailable-lock.json"
    Lockfile(lock_path).save()
    monkeypatch.setattr(paths, "lockfile_path", lambda: lock_path)
    r = client.post("/api/local-models/runtime/install", json={"backend": "cpu"})
    assert r.status_code == 400
    assert "not pinned" in r.json()["detail"]


# ── pause / resume against a real loopback range server ──────


_BIG_BODY = bytes(range(256)) * (8 * 1024 * 1024 // 256)  # 8 MiB, deterministic


