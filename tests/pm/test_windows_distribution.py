"""Selected Windows product closure; source and other targets keep their features."""
from pathlib import Path
import pytest
from pm import features

ROOT = Path(__file__).resolve().parents[2]


def test_product_selection_excludes_inference_families_and_keeps_shared_features():
    selected = features.distribution_extras(ROOT, target='win32-x64')
    assert {'audio-io', 'wake-sherpa', 'edge-tts', 'mcp', 'web', 'doc-extract', 'semantier-windows'} <= set(selected)
    assert not {'all', 'voice', 'wake', 'stt-whisper', 'kittentts', 'neutts'} & set(selected)
    assert features.distribution_extras(ROOT, target='linux-x64') is None


def test_product_policy_never_reacquires_excluded_extra_even_if_importable(monkeypatch):
    from pm import extras
    monkeypatch.setattr(features, 'runtime_distribution_extras', lambda: ['audio-io'])
    assert not extras.extra_supported('stt-whisper', importable=lambda _: True)
    assert extras.extra_supported('audio-io', importable=lambda _: True)


def test_selected_cache_keeps_only_resolved_and_build_requirements(tmp_path):
    from pm.uv_cache_prune import prune_uv_cache_to_lock
    project = tmp_path / 'project'; project.mkdir()
    (project / 'pyproject.toml').write_text('[build-system]\nrequires=["setuptools==83.0.0", "wheel"]\n')
    (project / 'uv.lock').write_text('[[package]]\nname="faster-whisper"\n')
    cache = tmp_path / 'cache'
    for name in ['faster_whisper', 'numpy', 'setuptools', 'wheel']:
        (cache / 'archive-v0' / name / (name+'-1.0.dist-info')).mkdir(parents=True)
    prune_uv_cache_to_lock(cache, project, selected_packages={'numpy'})
    assert not (cache / 'archive-v0/faster_whisper').exists()
    assert all((cache / 'archive-v0' / name).exists() for name in ['numpy', 'setuptools', 'wheel'])


def test_product_default_and_explicit_omitted_provider_do_not_fall_back(monkeypatch):
    from tools import transcription_tools as stt
    monkeypatch.setattr(features, 'runtime_distribution_extras', lambda: ['semantier-windows'])
    monkeypatch.setattr(stt, '_detect_local_backend', lambda: pytest.fail('automatic backend selection'))
    assert stt._get_provider({}) == 'sensevoice'
    assert stt._get_provider({'provider': 'local'}) == 'none'
    assert stt._get_provider({'provider': 'custom-saved'}) == 'custom-saved'


def test_ambiguous_saved_local_model_remains_local_and_config_is_not_rewritten(monkeypatch):
    from hermes_cli import config
    monkeypatch.setattr(features, 'runtime_distribution_extras', lambda: ['semantier-windows'])
    raw = {'stt': {'local': {'model': 'small'}}}
    merged = {'stt': {'provider': 'sensevoice', 'local': {'model': 'small'}}}
    config._preserve_product_stt_selection(merged, raw)
    assert merged['stt']['provider'] == 'local'
    assert raw == {'stt': {'local': {'model': 'small'}}}
