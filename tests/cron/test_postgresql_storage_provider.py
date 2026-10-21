"""Hosted cron authority cannot silently fall back to standalone JSON files."""
import pytest


def test_postgresql_missing_provider_refuses_before_file_fallback(monkeypatch,tmp_path):
    from cron import jobs,storage_provider
    monkeypatch.setenv('HERMES_STATE_BACKEND','postgresql')
    monkeypatch.setattr(storage_provider,'_provider',None)
    with jobs.use_cron_store(tmp_path):
        with pytest.raises(RuntimeError,match='PostgreSQL cron storage.*provider'):
            jobs.load_jobs()
    assert not (tmp_path/'cron').exists()


def test_registered_host_provider_and_standalone_sqlite_are_preserved(monkeypatch,tmp_path):
    from cron import jobs,storage_provider
    marker=object()
    monkeypatch.setenv('HERMES_STATE_BACKEND','postgresql')
    monkeypatch.setattr(storage_provider,'_provider',marker)
    assert storage_provider.get_storage_provider(required=True) is marker
    monkeypatch.setenv('HERMES_STATE_BACKEND','sqlite')
    monkeypatch.setattr(storage_provider,'_provider',None)
    with jobs.use_cron_store(tmp_path):assert jobs.load_jobs()==[]
