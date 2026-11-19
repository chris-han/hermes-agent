"""Frozen import consumes exactly the bytes bound by its operator request."""
from contextlib import contextmanager
import hashlib
import json
from types import SimpleNamespace
import sqlite3

import pytest

from hermes_state_postgresql_portability import PostgreSQLPortabilityMixin


class SnapshotReader(PostgreSQLPortabilityMixin):
    scope = SimpleNamespace(schema='fixture', identity=('tenant','workspace','native'))

    @contextmanager
    def _transaction(self, *, read_only=False):
        class Connection:
            def execute(self, query, params=()):
                result = {'permitted':True} if read_only else {'receipt_json':json.dumps({'source_digest':params[0]})}
                return SimpleNamespace(fetchone=lambda:result)
        yield Connection()

    def table(self, name):
        return name

    def _lock_conversation(self, *args):
        pass


def snapshot(path, value):
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE project_meta(key TEXT, value TEXT)')
        connection.execute('INSERT INTO project_meta VALUES(?,?)', ('key',value))
    return path.read_bytes()


def test_changed_frozen_source_is_refused(tmp_path):
    path = tmp_path/'projects.db'
    original = snapshot(path, 'original')
    path.write_bytes(b'changed after operator preflight')
    with pytest.raises(ValueError, match='snapshot hash'):
        SnapshotReader().import_sqlite_snapshot(path, family='projects', source_sha256=hashlib.sha256(original).hexdigest())


def test_path_replacement_after_hash_cannot_change_imported_content(tmp_path, monkeypatch):
    path = tmp_path/'projects.db'
    original = snapshot(path, 'original')
    replacement = snapshot(tmp_path/'replacement.db', 'changed')
    connect = sqlite3.connect

    def replacing_connect(*args, **kwargs):
        assert args[0] == ':memory:', 'Pinned imports must not reopen the mutable source pathname'
        path.write_bytes(replacement)
        return connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, 'connect', replacing_connect)
    receipt = SnapshotReader().import_sqlite_snapshot(path, family='projects', source_sha256=hashlib.sha256(original).hexdigest())
    content = {'contract':'native-postgresql-import-v1','family':'projects',
               'tables':{'project_meta':[{'key':'key','value':'original'}]},'sequences':{}}
    expected = hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
    assert receipt['source_digest'] == expected
