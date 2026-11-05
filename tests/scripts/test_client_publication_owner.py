"""Publisher entry points work without importing the frozen Runtime snapshot."""
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize('entry', ['publisher', 'stage', 'storage'])
def test_publication_entry_without_runtime(tmp_path, entry):
    root = Path(__file__).resolve().parents[2]
    program = '''
import importlib.abc
import runpy
import sys
from pathlib import Path

root, work, entry = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(root))
class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'agents', 'contracts'}:
            raise AssertionError('publisher imported Runtime: ' + fullname)
sys.meta_path.insert(0, NoRuntime())
if entry == 'stage':
    sys.argv = ['stage_local_native', '--help']
    runpy.run_path(str(root / 'scripts/qualification/stage_local_native.py'), run_name='__main__')
elif entry == 'storage':
    sys.argv = ['ecs_endpoint', '--help']
    runpy.run_module('client_operations.ecs_endpoint', run_name='__main__')
else:
    from client_operations.artifact_io import immutable_write, file_facts
    from client_operations.publication import Publisher
    from client_operations.publications import Publications
    from client_operations.targets import DeliveryTargets
    from client_operations.store import Store
    from client_operations.ecs_target import ECSTarget
    from client_operations.r2_target import R2Target
    from client_operations.contracts import Promotion, canonical
    target = work / 'payload'
    immutable_write(target, canonical({'payload': 'signed fixture'}))
    immutable_write(target, canonical({'payload': 'signed fixture'}))
    assert file_facts(target)[0] > 0
    assert Promotion(release_id='release-a', expected_revision=1, idempotency_key='retry-a').release_id == 'release-a'
    owner = Publisher(None, work / 'public', work, 'https://example.test', b'key', {}, 'acceptance')
    assert owner.empty_record()['head'] is None
    assert Path(__import__('client_operations.store', fromlist=['x']).__file__).with_name('schema.sql').is_file()
'''
    result = subprocess.run(
        [sys.executable, '-I', '-c', program, str(root), str(tmp_path), entry],
        cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr
