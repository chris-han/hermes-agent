from pathlib import Path
import xml.etree.ElementTree as ET
import pytest
from scripts.releases.semantier_candidate import stage
from scripts.bundles.release_artifacts import write_appinstaller


def test_manual_context_cannot_mint_ci_receipt(tmp_path,monkeypatch):
    monkeypatch.delenv('GITHUB_ACTIONS',raising=False)
    with pytest.raises(ValueError,match='CI context'):stage(tmp_path,tmp_path,tmp_path,{'en':'Unqualified'})


def test_single_msix_descriptor_preserves_pinned_update_policy(tmp_path):
    path=tmp_path/'fixed.appinstaller'
    write_appinstaller(path,identity='Semantier.Agent.Acceptance',publisher='CN=Semantier Technology, O=Semantier Technology',version='0.0.37.0',self_uri='http://localhost/fixed.appinstaller',artifact_uri='http://localhost/app.msix',update_policy='pinned',package_arch='x64')
    root=ET.fromstring(path.read_bytes());main=list(root)
    assert len(main)==1 and main[0].tag.endswith('MainPackage')
    assert main[0].get('ProcessorArchitecture')=='x64'
    write_appinstaller(path,identity='Upstream',publisher='Upstream',version='1.0.0.0',self_uri='https://example/fixed.appinstaller',artifact_uri='https://example/app.msixbundle')
    root=ET.fromstring(path.read_bytes())
    assert next(iter(root)).tag.endswith('MainBundle')
    assert list(root)[1].tag.endswith('UpdateSettings')


def test_parent_snapshot_hash_and_path_are_verified_without_source_rewrites(tmp_path):
    import hashlib,json
    from scripts.bundles.parent_composition import verify
    source=tmp_path/'src';source.mkdir();file=source/'contract.py';file.write_text('value=1\n')
    manifest={'parent_commit':'a'*40,'files':[{'path':'src/contract.py','sha256':hashlib.sha256(file.read_bytes()).hexdigest()}]}
    (tmp_path/'composition.json').write_text(json.dumps(manifest))
    assert verify(tmp_path)==manifest
    file.write_text('value=2\n')
    with pytest.raises(ValueError,match='integrity'):verify(tmp_path)
    manifest['files'][0]['path']='../outside.py'
    (tmp_path/'composition.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='integrity'):verify(tmp_path)


def test_staging_survives_unavailable_kernel_fast_copy(tmp_path, monkeypatch):
    """WSL sendfile can raise ENOMEM after a partial cross-filesystem copy."""
    import errno
    import hashlib
    import json
    import os
    from scripts.releases.semantier_windows import acceptance_request

    request = acceptance_request(commit='a'*40, controller='b'*40, version='0.0.44',
                                 source_version='0.22.2', base='http://127.0.0.1:8926')
    output = tmp_path/'output'
    native = tmp_path/'native'
    incoming = tmp_path/'incoming'
    distribution = output/'signed/distribution'
    distribution.mkdir(parents=True)
    native.mkdir()
    artifact = distribution/'candidate.msix'
    artifact.write_bytes(b'bounded transfer fixture\n' * 150000)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    archive = output/'signed/candidate.zip'
    archive.write_bytes(b'archive transfer fixture\n' * 150000)
    archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix('.zip.sha256').write_text(archive_digest+'  candidate.zip\n')
    publisher = 'CN=Semantier Technology, O=Semantier Technology'
    records = {
        output/'result.json': {'result': 'PASS', 'sourceCommit': request['commit']},
        output/'channel-request.json': request,
        output/'signed/receipt.json': {'signing': 'PASS', 'verification': 'PASS: fixture',
            'packageName': 'Semantier.Agent.Acceptance', 'publisher': publisher,
            'packageVersion': '0.0.44.0', 'signedSha256': digest,
            'archiveName': archive.name, 'trustedTimestamp': False},
        native/'installed-identity.json': {'version': '0.0.44.0', 'publisher': publisher,
            'stamp': {'commit': request['commit']},
            'packageFullName': 'Semantier.Agent.Acceptance_0.0.44.0_x64__fixture'},
        native/'native-install-exit.json': {'exitCode': 0},
        native/'artifact.json': {'sha256': digest},
    }
    for path, value in records.items():
        path.write_text(json.dumps(value))
    for key, value in {'GITHUB_ACTIONS': 'true', 'GITHUB_REPOSITORY': 'chris-han/hermes-agent',
                       'GITHUB_SHA': request['commit'], 'GITHUB_RUN_ID': 'fixture',
                       'GITHUB_RUN_ATTEMPT': '1', 'SEMANTIER_CLIENT_RECEIPT_KEY': 'x'*32}.items():
        monkeypatch.setenv(key, value)

    def unavailable(*args, **kwargs):
        raise OSError(errno.ENOMEM, 'kernel fast copy unavailable')

    def partial_then_unavailable(out_fd, in_fd, offset, count):
        if offset:
            raise OSError(errno.ENOMEM, 'kernel fast copy failed after partial transfer')
        os.lseek(in_fd, offset, os.SEEK_SET)
        return os.write(out_fd, os.read(in_fd, 4096))

    monkeypatch.setattr(os, 'sendfile', partial_then_unavailable, raising=False)
    monkeypatch.setattr(os, 'copy_file_range', unavailable, raising=False)
    result = stage(output, native, incoming, {'en': 'Unit fixture, not CI evidence'})
    candidate = json.loads(Path(result['candidate_path']).read_text())
    package = candidate['manifest']['packages'][0]
    assert (incoming/package['artifact']['key']).read_bytes() == artifact.read_bytes()
    assert (incoming/package['distributionArchive']['key']).read_bytes() == archive.read_bytes()
    with pytest.raises(ValueError, match='Fresh immutable'):
        stage(output, native, incoming, {'en': 'must not overwrite'})
