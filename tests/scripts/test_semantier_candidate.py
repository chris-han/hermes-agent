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
