import pytest
from scripts.releases.semantier_windows import acceptance_request,IDENTITY
from scripts.bundles.desktop_prepare import validate_channel_request


def test_consecutive_versions_keep_family_and_provenance():
    first=acceptance_request(commit='a'*40,controller='b'*40,version='0.0.36',source_version='0.22.2',base='http://127.0.0.1:8926')
    second=acceptance_request(commit='c'*40,controller='b'*40,version='0.0.37',source_version='0.22.2',base='http://127.0.0.1:8926')
    assert first['bundleEnv']==second['bundleEnv']=={'HERMES_HOME':None,'HERMES_DATA_DIR_SUFFIX':'-semantier-acceptance'}
    assert first['identity']==second['identity']==IDENTITY
    assert first['commit']!=second['commit'] and first['windowsVersion']!=second['windowsVersion']
    assert validate_channel_request(first)==first
    with pytest.raises(ValueError):validate_channel_request({**first,'semantierScope':'other'})
    with pytest.raises(ValueError):acceptance_request(commit='a'*40,controller='b'*40,version='0.0.36',source_version='0.22.2',base='http://untrusted.example')
