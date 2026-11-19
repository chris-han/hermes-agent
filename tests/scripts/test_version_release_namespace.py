import pytest
from hermes_cli import release_channels as protocol
from scripts.releases.semantier_windows import acceptance_request


def request():
    return acceptance_request(commit='a'*40, controller='b'*40, version='0.0.132',
                              source_version='0.0.0', base='https://cloud.semantier.cn')


def test_acceptance_emits_canonical_version_directory():
    value=request()
    assert value['buildId']==value['windowsVersion']=='0.0.132.0'
    assert protocol.build_prefix(value['buildId'])=='releases/channel-builds/0.0.132.0/'
    assert protocol.validate_request(value)==value


def test_version_namespace_must_match_signed_windows_version():
    value=request();value['buildId']='0.0.133.0'
    with pytest.raises(protocol.ChannelError):protocol.validate_request(value)


@pytest.mark.parametrize('value',['0.0.0132.0','0.0.65536.0','65536.2.3.4','1.2.3.65536','0.0.132','v0.0.132.0','0.0.132.0/../x'])
def test_noncanonical_versions_refused(value):
    with pytest.raises(protocol.ChannelError):protocol.build_prefix(value)


@pytest.mark.parametrize('value',['1.2.132.4','10.20.30.40','65535.65535.65535.65535'])
def test_all_four_version_components_are_variable(value):
    assert protocol.build_prefix(value)=='releases/channel-builds/'+value+'/'


def test_historical_guid_namespace_stays_readable():
    assert protocol.build_prefix('c'*32)=='releases/channel-builds/'+'c'*32+'/'


def test_version_namespace_binds_the_actual_version_independent_of_sequence():
    value=request();value.update(buildId='1.2.132.4',windowsVersion='1.2.132.4',version='1.2.132',releaseTag='v1.2.132')
    assert protocol.validate_request(value)==value
