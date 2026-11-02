import pytest
from scripts.releases.simple_native_slots import advance


def head(build, sequence):
    return {'buildId':build*32,'sequence':sequence,'manifestKey':'releases/channel-builds/'+build*32+'/build.json','sha256':build*64}


def test_previous_active_becomes_archive_without_changing_other_fields():
    old=head('a',1); new=head('b',2)
    before={'nativeHead':old,'revision':3,'nextSequence':3,'identity':{'x':1}}
    after=advance(before,new)
    assert after['nativeHead']==new and after['nativeArchive']==old
    assert after['identity']==before['identity'] and after['revision']==4
    assert before['nativeHead']==old


def test_stale_or_unallocated_sequence_refused():
    with pytest.raises(ValueError): advance({'nativeHead':head('a',2),'revision':1,'nextSequence':3},head('b',2))
    with pytest.raises(ValueError): advance({'nativeHead':head('a',1),'revision':1,'nextSequence':2},head('b',2))


def test_same_head_is_idempotent():
    record={'nativeHead':head('a',1),'nativeArchive':None,'revision':3,'nextSequence':2}
    assert advance(record,head('a',1))==record
