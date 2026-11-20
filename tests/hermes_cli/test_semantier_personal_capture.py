"""Bounded actual-profile capture; no managed admission is inferred locally."""
import pytest
from hermes_constants import set_hermes_home_override, reset_hermes_home_override


@pytest.fixture
def profile(tmp_path):
    token=set_hermes_home_override(tmp_path)
    (tmp_path/'skills'/'my-report').mkdir(parents=True)
    (tmp_path/'skills'/'my-report'/'SKILL.md').write_text('---\nname: my-report\ndescription: personal report\n---\nCheck the report and present a table.\n')
    (tmp_path/'memories').mkdir()
    (tmp_path/'memories'/'MEMORY.md').write_text('Use a concise table.')
    yield tmp_path
    reset_hermes_home_override(token)


def test_capture_is_exact_bounded_and_retains_revision(profile,monkeypatch):
    from hermes_cli.semantier_personal_capture import snapshot, capture
    monkeypatch.setenv('SEMANTIER_RUNTIME','1')
    scope={'organization_id':'org','workspace_id':'ws','owner_ref':'personal-user'}
    selected=snapshot('my-report')
    assert len(selected['contexts'])==1
    entry=selected['contexts'][0]
    first=capture(selected,entry,scope=scope,incarnation='a'*32)
    assert first.skill.files[0].content_bytes()==(profile/'skills'/'my-report'/'SKILL.md').read_bytes()
    assert first.decision_context.content=='Use a concise table.'
    assert first.decision_context.origin['source_hash']==first.skill.artifact_hash
    assert capture(selected,entry,scope=scope,incarnation='a'*32,previous=first.model_dump(mode='json'))==first
    (profile/'skills'/'my-report'/'SKILL.md').write_text('---\nname: my-report\ndescription: report\n---\nCheck the report before writing.\n')
    changed=snapshot('my-report')
    with pytest.raises(ValueError,match='ACK'):
        capture(changed,entry,scope=scope,incarnation='a'*32,previous=first.model_dump(mode='json'))
    ack={'artifact_kind':'personal_capability_import','artifact_ref':'ack-one','artifact_hash':'sha256:'+'1'*64}
    successor=capture(changed,entry,scope=scope,incarnation='a'*32,previous=first.model_dump(mode='json'),acknowledgement=ack)
    assert successor.skill.version==2
    assert successor.skill.predecessor==first.skill.pin()
    assert successor.decision_context==first.decision_context
    assert successor.context_basis==first.context_basis
    assert successor.predecessor.artifact_ref=='ack-one'
    with pytest.raises(ValueError,match='scope'):
        capture(changed,entry,scope=scope|{'owner_ref':'other'},incarnation='a'*32,previous=first.model_dump(mode='json'),acknowledgement=ack)


def test_capture_refuses_stale_context_and_disallowed_source_files(profile,monkeypatch):
    from hermes_cli.semantier_personal_capture import snapshot, capture
    monkeypatch.setenv('SEMANTIER_RUNTIME','1')
    selected=snapshot('my-report')
    entry=selected['contexts'][0]
    with pytest.raises(ValueError,match='context'):
        capture(selected,entry|{'content':'invented text'},scope={'organization_id':'org','workspace_id':'ws','owner_ref':'u'},incarnation='a'*32)
    (profile/'skills'/'my-report'/'credentials.db').write_bytes(b'secret')
    with pytest.raises(ValueError):
        snapshot('my-report')


def test_capture_never_harvests_other_skills_or_symlinks(profile,monkeypatch):
    from hermes_cli.semantier_personal_capture import snapshot
    monkeypatch.setenv('SEMANTIER_RUNTIME','1')
    other=profile/'skills'/'other'
    other.mkdir()
    (other/'private.txt').write_text('unselected')
    selected=snapshot('my-report')
    assert [f['relative_path'] for f in selected['files']]==['SKILL.md']
    (profile/'skills'/'my-report'/'linked').symlink_to(other,target_is_directory=True)
    with pytest.raises(ValueError):
        snapshot('my-report')
    monkeypatch.setenv('SEMANTIER_RUNTIME','0')
    (profile/'skills'/'my-report'/'linked').unlink()
    assert snapshot('my-report')['skill_id']=='my-report'


def test_native_routes_capture_current_profile_and_reject_unknown_fields(profile):
    """Router fixture qualifies input/scoping, not gateway authentication."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from hermes_cli.web_routers.skills import router
    app=FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        response=client.post('/api/skills/personal/snapshot',json={'skill_id':'my-report'})
        assert response.status_code==200,response.text
        selected=response.json()
        assert (client.post('/api/skills/personal/snapshot',json={'skill_id':'my-report','path':'/other'})).status_code==422
        result=client.post('/api/skills/personal/capture',json={'selected':selected,'context_record':selected['contexts'][0],
            'scope':{'organization_id':'org','workspace_id':'ws','owner_ref':'u'},'incarnation':'a'*32})
        assert result.status_code==200,result.text
        assert result.json()['skill']['skill_id']=='my-report'
        refused=client.post('/api/skills/personal/install?profile=current',json={'package':{},'confirmed':False})
        assert refused.status_code==403,refused.text


def test_capture_does_not_relabel_distributed_package_as_new_personal_authorship(profile,monkeypatch):
    from hermes_cli.semantier_personal_capture import snapshot
    from tools.skills_hub import HubLockFile
    monkeypatch.setattr(HubLockFile,'get_installed',lambda self,name:{'source':'semantier-personal','metadata':{'author_ref':'another-account'}})
    with pytest.raises(ValueError,match='retains its existing source'):
        snapshot('my-report')
