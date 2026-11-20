"""Actual native daemon transport; portable source proof, not Windows acceptance."""
import httpx
import pytest

from tests.gateway.test_native_http_auth import daemon,headers,ticket


@pytest.mark.platforms('linux')
def test_personal_capture_routes_use_current_native_grant_and_profile(tmp_path):
    def prepare(home):
        (home/'skills'/'my-report').mkdir(parents=True)
        (home/'skills'/'my-report'/'SKILL.md').write_text('---\nname: my-report\ndescription: report\n---\nCheck the report and present a table.\n')
        (home/'memories').mkdir()
        (home/'memories'/'MEMORY.md').write_text('Use a concise table.')
    with daemon(tmp_path,prepare=prepare) as (home,descriptor),httpx.Client(base_url=descriptor['api_origin'],trust_env=False) as client:
        path='/api/skills/personal/snapshot?profile=default'
        assert client.post(path,json={'skill_id':'my-report'}).status_code==401
        token=ticket(home,descriptor)
        response=client.post(path,headers=headers(token),json={'skill_id':'my-report'})
        assert response.status_code==200,response.text
        assert client.post(path,headers=headers(token),json={'skill_id':'my-report'}).status_code==401
        assert client.post('/api/skills/personal/snapshot?profile=other',headers=headers(ticket(home,descriptor)),json={'skill_id':'my-report'}).status_code==403
        selected=response.json()
        captured=client.post('/api/skills/personal/capture?profile=default',headers=headers(ticket(home,descriptor)),
            json={'selected':selected,'context_record':selected['contexts'][0],
                  'scope':{'organization_id':'org','workspace_id':'ws','owner_ref':'personal-user'},'incarnation':'a'*32})
        assert captured.status_code==200,captured.text
        assert captured.json()['decision_context']['content']=='Use a concise table.'
        assert captured.json()['skill']['files']==selected['files']
        refused=client.post('/api/skills/personal/install?profile=default',headers=headers(ticket(home,descriptor)),json={'package':{},'confirmed':False})
        assert refused.status_code==403,refused.text
