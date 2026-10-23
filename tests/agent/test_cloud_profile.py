"""Cloud profile request configuration through the profile-owned production bridge."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import pytest
import yaml


def test_profile_scoped_catalog_and_actual_request_fields(tmp_path, monkeypatch):
    from hermes_cli.config import load_config_readonly
    from agent.semantier_gateway import cloud_catalog, configure_cloud_request
    seen=[]
    profile={'id':'private-alias','revision':'r1','context_window':8000,'max_output_tokens':100,
        'capabilities':{'tools':None,'vision':False,'reasoning':False,'structured_output':False,'stream':True},
        'request_parameters':{'output':'max_tokens','reasoning_controls':[]}}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_GET(self):
            seen.append(self.headers['Authorization'])
            data=json.dumps({'object':'list','data':[profile]}).encode()
            self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}/v1'
    monkeypatch.setenv('CLOUD_TEST_KEY','scoped-test-key')
    monkeypatch.setenv('HERMES_HOME',str(tmp_path))
    cfg={'semantier_gateway':{'enabled':True,'base_url':base,'api_key_env':'CLOUD_TEST_KEY'},
         'model':{'default':'private-alias','provider':'custom','base_url':base}}
    (tmp_path/'config.yaml').write_text(yaml.safe_dump(cfg))
    try:
        assert cloud_catalog()['data'][0]['id']=='private-alias'
        assert cloud_catalog()['data'][0]['revision']=='r1' and len(seen)==1
        body=configure_cloud_request({'model':'private-alias','messages':[{'role':'user','content':'hello'}],
            'tools':[{'function':{'name':'test'}}],'reasoning_effort':'high','temperature':.7},base_url=base)
        assert body['max_tokens']==100 and 'tools' not in body and 'reasoning_effort' not in body
        assert body['temperature']==.7
        assert body['extra_headers']['X-Semantier-Capability-Revision']=='r1'
        other=tmp_path/'other';other.mkdir();monkeypatch.setenv('HERMES_HOME',str(other))
        (other/'config.yaml').write_text(yaml.safe_dump(cfg))
        cloud_catalog();assert len(seen)==2
        assert configure_cloud_request({'model':'byok'},base_url='http://127.0.0.1:2/v1')=={'model':'byok'}
    finally:
        server.shutdown();server.server_close();thread.join()
