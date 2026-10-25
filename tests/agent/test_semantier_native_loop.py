"""Real native Hermes tool + helper HTTP calls; external model behavior is deterministic."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import yaml


def test_native_read_tool_loop_and_real_helper_attribution(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.chdir(tmp_path)
    evidence = tmp_path / 'evidence.txt'
    evidence.write_text('native read evidence')
    requests = []

    class Upstream(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != '/api/cloud-provider/catalog':
                self.send_error(404)
                return
            payload = {'object':'list','data':[{'id':'fixture-model','label':'Fixture',
                'context_window':128000,'max_output_tokens':8192,'protocol':'chat',
                'revision':'fixture-r1','account_id':'fixture-account',
                'capabilities':{'tools':True,'stream':False,'vision':False,'reasoning':False,'structured_output':False},
                'request_parameters':{'output':'max_tokens','reasoning_controls':[]}}]}
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())
        def do_POST(self):
            if self.path != '/v1/chat/completions':
                self.send_error(404)
                return
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append((dict(self.headers), body))
            is_chat = self.headers.get('X-Semantier-Purpose') == 'chat'
            has_tool = any(message.get('role') == 'tool' for message in body['messages'])
            message = {'role': 'assistant', 'content': 'native read evidence' if is_chat else '{"title":"Read evidence"}'}
            reason = 'stop'
            if is_chat and not has_tool:
                message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                    'id': 'call-read', 'type': 'function', 'function': {
                        'name': 'read_file', 'arguments': json.dumps({'path': str(evidence)})}}]}
                reason = 'tool_calls'
            payload = {'id': 'completion-local', 'object': 'chat.completion', 'created': 1,
                       'model': body['model'], 'choices': [{'index': 0, 'message': message,
                                                          'finish_reason': reason}],
                       'usage': {'prompt_tokens': 20, 'completion_tokens': 10, 'total_tokens': 30}}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f'http://127.0.0.1:{server.server_port}/v1'
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump({
        'billing': {'backend': 'semantier'},
        'semantier_gateway': {'enabled': True, 'base_url': endpoint, 'api_key_env': 'TEST_GATEWAY_TOKEN'},
        'auxiliary': {'transient_retries': 0}, 'compression': {'enabled': False},
        'model': {'context_length': 128000},
    }))
    (tmp_path / '.env').write_text('TEST_GATEWAY_TOKEN=local-token\n')
    agent = None
    try:
        from run_agent import AIAgent
        from agent.title_generator import generate_title
        agent = AIAgent(model='fixture-model', provider='custom', base_url=endpoint, api_key='local-token',
                        session_id='native-session', enabled_toolsets=['file'], max_iterations=4,
                        skip_context_files=True, skip_memory=True, skip_background_review=True,
                        quiet_mode=True, cwd=str(tmp_path), max_tokens=128,
                        fallback_model={'provider': 'openai', 'model': 'unmetered'})
        assert agent._fallback_chain == []
        agent._disable_streaming = True
        result = agent.run_conversation('Read evidence.txt.', task_id='native-run')
        assert not result.get('error'), result
        assert any(message.get('role') == 'tool' and 'native read evidence' in str(message.get('content'))
                   for message in result['messages'])
        helper = generate_title('Read evidence.txt.', timeout=5, main_runtime={
            'session_id': 'native-session', 'run_id': 'native-run', 'provider': 'custom',
            'model': 'fixture-model', 'base_url': endpoint, 'api_key': 'local-token', 'api_mode': 'chat_completions'})
        assert helper == 'Read evidence'
        chat = [(headers, body) for headers, body in requests if headers.get('X-Semantier-Purpose') == 'chat']
        title = [(headers, body) for headers, body in requests if headers.get('X-Semantier-Purpose') == 'title']
        assert len(chat) == 2 and len(title) == 1
        assert all(headers['Authorization'] == 'Bearer local-token' for headers, _ in requests)
        assert all(headers['X-Semantier-Session-Id'] == 'native-session' for headers, _ in requests)
        assert title[0][0]['X-Semantier-Run-Id'] == 'native-run'
        assert all(headers.get('X-Semantier-Run-Id') for headers, _ in chat)
        assert len({headers['Idempotency-Key'] for headers, _ in requests}) == 3
        assert all(isinstance(body.get('max_tokens', body.get('max_completion_tokens')), int) for _, body in requests)
        allowed = {'model', 'messages', 'max_tokens', 'max_completion_tokens', 'stream', 'temperature',
                   'top_p', 'stop', 'tools', 'tool_choice', 'parallel_tool_calls', 'response_format',
                   'presence_penalty', 'frequency_penalty', 'seed', 'stream_options', 'reasoning_effort'}
        assert all(not (set(body) - allowed) for _, body in requests), [set(body) - allowed for _, body in requests]

    finally:
        if agent:
            agent.close()
        server.shutdown()
        thread.join()
        server.server_close()
