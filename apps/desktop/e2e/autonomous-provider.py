"""Transparent real-Qwen observer for the optional, explicitly authorized Desktop eval.

The real credential never enters the app, fixtures, argv or retained traces. This
service changes only the declared output cap/usage request; it never creates an
assistant reply or chooses a tool. Unsupported requests fail rather than mock.
"""
import hashlib
import json
import os
from pathlib import Path
import secrets
import re
from urllib.parse import urlsplit
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.error
import urllib.request

ROOT = Path(os.environ['SEMANTIER_EVAL_OUTPUT'])
ROOT.mkdir(parents=True, exist_ok=True)
SECRET_NAMES = ('DASHSCOPE_API_KEY', 'DASHSCOPE_BASE_URL')
values = {name: os.environ.get(name, '') for name in SECRET_NAMES}
with Path(os.environ['SEMANTIER_EVAL_CREDENTIALS_ENV']).open(encoding='utf-8-sig') as source:
    for line in source:
        name, separator, value = line.partition('=')
        name = name.strip().removeprefix('export ')
        if separator and name in values and not values[name]:
            values[name] = value.strip().strip('"').strip("'")
KEY, BASE = values[SECRET_NAMES[0]], values[SECRET_NAMES[1]].rstrip('/')
if not KEY or not BASE.startswith('https://'):
    raise SystemExit('Configured Qwen credential/HTTPS provider missing')
TOKEN = secrets.token_hex(24)  # fixture capability, not the real provider credential
LOCK = threading.Lock()
STATE = {'case': None, 'counts': {}, 'calls': [], 'failures': [], 'budget_per_case': 12}
HINT = (Path(__file__).resolve().parents[3] / 'semantier-runtime/src/prompts/agents/DESKTOP_BROWSER_CONTEXT.md').read_text(encoding='utf-8').strip()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def scrub(value):
    if isinstance(value, dict):
        return {k: '[omitted model reasoning]' if k in ('reasoning_content', 'reasoning') else scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value.replace(KEY, '[redacted]') if isinstance(value, str) else value


def save():
    (ROOT / 'provider-state.json').write_text(json.dumps(scrub(STATE), indent=2), encoding='utf-8')


def normalize_response(raw, content_type):
    if 'text/event-stream' not in content_type:
        reply = json.loads(raw)
        choice = (reply.get('choices') or [{}])[0]
        return {'model': reply.get('model'), 'message': choice.get('message'), 'usage': reply.get('usage'), 'finish_reason': choice.get('finish_reason')}
    content, tool_calls, usage, model, finish = [], {}, None, None, None
    for line in raw.decode('utf-8').splitlines():
        if not line.startswith('data: ') or line[6:] == '[DONE]':
            continue
        part = json.loads(line[6:])
        model = part.get('model') or model
        usage = part.get('usage') or usage
        for choice in part.get('choices', []):
            finish = choice.get('finish_reason') or finish
            delta = choice.get('delta', {})
            if delta.get('content'):
                content.append(delta['content'])
            for tool in delta.get('tool_calls', []):
                target = tool_calls.setdefault(tool['index'], {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                target['id'] += tool.get('id') or ''
                target['function']['name'] += tool.get('function', {}).get('name') or ''
                target['function']['arguments'] += tool.get('function', {}).get('arguments') or ''
    return {'model': model, 'usage': usage, 'finish_reason': finish, 'message': {'role': 'assistant', 'content': ''.join(content), 'tool_calls': list(tool_calls.values())}}


def forbidden_tool(calls):
    allowed = {'desktop_preview', 'drive_preview', 'browser_exec', 'todo_list', 'clarify', 'focus_pane', 'annotate_preview'}
    for call in calls or []:
        function = call.get('function', {})
        name = function.get('name', '')
        try:
            args = json.loads(function.get('arguments') or '{}')
        except ValueError:
            continue  # Canonical tool argument validation owns invalid JSON.
        if name == 'tool_call':
            nested = [{'function': {'name': item.get('name'), 'arguments': json.dumps(item.get('arguments', {}))}} for item in args.get('calls', [])]
            violation = forbidden_tool(nested)
            if violation:
                return violation
            continue
        if name not in allowed:
            return 'tool_outside_authorized_browser_fixture: ' + str(name)
        raw = json.dumps(args)
        for url in re.findall(r'https?://[^\s\"\'<>]+', raw):
            if urlsplit(url).hostname not in ('127.0.0.1', 'localhost'):
                return 'external_website_attempt'
        if re.search(r'(?:[Cc]:[\\/]+Users|import\s+(?:os|subprocess|shutil)|from\s+(?:pathlib|subprocess))', raw):
            return 'host_data_or_process_attempt'
    return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def authorized(self):
        supplied = self.headers.get('X-Fixture-Token') or self.headers.get('Authorization', '').removeprefix('Bearer ')
        return secrets.compare_digest(supplied, TOKEN)

    def reply(self, code, value):
        raw = json.dumps(value).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if not self.authorized():
            return self.reply(401, {'error': 'fixture observer capability required'})
        if self.path == '/__fixture/state':
            return self.reply(200, scrub(STATE))
        if self.path.rstrip('/') not in ('/v1/models', '/models'):
            return self.reply(404, {'error': 'unsupported observer route'})
        self.forward(None)

    def do_POST(self):
        if not self.authorized():
            return self.reply(401, {'error': 'fixture observer capability required'})
        data = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
        if self.path == '/__fixture/case':
            case = data.get('case')
            if case not in ('A', 'B', 'C', 'D', 'E', 'F', None):
                return self.reply(400, {'error': 'unknown frozen case'})
            with LOCK:
                STATE['case'] = case
                save()
            return self.reply(200, {'case': case})
        if self.path.rstrip('/') not in ('/v1/chat/completions', '/chat/completions'):
            return self.reply(404, {'error': 'unsupported observer route'})
        if data.get('model') != 'qwen3.5-plus':
            return self.reply(400, {'error': 'Only frozen qwen3.5-plus is authorized'})
        self.forward(data)

    def forward(self, body):
        record = None
        if body is not None:
            with LOCK:
                case = STATE['case']
                count = STATE['counts'].get(case, 0)
                if STATE['failures']:
                    return self.reply(400, {'error': 'Evaluation composition/scope stopped'})
                if not case or count >= STATE['budget_per_case']:
                    return self.reply(429, {'error': 'frozen trial call budget exhausted'})
                names = [tool.get('function', {}).get('name') for tool in body.get('tools', [])]
                system = '\n'.join(str(m.get('content', '')) for m in body.get('messages', []) if m.get('role') in ('system', 'developer'))
                if not {'desktop_preview', 'drive_preview', 'browser_exec'}.issubset(names) or HINT not in system:
                    STATE['failures'].append({'case': case, 'type': 'missing_effective_prompt_or_tools', 'at': stamp(), 'available_tools': names, 'hint_loaded': HINT in system})
                    save()
                    return self.reply(503, {'error': 'Evaluation composition missing required effective browser prompt/tools'})
                body['max_tokens'] = 4096
                if body.get('stream'):
                    body['stream_options'] = {**body.get('stream_options', {}), 'include_usage': True}
                STATE['counts'][case] = count + 1
                record = {'case': case, 'index': count + 1, 'at': stamp(), 'system_prompt_sha256': digest(system), 'browser_hint_sha256': digest(HINT), 'request': scrub(body), 'request_sha256': digest(json.dumps(body, sort_keys=True))}
                STATE['calls'].append(record)
                save()
        target = BASE + ('/chat/completions' if body is not None else '/models')
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(target, data=data, headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'})
        start = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=150) as response:
                code, content_type = response.status, response.headers.get('Content-Type', 'application/json')
                raw = response.read()
            if record is not None:
                with LOCK:
                    record.update(response=scrub(normalize_response(raw, content_type)), duration_ms=round((time.monotonic() - start) * 1000), http_status=code)
                    violation = forbidden_tool(record['response'].get('message', {}).get('tool_calls'))
                    if violation:
                        record['harness_delivery_blocked'] = violation
                        STATE['failures'].append({'case': record['case'], 'type': 'forbidden_action_attempt', 'reason': violation, 'at': stamp()})
                    save()
                if violation:
                    return self.reply(400, {'error': 'Evaluation stopped before unauthorized action'})
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except Exception as error:
            code = error.code if isinstance(error, urllib.error.HTTPError) else 502
            if record is not None:
                with LOCK:
                    record.update(http_status=code, transport_error=type(error).__name__, duration_ms=round((time.monotonic()-start)*1000))
                    save()
            self.reply(code, {'error': 'Real Qwen request failed', 'type': type(error).__name__, 'upstream_status': code})


server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
print(json.dumps({'port': server.server_address[1], 'token': TOKEN, 'model': 'qwen3.5-plus', 'provider_origin_sha256': digest(BASE), 'hint_sha256': digest(HINT)}), flush=True)
server.serve_forever()
