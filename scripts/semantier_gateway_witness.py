"""Controlled native Hermes witness for the connected offline gateway acceptance runner.

Run in the acceptance network namespace, with a real issued token in the subprocess env.
This script owns a NEW disposable Hermes profile and never reads a user's profile.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--session-id', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    endpoint = urlsplit(args.endpoint)
    if endpoint.scheme != 'http' or endpoint.hostname != '127.0.0.1' or endpoint.path.rstrip('/') != '/v1':
        parser.error('This offline witness requires an explicit loopback /v1 facade.')
    token = os.environ.get('SEMANTIER_MODEL_GATEWAY_TOKEN')
    if not token:
        parser.error('An actually issued gateway token is required.')
    home = args.home.resolve()
    workspace = args.workspace.resolve()
    evidence = args.evidence.resolve()
    if not evidence.is_relative_to(workspace) or not evidence.is_file():
        parser.error('Evidence must be an existing file inside the witness workspace.')
    home.mkdir(parents=True, exist_ok=True)
    if any(home.iterdir()):
        parser.error('The witness requires a new empty profile directory.')
    import yaml
    (home / 'config.yaml').write_text(yaml.safe_dump({
        'model': {'provider': 'custom', 'default': args.model, 'base_url': args.endpoint,
                  'api_key': '${SEMANTIER_MODEL_GATEWAY_TOKEN}', 'context_length': 128000},
        'billing': {'backend': 'semantier'},
        'semantier_gateway': {'enabled': True, 'base_url': args.endpoint,
                              'api_key_env': 'SEMANTIER_MODEL_GATEWAY_TOKEN'},
        'auxiliary': {'transient_retries': 0, 'title_generation': {
            'provider': 'custom', 'model': args.model, 'base_url': args.endpoint,
            'api_key': '${SEMANTIER_MODEL_GATEWAY_TOKEN}'}},
        'compression': {'enabled': False},
    }), encoding='utf-8')
    (home / '.env').write_text('SEMANTIER_MODEL_GATEWAY_TOKEN=' + token + '\n', encoding='utf-8')
    (home / '.env').chmod(0o600)
    # This is one isolated witness subprocess, not a shared multi-profile runtime.
    os.environ['HERMES_HOME'] = str(home)
    os.chdir(workspace)
    from run_agent import AIAgent
    from agent.title_generator import generate_title
    from agent.semantier_gateway import read_billing, SCHEMA_VERSION
    capture = io.StringIO()
    agent = None
    with redirect_stdout(capture):
        try:
            agent = AIAgent(model=args.model, provider='custom', base_url=args.endpoint,
                            api_key=token, session_id=args.session_id, enabled_toolsets=['file'],
                            max_iterations=4, max_tokens=128, skip_context_files=True,
                            skip_memory=True, skip_background_review=True, quiet_mode=True,
                            cwd=str(workspace))
            agent._disable_streaming = True
            result = agent.run_conversation(f'Read {evidence.name}.', task_id=args.run_id)
            tool_rows = [row for row in result.get('messages', []) if row.get('role') == 'tool']
            read_ok = any(evidence.read_text(encoding='utf-8-sig') in str(row.get('content')) for row in tool_rows)
            title = generate_title(f'Read {evidence.name}.', timeout=15, main_runtime={
                'provider': 'custom', 'model': args.model, 'base_url': args.endpoint,
                'api_key': token, 'api_mode': 'chat_completions',
                'session_id': args.session_id, 'run_id': args.run_id})
            billing = read_billing('state', schema_version=SCHEMA_VERSION)
            subscription = read_billing('subscription', schema_version=SCHEMA_VERSION)
        finally:
            if agent is not None:
                agent.close()
    safe = {'session_id': args.session_id, 'run_id': args.run_id, 'tool_read_ok': read_ok,
            'tool_rows': len(tool_rows), 'title': title, 'turn_error': result.get('error'),
            'billing': billing, 'subscription': subscription}
    print(json.dumps(safe, ensure_ascii=False))
    if result.get('error') or not read_ok or not title or not billing.get('ok') or not subscription.get('ok'):
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
