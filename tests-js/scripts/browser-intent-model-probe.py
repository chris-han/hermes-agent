"""Opt-in first-action probe against a real configured model; never executes tools.

Run under the credential-owning environment. Prompt is the candidate's effective
Desktop hint exported by its real prompt builder. This measures first-action
selection only, not browser execution or full autonomous E2E success.
"""
import argparse
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--provider', default='alibaba-cn')
    parser.add_argument('--model', default='qwen3.5-plus')
    args = parser.parse_args()
    from agent.auxiliary_client import resolve_provider_client
    from tools.preview_tool import PREVIEW_SCHEMA
    from tools.drive_preview_tool import ACT_PREVIEW_SCHEMA
    from tools.browser_use_cli import BROWSER_EXEC_SCHEMA
    from tools.clarify_tool import CLARIFY_SCHEMA

    client, model = resolve_provider_client(args.provider, args.model)
    if client is None:
        raise SystemExit('Configured model route is unavailable; no credential fallback attempted.')
    client = client.with_options(timeout=60, max_retries=0)
    prompt = args.prompt.read_text(encoding='utf-8')
    tools = [{'type': 'function', 'function': schema} for schema in (
        PREVIEW_SCHEMA, ACT_PREVIEW_SCHEMA, BROWSER_EXEC_SCHEMA, CLARIFY_SCHEMA)]
    cases = [
        ('visible', 'Open Amazon and show me systems engineering books.', {'desktop_preview'}),
        ('data_only', 'Just get the price of the INCOSE Systems Engineering Handbook from Amazon and report it. I do not need to see the page.', {'browser_exec'}),
        ('existing_page', 'What is the price on this page I have signed in to?', {'desktop_preview'}),
        ('ambiguous', 'Browse Amazon and get price information for a book about systems engineering.', {'clarify', 'desktop_preview'}),
    ]
    rows = []
    for case_id, user, allowed in cases:
        reply = client.chat.completions.create(model=model, messages=[
            {'role': 'system', 'content': 'You are Semantier Agent.\n' + prompt},
            {'role': 'user', 'content': user}], tools=tools, max_tokens=600,
            extra_body={'enable_thinking': False})
        message = reply.choices[0].message
        calls = [{'name': call.function.name, 'arguments': json.loads(call.function.arguments)}
                 for call in (message.tool_calls or [])]
        passed = bool(calls) and all(call['name'] in allowed for call in calls)
        if case_id == 'existing_page':
            passed = passed and calls[0]['arguments'].get('action') == 'read'
        if case_id == 'visible':
            passed = passed and calls[0]['arguments'].get('action') == 'open'
        rows.append({'case': case_id, 'user': user, 'calls': calls,
                     'content': message.content, 'pass': passed,
                     'usage': reply.usage.model_dump() if reply.usage else None})
    result = {'produced_at': datetime.now(UTC).isoformat(),
              'prompt_sha256': sha256(prompt.encode()).hexdigest(),
              'provider': args.provider, 'model': model, 'real_model': True,
              'tools_executed': False, 'full_autonomous_e2e': 'NOT_RUN', 'cases': rows}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'model': model, 'passed': sum(row['pass'] for row in rows), 'total': len(rows)}))
    return 0 if all(row['pass'] for row in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
