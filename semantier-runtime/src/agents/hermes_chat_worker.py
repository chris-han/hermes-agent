"""Private one-turn native API worker. Never accepts public identity directly."""
import asyncio
import json
import sys
import os


async def main():
    # Keep startup IPC separate from library prints and diagnostic logs.
    ready = sys.stdout
    sys.stdout = sys.stderr
    # Metadata only: never log prompts, credentials or response text.
    import httpx, time
    original_send = httpx.Client.send
    def measured_send(client, request, **kwargs):
        started = time.monotonic()
        if request.url.path.endswith('/chat/completions'):
            try:
                body = json.loads(request.content)
                fields = {'messages':len(body.get('messages',[])), 'recent_tool_calls':[t.get('function',{}).get('name') for m in body.get('messages',[])[-6:] for t in m.get('tool_calls',[])]}
                fields.update({k: body[k] for k in ('model','stream','reasoning_effort','enable_thinking','max_tokens','thinking_budget') if k in body})
                print('chat_provider request '+json.dumps({'host':request.url.host,'bytes':len(request.content),**fields}), file=sys.stderr, flush=True)
            except (ValueError, httpx.RequestNotRead):
                pass
        response = original_send(client,request,**kwargs)
        if request.url.path.endswith('/chat/completions'):
            print('chat_provider headers '+json.dumps({'status':response.status_code,'seconds':round(time.monotonic()-started,3)}),file=sys.stderr,flush=True)
        return response
    httpx.Client.send = measured_send
    from agents.hermes_embedded_api import EmbeddedHermesAPIService
    service = EmbeddedHermesAPIService()
    try:
        await service.ensure_started(reuse_provisioned_runtime=os.environ.get('SEMANTIER_CHAT_WORKER_PROVISIONED')=='1')
        ready.write('SEMANTIER_CHAT_READY='+json.dumps({'base_url':service.base_url,'api_key':service.api_key})+'\n')
        ready.flush()
        await asyncio.Event().wait()
    finally:
        await service.shutdown()


if __name__ == '__main__':
    asyncio.run(main())
