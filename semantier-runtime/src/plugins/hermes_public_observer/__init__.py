"""Public native observer only; never policy enforcement or core access."""
import atexit
import json
import os
from pathlib import Path
from queue import Queue, Full
import threading
from datetime import datetime, timezone

def register(ctx):
    home=Path(os.environ["HERMES_HOME"])
    run_id=os.environ["SEMANTIER_PUBLIC_RUN_ID"]
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}",run_id):
        raise ValueError("INVALID_RUN_ID")
    skill_id=os.environ["SEMANTIER_PUBLIC_SKILL_ID"]
    ctx.register_skill(skill_id,home/"skills"/skill_id/"SKILL.md","Versioned experimental procedure")
    queue=Queue(maxsize=128)
    state={"dropped":0,"errors":0,"sequence":0,"redacted":0}
    lock=threading.Lock()
    root=home/".semantier-public"
    root.mkdir(exist_ok=True)
    path=root/(run_id+".jsonl")
    def writer():
        try:
            with path.open("xb") as output:
                while True:
                    row=queue.get()
                    if row is None:break
                    if isinstance(row, threading.Event):
                        output.flush()
                        row.set()
                        continue
                    output.write(json.dumps(row,ensure_ascii=False).encode()+b"\n")
                    output.flush()
        except (OSError,ValueError):
            state["errors"]+=1
    worker=threading.Thread(target=writer,daemon=True)
    worker.start()
    def clean(value):
        if isinstance(value,dict):
            result={}
            for key,item in value.items():
                if str(key).lower() in {"headers","authorization","api_key","password","access_token","secret"}:
                    state["redacted"]+=1
                else:result[key]=clean(item)
            return result
        if isinstance(value,(tuple,list)):return [clean(item) for item in value]
        return value
    def observe(hook,**payload):
        try:
            safe={k:v for k,v in payload.items() if k in (
                "session_id","tool_name","tool_call_id","arguments","args",
                "result","status","duration_ms","error_type")}
            with lock:
                state["sequence"]+=1
                if state["sequence"]>512:state["dropped"]+=1;return None
                before=state["redacted"]
                safe=clean(safe)
                row={"run_id":run_id,"sequence":state["sequence"],"hook":hook,
                     "captured_at":datetime.now(timezone.utc).isoformat(),
                     "payload":json.loads(json.dumps(safe)),
                     "gaps":["material_fields_redacted"] if state["redacted"]>before else [],
                     "argument_basis":"as_exposed_by_public_hook; effective_arguments_not_established"}
                raw=json.dumps(row).encode()
                if len(raw)>16384:state["dropped"]+=1;return None
                queue.put_nowait(row)
        except (Full,TypeError,ValueError):
            state["dropped"]+=1
        return None
    ctx.register_hook("post_tool_call",lambda **values:observe("post_tool_call",**values))
    observe("registration")
    def turn_end(**values):
        observe("on_session_end", **values)
        barrier=threading.Event()
        try:
            queue.put_nowait(barrier)
            flushed=barrier.wait(.5)
        except Full:
            flushed=False
        try:
            (root/(run_id+".coverage.json")).write_text(json.dumps({
                **state, "closed": False, "collector_checkpoint": "turn_end",
                "unflushed": 0 if flushed else "unknown", "history_complete": False}))
        except OSError:
            pass
        return None
    ctx.register_hook("on_session_end",turn_end)
    closed=threading.Event()
    def close():
        if closed.is_set():return
        closed.set()
        try:queue.put_nowait(None)
        except Full:state["dropped"]+=1
        worker.join(timeout=.5)
        try:(root/(run_id+".coverage.json")).write_text(json.dumps({
            **state,"closed":not worker.is_alive(),"history_complete":False}))
        except OSError:pass
    ctx.on_unload(close)
    atexit.register(close)
