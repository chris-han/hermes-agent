"""Real browser_exec isolation against the Electron E2E's synthetic localhost site.

Uses only the fixture home and locally installed tool binaries. Never connects to
Electron CDP or copies a user browser profile. Cleanup covers this child's driven sessions.
"""
import json
import os
import tempfile
import sys
import uuid
from pathlib import Path

from tools.browser_use_cli import browser_exec
from tools.browser_tool_lifecycle import cleanup_all_browsers

url = sys.argv[1]
assert url.startswith("http://127.0.0.1:"), "synthetic localhost fixture required"
prefix = "affinity-" + uuid.uuid4().hex[:12]
# AF_UNIX limits include the socket name: keep only this fixture IPC root short.
runtime = tempfile.TemporaryDirectory(prefix="aff-", dir="/tmp" if os.name == "posix" else None)
os.environ["BH_RUNTIME_DIR"] = runtime.name
os.environ["BH_RUNTIME_DIR_SHARED"] = "1"


def read(session, route):
    result = json.loads(browser_exec(
        f"new_tab({(url + route)!r})\nwait_for_load()\nprint(js('document.body.innerText'))",
        session=prefix + session, task_id=prefix + session, timeout_s=30))
    assert result.get("success"), result
    return result.get("output", "")


try:
    before = read("a", "/account")
    assert "Sign in required" in before, before
    signed_in = read("a", "/login")
    assert "Synthetic account signed in" in signed_in, signed_in
    marker = Path(runtime.name) / 'invalid-helper-side-effect'
    rejected = json.loads(browser_exec(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\nfrom browser_helpers import goto_url\ngoto_url({url!r})",
        session=prefix + 'a', task_id=prefix + 'a'))
    assert rejected.get('code') == 'UNSUPPORTED_HELPER_IMPORT', rejected
    assert rejected['recovery']['same_context'] and 'goto_url' in rejected['supported_helpers']
    assert not marker.exists(), 'Rejected payload executed side effects'
    corrected = read("a", "/account")
    assert "Synthetic account signed in" in corrected, corrected
    other = read("b", "/account")
    assert "Sign in required" in other, other
    retained = read("a", "/account")
    assert "Synthetic account signed in" in retained, retained
    print(json.dumps({"independent_browser_exec": "PASS", "named_session_isolation": "PASS", "preexecution_rejection": "PASS", "same_context_correction": "PASS"}))
finally:
    cleanup_all_browsers()
    runtime.cleanup()
