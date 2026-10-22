from __future__ import annotations

import json
import logging
import re

from pydantic import ValidationError
from agents.tender_journey_tool_api import authenticated_tool_api
from agents.workspace_uploads import WorkspaceUploadError

logger = logging.getLogger(__name__)


async def _invoke(operation, args):
    api = None
    try:
        api = authenticated_tool_api()
        result = await api.invoke(operation=operation, payload=args)
        return json.dumps({"ok": True, "result": result}, ensure_ascii=False)
    except ValidationError:
        code = "INVALID_REQUEST"
    except KeyError:
        code = "JOURNEY_OR_ARTIFACT_NOT_FOUND"
    except PermissionError:
        code = "FORBIDDEN"
    except WorkspaceUploadError as exc:
        code = "UPLOAD_UNAVAILABLE" if exc.code == "NOT_FOUND" else "UPLOAD_REJECTED"
    except ValueError as exc:
        candidate = str(exc).split(":", 1)[0]
        code = candidate if re.fullmatch(r"[A-Z][A-Z0-9_]{1,100}", candidate) else "REQUEST_REJECTED"
    except Exception as exc:
        logger.error("Tender journey tool failed: operation=%s exception_type=%s", operation, type(exc).__name__)
        code = "INTERNAL_ERROR"
    error = {"code": code}
    if code == "UPLOAD_UNAVAILABLE":
        error["recovery"] = {"action": "list_uploads", "arguments": {}}
    if code == "STALE_JOURNEY_REVISION" and api is not None:
        try:
            error["recovery"] = await api.invoke(operation="resolve", payload={"journey_id": args.get("journey_id")})
        except Exception:
            pass
    return json.dumps({"ok": False, "error": error}, ensure_ascii=False)


async def tender_journey_resolve(args, **_kwargs):
    return await _invoke("resolve", args)


async def tender_journey_action(args, **_kwargs):
    return await _invoke("action", args)
