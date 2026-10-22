"""Authenticated presentation hints and bounded native-source retrieval, no authority."""
import json
import re
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

from agents.webapi.job_service import _kanban_connection, _kanban_task_dict, _list_workspace_cron_jobs
from agents.coordination_association import CoordinationAssociationStore

_ASSETS = Path(__file__).resolve().parents[2] / "prompts" / "chat"


def resolve_presentation_context(ctx, message, raw):
    catalog = json.loads((_ASSETS / "presentation-navigation.json").read_text())
    raw = raw if isinstance(raw, dict) else {}
    path = raw.get("path")
    paths = {urlsplit(row[0]).path for row in catalog}
    path = path if path in paths else None
    query = parse_qs(str(raw.get("search") or "")[:4000])
    selected = (query.get("work") or query.get("execution") or [None])[0]
    text = message.casefold().strip()
    explicit = next((kind for kind, words in (
        ("task", ("task", "tasks", "任务")), ("job", ("job", "jobs", "定时工作")),
        ("execution", ("mission", "missions", "execution", "executions", "执行")),
        ("agent", ("agent", "agents", "profile", "profiles", "智能体")), ("knowledge", ("knowledge", "ontology", "知识", "本体")),
    ) if any(re.search(r"(?<![a-z])" + re.escape(word) + r"(?![a-z])", text) for word in words)), None)
    page_kind = next((row[2] for row in catalog if urlsplit(row[0]).path == path), None)
    opening = re.fullmatch(r"(?:please\s+)?(?:open\s+|go to\s+|navigate to\s+|打开\s*|进入\s*|跳转到\s*)(.+?)[。.!]?", text)
    target = opening.group(1).strip() if opening else None
    navigation = next(({"kind": "navigate", "href": row[0]} for row in catalog if target in row[1]), None) if target else None
    candidates = []
    availability = {}

    def add(kind, identifier, title, state, href):
        key = kind + ":" + str(identifier)
        matched = str(identifier).casefold() in text or (bool(title) and str(title).casefold() in text)
        candidates.append({"source": kind, "id": str(identifier), "title": title,
            "state": state, "href": href, "selected": selected == key or selected == identifier,
            "matched": matched})

    # Each native owner scopes data to the authenticated workspace. No host scan.
    for kind in ("task", "job", "execution"):
        try:
            if kind == "task":
                with _kanban_connection(ctx) as (db, conn):
                    for task in db.list_tasks(conn, include_archived=False):
                        item = _kanban_task_dict(task)
                        add(kind, item["id"], item.get("title"), item.get("status"),
                            "/dashboard?" + urlencode({"view": "kanban", "source": "task", "work": "task:" + str(item["id"])}))
            elif kind == "job":
                for job in _list_workspace_cron_jobs(ctx, include_disabled=True):
                    add(kind, job["id"], job.get("name"), job.get("enabled"),
                        "/dashboard?" + urlencode({"view": "kanban", "source": "job", "work": "job:" + str(job["id"])}))
            else:
                for execution in CoordinationAssociationStore(ctx.hermes_home / "state.db").list_contexts(ctx.workspace_id):
                    add(kind, execution.coordination_id, execution.goal_ref, None,
                        "/orchestrator?" + urlencode({"execution": execution.coordination_id}))
            availability[kind] = "AVAILABLE"
        except (OSError, RuntimeError, ValueError):
            availability[kind] = "UNAVAILABLE"
    candidates.sort(key=lambda item: (
        explicit is not None and item["source"] != explicit,
        explicit is None and not item["selected"],
        explicit is None and item["source"] != page_kind,
        not item["matched"],
        item["source"], item["id"],
    ))
    if navigation is None and target:
        matches = [item for item in candidates if target in (
            item["id"].casefold(), str(item["title"] or "").casefold(),
            item["source"] + " " + item["id"].casefold(),
        ) and (explicit is None or item["source"] == explicit)]
        if len(matches) == 1:
            navigation = {"kind": "navigate", "href": matches[0]["href"]}
    return {"schema_version": "presentation_context.v1", "page": path,
        "selected": next((item["source"] + ":" + item["id"] for item in candidates if item["selected"]), None),
        "explicit_scope": explicit, "candidates": candidates[:20], "availability": availability,
        "navigation": navigation, "catalog": [{"href": row[0], "labels": row[1]} for row in catalog]}


def presentation_system_context(ctx, message, raw):
    if raw is None:
        return None
    resolved = resolve_presentation_context(ctx, message, raw)
    return (_ASSETS / "presentation-context.md").read_text() + "\n" + json.dumps(resolved, ensure_ascii=False)
