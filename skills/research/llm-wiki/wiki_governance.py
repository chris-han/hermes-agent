"""Governed sidecar assurance for the llm-wiki skill.

The wiki remains curation/context material. This utility creates deterministic
metadata for impact analysis and linting; it does not promote markdown into
Semantier runtime authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import uuid
import fcntl
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, UTC
from pathlib import Path
import re
from typing import Any

AUTHORITY_ORDER = {
    "canonical": 4,
    "derived": 3,
    "operational": 2,
    "reference": 1,
}
ALIGNMENT_STATES = {"CURRENT", "REVIEW_REQUIRED", "ALIGNED", "EXEMPT"}
FINDING_CLASSES = {"AUTO_FIX", "PROPOSE_FIX", "BLOCKING_CONFLICT"}
WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


@dataclass
class WikiPage:
    path: str
    full_path: Path
    frontmatter: dict[str, Any]
    body: str
    wikilinks: list[str] = field(default_factory=list)
    sha256: str = ""


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def wiki_root_from_env_or_arg(raw: str | None) -> Path:
    if raw:
        return Path(raw).expanduser().resolve()
    import os

    env = os.environ.get("WIKI_PATH")
    if env:
        return Path(env).expanduser().resolve()
    raise RuntimeError(
        "llm-wiki requires an explicit --wiki path or governed WIKI_PATH environment"
    )


def governance_root(wiki_root: Path) -> Path:
    return wiki_root / ".governance"


def parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if value in {"[]", ""}:
        return []
    if value == "{}":
        return {}
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [item.strip().strip("'\"") for item in inner.split(",") if item.strip()]
    return value.strip("'\"")


def parse_simple_yaml(raw: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    lines = raw.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if not line.strip() or line.lstrip().startswith("#") or line.startswith(" "):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if value.strip():
            data[key] = parse_scalar(value)
            continue
        items: list[str] = []
        mapping: dict[str, Any] = {}
        while index < len(lines) and lines[index].startswith("  "):
            child = lines[index].strip()
            index += 1
            if child.startswith("- "):
                items.append(child[2:].strip().strip("'\""))
            elif ":" in child:
                child_key, child_value = child.split(":", 1)
                mapping[child_key.strip()] = parse_scalar(child_value)
        data[key] = items if items else mapping
    return data


def render_simple_yaml(data: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, list):
            if value:
                lines.append(f"{key}:")
                lines.extend(f"  - {item}" for item in value)
            else:
                lines.append(f"{key}: []")
        elif isinstance(value, dict):
            if value:
                lines.append(f"{key}:")
                for child_key, child_value in value.items():
                    lines.append(f"  {child_key}: {child_value}")
            else:
                lines.append(f"{key}: {{}}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end < 0:
        return {}, text
    raw_fm = text[4:end]
    body = text[text.find("\n", end + 4) + 1 :]
    return parse_simple_yaml(raw_fm), body


def iter_wiki_pages(wiki_root: Path) -> list[WikiPage]:
    pages: list[WikiPage] = []
    if not wiki_root.exists():
        return pages
    for full_path in sorted(wiki_root.rglob("*.md")):
        relative = full_path.relative_to(wiki_root).as_posix()
        if relative.startswith((".governance/", "raw/")) or relative in {"log.md", "SCHEMA.md"}:
            continue
        raw = full_path.read_text(encoding="utf-8")
        frontmatter, body = split_frontmatter(raw)
        links = []
        for match in WIKILINK_RE.finditer(body):
            target = match.group(1).split("|")[0].split("#")[0].strip()
            if target:
                links.append(target)
        pages.append(
            WikiPage(
                path=relative,
                full_path=full_path,
                frontmatter=frontmatter,
                body=body,
                wikilinks=sorted(set(links)),
                sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
            )
        )
    return pages


def normalize_page_ref(ref: str) -> str:
    normalized = ref.strip().replace("\\", "/")
    if not normalized:
        return ""
    if not normalized.endswith(".md"):
        normalized = f"{normalized}.md"
    return normalized


def build_dependency_graph(wiki_root: Path, *, operation_id: str | None = None) -> dict[str, Any]:
    pages = iter_wiki_pages(wiki_root)
    page_ids = {page.path for page in pages}
    nodes = []
    edges = []
    for page in pages:
        authority = str(page.frontmatter.get("authority", "reference"))
        alignment_state = str(page.frontmatter.get("alignment_state", "CURRENT"))
        nodes.append(
            {
                "id": page.path,
                "authority": authority,
                "alignment_state": alignment_state,
                "sha256": page.sha256,
            }
        )
        for dep in page.frontmatter.get("depends_on", []) or []:
            target = normalize_page_ref(str(dep))
            edges.append({"source": page.path, "target": target, "type": "depends_on"})
        for governed in page.frontmatter.get("governs", []) or []:
            target = normalize_page_ref(str(governed))
            edges.append({"source": page.path, "target": target, "type": "governs"})
        for link in page.wikilinks:
            target = normalize_page_ref(link)
            if target in page_ids:
                edges.append({"source": page.path, "target": target, "type": "wikilink"})
    graph = {
        "schema_version": "llm-wiki-governance.v1",
        "generated_at": utc_now_iso(),
        "nodes": nodes,
        "edges": edges,
    }
    root = governance_root(wiki_root)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "dependency-graph.json", graph)
    append_audit(wiki_root, "graph_rebuild", "dependency-graph.json", operation_id=operation_id)
    return graph


def graph_adjacency(graph: dict[str, Any]) -> dict[str, list[str]]:
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in graph.get("edges", []):
        if edge.get("type") == "depends_on":
            adjacency[str(edge["source"])].append(str(edge["target"]))
    return adjacency


def detect_cycles(graph: dict[str, Any]) -> list[list[str]]:
    adjacency = graph_adjacency(graph)
    cycles: list[list[str]] = []
    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []

    def dfs(node: str) -> None:
        if node in visiting:
            start = stack.index(node)
            cycles.append(stack[start:] + [node])
            return
        if node in visited:
            return
        visiting.add(node)
        stack.append(node)
        for target in adjacency.get(node, []):
            dfs(target)
        stack.pop()
        visiting.remove(node)
        visited.add(node)

    for node in adjacency:
        dfs(node)
    return cycles


def load_contracts(wiki_root: Path) -> dict[str, dict[str, Any]]:
    contracts_dir = governance_root(wiki_root) / "contracts"
    contracts: dict[str, dict[str, Any]] = {}
    if not contracts_dir.exists():
        return contracts
    for path_item in sorted(contracts_dir.glob("*.yaml")):
        parsed = parse_simple_yaml(path_item.read_text(encoding="utf-8"))
        contract_id = str(parsed.get("contract_id") or path_item.stem)
        contracts[contract_id] = parsed
    return contracts


def audit_wiki(wiki_root: Path, *, change: dict[str, Any] | None = None) -> dict[str, Any]:
    graph = build_dependency_graph(wiki_root, operation_id=(change or {}).get("operation_id"))
    pages = {page.path: page for page in iter_wiki_pages(wiki_root)}
    node_authority = {
        node["id"]: str(node.get("authority", "reference")) for node in graph["nodes"]
    }
    findings: list[dict[str, Any]] = []

    for page in pages.values():
        authority = str(page.frontmatter.get("authority", "")).strip()
        if authority not in AUTHORITY_ORDER:
            findings.append(
                {
                    "class": "PROPOSE_FIX",
                    "code": "missing_authority",
                    "page": page.path,
                    "message": "Page missing valid governed authority frontmatter",
                }
            )
        state = str(page.frontmatter.get("alignment_state", "CURRENT"))
        if state not in ALIGNMENT_STATES:
            findings.append(
                {
                    "class": "AUTO_FIX",
                    "code": "invalid_alignment_state",
                    "page": page.path,
                    "message": "Alignment state is not recognized",
                }
            )

    for edge in graph["edges"]:
        if edge["type"] != "depends_on":
            continue
        source = edge["source"]
        target = edge["target"]
        if source not in pages or target not in pages:
            findings.append(
                {
                    "class": "PROPOSE_FIX",
                    "code": "missing_dependency",
                    "page": source,
                    "target": target,
                    "message": "Explicit dependency target is missing",
                }
            )
            continue
        if AUTHORITY_ORDER.get(node_authority[source], 0) > AUTHORITY_ORDER.get(
            node_authority[target], 0
        ):
            findings.append(
                {
                    "class": "BLOCKING_CONFLICT",
                    "code": "canonical_derived_direction_reversal",
                    "page": source,
                    "target": target,
                    "message": "Higher-authority page depends on a lower-authority page",
                }
            )

    for cycle in detect_cycles(graph):
        findings.append(
            {
                "class": "BLOCKING_CONFLICT",
                "code": "dependency_cycle",
                "cycle": cycle,
                "message": "Explicit governed dependency cycle detected",
            }
        )

    contracts = load_contracts(wiki_root)
    for page in pages.values():
        for contract_id in page.frontmatter.get("semantic_contracts", []) or []:
            contract = contracts.get(str(contract_id))
            if contract is None:
                findings.append(
                    {
                        "class": "BLOCKING_CONFLICT",
                        "code": "missing_semantic_contract",
                        "page": page.path,
                        "contract_id": contract_id,
                        "message": "Referenced semantic contract is missing",
                    }
                )
                continue
            for claim in contract.get("required_claims", []) or []:
                claim_text = str(claim)
                if claim_text and claim_text not in page.body:
                    findings.append(
                        {
                            "class": "PROPOSE_FIX",
                            "code": "required_claim_missing",
                            "page": page.path,
                            "contract_id": contract_id,
                            "message": "Required semantic-contract claim is missing",
                        }
                    )
            for claim in contract.get("prohibited_claims", []) or []:
                claim_text = str(claim)
                if claim_text and claim_text in page.body:
                    findings.append(
                        {
                            "class": "BLOCKING_CONFLICT",
                            "code": "prohibited_claim_present",
                            "page": page.path,
                            "contract_id": contract_id,
                            "message": "Prohibited semantic-contract claim is present",
                        }
                    )

    report = {
        "schema_version": "llm-wiki-assurance-report.v1",
        "generated_at": utc_now_iso(),
        "finding_count": len(findings),
        "findings": findings,
        "finding_classes": sorted(FINDING_CLASSES),
    }
    return publish_report(wiki_root, "assurance", report, change=change)


def downstream_dependents(graph: dict[str, Any], canonical_page: str) -> list[str]:
    reverse: dict[str, list[str]] = defaultdict(list)
    for edge in graph.get("edges", []):
        if edge.get("type") == "depends_on":
            reverse[str(edge["target"])].append(str(edge["source"]))
    result: list[str] = []
    queue = deque([canonical_page])
    seen = {canonical_page}
    while queue:
        current = queue.popleft()
        for dependent in reverse.get(current, []):
            if dependent in seen:
                continue
            seen.add(dependent)
            result.append(dependent)
            queue.append(dependent)
    return result


def mark_review_required(wiki_root: Path, relative_page: str, *, operation_id: str | None = None) -> None:
    full_path = wiki_root / relative_page
    raw = full_path.read_text(encoding="utf-8")
    frontmatter, body = split_frontmatter(raw)
    changed = frontmatter.get("alignment_state") != "REVIEW_REQUIRED"
    if changed:
        frontmatter["alignment_state"] = "REVIEW_REQUIRED"
        atomic_bytes(full_path, f"---\n{render_simple_yaml(frontmatter)}\n---\n{body}".encode())
    if changed or operation_id is not None:
        append_audit(wiki_root, "alignment_state_transition", relative_page, operation_id=operation_id)


def impact_canonical_change(wiki_root: Path, canonical_page: str, *, change: dict[str, Any] | None = None) -> dict[str, Any]:
    page = normalize_page_ref(canonical_page)
    graph = build_dependency_graph(wiki_root, operation_id=(change or {}).get("operation_id"))
    dependents = downstream_dependents(graph, page)
    for dependent in dependents:
        mark_review_required(wiki_root, dependent, operation_id=(change or {}).get("operation_id"))
    report = {
        "schema_version": "llm-wiki-impact-report.v1",
        "generated_at": utc_now_iso(),
        "canonical_page": page,
        "downstream_dependents": dependents,
        "event_driven": True,
    }
    return publish_report(wiki_root, "impact", report, change=change)


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".wiki-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True).encode("utf-8"))


def publish_report(wiki_root: Path, kind: str, report: dict[str, Any], *, change: dict[str, Any] | None = None) -> dict[str, Any]:
    """Immutable report identity; CLI calls without change identity are new attempts."""
    root = governance_root(wiki_root) / "reports"
    root.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(json.dumps(change, sort_keys=True).encode()).hexdigest() if change else uuid.uuid4().hex
    path = root / f"{kind}-{identity}.json"
    report = {**report, "change": change, "report_ref": path.relative_to(wiki_root).as_posix()}
    fd, temporary = tempfile.mkstemp(prefix=".report-", dir=root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True); handle.flush(); os.fsync(handle.fileno())
        try:
            os.link(temporary, path)  # atomic, exclusive; never overwrite an earlier report
            _sync_directory(root)
        except FileExistsError:
            existing = json.loads(path.read_text())
            if change is None or existing.get("change") != change:
                raise ValueError("report identity conflict")
            report = existing
    finally:
        Path(temporary).unlink(missing_ok=True)
    append_audit(wiki_root, "assurance_audit" if kind == "assurance" else "impact_report", report["report_ref"], operation_id=(change or {}).get("operation_id"))
    return report


def append_audit(wiki_root: Path, action: str, subject: str, *, operation_id: str | None = None) -> None:
    governance_root(wiki_root).mkdir(parents=True, exist_ok=True)
    log_path = governance_root(wiki_root) / "audit-log.jsonl"
    entry = {"ts": utc_now_iso(), "action": action, "subject": subject, "operation_id": operation_id}
    with log_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        if operation_id is not None:
            handle.seek(0)
            for line in handle:
                previous = json.loads(line)
                if all(previous.get(key) == entry[key] for key in ("action", "subject", "operation_id")):
                    return
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Governed llm-wiki assurance")
    parser.add_argument("command", choices=["rebuild-graph", "audit", "canonical-change"])
    parser.add_argument("--wiki", default=None)
    parser.add_argument("--page", default=None)
    args = parser.parse_args(argv)
    wiki_root = wiki_root_from_env_or_arg(args.wiki)
    if args.command == "rebuild-graph":
        build_dependency_graph(wiki_root)
    elif args.command == "audit":
        audit_wiki(wiki_root)
    elif args.command == "canonical-change":
        if not args.page:
            raise SystemExit("--page is required for canonical-change")
        impact_canonical_change(wiki_root, args.page)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
