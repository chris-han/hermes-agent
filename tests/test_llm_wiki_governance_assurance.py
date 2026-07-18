from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WIKI_GOVERNANCE_PATH = (
    REPO_ROOT / "skills" / "research" / "llm-wiki" / "wiki_governance.py"
)


def load_wiki_governance():
    spec = importlib.util.spec_from_file_location("llm_wiki_governance", WIKI_GOVERNANCE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_page(
    wiki_root: Path,
    relative_path: str,
    *,
    authority: str,
    depends_on: list[str] | None = None,
    alignment_state: str = "CURRENT",
    body: str = "",
) -> None:
    path = wiki_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    deps = depends_on or []
    lines = [
        "---",
        f"authority: {authority}",
    ]
    if deps:
        lines.append("depends_on:")
        lines.extend(f"  - {item}" for item in deps)
    else:
        lines.append("depends_on: []")
    lines.extend(
        [
            "governs: []",
            "semantic_contracts: []",
            "upstream_bindings: {}",
            f"alignment_state: {alignment_state}",
            "---",
            body,
        ]
    )
    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def test_build_dependency_graph_persists_explicit_edges_and_audit(tmp_path: Path) -> None:
    governance = load_wiki_governance()
    wiki_root = tmp_path / "wiki"
    write_page(
        wiki_root,
        "concepts/derived.md",
        authority="derived",
        depends_on=["concepts/canonical.md"],
        body="See [[concepts/canonical]].",
    )
    write_page(wiki_root, "concepts/canonical.md", authority="canonical", body="Doctrine")

    graph = governance.build_dependency_graph(wiki_root)

    graph_path = wiki_root / ".governance" / "dependency-graph.json"
    assert graph_path.exists()
    persisted = json.loads(graph_path.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == "llm-wiki-governance.v1"
    assert {
        "source": "concepts/derived.md",
        "target": "concepts/canonical.md",
        "type": "depends_on",
    } in graph["edges"]
    assert (wiki_root / ".governance" / "audit-log.jsonl").exists()


def test_canonical_change_marks_downstream_dependents_review_required(tmp_path: Path) -> None:
    governance = load_wiki_governance()
    wiki_root = tmp_path / "wiki"
    write_page(wiki_root, "concepts/canonical.md", authority="canonical", body="Doctrine")
    write_page(
        wiki_root,
        "concepts/derived.md",
        authority="derived",
        depends_on=["concepts/canonical.md"],
        body="Derived",
    )

    report = governance.impact_canonical_change(wiki_root, "concepts/canonical.md")

    assert report["event_driven"] is True
    assert report["downstream_dependents"] == ["concepts/derived.md"]
    raw = (wiki_root / "concepts" / "derived.md").read_text(encoding="utf-8")
    assert "alignment_state: REVIEW_REQUIRED" in raw


def test_audit_classifies_reversal_and_cycle_as_blocking(tmp_path: Path) -> None:
    governance = load_wiki_governance()
    wiki_root = tmp_path / "wiki"
    write_page(
        wiki_root,
        "concepts/canonical.md",
        authority="canonical",
        depends_on=["concepts/derived.md"],
        body="Doctrine",
    )
    write_page(
        wiki_root,
        "concepts/derived.md",
        authority="derived",
        depends_on=["concepts/canonical.md"],
        body="Derived",
    )

    report = governance.audit_wiki(wiki_root)

    codes = {finding["code"] for finding in report["findings"]}
    assert "canonical_derived_direction_reversal" in codes
    assert "dependency_cycle" in codes
    assert any(finding["class"] == "BLOCKING_CONFLICT" for finding in report["findings"])
