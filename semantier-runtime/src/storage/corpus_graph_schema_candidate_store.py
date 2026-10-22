"""Content-addressed workspace artifact store for schema candidates."""

from __future__ import annotations

import json
from pathlib import Path

from contracts.corpus_graph_schema import CorpusGraphSchemaCandidate


class CorpusGraphSchemaCandidateStore:
    def __init__(self, workspace_root: str | Path, *, workspace_id: str) -> None:
        root = Path(workspace_root).resolve()
        workspace_path = (
            root
            if root.name == workspace_id and root.parent.name == "workspaces"
            else root / "workspaces" / workspace_id
        )
        self.root = workspace_path / "artifacts" / "schema-candidates"

    def put(self, candidate: CorpusGraphSchemaCandidate) -> dict:
        digest = candidate.candidate_hash.removeprefix("sha256:")
        target = self.root / digest / f"{candidate.schema_version}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            candidate.as_dict(), ensure_ascii=False, sort_keys=True, indent=2
        )
        if target.exists() and target.read_text(encoding="utf-8") != encoded:
            raise ValueError("CORPUS_SCHEMA_CANDIDATE_ARTIFACT_CONFLICT")
        target.write_text(encoded, encoding="utf-8")
        return candidate.as_dict()

    def get_by_hash(self, candidate_hash: str) -> dict:
        if not candidate_hash.startswith("sha256:"):
            raise ValueError("candidate_hash must be a sha256 ref")
        root = self.root / candidate_hash.removeprefix("sha256:")
        matches = sorted(root.glob("corpus_graph_schema_candidate.*.json"))
        if len(matches) != 1:
            raise ValueError(f"schema candidate not found: {candidate_hash}")
        payload = json.loads(matches[0].read_text(encoding="utf-8"))
        return CorpusGraphSchemaCandidate.from_dict(payload).as_dict()

    def get_by_id(self, candidate_id: str) -> dict:
        if not isinstance(candidate_id, str) or not candidate_id.startswith("cgsc_"):
            raise ValueError("invalid schema candidate id")
        matches: list[dict] = []
        if self.root.is_dir():
            for path in self.root.glob("*/corpus_graph_schema_candidate.*.json"):
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("candidate_id") == candidate_id:
                    matches.append(payload)
        if len(matches) != 1:
            raise ValueError(f"schema candidate not found: {candidate_id}")
        return CorpusGraphSchemaCandidate.from_dict(matches[0]).as_dict()

    def add_successor_link(
        self, *, predecessor_candidate_ref: str, successor: CorpusGraphSchemaCandidate
    ) -> None:
        link_root = self.root / "successor-links" / predecessor_candidate_ref
        target = link_root / f"{successor.candidate_id}.json"
        payload = {
            "schema_version": "corpus_graph_schema_successor_link.v1",
            "predecessor_candidate_ref": predecessor_candidate_ref,
            "successor_candidate_ref": successor.candidate_id,
            "successor_candidate_hash": successor.candidate_hash,
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
        if target.exists() and target.read_text(encoding="utf-8") != encoded:
            raise ValueError("CORPUS_SCHEMA_SUCCESSOR_LINK_CONFLICT")
        target.write_text(encoded, encoding="utf-8")

    def successors(self, predecessor_candidate_ref: str) -> list[str]:
        root = self.root / "successor-links" / predecessor_candidate_ref
        if not root.is_dir():
            return []
        return sorted(
            str(json.loads(path.read_text(encoding="utf-8"))["successor_candidate_ref"])
            for path in root.glob("*.json")
        )


__all__ = ["CorpusGraphSchemaCandidateStore"]
