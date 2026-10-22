"""Append-only workspace artifact store for immutable SchemaDecision records."""

from __future__ import annotations

import json
from pathlib import Path

from contracts.schema_decision import SchemaDecision


class SchemaDecisionStore:
    def __init__(self, workspace_root: str | Path, *, workspace_id: str) -> None:
        root = Path(workspace_root).resolve()
        workspace_path = (
            root
            if root.name == workspace_id and root.parent.name == "workspaces"
            else root / "workspaces" / workspace_id
        )
        self.root = workspace_path / "artifacts" / "schema-decisions"

    def put(self, decision: SchemaDecision) -> dict:
        digest = decision.schema_decision_hash.removeprefix("sha256:")
        target = self.root / digest / f"{decision.schema_version}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            decision.as_dict(), ensure_ascii=False, sort_keys=True, indent=2
        )
        if target.exists() and target.read_text(encoding="utf-8") != encoded:
            raise ValueError("SCHEMA_DECISION_ARTIFACT_CONFLICT")
        target.write_text(encoded, encoding="utf-8")
        return decision.as_dict()

    def get(
        self,
        schema_decision_id: str,
        *,
        taxonomy_service: object | None = None,
    ) -> dict:
        return self.get_contract(
            schema_decision_id, taxonomy_service=taxonomy_service
        ).as_dict()

    def get_contract(
        self,
        schema_decision_id: str,
        *,
        taxonomy_service: object | None = None,
    ) -> SchemaDecision:
        if not schema_decision_id.startswith("schema-decision-"):
            raise ValueError("invalid schema_decision_id")
        matches = []
        if self.root.is_dir():
            for candidate in self.root.glob("*/schema_decision.v*.json"):
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                if payload.get("schema_decision_id") == schema_decision_id:
                    matches.append(payload)
        if len(matches) != 1:
            raise ValueError(f"schema decision not found: {schema_decision_id}")
        payload = matches[0]
        if payload.get("schema_version") == "schema_decision.v3":
            from services.knowledge_relation_taxonomy_service import (
                KnowledgeRelationTaxonomyService,
            )

            if type(taxonomy_service) is not KnowledgeRelationTaxonomyService:
                raise ValueError(
                    "decision v3 replay requires the exact governed taxonomy service"
                )
            return KnowledgeRelationTaxonomyService.replay_schema_decision_v3(
                taxonomy_service, payload
            )
        return SchemaDecision.from_dict(payload)


__all__ = ["SchemaDecisionStore"]
