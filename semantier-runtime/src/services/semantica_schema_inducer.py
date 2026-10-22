"""Thin deterministic adapter around the pinned Semantica ontology surface."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from eos.artifact_hashing import canonical_json, content_hash
from semantier.knowledge_feature_packs.semantica_adapter import (
    SEMANTICA_COMMIT,
    _ensure_vendored_semantica_importable,
)


class SemanticaSchemaInducer:
    """Generate and persist Semantica-owned ontology evaluation evidence."""

    def __init__(self, workspace_root: str | Path, *, workspace_id: str) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_id = workspace_id
        if not workspace_id.strip():
            raise ValueError("workspace_id is required")
        self.workspace_path = (
            self.workspace_root
            if self.workspace_root.name == workspace_id
            and self.workspace_root.parent.name == "workspaces"
            else self.workspace_root / "workspaces" / workspace_id
        )

    def induce(
        self,
        *,
        semantic_network: dict[str, Any],
        competency_questions: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Run real deterministic ontology, SHACL, and evaluator primitives."""

        if not isinstance(semantic_network, dict):
            raise ValueError("semantic_network must be an object")
        entities = semantic_network.get("entities")
        relationships = semantic_network.get("relationships")
        if not isinstance(entities, list) or not entities:
            raise ValueError("semantic_network.entities must not be empty")
        if not isinstance(relationships, list):
            raise ValueError("semantic_network.relationships must be a list")

        _ensure_vendored_semantica_importable()
        from semantica.ontology import OntologyEngine

        engine = OntologyEngine(min_occurrences=1)
        ontology = engine.from_data(
            {"entities": entities, "relationships": relationships},
            name="SemantierSchemaCandidate",
            validate=True,
        )
        ontology = json.loads(canonical_json(ontology))
        ontology.get("metadata", {}).pop("generated_at", None)

        data_graph = self._data_graph_turtle(semantic_network, ontology)
        shacl_report = engine.validate_graph(
            data_graph,
            ontology=ontology,
            data_graph_format="turtle",
            explain=True,
            abort_on_first=False,
        ).to_dict()
        evaluation = asdict(
            engine.evaluator.evaluate_ontology(
                ontology,
                competency_questions=list(competency_questions),
            )
        )

        input_hash = f"sha256:{content_hash({'semantic_network': semantic_network, 'competency_questions': list(competency_questions), 'semantica_commit': SEMANTICA_COMMIT})}"
        ontology_payload = {
            "schema_version": "semantica_ontology_candidate.v1",
            "semantica_commit": SEMANTICA_COMMIT,
            "generation_mode": "deterministic",
            "input_hash": input_hash,
            "ontology": ontology,
        }
        shacl_payload = {
            "schema_version": "semantica_shacl_report.v1",
            "semantica_commit": SEMANTICA_COMMIT,
            "input_hash": input_hash,
            "blocking_scope": "semantica_shape_conformance",
            "report": shacl_report,
        }
        evaluation_payload = {
            "schema_version": "semantica_ontology_evaluation.v1",
            "semantica_commit": SEMANTICA_COMMIT,
            "input_hash": input_hash,
            "authority": "advisory",
            "evaluation": evaluation,
        }

        ontology_ref, ontology_hash = self._persist("ontology", ontology_payload)
        shacl_ref, shacl_hash = self._persist("shacl", shacl_payload)
        evaluation_ref, evaluation_hash = self._persist(
            "evaluation", evaluation_payload
        )
        return {
            "semantica_commit": SEMANTICA_COMMIT,
            "generation_mode": "deterministic",
            "input_hash": input_hash,
            "ontology_candidate_ref": ontology_ref,
            "ontology_candidate_hash": ontology_hash,
            "shacl_report_ref": shacl_ref,
            "shacl_report_hash": shacl_hash,
            "ontology_evaluation_ref": evaluation_ref,
            "ontology_evaluation_hash": evaluation_hash,
            "shacl_blocking": True,
            "ontology_evaluation_advisory": True,
            "ontology": ontology,
            "shacl_report": shacl_report,
            "ontology_evaluation": evaluation,
        }

    def _persist(self, kind: str, payload: dict[str, Any]) -> tuple[str, str]:
        digest = content_hash(payload)
        relative = Path("artifacts") / "schema-discovery" / "semantica" / digest / f"{kind}.json"
        target = self.workspace_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
        if target.exists() and target.read_text(encoding="utf-8") != encoded:
            raise ValueError("SEMANTICA_SCHEMA_ARTIFACT_CONFLICT")
        target.write_text(encoded, encoding="utf-8")
        return relative.as_posix(), f"sha256:{digest}"

    @staticmethod
    def _data_graph_turtle(
        semantic_network: dict[str, Any], ontology: dict[str, Any]
    ) -> str:
        """Project input instances into a deterministic RDF graph for SHACL."""

        base = str(ontology.get("uri") or "https://semantica.dev/ontology/")
        if not base.endswith(("/", "#")):
            base += "/"
        class_names = {
            str(item.get("metadata", {}).get("inferred_from") or item.get("name")): str(
                item.get("name")
            )
            for item in ontology.get("classes", [])
        }
        entity_uris: dict[str, str] = {}
        lines = [f"@prefix sem: <{base}> ."]
        for index, entity in enumerate(semantic_network.get("entities", [])):
            identity = str(entity.get("id") or entity.get("name") or index)
            name = str(entity.get("name") or entity.get("text") or identity)
            entity_type = str(entity.get("type") or entity.get("entity_type") or "Entity")
            uri_name = quote(identity, safe="-._~")
            entity_uris[identity] = uri_name
            entity_uris[name] = uri_name
            class_name = class_names.get(entity_type, entity_type)
            lines.append(f"sem:{uri_name} a sem:{quote(class_name, safe='-._~')} .")
        for relation in semantic_network.get("relationships", []):
            source = entity_uris.get(str(relation.get("source")))
            target = entity_uris.get(str(relation.get("target")))
            predicate = str(
                relation.get("type")
                or relation.get("relationship_type")
                or "relatedTo"
            )
            if source and target:
                lines.append(
                    f"sem:{source} sem:{quote(predicate, safe='-._~')} sem:{target} ."
                )
        return "\n".join(lines) + "\n"


__all__ = ["SemanticaSchemaInducer"]
