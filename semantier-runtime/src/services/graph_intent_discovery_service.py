"""Source-grounded, non-authoritative GraphIntent proposal discovery."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable

from contracts.canonical_source_ir import CanonicalSourceDocument
from contracts.canonical_source_ir import content_hash as canonical_ir_content_hash
from contracts.evidence_location import SourceIdentity
from contracts.graph_intent_brief import (
    AlternativeConsidered,
    AuthorityBoundary,
    CompetencyQuestion,
    GraphIntentBrief,
    GraphIntentSourceBinding,
    HardNegativeQuestion,
    IntentTypeDescription,
    PreferredRepresentation,
    PrimaryRelationship,
    SourceRoleInterpretation,
    graph_intent_proposal_output_schema,
)
from eos.artifact_hashing import canonical_json, content_hash
from services.corpus_graph_schema_profile_registry import CorpusGraphSchemaProfileRegistry
from services.graph_schema_llm_evidence_runner import GraphSchemaLlmEvidenceRunner
from storage.graph_intent_brief_store import (
    GraphIntentBriefStore,
    GraphIntentPublicationTransaction,
)


_PROMPT = Path(__file__).resolve().parents[1] / "prompts/graph_intent_discovery_v1.txt"
_PARTIAL_POLICY = (
    Path(__file__).resolve().parents[1] / "prompts/graph_intent_partial_policy_v1.json"
)
_MACHINE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$")
_MATERIAL_SEMANTICS = frozenset(
    {
        "topology", "direction", "cardinality", "representation",
        "authority_interpretation", "grouping", "applicability",
        "evidence_ownership", "extraction_behavior",
    }
)
_CORE_CLAIMS = (
    "document_function", "represented_or_governed_subject",
    "primary_relationship", "preferred_representation", "authority_boundary",
)
_ROLE_FIELDS = (
    "canonical_objects", "lexical_evidence", "reference_examples",
    "classifications", "source_assessments", "remediation_fields",
    "contextual_fields",
)
_OBSERVABLE_PARTIAL_CLAIMS = frozenset({"document_function"})
_STRUCTURAL_HEADING_OBSERVED = "STRUCTURAL_HEADING_OBSERVED"
_GENERIC_STRUCTURE_INSUFFICIENT = "GENERIC_CANONICAL_STRUCTURE_INSUFFICIENT"
_LLM_UNAVAILABLE = "LLM_PROPOSAL_EVIDENCE_UNAVAILABLE_OR_MALFORMED"
_INVALID_TYPED_PROPOSAL = "INVALID_TYPED_PROPOSAL"
_UNSUPPORTED_MATERIAL_CLAIM = "UNSUPPORTED_MATERIAL_CLAIM"
_UNRELATED_CLAIM_EVIDENCE = "UNRELATED_CLAIM_EVIDENCE"
_UNKNOWN_CANONICAL_SOURCE_BLOCK = "UNKNOWN_CANONICAL_SOURCE_BLOCK"
_CATALOG_ADMISSION_FAILED = "CATALOG_ADMISSION_FAILED"
_CATALOG_HARD_NEGATIVE_ADMISSION_FAILED = "CATALOG_HARD_NEGATIVE_ADMISSION_FAILED"
_CATALOG_AUTHORITY_ADMISSION_FAILED = "CATALOG_AUTHORITY_ADMISSION_FAILED"
_FOREIGN_VOCABULARY_REJECTED = "FOREIGN_VOCABULARY_REJECTED"
_CATALOG_ERROR_CODES = frozenset(
    {
        _CATALOG_ADMISSION_FAILED,
        _CATALOG_HARD_NEGATIVE_ADMISSION_FAILED,
        _CATALOG_AUTHORITY_ADMISSION_FAILED,
        _FOREIGN_VOCABULARY_REJECTED,
    }
)


def _proposal_evidence_blocks(canonical: CanonicalSourceDocument):
    eligible_types = {"paragraph", "heading", "list_item", "cell", "legal_article"}
    return [
        block for block in canonical.blocks
        if block.block_type in eligible_types and str(block.content or "").strip()
    ]


class _ProposalValidationError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class GraphIntentDiscoveryService:
    """Compile a reviewable T6 proposal from source evidence, never catalog answers."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        intent_store: GraphIntentBriefStore,
        profile_registry: CorpusGraphSchemaProfileRegistry | None = None,
        llm_evidence_runner: GraphSchemaLlmEvidenceRunner | None = None,
    ) -> None:
        resolved_root = Path(workspace_root).resolve()
        if resolved_root != intent_store.workspace_path and not (
            intent_store.workspace_path.is_relative_to(resolved_root)
        ):
            raise ValueError("intent store must resolve below the governed workspace root")
        self.intent_store = intent_store
        self.profile_registry = profile_registry or CorpusGraphSchemaProfileRegistry()
        self.llm_evidence_runner = llm_evidence_runner
        self.proposal_root = intent_store.artifact_root / "graph-intent-proposals"
        self.assessment_root = (
            intent_store.artifact_root / "graph-intent-partial-assessments"
        )
        self.prompt_root = intent_store.artifact_root / "graph-intent-prompts"
        self.policy_root = intent_store.artifact_root / "graph-intent-policies"

    @staticmethod
    def _machine_id(value: Any, field: str) -> str:
        if (
            not isinstance(value, str)
            or value != value.strip()
            or not value
            or not value.isascii()
            or _MACHINE_ID.fullmatch(value) is None
        ):
            raise ValueError(
                f"{field} must be an exact ASCII-stable machine identifier"
            )
        return value

    @staticmethod
    def _semantic_category_sets(
        semantic_categories: dict[str, Any],
    ) -> tuple[frozenset[str], frozenset[str]]:
        if not isinstance(semantic_categories, dict) or set(
            semantic_categories
        ) != {"material", "non_material"}:
            raise ValueError("semantic category catalog is invalid")
        material = semantic_categories["material"]
        non_material = semantic_categories["non_material"]
        if (
            not isinstance(material, list)
            or not isinstance(non_material, list)
            or not material
        ):
            raise ValueError("semantic category catalog is invalid")
        material_set: set[str] = set()
        non_material_set: set[str] = set()
        for bucket, values in (
            ("material", material),
            ("non_material", non_material),
        ):
            target = material_set if bucket == "material" else non_material_set
            for index, item in enumerate(values):
                GraphIntentDiscoveryService._machine_id(
                    item, f"semantic_categories.{bucket}[{index}]"
                )
                target.add(item)
        if (
            len(material_set) != len(material)
            or len(non_material_set) != len(non_material)
            or material_set & non_material_set
        ):
            raise ValueError("semantic category catalog is invalid")
        return frozenset(material_set), frozenset(non_material_set)

    @classmethod
    def _current_semantic_categories(cls) -> dict[str, Any]:
        try:
            policy = json.loads(_PARTIAL_POLICY.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("graph intent partial policy must be UTF-8 JSON") from exc
        cls._validate_partial_policy(policy)
        return deepcopy(policy["semantic_categories"])

    @staticmethod
    def _affected_semantics(
        value: Any,
        field: str,
        semantic_categories: dict[str, Any],
    ) -> tuple[str, ...]:
        if not isinstance(value, list):
            raise ValueError(f"{field} must be a list of governed affected_semantics")
        material, non_material = GraphIntentDiscoveryService._semantic_category_sets(
            semantic_categories
        )
        allowed = material | non_material
        normalized: list[str] = []
        for index, item in enumerate(value):
            GraphIntentDiscoveryService._machine_id(item, f"{field}[{index}]")
            if item not in allowed:
                raise ValueError(
                    f"{field}[{index}] is not a governed affected_semantics value"
                )
            normalized.append(item)
        return tuple(normalized)

    @staticmethod
    def _is_material(
        affected_semantics: Iterable[str],
        semantic_categories: dict[str, Any],
    ) -> bool:
        material, _non_material = GraphIntentDiscoveryService._semantic_category_sets(
            semantic_categories
        )
        return any(item in material for item in affected_semantics)

    @classmethod
    def is_material_question(cls, affected_semantics: Iterable[str]) -> bool:
        semantic_categories = cls._current_semantic_categories()
        values = cls._affected_semantics(
            affected_semantics, "affected_semantics", semantic_categories
        )
        return cls._is_material(values, semantic_categories)

    def propose(
        self,
        *,
        source_identity: SourceIdentity,
        lineage_id: str,
        canonical_source_ir_ref: str,
        canonical_source_ir_hash: str,
        profile_id: str,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        llm_config: dict[str, Any] | None = None,
        operating_mode: str = "assisted",
        expected_profile_hash: str | None = None,
        expected_catalog_hash: str | None = None,
        expected_prompt_asset_hash: str | None = None,
        expected_config_hash: str | None = None,
        attempt_ref: str | None = None,
    ) -> GraphIntentBrief:
        if attempt_ref is not None:
            self._machine_id(attempt_ref, "attempt_ref")
        if operating_mode not in {"assisted", "auto_pilot"}:
            raise ValueError("operating_mode must be assisted or auto_pilot")
        profile = self.profile_registry.resolve(profile_id)
        catalog = self.profile_registry.graph_intent_catalog(
            profile_id,
            expected_profile_hash=expected_profile_hash,
            expected_catalog_hash=expected_catalog_hash,
        )
        if profile.graph_intent_catalog_hash is None:
            raise ValueError("registered profile lacks a GraphIntent catalog")
        binding = GraphIntentSourceBinding(
            organization_id=self.intent_store.organization_id,
            workspace_id=self.intent_store.workspace_id,
            lineage_id=lineage_id,
            source_identity_ref=source_identity.source_identity_ref,
            source_hash=source_identity.source_hash,
            source_version=source_identity.source_version,
            canonical_source_ir_ref=canonical_source_ir_ref,
            canonical_source_ir_hash=canonical_source_ir_hash,
        )
        canonical = self.intent_store.verify_source_binding(
            binding, source_identity=source_identity
        )
        blocks = _proposal_evidence_blocks(canonical)
        if not blocks:
            raise ValueError("canonical source IR contains no proposal evidence blocks")
        block_by_id = {block.block_id: block for block in blocks}
        evidence_id_by_alias = {
            f"block_{index:06d}": block.block_id
            for index, block in enumerate(blocks, start=1)
        }
        evidence_ref_by_id = {
            block_id: f"evidence:canonical-block:{block_id.removeprefix('sha256:')}"
            for block_id in block_by_id
        }
        structural_context = self._structural_context(canonical, blocks)
        matched_ids = self._matched_block_ids(catalog, blocks)
        config = (
            {"temperature": 0}
            if llm_config is None
            else deepcopy(llm_config)
        )
        if (
            config != {"temperature": 0}
            or type(config.get("temperature")) is not int
        ):
            raise ValueError("unsupported GraphIntent proposal config")
        partial_policy, partial_policy_hash, partial_policy_ref = (
            self._read_partial_policy()
        )
        semantic_categories = partial_policy["semantic_categories"]
        prompt_bytes = _PROMPT.read_bytes()
        try:
            prompt_template = prompt_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("graph intent prompt asset must be UTF-8") from exc
        prompt_hash = f"sha256:{sha256(prompt_bytes).hexdigest()}"
        prompt_ref = self._write_prompt_asset(prompt_bytes, prompt_hash)
        prompt = self._prompt(prompt_template, profile_id, catalog, blocks)
        llm = self._llm_evidence(
            provider=llm_provider, model=llm_model, prompt=prompt, config=config,
            output_schema=graph_intent_proposal_output_schema(catalog),
            profile_hash=profile.content_hash,
            catalog_hash=profile.graph_intent_catalog_hash,
            prompt_asset_hash=prompt_hash,
            prompt_asset_ref=prompt_ref,
            expected_prompt_asset_hash=expected_prompt_asset_hash,
            expected_config_hash=expected_config_hash,
            attempt_ref=attempt_ref,
        )
        typed = self._typed_runner_proposal(llm)
        unresolved: list[dict[str, Any]] = []
        minimum = int(catalog.get("minimum_source_evidence_matches", 1))
        incompatible_profile = len(matched_ids) < minimum
        typed_error_code: str | None = None
        claim_evidence: dict[str, list[str]]
        partial_assessment: dict[str, Any] | None = None
        partial_assessment_pin: dict[str, str] | None = None
        if not incompatible_profile and typed is not None:
            try:
                resolved_typed = self._resolve_evidence_aliases(
                    typed, evidence_id_by_alias=evidence_id_by_alias
                )
                resolved_typed = self._filter_admitted_source_role_ids(
                    resolved_typed, catalog=catalog
                )
                proposal, claim_block_ids = self._validate_typed_proposal(
                    resolved_typed,
                    catalog=catalog,
                    block_by_id=block_by_id,
                    semantic_categories=semantic_categories,
                )
                self._reject_foreign_vocabulary(proposal, active_catalog=catalog)
                claim_evidence = {
                    claim: [evidence_ref_by_id[item] for item in ids]
                    for claim, ids in claim_block_ids.items()
                    if ids
                }
            except (KeyError, TypeError, ValueError) as exc:
                typed_error_code = self._proposal_error_code(exc)
        elif typed is None:
            typed_error_code = _LLM_UNAVAILABLE
        if incompatible_profile or typed_error_code is not None:
            reason_codes: dict[str, str] = {}
            if incompatible_profile:
                reason_codes["profile_source_compatibility"] = (
                    "PROFILE_SOURCE_COMPATIBILITY_UNRESOLVED"
                )
            if typed_error_code is not None:
                reason_codes["proposal_evidence"] = typed_error_code
            partial_assessment, partial_assessment_pin = (
                self._stage_partial_assessment(
                    source_identity=source_identity,
                    blocks=blocks,
                    structural_context=structural_context,
                    profile_id=profile_id,
                    profile_hash=profile.content_hash,
                    catalog_hash=profile.graph_intent_catalog_hash,
                    partial_policy_hash=partial_policy_hash,
                    reason_codes=reason_codes,
                )
            )
            assessment_ref = partial_assessment_pin["evidence_ref"]
            proposal, claim_evidence, partial_questions = self._source_partial(
                partial_policy,
                block_by_id,
                evidence_ref_by_id,
                partial_assessment,
                assessment_ref,
            )
            unresolved.extend(partial_questions)
            if incompatible_profile:
                unresolved.append(
                    self._policy_question(
                        partial_policy,
                        "profile_source_compatibility",
                        partial_assessment,
                        assessment_ref,
                    )
                )
            if typed_error_code is not None:
                diagnostic = (
                    "catalog_admission"
                    if typed_error_code in _CATALOG_ERROR_CODES
                    else "proposal_evidence"
                )
                unresolved.append(
                    self._policy_question(
                        partial_policy,
                        diagnostic,
                        partial_assessment,
                        assessment_ref,
                        failure_code=typed_error_code,
                    )
                )
        for index, item in enumerate(proposal.pop("unresolved_questions", [])):
            affected = list(item["affected_semantics"])
            unresolved.append(
                self._question(
                    item["question_id"],
                    item["question"],
                    affected,
                    claim_evidence.get(f"unresolved_questions.{index}", []),
                    semantic_categories,
                )
            )
        if any(item["material"] and not item["evidence_refs"] for item in unresolved):
            raise ValueError("material unresolved question lacks source evidence")
        claim_evidence = {
            claim: refs for claim, refs in claim_evidence.items()
            if refs and not claim.startswith("unresolved_questions.")
        }
        for index, item in enumerate(unresolved):
            if item["evidence_refs"]:
                claim_evidence[f"unresolved_questions.{index}"] = list(
                    item["evidence_refs"]
                )
        inverse_refs = {ref: block_id for block_id, ref in evidence_ref_by_id.items()}
        used_source_refs = {
            ref
            for refs in claim_evidence.values()
            for ref in refs
            if ref in inverse_refs
        }
        source_block_evidence = {
            ref: {
                "block_id": inverse_refs[ref],
                "content_hash": block_by_id[inverse_refs[ref]].content_hash,
                "block_type": block_by_id[inverse_refs[ref]].block_type,
                "exact_text": str(block_by_id[inverse_refs[ref]].content),
                **deepcopy(structural_context[inverse_refs[ref]]),
            }
            for ref in sorted(used_source_refs)
        }
        evidence_payload = {
            "schema_version": "graph_intent_proposal_evidence.v1",
            "authority_tier": "T6", "authoritative": False,
            "source_pins": {
                "source_identity_ref": source_identity.source_identity_ref,
                "source_hash": source_identity.source_hash,
                "canonical_source_ir_ref": canonical_source_ir_ref,
                "canonical_source_ir_hash": canonical_source_ir_hash,
            },
            "profile_id": profile_id,
            "profile_hash": profile.content_hash,
            "catalog_hash": profile.graph_intent_catalog_hash,
            "partial_policy": {
                "asset_hash": partial_policy_hash,
                "asset_ref": partial_policy_ref,
            },
            "partial_assessment": partial_assessment_pin,
            "claim_evidence": claim_evidence,
            "source_block_evidence": source_block_evidence,
            "llm_evidence": llm,
        }
        evidence_hash = f"sha256:{content_hash(evidence_payload)}"
        evidence_ref = f"evidence:graph-intent-proposal:{evidence_hash[7:]}"
        pending_artifacts = [
            self._proposal_evidence_publication(evidence_payload, evidence_hash)
        ]
        if partial_assessment is not None and partial_assessment_pin is not None:
            pending_artifacts.insert(
                0,
                self._partial_assessment_publication(
                    partial_assessment, partial_assessment_pin
                ),
            )
        for target, encoded, error in pending_artifacts:
            self._require_artifact_no_conflict(target, encoded, error)
        material_unresolved = any(item["material"] for item in unresolved)
        actual_mode = (
            "assisted"
            if operating_mode == "auto_pilot" and material_unresolved
            else operating_mode
        )
        fallback_reason = (
            partial_policy["fallback_reason_templates"]["auto_pilot_material_ambiguity"]
            if operating_mode == "auto_pilot" and material_unresolved
            else None
        )
        dimensions = []
        for index, item in enumerate(proposal.get("material_dimensions", [])):
            refs = claim_evidence.get(f"material_dimensions.{index}", [])
            if refs:
                dimensions.append({**item, "evidence_refs": refs})
        content = {
            "source_binding": binding.as_dict(),
            "document_function": proposal["document_function"],
            "represented_or_governed_subject": proposal["represented_or_governed_subject"],
            "primary_relationship": proposal["primary_relationship"],
            "preferred_representation": proposal["preferred_representation"],
            "material_dimensions": dimensions,
            "source_role_interpretation": proposal["source_role_interpretation"],
            "authority_boundary": proposal["authority_boundary"],
            "competency_questions": proposal.get("competency_questions", []),
            "hard_negative_questions": proposal.get("hard_negative_questions", []),
            "alternatives_considered": proposal.get("alternatives_considered", []),
            "evidence_refs": sorted(
                {ref for refs in claim_evidence.values() for ref in refs} | {evidence_ref}
            ),
            "assumptions": proposal.get("assumptions", []),
            "unresolved_questions": unresolved,
            "clarification_refs": [],
            "review": {
                "state": "proposed", "reviewed_by": "system",
                "actor_ref": "system:graph-intent-discovery",
                "operating_mode": actual_mode,
                "policy_ref": (
                    "policy:graph-intent-partial:"
                    f"{partial_policy_hash.removeprefix('sha256:')}"
                ),
                "rationale_ref": None,
                "decision_evidence_refs": [evidence_ref],
                "fallback_reason": fallback_reason,
            },
            "predecessor_graph_intent_ref": None,
            "successor_graph_intent_refs": [],
        }
        self._reject_foreign_vocabulary(content, active_catalog=catalog)
        brief = GraphIntentBrief.create(content)
        self._reject_foreign_vocabulary(brief.as_dict(), active_catalog=catalog)
        with GraphIntentPublicationTransaction(
            self.intent_store.artifact_root
        ) as transaction:
            for target, encoded, error in pending_artifacts:
                transaction.require_no_conflict(target, encoded, error)
            stored = self.intent_store.create(brief, source_identity=source_identity)
            for target, encoded, error in pending_artifacts:
                self._publish_artifact(target, encoded, error)
        return stored

    def replay(
        self, graph_intent_ref: str, *, graph_intent_hash: str | None = None
    ) -> dict[str, Any]:
        brief = self.intent_store.get(
            graph_intent_ref, graph_intent_hash=graph_intent_hash
        )
        return {
            "brief": brief.as_dict(),
            "proposal_evidence": self.load_proposal_evidence(brief),
        }

    def load_proposal_evidence(self, brief: GraphIntentBrief) -> dict[str, Any]:
        refs = [
            ref for ref in brief.evidence_refs
            if ref.startswith("evidence:graph-intent-proposal:")
        ]
        if len(refs) != 1:
            raise ValueError("graph intent must pin exactly one proposal evidence artifact")
        digest = refs[0].rsplit(":", 1)[1]
        path = self.proposal_root / digest / "graph_intent_proposal_evidence.v1.json"
        payload = json.loads(
            self.intent_store.read_committed_artifact_by_hash(
                brief.graph_intent_id,
                "proposal_evidence",
                f"sha256:{digest}",
                expected_path=path.resolve(),
                expected_kind="graph_intent_discovery_proposal",
            )
        )
        if content_hash(payload) != digest:
            raise ValueError("graph intent proposal evidence hash mismatch")
        pins = payload["source_pins"]
        if (
            pins["source_identity_ref"]
            != brief.source_binding.source_identity_ref
            or pins["canonical_source_ir_ref"]
            != brief.source_binding.canonical_source_ir_ref
            or pins["source_hash"] != brief.source_binding.source_hash
            or pins["canonical_source_ir_hash"]
            != brief.source_binding.canonical_source_ir_hash
        ):
            raise ValueError("graph intent proposal source pins mismatch")
        canonical = self._load_bound_canonical_source_ir(brief)
        self.profile_registry.graph_intent_catalog(
            payload["profile_id"],
            expected_profile_hash=payload["profile_hash"],
            expected_catalog_hash=payload["catalog_hash"],
        )
        self._verify_historical_partial_policy(payload.get("partial_policy"))
        self._verify_partial_assessment_pin(
            payload.get("partial_assessment"),
            payload,
            brief=brief,
            canonical=canonical,
        )
        llm = payload["llm_evidence"]
        prompt_ref = llm.get("prompt_asset_ref")
        if not isinstance(prompt_ref, str) or not prompt_ref:
            raise ValueError("graph intent evidence lacks historical prompt ref")
        prompt_path = (self.intent_store.workspace_path / prompt_ref).resolve()
        if not prompt_path.is_relative_to(self.intent_store.artifact_root):
            raise ValueError("graph intent prompt ref escapes governed artifact root")
        prompt_bytes = prompt_path.read_bytes()
        prompt_hash = f"sha256:{sha256(prompt_bytes).hexdigest()}"
        if llm["prompt_asset_hash"] != prompt_hash:
            raise ValueError("graph intent prompt asset hash mismatch")
        if llm["config_hash"] != f"sha256:{content_hash(llm['config'])}":
            raise ValueError("graph intent proposal config hash mismatch")
        if "structured_output_schema" in llm:
            if llm.get("structured_output_schema_hash") != f"sha256:{content_hash(llm['structured_output_schema'])}":
                raise ValueError("graph intent structured output schema hash mismatch")
        attempts = llm.get("structured_output_attempts", [])
        for index, attempt in enumerate(attempts, start=1):
            if (
                attempt["attempt_number"] != index
                or (attempt["provider"], attempt["model"]) != (llm["provider"], llm["model"])
                or attempt["input_hash"] != f"sha256:{content_hash(attempt['input_text'])}"
                or attempt["instruction_hash"] != "sha256:" + sha256(attempt["instructions"].encode("utf-8")).hexdigest()
                or attempt["output_schema_hash"] != llm["structured_output_schema_hash"]
                or (index == 1 and attempt["input_hash"] != llm["formatted_input_hash"])
            ):
                raise ValueError("graph intent structured attempt binding mismatch")
        if attempts and ((llm["status"] == "passed") != (attempts[-1]["status"] == "passed")):
            raise ValueError("graph intent structured attempt outcome mismatch")
        self._verify_runner_output(llm)
        return payload

    def proposal_validation_failure(self, brief: GraphIntentBrief) -> str | None:
        """Read retained producer rejection separately from material review.

        A successful provider response may still fail proposal admission. This
        read uses committed evidence and never re-runs inference or current policy.
        """
        evidence = self.load_proposal_evidence(brief)
        if evidence["llm_evidence"]["status"] != "passed":
            return _LLM_UNAVAILABLE
        partial = evidence.get("partial_assessment")
        if partial is None:
            return None
        assessment = self.load_partial_assessment(
            partial["evidence_ref"], graph_intent_ref=brief.graph_intent_id
        )
        claims = assessment["claim_assessments"]
        for claim in ("proposal_evidence", "catalog_admission"):
            diagnostic = claims.get(claim)
            if diagnostic is not None:
                return diagnostic["reason_code"]
        return None

    def load_partial_assessment(
        self,
        evidence_ref: str,
        *,
        graph_intent_ref: str,
    ) -> dict[str, Any]:
        prefix = "evidence:graph-intent-partial-assessment:"
        if not isinstance(evidence_ref, str) or not evidence_ref.startswith(prefix):
            raise ValueError("graph intent partial assessment ref is malformed")
        digest = evidence_ref.removeprefix(prefix)
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("graph intent partial assessment ref is malformed")
        path = self.assessment_root / digest / "graph_intent_partial_assessment.v1.json"
        if not path.resolve().is_relative_to(self.assessment_root.resolve()):
            raise ValueError("graph intent partial assessment ref escapes governed root")
        payload = json.loads(
            self.intent_store.read_committed_artifact_by_hash(
                graph_intent_ref,
                "partial_assessment",
                f"sha256:{digest}",
                expected_path=path.resolve(),
                expected_kind="graph_intent_discovery_proposal",
            )
        )
        if content_hash(payload) != digest:
            raise ValueError("graph intent partial assessment hash mismatch")
        self._validate_partial_assessment(payload)
        return payload

    def _load_bound_canonical_source_ir(
        self, brief: GraphIntentBrief
    ) -> CanonicalSourceDocument:
        path = self.intent_store._resolve_canonical_source_ir(  # noqa: SLF001
            brief.source_binding.canonical_source_ir_ref
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        canonical = CanonicalSourceDocument.model_validate(payload)
        canonical_payload = canonical.model_dump(
            mode="json", by_alias=True, exclude_none=False
        )
        if (
            canonical_ir_content_hash(canonical_payload)
            != brief.source_binding.canonical_source_ir_hash
        ):
            raise ValueError("canonical_source_ir_hash mismatch")
        if canonical.source_identity_ref != brief.source_binding.source_identity_ref:
            raise ValueError("canonical source IR source identity mismatch")
        locators = [
            locator
            for mapping in canonical.source_map.mappings
            for locator in mapping.source_locators
        ]
        if not locators or any(
            locator.source_hash != brief.source_binding.source_hash
            or locator.source_identity_ref
            != brief.source_binding.source_identity_ref
            for locator in locators
        ):
            raise ValueError(
                "CanonicalSourceIR locator source_hash or source identity mismatch"
            )
        if (
            f"sha256:{path.parent.name}"
            != brief.source_binding.canonical_source_ir_hash
        ):
            raise ValueError("canonical_source_ir_ref content hash mismatch")
        return canonical

    @classmethod
    def _verify_partial_assessment_source_structure(
        cls,
        assessment: dict[str, Any],
        canonical: CanonicalSourceDocument,
    ) -> None:
        canonical_blocks = _proposal_evidence_blocks(canonical)
        block_by_id = {block.block_id: block for block in canonical_blocks}
        inspected_ids = [
            block["block_id"] for block in assessment["inspected_blocks"]
        ]
        eligible_ids = [block.block_id for block in sorted(
            canonical_blocks, key=lambda block: block.block_id
        )]
        if (
            len(inspected_ids) != len(set(inspected_ids))
            or inspected_ids != eligible_ids
        ):
            raise ValueError(
                "graph intent partial assessment inspected canonical source blocks mismatch"
            )
        structural_context = cls._structural_context(
            canonical,
            [block_by_id[block_id] for block_id in inspected_ids],
        )
        expected_blocks = [
            cls._assessment_block_payload(block_by_id[block_id], structural_context)
            for block_id in inspected_ids
        ]
        if assessment["inspected_blocks"] != expected_blocks:
            raise ValueError(
                "graph intent partial assessment inspected canonical source blocks mismatch"
            )
        expected_hash = (
            f"sha256:{content_hash(cls._assessment_structure_fingerprint(expected_blocks))}"
        )
        if (
            assessment["source_pins"]["canonical_source_structure_hash"]
            != expected_hash
        ):
            raise ValueError(
                "graph intent partial assessment canonical source structure hash mismatch"
            )

    @staticmethod
    def _assessment_block_payload(
        block: Any,
        structural_context: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        context = structural_context[block.block_id]
        return {
            "block_id": block.block_id,
            "content_hash": block.content_hash,
            "block_type": block.block_type,
            "role_code": context["role_code"],
            "structural_role": context["structural_role"],
            "parent_block_id": block.parent_block_id,
            "ordinal": context["ordinal"],
            "structure": deepcopy(context["structure"]),
            "source_mappings": deepcopy(context["source_mappings"]),
        }

    @staticmethod
    def _assessment_structure_fingerprint(
        inspected_blocks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "block_id": item["block_id"],
                "content_hash": item["content_hash"],
                "block_type": item["block_type"],
                "role_code": item["role_code"],
                "parent_block_id": item["parent_block_id"],
                "structure": item["structure"],
                "source_mappings": item["source_mappings"],
            }
            for item in inspected_blocks
        ]

    def _verify_partial_assessment_pin(
        self,
        value: Any,
        proposal_evidence: dict[str, Any],
        *,
        brief: GraphIntentBrief,
        canonical: CanonicalSourceDocument,
    ) -> None:
        if value is None:
            return
        if not isinstance(value, dict) or set(value) != {
            "artifact_hash",
            "artifact_ref",
            "evidence_ref",
        }:
            raise ValueError("graph intent partial assessment pin is malformed")
        artifact_hash = value["artifact_hash"]
        artifact_ref = value["artifact_ref"]
        evidence_ref = value["evidence_ref"]
        if (
            not isinstance(artifact_hash, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", artifact_hash) is None
            or not isinstance(artifact_ref, str)
            or not artifact_ref
            or not isinstance(evidence_ref, str)
        ):
            raise ValueError("graph intent partial assessment pin is malformed")
        target = (self.intent_store.workspace_path / artifact_ref).resolve()
        if not target.is_relative_to(self.assessment_root.resolve()):
            raise ValueError("graph intent partial assessment ref escapes governed root")
        assessment = self.load_partial_assessment(
            evidence_ref,
            graph_intent_ref=brief.graph_intent_id,
        )
        if f"sha256:{content_hash(assessment)}" != artifact_hash:
            raise ValueError("graph intent partial assessment artifact hash mismatch")
        if target != (
            self.assessment_root
            / artifact_hash.removeprefix("sha256:")
            / "graph_intent_partial_assessment.v1.json"
        ).resolve():
            raise ValueError("graph intent partial assessment artifact ref mismatch")
        if (
            assessment["profile_id"] != proposal_evidence["profile_id"]
            or assessment["profile_hash"] != proposal_evidence["profile_hash"]
            or assessment["catalog_hash"] != proposal_evidence["catalog_hash"]
            or assessment["policy_hash"]
            != proposal_evidence["partial_policy"]["asset_hash"]
        ):
            raise ValueError("graph intent partial assessment pins mismatch")
        if (
            assessment["source_pins"]["source_identity_ref"]
            != proposal_evidence["source_pins"]["source_identity_ref"]
            or assessment["source_pins"]["source_identity_ref"]
            != brief.source_binding.source_identity_ref
        ):
            raise ValueError("graph intent partial assessment source identity mismatch")
        if (
            assessment["source_pins"]["source_hash"]
            != proposal_evidence["source_pins"]["source_hash"]
            or assessment["source_pins"]["source_hash"]
            != brief.source_binding.source_hash
        ):
            raise ValueError("graph intent partial assessment source hash mismatch")
        self._verify_partial_assessment_source_structure(assessment, canonical)

    def _write_evidence(self, payload: dict[str, Any], digest_ref: str) -> None:
        target, encoded, error = self._proposal_evidence_publication(
            payload, digest_ref
        )
        self._publish_artifact(target, encoded, error)

    def _proposal_evidence_publication(
        self, payload: dict[str, Any], digest_ref: str
    ) -> tuple[Path, str, str]:
        digest = digest_ref.removeprefix("sha256:")
        target = self.proposal_root / digest / "graph_intent_proposal_evidence.v1.json"
        encoded = canonical_json(payload)
        return target, encoded, "GRAPH_INTENT_PROPOSAL_EVIDENCE_CONFLICT"

    def _partial_assessment_publication(
        self,
        payload: dict[str, Any],
        pin: dict[str, str],
    ) -> tuple[Path, str, str]:
        artifact_ref = pin["artifact_ref"]
        target = (self.intent_store.workspace_path / artifact_ref).resolve()
        if not target.is_relative_to(self.assessment_root.resolve()):
            raise ValueError("graph intent partial assessment ref escapes governed root")
        if pin["artifact_hash"] != f"sha256:{content_hash(payload)}":
            raise ValueError("graph intent partial assessment artifact hash mismatch")
        encoded = canonical_json(payload)
        return target, encoded, "GRAPH_INTENT_PARTIAL_ASSESSMENT_CONFLICT"

    @staticmethod
    def _require_artifact_no_conflict(
        target: Path, encoded: str, error: str
    ) -> None:
        if target.exists() and target.read_text(encoding="utf-8") != encoded:
            raise ValueError(error)

    @classmethod
    def _publish_artifact(cls, target: Path, encoded: str, error: str) -> bool:
        transaction = GraphIntentPublicationTransaction.current_for_path(target)
        if transaction is not None:
            return transaction.publish(target, encoded, error)
        return GraphIntentPublicationTransaction.publish_unlocked(target, encoded, error)

    def _stage_partial_assessment(
        self,
        *,
        source_identity: SourceIdentity,
        blocks: list[Any],
        structural_context: dict[str, dict[str, Any]],
        profile_id: str,
        profile_hash: str,
        catalog_hash: str,
        partial_policy_hash: str,
        reason_codes: dict[str, str],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        payload = self._partial_assessment(
            source_identity=source_identity,
            blocks=blocks,
            structural_context=structural_context,
            profile_id=profile_id,
            profile_hash=profile_hash,
            catalog_hash=catalog_hash,
            partial_policy_hash=partial_policy_hash,
            reason_codes=reason_codes,
        )
        digest = content_hash(payload)
        digest_ref = f"sha256:{digest}"
        evidence_ref = f"evidence:graph-intent-partial-assessment:{digest}"
        target = (
            self.assessment_root
            / digest
            / "graph_intent_partial_assessment.v1.json"
        )
        return payload, {
            "artifact_hash": digest_ref,
            "artifact_ref": target.relative_to(
                self.intent_store.workspace_path
            ).as_posix(),
            "evidence_ref": evidence_ref,
        }

    @classmethod
    def _validate_partial_assessment(cls, value: Any) -> None:
        required = {
            "schema_version",
            "authority_tier",
            "authoritative",
            "source_pins",
            "profile_id",
            "profile_hash",
            "catalog_hash",
            "policy_hash",
            "structural_criteria",
            "inspected_blocks",
            "claim_assessments",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("graph intent partial assessment shape is invalid")
        if value["schema_version"] != "graph_intent_partial_assessment.v1":
            raise ValueError("graph intent partial assessment version is invalid")
        if value["authority_tier"] != "T6" or value["authoritative"] is not False:
            raise ValueError("graph intent partial assessment authority is invalid")
        cls._machine_id(value["profile_id"], "partial_assessment.profile_id")
        for field in ("profile_hash", "catalog_hash", "policy_hash"):
            if re.fullmatch(r"sha256:[0-9a-f]{64}", str(value[field])) is None:
                raise ValueError(f"partial_assessment.{field} is invalid")
        source_pins = value["source_pins"]
        if not isinstance(source_pins, dict) or set(source_pins) != {
            "source_identity_ref",
            "source_hash",
            "canonical_source_structure_hash",
        }:
            raise ValueError("graph intent partial assessment source pins are invalid")
        cls._machine_id(
            source_pins["source_identity_ref"],
            "partial_assessment.source_identity_ref",
        )
        for field in ("source_hash", "canonical_source_structure_hash"):
            if re.fullmatch(r"sha256:[0-9a-f]{64}", str(source_pins[field])) is None:
                raise ValueError(f"partial_assessment.{field} is invalid")
        if not isinstance(value["structural_criteria"], dict):
            raise ValueError("graph intent partial assessment criteria are invalid")
        blocks = value["inspected_blocks"]
        if not isinstance(blocks, list) or not blocks:
            raise ValueError("graph intent partial assessment blocks are invalid")
        block_ids: set[str] = set()
        for index, block in enumerate(blocks):
            if not isinstance(block, dict) or set(block) != {
                "block_id",
                "content_hash",
                "block_type",
                "role_code",
                "structural_role",
                "parent_block_id",
                "ordinal",
                "structure",
                "source_mappings",
            }:
                raise ValueError("graph intent partial assessment block is invalid")
            cls._machine_id(block["block_id"], f"inspected_blocks[{index}].block_id")
            block_ids.add(block["block_id"])
            for field in ("content_hash", "block_type", "role_code", "structural_role"):
                if not isinstance(block[field], str) or not block[field].strip():
                    raise ValueError(
                        f"partial_assessment.inspected_blocks[{index}].{field} "
                        "is invalid"
                    )
            if block["parent_block_id"] is not None:
                cls._machine_id(
                    block["parent_block_id"],
                    f"inspected_blocks[{index}].parent_block_id",
                )
            if not isinstance(block["structure"], dict) or not isinstance(
                block["source_mappings"], list
            ):
                raise ValueError("graph intent partial assessment block is invalid")
        assessments = value["claim_assessments"]
        if not isinstance(assessments, dict):
            raise ValueError("graph intent partial assessment claims are invalid")
        for claim, assessment in assessments.items():
            cls._machine_id(claim, "partial_assessment.claim")
            if not isinstance(assessment, dict):
                raise ValueError("graph intent partial assessment claim is invalid")
            if assessment.get("status") == "observed":
                expected_keys = {
                    "status",
                    "reason_code",
                    "observed_block_ids",
                    "inspected_block_ids",
                }
            elif assessment.get("status") == "unresolved":
                expected_keys = {"status", "reason_code", "inspected_block_ids"}
            else:
                raise ValueError("graph intent partial assessment status is invalid")
            if set(assessment) != expected_keys:
                raise ValueError("graph intent partial assessment claim is invalid")
            cls._machine_id(
                assessment["reason_code"],
                "partial_assessment.claim.reason_code",
            )
            inspected = assessment["inspected_block_ids"]
            if (
                not isinstance(inspected, list)
                or not inspected
                or any(item not in block_ids for item in inspected)
            ):
                raise ValueError("partial assessment inspected block ids are invalid")
            if assessment["status"] == "observed":
                observed = assessment["observed_block_ids"]
                if (
                    not isinstance(observed, list)
                    or not observed
                    or any(item not in block_ids for item in observed)
                ):
                    raise ValueError(
                        "partial assessment observed block ids are invalid"
                    )

    def _write_prompt_asset(self, prompt_bytes: bytes, digest_ref: str) -> str:
        digest = digest_ref.removeprefix("sha256:")
        target = self.prompt_root / digest / "graph_intent_discovery_v1.txt"
        if target.exists() and target.read_bytes() != prompt_bytes:
            raise ValueError("GRAPH_INTENT_PROMPT_ARTIFACT_CONFLICT")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(prompt_bytes)
        return target.relative_to(self.intent_store.workspace_path).as_posix()

    def _read_partial_policy(self) -> tuple[dict[str, Any], str, str]:
        policy_bytes = _PARTIAL_POLICY.read_bytes()
        try:
            policy = json.loads(policy_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("graph intent partial policy must be UTF-8 JSON") from exc
        self._validate_partial_policy(policy)
        policy_hash = f"sha256:{sha256(policy_bytes).hexdigest()}"
        policy_ref = self._write_policy_asset(policy_bytes, policy_hash)
        return deepcopy(policy), policy_hash, policy_ref

    def _write_policy_asset(self, policy_bytes: bytes, digest_ref: str) -> str:
        digest = digest_ref.removeprefix("sha256:")
        target = self.policy_root / digest / "graph_intent_partial_policy_v1.json"
        if target.exists() and target.read_bytes() != policy_bytes:
            raise ValueError("GRAPH_INTENT_PARTIAL_POLICY_ARTIFACT_CONFLICT")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(policy_bytes)
        return target.relative_to(self.intent_store.workspace_path).as_posix()

    def _verify_historical_partial_policy(self, value: Any) -> None:
        if not isinstance(value, dict) or set(value) != {"asset_hash", "asset_ref"}:
            raise ValueError("graph intent evidence lacks a partial policy pin")
        asset_hash = value["asset_hash"]
        asset_ref = value["asset_ref"]
        if (
            not isinstance(asset_hash, str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", asset_hash)
            or not isinstance(asset_ref, str)
            or not asset_ref
        ):
            raise ValueError("graph intent partial policy pin is malformed")
        target = (self.intent_store.workspace_path / asset_ref).resolve()
        if not target.is_relative_to(self.policy_root.resolve()):
            raise ValueError("graph intent partial policy ref escapes governed root")
        policy_bytes = target.read_bytes()
        if f"sha256:{sha256(policy_bytes).hexdigest()}" != asset_hash:
            raise ValueError("graph intent partial policy asset hash mismatch")
        try:
            policy = json.loads(policy_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "historical graph intent partial policy is invalid"
            ) from exc
        self._validate_partial_policy(policy)

    @classmethod
    def _validate_partial_policy(cls, value: Any) -> None:
        required = {
            "schema_version",
            "semantic_categories",
            "partial_fields",
            "clarification_questions",
            "diagnostic_questions",
            "failure_messages",
            "fallback_reason_templates",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("graph intent partial policy shape is invalid")
        if value["schema_version"] != "graph_intent_partial_policy.v1":
            raise ValueError("graph intent partial policy version is invalid")

        def exact_object(item: Any, keys: set[str], field: str) -> dict[str, Any]:
            if not isinstance(item, dict) or set(item) != keys:
                raise ValueError(f"graph intent partial policy {field} is invalid")
            return item

        semantic_categories = value["semantic_categories"]
        material, _non_material = cls._semantic_category_sets(semantic_categories)
        if material != _MATERIAL_SEMANTICS:
            raise ValueError("graph intent partial policy material categories differ")

        def prose(
            item: Any,
            field: str,
            *,
            observation: bool = False,
            assessment: bool = False,
            failure: bool = False,
        ) -> str:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"graph intent partial policy {field} is invalid")
            if observation and not all(
                placeholder in item
                for placeholder in ("{structural_role}", "{exact_text}")
            ):
                raise ValueError(
                    f"graph intent partial policy {field} lacks observation variables"
                )
            if assessment and not all(
                placeholder in item
                for placeholder in ("{assessment_ref}", "{reason_code}")
            ):
                raise ValueError(
                    f"graph intent partial policy {field} lacks assessment variables"
                )
            if failure and "{failure_message}" not in item:
                raise ValueError(
                    f"graph intent partial policy {field} lacks failure message"
                )
            return item

        partial_fields = exact_object(
            value["partial_fields"], set(_CORE_CLAIMS), "partial_fields"
        )
        simple_specs = {
            "document_function": {
                "type_id",
                "description_template",
                "unresolved_description_template",
            },
            "represented_or_governed_subject": {"type_id", "description_template"},
            "primary_relationship": {
                "type_id",
                "source_type_id",
                "target_type_id",
                "direction",
                "description_template",
            },
            "preferred_representation": {"kind", "type_id", "rationale_template"},
            "authority_boundary": {
                "source_role",
                "may_support_screening_or_reasoning",
                "establishes_binding_conclusion",
                "rationale_template",
            },
        }
        for field, keys in simple_specs.items():
            spec = exact_object(partial_fields[field], keys, field)
            for key, item in spec.items():
                if key.endswith("_template"):
                    unresolved_observation_template = (
                        field == "document_function"
                        and key == "unresolved_description_template"
                    )
                    prose(
                        item,
                        f"{field}.{key}",
                        observation=(
                            field in _OBSERVABLE_PARTIAL_CLAIMS
                            and not unresolved_observation_template
                        ),
                        assessment=(
                            field not in _OBSERVABLE_PARTIAL_CLAIMS
                            or unresolved_observation_template
                        ),
                    )
                elif item is not None and key not in {
                    "may_support_screening_or_reasoning",
                    "establishes_binding_conclusion",
                }:
                    cls._machine_id(item, f"{field}.{key}")
            for key in (
                "may_support_screening_or_reasoning",
                "establishes_binding_conclusion",
            ):
                if key in spec and type(spec[key]) is not bool:
                    raise ValueError(
                        f"graph intent partial policy {field}.{key} is invalid"
                    )

        question_sets = {
            "clarification_questions": set(_CORE_CLAIMS),
            "diagnostic_questions": {
                "profile_source_compatibility",
                "proposal_evidence",
                "catalog_admission",
            },
        }
        for field, keys in question_sets.items():
            questions = exact_object(value[field], keys, field)
            for key, raw in questions.items():
                item = exact_object(
                    raw,
                    {"question_id", "question_template", "affected_semantics"},
                    f"{field}.{key}",
                )
                cls._machine_id(item["question_id"], f"{field}.{key}.question_id")
                prose(
                    item["question_template"],
                    f"{field}.{key}.question_template",
                    assessment=True,
                    failure=key in {"proposal_evidence", "catalog_admission"},
                )
                cls._affected_semantics(
                    item["affected_semantics"],
                    f"{field}.{key}.affected_semantics",
                    semantic_categories,
                )

        failure_messages = exact_object(
            value["failure_messages"],
            {
                _LLM_UNAVAILABLE,
                _INVALID_TYPED_PROPOSAL,
                _UNSUPPORTED_MATERIAL_CLAIM,
                _UNRELATED_CLAIM_EVIDENCE,
                _UNKNOWN_CANONICAL_SOURCE_BLOCK,
                _CATALOG_ADMISSION_FAILED,
                _CATALOG_HARD_NEGATIVE_ADMISSION_FAILED,
                _CATALOG_AUTHORITY_ADMISSION_FAILED,
                _FOREIGN_VOCABULARY_REJECTED,
            },
            "failure_messages",
        )
        for key, item in failure_messages.items():
            cls._machine_id(key, f"failure_messages.{key}")
            prose(item, f"failure_messages.{key}")
        fallbacks = exact_object(
            value["fallback_reason_templates"],
            {"auto_pilot_material_ambiguity"},
            "fallback_reason_templates",
        )
        prose(
            fallbacks["auto_pilot_material_ambiguity"],
            "fallback_reason_templates.auto_pilot_material_ambiguity",
        )

    def _typed_runner_proposal(self, llm: dict[str, Any]) -> dict[str, Any] | None:
        if llm["status"] != "passed":
            return None
        output = self._verify_runner_output(llm)
        ontology = output.get("ontology")
        if not isinstance(ontology, dict):
            return None
        proposal = ontology.get("graph_intent_proposal")
        return deepcopy(proposal) if isinstance(proposal, dict) else None

    def _verify_runner_output(self, llm: dict[str, Any]) -> dict[str, Any]:
        output_ref = llm.get("output_ref")
        output_hash = llm.get("output_hash")
        if not output_ref or not output_hash:
            if llm.get("status") == "passed":
                raise ValueError("passed LLM evidence lacks pinned runner output")
            return {}
        target = (self.intent_store.workspace_path / str(output_ref)).resolve()
        if not target.is_relative_to(self.intent_store.artifact_root):
            raise ValueError("runner output ref escapes governed artifact root")
        payload = json.loads(target.read_text(encoding="utf-8"))
        if f"sha256:{content_hash(payload)}" != output_hash:
            raise ValueError("runner output hash mismatch")
        return payload

    @staticmethod
    def _filter_admitted_source_role_ids(
        value: dict[str, Any], *, catalog: dict[str, Any]
    ) -> dict[str, Any]:
        proposal = deepcopy(value)
        roles = proposal.get("source_role_interpretation")
        claim_evidence = proposal.get("claim_evidence")
        admitted = set(map(str, catalog.get("admitted_type_ids") or []))
        if not isinstance(roles, dict) or not isinstance(claim_evidence, dict):
            return proposal
        for field in _ROLE_FIELDS:
            current = roles.get(field)
            if not isinstance(current, list):
                continue
            filtered = [item for item in current if isinstance(item, str) and item in admitted]
            roles[field] = filtered
            if not filtered:
                claim_evidence.pop(f"source_role_interpretation.{field}", None)
        return proposal

    def _validate_typed_proposal(
        self,
        value: dict[str, Any],
        *,
        catalog: dict[str, Any],
        block_by_id: dict[str, Any],
        semantic_categories: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, list[str]]]:
        proposal = deepcopy(value)
        required = {
            *_CORE_CLAIMS, "material_dimensions", "source_role_interpretation",
            "competency_questions", "hard_negative_questions",
            "alternatives_considered", "assumptions", "unresolved_questions",
            "claim_evidence",
        }
        if set(proposal) != required:
            raise ValueError("typed GraphIntent proposal is malformed")
        self._validate_typed_shape(
            proposal, semantic_categories=semantic_categories
        )
        claim_raw = proposal.pop("claim_evidence")
        if not isinstance(claim_raw, dict):
            raise ValueError("typed GraphIntent claim evidence is malformed")
        claim_ids: dict[str, list[str]] = {}
        for claim, ids in claim_raw.items():
            if (
                not isinstance(claim, str)
                or not claim
                or not isinstance(ids, list)
                or not ids
                or not all(isinstance(item, str) and item for item in ids)
            ):
                raise ValueError("typed GraphIntent claim evidence is malformed")
            self._machine_id(claim, "claim_evidence claim ref")
            for index, item in enumerate(ids):
                self._machine_id(item, f"claim_evidence.{claim}[{index}]")
            normalized = list(ids)
            if any(item not in block_by_id for item in normalized):
                raise _ProposalValidationError(_UNKNOWN_CANONICAL_SOURCE_BLOCK)
            claim_ids[claim] = list(dict.fromkeys(normalized))
        required_claims = set(_CORE_CLAIMS)
        required_claims.update(
            f"material_dimensions.{index}"
            for index, _item in enumerate(proposal["material_dimensions"])
        )
        roles = proposal["source_role_interpretation"]
        for field in _ROLE_FIELDS:
            if roles.get(field):
                required_claims.add(f"source_role_interpretation.{field}")
        for field in (
            "competency_questions", "hard_negative_questions",
            "alternatives_considered", "assumptions", "unresolved_questions",
        ):
            required_claims.update(
                f"{field}.{index}" for index, _item in enumerate(proposal[field])
            )
        if not required_claims.issubset(claim_ids):
            raise _ProposalValidationError(_UNSUPPORTED_MATERIAL_CLAIM)
        if not set(claim_ids).issubset(required_claims):
            raise _ProposalValidationError(_UNRELATED_CLAIM_EVIDENCE)
        admitted_types = set(map(str, catalog.get("admitted_type_ids") or []))
        admitted_relations = set(map(str, catalog.get("admitted_relation_ids") or []))
        admitted_kinds = set(
            map(str, catalog.get("admitted_representation_kinds") or [])
        )
        type_ids = {
            proposal["document_function"]["type_id"],
            proposal["represented_or_governed_subject"]["type_id"],
            *proposal["primary_relationship"]["source_types"],
            *proposal["primary_relationship"]["target_types"],
            *[item["type_id"] for item in proposal["material_dimensions"]],
            *[
                item for field in _ROLE_FIELDS
                for item in proposal["source_role_interpretation"].get(field, [])
            ],
        }
        representation_type = proposal["preferred_representation"].get("type_id")
        if representation_type:
            type_ids.add(representation_type)
        if (
            not type_ids.issubset(admitted_types)
            or proposal["primary_relationship"]["type_id"] not in admitted_relations
            or proposal["preferred_representation"]["kind"] not in admitted_kinds
        ):
            raise _ProposalValidationError(_CATALOG_ADMISSION_FAILED)
        negative_fields = ("question_id", "text", "distinction_to_preserve")
        expected_negatives = sorted(
            tuple(str(item[field]) for field in negative_fields)
            for item in catalog.get("hard_negative_questions", [])
        )
        actual_negatives = sorted(
            tuple(str(item[field]) for field in negative_fields)
            for item in proposal["hard_negative_questions"]
        )
        if expected_negatives != actual_negatives:
            raise _ProposalValidationError(_CATALOG_HARD_NEGATIVE_ADMISSION_FAILED)
        constraints = catalog.get("authority_constraints")
        if not isinstance(constraints, dict):
            raise ValueError("registered catalog lacks authority constraints")
        allowed_roles = constraints.get("allowed_source_roles")
        binding_allowed = constraints.get(
            "establishes_binding_conclusion_allowed"
        )
        authority = proposal["authority_boundary"]
        if (
            not isinstance(allowed_roles, list)
            or not allowed_roles
            or not all(isinstance(item, str) and item for item in allowed_roles)
            or not isinstance(binding_allowed, bool)
        ):
            raise ValueError("registered catalog authority constraints are malformed")
        if (
            authority.get("source_role") not in allowed_roles
            or (
                bool(authority.get("establishes_binding_conclusion"))
                and not binding_allowed
            )
        ):
            raise _ProposalValidationError(_CATALOG_AUTHORITY_ADMISSION_FAILED)
        proposal["document_function"] = IntentTypeDescription.from_dict(
            proposal["document_function"], "document_function"
        ).as_dict()
        proposal["represented_or_governed_subject"] = (
            IntentTypeDescription.from_dict(
                proposal["represented_or_governed_subject"],
                "represented_or_governed_subject",
            ).as_dict()
        )
        proposal["primary_relationship"] = PrimaryRelationship.from_dict(
            proposal["primary_relationship"]
        ).as_dict()
        proposal["preferred_representation"] = PreferredRepresentation.from_dict(
            proposal["preferred_representation"]
        ).as_dict()
        proposal["source_role_interpretation"] = (
            SourceRoleInterpretation.from_dict(
                proposal["source_role_interpretation"]
            ).as_dict()
        )
        proposal["authority_boundary"] = AuthorityBoundary.from_dict(
            proposal["authority_boundary"]
        ).as_dict()
        proposal["competency_questions"] = [
            CompetencyQuestion.from_dict(item).as_dict()
            for item in proposal["competency_questions"]
        ]
        proposal["hard_negative_questions"] = [
            HardNegativeQuestion.from_dict(item).as_dict()
            for item in proposal["hard_negative_questions"]
        ]
        proposal["alternatives_considered"] = [
            AlternativeConsidered.from_dict(item).as_dict()
            for item in proposal["alternatives_considered"]
        ]
        if not isinstance(proposal["assumptions"], list) or any(
            not isinstance(item, str) or not item.strip()
            for item in proposal["assumptions"]
        ):
            raise ValueError("typed GraphIntent assumptions are malformed")
        return proposal, claim_ids

    @staticmethod
    def _resolve_evidence_aliases(
        value: dict[str, Any], *, evidence_id_by_alias: dict[str, str]
    ) -> dict[str, Any]:
        proposal = deepcopy(value)
        claim_evidence = proposal.get("claim_evidence")
        if not isinstance(claim_evidence, dict):
            return proposal
        proposal["claim_evidence"] = {
            claim: [
                evidence_id_by_alias.get(item, item) if isinstance(item, str) else item
                for item in ids
            ]
            if isinstance(ids, list)
            else ids
            for claim, ids in claim_evidence.items()
        }
        return proposal

    @staticmethod
    def _validate_typed_shape(
        proposal: dict[str, Any],
        *,
        semantic_categories: dict[str, Any],
    ) -> None:
        def object_with_keys(
            value: Any, keys: set[str], field: str
        ) -> dict[str, Any]:
            if not isinstance(value, dict) or set(value) != keys:
                raise ValueError(f"typed GraphIntent {field} is malformed")
            return value

        def required_string(value: Any, field: str) -> str:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"typed GraphIntent {field} is malformed")
            return value

        def machine_string(value: Any, field: str) -> str:
            return GraphIntentDiscoveryService._machine_id(value, field)

        def list_of_strings(
            value: Any,
            field: str,
            *,
            allow_empty: bool = True,
            machine_ids: bool = False,
        ) -> list[str]:
            if (
                not isinstance(value, list)
                or (not allow_empty and not value)
                or not all(isinstance(item, str) and item.strip() for item in value)
            ):
                raise ValueError(f"typed GraphIntent {field} is malformed")
            if machine_ids:
                for index, item in enumerate(value):
                    machine_string(item, f"{field}[{index}]")
            return value

        def list_of_objects(value: Any, field: str) -> list[dict[str, Any]]:
            if not isinstance(value, list) or not all(
                isinstance(item, dict) for item in value
            ):
                raise ValueError(f"typed GraphIntent {field} is malformed")
            return value

        for field in ("document_function", "represented_or_governed_subject"):
            item = object_with_keys(proposal[field], {"type_id", "description"}, field)
            machine_string(item["type_id"], f"{field}.type_id")
            required_string(item["description"], f"{field}.description")

        relationship_value = proposal["primary_relationship"]
        relationship_fields = {
            "type_id", "source_types", "target_types", "direction",
            "semantic_description",
        }
        if isinstance(relationship_value, dict) and (
            "source_role_refs" in relationship_value
            or "target_role_refs" in relationship_value
        ):
            relationship_fields.update({"source_role_refs", "target_role_refs"})
        relationship = object_with_keys(
            relationship_value,
            relationship_fields,
            "primary_relationship",
        )
        machine_string(relationship["type_id"], "primary_relationship.type_id")
        list_of_strings(
            relationship["source_types"],
            "primary_relationship.source_types",
            allow_empty=False,
            machine_ids=True,
        )
        list_of_strings(
            relationship["target_types"],
            "primary_relationship.target_types",
            allow_empty=False,
            machine_ids=True,
        )
        machine_string(relationship["direction"], "primary_relationship.direction")
        if relationship["direction"] not in {
            "source_to_target", "target_to_source"
        }:
            raise ValueError("typed GraphIntent primary relationship direction is invalid")
        required_string(
            relationship["semantic_description"],
            "primary_relationship.semantic_description",
        )
        if "source_role_refs" in relationship:
            for field in ("source_role_refs", "target_role_refs"):
                list_of_strings(
                    relationship[field],
                    f"primary_relationship.{field}",
                    allow_empty=False,
                    machine_ids=True,
                )

        representation = object_with_keys(
            proposal["preferred_representation"],
            {"kind", "type_id", "rationale"},
            "preferred_representation",
        )
        machine_string(representation["kind"], "preferred_representation.kind")
        if representation["type_id"] is not None:
            machine_string(
                representation["type_id"], "preferred_representation.type_id"
            )
        required_string(
            representation["rationale"], "preferred_representation.rationale"
        )

        for index, item in enumerate(
            list_of_objects(proposal["material_dimensions"], "material_dimensions")
        ):
            item = object_with_keys(
                item, {"type_id", "description"}, f"material_dimensions.{index}"
            )
            machine_string(item["type_id"], f"material_dimensions.{index}.type_id")
            required_string(
                item["description"], f"material_dimensions.{index}.description"
            )

        roles = object_with_keys(
            proposal["source_role_interpretation"],
            set(_ROLE_FIELDS),
            "source_role_interpretation",
        )
        for field in _ROLE_FIELDS:
            list_of_strings(
                roles[field],
                f"source_role_interpretation.{field}",
                machine_ids=True,
            )

        authority = object_with_keys(
            proposal["authority_boundary"],
            {
                "source_role", "may_support_screening_or_reasoning",
                "establishes_binding_conclusion", "rationale",
            },
            "authority_boundary",
        )
        machine_string(authority["source_role"], "authority_boundary.source_role")
        required_string(authority["rationale"], "authority_boundary.rationale")
        for field in (
            "may_support_screening_or_reasoning",
            "establishes_binding_conclusion",
        ):
            if type(authority[field]) is not bool:
                raise ValueError(
                    f"typed GraphIntent authority_boundary.{field} is malformed"
                )

        item_specs = {
            "competency_questions": {"question_id", "text", "success_condition"},
            "hard_negative_questions": {
                "question_id", "text", "distinction_to_preserve",
            },
            "alternatives_considered": {
                "alternative_id", "description", "material_difference",
                "disposition",
            },
        }
        for field, keys in item_specs.items():
            for index, item in enumerate(list_of_objects(proposal[field], field)):
                item = object_with_keys(item, keys, f"{field}.{index}")
                for key in keys:
                    if key in {"question_id", "alternative_id", "disposition"}:
                        machine_string(item[key], f"{field}.{index}.{key}")
                    else:
                        required_string(item[key], f"{field}.{index}.{key}")

        list_of_strings(proposal["assumptions"], "assumptions")
        for index, item in enumerate(
            list_of_objects(proposal["unresolved_questions"], "unresolved_questions")
        ):
            item = object_with_keys(
                item,
                {"question_id", "question", "affected_semantics"},
                f"unresolved_questions.{index}",
            )
            machine_string(
                item["question_id"], f"unresolved_questions.{index}.question_id"
            )
            required_string(item["question"], f"unresolved_questions.{index}.question")
            GraphIntentDiscoveryService._affected_semantics(
                item["affected_semantics"],
                f"unresolved_questions.{index}.affected_semantics",
                semantic_categories,
            )

        if not isinstance(proposal["claim_evidence"], dict):
            raise ValueError("typed GraphIntent claim evidence is malformed")

    def _reject_foreign_vocabulary(
        self,
        value: Any,
        *,
        active_catalog: dict[str, Any],
    ) -> None:
        raw = active_catalog.get("forbidden_vocabulary")
        if (
            not isinstance(raw, list)
            or not all(
                isinstance(item, str) and item and item == item.strip() for item in raw
            )
            or len(set(raw)) != len(raw)
        ):
            raise ValueError("registered catalog forbidden vocabulary is malformed")
        forbidden = frozenset(raw)

        def contains_forbidden(item: Any) -> bool:
            if isinstance(item, str):
                return any(token in item for token in forbidden)
            if isinstance(item, dict):
                return any(
                    contains_forbidden(key) or contains_forbidden(child)
                    for key, child in item.items()
                )
            if isinstance(item, (list, tuple)):
                return any(contains_forbidden(child) for child in item)
            return False

        if contains_forbidden(value):
            raise _ProposalValidationError(_FOREIGN_VOCABULARY_REJECTED)

    @staticmethod
    def _proposal_error_code(exc: BaseException) -> str:
        if isinstance(exc, _ProposalValidationError):
            return exc.code
        return _INVALID_TYPED_PROPOSAL

    @staticmethod
    def _source_partial(
        policy: dict[str, Any],
        block_by_id: dict[str, Any],
        canonical_block_refs: dict[str, str],
        assessment: dict[str, Any],
        assessment_ref: str,
    ) -> tuple[dict[str, Any], dict[str, list[str]], list[dict[str, Any]]]:
        fields = policy["partial_fields"]
        semantic_categories = policy["semantic_categories"]
        assessment_blocks = {
            block["block_id"]: block for block in assessment["inspected_blocks"]
        }
        claim_assessments = assessment["claim_assessments"]

        def rendered(claim: str, template_field: str) -> str:
            claim_assessment = claim_assessments[claim]
            if claim_assessment["status"] == "observed":
                block_id = claim_assessment["observed_block_ids"][0]
                return GraphIntentDiscoveryService._render_policy_template(
                    fields[claim][template_field],
                    block_by_id=block_by_id,
                    observation={
                        "block_id": block_id,
                        "structural_role": assessment_blocks[block_id][
                            "structural_role"
                        ],
                    },
                )
            fallback_template = f"unresolved_{template_field}"
            return GraphIntentDiscoveryService._render_policy_template(
                fields[claim].get(fallback_template, fields[claim][template_field]),
                assessment_ref=assessment_ref,
                reason_code=claim_assessment["reason_code"],
            )

        def claim_refs(claim: str) -> list[str]:
            claim_assessment = claim_assessments[claim]
            if claim_assessment["status"] == "observed":
                return [
                    canonical_block_refs[block_id]
                    for block_id in claim_assessment["observed_block_ids"]
                ]
            return [assessment_ref]

        document = fields["document_function"]
        subject = fields["represented_or_governed_subject"]
        relationship = fields["primary_relationship"]
        representation = fields["preferred_representation"]
        authority = fields["authority_boundary"]
        questions: list[dict[str, Any]] = []
        for claim in _CORE_CLAIMS:
            spec = policy["clarification_questions"][claim]
            questions.append(
                GraphIntentDiscoveryService._question(
                    spec["question_id"],
                    GraphIntentDiscoveryService._render_policy_template(
                        spec["question_template"],
                        assessment_ref=assessment_ref,
                        reason_code=claim_assessments[claim]["reason_code"],
                    ),
                    list(spec["affected_semantics"]),
                    [assessment_ref],
                    semantic_categories,
                )
            )
        return (
            {
                "document_function": {
                    "type_id": document["type_id"],
                    "description": rendered(
                        "document_function", "description_template"
                    ),
                },
                "represented_or_governed_subject": {
                    "type_id": subject["type_id"],
                    "description": rendered(
                        "represented_or_governed_subject", "description_template"
                    ),
                },
                "primary_relationship": {
                    "type_id": relationship["type_id"],
                    "source_types": [relationship["source_type_id"]],
                    "target_types": [relationship["target_type_id"]],
                    "direction": relationship["direction"],
                    "semantic_description": rendered(
                        "primary_relationship", "description_template"
                    ),
                },
                "preferred_representation": {
                    "kind": representation["kind"],
                    "type_id": representation["type_id"],
                    "rationale": rendered(
                        "preferred_representation", "rationale_template"
                    ),
                },
                "material_dimensions": [],
                "source_role_interpretation": {field: [] for field in _ROLE_FIELDS},
                "authority_boundary": {
                    "source_role": authority["source_role"],
                    "may_support_screening_or_reasoning": authority[
                        "may_support_screening_or_reasoning"
                    ],
                    "establishes_binding_conclusion": authority[
                        "establishes_binding_conclusion"
                    ],
                    "rationale": rendered("authority_boundary", "rationale_template"),
                },
                "competency_questions": [], "hard_negative_questions": [],
                "alternatives_considered": [], "assumptions": [],
                "unresolved_questions": [],
            },
            {claim: claim_refs(claim) for claim in _CORE_CLAIMS},
            questions,
        )

    @staticmethod
    def _render_policy_template(
        template: str,
        *,
        block_by_id: dict[str, Any] | None = None,
        observation: dict[str, str] | None = None,
        assessment_ref: str = "",
        reason_code: str = "",
        failure_message: str = "",
        failure_code: str = "",
    ) -> str:
        structural_role = ""
        exact_text = ""
        if observation is not None:
            if block_by_id is None:
                raise ValueError("policy observation rendering requires source blocks")
            block = block_by_id[observation["block_id"]]
            structural_role = observation["structural_role"]
            exact_text = str(block.content)
        return template.format(
            structural_role=structural_role,
            exact_text=exact_text,
            assessment_ref=assessment_ref,
            reason_code=reason_code,
            failure_code=failure_code,
            failure_message=failure_message,
        )

    @staticmethod
    def _failure_message(policy: dict[str, Any], failure_code: str) -> str:
        messages = policy["failure_messages"]
        try:
            message = messages[failure_code]
        except KeyError as exc:
            raise ValueError("graph intent partial policy lacks failure code") from exc
        if not isinstance(message, str) or not message.strip():
            raise ValueError("graph intent partial policy failure message is invalid")
        return message

    @staticmethod
    def _policy_question(
        policy: dict[str, Any],
        key: str,
        assessment: dict[str, Any],
        assessment_ref: str,
        *,
        failure_code: str = "",
    ) -> dict[str, Any]:
        spec = policy["diagnostic_questions"][key]
        claim_key = (
            "profile_source_compatibility"
            if key == "profile_source_compatibility"
            else "proposal_evidence"
        )
        return GraphIntentDiscoveryService._question(
            spec["question_id"],
            GraphIntentDiscoveryService._render_policy_template(
                spec["question_template"],
                assessment_ref=assessment_ref,
                reason_code=assessment["claim_assessments"][claim_key][
                    "reason_code"
                ],
                failure_code=failure_code,
                failure_message=(
                    GraphIntentDiscoveryService._failure_message(
                        policy, failure_code
                    )
                    if failure_code
                    else ""
                ),
            ),
            list(spec["affected_semantics"]),
            [assessment_ref],
            policy["semantic_categories"],
        )

    @staticmethod
    def _question(
        question_id: str,
        text: str,
        affected: list[str],
        evidence_refs: list[str],
        semantic_categories: dict[str, Any],
    ) -> dict[str, Any]:
        GraphIntentDiscoveryService._machine_id(question_id, "question_id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("question text is required")
        normalized_affected = list(
            GraphIntentDiscoveryService._affected_semantics(
                affected, "affected_semantics", semantic_categories
            )
        )
        for index, ref in enumerate(evidence_refs):
            GraphIntentDiscoveryService._machine_id(
                ref, f"unresolved_questions[].evidence_refs[{index}]"
            )
        return {
            "question_id": question_id,
            "question": text,
            "material": GraphIntentDiscoveryService._is_material(
                normalized_affected, semantic_categories
            ),
            "affected_semantics": normalized_affected,
            "evidence_refs": list(evidence_refs),
        }

    @staticmethod
    def _matched_block_ids(catalog: dict[str, Any], blocks: list[Any]) -> list[str]:
        terms = tuple(str(item) for item in catalog.get("source_evidence_terms", []))
        return [
            block.block_id for block in blocks
            if any(term in str(block.content) for term in terms)
        ]

    @staticmethod
    def _structural_context(
        canonical: Any, blocks: list[Any]
    ) -> dict[str, dict[str, Any]]:
        mappings: dict[str, list[dict[str, Any]]] = {
            block.block_id: [] for block in blocks
        }
        for mapping in canonical.source_map.mappings:
            mapping_evidence = {
                "transformation_kind": mapping.transformation.kind,
                "transform_ref": mapping.transformation.transform_ref,
                "source_locators": [
                    {
                        "native_id": locator.native_id,
                        "ordinal_path": list(locator.ordinal_path),
                        "paragraph_id": locator.paragraph_id,
                        "table_id": locator.table_id,
                        "row_id": locator.row_id,
                        "column_id": locator.column_id,
                    }
                    for locator in mapping.source_locators
                ],
            }
            for block_id in mapping.block_ids:
                if block_id in mappings:
                    mappings[block_id].append(deepcopy(mapping_evidence))
        result: dict[str, dict[str, Any]] = {}
        for block in blocks:
            structure = dict(block.structure)
            if block.block_type == "heading":
                role_code = "heading"
                role = "source-mapped heading"
            elif block.block_type == "cell" and structure.get("rowOrdinal") == 1:
                role_code = "table_header_cell"
                role = "source-mapped table header cell"
            elif block.block_type == "cell":
                role_code = "table_body_cell"
                role = "source-mapped table body cell"
            elif block.block_type == "list_item" and block.parent_block_id:
                role_code = "nested_list_item"
                role = "source-mapped nested list item"
            elif block.block_type == "list_item":
                role_code = "root_list_item"
                role = "source-mapped root list item"
            elif block.block_type == "paragraph" and block.parent_block_id:
                role_code = "nested_paragraph"
                role = "source-mapped nested paragraph"
            elif block.block_type == "paragraph":
                role_code = "top_level_paragraph"
                role = "source-mapped top-level paragraph"
            else:
                role_code = str(block.block_type)
                role = f"source-mapped {block.block_type}"
            result[block.block_id] = {
                "role_code": role_code,
                "structural_role": role,
                "parent_block_id": block.parent_block_id,
                "ordinal": block.ordinal,
                "structure": structure,
                "source_mappings": sorted(mappings[block.block_id], key=canonical_json),
            }
        return result

    @classmethod
    def _structure_informed_context(
        cls,
        canonical: Any,
        blocks: list[Any],
        structural_relations: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Experimental projection: enrich canonical block context with explicit relations.

        The relations are evidence only. They do not alter CanonicalSourceIR, create
        semantic graph claims, or participate in admission unless a caller explicitly
        opts into this projection.
        """
        result = cls._structural_context(canonical, blocks)
        known = set(result)
        for block_id in known:
            result[block_id]["structural_relations"] = []
        for relation in structural_relations:
            source = str(relation.get("from", ""))
            target = str(relation.get("to", ""))
            kind = str(relation.get("kind", ""))
            if source not in known or target not in known:
                raise ValueError("structure-informed relation targets unknown canonical block")
            if kind not in {"owns", "header-for", "continues"}:
                raise ValueError("unsupported structure-informed relation")
            scope = str(relation.get("scope", ""))
            if kind == "header-for":
                if scope not in {"table-column", "table-row", "table-region"}:
                    raise ValueError("header-for relation requires a supported scope")
                source_locators = result[source]["source_mappings"]
                target_locators = result[target]["source_mappings"]
                source_positions = {
                    (loc["table_id"], loc["row_id"], loc["column_id"])
                    for mapping in source_locators for loc in mapping["source_locators"]
                }
                target_positions = {
                    (loc["table_id"], loc["row_id"], loc["column_id"])
                    for mapping in target_locators for loc in mapping["source_locators"]
                }
                compatible = any(
                    source_table is not None
                    and source_table == target_table
                    and (
                        scope == "table-region"
                        or (scope == "table-column" and source_column == target_column)
                        or (scope == "table-row" and source_row == target_row)
                    )
                    for source_table, source_row, source_column in source_positions
                    for target_table, target_row, target_column in target_positions
                )
                if not compatible:
                    raise ValueError("header-for relation contradicts canonical table coordinates")
            evidence = {
                "kind": kind,
                "from": source,
                "to": target,
                "basis": str(relation.get("basis", "")),
                **({"scope": scope} if scope else {}),
            }
            result[source]["structural_relations"].append(deepcopy(evidence))
            result[target]["structural_relations"].append(deepcopy(evidence))
        for block_id in known:
            result[block_id]["structural_relations"] = sorted(
                result[block_id]["structural_relations"], key=canonical_json
            )
        return result

    @staticmethod
    def _partial_assessment(
        *,
        source_identity: SourceIdentity,
        blocks: list[Any],
        structural_context: dict[str, dict[str, Any]],
        profile_id: str,
        profile_hash: str,
        catalog_hash: str,
        partial_policy_hash: str,
        reason_codes: dict[str, str],
    ) -> dict[str, Any]:
        ordered_blocks = sorted(blocks, key=lambda block: block.block_id)
        inspected_blocks = [
            GraphIntentDiscoveryService._assessment_block_payload(
                block, structural_context
            )
            for block in ordered_blocks
        ]
        structure_fingerprint = (
            GraphIntentDiscoveryService._assessment_structure_fingerprint(
                inspected_blocks
            )
        )
        all_block_ids = [item["block_id"] for item in inspected_blocks]
        heading_blocks = [
            block for block in inspected_blocks if block["role_code"] == "heading"
        ]
        observed_heading = (
            min(
                heading_blocks,
                key=lambda block: (
                    block["ordinal"]
                    if isinstance(block["ordinal"], int)
                    else 10**9,
                    block["block_id"],
                ),
            )
            if heading_blocks
            else None
        )
        claim_assessments: dict[str, dict[str, Any]] = {}
        if observed_heading is not None:
            claim_assessments["document_function"] = {
                "status": "observed",
                "reason_code": _STRUCTURAL_HEADING_OBSERVED,
                "observed_block_ids": [observed_heading["block_id"]],
                "inspected_block_ids": [observed_heading["block_id"]],
            }
        else:
            claim_assessments["document_function"] = {
                "status": "unresolved",
                "reason_code": "STRUCTURAL_HEADING_UNAVAILABLE",
                "inspected_block_ids": all_block_ids,
            }
        for claim in (
            "represented_or_governed_subject",
            "primary_relationship",
            "preferred_representation",
            "authority_boundary",
        ):
            claim_assessments[claim] = {
                "status": "unresolved",
                "reason_code": _GENERIC_STRUCTURE_INSUFFICIENT,
                "inspected_block_ids": all_block_ids,
            }
        for claim, reason_code in sorted(reason_codes.items()):
            GraphIntentDiscoveryService._machine_id(
                claim, "partial_assessment.reason_claim"
            )
            GraphIntentDiscoveryService._machine_id(
                reason_code, "partial_assessment.reason_code"
            )
            claim_assessments[claim] = {
                "status": "unresolved",
                "reason_code": reason_code,
                "inspected_block_ids": all_block_ids,
            }
        return {
            "schema_version": "graph_intent_partial_assessment.v1",
            "authority_tier": "T6",
            "authoritative": False,
            "source_pins": {
                "source_identity_ref": source_identity.source_identity_ref,
                "source_hash": source_identity.source_hash,
                "canonical_source_structure_hash": (
                    f"sha256:{content_hash(structure_fingerprint)}"
                ),
            },
            "profile_id": profile_id,
            "profile_hash": profile_hash,
            "catalog_hash": catalog_hash,
            "policy_hash": partial_policy_hash,
            "structural_criteria": {
                "document_function": {
                    "observable_roles": ["heading"],
                    "reason_code": _STRUCTURAL_HEADING_OBSERVED,
                },
                "semantic_core_claims": {
                    "unresolved_claims": [
                        "represented_or_governed_subject",
                        "primary_relationship",
                        "preferred_representation",
                        "authority_boundary",
                    ],
                    "reason_code": _GENERIC_STRUCTURE_INSUFFICIENT,
                },
                "diagnostic_claims": sorted(reason_codes),
            },
            "inspected_blocks": inspected_blocks,
            "claim_assessments": claim_assessments,
        }

    @staticmethod
    def _prompt(
        template: str,
        profile_id: str,
        catalog: dict[str, Any],
        blocks: list[Any],
    ) -> str:
        return template.format(
            profile_id=profile_id,
            profile_catalog=canonical_json(catalog),
            canonical_source_blocks=canonical_json(
                [
                    {
                        "block_id": f"block_{index:06d}",
                        "block_type": item.block_type,
                        "content": item.content,
                    }
                    for index, item in enumerate(blocks, start=1)
                ]
            ),
        )

    def _llm_evidence(
        self,
        *,
        provider: str | None,
        model: str | None,
        prompt: str,
        config: dict[str, Any],
        output_schema: dict[str, Any],
        profile_hash: str,
        catalog_hash: str,
        prompt_asset_hash: str,
        prompt_asset_ref: str,
        expected_prompt_asset_hash: str | None,
        expected_config_hash: str | None,
        attempt_ref: str | None = None,
    ) -> dict[str, Any]:
        schema_hash = f"sha256:{content_hash(output_schema)}"
        input_hash = f"sha256:{content_hash(prompt)}"
        config_hash = f"sha256:{content_hash(config)}"
        if (
            expected_prompt_asset_hash is not None
            and expected_prompt_asset_hash != prompt_asset_hash
        ):
            raise ValueError("expected prompt asset hash mismatch")
        if expected_config_hash is not None and expected_config_hash != config_hash:
            raise ValueError("expected proposal config hash mismatch")
        transport_fingerprint = str(
            getattr(self.llm_evidence_runner, "transport_fingerprint", "") or ""
        ).strip()
        base = {
            "authoritative": False, "provider": provider, "model": model,
            "profile_hash": profile_hash, "catalog_hash": catalog_hash,
            "config": config, "config_hash": config_hash,
            "prompt_asset_hash": prompt_asset_hash,
            "prompt_asset_ref": prompt_asset_ref,
            "formatted_input_hash": input_hash,
            "structured_output_schema": deepcopy(output_schema),
            "structured_output_schema_hash": schema_hash,
            "transport_fingerprint": transport_fingerprint,
            **({"attempt_ref": attempt_ref} if attempt_ref is not None else {}),
        }
        if not provider or not model or self.llm_evidence_runner is None:
            return {
                **base, "status": "unavailable",
                "output_ref": None, "output_hash": None,
            }
        request_id = content_hash(
            {
                "profile_hash": profile_hash, "catalog_hash": catalog_hash,
                "prompt_asset_hash": prompt_asset_hash,
                "formatted_input_hash": input_hash,
                "structured_output_schema_hash": schema_hash,
                "provider": provider.lower(), "model": model,
                "config_hash": config_hash,
                "transport_fingerprint": transport_fingerprint,
                **({"attempt_ref": attempt_ref} if attempt_ref is not None else {}),
            }
        )[:24]
        evidence_id = f"gipe_{request_id}"
        try:
            evidence = self.llm_evidence_runner.store.get_llm_proposal_evidence(
                evidence_id
            )
        except KeyError:
            evidence = self.llm_evidence_runner.run(
                provider=provider, model=model, source_text=prompt,
                name="GraphIntentProposalEvidence",
                base_uri="urn:semantier:graph-intent:proposal:",
                evidence_id=evidence_id,
                structured_output_schema=output_schema,
            )
        if evidence["provider"] != provider.lower() or evidence["model"] != model:
            raise ValueError("pinned LLM proposal evidence provider/model mismatch")
        result = {
            **base, "status": evidence["status"],
            "provider": evidence["provider"], "model": evidence["model"],
            "provider_contract_version": evidence["provider_contract_version"],
            "runner_evidence_ref": evidence["evidence_id"],
            "output_ref": evidence.get("proposal_ref"),
            "output_hash": evidence.get("proposal_hash"),
            **({"structured_output_attempts": deepcopy(evidence["diagnostics"]["structured_output_attempts"])}
               if evidence.get("diagnostics", {}).get("structured_output_attempts") else {}),
        }
        self._verify_runner_output(result)
        return result


__all__ = ["GraphIntentDiscoveryService"]
