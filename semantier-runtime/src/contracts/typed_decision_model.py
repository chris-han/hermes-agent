"""Provider-neutral contracts for bounded post-DLP decision scoring.

These artifacts are derived non-authority research/runtime projections. They bind an
already-governed DecisionLocalProjectionV2 to the exact typed candidate space from
DecisionContractV1. They do not create admissibility, authority, or findings.
"""
from __future__ import annotations

import math
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.decision_local_projection import SemanticCandidateV1
from eos.artifact_hashing import content_hash

_SHA = r"^sha256:[0-9a-f]{64}$"
_EXACT_TENDER_CANDIDATES = (
    SemanticCandidateV1(candidate_id="A", semantic_value="TRUE"),
    SemanticCandidateV1(candidate_id="B", semantic_value="FALSE"),
    SemanticCandidateV1(candidate_id="C", semantic_value="UNKNOWN"),
)


def _digest(value) -> str:
    return "sha256:" + content_hash(value)


_FRONTIER_POLICY = {
    "policy": "decision-contract-derived-typed-frontier",
    "version": "v1",
    "candidate_semantics": "reuse exact SemanticPredicateSpecV1 candidates",
    "tender_unknown_semantics": "C=UNKNOWN is the exhaustive unresolved state; no synthetic NONE_OF_THE_ABOVE",
}
FRONTIER_POLICY_HASH = _digest(_FRONTIER_POLICY)
FRONTIER_POLICY_REF = "typed-decision-frontier-policy:v1:" + FRONTIER_POLICY_HASH[7:]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TypedDecisionQuestionV1(_Frozen):
    predicate_id: str = Field(min_length=1)
    semantic_type_ref: str = Field(min_length=1)
    family_ref: str = Field(min_length=1)
    candidates: tuple[SemanticCandidateV1, ...] = _EXACT_TENDER_CANDIDATES

    @model_validator(mode="after")
    def exact_tender_candidates(self):
        if self.candidates != _EXACT_TENDER_CANDIDATES:
            raise ValueError("typed decision question candidates must be exact A=TRUE/B=FALSE/C=UNKNOWN")
        return self


class TypedDecisionFrontierV1(_Frozen):
    """Content-addressed typed question frontier derived from one exact DLP."""

    schema_version: Literal["typed_decision_frontier.v1"] = "typed_decision_frontier.v1"
    frontier_ref: str
    frontier_hash: str = Field(pattern=_SHA)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    dlp_ref: str = Field(min_length=1)
    dlp_hash: str = Field(pattern=_SHA)
    dlp_model_input_hash: str = Field(pattern=_SHA)
    decision_contract_ref: str = Field(min_length=1)
    decision_contract_hash: str = Field(pattern=_SHA)
    questions: tuple[TypedDecisionQuestionV1, ...] = Field(min_length=1)
    frontier_policy_ref: str = FRONTIER_POLICY_REF
    frontier_policy_hash: str = FRONTIER_POLICY_HASH

    @model_validator(mode="after")
    def verify_identity(self):
        if (self.frontier_policy_ref, self.frontier_policy_hash) != (
            FRONTIER_POLICY_REF,
            FRONTIER_POLICY_HASH,
        ):
            raise ValueError("untrusted typed decision frontier policy")
        ids = tuple(item.predicate_id for item in self.questions)
        if len(ids) != len(set(ids)):
            raise ValueError("typed decision frontier predicate IDs must be unique")
        if self.questions != tuple(sorted(self.questions, key=lambda item: item.predicate_id.encode("utf-8"))):
            raise ValueError("typed decision frontier questions must be in canonical predicate order")
        digest = _digest(self.hash_payload())
        if (self.frontier_ref, self.frontier_hash) != (
            "typed-decision-frontier:v1:" + digest[7:],
            digest,
        ):
            raise ValueError("typed decision frontier identity mismatch")
        return self

    def hash_payload(self) -> dict:
        return self.model_dump(mode="json", exclude={"frontier_ref", "frontier_hash"})

    @classmethod
    def from_projection(cls, projection) -> "TypedDecisionFrontierV1":
        payload = projection.model_input_payload()
        if projection.model_input_hash != _digest(payload):
            raise ValueError("DLP model-input hash mismatch")
        contract = projection.decision_contract
        questions = tuple(
            TypedDecisionQuestionV1(
                predicate_id=item.predicate_id,
                semantic_type_ref=item.semantic_type_ref,
                family_ref=item.family_ref,
                candidates=item.candidates,
            )
            for item in contract.semantic_predicates
        )
        questions = tuple(sorted(questions, key=lambda item: item.predicate_id.encode("utf-8")))
        body = {
            "schema_version": "typed_decision_frontier.v1",
            "organization_id": projection.legacy.organization_id,
            "workspace_id": projection.legacy.workspace_id,
            "dlp_ref": projection.projection_ref,
            "dlp_hash": projection.projection_hash,
            "dlp_model_input_hash": projection.model_input_hash,
            "decision_contract_ref": contract.contract_ref,
            "decision_contract_hash": contract.contract_hash,
            "questions": questions,
            "frontier_policy_ref": FRONTIER_POLICY_REF,
            "frontier_policy_hash": FRONTIER_POLICY_HASH,
        }
        provisional = cls.model_construct(
            frontier_ref="typed-decision-frontier:v1:pending",
            frontier_hash="sha256:" + "0" * 64,
            **body,
        )
        digest = _digest(provisional.hash_payload())
        return cls.model_validate(
            {
                **body,
                "frontier_ref": "typed-decision-frontier:v1:" + digest[7:],
                "frontier_hash": digest,
            }
        )

    def verify_against_projection(self, projection) -> None:
        expected = type(self).from_projection(projection)
        if self != expected:
            raise ValueError("typed decision frontier does not match exact DLP/DecisionContract")


class TypedDecisionModelInputV1(_Frozen):
    schema_version: Literal["typed_decision_model_input.v1"] = "typed_decision_model_input.v1"
    input_ref: str
    input_hash: str = Field(pattern=_SHA)
    dlp_ref: str = Field(min_length=1)
    dlp_hash: str = Field(pattern=_SHA)
    dlp_model_input_hash: str = Field(pattern=_SHA)
    dlp_model_input: dict
    typed_decision_frontier: TypedDecisionFrontierV1
    scoring_protocol: Literal["closed_candidate_sequence_score.v1"] = "closed_candidate_sequence_score.v1"

    @model_validator(mode="after")
    def verify_identity(self):
        if _digest(self.dlp_model_input) != self.dlp_model_input_hash:
            raise ValueError("typed decision input DLP payload hash mismatch")
        if (
            self.typed_decision_frontier.dlp_ref,
            self.typed_decision_frontier.dlp_hash,
            self.typed_decision_frontier.dlp_model_input_hash,
        ) != (self.dlp_ref, self.dlp_hash, self.dlp_model_input_hash):
            raise ValueError("typed decision input frontier/DLP binding mismatch")
        digest = _digest(self.hash_payload())
        if (self.input_ref, self.input_hash) != (
            "typed-decision-model-input:v1:" + digest[7:],
            digest,
        ):
            raise ValueError("typed decision model input identity mismatch")
        return self

    def hash_payload(self) -> dict:
        return self.model_dump(mode="json", exclude={"input_ref", "input_hash"})

    @classmethod
    def from_projection(
        cls, projection, frontier: TypedDecisionFrontierV1
    ) -> "TypedDecisionModelInputV1":
        frontier.verify_against_projection(projection)
        payload = projection.model_input_payload()
        body = {
            "schema_version": "typed_decision_model_input.v1",
            "dlp_ref": projection.projection_ref,
            "dlp_hash": projection.projection_hash,
            "dlp_model_input_hash": projection.model_input_hash,
            "dlp_model_input": payload,
            "typed_decision_frontier": frontier,
            "scoring_protocol": "closed_candidate_sequence_score.v1",
        }
        provisional = cls.model_construct(
            input_ref="typed-decision-model-input:v1:pending",
            input_hash="sha256:" + "0" * 64,
            **body,
        )
        digest = _digest(provisional.hash_payload())
        return cls.model_validate(
            {
                **body,
                "input_ref": "typed-decision-model-input:v1:" + digest[7:],
                "input_hash": digest,
            }
        )


ScoreSemantics = Literal["probability", "logit", "unnormalized_score", "choice_only"]


class CandidateScoreV1(_Frozen):
    candidate_id: Literal["A", "B", "C"]
    score: float

    @model_validator(mode="after")
    def finite_score(self):
        if not math.isfinite(self.score):
            raise ValueError("candidate score must be finite")
        return self


class QuestionDecisionResultV1(_Frozen):
    predicate_id: str = Field(min_length=1)
    selected_candidate_id: Literal["A", "B", "C"]
    score_semantics: ScoreSemantics
    candidate_scores: tuple[CandidateScoreV1, ...] = ()

    @model_validator(mode="after")
    def score_shape(self):
        ids = tuple(item.candidate_id for item in self.candidate_scores)
        if self.score_semantics == "choice_only":
            if self.candidate_scores:
                raise ValueError("choice-only result must not fabricate candidate scores")
            return self
        if ids != ("A", "B", "C"):
            raise ValueError("scored result must preserve canonical A/B/C candidate order")
        if self.score_semantics == "probability":
            if any(item.score < 0.0 or item.score > 1.0 for item in self.candidate_scores):
                raise ValueError("probability score must be in [0,1]")
            if abs(sum(item.score for item in self.candidate_scores) - 1.0) > 1e-6:
                raise ValueError("probability scores must sum to one")
        return self


class DecisionDistributionV1(_Frozen):
    """Successful model evidence over the frozen typed frontier.

    Transport/model failures produce no DecisionDistributionV1 and therefore cannot
    be silently converted into the semantic UNKNOWN candidate.
    """

    schema_version: Literal["decision_distribution.v1"] = "decision_distribution.v1"
    distribution_ref: str
    distribution_hash: str = Field(pattern=_SHA)
    input_ref: str = Field(min_length=1)
    input_hash: str = Field(pattern=_SHA)
    producer_ref: str = Field(min_length=1)
    producer_hash: str = Field(pattern=_SHA)
    producer_config_ref: str = Field(min_length=1)
    producer_config_hash: str = Field(pattern=_SHA)
    question_results: tuple[QuestionDecisionResultV1, ...] = Field(min_length=1)
    raw_result_ref: str = Field(min_length=1)
    raw_result_hash: str = Field(pattern=_SHA)

    @model_validator(mode="after")
    def verify_identity(self):
        ids = tuple(item.predicate_id for item in self.question_results)
        if len(ids) != len(set(ids)):
            raise ValueError("decision distribution predicate IDs must be unique")
        if self.question_results != tuple(
            sorted(self.question_results, key=lambda item: item.predicate_id.encode("utf-8"))
        ):
            raise ValueError("decision distribution results must be in canonical predicate order")
        digest = _digest(self.hash_payload())
        if (self.distribution_ref, self.distribution_hash) != (
            "decision-distribution:v1:" + digest[7:],
            digest,
        ):
            raise ValueError("decision distribution identity mismatch")
        return self

    def hash_payload(self) -> dict:
        return self.model_dump(mode="json", exclude={"distribution_ref", "distribution_hash"})

    @classmethod
    def create(
        cls,
        *,
        model_input: TypedDecisionModelInputV1,
        producer_ref: str,
        producer_hash: str,
        producer_config_ref: str,
        producer_config_hash: str,
        question_results: tuple[QuestionDecisionResultV1, ...],
        raw_result_ref: str,
        raw_result_hash: str,
    ) -> "DecisionDistributionV1":
        ordered = tuple(sorted(question_results, key=lambda item: item.predicate_id.encode("utf-8")))
        body = {
            "schema_version": "decision_distribution.v1",
            "input_ref": model_input.input_ref,
            "input_hash": model_input.input_hash,
            "producer_ref": producer_ref,
            "producer_hash": producer_hash,
            "producer_config_ref": producer_config_ref,
            "producer_config_hash": producer_config_hash,
            "question_results": ordered,
            "raw_result_ref": raw_result_ref,
            "raw_result_hash": raw_result_hash,
        }
        provisional = cls.model_construct(
            distribution_ref="decision-distribution:v1:pending",
            distribution_hash="sha256:" + "0" * 64,
            **body,
        )
        digest = _digest(provisional.hash_payload())
        result = cls.model_validate(
            {
                **body,
                "distribution_ref": "decision-distribution:v1:" + digest[7:],
                "distribution_hash": digest,
            }
        )
        result.verify_against(model_input)
        return result

    def verify_against(self, model_input: TypedDecisionModelInputV1) -> None:
        if (self.input_ref, self.input_hash) != (model_input.input_ref, model_input.input_hash):
            raise ValueError("decision distribution input binding mismatch")
        expected = {
            question.predicate_id: tuple(item.candidate_id for item in question.candidates)
            for question in model_input.typed_decision_frontier.questions
        }
        actual = {item.predicate_id: item for item in self.question_results}
        if set(actual) != set(expected):
            raise ValueError("decision distribution predicate coverage mismatch")
        for predicate_id, candidate_ids in expected.items():
            item = actual[predicate_id]
            if item.selected_candidate_id not in candidate_ids:
                raise ValueError("decision distribution selected candidate is outside frontier")
            if item.score_semantics != "choice_only":
                if tuple(score.candidate_id for score in item.candidate_scores) != candidate_ids:
                    raise ValueError("decision distribution candidate scores are outside frontier")


__all__ = [
    "FRONTIER_POLICY_HASH",
    "FRONTIER_POLICY_REF",
    "TypedDecisionQuestionV1",
    "TypedDecisionFrontierV1",
    "TypedDecisionModelInputV1",
    "CandidateScoreV1",
    "QuestionDecisionResultV1",
    "DecisionDistributionV1",
]
