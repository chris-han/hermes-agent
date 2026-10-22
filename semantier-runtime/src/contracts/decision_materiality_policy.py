"""Server-owned conservative selection policy; no evaluator or budget inputs."""
from dataclasses import dataclass
from hashlib import sha256
import json

_POLICY_JSON = '{"failure":"If complete conservative semantics cannot fit, emit TK_BUDGET_UNSATISFIED without changing policy or membership.","forbidden_inputs":["gold","hard_negative_family","prior_model_output","rescue_regression","case_specific_answer_hint","token_pressure"],"non_material":"Only ContextGraph assertions and evidence outside the exact WorkingContext selected closure may be classified NON_MATERIAL_TO_DECISION_CONTRACT; dependency matches must belong to the retained selected closure.","omission_accounting":"Every full evidence/assertion item absent as a standalone model item has an artifact identity, hash, allowed reason and represented_by where applicable. No TK_BUDGET omission.","omissions":["NON_MATERIAL_TO_DECISION_CONTRACT","DUPLICATE_PROVENANCE_EXPANSION","REPRESENTED_BY_MATERIAL_ASSERTION","INACTIVE_BY_SCOPE_OR_TIME"],"qualifiers":"Preserve selected assertion semantics, condition, exception, qualification, scope, precedence, temporal qualifiers, world/time basis, deterministic constraints and all dependency statuses without alteration.","representation":"Intern repeated node labels and provenance expansion into compact local handles. Keep exact frozen source spans in canonical locator order, with evidence selector and text hashes retained in full replay. Never paraphrase or summarize. A handle resolves only within its retained projection.","schema_version":"decision_materiality_policy.v1","selection":"Include every WorkingContext-selected assertion and every evidence item in its closure. Ambiguous material is included. No lexical relevance filter."}'


@dataclass(frozen=True, slots=True)
class DecisionMaterialityPolicyV1:
    """A fixed versioned policy, not caller-configurable selection authority."""

    @property
    def policy_hash(self) -> str:
        return "sha256:" + sha256(_POLICY_JSON.encode()).hexdigest()

    @property
    def policy_ref(self) -> str:
        return "decision-materiality-policy:v1:" + self.policy_hash[7:]

    def payload(self) -> dict:
        return json.loads(_POLICY_JSON)
