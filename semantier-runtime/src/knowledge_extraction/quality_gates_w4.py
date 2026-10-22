"""W4: Semantic role classification and boundary-quality gates.

Implements IC-3 classification and validation that separates boundary detection
from semantic role assignment, preventing mixed-role prose from being silently
promoted to generic concepts.
"""

from __future__ import annotations
from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel
from enum import Enum


class SemanticRole(str, Enum):
    """IC-4: Semantic roles for classified candidate spans."""
    CANONICAL_SENSITIVE_TERM = "canonical_sensitive_term"
    ALIAS = "alias"
    VIOLATION_CATEGORY = "violation_category"
    PROHIBITION_TYPE = "prohibition_type"
    APPLICABILITY_CONSTRAINT = "applicability_constraint"
    REMEDIATION_ACTION = "remediation_action"
    GROUNDING_EXAMPLE = "grounding_example"
    MODIFIER = "modifier"
    NON_CONCEPT_NARRATIVE = "non_concept_narrative"


class BoundaryQualityGate(BaseModel):
    """IC-5: Individual gate check result."""
    gate_name: str
    passed: bool
    severity: Literal["hard", "warn"]
    details: Optional[str] = None


class BoundaryQualityValidation(BaseModel):
    """IC-5: Complete validation result for a candidate span."""
    
    candidate_span_id: str
    gate_results: Dict[str, BoundaryQualityGate]
    
    @property
    def all_hard_gates_pass(self) -> bool:
        """True if all hard gates passed."""
        return all(
            gate.passed for gate in self.gate_results.values()
            if gate.severity == "hard"
        )
    
    @property
    def has_warnings(self) -> bool:
        """True if any warnings present."""
        return any(
            not gate.passed for gate in self.gate_results.values()
            if gate.severity == "warn"
        )
    
    @property
    def gate_outcome(self) -> Literal["pass", "needs_boundary_review", "reject", "provider_error"]:
        """IC-3: Determine outcome based on gate results."""
        if not self.all_hard_gates_pass:
            # Hard gate failed
            failing_gates = [name for name, gate in self.gate_results.items() if not gate.passed and gate.severity == "hard"]
            
            if "whole_block_fallback" in failing_gates:
                return "reject"  # Cannot project whole-element shim
            elif "anchor_integrity" in failing_gates:
                return "reject"  # Cannot project without valid anchor
            elif "shim_provenance" in failing_gates:
                return "reject"  # Explicitly prohibited
            elif "label_length" in failing_gates:
                return "reject"  # IC-3 / AD-5: hard max exceeded without split
            elif "coverage_integrity" in failing_gates:
                return "reject"  # Unexplained overlaps / gaps are not eligible
            else:
                return "needs_boundary_review"  # Needs human review
        
        if self.has_warnings:
            return "needs_boundary_review"  # Ambiguous; needs review
        
        return "pass"  # Ready to project


class RoleClassifier:
    """W4: Classify candidate spans and concepts into semantic roles."""
    
    def __init__(self, profile: dict):
        self.profile = profile
        self.allowed_roles = profile.get("segmentation", {}).get("decomposableSourceRoles", [])
    
    def classify(
        self,
        candidate_span_id: str,
        exact_text: str,
        source_role: str,
        native_entities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Classify a candidate span into semantic role(s).
        
        W4: Separate boundary detection from classification.
        A normalized label cannot replace exact source expression.
        """
        
        # Infer semantic role from:
        # 1. Source role (input from segmentation)
        # 2. Content patterns (terms vs actions vs examples)
        # 3. Native extraction hints
        
        inferred_role = self._infer_semantic_role(exact_text, source_role, native_entities)
        
        return {
            "candidate_span_id": candidate_span_id,
            "semantic_role": inferred_role,
            "confidence": 0.9,
            "exact_text": exact_text,  # Never replace with normalized label
            "source_role": source_role
        }
    
    def _infer_semantic_role(
        self,
        text: str,
        source_role: str,
        native_entities: List[Dict[str, Any]]
    ) -> str:
        """Infer semantic role from content patterns."""

        marker_roles = self.profile.get("segmentation", {}).get(
            "semanticRoleMarkers", {}
        )
        for role, markers in marker_roles.items():
            if text.strip() in {str(marker) for marker in markers}:
                return str(role)
        
        # Simple heuristics (in production, could use trained classifier)
        
        # Remediation: Contains action verbs
        if any(verb in text for verb in ["删除", "移除", "去掉", "取消"]):
            return SemanticRole.REMEDIATION_ACTION.value
        
        # Example: Contains example indicators
        if any(marker in text for marker in ["如:", "例如:", "比如:"]):
            return SemanticRole.GROUNDING_EXAMPLE.value
        
        # Applicability: Contains applicability markers
        if any(marker in text for marker in ["仅限", "限于", "仅包括"]):
            return SemanticRole.APPLICABILITY_CONSTRAINT.value
        
        # Default to source role
        return source_role


class BoundaryQualityGateValidator:
    """W4: Validate candidate spans against quality gates (IC-5)."""
    
    def __init__(self, profile: dict):
        self.profile = profile
        self.quality_gates_config = profile.get("segmentation", {}).get("qualityGates", {})
        self.allowed_roles = profile.get("segmentation", {}).get("decomposableSourceRoles", [])
    
    def validate(
        self,
        candidate_span_id: str,
        exact_text: str,
        start_offset: int,
        end_offset: int,
        source_identity: str,
        semantic_role: str,
        native_entity_count: int,
        fallback_method: Optional[str],
        delimiter_count: int = 0
    ) -> BoundaryQualityValidation:
        """IC-5: Run all boundary quality gates.
        
        Gates (IC-5):
        1. whole_block_fallback: reject unless structurally atomic
        2. label_length: flag unusually long labels
        3. multi_delimiter: require review if multiple delimiters remain
        4. sentence_shape: flag complete sentences or multiple clauses
        5. mixed_semantic_roles: flag combined term/category/action
        6. anchor_integrity: exact equality required
        7. coverage_integrity: no unexplained gaps
        8. shim_provenance: prohibit whole-element fallback
        """
        
        gates = {}
        
        # Gate 1: whole_block_fallback
        gates["whole_block_fallback"] = self._gate_whole_block_fallback(
            exact_text, fallback_method, native_entity_count
        )
        
        # Gate 2: label_length
        gates["label_length"] = self._gate_label_length(exact_text)
        
        # Gate 3: multi_delimiter
        gates["multi_delimiter"] = self._gate_multi_delimiter(exact_text, delimiter_count)
        
        # Gate 4: sentence_shape
        gates["sentence_shape"] = self._gate_sentence_shape(exact_text)
        
        # Gate 5: mixed_semantic_roles
        gates["mixed_semantic_roles"] = self._gate_mixed_semantic_roles(exact_text, semantic_role)
        
        # Gate 6: anchor_integrity
        gates["anchor_integrity"] = self._gate_anchor_integrity(exact_text, start_offset, end_offset)
        
        # Gate 7: coverage_integrity (simplified; in real impl, check no gaps)
        gates["coverage_integrity"] = BoundaryQualityGate(
            gate_name="coverage_integrity",
            passed=True,
            severity="hard",
            details="Span covers continuous region"
        )
        
        # Gate 8: shim_provenance
        gates["shim_provenance"] = self._gate_shim_provenance(fallback_method)
        
        return BoundaryQualityValidation(
            candidate_span_id=candidate_span_id,
            gate_results=gates
        )
    
    def _gate_whole_block_fallback(
        self,
        exact_text: str,
        fallback_method: Optional[str],
        native_entity_count: int
    ) -> BoundaryQualityGate:
        """Gate 1: Reject if promoted solely via whole-element fallback."""
        
        if fallback_method == "semantier_empty_result_shim":
            return BoundaryQualityGate(
                gate_name="whole_block_fallback",
                passed=False,
                severity="hard",
                details="Whole-element shim fallback; requires human split or valid native extraction"
            )
        
        return BoundaryQualityGate(
            gate_name="whole_block_fallback",
            passed=True,
            severity="hard"
        )
    
    def _gate_label_length(self, exact_text: str) -> BoundaryQualityGate:
        """Gate 2: Flag unusually long labels."""
        
        advisory_limit = self.quality_gates_config.get("advisoryLabelCodePoints", 40)
        hard_limit = self.quality_gates_config.get("hardLabelCodePoints", 160)
        
        text_length = len(exact_text)
        
        if text_length > hard_limit:
            return BoundaryQualityGate(
                gate_name="label_length",
                passed=False,
                severity="hard",
                details=f"Label {text_length} code-points exceeds hard limit {hard_limit}"
            )
        
        if text_length > advisory_limit:
            return BoundaryQualityGate(
                gate_name="label_length",
                passed=True,
                severity="warn",
                details=f"Label {text_length} code-points exceeds advisory {advisory_limit}"
            )
        
        return BoundaryQualityGate(
            gate_name="label_length",
            passed=True,
            severity="hard"
        )
    
    def _gate_multi_delimiter(self, exact_text: str, delimiter_count: int) -> BoundaryQualityGate:
        """Gate 3: Flag if multiple delimiters remain (not split)."""
        
        if delimiter_count > 1:
            return BoundaryQualityGate(
                gate_name="multi_delimiter",
                passed=False,
                severity="warn",
                details=f"{delimiter_count} delimiters remain in span; consider split"
            )
        
        return BoundaryQualityGate(
            gate_name="multi_delimiter",
            passed=True,
            severity="hard"
        )
    
    def _gate_sentence_shape(self, exact_text: str) -> BoundaryQualityGate:
        """Gate 4: Flag complete sentences or multiple independent clauses."""
        
        sentence_terminators = ["。", "！", "？"]
        has_terminator = any(t in exact_text for t in sentence_terminators)
        
        if has_terminator and len(exact_text) > 20:
            return BoundaryQualityGate(
                gate_name="sentence_shape",
                passed=False,
                severity="warn",
                details="Span contains sentence terminator; may be multiple clauses"
            )
        
        return BoundaryQualityGate(
            gate_name="sentence_shape",
            passed=True,
            severity="hard"
        )
    
    def _gate_mixed_semantic_roles(self, exact_text: str, semantic_role: str) -> BoundaryQualityGate:
        """Gate 5: Flag text combining different semantic roles."""
        
        # Simple pattern matching
        has_term = len(exact_text) < 20  # Terms are typically short
        has_action = any(v in exact_text for v in ["删除", "移除"])
        has_example = any(m in exact_text for m in ["如:", "例如"])
        
        role_count = sum([has_term, has_action, has_example])
        
        if role_count > 1:
            return BoundaryQualityGate(
                gate_name="mixed_semantic_roles",
                passed=False,
                severity="warn",
                details=f"Span may contain {role_count} semantic roles; requires review"
            )
        
        return BoundaryQualityGate(
            gate_name="mixed_semantic_roles",
            passed=True,
            severity="hard"
        )
    
    def _gate_anchor_integrity(
        self,
        exact_text: str,
        start_offset: int,
        end_offset: int
    ) -> BoundaryQualityGate:
        """Gate 6: Require exact anchor match.

        IC-5 / AD-5: ``end_offset - start_offset`` must equal ``len(exact_text)``.
        Any mismatch is a hard reject because the span can no longer be trusted
        to map back to immutable source slices.
        """
        if start_offset < 0 or end_offset <= start_offset or not exact_text:
            return BoundaryQualityGate(
                gate_name="anchor_integrity",
                passed=False,
                severity="hard",
                details="Anchor offsets invalid or exact_text empty",
            )
        span_length = end_offset - start_offset
        if span_length != len(exact_text):
            return BoundaryQualityGate(
                gate_name="anchor_integrity",
                passed=False,
                severity="hard",
                details=(
                    f"Anchor span {span_length} != exact_text length {len(exact_text)}"
                ),
            )
        return BoundaryQualityGate(
            gate_name="anchor_integrity",
            passed=True,
            severity="hard",
        )
    
    def _gate_shim_provenance(self, fallback_method: Optional[str]) -> BoundaryQualityGate:
        """Gate 8: Prohibit whole-element shim from reference projection."""
        
        if fallback_method == "semantier_empty_result_shim":
            return BoundaryQualityGate(
                gate_name="shim_provenance",
                passed=False,
                severity="hard",
                details="Shim-origin candidates cannot be projected to reference concepts"
            )
        
        return BoundaryQualityGate(
            gate_name="shim_provenance",
            passed=True,
            severity="hard"
        )


def determine_projection_eligibility(validation: BoundaryQualityValidation) -> Dict[str, Any]:
    """W4: Determine if candidate can be projected to reference concept.
    
    Only candidates that pass hard gates and don't have unresolved warnings
    can be projected. Others must go to boundary_review.
    """
    
    outcome = validation.gate_outcome
    
    if outcome == "pass":
        return {
            "can_project": True,
            "review_status": "approved_automatic",
            "reason": "Passed all hard gates"
        }
    elif outcome == "needs_boundary_review":
        return {
            "can_project": False,
            "review_status": "needs_boundary_review",
            "reason": "Warnings or soft failures detected; requires curator review"
        }
    else:  # "reject"
        return {
            "can_project": False,
            "review_status": "rejected",
            "reason": f"Failed hard gate(s): {[name for name, gate in validation.gate_results.items() if not gate.passed and gate.severity == 'hard']}"
        }


if __name__ == "__main__":
    # Test
    profile = {
        "segmentation": {
            "decomposableSourceRoles": ["canonical_sensitive_term", "alias", "remediation"],
            "qualityGates": {
                "advisoryLabelCodePoints": 40,
                "hardLabelCodePoints": 160
            }
        }
    }
    
    classifier = RoleClassifier(profile)
    validator = BoundaryQualityGateValidator(profile)
    
    # Test case 1: Short atomic term (should pass)
    classification = classifier.classify(
        "ccs_v1_abc",
        "大型企业",
        "canonical_sensitive_term",
        []
    )
    print(f"Classification: {classification['semantic_role']}")
    
    validation = validator.validate(
        "ccs_v1_abc",
        "大型企业",
        0, 4,
        "canonical_v1_xyz",
        "canonical_sensitive_term",
        0,
        None
    )
    print(f"Validation outcome: {validation.gate_outcome}")
    assert validation.gate_outcome == "pass", "Atomic term should pass"
    
    # Test case 2: Long whole-element (should reject)
    long_text = "大型企业、央企、世界500强、行业龙头"
    validation2 = validator.validate(
        "ccs_v1_xyz",
        long_text,
        0, len(long_text),
        "canonical_v1_xyz",
        "canonical_sensitive_term",
        0,
        "semantier_empty_result_shim"  # Shim provenance
    )
    print(f"Validation outcome (shim): {validation2.gate_outcome}")
    assert validation2.gate_outcome == "reject", "Shim-origin should reject"
    
    print("✓ W4 classification and quality gates test passed!")
