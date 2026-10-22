"""W3: Refactored Semantica adapter - removes whole-element shim, adds explicit fallback policy.

Implements IC-3: Run Semantica extraction against candidate spans instead of
unsplit blocks. Never automatically promote whole-element compatibility output.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass
from datetime import datetime
from pydantic import BaseModel
import json


@dataclass
class ExtractionRequest:
    """Request to extract concepts from a candidate span."""
    extraction_run_id: str
    candidate_span_id: str
    source_identity: str
    exact_text: str
    profile_version: str
    provider: Literal["in_process_semantica", "legacy_regex"]


@dataclass
class ExtractionResponse:
    """Response from Semantica extraction provider."""
    extraction_run_id: str
    candidate_span_id: str
    provider: str
    provider_version: str
    native_entities: List[Dict[str, Any]]  # What Semantica actually returned
    empty_native_result: bool  # True if provider returned no entities
    fallback_reason: Optional[str] = None
    fallback_method: Optional[str] = None


class InProcessSemanticaAdapter:
    """IC-3: Refactored adapter - never creates whole-element shim."""
    
    def __init__(self, provider_version: str = "pinned_v1"):
        self.provider_version = provider_version
    
    def extract(self, request: ExtractionRequest) -> ExtractionResponse:
        """Extract from a candidate span.
        
        IC-3 Contract:
        - Run native extraction against the exact candidate span text
        - Preserve empty native result as a proposal requiring grounding
        - NEVER promote text[0:len(text)] just because extraction is empty
        - Separate native output, segmentation diagnostics, and fallback policies
        """
        
        # Step 1: Run native Semantica extraction
        native_entities = self._call_native_provider(request.exact_text)
        
        # Step 2: Check result
        empty_native_result = len(native_entities) == 0
        
        # Step 3: Handle fallback (IC-3 explicit policy)
        fallback_reason = None
        fallback_method = None
        
        if empty_native_result:
            # IC-3: Never create whole-element shim for reference_graph_build
            # Instead: preserve empty result as a boundary proposal requiring grounding
            fallback_reason = "semantica_no_entities"
            fallback_method = None  # No fallback promotion
        
        response = ExtractionResponse(
            extraction_run_id=request.extraction_run_id,
            candidate_span_id=request.candidate_span_id,
            provider="in_process_semantica",
            provider_version=self.provider_version,
            native_entities=native_entities,
            empty_native_result=empty_native_result,
            fallback_reason=fallback_reason,
            fallback_method=fallback_method
        )
        
        return response
    
    def _call_native_provider(self, text: str) -> List[Dict[str, Any]]:
        """Call the pinned Semantica provider.
        
        IC-3: Provider failure must not silently fall back to another provider.
        """
        try:
            # TODO: Call actual Semantica in-process extractor
            # For now, stub that returns empty list for Chinese text
            if any(ord(c) >= 0x4E00 and ord(c) <= 0x9FFF for c in text):
                # Chinese text; current implementation returns no entities
                return []
            else:
                # Non-Chinese; would call native regex
                return []
        except Exception as e:
            # IC-3: Record error; never silently retry with another provider
            raise RuntimeError(f"Semantica provider error: {e}")


class FallbackPolicy(BaseModel):
    """IC-3: Explicit fallback policy by workflow."""
    
    workflow: Literal["reference_graph_build", "legacy_candidate_assertion"]
    fallback_enabled: bool
    fallback_method: Optional[Literal["whole_element_shim", "none"]]
    legacy_flag: Optional[str] = None  # e.g., "legacy_whole_element_fallback=true"


class ReferenceGraphBuildFallback(FallbackPolicy):
    """IC-3: Fallback policy for reference_graph_build (no shim!)."""
    
    workflow: Literal["reference_graph_build"] = "reference_graph_build"
    fallback_enabled: bool = False  # IC-3: Never fallback for new references
    fallback_method: Literal["none"] = "none"
    
    def apply_fallback(self, result: ExtractionResponse) -> Dict[str, Any]:
        """IC-3: Fallback policy for reference_graph_build.
        
        - Never promote whole-element shim
        - Preserve empty result as boundary proposal
        - Record diagnostics for review
        """
        
        if result.empty_native_result:
            # IC-3: Empty result becomes a boundary proposal, NOT a concept
            return {
                "outcome": "needs_boundary_review",
                "reason": "empty_native_extraction",
                "can_project_as_concept": False,
                "requires_human_grounding": True,
                "diagnostics": {
                    "fallback_reason": result.fallback_reason,
                    "native_entity_count": 0,
                    "candidate_span_id": result.candidate_span_id
                }
            }
        else:
            # Native extraction succeeded
            return {
                "outcome": "native_extraction_success",
                "entities": result.native_entities,
                "can_project_as_concept": True,
                "requires_human_grounding": False
            }


class LegacyCandidateAssertionFallback(FallbackPolicy):
    """IC-3: Backward compatibility shim for legacy_candidate_assertion (temporary)."""
    
    workflow: Literal["legacy_candidate_assertion"] = "legacy_candidate_assertion"
    fallback_enabled: bool = True  # Temporary, for migration
    fallback_method: Literal["whole_element_shim"] = "whole_element_shim"
    legacy_flag: str = "legacy_whole_element_fallback=true"
    
    def apply_fallback(self, result: ExtractionResponse) -> Dict[str, Any]:
        """IC-3: Legacy fallback for backward compatibility (deprecated).
        
        IC-7: Remove after two verified release cycles.
        """
        
        if result.empty_native_result:
            # Legacy: Allow whole-element promotion ONLY for legacy workflow
            return {
                "outcome": "legacy_fallback_whole_element",
                "legacy_flag": self.legacy_flag,
                "can_project_as_concept": False,  # IC-3: Prohibit reference projection
                "requires_human_grounding": True,
                "diagnostics": {
                    "fallback_method": "semantier_empty_result_shim",
                    "deprecation_warning": "legacy_whole_element_fallback will be removed in v2.0"
                }
            }
        else:
            return {
                "outcome": "native_extraction_success",
                "entities": result.native_entities,
                "can_project_as_concept": True
            }


class ExtractedArtifact(BaseModel):
    """IC-3: Artifact structure that separates native/segmented/fallback outputs."""
    
    extraction_run_id: str
    canonical_elements: List[Dict[str, Any]]
    candidate_spans: List[Dict[str, Any]]  # From segmentation (W2)
    nativeProviderOutput: List[ExtractionResponse]  # What Semantica returned
    fallbackDiagnostics: List[Dict[str, Any]]  # Fallback decisions
    boundaryValidation: List[Dict[str, Any]]  # Quality gates (W4)
    candidateGraph: Dict[str, Any]  # Reference concepts ready for grounding


def build_reference_graph_with_new_profile(
    canonical_elements: List[Dict[str, Any]],
    segmenter,  # ReferenceConceptCandidateSegmenter
    adapter: InProcessSemanticaAdapter,
    fallback_policy: ReferenceGraphBuildFallback,
    extraction_run_id: str
) -> ExtractedArtifact:
    """IC-3: Reference graph build with segmentation and no shim.
    
    Pipeline:
    1. Segment each canonical element into candidate spans
    2. Extract from each candidate span (not element)
    3. Apply fallback policy (no shim!)
    4. Prepare candidates for quality gates (W4)
    """
    
    all_candidate_spans = []
    all_extractions = []
    all_fallback_diagnostics = []
    
    # For each canonical element
    for element in canonical_elements:
        block_id = element.get("block_id", "unknown")
        source_identity = element.get("source_identity", "unknown")
        exact_text = element.get("exact_text", "")
        source_role = element.get("source_role", "canonical_sensitive_term")
        
        # Step 1: Segment
        segmentation_result = segmenter.segment(
            source_block_id=block_id,
            source_identity=source_identity,
            exact_text=exact_text,
            source_role=source_role
        )
        all_candidate_spans.extend(segmentation_result.candidate_spans)
        
        # Step 2: Extract from each candidate
        for span in segmentation_result.candidate_spans:
            candidate_span_id = span["candidate_span_id"]
            span_text = span["exact_text"]
            
            request = ExtractionRequest(
                extraction_run_id=extraction_run_id,
                candidate_span_id=candidate_span_id,
                source_identity=source_identity,
                exact_text=span_text,
                profile_version=segmenter.profile_version,
                provider="in_process_semantica"
            )
            
            response = adapter.extract(request)
            all_extractions.append(response)
            
            # Step 3: Apply fallback policy
            fallback_result = fallback_policy.apply_fallback(response)
            all_fallback_diagnostics.append({
                "candidate_span_id": candidate_span_id,
                "fallback_outcome": fallback_result,
                "source_text": span_text[:100]  # Truncate for diagnostics
            })
    
    # Build artifact
    artifact = ExtractedArtifact(
        extraction_run_id=extraction_run_id,
        canonical_elements=canonical_elements,
        candidate_spans=all_candidate_spans,
        nativeProviderOutput=[r.__dict__ for r in all_extractions],
        fallbackDiagnostics=all_fallback_diagnostics,
        boundaryValidation=[],  # Populated in W4
        candidateGraph={}  # Populated after W4
    )
    
    return artifact


if __name__ == "__main__":
    # Test: Extract from Chinese text (should return empty, not shim)
    adapter = InProcessSemanticaAdapter()
    
    request = ExtractionRequest(
        extraction_run_id="test_run_1",
        candidate_span_id="ccs_v1_abc123",
        source_identity="canonical_v1_xyz",
        exact_text="大型企业",
        profile_version="1.0.0",
        provider="in_process_semantica"
    )
    
    response = adapter.extract(request)
    print(f"Extraction response: empty={response.empty_native_result}, entities={response.native_entities}")
    
    # Test fallback policy
    fallback_policy = ReferenceGraphBuildFallback()
    result = fallback_policy.apply_fallback(response)
    print(f"Fallback result: {result}")
    
    # Assert: No whole-element shim in new profile
    assert result["can_project_as_concept"] == False, "Empty result should NOT be promoted to concept"
    assert result["requires_human_grounding"] == True, "Empty result should require grounding"
    print("✓ IC-3 fallback policy test passed!")
