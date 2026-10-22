"""Small, governed cognitive-context adapter for the Semantier runtime.

The package deliberately keeps memory, knowledge, context construction, and
learning in one SQLite-backed boundary.  It does not replace Semantica or the
existing governance stores; it gives new cognition code stable contracts over
those substrates.
"""

from .models import (
    CognitiveItem,
    DecisionContext,
    EvidenceRecord,
    LearningCandidate,
    ItemType,
    Scope,
    Status,
)
from .service import CognitiveContextService, LearningConfig
from .store import CognitiveContextStore
from .telemetry import LearningTelemetryAdapter, LearningTelemetryResult
from .models import DecisionStep, DecisionStepConflictError, LearningProposal, LearningProposalProvider

__all__ = [
    "CognitiveContextService",
    "CognitiveContextStore",
    "CognitiveItem",
    "DecisionContext",
    "EvidenceRecord",
    "ItemType",
    "LearningCandidate",
    "LearningConfig",
    "LearningProposal",
    "LearningProposalProvider",
    "LearningTelemetryAdapter",
    "LearningTelemetryResult",
    "DecisionStep",
    "DecisionStepConflictError",
    "Scope",
    "Status",
]
