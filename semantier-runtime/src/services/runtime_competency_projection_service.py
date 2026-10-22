"""Deterministic pre-F_C competency triage."""
from contracts.runtime_competency_projection import GovernedCompetencyBindingV1, RuntimeCompetencyProjectionV1
from contracts.semantic_kernel import WorkingContext

class RuntimeCompetencyProjectionService:
    @staticmethod
    def project(*, working_context: WorkingContext, candidates: tuple[GovernedCompetencyBindingV1,...]):
        keys=[x.competency_key for x in candidates]
        if len(keys)!=len(set(keys)):
            raise ValueError("DUPLICATE_COMPETENCY_KEY")
        result=tuple(RuntimeCompetencyProjectionV1.create(
            working_context=working_context,binding=b
        ) for b in sorted(candidates,key=lambda x:x.competency_key))
        return result

    @staticmethod
    def rank_bounded(*, candidates: tuple[RuntimeCompetencyProjectionV1, ...],
                     scores: dict[str, float]) -> tuple[RuntimeCompetencyProjectionV1, ...]:
        """Research seam for learned triage ranking; ranking can never broaden candidates."""
        keys = {item.competency_key for item in candidates}
        if not set(scores).issubset(keys):
            raise ValueError("TRIAGE_RANKING_CANNOT_BROADEN_CANDIDATES")
        return tuple(sorted(candidates, key=lambda item: (-scores.get(item.competency_key, float("-inf")),
                                                          item.competency_key)))
