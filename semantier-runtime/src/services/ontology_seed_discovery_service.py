from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import time

from contracts.graph_intent_brief import GraphIntentBrief
from contracts.ontology_build_inputs import OntologyBuildInputEnvelopeV1
from contracts.ontology_building import CompetencyContractV1, OntologyBuildManifestV1
from contracts.ontology_catalog import CatalogProviderError, OntologyCatalogCandidate, OntologyCatalogProvider
from contracts.ontology_seed_state import (
    OntologySeedIgnoreReceiptV1, OntologySeedStateV1, SeedPolicySnapshotV1,
    SeedProviderFailureV1,
)
from services.ontology_registry_catalog_provider import SemanticaRegistryCatalogProvider
from services.ontology_seed_matcher import (
    OntologySeedDemandService, OntologySeedMatchPolicyV1, OntologySeedMatcher,
)
from storage.ontology_build_store import OntologyBuildStore


@dataclass(frozen=True)
class ExternalCatalogBinding:
    provider_id: str
    provider: OntologyCatalogProvider


class OntologySeedDiscoveryService:
    def __init__(self, *, max_external_candidates: int = 20,
                 max_provider_failures: int = 10, discovery_deadline_seconds: float = 30.0) -> None:
        if (min(max_external_candidates, max_provider_failures) < 1
                or not math.isfinite(discovery_deadline_seconds)
                or discovery_deadline_seconds <= 0):
            raise ValueError("discovery bounds and deadline must be positive")
        self._max_external_candidates = max_external_candidates
        self._max_provider_failures = max_provider_failures
        self._deadline = discovery_deadline_seconds

    def begin(
        self, *, store: OntologyBuildStore, organization_id: str, workspace_id: str,
        build_ref: str, expected_state_ref: str | None, expected_state_hash: str | None,
        created_at: datetime, policy: OntologySeedMatchPolicyV1 | None = None,
        competency_contract_ref: str | None = None,
        competency_contract_hash: str | None = None,
    ) -> OntologySeedStateV1:
        manifest = store.get(organization_id=organization_id, workspace_id=workspace_id,
                             artifact_ref=build_ref, contract_type=OntologyBuildManifestV1)
        if manifest is None:
            raise LookupError("ontology build manifest not found")
        input_pin = self._manifest_pin(manifest, "ontology_build_inputs")
        envelope = store.get(organization_id=organization_id, workspace_id=workspace_id,
                             artifact_ref=input_pin.artifact_ref,
                             contract_type=OntologyBuildInputEnvelopeV1)
        if envelope is None or envelope.envelope_hash != input_pin.artifact_hash:
            raise ValueError("build manifest input envelope pin is missing or corrupt")
        graph_intent = GraphIntentBrief.from_dict(envelope.graph_intent)
        graph_pin = self._manifest_pin(manifest, "graph_intent_brief")
        if (graph_intent.graph_intent_id, graph_intent.graph_intent_hash) != (
            graph_pin.artifact_ref, graph_pin.artifact_hash,
        ):
            raise ValueError("archived GraphIntent does not match build manifest")
        competency = None
        if (competency_contract_ref is None) != (competency_contract_hash is None):
            raise ValueError("competency contract ref/hash must be paired")
        if competency_contract_ref is not None:
            competency_pin = self._manifest_pin(manifest, "competency_contract")
            if (competency_pin.artifact_ref, competency_pin.artifact_hash) != (
                competency_contract_ref, competency_contract_hash,
            ):
                raise ValueError("competency contract does not match build manifest")
            competency = store.get(organization_id=organization_id, workspace_id=workspace_id,
                                   artifact_ref=competency_contract_ref,
                                   contract_type=CompetencyContractV1)
            if competency is None:
                raise ValueError("manifest-pinned competency contract is missing")
        effective_policy = OntologySeedMatchPolicyV1.model_validate(
            (policy or OntologySeedMatchPolicyV1()).model_dump(mode="json")
        )
        query = OntologySeedDemandService().derive(
            graph_intent=graph_intent, competency_contract=competency,
            organization_id=organization_id, workspace_id=workspace_id,
            build_ref=build_ref, created_at=created_at,
        )
        predecessor = store.get_seed_state_head(
            organization_id=organization_id, workspace_id=workspace_id, build_ref=build_ref
        )
        revision = 1 if predecessor is None else predecessor.revision + 1
        state = OntologySeedStateV1.create(
            organization_id=organization_id, workspace_id=workspace_id,
            build_ref=build_ref, build_hash=manifest.build_hash, revision=revision,
            predecessor_state_ref=predecessor.state_ref if predecessor else None,
            predecessor_state_hash=predecessor.state_hash if predecessor else None,
            search_status="searching", query=query,
            policy=SeedPolicySnapshotV1(
                policy_version=effective_policy.policy_version,
                policy_hash=effective_policy.policy_hash,
                config=effective_policy.model_dump(mode="json"),
            ), recommendations=(), ignored_recommendation_refs=(), external_candidates=(),
            provider_failures=(), selected_recommendation=None, imported_snapshot=None,
            alignment_actions=(), source_grounded_delta=None, created_at=created_at,
        )
        store.append_seed_state(
            state, expected_state_ref=expected_state_ref, expected_state_hash=expected_state_hash
        )
        return state

    async def run(
        self, *, store_path: Path, organization_id: str, workspace_id: str,
        build_ref: str, searching_state_ref: str, searching_state_hash: str,
        external_providers: tuple[ExternalCatalogBinding, ...] = (),
        completed_at: datetime | None = None,
    ) -> OntologySeedStateV1:
        store = OntologyBuildStore(store_path)
        try:
            state = store.get(organization_id=organization_id, workspace_id=workspace_id,
                              artifact_ref=searching_state_ref, contract_type=OntologySeedStateV1)
            if state is None or state.state_hash != searching_state_hash or state.build_ref != build_ref:
                raise ValueError("searching state pin is missing or stale")
            head = store.get_seed_state_head(organization_id=organization_id,
                                             workspace_id=workspace_id, build_ref=build_ref)
            if head is None or (head.state_ref, head.state_hash) != (state.state_ref, state.state_hash):
                raise ValueError("search completion is obsolete")
            if state.search_status != "searching" or state.query is None or state.policy is None:
                raise ValueError("state is not an active search")
            policy = OntologySeedMatchPolicyV1.model_validate(state.policy.config)
            if policy.policy_hash != state.policy.policy_hash:
                raise ValueError("stored effective policy is corrupt")
            external: list[OntologyCatalogCandidate] = []
            failures: list[SeedProviderFailureV1] = []
            recommendations = ()

            def registry_profiles():
                worker_store = OntologyBuildStore(store_path)
                try:
                    async def resolve():
                        registry = SemanticaRegistryCatalogProvider(
                            store=worker_store, organization_id=organization_id,
                            workspace_id=workspace_id, max_results=policy.retrieval_limit,
                        )
                        resolved = []
                        for entry in await registry.search(state.query):
                            if entry.matching_eligible and entry.profile_ref is not None:
                                resolved.append(await registry.fetch_profile(
                                    entry.snapshot_ref, entry.profile_ref
                                ))
                        return tuple(resolved)

                    return asyncio.run(resolve())
                finally:
                    worker_store.close()

            async def discover():
                profiles = await asyncio.to_thread(registry_profiles)
                for binding in external_providers:
                    try:
                        candidates = await binding.provider.search(state.query)
                    except CatalogProviderError:
                        failures.append(SeedProviderFailureV1(
                            provider_id=binding.provider_id, reason_code="PROVIDER_FAILURE"
                        ))
                        continue
                    external.extend(candidates)
                    if len(external) >= self._max_external_candidates:
                        break
                match = await asyncio.to_thread(
                    OntologySeedMatcher(policy=policy).match,
                    query=state.query, profiles=profiles,
                    current_graph_intent_ref=state.query.graph_intent_ref,
                    current_graph_intent_hash=state.query.graph_intent_hash,
                    current_competency_contract_ref=state.query.competency_contract_ref,
                    current_competency_contract_hash=state.query.competency_contract_hash,
                    current_competency_contract_content_hash=(
                        state.query.competency_contract_content_hash
                    ),
                )
                return match.recommendations

            started = time.monotonic()
            try:
                async with asyncio.timeout(self._deadline):
                    recommendations = await discover()
                    if time.monotonic() - started > self._deadline:
                        raise TimeoutError
            except TimeoutError:
                recommendations = ()
                failures.append(SeedProviderFailureV1(
                    provider_id="discovery", reason_code="DISCOVERY_DEADLINE"
                ))
            failures = failures[: self._max_provider_failures]
            deduplicated = {item.external_ref: item for item in external}
            external_candidates = tuple(
                deduplicated[key] for key in sorted(deduplicated)
            )[: self._max_external_candidates]
            status = "matched" if recommendations else (
                "searched" if external_candidates else "no_match"
            )
            successor = OntologySeedStateV1.create(
                organization_id=organization_id, workspace_id=workspace_id,
                build_ref=build_ref, build_hash=state.build_hash, revision=state.revision + 1,
                predecessor_state_ref=state.state_ref, predecessor_state_hash=state.state_hash,
                search_status=status, query=state.query, policy=state.policy,
                recommendations=recommendations, ignored_recommendation_refs=(),
                external_candidates=external_candidates, provider_failures=tuple(failures),
                selected_recommendation=None, imported_snapshot=None, alignment_actions=(),
                source_grounded_delta=None, created_at=completed_at or datetime.now(timezone.utc),
            )
            store.append_seed_state(successor, expected_state_ref=state.state_ref,
                                    expected_state_hash=state.state_hash)
            return successor
        finally:
            store.close()

    def ignore(
        self, *, store: OntologyBuildStore, organization_id: str, workspace_id: str,
        build_ref: str, recommendation_ref: str,
        expected_state_ref: str, expected_state_hash: str,
        reason_code: str, ignored_at: datetime,
    ) -> tuple[OntologySeedStateV1, OntologySeedIgnoreReceiptV1]:
        current = store.get_seed_state_head(
            organization_id=organization_id, workspace_id=workspace_id, build_ref=build_ref
        )
        if current is None or (current.state_ref, current.state_hash) != (
            expected_state_ref, expected_state_hash,
        ):
            raise ValueError("ignore expected state is stale")
        matches = [item for item in current.visible_recommendations
                   if item.recommendation_ref == recommendation_ref and item.status == "PROPOSED"]
        if len(matches) != 1:
            raise ValueError("ignore requires one exact visible proposed recommendation")
        recommendation = matches[0]
        successor = OntologySeedStateV1.create(
            **current.model_dump(mode="python", exclude={
                "schema_version", "state_ref", "state_hash", "revision",
                "predecessor_state_ref", "predecessor_state_hash", "created_at",
                "ignored_recommendation_refs",
            }),
            revision=current.revision + 1, predecessor_state_ref=current.state_ref,
            predecessor_state_hash=current.state_hash,
            ignored_recommendation_refs=tuple((*current.ignored_recommendation_refs, recommendation_ref)),
            created_at=ignored_at,
        )
        receipt = OntologySeedIgnoreReceiptV1.create(
            organization_id=organization_id, workspace_id=workspace_id, build_ref=build_ref,
            query_ref=current.query.query_ref, query_hash=current.query.query_hash,
            recommendation_ref=recommendation.recommendation_ref,
            recommendation_hash=recommendation.recommendation_hash,
            predecessor_state_ref=current.state_ref, predecessor_state_hash=current.state_hash,
            successor_state_ref=successor.state_ref, successor_state_hash=successor.state_hash,
            reason_code=reason_code, ignored_at=ignored_at,
        )
        store.append_seed_state(successor, expected_state_ref=current.state_ref,
                                expected_state_hash=current.state_hash, related_artifacts=(receipt,))
        return successor, receipt

    @staticmethod
    def _manifest_pin(manifest: OntologyBuildManifestV1, artifact_kind: str):
        matches = [item for item in manifest.child_artifacts if item.artifact_kind == artifact_kind]
        if len(matches) != 1:
            raise ValueError(f"build manifest requires exact {artifact_kind} pin")
        return matches[0]
