"""Repo-owned registry for governed corpus graph-schema induction profiles."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from types import MappingProxyType

from contracts.knowledge_relation_taxonomy import KnowledgeRelationProfileCatalogV1
from eos.artifact_hashing import content_hash


_PROFILE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_PROFILE_METADATA = {
    "tender_corpus_schema_induction.v0_1": {
        "source_kind": "docx",
        "display_name_en": "Tender corpus schema induction",
        "display_name_zh": "招投标语料图模式归纳",
    },
    "ai_agents_workshop_schema_induction.v0_1": {
        "source_kind": "docx",
        "display_name_en": "AI agents workshop schema induction",
        "display_name_zh": "AI 智能体研讨会图模式归纳",
    },
}


@dataclass(frozen=True)
class CorpusGraphSchemaProfile:
    profile_id: str
    source_kind: str
    display_name_en: str
    display_name_zh: str
    path: Path
    content_hash: str
    graph_intent_catalog_hash: str | None
    knowledge_relation_catalog_hash: str | None


class CorpusGraphSchemaProfileRegistry:
    """Resolve exact profile IDs to immutable files below the repo profile root."""

    __slots__ = (
        "_payload_json_by_profile",
        "_profiles",
        "_profiles_root",
        "_sealed",
    )

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("governed profile registry composition is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, *, repository_root: str | Path | None = None) -> None:
        object.__setattr__(self, "_sealed", False)
        root = (
            Path(repository_root).resolve()
            if repository_root is not None
            else Path(__file__).resolve().parents[2]
        )
        object.__setattr__(
            self,
            "_profiles_root",
            (root / "src/semantier/knowledge_feature_packs/profiles").resolve(),
        )
        profiles, payload_json_by_profile = self._load_profiles()
        object.__setattr__(self, "_profiles", MappingProxyType(profiles))
        object.__setattr__(
            self,
            "_payload_json_by_profile",
            MappingProxyType(payload_json_by_profile),
        )
        object.__setattr__(self, "_sealed", True)

    @property
    def profiles_root(self) -> Path:
        return self._profiles_root

    def list_profiles(self) -> tuple[CorpusGraphSchemaProfile, ...]:
        return tuple(
            CorpusGraphSchemaProfileRegistry.resolve(self, key)
            for key in sorted(self._profiles)
        )

    def resolve(self, profile_id: str) -> CorpusGraphSchemaProfile:
        if not isinstance(profile_id, str) or not _PROFILE_ID.fullmatch(profile_id):
            raise ValueError("profile_id must be an exact ASCII-stable identifier")
        try:
            profile = self._profiles[profile_id]
        except KeyError as exc:
            raise ValueError(f"unknown governed profile_id: {profile_id}") from exc
        current = json.loads(profile.path.read_text(encoding="utf-8"))
        if f"sha256:{content_hash(current)}" != profile.content_hash:
            raise ValueError("governed profile content hash mismatch")
        return profile

    def profile_payload(
        self, profile_id: str, *, expected_profile_hash: str | None = None
    ) -> dict:
        profile = CorpusGraphSchemaProfileRegistry.resolve(self, profile_id)
        if (
            expected_profile_hash is not None
            and profile.content_hash != expected_profile_hash
        ):
            raise ValueError("expected profile content hash mismatch")
        return json.loads(self._payload_json_by_profile[profile_id])

    def graph_intent_catalog(
        self,
        profile_id: str,
        *,
        expected_profile_hash: str | None = None,
        expected_catalog_hash: str | None = None,
    ) -> dict:
        profile = CorpusGraphSchemaProfileRegistry.resolve(self, profile_id)
        catalog = CorpusGraphSchemaProfileRegistry.profile_payload(
            self,
            profile_id, expected_profile_hash=expected_profile_hash
        ).get("graph_intent_catalog")
        if not isinstance(catalog, dict):
            raise ValueError(f"governed profile has no graph intent catalog: {profile_id}")
        if profile.graph_intent_catalog_hash is None:
            raise ValueError(f"governed profile has no graph intent catalog: {profile_id}")
        actual = f"sha256:{content_hash(catalog)}"
        if actual != profile.graph_intent_catalog_hash:
            raise ValueError("governed profile catalog hash mismatch")
        if expected_catalog_hash is not None and actual != expected_catalog_hash:
            raise ValueError("expected catalog content hash mismatch")
        return catalog

    def require_additive_type_successor(self, profile_id: str, *, predecessor_id: str,
                                       predecessor_hash: str, predecessor_catalog_hash: str) -> None:
        """Verify one explicit type-only migration without changing historical pins."""
        old = self.profile_payload(predecessor_id, expected_profile_hash=predecessor_hash)
        self.graph_intent_catalog(predecessor_id, expected_profile_hash=predecessor_hash,
            expected_catalog_hash=predecessor_catalog_hash)
        new = self.profile_payload(profile_id)
        expected = dict(profile_id=predecessor_id, profile_hash=predecessor_hash,
            catalog_hash=predecessor_catalog_hash, change_kind='additive_graph_intent_types.v1')
        if profile_id == predecessor_id or new.pop('predecessor_profile', None) != expected:
            raise ValueError('profileId does not name an exact additive catalog successor')
        old_catalog, new_catalog = old.pop('graph_intent_catalog'), new.pop('graph_intent_catalog')
        old_types, new_types = old_catalog.pop('admitted_type_ids'), new_catalog.pop('admitted_type_ids')
        if (not isinstance(new_types, list) or not all(isinstance(t, str) and _PROFILE_ID.fullmatch(t) for t in new_types)
            or len(new_types) != len(set(new_types)) or not set(old_types) < set(new_types)
            or old_catalog != new_catalog):
            raise ValueError('profile successor must only add graph intent types')
        for key in ('profile_id', 'source_kind', 'display_name_en', 'display_name_zh', 'predecessor_profile'):
            old.pop(key, None)
            new.pop(key, None)
        if old != new or self.resolve(profile_id).source_kind != self.resolve(predecessor_id).source_kind:
            raise ValueError('profile successor changes non-catalog semantics')

    def knowledge_relation_catalog(
        self,
        profile_id: str,
        *,
        expected_profile_hash: str | None = None,
        expected_catalog_hash: str | None = None,
    ) -> dict:
        """Resolve the one profile-local relation catalog by exact profile/hash pins."""

        profile = CorpusGraphSchemaProfileRegistry.resolve(self, profile_id)
        catalog = CorpusGraphSchemaProfileRegistry.profile_payload(
            self,
            profile_id, expected_profile_hash=expected_profile_hash
        ).get("knowledge_relation_catalog")
        if not isinstance(catalog, dict):
            raise ValueError(
                f"governed profile has no knowledge relation catalog: {profile_id}"
            )
        if profile.knowledge_relation_catalog_hash is None:
            raise ValueError(
                f"governed profile has no knowledge relation catalog: {profile_id}"
            )
        validated = KnowledgeRelationProfileCatalogV1.model_validate(catalog)
        actual = validated.catalog_hash()
        if actual != profile.knowledge_relation_catalog_hash:
            raise ValueError("governed profile relation catalog hash mismatch")
        if expected_catalog_hash is not None and actual != expected_catalog_hash:
            raise ValueError("expected catalog content hash mismatch")
        return validated.model_dump(mode="json", by_alias=True)

    def _load_profiles(
        self,
    ) -> tuple[dict[str, CorpusGraphSchemaProfile], dict[str, str]]:
        profiles: dict[str, CorpusGraphSchemaProfile] = {}
        payload_json_by_profile: dict[str, str] = {}
        if not self.profiles_root.is_dir():
            return profiles, payload_json_by_profile
        for path in sorted(self.profiles_root.glob("*.json")):
            resolved = path.resolve()
            if not resolved.is_relative_to(self.profiles_root):
                raise ValueError("profile path escapes the governed profiles root")
            payload = json.loads(resolved.read_text(encoding="utf-8"))
            profile_id = payload.get("profile_id")
            if (
                not isinstance(profile_id, str)
                or not _PROFILE_ID.fullmatch(profile_id)
                or path.stem != profile_id
            ):
                raise ValueError(f"invalid governed profile_id in {path.name}")
            metadata = _PROFILE_METADATA.get(profile_id, payload)
            source_kind = metadata.get("source_kind")
            display_name_en = metadata.get("display_name_en")
            display_name_zh = metadata.get("display_name_zh")
            if not all(
                isinstance(value, str) and value.strip()
                for value in (source_kind, display_name_en, display_name_zh)
            ):
                raise ValueError(f"profile metadata is incomplete for {profile_id}")
            if not _PROFILE_ID.fullmatch(source_kind):
                raise ValueError(f"source_kind must be ASCII-stable for {profile_id}")
            profiles[profile_id] = CorpusGraphSchemaProfile(
                profile_id=profile_id,
                source_kind=source_kind,
                display_name_en=display_name_en,
                display_name_zh=display_name_zh,
                path=resolved,
                content_hash=f"sha256:{content_hash(payload)}",
                graph_intent_catalog_hash=(
                    f"sha256:{content_hash(payload['graph_intent_catalog'])}"
                    if isinstance(payload.get("graph_intent_catalog"), dict)
                    else None
                ),
                knowledge_relation_catalog_hash=(
                    KnowledgeRelationProfileCatalogV1.model_validate(
                        payload["knowledge_relation_catalog"]
                    ).catalog_hash()
                    if isinstance(payload.get("knowledge_relation_catalog"), dict)
                    else None
                ),
            )
            payload_json_by_profile[profile_id] = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        return profiles, payload_json_by_profile
