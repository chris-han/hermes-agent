"""Fail-closed registry for selectable journeys and retired skill references."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


CANONICAL_JOURNEY_REF = "knowledge-graph-building@1.0.0"
CANONICAL_JOURNEY_NAME = "knowledge-graph-building"
COMPILER_CAPABILITY_ID = "corpus-graph-schema-induction"
TENDER_JOURNEY_REF = "tender-review-journey@1.0.0"
TENDER_JOURNEY_NAME = "tender-review-journey"

_SKILLS_ROOT = Path(__file__).resolve().parents[1] / "skills"


class SkillRegistryError(ValueError):
    """Raised when a caller attempts an unknown or retired selection."""


@dataclass(frozen=True, slots=True)
class SkillRegistryRecord:
    skill_ref: str
    name: str
    display_name: str
    source_path: str
    source_ref: str
    skill_md_sha256: str
    lifecycle: Literal["active", "retired"]
    selectable: bool
    read_only: bool
    capability_id: str | None = None
    selection_policy: "SkillSelectionPolicy | None" = None


@dataclass(frozen=True, slots=True)
class RequestIntent:
    all_phrases: tuple[str, ...]
    any_phrases: tuple[str, ...] = ()
    context_phrases: tuple[str, ...] = ()

    def matches(self, request: str) -> bool:
        if not all(_contains_phrase(request, phrase) for phrase in self.all_phrases):
            return False
        if self.any_phrases and not any(
            _contains_phrase(request, phrase) for phrase in self.any_phrases
        ):
            return False
        return not self.context_phrases or any(
            _contains_phrase(request, phrase) for phrase in self.context_phrases
        )


@dataclass(frozen=True, slots=True)
class SkillSelectionPolicy:
    intents: tuple[RequestIntent, ...]
    exclude_phrases: tuple[str, ...]

    def matches(self, user_request: str) -> bool:
        normalized = _normalize_phrase(user_request)
        if any(_contains_phrase(normalized, phrase) for phrase in self.exclude_phrases):
            return False
        return any(intent.matches(normalized) for intent in self.intents)


def _normalize_phrase(value: str) -> str:
    normalized = "".join(
        character if character.isalnum() else " " for character in value.casefold()
    )
    return " ".join(normalized.split())


def _contains_phrase(normalized_request: str, phrase: str) -> bool:
    normalized_phrase = _normalize_phrase(phrase)
    if any("\u4e00" <= character <= "\u9fff" for character in normalized_phrase):
        return normalized_phrase in normalized_request
    return f" {normalized_phrase} " in f" {normalized_request} "


def _frontmatter(path: Path) -> dict[str, object]:
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---\n") or "\n---\n" not in raw[4:]:
        raise SkillRegistryError(f"canonical skill frontmatter is missing: {path}")
    frontmatter_text = raw[4:].split("\n---\n", 1)[0]
    parsed = yaml.safe_load(frontmatter_text) or {}
    if not isinstance(parsed, dict):
        raise SkillRegistryError(f"canonical skill frontmatter is invalid: {path}")
    return parsed


def _string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise SkillRegistryError(f"canonical selection {field} must be a string list")
    return tuple(item.strip() for item in value)


def _selection_policy(path: Path) -> SkillSelectionPolicy:
    metadata = _frontmatter(path).get("metadata")
    semantier = metadata.get("semantier") if isinstance(metadata, dict) else None
    selection = semantier.get("selection") if isinstance(semantier, dict) else None
    if not isinstance(selection, dict):
        raise SkillRegistryError("canonical skill selection metadata is missing")
    raw_intents = selection.get("intents")
    if not isinstance(raw_intents, list) or not raw_intents:
        raise SkillRegistryError("canonical skill selection intents are missing")
    intents: list[RequestIntent] = []
    for raw_intent in raw_intents:
        if not isinstance(raw_intent, dict):
            raise SkillRegistryError("canonical selection intent must be an object")
        intents.append(
            RequestIntent(
                all_phrases=_string_tuple(
                    raw_intent.get("all_phrases"), field="all_phrases"
                ),
                any_phrases=(
                    _string_tuple(raw_intent["any_phrases"], field="any_phrases")
                    if "any_phrases" in raw_intent
                    else ()
                ),
                context_phrases=(
                    _string_tuple(
                        raw_intent["context_phrases"], field="context_phrases"
                    )
                    if "context_phrases" in raw_intent
                    else ()
                ),
            )
        )
    return SkillSelectionPolicy(
        intents=tuple(intents),
        exclude_phrases=_string_tuple(
            selection.get("exclude_phrases"), field="exclude_phrases"
        ),
    )


def _record(
    *,
    skill_ref: str,
    name: str,
    display_name: str,
    relative_path: str,
    lifecycle: Literal["active", "retired"],
    selectable: bool,
    capability_id: str | None = None,
    include_selection_policy: bool = False,
) -> SkillRegistryRecord:
    path = _SKILLS_ROOT / relative_path
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return SkillRegistryRecord(
        skill_ref=skill_ref,
        name=name,
        display_name=display_name,
        source_path=str(path),
        source_ref=f"src/skills/{relative_path}",
        skill_md_sha256=f"sha256:{digest}",
        lifecycle=lifecycle,
        selectable=selectable,
        read_only=not selectable,
        capability_id=capability_id,
        selection_policy=_selection_policy(path) if include_selection_policy else None,
    )


def _canonical_record() -> SkillRegistryRecord:
    return _record(
        skill_ref=CANONICAL_JOURNEY_REF,
        name=CANONICAL_JOURNEY_NAME,
        display_name="Build Knowledge Graph",
        relative_path="governance/knowledge-graph-building/SKILL.md",
        lifecycle="active",
        selectable=True,
        capability_id=COMPILER_CAPABILITY_ID,
        include_selection_policy=True,
    )


def _tender_record() -> SkillRegistryRecord:
    return _record(
        skill_ref=TENDER_JOURNEY_REF,
        name=TENDER_JOURNEY_NAME,
        display_name="Tender Review Journey",
        relative_path="governance/tender-review-journey/SKILL.md",
        lifecycle="active",
        selectable=True,
        include_selection_policy=True,
    )


def _historical_records() -> dict[str, SkillRegistryRecord]:
    return {
        ref: _record(
            skill_ref=ref,
            name=name,
            display_name=display,
            relative_path=path,
            lifecycle="retired",
            selectable=False,
            capability_id=(
                COMPILER_CAPABILITY_ID if name == COMPILER_CAPABILITY_ID else None
            ),
        )
        for ref, name, display, path in (
            (
                "corpus-graph-schema-induction@1.0.0",
                "corpus-graph-schema-induction",
                "Corpus Graph Schema Induction",
                "retired/corpus-graph-schema-induction/1.0.0/SKILL.md",
            ),
            (
                "graph-schema-discovery@1.3.0",
                "graph-schema-discovery",
                "Graph Schema Discovery",
                "retired/graph-schema-discovery/1.3.0/SKILL.md",
            ),
        )
    }


def select_for_new_run(
    *,
    skill_ref: str | None = None,
    user_request: str | None = None,
) -> SkillRegistryRecord | None:
    """Resolve only active journey selections; retired/unknown refs fail closed."""
    canonical = _canonical_record()
    tender = _tender_record()
    active = (tender, canonical)
    if skill_ref is not None:
        for record in active:
            if skill_ref == record.skill_ref:
                return record
        if skill_ref in _historical_records():
            raise SkillRegistryError(
                f"skill ref is retired and not selectable: {skill_ref}"
            )
        raise SkillRegistryError(f"unknown skill ref: {skill_ref}")
    request = (user_request or "").strip()
    if not request:
        return canonical
    for record in active:
        if record.selection_policy is None:
            raise SkillRegistryError(f"active skill selection policy is missing: {record.skill_ref}")
        if record.selection_policy.matches(request):
            return record
    return None


def resolve_historical_ref(skill_ref: str) -> SkillRegistryRecord:
    """Resolve an exact retired ref for read-only replay/audit use."""
    try:
        return _historical_records()[skill_ref]
    except KeyError as exc:
        raise SkillRegistryError(f"unknown historical skill ref: {skill_ref}") from exc
