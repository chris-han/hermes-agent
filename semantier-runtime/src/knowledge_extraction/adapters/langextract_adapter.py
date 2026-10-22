"""LangExtract implementation of the provider-neutral extraction seam."""

from __future__ import annotations

from hashlib import sha256
import importlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from contracts.knowledge_feature_pack import content_hash
from contracts.knowledge_graph_phase1 import SourceAnchor
from contracts.knowledge_reasoning_phase1 import (
    SemanticOccurrence,
    SourceAlignment,
    SourceAlignmentStatus,
)
from knowledge_extraction.contracts import SourceExtractionRequest, SourceExtractionResult


class ExtractionProviderUnavailable(RuntimeError):
    pass


class LangExtractAdapter:
    provider = "langextract"

    def __init__(
        self,
        *,
        artifact_writer: Callable[[str, str], str],
        profile_root: str | Path | None = None,
        extract_fn: Callable[..., Any] | None = None,
        provider_version: str | None = None,
        provider_name: str = "langextract",
    ) -> None:
        self._artifact_writer = artifact_writer
        self.provider = provider_name
        self._profile_root = Path(profile_root) if profile_root else Path(__file__).parents[1] / "profiles"
        self._extract_fn = extract_fn
        if provider_version is not None:
            self._provider_version = provider_version
        else:
            try:
                self._provider_version = version("langextract")
            except PackageNotFoundError:
                self._provider_version = "unavailable"

    def capability(self) -> dict[str, str]:
        try:
            self._resolve_extract_fn()
        except ExtractionProviderUnavailable as exc:
            return {"provider": self.provider, "state": "unavailable", "detail": str(exc)}
        return {"provider": self.provider, "state": "healthy", "detail": self._provider_version}

    def extract(self, request: SourceExtractionRequest) -> SourceExtractionResult:
        profile = _load_profile(self._profile_root, request.extraction_profile)
        configured_model_ref = request.provider_options.get("model_ref", profile["model_ref"])
        if not isinstance(configured_model_ref, str) or not configured_model_ref.strip():
            raise ValueError("LangExtract model_ref must be a non-empty string")
        model_ref = configured_model_ref.strip()
        prompt_ref = f"{request.extraction_profile}:{profile['profile_hash']}"
        try:
            extract_fn = self._resolve_extract_fn()
            examples = profile["examples"]
            if self._extract_fn is None:
                examples = _to_langextract_examples(
                    examples,
                    importlib.import_module("langextract").data,
                )
            provider_result = extract_fn(
                text_or_documents=request.text,
                prompt_description=profile["instruction"],
                examples=examples,
                model_id=model_ref,
                **dict(request.provider_options.get("langextract_options") or {}),
            )
        except ExtractionProviderUnavailable:
            raise
        except Exception as exc:
            raise ExtractionProviderUnavailable(f"LangExtract invocation failed: {exc}") from exc

        raw_payload = _provider_payload(provider_result)
        raw = json.dumps(raw_payload, ensure_ascii=False, sort_keys=True, default=str)
        raw_hash = f"sha256:{sha256(raw.encode('utf-8')).hexdigest()}"
        raw_ref = self._artifact_writer(raw, raw_hash)
        occurrences = tuple(
            _occurrence(
                extraction=item,
                ordinal=index,
                request=request,
                raw_output_ref=raw_ref,
                model_ref=model_ref,
                prompt_ref=prompt_ref,
                provider_name=self.provider,
            )
            for index, item in enumerate(getattr(provider_result, "extractions", ()) or ())
        )
        warnings = tuple(str(item) for item in getattr(provider_result, "warnings", ()) or ())
        return SourceExtractionResult(
            extraction_run_id=request.extraction_run_id,
            provider=self.provider,
            provider_version=self._provider_version,
            model_ref=model_ref,
            prompt_ref=prompt_ref,
            raw_output_ref=raw_ref,
            raw_output_hash=raw_hash,
            occurrences=occurrences,
            warnings=warnings,
            capability_state="partial" if warnings else "healthy",
        )

    def _resolve_extract_fn(self) -> Callable[..., Any]:
        if self._extract_fn is not None:
            return self._extract_fn
        try:
            module = importlib.import_module("langextract")
        except ImportError as exc:
            raise ExtractionProviderUnavailable(
                "LangExtract is unavailable; install the knowledge-langextract optional dependency"
            ) from exc
        return module.extract


def _load_profile(root: Path, profile_name: str) -> dict[str, Any]:
    profile_dir = root / profile_name
    if not profile_dir.is_dir():
        raise ValueError(f"extraction profile not found: {profile_name}")
    instruction = (profile_dir / "instruction.md").read_text(encoding="utf-8")
    examples = json.loads((profile_dir / "examples.json").read_text(encoding="utf-8"))
    metadata = json.loads((profile_dir / "profile.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("extraction profile metadata must be an object")
    if metadata.get("schema_version") != "knowledge_extraction_profile.v1":
        raise ValueError("unsupported extraction profile schema_version")
    if metadata.get("profile") != profile_name:
        raise ValueError("extraction profile name does not match its directory")
    if not isinstance(metadata.get("model_ref"), str) or not metadata["model_ref"].strip():
        raise ValueError("extraction profile model_ref must be a non-empty string")
    if not isinstance(examples, list):
        raise ValueError("extraction profile examples must be a list")
    return {
        **metadata,
        "instruction": instruction,
        "examples": examples,
        "profile_hash": content_hash({"instruction": instruction, "examples": examples, "metadata": metadata}),
    }


def _to_langextract_examples(raw_examples: list[Any], data: Any) -> list[Any]:
    converted = []
    for example_index, raw_example in enumerate(raw_examples):
        if not isinstance(raw_example, Mapping):
            raise ValueError(f"LangExtract example {example_index} must be an object")
        text = raw_example.get("text")
        raw_extractions = raw_example.get("extractions")
        if not isinstance(text, str) or not text:
            raise ValueError(f"LangExtract example {example_index} text must be non-empty")
        if not isinstance(raw_extractions, list):
            raise ValueError(f"LangExtract example {example_index} extractions must be a list")
        group_indexes: dict[str, int] = {}
        extractions = []
        for extraction_index, raw_extraction in enumerate(raw_extractions):
            if not isinstance(raw_extraction, Mapping):
                raise ValueError(
                    f"LangExtract example {example_index} extraction {extraction_index} must be an object"
                )
            extraction_class = raw_extraction.get("extraction_class")
            extraction_text = raw_extraction.get("extraction_text")
            attributes = raw_extraction.get("attributes")
            if not isinstance(extraction_class, str) or not extraction_class:
                raise ValueError("LangExtract extraction_class must be a non-empty string")
            if not isinstance(extraction_text, str) or not extraction_text:
                raise ValueError("LangExtract extraction_text must be a non-empty string")
            if attributes is not None and not isinstance(attributes, Mapping):
                raise ValueError("LangExtract extraction attributes must be an object")
            copied_attributes = dict(attributes) if attributes is not None else None
            group_index = raw_extraction.get("group_index")
            group_ref = copied_attributes.get("group_ref") if copied_attributes else None
            if group_index is not None and (not isinstance(group_index, int) or isinstance(group_index, bool)):
                raise ValueError("LangExtract group_index must be an integer")
            if group_index is None and group_ref is not None:
                if not isinstance(group_ref, str) or not group_ref:
                    raise ValueError("LangExtract group_ref must be a non-empty string")
                group_index = group_indexes.setdefault(group_ref, len(group_indexes))
            extractions.append(
                data.Extraction(
                    extraction_class=extraction_class,
                    extraction_text=extraction_text,
                    attributes=copied_attributes,
                    group_index=group_index,
                )
            )
        converted.append(data.ExampleData(text=text, extractions=extractions))
    return converted


def _provider_payload(result: Any) -> dict[str, Any]:
    return {
        "extractions": [
            {
                "extraction_class": getattr(item, "extraction_class", None),
                "extraction_text": getattr(item, "extraction_text", None),
                "attributes": getattr(item, "attributes", None),
                "char_interval": _interval_payload(getattr(item, "char_interval", None)),
                "alignment_status": str(getattr(item, "alignment_status", "") or ""),
                "group_index": getattr(item, "group_index", None),
                "extraction_index": getattr(item, "extraction_index", None),
            }
            for item in getattr(result, "extractions", ()) or ()
        ]
    }


def _interval_payload(interval: Any) -> dict[str, int] | None:
    if interval is None:
        return None
    return {"start": int(interval.start_pos), "end": int(interval.end_pos)}


def _occurrence(
    *, extraction: Any, ordinal: int, request: SourceExtractionRequest,
    raw_output_ref: str, model_ref: str, prompt_ref: str,
    provider_name: str,
) -> SemanticOccurrence:
    interval = getattr(extraction, "char_interval", None)
    extraction_text = str(getattr(extraction, "extraction_text", "") or "")
    if interval is None:
        start, end = 0, len(request.text)
        status = SourceAlignmentStatus.UNGROUNDED
        provider_start = provider_end = None
    else:
        start, end = int(interval.start_pos), int(interval.end_pos)
        if start < 0 or end > len(request.text) or end <= start:
            raise ValueError("LangExtract interval is outside canonical text")
        provider_start, provider_end = start, end
        status = _alignment_status(
            getattr(extraction, "alignment_status", None),
            exact=request.text[start:end] == extraction_text,
        )
    exact_text = request.text[start:end]
    anchor_basis = {"document_id": request.document_id, "start": start, "end": end, "hash": request.canonical_document_hash}
    anchor = SourceAnchor(
        anchor_id=f"source_anchor_{content_hash(anchor_basis)[7:31]}",
        document_hash=request.canonical_document_hash,
        document_part="paragraph",
        paragraph_index=0,
        start_offset=start,
        end_offset=end,
        exact_text=exact_text,
        local_text_hash=content_hash(exact_text),
    )
    identity = {"run": request.extraction_run_id, "ordinal": ordinal, "anchor": anchor.anchor_id}
    confidence = (getattr(extraction, "attributes", None) or {}).get("confidence")
    return SemanticOccurrence(
        occurrence_id=f"occ_{content_hash(identity)[7:31]}",
        extraction_run_id=request.extraction_run_id,
        document_id=request.document_id,
        source_anchor=anchor,
        extraction_class=str(getattr(extraction, "extraction_class", "unknown") or "unknown"),
        extraction_attributes=dict(getattr(extraction, "attributes", None) or {}),
        alignment=SourceAlignment(
            provider=provider_name,
            status=status,
            provider_start=provider_start,
            provider_end=provider_end,
        ),
        machine_confidence=float(confidence) if confidence is not None else None,
        model_ref=model_ref,
        prompt_ref=prompt_ref,
        raw_output_ref=raw_output_ref,
        group_ref=(
            str(getattr(extraction, "group_index"))
            if getattr(extraction, "group_index", None) is not None else None
        ),
        extraction_index=getattr(extraction, "extraction_index", None),
    )


def _alignment_status(value: Any, *, exact: bool) -> SourceAlignmentStatus:
    normalized = str(getattr(value, "value", value) or "").lower()
    if exact:
        return SourceAlignmentStatus.EXACT
    return {
        "match_fuzzy": SourceAlignmentStatus.FUZZY,
        "fuzzy": SourceAlignmentStatus.FUZZY,
        "match_greater": SourceAlignmentStatus.GREATER,
        "greater": SourceAlignmentStatus.GREATER,
        "match_lesser": SourceAlignmentStatus.LESSER,
        "lesser": SourceAlignmentStatus.LESSER,
    }.get(normalized, SourceAlignmentStatus.FUZZY)
