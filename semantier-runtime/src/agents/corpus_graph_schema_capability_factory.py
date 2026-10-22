"""Agent-layer composition for the shared ContextGraph schema capability."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from jsonschema import ValidationError

from agents.hermes_current_agent_llm import HermesCurrentAgentStructuredLlm
from contracts.graph_intent_brief import graph_intent_proposal_output_schema
from contracts.ai_grounding import StructuredGenerationError
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from runtime_paths import platform_runtime_root
from eos.artifact_hashing import content_hash
from storage.knowledge_builder_store import KnowledgeBuilderStore
from services.corpus_graph_schema_capability_service import CorpusGraphSchemaCapabilityService
from services.graph_schema_llm_evidence_runner import (
    GraphSchemaLlmEvidenceRunner,
    resolve_runtime_llm_provider_model,
)


_GRAPH_INTENT_INSTRUCTIONS_PATH = (
    Path(__file__).resolve().parents[1] / "prompts/graph_intent_provider_v2.txt"
)

_GRAPH_INTENT_REPAIR_PATH = (
    Path(__file__).resolve().parents[1] / "prompts/graph_intent_schema_repair_v1.txt"
)


class _HermesGraphIntentStructuredProvider:
    def __init__(
        self,
        *,
        inference_home: Path,
        expected_provider: str,
        expected_model: str,
        instructions: str | None = None,
        repair_template: str | None = None,
    ) -> None:
        self.inference_home = inference_home.expanduser().resolve()
        self.expected_provider = expected_provider
        self.expected_model = expected_model
        self.instructions = (
            instructions if instructions is not None
            else _GRAPH_INTENT_INSTRUCTIONS_PATH.read_text(encoding="utf-8")
        )
        if not self.instructions.strip():
            raise ValueError("GraphIntent provider instruction asset is empty")
        self.repair_template = (
            repair_template if repair_template is not None
            else _GRAPH_INTENT_REPAIR_PATH.read_text(encoding="utf-8")
        )
        if not self.repair_template.strip():
            raise ValueError("GraphIntent repair instruction asset is empty")
        self.generation_attempts: list[dict[str, Any]] = []
        self.llm = HermesCurrentAgentStructuredLlm(hermes_home=self.inference_home)

    def generate_structured(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if model != self.expected_model or temperature != 0:
            raise ValueError("GraphIntent current-agent invocation drifted from pinned model/config")
        home_marker = set_hermes_home_override(self.inference_home)
        try:
            result = asyncio.run(self._generate(prompt, json_schema=json_schema))
        finally:
            reset_hermes_home_override(home_marker)
        return result

    async def _generate(
        self, prompt: str, *, json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        schema = json_schema if json_schema is not None else graph_intent_proposal_output_schema()
        self.generation_attempts = []
        attempt_input = prompt
        for attempt_number in (1, 2):
            attempt = {
                "attempt_number": attempt_number,
                "produced_at": datetime.now(timezone.utc).isoformat(),
                "usage": None,
                "provider": self.expected_provider, "model": self.expected_model,
                "instructions": self.instructions,
                "instruction_hash": "sha256:" + sha256(self.instructions.encode("utf-8")).hexdigest(),
                "input_text": attempt_input,
                "input_hash": "sha256:" + content_hash(attempt_input),
                "output_schema_hash": "sha256:" + content_hash(schema),
                "repair_template_hash": "sha256:" + sha256(self.repair_template.encode("utf-8")).hexdigest(),
            }
            self.generation_attempts.append(attempt)
            try:
                generated = await self.llm.generate_structured(
                    instructions=self.instructions,
                    input_text=attempt_input,
                    json_schema=schema,
                    schema_name="semantier_graph_intent_proposal_evidence_v1",
                    purpose="contextgraph_graph_intent_proposal",
                )
                attempt.update(usage=dict(generated.usage), output_text=generated.text,
                    output_hash="sha256:" + sha256(generated.text.encode("utf-8")).hexdigest())
                if (generated.provider, generated.model) != (
                    self.expected_provider, self.expected_model,
                ):
                    raise ValueError("GraphIntent current-agent provider/model attribution drifted")
            except Exception as exc:
                raw = getattr(exc, "raw_response", None)
                if isinstance(raw, dict):
                    attempt["raw_response"] = raw
                validation = exc.__cause__
                if isinstance(exc, StructuredGenerationError) and isinstance(validation, ValueError):
                    validation = validation.__cause__
                if not isinstance(validation, ValidationError):
                    attempt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                    raise
                diagnostic = {
                    "path": list(validation.absolute_path),
                    "keyword": validation.validator,
                    "message": validation.message,
                }
                attempt.update(status="schema_rejected", validation_diagnostic=diagnostic)
                if attempt_number == 2:
                    raise
                attempt_input = self.repair_template.format(
                    validation_error=json.dumps(diagnostic, ensure_ascii=False, sort_keys=True),
                    original_request=prompt,
                )
            else:
                attempt["status"] = "passed"
                return generated.parsed
        raise AssertionError("GraphIntent schema repair exhausted without an outcome")


class _HermesGraphIntentGenerator:
    def __init__(
        self, *, inference_home: Path, provider: str, model: str,
        instructions: str | None = None,
        repair_template: str | None = None,
    ) -> None:
        self.provider = _HermesGraphIntentStructuredProvider(
            inference_home=inference_home,
            expected_provider=provider,
            expected_model=model,
            instructions=instructions,
            repair_template=repair_template,
        )


def _managed_auth_available(inference_home: Path, provider: str) -> bool:
    home_marker = set_hermes_home_override(inference_home.expanduser().resolve())
    try:
        from hermes_cli.auth import get_auth_status

        return bool(get_auth_status(provider).get("logged_in"))
    except Exception:
        return False
    finally:
        reset_hermes_home_override(home_marker)


def build_corpus_graph_schema_capability(
    *,
    workspace_root: str | Path,
    workspace_id: str,
    user_id: str,
    organization_id: str,
    inference_home: str | Path | None = None,
    source_registry: KnowledgeBuilderStore | None = None,
    graph_intent_inference_enabled: bool = True,
) -> CorpusGraphSchemaCapabilityService:
    """Compose Web and Chat against one shared Hermes inference owner."""
    if not graph_intent_inference_enabled:
        return CorpusGraphSchemaCapabilityService(workspace_root=workspace_root, workspace_id=workspace_id,
            user_id=user_id, organization_id=organization_id, source_registry=source_registry,
            graph_intent_inference_enabled=False)

    resolved_inference_home = Path(
        inference_home if inference_home is not None else platform_runtime_root()
    ).expanduser().resolve()
    provider_model = resolve_runtime_llm_provider_model(
        hermes_home=resolved_inference_home
    )
    if provider_model is None:
        return CorpusGraphSchemaCapabilityService(
            workspace_root=workspace_root,
            workspace_id=workspace_id,
            user_id=user_id,
            organization_id=organization_id,
            graph_intent_llm_provider_model=None,
            source_registry=source_registry,
        )

    binding = HermesCurrentAgentStructuredLlm._load_auxiliary_binding(
        resolved_inference_home
    )
    # Pin the transport owner's resolved identity, not the raw configuration alias.
    # The attribution guard below must compare two views of this same binding.
    provider, model = binding["provider"], binding["model"]
    instructions = _GRAPH_INTENT_INSTRUCTIONS_PATH.read_text(encoding="utf-8")
    repair_template = _GRAPH_INTENT_REPAIR_PATH.read_text(encoding="utf-8")
    transport_fingerprint = "sha256:" + content_hash(
        {
            "provider": binding["provider"],
            "model": binding["model"],
            "base_url": binding.get("base_url") or "",
            "timeout": binding.get("timeout"),
            "extra_body": binding.get("extra_body") or {},
            "evidence_schema_hash": "sha256:" + content_hash(graph_intent_proposal_output_schema()),
            "instruction_asset_hash": "sha256:" + sha256(instructions.encode("utf-8")).hexdigest(),
            "repair_template_hash": "sha256:" + sha256(repair_template.encode("utf-8")).hexdigest(),
            "maximum_structured_attempts": 2,
        }
    )
    runner = GraphSchemaLlmEvidenceRunner(
        workspace_root,
        workspace_id=workspace_id,
        generator_factory=lambda requested_provider, requested_model: _HermesGraphIntentGenerator(
            inference_home=resolved_inference_home,
            provider=requested_provider,
            model=requested_model,
            instructions=instructions,
            repair_template=repair_template,
        ),
        managed_auth_available=lambda requested_provider: _managed_auth_available(
            resolved_inference_home, requested_provider
        ),
        transport_fingerprint=transport_fingerprint,
    )
    return CorpusGraphSchemaCapabilityService(
        workspace_root=workspace_root,
        workspace_id=workspace_id,
        user_id=user_id,
        organization_id=organization_id,
        graph_intent_llm_runner=runner,
        graph_intent_llm_provider_model=(provider, model),
        source_registry=source_registry,
    )


__all__ = ["build_corpus_graph_schema_capability"]
