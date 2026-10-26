"""Live Semantica LLM proposal evidence runner for graph schema discovery."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from eos.artifact_hashing import canonical_json, content_hash
from semantier.knowledge_feature_packs.semantica_adapter import (
    SEMANTICA_COMMIT,
    _ensure_vendored_semantica_importable,
)
from storage.graph_schema_discovery_store import GraphSchemaDiscoveryStore


SKILL_MAJOR_VERSION = "knowledge-graph-building@1.0.0"
PROVIDER_CONTRACT_VERSION = "semantica-llm-ontology@1"

DEEPSEEK_MAAS_MODEL = "deepseek-v4.1-flash-1017"
DEEPSEEK_MAAS_BASE_URL = "https://ws-n7w2em1zmym5ysta.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"


def _provider_model(provider: str, model: str) -> tuple[str, str]:
    if provider.lower() == "deepseek":
        return "alibaba", DEEPSEEK_MAAS_MODEL
    return provider.lower(), model


PROVIDER_ENV_KEYS = {
    "alibaba": "DASHSCOPE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DASHSCOPE_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "novita": "NOVITA_API_KEY",
    "ollama": "",
    "openai": "OPENAI_API_KEY",
}


GeneratorFactory = Callable[[str, str], Any]
ManagedAuthAvailable = Callable[[str], bool]


def resolve_active_workspace_path(
    workspace_root: str | Path, *, workspace_id: str
) -> Path:
    """Resolve either a runtime root or an already-active workspace root."""

    root = Path(workspace_root).resolve()
    return (
        root
        if root.name == workspace_id and root.parent.name == "workspaces"
        else root / "workspaces" / workspace_id
    )


def resolve_runtime_llm_provider_model(
    *,
    environ: Mapping[str, str] | None = None,
    hermes_home: str | Path | None = None,
    workspace_root: str | Path | None = None,
) -> tuple[str, str] | None:
    env = environ if environ is not None else os.environ
    provider = str(env.get("HERMES_INFERENCE_PROVIDER") or "").strip()
    model = str(
        env.get("HERMES_INFERENCE_MODEL") or env.get("HERMES_MODEL") or ""
    ).strip()
    hosted = env.get("SEMANTIER_STORAGE_BACKEND") == "postgresql"

    config_paths: list[Path] = []
    if hermes_home is not None:
        config_paths.append(Path(hermes_home).expanduser().resolve() / "config.yaml")
    elif env.get("HERMES_HOME"):
        config_paths.append(
            Path(str(env["HERMES_HOME"])).expanduser().resolve() / "config.yaml"
        )
    if workspace_root is not None:
        config_paths.append(
            Path(workspace_root).resolve() / ".semantier-home" / "config.yaml"
        )

    for config_path in config_paths:
        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        config = loaded if isinstance(loaded, dict) else {}
        auxiliary = config.get("auxiliary")
        task = (
            auxiliary.get("contextgraph_ai_ground")
            if isinstance(auxiliary, dict)
            else None
        )
        task = task if isinstance(task, dict) else {}
        model_config = config.get("model")
        model_config = model_config if isinstance(model_config, dict) else {}
        cloud = config.get("semantier_gateway")
        managed = (hosted or model_config.get("provider") == "semantier-cloud"
                   or isinstance(cloud, dict) and cloud.get("enabled") is True)
        if not managed and provider and model:
            return _provider_model(provider, model)
        resolved_provider = str(task.get("provider") or ("" if managed else provider)).strip()
        resolved_model = str(task.get("model") or ("" if managed else model)).strip()
        if resolved_provider in {"", "auto"}:
            resolved_provider = str(model_config.get("provider") or "").strip()
        if not resolved_model:
            resolved_model = str(
                model_config.get("default") or model_config.get("model") or ""
            ).strip()
        if resolved_provider and resolved_provider != "auto" and resolved_model:
            return _provider_model(resolved_provider, resolved_model)
        if managed:
            return None
    if not hosted and provider and model:
        return _provider_model(provider, model)
    return None


class GraphIntentReexecutionEvidence:
    """Durable stages in the existing proposal evidence owner; no transport owner."""

    def __init__(self, store, context, control, transport_fingerprint):
        self.store, self.context, self.control = store, context, control
        self.transport_fingerprint = transport_fingerprint
        self.prefix = "gire_" + content_hash(dict(organization=context.organization_id,
            workspace=context.workspace_id, actor=context.user_id, command=control.command_id))

    def read(self, stage):
        try:
            row = self.store.get_llm_proposal_evidence(self.prefix + "_" + stage)
        except KeyError:
            return None
        data = row["diagnostics"]
        if data.get("sha256") != content_hash(data.get("payload")):
            raise ValueError("REEXECUTION_EVIDENCE_INTEGRITY")
        payload = data["payload"]
        if stage in {"prepare", "observe"}:
            from contracts.model_gateway import ModelReexecutionRecordV1, canonical_hash
            record = ModelReexecutionRecordV1.model_validate(payload["record"] if stage == "prepare" else payload)
            if record.control != self.control or (record.organization_id,record.workspace_id,record.actor_ref) != (
                self.context.organization_id,self.context.workspace_id,"user:"+self.context.user_id):
                raise ValueError("REEXECUTION_MATERIAL_CONFLICT")
            if stage == "prepare" and (record.request_hash != canonical_hash(payload["command"]["request"])
                    or record.material_hash != canonical_hash(payload["material"])):
                raise ValueError("REEXECUTION_EVIDENCE_INTEGRITY")
        return payload

    def retain(self, stage, payload):
        prior = self.read(stage)
        if prior is not None:
            if prior != payload:
                raise ValueError("REEXECUTION_MATERIAL_CONFLICT")
            return prior
        try:
            self.store.record_llm_proposal_evidence(
                semantica_commit=SEMANTICA_COMMIT, skill_major_version=SKILL_MAJOR_VERSION,
                provider_contract_version="graph-intent-reexecution.v1", provider="semantier_gateway",
                model="durable-command", status="unavailable", evidence_id=self.prefix + "_" + stage,
                diagnostics={"stage": stage, "payload": payload, "sha256": content_hash(payload)})
        except Exception:
            # A concurrent delivery may win the unique evidence id. Compare its
            # complete immutable value; never suppress an unrelated store error.
            prior = self.read(stage)
            if prior is None:
                raise
            if prior != payload:
                raise ValueError("REEXECUTION_MATERIAL_CONFLICT")
        return payload

    def bind_material(self, *, prompt, instructions, schema, provider, model):
        material = dict(input=prompt, instructions=instructions, output_schema=schema)
        self.retain("material", dict(control=self.control.model_dump(mode="json"),
            organization_id=self.context.organization_id, workspace_id=self.context.workspace_id,
            actor_ref="user:" + self.context.user_id, provider=provider, model=model,
            transport_fingerprint=self.transport_fingerprint, material=material))
        return material

    def prepare(self, command, material):
        from contracts.model_gateway import ModelReexecutionRecordV1, canonical_hash
        from datetime import datetime, timezone
        prior = self.read("prepare")
        # The material row wins atomically before PREPARE. Its persisted time
        # keeps concurrent first deliveries byte-identical, rather than treating
        # different local clock readings as changed command intent.
        prepared_at = (prior["record"]["created_at"] if prior else
            datetime.fromisoformat(self.store.get_llm_proposal_evidence(
                self.prefix + "_material")["observed_at"]).astimezone(timezone.utc).isoformat().replace("+00:00", "Z"))
        basis = dict(schema_version="model_reexecution_record.v1",
            organization_id=self.context.organization_id, workspace_id=self.context.workspace_id,
            actor_ref="user:" + self.context.user_id, control=self.control.model_dump(mode="json"),
            command_hash=command.command_hash(self.control.predecessor_id,self.context.user_id,self.context.workspace_id),
            request_hash=canonical_hash(command.request), material_hash=canonical_hash(material),
            schema_pin=dict(artifact_kind="graph_intent_output_schema", artifact_ref="graph-intent-output-schema:" + content_hash(material["output_schema"]),
                artifact_hash="sha256:" + content_hash(material["output_schema"])), lineage=None,
            created_at=prepared_at)
        h=content_hash(basis)
        record=ModelReexecutionRecordV1(artifact_ref="model-reexecution:"+h,artifact_hash="sha256:"+h,**basis)
        self.retain("prepare",dict(record=record.model_dump(mode="json"),command=command.model_dump(mode="json"),material=material))

    def observe(self, lineage):
        from contracts.model_gateway import ModelReexecutionRecordV1
        if not lineage:
            raise ValueError("authoritative reexecution lineage required")
        basis={**self.read("prepare")["record"],"lineage":lineage}
        basis.pop("artifact_ref");basis.pop("artifact_hash")
        h=content_hash(basis)
        record=ModelReexecutionRecordV1(artifact_ref="model-reexecution:"+h,artifact_hash="sha256:"+h,**basis)
        self.retain("observe",record.model_dump(mode="json"))


class GraphSchemaLlmEvidenceRunner:
    """Capture pinned live LLM proposal evidence without making it authority."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        workspace_id: str,
        store: GraphSchemaDiscoveryStore | None = None,
        environ: Mapping[str, str] | None = None,
        generator_factory: GeneratorFactory | None = None,
        managed_auth_available: ManagedAuthAvailable | None = None,
        transport_fingerprint: str | None = None,
        reexecution_control=None,
    ) -> None:
        if environ is not None and not isinstance(environ, Mapping):
            raise ValueError("LLM proposal evidence environ must be a mapping")
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_id = workspace_id
        self.workspace_path = resolve_active_workspace_path(
            self.workspace_root,
            workspace_id=workspace_id,
        )
        self.store = store or GraphSchemaDiscoveryStore(
            self.workspace_root,
            workspace_id=workspace_id,
        )
        self.environ = environ if environ is not None else os.environ
        self.generator_factory = generator_factory
        self.managed_auth_available = managed_auth_available
        self.transport_fingerprint = str(transport_fingerprint or "").strip()
        self.reexecution_control = reexecution_control

    def run(
        self,
        *,
        provider: str,
        model: str,
        source_text: str,
        run_id: str | None = None,
        name: str = "TenderSensitiveGraphSchemaProposal",
        base_uri: str = "urn:semantier:ontology:tender-sensitive:",
        semantica_commit: str = SEMANTICA_COMMIT,
        skill_major_version: str = SKILL_MAJOR_VERSION,
        provider_contract_version: str = PROVIDER_CONTRACT_VERSION,
        evidence_id: str | None = None,
        structured_output_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._require(provider=provider, model=model, source_text=source_text)
        if run_id is not None:
            self._require_safe_artifact_id(run_id, label="run_id")
        if evidence_id is not None:
            self._require_safe_artifact_id(evidence_id, label="evidence_id")
        provider_key, model = _provider_model(provider, model)
        env_key = PROVIDER_ENV_KEYS.get(provider_key)
        has_managed_auth = bool(
            self.managed_auth_available is not None
            and self.managed_auth_available(provider_key)
        )
        if env_key and not self.environ.get(env_key) and not has_managed_auth:
            return self.store.record_llm_proposal_evidence(
                run_id=run_id,
                semantica_commit=semantica_commit,
                skill_major_version=skill_major_version,
                provider_contract_version=provider_contract_version,
                provider=provider_key,
                model=model,
                status="unavailable",
                evidence_id=evidence_id,
                diagnostics={
                    "reason": "provider credential is not configured",
                    "missing_env": env_key,
                },
            )

        generator = None
        try:
            generator = self._generator(provider_key, model)
            if (
                name == "GraphIntentProposalEvidence"
                and callable(
                    getattr(getattr(generator, "provider", None), "generate_structured", None)
                )
            ):
                ontology = generator.provider.generate_structured(
                    source_text,
                    model=model,
                    temperature=0,
                    **({"json_schema": structured_output_schema} if structured_output_schema is not None else {}),
                )
            else:
                ontology = generator.generate_ontology_from_text(
                    source_text,
                    name=name,
                    base_uri=base_uri,
                    version="1.0",
                    temperature=0,
                )
            if not isinstance(ontology, dict):
                raise ValueError("LLM proposal output must be an object")
            if self.reexecution_control is not None and evidence_id is not None:
                try:
                    return self.store.get_llm_proposal_evidence(evidence_id)
                except KeyError:
                    pass
            proposal = self._proposal_payload(
                ontology=ontology,
                provider=provider_key,
                model=model,
                semantica_commit=semantica_commit,
                skill_major_version=skill_major_version,
                provider_contract_version=provider_contract_version,
                run_id=run_id,
            )
            proposal_hash = f"sha256:{content_hash(proposal)}"
            proposal_ref = self._write_proposal(proposal, proposal_hash, run_id=run_id)
            class_count = len(ontology.get("classes") or [])
            property_count = len(ontology.get("properties") or [])
            structured_graph_intent = isinstance(
                ontology.get("graph_intent_proposal"), dict
            )
            status = (
                "passed"
                if class_count > 0 or structured_graph_intent
                else "disagreement"
            )
            diagnostics = {
                "class_count": class_count,
                "property_count": property_count,
                "structured_graph_intent": structured_graph_intent,
                "artifact_ref": proposal_ref,
                **self._structured_attempt_diagnostics(generator),
            }
            if status == "disagreement":
                diagnostics["reason"] = (
                    "LLM proposal did not produce any ontology classes"
                )
            return self.store.record_llm_proposal_evidence(
                run_id=run_id,
                semantica_commit=semantica_commit,
                skill_major_version=skill_major_version,
                provider_contract_version=provider_contract_version,
                provider=provider_key,
                model=model,
                status=status,
                proposal_ref=proposal_ref,
                proposal_hash=proposal_hash,
                evidence_id=evidence_id,
                diagnostics=diagnostics,
            )
        except Exception as exc:
            if str(exc) == "REEXECUTION_MATERIAL_CONFLICT":
                raise
            if self.reexecution_control is not None and evidence_id is not None:
                try:
                    return self.store.get_llm_proposal_evidence(evidence_id)
                except KeyError:
                    pass
            if provider_key == "ollama" and (
                isinstance(exc, ModuleNotFoundError) or "ollama" in str(exc).lower()
            ):
                return self.store.record_llm_proposal_evidence(
                    run_id=run_id,
                    semantica_commit=semantica_commit,
                    skill_major_version=skill_major_version,
                    provider_contract_version=provider_contract_version,
                    provider=provider_key,
                    model=model,
                    status="unavailable",
                    evidence_id=evidence_id,
                    diagnostics={
                        "reason": "open-source local LLM provider is not available",
                        "provider": provider_key,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
            return self.store.record_llm_proposal_evidence(
                run_id=run_id,
                semantica_commit=semantica_commit,
                skill_major_version=skill_major_version,
                provider_contract_version=provider_contract_version,
                provider=provider_key,
                model=model,
                status="failed",
                evidence_id=evidence_id,
                diagnostics={
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    **self._structured_attempt_diagnostics(generator),
                },
            )

    @staticmethod
    def _structured_attempt_diagnostics(generator: Any) -> dict[str, Any]:
        attempts = getattr(getattr(generator, "provider", None), "generation_attempts", None)
        return {"structured_output_attempts": deepcopy(attempts)} if attempts else {}

    def _generator(self, provider: str, model: str) -> Any:
        provider, model = _provider_model(provider, model)
        if self.generator_factory is not None:
            return self.generator_factory(provider, model)
        _ensure_vendored_semantica_importable()
        from semantica.ontology.llm_generator import LLMOntologyGenerator

        if provider == "alibaba" or (
            provider == "openai" and self.environ.get("OPENAI_BASE_URL")
        ):
            from semantica.semantic_extract import providers

            generator = LLMOntologyGenerator(provider=None, model=model)
            generator.provider_name = provider
            api_key_env = (
                "DASHSCOPE_API_KEY" if provider == "alibaba" else "OPENAI_API_KEY"
            )
            base_url = (
                self.environ.get("DASHSCOPE_BASE_URL")
                or "https://dashscope.aliyuncs.com/compatible-mode/v1"
                if provider == "alibaba"
                else self.environ.get("OPENAI_BASE_URL")
            )
            if provider == "alibaba" and model == DEEPSEEK_MAAS_MODEL:
                base_url = DEEPSEEK_MAAS_BASE_URL
            generator.provider = providers.OpenAIProvider(
                api_key=self.environ.get(api_key_env),
                base_url=base_url,
                model=model,
            )
            return generator

        return LLMOntologyGenerator(provider=provider, model=model)

    def _write_proposal(
        self,
        proposal: dict[str, Any],
        proposal_hash: str,
        *,
        run_id: str | None,
    ) -> str:
        safe_run_id = run_id or "ad-hoc"
        self._require_safe_artifact_id(safe_run_id, label="run_id")
        filename = f"{safe_run_id}-{proposal_hash.removeprefix('sha256:')[:16]}.json"
        relative_ref = f"artifacts/knowledge-graph-building/llm-proposals/{filename}"
        path = self.workspace_path / relative_ref
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(canonical_json(proposal), encoding="utf-8")
        return relative_ref

    @staticmethod
    def _require_safe_artifact_id(value: str, *, label: str) -> None:
        if not isinstance(value, str):
            raise ValueError(
                f"LLM proposal evidence {label} must be a safe artifact id"
            )
        path = Path(value)
        if (
            not value.strip()
            or path.is_absolute()
            or "/" in value
            or "\\" in value
            or ".." in path.parts
        ):
            raise ValueError(
                f"LLM proposal evidence {label} must be a safe artifact id"
            )

    @staticmethod
    def _proposal_payload(
        *,
        ontology: dict[str, Any],
        provider: str,
        model: str,
        semantica_commit: str,
        skill_major_version: str,
        provider_contract_version: str,
        run_id: str | None,
    ) -> dict[str, Any]:
        return {
            "schema_version": "semantica_llm_ontology_proposal.v1",
            "run_id": run_id,
            "provider": provider,
            "model": model,
            "semantica_commit": semantica_commit,
            "skill_major_version": skill_major_version,
            "provider_contract_version": provider_contract_version,
            "ontology": ontology,
        }

    @staticmethod
    def _require(**values: str) -> None:
        missing = sorted(
            name
            for name, value in values.items()
            if not isinstance(value, str) or not value.strip()
        )
        if missing:
            raise ValueError(f"LLM proposal evidence run missing fields: {missing}")
