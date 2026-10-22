"""Current-agent Hermes structured completion adapter for Semantier features."""

from __future__ import annotations

from dataclasses import asdict
import asyncio
from copy import deepcopy
from contextvars import ContextVar
from hashlib import sha256
import os
from pathlib import Path
import posixpath
from urllib.parse import unquote, urlsplit
from typing import Any, Mapping

import yaml

from agent.auxiliary_client import async_call_llm, _scoped_key_env
from agent.plugin_llm import PluginLlm, PluginLlmTextInput, _extract_text

from contracts.ai_grounding import StructuredGenerationError, StructuredGenerationResult, NoCompatibleModelError


class HermesCurrentAgentStructuredLlm:
    _AUXILIARY_TASK = "contextgraph_ai_ground"

    def __init__(
        self,
        plugin_llm: PluginLlm | None = None,
        *,
        hermes_home: Path | None = None,
        require_primary_binding: bool = True,
    ) -> None:
        self._hermes_home = hermes_home
        self._raw_response = ContextVar("semantier_structured_raw_response", default=None)
        self._compatibility_profiles = []
        if plugin_llm is not None:
            self._llm = plugin_llm
            return
        if hermes_home is None:
            raise StructuredGenerationError("AI Ground workspace Hermes home is required")
        binding = self._load_auxiliary_binding(hermes_home, require_primary_binding=require_primary_binding)
        self._compatibility_profiles = deepcopy(binding.get("compatible_execution_profiles", []))
        from agents import compatible_model_transport
        from contracts import runtime_judgment_execution
        import agent.plugin_llm
        import agent.anthropic_adapter
        from importlib.metadata import version
        self._compatibility_adapter_hash = "sha256:" + sha256(
            (version("anthropic") + ":" + version("httpx")).encode() + b"".join(
            Path(path).read_bytes() for path in (str(Path(__file__)), compatible_model_transport.__file__,
                agent.plugin_llm.__file__, agent.anthropic_adapter.__file__,
                runtime_judgment_execution.__file__))).hexdigest()

        async def call_workspace_auxiliary(**kwargs: Any) -> tuple[str, str, Any]:
            if binding.get("generic_gateway_primary"):
                raise StructuredGenerationError("qualified execution cannot use the generic model gateway")
            if binding["provider"] in {"", "auto"} or not binding["model"]:
                raise StructuredGenerationError("primary auxiliary binding is unresolved")
            extra_body = dict(binding.get("extra_body") or {})
            extra_body.update(kwargs.get("extra_body") or {})
            route_info: dict[str, str] = {}
            response = await async_call_llm(
                task=self._AUXILIARY_TASK,
                provider=binding["provider"],
                model=binding["model"],
                base_url=binding.get("base_url") or None,
                api_key=binding.get("api_key") or None,
                messages=kwargs["messages"],
                temperature=kwargs.get("temperature"),
                max_tokens=kwargs.get("max_tokens"),
                timeout=kwargs.get("timeout") or binding.get("timeout"),
                extra_body=extra_body,
                route_info=route_info,
            )
            resolved_provider = str(route_info.get("provider") or "").strip()
            resolved_model = str(route_info.get("model") or "").strip()
            if (resolved_provider, resolved_model) != (binding["provider"], binding["model"]):
                raise StructuredGenerationError(
                    "AI Ground provider/model drifted from the exact configured binding"
                )
            capture = self._raw_response.get()
            if capture is not None:
                capture.update(text=_extract_text(response), provider=resolved_provider, model=resolved_model)
                usage = getattr(response, "usage", None)
                capture["usage"] = usage.model_dump() if hasattr(usage, "model_dump") else (vars(usage) if hasattr(usage, "__dict__") else None)
            return resolved_provider, resolved_model, response

        self._llm = PluginLlm(
            plugin_id="tender_contextgraph_builder",
            async_caller=call_workspace_auxiliary,
        )

    @classmethod
    def _load_auxiliary_binding(cls, hermes_home: Path, *, require_primary_binding: bool = True) -> dict[str, Any]:
        config_path = hermes_home.expanduser().resolve() / "config.yaml"
        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise StructuredGenerationError(
                f"AI Ground auxiliary configuration unavailable: {config_path}"
            ) from exc
        config = loaded if isinstance(loaded, dict) else {}
        auxiliary = config.get("auxiliary")
        task = auxiliary.get(cls._AUXILIARY_TASK) if isinstance(auxiliary, dict) else None
        task = task if isinstance(task, dict) else {}
        model_config = config.get("model")
        model_config = model_config if isinstance(model_config, dict) else {}
        provider = str(task.get("provider") or "").strip()
        model = str(task.get("model") or "").strip()
        if provider in {"", "auto"}:
            provider = str(model_config.get("provider") or "").strip()
        if not model:
            model = str(model_config.get("default") or model_config.get("model") or "").strip()
        if os.environ.get("SEMANTIER_F10_FIXTURE") == "1" and (
            not provider or provider == "auto" or not model
        ):
            provider = os.environ.get(
                "SEMANTIER_F10_AUXILIARY_PROVIDER", ""
            ).strip()
            model = os.environ.get("SEMANTIER_F10_AUXILIARY_MODEL", "").strip()
        fixture_base_url = ""
        fixture_api_key = ""
        if os.environ.get("SEMANTIER_F10_FIXTURE") == "1":
            fixture_base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
            fixture_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if provider == "openai" and (task.get("base_url") or fixture_base_url):
                provider = "custom"
        if require_primary_binding and (not provider or provider == "auto" or not model):
            raise StructuredGenerationError(
                "auxiliary.contextgraph_ai_ground must resolve a provider and model: "
                f"{config_path}"
            )
        base_url = str(task.get("base_url") or model_config.get("base_url") or fixture_base_url).strip()
        generic_gateway_primary = False
        gateway = config.get("semantier_gateway") or {}
        if isinstance(gateway, dict) and gateway.get("enabled") is True and gateway.get("base_url"):
            def endpoint_identity(value):
                parsed = urlsplit(str(value or ""))
                return (parsed.scheme.lower(), parsed.hostname,
                        parsed.port or {"https": 443, "http": 80}.get(parsed.scheme.lower()),
                        posixpath.normpath(unquote(parsed.path)).rstrip("/"))

            gateway_endpoint = endpoint_identity(gateway["base_url"])
            profiles = task.get("compatible_execution_profiles") or []
            endpoints = [base_url]
            profile_endpoints = [profile.get("base_url", "") for profile in profiles if isinstance(profile, dict)]
            configured_provider = (config.get("providers") or {}).get(provider, {})
            if isinstance(configured_provider, dict):
                endpoints.append(configured_provider.get("base_url") or configured_provider.get("api")
                                 or configured_provider.get("url") or "")
            generic_gateway_primary = any(
                endpoint and endpoint_identity(endpoint) == gateway_endpoint for endpoint in endpoints)
            if ((require_primary_binding and generic_gateway_primary)
                    or any(endpoint and endpoint_identity(endpoint) == gateway_endpoint for endpoint in profile_endpoints)):
                raise StructuredGenerationError("qualified execution cannot use the generic model gateway")
        api_key = str(task.get("api_key") or "").strip()
        key_env = str(task.get("key_env") or task.get("api_key_env") or "").strip()
        if not api_key and key_env:
            api_key = _scoped_key_env(key_env)
            if not api_key:
                raise StructuredGenerationError("configured credential is unavailable")
        return {
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "generic_gateway_primary": generic_gateway_primary,
            "api_key": api_key or fixture_api_key,
            "timeout": task.get("timeout"),
            "extra_body": task.get("extra_body") if isinstance(task.get("extra_body"), dict) else {},
            "compatible_execution_profiles": task.get("compatible_execution_profiles", []),
        }

    async def generate_compatible_structured(
        self, *, instructions: str, input_text: str, json_schema: Mapping[str, Any],
        record, schema_name: str = "runtime_semantic_judgment_v1", purpose: str = "runtime_semantic_judgment",
    ) -> StructuredGenerationResult:
        """Use only configured profiles and count the final Messages SDK body.

        The caller owns scoped projection verification and durable receipts.
        Generic auxiliary fallbacks cannot bypass this explicit profile list.
        """
        import httpx
        from agent.auxiliary_client import resolve_provider_client, AnthropicAuxiliaryClient
        from agents.compatible_model_transport import (
            CompleteRequestCompatibilityTransport, CompatibilityFailure, PROFILE_FIELDS, _texts,
            profile_resolved,
        )

        if self._hermes_home is None:
            raise StructuredGenerationError("COMPATIBILITY_PROFILE_UNRESOLVED")
        profiles = deepcopy(self._compatibility_profiles)
        if not isinstance(profiles, list) or not profiles:
            raise NoCompatibleModelError("NO_COMPATIBLE_MODEL: no authorized execution profiles")
        receipts = []
        raw_response = {}
        import threading
        owner_loop = asyncio.get_running_loop()
        owner_thread = threading.get_ident()

        def retain_on_owner(snapshot):
            record(deepcopy(snapshot))
            receipts.append(snapshot)

        async def retain_async(snapshot):
            retain_on_owner(snapshot)

        def retain(receipt):
            snapshot = deepcopy(receipt)
            # Persist on the caller's loop before the transport invokes the model.
            if threading.get_ident() == owner_thread:
                retain_on_owner(snapshot)
            else:
                asyncio.run_coroutine_threadsafe(retain_async(snapshot), owner_loop).result()

        async def invoke(**kwargs):
            for configured in profiles:
                if not isinstance(configured, dict):
                    retain(dict(schema_version="complete_request_compatibility.v1",
                        profile={}, outcome="NO_COMPATIBLE_MODEL", reason="PROFILE_UNRESOLVED"))
                    continue
                profile = {key: value for key, value in configured.items() if key in PROFILE_FIELDS}
                if (set(configured) != PROFILE_FIELDS | {"api_key_env"}
                    or not profile_resolved(profile) or not isinstance(configured["api_key_env"], str)):
                    retain(dict(schema_version="complete_request_compatibility.v1",
                        profile=profile, outcome="NO_COMPATIBLE_MODEL", reason="PROFILE_UNRESOLVED"))
                    continue
                credential = os.environ.get(configured.get("api_key_env", ""), "")
                if not credential:
                    retain(dict(schema_version="complete_request_compatibility.v1",
                        profile=profile, outcome="NO_COMPATIBLE_MODEL", reason="CREDENTIAL_UNAVAILABLE"))
                    continue
                client, model = resolve_provider_client(profile["provider"], profile["model"],
                    explicit_base_url=profile["base_url"], explicit_api_key=credential,
                    api_mode="anthropic_messages")
                if not isinstance(client, AnthropicAuxiliaryClient) or model != profile["model"]:
                    if client is not None:
                        client.close()
                    retain(dict(schema_version="complete_request_compatibility.v1",
                        profile=profile, outcome="NO_COMPATIBLE_MODEL", reason="ADAPTER_UNSUPPORTED"))
                    continue
                http = httpx.Client(transport=CompleteRequestCompatibilityTransport(
                    transport=httpx.HTTPTransport(retries=0), profile=profile,
                    required_texts=tuple(_texts({"messages": kwargs["messages"]}).elements()),
                    record=retain), timeout=120, follow_redirects=False)
                sdk = client._real_client.with_options(http_client=http, max_retries=0)
                guarded = AnthropicAuxiliaryClient(sdk, model, credential, profile["base_url"])
                try:
                    response = await asyncio.to_thread(guarded.chat.completions.create,
                        model=model, messages=kwargs["messages"], temperature=kwargs.get("temperature"),
                        max_tokens=profile.get("required_output_reserve"),
                        extra_body={**(kwargs.get("extra_body") or {}), "thinking": {"type": "disabled"}})
                    raw_response.update(text=_extract_text(response), provider=profile['provider'], model=model)
                    if response.choices[0].finish_reason != "stop":
                        raise StructuredGenerationError("COMPATIBILITY_OUTPUT_INCOMPLETE", raw_response=raw_response)
                    return profile["provider"], model, response
                except Exception as exc:
                    cause = exc
                    while cause.__cause__ is not None:
                        cause = cause.__cause__
                    if not isinstance(cause, CompatibilityFailure):
                        raise
                    # Recount the unchanged PluginLlm material under the next
                    # configured profile. Never use an implicit provider fallback.
                finally:
                    guarded.close()
                    client.close()
            raise NoCompatibleModelError("NO_COMPATIBLE_MODEL")

        plugin = PluginLlm(plugin_id="tender_contextgraph_builder", async_caller=invoke)
        try:
            result = await plugin.acomplete_structured(instructions=instructions,
                input=[PluginLlmTextInput(text=input_text)], json_schema=dict(json_schema),
                json_mode=True, schema_name=schema_name, temperature=0.0,
                purpose=purpose)
        except ValueError as exc:
            if not raw_response:
                raise
            raise StructuredGenerationError("Hermes structured response failed validation",
                raw_response=raw_response) from exc
        if result.content_type != "json" or not isinstance(result.parsed, dict):
            raise StructuredGenerationError("Hermes returned no parsed JSON object", raw_response=raw_response or None)
        return StructuredGenerationResult(parsed=result.parsed, text=result.text,
            provider=result.provider, model=result.model, agent_id=result.agent_id,
            usage=asdict(result.usage), audit={**dict(result.audit), "compatibility": receipts[-1]})

    @property
    def compatibility_execution_identity(self):
        from contracts.runtime_judgment_execution import PROFILE_FIELDS
        profiles = self._compatibility_profiles
        safe = [{key: deepcopy(value) for key, value in item.items()
                 if key in PROFILE_FIELDS | {"api_key_env"}}
                if isinstance(item, dict) else {} for item in profiles] if isinstance(profiles, list) else []
        valid_shape = isinstance(profiles, list) and all(isinstance(item, dict)
            and set(item) == PROFILE_FIELDS | {"api_key_env"} for item in profiles)
        return {"profiles": safe, "configuration_valid": valid_shape,
                "adapter_hash": getattr(self, "_compatibility_adapter_hash", None)}

    async def generate_structured(
        self,
        *,
        instructions: str,
        input_text: str,
        json_schema: Mapping[str, Any],
        schema_name: str = "semantier_ai_grounding_batch_v1",
        purpose: str = _AUXILIARY_TASK,
    ) -> StructuredGenerationResult:
        capture = {}
        token = self._raw_response.set(capture)
        try:
            result = await self._llm.acomplete_structured(
                instructions=instructions,
                input=[PluginLlmTextInput(text=input_text)],
                json_schema=dict(json_schema),
                json_mode=True,
                schema_name=schema_name,
                temperature=0.0,
                purpose=purpose,
            )
            if result.content_type != "json" or not isinstance(result.parsed, dict):
                raise StructuredGenerationError("Hermes returned no parsed JSON object",
                    raw_response=capture or None)
        except ValueError as exc:
            if not capture:
                raise
            raise StructuredGenerationError("Hermes structured response failed validation",
                raw_response=capture) from exc
        finally:
            self._raw_response.reset(token)
        if not result.provider or not result.model:
            raise StructuredGenerationError("Hermes returned incomplete provider attribution")
        return StructuredGenerationResult(
            parsed=result.parsed,
            text=result.text,
            provider=result.provider,
            model=result.model,
            agent_id=result.agent_id,
            usage=asdict(result.usage),
            audit=dict(result.audit),
        )
