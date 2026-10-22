"""Compose source schema production with the existing current-agent adapter."""
from pathlib import Path

from agents.hermes_current_agent_llm import HermesCurrentAgentStructuredLlm
from runtime_paths import platform_runtime_root
from services.normative_schema_production_service import NormativeSchemaProductionService
from services.normative_semantic_production_service import NormativeSemanticProductionService
from services.source_validation_case_production_service import SourceValidationCaseProductionService


def build_normative_schema_producer(*, inference_home: Path | None = None):
    home = inference_home or platform_runtime_root()
    binding = HermesCurrentAgentStructuredLlm._load_auxiliary_binding(home)
    return NormativeSchemaProductionService(
        llm=HermesCurrentAgentStructuredLlm(hermes_home=home),
        provider=binding["provider"], model=binding["model"],
    )


def build_normative_semantic_producer(*, inference_home: Path | None = None):
    home = inference_home or platform_runtime_root()
    binding = HermesCurrentAgentStructuredLlm._load_auxiliary_binding(home)
    return NormativeSemanticProductionService(
        llm=HermesCurrentAgentStructuredLlm(hermes_home=home),
        provider=binding["provider"], model=binding["model"],
    )


def build_source_validation_case_producer(*, inference_home: Path | None = None):
    home = inference_home or platform_runtime_root()
    binding = HermesCurrentAgentStructuredLlm._load_auxiliary_binding(home)
    return SourceValidationCaseProductionService(
        llm=HermesCurrentAgentStructuredLlm(hermes_home=home),
        provider=binding["provider"], model=binding["model"],
    )
