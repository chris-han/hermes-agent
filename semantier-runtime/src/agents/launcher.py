from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Sequence

import yaml

from agents.bootstrap_repair import record_clean_eos_shutdown
from agents.runtime_memory_boundary import sanitize_user_memory_profile
from agents.hermes_bundled_skill_adapter import forbidden_global_external_skill_roots

_ANALYTICS_PLUGIN_NAME = "governed_analytics"
_DATA_ENGINEERING_PLUGIN_NAME = "governed_data_engineering"
_ROUTING_GUARD_PLUGIN_NAME = "semantier_routing_guard"
_REAL_COMPANY_ONBOARDING_PLUGIN_NAME = "real_company_onboarding"
_NIHAIXIA_PLUGIN_NAME = "nihaixia"
_FEISHU_MEETING_COORDINATOR_PLUGIN_NAME = "feishu_meeting_coordinator"
_AUTO_RESUME_SCREENING_PLUGIN_NAME = "auto_resume_screening"
_REGULATORY_SOURCE_MONITOR_PLUGIN_NAME = "regulatory_source_monitor"
_VC_GITHUB_OPPORTUNITY_RADAR_PLUGIN_NAME = "vc_github_opportunity_radar"
_DOCUMENT_EXTRACTION_PLUGIN_NAME = "document_extraction"
_FINANCIAL_DOCUMENTS_PLUGIN_NAME = "financial_documents"
_AUTOMATE_EXCEL_PLUGIN_NAME = "automate_excel"
_KNOWLEDGE_PARSER_PLUGIN_NAME = "knowledge_parser"
_KNOWLEDGE_SENSITIVE_LABELING_PLUGIN_NAME = "knowledge_sensitive_labeling"
_TENDER_SENSITIVE_TERM_DETECTION_MODELING_PLUGIN_NAME = "tender_sensitive_term_detection_modeling"
_GOVERNED_GRAPH_PLUGIN_NAME = "governed_graph"
_TENDER_RUNTIME_EXPERT_PLUGIN_NAME = "tender_contextgraph_runtime"
_CORPUS_GRAPH_SCHEMA_INDUCTION_PLUGIN_NAME = "corpus_graph_schema_induction"
_TENDER_JOURNEY_PLUGIN_NAME = "tender_journey"
_RUNTIME_PLUGIN_NAMES: tuple[str, ...] = (
    _ANALYTICS_PLUGIN_NAME,
    _DATA_ENGINEERING_PLUGIN_NAME,
    _ROUTING_GUARD_PLUGIN_NAME,
    _REAL_COMPANY_ONBOARDING_PLUGIN_NAME,
    _NIHAIXIA_PLUGIN_NAME,
    _DOCUMENT_EXTRACTION_PLUGIN_NAME,
    _FINANCIAL_DOCUMENTS_PLUGIN_NAME,
    _AUTOMATE_EXCEL_PLUGIN_NAME,
    _KNOWLEDGE_PARSER_PLUGIN_NAME,
    _KNOWLEDGE_SENSITIVE_LABELING_PLUGIN_NAME,
    _TENDER_SENSITIVE_TERM_DETECTION_MODELING_PLUGIN_NAME,
    _GOVERNED_GRAPH_PLUGIN_NAME,
    _CORPUS_GRAPH_SCHEMA_INDUCTION_PLUGIN_NAME,
    _TENDER_JOURNEY_PLUGIN_NAME,
)
_OPTIONAL_RUNTIME_PLUGIN_NAMES: tuple[str, ...] = (
    "sensevoice_stt",
    _FEISHU_MEETING_COORDINATOR_PLUGIN_NAME,
    _AUTO_RESUME_SCREENING_PLUGIN_NAME,
    _REGULATORY_SOURCE_MONITOR_PLUGIN_NAME,
    _VC_GITHUB_OPPORTUNITY_RADAR_PLUGIN_NAME,
    _AUTOMATE_EXCEL_PLUGIN_NAME,
)
_PLUGIN_SKILL_EXTERNAL_DIR_NAMES: tuple[str, ...] = tuple(
    dict.fromkeys((*_RUNTIME_PLUGIN_NAMES, *_OPTIONAL_RUNTIME_PLUGIN_NAMES))
)
_MEETING_COORDINATOR_PROFILE_NAMES: tuple[str, ...] = (
    "meeting-coordinator",
)
_TENDER_RUNTIME_EXPERT_PROFILE_NAME = "tender-runtime-expert"
_RUNTIME_SKILL_PATHS: tuple[tuple[str, ...], ...] = (
    ("productivity", "expense-reimbursement"),
    ("governance", "knowledge-graph-building"),
    ("governance", "tender-review-journey"),
)
_LEGACY_RUNTIME_SKILL_PATHS: tuple[tuple[str, ...], ...] = (
    ("semantier", "governed-query"),
    ("read-sim-json",),
    ("governance", "corpus-graph-schema-induction"),
    ("governance", "graph-schema-discovery"),
)
_DEFAULT_RUNTIME_ENV: dict[str, str] = {
    "SEMANTIER_LOCAL_STATE_DIR": ".semantier-home",
    "SEMANTIER_AUTH_DB_PATH": ".semantier-home/auth.db",
    "SEMANTIER_EXPOSE_HERMES_BUNDLED_SKILLS": "1",
}
_REMOVED_RUNTIME_ENV_KEYS: tuple[str, ...] = (
    "API_SERVER_ENABLED",
    "API_SERVER_HOST",
    "API_SERVER_PORT",
    "HERMES_HOME",
)
_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts" / "agents"


def _load_prompt(filename: str) -> str:
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8")


def load_agent_prompt(filename: str) -> str:
    return _load_prompt(filename)


def _default_soul_md() -> str:
    return _load_prompt("SOUL.md")


def _default_governance_md() -> str:
    return _load_prompt("GOVERNANCE.md")


def load_default_soul_md() -> str:
    return _default_soul_md()


def load_default_governance_md() -> str:
    return _default_governance_md()
RUNTIME_EMBED_GATEWAY_ENV = "SEMANTIER_RUNTIME_EMBED_GATEWAY"


def _missing_hermes_dependency(exc: Exception) -> ImportError:
    err = ImportError(
        "hermes-agent is not installed. Add it as a workspace dependency and sync the environment."
    )
    err.__cause__ = exc
    return err


def _run_with_argv(entrypoint: Callable[[], int | None], argv: Sequence[str]) -> int:
    previous_argv = sys.argv[:]
    try:
        sys.argv = list(argv)
        result = entrypoint()
        return 0 if result is None else int(result)
    finally:
        sys.argv = previous_argv


def _packaged_hermes_root() -> Path | None:
    from pm.features import runtime_distribution_extras
    if runtime_distribution_extras() is None:
        return None
    from pm.paths import repo_root
    root = repo_root().resolve()
    if Path(__file__).resolve().parents[2] != root / "semantier-runtime":
        raise RuntimeError("packaged Semantier source is outside its bound composition")
    if not (root / "semantier-runtime/composition.json").is_file():
        raise RuntimeError("packaged Semantier composition manifest is missing")
    return root


def _repo_runtime_root() -> Path:
    if _packaged_hermes_root() is not None:
        from hermes_constants import get_default_hermes_root
        return Path(os.environ.get("SEMANTIER_LOCAL_STATE_DIR") or
                    get_default_hermes_root() / ".semantier-home").resolve()
    return Path(__file__).resolve().parents[2] / ".semantier-home"


def _repo_hermes_agent_root() -> Path:
    return _packaged_hermes_root() or Path(__file__).resolve().parents[2] / "hermes-agent"


def _repo_source_root() -> Path:
    return Path(__file__).resolve().parents[2] / "src"


def _assert_runtime_checkout_coherence() -> None:
    """Fail closed when a checkout-local venv imports Semantier from another checkout."""
    prefix = Path(sys.prefix)
    if prefix.name not in {".venv", "venv"}:
        return
    checkout = prefix.parent.resolve()
    if not (checkout / "pyproject.toml").exists():
        return
    launcher_checkout = Path(__file__).resolve().parents[2]
    if launcher_checkout != checkout:
        raise RuntimeError(
            "Semantier checkout/venv mismatch: "
            f"launcher={launcher_checkout} venv_checkout={checkout}. "
            "Recreate the checkout-local .venv instead of sharing or symlinking another checkout venv."
        )


def _ensure_repo_hermes_agent_on_path() -> None:
    """Prefer the repository Hermes Agent checkout for Semantier runs.

    ``uv`` installs path dependencies as wheels unless they are explicitly
    editable. The Semantier repo carries local Hermes Agent changes, so the
    runtime wrapper must place the checkout ahead of site-packages before
    importing ``hermes_cli`` or ``gateway.run``. Keep repo source paths ahead
    of the checkout because Semantier owns top-level packages such as
    ``plugins`` that would otherwise be shadowed by Hermes Agent packages.
    """
    hermes_agent_root = _repo_hermes_agent_root()
    if not hermes_agent_root.exists():
        return
    resolved = hermes_agent_root.resolve()
    existing_checkout_indexes = []
    for index, existing in enumerate(sys.path):
        try:
            if Path(existing).resolve() == resolved:
                existing_checkout_indexes.append(index)
        except (OSError, RuntimeError):
            continue
    for index in reversed(existing_checkout_indexes):
        del sys.path[index]
    repo_roots = {
        Path(__file__).resolve().parents[2].resolve(),
        _repo_source_root().resolve(),
    }
    insert_at = 0
    for index, existing in enumerate(sys.path):
        try:
            if Path(existing or ".").resolve() in repo_roots:
                insert_at = index + 1
        except (OSError, RuntimeError):
            continue
    sys.path.insert(insert_at, str(hermes_agent_root))
    # Freeze platform-owned reference names before native profile dotenv/config
    # can change process environment. Only names are retained, never values.
    from tools.environments.local_env_policy import protect_platform_secret_names
    protect_platform_secret_names(os.environ)


def _repo_plugin_source_root() -> Path:
    return Path(__file__).resolve().parents[1] / "plugins"


def _repo_skill_source_root() -> Path:
    return Path(__file__).resolve().parents[1] / "skills"


def _repo_marketplace_plugin_source_root() -> Path:
    return Path(__file__).resolve().parents[2] / "semantier-skills" / "plugins"


def _repo_policy_candidate_source_root() -> Path:
    return Path(__file__).resolve().parents[2] / "bootstrap" / "common_knowledge" / "policy_candidates"


def _repo_doctrine_artifact_source_root() -> Path:
    return Path(__file__).resolve().parents[2] / "bootstrap" / "common_knowledge" / "t3-doctrine"


def _plugin_source_dir(plugin_name: str) -> Path:
    return _plugin_source_dir_with_fallback(plugin_name, allow_marketplace_fallback=True)


def _plugin_source_dir_with_fallback(
    plugin_name: str,
    *,
    allow_marketplace_fallback: bool,
) -> Path:
    source_dir = _repo_plugin_source_root() / plugin_name
    if source_dir.exists() or not allow_marketplace_fallback:
        return source_dir
    return _repo_marketplace_plugin_source_root() / plugin_name


def _read_runtime_config(runtime_root: Path) -> dict:
    config_path = runtime_root / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _write_runtime_config(runtime_root: Path, config: dict) -> None:
    runtime_root.mkdir(parents=True, exist_ok=True)
    (runtime_root / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _load_db_workspace_config(workspace_id: str | None) -> dict:
    if not workspace_id:
        return {}

    from agents.auth_db import get_workspace_config

    return get_workspace_config(workspace_id)


def _merge_configs(base: dict, override: dict) -> dict:
    merged: dict[str, object] = {}
    for key, value in base.items():
        if isinstance(value, dict):
            merged[key] = dict(value)
        else:
            merged[key] = value
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _merge_configs(existing, value)
        else:
            merged[key] = value
    return merged


def _save_db_workspace_config(workspace_id: str | None, config: dict) -> None:
    if not workspace_id:
        return

    from agents.auth_db import save_workspace_config

    save_workspace_config(workspace_id, config)


def _migrate_workspace_config_file_to_db(runtime_root: Path, workspace_id: str | None) -> None:
    if not workspace_id:
        return
    if _load_db_workspace_config(workspace_id):
        return
    config = _read_runtime_config(runtime_root)
    if config:
        _save_db_workspace_config(workspace_id, config)


def _restore_workspace_configured_plugins(runtime_root: Path, config: dict) -> None:
    plugins_cfg = config.get("plugins")
    if not isinstance(plugins_cfg, dict):
        return
    enabled_plugins = plugins_cfg.get("enabled")
    if not isinstance(enabled_plugins, list):
        return
    for raw_plugin_name in enabled_plugins:
        plugin_name = str(raw_plugin_name or "").strip()
        if plugin_name not in _OPTIONAL_RUNTIME_PLUGIN_NAMES:
            continue
        source_dir = _plugin_source_dir(plugin_name)
        if not source_dir.exists():
            continue
        _install_repo_local_plugin(runtime_root, plugin_name)


def _plugin_skill_metadata(plugin_dir: Path) -> tuple[str, Path] | None:
    skill_path = plugin_dir / "SKILL.md"
    if not skill_path.exists():
        return None
    try:
        raw = skill_path.read_text(encoding="utf-8")
    except OSError:
        return None

    metadata: dict[str, object] = {}
    if raw.startswith("---\n"):
        end = raw.find("\n---", 4)
        if end != -1:
            try:
                parsed = yaml.safe_load(raw[4:end]) or {}
            except yaml.YAMLError:
                parsed = {}
            if isinstance(parsed, dict):
                metadata = parsed

    skill_name = str(metadata.get("name") or "").strip() or plugin_dir.name
    if not skill_name:
        return None
    return skill_name, skill_path


def _materialized_plugin_skill_dir(runtime_root: Path, *, skill_name: str) -> Path:
    return runtime_root / "skills" / "productivity" / skill_name


def _plugin_skill_external_dir(runtime_root: Path, plugin_name: str) -> Path | None:
    plugin_dir = runtime_root / "plugins" / plugin_name
    if (plugin_dir / "SKILL.md").exists():
        return plugin_dir
    return None


def _plugin_skill_external_dirs(runtime_root: Path) -> list[Path]:
    return [
        path
        for plugin_name in _PLUGIN_SKILL_EXTERNAL_DIR_NAMES
        if (path := _plugin_skill_external_dir(runtime_root, plugin_name)) is not None
    ]


def _install_repo_local_plugin(
    runtime_root: Path,
    plugin_name: str,
    *,
    allow_marketplace_fallback: bool = True,
) -> None:
    source_dir = _plugin_source_dir_with_fallback(
        plugin_name,
        allow_marketplace_fallback=allow_marketplace_fallback,
    )
    if not source_dir.exists():
        raise FileNotFoundError(
            f"required plugin source missing at {source_dir}; ensure repository checkout is complete"
        )

    plugins_dir = runtime_root / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)

    target_dir = plugins_dir / plugin_name
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
    _remove_materialized_plugin_skill(runtime_root, plugin_name)

def _install_repo_local_plugins(
    runtime_root: Path,
) -> None:
    """Backward-compatible helper used by tests and older callers."""
    for plugin_name in _RUNTIME_PLUGIN_NAMES:
        _install_repo_local_plugin(
            runtime_root,
            plugin_name,
            allow_marketplace_fallback=False,
        )


def _remove_materialized_plugin_skill(runtime_root: Path, plugin_name: str) -> None:
    plugin_dir = runtime_root / "plugins" / plugin_name
    metadata = _plugin_skill_metadata(plugin_dir)
    if metadata is None:
        return
    skill_name, _skill_path = metadata
    target_dir = _materialized_plugin_skill_dir(runtime_root, skill_name=skill_name)
    if target_dir.exists() and target_dir.is_dir():
        shutil.rmtree(target_dir)


def _remove_materialized_optional_plugin_skills(runtime_root: Path) -> None:
    for plugin_name in _OPTIONAL_RUNTIME_PLUGIN_NAMES:
        _remove_materialized_plugin_skill(runtime_root, plugin_name)


def _install_repo_local_skill(runtime_root: Path, relative_parts: tuple[str, ...]) -> None:
    source_dir = _repo_skill_source_root().joinpath(*relative_parts)
    if not source_dir.exists():
        return

    skills_dir = runtime_root / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    target_dir = skills_dir.joinpath(*relative_parts)
    if target_dir.exists() and target_dir.is_dir():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)


def _remove_repo_local_skill(runtime_root: Path, relative_parts: tuple[str, ...]) -> None:
    skills_dir = runtime_root / "skills"
    target_dir = skills_dir.joinpath(*relative_parts)
    if target_dir.exists() and target_dir.is_dir():
        shutil.rmtree(target_dir)


def _remove_legacy_repo_local_skills(runtime_root: Path) -> None:
    for relative_parts in _LEGACY_RUNTIME_SKILL_PATHS:
        _remove_repo_local_skill(runtime_root, relative_parts)


def _remove_managed_workspace_skill_copies(runtime_root: Path) -> None:
    for relative_parts in _RUNTIME_SKILL_PATHS:
        _remove_repo_local_skill(runtime_root, relative_parts)
    _remove_legacy_repo_local_skills(runtime_root)


def _sync_managed_workspace_plugin_links(
    runtime_root: Path, shared_plugins_external_dir: Path | None,
) -> None:
    """Expose shared code to Hermes' native home/plugins discovery, without copies.

    Links are discovery locations only. Native enabled/disabled gates and the
    authenticated execution boundary still control tool availability and IO.
    """
    plugins_dir = runtime_root / "plugins"
    for plugin_name in _RUNTIME_PLUGIN_NAMES:
        target_dir = plugins_dir / plugin_name
        source = shared_plugins_external_dir / plugin_name if shared_plugins_external_dir else None
        if target_dir.is_symlink():
            if source and (source / "plugin.yaml").is_file() and target_dir.resolve() == source.resolve():
                continue
            target_dir.unlink()
        elif target_dir.exists() and target_dir.is_dir():
            shutil.rmtree(target_dir)
        if source and (source / "plugin.yaml").is_file():
            plugins_dir.mkdir(parents=True, exist_ok=True)
            if os.name == "nt":
                shutil.copytree(source.resolve(), target_dir)
            else:
                target_dir.symlink_to(source.resolve(), target_is_directory=True)


def _remove_obsolete_hermes_local_dirs(runtime_root: Path) -> None:
    """Remove legacy Hermes runtime homes superseded by Semantier runtime roots."""
    candidates = (
        runtime_root / ".hermes-local",
        runtime_root.parent / ".hermes-local",
        _repo_hermes_agent_root() / ".hermes-local",
    )
    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_dir():
                shutil.rmtree(candidate)
        except OSError:
            continue


def _load_repo_doctrine_artifact_files(runtime_root: Path) -> None:
    source_root = _repo_doctrine_artifact_source_root()
    if not source_root.exists():
        return

    from eos.artifact_hashing import content_hash
    from eos.db import init_db
    from storage.kgl_store import KGLStore

    def _transition_guards(to_status: str) -> dict[str, bool] | None:
        if to_status == "VALIDATED":
            return {
                "schema_validation_pass": True,
                "source_registration_present": True,
                "authority_origin_valid": True,
            }
        if to_status == "APPROVED":
            return {
                "precedence_check_complete": True,
                "no_unresolved_higher_tier_conflict": True,
            }
        if to_status == "ACTIVE":
            return {
                "replay_dry_run_pass": True,
                "authority_origin_verified": True,
            }
        return None

    db_path = runtime_root / "eos.db"
    init_db(db_path)
    store = KGLStore(db_path=db_path)

    for path in sorted(source_root.rglob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            continue
        if not raw.get("semantic_tier") or not raw.get("source_type") or not raw.get("claims"):
            continue

        semantic_tier = str(raw.get("semantic_tier") or "").strip()
        if semantic_tier != "T3":
            continue

        relative_ref = str(path.relative_to(source_root.parent))
        artifact = store.propose_artifact(
            source_type=str(raw.get("source_type") or "").strip(),
            authority_domain=str(raw.get("authority_domain") or "").strip(),
            tag_authority_level=str(raw.get("tag_authority_level") or "").strip(),
            semantic_tier=semantic_tier,
            source_ref=str(raw.get("source_ref") or relative_ref),
            source_hash=content_hash(raw),
            authority_origin=str(raw.get("authority_origin") or "").strip() or None,
            source_registration={
                "source_uri": str(raw.get("source_ref") or relative_ref),
                "jurisdiction": str(raw.get("jurisdiction") or ""),
                "effective_from": str(raw.get("effective_from") or ""),
                "effective_to": raw.get("effective_to"),
                "source_version": str(raw.get("source_version") or ""),
                "anchors": list(raw.get("anchors") or []),
                "extraction_method": str(raw.get("extraction_method") or ""),
                "curator": str(raw.get("curator") or ""),
                "ambiguities": list(raw.get("ambiguities") or []),
                "source_provenance": raw.get("source_provenance"),
            },
            extracted_claims=list(raw.get("claims") or []),
            extracted_constraints=list(raw.get("constraints") or []),
            extracted_projection_rules=list(raw.get("projection_rules") or []),
        )

        current_status = str(artifact.get("ingestion_status") or "PROPOSED")
        transitions = (
            ("VALIDATED", ("domain_owner",)),
            ("APPROVED", ("domain_owner",)),
            ("ACTIVE", ("governance_chair", "domain_owner")),
        )
        for to_status, actor_roles in transitions:
            if current_status == to_status:
                continue
            valid_next = {
                "PROPOSED": "VALIDATED",
                "VALIDATED": "APPROVED",
                "APPROVED": "ACTIVE",
            }.get(current_status)
            if valid_next != to_status:
                continue
            artifact = store.transition_artifact(
                artifact["artifact_id"],
                to_status,  # type: ignore[arg-type]
                actor="bootstrap",
                notes=f"Loaded from {relative_ref}",
                actor_roles=actor_roles,
                guard_results=_transition_guards(to_status),
            )
            current_status = str(artifact.get("ingestion_status") or current_status)


def _load_repo_policy_candidate_files(runtime_root: Path) -> None:
    source_root = _repo_policy_candidate_source_root()
    if not source_root.exists():
        return

    from eos.artifact_hashing import canonical_json, content_hash
    from eos.db import init_db
    from storage.governance_store import GovernanceStore

    db_path = runtime_root / "eos.db"
    init_db(db_path)
    store = GovernanceStore(db_path=db_path)

    for path in sorted(source_root.rglob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"policy candidate file must contain a YAML object: {path}")

        content = raw.get("content")
        if not isinstance(content, dict):
            raise ValueError(f"policy candidate file missing object 'content': {path}")

        name = str(raw.get("name") or "").strip()
        category = str(raw.get("category") or "").strip()
        semantic_tier = str(raw.get("semantic_tier") or "").strip()
        if not name or not category or not semantic_tier:
            raise ValueError(
                f"policy candidate file must define name/category/semantic_tier: {path}"
            )

        content_yaml = canonical_json(content)
        row = store.propose(
            name=name,
            category=category,
            semantic_tier=semantic_tier,
            content_yaml=content_yaml,
            content_hash=content_hash(content),
            proposed_by=str(raw.get("proposed_by") or "bootstrap"),
            scoped_owner=raw.get("scoped_owner"),
            rollback_window_hours=raw.get("rollback_window_hours"),
            rollback_strategy=raw.get("rollback_strategy"),
            notes=str(raw.get("notes") or f"Loaded from {path.relative_to(source_root.parent)}"),
            jurisdiction_review=raw.get("jurisdiction_review"),
        )

        activation = raw.get("activation") or {}
        if not isinstance(activation, dict):
            raise ValueError(f"policy candidate file 'activation' must be an object: {path}")
        desired_status = str(activation.get("status") or "ACTIVE").upper()
        if desired_status not in {"PENDING", "APPROVED", "ACTIVE"}:
            raise ValueError(
                f"policy candidate activation.status must be PENDING/APPROVED/ACTIVE: {path}"
            )

        if desired_status == "PENDING" or row["status"] == desired_status:
            continue

        if row["status"] == "PENDING" and semantic_tier == "T5":
            row = store.fast_track_activate(
                row["candidate_id"],
                actor=str(activation.get("actor") or raw.get("scoped_owner") or "bootstrap"),
                actor_roles=tuple(activation.get("actor_roles") or ("delegated_curator",)),
                notes=str(raw.get("notes") or f"Loaded from {path.relative_to(source_root.parent)}"),
            )
            if desired_status == "ACTIVE":
                continue

        if row["status"] == "PENDING":
            replay_result = activation.get("replay_result") or {
                "regressions": [],
                "tested_claims": 0,
            }
            replay_tested_at = replay_result.get("tested_at")
            row = store.record_replay_result(
                row["candidate_id"],
                regressions=list(replay_result.get("regressions") or []),
                tested_claims=int(replay_result.get("tested_claims") or 0),
                tested_at=str(replay_tested_at) if replay_tested_at else None,
            )
            row = store.record_approval(
                row["candidate_id"],
                approved_by=str(activation.get("approved_by") or "bootstrap"),
                actor_roles=tuple(activation.get("approval_actor_roles") or ()),
            )

        if desired_status == "ACTIVE" and row["status"] == "APPROVED":
            store.activate(
                row["candidate_id"],
                actor_roles=tuple(activation.get("activation_actor_roles") or ()),
            )


def _ensure_repo_local_runtime_env(runtime_root: Path) -> None:
    env_path = runtime_root / ".env"
    entries: dict[str, str] = {}
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            entries[key] = value

    changed = False
    for key in _REMOVED_RUNTIME_ENV_KEYS:
        if key in entries:
            entries.pop(key, None)
            changed = True

    for key, value in _DEFAULT_RUNTIME_ENV.items():
        # The authenticated offline F10 harness owns absolute database paths
        # in its isolated Hermes-home dotenv. Do not replace those paths with
        # repository-relative defaults during launcher bootstrap.
        if (
            os.environ.get("SEMANTIER_F10_FIXTURE") == "1"
            and key
            in {
                "SEMANTIER_AUTH_DB_PATH",
                "SEMANTIER_EOS_DB_PATH",
                "SEMANTIER_FALKORDB_LITE_PATH",
                "SEMANTIER_LOCAL_STATE_DIR",
            }
        ):
            continue
        if key == "SEMANTIER_LOCAL_STATE_DIR":
            configured_root = os.environ.get(key) or str(_repo_runtime_root())
            value = str(Path(configured_root).expanduser().resolve())
        elif key == "SEMANTIER_AUTH_DB_PATH":
            configured_auth_db = os.environ.get(key)
            if configured_auth_db and Path(configured_auth_db).is_absolute():
                value = str(Path(configured_auth_db).expanduser().resolve())
        if entries.get(key) != value:
            entries[key] = value
            changed = True

    if changed or not env_path.exists():
        body = "".join(f"{key}={entries[key]}\n" for key in sorted(entries))
        env_path.write_text(body, encoding="utf-8")


def _ensure_runtime_profile(runtime_root: Path, profile_name: str) -> Path:
    profile_dir = runtime_root / "profiles" / profile_name
    if not profile_dir.exists():
        previous_home = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = str(runtime_root)
        try:
            _ensure_repo_hermes_agent_on_path()
            try:
                from hermes_cli.profiles import create_profile
                profile_dir = create_profile(profile_name, no_alias=True, no_skills=True)
            except (ImportError, ModuleNotFoundError):
                # Bootstrap tests and minimal installations may not ship the
                # optional profile helper.  A profile is still a filesystem
                # boundary; Hermes can materialize its aliases later.
                profile_dir.mkdir(parents=True, exist_ok=True)
        finally:
            if previous_home is None:
                os.environ.pop("HERMES_HOME", None)
            else:
                os.environ["HERMES_HOME"] = previous_home
    profile_dir.mkdir(parents=True, exist_ok=True)
    return profile_dir


def _ensure_meeting_coordinator_profiles(runtime_root: Path) -> None:
    if not (
        runtime_root
        / "plugins"
        / _FEISHU_MEETING_COORDINATOR_PLUGIN_NAME
        / "plugin.yaml"
    ).exists():
        return

    for profile_name in _MEETING_COORDINATOR_PROFILE_NAMES:
        profile_dir = _ensure_runtime_profile(runtime_root, profile_name)
        _ensure_repo_local_soul(profile_dir)
        _ensure_repo_local_governance(profile_dir)
        _install_repo_local_plugins(profile_dir)
        _install_repo_local_plugin(profile_dir, _FEISHU_MEETING_COORDINATOR_PLUGIN_NAME)
        _ensure_repo_local_runtime_config(profile_dir)
        _ensure_repo_local_runtime_env(profile_dir)
        sanitize_user_memory_profile(profile_dir)


def _ensure_tender_runtime_expert_profile(runtime_root: Path) -> None:
    """Materialize the isolated Phase1-001 profile, never in Main Hermes."""
    source_dir = _plugin_source_dir_with_fallback(
        _TENDER_RUNTIME_EXPERT_PLUGIN_NAME, allow_marketplace_fallback=False
    )
    if not (source_dir / "plugin.yaml").exists():
        return
    profile_dir = _ensure_runtime_profile(
        runtime_root, _TENDER_RUNTIME_EXPERT_PROFILE_NAME
    )
    _ensure_repo_local_soul(profile_dir)
    _ensure_repo_local_governance(profile_dir)
    _install_repo_local_plugin(
        profile_dir, _TENDER_RUNTIME_EXPERT_PLUGIN_NAME, allow_marketplace_fallback=False
    )
    config = _read_runtime_config(profile_dir)
    plugins = config.setdefault("plugins", {})
    enabled = plugins.setdefault("enabled", [])
    if _TENDER_RUNTIME_EXPERT_PLUGIN_NAME not in enabled:
        enabled.append(_TENDER_RUNTIME_EXPERT_PLUGIN_NAME)
    toolsets = config.setdefault("platform_toolsets", {})
    for platform in ("cli", "api_server", "acp", "feishu", "weixin"):
        names = toolsets.setdefault(platform, [])
        if _TENDER_RUNTIME_EXPERT_PLUGIN_NAME not in names:
            names.append(_TENDER_RUNTIME_EXPERT_PLUGIN_NAME)
    profile_path = profile_dir / "PROFILE.md"
    profile_path.write_text(
        "# tender-runtime-expert\n\n"
        "You are the read-only Tender Runtime Expert. Use only the registered "
        "tender_contextgraph_runtime toolset. Never mutate candidates, checkpoint, "
        "submit governance evidence, approve, admit, or change authority bindings.\n",
        encoding="utf-8",
    )
    _write_runtime_config(profile_dir, config)
    _ensure_repo_local_runtime_env(profile_dir)
    sanitize_user_memory_profile(profile_dir)


_UPSTREAM_HERMES_SOUL_PREFIX = "You are Hermes Agent, an intelligent AI assistant created by Nous Research."


def _ensure_repo_local_soul(runtime_root: Path) -> None:
    soul_path = runtime_root / "SOUL.md"
    if soul_path.exists():
        existing = soul_path.read_text(encoding="utf-8")
        if not existing.startswith(_UPSTREAM_HERMES_SOUL_PREFIX):
            return
    soul_path.write_text(_default_soul_md(), encoding="utf-8")


def _ensure_repo_local_governance(runtime_root: Path) -> None:
    governance_path = runtime_root / "SEMANTIER_GOVERNANCE.md"
    governance_path.write_text(_default_governance_md(), encoding="utf-8")


def _repo_state_evolution_worker_root() -> Path:
    return Path(__file__).resolve().parents[2] / "hermes-workspace" / "dist-state-evolution"


def _install_state_evolution_worker(runtime_root: Path):
    """Install immutable generated worker code through the shared bootstrap owner.

    Minimal installations may omit this optional build. Invoking State Evolution
    then fails explicitly rather than reading an unmanaged repository fallback.
    """
    import tempfile
    from contracts.state_evolution import evolution_hash
    from services.state_evolution_service import verify_evolution_worker_assets
    source = _repo_state_evolution_worker_root()
    if not (source / "manifest.json").is_file():
        return None
    manifest = verify_evolution_worker_assets(source)
    parent = runtime_root / "state-evolution-workers"
    if not parent.resolve().is_relative_to(runtime_root.resolve()):
        raise ValueError("STATE_EVOLUTION_WORKER_ASSET_PATH_INVALID")
    parent.mkdir(parents=True, exist_ok=True)
    digest = manifest["buildHash"].split(":")[1]
    target = parent / digest
    if not target.resolve().is_relative_to(parent.resolve()):
        raise ValueError("STATE_EVOLUTION_WORKER_ASSET_PATH_INVALID")
    if not target.exists():
        with tempfile.TemporaryDirectory(prefix="install-", dir=parent) as staging:
            staged = Path(staging) / digest
            shutil.copytree(source, staged)
            verify_evolution_worker_assets(staged)
            try:
                staged.rename(target)
            except FileExistsError:
                verify_evolution_worker_assets(target)
    verify_evolution_worker_assets(target)
    pin = {"artifact_kind":"state_evolution_worker_build", "artifact_ref":"state-evolution-worker:"+digest,
           "artifact_hash":evolution_hash(manifest)}
    with tempfile.NamedTemporaryFile(mode="w", dir=parent, prefix="current-", delete=False) as selector:
        json.dump(pin, selector)
        selector.write("\n")
        selector_path = Path(selector.name)
    selector_path.replace(parent / "current.json")
    return pin


def _bootstrap_runtime_root(
    runtime_root: Path,
    *,
    install_repo_skills: bool = True,
    shared_skills_external_dir: Path | None = None,
) -> None:
    runtime_root.mkdir(parents=True, exist_ok=True)
    _remove_obsolete_hermes_local_dirs(runtime_root)
    _install_state_evolution_worker(runtime_root)
    _ensure_repo_local_soul(runtime_root)
    _ensure_repo_local_governance(runtime_root)
    _install_repo_local_plugins(runtime_root)
    if _packaged_hermes_root() is not None:
        _install_repo_local_plugin(runtime_root, "sensevoice_stt", allow_marketplace_fallback=False)
        _install_packaged_sensevoice_assets(runtime_root)
    if install_repo_skills:
        for skill_path in _RUNTIME_SKILL_PATHS:
            _install_repo_local_skill(runtime_root, skill_path)
        _remove_legacy_repo_local_skills(runtime_root)
    else:
        _remove_managed_workspace_skill_copies(runtime_root)
    _ensure_repo_local_runtime_config(
        runtime_root,
        shared_skills_external_dir=shared_skills_external_dir,
    )
    _ensure_meeting_coordinator_profiles(runtime_root)
    _ensure_tender_runtime_expert_profile(runtime_root)
    _ensure_repo_local_runtime_env(runtime_root)
    _load_repo_doctrine_artifact_files(runtime_root)
    _load_repo_policy_candidate_files(runtime_root)
    sanitize_user_memory_profile(runtime_root)


_WORKSPACE_BOOTSTRAP_LOCKS: dict[Path, threading.RLock] = {}
_WORKSPACE_BOOTSTRAP_LOCKS_GUARD = threading.Lock()
_WORKSPACE_BOOTSTRAP_LOCAL = threading.local()


@contextmanager
def _workspace_bootstrap_lock(runtime_root: Path):
    """Serialize one workspace transaction, including same-thread reentry.

    POSIX uses flock; Windows uses a byte-range lock. Unsupported platforms
    fail closed rather than silently dropping cross-process serialization.
    The lock file stays in place so waiters always lock the same inode.
    """
    root = runtime_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with _WORKSPACE_BOOTSTRAP_LOCKS_GUARD:
        lock = _WORKSPACE_BOOTSTRAP_LOCKS.setdefault(root, threading.RLock())
    with lock:
        held = getattr(_WORKSPACE_BOOTSTRAP_LOCAL, 'held', None)
        if held is None:
            held = _WORKSPACE_BOOTSTRAP_LOCAL.held = set()
        if root in held:
            yield
            return
        with (root / '.semantier-bootstrap.lock').open('a+b') as handle:
            if os.name == 'posix':
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                release = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            elif os.name == 'nt':
                import errno
                import msvcrt
                if os.fstat(handle.fileno()).st_size == 0:
                    handle.write(b'0')
                    handle.flush()
                handle.seek(0)
                while True:
                    try:
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError as exc:
                        if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                            raise
                        time.sleep(0.05)
                def release() -> None:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                raise RuntimeError('workspace bootstrap requires a supported process lock')
            held.add(root)
            try:
                yield
            finally:
                held.remove(root)
                release()


def _ensure_workshop_teaching_materials(runtime_root: Path) -> None:
    """Install public teaching inputs, never another user's sessions or results."""
    repository = Path(__file__).resolve().parents[2]
    destination = runtime_root / "demo-materials"
    if destination.is_symlink():
        raise PermissionError("TEACHING_MATERIAL_PATH_ESCAPE")
    destination.mkdir(parents=True, exist_ok=True)
    sources = list((repository / "docs/workshop/materials").iterdir())
    sources += list((repository / "src/catalogs/workshop/expense-policy").iterdir())
    for source in sources:
        if not source.is_file() or source.suffix not in {".md", ".pdf", ".json"}:
            continue
        name = "expense-policy-" + source.name if source.parent.name == "expense-policy" else source.name
        target = destination / name
        if target.is_symlink() or target.with_name(target.name + ".installing").is_symlink():
            raise PermissionError("TEACHING_MATERIAL_PATH_ESCAPE")
        raw = source.read_bytes()
        if not target.is_file() or target.read_bytes() != raw:
            temporary = target.with_name(target.name + ".installing")
            temporary.write_bytes(raw)
            temporary.replace(target)


def bootstrap_workspace_runtime_home(
    runtime_root: Path,
    *,
    shared_skills_external_dir: Path | None = None,
    shared_plugins_external_dir: Path | None = None,
    workspace_id: str | None = None,
) -> None:
    """Bootstrap a workspace-local Hermes home with shared Semantier surfaces.

    Workspace homes need the shared plugin/config/env surface so gateway and UI
    sessions expose the governed toolsets, but they should not create their own
    governed doctrine/policy stores. Those remain owned by the shared runtime.
    """
    with _workspace_bootstrap_lock(runtime_root):
        runtime_root.mkdir(parents=True, exist_ok=True)
        _migrate_workspace_config_file_to_db(runtime_root, workspace_id)
        runtime_config = _read_runtime_config(runtime_root)
        db_config = _load_db_workspace_config(workspace_id)
        if db_config:
            _restore_workspace_configured_plugins(runtime_root, db_config)
            runtime_config = _merge_configs(
                dict(db_config),
                runtime_config,
            )
        _write_runtime_config(runtime_root, runtime_config)
        _remove_obsolete_hermes_local_dirs(runtime_root)
        _ensure_workshop_teaching_materials(runtime_root)
        _ensure_repo_local_soul(runtime_root)
        _ensure_repo_local_governance(runtime_root)
        _sync_managed_workspace_plugin_links(runtime_root, shared_plugins_external_dir)
        _remove_materialized_optional_plugin_skills(runtime_root)
        _remove_managed_workspace_skill_copies(runtime_root)
        for child in ("skills", "memories", "profiles", "cron", "logs", "home", "sessions"):
            (runtime_root / child).mkdir(parents=True, exist_ok=True)
        _ensure_repo_local_runtime_config(
            runtime_root,
            shared_skills_external_dir=shared_skills_external_dir,
            shared_plugins_external_dir=shared_plugins_external_dir,
        )
        _save_db_workspace_config(workspace_id, _read_runtime_config(runtime_root))
        _ensure_meeting_coordinator_profiles(runtime_root)
        _ensure_repo_local_runtime_env(runtime_root)
        sanitize_user_memory_profile(runtime_root)


_SEMANTIER_PLATFORM_TOOLSETS: dict[str, str] = {
    "cli": "hermes-cli",
    "weixin": "hermes-weixin",
    "feishu": "hermes-feishu",
    "api_server": "hermes-api-server",
    "acp": "hermes-acp",
}

_SUPPORTED_GATEWAYS: tuple[str, ...] = ("cli", "weixin", "feishu", "acp")


def _ensure_repo_local_runtime_config(
    runtime_root: Path,
    *,
    shared_skills_external_dir: Path | None = None,
    shared_plugins_external_dir: Path | None = None,
) -> None:
    """Ensure the repo-local Hermes config enables the business analytics surfaces.

    This only touches the repo-local runtime used by the Semantier wrappers.
    Explicit non-repo ``HERMES_HOME`` overrides are left alone.
    """
    runtime_root.mkdir(parents=True, exist_ok=True)
    config_path = runtime_root / "config.yaml"
    config: dict = {}
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            config = loaded

    changed = False
    forbidden_skill_external_dirs = forbidden_global_external_skill_roots(runtime_root)

    plugins_cfg = config.get("plugins")
    if not isinstance(plugins_cfg, dict):
        plugins_cfg = {}
        config["plugins"] = plugins_cfg
        changed = True

    enabled_plugins = plugins_cfg.get("enabled")
    if not isinstance(enabled_plugins, list):
        enabled_plugins = []
        plugins_cfg["enabled"] = enabled_plugins
        changed = True
    for plugin_name in _RUNTIME_PLUGIN_NAMES:
        if plugin_name not in enabled_plugins:
            enabled_plugins.append(plugin_name)
            changed = True
    plugin_roots = [runtime_root]
    if shared_plugins_external_dir is not None:
        plugin_roots.append(shared_plugins_external_dir.parent)

    installed_optional_plugins = [
        plugin_name
        for plugin_name in _OPTIONAL_RUNTIME_PLUGIN_NAMES
        if any((root / "plugins" / plugin_name / "plugin.yaml").exists() for root in plugin_roots)
    ]
    for plugin_name in installed_optional_plugins:
        if plugin_name not in enabled_plugins:
            enabled_plugins.append(plugin_name)
            changed = True

    platform_toolsets = config.get("platform_toolsets")
    if not isinstance(platform_toolsets, dict):
        platform_toolsets = {}
        config["platform_toolsets"] = platform_toolsets
        changed = True

    for platform, default_toolset in _SEMANTIER_PLATFORM_TOOLSETS.items():
        toolsets = platform_toolsets.get(platform)
        if not isinstance(toolsets, list):
            toolsets = [default_toolset]
            platform_toolsets[platform] = toolsets
            changed = True
        if default_toolset not in toolsets:
            toolsets.append(default_toolset)
            changed = True
        if _ANALYTICS_PLUGIN_NAME not in toolsets:
            toolsets.append(_ANALYTICS_PLUGIN_NAME)
            changed = True
        if _DATA_ENGINEERING_PLUGIN_NAME not in toolsets:
            toolsets.append(_DATA_ENGINEERING_PLUGIN_NAME)
            changed = True

    for platform, toolsets in platform_toolsets.items():
        if platform in _SEMANTIER_PLATFORM_TOOLSETS or not isinstance(toolsets, list):
            continue
        if _ANALYTICS_PLUGIN_NAME not in toolsets:
            toolsets.append(_ANALYTICS_PLUGIN_NAME)
            changed = True
        if _DATA_ENGINEERING_PLUGIN_NAME not in toolsets:
            toolsets.append(_DATA_ENGINEERING_PLUGIN_NAME)
            changed = True

    # Hard guarantee: policy/runtime tools must be enabled on all 3 supported gateways.
    for gateway in _SUPPORTED_GATEWAYS:
        default_toolset = _SEMANTIER_PLATFORM_TOOLSETS[gateway]
        toolsets = platform_toolsets.get(gateway)
        if not isinstance(toolsets, list):
            toolsets = [default_toolset]
            platform_toolsets[gateway] = toolsets
            changed = True
        if default_toolset not in toolsets:
            toolsets.append(default_toolset)
            changed = True
        if _ANALYTICS_PLUGIN_NAME not in toolsets:
            toolsets.append(_ANALYTICS_PLUGIN_NAME)
            changed = True
        if _DATA_ENGINEERING_PLUGIN_NAME not in toolsets:
            toolsets.append(_DATA_ENGINEERING_PLUGIN_NAME)
            changed = True

    if _FEISHU_MEETING_COORDINATOR_PLUGIN_NAME in installed_optional_plugins:
        feishu_toolsets = platform_toolsets.get("feishu")
        if not isinstance(feishu_toolsets, list):
            feishu_toolsets = [_SEMANTIER_PLATFORM_TOOLSETS["feishu"]]
            platform_toolsets["feishu"] = feishu_toolsets
            changed = True
        if "meeting-coordinator" not in feishu_toolsets:
            feishu_toolsets.append("meeting-coordinator")
            changed = True

    for optional_tool_plugin in (
        _REAL_COMPANY_ONBOARDING_PLUGIN_NAME,
        _AUTO_RESUME_SCREENING_PLUGIN_NAME,
        _DOCUMENT_EXTRACTION_PLUGIN_NAME,
        _FINANCIAL_DOCUMENTS_PLUGIN_NAME,
        _AUTOMATE_EXCEL_PLUGIN_NAME,
        _TENDER_SENSITIVE_TERM_DETECTION_MODELING_PLUGIN_NAME,
        _VC_GITHUB_OPPORTUNITY_RADAR_PLUGIN_NAME,
        _CORPUS_GRAPH_SCHEMA_INDUCTION_PLUGIN_NAME,
        _TENDER_JOURNEY_PLUGIN_NAME,
    ):
        if (
            optional_tool_plugin in _OPTIONAL_RUNTIME_PLUGIN_NAMES
            and optional_tool_plugin not in installed_optional_plugins
        ):
            continue
        for platform, default_toolset in _SEMANTIER_PLATFORM_TOOLSETS.items():
            toolsets = platform_toolsets.get(platform)
            if not isinstance(toolsets, list):
                toolsets = [default_toolset]
                platform_toolsets[platform] = toolsets
                changed = True
            if optional_tool_plugin not in toolsets:
                toolsets.append(optional_tool_plugin)
                changed = True

    skills_cfg = config.get("skills")
    if not isinstance(skills_cfg, dict):
        skills_cfg = {}
        config["skills"] = skills_cfg
        changed = True

    plugin_skill_dirs = _plugin_skill_external_dirs(runtime_root)
    if shared_plugins_external_dir is not None:
        shared_plugin_runtime_root = shared_plugins_external_dir.parent
        plugin_skill_dirs.extend(_plugin_skill_external_dirs(shared_plugin_runtime_root))
    external_dirs = skills_cfg.get("external_dirs")
    if plugin_skill_dirs or isinstance(external_dirs, list):
        if not isinstance(external_dirs, list):
            external_dirs = []
            skills_cfg["external_dirs"] = external_dirs
            changed = True
        plugin_dirs = {
            str((runtime_root / "plugins" / plugin_name).resolve())
            for plugin_name in _PLUGIN_SKILL_EXTERNAL_DIR_NAMES
        }
        if shared_plugins_external_dir is not None:
            plugin_dirs.update(
                str((shared_plugins_external_dir / plugin_name).resolve())
                for plugin_name in _PLUGIN_SKILL_EXTERNAL_DIR_NAMES
            )
        normalized_external_dirs: list[str] = []
        for item in external_dirs:
            text = str(item or "").strip()
            if not text:
                continue
            try:
                resolved_path = Path(text).expanduser().resolve()
                resolved = str(resolved_path)
            except OSError:
                resolved_path = None
                resolved = text
            if resolved_path is not None and resolved_path in forbidden_skill_external_dirs:
                continue
            if resolved in plugin_dirs:
                continue
            if resolved not in normalized_external_dirs:
                normalized_external_dirs.append(resolved)
        for plugin_skill_dir in plugin_skill_dirs:
            resolved = str(plugin_skill_dir.resolve())
            if resolved not in normalized_external_dirs:
                normalized_external_dirs.append(resolved)
        if normalized_external_dirs != external_dirs:
            skills_cfg["external_dirs"] = normalized_external_dirs
            changed = True

    # Hermes memory is convenience context only. Semantier authority-bearing
    # context must still come from governed sources, not USER.md / MEMORY.md.
    memory_cfg = config.get("memory")
    if not isinstance(memory_cfg, dict):
        memory_cfg = {}
        config["memory"] = memory_cfg
        changed = True
    if memory_cfg.get("memory_enabled") is not True:
        memory_cfg["memory_enabled"] = True
        changed = True
    if memory_cfg.get("user_profile_enabled") is not True:
        memory_cfg["user_profile_enabled"] = True
        changed = True

    approvals_cfg = config.get("approvals")
    if not isinstance(approvals_cfg, dict):
        approvals_cfg = {}
        config["approvals"] = approvals_cfg
        changed = True
    if approvals_cfg.get("mode") != "smart":
        approvals_cfg["mode"] = "smart"
        changed = True
    if approvals_cfg.get("cron_mode") != "approve":
        approvals_cfg["cron_mode"] = "approve"
        changed = True

    auxiliary_cfg = config.get("auxiliary")
    if not isinstance(auxiliary_cfg, dict):
        auxiliary_cfg = {}
        config["auxiliary"] = auxiliary_cfg
        changed = True
    ai_ground_cfg = auxiliary_cfg.get("contextgraph_ai_ground")
    if not isinstance(ai_ground_cfg, dict):
        auxiliary_cfg["contextgraph_ai_ground"] = {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 120,
            "extra_body": {"enable_thinking": False},
        }
        changed = True

    if shared_skills_external_dir is not None:
        external_dirs = skills_cfg.get("external_dirs")
        if not isinstance(external_dirs, list):
            external_dirs = []
            skills_cfg["external_dirs"] = external_dirs
            changed = True
        shared_dir = str(shared_skills_external_dir.resolve())
        normalized_external_dirs: list[str] = []
        for item in external_dirs:
            text = str(item or "").strip()
            if not text:
                continue
            try:
                resolved_path = Path(text).expanduser().resolve()
                resolved = str(resolved_path)
            except OSError:
                resolved_path = None
                resolved = text
            if resolved_path is not None and resolved_path in forbidden_skill_external_dirs:
                continue
            if resolved == str((runtime_root / "skills").resolve()):
                continue
            if resolved not in normalized_external_dirs:
                normalized_external_dirs.append(resolved)
        if shared_dir not in normalized_external_dirs:
            normalized_external_dirs.insert(0, shared_dir)
        if normalized_external_dirs != external_dirs:
            skills_cfg["external_dirs"] = normalized_external_dirs
            changed = True

    if changed:
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )


def _governed_transcription_temporary_directory(*, prefix: str, ignore_cleanup_errors: bool = False):
    """Keep native oversized-audio staging inside the calling task's artifacts."""
    import tempfile
    from gateway.execution_boundary import BoundaryPathRejected, GovernedExecutionBoundaryRequired, execution_scratch_dir

    if not isinstance(prefix, str) or any(char in prefix for char in ("/", "\\", ":", "\x00")):
        raise BoundaryPathRejected("Transcription scratch prefix must be a simple filename")
    scratch = execution_scratch_dir()
    if scratch is None:
        raise GovernedExecutionBoundaryRequired("Transcription staging requires an active execution boundary")
    return tempfile.TemporaryDirectory(prefix=prefix, dir=scratch,
        ignore_cleanup_errors=ignore_cleanup_errors)


def _bind_native_transcription_scratch() -> None:
    # Adapt only this pinned native module's imported factory. The stdlib
    # factory and process-global temp directory remain untouched.
    from tools import transcription_chunking

    transcription_chunking.TemporaryDirectory = _governed_transcription_temporary_directory


def _install_packaged_sensevoice_assets(runtime_root: Path) -> None:
    """Provision pinned shared assets once; never overwrite a differing user asset."""
    import hashlib
    from plugins.sensevoice_stt.provider import ASSETS
    from storage.governed_file_io import exclusive_file_lock, sync_directory, clean_publication_temps
    import uuid

    root = _packaged_hermes_root()
    if root is None:
        return
    source = root / "semantier-runtime/assets/sensevoice"
    target = runtime_root / "models/sensevoice"
    target.mkdir(parents=True, exist_ok=True)
    if not target.resolve().is_relative_to(runtime_root.resolve()):
        raise ValueError("packaged SenseVoice assets escape platform root")
    with exclusive_file_lock(target / '.provision.lock'):
        for name, (size, expected) in ASSETS.items():
            destination = target / name
            if not destination.resolve().is_relative_to(target.resolve()):
                raise ValueError('SenseVoice asset escapes platform root')
            clean_publication_temps(destination)
            candidate = destination if destination.exists() else source / name
            with candidate.open('rb') as stream:
                if candidate.stat().st_size != size or hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                    raise ValueError(f"SenseVoice pinned asset integrity mismatch: {name}")
            if destination.exists():
                continue
            stage = target / ('.' + name + '.' + uuid.uuid4().hex + '.tmp')
            try:
                with candidate.open('rb') as src, stage.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
                    dst.flush(); os.fsync(dst.fileno())
                os.replace(stage, destination)
                sync_directory(target)
            finally:
                stage.unlink(missing_ok=True)
        sync_directory(target)  # repeat sync on restart after publication-before-flush failure


def bootstrap_packaged_runtime() -> None:
    """The native Hermes entry composes the parent through this sole bootstrap owner."""
    if _packaged_hermes_root() is None:
        raise RuntimeError("packaged bootstrap requires the declared Windows product")
    from hermes_constants import get_hermes_home
    _ensure_repo_local_runtime()
    platform = _repo_runtime_root()
    home = get_hermes_home().resolve()
    if home != platform:
        bootstrap_workspace_runtime_home(home, shared_skills_external_dir=platform / 'skills',
                                         shared_plugins_external_dir=platform / 'plugins')
        _install_repo_local_plugin(home, 'sensevoice_stt', allow_marketplace_fallback=False)
        _ensure_repo_local_runtime_config(home, shared_skills_external_dir=platform / 'skills',
                                          shared_plugins_external_dir=platform / 'plugins')


def _ensure_repo_local_runtime(
    *, db_check: str = "auto", consume_db_health_marker: bool = False
) -> None:
    """Default Semantier wrappers to the repo-local Hermes runtime.

    Explicit environment overrides still win. The shared runtime root is not
    exported as ``HERMES_HOME``; authenticated execution must bind that per
    governed workspace.
    """
    runtime_root = _repo_runtime_root()
    runtime_root.mkdir(parents=True, exist_ok=True)
    # Shared plugin discovery requires an absolute platform root, independent
    # of the authenticated workspace's working directory.
    configured_root = os.environ.get("SEMANTIER_LOCAL_STATE_DIR") or str(runtime_root)
    os.environ["SEMANTIER_LOCAL_STATE_DIR"] = str(Path(configured_root).expanduser().resolve())
    # Native Kanban has a separate shared-home override; request-bound
    # HERMES_HOME must never select a workspace-local coordination database.
    os.environ["HERMES_KANBAN_HOME"] = os.environ["SEMANTIER_LOCAL_STATE_DIR"]
    os.environ.setdefault("SEMANTIER_EXPOSE_HERMES_BUNDLED_SKILLS", "1")
    from agents.bootstrap_repair import quarantine_malformed_eos_db

    quarantine_malformed_eos_db(
        runtime_root / "eos.db",
        check_mode=db_check,
        consume_clean_marker=consume_db_health_marker,
    )
    _bootstrap_runtime_root(runtime_root)
    _ensure_repo_hermes_agent_on_path()
    _bind_native_transcription_scratch()
    from agents.cron_home_provider import register_semantier_cron_home_provider

    register_semantier_cron_home_provider()
    # Hermes imports used during provider registration may put their checkout
    # back at sys.path[0]. Reassert the composed plugin namespace contract
    # before the ASGI application imports Semantier-owned plugins.
    _ensure_repo_hermes_agent_on_path()


def _terminate_hermes_gateway_processes() -> None:
    """Kill any running hermes gateway processes before starting a replacement.

    Matches processes whose command line contains ``hermes`` and ``gateway run``
    (e.g. ``hermes_cli.main gateway run --replace`` or the standalone
    ``hermes gateway run`` binary) but excludes the current process so we
    don't kill ourselves.
    """
    own_pid = os.getpid()
    try:
        result = subprocess.run(
            ["pgrep", "-f", "hermes.*gateway.run"],
            capture_output=True,
            text=True,
        )
        pids = [int(p) for p in result.stdout.split() if p.strip()]
    except Exception:
        return

    for pid in pids:
        if pid == own_pid:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue

    if pids:
        # Brief wait to allow graceful shutdown before continuing.
        time.sleep(1.0)
        for pid in pids:
            if pid == own_pid:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def _runtime_owner_path(runtime_root: Path) -> Path:
    return runtime_root / "semantier-runtime-owner.json"


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _load_runtime_owner(runtime_root: Path) -> dict | None:
    owner_path = _runtime_owner_path(runtime_root)
    if not owner_path.exists():
        return None
    try:
        loaded = json.loads(owner_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _clear_runtime_owner(runtime_root: Path) -> None:
    _runtime_owner_path(runtime_root).unlink(missing_ok=True)


def _write_runtime_owner(
    runtime_root: Path,
    *,
    pid: int | None = None,
    port: int | None = None,
    command: str | None = None,
) -> None:
    payload: dict[str, object] = {"pid": pid if pid is not None else os.getpid()}
    try:
        payload["pgid"] = os.getpgid(int(payload["pid"]))
    except Exception:
        pass
    if port is not None:
        payload["port"] = int(port)
    if command:
        payload["command"] = command
    _runtime_owner_path(runtime_root).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _terminate_pid_or_group(pid: int, *, pgid: int | None = None) -> None:
    try:
        resolved_pgid = pgid if pgid is not None else os.getpgid(pid)
    except Exception:
        resolved_pgid = None

    if resolved_pgid and resolved_pgid > 1:
        try:
            os.killpg(resolved_pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        time.sleep(1.0)
        try:
            os.killpg(resolved_pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        return

    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    time.sleep(1.0)
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _terminate_semantier_webapi_processes(*, port: int = 8899) -> None:
    own_pid = os.getpid()
    try:
        own_pgid = os.getpgid(own_pid)
    except Exception:
        own_pgid = None
    candidate_pids: list[int] = []

    def _is_self_or_current_group(pid: int) -> bool:
        if pid == own_pid:
            return True
        if own_pgid is None:
            return False
        try:
            return os.getpgid(pid) == own_pgid
        except Exception:
            return False

    try:
        result = subprocess.run(
            ["ss", "-ltnp"],
            capture_output=True,
            text=True,
            check=False,
        )
        for match in re.finditer(rf":{port}\b.*pid=(\d+)", result.stdout or ""):
            pid = int(match.group(1))
            if not _is_self_or_current_group(pid) and pid not in candidate_pids:
                candidate_pids.append(pid)
    except Exception:
        pass

    for pid in candidate_pids:
        _terminate_pid_or_group(pid)


def run_agent_cli(argv: Sequence[str] | None = None) -> int:
    _ensure_repo_local_runtime()
    try:
        from run_agent import main as hermes_agent_main  # type: ignore[import]
    except Exception as exc:
        raise _missing_hermes_dependency(exc)

    agent_argv = ["semantier-agent", *(argv if argv is not None else sys.argv[1:])]
    return _run_with_argv(hermes_agent_main, agent_argv)


def run_gateway_cli(argv: Sequence[str] | None = None) -> int:
    """Shorthand entry-point for the gateway sub-command.

    ``semantier-gateway [run|setup|...]`` is equivalent to
    ``semantier gateway [run|setup|...]``.  All non-gateway hermes commands
    pass through directly (e.g. ``semantier-gateway pairing approve weixin <CODE>``).
    """
    raw_args = list(argv if argv is not None else sys.argv[1:])
    # Strip redundant leading "gateway" from legacy invocations.
    if raw_args and raw_args[0] == "gateway":
        raw_args = raw_args[1:]
    return run_hermes_cli(["gateway", *raw_args], program_name="semantier-gateway")


def run_webapi_cli(argv: Sequence[str] | None = None, *, program_name: str = "semantier webapi") -> int:
    return run_runtime_cli(argv, program_name=program_name)


def run_runtime_cli(argv: Sequence[str] | None = None, *, program_name: str = "semantier") -> int:
    _assert_runtime_checkout_coherence()
    parser = argparse.ArgumentParser(prog=program_name)
    parser.add_argument("--host", default=os.getenv("SEMANTIER_RUNTIME_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("SEMANTIER_RUNTIME_PORT", "8899")))
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--db-check",
        choices=("auto", "quick", "full"),
        default="auto",
        help="eos.db validation depth (default: auto)",
    )
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])

    runtime_root = _repo_runtime_root()
    runtime_root.mkdir(parents=True, exist_ok=True)

    owner = _load_runtime_owner(runtime_root)
    if owner:
        pid = int(owner.get("pid", 0) or 0)
        if pid and _pid_is_alive(pid):
            if not args.replace:
                raise RuntimeError(f"semantier runtime already active under pid {pid}; use --replace")
            owner_pgid = int(owner.get("pgid", 0) or 0) or None
            _terminate_pid_or_group(pid, pgid=owner_pgid)
        _clear_runtime_owner(runtime_root)

    if args.replace:
        _terminate_semantier_webapi_processes(port=args.port)

    _ensure_repo_local_runtime(
        db_check=args.db_check,
        consume_db_health_marker=True,
    )

    import uvicorn

    previous_embed = os.environ.get(RUNTIME_EMBED_GATEWAY_ENV)
    os.environ[RUNTIME_EMBED_GATEWAY_ENV] = "1"
    _write_runtime_owner(
        runtime_root,
        port=args.port,
        command=f"{program_name} run",
    )
    completed_cleanly = False
    try:
        uvicorn.run(
            "semantier.agents.gateway:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            ws="websockets-sansio",
        )
        completed_cleanly = True
    finally:
        try:
            if completed_cleanly:
                record_clean_eos_shutdown(runtime_root / "eos.db")
        finally:
            try:
                _clear_runtime_owner(runtime_root)
            finally:
                if previous_embed is None:
                    os.environ.pop(RUNTIME_EMBED_GATEWAY_ENV, None)
                else:
                    os.environ[RUNTIME_EMBED_GATEWAY_ENV] = previous_embed

    return 0


def run_hermes_cli(argv: Sequence[str] | None = None, *, program_name: str = "semantier") -> int:
    """Full hermes CLI exposed as the ``semantier`` command.

    Routing rules (in order):
    1. No args or first arg is a gateway sub-command (``run``, ``setup``) and
       the call came via ``semantier-gateway`` → route to ``hermes gateway``.
    2. First arg is ``gateway`` → route to ``hermes gateway <rest>``.
    3. Anything else → pass directly to ``hermes <args>``.

    This means every hermes command is reachable via ``semantier``::

        semantier gateway run --replace
        semantier pairing approve weixin <CODE>
        semantier status
        semantier skills list
    """
    raw_args = list(argv if argv is not None else sys.argv[1:])

    _assert_runtime_checkout_coherence()
    _ensure_repo_local_runtime()

    # Semantier-native bootstrap orchestration commands.
    if raw_args and raw_args[0] == "bootstrap":
        return run_bootstrap_cli(raw_args[1:], program_name=program_name)
    if raw_args[:2] == ["webapi", "run"]:
        return run_webapi_cli(raw_args[1:], program_name=program_name)
    if raw_args and raw_args[0] == "run":
        return run_runtime_cli(raw_args[1:], program_name=program_name)

    # Gateway-specific sub-commands that must be routed under "hermes gateway".
    _GATEWAY_SUBCOMMANDS = {"run", "setup"}

    # Kill competing hermes gateway processes before starting a replacement.
    _is_gateway_run = (
        raw_args[:2] == ["gateway", "run"]
        or (raw_args and raw_args[0] in _GATEWAY_SUBCOMMANDS)
    )
    if _is_gateway_run:
        try:
            os.environ.setdefault("SEMANTIER_AUTH_DB_PATH", ".semantier-home/auth.db")
            from agents.hermes_embedded_gateway import (
                _hydrate_saved_feishu_gateway_env,
                _hydrate_saved_weixin_gateway_env,
            )

            _hydrate_saved_feishu_gateway_env()
            _hydrate_saved_weixin_gateway_env(_repo_runtime_root())
        except Exception:
            pass

    try:
        from hermes_cli.main import main as hermes_main  # type: ignore[import]
    except Exception as exc:
        raise _missing_hermes_dependency(exc)

    if "--replace" in raw_args and _is_gateway_run:
        _terminate_hermes_gateway_processes()

    if raw_args and raw_args[0] == "gateway":
        # Explicit "gateway" prefix — keep as-is for hermes routing.
        hermes_argv = [program_name, *raw_args]
    elif not raw_args or raw_args[0] in _GATEWAY_SUBCOMMANDS:
        # Bare invocation or gateway sub-command shorthand.
        hermes_argv = [program_name, "gateway", *raw_args]
    else:
        # All other hermes top-level commands pass through directly.
        hermes_argv = [program_name, *raw_args]

    return _run_with_argv(hermes_main, hermes_argv)


def run_bootstrap_cli(argv: Sequence[str] | None = None, *, program_name: str = "semantier") -> int:
    """Semantier-native bootstrap orchestration.

    Commands:
        semantier bootstrap [seed-args...]
        semantier bootstrap cleanup [cleanup-args...]
    """
    raw_args = list(argv if argv is not None else sys.argv[1:])
    repo_root = Path(__file__).resolve().parents[2]
    bootstrap_script = repo_root / "bootstrap" / "bootstrap.sh"
    cleanup_script = repo_root / "bootstrap" / "cleanup.sh"

    if not raw_args or raw_args[0] in {"-h", "--help"}:
        print(
            "Semantier bootstrap commands\n\n"
            f"Usage:\n"
            f"  {program_name} bootstrap [seed args...]\n"
            f"  {program_name} bootstrap cleanup [cleanup args...]\n\n"
            "Examples:\n"
            f"  {program_name} bootstrap --replace\n"
            f"  {program_name} bootstrap --dry-run\n"
            f"  {program_name} bootstrap cleanup --dry-run\n"
        )
        return 0

    if raw_args[0] == "cleanup" or raw_args[0] == "--cleanup":
        cmd = ["bash", str(cleanup_script), *raw_args[1:]]
    else:
        if "--replace" in raw_args:
            from agents.bootstrap_repair import quarantine_malformed_eos_db

            runtime_root = Path(os.environ.get("SEMANTIER_LOCAL_STATE_DIR") or ".semantier-home")
            if not runtime_root.is_absolute():
                runtime_root = repo_root / runtime_root
            quarantine_malformed_eos_db(runtime_root / "eos.db")
        cmd = ["bash", str(bootstrap_script), *raw_args]

    completed = subprocess.run(cmd, cwd=str(repo_root), check=False)
    return int(completed.returncode)


def run_industry_sim(argv: Sequence[str] | None = None) -> int:
    """``semantier-industry-sim`` — industry simulator dataset CLI.

    Delegates to the shared dataset registry under ``bootstrap/industry_simulator``.
    All flags are passed through unchanged::

        semantier-industry-sim --output stress
        semantier-industry-sim --output fraud
        semantier-industry-sim --output all  # writes to .semantier-home/sim.json
        semantier-industry-sim --dataset apparel_trade_3_year --output cashflow
        semantier-industry-sim --seed 7 --black-swan-year 2025 --output injury
    """
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    src_dir = repo_root / "src"

    for p in (str(src_dir),):
        if p not in sys.path:
            sys.path.insert(0, p)

    from bootstrap.industry_simulator.cli import dataset_cli_main

    run_argv = ["semantier-industry-sim", *(argv if argv is not None else sys.argv[1:])]
    return _run_with_argv(
        lambda: dataset_cli_main(prog="semantier-industry-sim", default_dataset_id="construction_3_year"),
        run_argv,
    )


def run_smb_sim(argv: Sequence[str] | None = None) -> int:
    """Deprecated compatibility alias for the old simulator entrypoint."""
    return run_industry_sim(argv)


def run_api(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="semantier-api")
    parser.add_argument("--host", default=os.getenv("SEMANTIER_API_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("SEMANTIER_API_PORT", "8000")))
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])

    import uvicorn

    uvicorn.run(
        "semantier.agents.gateway:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        ws="websockets-sansio",
    )
    return 0
