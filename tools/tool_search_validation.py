"""Local argument validation for ``tool_call`` against a deferred tool's schema."""

from __future__ import annotations

import copy
from decimal import Decimal, InvalidOperation
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from tools.registry import tool_error
from tools.tool_search_catalog import BRIDGE_TOOL_NAMES, _registry_entry

logger = logging.getLogger("tools.tool_search")

_SCHEMA_LITERAL_KEYS = frozenset({"const", "default", "enum", "example", "examples"})


def _schema_for_local_validation(node: Any) -> Any:
    """JSON-Schema-compatible copy honoring OpenAPI ``nullable: true`` (the normal coercion
    path accepts that shape, so local validation must too)."""
    if isinstance(node, list):
        return [_schema_for_local_validation(item) for item in node]
    if not isinstance(node, dict):
        return node
    # Literal keywords hold instance data, not schemas: copy byte-for-byte.
    normalized = {key: (copy.deepcopy(value) if key in _SCHEMA_LITERAL_KEYS
                        else _schema_for_local_validation(value))
                  for key, value in node.items() if key != "nullable"}
    if node.get("nullable") is not True:
        return normalized
    schema_type = normalized.get("type")
    if isinstance(schema_type, str):
        schema_type = [schema_type]
    if isinstance(schema_type, list):
        if "null" not in schema_type:
            normalized["type"] = [*schema_type, "null"]
        return normalized
    # No ``type`` to extend ($ref/combinator): wrap so local refs still resolve from the
    # root while null stays an explicit alternative.
    return {"anyOf": [normalized, {"type": "null"}]}


def _schema_has_external_ref(node: Any) -> bool:
    """True when *node* contains a non-local ``$ref`` — local validation must never turn a
    tool call into an implicit network/file fetch (fail open)."""
    if isinstance(node, list):
        return any(_schema_has_external_ref(item) for item in node)
    if not isinstance(node, dict):
        return False
    ref = node.get("$ref")
    return (isinstance(ref, str) and not ref.startswith("#")) or any(
        _schema_has_external_ref(value) for key, value in node.items()
        if key not in _SCHEMA_LITERAL_KEYS)


def _validation_path(error: Any) -> str:
    """Format a jsonschema error path as a compact argument path."""
    path = "arguments"
    for part in getattr(error, "absolute_path", ()):
        if isinstance(part, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part):
            path += f".{part}"
        else:
            path += f"[{part if isinstance(part, int) else json.dumps(part, ensure_ascii=False)}]"
    return path


def is_browser_call(name: str) -> bool:
    from tools.registry import registry
    entry = registry.get_entry(name)
    return name in ('desktop_preview', 'drive_preview', 'annotate_preview', 'browser_exec') or bool(entry and entry.toolset in ('browser', 'browser_use'))


def _validation_error(message: str, *, path: str, constraint: str, parameters: Any, name: str = '') -> str:
    if is_browser_call(name):
        code = 'MISSING_REQUIRED_FIELD' if constraint in ('required', 'anyOf') else 'UNKNOWN_ACTION' if constraint == 'enum' and path.endswith('.action') else 'INVALID_ELEMENT_REF' if constraint == 'pattern' and path.endswith('.ref') else 'INVALID_ARGUMENT'
        return tool_error(f"Browser arguments rejected at {path} ({constraint}); tool was not invoked.",
            success=False, code=code, path=path, constraint=constraint, parameters=parameters, rejected_tool=name,
            recovery={'same_context': True, 'next_action': 'correct_arguments_using_parameters'}, retryable=True)
    return tool_error(message, path=path, constraint=constraint, parameters=parameters,
                      hint="Retry tool_call with 'arguments' matching the parameters schema above.")


def _contract_unavailable(name, reason):
    return tool_error('Browser argument contract unavailable; tool was not invoked.', success=False, code='ARGUMENT_CONTRACT_UNAVAILABLE',
        path='arguments', rejected_tool=name, reason=reason, retryable=False,
        recovery={'same_context': True, 'next_action': 'restore_native_tool_contract'})


def browser_call_arguments(name, args):
    if not is_browser_call(name):
        return args, None
    error = validate_deferred_call_args(name, args)
    if error is not None:
        return args, error
    from model_tools import coerce_tool_args
    return coerce_tool_args(name, copy.deepcopy(args)), None


def tool_call_argument_error(message):
    return tool_error(message, success=False, code='MISSING_REQUIRED_FIELD' if "requires a 'name'" in message else 'INVALID_ARGUMENT',
        path='arguments.calls', recovery={'same_context': True, 'next_action': 'supply_calls_array_with_name_and_arguments'}, retryable=True)


def validate_deferred_call_args(name: str, args: dict[str, Any]) -> Optional[str]:
    """Validate ``tool_call`` arguments against the deferred tool's schema. Models invoke
    deferred tools "blind" (schema unseen) and omit required args; without this, the opaque
    downstream failure makes cheap models loop. Required-field probe first, then the same
    schema-guided coercion normal dispatch applies, then jsonschema on the repaired copy.
    Browser schemas fail closed; other tools retain existing fail-open behavior. Returns a JSON
    error string when invalid, ``None`` when the call should dispatch.

    This restores the concrete-schema checks that the provider cannot perform through the generic
    ``arguments: object`` bridge. See #5149.
    """
    strict = is_browser_call(name)
    try:
        from tools.registry import registry as _registry
        schema = _registry.get_schema(name)
        if not isinstance(schema, dict):
            return _contract_unavailable(name, 'missing_schema') if strict else None
        fn = schema.get("function") if schema.get("type") == "function" else schema
        params = fn.get("parameters") if isinstance(fn, dict) else None
        if not isinstance(params, dict):
            return _contract_unavailable(name, 'missing_parameters') if strict else None
        if strict and not isinstance(args, dict):
            return _validation_error('Expected an arguments object; tool was not invoked.', name=name,
                                     path='arguments', constraint='type', parameters=params)
        required = params.get("required")
        missing = ([r for r in required if isinstance(r, str) and r not in args]
                   if isinstance(required, list) else [])
        if missing:
            return _validation_error(
                f"tool_call to '{name}' is missing required argument(s): "
                f"{', '.join(missing)}. The tool was NOT invoked.",
                path="arguments", constraint="required", parameters=params, name=name)
        validation_schema = _schema_for_local_validation(params)
        if _schema_has_external_ref(validation_schema):
            logger.debug("Skipping local deferred-argument validation for %s: external $ref", name)
            return _contract_unavailable(name, 'external_schema_reference') if strict else None
        # Validate the repaired shape dispatch will see; copy because coerce_tool_args may
        # normalize in place (dispatch re-coerces canonically).
        try:
            from model_tools import coerce_tool_args
            candidate_args = coerce_tool_args(name, copy.deepcopy(args))
        except Exception:
            logger.debug("Deferred-argument coercion failed for %s", name, exc_info=True)
            candidate_args = dict(args)
        if strict:
            for key, value in args.items():
                normalized = candidate_args.get(key)
                if isinstance(value, str) and isinstance(normalized, (int, float)) and not isinstance(normalized, bool):
                    try:
                        if Decimal(value) != Decimal(str(normalized)):
                            return _validation_error('Lossful browser argument coercion rejected.', name=name,
                                path='arguments.' + key, constraint='lossless_coercion', parameters=params)
                    except InvalidOperation:
                        return _validation_error('Ambiguous browser argument coercion rejected.', name=name,
                            path='arguments.' + key, constraint='lossless_coercion', parameters=params)
        try:
            from jsonschema.exceptions import best_match
            from jsonschema.validators import validator_for
        except ImportError:
            logger.debug("jsonschema unavailable; keeping required-only validation for %s", name)
            return _contract_unavailable(name, 'missing_validator') if strict else None
        validator_cls = validator_for(validation_schema)
        validator_cls.check_schema(validation_schema)
        validation_error = best_match(validator_cls(validation_schema).iter_errors(candidate_args))
        if validation_error is None:
            return None
        path = _validation_path(validation_error)
        constraint = str(getattr(validation_error, "validator", None) or "schema")
        detail = re.sub(r"\s+", " ", str(validation_error.message)).strip()
        if len(detail) > 600:
            detail = detail[:597] + "..."
        return _validation_error(
            f"tool_call to '{name}' failed argument validation at {path} "
            f"({constraint}): {detail}. The tool was NOT invoked.",
            path=path, constraint=constraint, parameters=params, name=name)
    except Exception:  # Browser calls fail closed on validator faults.
        logger.debug("validate_deferred_call_args failed for %s", name, exc_info=True)
        return _contract_unavailable(name, 'invalid_contract') if strict else None


def normalize_tool_call_entries(args: dict[str, Any]) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Normalize ``tool_call`` arguments into a ``calls[]`` list of entries.

    Accepts the advertised batch shape ``{"calls": [{"name", "arguments"}, ...]}``
    and, tolerantly, the legacy single shape ``{"name": ..., "arguments": ...}``
    (a single call is a batch of one). Each entry's ``arguments`` is coerced to
    a dict (JSON strings parsed, ``None`` → ``{}``). Returns ``(entries, None)``
    or ``([], error_message)``.
    """
    raw_calls = args.get("calls")
    if raw_calls is None:
        # Legacy single shape.
        if not str(args.get("name") or "").strip():
            return [], "tool_call requires 'calls' (an array of {name, arguments})"
        raw_calls = [{"name": args.get("name"), "arguments": args.get("arguments")}]
    if isinstance(raw_calls, str):
        # Tolerate the model emitting the batch envelope as a JSON string —
        # mirror the per-entry `arguments` handling below (#114484).
        try:
            raw_calls = json.loads(raw_calls)
        except json.JSONDecodeError as e:
            return [], f"tool_call 'calls' is not valid JSON: {e}"
    if isinstance(raw_calls, dict):
        raw_calls = [raw_calls]
    if not isinstance(raw_calls, list) or not raw_calls:
        return [], "tool_call 'calls' must be a non-empty array of {name, arguments}"

    entries: list[dict[str, Any]] = []
    for position, raw in enumerate(raw_calls):
        if not isinstance(raw, dict):
            return [], f"tool_call calls[{position}] must be an object with 'name' and 'arguments'"
        name = str(raw.get("name") or "").strip()
        if not name:
            return [], f"tool_call calls[{position}] requires a 'name'"
        if name in BRIDGE_TOOL_NAMES:
            return [], f"tool_call cannot invoke '{name}' (it is itself a bridge tool)"
        raw_args = raw.get("arguments")
        if raw_args is None or (isinstance(raw_args, str) and not raw_args.strip()):
            # "" / whitespace is how some OpenAI-compatible gateways spell "no arguments" for a
            # parameterless tool (#83937); the loop already treats an empty outer arguments string
            # as {} (turn_tool_validation), and a missing required param still surfaces below via
            # validate_deferred_call_args instead of an opaque JSON parse error.
            raw_args = {}
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError as e:
                return [], f"tool_call calls[{position}].arguments is not valid JSON: {e}"
        if not isinstance(raw_args, dict):
            return [], f"tool_call calls[{position}].arguments must be an object"
        entries.append({"name": name, "arguments": raw_args})
    return entries, None


_ECHO_ARGS_MAX_CHARS = 1500


def local_batch_error(entries: list[dict[str, Any]]) -> str:
    """Rejection for a multi-entry batch that names a local tool. Restates the valid
    shape with the caller's OWN first entry: small models re-send an identical batch
    when told only the constraint, and the echoed payload is what gets them unstuck."""
    first = entries[0]
    args = json.dumps(first.get("arguments", {}), ensure_ascii=False, separators=(",", ":"))
    if len(args) > _ECHO_ARGS_MAX_CHARS:
        args = "{...}"  # keep the correction readable; the model still has its own arguments
    retry = '{{"calls":[{{"name":{},"arguments":{}}}]}}'.format(json.dumps(first["name"], ensure_ascii=False), args)
    remaining = (f" then issue the remaining {len(entries) - 1} call(s) as separate tool_call invocations"
                 if len(entries) > 1 else "")
    return (
        f"tool_call takes exactly one entry for local tools; you sent {len(entries)}. "
        f"Retry with only: {retry}{remaining}. Only connectors__ names may be batched together."
    )


def not_deferrable_error(name: str) -> str:
    """Rejection for a ``tool_call`` naming something that is not a deferred tool.
    Two different mistakes reach here and need opposite corrections: a directly-listed
    tool (call it without the bridge) vs. an unknown name — typically a deferred MCP tool
    cited by its bare suffix instead of the full ``mcp__<server>__{tool}`` name. Telling
    the second group 'call it directly' is the opposite of what they must do."""
    from tools.tool_search import _core_tool_names  # late: tool_search imports this module
    if name in _core_tool_names() or _registry_entry(name) is not None:
        return (f"'{name}' is a directly-listed tool, not a deferred one. "
                "Call it directly instead of via tool_call.")
    suffix = f"__{name}"
    try:
        from tools.registry import registry
        candidates = sorted(n for n in registry.get_all_tool_names() if n.endswith(suffix))
    except Exception:
        candidates = []
    hint = (f" Did you mean {', '.join(repr(c) for c in candidates)}?" if candidates
            else " Use tool_search to find the exact name.")
    return (f"'{name}' is not a known tool name. Deferred tools must be invoked through tool_call "
            f"by the exact name tool_search returns (e.g. mcp__<server>__{{tool}}).{hint}")
