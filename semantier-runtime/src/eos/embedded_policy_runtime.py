from __future__ import annotations

import hashlib
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_PATH_RE = re.compile(r"input(?:\.[A-Za-z_][A-Za-z0-9_]*|\[[0-9]+\])+")
_NUM_RE = re.compile(r"^-?(?:[0-9]+(?:\.[0-9]+)?)$")

# Module-level bundle cache keyed by "<absolute_path>:<mtime_ns>".
# Shared across all callers (all gateways in the same process) so each policy
# file is parsed at most once per on-disk change.
_bundle_cache: dict[str, "EmbeddedPolicyBundle"] = {}
_bundle_cache_lock = threading.Lock()


@dataclass(frozen=True)
class EmbeddedPolicyBundle:
    """Hash-pinned embedded policy bundle.

    This is a zero-dependency Rego-subset runtime for deterministic replay.
    """

    policy_ref: str
    policy_sha256: str
    allow_expression: str


def _split_once(expr: str, op: str) -> tuple[str, str] | None:
    idx = expr.find(op)
    if idx < 0:
        return None
    return expr[:idx].strip(), expr[idx + len(op) :].strip()


def _parse_allow_expression(policy_text: str) -> str:
    in_allow = False
    for raw in policy_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("allow") and line.endswith("{"):
            in_allow = True
            continue
        if in_allow and line == "}":
            break
        if in_allow:
            return line
    raise ValueError("No allow expression found in policy text")


def load_embedded_policy(policy_path: str | Path) -> EmbeddedPolicyBundle:
    """Load (or return cached) an EmbeddedPolicyBundle for *policy_path*.

    The cache key is ``<absolute_path>:<mtime_ns>`` so any on-disk change
    automatically invalidates the entry while keeping the parsed bundle alive
    across all gateway calls that share the same process.
    """
    p = Path(policy_path).resolve()
    mtime_ns = p.stat().st_mtime_ns
    cache_key = f"{p}:{mtime_ns}"

    with _bundle_cache_lock:
        if cache_key in _bundle_cache:
            return _bundle_cache[cache_key]

    text = p.read_text(encoding="utf-8")
    bundle = EmbeddedPolicyBundle(
        policy_ref=str(p),
        policy_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        allow_expression=_parse_allow_expression(text),
    )

    with _bundle_cache_lock:
        _bundle_cache[cache_key] = bundle
        # Evict stale entries for the same path (previous mtime).
        stale = [k for k in _bundle_cache if k.startswith(f"{p}:") and k != cache_key]
        for k in stale:
            del _bundle_cache[k]

    return bundle


def _resolve_input_path(input_obj: dict[str, Any], path_expr: str) -> Any:
    # path_expr format: input.semantic_claim.resources[0].delta
    if not path_expr.startswith("input"):
        raise ValueError(f"Unsupported term: {path_expr}")

    cur: Any = input_obj
    rest = path_expr[len("input") :]
    i = 0
    while i < len(rest):
        if rest[i] == ".":
            j = i + 1
            while j < len(rest) and (rest[j].isalnum() or rest[j] == "_"):
                j += 1
            key = rest[i + 1 : j]
            if not isinstance(cur, dict) or key not in cur:
                raise KeyError(f"Missing key in input path: {key}")
            cur = cur[key]
            i = j
            continue
        if rest[i] == "[":
            j = rest.find("]", i)
            if j < 0:
                raise ValueError(f"Malformed index in path: {path_expr}")
            idx = int(rest[i + 1 : j])
            if not isinstance(cur, list):
                raise TypeError(f"Expected list for index access in path: {path_expr}")
            cur = cur[idx]
            i = j + 1
            continue
        raise ValueError(f"Unsupported path token in: {path_expr}")
    return cur


def _to_number(value: Any) -> float:
    if isinstance(value, bool):
        # Normalize Python/JSON booleans so gate-check inputs like
        # {"four_flow_consistent": True} work without callers having to cast.
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and _NUM_RE.match(value.strip()):
        return float(value.strip())
    raise TypeError(f"Term is not numeric: {value!r}")


def _eval_arith(expr: str, input_obj: dict[str, Any]) -> float:
    # Minimal deterministic arithmetic evaluator supporting + and - with input paths and literals.
    tokens = re.split(r"(\+|\-)", expr)
    if not tokens:
        raise ValueError("Empty arithmetic expression")

    def eval_term(term: str) -> float:
        t = term.strip()
        if not t:
            raise ValueError("Empty term")
        if _NUM_RE.match(t):
            return float(t)
        if _PATH_RE.fullmatch(t):
            return _to_number(_resolve_input_path(input_obj, t))
        raise ValueError(f"Unsupported term syntax: {t}")

    value = eval_term(tokens[0])
    i = 1
    while i + 1 < len(tokens):
        op = tokens[i].strip()
        rhs = eval_term(tokens[i + 1])
        if op == "+":
            value += rhs
        elif op == "-":
            value -= rhs
        else:
            raise ValueError(f"Unsupported operator: {op}")
        i += 2
    return value


def evaluate_embedded_policy(bundle: EmbeddedPolicyBundle, input_obj: dict[str, Any]) -> dict[str, Any]:
    expr = bundle.allow_expression

    split = _split_once(expr, "==")
    if split is None:
        raise ValueError(f"Only '==' allow expressions are currently supported: {expr}")

    lhs, rhs = split
    lhs_val = _eval_arith(lhs, input_obj)
    rhs_val = _eval_arith(rhs, input_obj)
    allowed = lhs_val == rhs_val

    return {
        "allow": allowed,
        "policy_ref": bundle.policy_ref,
        "policy_sha256": bundle.policy_sha256,
        "allow_expression": bundle.allow_expression,
        "lhs_value": lhs_val,
        "rhs_value": rhs_val,
        "deterministic": True,
    }


def evaluate_embedded_policy_batch(
    policy_path: str | Path,
    inputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate *inputs* against a single policy in one call path.

    The bundle is loaded (and cached) once; all inputs are evaluated against
    the same parsed expression.  This is the preferred entry point for the
    case-matrix simulator so gateway calls and simulator calls share the cache.

    Args:
        policy_path: Path to a .rego file.
        inputs: List of input dicts, each in the shape expected by the policy.

    Returns:
        List of result dicts (same shape as :func:`evaluate_embedded_policy`)
        with an additional ``input_index`` key for traceability.
    """
    bundle = load_embedded_policy(policy_path)
    results: list[dict[str, Any]] = []
    for idx, input_obj in enumerate(inputs):
        result = evaluate_embedded_policy(bundle, input_obj)
        result["input_index"] = idx
        results.append(result)
    return results


def clear_policy_cache() -> int:
    """Evict all cached bundles.  Returns the number of entries removed.

    Primarily useful in tests that write temporary policy files.
    """
    with _bundle_cache_lock:
        count = len(_bundle_cache)
        _bundle_cache.clear()
    return count
