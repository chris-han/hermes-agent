import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.semantier_agent import SemantierAgent
from eos.artifact_hashing import content_hash
from runtime_paths import workspace_hermes_home_path


_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts" / "eos"


def _load_prompt_template(filename: str) -> str:
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


def _parse_hermes_output(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract J_t and π_t from the final assistant message in a Hermes run.

    Hermes `run_conversation()` returns a `messages` list in the internal
    OpenAI-wire format:
        [
            {"role": "user",      "content": "<signal prompt>"},
            {"role": "assistant", "content": "<justification text or JSON>",
             "tool_calls": [{"function": {"name": ..., "arguments": ...}}]},
            ...
        ]

    The last assistant message is the agent's final output:
    - `content`    → J_t: the natural-language (or JSON) justification
    - `tool_calls` → π_t: the proposed execution plan as structured tool calls

    Returns a dict with keys `J_t`, `pi_t`, `trajectory_hash`.
    """
    # Find the last assistant message (skip trailing tool-result messages).
    last_assistant: Optional[Dict[str, Any]] = None
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            last_assistant = msg
            break

    if last_assistant is None:
        raise ValueError("Hermes run produced no assistant message")

    # --- Parse J_t from `content` ---
    raw_content = last_assistant.get("content") or ""
    # Strip <think>/<reasoning> blocks — they are internal scratchpad, not J_t.
    stripped = re.sub(
        r"<(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)\b[^>]*>.*?"
        r"</(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)>",
        "",
        raw_content,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    # Attempt to parse structured JSON from the content block; fall back to
    # wrapping in a plain `claim` field so downstream validators receive a dict.
    J_t: Dict[str, Any]
    try:
        J_t = json.loads(stripped)
        if not isinstance(J_t, dict):
            raise ValueError("not a dict")
    except (json.JSONDecodeError, ValueError):
        J_t = {"claim": stripped}

    # --- Parse π_t from `tool_calls` ---
    raw_tool_calls = last_assistant.get("tool_calls") or []
    steps = []
    for tc in raw_tool_calls:
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        name = fn.get("name", "")
        args_raw = fn.get("arguments", "{}")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except json.JSONDecodeError:
            args = {"_raw": args_raw}
        steps.append({"tool": name, "arguments": args})

    # π_t as a program structure matching what `validate_execution` expects.
    pi_t: Dict[str, Any] = {"steps": steps}

    # trajectory_hash: H over the full messages list (canonical JSON).
    trajectory_hash = content_hash(messages)

    return {"J_t": J_t, "pi_t": pi_t, "trajectory_hash": trajectory_hash}


class HermesClient:
    """Hermes Agent adapter.

    Calls the real Hermes `AIAgent` when available; falls back to a
    deterministic stub for tests and environments where the Hermes submodule
    is not installed.  Both paths expose the same interface so callers are
    unaffected by which backend is active.

    The key design principle (per paper §4):
        Hermes *generates* J_t and π_t via agentic reasoning.
        Semantier *validates* them via the governance gate.
    This adapter is the bridge: it drives a Hermes run and extracts the
    structured (J_t, π_t) pair from the agent's final output message.
    """

    def __init__(
        self,
        hermes_base_url: Optional[str] = None,
        hermes_api_key: Optional[str] = None,
        hermes_model: Optional[str] = None,
    ):
        self._base_url = hermes_base_url
        self._api_key = hermes_api_key
        self._model = hermes_model or "anthropic/claude-sonnet-4-5"

    # ------------------------------------------------------------------
    # Primary public interface
    # ------------------------------------------------------------------

    def run(
        self,
        signal: Dict[str, Any],
        *,
        workspace_id: str | None = None,
        hermes_home: Path | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        organization_id: str | None = None,
        authority_context_ref: str | None = None,
    ) -> Dict[str, Any]:
        """Drive a Hermes agentic run for *signal* and return structured outputs.

        Returns:
            {
                "J_t":              dict,   # justification (semantic claim + pins)
                "pi_t":             dict,   # execution program (steps list)
                "trajectory_hash":  str,    # H(canonical(messages))
                "messages":         list,   # full Hermes message history
            }
        """
        try:
            return self._run_via_hermes_agent(
                signal,
                workspace_id=workspace_id,
                hermes_home=hermes_home,
                session_id=session_id,
                user_id=user_id,
                organization_id=organization_id,
                authority_context_ref=authority_context_ref,
            )
        except Exception:
            return self._run_stub(signal)

    # ------------------------------------------------------------------
    # Real Hermes path (via semantier wrapper)
    # ------------------------------------------------------------------

    def _run_via_hermes_agent(
        self,
        signal: Dict[str, Any],
        *,
        workspace_id: str | None = None,
        hermes_home: Path | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        organization_id: str | None = None,
        authority_context_ref: str | None = None,
    ) -> Dict[str, Any]:
        """Call Hermes through semantier's wrapper and parse its output."""
        agent = SemantierAgent(
            base_url=self._base_url,
            api_key=self._api_key,
            model=self._model,
        )
        prompt = _build_signal_prompt(signal)
        resolved_hermes_home = hermes_home
        if resolved_hermes_home is None and workspace_id:
            resolved_hermes_home = workspace_hermes_home_path(workspace_id)
        result = agent.run_conversation(
            prompt,
            hermes_home=resolved_hermes_home,
            session_id=session_id,
            user_id=user_id,
            workspace_id=workspace_id,
            organization_id=organization_id,
            authority_context_ref=authority_context_ref,
        )
        messages: List[Dict[str, Any]] = result.get("messages", [])
        parsed = _parse_hermes_output(messages)
        parsed["messages"] = messages
        return parsed

    # ------------------------------------------------------------------
    # Deterministic stub (fallback / offline tests)
    # ------------------------------------------------------------------

    def _run_stub(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Deterministic fallback used when the Hermes submodule is unavailable."""
        amount = str(signal.get("amount", "0"))
        currency = signal.get("currency", "USD")
        invoice_id = signal.get("invoice_id")
        payment_id = signal.get("payment_id")
        counterparty = signal.get("counterparty")
        invoice_counterparty = signal.get("invoice_counterparty", counterparty)
        payment_counterparty = signal.get("payment_counterparty", counterparty)

        resources = [
            {"type": "cash", "delta": amount, "currency": currency, "ref": payment_id},
        ]
        if invoice_id:
            resources.append(
                {"type": "receivable", "delta": f"-{amount}", "currency": currency, "ref": invoice_id}
            )

        J_t: Dict[str, Any] = {
            "claim": "cash_receipt_settlement",
            "semantic_claim": {
                "event": {"type": "cash_receipt", "ref": payment_id},
                "resources": resources,
                "agents": [
                    {"role": "payer", "id": payment_counterparty},
                    {"role": "receiver", "id": "org_001"},
                ],
                "duality": {"settles": invoice_id},
                "counterparties": {
                    "invoice": invoice_counterparty,
                    "payment": payment_counterparty,
                },
            },
            "pins": {},
        }

        pi_t: Dict[str, Any] = {
            "steps": [
                {"tool": "match_payment_to_invoice", "arguments": {"invoice_id": invoice_id}},
                {"tool": "post_journal_entry", "arguments": {"amount": amount, "currency": currency}},
            ]
        }

        messages = [
            {"role": "user", "content": _build_signal_prompt(signal)},
            {"role": "assistant", "content": json.dumps(J_t)},
        ]
        return {
            "J_t": J_t,
            "pi_t": pi_t,
            "trajectory_hash": _canonical_hash(messages),
            "messages": messages,
        }


def _build_signal_prompt(signal: Dict[str, Any]) -> str:
    """Render a signal dict into a structured prompt for Hermes."""
    return _load_prompt_template("SIGNAL_PROMPT.md").format(
        signal_json=json.dumps(signal, indent=2, ensure_ascii=False),
    )
