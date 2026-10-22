# Prompt Engineering in Semantier Agent System

This document explains how the prompt files under `src/prompts/` are used, composed, and injected into the Semantier agent system for a typical user request.

---

## Prompt Catalog

| Location | File | Role |
|---|---|---|
| `agents/` | `GOVERNANCE.md` | Immutable governance constraints — prepended to every conversation |
| `agents/` | `SOUL.md` | Agent identity, communication style, and tone directives |
| `agents/` | `WRAPPER_SYSTEM_INSTRUCTION.md` | Base system instruction; wraps SOUL directives for runtime injection |
| `agents/` | `DIGITAL_CFO_USER_GUIDANCE.md` | Base digital-CFO positioning and example business questions the agent should prioritize |
| `agents/` | `ACTIVE_ORGANIZATION_CONTEXT.md` | Template — injected when the user has an active org membership |
| `agents/` | `ORGANIZATION_TOOL_ROUTING_CONTEXT.md` | Template — injected for analytics queries on non-default org |
| `agents/` | `DEMO_DATASET_CONTEXT.md` | Template — injected when the active org uses a DEMO dataset |
| `agents/` | `DEFAULT_ANALYTICS_SAMPLE_CONTEXT.md` | Template — injected for analytics queries on the default sample org |
| `agents/` | `governed_context.json` | Intent classification marker lists (analytics, company, 3-year queries) |
| `eos/` | `SIGNAL_PROMPT.md` | Template — drives structured J_t justification from EOS event signals |
| `semantic_completion/` | `amount.md`, `currency.md`, … | Interactive field-collection prompts for the semantic completion flow |

---

## Component Diagram

```mermaid
graph TB
    subgraph PromptFiles["Prompt Files (src/prompts/)"]
        direction TB
        GOV["agents/GOVERNANCE.md\n(immutable constraints)"]
        SOUL["agents/SOUL.md\n(identity & tone)"]
        WRAP["agents/WRAPPER_SYSTEM_INSTRUCTION.md\n(base system instruction)"]
        CFO["agents/DIGITAL_CFO_USER_GUIDANCE.md\n(digital CFO positioning)"]
        AOC["agents/ACTIVE_ORGANIZATION_CONTEXT.md\n(template)"]
        OTR["agents/ORGANIZATION_TOOL_ROUTING_CONTEXT.md\n(template)"]
        DDC["agents/DEMO_DATASET_CONTEXT.md\n(template)"]
        DAS["agents/DEFAULT_ANALYTICS_SAMPLE_CONTEXT.md\n(template)"]
        GCJ["agents/governed_context.json\n(intent markers)"]
        SIG["eos/SIGNAL_PROMPT.md\n(EOS signal template)"]
        SC["semantic_completion/*.md\n(field prompts)"]
    end

    subgraph AssemblyLayer["Prompt Assembly Layer (src/agents/)"]
        direction TB
        LAUNCHER["launcher.py\nload_agent_prompt()\nload_default_soul_md()\nload_default_governance_md()"]
        GC["governed_context.py\nbuild_governed_runtime_context_prompt()\nIntent classifiers (_is_analytics_query etc.)"]
        SA["semantier_agent.py\n_compose_effective_system_message()\nWRAPPER_SYSTEM_INSTRUCTION constant"]
    end

    subgraph IdentityLayer["Identity & Authority Layer"]
        GI["gateway_identity.py\nget_user_organization_context()\nresolve_user_id_for_workspace()"]
        AUTH["auth_db.py\nOrganization / User records"]
    end

    subgraph RuntimeLayer["Runtime Layer"]
        AGENT["SemantierAgent\nrun_conversation()"]
        HB["_hermes_home_bound()\nWorkspace isolation context"]
        AI["hermes-agent AIAgent\nrun_conversation(prompt, system_message=...)"]
    end

    subgraph OtherFlows["Other Prompt Flows"]
        EOS["eos/runtime_entry.py\nEOS governance execution"]
        SCS["SemanticCompletionService\nInteractive field collection"]
    end

    GOV --> LAUNCHER
    SOUL --> LAUNCHER
    WRAP --> LAUNCHER
    CFO --> LAUNCHER
    GCJ --> GC
    AOC --> GC
    OTR --> GC
    DDC --> GC
    DAS --> GC

    LAUNCHER --> SA
    GC --> SA
    GI --> GC
    AUTH --> GI

    SA --> AGENT
    AGENT --> HB
    HB --> AI

    SIG --> EOS
    SC --> SCS
```

---

## Prompt Assembly: Layer Structure

For every call to `SemantierAgent.run_conversation()`, the effective system message is assembled in two layers:

### Layer 1 — Static Base (always included)

Built once at module load time in `semantier_agent.py`:

```
WRAPPER_SYSTEM_INSTRUCTION =
    GOVERNANCE.md          ← immutable governance constraints
    +
    WRAPPER_SYSTEM_INSTRUCTION.md   ← operational directives (identity, confidentiality, routing)
    +
    DIGITAL_CFO_USER_GUIDANCE.md    ← user-facing digital CFO positioning and in-scope business questions
```

> `SOUL.md` is available for skills and downstream persona use but is not directly concatenated into the wrapper; it lives as a separate loadable artifact via `load_default_soul_md()`.

### Layer 2 — Governed Context (conditional, per-request)

Built dynamically by `build_governed_runtime_context_prompt()` in `governed_context.py`.  
Each sub-section is gated by **identity resolution** and **intent classification**:

Relative analytics periods are resolved by `agents.temporal_resolution` before
prompt assembly. Prompt text may carry the resolved period IDs for user-facing
guidance, but the date math is not delegated to the LLM.

| Prompt | Condition |
|---|---|
| `ACTIVE_ORGANIZATION_CONTEXT.md` | `membership_status == "active"` |
| `ORGANIZATION_TOOL_ROUTING_CONTEXT.md` | active + analytics query + non-default org |
| `DEMO_DATASET_CONTEXT.md` | active + `dataset_type == "DEMO"` + analytics query |
| `DEFAULT_ANALYTICS_SAMPLE_CONTEXT.md` | active + default org + analytics query + no explicit company marker |

### Final Assembly

```python
# semantier_agent.py — _compose_effective_system_message()
parts = [WRAPPER_SYSTEM_INSTRUCTION]
if governed_context:
    parts.append(governed_context)
system_message = "\n\n".join(parts)
```

---

## Sequence Diagram — Prompt Assembly Lifecycle for a Typical User Request

```mermaid
sequenceDiagram
    actor User
    participant Gateway as Gateway / Ingress<br/>(webapi_gateway / feishu / weixin)
    participant Agent as SemantierAgent<br/>semantier_agent.py
    participant HB as _hermes_home_bound()<br/>Workspace Isolation
    participant Launcher as launcher.py<br/>Prompt Loader
    participant GC as governed_context.py<br/>Context Builder
    participant GI as gateway_identity.py<br/>Identity Authority
    participant AI as hermes AIAgent<br/>LLM Runtime

    User->>Gateway: User request (message + session context)

    Gateway->>Agent: run_conversation(prompt, hermes_home=...)

    Note over Agent: WRAPPER_SYSTEM_INSTRUCTION already<br/>loaded at module init time:<br/>GOVERNANCE.md + WRAPPER_SYSTEM_INSTRUCTION.md + DIGITAL_CFO_USER_GUIDANCE.md

    Agent->>HB: Enter _hermes_home_bound(target_home)
    HB->>HB: Pin HERMES_HOME env var + context var<br/>Bootstrap workspace runtime home

    Agent->>GC: build_governed_runtime_context_prompt(<br/>  workspace_id, user_message)

    GC->>GI: resolve_user_id_for_workspace(workspace_id)
    GI-->>GC: user_id (from governed auth store)

    GC->>GI: get_user_organization_context(user_id)
    GI-->>GC: org_context {org_id, name, status, dataset_type}

    Note over GC: Intent classification:<br/>_is_analytics_query(user_message)<br/>_mentions_explicit_company_selection()<br/>_is_three_year_query()

    alt membership_status == "active"
        GC->>GC: Load ACTIVE_ORGANIZATION_CONTEXT.md<br/>.format(organization_name=...)
    end

    alt active + analytics query + non-default org
        GC->>GC: Load ORGANIZATION_TOOL_ROUTING_CONTEXT.md<br/>.format(organization_id=..., three_year_guidance=...)
    end

    alt active + DEMO dataset + analytics query
        GC->>GC: Load DEMO_DATASET_CONTEXT.md<br/>.format(organization_name=...)
    end

    alt active + default org + analytics query + no explicit company
        GC->>GC: Load DEFAULT_ANALYTICS_SAMPLE_CONTEXT.md<br/>.format(organization_name=...)
    end

    GC-->>Agent: governed_context (joined prompt fragments or None)

    Agent->>Agent: _compose_effective_system_message()<br/>= WRAPPER_SYSTEM_INSTRUCTION<br/>+ (governed_context if present)

    Agent->>AI: AIAgent.run_conversation(<br/>  prompt=user_message,<br/>  system_message=assembled_system_message)

    Note over AI: LLM executes under assembled constraints.<br/>Tool calls are scoped by platform toolsets<br/>(runtime_toolsets.py → hermes-agent platform registry).

    AI-->>Agent: response dict (with trajectory)
    Agent-->>HB: Exit context (restore HERMES_HOME)
    Agent-->>Gateway: response
    Gateway-->>User: Final answer
```

---

## Other Prompt Flows

### EOS Signal Prompt (`eos/SIGNAL_PROMPT.md`)

Used independently by the EOS governance execution path (`eos/runtime_entry.py`).  
The template receives a `{signal_json}` payload and instructs the model to produce a structured **J_t** (justification triple) with REA semantic claims, ontology version pins, and resource/agent/duality fields.  
This flow is entirely separate from the agent conversation system message.

### Semantic Completion Field Prompts (`semantic_completion/*.md`)

Used by `SemanticCompletionService` for the interactive multi-turn financial event entry flow (Weixin/Feishu channel).  
Each file contains a short prompt string displayed to the user when a required field (`amount`, `currency`, `counterparty`, `invoice_ref`, `description`) is missing.  
These are not injected into the LLM system message; they are user-facing turn prompts surfaced by the API layer.

---

## Governance Invariants

All prompt assembly respects these non-negotiable rules from `AGENTS.md`:

- **Identity is never inferred** — user_id and org_context are resolved only from governed authority sources (`gateway_identity.py` → `auth_db.py`), never from LLM inference or user self-claim.
- **Governance is always first** — `GOVERNANCE.md` is prepended before any other content and cannot be overridden by later prompt fragments.
- **Source code is never disclosed** — `WRAPPER_SYSTEM_INSTRUCTION.md` and `GOVERNANCE.md` both enforce the confidentiality constraint on internal implementation files.
- **Context wins over model memory** — when `governed_context` conflicts with prior assistant messages, the governed context takes precedence (enforced via explicit prompt instruction in `ACTIVE_ORGANIZATION_CONTEXT.md`).
