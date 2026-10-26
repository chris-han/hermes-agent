---
sidebar_position: 4
title: Local Models
description: Run models entirely on your own machine — no account, no API key, nothing leaves your computer.
---

# Local Models

Hermes can run open models entirely on your own machine. It downloads and
manages the inference engine (llama.cpp), picks the right build of each
model for your hardware, and handles memory so you never configure context
sizes, GPU layers, or quantization. You pick a model; Hermes does the rest.

Nothing leaves your computer: no account, no API key, and no network access
after a model is downloaded.

## Terminal setup and desktop endpoints

The desktop app no longer installs, downloads or manages local models. Its
**Settings → Model service → Accounts** offers Custom endpoint and Semantier Cloud.
Use **Custom endpoint** for an independently running OpenAI-compatible local or
remote server, including `http://localhost:8080/v1`. Existing model files and
user caches are preserved.

From a terminal, `hermes model` → **Local models** retains the shared runtime setup:
it lists the catalog for your machine, installs the pinned PM engine, downloads
the model with live progress, starts the server and makes it your default.
Models already on disk are offered too. Ctrl+C pauses a download; choosing the
model again resumes it. No desktop download controls or local-model launch flag
are provided.

## How Hermes picks what to download

Every model in the catalog is priced against **your machine** before you
download anything. Each row shows:

- **Memory fit** — green (*Fits your GPU*: runs entirely in GPU memory),
  amber (*Uses system RAM*: works, but slower), or red (*Too big for this
  machine*).
- **Context** — the window the model starts with and the maximum it can
  grow to.
- The download size of the build selected for your hardware.

Models ship in several quality grades (quantizations). Hermes picks the
highest-quality build that runs fully on your GPU; machines with less
memory get a more compact build of the same model with the same
guarantees. Below 4-bit the quality loss is too severe, so Hermes never
offers builds smaller than that — a machine that can't run the 4-bit
build spilled to system RAM simply can't run that model.

Models that don't fit stay visible with the reason, so you always know
what a hardware upgrade would unlock.

Hermes recommends the highest-quality model that runs entirely in GPU or
unified memory at a predicted 20 tokens per second or more. A machine's
maker can set a different default, and Hermes recommends that model
whenever it fits. When no model reaches the floor, Hermes recommends none,
and you can still choose any model that fits.

## How memory management works

Local models live or die by memory placement, so Hermes manages it
end-to-end and exposes no knobs:

- **Models start at a context window that fully fits your GPU** and grow
  toward their native maximum as your conversation needs more room. You
  may see "Context window grown" in the status feed during long sessions
  — that's the window expanding, not an error.
- **Every recommended model gets at least a 64K context window.** When a
  model is larger than your GPU's memory, Hermes deliberately places the
  overflow in system RAM in the order that hurts least (expert weights
  first, never the attention cache), trading some speed to protect the
  context guarantee.
- **Memory fit includes the launch configuration**, not just the model file:
  context state, runtime buffers, the vision projector, and MTP buffers all
  count. For multi-token prediction (MTP), Hermes uses smaller batches when
  larger batches would spill at the same context window. MTP stays enabled.
  The same calculation runs when a grown window is restored after restart.
- **Conversation compression follows a growth check.** If a larger window
  cannot fit, generation is too slow, or the native maximum is reached,
  Hermes compresses instead of claiming a window the server did not receive.
- Idle models are unloaded after 15 minutes to free GPU memory; they
  reload automatically on the next message.

## Using your own llama-server

If a llama-server is already running on your machine, Hermes detects it
and uses it instead of starting its own. Point a custom endpoint at any
OpenAI-compatible server for full manual control — the managed runtime is
a default, not a requirement. You can enter the server root (for example
`http://127.0.0.1:8080`) or the full `/v1` URL: the endpoint test tries
both and saves the variant that actually served `/models`, so chat
requests go to the same prefix the model list came from. For manual setups (Ollama, MLX, custom
builds, headless CLI machines), see
[Run Hermes Locally with Ollama](../guides/local-ollama-setup.md) and
[Run Local LLMs on Mac](../guides/local-llm-on-mac.md).

## Configuration

The managed runtime is controlled by the `local_runtime` section of
`config.yaml` for CLI and headless use:

```yaml
local_runtime:
  enabled: false     # true = start the managed server with Hermes.
  backend: auto      # auto | cuda | metal | vulkan | hip | cpu
  detect_ports: [8081]  # extra ports to probe for a llama-server you run
                        # yourself (the default probe is :8080 only)
```

Running `llama-server` yourself on a fixed port works with the same
`model.provider: llamacpp` selection — either list the port in
`local_runtime.detect_ports`, or define the endpoint explicitly under
`providers:` (an explicit entry wins over server detection):

```yaml
providers:
  llamacpp:
    base_url: http://127.0.0.1:8081/v1
    model: my-model
```

The `/model` → **Local** picker row and `provider: llamacpp` resolve to that
server; with no server reachable the error names the local runtime ("the local
model server isn't running") instead of an unknown-provider or missing-API-key
message.

Engine versions come only from PM's lockfile, not a `local_runtime.tag`
override. Boot uses an installed PM engine without downloading. When a new
pin is available, use the terminal setup flow to install it.

Models live in the machine-shared `models/` directory. Engine binaries live
in PM's store; `runtimes/llamacpp/` holds mutable presets and server state.
Selecting a local model as your main model uses the standard
`model.provider: llamacpp` + `model.default` settings.

## Requirements and limits

- **Windows:** CUDA on supported NVIDIA targets, Vulkan on x64, or CPU.
  **Linux:** CUDA on x64 and arm64 NVIDIA (needs a CUDA 13 driver), Vulkan, or CPU.
  **macOS:** Metal or CPU. HIP/ROCm is an explicit choice on supported x64
  targets. Unsupported backend/target pairs fail before any download.
- `backend: auto` prefers CUDA for NVIDIA and Vulkan for AMD and Intel GPUs,
  then falls back to the builds listed above for your platform (CPU when no
  GPU is found). Behind a Vulkan or HIP engine, a discrete card is sized from
  its own memory (system RAM is spill space) and each launch subtracts what
  other programs hold on it; an integrated GPU shares system RAM and is sized
  from that. Quickstart installs the engine before choosing the model, so the
  pick reflects the card.
- A GPU with 8 GB+ of memory runs the small catalog models comfortably;
  16 GB+ runs the 27–35B models at high quality.
- Model completeness is checked against the server's response, not catalog
  size estimates. Interrupted transfers retain partials for resume; incomplete
  files are not published. Engine archives are SHA-256 verified before use.
- Deleting a model removes every file it staged, including vision
  adapters and speculative-decoding companions.
