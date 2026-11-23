# Desktop perf harness

One systematized way to measure desktop rendering/interaction performance,
diff it against a committed baseline, and fail on regressions. It replaces the
dozen one-off `measure-*` / `profile-*` scripts that each reinvented the CDP
client, arg parsing, stats, and output (and never had a baseline).

## Quick start

```bash
# Isolated instance (recommended) — no running app or LLM credits needed.
# Its own --user-data-dir + HERMES_HOME means it never collides with `hgui`.
npm run perf -- --spawn

# Or: launch an isolated instance once, attach repeatedly (faster iteration).
npm run perf:serve            # leaves an instance on :9222
npm run perf                  # attaches, runs the CI suite, gates on baseline

# One scenario, with a CPU profile:
npm run perf -- stream --cpuprofile --tokens 800

# Representative PRODUCTION numbers (minified React, not the ~3x-slower dev build):
npm run perf -- cold-start stream keystroke transcript --spawn --prod

# Re-capture the baseline on your reference device, then commit baseline.json:
npm run perf -- cold-start stream keystroke transcript --spawn --prod --update-baseline
```

## Profiling an existing workspace

```bash
node scripts/perf/run.mjs live-window --seconds 15 --json ~/.hermes/cache/scratch/live-window.json
# Attribution is a separate pass, not an FPS comparison:
node scripts/perf/run.mjs live-window --seconds 10 --cpuprofile ~/.hermes/cache/scratch
```

`live-window` never opens/closes tabs, seeds messages, moves focus, or forces GC.
It records frame intervals, long tasks, visible/mounted transcript counts, and
heap/DOM counters. Keep the same panes and workload for before/after captures.
Record restarts, HMR, and code updates separately: a fresh renderer is not a
valid after measurement for a long-running renderer. Heap counters alone do not
prove a leak. Heavy render/atom counters must be stopped before timing.

## Dev vs prod

By default the harness measures the **dev** renderer (fast to spin up, good for
relative regression checks). Pass `--prod` (with `--spawn`) to build a
production renderer *with the probe included* (`VITE_PERF_PROBE=1`) and measure
minified React — the representative shipped numbers. The committed baseline is
captured with `--prod`.

## Why isolation matters

The measurement this harness exists to run was historically blocked: a running
`hgui` holds the Electron single-instance lock, so a second instance quit
immediately. `--spawn` / `perf:serve` launch with their own `--user-data-dir`
(separate lock scope), their own `HERMES_HOME` (separate backend + sessions),
and their own `--remote-debugging-port`. Synthetic scenarios drive `$messages`
directly via `window.__PERF_DRIVE__`, so no LLM credits are spent.

## Scenarios

| scenario | tier | measures | replaces |
|---|---|---|---|
| `stream` | ci | streaming longtasks, frame p95/p99, mutation cadence | measure-synthetic-stream, profile-synth-stream, profile-long-stream |
| `stream --real` | backend | same, from a real LLM stream | measure-real-stream, profile-real-stream |
| `keystroke` | ci | composer keystroke → paint latency | measure-latency, profile-typing, leak-typing |
| `transcript` | ci | large-transcript mount + paint cost | (new) |
| `render-churn` | ci | per-component render attribution + store churn while N tabs stream | (new) |
| `idle-cost` | report | busy-but-silent tiles: idle commit rate, + fps while resizing / typing | (new) |
| `right-pane` | report | file tree + persistent xterm tabs under chat/terminal output and split dragging | (new) |
| `cold-start` | cold | launch → CDP → driver → first paint (fresh spawn/run) | (new) |
| `first-token` | backend | Enter → first assistant-bubble DOM growth (approximation) | (new) |
| `submit` | backend | Enter → cleared → user msg painted, scroll jump | measure-submit, measure-jump |
| `session-switch` | backend | route → first-paint → settle | profile-session-switch |
| `session-load` | backend | how far a session's transcript moves after first paint | (new) |
| `profile-switch` | backend | rail click → sidebar settled | measure-profile-switch |

`ci` + `cold` scenarios need no backend/credits and are gated against
`baseline.json` (`cold-start` requires `--spawn` since it measures a fresh
launch, and must be run in its own invocation). `backend` scenarios need a live
backend (and `--spawn` or a real session/credits) and are report-only.

CPU profiling is a cross-cutting `--cpuprofile` flag on any scenario (it wraps
the run in `Profiler.start/stop` and prints a top-self-time table), replacing
every standalone `profile-*` script.

## Windows first-visible-content MVL protocol

Measure one focused native Electron window, one disposable local test identity,
one fresh SessionDB and one conversation with sequential turns. Prefer the
registered package and verify its executable and renderer hashes. If an
instrumented source build is needed, identify it separately and use a minified
production renderer with the ordinary GPU and sandbox settings. Check process
ownership, isolated HOME and USERPROFILE, Hermes home, Electron user data and
loopback port bindings before the first Enter. Use only the mock key and
loopback model in the core E2E fixture. The default perf spawn path can seed
credentials from the caller's Hermes home, so it is unsuitable for this MVL.

Extend the existing `first-token` scenario and CDP client to capture the
following ordered observations. The existing DOM-growth result is a rough
diagnostic and cannot satisfy this protocol.

| Event | Evidence |
|---|---|
| Enter | Trusted, single CDP keypress observed by a capture-phase composer listener |
| Gateway acceptance | Accepted `prompt.submit` after durable user-row staging, with RPC, session and turn identity |
| Agent request | Outbound inference attempt at the agent client boundary |
| Provider receive / first text | Core fixture receipt and first nonempty content write, including delay-start mark |
| Gateway delta | First matching content delta at the server emission boundary |
| Renderer receive | Matching delta at the renderer WebSocket handler or renderer-side CDP frame receipt |
| First visible paint | First compositor frame containing the exact scripted text in the current assistant message |

Carry a unique `U<turn>-<run nonce>` prompt marker through the RPC and provider,
and use a distinct response marker in the answer. Join the delta and completion
on session, turn, and sequence. Record UTC timestamps for provenance. Measure
with monotonic clocks and retain raw values. Calibrate each clock against a
reference monotonic clock before and after the run using bracketed exchanges.
Retain offset bounds and reject orderings or rankings that cannot be resolved
within the calibration error. A DOM mutation or animation-frame callback alone
does not verify a compositor frame. Retain timestamped visual frame evidence
around the transition and report the frame interval when the first presentation
is only bounded by two frames.

Send reasoning and empty content as controls before the unique nonempty text,
then apply the core provider's 200 ms first-text timer. Reject markers in user
bubbles or reasoning, hidden and clipped nodes, placeholders and status text.
Require the scripted provider output, first delta, painted assistant text and
final completion to agree. Wait for renderer readiness and the correlated
completion plus composer idle between turns. Run two warmups followed by ten
measured turns. Preserve failed attempts in their original order with their
timeout or rejection reasons.

Report raw events and stage intervals per turn, the conventional median
(average of the two middle observations), nearest-rank p95 (maximum at ten),
and timeout count. Rank stages by median and show the injected provider delay
separately when identifying the largest actionable stage. Stop after that
finding. Acceptance requires all ten measured turns and zero false detections.

## Adding a scenario

Create `scenarios/<name>.mjs` exporting `{ name, tier, description, run(cdp, opts) }`
where `run` returns `{ metrics, detail }` (metrics = flat numbers, lower is
better), then register it in `scenarios/index.mjs`. If it's `ci`, add a
`baseline.json` entry (or run `--update-baseline`).

## Layout

- `lib/cdp.mjs` — the one CDP client + target discovery + typing + CPU-profile wrapper + DOM selectors.
- `lib/stats.mjs` — percentiles, histograms, CPU-profile self-time ranking.
- `lib/baseline.mjs` — load/compare/update the baseline + regression gate.
- `lib/launch.mjs` — attach, or spawn a fully isolated instance.
- `scenarios/` — one module per measurement.
- `run.mjs` — entrypoint. `serve.mjs` — standalone isolated launcher.

## Not migrated (kept as dev utilities)

`eval.mjs`, `reload.mjs`, `reload-renderer.mjs`, `probe-renderer.mjs`,
`probe-thread.mjs`, `click-session.mjs`, `diag-*.mjs` are interactive dev
helpers, not benchmarks. They can adopt `lib/cdp.mjs` in a follow-up.
