# Source Electron fixture runner

From the Hermes repository root, using existing installed dependencies:

```bash
node apps/desktop/scripts/run-fixture-e2e.mjs --python /absolute/path/to/venv/bin/python
```

The default is a real Electron bootstrap smoke. It checks canonical `default`,
qualifies failure diagnostics against actual accessible UI, and stops its gateway.
No model invocation is needed. The caught `diagnostic.counterexample` stage is an
intentional diagnostic test, not a failed application journey.

Run existing synthetic journeys through the same launcher:

```bash
node apps/desktop/scripts/run-fixture-e2e.mjs --python /absolute/path/to/venv/bin/python \
  e2e/cloud-sign-in.spec.ts e2e/browser-context-affinity.spec.ts
```

Or, from `apps/desktop`, use `npm run e2e:fixture -- --python /absolute/python`.
`--help` lists options. `--output /new-or-empty/directory` keeps artifacts at a
chosen location; otherwise each invocation gets a unique `test-results/fixture-*`
directory. `--timeout-ms` bounds the whole Playwright invocation (default 600000).

The runner derives all source paths from its own location, validates the source
before launch, bundles only development main/preload with the existing bundler,
starts fresh Vite middleware on an OS-assigned loopback port with HMR disabled,
and runs one Playwright worker with zero retries. Linux uses fixture-owned Xvfb.
It does not run `npm run build`, install dependencies or create Windows packages.

A source `.env` is refused because gateway startup loads it independently of
`HERMES_HOME`. Use a clean worktree, not a copied production home. The selected
Python is mapped to both `HERMES_DESKTOP_PYTHON` (the actual Desktop override) and
`HERMES_PYTHON` (fixture CLI helpers). Inherited credentials, runtime/session flags,
fake Desktop boot flags and `ELECTRON_RUN_AS_NODE` are stripped.

`FIXTURE_STAGE` lines identify completed/failing boundaries. `FIXTURE_RESULT`
prints status, executed/skipped counts and the artifact directory. A run that
only skips tests fails qualification. The directory contains:

- `runner.json`: selected source/Python, interpreter version, renderer URL,
  stage durations, the short OS temporary root, and retained cleanup failures.
  Temporary homes are separate from artifact paths to respect Chromium socket limits.
- `results.json` and `playwright/`: actual Playwright results, traces/screenshots.
- Per-test `fixture-stages.json`: bounded error, URL without credentials/query,
  and actual accessibility snapshot captured before teardown. Known bearer and
  password-field values are redacted. These fixtures are for synthetic data;
  arbitrary application content is not a general secret-scrubbing guarantee.

For another synthetic source spec, reuse `createSandbox`, `buildAppEnv` and
`DesktopFixtureSession`:

```ts
const sandbox = createSandbox('feature')
const env = buildAppEnv(sandbox, {
  HERMES_PYTHON: process.env.HERMES_PYTHON!,
  HERMES_DESKTOP_DEV_SERVER: process.env.HERMES_DESKTOP_DEV_SERVER!
})
const fixture = new DesktopFixtureSession(sandbox, env, testInfo)
try {
  const { page } = await fixture.launch()
  await fixture.stage('feature.ready', async () => {
    // Assert real UI/API state; avoid importing renderer stores through HMR.
  })
  // fixture.button('Begin') accepts both Begin and the decorated [ Begin ].
  // fixture.restart() stops/relaunches Electron and gateway against the same home.
} finally {
  await fixture.close()
}
```

Keep feature-owned service/model fixtures in their specs. Cloud uses local mock
Cloud with native DPAPI on Windows and a fixture-only Fernet adapter on Linux; browser affinity uses a scripted model.
The Windows Cloud journey verifies native encrypted persistence and restart. Production OAuth and autonomous LLM routing remain outside this synthetic lane.

Cleanup validates ownership/source/home before using the supported native
`gateway stop` command. A failed stop retains the home and fails the run; process
identity is never guessed from command-line substrings. The runner also recovers
marked homes under its own temporary root after worker interruption. Shared
browsers and development servers are not stopped. Existing legacy specs should
use the shared session or source fixture teardown to get these guarantees.


## Optional authorized real-model qualification

`e2e/autonomous-browser.spec.ts` uses actual Electron chats and the existing
canonical Agent/gateway. It is skipped unless `SEMANTIER_AUTONOMOUS_BROWSER_RUN=1`.
It never mocks model choices. Use an isolated source worktree without `.env`.
Set `SEMANTIER_EVAL_CREDENTIALS_ENV` to an existing authorized Windows-local
Runtime env file containing DASHSCOPE_API_KEY and DASHSCOPE_BASE_URL; the observer
reads only those entries and never passes the real key into Electron.
Set `SEMANTIER_EVAL_OUTPUT` to a new evidence directory, and bind
`SEMANTIER_EVAL_SOURCE_SHA`, `SEMANTIER_EVAL_RUNTIME_SHA`, `SEMANTIER_EVAL_RUN_ID`
and `SEMANTIER_EVAL_PREFLIGHT_PROMPT_HASHES` before the frozen run. The model is
strictly qwen3.5-plus. Keep native tool_search/deferred defaults and consent on.

First run with `SEMANTIER_EVAL_PREFLIGHT_ONLY=1` (and optionally
`SEMANTIER_EVAL_LOCALE=en` or `zh`) to exercise supported consent, actual composer
submission, scoped tool catalog, and restart with **zero upstream inference**.
Remove that flag only after evaluator self-tests and readiness pass. Run using
only the existing CLI flags:

```powershell
node apps/desktop/scripts/run-fixture-e2e.mjs --python <native-python.exe> --output <new-run-directory> --timeout-ms 2100000 e2e/autonomous-browser.spec.ts
```

The six cases cap each at12upstream requests,4096output tokens/request and240s.
The observer forwards real provider bytes, buffering streaming responses for
redacted traces; observed streaming latency therefore differs from normal UI.
The test emits measurement rows: a green runner exit does **not** establish
qualification. Review actual tool results, final responses, webview/server state,
ownership and all six case outcomes against the frozen contract. Never score
zero model calls as autonomous success. Discovery/wrappers are retained and the
first actual browser operation is scored separately. No production website or
real-profile cookie copying is permitted.

Evaluator self-tests use the existing Electron Vitest project:
`vitest run scripts/autonomous-evaluator.test.mjs`.
Observer parser/scope tests run with native Python:
`python apps/desktop/e2e/autonomous-provider.test.py`.
