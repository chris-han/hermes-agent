// Build-time guard: refuse to hand a half-built renderer to electron-builder.
//
// `npm run pack` / `npm run dist*` are `npm run build && npm run builder`.
// If the `build` step (tsc -b && vite build) fails but packaging proceeds
// anyway — a stale checkout that fails typecheck, an interrupted vite build,
// or npm not short-circuiting `&&` in some shells — electron-builder happily
// packages an app with an empty or missing `dist/`. The result launches but
// blank-pages with `ERR_FILE_NOT_FOUND` for dist/index.html, with no clue why.
//
// The desktop compiler (scripts/build/desktop.mjs) runs it on the scratch
// product before publishing dist/, so every `npm run build` path inherits it
// once. It is deliberately not also a `postbuild` hook: that re-ran the same
// check on the bytes just verified, doubling its cost in every update.
// It fails loud and early instead of shipping a broken bundle.
// See issues #39484 (renderer blank page) and #41327 / #39472 (dashboard 404).

import { existsSync, readFileSync, statSync, readdirSync } from "fs"
import { spawnSync } from "child_process"
import { join, resolve } from "path"
import { isMain } from "./utils.mjs"

const ROUTER_CONTEXT_ERROR = "may be used only in the context of a"

// @tanstack/react-query carries module-level React context (QueryClientContext).
// The entry's QueryClientProvider and every lazy chunk's useQuery must share ONE
// runtime instance; if a build ever emits a second copy, the provider's context
// is invisible to the other copy and useQuery throws "No QueryClient set" — the
// packaged app error-boundaries on launch (#95560). Same single-instance
// invariant as the react-router check above, same failure class.
const QUERY_CLIENT_CONTEXT_ERROR = "No QueryClient set, use QueryClientProvider to set one"

// Pure check — returns { ok: true } or { ok: false, error: "...", kind?: "bundle" | "harness" }.
// `kind: "harness"` marks a failure of the CHECK itself (node would not start, the checker
// produced no verdict) rather than a defective bundle, so main() can point at the right cause
// instead of telling the reader to re-run a build that was never the problem.
// Kept side-effect-free so it can be unit tested without spawning a process.
export function checkDistBuilt(distDir) {
  if (!existsSync(distDir) || !statSync(distDir).isDirectory()) {
    return { ok: false, error: `no dist directory at ${distDir}` }
  }

  const indexHtml = join(distDir, "index.html")
  if (!existsSync(indexHtml) || !statSync(indexHtml).isFile()) {
    return { ok: false, error: `dist/index.html is missing at ${indexHtml}` }
  }
  if (statSync(indexHtml).size === 0) {
    return { ok: false, error: `dist/index.html is empty at ${indexHtml}` }
  }

  // index.html alone isn't enough — vite emits hashed JS into dist/assets.
  // An index.html with no script bundle still blank-pages.
  const assetsDir = join(distDir, "assets")
  const hasAssets =
    existsSync(assetsDir) &&
    statSync(assetsDir).isDirectory() &&
    readdirSync(assetsDir).some(name => name.endsWith(".js"))
  if (!hasAssets) {
    return { ok: false, error: `dist/assets has no built JS bundle (expected vite output under ${assetsDir})` }
  }

  const routerContextAssets = readdirSync(assetsDir)
    .filter(name => name.endsWith(".js"))
    .filter(name => readFileSync(join(assetsDir, name), "utf8").includes(ROUTER_CONTEXT_ERROR))

  if (routerContextAssets.length > 1) {
    return {
      ok: false,
      error: `react-router context invariant found in multiple JS assets: ${routerContextAssets.join(", ")}`
    }
  }

  const queryClientContextAssets = readdirSync(assetsDir)
    .filter(name => name.endsWith(".js"))
    .filter(name => readFileSync(join(assetsDir, name), "utf8").includes(QUERY_CLIENT_CONTEXT_ERROR))

  if (queryClientContextAssets.length > 1) {
    return {
      ok: false,
      error:
        `@tanstack/react-query context invariant found in multiple JS assets: ` +
        `${queryClientContextAssets.join(", ")} — duplicate react-query runtimes make the ` +
        `QueryClientProvider's context invisible to useQuery in other chunks (` +
        `"No QueryClient set" on launch, #95560)`
    }
  }

  // Parse-validate every emitted chunk as an ES module. Corrupted-silent-fail
  // bundles (a dropped identifier token mid-file) produce invalid syntax that
  // only explodes at module-evaluation time in Electron's renderer.
  const chunkParse = verifyChunksParse(assetsDir)
  if (!chunkParse.ok) {
    return chunkParse
  }

  return { ok: true }
}

// Renderer chunks are emitted as ESM (`<script type="module">` in index.html).
// A silent bundler failure can emit syntactically invalid chunks that parse fine
// as CJS-ish text but throw on module evaluation in Electron — the app then
// white-screens with `Uncaught SyntaxError` in the renderer console (observed
// 2026-09: the update-produced bundle was missing a 10-byte identifier token,
// `{$:n,}` vs `{categories:n,}`, leaving an invalid destructuring pattern).
// Parse each emitted chunk as an ES module before packaging so a corrupted
// build fails loudly and the update retry rebuilds instead of shipping it.
//
// The parsing itself runs in ONE child process (`check-chunks-parse.mjs`,
// vm.SourceTextModule), not one spawn per chunk: with ~1000 chunks, a single
// stalled process creation on Windows (AV, memory pressure) used to abort the
// whole update — the guard must not be the least reliable step in the build.
function verifyChunksParse(assetsDir) {
  const nodeBin = process.env.NODE || process.execPath || "node"
  const checker = join(import.meta.dirname, "check-chunks-parse.mjs")
  const probe = spawnSync(nodeBin, ["--experimental-vm-modules", checker, assetsDir], {
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
    timeout: 300_000,
  })
  if (probe.error) {
    return {
      ok: false,
      kind: "harness",
      error: `could not run node to syntax-check the renderer chunks: ${probe.error.message}`,
    }
  }

  let verdict = null
  try {
    verdict = JSON.parse(String(probe.stdout || "").trim() || "null")
  } catch {
    verdict = null
  }
  if (!verdict) {
    const detail = String(probe.stderr || "").trim().split("\n").filter(Boolean).slice(-4).join(" / ")
    return {
      ok: false,
      kind: "harness",
      error: `node could not syntax-check the renderer chunks (exit ${probe.status}) — ${detail}`,
    }
  }
  if (verdict.ok) {
    return { ok: true }
  }
  if (verdict.kind === "harness" || !verdict.name) {
    // The check itself did not run (blocked/stalled node, no verdict, or the checker reporting
    // that it could not read a chunk or use the vm module). Never call that a bundle defect.
    const where = verdict.name ? ` (on ${verdict.name})` : ""
    return {
      ok: false,
      kind: "harness",
      error: `the renderer chunk check could not run${where}: ${verdict.detail}`,
    }
  }
  return {
    ok: false,
    kind: "bundle",
    error: `built chunk is not valid ES module syntax: ${verdict.name} — ${verdict.detail}. ` +
      `A renderer chunk failed to parse, so packaging would ship an app that ` +
      `white-screens with "Uncaught SyntaxError" on launch. Re-run the build.`,
  }
}

// A standalone development bundle still loads source-adjacent product identity.
// Probe from the emitted main's location, including that module's own imports.
export function checkDesktopLaunch(distDir) {
  const renderer = checkDistBuilt(distDir)
  if (!renderer.ok) return renderer
  for (const name of ["electron-main.mjs", "electron-preload.js", "preview-guest-preload.js"]) {
    if (!existsSync(join(distDir, name))) return { ok: false, error: `launch asset missing: ${name}` }
  }
  const html = readFileSync(join(distDir, "index.html"), "utf8")
  for (const match of html.matchAll(/(?:src|href)=["']((?:\.\/)?assets\/[^"']+)["']/g)) {
    const asset = match[1].split(/[?#]/)[0]
    if (!existsSync(join(distDir, asset))) return { ok: false, error: `HTML launch asset missing: ${asset}` }
  }
  const main = join(distDir, "electron-main.mjs")
  const code = readFileSync(main, "utf8")
  if (code.includes("typeof __HERMES_PRODUCT_IDENTITY__")) {
    const reference = code.match(/["'](\.{1,2}\/product-identity\.cjs)["']/)?.[1]
    if (!reference) return { ok: false, error: "development identity reference is missing" }
    const probe = spawnSync(process.execPath, ["-e",
      "require('node:module').createRequire(process.argv[1])(process.argv[2])", resolve(main), reference],
      { encoding: "utf8", timeout: 15_000, windowsHide: true })
    if (probe.error || probe.status !== 0) return {
      ok: false, kind: probe.error ? "harness" : "bundle",
      error: `development identity dependency failed from ${main}: ${probe.error?.message || String(probe.stderr).trim().slice(0, 400)}`,
    }
  }
  return { ok: true }
}

function main() {
  const desktopRoot = resolve(import.meta.dirname, "..")
  const at = process.argv.indexOf("--dist")
  const distDir = at >= 0 ? resolve(process.argv[at + 1]) : join(desktopRoot, "dist")
  const result = process.argv.includes("--launch") ? checkDesktopLaunch(distDir) : checkDistBuilt(distDir)

  if (!result.ok) {
    console.error(`\n✗ assert-dist-built: ${result.error}`)
    if (result.kind === "harness") {
      // The check could not run — the bundle is not implicated. Saying "re-run the build" here
      // sent a past update investigation after a bundle defect that did not exist.
      console.error("  This is a failure of the CHECK, not of the bundle: node did not run to")
      console.error("  completion. On Windows that is usually a stalled process creation")
      console.error("  (antivirus/EDR interference, memory pressure) and clears on a retry.")
      console.error("  If it repeats, check the node binary and any AV exclusion for the")
      console.error(`    build tree, then re-run: cd ${desktopRoot} && npm run build\n`)
      process.exit(1)
    }
    if (process.argv.includes("--launch")) {
      console.error("  Final desktop staging is incomplete. Rebuild or restage from the authoritative source with the correct bundle mode.")
      process.exit(1)
    }
    console.error("  The renderer bundle is missing or incomplete, so packaging")
    console.error("  would produce an app that launches to a blank page.")
    console.error("  Re-run the build and check the tsc/vite output above for the")
    console.error("  real failure, then package again:")
    console.error(`    cd ${desktopRoot} && npm run build\n`)
    process.exit(1)
  }

  console.log(process.argv.includes("--launch")
    ? "✓ assert-dist-built: final launch assets and identity dependencies verified"
    : "✓ assert-dist-built: dist/index.html + assets present")
}

if (isMain(import.meta.url)) {
  main()
}

export default { checkDistBuilt, checkDesktopLaunch }
