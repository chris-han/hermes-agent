import babel from '@rolldown/plugin-babel'
import react, { reactCompilerPreset } from '@vitejs/plugin-react'
import { defineConfig, type Plugin } from 'vite'

/** React Compiler preset scoped to modules that can actually contain
 *  components/hooks (JSX syntax or a react-ish import). The preset's default
 *  code filter matches any PascalCase/use* declaration — effectively every TS
 *  module — which made the babel pass parse all ~1.5k source files when only
 *  ~750 are React-bearing. */
function compilerPreset() {
  const preset = reactCompilerPreset()

  // Babel's independent resolver anchors plugin names to the source file,
  // so external dependency builds must pass the resolved compiler explicitly.
  if (process.env.HERMES_DESKTOP_BUILD_DIR) {
    preset.preset = () => ({ plugins: [[resolveAppPackage('babel-plugin-react-compiler'), {}]] })
  }

  preset.rolldown.filter.code = /\/>|<\/|from\s*['"][^'"]*react/

  return preset
}

/** The production renderer build memoizes this pass on disk (scripts/compiler-cache.mjs); the
 *  lockfile and this config (the preset and its filter) pin the toolchain it keys on. The
 *  cache lives under node_modules: ignored, and outside every freshness hash.
 *  External-output builds skip it to avoid writing into the source checkout. */
async function cachedCompilerPass(command: string) {
  return withCompilerCache(await babel({ presets: [compilerPreset()] }), {
    command,
    enabled: !process.env.HERMES_DESKTOP_BUILD_DIR,
    cacheRoot: path.join(__dirname, 'node_modules/.cache/hermes-react-compiler'),
    base: __dirname,
    toolchain: [path.resolve(__dirname, '../../package-lock.json'), fileURLToPath(import.meta.url)]
  })
}

import fs from 'fs'
import { createRequire } from 'module'
import path from 'path'
import { fileURLToPath } from 'url'

import tailwindcss from '@tailwindcss/vite'

import { withCompilerCache } from './scripts/compiler-cache.mjs'

// The runner loads this as ESM without the default bundler's CJS globals.
const __dirname: string = path.dirname(fileURLToPath(import.meta.url))

// `hgui` symlinks a worktree's node_modules to the main checkout. Vite realpaths
// those before enforcing server.fs.allow, so codicon/font assets resolve outside
// the worktree root and 404. Whitelist the real node_modules locations.
const real = (p: string): string | null => {
  try {
    return fs.realpathSync(p)
  } catch {
    return null
  }
}

const fsAllow = [
  ...new Set(
    [
      path.resolve(__dirname, '../..'),
      real(path.resolve(__dirname, 'node_modules')),
      real(path.resolve(__dirname, '../../node_modules'))
    ].filter((p): p is string => p !== null)
  )
]

// React refuses to run when `react` and `react-dom` come from two different
// installed copies ("Minified React error #527" — a blank window, since it
// throws before the first paint). Both packages are pinned to one version in
// this workspace's package.json, but npm hoists whatever *it* considers
// compatible to the monorepo root: a root dependency whose react peer is a
// loose range (e.g. `^18 || ^19`) pulls the newest react up there, while
// react-dom stays at the pinned one. `^19.2.7` accepts `19.2.8`, so npm never
// warns. Resolving from this workspace instead of a hardcoded root path yields
// the versions declared here — npm nests a copy under the workspace exactly
// when the hoisted one differs, so the pair can only ever match.
const requireFromApp = createRequire(path.join(__dirname, 'vite.config.ts'))

// The external tree is selected only by the controlled build job. Node's
// resolution hooks do not cover Vite's resolver or Tailwind's CSS resolver.
const sourceRoot = path.resolve(__dirname, '../..')
const selectedNodeRoot = process.env.HERMES_DESKTOP_BUILD_DIR && process.env.HERMES_DESKTOP_NODE_ROOT

const externalNodeRoot = (() => {
  if (!selectedNodeRoot) {return null}

  if (!path.isAbsolute(selectedNodeRoot)) {throw new Error('External Desktop dependency root must be absolute')}
  const deps = fs.realpathSync(selectedNodeRoot)
  const source = fs.realpathSync(sourceRoot)

  const contains = (parent: string, child: string) => {
    const relative = path.relative(parent, child)

    return !relative || (!relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative))
  }

  if (contains(source, deps) || contains(deps, source)) {throw new Error('External Desktop dependencies must not overlap source')}

  return deps
})()

function resolveAppPackage(name: string): string {
  const resolved = requireFromApp.resolve(name)

  if (externalNodeRoot) {
    const relative = path.relative(externalNodeRoot, fs.realpathSync(resolved))

    if (!relative || relative.startsWith(`..${path.sep}`) || relative === '..' || path.isAbsolute(relative)) {
      throw new Error(`Desktop package is outside selected dependency root: ${name}`)
    }
  }

  return resolved
}

const reactDir = path.dirname(resolveAppPackage('react/package.json'))
const reactDomDir = path.dirname(resolveAppPackage('react-dom/package.json'))

const externalCssAliases: Record<string, string> = externalNodeRoot ? {
  'react-router': path.dirname(resolveAppPackage('react-router/package.json')),
  '@tanstack/react-query': path.dirname(resolveAppPackage('@tanstack/react-query/package.json')),
  tailwindcss: path.dirname(resolveAppPackage('tailwindcss/package.json')),
  'tw-shimmer': resolveAppPackage('tw-shimmer'),
  katex: path.dirname(resolveAppPackage('katex/package.json')),
  '@tailwindcss/typography': resolveAppPackage('@tailwindcss/typography'),
  '@vscode/codicons': path.dirname(resolveAppPackage('@vscode/codicons/package.json')),
  '@fontsource/manrope': path.dirname(resolveAppPackage('@fontsource/manrope/package.json')),
  '@fontsource/noto-sans-sc': path.dirname(resolveAppPackage('@fontsource/noto-sans-sc/package.json'))
} : {}

function externalNodeResolver(): Plugin | null {
  if (!externalNodeRoot) {return null}
  const deps = externalNodeRoot

  return {
    name: 'hermes:external-node-dependencies',
    enforce: 'pre',
    async resolveId(id, importer, options) {
      if (!importer || id.startsWith('\0') || id.startsWith('.') || id.startsWith('/') || id.includes(':') ||
          id === 'vite/modulepreload-polyfill' || id.startsWith('@/') || id.startsWith('@hermes/')) {return null}

      const relative = path.relative(sourceRoot, importer.split('?')[0])

      if (!relative || relative.startsWith(`..${path.sep}`) || relative === '..' || path.isAbsolute(relative)) {return null}

      let anchor = path.join(deps, path.dirname(relative))

      while (!fs.existsSync(anchor) && anchor !== deps) {anchor = path.dirname(anchor)}

      const resolved = await this.resolve(id, path.join(anchor, 'package.json'), { ...options, skipSelf: true })

      if (!resolved) {throw new Error(`Desktop import is unavailable in selected dependency root: ${id}`)}

      if (!resolved.external && path.isAbsolute(resolved.id.split('?')[0])) {
        const target = fs.realpathSync(resolved.id.split('?')[0])
        const within = path.relative(deps, target)

        if (!within || within.startsWith(`..${path.sep}`) || within === '..' || path.isAbsolute(within)) {
          throw new Error(`Desktop import is outside selected dependency root: ${id}`)
        }
      }

      return resolved
    },
    transform(code, id) {
      const file = path.normalize(id.split('?')[0])
      const styles = path.normalize(path.join(__dirname, 'src/styles.css'))

      if ((process.platform === 'win32' ? file.toLowerCase() : file) !==
          (process.platform === 'win32' ? styles.toLowerCase() : styles)) {return null}

      const old = '../../../node_modules/@nous-research/ui/dist/fonts/Collapse-Bold.woff2'

      if (!code.includes(old)) {return null}
      const font = resolveAppPackage('@nous-research/ui/fonts/Collapse-Bold.woff2')
      const relative = path.relative(path.dirname(file), font).split(path.sep).join('/')

      return code.replace(old, relative.startsWith('.') ? relative : `./${relative}`)
    }
  }
}

// The dev-only render/state churn counters (src/debug) must be imported
// STATICALLY above react-dom — react-dom captures the devtools hook at module
// init, so a dynamic import lands too late and observes zero commits. A static
// side-effect import can't be tree-shaken, so instead the whole graph is
// aliased out of any non-dev build. `command === 'serve'` covers `vite dev`;
// the perf harness opts a production build back in with VITE_PERF_PROBE=1.
const debugEntry = (command: string, env: Record<string, string>) =>
  command === 'serve' || env.VITE_PERF_PROBE === '1'
    ? path.resolve(__dirname, './src/debug/dev-only.ts')
    : path.resolve(__dirname, './src/debug/dev-only.noop.ts')

// The emoji picker (frimousse) fetches `<emojibaseUrl>/<locale>/data.json` at
// runtime. Its default is a CDN; Electron must work offline, so serve the
// bundled emojibase-data package at a stable local path instead — middleware
// in dev, emitted assets in the build. Only the files a locale actually needs.
const emojibaseDir =
  process.env.HERMES_DESKTOP_BUILD_DIR
    ? real(path.dirname(resolveAppPackage('emojibase-data/package.json')))
    : real(path.resolve(__dirname, 'node_modules/emojibase-data')) ??
      real(path.resolve(__dirname, '../../node_modules/emojibase-data'))

const EMOJIBASE_PATH = /^[a-z-]+\/(data|messages|shortcodes\/emojibase)\.json$/

const emojibaseAssets = () => ({
  name: 'hermes:emojibase-assets',
  configureServer(server: {
    middlewares: { use: (route: string, handler: (req: any, res: any, next: () => void) => void) => void }
  }) {
    server.middlewares.use('/emojibase', (req, res, next) => {
      const rel = (req.url ?? '').split('?')[0].replace(/^\/+/, '')

      if (!emojibaseDir || !EMOJIBASE_PATH.test(rel)) {
        return next()
      }

      fs.readFile(path.join(emojibaseDir, rel), (err: unknown, buf: Buffer) => {
        if (err) {
          return next()
        }

        res.setHeader('Content-Type', 'application/json')
        res.setHeader('Cache-Control', 'public, max-age=31536000, immutable')
        res.end(buf)
      })
    })
  },
  generateBundle(this: { emitFile: (asset: { type: 'asset'; fileName: string; source: Uint8Array }) => void }) {
    if (!emojibaseDir) {
      return
    }

    for (const rel of ['en/data.json', 'en/messages.json', 'en/shortcodes/emojibase.json']) {
      this.emitFile({
        type: 'asset',
        fileName: `emojibase/${rel}`,
        source: fs.readFileSync(path.join(emojibaseDir, rel))
      })
    }
  }
})

export default defineConfig(({ command }) => ({
  base: './',
  plugins: [externalNodeResolver(), react(), cachedCompilerPass(command), tailwindcss(), emojibaseAssets()],
  css: {
    // Pin an explicit (empty) PostCSS config. Tailwind is handled entirely by
    // `@tailwindcss/vite`, so the renderer needs no PostCSS plugins — and
    // without this, Vite's `postcss-load-config` walks UP the filesystem
    // looking for a stray `postcss.config.*` / `tailwind.config.*`. The desktop
    // build runs from inside the user's home tree (e.g.
    // `C:\Users\<name>\AppData\Local\hermes\hermes-agent\apps\desktop`), so an
    // unrelated Tailwind v3 config higher up the tree gets picked up and
    // reprocesses our v4 stylesheet, failing the build with
    // "`@layer base` is used but no matching `@tailwind base` directive is
    // present." Pinning the config makes the build hermetic.
    postcss: { plugins: [] }
  },
  build: {
    // Validate the packaged generation with metadata checks at launch, without
    // reading every lazy vendor chunk (and triggering on-access AV scans).
    manifest: 'renderer-manifest.json',
    // The renderer intentionally ships FEW chunks (not one, not thousands):
    //   · `codeSplitting: false` (the old setup) inlines every `lazy()` /
    //     dynamic import into the entry, so heavyweight lazy-only deps
    //     (mermaid, shiki grammars, katex) are parsed + evaluated on every
    //     cold start even though nothing rendered them. By the time the
    //     bundle hit ~28 MB that eval was ~1s of launch on an M-series.
    //   · Default splitting emits a chunk per shiki grammar/theme — thousands
    //     of files, which electron-builder OOMs scanning (#38888).
    // `advancedChunks` is the middle ground: heavyweight libraries merge into
    // a handful of named vendor chunks loaded on first use, app-level dynamic
    // imports stay lazy, and the file count stays in the tens.
    chunkSizeWarningLimit: 25000,
    rolldownOptions: {
      output: {
        advancedChunks: {
          groups: [
            // Shared foundations FIRST (first match wins): an unmatched
            // module shared by the entry and a heavy chunk gets merged INTO
            // the heavy chunk, and the entry then statically imports 19 MB of
            // shiki just to reach react/hast utils — putting the heavy chunk
            // right back on the boot path.
            //
            // @tanstack/react-query is here for the same reason react-router
            // is: it carries MODULE-LEVEL context (QueryClientContext) that
            // the entry's QueryClientProvider and every lazy chunk's useQuery
            // must share. Left to rolldown's merge heuristics, an unmatched
            // shared module can be inlined into a lazy chunk — the packaged
            // app then runs TWO react-query runtimes, the provider's context
            // is invisible to the other copy, and useQuery throws "No
            // QueryClient set, use QueryClientProvider to set one" on the
            // launch path (#95560). Grouping it forces one shared instance.
            {
              name: 'vendor-react',
              test: /node_modules[\\/](react|react-dom|scheduler|react-router|@tanstack[\\/]react-query)[\\/]/
            },
            {
              name: 'vendor-md',
              test: /node_modules[\\/](property-information|hast-util-[^\\/]+|mdast-util-[^\\/]+|micromark[^\\/]*|unist-util-[^\\/]+|vfile[^\\/]*|unified|stringify-entities|space-separated-tokens|comma-separated-tokens|zwitch|html-void-elements|devlop|style-to-js|style-to-object|clsx)[\\/]/
            },
            // Shared utility packages the entry ALSO uses — kept out of the
            // heavy groups for the same boot-path reason.
            {
              name: 'vendor-util',
              test: /node_modules[\\/](lodash-es|es-toolkit|uuid|dayjs|d3-array|d3-color|d3-force|d3-interpolate|d3-time[^\\/]*|dompurify|stylis)[\\/]/
            },
            // One chunk per heavyweight, lazy-only library family.
            // @streamdown/code lives WITH shiki because it statically imports
            // the full shiki bundle.
            {
              name: 'mermaid',
              test: /node_modules[\\/](mermaid|cytoscape|dagre|khroma|elkjs|d3|d3-[^\\/]+|@mermaid-js)[\\/]/
            },
            {
              name: 'shiki',
              test: /node_modules[\\/](shiki|@shikijs|react-shiki|@streamdown[\\/]code|oniguruma-to-es|oniguruma-parser|regex(-[^\\/]+)?)[\\/]/
            },
            { name: 'katex', test: /node_modules[\\/]katex[\\/]/ }
          ]
        }
      }
    }
  },
  // driver.js only enters the graph through the tour's DYNAMIC import chain
  // (lib/tour/run-tour.ts), so the dep scanner never sees it at startup. Left
  // alone, first use registers it as a missing dep at runtime — which (a)
  // esbuild-prebundles the `?raw` IIFE import as a JS module, breaking the
  // raw-text transform ("does not provide an export named 'default'"), and
  // (b) triggers Vite's "new dependencies optimized" full page reload mid-
  // session. It's pure ESM with no CJS deps, so serving it unoptimized is
  // free. Query and bare forms all listed — exclusion matches exact ids.
  optimizeDeps: {
    exclude: [
      'driver.js',
      'driver.js/dist/driver.js.iife.js',
      'driver.js/dist/driver.js.iife.js?raw',
      'driver.js/dist/driver.css?raw'
    ]
  },
  resolve: {
    alias: {
      ...externalCssAliases,
      '@/debug/dev-only': debugEntry(command, process.env as Record<string, string>),
      '@': path.resolve(__dirname, './src'),
      '@hermes/plugin-sdk': path.resolve(__dirname, './src/sdk/index.ts'),
      '@hermes/shared/billing': path.resolve(__dirname, '../shared/src/billing-types.ts'),
      '@hermes/shared/color': path.resolve(__dirname, '../shared/src/color.ts'),
      '@hermes/shared': path.resolve(__dirname, '../shared/src'),
      // The tour tool's preview surface injects driver.js's prebuilt IIFE into
      // the pane's guest page as raw source; the package's exports map doesn't
      // expose that dist file (nor ./package.json), so resolve the main entry
      // (dist/driver.js.cjs) and point at its sibling. Both keys on purpose:
      // alias matching is exact, and the id reaches it with the `?raw` query
      // still attached in dev but stripped in some build paths.
      'driver.js/dist/driver.js.iife.js?raw': `${path.join(
        path.dirname(resolveAppPackage('driver.js')),
        'driver.js.iife.js'
      )}?raw`,
      'driver.js/dist/driver.js.iife.js': path.join(
        path.dirname(resolveAppPackage('driver.js')),
        'driver.js.iife.js'
      ),
      react: reactDir,
      'react-dom': reactDomDir,
      'react/jsx-dev-runtime': path.join(reactDir, 'jsx-dev-runtime.js'),
      'react/jsx-runtime': path.join(reactDir, 'jsx-runtime.js')
    },
    dedupe: externalNodeRoot ? [] : ['react', 'react-dom', 'react-router', '@tanstack/react-query']
  },
  server: {
    host: '127.0.0.1',
    port: 5174,
    strictPort: true,
    fs: {
      allow: fsAllow
    }
  },
  preview: {
    host: '127.0.0.1',
    port: 4174
  }
}))
