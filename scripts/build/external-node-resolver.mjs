// Explicit build-only Node resolution for a dependency tree outside the source.
// The preparation manager must bind both roots before enabling this hook.
import fs from 'node:fs'
import { createRequire, isBuiltin, registerHooks } from 'node:module'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

function root(name) {
  const selected = process.env[name]
  if (!selected || !path.isAbsolute(selected)) throw new Error(`${name} must name an absolute directory`)
  const resolved = fs.realpathSync(selected)
  if (!fs.statSync(resolved).isDirectory()) throw new Error(`${name} must name a directory`)
  return resolved
}

function within(parent, child) {
  const relative = path.relative(parent, child)
  return relative === '' || (relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative))
}

const source = root('HERMES_NODE_SOURCE_ROOT')
const dependencies = root('HERMES_NODE_DEPENDENCY_ROOT')
if (within(source, dependencies) || within(dependencies, source)) {
  throw new Error('External Node dependencies must not overlap source')
}

function admit(result) {
  if (!result.url.startsWith('file:')) return result
  const target = fs.realpathSync(fileURLToPath(result.url))
  if (within(dependencies, target)) return result
  // npm workspace links may point back to source, but must not select an
  // unrelated checkout-local dependency installation.
  const relative = path.relative(source, target)
  if (within(source, target) && !relative.split(path.sep).includes('node_modules')) return result
  throw new Error('External Node resolution escaped the selected roots')
}

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (!context.parentURL?.startsWith('file:') || specifier.startsWith('.') || specifier.startsWith('/') ||
        specifier.includes(':') || isBuiltin(specifier)) return nextResolve(specifier, context)
    const parent = fileURLToPath(context.parentURL)
    if (!within(source, parent)) return nextResolve(specifier, context)
    const anchor = path.join(dependencies, path.relative(source, parent))
    if (context.conditions.includes('require')) {
      return admit({ url: pathToFileURL(createRequire(anchor).resolve(specifier)).href, shortCircuit: true })
    }
    return admit(nextResolve(specifier, { ...context, parentURL: pathToFileURL(anchor).href }))
  }
})
