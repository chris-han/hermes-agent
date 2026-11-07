// msix-shared.mjs — shared MSIX packaging utilities
// Stub for development builds

import fs from 'node:fs'
import path from 'node:path'

/**
 * Returns channel build request info
 * @returns {{ semantierScope?: string } | null}
 */
export function channelBuildRequest() {
  const scope = process.env.SEMANTIER_DESKTOP_SCOPE || ''
  if (scope) {
    return { semantierScope: scope }
  }
  return null
}

/**
 * Parse version string to MSIX format
 * @param {string} version
 * @returns {string}
 */
export function parseVersion(version) {
  const match = /^v?(\d+)\.(\d+)\.(\d+)(?:-(.+))?$/.exec(version)
  if (!match) return '0.0.0.0'
  const [, major, minor, patch] = match
  return `${major}.${minor}.${patch}.0`
}

/**
 * Get app identity for MSIX packaging
 * @param {string} desktopPath
 * @param {string} tag
 * @returns {{ version: string, fileVersion: string }}
 */
export function appIdentity(desktopPath, tag) {
  const pkgPath = path.join(desktopPath, 'package.json')
  const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'))
  const baseVersion = tag || pkg.version || '0.0.0'
  
  return {
    version: parseVersion(baseVersion),
    fileVersion: baseVersion
  }
}
