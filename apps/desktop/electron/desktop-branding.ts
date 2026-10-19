import fs from 'node:fs'
import path from 'node:path'

import { nativeImage } from 'electron'

import { DEFAULT_DESKTOP_BRANDING, type DesktopBranding } from './desktop-branding-types'

const MAX_BYTES = 2 * 1024 * 1024

export function validateDesktopBranding(input: unknown): DesktopBranding {
  if (!input || typeof input !== 'object') {
    throw new Error('Invalid branding')
  }

  const value = input as DesktopBranding

  if (
    value.version !== 1 ||
    [value.displayName, value.displayNameZh === undefined ? '' : value.displayNameZh].some(
      name => typeof name !== 'string' || name.length > 120 ||
        Array.from(name).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127)
    )
  ) {
    throw new Error('Invalid display name')
  }

  let logo: string | null = null

  if (value.logo !== null) {
    if (typeof value.logo !== 'string' || value.logo.length > Math.ceil(MAX_BYTES / 3) * 4 + 32) {
      throw new Error('Invalid logo size')
    }

    const match = /^data:image\/(png|jpeg|webp);base64,([A-Za-z0-9+/]+={0,2})$/.exec(value.logo)

    if (!match) {
      throw new Error('Invalid logo format')
    }

    const bytes = Buffer.from(match[2], 'base64')

    const formats: Record<string, boolean> = {
      png: bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])),
      jpeg: bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255,
      webp: bytes.toString('ascii', 0, 4) === 'RIFF' && bytes.toString('ascii', 8, 12) === 'WEBP'
    }

    if (bytes.length > MAX_BYTES || !formats[match[1]]) {
      throw new Error('Invalid logo bytes')
    }

    const image = nativeImage.createFromBuffer(bytes)
    const { width, height } = image.getSize()

    if (image.isEmpty() || width < 32 || height < 32 || width > 4096 || height > 4096) {
      throw new Error('Invalid logo dimensions')
    }

    const canonical = image.toPNG()

    if (canonical.length > MAX_BYTES) {
      throw new Error('Logo output too large')
    }

    logo = `data:image/png;base64,${canonical.toString('base64')}`
  }

  return { version: 1, displayName: value.displayName, displayNameZh: value.displayNameZh ?? '', logo }
}

export function createDesktopBrandingStore(file: string, changed: (value: DesktopBranding) => void = () => {}) {
  let current: DesktopBranding = { ...DEFAULT_DESKTOP_BRANDING }
  let loadError: unknown

  try {
    current = validateDesktopBranding(JSON.parse(fs.readFileSync(file, 'utf8')))
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
      loadError = error
    }
  }

  let pending = Promise.resolve()

  return {
    initial: () => ({ ...current }),
    get: () => {
      if (loadError) {
        throw new Error('Could not load branding')
      }

      return { ...current }
    },
    set(input: unknown) {
      const operation = pending.then(() => {
        const next = validateDesktopBranding(input)
        fs.mkdirSync(path.dirname(file), { recursive: true })
        fs.writeFileSync(file + '.tmp', JSON.stringify(next), { encoding: 'utf8', mode: 0o600 })
        fs.renameSync(file + '.tmp', file)
        current = next
        loadError = undefined
        changed({ ...next })

        return { ...next }
      })

      pending = operation.then(
        () => undefined,
        () => undefined
      )

      return operation
    }
  }
}
