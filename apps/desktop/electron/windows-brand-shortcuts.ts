import { createHash, randomUUID } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

import { app, nativeImage, shell, type ShortcutDetails } from 'electron'

import { appIconCandidates, resolveAppIcon } from './app-icon'
import { validateDesktopBranding } from './desktop-branding'
import type { DesktopBranding } from './desktop-branding-types'
import { PRODUCT_IDENTITY } from './product-identity'
import { runChannelPowerShell } from './updater/channel-native'
import { validBrandShortcutName } from './windows-brand-shortcuts-types'
import type { BrandShortcutRequest, BrandShortcutResult, ShortcutDestination } from './windows-brand-shortcuts-types'

const DESTINATIONS = ['desktop', 'startMenu'] as const

export function validateShortcutName(value: unknown): string {
  if (!validBrandShortcutName(value)) {
    throw new Error('Invalid shortcut name')
  }
  return value
}

/** Windows ICO supports PNG entries; no executable encoder or dependency is needed. */
export function pngsToIco(images: { size: number; png: Buffer }[]): Buffer {
  const header = Buffer.alloc(6 + 16 * images.length)
  header.writeUInt16LE(1, 2)
  header.writeUInt16LE(images.length, 4)
  let offset = header.length
  images.forEach(({ size, png }, index) => {
    const row = 6 + index * 16
    header[row] = header[row + 1] = size === 256 ? 0 : size
    header.writeUInt16LE(1, row + 4)
    header.writeUInt16LE(32, row + 6)
    header.writeUInt32LE(png.length, row + 8)
    header.writeUInt32LE(offset, row + 12)
    offset += png.length
  })
  return Buffer.concat([header, ...images.map(image => image.png)])
}

function brandingIcon(branding: DesktopBranding): Buffer {
  const fallback = resolveAppIcon(
    appIconCandidates({
      isWindows: true,
      appRoot: app.getAppPath(),
      resourcesPath: process.resourcesPath,
      unpackedPathFor: p => p.replace(/\.asar(?=[\\/])/, '.asar.unpacked')
    })
  )
  const source = branding.logo
    ? nativeImage.createFromDataURL(branding.logo)
    : fallback
      ? nativeImage.createFromPath(fallback)
      : null
  if (!source || source.isEmpty()) {
    throw new Error('Brand icon unavailable')
  }
  const { width, height } = source.getSize()
  return pngsToIco(
    [16, 24, 32, 48, 64, 128, 256].map(size => {
      const ratio = size / Math.max(width, height)
      const fitted = source.resize({
        width: Math.max(1, Math.round(width * ratio)),
        height: Math.max(1, Math.round(height * ratio)),
        quality: 'best'
      })
      const dimensions = fitted.getSize()
      const canvas = Buffer.alloc(size * size * 4)
      const pixels = fitted.toBitmap()
      const left = Math.floor((size - dimensions.width) / 2)
      const top = Math.floor((size - dimensions.height) / 2)
      for (let row = 0; row < dimensions.height; row++) {
        pixels.copy(
          canvas,
          ((top + row) * size + left) * 4,
          row * dimensions.width * 4,
          (row + 1) * dimensions.width * 4
        )
      }
      return { size, png: nativeImage.createFromBitmap(canvas, { width: size, height: size }).toPNG() }
    })
  )
}

/** Registered MSIX activation survives package updates; never link into WindowsApps. */
async function runningShortcutTarget(): Promise<ShortcutDetails> {
  const result = await runChannelPowerShell(
    String.raw`
$ErrorActionPreference='Stop'
$p=[Console]::In.ReadToEnd() | ConvertFrom-Json
$packages=@(Get-AppxPackage -Name $p.identity | Where-Object { $_.Name -ceq $p.identity -and $p.executable.StartsWith($_.InstallLocation.TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase) })
if ($packages.Count -gt 1) { throw 'Ambiguous running package' }
if ($packages.Count -eq 1) {
 $pkg=$packages[0]; $manifest=Get-AppxPackageManifest -Package $pkg.PackageFullName
 $apps=@($manifest.Package.Applications.Application | Where-Object { [IO.Path]::GetFullPath((Join-Path $pkg.InstallLocation $_.Executable)) -ieq $p.executable })
 if ($apps.Count -ne 1) { throw 'Running app not in registered package' }
 @{ appUserModelId=($pkg.PackageFamilyName+'!'+$apps[0].Id); target=(Join-Path $env:SystemRoot 'explorer.exe'); args=('shell:AppsFolder\'+$pkg.PackageFamilyName+'!'+$apps[0].Id) } | ConvertTo-Json -Compress
} else {
 if ($p.packaged -or $p.executable -match '\\WindowsApps\\') { throw 'Running package unavailable' }
 @{ appUserModelId=$p.appId; target=$p.executable; args='' } | ConvertTo-Json -Compress
}
`,
    JSON.stringify({
      identity: PRODUCT_IDENTITY.msixAppIdWithOrg,
      executable: process.execPath,
      packaged: Boolean(process.windowsStore),
      appId: PRODUCT_IDENTITY.appId
    })
  )
  return JSON.parse(result.stdout) as ShortcutDetails
}

function sameLink(a: ShortcutDetails, b: ShortcutDetails): boolean {
  return (
    ['target', 'icon', 'cwd'].every(
      key =>
        String(a[key as keyof ShortcutDetails] || '').toLowerCase() ===
        String(b[key as keyof ShortcutDetails] || '').toLowerCase()
    ) &&
    ['args', 'description', 'appUserModelId', 'iconIndex'].every(
      key => (a[key as keyof ShortcutDetails] ?? '') === (b[key as keyof ShortcutDetails] ?? '')
    )
  )
}

export interface ShortcutServiceOptions {
  directory: string
  roots: Record<ShortcutDestination, string>
  branding: () => DesktopBranding
  target?: () => Promise<ShortcutDetails>
}

export function createBrandShortcutService(options: ShortcutServiceOptions) {
  const registryFile = path.join(options.directory, 'owned-shortcuts.v1.json')
  let pending = Promise.resolve<unknown>(undefined)
  const records = (): Record<string, ShortcutDetails> => {
    try {
      return JSON.parse(fs.readFileSync(registryFile, 'utf8'))
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
        return {}
      }
      throw error
    }
  }
  const linkPath = (name: string, destination: ShortcutDestination) => {
    if (!DESTINATIONS.includes(destination)) {
      throw new Error('Invalid shortcut destination')
    }
    return path.join(options.roots[destination], validateShortcutName(name) + '.lnk')
  }
  const owned = (file: string, all: Record<string, ShortcutDetails>) => {
    if (!all[file]) {
      return false
    }
    try {
      if (fs.lstatSync(file).isSymbolicLink()) {
        return false
      }
      return sameLink(shell.readShortcutLink(file), all[file])
    } catch {
      return false
    } // An unreadable or replaced link is never safe to update.
  }

  return {
    create(request: BrandShortcutRequest): Promise<BrandShortcutResult[]> {
      const task = pending.then(async () => {
        validateShortcutName(request?.name)
        if (
          !Array.isArray(request.destinations) ||
          !request.destinations.length ||
          request.destinations.length > 2 ||
          new Set(request.destinations).size !== request.destinations.length ||
          request.destinations.some(d => !DESTINATIONS.includes(d))
        ) {
          throw new Error('Invalid shortcut destination')
        }
        const branding = options.branding()
        if (JSON.stringify(validateDesktopBranding(request.branding)) !== JSON.stringify(branding)) {
          throw new Error('Branding changed; refresh and retry')
        }
        const target = await (options.target ?? runningShortcutTarget)()
        // A concurrent branding save must not relabel a shortcut after native target resolution.
        if (JSON.stringify(options.branding()) !== JSON.stringify(branding)) {
          throw new Error('Branding changed; refresh and retry')
        }
        const iconBytes = brandingIcon(branding)
        fs.mkdirSync(options.directory, { recursive: true })
        const icon = path.join(options.directory, createHash('sha256').update(iconBytes).digest('hex') + '.ico')
        if (!fs.existsSync(icon)) {
          fs.writeFileSync(icon, iconBytes, { flag: 'wx' })
        }
        const all = records()
        const results: BrandShortcutResult[] = []
        for (const destination of request.destinations) {
          const file = linkPath(request.name, destination)
          let temporary = ''
          let before: Buffer | null = null
          let written = false
          try {
            const exists = fs.existsSync(file)
            if (exists && !owned(file, all)) {
              results.push({ destination, status: 'conflict' })
              continue
            }
            if (exists) {
              before = fs.readFileSync(file)
            }
            fs.mkdirSync(options.roots[destination], { recursive: true })
            temporary = path.join(options.roots[destination], '.' + randomUUID() + '.lnk')
            const details = { ...target, icon, iconIndex: 0, description: 'Semantier Agent custom shortcut' }
            if (!shell.writeShortcutLink(temporary, 'create', details)) {
              throw new Error('Shortcut write failed')
            }
            if (!sameLink(shell.readShortcutLink(temporary), details)) {
              throw new Error('Shortcut verification failed')
            }
            if (exists) {
              if (!owned(file, all)) {
                results.push({ destination, status: 'conflict' })
                continue
              }
              fs.renameSync(temporary, file)
            } else {
              fs.copyFileSync(temporary, file, fs.constants.COPYFILE_EXCL)
            }
            written = true
            const updated = { ...all, [file]: shell.readShortcutLink(file) }
            const registryTemp = registryFile + '.tmp'
            fs.writeFileSync(registryTemp, JSON.stringify(updated), { mode: 0o600 })
            fs.renameSync(registryTemp, registryFile)
            Object.assign(all, updated)
            results.push({ destination, status: exists ? 'updated' : 'created' })
          } catch {
            // No apparent success when recording ownership failed. Restore only our own write.
            if (written) {
              if (before) {
                fs.writeFileSync(file, before)
              } else {
                fs.unlinkSync(file)
              }
            }
            results.push({ destination, status: 'failed' })
          } finally {
            if (temporary && fs.existsSync(temporary)) {
              fs.unlinkSync(temporary)
            }
          }
        }
        return results
      })
      pending = task.catch(() => undefined)
      return task
    },
    reveal(name: string, destination: ShortcutDestination) {
      const file = linkPath(name, destination)
      if (!fs.existsSync(file) || !owned(file, records())) {
        throw new Error('Shortcut is not owned')
      }
      shell.showItemInFolder(file)
    }
  }
}

export function defaultBrandShortcutService(branding: () => DesktopBranding) {
  return createBrandShortcutService({
    directory: path.join(app.getPath('userData'), 'branding-shortcuts'),
    roots: {
      desktop: app.getPath('desktop'),
      startMenu: path.join(app.getPath('appData'), 'Microsoft', 'Windows', 'Start Menu', 'Programs')
    },
    branding
  })
}
