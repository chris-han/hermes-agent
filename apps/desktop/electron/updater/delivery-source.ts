import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import path from 'node:path'

import { load } from 'js-yaml'

import { channelPublicBase } from './channel-protocol'

export type DeliverySourceId = 'channel1' | 'channel2' | 'channel3'
export interface DeliverySourceStatus {
  selected: DeliverySourceId
  checkStatus?: 'checking' | 'completed' | 'failed'
  options: { id: DeliverySourceId; publicBase: string }[]
}
const ids: DeliverySourceId[] = ['channel1', 'channel2', 'channel3']
const cloudflare = 'https://update2.semantier.com'

function sourceId(value: unknown): DeliverySourceId {
  if (!ids.includes(value as DeliverySourceId)) {throw new Error('Invalid update delivery source')}

  return value as DeliverySourceId
}

function local(base: string): boolean {
  return ['localhost', '127.0.0.1', '[::1]'].includes(new URL(base).hostname)
}

/** Install-wide preference: changing a chat profile must not change the app update owner. */
export class DeliverySources {
  constructor(private readonly file: string, private readonly config: string, private readonly stampedBase: string) {}

  private selected(): DeliverySourceId {
    if (existsSync(this.file)) {
      return sourceId(JSON.parse(readFileSync(this.file, 'utf8')).source)
    }

    return this.stampedBase === cloudflare ? 'channel2' : this.stampedBase && local(this.stampedBase) ? 'channel3' : 'channel1'
  }

  private endpoints(): Record<DeliverySourceId, string> {
    const config: unknown = existsSync(this.config) ? load(readFileSync(this.config, 'utf8')) : null

    const updates = config && typeof config === 'object' && 'updates' in config && config.updates && typeof config.updates === 'object'
      ? config.updates as Record<string, unknown> : {}

    const configured = typeof updates.desktop_feed_base_url === 'string' ? updates.desktop_feed_base_url.trim() : ''
    const dev = typeof updates.desktop_local_feed_base_url === 'string' ? updates.desktop_local_feed_base_url.trim() : ''
    const primary = configured || this.stampedBase

    return {
      channel1: primary && primary !== cloudflare && !local(primary) ? primary : '',
      channel2: cloudflare,
      channel3: dev || (this.stampedBase && local(this.stampedBase) ? this.stampedBase : '')
    }
  }

  status(): DeliverySourceStatus {
    const endpoints = this.endpoints()

    return { selected: this.selected(), options: ids.map(id => ({ id, publicBase: endpoints[id] })) }
  }

  base(): string {
    const selected = this.selected()
    const base = this.endpoints()[selected]

    if (!base) {throw new Error('Update source endpoint is not configured: ' + selected)}
    channelPublicBase(base)

    if (selected === 'channel3' ? !local(base) : new URL(base).protocol !== 'https:') {
      throw new Error('Update source requires ' + (selected === 'channel3' ? 'a loopback endpoint' : 'HTTPS'))
    }

    return base
  }

  select(value: unknown): void {
    const source = sourceId(value)
    mkdirSync(path.dirname(this.file), { recursive: true })
    const pending = this.file + '.tmp'
    writeFileSync(pending, JSON.stringify({ source }) + '\n', { mode: 0o600 })
    renameSync(pending, this.file)
  }
}
