import { createHash } from 'node:crypto'

import {
  type ActiveChannel,
  buildPrefix,
  type ChannelBuild,
  channelKey,
  type ChannelManifest,
  type ChannelPackage,
  channelPublicBase,
  decodeChannelManifest,
  decodeChannelRecord,
  decodeNativePackageGraph,
  type NativePackageGraph,
  type ReceiverKind,
  type RetiredChannel,
  sameChannelIdentity,
  validateChannelName
} from './channel-protocol'

export interface ChannelTarget {
  channel: ActiveChannel
  manifest: ChannelManifest
  package: ChannelPackage
  manifestSha256: string
  artifactUrl: string
  feedUrl: string
  nativeGraph?: NativePackageGraph
}
export interface ChannelRetirement {
  source: ChannelBuild
  target: ChannelTarget
  /** Publisher-pinned at retire() time; in-place shares the destination's identity. */
  receiverKind: ReceiverKind
}
export type ChannelResolution =
  | { kind: 'active'; target: ChannelTarget }
  | { kind: 'empty'; channel: ActiveChannel }
  | { kind: 'retirement'; retirement: ChannelRetirement }

export interface ChannelResolverDeps {
  build: ChannelBuild
  platform: ChannelPackage['platform']
  arch: ChannelPackage['arch']
  /** Locally trusted signing identity, not a value supplied by the feed. */
  signer: string
  fetch?: typeof fetch
  /** Set only after local host/trust and association or migration eligibility checks. */
  nativePackageUri?: boolean
}

function assertVersionFloor(version: string, floor: string): void {
  // Retirement floors are release versions, never preview sequence quads.
  if (!/^\d+\.\d+\.\d+$/.test(version) || !/^\d+\.\d+\.\d+$/.test(floor)) {
    throw new Error('Retirement version floor is not comparable')
  }

  const current = version.split('.').map(BigInt)
  const minimum = floor.split('.').map(BigInt)

  for (let index = 0; index < 3; index += 1) {
    if (current[index] > minimum[index]) {
      return
    }

    if (current[index] < minimum[index]) {
      throw new Error('Destination is below retirement minimum version')
    }
  }
}

/** Metadata reads never download an application or follow redirects. */
export class ChannelResolver {
  private readonly base: string
  private readonly fetcher: typeof fetch

  constructor(private readonly deps: ChannelResolverDeps) {
    this.base = channelPublicBase(deps.build.publicBase)
    this.fetcher = deps.fetch ?? fetch
    validateChannelName(deps.build.channel)
  }

  async read(key: string, digest?: string): Promise<string> {
    const url = `${this.base}/${channelKey(key)}`

    const response = await this.fetcher(url, {
      redirect: 'error',
      signal: AbortSignal.timeout(30_000),
      cache: 'no-store',
      headers: { 'Cache-Control': 'no-cache' }
    })

    if (!response.ok || response.url !== url) {
      throw new Error(`Channel read unavailable: HTTP ${response.status}`)
    }

    const maximum = 4 * 1024 * 1024
    const reader = response.body?.getReader()

    if (!reader) {
      throw new Error('Channel metadata has no body')
    }

    const chunks: Uint8Array[] = []
    let size = 0

    try {
      while (true) {
        const { done, value } = await reader.read()

        if (done) {
          break
        }

        size += value.byteLength

        if (size > maximum) {
          throw new Error('Channel metadata exceeds size limit')
        }

        chunks.push(value)
      }
    } finally {
      await reader.cancel()
    }

    const body = Buffer.concat(chunks)

    if (digest && createHash('sha256').update(body).digest('hex') !== digest) {
      throw new Error('Channel metadata SHA256 mismatch')
    }

    return new TextDecoder('utf-8', { fatal: true }).decode(body)
  }

  async resolve(): Promise<ChannelResolution> {
    let record = decodeChannelRecord(await this.read(`releases/channels/${this.deps.build.channel}.json`))
    this.assertRecord(record, this.deps.build.channel)

    if (!sameChannelIdentity(record.identity, this.deps.build.identity)) {
      throw new Error('Installed channel identity changed')
    }

    if (record.state === 'active') {
      const native = this.deps.platform === 'win32' && this.deps.arch === 'x64' && this.deps.nativePackageUri === true

      if (!record.head && record.nativeHead && !native) {
        throw new Error('Native channel is not supported by this installation')
      }

      return record.head || record.nativeHead
        ? {
            kind: 'active',
            target: await this.target(record, native)
          }
        : { kind: 'empty', channel: record }
    }

    const retired: RetiredChannel = record

    if (retired.destination === retired.name) {
      throw new Error('Channel retirement cycle')
    }

    record = decodeChannelRecord(await this.read(`releases/channels/${retired.destination}.json`))
    this.assertRecord(record, retired.destination)

    if (record.state !== 'active' || record.policy !== 'stable-release') {
      throw new Error('Retirement requires a directly active stable-release destination')
    }

    if (!record.head || retired.destinationHead.sequence > record.head.sequence) {
      throw new Error('Retirement destination exceeds published head')
    }

    // Stable may advance while preview is offline; preserve its first receiver.
    const target: ChannelTarget = await this.target({ ...record, head: retired.destinationHead })

    if (target.manifest.receiverProtocol !== retired.receiverProtocol) {
      throw new Error('Stable build has no supported retirement receiver')
    }

    assertVersionFloor(target.manifest.request.sourceVersion, retired.minimumVersion)

    return { kind: 'retirement', retirement: { source: this.deps.build, target, receiverKind: retired.receiver.kind } }
  }

  private assertRecord(record: ActiveChannel | RetiredChannel, name: string): void {
    if (record.name !== name) {
      throw new Error('Channel name does not match object key')
    }

    if (record.repository.toLowerCase() !== this.deps.build.repository.toLowerCase()) {
      throw new Error('Channel repository authority mismatch')
    }
  }

  private deliveryHead(channel: ActiveChannel, native: boolean): ActiveChannel['head'] {
    return native && channel.nativeHead ? channel.nativeHead : channel.head
  }

  private async target(channel: ActiveChannel, native = false): Promise<ChannelTarget> {
    const head = this.deliveryHead(channel, native)

    if (!head) {
      throw new Error('No destination build published')
    }

    const manifest = decodeChannelManifest(await this.read(head.manifestKey, head.sha256))

    if (manifest.schema !== (head === channel.nativeHead ? 2 : 1)) {
      throw new Error('Channel manifest schema/head mismatch')
    }

    const request = manifest.request

    if (
      request.buildId !== head.buildId ||
      request.sequence !== head.sequence ||
      request.channel !== channel.name ||
      request.repository.toLowerCase() !== channel.repository.toLowerCase() ||
      request.publicBase !== this.base ||
      !sameChannelIdentity(request.identity, channel.identity)
    ) {
      throw new Error('Channel manifest request binding mismatch')
    }

    if (
      channel.policy === 'preview' &&
      (request.version !== `0.0.${request.sequence}` ||
        request.windowsVersion !== `0.${Math.floor(request.sequence / 65536)}.${request.sequence % 65536}.0`)
    ) {
      throw new Error('Channel package version does not match sequence')
    }

    this.assertPackages(channel, manifest)

    const entry = manifest.packages.find(
      (item: ChannelPackage): boolean => item.platform === this.deps.platform && item.arch === this.deps.arch
    )

    if (!entry) {
      throw new Error('Channel has no package for this platform and architecture')
    }

    const signer = entry.platform === 'darwin' ? entry.teamId : entry.publisher

    if (!this.deps.signer || signer !== this.deps.signer) {
      throw new Error('Channel package signing identity mismatch')
    }

    const nativeGraph = await this.nativeGraph(entry, request)

    return {
      ...(nativeGraph ? { nativeGraph } : {}),
      channel,
      manifest,
      package: entry,
      manifestSha256: head.sha256,
      artifactUrl: `${this.base}/${entry.artifact.key}`,
      feedUrl: `${this.base}/${entry.feed.key}`
    }
  }

  private async nativeGraph(
    entry: ChannelPackage,
    request: ChannelManifest['request']
  ): Promise<NativePackageGraph | undefined> {
    if (!entry.native) {
      return undefined
    }
    const body = await this.read(entry.native.graph.key, entry.native.graph.sha256)

    if (Buffer.byteLength(body) !== entry.native.graph.size) {
      throw new Error('Native graph descriptor size mismatch')
    }

    return decodeNativePackageGraph(body, request, entry)
  }

  private assertNativePrefix(entry: ChannelPackage, buildId: string): void {
    if (
      entry.native &&
      (!entry.native.graph.key.startsWith(buildPrefix(buildId)) || !entry.artifact.key.startsWith(buildPrefix(buildId)))
    ) {
      throw new Error('Native graph outside immutable build namespace')
    }
  }

  private assertPackages(channel: ActiveChannel, manifest: ChannelManifest): void {
    const request = manifest.request
    const prefixes = [buildPrefix(request.buildId)]

    if (channel.policy !== 'preview') {
      if (!request.releaseTag || request.version !== request.releaseTag.slice(1)) {
        throw new Error('Protected release version mismatch')
      }

      // The archive prefix is the attempt ref when the request names one; a
      // bare releaseTag fallback never holds attempt artifacts (fail closed).
      prefixes.push(`releases/tag/${request.archiveRef ?? request.releaseTag}/`)
    }

    for (const entry of manifest.packages) {
      this.assertNativePrefix(entry, request.buildId)
      const expectedIdentity = entry.platform === 'darwin' ? request.identity.appId : request.identity.msixAppIdWithOrg
      const expectedVersion = entry.platform === 'darwin' ? request.version : request.windowsVersion

      if (
        (entry.artifact.chunks && entry.artifact.chunks.buildId !== request.buildId) ||
        entry.identity !== expectedIdentity ||
        entry.version !== expectedVersion ||
        entry.feed.channel !== 'stable' ||
        !prefixes.some((prefix: string): boolean => entry.artifact.key.startsWith(prefix)) ||
        !prefixes.some((prefix: string): boolean => entry.feed.key.startsWith(prefix))
      ) {
        throw new Error('Channel package identity, version or immutable prefix mismatch')
      }

      if (entry.platform === 'darwin' && !entry.feed.key.endsWith(`/${entry.feed.channel}-mac.yml`)) {
        throw new Error('Invalid native macOS feed descriptor')
      }

      if (entry.platform === 'win32' && !entry.native && !entry.feed.key.endsWith('.appinstaller')) {
        throw new Error('Invalid native Windows feed descriptor')
      }
    }
  }
}
