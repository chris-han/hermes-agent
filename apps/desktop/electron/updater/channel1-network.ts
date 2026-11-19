import { nativeUpdateFetch } from './network'

/** Metadata keeps OS proxy policy; canonical OSS bytes use isolated HTTPS. */
export function createDesktopUpdateFetches(
  platform: string,
  systemFetch: typeof fetch,
  sessions: { fromPartition: (partition: string, options: { cache: boolean }) => DirectSession }
): { metadata: typeof fetch; artifact: typeof fetch } {
  const metadata = nativeUpdateFetch(systemFetch)

  const artifact =
    platform === 'win32'
      ? channel1ArtifactFetch(metadata, () => {
          return sessions.fromPartition('semantier-channel1-artifacts', { cache: false })
        })
      : metadata

  return { metadata, artifact }
}

interface DirectSession {
  setProxy: (config: { mode: 'direct' }) => Promise<void>
  fetch: (input: string | Request, init?: RequestInit) => Promise<Response>
}

/** Only canonical Channel 1 packages use this anonymous, isolated direct session.
 * Failure propagates; metadata and other channels retain their existing policy.
 */
export function channel1ArtifactFetch(fallback: typeof fetch, create: () => DirectSession): typeof fetch {
  let direct: Promise<typeof fetch> | undefined

  return async (input, init) => {
    const url = new URL(typeof input === 'string' ? input : input instanceof URL ? input.href : input.url)

    if (url.origin !== 'https://assets.semantier.cn') {
      return fallback(input, init)
    }

    if (
      url.username ||
      url.password ||
      url.search ||
      url.hash ||
      !/^\/releases\/channel-builds\/[a-f0-9]{32}\/SemantierAgent\.msix$/.test(url.pathname)
    ) {
      throw new Error('Invalid Channel 1 pinned package URL')
    }

    direct ??= (async () => {
      const session = create()
      await session.setProxy({ mode: 'direct' })

      return nativeUpdateFetch((request, options) =>
        session.fetch(request instanceof URL ? request.href : request, options)
      )
    })()

    return (await direct)(input, init)
  }
}
