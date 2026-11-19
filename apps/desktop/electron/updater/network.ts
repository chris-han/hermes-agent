/** Adapt Electron's system-network fetch, which omits Response.url, without
 * allowing redirects or browser cookies into anonymous update downloads. */
export function nativeUpdateFetch(nativeFetch: typeof fetch): typeof fetch {
  return async (input, init) => {
    if (init?.redirect !== 'error') {
      throw new Error('Update transport requires redirect rejection')
    }
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
    const response = await nativeFetch(input, { ...init, credentials: 'omit' })

    if (response.redirected || (response.url && response.url !== url)) {
      throw new Error('Update response origin mismatch')
    }

    // Redirects were forbidden by the native transport. The original request
    // therefore owns the response URL even when Electron omits that metadata.
    if (!response.url) {
      Object.defineProperty(response, 'url', { value: url })
    }

    return response
  }
}
