/** The server owns the allowlisted origin; clients own only these fixed task paths. */
const tasks = { manage: '/billing', plans: '/pricing', topUp: '/billing/top-up', orders: '/billing/orders' } as const

export function semantierPortalTask(origin: string | null, organizationId: string, accountId: string, task: keyof typeof tasks): string | null {
  if (!origin || !organizationId || !accountId) {return null}

  try {
    const url = new URL(origin)

    if (url.username || url.password || url.search || url.hash || !['', '/'].includes(url.pathname)
      || (url.protocol !== 'https:' && !(url.protocol === 'http:' && ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname)))) {
      return null
    }

    url.pathname = tasks[task]
    url.search = new URLSearchParams({ organization_id: organizationId, account_id: accountId }).toString()

    return url.toString()
  } catch {
    return null
  }
}

/** Presentation-only return. It cannot authorize accounts, devices or arbitrary URLs. */
export function isSemantierPortalReturn(initial: string, destination: string): boolean {
  try {
    const source = new URL(initial)
    const next = new URL(destination)

    return source.searchParams.get('entry') === 'windows' && !next.username && !next.password &&
      ['https:', 'http:'].includes(source.protocol) && next.origin === source.origin &&
      next.pathname.startsWith('/billing') && next.hash === '#semantier-return-to-app'
  } catch {
    return false
  }
}
