// Format an api-request failure for desktop.log. The renderer only ever sees
// the invoke rejection; the real stack lives here in main, so persist it
// before rethrowing. Clamp: paths and error detail must not bloat the log.
export function formatApiRequestFailure(
  request: { method?: string; path?: string } | null | undefined,
  error: unknown
): string {
  const method = String(request?.method ?? 'GET').toUpperCase()
  const path = String(request?.path ?? '(no path)').slice(0, 500)
  const detail = error instanceof Error ? (error.stack ?? error.message) : String(error)

  return `[hermes:api ${method} ${path}] ${detail}`.slice(0, 6000)
}
