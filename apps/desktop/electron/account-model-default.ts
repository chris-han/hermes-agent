/** Resolve preference only against the installed device's current authorized catalog. */
export function resolveAccountModelDefault(preference: unknown, recommendation: unknown, readiness: unknown): string | null {
  const pref = preference as { account_id?: unknown; model?: unknown } | null
  const rec = recommendation as { recommended_model?: unknown } | null
  const ready = readiness as { account_id?: unknown; account_provisioning?: { inference_ready?: unknown };
    catalog?: { status?: unknown; data?: unknown } } | null
  if (!pref || !ready || typeof pref.account_id !== 'string' || !pref.account_id || pref.account_id !== ready.account_id ||
      ready.account_provisioning?.inference_ready !== true || ready.catalog?.status !== 'available' || !Array.isArray(ready.catalog.data)) {return null}
  const model = pref.model === null || pref.model === '' || pref.model === 'platform_recommended' ? rec?.recommended_model : pref.model
  if (typeof model !== 'string' || !model.trim() || model.length > 256 || model !== model.trim()) {return null}
  const matches = ready.catalog.data.filter((entry: unknown) => {
    if (!entry || typeof entry !== 'object') {return false}
    const row = entry as { id?: unknown; account_id?: unknown; access?: unknown; modality?: unknown }
    return row.id === model && row.account_id === pref.account_id && row.access === 'authorized' &&
      (row.modality === undefined || row.modality === 'chat')
  })
  return matches.length === 1 ? model : null
}
