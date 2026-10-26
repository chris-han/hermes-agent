import { LOCAL_CONNECTION_ID } from '@hermes/shared'
import { atom, computed } from 'nanostores'

import { $defaultProfileRoute } from '@/store/default-profile'
import { $activeGatewayProfile, $profiles, normalizeProfileKey } from '@/store/profile'
import { $connection } from '@/store/session'

// ── Shared settings "Applies to" scope ──────────────────────────────────────
// One selection shared by every config-backed settings page (Model, Workspace,
// Safety, Memory & Context, Voice, Tools & Keys) and the Messaging overlay, so
// picking a profile on one page carries to the next instead of resetting per
// page. `null` means "follow the app's active profile" — the default, which
// keeps single-profile users on the exact pre-existing code path (requests
// fall back to the app-wide active profile in api/client.ts `profileScoped`).
export const $settingsScopeOverride = atom<null | string>(null)

// The profile the settings pages are currently editing (a concrete key).
export const $settingsScopeProfile = computed([$settingsScopeOverride, $activeGatewayProfile], (override, active) =>
  normalizeProfileKey(override ?? active)
)

// The native startup owner already selects the managed desktop home. Hermes'
// is_default still identifies its legacy root; it must not make that root a
// competing desktop default. Never borrow a local preference on another source.
const $desktopLocalDefault = computed([$defaultProfileRoute, $connection, $profiles], (route, connection, profiles) => {
  const localSource = connection?.mode === 'local' &&
    (!connection.connectionId || connection.connectionId === LOCAL_CONNECTION_ID)

  const localPreference = route && (!route.connectionId || route.connectionId === LOCAL_CONNECTION_ID)

  return localSource && localPreference && route.profile === 'semantier-local' &&
    profiles.some(profile => profile.name === route.profile) ? route.profile : null
})

// Ordinary Settings need no second root choice for a managed single-profile
// desktop. An explicitly selected legacy root remains visible and keeps its
// exact request scope; advanced profile access and all stored data stay intact.
export const $settingsScopeProfiles = computed([$profiles, $desktopLocalDefault, $settingsScopeProfile], (profiles, desktopDefault, selected) =>
  desktopDefault && selected !== 'default'
    ? profiles.filter(profile => !(profile.is_default && normalizeProfileKey(profile.name) === 'default'))
    : profiles
)

// Whether the settings pages are editing a profile OTHER than the default
// one. The scope follows the app's active profile when no override is set —
// which, after opening any Bot Mode chat, is the BOT's profile — so an edit
// can land in profiles/<bot>/config.yaml while the user believes they are
// editing their main config (#89190/#89162 class). Surfaces render this
// loudly. Until the roster has loaded (no is_default entry yet) the root
// profile's canonical key is assumed, so an unknown default fails loud, not
// quiet.
export const $settingsScopeEditsNonDefault = computed([$settingsScopeProfile, $profiles, $desktopLocalDefault], (selected, profiles, desktopDefault) => {
  const defaultProfile = profiles.find(profile => profile.is_default)

  return selected !== (desktopDefault ?? normalizeProfileKey(defaultProfile?.name))
})

// ── Request-scope form (THE value to hand to API helpers) ──────────────────
// The store contract and the API contract disagree about `null`:
//   - here, `null` means "follow the app's active profile" (no override);
//   - in api/client.ts `profileScoped()`/`capabilityScoped()`, `null` means
//     "deliberately suppress the active profile and target primary/default" —
//     only `undefined` falls back to the active profile.
// Passing the raw override into an API helper therefore silently retargets
// every read/write to the primary profile whenever no override is set — the
// "model change reverts when I re-enter the tab" class of bug (#90549: the
// page WROTE the right profile but READ primary back).
//
// It must also never be `undefined` for a profile the pages actually display.
// `undefined` makes `profileScoped()` omit `?profile=` altogether, and the
// backend resolves an omitted profile to the home it was LAUNCHED with — not
// the profile this store says we are editing. Those differ the moment a pooled
// desktop backend serves a profile other than its launch home (`hermes
// --profile A serve`, editing B): every settings page then READ A's values and
// WROTE them back to A, while the "Changes on this page apply to 'B'" note —
// which reads the concrete $settingsScopeProfile — kept naming B (#118432).
// Send the concrete key the pages render. Only a name with no profile
// directory behind it (`custom`, a HERMES_HOME outside profiles/) stays
// `undefined`, where the ambient path is the only correct answer.
export const $settingsRequestProfile = computed($settingsScopeProfile, (selected): string | undefined =>
  selected === 'custom' ? undefined : selected
)

// Select the profile the settings pages should edit. Picking the app's active
// profile stores `null` (no override) so the scope keeps following the app on
// profile switches — and requests keep their unscoped default shape.
export function setSettingsScope(name: string): void {
  const key = normalizeProfileKey(name)

  $settingsScopeOverride.set(key === normalizeProfileKey($activeGatewayProfile.get()) ? null : key)
}

// An app-wide profile switch re-homes every settings surface to the new
// backend; a surviving override would silently keep edits pointed at the
// previous target. Same drop-the-override contract as the Capabilities
// selector (app/capabilities useOnProfileSwitch).
let lastActiveProfile = normalizeProfileKey($activeGatewayProfile.get())

$activeGatewayProfile.subscribe(value => {
  const key = normalizeProfileKey(value)

  if (key !== lastActiveProfile) {
    lastActiveProfile = key
    $settingsScopeOverride.set(null)
  }
})
