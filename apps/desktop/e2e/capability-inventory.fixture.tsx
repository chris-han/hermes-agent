import '../src/styles.css'

import { QueryClientProvider } from '@tanstack/react-query'
import { useState } from 'react'
import { createRoot } from 'react-dom/client'
import { MemoryRouter } from 'react-router'

import { CapabilitiesView } from '../src/app/capabilities'
import { ConnectorsDirectory } from '../src/app/capabilities/connectors/connectors-directory'
import { deriveCards, EMPTY_CONNECTORS_FILTER } from '../src/app/capabilities/connectors/derive'
import { CapabilityInventory } from '../src/app/capabilities/inventory'
import { NativePluginCatalog } from '../src/app/capabilities/plugins/native-catalog'
import { I18nProvider } from '../src/i18n'
import { queryClient } from '../src/lib/query-client'
import { ThemeProvider } from '../src/themes/context'

const params = new URLSearchParams(location.search)
const count = Number(params.get('count') || 14)
const locale = params.get('locale') || 'en'
const calls: any[] = []
let failToggle = false
let failCatalog = sessionStorage.getItem('fixture-offline') === 'true'
const items = new Map<string, any[]>()

function inventory(profile: string) {
  if (!items.has(profile))
    {items.set(
      profile,
      Array.from({ length: count }, (_, i) => ({
        name: `${profile}-skill-${String(i).padStart(3, '0')}`,
        description: `Long bilingual research skill 习得技能检索与文档整理 ${i}`,
        category: i % 2 ? 'research' : 'writing',
        provenance: i % 3 ? 'bundled' : 'agent',
        enabled: true,
        usage: i
      }))
    )}

  return items.get(profile)!
}

window.hermesDesktop = {
  api: async (request: any) => {
    calls.push(structuredClone(request))
    const profile = request.profile || 'default'

    if (request.path === '/api/profiles')
      {return { profiles: [{ name: 'default', is_default: true }, { name: 'other' }] }}

    if (request.path === '/api/skills') {return structuredClone(inventory(profile))}

    if (request.path.includes('/hub/official')) {return { skills: [] }}

    if (request.path.includes('/hub/sources')) {
      if (failCatalog) {throw new Error('Fixture catalog offline')}

      return {
        featured: [
          {
            name: 'Catalog research',
            identifier: 'fixture/research',
            description: 'Catalog metadata supplied by backend fixture',
            source: 'fixture',
            repo: 'https://example.invalid/fixture',
            tags: []
          }
        ],
        sources: [],
        installed: {}
      }
    }

    if (request.path.includes('/hub/search')) {
      if (failCatalog) {throw new Error('Fixture catalog offline')}

      return {
        results: [
          {
            name: 'Catalog research',
            identifier: 'fixture/research',
            description: 'Catalog result',
            source: 'fixture',
            repo: null,
            tags: []
          }
        ],
        timed_out: [],
        installed: {},
        source_counts: {}
      }
    }

    if (request.path.includes('/hub/preview')) {return { skill_md: '# Research catalog preview' }}

    if (request.path.includes('/hub/install')) {return { name: 'fixture-action' }}

    if (request.path.includes('/actions/'))
      {return { running: false, exit_code: 0, lines: ['Installed'], name: 'fixture-action' }}

    if (request.path.includes('/skills/content'))
      {return { content: '# Skill content\n\nExact owner: ' + profile, name: 'skill' }}

    if (request.path.startsWith('/api/learning/node')) {
      if (request.method === 'PUT') {return { ok: true }}

      if (request.method === 'DELETE') {
        items.set(
          profile,
          inventory(profile).filter(row => row.name !== request.body.id)
        )

        return { ok: true }
      }

      return { ok: true, kind: 'skill', content: '# Editable learned skill', label: profile }
    }

    if (request.path.includes('/skills/toggle')) {
      if (failToggle) {throw new Error('Fixture toggle rejected')}
      const item = inventory(profile).find(row => row.name === request.body.name)

      if (item) {item.enabled = request.body.enabled}

      return { ok: true }
    }

    if (request.path.includes('/tools/toolsets') && !request.path.includes('/config'))
      {return [
        {
          name: 'web',
          label: 'Web Search',
          enabled: true,
          available: true,
          configured: true,
          tools: ['web_search'],
          description: 'Research tools'
        }
      ]}

    if (request.path.includes('/config')) {return { has_category: false, display: { language: locale } }}

    if (request.path.includes('/analytics')) {return { tools: [] }}

    return {}
  }
} as any
Object.assign(window, {
  fixture: {
    calls,
    rejectToggle: (value: boolean) => {
      failToggle = value
    },
    offline: (value: boolean) => {
      failCatalog = value
      sessionStorage.setItem('fixture-offline', String(value))
    }
  }
})

function Fixture() {
  const [profile, setProfile] = useState('default')
  const [connection, setConnection] = useState('local')
  const [plugins, setPlugins] = useState(false)
  const [connectors, setConnectors] = useState(false)
  const [filter, setFilter] = useState(EMPTY_CONNECTORS_FILTER)

  return (
    <div className="flex h-screen flex-col">
      <div className="flex shrink-0 flex-wrap gap-3 p-2">
        <button onClick={() => setProfile(p => (p === 'default' ? 'other' : 'default'))}>Switch fixture profile</button>
        <button onClick={() => setConnection(p => (p === 'local' ? 'remote-fixture' : 'local'))}>
          Switch fixture connection
        </button>
        <button onClick={() => setPlugins(p => !p)}>Plugin catalog fixture</button>
        <button
          onClick={() => {
            setConnectors(value => !value)
            setPlugins(false)
          }}
        >
          Connectors fixture
        </button>
        <span data-testid="scope">
          {connection}::{profile}
        </span>
      </div>
      <div className="min-h-0 flex-1">
        {connectors ? (
          <CapabilityInventory tab="connectors">
            <div className="flex h-full p-3">
              <ConnectorsDirectory
                cards={deriveCards({
                  hosted: [],
                  bundled: [],
                  local: Array.from({ length: count }, (_, index) => ({
                    name: `connector-${index}`,
                    description: 'Long bilingual connector 连接器',
                    target: 'fixture',
                    enabled: true,
                    status: 'ok' as const,
                    toolsTotal: 5
                  }))
                })}
                filter={filter}
                onFilterChange={setFilter}
                onOpen={card => calls.push({ openConnector: card.slug })}
              />
            </div>
          </CapabilityInventory>
        ) : plugins ? (
          <CapabilityInventory tab="plugins">
            <NativePluginCatalog
              connectionId={connection}
              profile={profile}
              request={async () =>
                ({
                  available: true,
                  entries: Array.from({ length: count }, (_, index) => ({
                    name: `fixture-plugin-${index}`,
                    description: 'Native plugin catalog',
                    repo: 'https://example.invalid/plugin',
                    sha: 'a'.repeat(40)
                  }))
                }) as any
              }
            />
          </CapabilityInventory>
        ) : (
          <CapabilitiesView embedded fixedConnection={connection} fixedProfile={profile} />
        )}
      </div>
    </div>
  )
}

createRoot(document.getElementById('root')!).render(
  <MemoryRouter>
    <QueryClientProvider client={queryClient}>
      <I18nProvider configClient={null} initialLocale={locale}>
        <ThemeProvider>
          <Fixture />
        </ThemeProvider>
      </I18nProvider>
    </QueryClientProvider>
  </MemoryRouter>
)
