import { isRecord } from '@hermes/shared/i18n'
import type { WorkspaceScope } from '../../electron/semantier-workspace-client'
import { persistString, storedString } from '@/lib/storage'
import type { I18nConfigClient } from './context'

/** Cloud chrome preferences belong to Desktop; they never write Native config. */
export function workspaceLocaleClient(origin: string, scope?: WorkspaceScope): I18nConfigClient | null {
  if (!scope?.userId || !scope.workspaceId || !scope.organizationId) { return null }
  const key = `hermes.workspace.display.language:${JSON.stringify([
    new URL(origin).origin, scope.userId, scope.workspaceId, scope.organizationId
  ])}`
  return {
    getConfig: async () => {
      const language = storedString(key)
      return language ? { display: { language } } : {}
    },
    saveConfig: async config => {
      const language = isRecord(config.display) ? config.display.language : undefined
      if (typeof language !== 'string' || !language.trim()) { return { ok: false } }
      persistString(key, language)
      return { ok: storedString(key) === language }
    }
  }
}
