import { modelProviderName } from '@/i18n/model-service'
import type { ModelOptionProvider } from '@hermes/shared'
import { useStore } from '@nanostores/react'

import { useI18n } from '@/i18n'
import { $refreshingModelSources, modelDiscoveryRefreshKey } from '@/store/model-discovery'

const messages: Record<string, Record<string, string>> = {
  refreshing: {
    en: 'Refreshing provider models…', zh: '正在刷新服务商模型…', 'zh-hant': '正在重新整理供應商模型…',
    ja: 'プロバイダーのモデルを更新しています…', fr: 'Actualisation des modèles du fournisseur…',
    de: 'Anbietermodelle werden aktualisiert…', es: 'Actualizando los modelos del proveedor…',
    ru: 'Обновление моделей провайдера…', ar: 'جارٍ تحديث نماذج المزود…'
  },
  cached: {
    en: 'Showing a verified cached provider list.', zh: '显示已验证的服务商缓存列表。', 'zh-hant': '顯示已驗證的供應商快取清單。',
    ja: '検証済みのキャッシュ一覧を表示しています。', fr: 'Liste du fournisseur vérifiée en cache.',
    de: 'Verifizierte Anbieterliste aus dem Cache.', es: 'Lista verificada del proveedor en caché.',
    ru: 'Показан проверенный список провайдера из кеша.', ar: 'تُعرض قائمة المزود الموثقة من ذاكرة التخزين المؤقت.'
  },
  stale: {
    en: 'Refresh failed; showing the last verified provider list.', zh: '刷新失败；显示上次验证的服务商列表。', 'zh-hant': '重新整理失敗；顯示上次驗證的供應商清單。',
    ja: '更新に失敗したため、最後に検証した一覧を表示しています。', fr: 'Actualisation échouée ; dernière liste vérifiée affichée.',
    de: 'Aktualisierung fehlgeschlagen; letzte verifizierte Liste wird angezeigt.', es: 'Falló la actualización; se muestra la última lista verificada.',
    ru: 'Обновление не удалось; показан последний проверенный список.', ar: 'فشل التحديث؛ تُعرض آخر قائمة موثقة للمزود.'
  },
  missing: {
    en: 'Waiting for the provider model list.', zh: '正在等待服务商模型列表。', 'zh-hant': '正在等待供應商模型清單。',
    ja: 'プロバイダーのモデル一覧を待機しています。', fr: 'En attente de la liste des modèles du fournisseur.',
    de: 'Warten auf die Modellliste des Anbieters.', es: 'Esperando la lista de modelos del proveedor.',
    ru: 'Ожидание списка моделей провайдера.', ar: 'في انتظار قائمة نماذج المزود.'
  },
  error: {
    en: 'Model list unavailable. Refresh to try again.', zh: '模型列表不可用。请刷新重试。', 'zh-hant': '模型清單無法使用。請重新整理再試。',
    ja: 'モデル一覧を取得できません。更新して再試行してください。', fr: 'Liste indisponible. Actualisez pour réessayer.',
    de: 'Modellliste nicht verfügbar. Zum Wiederholen aktualisieren.', es: 'Lista no disponible. Actualiza para reintentar.',
    ru: 'Список моделей недоступен. Обновите для повтора.', ar: 'قائمة النماذج غير متاحة. حدّث للمحاولة مجددًا.'
  },
  unsupported: {
    en: 'This provider does not support model listing.', zh: '此服务商不支持查询模型列表。', 'zh-hant': '此供應商不支援查詢模型清單。',
    ja: 'このプロバイダーはモデル一覧に対応していません。', fr: 'Ce fournisseur ne permet pas de lister les modèles.',
    de: 'Dieser Anbieter unterstützt keine Modellliste.', es: 'Este proveedor no permite listar modelos.',
    ru: 'Этот провайдер не поддерживает список моделей.', ar: 'لا يدعم هذا المزود عرض قائمة النماذج.'
  },
  empty: {
    en: 'The provider returned no models.', zh: '服务商未返回模型。', 'zh-hant': '供應商未傳回模型。',
    ja: 'プロバイダーからモデルが返されませんでした。', fr: 'Le fournisseur n’a renvoyé aucun modèle.',
    de: 'Der Anbieter hat keine Modelle zurückgegeben.', es: 'El proveedor no devolvió modelos.',
    ru: 'Провайдер не вернул моделей.', ar: 'لم يُرجع المزود أي نماذج.'
  }
}

export function ModelDiscoveryRefreshStatus({ ownerKey, profile }: { ownerKey?: string; profile: string }) {
  const { locale } = useI18n()
  const refreshing = useStore($refreshingModelSources)
  if (!ownerKey || !refreshing.has(modelDiscoveryRefreshKey(ownerKey, profile))) {return null}
  return <div className="px-3 py-1 text-xs text-(--ui-text-tertiary)" role="status">
    {messages.refreshing[locale] ?? messages.refreshing.en}
  </div>
}

export function ModelDiscoveryStatus({ provider }: { provider: ModelOptionProvider }) {
  const { locale } = useI18n()
  const state = provider.discovery_state === 'fresh' && !(provider.models?.length) ? 'empty' : provider.discovery_state
  if (!state || state === 'fresh') {return null}
  const message = messages[state]?.[locale] ?? messages[state]?.en
  if (!message) {return null}
  return <div className="px-3 py-1 text-xs text-(--ui-text-tertiary)" role="status">{modelProviderName(provider, locale)}: {message}</div>
}
