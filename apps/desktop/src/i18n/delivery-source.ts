import type { BundledLocale } from './types'
interface Copy { title: string; channel1: string; channel2: string; channel3: string; missing: string; failed: string }
export const deliverySourceCopy: Record<string, Copy> = {
  en: { title: 'Update source', channel1: 'Channel 1 · Alibaba Cloud ECS', channel2: 'Channel 2 · Cloudflare', channel3: 'Channel 3 · Local development', missing: 'Endpoint not configured', failed: 'Could not change update source. Try again.' },
  zh: { title: '更新来源', channel1: '通道 1 · 阿里云 ECS', channel2: '通道 2 · Cloudflare', channel3: '通道 3 · 本地开发', missing: '尚未配置更新地址', failed: '无法切换更新来源，请重试。' },
  'zh-hant': { title: '更新來源', channel1: '通道 1 · 阿里雲 ECS', channel2: '通道 2 · Cloudflare', channel3: '通道 3 · 本機開發', missing: '尚未設定更新位址', failed: '無法切換更新來源，請重試。' },
  ja: { title: '更新元', channel1: 'チャネル 1 · Alibaba Cloud ECS', channel2: 'チャネル 2 · Cloudflare', channel3: 'チャネル 3 · ローカル開発', missing: 'エンドポイント未設定', failed: '更新元を変更できませんでした。再試行してください。' },
  de: { title: 'Update-Quelle', channel1: 'Kanal 1 · Alibaba Cloud ECS', channel2: 'Kanal 2 · Cloudflare', channel3: 'Kanal 3 · Lokale Entwicklung', missing: 'Endpunkt nicht konfiguriert', failed: 'Update-Quelle konnte nicht geändert werden. Erneut versuchen.' },
  fr: { title: 'Source des mises à jour', channel1: 'Canal 1 · Alibaba Cloud ECS', channel2: 'Canal 2 · Cloudflare', channel3: 'Canal 3 · Développement local', missing: 'Point de terminaison non configuré', failed: 'Impossible de changer de source. Réessayez.' },
  es: { title: 'Origen de actualizaciones', channel1: 'Canal 1 · Alibaba Cloud ECS', channel2: 'Canal 2 · Cloudflare', channel3: 'Canal 3 · Desarrollo local', missing: 'Dirección sin configurar', failed: 'No se pudo cambiar el origen. Reinténtalo.' },
  ru: { title: 'Источник обновлений', channel1: 'Канал 1 · Alibaba Cloud ECS', channel2: 'Канал 2 · Cloudflare', channel3: 'Канал 3 · Локальная разработка', missing: 'Адрес не настроен', failed: 'Не удалось изменить источник. Повторите попытку.' },
  ar: { title: 'مصدر التحديثات', channel1: 'القناة 1 · Alibaba Cloud ECS', channel2: 'القناة 2 · Cloudflare', channel3: 'القناة 3 · التطوير المحلي', missing: 'لم يتم إعداد العنوان', failed: 'تعذر تغيير مصدر التحديثات. حاول مجددًا.' }
} satisfies Record<BundledLocale, Copy>
