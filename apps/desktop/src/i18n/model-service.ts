import type { BundledLocale } from './types'

export interface ModelServiceCopy {
  providerName: string
  title: string
  custom: string
  customDescription: string
  cloudDescription: string
  signIn: string
  advancedRuntime: string
  setup: string
  plan: string
  preparing: string
  expired: string
  credit: string
  reauth: string
  unavailable: string
  denied: string
  ready: string
  configure: string
  retry: string
  noKey: string
  coverage: string
  saveUse: string
  cancel: string
  details: string
}

export const modelServiceCopy: Record<string, ModelServiceCopy> = {
  "en": {
    providerName: 'Semantier Model Provider',
    title: "Model service",
    custom: "Custom endpoint",
    customDescription: "Save an OpenAI-compatible provider URL and API key. Choose a model afterward.",
    cloudDescription: "Sign in to use your Semantier models and account credit.",
    signIn: "Sign in to Semantier Model Provider",
    advancedRuntime: "Advanced · Agent runtime",
    setup: "Connect a model service to start",
    plan: "Choose a plan to use Semantier Model Provider",
    preparing: "Setting up model access",
    expired: "Renew model access",
    credit: "Add credit",
    reauth: "Sign in again",
    unavailable: "Connection unavailable",
    denied: "Access denied for this account",
    ready: "Connected",
    configure: "Configure model",
    retry: "Retry",
    noKey: "This endpoint requires no API key",
    coverage: "Checks model discovery and authentication only. No inference request is sent.",
    saveUse: "Save and use",
    cancel: "Cancel",
    details: "Details",
  },
  "zh": {
    providerName: 'Semantier 模型服务',
    title: "模型服务",
    custom: "自定义端点",
    customDescription: "保存兼容 OpenAI 的服务商地址和 API 密钥，之后再选择模型。",
    cloudDescription: "登录后使用 Semantier 模型与账户额度。",
    signIn: "登录 Semantier 模型服务",
    advancedRuntime: "高级 · Agent 运行位置",
    setup: "连接模型服务即可开始",
    plan: "选择套餐以使用 Semantier 模型服务",
    preparing: "正在开通模型服务",
    expired: "续订模型服务",
    credit: "充值",
    reauth: "重新登录",
    unavailable: "暂时无法连接",
    denied: "此账户无权访问",
    ready: "已连接",
    configure: "配置模型",
    retry: "重试",
    noKey: "此端点无需 API 密钥",
    coverage: "仅检查模型列表与身份验证，不发送推理请求。",
    saveUse: "保存并使用",
    cancel: "取消",
    details: "详细信息",
  },
  "zh-hant": {
    providerName: 'Semantier 模型服務',
    title: "模型服務",
    custom: "自訂端點",
    customDescription: "儲存相容 OpenAI 的供應商網址與 API 金鑰，之後再選擇模型。",
    cloudDescription: "登入後使用 Semantier 模型與帳戶額度。",
    signIn: "登入 Semantier 模型服務",
    advancedRuntime: "進階 · Agent 執行位置",
    setup: "連線模型服務即可開始",
    plan: "選擇方案以使用 Semantier 模型服務",
    preparing: "正在開通模型服務",
    expired: "續訂模型服務",
    credit: "儲值",
    reauth: "重新登入",
    unavailable: "暫時無法連線",
    denied: "此帳戶無權存取",
    ready: "已連線",
    configure: "設定模型",
    retry: "重試",
    noKey: "此端點不需要 API 金鑰",
    coverage: "僅檢查模型清單與驗證，不傳送推論請求。",
    saveUse: "儲存並使用",
    cancel: "取消",
    details: "詳細資訊",
  },
  "ja": {
    providerName: 'Semantier Model Provider',
    title: "モデルサービス",
    custom: "カスタムエンドポイント",
    customDescription: "OpenAI互換のモデルURL、APIキー、モデルを設定します。Agentはローカルで動作します。",
    cloudDescription: "ログインしてSemantierモデルとアカウント残高を利用します。",
    signIn: "Semantier Model Providerにログイン",
    advancedRuntime: "詳細 · Agentの実行環境",
    setup: "モデルサービスに接続してください",
    plan: "Semantier Model Providerのプランを選択",
    preparing: "モデルアクセスを準備中",
    expired: "モデルアクセスを更新",
    credit: "残高を追加",
    reauth: "再ログイン",
    unavailable: "接続できません",
    denied: "このアカウントはアクセスできません",
    ready: "接続済み",
    configure: "モデルを設定",
    retry: "再試行",
    noKey: "このエンドポイントはAPIキー不要",
    coverage: "モデル一覧と認証のみ確認します。推論リクエストは送信しません。",
    saveUse: "保存して使用",
    cancel: "キャンセル",
    details: "詳細",
  },
  "fr": {
    providerName: 'Semantier Model Provider',
    title: "Service de modèles",
    custom: "Point de terminaison personnalisé",
    customDescription: "Configurez une URL de modèle compatible OpenAI, une clé API et un modèle. L’agent reste local.",
    cloudDescription: "Connectez-vous pour utiliser vos modèles Semantier et votre crédit.",
    signIn: "Se connecter à Semantier Model Provider",
    advancedRuntime: "Avancé · Exécution de l’agent",
    setup: "Connectez un service de modèles",
    plan: "Choisissez un forfait Semantier Model Provider",
    preparing: "Préparation de l’accès aux modèles",
    expired: "Renouveler l’accès",
    credit: "Ajouter du crédit",
    reauth: "Se reconnecter",
    unavailable: "Connexion indisponible",
    denied: "Accès refusé pour ce compte",
    ready: "Connecté",
    configure: "Configurer le modèle",
    retry: "Réessayer",
    noKey: "Ce point de terminaison ne nécessite pas de clé API",
    coverage: "Vérifie uniquement le catalogue et l’authentification. Aucune inférence.",
    saveUse: "Enregistrer et utiliser",
    cancel: "Annuler",
    details: "Détails",
  },
  "de": {
    providerName: 'Semantier Model Provider',
    title: "Modelldienst",
    custom: "Benutzerdefinierter Endpunkt",
    customDescription: "OpenAI-kompatible Modell-URL, API-Schlüssel und Modell einrichten. Der Agent läuft weiterhin lokal.",
    cloudDescription: "Anmelden, um Semantier-Modelle und Kontoguthaben zu nutzen.",
    signIn: "Bei Semantier Model Provider anmelden",
    advancedRuntime: "Erweitert · Agent-Laufzeit",
    setup: "Einen Modelldienst verbinden",
    plan: "Semantier-Cloud-Tarif auswählen",
    preparing: "Modellzugriff wird eingerichtet",
    expired: "Modellzugriff verlängern",
    credit: "Guthaben aufladen",
    reauth: "Erneut anmelden",
    unavailable: "Verbindung nicht verfügbar",
    denied: "Zugriff für dieses Konto verweigert",
    ready: "Verbunden",
    configure: "Modell einrichten",
    retry: "Erneut versuchen",
    noKey: "Dieser Endpunkt benötigt keinen API-Schlüssel",
    coverage: "Prüft nur Modellkatalog und Authentifizierung. Keine Inferenzanfrage.",
    saveUse: "Speichern und verwenden",
    cancel: "Abbrechen",
    details: "Details",
  },
  "es": {
    providerName: 'Semantier Model Provider',
    title: "Servicio de modelos",
    custom: "Endpoint personalizado",
    customDescription: "Configura una URL compatible con OpenAI, clave API y modelo. El agente sigue ejecutándose localmente.",
    cloudDescription: "Inicia sesión para usar tus modelos Semantier y el saldo de tu cuenta.",
    signIn: "Iniciar sesión en Semantier Model Provider",
    advancedRuntime: "Avanzado · Ejecución del agente",
    setup: "Conecta un servicio de modelos",
    plan: "Elige un plan de Semantier Model Provider",
    preparing: "Preparando el acceso a modelos",
    expired: "Renovar el acceso",
    credit: "Añadir saldo",
    reauth: "Volver a iniciar sesión",
    unavailable: "Conexión no disponible",
    denied: "Acceso denegado para esta cuenta",
    ready: "Conectado",
    configure: "Configurar modelo",
    retry: "Reintentar",
    noKey: "Este endpoint no necesita clave API",
    coverage: "Solo comprueba el catálogo y la autenticación. No envía solicitudes de inferencia.",
    saveUse: "Guardar y usar",
    cancel: "Cancelar",
    details: "Detalles",
  },
  "ru": {
    providerName: 'Semantier Model Provider',
    title: "Сервис моделей",
    custom: "Свой эндпоинт",
    customDescription: "Укажите совместимый с OpenAI URL, API-ключ и модель. Агент продолжит работать локально.",
    cloudDescription: "Войдите для доступа к моделям Semantier и средствам аккаунта.",
    signIn: "Войти в Semantier Model Provider",
    advancedRuntime: "Дополнительно · Среда агента",
    setup: "Подключите сервис моделей",
    plan: "Выберите тариф Semantier Model Provider",
    preparing: "Подготовка доступа к моделям",
    expired: "Продлить доступ",
    credit: "Пополнить баланс",
    reauth: "Войти снова",
    unavailable: "Подключение недоступно",
    denied: "Нет доступа к этому аккаунту",
    ready: "Подключено",
    configure: "Настроить модель",
    retry: "Повторить",
    noKey: "Эндпоинт не требует API-ключа",
    coverage: "Проверяются только каталог и авторизация. Запросы к модели не отправляются.",
    saveUse: "Сохранить и использовать",
    cancel: "Отмена",
    details: "Подробности",
  },
  "ar": {
    providerName: 'Semantier Model Provider',
    title: "خدمة النماذج",
    custom: "نقطة نهاية مخصصة",
    customDescription: "أدخل عنوان نموذج متوافق مع OpenAI ومفتاح API والنموذج. يبقى الوكيل محلياً.",
    cloudDescription: "سجّل الدخول لاستخدام نماذج Semantier ورصيد حسابك.",
    signIn: "تسجيل الدخول إلى Semantier Model Provider",
    advancedRuntime: "متقدم · بيئة تشغيل الوكيل",
    setup: "اربط خدمة نماذج للبدء",
    plan: "اختر خطة Semantier Model Provider",
    preparing: "جارٍ إعداد الوصول للنماذج",
    expired: "تجديد الوصول للنماذج",
    credit: "إضافة رصيد",
    reauth: "تسجيل الدخول مجدداً",
    unavailable: "الاتصال غير متاح",
    denied: "الوصول مرفوض لهذا الحساب",
    ready: "متصل",
    configure: "إعداد النموذج",
    retry: "إعادة المحاولة",
    noKey: "لا تتطلب نقطة النهاية مفتاح API",
    coverage: "يفحص قائمة النماذج والمصادقة فقط، دون إرسال طلب استدلال.",
    saveUse: "حفظ واستخدام",
    cancel: "إلغاء",
    details: "التفاصيل",
  },
} satisfies Record<BundledLocale, ModelServiceCopy>

/** Managed service presentation; wire/provider identity and custom names stay intact. */
export function modelProviderName(provider: { slug?: string; id?: string; name: string }, locale = 'en'): string {
  return (provider.slug ?? provider.id) === 'semantier-cloud'
    ? (modelServiceCopy[locale] ?? modelServiceCopy.en).providerName
    : provider.name
}

const providerJourneyMessages = {
  en: { add: 'Add Provider', listUrl: 'Model List API URL (optional)', enabled: 'Enabled',
    manual: 'Manual model ID · availability unverified', unverified: 'Unverified', enterModel: 'Enter model ID',
    disabled: 'Disabled', keySaved: 'Key saved', noKey: 'No key', test: 'Test Model',
    cost: 'Sends one small inference request through the selected provider. Tokens and charges may apply. Your conversation is preserved.',
    observed: 'Model response observed.' },
  zh: { add: '添加服务商', listUrl: '模型列表 API URL（可选）', enabled: '启用服务商',
    manual: '手动输入 · 模型可用性尚未验证', unverified: '未验证', enterModel: '手动输入模型 ID',
    disabled: '已停用', keySaved: '密钥已保存', noKey: '无密钥', test: '测试模型',
    cost: '将通过所选服务商发送一次小型推理请求，可能消耗 Token 并产生费用。不会修改当前会话。',
    observed: '已观察到模型回复。' },
  'zh-hant': { add: '新增供應商', listUrl: '模型清單 API URL（選填）', enabled: '啟用供應商',
    manual: '手動輸入 · 模型可用性尚未驗證', unverified: '未驗證', enterModel: '手動輸入模型 ID',
    disabled: '已停用', keySaved: '金鑰已儲存', noKey: '無金鑰', test: '測試模型',
    cost: '將透過所選供應商傳送一次小型推論請求，可能消耗 Token 並產生費用。不會修改目前對話。',
    observed: '已觀察到模型回覆。' }
}

export function providerJourneyCopy(locale: string) {
  return providerJourneyMessages[locale as keyof typeof providerJourneyMessages] ?? providerJourneyMessages.en
}

export function providerTestStatus(locale: string, status: string): string {
  const english: Record<string, string> = {
    catalog_verified: 'Model catalog verified. Inference has not been tested.',
    empty: 'The model catalog is empty. You can enter a model ID manually.',
    invalid_key: 'The provider rejected the API key (401).', forbidden: 'The provider denied this request (403).',
    unsupported: 'No supported model catalog. Enter a model ID manually.',
    rate_limited: 'The provider rate limited the request (429). Try again later.',
    timeout: 'The provider request timed out.', unreachable: 'The provider could not be reached.',
    invalid_url: 'Invalid or blocked provider URL.', model_unavailable: 'The provider rejected this model (404).',
    empty_response: 'The provider returned no model response.', response_observed: 'Model response observed.',
    error: 'The provider request failed. Check the provider configuration and retry.',
    unknown: 'The provider result could not be verified.', unavailable: 'This provider route is unavailable.'
  }

  const chinese: Record<string, string> = {
    catalog_verified: '模型列表已验证；尚未测试推理。', empty: '模型列表为空，仍可手动输入模型 ID。',
    invalid_key: '服务商拒绝了 API 密钥（401）。', forbidden: '服务商拒绝了此请求（403）。',
    unsupported: '无可用的模型列表 API，请手动输入模型 ID。', rate_limited: '服务商限制了请求频率（429），请稍后重试。',
    timeout: '服务商请求超时。', unreachable: '无法连接到服务商。', invalid_url: '服务商 URL 无效或被网络安全规则阻止。',
    model_unavailable: '服务商不支持此模型（404）。', empty_response: '服务商未返回模型回复。',
    response_observed: '已观察到模型回复。', error: '服务商请求失败，请检查配置后重试。',
    unknown: '无法验证服务商结果。', unavailable: '此服务商路由不可用。'
  }

  const traditional: Record<string, string> = {
    catalog_verified: '模型清單已驗證；尚未測試推論。', empty: '模型清單為空，仍可手動輸入模型 ID。',
    invalid_key: '供應商拒絕了 API 金鑰（401）。', forbidden: '供應商拒絕了此請求（403）。',
    unsupported: '無可用的模型清單 API，請手動輸入模型 ID。', rate_limited: '供應商限制了請求頻率（429），請稍後重試。',
    timeout: '供應商請求逾時。', unreachable: '無法連線至供應商。', invalid_url: '供應商 URL 無效或遭網路安全規則阻擋。',
    model_unavailable: '供應商不支援此模型（404）。', empty_response: '供應商未傳回模型回覆。',
    response_observed: '已觀察到模型回覆。', error: '供應商請求失敗，請檢查設定後重試。',
    unknown: '無法驗證供應商結果。', unavailable: '此供應商路由無法使用。'
  }

  const messages = ({ zh: chinese, 'zh-hant': traditional } as Record<string, Record<string, string>>)[locale] ?? english

  return messages[status] ?? messages.unknown
}
