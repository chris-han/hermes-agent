import type { BundledLocale } from './types'

export interface ModelServiceCopy {
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
  en: {
    title: 'Model service',
    custom: 'Custom endpoint',
    customDescription: 'Use an OpenAI-compatible model URL, API key and model. The agent keeps running locally.',
    cloudDescription: 'Sign in to use your Semantier models and account credit.',
    signIn: 'Sign in to Semantier Cloud',
    advancedRuntime: 'Advanced · Agent runtime',
    setup: 'Connect a model service to start',
    plan: 'Choose a plan to use Semantier Cloud',
    preparing: 'Setting up model access',
    expired: 'Renew model access',
    credit: 'Add credit',
    reauth: 'Sign in again',
    unavailable: 'Connection unavailable',
    denied: 'Access denied for this account',
    ready: 'Connected',
    configure: 'Configure model',
    retry: 'Retry',
    noKey: 'This endpoint requires no API key',
    coverage: 'Checks model discovery and authentication only. No inference request is sent.',
    saveUse: 'Save and use',
    cancel: 'Cancel',
    details: 'Details'
  },
  zh: {
    title: '模型服务',
    custom: '自定义端点',
    customDescription: '填写兼容 OpenAI 的模型服务地址、API 密钥和模型。Agent 仍在本地运行。',
    cloudDescription: '登录后使用 Semantier 模型与账户额度。',
    signIn: '登录 Semantier Cloud',
    advancedRuntime: '高级 · Agent 运行位置',
    setup: '连接模型服务即可开始',
    plan: '选择套餐以使用 Semantier Cloud',
    preparing: '正在开通模型服务',
    expired: '续订模型服务',
    credit: '充值',
    reauth: '重新登录',
    unavailable: '暂时无法连接',
    denied: '此账户无权访问',
    ready: '已连接',
    configure: '配置模型',
    retry: '重试',
    noKey: '此端点无需 API 密钥',
    coverage: '仅检查模型列表与身份验证，不发送推理请求。',
    saveUse: '保存并使用',
    cancel: '取消',
    details: '详细信息'
  },
  'zh-hant': {
    title: '模型服務',
    custom: '自訂端點',
    customDescription: '填寫相容 OpenAI 的模型服務網址、API 金鑰與模型。Agent 仍在本機執行。',
    cloudDescription: '登入後使用 Semantier 模型與帳戶額度。',
    signIn: '登入 Semantier Cloud',
    advancedRuntime: '進階 · Agent 執行位置',
    setup: '連線模型服務即可開始',
    plan: '選擇方案以使用 Semantier Cloud',
    preparing: '正在開通模型服務',
    expired: '續訂模型服務',
    credit: '儲值',
    reauth: '重新登入',
    unavailable: '暫時無法連線',
    denied: '此帳戶無權存取',
    ready: '已連線',
    configure: '設定模型',
    retry: '重試',
    noKey: '此端點不需要 API 金鑰',
    coverage: '僅檢查模型清單與驗證，不傳送推論請求。',
    saveUse: '儲存並使用',
    cancel: '取消',
    details: '詳細資訊'
  },
  ja: {
    title: 'モデルサービス',
    custom: 'カスタムエンドポイント',
    customDescription: 'OpenAI互換のモデルURL、APIキー、モデルを設定します。Agentはローカルで動作します。',
    cloudDescription: 'ログインしてSemantierモデルとアカウント残高を利用します。',
    signIn: 'Semantier Cloudにログイン',
    advancedRuntime: '詳細 · Agentの実行環境',
    setup: 'モデルサービスに接続してください',
    plan: 'Semantier Cloudのプランを選択',
    preparing: 'モデルアクセスを準備中',
    expired: 'モデルアクセスを更新',
    credit: '残高を追加',
    reauth: '再ログイン',
    unavailable: '接続できません',
    denied: 'このアカウントはアクセスできません',
    ready: '接続済み',
    configure: 'モデルを設定',
    retry: '再試行',
    noKey: 'このエンドポイントはAPIキー不要',
    coverage: 'モデル一覧と認証のみ確認します。推論リクエストは送信しません。',
    saveUse: '保存して使用',
    cancel: 'キャンセル',
    details: '詳細'
  },
  fr: {
    title: 'Service de modèles',
    custom: 'Point de terminaison personnalisé',
    customDescription: 'Configurez une URL de modèle compatible OpenAI, une clé API et un modèle. L’agent reste local.',
    cloudDescription: 'Connectez-vous pour utiliser vos modèles Semantier et votre crédit.',
    signIn: 'Se connecter à Semantier Cloud',
    advancedRuntime: 'Avancé · Exécution de l’agent',
    setup: 'Connectez un service de modèles',
    plan: 'Choisissez un forfait Semantier Cloud',
    preparing: 'Préparation de l’accès aux modèles',
    expired: 'Renouveler l’accès',
    credit: 'Ajouter du crédit',
    reauth: 'Se reconnecter',
    unavailable: 'Connexion indisponible',
    denied: 'Accès refusé pour ce compte',
    ready: 'Connecté',
    configure: 'Configurer le modèle',
    retry: 'Réessayer',
    noKey: 'Ce point de terminaison ne nécessite pas de clé API',
    coverage: 'Vérifie uniquement le catalogue et l’authentification. Aucune inférence.',
    saveUse: 'Enregistrer et utiliser',
    cancel: 'Annuler',
    details: 'Détails'
  },
  de: {
    title: 'Modelldienst',
    custom: 'Benutzerdefinierter Endpunkt',
    customDescription:
      'OpenAI-kompatible Modell-URL, API-Schlüssel und Modell einrichten. Der Agent läuft weiterhin lokal.',
    cloudDescription: 'Anmelden, um Semantier-Modelle und Kontoguthaben zu nutzen.',
    signIn: 'Bei Semantier Cloud anmelden',
    advancedRuntime: 'Erweitert · Agent-Laufzeit',
    setup: 'Einen Modelldienst verbinden',
    plan: 'Semantier-Cloud-Tarif auswählen',
    preparing: 'Modellzugriff wird eingerichtet',
    expired: 'Modellzugriff verlängern',
    credit: 'Guthaben aufladen',
    reauth: 'Erneut anmelden',
    unavailable: 'Verbindung nicht verfügbar',
    denied: 'Zugriff für dieses Konto verweigert',
    ready: 'Verbunden',
    configure: 'Modell einrichten',
    retry: 'Erneut versuchen',
    noKey: 'Dieser Endpunkt benötigt keinen API-Schlüssel',
    coverage: 'Prüft nur Modellkatalog und Authentifizierung. Keine Inferenzanfrage.',
    saveUse: 'Speichern und verwenden',
    cancel: 'Abbrechen',
    details: 'Details'
  },
  es: {
    title: 'Servicio de modelos',
    custom: 'Endpoint personalizado',
    customDescription:
      'Configura una URL compatible con OpenAI, clave API y modelo. El agente sigue ejecutándose localmente.',
    cloudDescription: 'Inicia sesión para usar tus modelos Semantier y el saldo de tu cuenta.',
    signIn: 'Iniciar sesión en Semantier Cloud',
    advancedRuntime: 'Avanzado · Ejecución del agente',
    setup: 'Conecta un servicio de modelos',
    plan: 'Elige un plan de Semantier Cloud',
    preparing: 'Preparando el acceso a modelos',
    expired: 'Renovar el acceso',
    credit: 'Añadir saldo',
    reauth: 'Volver a iniciar sesión',
    unavailable: 'Conexión no disponible',
    denied: 'Acceso denegado para esta cuenta',
    ready: 'Conectado',
    configure: 'Configurar modelo',
    retry: 'Reintentar',
    noKey: 'Este endpoint no necesita clave API',
    coverage: 'Solo comprueba el catálogo y la autenticación. No envía solicitudes de inferencia.',
    saveUse: 'Guardar y usar',
    cancel: 'Cancelar',
    details: 'Detalles'
  },
  ru: {
    title: 'Сервис моделей',
    custom: 'Свой эндпоинт',
    customDescription: 'Укажите совместимый с OpenAI URL, API-ключ и модель. Агент продолжит работать локально.',
    cloudDescription: 'Войдите для доступа к моделям Semantier и средствам аккаунта.',
    signIn: 'Войти в Semantier Cloud',
    advancedRuntime: 'Дополнительно · Среда агента',
    setup: 'Подключите сервис моделей',
    plan: 'Выберите тариф Semantier Cloud',
    preparing: 'Подготовка доступа к моделям',
    expired: 'Продлить доступ',
    credit: 'Пополнить баланс',
    reauth: 'Войти снова',
    unavailable: 'Подключение недоступно',
    denied: 'Нет доступа к этому аккаунту',
    ready: 'Подключено',
    configure: 'Настроить модель',
    retry: 'Повторить',
    noKey: 'Эндпоинт не требует API-ключа',
    coverage: 'Проверяются только каталог и авторизация. Запросы к модели не отправляются.',
    saveUse: 'Сохранить и использовать',
    cancel: 'Отмена',
    details: 'Подробности'
  },
  ar: {
    title: 'خدمة النماذج',
    custom: 'نقطة نهاية مخصصة',
    customDescription: 'أدخل عنوان نموذج متوافق مع OpenAI ومفتاح API والنموذج. يبقى الوكيل محلياً.',
    cloudDescription: 'سجّل الدخول لاستخدام نماذج Semantier ورصيد حسابك.',
    signIn: 'تسجيل الدخول إلى Semantier Cloud',
    advancedRuntime: 'متقدم · بيئة تشغيل الوكيل',
    setup: 'اربط خدمة نماذج للبدء',
    plan: 'اختر خطة Semantier Cloud',
    preparing: 'جارٍ إعداد الوصول للنماذج',
    expired: 'تجديد الوصول للنماذج',
    credit: 'إضافة رصيد',
    reauth: 'تسجيل الدخول مجدداً',
    unavailable: 'الاتصال غير متاح',
    denied: 'الوصول مرفوض لهذا الحساب',
    ready: 'متصل',
    configure: 'إعداد النموذج',
    retry: 'إعادة المحاولة',
    noKey: 'لا تتطلب نقطة النهاية مفتاح API',
    coverage: 'يفحص قائمة النماذج والمصادقة فقط، دون إرسال طلب استدلال.',
    saveUse: 'حفظ واستخدام',
    cancel: 'إلغاء',
    details: 'التفاصيل'
  }
} satisfies Record<BundledLocale, ModelServiceCopy>
