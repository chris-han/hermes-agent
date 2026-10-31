import type { BundledLocale } from './types'

interface WindowCloseCopy {
  title: string
  question: string
  detail: string
  keep: string
  quit: string
  backgroundUnavailable: string
  unavailable: string
}

export const windowCloseCopy: Record<string, WindowCloseCopy> = {
  en: {
    title: 'Close Semantier Agent',
    question: 'Keep Semantier Agent running in the background?',
    detail:
      'Keep running hides this window in the system tray. Quit exits the app and closes connections it manages. A separately managed gateway keeps running.',
    keep: 'Keep running in background',
    quit: 'Quit Semantier Agent',
    backgroundUnavailable: 'The system tray is unavailable. The window will stay open. Try again or choose Quit.',
    unavailable: 'This close request is no longer available. Cancel and try closing the window again.'
  },
  zh: {
    title: '关闭 Semantier Agent',
    question: '是否让 Semantier Agent 继续在后台运行？',
    detail: '后台运行会将此窗口隐藏到系统托盘。退出会关闭应用及其管理的连接，独立管理的网关会继续运行。',
    keep: '在后台运行',
    quit: '退出 Semantier Agent',
    backgroundUnavailable: '系统托盘不可用，窗口将保持打开。请重试或选择退出。',
    unavailable: '此关闭请求已失效。请取消后重新关闭窗口。'
  },
  'zh-hant': {
    title: '關閉 Semantier Agent',
    question: '是否讓 Semantier Agent 繼續在背景執行？',
    detail: '背景執行會將此視窗隱藏至系統匣。結束會關閉應用程式及其管理的連線，獨立管理的閘道仍會繼續執行。',
    keep: '在背景執行',
    quit: '結束 Semantier Agent',
    backgroundUnavailable: '系統匣無法使用，視窗將保持開啟。請重試或選擇結束。',
    unavailable: '此關閉要求已失效。請取消後重新關閉視窗。'
  },
  ja: {
    title: 'Semantier Agent を閉じる',
    question: 'Semantier Agent をバックグラウンドで実行し続けますか？',
    detail:
      'バックグラウンド実行ではウィンドウがシステムトレイに隠れます。終了するとアプリと管理中の接続が閉じます。別途管理されるゲートウェイは実行を続けます。',
    keep: 'バックグラウンドで実行',
    quit: 'Semantier Agent を終了',
    backgroundUnavailable: 'システムトレイを利用できません。ウィンドウは開いたままです。再試行するか終了してください。',
    unavailable: 'この終了要求は無効になりました。キャンセルして、もう一度閉じてください。'
  },
  ar: {
    title: 'إغلاق Semantier Agent',
    question: 'هل تريد إبقاء Semantier Agent قيد التشغيل في الخلفية؟',
    detail:
      'يخفي التشغيل في الخلفية النافذة في علبة النظام. ينهي الخروج التطبيق واتصالاته المُدارة. تستمر البوابة المُدارة بشكل مستقل في العمل.',
    keep: 'التشغيل في الخلفية',
    quit: 'إنهاء Semantier Agent',
    backgroundUnavailable: 'علبة النظام غير متاحة. ستبقى النافذة مفتوحة. أعد المحاولة أو اختر الخروج.',
    unavailable: 'لم يعد طلب الإغلاق متاحًا. ألغِ الطلب وحاول إغلاق النافذة مجددًا.'
  },
  ru: {
    title: 'Закрыть Semantier Agent',
    question: 'Оставить Semantier Agent работать в фоне?',
    detail:
      'Фоновый режим скрывает окно в системном трее. Выход завершает приложение и управляемые им подключения. Независимый шлюз продолжит работать.',
    keep: 'Оставить в фоне',
    quit: 'Выйти из Semantier Agent',
    backgroundUnavailable: 'Системный трей недоступен. Окно останется открытым. Повторите попытку или выберите выход.',
    unavailable: 'Этот запрос закрытия устарел. Отмените его и закройте окно снова.'
  },
  fr: {
    title: 'Fermer Semantier Agent',
    question: 'Garder Semantier Agent actif en arrière-plan ?',
    detail:
      'Le mode arrière-plan masque la fenêtre dans la zone de notification. Quitter ferme l’application et les connexions qu’elle gère. Une passerelle gérée séparément continue de fonctionner.',
    keep: 'Garder en arrière-plan',
    quit: 'Quitter Semantier Agent',
    backgroundUnavailable:
      'La zone de notification est indisponible. La fenêtre reste ouverte. Réessayez ou choisissez Quitter.',
    unavailable: 'Cette demande de fermeture a expiré. Annulez et fermez à nouveau la fenêtre.'
  },
  de: {
    title: 'Semantier Agent schließen',
    question: 'Semantier Agent im Hintergrund weiterlaufen lassen?',
    detail:
      'Der Hintergrundbetrieb verbirgt das Fenster im Infobereich. Beenden schließt die App und ihre verwalteten Verbindungen. Ein separat verwaltetes Gateway läuft weiter.',
    keep: 'Im Hintergrund weiterlaufen',
    quit: 'Semantier Agent beenden',
    backgroundUnavailable:
      'Der Infobereich ist nicht verfügbar. Das Fenster bleibt geöffnet. Erneut versuchen oder Beenden wählen.',
    unavailable: 'Diese Schließanfrage ist nicht mehr verfügbar. Abbrechen und das Fenster erneut schließen.'
  },
  es: {
    title: 'Cerrar Semantier Agent',
    question: '¿Mantener Semantier Agent en segundo plano?',
    detail:
      'El modo en segundo plano oculta la ventana en la bandeja del sistema. Salir cierra la aplicación y las conexiones que administra. Una puerta de enlace independiente sigue funcionando.',
    keep: 'Mantener en segundo plano',
    quit: 'Salir de Semantier Agent',
    backgroundUnavailable:
      'La bandeja del sistema no está disponible. La ventana seguirá abierta. Reintenta o elige Salir.',
    unavailable: 'Esta solicitud de cierre ya no está disponible. Cancela y vuelve a cerrar la ventana.'
  }
} satisfies Record<BundledLocale, WindowCloseCopy>
