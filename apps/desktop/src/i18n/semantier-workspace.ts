const en = {
  title: 'Semantier Cloud',
  description: 'Open the same conversations and Winman as Hermes Workspace.',
  address: 'Workspace address',
  signIn: 'Sign in and open workspace',
  open: 'Open workspace',
  signOut: 'Sign out',
  signedIn: 'Signed in',
  checking: 'Checking account…',
  signedOut: 'Sign in to recover your workspace.',
  waiting: 'Connecting…',
  retry: 'Retry',
  unavailable: 'Workspace identity is unavailable. Try again.',
  forbidden: 'Workspace access is denied for this account or organization.',
  failed: 'Could not open the workspace. Check the address and try again.',
  other: 'Other cloud agents',
  identityRequired: 'Sign-in did not return an authenticated workspace.',
  accountTitle: 'Semantier Account',
  accountDescription: 'Sign in to use your account without changing the agent runtime.',
  accountSignIn: 'Sign in',
  accountSignedOut: 'Sign in to use eligible account services.',
  accountRetry: 'Check account'
}

const zh: typeof en = {
  title: 'Semantier Cloud',
  description: '打开与 Hermes Workspace 相同的会话和僚机。',
  address: '工作区地址',
  signIn: '登录并打开工作区',
  open: '打开工作区',
  signOut: '退出登录',
  signedIn: '已登录',
  checking: '正在检查账户…',
  signedOut: '登录以恢复你的工作区。',
  waiting: '正在连接…',
  retry: '重试',
  unavailable: '暂时无法验证工作区身份，请重试。',
  forbidden: '此账户或组织无权访问工作区。',
  failed: '无法打开工作区。请检查地址后重试。',
  other: '其他云端智能体',
  identityRequired: '登录未返回已认证的工作区。',
  accountTitle: 'Semantier 账户',
  accountDescription: '登录账户不会更改智能体的运行位置。',
  accountSignIn: '登录',
  accountSignedOut: '登录后可使用账户有权使用的服务。',
  accountRetry: '检查账户'
}

const hant: typeof en = {
  title: 'Semantier Cloud',
  description: '開啟與 Hermes Workspace 相同的會話和僚機。',
  address: '工作區地址',
  signIn: '登入並開啟工作區',
  open: '開啟工作區',
  signOut: '登出',
  signedIn: '已登入',
  checking: '正在檢查帳戶…',
  signedOut: '登入以恢復你的工作區。',
  waiting: '正在連線…',
  retry: '重試',
  unavailable: '暫時無法驗證工作區身分，請重試。',
  forbidden: '此帳戶或組織無權存取工作區。',
  failed: '無法開啟工作區。請檢查地址後重試。',
  other: '其他雲端智能體',
  identityRequired: '登入未傳回已驗證的工作區。',
  accountTitle: 'Semantier 帳戶',
  accountDescription: '登入帳戶不會更改智能體的執行位置。',
  accountSignIn: '登入',
  accountSignedOut: '登入後可使用帳戶有權使用的服務。',
  accountRetry: '檢查帳戶'
}

export function semantierWorkspaceCopy(locale: string): typeof en {
  return /^zh[-_]hant|^zh[-_](tw|hk)/i.test(locale) ? hant : /^zh/i.test(locale) ? zh : en
}
