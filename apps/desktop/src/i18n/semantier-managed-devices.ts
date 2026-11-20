const en = {
  title: 'Windows PCs',
  included: 'Two Windows PCs are included. Web and Winman do not use PC seats.',
  view: 'View Windows PCs',
  refresh: 'Refresh PCs',
  loading: 'Loading PCs…',
  empty: 'No managed Windows PCs are registered.',
  failed: 'Could not update the device list. Refresh to check the current server status.',
  remove: 'Remove PC',
  confirm: 'Confirm removal',
  cancel: 'Cancel',
  question: 'Remove this PC’s managed access?',
  active: 'Active',
  revoked: 'Revoked',
  verified: 'TPM verified by server',
  unknown: 'Hardware not verified',
  seen: 'Last seen',
  checkout: 'Additional-seat checkout is unavailable.',
  local: 'Local Hermes remains available without managed enrollment.',
  enrollment: 'Enrollment is unavailable until hardware attestation can be verified.'
}

const zh: typeof en = {
  title: 'Windows 电脑',
  included: '账户包含两台 Windows 电脑。网页和僚机不占用电脑名额。',
  view: '查看 Windows 电脑',
  refresh: '刷新电脑列表',
  loading: '正在加载电脑…',
  empty: '尚未注册托管 Windows 电脑。',
  failed: '无法更新设备列表。请刷新以查看服务器上的当前状态。',
  remove: '移除电脑',
  confirm: '确认移除',
  cancel: '取消',
  question: '移除这台电脑的托管访问权限？',
  active: '有效',
  revoked: '已撤销',
  verified: '服务器已验证 TPM',
  unknown: '硬件未验证',
  seen: '最近在线',
  checkout: '暂不支持购买额外电脑名额。',
  local: '无需托管注册，仍可使用本地 Hermes。',
  enrollment: '硬件证明可验证后才能注册。'
}

const hant: typeof en = {
  title: 'Windows 電腦',
  included: '帳戶包含兩台 Windows 電腦。網頁和僚機不佔用電腦名額。',
  view: '查看 Windows 電腦',
  refresh: '重新整理電腦清單',
  loading: '正在載入電腦…',
  empty: '尚未註冊受管 Windows 電腦。',
  failed: '無法更新裝置清單。請重新整理以查看伺服器上的目前狀態。',
  remove: '移除電腦',
  confirm: '確認移除',
  cancel: '取消',
  question: '移除這台電腦的受管存取權限？',
  active: '有效',
  revoked: '已撤銷',
  verified: '伺服器已驗證 TPM',
  unknown: '硬體未驗證',
  seen: '最近上線',
  checkout: '暫不支援購買額外電腦名額。',
  local: '無需受管註冊，仍可使用本機 Hermes。',
  enrollment: '硬體證明可驗證後才能註冊。'
}

export function managedDeviceCopy(locale: string): typeof en {
  return /^zh[-_]hant|^zh[-_](tw|hk)/i.test(locale) ? hant : /^zh/i.test(locale) ? zh : en
}
