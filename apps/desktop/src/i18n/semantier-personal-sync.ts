const en = {
  title: 'Personal Skills',
  intro:
    'Share one Skill and one selected memory entry from this PC’s default local profile. DEVELOPMENT sync uses your signed-in account, without device enrollment.',
  name: 'Local Skill name',
  preview: 'Preview local source',
  context: 'Select one decision-context entry',
  share: 'Share this exact source',
  sources: 'Refresh personal sources',
  empty: 'No personal source revisions yet.',
  prepare: 'Review this revision',
  enable: 'Enable in Cloud',
  install: 'Install approved revision on this PC',
  replace: 'Replace the exact previous personal revision',
  pending: 'Working…',
  failed: 'Could not complete this step. Your retained source is preserved; refresh and retry.',
  shared: 'Source saved in Cloud. Review it before enabling.',
  enabled: 'Cloud admission completed. Normal native package checks still apply.',
  installed: 'Installed for the next session.',
  review:
    'Review the exact Skill and context below. Enabling uses the normal cloud assessment and publication checks; installation grants no premium plugin entitlement.',
  noContext: 'No selectable memory entries. Add a personal context entry in local Memory, then preview again.',
  files: 'Selected files',
  more: 'More source history exists.',
  assurance: 'Hardware: NOT_VERIFIED · No purchase or premium entitlement is granted.'
}

const zh: typeof en = {
  title: '个人技能',
  intro: '从本机默认本地档案分享一个技能及一条选定的记忆。开发环境同步使用已登录账户，无需设备注册。',
  name: '本地技能名称',
  preview: '预览本地来源',
  context: '选择一条决策上下文',
  share: '分享此精确来源',
  sources: '刷新个人来源',
  empty: '暂无个人来源版本。',
  prepare: '审阅此版本',
  enable: '在云端启用',
  install: '在本机安装已批准版本',
  replace: '替换精确的上一版个人技能',
  pending: '处理中…',
  failed: '此步骤未完成。已保留来源，请刷新并重试。',
  shared: '来源已保存至云端，请审阅后启用。',
  enabled: '云端准入已完成，仍须通过正常的本地包检查。',
  installed: '已安装，下次会话可用。',
  review: '请审阅下列精确技能和上下文。启用使用正常的云端评估与发布检查；安装不授予付费插件权益。',
  noContext: '没有可选记忆。请先在本地记忆中添加个人上下文，再次预览。',
  files: '选定文件',
  more: '还有更多来源历史。',
  assurance: '硬件：NOT_VERIFIED · 不产生购买或付费权益。'
}

const hant: typeof en = {
  title: '個人技能',
  intro: '從本機預設本機設定檔分享一個技能及一條選定記憶。開發環境同步使用已登入帳戶，無需裝置註冊。',
  name: '本機技能名稱',
  preview: '預覽本機來源',
  context: '選擇一條決策上下文',
  share: '分享此精確來源',
  sources: '重新整理個人來源',
  empty: '尚無個人來源版本。',
  prepare: '審閱此版本',
  enable: '在雲端啟用',
  install: '在本機安裝已核准版本',
  replace: '取代精確的上一版個人技能',
  pending: '處理中…',
  failed: '此步驟未完成。來源已保留，請重新整理並重試。',
  shared: '來源已儲存至雲端，請審閱後啟用。',
  enabled: '雲端准入已完成，仍須通過正常本機套件檢查。',
  installed: '已安裝，下次工作階段可用。',
  review: '請審閱下列精確技能和上下文。啟用使用正常雲端評估與發佈檢查；安裝不授予付費外掛權益。',
  noContext: '沒有可選記憶。請先在本機記憶中新增個人上下文，再次預覽。',
  files: '選定檔案',
  more: '還有更多來源歷史。',
  assurance: '硬體：NOT_VERIFIED · 不產生購買或付費權益。'
}

export function personalSyncCopy(locale: string) {
  return locale === 'zh-Hant' ? hant : locale.startsWith('zh') ? zh : en
}
