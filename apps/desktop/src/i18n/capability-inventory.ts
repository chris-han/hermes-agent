import { useI18n } from './context'

const en = {
  archiveTitle: (name: string) => `Archive ${name}?`,
  archiveDescription: 'The skill is archived and can be restored with `hermes curator restore`.',
  list: 'List',
  cards: 'Cards',
  view: 'Inventory view',
  learned: 'Learned Skills',
  learnedItem: 'Learned Skill',
  other: 'Other Installed Skills',
  marketplace: 'Skill Marketplace',
  all: 'All sources',
  enabled: 'Enabled',
  disabled: 'Disabled',
  status: 'All states',
  name: 'Name',
  usage: 'Usage',
  source: 'Source',
  category: 'All categories',
  unavailable: 'Catalog unavailable. Installed capabilities remain available.',
  retry: 'Retry',
  back: 'Back to inventory',
  partial: 'Some sources did not respond.',
  builtin: 'Built-in',
  external: 'External',
  hub: 'Marketplace Source'
}

const zh: typeof en = {
  archiveTitle: (name: string) => `归档 ${name}？`,
  archiveDescription: '归档后可通过 `hermes curator restore` 恢复此技能。',
  list: '列表',
  cards: '卡片',
  view: '库存视图',
  learned: '习得技能',
  learnedItem: '习得技能',
  other: '其他已安装技能',
  marketplace: '技能市场',
  all: '全部来源',
  enabled: '已启用',
  disabled: '已停用',
  status: '全部状态',
  name: '名称',
  usage: '使用次数',
  source: '来源',
  category: '全部分类',
  unavailable: '市场暂不可用，已安装的技能与工具仍可管理。',
  retry: '重试',
  back: '返回列表',
  partial: '部分来源未响应。',
  builtin: '内置技能',
  external: '外部技能',
  hub: '市场来源'
}

const hant: typeof en = {
  ...zh,
  archiveTitle: (name: string) => `封存 ${name}？`,
  archiveDescription: '封存後可透過 `hermes curator restore` 還原此技能。',
  list: '列表',
  cards: '卡片',
  learned: '習得技能',
  learnedItem: '習得技能',
  other: '其他已安裝技能',
  marketplace: '技能市場',
  all: '全部來源',
  enabled: '已啟用',
  disabled: '已停用',
  status: '全部狀態',
  name: '名稱',
  usage: '使用次數',
  source: '來源',
  category: '全部分類',
  unavailable: '市場暫不可用，已安裝的技能與工具仍可管理。',
  retry: '重試',
  back: '返回列表',
  partial: '部分來源未回應。',
  builtin: '內建技能',
  external: '外部技能',
  hub: '市場來源'
}

export function useInventoryCopy() {
  const { locale } = useI18n()

  return locale === 'zh-Hant' ? hant : locale.startsWith('zh') ? zh : en
}
