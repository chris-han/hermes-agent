import { readKey, writeKey } from '@/lib/storage'

export const marketplaceSourceKey = (scope: string) => `hermes.desktop.marketplaceSource.v1:${scope}`
export const readMarketplaceSource = (scope: string) => readKey(marketplaceSourceKey(scope)) ?? 'semantier'
export const saveMarketplaceSource = (scope: string, value: string) => writeKey(marketplaceSourceKey(scope), value)

export function moveMarketplaceSource(from: string, to?: string) {
  const saved = readKey(marketplaceSourceKey(from))

  if (saved !== null && to && readKey(marketplaceSourceKey(to)) === null) {
    writeKey(marketplaceSourceKey(to), saved)
  }

  writeKey(marketplaceSourceKey(from), null)
}

export const marketplaceSources = [
  { value: 'semantier', label: 'Semantier Official' },
  { value: 'hermes-index', label: 'Hermes Index' },
  { value: 'official', label: 'Nous Official' },
  { value: 'browse-sh', label: 'browse.sh' },
  { value: 'url', label: 'Direct URL' }
]
