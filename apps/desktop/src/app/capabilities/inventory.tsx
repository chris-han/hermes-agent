import { useStore } from '@nanostores/react'
import { type ComponentProps, createContext, type ReactNode, useContext, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { useInventoryCopy } from '@/i18n/capability-inventory'
import { Codecs, persistentAtom } from '@/lib/persisted'
import { cn } from '@/lib/utils'

import { CapRow } from '../master-detail'

const ViewContext = createContext(false)
const DetailContext = createContext<() => void>(() => {})
const preferences = new Map<string, ReturnType<typeof preference>>()

const preference = (tab: string) =>
  persistentAtom(`hermes.desktop.capabilities.${tab}.cards`, tab === 'marketplace', Codecs.bool)

function preferenceFor(tab: string) {
  let atom = preferences.get(tab)

  if (!atom) {
    atom = preference(tab)
    preferences.set(tab, atom)
  }

  return atom
}

export function CapabilityInventory({ tab, children }: { tab: string; children: ReactNode }) {
  const atom = preferenceFor(tab)
  const cards = useStore(atom)
  const c = useInventoryCopy()
  const [detailOpen, setDetailOpen] = useState(false)
  const returnFocus = useRef<HTMLElement | null>(null)
  const inventoryRoot = useRef<HTMLDivElement>(null)

  return (
    <ViewContext.Provider value={cards}>
      <DetailContext.Provider
        value={() => {
          returnFocus.current =
            window.document.activeElement instanceof HTMLElement ? window.document.activeElement : null
          setDetailOpen(true)
        }}
      >
        <div
          className="flex h-full min-h-0 flex-col"
          data-capability-inventory
          data-detail-open={detailOpen}
          data-view={cards ? 'cards' : 'list'}
          ref={inventoryRoot}
        >
          <div aria-label={c.view} className="flex shrink-0 justify-end gap-1 px-3 py-1" role="group">
            <SegmentedControl
              animated
              onChange={view => atom.set(view === 'cards')}
              options={[{ id: 'list', label: c.list }, { id: 'cards', label: c.cards }]}
              value={cards ? 'cards' : 'list'}
            />
          </div>
          {detailOpen && (
            <Button
              className="min-[721px]:hidden"
              onClick={() => {
                setDetailOpen(false)
                requestAnimationFrame(() => {
                  const target = returnFocus.current?.isConnected
                    ? returnFocus.current
                    : inventoryRoot.current?.querySelector<HTMLElement>('aside button')

                  target?.focus()
                })
              }}
              size="xs"
            >
              {c.back}
            </Button>
          )}
          <div className="min-h-0 flex-1">{children}</div>
        </div>
      </DetailContext.Provider>
    </ViewContext.Provider>
  )
}

export function useInventoryCards() {
  return useContext(ViewContext)
}

export function InventoryItems({ children }: { children: ReactNode }) {
  const cards = useInventoryCards()

  return (
    <div
      className={cards ? 'grid grid-cols-1 gap-2 min-[1100px]:grid-cols-2' : 'flex flex-col'}
      data-testid="inventory-items"
      data-view={cards ? 'cards' : 'list'}
    >
      {children}
    </div>
  )
}

export function InventoryRow({ description, ...props }: ComponentProps<typeof CapRow> & { description?: ReactNode }) {
  const cards = useInventoryCards()
  const openDetail = useContext(DetailContext)

  const select = () => {
    props.onSelect()
    openDetail()
  }

  // List rows need the subtitle for scanability; cards already render the
  // description in a bounded preview. Do not repeat an identical, unbounded
  // subtitle under it (e.g. the A2A toolset's multi-paragraph description).
  const cardSubtitle = typeof description === 'string' && props.subtitle === description ? null : props.subtitle

  if (!cards) {
    return <CapRow {...props} onSelect={select} />
  }

  return (
    <div
      className={cn(
        'flex min-w-0 flex-col gap-3 rounded-lg border border-(--ui-stroke-secondary) p-3 [content-visibility:auto] [contain-intrinsic-size:auto_112px]',
        props.active && 'bg-(--ui-row-active-background)'
      )}
    >
      <button
        aria-label={props.title}
        aria-pressed={props.active}
        className="min-w-0 flex-1 cursor-pointer text-left focus-visible:outline-2 focus-visible:outline-primary"
        onClick={select}
        type="button"
      >
        <span className="block break-words text-sm font-medium">{props.title}</span>
        {description && (
          <span className="mt-1 block line-clamp-2 break-words text-xs text-(--ui-text-tertiary)">{description}</span>
        )}
        {cardSubtitle != null && (
          <span className="mt-1 flex min-w-0 flex-wrap gap-1 text-xs text-(--ui-text-tertiary)">{cardSubtitle}</span>
        )}
      </button>
      <div className="flex items-center justify-between gap-2 text-xs text-(--ui-text-tertiary)">
        <span>{props.meta}</span>
        {props.action ??
          (props.onToggle && (
            <Switch
              aria-label={props.toggleLabel ?? props.title}
              checked={props.enabled}
              disabled={props.busy}
              onCheckedChange={props.onToggle}
              size="xs"
            />
          ))}
      </div>
    </div>
  )
}

export function InventorySelect({
  label,
  value,
  onChange,
  options
}: {
  label: string
  value: string
  onChange: (value: string) => void
  options: { value: string; label: string }[]
}) {
  return (
    <Select onValueChange={onChange} value={value}>
      <SelectTrigger aria-label={label} className="w-auto max-w-full text-xs" size="sm">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {options.map(option => (
          <SelectItem key={option.value} value={option.value}>
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}
