import { motion, useReducedMotion } from 'motion/react'
import { useId } from 'react'

import type { IconComponent } from '@/lib/icons'
import { cn } from '@/lib/utils'

export interface SegmentedControlOption<T extends string> {
  id: T
  label: string
  icon?: IconComponent
}

interface SegmentedControlProps<T extends string> {
  options: readonly SegmentedControlOption<T>[]
  value: T
  onChange: (id: T) => void
  className?: string
  /** Dims the whole track and blocks selection (e.g. gated behind a prerequisite). */
  disabled?: boolean
  animated?: boolean
}

/**
 * Grouped one-row toggle used for small mutually-exclusive choices
 * (color mode, tool-call display, usage period, etc.). Flat by design —
 * no per-option borders, just a tinted track with a raised active pill.
 */
export function SegmentedControl<T extends string>({
  animated = false,
  className,
  disabled = false,
  onChange,
  options,
  value
}: SegmentedControlProps<T>) {
  const indicatorId = useId()
  const reducedMotion = useReducedMotion()

  return (
    <div
      className={cn(
        'inline-grid w-fit auto-cols-fr grid-flow-col gap-0.5 rounded-[5px] bg-(--ui-bg-tertiary) p-0.5',
        disabled && 'opacity-50',
        className
      )}
    >
      {options.map(({ id, label, icon: Icon }) => {
        const active = value === id

        return (
          <button
            aria-pressed={active}
            className={cn(
              'relative isolate flex min-w-0 items-center justify-center gap-1 rounded-[3px] px-2.5 py-0.5 text-[0.6875rem] font-medium whitespace-nowrap transition-colors disabled:cursor-default',
              active
                ? animated
                  ? 'text-primary-foreground'
                  : 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
            )}
            disabled={disabled}
            key={id}
            onClick={() => onChange(id)}
            type="button"
          >
            {animated && active && (
              <motion.span
                aria-hidden
                className="absolute inset-0 -z-10 rounded-[inherit] bg-primary"
                data-slot="segmented-control-indicator"
                layoutId={`segmented-control-${indicatorId}`}
                transition={reducedMotion ? { duration: 0 } : { type: 'spring', stiffness: 500, damping: 35 }}
              />
            )}
            {Icon && <Icon className="size-3 shrink-0" />}
            {/* A squeezed track ellipsizes a label inside its own cell; it never
                wraps or paints across the neighbouring option. */}
            <span className="min-w-0 truncate">{label}</span>
          </button>
        )
      })}
    </div>
  )
}
