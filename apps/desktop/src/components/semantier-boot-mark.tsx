import { useEffect, useState } from 'react'

import { DecodeText } from '@/components/ui/decode-text'
import { prefersReducedMotion } from '@/hooks/use-media-query'

const WORDS = ['SĒMANTIKÓS', 'SÉMANTIQUE', 'SEMANTIER'] as const
// Each word gets its decode time plus a readable hold. Never delay readiness.
const WORD_MS = 1400

interface SemantierBootMarkProps {
  active: boolean
  className?: string
}

export function SemantierBootMark({ active, className }: SemantierBootMarkProps) {
  const [stage, setStage] = useState(0)
  const animate = active && !prefersReducedMotion()

  useEffect(() => {
    if (!animate) {
      setStage(0)

      return
    }

    if (stage === WORDS.length - 1) {
      return
    }

    const id = window.setTimeout(() => setStage(stage + 1), WORD_MS)

    return () => window.clearTimeout(id)
  }, [animate, stage])

  const word = animate ? WORDS[stage] : WORDS[2]

  return (
    <span aria-label="SEMANTIER" className={className} role="img">
      <DecodeText active={animate} aria-hidden="true" cursor data-boot-word="" key={word} prefix={4} text={word} />
    </span>
  )
}
