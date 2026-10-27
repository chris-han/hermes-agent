import '../src/styles.css'

import React from 'react'
import { createRoot } from 'react-dom/client'

import { UpdateDeliverySourceRow } from '../src/app/settings/update-delivery-source-row'

let selected = (localStorage.getItem('fixture-update-source') ?? 'channel1') as 'channel1' | 'channel2' | 'channel3'

const options = [
  { id: 'channel1' as const, publicBase: 'https://ecs.example.com' },
  { id: 'channel2' as const, publicBase: 'https://update2.semantier.com' },
  { id: 'channel3' as const, publicBase: 'http://127.0.0.1:8926' }
]

window.hermesDesktop = { updates: {
  getDeliverySources: async () => ({ selected, options }),
  setDeliverySource: async source => {
    selected = source
    localStorage.setItem('fixture-update-source', source)

    return { selection: { selected, options }, status: { supported: true, updateAvailable: false } }
  }
} } as typeof window.hermesDesktop
createRoot(document.getElementById('root')!).render(<main className="mx-auto w-full max-w-2xl p-6"><UpdateDeliverySourceRow /></main>)
