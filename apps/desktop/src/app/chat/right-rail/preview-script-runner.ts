/**
 * PREVIEW SCRIPT RUNNER REGISTRY — the one way anything in the app reaches into
 * the preview pane's guest page, the script analog of preview-nav's handle
 * registry.
 *
 * A live browser pane registers its webview's `executeJavaScript` here, keyed
 * by tab id; `activePreviewScriptRunner` resolves the ACTIVE tab among the
 * requesting session's tabs. Both guest-page features ride it — the tour tool (preview-tour.ts)
 * and the interaction tool (preview-act.ts) — so their heavy payloads stay out
 * of the pane component's static import graph and only load when used.
 */

import type { PreviewOwner } from '@/store/preview-ownership'

import { activePreviewTabFor } from './preview-active-tab'
import type { PreviewInputEvent } from './preview-input'

/** Runs JS source in the pane's guest page, resolving its completion value. */
export type PreviewScriptRunner = (code: string) => Promise<unknown>

const runners = new Map<string, PreviewScriptRunner>()

/** Register a live preview's script runner; returns an idempotent unregister. */
export function registerPreviewScriptRunner(tabId: string, runner: PreviewScriptRunner): () => void {
  runners.set(tabId, runner)

  return () => {
    if (runners.get(tabId) === runner) {
      runners.delete(tabId)
    }
  }
}

/** The script runner of the ACTIVE tab among those `owner` (the requesting
 *  session's stored id; omitted = the focused session) may see. Null = no live
 *  page behind it. */
export function activePreviewScriptRunner(owner?: PreviewOwner): PreviewScriptRunner | null {
  const tab = activePreviewTabFor(owner)

  return (tab && runners.get(tab.id)) || null
}

/** Internal frame routing; tokens identify a document, never an agent-supplied
 * JavaScript expression. The existing preview engine supplies every script. */
export interface PreviewFrameRunner {
  list: () => Promise<{ token: string }[]>
  input: (token: string, event: PreviewInputEvent, point?: { x: number; y: number }) => Promise<void>
  run: (token: string, code: string) => Promise<unknown>
  mapPoint: (token: string, point: { x: number; y: number }) => Promise<{ visible: boolean; x: number; y: number }>
}

const frameRunners = new Map<string, PreviewFrameRunner>()

export function registerPreviewFrameRunner(tabId: string, runner: PreviewFrameRunner): () => void {
  frameRunners.set(tabId, runner)

  return () => { if (frameRunners.get(tabId) === runner) {frameRunners.delete(tabId)} }
}

export function activePreviewFrameRunner(owner?: PreviewOwner): PreviewFrameRunner | null {
  const tab = activePreviewTabFor(owner)

  return (tab && frameRunners.get(tab.id)) || null
}

type FrameBridge = (payload: { webContentsId: number; operation: 'list' | 'run' | 'mapPoint' | 'input' | 'dispose'; token?: string; code?: string; point?: { x: number; y: number }; event?: PreviewInputEvent }) => Promise<any>
type Guest = { getWebContentsId?: () => number }

/** Bind every trip to the exact webview that started it. */
export function createPreviewFrameRunner(getGuest: () => Guest | null, bridge: FrameBridge): PreviewFrameRunner {
  const invoke = async (payload: { operation: 'list' | 'run' | 'mapPoint' | 'input'; token?: string; code?: string; point?: { x: number; y: number }; event?: PreviewInputEvent }) => {
    const guest = getGuest()
    const webContentsId = guest?.getWebContentsId?.()

    if (!guest || typeof webContentsId !== 'number') {throw new Error('PANE_UNAVAILABLE')}
    const result = await bridge({ ...payload, webContentsId })

    if (getGuest() !== guest || guest.getWebContentsId?.() !== webContentsId) {throw new Error('STALE_TARGET')}

    return result
  }

  return {
    list: () => invoke({ operation: 'list' }),
    input: (token, event, point) => invoke({ operation: 'input', token, event, point }),
    run: (token, code) => invoke({ operation: 'run', token, code }),
    mapPoint: (token, point) => invoke({ operation: 'mapPoint', token, point })
  }
}

export async function readPreviewFrameText(runner: PreviewFrameRunner): Promise<string> {
  const frames = await runner.list()
  const pieces: string[] = []

  for (const frame of frames) {
    const text = await runner.run(frame.token, 'document.body ? document.body.innerText : ""')

    if (typeof text === 'string' && text.trim()) {pieces.push(text)}
  }

  return pieces.join('\n')
}
