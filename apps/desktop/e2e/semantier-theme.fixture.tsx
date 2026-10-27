import '../src/styles.css'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useState } from 'react'
import { createRoot } from 'react-dom/client'

import { AppearanceSettings } from '../src/app/settings/appearance-settings'
import { Button } from '../src/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogTitle, DialogTrigger } from '../src/components/ui/dialog'
import { Input } from '../src/components/ui/input'
import { Brain, CheckCircle2, Send, Settings } from '../src/lib/icons'
import { ThemeProvider, useTheme } from '../src/themes/context'

const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })

function Surface() {
  const theme = useTheme()
  const [settings, showSettings] = useState(false)

  return (
    <div className="grid h-dvh min-w-0 grid-rows-[48px_1fr] sm:grid-cols-[224px_minmax(0,1fr)] sm:grid-rows-[1fr]">
      <aside className="flex gap-2 border-b border-(--dt-border) bg-(--ui-bg-sidebar) p-3 sm:flex-col sm:border-r sm:border-b-0">
        <strong className="mr-auto text-sm">semantier</strong>
        <Button onClick={() => showSettings(false)} variant="ghost">
          Chat
        </Button>
        <Button onClick={() => showSettings(true)} variant="ghost">
          <Settings />
          Appearance settings
        </Button>
      </aside>
      <section className="flex min-h-0 min-w-0 flex-col bg-(--ui-chat-surface-background)">
        <header className="flex shrink-0 flex-wrap items-center gap-2 border-b border-(--dt-border) bg-(--ui-bg-chrome) px-4 py-2">
          <span className="mr-auto text-sm">{settings ? 'Appearance' : 'Semantier Local'}</span>
          <Button aria-label="Light appearance" onClick={() => theme.setMode('light')} size="sm" variant="ghost">
            Light
          </Button>
          <Button aria-label="Dark appearance" onClick={() => theme.setMode('dark')} size="sm" variant="ghost">
            Dark
          </Button>
          <Button aria-label="System appearance" onClick={() => theme.setMode('system')} size="sm" variant="ghost">
            System
          </Button>
        </header>
        <div className="min-h-0 min-w-0 flex-1 overflow-auto p-4 sm:p-6">
          {settings ? (
            <AppearanceSettings subpage="theme" />
          ) : (
            <div className="mx-auto flex max-w-3xl flex-col gap-6">
              <article className="rounded-lg border border-(--dt-border) bg-card p-4" data-testid="content-panel">
                <h1 className="mb-3 text-2xl font-semibold">Your workspace, in focus.</h1>
                <p className="mb-3 text-sm">Review evidence, work with tools, and keep the conversation in view.</p>
                <p className="text-sm" data-testid="chinese">
                  司序科技 · 让组织知识成为可持续运作的智能。
                </p>
                <p className="mt-4 text-xs text-(--ui-text-tertiary)" data-testid="quiet">
                  Local workspace · Ready for your next task
                </p>
                <div className="mt-4 flex flex-wrap gap-4 text-xs">
                  <span className="inline-flex items-center gap-1 text-(--ui-success)">
                    <CheckCircle2 />
                    Connected
                  </span>
                  <span className="text-(--ui-warning)">Review needed</span>
                  <span className="text-(--ui-info)">Reference available</span>
                  <span className="inline-flex items-center gap-1 rounded-sm bg-(--ui-intelligence) px-2 py-1 text-(--ui-intelligence-foreground)">
                    <Brain />
                    Semantier
                  </span>
                </div>
                <pre className="mt-4 overflow-auto rounded-md bg-muted p-3 font-mono text-xs">
                  profile: semantier-local
                  <br />
                  version: 0.0.26 · 0123456789
                </pre>
                <div className="mt-4 flex flex-wrap gap-2">
                  <Button data-testid="primary">
                    <Send />
                    Continue
                  </Button>
                  <Button variant="secondary">Open files</Button>
                  <Button disabled variant="secondary">
                    Unavailable
                  </Button>
                  <Button data-testid="danger" variant="destructive">
                    Delete selection
                  </Button>
                  <Dialog>
                    <DialogTrigger asChild>
                      <Button variant="outline">Open dialog</Button>
                    </DialogTrigger>
                    <DialogContent>
                      <DialogTitle>Workspace preferences</DialogTitle>
                      <DialogDescription>Settings remain local to this profile.</DialogDescription>
                      <Input aria-label="Dialog field" placeholder="Organization name" />
                    </DialogContent>
                  </Dialog>
                </div>
              </article>
              <div
                className="rounded-lg border border-(--dt-border) bg-card p-3 shadow-composer"
                data-slot="composer-root"
              >
                <Input aria-label="Message" placeholder="Ask Semantier…" />
              </div>
            </div>
          )}
        </div>
      </section>
    </div>
  )
}

createRoot(document.getElementById('root')!).render(
  <QueryClientProvider client={queryClient}>
    <ThemeProvider>
      <Surface />
    </ThemeProvider>
  </QueryClientProvider>
)
