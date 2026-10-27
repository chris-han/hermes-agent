import type { UpdaterApplyResultWire, UpdaterStatusWire, UpdaterStrategy } from './index'

/** Preserve one native selection across check/apply and one owner through handoff. */
export class UpdateOperation {
  private strategy: Promise<UpdaterStrategy | null> | undefined
  private applying: boolean = false
  private checking: number = 0

  constructor(private readonly initialize: () => Promise<UpdaterStrategy | null>) {}

  resolve(): Promise<UpdaterStrategy | null> {
    this.strategy ??= this.initialize().catch((error: Error): never => {
      this.strategy = undefined
      throw error
    })

    return this.strategy
  }

  reset(): void {
    this.changeSelection(() => {})
  }

  async check(run: () => Promise<UpdaterStatusWire>): Promise<UpdaterStatusWire> {
    if (this.applying) { throw new Error('An update is already in progress.') }
    this.checking++

    try { return await run() } finally { this.checking-- }
  }

  changeSelection(save: () => void): void {
    if (this.applying || this.checking) { throw new Error('An update operation is already in progress.') }
    save()
    this.strategy = undefined
  }

  async apply(run: () => Promise<UpdaterApplyResultWire>): Promise<UpdaterApplyResultWire> {
    if (this.applying || this.checking) {
      throw new Error('An update is already in progress.')
    }

    this.applying = true
    let handedOff: boolean = false

    try {
      const result: UpdaterApplyResultWire = await run()
      handedOff = result.handedOff === true

      return result
    } finally {
      if (!handedOff) {
        this.applying = false
      }
    }
  }
}
