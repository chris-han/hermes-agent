import type { UpdaterApplyResultWire, UpdaterStatusWire, UpdaterStrategy } from './index'

/** Preserve one native selection across check/apply and one owner through handoff. */
export class UpdateOperation {
  private strategy: Promise<UpdaterStrategy | null> | undefined
  private applying: boolean = false
  private checking: number = 0
  private lastCheckStatus: 'failed' | 'completed' | undefined

  get checkStatus(): 'checking' | 'failed' | 'completed' | undefined {
    return this.checking ? 'checking' : this.lastCheckStatus
  }

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

    try {
      const result = await run()
      this.lastCheckStatus = result.error ? 'failed' : 'completed'

      return result
    } catch (error) {
      this.lastCheckStatus = 'failed'
      throw error
    } finally { this.checking-- }
  }

  changeSelection(save: () => void): void {
    if (this.applying || this.checking) { throw new Error('An update operation is already in progress.') }
    save()
    this.strategy = undefined
    this.lastCheckStatus = undefined
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
