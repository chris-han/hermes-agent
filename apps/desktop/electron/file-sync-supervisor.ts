import { fork } from 'node:child_process'
import { createHash, randomUUID } from 'node:crypto'
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

import { normalizeSemantierWorkspaceOrigin } from './semantier-workspace-client'

export const FILEX_PIN = {
  version: '0.61.1',
  commit: '2964aba36586b32d142d8fa9edba37d9088541d0',
  linux: '331004fe75de55b1a6e08165dc172a8ae357cb1799862e89b2bb59b773df5f11',
  win32: '190305d0e07c882e1304ebbddccc338a805ad00e3413dd48249f8d01f8729ed5'
} as const

type Identity = { userId: string; workspaceId: string }
type Credential = Identity & { remoteRoot: string; token: string; enginePath: string; version: string }
type Binding = Identity & {
  id: string
  serverOrigin: string
  canonicalLocalFolder: string
  remoteRoot: string
  sealedToken: string
  enabled: boolean
  paired: boolean
}
type EngineEvent = { event?: string; code?: string; params?: { conflicts?: number; failed?: number; held?: number } }
type Run = (
  args: string[],
  env: NodeJS.ProcessEnv,
  onEvent: (event: EngineEvent) => void,
  signal: AbortSignal
) => Promise<{ code: number; events?: EngineEvent[]; stdout?: string }>
type Options = {
  stateDir: string
  binary: string
  helper: string
  identity: (origin: string) => Promise<Identity>
  binding: (origin: string) => Promise<Credential>
  seal: (value: string) => string
  verifyBinary?: () => Promise<void>
  run?: Run
  forbiddenRoots?: string[]
}
export type FileSyncStatus = {
  state: 'Paused' | 'Syncing' | 'Synced' | 'Conflict' | 'Error'
  folder?: string
  message?: string
}
const overlaps = (a: string, b: string) => a === b || a.startsWith(b + path.sep) || b.startsWith(a + path.sep)
const canonicalKey = (p: string) => (process.platform === 'win32' ? p.toLowerCase() : p)

export class FileSyncSupervisor {
  readonly registryPath: string
  private generation = 0
  private busy = false
  private closing = false
  private quitReady = false
  private quitPending: Promise<void> | null = null
  private controller: AbortController | null = null
  private running: Promise<unknown> | null = null
  private current: Binding | null = null
  private stopChild: (() => void) | null = null
  private view: FileSyncStatus = { state: 'Paused' }
  constructor(public options: Options) {
    this.registryPath = path.join(options.stateDir, 'folder-owners.json')
  }
  status() {
    return { ...this.view }
  }
  async owners(): Promise<Binding[]> {
    try {
      const registry = JSON.parse(await fs.readFile(this.registryPath, 'utf8'))

      if (registry.version !== 1 || !Array.isArray(registry.bindings)) {
        throw new Error('Invalid File Sync ownership registry. Preserve it for recovery.')
      }

      return registry.bindings
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
        return []
      }

      throw error
    }
  }
  private async write(bindings: Binding[]) {
    await fs.mkdir(this.options.stateDir, { recursive: true, mode: 0o700 })
    const temp = this.registryPath + '.' + randomUUID() + '.tmp'
    const handle = await fs.open(temp, 'wx', 0o600)

    try {
      await handle.writeFile(JSON.stringify({ version: 1, bindings }))
      await handle.sync()
    } finally {
      await handle.close()
    }

    await fs.rename(temp, this.registryPath)
  }
  async accountChanging(origin: string) {
    normalizeSemantierWorkspaceOrigin(origin) // Validate even while another origin is pending.
    this.generation++
    await this.stop()
  }
  private async stop() {
    if (this.controller?.signal.aborted) {
      this.stopChild?.()
    } else {
      this.controller?.abort()
    }

    let timeout: ReturnType<typeof setTimeout> | undefined

    try {
      await Promise.race([
        this.running,
        new Promise((_, reject) => {
          timeout = setTimeout(
            () => reject(new Error('The sync process has not stopped. Account change was refused.')),
            8000
          )
        })
      ])
    } finally {
      clearTimeout(timeout)
    }

    this.controller = null
    this.running = null
    this.view = { state: 'Paused', folder: this.current?.canonicalLocalFolder }
  }
  async shutdown() {
    this.closing = true
    this.generation++
    await this.stop()
  }
  async pause() {
    this.generation++
    await this.stop()

    if (this.current) {
      const owners = await this.owners()
      await this.write(owners.map(x => (x.id === this.current?.id ? { ...x, enabled: false } : x)))
    }
  }
  async restore() {
    const record = (await this.owners()).find(x => x.enabled)

    if (record) {
      await this.enable(record.serverOrigin, record.canonicalLocalFolder)
    }
  }
  async enable(rawOrigin: string, folder: string) {
    if(this.closing){throw new Error('File Sync is closing with the application.')}

    if (this.busy) {
      throw new Error('A File Sync operation is already in progress.')
    }

    this.busy = true
    const generation = ++this.generation

    const check = () => {
      if (this.generation !== generation) {
        throw new Error('The account changed. Reconnect File Sync.')
      }
    }

    try {
      await this.stop()
      check()
      const origin = normalizeSemantierWorkspaceOrigin(rawOrigin)
      const canonical = await this.localFolder(folder)
      check()
      const identity = await this.options.identity(origin)
      check()

      const owners = await this.owners()
      check()
      const match = this.matchOwner(owners, origin, identity, canonical)

      if (!match && (await fs.readdir(canonical)).length) {
        throw new Error('Choose an empty folder for a new File Sync binding. Existing files are preserved.')
      }

      check()
      const credential = await this.options.binding(origin)
      check()

      this.validateCredential(credential, identity, match)

      const record: Binding = {
        ...identity,
        id: match?.id || randomUUID(),
        serverOrigin: origin,
        canonicalLocalFolder: canonical,
        remoteRoot: credential.remoteRoot,
        sealedToken: this.options.seal(credential.token),
        enabled: true,
        paired: match?.paired || false
      }

      check()

      await this.verifyDependency()
      check()

      const store = () =>
        this.write([...owners.filter(x => x.id !== record.id).map(x => ({ ...x, enabled: false })), record])

      // Commit ownership before the first engine invocation, even if pairing fails.
      await store()
      check()
      this.current = record
      const state = path.join(this.options.stateDir, record.id)
      await fs.mkdir(state, { recursive: true, mode: 0o700 })
      check()

      const env = this.environment(origin, credential, state)
      const controller = new AbortController()
      this.controller = controller
      const run = this.options.run || this.run.bind(this)
      await this.preparePair(run, env, record, controller.signal, store, check)
      const current = await this.options.identity(origin)
      check()

      if (current.userId !== record.userId || current.workspaceId !== record.workspaceId) {
        throw new Error('The account changed before sync started.')
      }

      this.view = { state: 'Syncing', folder: canonical }
      this.running = run(
        ['sync', 'run', '--json', '--account', record.id, '--watch', '5s', '--live=false', '--full-every', '30m'],
        env,
        event => this.watchEvent(event, generation, canonical),
        controller.signal
      ).then(() => {
        if (this.generation === generation) {
          this.view = {
            state: 'Error',
            folder: canonical,
            message: 'The sync service stopped. Resume to reconnect; files are preserved.'
          }
        }
      })

      return this.status()
    } catch (error) {
      if (this.generation === generation) {
        this.view = { state: 'Error', message: error instanceof Error ? error.message : 'File Sync unavailable' }
      }

      throw error
    } finally {
      this.busy = false
    }
  }
  private async localFolder(folder: string) {
    const canonical = await fs.realpath(folder)

    if (!(await fs.stat(canonical)).isDirectory()) {
      throw new Error('Select an available folder.')
    }

    const reserved = [
      this.options.stateDir,
      path.join(os.homedir(), '.hermes'),
      path.join(os.homedir(), '.semantier-home'),
      ...(this.options.forbiddenRoots || [])
    ]

    if (
      canonical === path.parse(canonical).root ||
      reserved.some(p => overlaps(canonicalKey(canonical), canonicalKey(path.resolve(p))))
    ) {
      throw new Error('Choose a folder outside application state and credentials.')
    }

    return canonical
  }

  private matchOwner(owners: Binding[], origin: string, identity: Identity, canonical: string) {
    if (!identity.userId || !identity.workspaceId) {
      throw new Error('Sign in to your personal Workspace first.')
    }

    const match = owners.find(x => overlaps(canonicalKey(x.canonicalLocalFolder), canonicalKey(canonical)))

    if (
      match &&
      (match.serverOrigin !== origin ||
        match.userId !== identity.userId ||
        match.workspaceId !== identity.workspaceId ||
        canonicalKey(match.canonicalLocalFolder) !== canonicalKey(canonical))
    ) {
      throw new Error(
        'This folder belongs to another account or folder binding. Choose a separate folder; existing files are preserved.'
      )
    }

    return match
  }

  private validateCredential(credential: Credential, identity: Identity, match: Binding | undefined) {
    if (
      credential.userId !== identity.userId ||
      credential.workspaceId !== identity.workspaceId ||
      credential.version !== FILEX_PIN.version ||
      credential.enginePath !== '/api/file-sync/engine'
    ) {
      throw new Error('The account changed or the sync service is incompatible.')
    }

    if (match && match.remoteRoot !== credential.remoteRoot) {
      throw new Error('Remote folder binding changed. Preserve local data before reconnecting.')
    }
  }

  private async verifyDependency() {
    if (this.options.verifyBinary) {
      await this.options.verifyBinary()
    } else {
      const expected = FILEX_PIN[process.platform as 'linux' | 'win32']

      if (
        !expected ||
        createHash('sha256')
          .update(await fs.readFile(this.options.binary))
          .digest('hex') !== expected
      ) {
        throw new Error('Install the pinned filex 0.61.1 CLI before enabling File Sync.')
      }
    }
  }

  private environment(origin: string, credential: Credential, state: string) {
    const env: NodeJS.ProcessEnv = {
      FILEX_URL: origin + credential.enginePath,
      FILEX_TOKEN: credential.token,
      FILEX_SYNC_DIR: state,
      FILEX_CLI_CONFIG: path.join(state, 'cli.yaml'),
      FILEX_LANG: 'en',
      ELECTRON_RUN_AS_NODE: '1'
    }

    for (const key of ['PATH', 'HOME', 'USERPROFILE', 'APPDATA', 'LOCALAPPDATA', 'TEMP', 'TMP', 'SystemRoot', 'LANG']) {
      if (process.env[key]) {
        env[key] = process.env[key]
      }
    }

    return env
  }

  private async preparePair(
    run: Run,
    env: NodeJS.ProcessEnv,
    record: Binding,
    signal: AbortSignal,
    store: () => Promise<void>,
    check: () => void
  ) {
    const listing = run(['sync', 'list', '--json'], env, () => {}, signal)
    this.running = listing
    const listed = await listing
    check()

    if (listed.code !== 0) {
      throw new Error('Cannot read sync pairing state. Preserve it for recovery.')
    }

    const pairs = JSON.parse(listed.stdout || '[]')

    if (!Array.isArray(pairs) || pairs.length > 1) {
      throw new Error('Unexpected sync pairs. Preserve the existing baseline for recovery.')
    }

    if (pairs.length === 1) {
      const pair = pairs[0]

      if (
        pair.local !== record.canonicalLocalFolder ||
        pair.remote !== record.remoteRoot ||
        pair.account !== record.id
      ) {
        throw new Error('Sync pair ownership differs. No engine was started.')
      }

      record.paired = true
      await store()
      check()
    } else if (record.paired) {
      throw new Error(
        'The sync baseline or pair is missing. Restore it before resuming; no automatic reset was performed.'
      )
    }

    if (!record.paired) {
      const adding = run(
        ['sync', 'add', record.canonicalLocalFolder, record.remoteRoot, '--account', record.id],
        env,
        () => {},
        signal
      )

      this.running = adding
      const result = await adding
      check()

      if (result.code !== 0) {
        throw new Error('File Sync pairing failed; files and ownership are preserved.')
      }

      record.paired = true
      await store()
      check()
    }
  }

  private watchEvent(event: EngineEvent, generation: number, canonical: string) {
    if (this.generation !== generation) {
      return
    }

    if (event.event === 'pass') {
      this.view = {
        state: event.params?.failed ? 'Error' : event.params?.conflicts || event.params?.held ? 'Conflict' : 'Synced',
        folder: canonical
      }
    } else if (event.event === 'error') {
      this.view = {
        state: 'Error',
        folder: canonical,
        message: 'File Sync needs reconnection. Your files are preserved.'
      }
    } else if (event.event === 'progress' && this.view.state !== 'Conflict') {
      this.view = { state: 'Syncing', folder: canonical }
    }
  }

  holdQuit(event: { preventDefault: () => void }, reenter: () => void, failed: () => void) {
    if (this.quitReady) {return false}
    event.preventDefault()

    if (!this.quitPending)
      {this.quitPending = this.shutdown()
        .then(() => {
          this.quitReady = true
          reenter()
        })
        .catch(() => {
          this.quitPending = null
          failed()
        })}

    return true
  }

  private run(
    args: string[],
    env: NodeJS.ProcessEnv,
    onEvent: (event: EngineEvent) => void,
    signal: AbortSignal
  ): Promise<{ code: number; stdout: string }> {
    if (signal.aborted) {
      return Promise.resolve({ code: 1, stdout: '' })
    }

    return new Promise(resolve => {
      let child: ReturnType<typeof fork>

      try {
        child = fork(this.options.helper, [], { env, stdio: ['ignore', 'pipe', 'ignore', 'ipc'] })
      } catch {
        resolve({ code: 1, stdout: '' })

        return
      }

      let pending = '',
        stdout = ''

      const stop = () => {
        if (child.connected) {
          child.disconnect()
        } else {
          child.kill('SIGTERM')
        }
      }

      this.stopChild = stop
      signal.addEventListener('abort', stop, { once: true })
      child.stdout?.on('data', chunk => {
        const text = String(chunk)

        if (stdout.length < 1024 * 1024) {
          stdout += text
        }

        pending += text

        if (pending.length > 65536) {
          pending = ''

          return
        }

        const lines = pending.split('\n')
        pending = lines.pop() || ''

        for (const line of lines) {
          try {
            onEvent(JSON.parse(line))
          } catch {
            /* non-JSON diagnostics never define product state */
          }
        }
      })

      const finish = (code: number) => {
        signal.removeEventListener('abort', stop)

        if (this.stopChild === stop) {
          this.stopChild = null
        }

        resolve({ code, stdout })
      }

      child.on('error', () => {
        if (!child.pid) {
          finish(1)
        }
      })
      child.once('exit', code => finish(code ?? 1))
      child.send({ binary: this.options.binary, args }, error => {
        if (error) {
          stop()
        }
      })

      if (signal.aborted) {
        stop()
      }
    })
  }
}
