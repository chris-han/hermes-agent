import { spawn } from 'node:child_process'

type Runtime = { command: string; cwd: string; env: NodeJS.ProcessEnv }

/** Run the fixed native restriction owner without starting or contacting the model service. */
export async function runAccountFence(runtime: Runtime, account: Record<string, string> | null): Promise<void> {
  const input = JSON.stringify({ account })
  if (Buffer.byteLength(input) > 8192) {throw new Error('Account context is unavailable.')}
  return new Promise((resolve, reject) => {
    const child = spawn(runtime.command, ['-B', '-m', 'hermes_cli.cloud_account_fence'], {
      cwd: runtime.cwd, env: { ...runtime.env, PYTHONUTF8: '1' }, shell: false, windowsHide: true,
      stdio: ['pipe', 'pipe', 'pipe']
    })
    let output = ''
    let timedOut = false
    const timer = setTimeout(() => {timedOut = true; child.kill()}, 30_000)
    child.stdout.setEncoding('utf8')
    child.stdout.on('data', (chunk: string) => {
      output += chunk
      if (Buffer.byteLength(output) > 8192) {timedOut = true; child.kill()}
    })
    child.stderr.resume()
    child.stdin.on('error', () => {})
    child.on('error', () => {clearTimeout(timer); reject(new Error('Account context is unavailable.'))})
    child.on('close', code => {
      clearTimeout(timer)
      try {
        if (!timedOut && code === 0 && JSON.parse(output)?.ok === true) {resolve(); return}
      } catch { /* An invalid response is a failure. */ }
      reject(new Error('Account context is unavailable.'))
    })
    child.stdin.end(input)
  })
}
