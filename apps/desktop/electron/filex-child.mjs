// Lifecycle only: filex owns all watching, planning, transfers and conflicts.
import { spawn, execFile } from 'node:child_process'
import { pathToFileURL } from 'node:url'

export function superviseFilexChild(parent, os) {
  let child
  let closing = false
  function stop() {
    if (closing) return
    closing = true
    if (!child?.pid) return os.exit(0)
    if (os.platform === 'win32') {
      os.execFile('taskkill', ['/PID', String(child.pid), '/T', '/F'], { windowsHide: true }, error => {
        // Child exit proves termination. Failure keeps the helper alive, so
        // main refuses the account transition and can retry a SIGTERM.
        if (error && child.exitCode === null) closing = false
      })
    } else {
      try {
        os.killGroup(-child.pid, 'SIGTERM')
      } catch {
        closing = false
      }
      const timer = setTimeout(() => {
        try {
          os.killGroup(-child.pid, 'SIGKILL')
        } catch {
          closing = false
        }
      }, 5000)
      child.once('exit', () => clearTimeout(timer))
    }
  }
  parent.once('disconnect', stop)
  parent.on('SIGTERM', stop)
  parent.on('SIGINT', stop)
  parent.once('message', ({ binary, args }) => {
    child = os.spawn(binary, args, {
      env: parent.env,
      detached: os.platform !== 'win32',
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe']
    })
    parent.stdout.on?.('error', stop)
    child.stdout.on?.('error', stop)
    child.stderr.on?.('error', stop)
    child.stdout.pipe(parent.stdout)
    child.stderr.resume()
    child.once('error', () => os.exit(1))
    child.once('exit', code => os.exit(code ?? 1))
  })
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  superviseFilexChild(process, {
    platform: process.platform,
    spawn,
    execFile,
    killGroup: process.kill,
    exit: code => process.exit(code)
  })
}
