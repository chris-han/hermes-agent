import { spawn } from 'node:child_process'

import type { KeyRequest, PublicProof } from './managed-device-enrollment'

export interface NativeKeyRuntime {
  command: string
  cwd: string
  env: NodeJS.ProcessEnv
}

/** A private, owned helper process; never a renderer-selected executable or shell. */
export async function runManagedDeviceKey(runtime: NativeKeyRuntime, request: KeyRequest): Promise<PublicProof> {
  const wire = JSON.stringify(request)

  if (
    Buffer.byteLength(wire) > 32768 ||
    !['create', 'public', 'sign_enrollment'].includes(request.operation) ||
    !/^[a-f0-9]{32}$/.test(request.key_id)
  ) {
    throw new Error('Invalid native device request')
  }

  return new Promise((resolve, reject) => {
    const child = spawn(runtime.command, ['-B', '-m', 'hermes_cli.semantier_device_key'], {
      cwd: runtime.cwd,
      env: { ...runtime.env, PYTHONUTF8: '1' },
      shell: false,
      windowsHide: true,
      stdio: ['pipe', 'pipe', 'pipe']
    })

    let output = ''
    let stderrSize = 0
    let settled = false

    const fail = (code: string) => {
      if (settled) {
        return
      }

      settled = true
      clearTimeout(timer)
      reject(Object.assign(new Error('Native managed device key is unavailable'), { nativeDeviceCode: code }))
    }

    const timer = setTimeout(() => {
      child.kill()
      fail('NATIVE_DEVICE_KEY_TIMEOUT')
    }, 30000)

    child.on('error', () => fail('NATIVE_DEVICE_KEY_UNAVAILABLE'))
    child.stdout.setEncoding('utf8')
    child.stdout.on('data', (chunk: string) => {
      if (settled) {
        return
      }

      output += chunk

      if (Buffer.byteLength(output) > 32768) {
        child.kill()
        fail('NATIVE_DEVICE_KEY_PROTOCOL')
      }
    })
    child.stderr.on('data', (chunk: Buffer) => {
      if (settled) {
        return
      }

      stderrSize += chunk.length

      if (stderrSize > 4096) {
        child.kill()
        fail('NATIVE_DEVICE_KEY_PROTOCOL')
      }
    })
    child.stdin.on('error', () => {
      /* close/error owns the durable outcome; never replay */
    })
    child.on('close', code => {
      if (settled) {
        return
      }

      let reply

      try {
        reply = JSON.parse(output)
      } catch {
        fail('NATIVE_DEVICE_KEY_PROTOCOL')

        return
      }

      if (code !== 0 || reply?.ok !== true) {
        fail(
          ['NATIVE_DEVICE_KEY_UNAVAILABLE', 'INVALID_NATIVE_DEVICE_KEY_REQUEST'].includes(reply?.error)
            ? reply.error
            : 'NATIVE_DEVICE_KEY_PROTOCOL'
        )

        return
      }

      const result = reply.result

      if (
        !result ||
        typeof result !== 'object' ||
        Object.keys(result).some(k => !['key_id', 'public_key', 'signature', 'assurance'].includes(k)) ||
        result.key_id !== request.key_id ||
        result.assurance !== 'NOT_VERIFIED' ||
        typeof result.public_key !== 'string' ||
        !result.public_key ||
        result.public_key.length > 2048 ||
        (request.operation === 'sign_enrollment' &&
          (typeof result.signature !== 'string' || result.signature.length > 112))
      ) {
        fail('NATIVE_DEVICE_KEY_PROTOCOL')

        return
      }

      settled = true
      clearTimeout(timer)
      resolve(result)
    })
    child.stdin.end(wire)
  })
}
