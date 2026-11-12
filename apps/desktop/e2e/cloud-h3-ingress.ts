/** Disposable TLS/QUIC boundary for the synthetic Cloud service; never installs a CA. */
import { execFile, execFileSync, spawn } from 'node:child_process'
import fs from 'node:fs'
import net from 'node:net'
import path from 'node:path'
import { promisify } from 'node:util'

export async function startCloudH3Ingress(directory: string, upstream: string, python: string) {
  const executable = process.env.HERMES_E2E_CLOUD_CADDY

  if (!executable) {
    return null
  }
  const listener = net.createServer()
  await new Promise<void>(resolve => listener.listen(0, '127.0.0.1', resolve))
  const port = (listener.address() as net.AddressInfo).port
  await new Promise<void>(resolve => listener.close(() => resolve()))
  const certificate = path.join(directory, 'cloud-cert.pem')
  const key = path.join(directory, 'cloud-key.pem')
  execFileSync(
    python,
    [
      '-c',
      `
from datetime import datetime, timezone, timedelta
from pathlib import Path
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
name=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,'localhost')])
now=datetime.now(timezone.utc)
cert=(x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
.serial_number(x509.random_serial_number()).not_valid_before(now-timedelta(minutes=1)).not_valid_after(now+timedelta(days=1))
.add_extension(x509.SubjectAlternativeName([x509.DNSName('localhost')]),critical=False)
.sign(key,hashes.SHA256()))
Path(${JSON.stringify(certificate)}).write_bytes(cert.public_bytes(serialization.Encoding.PEM))
Path(${JSON.stringify(key)}).write_bytes(key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))
`
    ],
    { timeout: 15_000 }
  )
  const log = path.join(directory, 'h3-access.jsonl')
  const config = path.join(directory, 'caddy.json')
  fs.writeFileSync(
    config,
    JSON.stringify({
      admin: { disabled: true },
      apps: {
        tls: { certificates: { load_files: [{ certificate, key }] } },
        http: {
          servers: {
            fixture: {
              listen: [`127.0.0.1:${port}`],
              protocols: ['h1', 'h2', 'h3'],
              automatic_https: { disable: true },
              tls_connection_policies: [{}],
              logs: {},
              routes: [
                { handle: [{ handler: 'reverse_proxy', upstreams: [{ dial: upstream.replace('http://', '') }] }] }
              ]
            }
          }
        }
      },
      logging: { logs: { default: { writer: { output: 'file', filename: log }, encoder: { format: 'json' } } } }
    })
  )
  const child = spawn(executable, ['run', '--config', config], { stdio: 'ignore', windowsHide: true })
  let spawnError: Error | undefined
  child.on('error', error => {
    spawnError = error
  })
  const origin = `https://localhost:${port}`

  const close = async () => {
    if (child.exitCode !== null || !child.pid) {
      return
    }
    const stopped = new Promise<void>(resolve => child.once('exit', () => resolve()))
    child.kill()
    await stopped
  }

  try {
    // This check exercises native libcurl, pinned CA and H3; HTTP status is immaterial.
    for (let attempt = 0; ; attempt++) {
      if (spawnError) {
        throw spawnError
      }

      try {
        await promisify(execFile)(
          python,
          [
            '-c',
            `
from curl_cffi import requests
with requests.Session(http_version='v3only',trust_env=False,verify=${JSON.stringify(certificate)}) as c:
 r=c.get(${JSON.stringify(origin + '/h3-probe')},timeout=2)
 assert r.http_version==30
`
          ],
          { timeout: 5_000, stdio: 'pipe' }
        )

        break
      } catch (error) {
        if (attempt >= 15 || child.exitCode !== null) {
          throw error
        }
        await new Promise(resolve => setTimeout(resolve, 200))
      }
    }

    return { origin, certificate, log, close }
  } catch (error) {
    await close()
    throw error
  }
}
