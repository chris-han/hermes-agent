// Verify the receipt-bound native input tree immediately before signing.
import { nativeInputsReason } from './prepared-native-deps.mjs'
const [source, nativeDeps, nativeToolchain] = process.argv.slice(2)
if (!source || !nativeDeps || !nativeToolchain) throw new Error('Pinned native inputs and toolchain are required')
const reason = nativeInputsReason({ source, nativeDeps, nativeToolchain, platform: 'win32', arch: 'x64' })
if (reason !== null) throw new Error('Prepared native receipt refused: ' + reason)
console.log('PREPARED_WINDOWS_NATIVE_RECEIPT_VERIFIED')
