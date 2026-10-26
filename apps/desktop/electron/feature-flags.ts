export function isCanaryTag(tag: string | null | undefined): boolean {
  return /\+canary\.20\d{6}T\d{6}Z$/.test(tag || '')
}
