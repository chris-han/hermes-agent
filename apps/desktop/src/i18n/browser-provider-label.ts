/** Display labels only: backend provider names remain stable API selection keys. */
export function browserProviderLabel(name: string, locale: string): string {
  const chinese = locale.startsWith('zh')

  if (name === 'Local Browser') {
    return chinese ? '内置 Chromium（默认）' : 'Built-in Chromium (default)'
  }

  return name.endsWith(' (system)') && chinese ? name.replace(' (system)', '（系统浏览器）') : name
}
