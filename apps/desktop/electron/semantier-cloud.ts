/** Process composition selection; model providers and credentials are independent. */
export function desktopCloudEnvironment(
  parent: NodeJS.ProcessEnv,
  backend: NodeJS.ProcessEnv = {},
): NodeJS.ProcessEnv {
  const parentMode = parent.SEMANTIER_CLOUD
  const backendMode = backend.SEMANTIER_CLOUD
  for (const mode of [parentMode, backendMode]) {
    if (mode !== undefined && mode !== '0' && mode !== '1') {
      throw new Error('SEMANTIER_CLOUD must be exactly 0 or 1')
    }
  }
  if (parentMode !== undefined && backendMode !== undefined && parentMode !== backendMode) {
    throw new Error('Conflicting SEMANTIER_CLOUD composition')
  }
  return {...parent, ...backend, SEMANTIER_CLOUD: parentMode ?? backendMode ?? '0'}
}
