/** Process composition selection; model providers and credentials are independent. */
export function desktopCloudEnvironment(
  parent: NodeJS.ProcessEnv,
  backend: NodeJS.ProcessEnv = {},
): NodeJS.ProcessEnv {
  const parentMode = parent.SEMANTIER_RUNTIME
  const backendMode = backend.SEMANTIER_RUNTIME

  for (const mode of [parentMode, backendMode]) {
    if (mode !== undefined && mode !== '0' && mode !== '1') {
      throw new Error('SEMANTIER_RUNTIME must be exactly 0 or 1')
    }
  }

  if (parentMode !== undefined && backendMode !== undefined && parentMode !== backendMode) {
    throw new Error('Conflicting SEMANTIER_RUNTIME composition')
  }

  return {...parent, ...backend, SEMANTIER_RUNTIME: parentMode ?? backendMode ?? '0'}
}
