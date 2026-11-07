import { useStore } from '@nanostores/react'
import { useNavigate } from 'react-router'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { modelServiceCopy } from '@/i18n/model-service'
import { openExternalLink } from '@/lib/external-link'
import type { RuntimeReadinessResult } from '@/lib/runtime-readiness'
import { $desktopBoot } from '@/store/boot'
import { $desktopOnboarding, refreshOnboarding, startManualProviderOAuth } from '@/store/onboarding'
import { captureOnboardingScope, type OnboardingScope } from '@/store/onboarding-scope'
import { setSettingsScope } from '@/store/settings-scope'

interface ModelServiceNoticeProps {
  profile?: string | null
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
}

export function ModelServiceNotice(props: ModelServiceNoticeProps) {
  const onboarding = useStore($desktopOnboarding)
  const boot = useStore($desktopBoot)
  const readiness = onboarding.readiness
  const captured = onboarding.readinessScope
  const current = captureOnboardingScope(props.profile)

  if (
    !readiness ||
    readiness.ready ||
    boot.running ||
    boot.error ||
    !captured ||
    JSON.stringify(current) !== JSON.stringify(captured)
  ) {
    return null
  }

  return <ServiceRecovery {...props} readiness={readiness} scope={captured} />
}

function ServiceRecovery({
  readiness,
  scope,
  requestGateway
}: ModelServiceNoticeProps & {
  readiness: RuntimeReadinessResult
  scope: OnboardingScope
}) {
  const { locale } = useI18n()
  const copy = modelServiceCopy[locale] ?? modelServiceCopy.en
  const navigate = useNavigate()

  const state =
    readiness.modelService?.state ??
    (readiness.source === 'fallback' || readiness.checksDisagree ? 'unavailable' : 'setup')

  const setup = state === 'setup' || state === 'reauth'
  const portal = readiness.modelService?.portalUrl

  const openCustom = () => {
    setSettingsScope(scope.profile || 'default')
    navigate('/settings?tab=providers&pview=custom-endpoints')
  }

  const retry = () =>
    void refreshOnboarding(
      { requestGateway, scope },
      () => JSON.stringify(captureOnboardingScope(scope.profile)) === JSON.stringify(scope)
    )

  return (
    <section className="flex flex-wrap items-center justify-between gap-2 py-2 text-xs text-(--ui-text-secondary)">
      <div className="min-w-0 flex-1">
        <p aria-live="polite" role="status">
          {copy[state]}
        </p>
        {(readiness.modelService?.detail || readiness.reason) && (
          <details>
            <summary>{copy.details}</summary>
            <p className="break-words">{readiness.modelService?.detail || readiness.reason}</p>
          </details>
        )}
      </div>
      <div className="flex flex-wrap gap-2">
        <Button
          onClick={() =>
            setup
              ? startManualProviderOAuth('semantier-cloud', scope)
              : portal
                ? void openExternalLink(portal)
                : retry()
          }
          size="sm"
        >
          {setup ? copy.signIn : portal ? copy[state] : copy.retry}
        </Button>
        <Button onClick={openCustom} size="sm" variant="text">
          {copy.custom}
        </Button>
      </div>
    </section>
  )
}
