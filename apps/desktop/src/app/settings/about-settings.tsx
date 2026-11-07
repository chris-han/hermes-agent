import { useStore } from '@nanostores/react'
import { type ReactElement, useEffect } from 'react'

import { SemantierAttribution } from '@/components/organization-brand'
import { SemantierFeedbackSettings } from '@/components/semantier-feedback'
import { UpdateStatusCard, VersionHero } from '@/components/update-status'
import { VersionDetails } from '@/components/version-details'
import { useI18n } from '@/i18n'
import { RefreshCw } from '@/lib/icons'
import { $connection } from '@/store/session'
import { $desktopVersion, $updateStatus, checkBackendUpdates, refreshDesktopVersion } from '@/store/updates'

import { SectionHeading, SettingsContent } from './primitives'
import { SETTING_IDS, settingElementId } from './settings-manifest'
import { UninstallSection } from './uninstall-section'
import { UpdateChannelRow } from './update-channel-row'
import { UpdateDeliverySourceRow } from './update-delivery-source-row'
import { useSettingDeepLink } from './use-setting-deep-link'

interface AboutSettingsProps {
  subpage?: string
}

export function AboutSettings({ subpage }: AboutSettingsProps = {}): ReactElement {
  useSettingDeepLink('about', page => subpage === undefined || page === subpage)

  if (subpage === 'uninstall') {
    return (
      <SettingsContent>
        <UninstallSection />
      </SettingsContent>
    )
  }

  return <AppUpdatesSettings includeUninstall={subpage === undefined} />
}

interface AppUpdatesSettingsProps {
  includeUninstall: boolean
}

function AppUpdatesSettings({ includeUninstall }: AppUpdatesSettingsProps): ReactElement {
  const { t } = useI18n()
  const version = useStore($desktopVersion)
  const update = useStore($updateStatus)
  const { locale } = useI18n()
  const connection = useStore($connection)
  const remote = connection?.mode === 'remote'

  // Refresh the running version when About opens or the active gateway changes.
  useEffect((): void => {
    void refreshDesktopVersion()

    if (remote) {
      void checkBackendUpdates()
    }
  }, [connection, remote])

  return (
    <SettingsContent>
      <VersionHero version={version} />
      <div className="text-center">
        <p className="text-xs text-muted-foreground">
          {t.branding.organizationId}: {t.branding.unbound}
        </p>
        <SemantierAttribution />
      </div>
      <div className="mx-auto mt-4 w-full max-w-2xl">
        <SectionHeading icon={RefreshCw} title={t.settings.about.updates} />
        <div className="grid gap-3" id={settingElementId(SETTING_IDS.about.updates)}>
          <UpdateDeliverySourceRow />
          <UpdateStatusCard target="client" />
          {update?.releaseNotes && (
            <details>
              <summary>{t.updates.releaseNotes}</summary>
              <pre className="whitespace-pre-wrap font-sans">
                {update.releaseNotes[locale.startsWith('zh') ? 'zh' : 'en'] ?? Object.values(update.releaseNotes)[0]}
              </pre>
            </details>
          )}
          <UpdateChannelRow />
          {/* Client and remote backend updates are independent. Only the client has release notes. */}
          {remote && <UpdateStatusCard showReleaseNotes={false} target="backend" />}
        </div>
        {version && <VersionDetails version={version} />}
        <SemantierFeedbackSettings />
        {includeUninstall && <UninstallSection />}
      </div>
    </SettingsContent>
  )
}
