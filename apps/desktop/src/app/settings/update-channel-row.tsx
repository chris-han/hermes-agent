import { useStore } from '@nanostores/react'
import { type ReactElement, useState } from 'react'

import { SegmentedControl } from '@/components/ui/segmented-control'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { confirm } from '@/store/confirm'
import { notifyError } from '@/store/notifications'
import {
  $updateApply,
  $updateChecking,
  $updateStatus,
  checkUpdates,
  setUpdateChannel,
  sourceUpdateChannel,
  type UpdateChannel
} from '@/store/updates'

import { ListRow } from './primitives'

/**
 * The native owner declares available packaged channels; source checkouts
 * retain their existing stable/main policy.
 */
export function UpdateChannelRow(): ReactElement | null {
  const { t } = useI18n()
  const a = t.settings.about.channel
  const status = useStore($updateStatus)
  const checking = useStore($updateChecking)
  const applying = useStore($updateApply).applying
  const packaged = status?.releaseChannel !== undefined
  // The confirmed choice paints at once; the check that follows the save can be slow.
  const [pending, setPending] = useState<UpdateChannel | null>(null)
  const saved = status?.releaseChannel ?? sourceUpdateChannel(status)
  const channel = pending ?? saved

  if (channel === null) {
    return null
  }

  const change = async (next: UpdateChannel): Promise<void> => {
    if (next === channel) {
      return
    }

    // Main → stable can land on an older commit than the one running now.
    if (
      !packaged && next === 'stable' &&
      !(await confirm({
        title: a.stableConfirmTitle,
        description: a.stableConfirmBody,
        confirmLabel: a.stableConfirm
      }))
    ) {
      return
    }

    triggerHaptic('crisp')
    setPending(next)

    try {
      await setUpdateChannel(next)
    } catch (error) {
      // A throw may still follow a save, so re-read the install instead of guessing.
      notifyError(error, a.failed)
      void checkUpdates()
    } finally {
      setPending(null)
    }
  }

  return (
    <ListRow
      action={
        <SegmentedControl
          disabled={checking || applying || pending !== null}
          onChange={id => void change(id)}
          options={packaged ? [
            { id: 'production', label: a.production },
            { id: 'development', label: a.development }
          ] : [
            { id: 'stable', label: a.stable },
            { id: 'main', label: a.main }
          ]}
          value={channel}
        />
      }
      description={packaged ? a.packagedDescription : a.description}
      title={a.title}
    />
  )
}
