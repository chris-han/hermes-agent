import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import type { DesktopUpdateApplyResult } from '@/global'
import { useI18n } from '@/i18n'
import { $updateWorkReview, applyUpdates } from '@/store/updates'

export function UpdateWorkReviewDialog({ review }: { review: NonNullable<DesktopUpdateApplyResult['workReview']> }) {
  const { t } = useI18n()
  const copy = t.semantierFeedback

  return <ConfirmDialog
    cancelLabel={copy.updateWorkLater}
    confirmLabel={copy.updateWorkConfirm}
    description={`${review.currentVersion} → ${review.targetVersion}\n\n${review.stopsWork ? copy.updateWorkOwned : copy.updateWorkAttached}`}
    dismissOnConfirm
    onClose={() => {
      if ($updateWorkReview.get()?.token === review.token) { $updateWorkReview.set(undefined) }
    }}
    onConfirm={async () => {
      const result = await applyUpdates({ workReviewToken: review.token })

      if (result.workReview) { throw new Error(copy.updateWorkChanged) }

      if (!result.ok) { throw new Error(result.message || copy.updateWorkChanged) }
    }}
    open
    title={copy.updateWorkTitle}
  >
    <p>{copy.updateWorkCount}: {review.work.count}</p>
    <ul className="break-words text-sm text-muted-foreground">{review.work.titles.slice(0, 4).map(title => <li key={title}>{title}</li>)}</ul>
  </ConfirmDialog>
}

