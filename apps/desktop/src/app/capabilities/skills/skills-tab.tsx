import { compactNumber } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { type ReactNode, useCallback, useMemo, useRef, useState } from 'react'

import { ArchiveSkillConfirmDialog } from '@/app/learning/archive-skill-confirm-dialog'
import { CodeEditor } from '@/components/chat/code-editor'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  editLearningNode,
  getLearningNode,
  getOfficialSkills,
  type ProfileScope,
  profileScopeKey,
  setSkillEnabled
} from '@/hermes'
import { useI18n } from '@/i18n'
import { useInventoryCopy } from '@/i18n/capability-inventory'
import { Loader2 } from '@/lib/icons'
import { Codecs, persistentAtom } from '@/lib/persisted'
import { queryClient } from '@/lib/query-client'
import { invalidateSlashCompletions } from '@/lib/slash-completion-cache'
import { useStoreSelector } from '@/lib/use-session-slice'
import {
  $hubActions,
  hubActionKey,
  installHubSkill,
  notifyHubActionFailed,
  OFFICIAL_SKILLS_KEY
} from '@/store/hub-actions'
import { notify, notifyError } from '@/store/notifications'
import type { OfficialSkillInfo, SkillInfo } from '@/types/hermes'

import {
  DetailColumn,
  DetailPane,
  ListColumn,
  ListStrip,
  ListStripMenu,
  type ListStripMenuToggle,
  MasterDetail
} from '../../master-detail'
import { prettyName } from '../../settings/helpers'
import { InventoryRow as CapRow, InventoryItems, InventorySelect } from '../inventory'
import { CapabilityEmpty, SortButton } from '../primitives'

import { OfficialSkillDetail } from './official-skill-detail'
import { SkillDetail } from './skill-detail'
import { categoryFor, filteredOfficial, filteredSkills, skillsQueryKey, usageOf } from './skills-data'

// Sort direction for the Skills list — persisted so the tab remembers
// most/least-used across navigations and restarts.
const $skillsSortDesc = persistentAtom('hermes.desktop.capabilities.skillsSortDesc', true, Codecs.bool)

// Row subtitle: category, with non-default origins badged.
function skillSubtitle(skill: SkillInfo, learned: string): ReactNode {
  const category = prettyName(categoryFor(skill))
  const provenance = skill.provenance

  return (
    <>
      <span className="truncate">{category}</span>
      {provenance === 'agent' && (
        <Badge className="shrink-0 normal-case" variant="default">
          {learned}
        </Badge>
      )}
      {provenance === 'hub' && (
        <Badge className="shrink-0 normal-case" variant="muted">
          hub
        </Badge>
      )}
    </>
  )
}

interface SkillsTabProps {
  /** The scope's skill list, straight from the shell's query. */
  skills: SkillInfo[]
  /** The (connection, profile) scope every read and write routes to. */
  profile: ProfileScope
  query: string
  initialSelected?: string
  /** Page-level refresh: a saved skill edit reloads the same way the refresh
   *  hotkey does, counts and slash completions included. */
  onRefresh: () => void
}

/** The Skills tab: installed skills, official optional skills, and learned-skill editing. */
export function SkillsTab({ onRefresh, profile, query, skills, initialSelected }: SkillsTabProps) {
  const { t } = useI18n()
  const c = useInventoryCopy()
  const [source, setSource] = useState('all')
  const [status, setStatus] = useState('all')
  const [sort, setSort] = useState('usage')
  const [category, setCategory] = useState('all')
  const skillsSortDesc = useStore($skillsSortDesc)
  const [bulkBusy, setBulkBusy] = useState(false)
  const pendingWrites = useRef(new Set<string>())
  const [pending, setPending] = useState<Set<string>>(new Set())
  const [selectedSkill, setSelectedSkill] = useState<string | null>(initialSelected ?? null)
  const [selectedOfficial, setSelectedOfficial] = useState<string | null>(null)

  const { data: officialData } = useQuery({
    queryKey: [...OFFICIAL_SKILLS_KEY, profileScopeKey(profile)],
    queryFn: () => getOfficialSkills(profile),
    staleTime: 60_000,
    retry: false
  })

  // Learned/local skills are editable + archivable, mirroring the memory
  // graph (same /api/learning/node endpoints — delete archives, restorable
  // via `hermes curator restore`).
  const [skillEditor, setSkillEditor] = useState<null | { content: string; name: string }>(null)
  const [skillDraft, setSkillDraft] = useState('')
  const [skillSaving, setSkillSaving] = useState(false)
  const [archiveTarget, setArchiveTarget] = useState<null | string>(null)

  // Optimistic write-through against the scoped Skills key: toggles/bulk/
  // archive repaint instantly; the next background refetch reconciles.
  const setSkills = useCallback(
    (fn: (cur: SkillInfo[] | undefined) => SkillInfo[] | undefined) =>
      queryClient.setQueryData<SkillInfo[]>(skillsQueryKey(profile), prev => fn(prev) ?? prev),
    [profile]
  )

  const visibleSkills = useMemo(
    () =>
      filteredSkills(skills, query, skillsSortDesc)
        .filter(
          row =>
            (category === 'all' || categoryFor(row) === category) &&
            (source === 'all' || row.provenance === source) &&
            (status === 'all' || row.enabled === (status === 'enabled'))
        )
        .sort((a, b) => (sort === 'name' ? a.name.localeCompare(b.name) : 0)),
    [query, skills, skillsSortDesc, source, status, sort, category]
  )

  // Installed-name set stays unfiltered so search cannot make a skill look absent.
  const installedSkillNames = useMemo(() => new Set(skills.map(s => s.name)), [skills])

  const visibleOfficial = useMemo(() => {
    const catalog = (officialData?.skills ?? []).filter(
      skill => !skill.installed && !installedSkillNames.has(skill.name)
    )

    return filteredOfficial(catalog, query)
  }, [installedSkillNames, officialData, query])

  const runningInstallKey = useStoreSelector($hubActions, actions =>
    Object.keys(actions)
      .filter(key => actions[key]?.running)
      .sort()
      .join('|')
  )

  const runningInstalls = useMemo(() => new Set(runningInstallKey.split('|').filter(Boolean)), [runningInstallKey])

  // Keep a valid selection: fall back to the first visible row when the
  // current selection is filtered out (or nothing is selected yet).
  const activeSkill = useMemo(
    () => visibleSkills.find(s => s.name === selectedSkill) ?? visibleSkills[0] ?? null,
    [selectedSkill, visibleSkills]
  )

  const activeOfficial = useMemo(
    () => visibleOfficial.find(skill => skill.identifier === selectedOfficial) ?? null,
    [selectedOfficial, visibleOfficial]
  )

  function handleInstallOfficial(skill: OfficialSkillInfo) {
    notify({ kind: 'success', title: t.skills.hub.installStarted(skill.name), message: t.skills.hub.actionLog })
    void installHubSkill(skill.identifier, profile).catch(err =>
      notifyHubActionFailed(err, t.skills.hub.actionFailed, skill.name, profile)
    )
  }

  async function handleToggleSkill(skill: SkillInfo, enabled: boolean) {
    if (pendingWrites.current.has(skill.name)) {
      return
    }

    pendingWrites.current.add(skill.name)
    setPending(new Set(pendingWrites.current))
    const previous = { ...skill }
    setSkills(current => current?.map(row => (row.name === skill.name ? { ...row, enabled } : row)) ?? current)

    try {
      await setSkillEnabled(skill.name, enabled, profile)
      // A disabled skill loses its `/name` command, so the composer's cached
      // `/` list has to be dropped along with the row repaint.
      invalidateSlashCompletions()
    } catch (err) {
      setSkills(
        current =>
          current?.map(row => (row.name === skill.name ? { ...row, enabled: previous.enabled } : row)) ?? current
      )
      notifyError(err, t.skills.failedToUpdate(skill.name))
    } finally {
      pendingWrites.current.delete(skill.name)
      setPending(new Set(pendingWrites.current))
    }
  }

  // Sequential on purpose: each toggle is a config read-modify-write on the
  // backend; parallel calls would race the disabled-list save.
  async function bulkApply(targets: SkillInfo[], enabled: boolean) {
    if (bulkBusy || pendingWrites.current.size > 0 || targets.length === 0) {
      return
    }

    setBulkBusy(true)

    let done = 0

    try {
      for (const row of targets) {
        await setSkillEnabled(row.name, enabled, profile)
        setSkills(cur => cur?.map(r => (r.name === row.name ? { ...r, enabled } : r)) ?? cur)
        done += 1
      }

      notify({ kind: 'success', title: t.skills.bulkUpdated(done), message: '' })
    } catch (err) {
      notifyError(err, t.skills.failedToUpdate(t.skills.tabSkills))
    } finally {
      invalidateSlashCompletions()
      setBulkBusy(false)
    }
  }

  // Bulk actions ("All" master switch, "Disable unused") and the master-switch
  // state target the WHOLE tab, never the search-filtered view — a tab-wide
  // control that silently scoped to the current query would be a lie.
  const allEnabled = skills.length > 0 && skills.every(s => s.enabled)

  // One switch line covering enable-all/disable-all.
  const bulkSwitch: ListStripMenuToggle = {
    checked: allEnabled,
    disabled: bulkBusy || pending.size > 0,
    label: t.skills.all,
    onToggle: checked =>
      void bulkApply(
        skills.filter(row => row.enabled !== checked),
        checked
      )
  }

  // "Never used" = zero recorded activity. The pruning move for a 100+ skill
  // install: keep the workhorses, shed the noise.
  const disableUnused = () =>
    bulkApply(
      skills.filter(skill => skill.enabled && usageOf(skill) === 0),
      false
    )

  const openSkillEditor = async (name: string) => {
    try {
      const node = await getLearningNode(name, profile)

      setSkillEditor({ content: node.content, name })
      setSkillDraft(node.content)
    } catch (err) {
      notifyError(err, name)
    }
  }

  const saveSkillEdit = async () => {
    if (!skillEditor || skillSaving) {
      return
    }

    setSkillSaving(true)

    try {
      await editLearningNode(skillEditor.name, skillDraft, profile)
      notify({
        kind: 'success',
        title: t.skills.skillUpdated,
        message: t.skills.appliesToNewSessions(skillEditor.name)
      })
      setSkillEditor(null)
      onRefresh()
    } catch (err) {
      notifyError(err, skillEditor.name)
    } finally {
      setSkillSaving(false)
    }
  }

  const skillEditorPane = skillEditor && (
    <DetailPane
      actions={
        <Button disabled={skillSaving} onClick={() => void saveSkillEdit()} size="xs">
          {skillSaving ? t.common.saving : t.common.save}
        </Button>
      }
      id="skill-editor"
      onClose={() => setSkillEditor(null)}
      title={<span className="text-[0.68rem] font-normal text-muted-foreground/60">{skillEditor.name}/SKILL.md</span>}
    >
      <CodeEditor
        disabled={skillSaving}
        filePath="SKILL.md"
        initialValue={skillEditor.content}
        key={skillEditor.name}
        onCancel={() => setSkillEditor(null)}
        onChange={setSkillDraft}
        onSave={() => void saveSkillEdit()}
      />
    </DetailPane>
  )

  return (
    <>
      {skills.length === 0 && visibleOfficial.length === 0 ? (
        <CapabilityEmpty noun="skills" query={query} />
      ) : (
        <MasterDetail pane={skillEditorPane} resizeId="capabilities-split" split="wide">
          <ListColumn
            header={
              <ListStrip
                left={<SortButton desc={skillsSortDesc} onFlip={() => $skillsSortDesc.set(!$skillsSortDesc.get())} />}
                right={
                  <ListStripMenu
                    items={[
                      {
                        disabled: bulkBusy || pending.size > 0,
                        label: t.skills.disableUnused,
                        onSelect: () => void disableUnused()
                      }
                    ]}
                    label={t.skills.tabSkills}
                    toggle={bulkSwitch}
                  />
                }
              />
            }
          >
            <div className="flex flex-wrap gap-2 pb-2">
              <InventorySelect
                label={c.source}
                onChange={setSource}
                options={[
                  { value: 'all', label: c.all },
                  ...[...new Set(skills.map(row => row.provenance))]
                    .filter((value): value is NonNullable<SkillInfo['provenance']> => Boolean(value))
                    .map(value => ({
                      value,
                      label:
                        value === 'agent'
                          ? c.learned
                          : value === 'bundled'
                            ? c.builtin
                            : value === 'external'
                              ? c.external
                              : c.hub
                    }))
                ]}
                value={source}
              />
              <InventorySelect
                label={c.status}
                onChange={setStatus}
                options={[
                  { value: 'all', label: c.status },
                  { value: 'enabled', label: c.enabled },
                  { value: 'disabled', label: c.disabled }
                ]}
                value={status}
              />
              <InventorySelect
                label={c.name}
                onChange={setSort}
                options={[
                  { value: 'usage', label: c.usage },
                  { value: 'name', label: c.name }
                ]}
                value={sort}
              />
              <InventorySelect
                label={c.category}
                onChange={setCategory}
                options={[
                  { value: 'all', label: c.category },
                  ...[...new Set(skills.map(categoryFor))].sort().map(value => ({ value, label: prettyName(value) }))
                ]}
                value={category}
              />
            </div>
            {[true, false].map(learned => {
              const group = visibleSkills.filter(skill => (skill.provenance === 'agent') === learned)

              return group.length ? (
                <section key={String(learned)}>
                  <h2 className="px-2 pt-3 pb-1 text-xs font-medium">{learned ? c.learned : c.other}</h2>
                  <InventoryItems>
                    {group.map(skill => (
                      <CapRow
                        active={activeOfficial === null && activeSkill?.name === skill.name}
                        busy={bulkBusy || pending.has(skill.name)}
                        description={skill.description}
                        enabled={skill.enabled}
                        key={skill.name}
                        meta={usageOf(skill) > 0 ? `×${compactNumber(usageOf(skill))}` : undefined}
                        onSelect={() => {
                          setSelectedSkill(skill.name)
                          setSelectedOfficial(null)
                        }}
                        onToggle={enabled => void handleToggleSkill(skill, enabled)}
                        subtitle={skillSubtitle(skill, c.learnedItem)}
                        title={skill.name}
                        toggleLabel={skill.name}
                      />
                    ))}
                  </InventoryItems>
                </section>
              ) : null
            })}
            {visibleOfficial.length > 0 && (
              <div className="flex h-7 shrink-0 items-end px-2 pb-1 text-[0.62rem] font-medium uppercase tracking-wide text-(--ui-text-quaternary)">
                {t.skills.officialCatalog}
              </div>
            )}
            {visibleOfficial.map(skill => {
              const installing = runningInstalls.has(hubActionKey(skill.identifier, profile))

              return (
                <CapRow
                  action={
                    <Button disabled={installing} onClick={() => handleInstallOfficial(skill)} size="xs" variant="text">
                      {installing && <Loader2 className="size-3 animate-spin" />}
                      {installing ? t.skills.hub.installing : t.skills.hub.install}
                    </Button>
                  }
                  active={activeOfficial?.identifier === skill.identifier}
                  enabled={false}
                  key={skill.identifier}
                  onSelect={() => setSelectedOfficial(skill.identifier)}
                  subtitle={prettyName(skill.category)}
                  title={skill.name}
                />
              )
            })}
          </ListColumn>
          <DetailColumn footer={t.skills.changesApplyNewSessions}>
            {activeOfficial ? (
              <OfficialSkillDetail
                installing={runningInstalls.has(hubActionKey(activeOfficial.identifier, profile))}
                onInstall={() => handleInstallOfficial(activeOfficial)}
                profile={profile}
                skill={activeOfficial}
              />
            ) : (
              activeSkill && (
                <SkillDetail
                  onArchive={() => setArchiveTarget(activeSkill.name)}
                  onEdit={() => void openSkillEditor(activeSkill.name)}
                  profile={profile}
                  skill={activeSkill}
                />
              )
            )}
          </DetailColumn>
        </MasterDetail>
      )}
      {archiveTarget && (
        <ArchiveSkillConfirmDialog
          onApply={() => {
            const name = archiveTarget
            const archived = skills.find(skill => skill.name === name)
            const position = skills.findIndex(skill => skill.name === name)

            setSkills(current => current?.filter(skill => skill.name !== name) ?? current)
            invalidateSlashCompletions()

            if (skillEditor?.name === name) {
              setSkillEditor(null)
            }

            return () =>
              setSkills(current => {
                if (!archived || !current || current.some(skill => skill.name === name)) {
                  return current
                }

                const restored = [...current]
                restored.splice(Math.max(0, position), 0, archived)

                return restored
              })
          }}
          onClose={() => setArchiveTarget(null)}
          onFailure={(err, name) => notifyError(err, name)}
          open
          profile={profile}
          skillId={archiveTarget}
          skillName={archiveTarget}
        />
      )}
    </>
  )
}
