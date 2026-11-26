import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { activeGatewayProfileKey, requestGatewayForAgent } from '@/store/gateway'

import type {
  TutorCitation, TutorProgress, WorkspaceProject, WorkspaceProjectContinuation, WorkspaceProjectDraft,
  WorkspaceProjectHead, WorkspaceProjectMaterials, WorkspaceProjectPin, WorkspaceScope
} from '../../../electron/semantier-workspace-client'

const fields = ['goal', 'question', 'claim', 'rationale'] as const
const initial = { goal: '', question: '', claim: '', rationale: '' }

const headPin = (head: WorkspaceProjectHead): WorkspaceProjectPin => ({
  artifact_kind: 'project_semantic_head', artifact_ref: head.artifact_ref, artifact_hash: head.artifact_hash
})

function TutorCitations({ sources, locale }: { sources: TutorCitation[]; locale: string }) {
  return <>{sources.map(source => <details key={source.evidence.artifact_ref}>
    <summary>{locale === 'zh' ? '查看原文' : 'View source passage'} · {source.source_block_id}</summary>
    <blockquote className="border-l pl-3">{source.locator?.selector.exact_quote}</blockquote>
    <small className="block break-all">{source.evidence.artifact_ref} · {source.source_version}</small>
  </details>)}</>
}

function TutorResume({ origin, scope, project, plan, title }: {
  origin: string; scope: WorkspaceScope; project: string; plan: string; title: string
}) {
  const { locale } = useI18n()
  const [open, setOpen] = useState(false)
  const [progress, setProgress] = useState<TutorProgress | null>(null)
  const [failed, setFailed] = useState(false)
  const { userId, organizationId, workspaceId } = scope
  useEffect(() => {
    if (!open) {return}
    let active = true
    setFailed(false)
    setProgress(null)
    void window.hermesDesktop.workspace!.tutorProgress(origin, { userId, organizationId, workspaceId }, project, plan)
      .then(async value => {
        if (!active) {return}

        const objectives = value.objectives.map(obj => ({status: obj.status, knowledge_type: obj.knowledge_type,
          assessments: obj.assessments.map(({result, assistance, submitted_at, assessment}) => ({result, assistance, submitted_at, assessment}))}))

        const local = await requestGatewayForAgent<{mastery: {estimate: number | null}[]; release: TutorProgress['release']}>(
          'local', activeGatewayProfileKey(), 'plugins.semantier.tutor.progress',
          {project_ref: project, plan_ref: plan, release: value.release, objectives}, 20000)

        if (local.release.package_ref !== value.release.package_ref || local.mastery.length !== objectives.length)
          {throw new Error('Tutor release or progress mismatch')}

        if (active) {setProgress({...value, objectives: value.objectives.map((obj,index) => ({...obj, mastery: local.mastery[index]}))})}
      }).catch(() => { if (active) {setFailed(true)} })

    return () => { active = false }
  }, [open, origin, userId, organizationId, workspaceId, project, plan])

  return <details onToggle={event => setOpen(event.currentTarget.open)}>
    <summary>{title}</summary>
    {failed && <p role="alert">{locale === 'zh' ? '无法读取学习进度。' : 'Learning progress unavailable.'}</p>}
    {progress?.objectives.map(obj => <section key={obj.objective_ref}>
      <h3>{obj.title ?? (locale === 'zh' ? '材料不可用' : 'Material unavailable')}
        {obj.archived && (locale === 'zh' ? '（历史目标）' : ' (archived)')}</h3>
      {obj.mastery?.estimate != null && <p>{locale === 'zh' ? '掌握度估计' : 'Estimated mastery'} · {Math.round(obj.mastery.estimate * 100)}%</p>}
      {obj.lessons.map(item => <div key={item.lesson.artifact_ref}><p>{item.explanation}</p>
        <TutorCitations locale={locale} sources={item.sources} />
      </div>)}
      {obj.assessments.map(item => <div key={item.attempt_ref}><p>{item.result} · {item.feedback}</p>
        <TutorCitations locale={locale} sources={item.sources} /></div>)}
    </section>)}
  </details>
}

export function SemantierDesktopProjects({ origin, scope }: { origin: string; scope: WorkspaceScope }) {
  const { locale } = useI18n()
  const t = (en: string, zh: string) => locale === 'zh' ? zh : en
  const api = window.hermesDesktop.workspace
  const { userId, workspaceId, organizationId } = scope
  const [projects, setProjects] = useState<WorkspaceProject[]>([])
  const [projectReads, retryProjectReads] = useState(0)
  const [readFailure, setReadFailure] = useState<'projects' | 'materials' | 'scope' | null>(null)
  const [materials, setMaterials] = useState<WorkspaceProjectMaterials | null>(null)
  const [selected, select] = useState<WorkspaceProjectContinuation | null>(null)
  const [graphRef, setGraphRef] = useState('')
  const [noteRef, setNoteRef] = useState('')
  const [consent, setConsent] = useState(false)
  const [draft, setDraft] = useState<WorkspaceProjectDraft>(initial)
  const [history, setHistory] = useState<WorkspaceProjectHead | null>(null)
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState<'error' | 'conflict' | null>(null)
  const pending = useRef<{ requestId: string; signature: string } | null>(null)
  const head = selected?.head
  const visible = history ?? head
  const previous = visible?.parent
  const captions = { goal: t('Goal', '目标'), question: t('Open question', '待解决的问题'), claim: t('Decision', '决策'), rationale: t('Rationale', '理由') }

  useEffect(() => {
    if (!api) {return}
    let active = true
    const current = { userId, workspaceId, organizationId }
    function failRead(error: unknown, stage: 'projects' | 'materials') {
      if (!active) {return}
      const message = error instanceof Error ? error.message : ''
      if (/account or workspace changed|\((401|403)\)/i.test(message)) {
        setProjects([])
        setMaterials(null)
        select(null)
        setHistory(null)
        setDraft(initial)
        setReadFailure('scope')
      } else {
        setReadFailure(stage)
      }
    }

    setReadFailure(null)
    void api.projects(origin, current).then(async rows => {
      if (!active) {return}
      setProjects(rows)
      try {
        const choices = await api.projectMaterials(origin, current)
        if (active) {setMaterials(choices)}
      } catch (error) {
        failRead(error, 'materials')
      }
    }).catch(error => failRead(error, 'projects'))

    return () => { active = false }
  }, [api, origin, userId, workspaceId, organizationId, projectReads])

  async function run(operation: () => Promise<void>) {
    setBusy(true)
    setStatus(null)

    try {
      await operation()
    } catch (error) {
      const message = error instanceof Error ? error.message : ''

      if (message.includes('PROJECT_HEAD_CONFLICT')) {
        setStatus('conflict')
      } else {
        if (/changed|\((401|403|404)\)/i.test(message)) {
          select(null)
          setHistory(null)
          setDraft(initial)
          setProjects([])
          setMaterials(null)
          pending.current = null
        }

        setStatus('error')
      }
    } finally {
      setBusy(false)
    }
  }

  async function load(ref: string, keepDraft = false) {
    const project = await api!.continueProject(origin, scope, ref)
    select(project)
    setHistory(null)

    if (!keepDraft) {
      setDraft(project.head ?? { ...initial, goal: project.working_context.task_goal })
      pending.current = null
    }
  }

  function create() {
    const graph = materials?.graphs.find(item => item.pin.artifact_ref === graphRef)

    if (!api || !graph || !consent) {return}
    void run(async () => {
      const project = await api.createProject(origin, scope, graph.pin, true)
      setProjects(rows => [...rows, project])
      await load(project.project_ref)
      setConsent(false)
    })
  }

  function save() {
    const note = head?.note ?? materials?.notes.find(item => item.pin.artifact_ref === noteRef)?.pin

    if (!api || !selected || !note || fields.some(key => !draft[key].trim()) || status === 'conflict') {return}

    const input = { ...Object.fromEntries(fields.map(key => [key, draft[key]])) as WorkspaceProjectDraft,
      expected_head: head ? headPin(head) : null, note }

    const signature = JSON.stringify([selected.project.project_ref, input])
    const requestId = pending.current?.signature === signature ? pending.current.requestId : crypto.randomUUID()
    pending.current = { requestId, signature }
    void run(async () => {
      const saved = await api.commitProject(origin, scope, selected.project.project_ref, { ...input, request_id: requestId })
      setDraft(saved)
      select({ ...selected, head: saved })
      pending.current = null
    })
  }

  if (!api) {return null}

  return (
    <section aria-label={t('Project continuity', '项目连续性')} className="grid gap-3">
      {status && <p className="text-xs text-destructive" role="alert">
        {status === 'conflict' ? t('Another edit advanced this project. Load the latest decision before saving.', '项目已有更新。请加载最新决策后再保存。') : t('Project unavailable. Check the Workspace account and try again.', '项目不可用。请检查工作区账户后重试。')}
      </p>}
      {busy && <p className="text-xs" role="status">{t('Updating project…', '正在更新项目…')}</p>}
      {readFailure && <div className="text-xs text-destructive" role="alert">
        {readFailure === 'materials'
          ? t('Project materials unavailable. You can still open an existing project.', '项目材料不可用。你仍可打开现有项目。')
          : t('Projects unavailable. Check the Workspace account and try again.', '项目不可用。请检查工作区账户后重试。')}
        <Button onClick={() => { pending.current = null; retryProjectReads(value => value + 1) }} type="button" variant="outline">{t('Retry', '重试')}</Button>
      </div>}
      {!materials && !status && !readFailure && <p className="text-xs" role="status">{t('Loading projects…', '正在加载项目…')}</p>}
      {materials && !materials.graphs.length && !projects.length && <p className="text-xs">{t('No admitted project context is available.', '没有可用的已接纳项目上下文。')}</p>}
      {projects.length > 0 && <label className="grid gap-1 text-xs">
        {t('Project', '项目')}
        <select disabled={busy} onChange={event => void run(() => load(event.target.value))} value={selected?.project.project_ref ?? ''}>
          <option disabled value="">{t('Choose a project', '选择项目')}</option>
          {projects.map(project => <option key={project.project_ref}>{project.project_ref}</option>)}
        </select>
      </label>}
      {!!materials?.graphs.length && <details>
        <summary className="cursor-pointer text-sm">{t('Start a project', '开始项目')}</summary>
        <div className="mt-2 grid gap-2">
          <label className="grid gap-1 text-xs">{t('Admitted context', '已接纳的上下文')}
            <select disabled={busy} onChange={event => { setGraphRef(event.target.value); setConsent(false) }} value={graphRef}>
              <option disabled value="">{t('Choose the admitted context', '选择已接纳的上下文')}</option>
              {materials.graphs.map(item => <option key={item.pin.artifact_ref} value={item.pin.artifact_ref}>{item.label}</option>)}
            </select>
          </label>
          <label className="flex items-center gap-2 text-xs">
            <input checked={consent} disabled={!graphRef || busy} onChange={event => setConsent(event.target.checked)} type="checkbox" />
            {t('Share this synthetic context in the Workspace project.', '在工作区项目中共享此合成上下文。')}
          </label>
          <div><Button disabled={!consent || busy} onClick={create} type="button">{t('Create project', '创建项目')}</Button></div>
        </div>
      </details>}
      {selected && <>
        {!!selected.tutor_plans?.length && <section>
          <h2>{t('Learning plans', '学习计划')}</h2>
          {selected.tutor_plans.map(plan => <TutorResume key={plan.plan_ref} origin={origin} plan={plan.plan_ref}
            project={selected.project.project_ref} scope={scope} title={plan.goal} />)}
        </section>}
        <div className="flex flex-wrap gap-2">
          <Button disabled={busy} onClick={() => void run(() => load(selected.project.project_ref, true))} type="button" variant="outline">{t('Load latest decision', '加载最新决策')}</Button>
          {previous && <Button disabled={busy} onClick={() => void run(async () => {
            setHistory((await api.replayProject(origin, scope, selected.project.project_ref, previous)).head)
          })} type="button" variant="outline">{t('Previous decision', '上一条决策')}</Button>}
        </div>
        {visible && <div className="grid gap-1 text-xs">
          <h3 className="font-semibold">{history ? t('Historical decision', '历史决策') : t('Current decision', '当前决策')} · {visible.revision}</h3>
          {fields.map(key => <p className="break-words" key={key}><strong>{captions[key]}</strong> {visible[key]}</p>)}
          <details><summary>{t('Note and evidence pins', '笔记与证据引用')}</summary>
            {[visible.note, visible.graph, ...visible.evidence, visible.decision].map(pin => <p className="break-all" key={pin.artifact_ref}>{pin.artifact_ref}</p>)}
          </details>
        </div>}
        {!history && <>
          {!head && <label className="grid gap-1 text-xs">{t('Authored Note', '已撰写的笔记')}
            <select disabled={busy} onChange={event => setNoteRef(event.target.value)} value={noteRef}>
              <option disabled value="">{t('Choose the Note', '选择笔记')}</option>
              {materials?.notes.map(item => <option key={item.pin.artifact_ref} value={item.pin.artifact_ref}>{item.label}</option>)}
            </select>
          </label>}
          {fields.map(key => <label className="grid gap-1 text-xs" key={key}>{captions[key]}
            <Input disabled={busy} maxLength={key === 'goal' || key === 'question' ? 2000 : 8000} onChange={event => setDraft({ ...draft, [key]: event.target.value })}
              value={draft[key]} />
          </label>)}
          <div><Button disabled={busy || status === 'conflict' || (!head && !noteRef) || fields.some(key => !draft[key].trim())} onClick={save} type="button">{t('Save decision', '保存决策')}</Button></div>
        </>}
      </>}
    </section>
  )
}
