import { act, createElement } from 'react'
import { createRoot } from 'react-dom/client'
import { beforeEach, expect, it, vi } from 'vitest'

import { actEngineSource, actInPage, type PreviewActHolder } from './act-in-page'

beforeEach(() => {
  vi.restoreAllMocks()
  document.body.innerHTML = ''
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockReturnValue({
    left: 0,
    top: 0,
    right: 80,
    bottom: 30,
    width: 80,
    height: 30
  } as DOMRect)
  Element.prototype.scrollIntoView = vi.fn()
  delete (document as unknown as { elementFromPoint?: unknown }).elementFromPoint
})
const scan = (holder: PreviewActHolder = {}) => actInPage(document, holder, { kind: 'elements', full: true }).elements!

it('discovers empty Alibaba-like fields from unambiguous form rows and stable identities', () => {
  document.body.innerHTML = `
    <div class="next-form-item"><label>预算名称</label><div><input></div></div>
    <div class="next-form-item"><label>预算备注</label><div><textarea></textarea></div></div>
    <input id="budget-code"><input><input>
    <div class="next-form-item"><label>Ambiguous</label><input><input></div>
    <input type="number" value="100">`
  expect(scan().map(e => e.label)).toEqual(['预算名称', '预算备注', 'Unlabeled input: budget-code', '100'])
})

it('discovers property-bound React click targets and preserves refs through replacements', () => {
  const render = () => {
    document.body.innerHTML = '<div><span>+新增</span></div><table><tbody><tr><td><a>编辑</a></td></tr></tbody></table>'

    for (const el of document.querySelectorAll<HTMLElement>('span,a')) {
      el.onclick = vi.fn()
    }
  }

  render()
  const holder: PreviewActHolder = {}
  const first = scan(holder)
  expect(first.map(e => e.label)).toEqual(['+新增', '编辑'])
  render()
  expect(scan(holder).map(e => e.ref)).toEqual(first.map(e => e.ref))
  expect(actInPage(document, holder, { kind: 'click', ref: first[1].ref }).success).toBe(true)
  expect(document.querySelector<HTMLElement>('a')!.onclick).toHaveBeenCalledOnce()
})

it('never rebinds duplicate row actions by their document order', () => {
  const render = () => {
    document.body.innerHTML = '<button>编辑</button><button>编辑</button>'
  }

  render()
  const holder: PreviewActHolder = {}
  const old = scan(holder).map(e => e.ref)
  render()
  expect(scan(holder).every(e => !old.includes(e.ref))).toBe(true)
  expect(actInPage(document, holder, { kind: 'click', ref: old[0] }).success).toBe(false)
})

it('keeps hidden and occluded property-bound controls out of the inventory', () => {
  document.body.innerHTML = '<span style="display:none">Hidden</span><span>Covered</span><div id="wall"></div>'

  for (const el of document.querySelectorAll<HTMLElement>('span')) {
    el.onclick = vi.fn()
  }

  Object.defineProperty(document, 'elementFromPoint', {
    configurable: true,
    value: () => document.getElementById('wall')
  })
  expect(scan()).toEqual([])
})

it('applies naming in each nested document without crossing frame or shadow ownership', () => {
  document.body.innerHTML = '<iframe></iframe><div id="host"></div>'
  const child = document.querySelector('iframe')!.contentDocument!
  child.body.innerHTML = '<iframe></iframe>'
  const nested = child.querySelector('iframe')!.contentDocument!
  nested.body.innerHTML = '<label>预算名称<input></label>'
  nested.querySelector('input')!.getBoundingClientRect = () =>
    ({ left: 0, top: 0, right: 80, bottom: 30, width: 80, height: 30 }) as DOMRect
  document.getElementById('host')!.attachShadow({ mode: 'open' }).innerHTML = '<button>Shadow</button>'
  const injected = new Function('return ' + actEngineSource())() as typeof actInPage
  const holder: PreviewActHolder = {}
  expect(injected(nested, holder, { kind: 'elements' }).elements![0].label).toBe('预算名称')
  expect(scan()).toEqual([])
  expect(actInPage(document, holder, { kind: 'click', ref: holder.book![0].ref }).success).toBe(false)
})

it('discovers actual React custom actions after a keyed rerender', async () => {
  const root = createRoot(document.body)
  const clicked = vi.fn()

  const render = (key: string) =>
    act(async () => root.render(createElement('span', { key, onClick: clicked }, '+新增')))

  try {
    await render('first')
    const holder: PreviewActHolder = {}
    const first = scan(holder)[0]
    expect(first.label).toBe('+新增')
    await render('replacement')
    expect(scan(holder)[0].ref).toBe(first.ref)
    await act(async () => {
      actInPage(document, holder, { kind: 'click', ref: first.ref })
    })
    expect(clicked).toHaveBeenCalledOnce()
  } finally {
    await act(async () => root.unmount())
  }
})

it('keeps structurally identified row actions stable when rows reorder', () => {
  const render = (ids: string[]) => {
    document.body.innerHTML = ids.map(id => `<a id="${id}">编辑</a>`).join('')

    for (const el of document.querySelectorAll<HTMLElement>('a')) { el.onclick = vi.fn() }
  }

  render(['row-a-edit', 'row-b-edit'])
  const holder: PreviewActHolder = {}
  const old = scan(holder)
  render(['row-b-edit', 'row-a-edit'])
  expect(scan(holder).map(e => e.ref)).toEqual([old[1].ref, old[0].ref])
})
