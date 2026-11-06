import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { stubResizeObserver } from '@/test/jsdom'
import type { ConfigFieldSchema } from '@/types/hermes'

import { ConfigField } from './config-field'
import { rankSearchOption, SearchableSelect } from './searchable-select'

beforeAll(() => {
  stubResizeObserver()
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('rankSearchOption', () => {
  it('ranks a final-segment match above a mid-path match', () => {
    // "york" hits the city segment of America/New_York (score 2) but only a
    // mid-path segment of America/New_York/Special (score 1).
    expect(rankSearchOption('America/New_York', 'york')).toBe(2)
    expect(rankSearchOption('America/New_York/Special', 'york')).toBe(1)
    expect(rankSearchOption('America/New_York', 'york')).toBeGreaterThan(
      rankSearchOption('America/New_York/Special', 'york')
    )
  })

  it('is case-insensitive', () => {
    expect(rankSearchOption('Asia/Kolkata', 'KOLKATA')).toBe(2)
    expect(rankSearchOption('ASIA/KOLKATA', 'kolkata')).toBe(2)
  })

  it('scores a slashless option by plain substring', () => {
    expect(rankSearchOption('UTC', 'ut')).toBe(1)
    expect(rankSearchOption('UTC', 'xyz')).toBe(0)
  })
})

describe('SearchableSelect', () => {
  const options = ['America/New_York', 'Asia/Kolkata', 'Europe/Berlin', 'UTC']

  it('filters the complete loaded list and restores it when the query is cleared', () => {
    const options = Array.from({ length: 650 }, (_, index) => `vendor/model-${String(index).padStart(3, '0')}`)
    render(<SearchableSelect aria-label="Model" onChange={vi.fn()} options={options} value="" />)
    fireEvent.click(screen.getByRole('combobox', { name: 'Model' }))
    const input = screen.getByPlaceholderText('Search…')
    expect(screen.getAllByRole('option')).toHaveLength(650)
    fireEvent.change(input, { target: { value: 'MODEL-649' } })
    expect(screen.getAllByRole('option')).toHaveLength(1)
    expect(screen.getByRole('option').textContent).toContain('model-649')
    fireEvent.change(input, { target: { value: 'no-such-model' } })
    expect(screen.getByText('No results found.')).toBeTruthy()
    fireEvent.change(input, { target: { value: '' } })
    expect(screen.getAllByRole('option')).toHaveLength(650)
  })

  it('searches labels and stable ids, and selects the correct id when labels match', () => {
    const onChange = vi.fn()
    render(<SearchableSelect aria-label="Provider" onChange={onChange} options={[
      { value: 'east-provider', label: 'Example' },
      { value: 'west-provider', label: 'Example' }
    ]} value="" />)
    fireEvent.click(screen.getByRole('combobox', { name: 'Provider' }))
    fireEvent.change(screen.getByPlaceholderText('Search…'), { target: { value: 'WEST' } })
    fireEvent.click(screen.getByText('Example'))
    expect(onChange).toHaveBeenCalledWith('west-provider')
  })

  it('keeps a disabled selected option visible but cannot select it', () => {
    const onChange = vi.fn()
    render(<SearchableSelect aria-label="Model" onChange={onChange} options={[
      { value: 'old', label: 'Old model', disabled: true },
      { value: 'new', label: 'New model' }
    ]} value="old" />)
    fireEvent.click(screen.getByRole('combobox', { name: 'Model' }))
    const old = screen.getAllByText('Old model').at(-1)!
    fireEvent.click(old)
    expect(onChange).not.toHaveBeenCalled()
    fireEvent.click(screen.getByText('New model'))
    expect(onChange).toHaveBeenCalledWith('new')
  })

  it('selects a filtered option with Enter', () => {
    const onChange = vi.fn()
    render(<SearchableSelect aria-label="Provider" onChange={onChange} options={[
      { value: 'alpha', label: 'Alpha service' },
      { value: 'beta', label: 'Beta service' }
    ]} value="" />)
    fireEvent.click(screen.getByRole('combobox', { name: 'Provider' }))
    const search = screen.getByPlaceholderText('Search…')
    fireEvent.change(search, { target: { value: 'BETA' } })
    fireEvent.keyDown(search, { key: 'Enter' })
    expect(onChange).toHaveBeenCalledWith('beta')
  })

  it('opens, filters, and selects an option', () => {
    const onChange = vi.fn()

    render(<SearchableSelect onChange={onChange} options={options} placeholder="Search…" value="" />)

    fireEvent.click(screen.getByRole('combobox'))
    fireEvent.change(screen.getByPlaceholderText('Search…'), { target: { value: 'kolkata' } })
    fireEvent.click(screen.getByText('Asia/Kolkata'))

    expect(onChange).toHaveBeenCalledWith('Asia/Kolkata')
  })

  it('renders the clear item when clearLabel is set and selecting it resets to blank', () => {
    const onChange = vi.fn()

    render(<SearchableSelect clearLabel="System default" onChange={onChange} options={options} value="Asia/Kolkata" />)

    fireEvent.click(screen.getByRole('combobox'))
    fireEvent.click(screen.getByText('System default'))

    expect(onChange).toHaveBeenCalledWith('')
  })

  it('omits the clear item without clearLabel', () => {
    render(<SearchableSelect onChange={vi.fn()} options={options} value="" />)

    fireEvent.click(screen.getByRole('combobox'))

    expect(screen.queryByText('System default')).toBeNull()
  })

  // #99751: the settings action cell shrink-wraps to content, and the popover
  // floors its width at the trigger's (--radix-popover-trigger-width). Without
  // a positive min-width floor on the trigger, a blank value collapses the
  // whole control — trigger AND list — to the "Search…" placeholder (~70px);
  // controlVariants' `min-w-0` is no floor at all. jsdom does not compute
  // Tailwind layout, so the class contract is the assertable unit.
  it('keeps a positive min-width floor on the trigger when the value is blank', () => {
    for (const value of ['', 'Asia/Kolkata']) {
      const { unmount } = render(<SearchableSelect onChange={vi.fn()} options={options} value={value} />)

      const classes = screen
        .getByRole('combobox')
        .className.split(/\s+/)
        .filter(cls => /^min-w-(?!0$)\d/.test(cls))

      expect(classes.length).toBeGreaterThan(0)
      unmount()
    }
  })
})

describe('ConfigField searchable routing', () => {
  const searchableSchema: ConfigFieldSchema = {
    type: 'select',
    searchable: true,
    clearable: true,
    options: ['America/New_York', 'UTC']
  }

  it('routes searchable select schemas to SearchableSelect, not a free-text input', () => {
    const { container } = render(
      <ConfigField onChange={vi.fn()} schema={searchableSchema} schemaKey="timezone" value="UTC" />
    )

    // The searchable trigger renders; the generic free-text <Input> does not.
    expect(container.querySelector('[data-slot="searchable-select-trigger"]')).not.toBeNull()
    expect(container.querySelector('input[type="text"]')).toBeNull()
  })

  it('keeps plain string schemas on the free-text input', () => {
    const { container } = render(
      <ConfigField onChange={vi.fn()} schema={{ type: 'string' }} schemaKey="some.other.key" value="hello" />
    )

    expect(container.querySelector('[data-slot="searchable-select-trigger"]')).toBeNull()
    expect(screen.getByDisplayValue('hello')).not.toBeNull()
  })

  it('surfaces the clear item via schema.clearable and resets to blank', () => {
    const onChange = vi.fn()

    render(<ConfigField onChange={onChange} schema={searchableSchema} schemaKey="timezone" value="UTC" />)

    fireEvent.click(screen.getByRole('combobox'))
    fireEvent.click(screen.getByText('System default'))

    expect(onChange).toHaveBeenCalledWith('')
  })
})

describe('ConfigField list editing', () => {
  const listSchema: ConfigFieldSchema = { type: 'list' }

  it('keeps unfinished comma input as a draft until blur commits it', () => {
    const onChange = vi.fn()

    render(<ConfigField onChange={onChange} schema={listSchema} schemaKey="test.list" value={['first']} />)

    const input = screen.getByRole('textbox') as HTMLInputElement
    fireEvent.focus(input)
    fireEvent.change(input, { target: { value: 'first,' } })

    expect(input.value).toBe('first,')
    expect(onChange).not.toHaveBeenCalled()

    fireEvent.blur(input)

    expect(onChange).toHaveBeenCalledTimes(1)
    expect(onChange).toHaveBeenCalledWith(['first'])
    expect(input.value).toBe('first')
  })

  it('commits exactly once when Enter blurs the list field', () => {
    const onChange = vi.fn()

    render(<ConfigField onChange={onChange} schema={listSchema} schemaKey="test.list" value={['first']} />)

    const input = screen.getByRole('textbox') as HTMLInputElement
    input.focus()
    fireEvent.change(input, { target: { value: 'first, second' } })
    fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' })

    expect(onChange).toHaveBeenCalledTimes(1)
    expect(onChange).toHaveBeenCalledWith(['first', 'second'])
    expect(input.value).toBe('first, second')
  })
})
