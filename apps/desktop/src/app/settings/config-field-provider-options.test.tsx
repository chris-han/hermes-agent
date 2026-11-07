import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { ConfigField } from './config-field'

afterEach(cleanup)

it('keeps backend-discovered optional STT providers selectable beside the default', async () => {
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.scrollIntoView = vi.fn()
  const onChange = vi.fn()
  render(
    <ConfigField
      enumOptions={['local']}
      onChange={onChange}
      schema={{ type: 'select', options: ['local', 'sensevoice'] }}
      schemaKey="stt.provider"
      value="local"
    />
  )
  fireEvent.keyDown(screen.getByRole('combobox'), { key: 'ArrowDown' })
  const optional = await screen.findByRole('option', { name: /sensevoice/i })
  expect(screen.getByRole('option', { name: /local/i })).toBeTruthy()
  fireEvent.click(optional)
  expect(onChange).toHaveBeenCalledWith('sensevoice')
})
