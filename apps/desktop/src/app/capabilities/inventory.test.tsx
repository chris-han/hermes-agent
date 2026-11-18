import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { CapabilityInventory, InventoryItems, InventoryRow } from './inventory'

afterEach(cleanup)
it('remembers each tab view without losing the selected item', () => {
  const tab = (name: string) =>
    render(
      <CapabilityInventory tab={name}>
        <InventoryItems>
          <InventoryRow active enabled onSelect={() => {}} title="Research" />
        </InventoryItems>
      </CapabilityInventory>
    )

  let view = tab('test-skills')
  fireEvent.click(screen.getByRole('button', { name: 'Cards' }))
  expect(screen.getByTestId('inventory-items').getAttribute('data-view')).toBe('cards')
  expect(screen.getByRole('button', { name: 'Research' }).getAttribute('aria-pressed')).toBe('true')
  view.unmount()
  view = tab('test-tools')
  expect(screen.getByTestId('inventory-items').getAttribute('data-view')).toBe('list')
  view.unmount()
  tab('test-skills')
  expect(screen.getByTestId('inventory-items').getAttribute('data-view')).toBe('cards')
})

it('clamps a long toolset description once in cards while list keeps its compact subtitle', () => {
  const description = 'A2A supports agent discovery and communication. '.repeat(30)

  const view = render(
    <CapabilityInventory tab="test-a2a-card-clamp">
      <InventoryItems>
        <InventoryRow active description={description} enabled onSelect={() => {}} subtitle={description} title="A2A" />
      </InventoryItems>
    </CapabilityInventory>
  )

  expect(screen.getAllByText(/A2A supports agent discovery/)).toHaveLength(1)
  fireEvent.click(screen.getByRole('button', { name: 'Cards' }))
  const summary = screen.getByText(/A2A supports agent discovery/)
  expect(summary.className).toContain('line-clamp-2')
  expect(screen.getAllByText(/A2A supports agent discovery/)).toHaveLength(1)
  view.unmount()
})

it('uses one sliding Asimov selection indicator while preserving keyboard buttons', () => {
  const { container } = render(<CapabilityInventory tab="sliding-view"><span>Inventory</span></CapabilityInventory>)
  expect(container.querySelector('[data-slot="segmented-control-indicator"]')).not.toBeNull()
  fireEvent.click(screen.getByRole('button', { name: 'Cards' }))
  expect(screen.getByRole('button', { name: 'Cards' }).getAttribute('aria-pressed')).toBe('true')
  expect(container.querySelectorAll('[data-slot="segmented-control-indicator"]')).toHaveLength(1)
})
