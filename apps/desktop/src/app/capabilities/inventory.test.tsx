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
