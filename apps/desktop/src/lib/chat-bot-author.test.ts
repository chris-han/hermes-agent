import { expect, it } from 'vitest'
import { toChatMessages } from './chat-messages'
import { toRuntimeMessage, coalesceToolOnlyAssistants } from './chat-runtime'

const metadata = { author_bot_id: 'bot_a', author_display_name: 'My custom Winman' }

it('retains the server Bot author across hydration and renderer conversion, ignoring user metadata', () => {
  const rows = toChatMessages([
    { role: 'user', content: 'spoof', display_metadata: metadata },
    { role: 'assistant', content: 'Reminder', display_metadata: metadata }
  ])
  expect(rows[0]).not.toHaveProperty('botAuthor')
  expect(rows[1]).toMatchObject({ botAuthor: { botId: 'bot_a', displayName: 'My custom Winman' } })
  expect(toRuntimeMessage(rows[1]).metadata?.custom).toMatchObject({ botAuthor: { botId: 'bot_a', displayName: 'My custom Winman' } })
})

it('does not fold a Bot business message into a Main Agent tool response', () => {
  const rows = toChatMessages([
    { role: 'assistant', content: 'Main response', tool_calls: [{ id: 'c1', type: 'function', function: { name: 'read_file', arguments: '{}' } }] },
    { role: 'assistant', content: 'Reminder', display_metadata: metadata }
  ])
  expect(rows).toHaveLength(2)
  expect(rows[1]).toMatchObject({ botAuthor: { botId: 'bot_a' } })
})

it('keeps tool-only messages from different Bot authors separate', () => {
  const base = { role: 'assistant' as const, parts: [{ type: 'text' as const, text: 'Reminder' }], botAuthor: { botId: 'bot_a', displayName: 'Winman' } }
  const tool = { id: 't', role: 'assistant' as const, parts: [{ type: 'tool-call' as const, toolCallId: 'c', toolName: 'read_file', args: {}, argsText: '{}' }] }
  expect(coalesceToolOnlyAssistants([{ ...base, id: 'a' }, tool], new WeakMap())).toHaveLength(2)
})

it('preserves explicitly changed author names on historical Bot messages', () => {
  const rows = toChatMessages([
    { role: 'assistant', content: 'Earlier', tool_calls: [{ id: 'c1', type: 'function', function: { name: 'read_file', arguments: '{}' } }], display_metadata: metadata },
    { role: 'assistant', content: 'Later', display_metadata: { ...metadata, author_display_name: 'Renamed Winman' } }
  ])
  expect(rows).toHaveLength(2)
  expect(rows.map(row => row.botAuthor?.displayName)).toEqual(['My custom Winman', 'Renamed Winman'])
})
