import { useStore } from '@nanostores/react'

import { $activeGatewayRoute, $gateway } from '@/store/gateway'
import { $connection } from '@/store/session'

const sockets = new WeakMap<object, number>()
const descriptors = new WeakMap<object, number>()
let nextSocketId = 0
let nextDescriptorId = 0

/** Socket generation + profile prevents account data surviving a source switch. */
export function useBillingScope(): string {
  const gateway = useStore($gateway)
  const profile = useStore($activeGatewayRoute)
  const connection = useStore($connection)

  if (connection && !descriptors.has(connection)) {
    descriptors.set(connection, ++nextDescriptorId)
  }

  if (gateway && !sockets.has(gateway)) {sockets.set(gateway, ++nextSocketId)}

  return `${gateway ? sockets.get(gateway) : 'disconnected'}:${profile}:${connection ? descriptors.get(connection) : 'unbound'}`
}
