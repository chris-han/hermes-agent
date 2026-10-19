"""Authenticated profile-owned Billing reads on the canonical native transport."""
import asyncio
from functools import partial
from pathlib import Path

from gateway.session_config import _authorize


def handlers(connection):
    return {'billing.state': partial(read, connection, 'state'),
            'subscription.state': partial(read, connection, 'subscription')}


async def read(connection, resource, ref, params):
    _authorize(connection, ref, params)

    def scoped_read():
        from gateway.run import _profile_runtime_scope
        from agent.semantier_gateway import read_billing
        with _profile_runtime_scope(Path(connection.authority.profile_id)):
            return read_billing(resource, schema_version=params.get('schema_version'))
    return await asyncio.to_thread(scoped_read)
