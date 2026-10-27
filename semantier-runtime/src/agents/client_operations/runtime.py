"""Explicit optional composition, migration and bounded maintenance task."""
import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from .store import Store
from .publication import Publisher
from .feedback import Feedback


class ClientOperations:
    def __init__(self,env):
        self.channel=env['SEMANTIER_CLIENT_CHANNEL']
        self.environment=env['SEMANTIER_CLIENT_ENVIRONMENT']
        self.store=Store(env['SEMANTIER_DATABASE_URL'],env.get('SEMANTIER_CLIENT_SCHEMA','semantier_client_operations'))
        key=env.get('SEMANTIER_CLIENT_RECEIPT_KEY','').encode()
        identity=json.loads(env['SEMANTIER_CLIENT_IDENTITY_JSON'])
        target=env.get('SEMANTIER_CLIENT_PUBLICATION_TARGET','filesystem')
        if target not in {'filesystem','r2'}:raise ValueError('unknown client publication target')
        remote=None
        if target=='r2':
            from .r2_target import R2Target
            remote=R2Target.configured()
            if env['SEMANTIER_CLIENT_PUBLIC_BASE']!='https://update2.semantier.com':
                raise ValueError('R2 update2 publication base required')
        self.publisher=Publisher(self.store,env['SEMANTIER_CLIENT_PUBLIC_ROOT'],env['SEMANTIER_CLIENT_INCOMING_ROOT'],env['SEMANTIER_CLIENT_PUBLIC_BASE'],key,identity,self.channel,remote_target=remote)
        self.feedback=Feedback(self.store,self.channel)
        self.logs=Path(env['SEMANTIER_CLIENT_LOG_ROOT'])
        from .authorization import FeedbackAuthorizations
        self.authorization=FeedbackAuthorizations(self.store,self.channel,bytes.fromhex(env['SEMANTIER_CLIENT_AUTH_KEY']),env['SEMANTIER_CLIENT_PORTAL_ORIGIN'])

    def check(self):
        with self.store.connection() as c:
            if c.execute('SELECT max(version) AS v FROM schema_version').fetchone()['v']!=2:
                raise ValueError('client operations explicit migration required')
            c.execute('SELECT 1 FROM grants LIMIT 0')

    def maintain(self):
        from .distribution import ingest_logs
        from .ci import reconcile_candidates
        reconcile_candidates(self)
        ingest_logs(self.store,self.logs)
        self.store.retain()


async def start(app,env=None):
    env=os.environ if env is None else env
    app.state.client_operations=None;app.state.client_operations_task=None
    if env.get('SEMANTIER_CLIENT_OPERATIONS_ENABLED')!='1':return
    service=ClientOperations(env);await asyncio.to_thread(service.check)
    await asyncio.to_thread(service.publisher.initialize_channel)
    app.state.client_operations=service
    async def tick():
        while True:
            try:await asyncio.to_thread(service.maintain)
            except Exception:logging.getLogger(__name__).exception('client operations maintenance unavailable')
            await asyncio.sleep(30)
    app.state.client_operations_task=asyncio.create_task(tick())


async def stop(app):
    task=getattr(app.state,'client_operations_task',None)
    if task:
        task.cancel()
        try:await task
        except asyncio.CancelledError:pass


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['migrate','grant']);parser.add_argument('--app-role',default='semantier_app');parser.add_argument('--user-id');parser.add_argument('--capability',action='append',default=[]);args=parser.parse_args()
    store=Store(os.environ.get('SEMANTIER_MIGRATION_DATABASE_URL') or os.environ['SEMANTIER_DATABASE_URL'],os.environ.get('SEMANTIER_CLIENT_SCHEMA','semantier_client_operations'))
    if args.command=='migrate':print(json.dumps(store.migrate(args.app_role)))
    else:
        if not args.user_id:parser.error('--user-id required')
        from agents.gateway_identity import get_user_record
        if not get_user_record(args.user_id):parser.error('existing governed user required')
        store.grant(args.user_id,args.capability,os.environ['SEMANTIER_CLIENT_CHANNEL']);print('Explicit platform grants recorded')


if __name__=='__main__':main()
