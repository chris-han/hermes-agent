"""Explicit optional composition, migration and bounded maintenance task."""
import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from .store import Store
from .publication import Publisher, OperationError
from .feedback import Feedback
from .targets import DeliveryTargets
from .publications import Publications


class ClientOperations:
    def __init__(self,env):
        self.channel=env['SEMANTIER_CLIENT_CHANNEL']
        self.environment=env['SEMANTIER_CLIENT_ENVIRONMENT']
        self.store=Store(env['SEMANTIER_DATABASE_URL'],env.get('SEMANTIER_CLIENT_SCHEMA','semantier_client_operations'))
        key=env.get('SEMANTIER_CLIENT_RECEIPT_KEY','').encode()
        identity=json.loads(env['SEMANTIER_CLIENT_IDENTITY_JSON'])
        target=env.get('SEMANTIER_CLIENT_PUBLICATION_TARGET','filesystem')
        if target not in {'filesystem','r2'}:raise ValueError('unknown client publication target')
        primary_id = env.get('SEMANTIER_CLIENT_DELIVERY_TARGET_ID', 'channel2' if target=='r2' else 'channel1')
        self.primary_target_id = primary_id
        specs = {primary_id: {'kind': target, 'public_root': env['SEMANTIER_CLIENT_PUBLIC_ROOT'],
                             'incoming_root': env['SEMANTIER_CLIENT_INCOMING_ROOT'],
                             'public_base': env['SEMANTIER_CLIENT_PUBLIC_BASE']}}
        specs.update(json.loads(env.get('SEMANTIER_CLIENT_DELIVERY_TARGETS_JSON', '{}')))
        roots = [Path(spec['public_root']).resolve() for spec in specs.values()]
        if len(set(roots)) != len(roots):
            raise ValueError('delivery targets require independent public roots')
        publishers, errors = {}, {}
        for target_id, spec in specs.items():
            if target_id not in {'channel1', 'channel2', 'channel3'}:
                raise ValueError('unsupported delivery target')
            kind = spec['kind']
            if kind not in {'filesystem', 'r2'} or (kind=='r2') != (target_id=='channel2'):
                raise ValueError('delivery target owner mismatch')
            if kind=='r2' and spec['public_base']!='https://update2.semantier.com':
                raise ValueError('R2 update2 publication base required')
            remote = None
            if kind=='r2':
                from .r2_target import R2Target
                try:
                    remote = R2Target.configured()
                except (OperationError, ValueError, RuntimeError, SystemExit):
                    # A missing provider capability is not another filesystem
                    # publisher and cannot disable independent configured owners.
                    errors[target_id] = 'delivery_target_credentials_unavailable'
                    continue
            publishers[target_id] = Publisher(self.store, spec['public_root'], spec['incoming_root'],
                spec['public_base'], key, identity, self.channel, remote_target=remote, target_id=target_id)
        self.targets = DeliveryTargets(self.store, self.environment, publishers, self.channel,
            errors=errors, legacy_target_id=primary_id, origins={key: spec['public_base'] for key,spec in specs.items()})
        self.publications = Publications(self.targets)
        self.feedback=Feedback(self.store,self.channel)
        self.logs=Path(env['SEMANTIER_CLIENT_LOG_ROOT'])
        from .authorization import FeedbackAuthorizations
        self.authorization=FeedbackAuthorizations(self.store,self.channel,bytes.fromhex(env['SEMANTIER_CLIENT_AUTH_KEY']),env['SEMANTIER_CLIENT_PORTAL_ORIGIN'])

    @property
    def publisher(self):
        return self.targets.owner(self.primary_target_id)

    def check(self):
        with self.store.connection() as c:
            if c.execute('SELECT max(version) AS v FROM schema_version').fetchone()['v']!=3:
                raise ValueError('client operations explicit migration required')
            c.execute('SELECT 1 FROM grants LIMIT 0')

    def maintain(self):
        from .distribution import ingest_logs
        from .ci import reconcile_candidates
        try:
            reconcile_candidates(self)
        except OperationError as error:
            if error.status!=503:
                raise
            logging.getLogger(__name__).warning('Primary candidate ingest unavailable: %s',error.code)
        self.publications.maintain()
        with self.store.connection() as c:
            preparing = c.execute("SELECT operation_id FROM operations WHERE channel=%s AND status='preparing' ORDER BY created_at LIMIT 10", (self.channel,)).fetchall()
        for row in preparing:
            try:
                self.publisher.reconcile(row['operation_id'])
            except OperationError as error:
                if error.status!=503:
                    raise
        ingest_logs(self.store,self.logs)
        self.store.retain()


async def start(app,env=None):
    env=os.environ if env is None else env
    app.state.client_operations=None;app.state.client_operations_task=None
    if env.get('SEMANTIER_CLIENT_OPERATIONS_ENABLED')!='1':return
    service=ClientOperations(env);await asyncio.to_thread(service.check)
    for target_id,publisher in service.targets.publishers.items():
        failures=(OperationError,OSError)+( (publisher.remote_target.transport.R2RequestError,) if publisher.remote_target else () )
        try:
            await asyncio.to_thread(publisher.initialize_channel)
        except failures as error:
            service.targets.errors[target_id]=error.code if isinstance(error,OperationError) else 'delivery_target_unavailable'
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
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['migrate','grant','grant-targets']);parser.add_argument('--app-role',default='semantier_app');parser.add_argument('--user-id');parser.add_argument('--capability',action='append',default=[]);parser.add_argument('--legacy-target-id', choices=['channel1','channel2','channel3']);parser.add_argument('--delivery-target',action='append',default=[]);parser.add_argument('--action',action='append',default=[]);args=parser.parse_args()
    store=Store(os.environ.get('SEMANTIER_MIGRATION_DATABASE_URL') or os.environ['SEMANTIER_DATABASE_URL'],os.environ.get('SEMANTIER_CLIENT_SCHEMA','semantier_client_operations'))
    if args.command=='migrate':print(json.dumps(store.migrate(args.app_role,legacy_target_id=args.legacy_target_id)))
    else:
        if not args.user_id:parser.error('--user-id required')
        from agents.gateway_identity import get_user_record
        if not get_user_record(args.user_id):parser.error('existing governed user required')
        if args.command=='grant-targets':
            store.grant_targets(args.user_id,args.delivery_target,args.action,os.environ['SEMANTIER_CLIENT_CHANNEL'])
        else:
            store.grant(args.user_id,args.capability,os.environ['SEMANTIER_CLIENT_CHANNEL'])
        print('Explicit platform grants recorded')


if __name__=='__main__':main()
