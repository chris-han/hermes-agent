"""Scoped board projection of existing core-admitted meeting responsibilities."""
import json
import re

from agents.durable_responsibility import DurableResponsibilityStore
from contracts.meeting_delivery import digest
from agents.coordination_association import CoordinationAssociationStore, WorkRef

_ID = re.compile(r'^[A-Za-z0-9_.:-]{1,128}$')


def selected_core_meeting(store, case_id):
    """Recognize an admitted case even while its association handoff is pending."""
    with store._connect() as conn:
        row = conn.execute('''SELECT 1 FROM meeting_time_negotiations AS case_row
            JOIN meeting_core_admissions AS admission
              ON admission.monitor_id=case_row.monitor_id
             AND admission.event_revision_id=case_row.event_revision_id
            WHERE case_row.negotiation_id=%s''', (case_id,)).fetchone()
    return row is not None


class CoreMeetingBoardService:
    def __init__(self, store):
        self.store = store
        self.responsibilities = DurableResponsibilityStore(store.path)

    def reserve_meeting(self, context, *, monitor_id, revision_id, participant_id):
        """Reserve a fresh scoped domain creation, before its first write."""
        from agents.core_file_scope import CoreFileScope
        CoreFileScope.issue(context, 'meeting-board-admission')
        if not all(isinstance(v, str) and _ID.fullmatch(v) for v in (monitor_id, revision_id, participant_id)):
            raise ValueError('invalid meeting identity')
        marker = digest({'kind':'meeting-board.v1', 'principal':context.user_id,
            'organization':context.organization_id, 'workspace':context.workspace_id,
            'monitor':monitor_id, 'revision':revision_id, 'participant':participant_id})
        with self.store._connect() as conn:
            monitor=conn.execute('SELECT * FROM meeting_rsvp_monitors WHERE monitor_id=%s FOR UPDATE',(monitor_id,)).fetchone()
            if (not monitor or monitor['workspace_id'] != context.workspace_id
                or monitor['creator_user_id'] != context.user_id or monitor['event_revision_id'] != revision_id):
                raise KeyError(monitor_id)
            old=conn.execute('SELECT request_hash FROM meeting_core_admissions WHERE monitor_id=%s AND event_revision_id=%s', (monitor_id,revision_id)).fetchone()
            if old is None:
                if conn.execute('SELECT 1 FROM meeting_time_negotiations WHERE monitor_id=%s', (monitor_id,)).fetchone():
                    raise PermissionError('existing meeting cannot be adopted')
                conn.execute('INSERT INTO meeting_core_admissions VALUES(%s,%s,%s)', (monitor_id,revision_id,marker))
            elif old['request_hash'] != marker:
                raise PermissionError('meeting admission conflict')
        return marker

    def associate_meeting(self, context, case_id, marker):
        """Associate an admitted new case to its already-existing responsibility."""
        case=self.store.get_negotiation(case_id)
        if (not context.authenticated or case['creator_user_id'] != context.user_id
            or case['organization_id'] != context.organization_id or case['workspace_id'] != context.workspace_id):
            raise KeyError(case_id)
        expected=digest({'kind':'meeting-board.v1', 'principal':context.user_id,
            'organization':context.organization_id, 'workspace':context.workspace_id,
            'monitor':case['monitor_id'], 'revision':case['event_revision_id'],
            'participant':case['declined_attendee_user_id']})
        with self.store._connect() as conn:
            row=conn.execute('SELECT request_hash FROM meeting_core_admissions WHERE monitor_id=%s AND event_revision_id=%s',
                (case['monitor_id'],case['event_revision_id'])).fetchone()
        if not row or row['request_hash'] != marker or marker != expected:
            raise PermissionError('fresh meeting reservation required')
        rid=self.responsibilities.derive_id(responsibility_kind='feishu_meeting_negotiation',workspace_id=context.workspace_id,
            domain_ref=case_id,workflow_id='feishu-meeting-coordinator',workflow_instance_id=case_id)
        if self.responsibilities.get(rid) is None:
            raise RuntimeError('meeting responsibility missing')
        lock_conn=self.store._connect()
        try:
            lock_conn.execute('SELECT pg_advisory_lock(hashtextextended(%s,0))',('meeting-board-association:'+case_id,))
            CoordinationAssociationStore(self.store.path).ensure_context(workspace_id=context.workspace_id,
                work_ref=WorkRef('meeting-board.v1',case_id),goal_ref='feishu-meeting-coordinator',
                goal_revision=case['workflow_binding_hash'],principal_ref=context.user_id,opened_at=case['created_at'])
        finally:
            lock_conn.close()
        return self.get(context,case_id)

    def get(self, context, case_id):
        if not isinstance(case_id, str) or not _ID.fullmatch(case_id):
            raise KeyError(case_id)
        with self.store._connect() as conn:
            row = conn.execute('SELECT * FROM meeting_core_delivery_bindings WHERE negotiation_id=%s', (case_id,)).fetchone()
        if row is None:
            with self.store._connect() as conn:
                board=conn.execute("SELECT * FROM semantier_coordination_contexts WHERE workspace_id=%s AND work_source_kind='meeting-board.v1' AND work_source_id=%s AND principal_ref=%s",(context.workspace_id,case_id,context.user_id)).fetchone()
            if board is None:
                raise KeyError(case_id)
            case=self.store.get_negotiation(case_id)
            rid=self.responsibilities.derive_id(responsibility_kind='feishu_meeting_negotiation',workspace_id=context.workspace_id,
                domain_ref=case_id,workflow_id='feishu-meeting-coordinator',workflow_instance_id=case_id)
            association={'principal_id':board['principal_ref'],'organization_id':case['organization_id'],
                'workspace_id':board['workspace_id'],'negotiation_id':case_id,'responsibility_id':rid,
                'workflow_binding_hash':board['goal_revision']}
            revoked=False
        else:
            association = json.loads(row['association_json'])
            if digest(association) != row['association_hash']:
                raise RuntimeError('meeting association integrity failure')
            revoked=bool(row['revoked_at'])
        if (not context.authenticated or association.get('principal_id') != context.user_id
            or association.get('organization_id') != context.organization_id
            or association.get('workspace_id') != context.workspace_id):
            raise KeyError(case_id)
        case = self.store.get_negotiation(case_id)
        responsibility = self.responsibilities.get(association['responsibility_id'])
        if (case['creator_user_id'] != context.user_id or case['organization_id'] != context.organization_id
            or case['workspace_id'] != context.workspace_id or association.get('negotiation_id') != case_id
            or association.get('workflow_binding_hash') != case['workflow_binding_hash']
            or responsibility is None
            or responsibility.responsibility_kind != 'feishu_meeting_negotiation'
            or responsibility.workspace_id != context.workspace_id or responsibility.domain_ref != case_id
            or responsibility.workflow_id != 'feishu-meeting-coordinator'
            or responsibility.workflow_instance_id != case_id):
            raise RuntimeError('meeting responsibility binding mismatch')
        metadata=self._annotations(case_id)
        payload=json.loads(case.get('payload_json') or '{}')
        history=self.store.list_negotiation_events(case_id)
        options = None
        if case['status'] == 'awaiting_requester_decision':
            options = {'original': {'start_time':case['original_start_time'], 'end_time':case['original_end_time'], 'timezone':case['timezone']},
                       'candidates': [{k:row[k] for k in ('slot_id','start_time','end_time','timezone')}
                                      for row in self.store.list_candidate_slots(case_id)]}
        return {'title':payload.get('meeting_title') or payload.get('title') or case['event_id'],
                'decision_options':options,
                'history':history, 'result':{'selected_slot':json.loads(case['selected_slot_json']) if case.get('selected_slot_json') else None,
                    'external_effect':'not_applied' if case.get('finalize_status') == 'not_started' else case.get('finalize_status')},
                'comments':metadata['comment'], 'references':metadata['reference'], 'kind':'meeting', 'id':'meeting:'+case_id, 'negotiation_id':case_id,
                'responsibility_id':responsibility.responsibility_id,
                'accountable_owner':{'user_id':context.user_id},
                'responsible_executor':{'kind':'meeting_domain'},
                'status':case['status'], 'lifecycle_state':responsibility.lifecycle_state,
                'checkpoint_version':responsibility.checkpoint_version,
                'revoked':revoked}

    def _annotations(self, case_id):
        with self.store._connect() as conn:
            rows=conn.execute('SELECT * FROM meeting_board_annotations WHERE negotiation_id=%s ORDER BY entry_kind,entry_version',(case_id,)).fetchall()
        values={'comment':[], 'reference':[]}
        for row in rows:
            item=json.loads(row['payload_json'])
            item.update(request_id=row['request_id'],version=row['entry_version'],author_id=row['author_id'],created_at=row['created_at'])
            values[row['entry_kind']].append(item)
        return values

    def _append(self, context, case_id, data, entry_kind):
        self.get(context,case_id)  # Scope and association checks precede mutation.
        keys={'request_id','expected_version','content'} if entry_kind=='comment' else {'request_id','expected_version','kind','ref'}
        if (not isinstance(data,dict) or set(data)!=keys or not isinstance(data['request_id'],str)
            or not _ID.fullmatch(data['request_id']) or type(data['expected_version']) is not int or data['expected_version']<0):
            raise ValueError('invalid annotation')
        if entry_kind=='comment':
            if not isinstance(data['content'],str) or not data['content'].strip() or len(data['content'].encode())>8192:
                raise ValueError('invalid comment')
            payload={'content':data['content']}
        else:
            if data['kind'] not in ('work', 'meeting') or not isinstance(data['ref'],str) or not _ID.fullmatch(data['ref']):
                raise ValueError('invalid reference')
            payload={'kind':data['kind'],'ref':data['ref']}
            if data['kind'] == 'meeting':
                self.get(context, data['ref'])
        with self.store._connect() as conn:
            conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))',('meeting-board-annotation:'+case_id+':'+entry_kind,))
            prior=conn.execute('SELECT * FROM meeting_board_annotations WHERE negotiation_id=%s AND entry_kind=%s AND request_id=%s',
                (case_id,entry_kind,data['request_id'])).fetchone()
            if prior:
                if json.loads(prior['payload_json'])!=payload or prior['entry_version']!=data['expected_version']+1:
                    raise ValueError('annotation identity conflict')
                return {**payload,'request_id':prior['request_id'],'version':prior['entry_version'],'author_id':prior['author_id'],'created_at':prior['created_at']}
            count=conn.execute('SELECT count(*) AS n FROM meeting_board_annotations WHERE negotiation_id=%s AND entry_kind=%s',(case_id,entry_kind)).fetchone()['n']
            if count!=data['expected_version']:
                raise ValueError('annotation version conflict')
            if count>=256:
                raise ValueError('annotation limit reached')
            if entry_kind=='reference' and data['kind']=='work':
                from services.core_work_service import CoreWorkService
                CoreWorkService()._load(context,data['ref'])
            from agents.job_store import utc_now_iso
            created=utc_now_iso();version=count+1
            conn.execute('INSERT INTO meeting_board_annotations VALUES(%s,%s,%s,%s,%s,%s,%s)',
                (case_id,entry_kind,data['request_id'],version,context.user_id,json.dumps(payload,sort_keys=True),created))
        return {**payload,'request_id':data['request_id'],'version':version,'author_id':context.user_id,'created_at':created}

    def comment(self, context, case_id, data):
        return self._append(context,case_id,data,'comment')

    def reference(self, context, case_id, data):
        return self._append(context,case_id,data,'reference')

    def list(self, context, *, limit=25, cursor=""):
        if (type(limit) is not int or not 1 <= limit <= 100 or not isinstance(cursor, str)
            or (cursor and not _ID.fullmatch(cursor))):
            raise ValueError('invalid meeting page')
        with self.store._connect() as conn:
            rows = conn.execute("""SELECT negotiation_id FROM (SELECT negotiation_id FROM meeting_core_delivery_bindings
                WHERE association_json::jsonb->>'principal_id'=%s
                AND association_json::jsonb->>'workspace_id'=%s
                AND association_json::jsonb->>'organization_id'=%s
                UNION SELECT work_source_id AS negotiation_id FROM semantier_coordination_contexts
                WHERE work_source_kind='meeting-board.v1' AND principal_ref=%s AND workspace_id=%s) AS owned
                WHERE negotiation_id>%s ORDER BY negotiation_id LIMIT %s""",
                (context.user_id, context.workspace_id, context.organization_id, context.user_id, context.workspace_id, cursor, limit+1)).fetchall()
        return {'items':[self.get(context, row['negotiation_id']) for row in rows[:limit]],
                'next_cursor':rows[limit-1]['negotiation_id'] if len(rows)>limit else None}
