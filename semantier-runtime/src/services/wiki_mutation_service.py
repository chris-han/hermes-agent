"""Authenticated documentary wiki mutations; no generic filesystem permission."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from urllib.parse import quote

from agents.gateway_identity import get_user_record, get_user_organization_context
from runtime_paths import _workspaces_root, workspace_session_segment
from storage.wiki_mutation_store import WikiMutationStore, utc


PROFILE_TOOLS = ('read_file', 'search_files', 'wiki_read', 'wiki_mutate', 'wiki_reconcile')


def current_entitlement(user_id, workspace_id, capability="context.read"):
    user = get_user_record(user_id)
    org = get_user_organization_context(user_id)
    if not user or user.get('workspace_id') != workspace_id or not org or not org.get('organization_id'):
        raise PermissionError('active governed workspace membership required')
    grant = {"capability": capability, "decision": "allow", "scope": {
        "organization_id": org["organization_id"], "workspace_id": workspace_id}}
    return org, grant


def validated_wiki_root(workspace):
    root = _workspaces_root()/workspace/"wiki"
    current = root
    while current != current.parent:
        if current.is_symlink(): raise PermissionError("symlink workspace/wiki root rejected")
        current = current.parent
    return root


def governance_module():
    name = '_semantier_documentary_wiki_governance'
    if name not in sys.modules:
        path=Path(__file__).resolve().parents[2]/'hermes-agent/skills/research/llm-wiki/wiki_governance.py'
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        spec.loader.exec_module(module)
    return sys.modules[name]


def digest(data):
    return hashlib.sha256(data).hexdigest() if data is not None else None


class WikiMutationService:
    def __init__(self, boundary):
        if boundary is None or boundary.audit_metadata.get('wiki_task_profile') != 'documentary-wiki.v1':
            raise PermissionError('authenticated server-owned wiki profile required')
        self.workspace=boundary.workspace_id; self.actor=boundary.user_id; self.session=boundary.session_id
        self.org,self.grant=current_entitlement(self.actor,self.workspace)
        self.store=WikiMutationStore()
        local=workspace_session_segment(self.workspace,self.session)
        if self.store.binding(self.workspace,local) != self.actor:
            raise PermissionError('wiki task binding required')
        self.root=validated_wiki_root(self.workspace)
        self._safe_root()

    def _safe_root(self):
        # No symlink anywhere in the active wiki: assurance recursively opens
        # documents, contracts and sidecars, not only the mutated path.
        current=self.root
        while current != current.parent:
            if current.is_symlink(): raise PermissionError('symlink wiki root rejected')
            current=current.parent
        if self.root.exists():
            for path in self.root.rglob('*'):
                if path.is_symlink(): raise PermissionError('wiki symlink rejected')

    def path(self, relative):
        if not isinstance(relative,str) or '\\' in relative or not relative or relative.startswith('/'):
            raise PermissionError('relative wiki path required')
        parts=relative.split('/')
        if any(part in ('','.', '..') or part.startswith('.') for part in parts):
            raise PermissionError('unsafe wiki path')
        path=self.root.joinpath(*parts)
        self._safe_root()
        if not path.resolve().is_relative_to(self.root.resolve()): raise PermissionError('path outside wiki')
        return path

    def read(self, relative):
        path=self.path(relative)
        data=path.read_bytes()
        if len(data)>2*1024*1024: raise ValueError('wiki read exceeds 2 MiB')
        return {'path':relative,'content':data.decode('utf-8'),'sha256':digest(data), 'pending_operations':[{'operation_id':r['operation_id'],'path':r['path'],'state':r['assurance']['state']} for r in self.store.pending(self.workspace) if r['actor']==self.actor]}

    def _checkpoint(self, stage):
        """Fault-injection seam; production has no callback or retry loop."""

    def mutate(self, args, tool_call_id=None):
        if set(args) - {'operation_id','path','expected_sha256','content','old_text','new_text'}:
            raise ValueError('unknown documentary mutation fields')
        self.org, self.grant = current_entitlement(self.actor, self.workspace, "knowledge.propose")
        operation=args.get('operation_id')
        if not isinstance(operation,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',operation): raise ValueError('stable ASCII operation_id required')
        relative=args.get('path'); path=self.path(relative)
        if path.suffix.lower() not in ('.md','.markdown','.txt','.json','.yaml','.yml'):
            raise PermissionError('documentary extension required')
        expected=args.get('expected_sha256')
        if 'expected_sha256' not in args or (expected is not None and not re.fullmatch(r'[a-f0-9]{64}',str(expected))): raise ValueError('expected_sha256 required: null creates, hash updates')
        patch='old_text' in args or 'new_text' in args
        if patch and 'content' in args: raise ValueError('choose content or unique text patch')
        if not patch and not isinstance(args.get('content'),str): raise ValueError('content required')
        request_hash=digest(json.dumps(args,sort_keys=True).encode())
        with self.store.workspace_lock(self.workspace):
            self._safe_root()
            previous=self.store.get(self.workspace,operation)
            if previous:
                if previous['request_hash']!=request_hash or previous['actor']!=self.actor: raise ValueError('operation identity payload conflict')
                self.store.attempt(self.workspace,operation,'EXACT_RETRY')
                return self._reconcile_record(previous)
            old=path.read_bytes() if path.exists() else None
            if relative.startswith('raw/') and old is not None: raise PermissionError('raw sources are immutable')
            if digest(old)!=expected: raise ValueError('stale wiki version')
            if patch:
                before=args.get('old_text');after=args.get('new_text')
                if not isinstance(before,str) or not before or not isinstance(after,str) or old is None: raise ValueError('valid text patch required')
                text=old.decode('utf-8')
                if text.count(before)!=1: raise ValueError('patch must match exactly once')
                new=text.replace(before,after,1).encode()
            else: new=args['content'].encode('utf-8')
            if len(new)>2*1024*1024: raise ValueError('wiki mutation exceeds 2 MiB')
            m=governance_module()
            old_fm=m.split_frontmatter((old or b'').decode('utf-8'))[0]
            new_fm=m.split_frontmatter(new.decode('utf-8'))[0]
            canonical=(old_fm.get('authority')=='canonical' or new_fm.get('authority')=='canonical') and old!=new
            record={'schema_version':'documentary-wiki-mutation.v1','operation_id':operation,'request_hash':request_hash,'workspace':self.workspace,'organization':self.org['organization_id'],'actor':self.actor,'session':self.session,'tool_call_id':tool_call_id,'path':relative,'before_sha256':digest(old),'after_sha256':digest(new),'canonical_change':canonical,'created_at':utc(),'write_success':False,'assurance':{'state':'INTENT','last_error':None,'report_refs':[]},'navigation':{'status':'not_checked'},'review_required':self.grant['decision']=='allow_with_review','changed':old!=new}
            self._checkpoint('before_intent')
            self.store.save(self.workspace,record)
            self._checkpoint('after_intent')
            if old!=new:
                path.parent.mkdir(parents=True,exist_ok=True)
                m.atomic_bytes(path,new)
            self._checkpoint('after_file_commit')
            record['write_success']=True;record['assurance']['state']='PENDING'
            self.store.save(self.workspace,record)
            return self._finish(record)

    def reconcile(self, operation):
        self.org, self.grant = current_entitlement(self.actor, self.workspace, "knowledge.propose")
        with self.store.workspace_lock(self.workspace):
            record=self.store.get(self.workspace,operation)
            if record is None or record['actor']!=self.actor: raise PermissionError('owned operation required')
            self.store.attempt(self.workspace,operation,'RECONCILE')
            return self._reconcile_record(record)

    def _reconcile_record(self, record):
        if record['assurance']['state'] not in ('INTENT','PENDING'): return record
        path=self.path(record['path'])
        current=digest(path.read_bytes()) if path.exists() else None
        if current!=record['after_sha256']:
            record['write_success']=False
            record['assurance']['state']='NOT_APPLIED' if current==record['before_sha256'] else 'CONFLICT'
            self.store.save(self.workspace,record)
            return record
        record['write_success']=True
        record['assurance']['state']='PENDING'
        self.store.save(self.workspace,record)
        return self._finish(record)

    def _navigation(self, record):
        path=self.path(record['path'])
        if path.suffix.lower() not in ('.md','.markdown') or record['path'] in ('log.md','SCHEMA.md') or record['path'].startswith('raw/'):
            return {'status':'not_applicable','unresolved_links':[]}
        m=governance_module();pages=m.iter_wiki_pages(self.root)
        ids={p.path for p in pages};refs=[];inbound=[]
        body=m.split_frontmatter(path.read_text())[1]
        for match in m.WIKILINK_RE.finditer(body):
            target=m.normalize_page_ref(match.group(1).split('|')[0].split('#')[0])
            if target not in ids: refs.append(target)
        for page in pages:
            if record['path'] in {m.normalize_page_ref(link) for link in page.wikilinks}: inbound.append(page.path)
        return {'status':'checked','unresolved_links':sorted(set(refs)),'inbound_links':sorted(inbound)}

    def _assure(self, record):
        m=governance_module()
        refs=[]
        change={key:record[key] for key in ('operation_id','path','before_sha256','after_sha256','workspace')}
        if record['canonical_change']:
            self._checkpoint('during_dependents')
            report=m.impact_canonical_change(self.root,record['path'],change=change)
            refs.append(report['report_ref'])
        if record['changed'] and Path(record['path']).suffix.lower()=='.md' and not record['path'].startswith('raw/') and record['path'] not in ('log.md','SCHEMA.md'):
            report=m.audit_wiki(self.root,change=change)
            refs.append(report['report_ref'])
        self._checkpoint('after_report')
        return refs

    def _finish(self, record):
        try:
            self._safe_root()
            try: record['navigation']=self._navigation(record)
            except Exception as exc: record['navigation']={'status':'not_checked','diagnostic_error':str(exc)}
            refs=self._assure(record)
            record['assurance']={'state':'PENDING','last_error':None,'report_refs':refs}
            self.store.save(self.workspace,record)
            self._checkpoint('before_complete')
            record['assurance']['state']='COMPLETE' if refs else 'NOT_APPLICABLE'
        except Exception as exc:
            record['assurance']['state']='PENDING';record['assurance']['last_error']=str(exc)
        record['navigation_url']='/memory?lens=documents&knowledge_view=wiki&knowledge_path='+quote(record['path'],safe='')
        record['updated_at']=utc()
        self.store.save(self.workspace,record)
        return record
