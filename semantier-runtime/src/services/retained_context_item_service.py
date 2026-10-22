"""Item-addressed changes over Hermes' lock/parser/atomic projection seam."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from contracts.retained_context_items import RetainedContextItemChangeV1
from runtime_paths import workspace_hermes_home_path


def _digest(value):
    return 'sha256:' + hashlib.sha256(value.encode('utf-8')).hexdigest()


class RetainedContextItemService:
    def __init__(self, *, store):
        self.store = store

    @staticmethod
    def _check(context):
        if not (context.authenticated and context.user_id and context.organization_id
                and context.workspace_id):
            raise PermissionError('active authenticated workspace context required')
        expected = workspace_hermes_home_path(context.workspace_id).resolve()
        if Path(context.hermes_home).resolve() != expected or Path(context.workspace_root).resolve() != expected:
            raise PermissionError('retained context requires governed workspace runtime home')
        for path in (expected/'memories', expected/'memories'/'MEMORY.md', expected/'memories'/'USER.md'):
            if path.is_symlink() or (path != expected and expected not in path.resolve().parents):
                raise PermissionError('memory projection must stay inside governed workspace')

    @contextmanager
    def _bound(self, context):
        self._check(context)
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override
        token = set_hermes_home_override(Path(context.hermes_home))
        try:
            yield
        finally:
            reset_hermes_home_override(token)

    def _changes(self, context):
        pins = self.store.list_artifact_pins(organization_id=context.organization_id,
            workspace_id=context.workspace_id, schema_version='retained_context_item_change.v1')
        changes = [self.store.get(organization_id=context.organization_id,
            workspace_id=context.workspace_id, artifact_ref=pin.artifact_ref,
            contract_type=RetainedContextItemChangeV1) for pin in pins]
        # Workspace memory is shared by its governed membership; owner_ref records
        # the actor without implying personal file isolation that Hermes lacks.
        return sorted(changes, key=lambda x: (x.item_ref, x.revision))

    @staticmethod
    def _ref(context, source, content):
        return 'retained_context_item:' + _digest('\x00'.join((context.organization_id,
            context.workspace_id, source, content))).split(':')[1]

    @staticmethod
    def _revision(change):
        return change.artifact_hash

    @staticmethod
    def _projection(entries, changes):
        grouped = {}
        for change in changes:
            grouped.setdefault(change.item_ref, []).append(change)
        historical_owners = {}
        for item_ref, chain in grouped.items():
            for change in chain:
                for content in (change.prior_content, change.content):
                    if content is None:
                        continue
                    identity = (change.source_file, content)
                    if identity in historical_owners and historical_owners[identity] != item_ref:
                        raise ValueError('overlapping managed item histories')
                    historical_owners[identity] = item_ref
        result = list(entries)
        for chain in grouped.values():
            revisions = [change.revision for change in chain]
            if revisions != list(range(1, len(chain)+1)):
                raise ValueError('ambiguous managed item revision chain')
            for previous, successor in zip(chain, chain[1:]):
                if successor.expected_revision != previous.artifact_hash or successor.prior_content != previous.content:
                    raise ValueError('ambiguous managed item predecessor chain')
            historical = {change.prior_content for change in chain}
            historical.update(change.content for change in chain if change.content is not None)
            positions = [i for i, content in enumerate(result) if content in historical]
            position = min(positions) if positions else len(result)
            result = [content for content in result if content not in historical]
            if chain[-1].content is not None:
                result.insert(min(position,len(result)), chain[-1].content)
        return result

    def reconcile(self, *, context):
        from tools.memory_tool import MemoryStore
        with self._bound(context):
            for source, target in (('MEMORY.md','memory'), ('USER.md','user')):
                if not any(c.source_file == source for c in self._changes(context)):
                    continue
                memory = MemoryStore()
                def reconcile_locked(entries, limit):
                    changes = [change for change in self._changes(context) if change.source_file == source]
                    result = self._projection(entries, changes)
                    return result, 'Managed retained context reconciled.'
                result = memory._mutate(target, reconcile_locked)
                if not result.get('success'):
                    raise ValueError(result.get('error', 'memory reconciliation failed'))

    def _row(self, *, ref, revision, source, content, available=True, pending=False):
        return {'ref':ref, 'item_ref':ref, 'revision':revision, 'source_file':source,
            'content':content, 'leading_answer': ('Future context publication is pending.' if pending else
                'Available to a future session.' if available else 'Stopped using this context item.'),
            'runtime_availability': 'pending' if pending else 'available' if available else 'suppressed',
            'recorded_use':'unknown', 'available_actions':['correct','suppress'] if available else []}

    def list_items(self, *, context):
        from tools.memory_tool import load_on_disk_store
        self.reconcile(context=context)
        with self._bound(context):
            memory=load_on_disk_store()
            changes=self._changes(context)
            latest={change.item_ref:change for change in changes}
            rows=[]
            for source,target in (('MEMORY.md','memory'),('USER.md','user')):
                managed={change.content:change for change in latest.values()
                    if change.source_file==source and change.content is not None}
                for content in memory._entries_for(target):
                    change=managed.get(content)
                    rows.append(self._row(ref=change.item_ref if change else self._ref(context,source,content),
                        revision=self._revision(change) if change else _digest(content), source=source,content=content))
                    if not memory.target_enabled(target):
                        rows[-1]['runtime_availability']='unavailable'
                        rows[-1]['leading_answer']='Retained, but this memory source is disabled for future sessions.'
            rows.extend(self._row(ref=change.item_ref,revision=self._revision(change),source=change.source_file,
                content=None,available=False) for change in latest.values() if change.content is None)
            return rows

    def correct(self, *, context, item_ref, expected_revision, new_content, request_id, origin=None):
        return self._change(context=context,item_ref=item_ref,expected_revision=expected_revision,
            content=new_content,request_id=request_id,origin=origin)

    def suppress(self, *, context, item_ref, expected_revision, request_id, origin=None):
        return self._change(context=context,item_ref=item_ref,expected_revision=expected_revision,
            content=None,request_id=request_id,origin=origin)

    def _change(self, *, context, item_ref, expected_revision, content, request_id, origin):
        from tools.memory_tool import MemoryStore, load_on_disk_store, ENTRY_DELIMITER, _scan_memory_content
        from agents.runtime_memory_boundary import _USER_MEMORY_BLOCKED_LINE_RE
        if not request_id or not isinstance(request_id,str):
            raise ValueError('request_id required')
        if content is not None:
            content=content.strip()
            if not content or ENTRY_DELIMITER in content or '\n§' in content or _USER_MEMORY_BLOCKED_LINE_RE.search(content):
                raise ValueError('invalid or authority-bearing context content')
            if error:=_scan_memory_content(content):
                raise ValueError(error)
        with self._bound(context):
            existing=self._changes(context)
            chain=[change for change in existing if change.item_ref==item_ref]
            source=chain[-1].source_file if chain else None
            if source is None:
                for filename,target in (('MEMORY.md','memory'),('USER.md','user')):
                    memory=MemoryStore()
                    for candidate in memory._read_file(memory._path_for(target)):
                        if self._ref(context,filename,candidate)==item_ref:
                            source=filename
            if source is None:
                raise ValueError('stale or unknown context item')
            target='user' if source=='USER.md' else 'memory'
            memory=load_on_disk_store()
            retained=[]
            def change_locked(entries,limit):
                changes=self._changes(context)
                retries=[change for change in changes if change.request_id==request_id]
                if retries:
                    change=retries[0]
                    if (change.item_ref,change.expected_revision,change.content,change.owner_ref,change.origin)!=(
                            item_ref,expected_revision,content,context.user_id,origin):
                        raise ValueError('request_id reused for different mutation')
                    retained.append(change)
                    return self._projection(entries,[c for c in changes if c.source_file==source]), 'Retained change retried.'
                chain=[change for change in changes if change.item_ref==item_ref]
                # Validate against the exact raw snapshot while holding Hermes' file lock;
                # its live parser deduplicates entries and cannot prove unique identity.
                raw,ok=memory._read_raw_checked(memory._path_for(target))
                if not ok:
                    raise ValueError('memory source unreadable')
                raw_entries=memory._parse_entries(raw)
                if chain:
                    previous=chain[-1]
                    if previous.owner_ref != context.user_id:
                        raise PermissionError('context item belongs to another actor')
                    prior=previous.content
                    if previous.artifact_hash!=expected_revision or prior is None:
                        raise ValueError('stale expected revision')
                else:
                    matches=[entry for entry in raw_entries if self._ref(context,source,entry)==item_ref]
                    if len(matches)>1:
                        raise ValueError('ambiguous context item')
                    if len(matches)!=1 or _digest(matches[0])!=expected_revision:
                        raise ValueError('stale expected revision')
                    prior=matches[0]
                if raw_entries.count(prior)>1:
                    raise ValueError('ambiguous context item')
                if prior not in entries:
                    raise ValueError('stale source content')
                if content is not None and any(
                    change.item_ref != item_ref and change.source_file == source
                    and content in (change.prior_content, change.content)
                    for change in changes
                ):
                    raise ValueError('historical correction aliases another item identity')
                if content is not None and content!=prior and content in entries:
                    raise ValueError('ambiguous correction merges sibling identity')
                result=[content if entry==prior else entry for entry in entries] if content is not None else [entry for entry in entries if entry!=prior]
                if len(ENTRY_DELIMITER.join(result))>limit:
                    raise ValueError('memory character limit exceeded')
                change=RetainedContextItemChangeV1.create(organization_id=context.organization_id,
                    workspace_id=context.workspace_id,owner_ref=context.user_id,item_ref=item_ref,source_file=source,
                    revision=len(chain)+1,expected_revision=expected_revision,prior_content=prior,content=content,
                    action='suppress' if content is None else 'correct',request_id=request_id,
                    recorded_at=datetime.now(timezone.utc),origin=origin)
                self.store.put(change)  # Commit recoverable intent before projection; still inside the lock.
                retained.append(change)
                return result, 'Retained context item changed.'
            try:
                result=memory._mutate(target,change_locked)
            except OSError:
                if not retained:
                    raise
                result={'success':False}
            if not result.get('success') and not retained:
                raise ValueError(result.get('error','memory mutation failed'))
            change=retained[0]
            future=load_on_disk_store()
            future_entries=future._entries_for(target)
            historical={c.prior_content for c in self._changes(context) if c.item_ref==item_ref}
            confirmed=(content in future_entries if content is not None else True) and not any(
                old in future_entries for old in historical if old!=content)
            row = self._row(ref=item_ref,revision=change.artifact_hash,source=source,content=content,
                available=content is not None,pending=not confirmed)
            if confirmed and content is not None and not future.target_enabled(target):
                row['runtime_availability']='unavailable'
                row['leading_answer']='Retained, but this memory source is disabled for future sessions.'
            return row

    def history(self, *, context, item_ref):
        self._check(context)
        return [change.model_dump(mode='json') for change in self._changes(context) if change.item_ref==item_ref]
