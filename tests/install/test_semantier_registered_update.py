import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('driver',Path(__file__).with_name('semantier_registered_update.py'))
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)

def test_retained_validation_binds_both_observed_data_roots():
    p=d.Windows.__new__(d.Windows)
    p.request={}
    owner={'sid':'actor','session_sid':'actor','session':1,'root_id':'custody',
           'purpose':d.RETAINED_BASELINE,'desktop_root_id':'desktop','hermes_root_id':'home'}
    p.observe=lambda: dict(owner)
    p.validate(owner)
    for key in ('desktop_root_id','hermes_root_id'):
        changed=dict(owner);changed[key]='replacement'
        p.observe=lambda: changed
        with pytest.raises(ValueError,match='physical data root changed'):p.validate(owner)

def test_retained_verification_reuse_rejects_altered_evidence(tmp_path):
    evidence=tmp_path/'evidence.json';evidence.write_text('{"scope":"EXACT_INSTALLED_BASELINE"}')
    p=d.Windows.__new__(d.Windows)
    p.request={'purpose':d.RETAINED_BASELINE,'baseline_verification':str(evidence)}
    with pytest.raises(ValueError,match='evidence changed'):p.call('verify-existing-baseline')

def test_retained_snapshots_use_real_hermes_root_not_fixture(tmp_path, monkeypatch):
    custody=tmp_path/'custody';custody.mkdir()
    home=tmp_path/'actual-home';home.mkdir();(home/'state.db').write_bytes(b'data')
    calls=[]
    class Owner:
        @staticmethod
        def unlinked(path): return path
        @staticmethod
        def backup_database_snapshot(source, destination, prior, tables):
            calls.append(source)
            return {'tables':{}}
    @d.contextlib.contextmanager
    def hold(path, identity):
        assert path == home and identity == 'physical-home'
        yield
    monkeypatch.setattr(d,'held_windows_root',hold)
    p=d.Windows.__new__(d.Windows)
    p.request={'purpose':d.RETAINED_BASELINE,'root':str(custody),'hermes_root':str(home),
               'semantic_observer':{'sha256':'hash'},
               'semantic_databases':[{'path':'hermes/state.db','tables':{'sessions':['id']}}]}
    p.semantic_owner=lambda: Owner()
    p.observe=lambda: {'hermes_root_id':'physical-home'}
    p.capture_semantics()
    assert calls == [home/'state.db']
    p.request['semantic_databases'][0]['path']='userdata/hermes-home/state.db'
    with pytest.raises(ValueError,match='relative semantic database path'):p.capture_semantics()

def test_verifier_selects_driver_python_only_for_read_only_child(monkeypatch):
    calls=[]
    def run(*args, **kwargs):
        calls.append(kwargs)
        return type('Result', (), {'returncode': 0, 'stdout': '{}', 'stderr': ''})()
    monkeypatch.setattr(d.subprocess, 'run', run)
    platform=d.Windows.__new__(d.Windows)
    platform.request={};platform.powershell='powershell'
    platform.call('verify-existing-baseline')
    platform.call('observe')
    assert calls[0]['env']['HERMES_PYTHON'] == d.sys.executable
    assert calls[1]['env'] is None

class Platform:
    def __init__(self):
        self.state={'sid':'test','session':2,'root_id':'root','package':None,'process':None};self.calls=[];self.fail=None
    def observe(self): return __import__('copy').deepcopy(self.state)
    def validate(self, owner):
        if any(self.state[k]!=owner[k] for k in ('sid','session','root_id')):raise ValueError('context changed')
    def perform(self, action, args):
        self.calls.append(action)
        if self.fail==action:raise RuntimeError('operation failed')
        if action=='register':self.state['package']=args
        if action=='remove':self.state['package']=None;self.state['process']=None
        if action=='launch':self.state['process']={'pid':1,'birth':'a'}
        return self.observe()

def seed(t,p):
    package=p.state['package'];process=p.state['process']
    p.state['package']=None;p.state['process']=None;t.initialize(p.observe())
    if package is not None:
        t.operation('register',package)
        if process is not None:t.operation('launch',{})
        p.calls=[]

@pytest.fixture
def tx(tmp_path):
    p=Platform();return d.Transaction(tmp_path/'journal.jsonl',p),p

def test_intent_before_action_and_crash_reconcile(tx):
    t,p=tx;owner=p.observe();t.initialize(owner)
    target={'full':'baseline','hash':'a'}
    t.fault='before_action'
    with pytest.raises(d.Interrupted):t.operation('register',target)
    assert not p.calls
    assert t.reconcile()=='not_performed'
    t.fault='after_action'
    with pytest.raises(d.Interrupted):t.operation('register',target)
    assert p.calls==['register']
    assert t.reconcile()=='performed'
    assert p.calls==['register']

def test_changed_identity_preserves_ambiguous_state(tx):
    t,p=tx;seed(t,p);t.fault='after_action'
    with pytest.raises(d.Interrupted):t.operation('register',{'full':'owned'})
    p.state['package']={'full':'unrelated'}
    with pytest.raises(ValueError):t.reconcile()
    assert p.calls==['register']

def test_context_and_failed_operation(tx):
    t,p=tx;seed(t,p);p.state['root_id']='other'
    with pytest.raises(ValueError):t.operation('register',{})
    assert not p.calls

def test_concurrent_lock(tmp_path):
    with d.exclusive(tmp_path/'lock'):
        with pytest.raises((BlockingIOError,OSError)):
            with d.exclusive(tmp_path/'lock'):pass

def test_failed_operation_retains_pending_and_does_not_uninstall(tx):
    t,p=tx;seed(t,p);p.fail='register'
    with pytest.raises(RuntimeError):t.operation('register',{'full':'owned'})
    assert t.pending()['action']=='register'
    assert t.reconcile()=='not_performed'
    assert p.calls==['register']

@pytest.mark.parametrize('key,value',[('sid','other'),('session',3),('root_id','replacement')])
def test_recovery_refuses_context_replacement(tx,key,value):
    t,p=tx;seed(t,p);t.fault='after_action'
    with pytest.raises(d.Interrupted):t.operation('register',{'full':'owned'})
    p.state[key]=value
    with pytest.raises(ValueError):t.reconcile()
    assert p.calls==['register']

def test_cleanup_with_changed_process_preserves_uncertainty(tx):
    t,p=tx;p.state['package']={'full':'owned'};p.state['process']={'pid':1,'birth':'a'};seed(t,p)
    t.fault='before_action'
    with pytest.raises(d.Interrupted):t.operation('remove',{})
    p.state['process']={'pid':1,'birth':'different'}
    with pytest.raises(ValueError):t.reconcile()
    assert p.calls==[]

def test_baseline_remains_registered_after_launch_failure(tx):
    t,p=tx;seed(t,p);t.operation('register',{'full':'owned'})
    p.fail='launch'
    with pytest.raises(RuntimeError):t.operation('launch',{})
    assert p.state['package']=={'full':'owned'}
    assert 'remove' not in p.calls

class UpgradePlatform(Platform):
    def __init__(self):
        super().__init__();self.state['package']={'version':'0.0.58.0','family':'family'};self.state['process']={'pid':1,'birth':'a'}
        self.target={'version':'0.0.59.0','family':'family'};self.relaunch=True;self.changed=False
    def successor(self,args,before,after):return after['package']==args and after['process'] is not None
    def call(self,action,**kwargs):
        if action=='artifact':return self.target
        if action=='ready':
            if not self.state['process']:raise ValueError('not ready')
            return {}
        if action=='conservation':return {'data':'changed' if self.changed and self.state['package']==self.target else 'original'}
    def perform(self,action,args):
        super().perform(action,args)
        if action=='update':self.state['package']=args;self.state['process']={'pid':2,'birth':'b'} if self.relaunch else None

def test_successor_without_relaunch_is_partial_not_acceptance(tmp_path):
    p=UpgradePlatform();p.relaunch=False;t=d.Transaction(tmp_path/'j',p);seed(t,p);t.record('baseline_ready',facts={'healthy':True},observed=p.observe())
    with pytest.raises(ValueError):d.run_upgrade(t,p)
    assert t.reconcile()=='performed_partial'
    assert not any(r['kind']=='acceptance' for r in t.events())

def test_conservation_failure_and_success_through_upgrade(tmp_path):
    for changed in (True,False):
        p=UpgradePlatform();p.changed=changed;t=d.Transaction(tmp_path/str(changed),p);seed(t,p);t.record('baseline_ready',facts={'healthy':True},observed=p.observe())
        if changed:
            with pytest.raises(ValueError,match='Conservation'):d.run_upgrade(t,p)
        else:d.run_upgrade(t,p)
        assert any(r['kind']=='acceptance' for r in t.events()) is (not changed)
        assert 'remove' not in p.calls


def test_retained_upgrade_compares_database_semantics_not_sqlite_file_bytes(tmp_path):
    class RetainedUpgrade(UpgradePlatform):
        def __init__(self):
            super().__init__()
            self.request={'purpose':d.RETAINED_BASELINE,'semantic_databases':[{'path':'userdata/state.db'}]}
        def call(self,action,**kwargs):
            if action=='verify-existing-baseline': return {'scope':'EXACT_INSTALLED_BASELINE','observed':self.observe()}
            if action=='ready': return {'healthy':True,'gateway':{'pid':12,'birth':'a'}}
            if action=='conservation': return {'chats:userdata/state.db':self.state['package']['version'], 'theme:userdata/theme.json':'same'}
            return super().call(action,**kwargs)
        def capture_semantics(self): return {'scope':'fixture database backup'}
        def verify_semantics(self,before): return {'scope':'fixture semantic continuity'}
    p=RetainedUpgrade();t=d.Transaction(tmp_path/'j',p);owner=p.observe()
    t.initialize({**owner,'purpose':d.RETAINED_BASELINE})
    t.record('baseline_adopted',observed=owner,verification={'scope':'fixture'})
    t.record('baseline_ready',facts={'healthy':True,'gateway':{'pid':12,'birth':'a'}},observed=owner)
    d.run_upgrade(t,p)
    assert any(row['kind']=='semantic_preservation' for row in t.events())
    assert any(row['kind']=='acceptance' for row in t.events())


def test_retained_upgrade_refuses_when_consistent_backup_is_unavailable(tmp_path):
    class MissingBackup(UpgradePlatform):
        def __init__(self):
            super().__init__();self.request={'purpose':d.RETAINED_BASELINE}
        def capture_semantics(self):return None
    p=MissingBackup();t=d.Transaction(tmp_path/'j',p);owner=p.observe()
    t.initialize({**owner,'purpose':d.RETAINED_BASELINE})
    t.record('baseline_adopted',observed=owner,verification={'scope':'fixture'})
    t.record('baseline_ready',facts={'healthy':True},observed=owner)
    with pytest.raises(ValueError,match='consistent semantic backups'):d.run_upgrade(t,p)
    assert 'update' not in p.calls and t.pending() is None

def test_unlinked_outcome_and_empty_package_intent_refused(tx):
    t,p=tx;seed(t,p);t.record('outcome',intent_at='missing',action='register',status='performed',observed=p.observe())
    with pytest.raises(ValueError):t.reconcile()

def test_journal_links_refused_before_write(tmp_path):
    original=tmp_path/'original';original.write_text('retain',encoding='utf-8')
    link=tmp_path/'link';link.symlink_to(original)
    with pytest.raises(ValueError):
        with d.safe_file(link,'a'):pass
    hard=tmp_path/'hard';__import__('os').link(original,hard)
    with pytest.raises(ValueError):
        with d.safe_file(hard,'a'):pass
    assert original.read_text(encoding='utf-8-sig')=='retain'

def test_cleanup_interrupted_after_close_can_reconcile_package_removal(tx):
    t,p=tx;p.state['package']={'full':'owned'};p.state['process']={'pid':1,'birth':'a'};seed(t,p);t.fault='before_action'
    with pytest.raises(d.Interrupted):t.operation('remove',{})
    p.state['process']=None
    assert t.reconcile()=='not_performed'
    t.fault=None;t.operation('remove',{})
    assert p.state['package'] is None

def test_download_watcher_finishes_before_transaction_returns(tmp_path):
    import threading
    w=d.Windows.__new__(d.Windows);w.request={'root':str(tmp_path)};w.precondition={}
    entered=threading.Event();release=threading.Event()
    def call(action,**kwargs):
        if action=='download':entered.set();release.wait(2);return {'sha':'observed'}
        assert entered.wait(2);release.set();return {'updated':True}
    w.call=call
    assert w.perform('update',{})=={'updated':True}
    assert (tmp_path/'download-readback.json').exists()

def test_settled_registration_replacement_cannot_be_adopted_for_update(tmp_path):
    p=UpgradePlatform();t=d.Transaction(tmp_path/'j',p);seed(t,p)
    t.record('baseline_ready',facts={'healthy':True},observed=p.observe())
    p.state['package']={'version':'0.0.58.0','family':'other'}
    with pytest.raises(ValueError,match='replaced'):d.run_upgrade(t,p)
    assert 'update' not in p.calls

def test_settled_process_replacement_cannot_be_cleaned(tx):
    t,p=tx;seed(t,p);t.operation('register',{'full':'owned'});t.operation('launch',{})
    p.state['process']={'pid':1,'birth':'reused'}
    with pytest.raises(ValueError,match='replaced'):t.operation('remove',{})
    assert 'remove' not in p.calls


def test_user_retained_trial_never_removes_registration(tmp_path):
    p=Platform();p.state['package']={'full':'user-retained'}
    t=d.Transaction(tmp_path/'journal',p)
    t.initialize({**p.observe(),'purpose':'caddy-chrish-existing-baseline-v2'})
    with pytest.raises(ValueError,match='retained'):
        t.operation('remove',{})
    with pytest.raises(ValueError,match='retained'):
        d.run_cleanup(t,p,{'purpose':'caddy-chrish-existing-baseline-v2'})
    assert p.state['package']=={'full':'user-retained'}
    intent=t.record('intent',action='update',args={'full':'successor'},before=p.observe())
    p.state['package']={'full':'successor'}
    t.record('outcome',intent_at=intent['at'],action='update',status='performed',observed=p.observe())
    with pytest.raises(ValueError,match='retained'):
        d.run_cleanup(t,p,{})
    assert p.state['package']=={'full':'successor'}
    assert p.calls==[]


def test_existing_baseline_adoption_checks_actual_observation_before_recording(tmp_path):
    import copy
    class AdoptPlatform(Platform):
        def __init__(self):
            super().__init__()
            self.request={'purpose':d.RETAINED_BASELINE}
            self.state.update(sid=d.RETAINED_SID,session_sid=d.RETAINED_SID,
                              package={'full':'A62','version':'0.0.62.0'},process={'pid':11,'birth':'a'},related=[])
            self.calls=[];self.bad=False
        def call(self,action,**kwargs):
            self.calls.append(action)
            if action=='artifact': return {'full':'A62','version':'0.0.62.0','related':[]}
            if action=='verify-existing-baseline':
                state=copy.deepcopy(self.state)
                if self.bad: state['package']['version']='wrong'
                return {'scope':'EXACT_INSTALLED_BASELINE','observed':state}
            if action=='ready': return {'healthy':True,'gateway':{'pid':12,'birth':'a'}}
            raise AssertionError(action)
    p=AdoptPlatform();t=d.Transaction(tmp_path/'journal',p);p.bad=True
    wrong_actor=p.observe();wrong_actor['sid']='S-1-5-21-1-2-3-4'
    with pytest.raises(ValueError,match='actor'):d.run_adopt_baseline(t,p,wrong_actor)
    assert p.calls==[] and not t.events()
    with pytest.raises(ValueError):d.run_adopt_baseline(t,p,p.observe())
    assert not t.events()
    p.bad=False;p.calls=[];d.run_adopt_baseline(t,p,p.observe())
    assert p.calls==['artifact','verify-existing-baseline','ready']
    assert [r['kind'] for r in t.events()]==['owner','baseline_adopted','baseline_ready']
    t.assert_owned(p.observe(),require_process=True)
    p.bad=True
    with pytest.raises(ValueError,match='verification'):
        t.operation('update',{'full':'successor','version':'0.0.63.0'})
    assert t.pending() is None
    assert 'update' not in p.calls
    p.bad=False
    original_call=p.call
    p.call=lambda action,**kwargs: {'healthy':True,'gateway':{'pid':99,'birth':'replacement'}} if action=='ready' else original_call(action,**kwargs)
    with pytest.raises(ValueError,match='gateway changed'):
        t.operation('update',{'full':'successor','version':'0.0.63.0'})
    assert t.pending() is None and 'update' not in p.calls
    with pytest.raises(ValueError,match='retained'):t.operation('remove',{})
    assert p.state['package']['full']=='A62'


@pytest.mark.parametrize('session_sid',[None,'different-signed-in-user'])
def test_windows_observation_requires_actual_wts_account(session_sid):
    w=d.Windows.__new__(d.Windows)
    state={'sid':'test-account','session':2,'root_id':'owned','package':None,'process':None}
    if session_sid is not None:state['session_sid']=session_sid
    w.call=lambda action:state
    with pytest.raises(ValueError,match='WTS'):w.observe()


def test_windows_recovery_rejects_missing_recorded_wts_binding():
    w=d.Windows.__new__(d.Windows)
    state={'sid':'test-account','session_sid':'test-account','session':2,'root_id':'owned','package':None,'process':None}
    w.call=lambda action:state
    owner={k:v for k,v in state.items() if k!='session_sid'}
    with pytest.raises(ValueError,match='WTS'):w.validate(owner)
    w.validate(state)


class RelatedPlatform(Platform):
    def __init__(self):
        super().__init__();self.state['related']=[]
    def perform(self,action,args):
        self.calls.append(action)
        if action=='register':
            self.state['package']={k:v for k,v in args.items() if k!='related'}
            self.state['related']=args.get('related',[])
        elif action=='remove':self.state['package']=None;self.state['process']=None


def test_related_registration_is_part_of_durable_outcome(tmp_path):
    p=RelatedPlatform();t=d.Transaction(tmp_path/'j',p);t.initialize(p.observe())
    target={'full':'main','related':[{'full':'runtime_1'}]}
    t.operation('register',target)
    assert t.events()[-1]['observed']['related']==target['related']
    p.state['related']=[{'full':'runtime_2'}]
    with pytest.raises(ValueError,match='Related'):t.operation('remove',{})
    assert p.calls==['register']


def test_related_resources_surviving_main_removal_are_not_cleanup_success(tmp_path):
    p=RelatedPlatform();t=d.Transaction(tmp_path/'j',p);t.initialize(p.observe())
    t.operation('register',{'full':'main','related':[{'full':'runtime_1'}]})
    with pytest.raises(ValueError,match='outcome'):t.operation('remove',{})
    with pytest.raises(ValueError,match='Uncertain'):t.reconcile()
    assert t.pending()['action']=='remove'


def test_legacy_registration_cannot_adopt_unexpected_related_package(tmp_path):
    p=RelatedPlatform();t=d.Transaction(tmp_path/'j',p);t.initialize(p.observe())
    original=p.perform
    def perform(action,args):
        original(action,args)
        if action=='register':p.state['related']=[{'full':'unqualified'}]
    p.perform=perform
    with pytest.raises(ValueError,match='outcome'):t.operation('register',{'full':'legacy'})
    with pytest.raises(ValueError,match='Uncertain'):t.reconcile()


def test_related_registration_crash_reconciles_exact_set_only(tmp_path):
    p=RelatedPlatform();t=d.Transaction(tmp_path/'j',p);t.initialize(p.observe())
    t.fault='after_action'
    with pytest.raises(d.Interrupted):t.operation('register',{'full':'main','related':[{'full':'runtime_1'}]})
    p.state['related']=[]
    with pytest.raises(ValueError,match='Uncertain'):t.reconcile()
    p.state['related']=[{'full':'runtime_1'}]
    assert t.reconcile()=='performed'
    assert p.calls==['register']


def test_unknown_related_observation_does_not_pass_removal():
    with pytest.raises(ValueError,match='Related'):
        d.Transaction.__new__(d.Transaction).expected('remove',{}, {}, {'package':None,'process':None,'related':None})


def test_graph_prediction_binds_exact_static_members_then_records_local_incarnations(tmp_path):
    p=RelatedPlatform();t=d.Transaction(tmp_path/'j',p);t.initialize(p.observe())
    target={'full':'main','related':[{'full':'runtime_1','family':'same_family'}]}
    original=p.perform
    def perform(action,args):
        original(action,args)
        if action=='register':p.state['related']=[{'full':'runtime_1','family':'same_family','root_id':'actual-volume:file','install_path':'actual installed location'}]
    p.perform=perform
    t.operation('register',target)
    assert t.events()[-1]['observed']['related'][0]['root_id']=='actual-volume:file'
    p.state['related'][0]['root_id']='replaced:file'
    with pytest.raises(ValueError,match='Related'):t.operation('remove',{})
    assert p.calls==['register']


class NativeQualificationPlatform(Platform):
    def proof_binding(self):return 'fixture-proof-binding'
    def proof_bindings(self,current):return {current}
    def __init__(self):
        super().__init__();self.request={'purpose':d.NATIVE_QUALIFICATION}
    def call(self,action,**kwargs):
        self.calls.append(action)
        return {'artifact':{'full':'native'},'native-proof':{'scope':'local-only'},'ready':{'gateway':None}}[action]


def test_native_qualification_does_not_create_a_migration_baseline(tmp_path):
    p=NativeQualificationPlatform();t=d.Transaction(tmp_path/'j',p)
    d.run_native_qualification(t,p,p.observe())
    rows=t.events()
    assert rows[0]['owner']['purpose']==d.NATIVE_QUALIFICATION
    assert not any(row['kind']=='baseline_ready' for row in rows)
    with pytest.raises(ValueError,match='Baseline readiness'):d.run_upgrade(t,p)
    assert 'update' not in p.calls


def test_native_qualification_refuses_to_adopt_a_regular_transaction(tmp_path):
    p=NativeQualificationPlatform();t=d.Transaction(tmp_path/'j',p);t.initialize(p.observe())
    with pytest.raises(ValueError,match='purpose'):d.run_native_qualification(t,p,p.observe())
    assert not p.calls


def test_native_graph_binding_uses_original_bom_and_crlf_bytes(tmp_path):
    import base64,hashlib,json
    key='releases/channel-builds/'+'a'*32+'/main.msix'
    graph={'schema':1,'transport':'windows-package-uri','root':key,'payloadInventorySha256':'b'*64,'objects':[
        {'key':key,'kind':'main-package','identity':'Semantier.Agent.Acceptance','publisher':'CN=Semantier Technology, O=Semantier Technology','version':'0.0.59.0','arch':'x64','size':1,'sha256':'c'*64,'nativeDigest':base64.b64encode(b'd'*32).decode(),'references':[]}]}
    raw=('\ufeff'+json.dumps(graph,indent=2).replace('\n','\r\n')).encode('utf-8')
    file=tmp_path/'native-graph.json';file.write_bytes(raw)
    request={'target':{'path':str(tmp_path/'main.msix'),'sha256':'c'*64,'build_id':'a'*32,'version':'0.0.59.0','native_graph':{'path':str(file),'sha256':hashlib.sha256(raw).hexdigest()}}}
    assert d.load_native_target(request)==graph
    file.write_bytes(raw+b'changed')
    with pytest.raises(ValueError,match='source digest'):d.load_native_target(request)


@pytest.mark.parametrize('value', ['https://example.com','http://127.0.0.1:70000','http://user@127.0.0.1:80','http://127.0.0.1:80/path','http://127.0.0.1:80?x=y'])
def test_native_engineering_origin_refuses_unbound_address(value):
    with pytest.raises(ValueError,match='origin'):
        d.validate_native_origin({'purpose':d.NATIVE_QUALIFICATION,'native_origin':value})


def test_native_engineering_origin_requires_its_separate_purpose():
    with pytest.raises(ValueError,match='purpose'):
        d.validate_native_origin({'native_origin':'http://127.0.0.1:8926'})
    assert d.validate_native_origin({'purpose':d.NATIVE_QUALIFICATION,'native_origin':'http://127.0.0.1:8926'})=='http://127.0.0.1:8926'


def test_native_resume_refuses_absent_registration_without_deployment(tmp_path):
    p=NativeQualificationPlatform();t=d.Transaction(tmp_path/'j',p)
    with pytest.raises(ValueError,match='Existing native registration'):
        d.run_native_qualification(t,p,p.observe(),require_existing=True)
    assert not p.calls and not t.events()


def test_native_resume_reconciles_late_registration_without_registering_again(tmp_path):
    p=NativeQualificationPlatform();t=d.Transaction(tmp_path/'j',p)
    t.initialize({**p.observe(),'purpose':d.NATIVE_QUALIFICATION})
    t.fault='after_action'
    with pytest.raises(d.Interrupted):t.operation('register',{'full':'native'})
    t.fault=None;p.calls=[]
    d.run_native_qualification(t,p,p.observe(),require_existing=True)
    assert 'register' not in p.calls
    assert t.pending() is None
    assert any(row['kind']=='outcome' and row['status']=='performed' for row in t.events())


def test_native_resume_reuses_exact_bound_proof_after_launch_failure(tmp_path):
    p=NativeQualificationPlatform();p.proof_binding=lambda:'binding-v1'
    t=d.Transaction(tmp_path/'j',p);p.fail='launch'
    with pytest.raises(RuntimeError):d.run_native_qualification(t,p,p.observe())
    p.fail=None;p.calls=[]
    d.run_native_qualification(t,p,p.observe(),require_existing=True)
    assert 'native-proof' not in p.calls
    assert 'register' not in p.calls
    assert p.calls.count('launch')==1


def test_native_resume_reverifies_when_proof_binding_changes(tmp_path):
    p=NativeQualificationPlatform();p.proof_binding=lambda:'binding-v1'
    t=d.Transaction(tmp_path/'j',p);d.run_native_qualification(t,p,p.observe())
    p.proof_binding=lambda:'binding-v2';p.calls=[]
    d.run_native_qualification(t,p,p.observe(),require_existing=True)
    assert p.calls.count('native-proof')==1
    assert 'register' not in p.calls and 'launch' not in p.calls


@pytest.mark.parametrize('changed',['request','host','powershell','verifier'])
def test_native_proof_binding_invalidates_changed_execution_inputs(tmp_path,monkeypatch,changed):
    p=d.Windows.__new__(d.Windows);p.request={'target':{'sha256':'original'}}
    binary=tmp_path/'powershell.exe';binary.write_bytes(b'original host binary');p.powershell=str(binary)
    monkeypatch.setattr(d.sys,'getwindowsversion',lambda:(10,0,1,2,''),raising=False)
    original=p.proof_binding()
    assert p.proof_binding()==original
    if changed=='request':p.request['target']['sha256']='changed'
    elif changed=='host':monkeypatch.setattr(d.sys,'getwindowsversion',lambda:(10,0,2,2,''))
    elif changed=='powershell':binary.write_bytes(b'changed host binary')
    else:
        read=Path.read_bytes
        monkeypatch.setattr(Path,'read_bytes',lambda file:read(file)+(b'changed' if file.name=='native-payload-probe.py' else b''))
    assert p.proof_binding()!=original


@pytest.fixture
def prior_proof_source(tmp_path,monkeypatch):
    import hashlib,json
    p=d.Windows.__new__(d.Windows);p.request={'target':{'sha256':'same-target'}}
    binary=tmp_path/'powershell.exe';binary.write_bytes(b'host');p.powershell=str(binary)
    monkeypatch.setattr(d.sys,'getwindowsversion',lambda:(10,0,1,2,''),raising=False)
    current=Path(d.__file__).resolve().parents[2];root=tmp_path/'prior'
    for name in p.proof_sources(current):
        file=root/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes((current/name).read_bytes())
    platform=root/'tests/install/e2e-assets/semantier-registered-platform.ps1'
    text=platform.read_text(encoding='utf-8-sig')
    platform.write_text(text.replace(' function Observe {',' # prior observation implementation\n function Observe {'),encoding='utf-8')
    def bind():
        rows=[{'path':name,'sha256':hashlib.sha256((root/name).read_bytes()).hexdigest()} for name in p.proof_sources(current)]
        file=root/'source-binding.json';raw=json.dumps({'files':rows}).encode();file.write_bytes(raw)
        p.proof_source=(file,hashlib.sha256(raw).hexdigest())
    bind()
    return p,root,bind


def test_prior_proof_source_reuses_only_unchanged_claim_verifiers(prior_proof_source):
    p,root,_=prior_proof_source
    original=p.proof_binding(root);current=p.proof_binding()
    assert current!=original
    assert p.proof_bindings(current)=={current,original}


def test_prior_proof_source_rejects_changed_claim_verifier(prior_proof_source):
    p,root,bind=prior_proof_source
    file=root/'tests/install/e2e-assets/native-payload-probe.py'
    file.write_bytes(file.read_bytes()+b'\n# different verifier\n');bind()
    assert p.proof_bindings(p.proof_binding())=={p.proof_binding()}


@pytest.mark.parametrize('changed',['request','host'])
def test_prior_proof_source_cannot_rebind_changed_inputs(prior_proof_source,monkeypatch,changed):
    p,root,_=prior_proof_source;original=p.proof_binding(root)
    if changed=='request':p.request['target']['sha256']='different-target'
    else:monkeypatch.setattr(d.sys,'getwindowsversion',lambda:(10,0,2,2,''))
    assert original not in p.proof_bindings(p.proof_binding())


def test_prior_proof_source_refuses_unpinned_manifest(prior_proof_source):
    p,_,_=prior_proof_source;file,_=p.proof_source;p.proof_source=(file,'0'*64)
    with pytest.raises(ValueError,match='source binding mismatch'):p.proof_bindings(p.proof_binding())


def test_prior_proof_source_refuses_changed_staged_bytes(prior_proof_source):
    p,root,_=prior_proof_source
    file=root/'tests/install/semantier_registered_update.py';file.write_bytes(file.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='source bytes changed'):p.proof_bindings(p.proof_binding())


def native_handoff_case():
    request={'authorized_sid':'actor','selected_delivery_base':'https://s3.semantier.com',
             'native_handoff_source_sha256':'a'*64,'native_digest_source_sha256':'9'*64,'native_graph_value':{'root':'bound','objects':[{'sha256':'b'*64}]},
             'target':{'manifest_sha256':'c'*64,'native_graph':{'sha256':'d'*64},'build_id':'e'*32,'commit':'f'*40,'version':'0.0.61.0'}}
    before={'package':{'full':'baseline','version':'0.0.60.0','family':'same-family'},'process':{'pid':1,'birth':'2026-10-29T14:00:00.0000000+00:00'},'session':1}
    process={'pid':2,'birth':'2026-10-29T14:01:03.0000000+00:00','exe':'registered.exe','exe_id':'file-id'}
    intent={'authorizedSid':'actor','session':1,'manifestSha256':'c'*64,'graphSha256':'d'*64,'graph':request['native_graph_value'],
            'sourceSha256':'a'*64,'digestSourceSha256':'9'*64,'base':request['selected_delivery_base'],'targetBuildId':'e'*32,'targetCommit':'f'*40,
            'version':'0.0.61.0','currentVersion':'0.0.60.0','fullName':'baseline','familyName':'same-family'}
    proof={'transport':'windows-package-uri','intent':intent,'deployment':{'status':'OS_SUCCESS','manifestSha256':'c'*64,'activityId':'11111111-1111-1111-1111-111111111111','producedAt':'2026-10-29T14:01:01Z'},
           'relaunch_intent':{'manifestSha256':'c'*64,'producedAt':'2026-10-29T14:01:02Z'},
           'completion':{'status':'TARGET_GRAPH_REGISTERED_PROCESS_LAUNCHED','manifestSha256':'c'*64,'fullName':'successor','pid':2,'birth':process['birth'],'exe':'registered.exe','exeId':'file-id','sid':'actor','session':1,'producedAt':'2026-10-29T14:01:04Z'},
           'observed':{'sid':'actor','session':1,'process':process,'package':{'full':'successor','family':'same-family','commit':'f'*40,'build_id':'e'*32,'version':'0.0.61.0'}}}
    return request,before,proof


def test_native_handoff_binds_real_production_owner_and_successor_incarnation():
    request,before,proof=native_handoff_case()
    assert d.verify_native_handoff(proof,request,before)==proof


@pytest.mark.parametrize('part,key,value',[
    ('intent','sourceSha256','wrong'),('intent','digestSourceSha256','wrong'),('intent','base','https://other.invalid'),('intent','authorizedSid','other'),
    ('intent','graph',{'root':'different'}),('intent','graphSha256','0'*64),('deployment','status','UNKNOWN'),
    ('completion','pid',3),('completion','birth','2026-10-29T13:00:00Z'),('completion','exeId','other'),
    ('completion','status','TARGET_GRAPH_REGISTERED_ACTIVATION_REQUESTED'),('relaunch_intent','producedAt','2026-10-29T14:01:09Z')])
def test_native_handoff_refuses_unbound_or_manual_successor_claims(part,key,value):
    request,before,proof=native_handoff_case();proof[part][key]=value
    with pytest.raises(ValueError):d.verify_native_handoff(proof,request,before)


def test_readonly_precondition_refusal_does_not_leave_an_action_intent(tx):
    t,p=tx;seed(t,p)
    def refuse(*_args):raise ValueError('missing native binding')
    p.check=refuse
    with pytest.raises(ValueError,match='missing native binding'):t.operation('register',{'full':'target'})
    assert t.pending() is None and p.calls==[]


@pytest.mark.parametrize('target_registered',[False,True])
def test_native_reconciliation_requires_dead_waiter_typed_completion_and_exact_os_set(target_registered):
    request,before,proof=native_handoff_case()
    proof['waiterAlive']=False;proof['deploymentIntentPresent']=True
    proof['observed']['process']=None
    target=dict(proof['observed']['package'])
    if not target_registered:
        proof.pop('deployment')
        proof['failure']={'status':'FAILURE','operationCompleted':True,'hresult':'80080219','manifestSha256':request['target']['manifest_sha256']}
        proof['observed']['package']=dict(before['package'])
    now=__import__('copy').deepcopy(proof['observed'])
    expected='TARGET_GRAPH_REGISTERED' if target_registered else 'BASELINE_INTACT_FAILED'
    assert d.verify_native_reconciliation(proof,request,before,now,target)==expected
    proof['waiterAlive']=True
    with pytest.raises(ValueError):d.verify_native_reconciliation(proof,request,before,now,target)
    proof['waiterAlive']=False
    if target_registered:proof['deployment']['status']='UNKNOWN'
    else:proof['failure']['operationCompleted']=False
    with pytest.raises(ValueError):d.verify_native_reconciliation(proof,request,before,now,target)


def test_native_failed_reconciliation_does_not_release_changed_related_packages():
    request,before,proof=native_handoff_case();proof['waiterAlive']=False;proof['deploymentIntentPresent']=True;proof.pop('deployment')
    proof['failure']={'status':'FAILURE','operationCompleted':True,'hresult':'80080219','manifestSha256':request['target']['manifest_sha256']}
    proof['observed'].update(package=dict(before['package']),process=None,related=[{'full':'unexpected-resource'}])
    with pytest.raises(ValueError):d.verify_native_reconciliation(proof,request,before,proof['observed'],{})


def test_native_reconciliation_releases_only_proven_not_started_attempt():
    request,before,proof=native_handoff_case();proof.pop('deployment')
    proof.update(waiterAlive=False,deploymentIntentPresent=False)
    proof['observed'].update(package=dict(before['package']),process=None)
    assert d.verify_native_reconciliation(proof,request,before,proof['observed'],{})=='NOT_STARTED'
    proof['deploymentIntentPresent']=True
    with pytest.raises(ValueError):d.verify_native_reconciliation(proof,request,before,proof['observed'],{})


@pytest.mark.parametrize('target_registered',[False,True])
def test_transaction_uses_production_windows_reconciliation_and_acknowledges_before_outcome(tx,target_registered):
    import types
    from copy import deepcopy
    t,p=tx;request,original,proof=native_handoff_case()
    p.state.update(sid='actor',session=1,package=original['package'],process=original['process'])
    seed(t,p);p.request=request
    target=deepcopy(proof['observed']['package'])
    proof.update(waiterAlive=False,deploymentIntentPresent=True,directory='owned-native-attempt',fileSha256={'intent.json':'1'*64})
    if not target_registered:
        proof.pop('deployment');proof['failure']={'status':'FAILURE','operationCompleted':True,'hresult':'80080219','manifestSha256':request['target']['manifest_sha256']}
    def interrupted(action,args):
        assert action=='update'
        p.state.update(package=target if target_registered else original['package'],process=None)
        raise d.Interrupted('external transport interrupted after updater handoff')
    p.perform=interrupted
    with pytest.raises(d.Interrupted):t.operation('update',target)
    proof['observed']=p.observe()
    calls=[]
    def transport(action,**kwargs):
        calls.append((action,kwargs))
        if action=='native-reconcile':return deepcopy(proof)
        assert action=='native-reconciled'
        assert kwargs['expected_before']==p.observe()
        assert kwargs['native_proof_hashes']==proof['fileSha256']
        return {'status':kwargs['native_status']}
    p.call=transport;p.reconcile_native=types.MethodType(d.Windows.reconcile_native,p)
    assert t.reconcile()==('performed_partial' if target_registered else 'not_performed')
    assert [row[0] for row in calls]==['native-reconcile','native-reconciled']
    assert t.pending() is None
    assert t.reconcile()=='settled' and len(calls)==2


def test_native_sdk_success_cannot_hide_a_contradictory_failure_receipt():
    request,before,proof=native_handoff_case()
    proof['failure']={'status':'UNKNOWN'}
    with pytest.raises(ValueError):d.verify_native_handoff(proof,request,before)
    proof.update(waiterAlive=False,deploymentIntentPresent=True)
    proof['observed']['process']=None
    with pytest.raises(ValueError):d.verify_native_reconciliation(proof,request,before,proof['observed'],proof['observed']['package'])


def native_side_spec(tmp_path,build,version):
    import base64,json,hashlib
    prefix='releases/channel-builds/'+build+'/'
    def row(name,kind,arch,**extra):
        return dict(key=prefix+name,kind=kind,identity='Semantier.Agent.Acceptance',publisher='CN=Semantier Technology, O=Semantier Technology',version=version,arch=arch,
            size=1,sha256='a'*64,nativeDigest=base64.b64encode(b'd'*32).decode(),references=[],**extra)
    main=row('main.msix','main-package','x64');asset=row('asset.msix','asset-package','neutral',resourceId='runtime0001',allowExecution=True)
    root=row('root.msixbundle','main-bundle','neutral');root['references']=[main['key'],asset['key']]
    graph=dict(schema=1,transport='windows-package-uri',root=root['key'],payloadInventorySha256='e'*64,objects=[root,main,asset])
    directory=tmp_path/build;directory.mkdir();path=directory/'native-graph.json';path.write_text(json.dumps(graph))
    return dict(path=str(directory/'root.msixbundle'),sha256='a'*64,build_id=build,version=version,native_graph=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())),graph


def test_native_baseline_and_target_graphs_load_independently_without_local_origin(tmp_path):
    baseline,a=native_side_spec(tmp_path,'a'*32,'0.0.61.0');target,b=native_side_spec(tmp_path,'b'*32,'0.0.62.0')
    request=dict(baseline=baseline,target=target,selected_delivery_base='https://s3.semantier.com')
    assert d.load_native_side(request,'baseline')==a
    assert d.load_native_target(request)==b
    assert request['baseline']['native_graph']['sha256']!=request['target']['native_graph']['sha256']


def test_native_baseline_loader_refuses_changed_digest_and_wd1_still_requires_loopback(tmp_path):
    baseline,_=native_side_spec(tmp_path,'a'*32,'0.0.61.0')
    baseline['native_graph']['sha256']='f'*64
    with pytest.raises(ValueError,match='digest'):d.load_native_side({'baseline':baseline},'baseline')
    with pytest.raises(ValueError,match='origin'):d.load_native_target({'target':baseline,'purpose':d.NATIVE_QUALIFICATION,'native_origin':'https://s3.semantier.com'})


@pytest.mark.parametrize('status,completed,expected',[
    ('SUCCESS',True,'performed'),('RECONCILED_OS_SUCCESS',True,'performed'),
    ('FAILURE',True,'not_performed'),('FAILURE',False,None),('UNKNOWN',True,None),
])
def test_native_registration_requires_completed_sdk_proof(status,completed,expected):
    platform=object.__new__(d.Windows)
    platform.request={'baseline':{'native_graph':{'sha256':'a'*64}}}
    now={'sid':'test','session_sid':'test','session':2,'root_id':'root','package':None,'process':None,'related':[]}
    platform.call=lambda action,**kwargs:{'status':status,'operationCompleted':completed,'observed':dict(now)}
    intent={'at':'2026-10-29T00:00:00Z','args':{'full':'baseline'}}
    if expected is None:
        with pytest.raises(ValueError,match='UNKNOWN'):platform.reconcile_registration(intent,now)
    else:assert platform.reconcile_registration(intent,now)['status']==expected


def test_native_registration_recovery_refuses_observation_race():
    platform=object.__new__(d.Windows);platform.request={'baseline':{'native_graph':{}}}
    platform.call=lambda action,**kwargs:{'status':'SUCCESS','operationCompleted':True,'observed':{'root_id':'replacement'}}
    with pytest.raises(ValueError,match='observation changed'):
        platform.reconcile_registration({'at':'2026-10-29T00:00:00Z','args':{}},{'root_id':'original'})


@pytest.mark.parametrize('status,registered,expected',[
    ('performed',True,'performed'),('not_performed',False,'not_performed'),
    ('performed',False,None),('not_performed',True,None),('UNKNOWN',False,None),
])
def test_native_registration_journal_does_not_infer_absence(tx,status,registered,expected):
    t,p=tx;seed(t,p);t.fault='after_action' if registered else 'before_action'
    with pytest.raises(d.Interrupted):t.operation('register',{'full':'owned'})
    p.reconcile_registration=lambda intent,now:{'status':status,'proof':{'operationCompleted':status!='UNKNOWN'}}
    if expected is None:
        with pytest.raises(ValueError):t.reconcile()
        assert t.pending() is not None
    else:
        assert t.reconcile()==expected
        assert t.pending() is None


def test_native_upgrade_without_semantic_fixtures_refuses_before_update(tmp_path):
    from types import MethodType
    p=UpgradePlatform();p.request={'root':str(tmp_path),'target':{'native_graph':{'sha256':'a'*64}}}
    p.capture_semantics=MethodType(d.Windows.capture_semantics,p)
    t=d.Transaction(tmp_path/'j',p);seed(t,p)
    t.record('baseline_ready',facts={'healthy':True},observed=p.observe())
    with pytest.raises(ValueError,match='semantic'):d.run_upgrade(t,p)
    assert 'update' not in p.calls
    assert not any(row['kind']=='acceptance' for row in t.events())


def test_wd1_purpose_cannot_authorize_production_native_update():
    with pytest.raises(ValueError,match='does not authorize'):
        d.validate_native_update_inputs({'purpose':d.NATIVE_QUALIFICATION})
