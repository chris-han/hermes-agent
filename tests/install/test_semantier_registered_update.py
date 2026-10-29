import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('driver',Path(__file__).with_name('semantier_registered_update.py'))
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)

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
