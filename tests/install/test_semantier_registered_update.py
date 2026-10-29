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
