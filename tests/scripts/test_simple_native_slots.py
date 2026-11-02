import pytest
from scripts.releases.simple_native_slots import advance


def head(build, sequence):
    return {'buildId':build*32,'sequence':sequence,'manifestKey':'releases/channel-builds/'+build*32+'/build.json','sha256':build*64}


def test_previous_active_becomes_archive_without_changing_other_fields():
    old=head('a',1); new=head('b',2)
    before={'nativeHead':old,'revision':3,'nextSequence':3,'identity':{'x':1}}
    after=advance(before,new)
    assert after['nativeHead']==new and after['nativeArchive']==old
    assert after['identity']==before['identity'] and after['revision']==4
    assert before['nativeHead']==old


def test_stale_or_unallocated_sequence_refused():
    with pytest.raises(ValueError): advance({'nativeHead':head('a',2),'revision':1,'nextSequence':3},head('b',2))
    with pytest.raises(ValueError): advance({'nativeHead':head('a',1),'revision':1,'nextSequence':2},head('b',3))


def test_rollback_swaps_existing_slots_and_keeps_allocation_high_water():
    from scripts.releases.simple_native_slots import rollback
    record={'nativeHead':head('b',2),'nativeArchive':head('a',1),'revision':3,'nextSequence':3}
    swapped=rollback(record)
    assert swapped['nativeHead']==head('a',1) and swapped['nativeArchive']==head('b',2)
    assert swapped['revision']==4 and swapped['nextSequence']==3 and record['nativeHead']==head('b',2)
    promoted=advance(swapped,head('c',3))
    assert promoted['nativeHead']==head('c',3) and promoted['nativeArchive']==head('a',1)
    assert promoted['nativeRetirePending']==head('b',2)
    with pytest.raises(ValueError): advance(swapped,head('d',2))


@pytest.mark.parametrize('record', [
    {'nativeHead':head('a',1),'revision':1,'nextSequence':2},
    {'nativeHead':head('b',2),'nativeArchive':head('a',1),'nativeRetirePending':head('c',3),'revision':1,'nextSequence':4},
    {'nativeHead':head('a',1),'nativeArchive':head('a',1),'revision':1,'nextSequence':2},
])
def test_rollback_requires_two_distinct_settled_slots(record):
    from scripts.releases.simple_native_slots import rollback
    with pytest.raises(ValueError): rollback(record)


def test_rollback_cas_recovers_lost_reply_without_swapping_twice():
    import json
    from scripts.releases.simple_native_slots import commit_rollback
    initial={'nativeHead':head('b',2),'nativeArchive':head('a',1),'revision':3,'nextSequence':3}
    class Store:
        def __init__(self): self.record=initial; self.etag='old'; self.writes=0
        def get(self, key): return json.dumps(self.record).encode(), self.etag
        def put(self, key, body, etag):
            assert etag==self.etag
            self.record=json.loads(body); self.etag='new'; self.writes+=1
            raise OSError('lost response')
    store=Store()
    with pytest.raises(OSError, match='lost response'):
        commit_rollback(store,'channel.json',initial,'old')
    result=commit_rollback(store,'channel.json',initial,'old')
    assert result['nativeHead']==head('a',1) and result['nativeArchive']==head('b',2)
    assert store.writes==1


def test_rollback_cas_refuses_concurrent_record_change():
    import json
    from scripts.releases.simple_native_slots import commit_rollback
    initial={'nativeHead':head('b',2),'nativeArchive':head('a',1),'revision':3,'nextSequence':3}
    changed={**initial,'revision':4}
    class Store:
        def get(self,key):return json.dumps(changed).encode(),'other'
        def put(self,*args):pytest.fail('concurrent record must not be overwritten')
    with pytest.raises(ValueError, match='channel changed'):
        commit_rollback(Store(),'channel.json',initial,'old')


def test_rollback_cas_requires_exact_readback():
    import json
    from scripts.releases.simple_native_slots import commit_rollback
    initial={'nativeHead':head('b',2),'nativeArchive':head('a',1),'revision':3,'nextSequence':3}
    class Store:
        def __init__(self):self.first=True
        def get(self,key):
            if self.first:return json.dumps(initial).encode(),'old'
            return json.dumps({**initial,'revision':5}).encode(),'changed'
        def put(self,key,body,etag):self.first=False
    with pytest.raises(RuntimeError, match='readback'):
        commit_rollback(Store(),'channel.json',initial,'old')


def test_same_head_is_idempotent():
    record={'nativeHead':head('a',1),'nativeArchive':None,'revision':3,'nextSequence':2}
    assert advance(record,head('a',1))==record


def test_promotion_refuses_changed_channel_before_head_write():
    from scripts.releases.simple_native_slots import commit_slot
    class Store:
        def get(self, key):
            return b'{"nativeHead":null,"revision":2,"nextSequence":2}', 'new-etag'
        def put(self, *args):
            pytest.fail('stale preview must not mutate head')
    with pytest.raises(ValueError, match='changed'):
        commit_slot(Store(), 'releases/channels/test.json', 'old-etag', head('a',1))


def test_promotion_commits_active_and_archive_in_one_conditional_write():
    import json
    from scripts.releases.simple_native_slots import commit_slot
    record={'nativeHead':head('a',1),'revision':2,'nextSequence':3}
    class Store:
        def get(self, key): return json.dumps(record).encode(), 'etag'
        def put(self, key, body, etag):
            assert etag == 'etag'
            written=json.loads(body)
            assert written['nativeHead']==head('b',2)
            assert written['nativeArchive']==record['nativeHead']
    commit_slot(Store(), 'releases/channels/test.json', 'etag', head('b',2))


def test_lock_does_not_release_a_replaced_owner():
    import json
    from scripts.releases.simple_native_slots import publication_lock
    class Store:
        def __init__(self): self.value=None; self.deleted=False
        def put(self, key, body, etag=None):
            assert self.value is None and etag is None
            self.value=(body, 'ours')
        def get(self, key): return self.value
        def delete(self, key, etag): self.deleted=True
    store=Store()
    with publication_lock(store):
        store.value=(json.dumps({'owner':'someone-else'}).encode(),'theirs')
    assert not store.deleted


def test_next_sequence_allocates_atomically_with_head():
    result=advance({'nativeHead':head('a',1),'revision':2,'nextSequence':2},head('b',2))
    assert result['nextSequence']==3 and result['nativeHead']==head('b',2)


def test_displaced_archive_is_durably_recorded_for_cleanup():
    record={'nativeHead':head('b',2),'nativeArchive':head('a',1),'revision':3,'nextSequence':4}
    result=advance(record,head('c',3))
    assert result['nativeHead']==head('c',3) and result['nativeArchive']==head('b',2)
    assert result['nativeRetirePending']==head('a',1)
    with pytest.raises(ValueError, match='pending'):
        advance(result,head('d',4))


def test_native_manifest_rejects_changed_signed_graph():
    import json
    from test_semantier_windows_channel import native_graph
    from scripts.releases.simple_native_slots import native_manifest
    graph=native_graph()
    request={'buildId':'a'*32,'commit':'d'*40,'windowsVersion':'0.0.59.0',
             'identity':{'msixAppIdWithOrg':'Semantier.Agent.Acceptance'}}
    receipt={'signing':'PASS','verification':'PASS: fixture','sourceCommit':'d'*40,
             'packageName':'Semantier.Agent.Acceptance','packageVersion':'0.0.59.0',
             'publisher':'CN=Semantier Technology, O=Semantier Technology',
             'nativeGraph':graph,'signedSha256':'b'*64}
    manifest,_=native_manifest(request,receipt,json.dumps(graph).encode())
    assert manifest['packages'][0]['native']['graph']['key'].endswith('/native-graph.json')
    receipt['signedSha256']='e'*64
    with pytest.raises(ValueError,match='root differs'):
        native_manifest(request,receipt,json.dumps(graph).encode())


def test_failed_upload_never_changes_channel_head():
    from scripts.releases.simple_native_slots import publish_prepared
    class Driver:
        def validate(self, *args): pass
        def ensure_capacity(self, *args): pass
        def upload(self, *args): raise OSError('upload failed')
        def read_channel(self, name): return ({'nativeHead':head('a',1),'nextSequence':2,'revision':1}, 'tag')
        def commit(self, *args): pytest.fail('head changed after failed upload')
    manifest={'request':{'buildId':'b'*32,'sequence':2,'channel':'test'},'packages':[]}
    with pytest.raises(OSError): publish_prepared(Driver(), manifest, {'object':'path'})


def test_head_changes_only_after_every_immutable_object_is_verified():
    from scripts.releases.simple_native_slots import publish_prepared
    calls=[]
    class Driver:
        def read_channel(self, name): return ({'nativeHead':head('a',1),'nextSequence':2,'revision':1}, 'tag')
        def validate(self, *args): calls.append('validate')
        def ensure_capacity(self, *args): calls.append('capacity')
        def upload(self, key, path): calls.append('upload:'+key)
        def upload_metadata(self, key, body): calls.append('manifest')
        def commit(self, channel, etag, value): calls.append('commit'); return value
    manifest={'request':{'buildId':'b'*32,'sequence':2,'channel':'test'},'packages':[]}
    result=publish_prepared(Driver(), manifest, {'b':'b','a':'a'})
    assert calls == ['validate','capacity','upload:a','upload:b','manifest','commit']
    assert result['buildId']=='b'*32


def test_cleanup_keeps_shared_archive_and_clears_only_pending_marker():
    from scripts.releases.simple_native_slots import cleanup_pending
    record={'nativeHead':head('c',3),'nativeArchive':head('b',2),'nativeRetirePending':head('a',1),'revision':4,'nextSequence':4}
    class Driver:
        def read_channel(self, channel): return record, 'tag'
        def can_retire(self, channel, build): return False
        def delete_build(self, build): pytest.fail('shared build must not be deleted')
        def finish_cleanup(self, channel, etag, build):
            assert etag=='tag' and build=='a'*32
            return 'done'
    assert cleanup_pending(Driver(),'test')=='done'


def test_cleanup_never_clears_marker_after_delete_failure():
    from scripts.releases.simple_native_slots import cleanup_pending
    record={'nativeRetirePending':head('a',1)}
    class Driver:
        def read_channel(self, channel): return record, 'tag'
        def can_retire(self, channel, build): return True
        def delete_build(self, build): raise OSError('uncertain delete')
        def finish_cleanup(self, *args): pytest.fail('uncertain cleanup cannot be acknowledged')
    with pytest.raises(OSError): cleanup_pending(Driver(),'test')


def test_capacity_refuses_crossing_fixed_storage_budget():
    from scripts.releases.simple_native_slots import check_capacity
    with pytest.raises(ValueError, match='capacity'):
        check_capacity({'existing':9_999_000_000}, {'new':{'size':2_000_000}}, 0)
    assert check_capacity({'existing':10}, {'existing':{'size':10}, 'new':{'size':20}}, 0)==30


def test_capacity_check_refuses_unresolved_multipart_upload(monkeypatch):
    from scripts.releases import r2
    from scripts.releases.r2_scope import R2Scope
    from scripts.releases.simple_native_slots import R2Driver
    class Response:
        _body=b'<ListMultipartUploadsResult><IsTruncated>false</IsTruncated><Upload><Key>x</Key></Upload></ListMultipartUploadsResult>'
    monkeypatch.setattr(r2,'signed_request',lambda *a,**kw:Response())
    class Store:
        scope=R2Scope();base='http://127.0.0.1:1';bucket='fixture';creds={}
    with pytest.raises(RuntimeError,match='multipart'):
        R2Driver(Store(),'http://127.0.0.1:1').ensure_no_uploads()


def test_prepare_verifies_signed_graph_and_rejects_changed_artifact(tmp_path):
    import hashlib
    import json
    from test_semantier_windows_channel import native_graph
    from scripts.releases.simple_native_slots import prepare
    request={'buildId':'a'*32,'commit':'d'*40,'windowsVersion':'0.0.59.0',
             'identity':{'msixAppIdWithOrg':'Semantier.Agent.Acceptance'}}
    graph=native_graph(); dist=tmp_path/'signed/distribution';dist.mkdir(parents=True)
    for row in graph['objects']:
        payload=('synthetic '+row['key']).encode();row['size']=len(payload);row['sha256']=hashlib.sha256(payload).hexdigest()
        (dist/row['key'].rsplit('/',1)[1]).write_bytes(payload)
    signed={'signing':'PASS','verification':'PASS: synthetic','sourceCommit':'d'*40,
            'packageName':'Semantier.Agent.Acceptance','packageVersion':'0.0.59.0',
            'publisher':'CN=Semantier Technology, O=Semantier Technology','nativeGraph':graph,
            'signedSha256':graph['objects'][0]['sha256'],'certificateThumbprint':'A'*40}
    (tmp_path/'channel-request.json').write_text(json.dumps(request));(tmp_path/'result.json').write_text(json.dumps({'result':'PASS','sourceCommit':'d'*40}))
    (tmp_path/'signed/receipt.json').write_text(json.dumps(signed));(dist/'native-graph.json').write_text(json.dumps(graph))
    calls=[];_manifest,_,files=prepare(tmp_path,'A'*40,lambda paths,thumb: calls.append((paths,thumb)))
    assert len(files)==len(graph['objects'])+1 and calls[0][1]=='A'*40
    target=dist/graph['objects'][0]['key'].rsplit('/',1)[1]
    target.write_bytes(b'X'*graph['objects'][0]['size'])
    with pytest.raises(ValueError,match='hash mismatch'):
        prepare(tmp_path,'A'*40,lambda *_: pytest.fail('unverified content reached signer'))


def test_inflight_release_blocks_new_upload_without_database(monkeypatch):
    import json
    from scripts.releases.simple_native_slots import R2Driver
    class Control:
        def get(self, key): return json.dumps({'inflight':['a'*32]}).encode(), 'etag'
    class Store: pass
    with pytest.raises(RuntimeError,match='in.flight'):
        R2Driver(Store(),'https://example.test',Control()).ensure_no_uploads()


def test_release_notes_follow_updater_contract():
    from scripts.releases.simple_native_slots import validate_notes
    assert validate_notes({'en':'Hello\nworld','zh-cn':'更新'})['en']=='Hello\nworld'
    for value in ({'EN':'bad'}, {'en':''}, {'en':'bad\x00text'}, {'en':'x'*16001}, {'en':'😀'*9000}, {'en':1}):
        with pytest.raises(ValueError): validate_notes(value)


def test_legacy_head_requires_explicit_retirement():
    from scripts.releases.simple_native_slots import advance
    record={'head':head('a',1),'nativeHead':head('b',2),'revision':1,'nextSequence':3}
    with pytest.raises(ValueError,match='legacy'):
        advance(record,head('c',3))


def test_duplicate_graph_metadata_is_rejected():
    from scripts.releases.simple_native_slots import native_manifest
    with pytest.raises(ValueError,match='Invalid channel JSON'):
        native_manifest({'identity':{'msixAppIdWithOrg':'x'}},{},b'{"schema":1,"schema":1}')
