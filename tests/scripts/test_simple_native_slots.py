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
    install=manifest['installation']
    assert install['schema']==1 and install['root']==graph['root']
    assert install['objects']==[{key: row[key] for key in ('key','size','sha256')} for row in graph['objects']]
    receipt['signedSha256']='e'*64
    with pytest.raises(ValueError,match='root differs'):
        native_manifest(request,receipt,json.dumps(graph).encode())


def test_installation_proof_binds_release_origin_and_exact_closure():
    import base64
    import json
    from scripts.releases.simple_native_slots import add_installation_proof
    manifest={'request':{'buildId':'a'*32,'channel':'semantier-acceptance','sequence':7,
              'publicBase':'https://download.semantier.com','windowsVersion':'0.0.80.0',
              'identity':{'msixAppIdWithOrg':'Semantier.Agent.Acceptance'}},
              'packages':[{'publisher':'CN=Semantier Technology, O=Semantier Technology'}],
              'installation':{'schema':1,'root':'root','objects':[{'key':'root','size':10,'sha256':'b'*64}]}}
    observed=[]
    proof=add_installation_proof(manifest,lambda data: observed.append(data) or b'fixture signature')
    signed=json.loads(observed[0])
    assert signed == {'schema':1,'buildId':'a'*32,'channel':'semantier-acceptance','sequence':7,
                      'origin':'https://download.semantier.com','version':'0.0.80.0',
                      'identity':'Semantier.Agent.Acceptance','publisher':manifest['packages'][0]['publisher'],
                      'installation':manifest['installation']}
    assert base64.b64decode(proof['payload'],validate=True)==observed[0]
    assert base64.b64decode(proof['signature'],validate=True)==b'fixture signature'
    assert manifest['installationProof']==proof
    manifest['request']['publicBase']='https://evil.example'
    with pytest.raises(ValueError,match='origin'):
        add_installation_proof(manifest,lambda _: b'signature')


def test_new_channel2_origin_cannot_publish_without_setup_catalog(monkeypatch,tmp_path):
    from scripts.releases import simple_native_slots as slots
    monkeypatch.setattr(slots,'prepare',lambda *_:({'request':{'publicBase':'https://download.semantier.com'}},{},{}))
    monkeypatch.setattr(slots,'sign_windows_installation_proof',lambda *_:pytest.fail('unqualified signing reached'))
    with pytest.raises(ValueError,match='setup and catalog'):
        slots.main(['--output',str(tmp_path),'--thumbprint','A'*40])


def test_legacy_projection_binds_active_manifest_graph_and_channel():
    import base64
    import hashlib
    import json
    from test_semantier_windows_channel import native_graph
    from scripts.releases.simple_native_slots import legacy_installation_projection
    graph=native_graph();graph_bytes=json.dumps(graph,separators=(',',':')).encode();root=graph['objects'][0]
    request={'buildId':'a'*32,'channel':'semantier-acceptance','sequence':79,'windowsVersion':'0.0.59.0',
        'publicBase':'https://s3.semantier.com','identity':{'msixAppIdWithOrg':'Semantier.Agent.Acceptance'}}
    descriptor={'key':root['key'].rsplit('/',1)[0]+'/native-graph.json','sha256':hashlib.sha256(graph_bytes).hexdigest(),'size':len(graph_bytes)}
    package={'platform':'win32','arch':'x64','variant':'native-msix-assets','publisher':root['publisher'],'identity':root['identity'],'version':root['version'],
        'artifact':{k:root[k] for k in ('key','size','sha256')},'native':{'graph':descriptor}}
    raw=json.dumps({'schema':2,'request':request,'packages':[package]},separators=(',',':')).encode()
    head={'buildId':'a'*32,'sequence':79,'manifestKey':'releases/channel-builds/'+'a'*32+'/build.json','sha256':hashlib.sha256(raw).hexdigest()}
    verified=[]
    result=legacy_installation_projection(raw,graph_bytes,head,'https://download.semantier.com',lambda g:verified.append(g),lambda _:b'signature')
    payload=json.loads(base64.b64decode(result['installationProof']['payload']))
    assert verified==[graph] and payload['manifestSha256']==head['sha256']
    assert payload['origin']=='https://download.semantier.com' and payload['installation']==result['installation']
    for bad_head in ({**head,'sha256':'0'*64},{**head,'sequence':78}):
        with pytest.raises(ValueError): legacy_installation_projection(raw,graph_bytes,bad_head,'https://download.semantier.com',lambda _:pytest.fail('invalid graph reached verifier'),lambda _:b'sig')
    with pytest.raises(ValueError): legacy_installation_projection(raw,graph_bytes+b' ',head,'https://download.semantier.com',lambda _:pytest.fail('invalid graph reached verifier'),lambda _:b'sig')
    with pytest.raises(RuntimeError): legacy_installation_projection(raw,graph_bytes,head,'https://download.semantier.com',lambda _:(_ for _ in ()).throw(RuntimeError('signature rejected')),lambda _:pytest.fail('unsigned objects reached signer'))


def test_windows_installation_signer_with_disposable_certificate(monkeypatch):
    import base64
    from datetime import datetime, timedelta, timezone
    import hashlib
    from pathlib import Path
    import subprocess
    import tempfile
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    from cryptography.hazmat.primitives.serialization import pkcs12
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
    from scripts.releases.simple_native_slots import sign_windows_installation_proof
    powershell=Path('/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe')
    work=Path('/mnt/d/repo/deliverables/work')
    if not powershell.is_file() or not work.is_dir(): pytest.skip('Windows signing fixture unavailable')
    key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
    subject=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,'Disposable signing fixture')])
    now=datetime.now(timezone.utc)
    cert=(x509.CertificateBuilder().subject_name(subject).issuer_name(subject).public_key(key.public_key())
          .serial_number(x509.random_serial_number()).not_valid_before(now-timedelta(minutes=1))
          .not_valid_after(now+timedelta(days=1))
          .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CODE_SIGNING]),critical=False).sign(key,hashes.SHA256()))
    original_run=subprocess.run
    with tempfile.TemporaryDirectory(prefix='proof-signing-0080-',dir=work) as tmp:
        pfx=Path(tmp)/'fixture.pfx'
        pfx.write_bytes(pkcs12.serialize_key_and_certificates(b'fixture',key,cert,None,serialization.BestAvailableEncryption(b'fixture-only')))
        windows_path=subprocess.check_output(['wslpath','-w',str(pfx)],text=True).strip().replace("'","''")
        def run_fixture(command,**kwargs):
            script=base64.b64decode(command[-1]).decode('utf-16le')
            old="$cert=Get-Item -LiteralPath 'Cert:\\CurrentUser\\My\\57BF15739B4AA447842EAF388856A01B4E5029FF'"
            replacement="$cert=[Security.Cryptography.X509Certificates.X509Certificate2]::new('"+windows_path+"','fixture-only',[Security.Cryptography.X509Certificates.X509KeyStorageFlags]::EphemeralKeySet)"
            assert old in script
            script=script.replace(old,replacement).replace('10A6B55E9A03BBC0DCEB8D899C1776BF1FBAD963C82050C16534DA4983C9661D',hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest().upper())
            return original_run([*command[:-1],base64.b64encode(script.encode('utf-16le')).decode()],**kwargs)
        monkeypatch.setattr(subprocess,'run',run_fixture)
        payload=b'disposable installation proof fixture'
        signature=sign_windows_installation_proof(payload,'57BF15739B4AA447842EAF388856A01B4E5029FF')
        key.public_key().verify(signature,payload,padding.PKCS1v15(),hashes.SHA256())


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
    calls=[];_manifest,_,files=prepare(tmp_path,'A'*40,lambda paths,thumb,facts: calls.append((paths,thumb,facts)))
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


def test_derived_views_are_bound_to_record_and_reconcile_after_partial_write():
    from hermes_cli.release_channels import canonical_json, decode_json
    from scripts.releases.simple_native_slots import reconcile_slot_views, validate_slot_view
    record={'name':'stable','nativeHead':head('b',2),'nativeArchive':head('a',1),'revision':3,'nextSequence':3}
    class Store:
        def __init__(self): self.objects={}; self.interrupt=True
        def get(self,key): return self.objects.get(key)
        def put(self,key,body,etag=None):
            old=self.objects.get(key)
            assert (old is None and etag is None) or (old is not None and old[1]==etag)
            if key=='archive/manifest.json' and self.interrupt:
                self.interrupt=False
                raise OSError('interrupted')
            self.objects[key]=(body, str(len(self.objects)+1))
    store=Store()
    with pytest.raises(OSError): reconcile_slot_views(store,'stable',record)
    reconcile_slot_views(store,'stable',record)
    active=decode_json(store.get('active/manifest.json')[0])
    archive=decode_json(store.get('archive/manifest.json')[0])
    assert validate_slot_view(active,'stable',record,'active') == record['nativeHead']
    assert validate_slot_view(archive,'stable',record,'archive') == record['nativeArchive']
    changed={**record,'revision':4,'nativeHead':record['nativeArchive'],'nativeArchive':record['nativeHead']}
    with pytest.raises(ValueError, match='stale'): validate_slot_view(active,'stable',changed,'active')
    reconcile_slot_views(store,'stable',changed)
    assert validate_slot_view(decode_json(store.get('active/manifest.json')[0]),'stable',changed,'active')==changed['nativeHead']


def test_derived_empty_archive_is_explicit_and_wrong_channel_is_rejected():
    from scripts.releases.simple_native_slots import slot_view, validate_slot_view
    record={'nativeHead':head('a',1),'nativeArchive':None,'revision':1,'nextSequence':2}
    view=slot_view('stable',record,'archive')
    assert validate_slot_view(view,'stable',record,'archive') is None
    with pytest.raises(ValueError, match='stale'): validate_slot_view(view,'other',record,'archive')


def test_r2_store_accepts_only_the_two_public_slot_metadata_keys():
    from scripts.releases.channels import R2ChannelStore
    assert R2ChannelStore.metadata_key('active/manifest.json') == 'active/manifest.json'
    assert R2ChannelStore.metadata_key('archive/manifest.json') == 'archive/manifest.json'
    assert R2ChannelStore.metadata_key('releases/channels/stable.json') == 'releases/channels/stable.json'
    with pytest.raises(ValueError): R2ChannelStore.metadata_key('active/other.json')
    with pytest.raises(ValueError): R2ChannelStore.metadata_key('active/../archive/manifest.json')


def test_driver_reconciles_views_only_with_lock_and_current_channel_record():
    from hermes_cli.release_channels import canonical_json, decode_json
    from scripts.releases.simple_native_slots import R2Driver
    record={'nativeHead':head('a',1),'nativeArchive':None,'revision':1,'nextSequence':2}
    class Store:
        def __init__(self): self.values={}; self.writes=0
        def get(self,key): return self.values.get(key)
        def put(self,key,body,etag=None):
            old=self.values.get(key)
            assert (old is None and etag is None) or (old is not None and old[1]==etag)
            self.writes+=1; self.values[key]=(body,str(self.writes))
    store=Store()
    driver=R2Driver(store,'https://download.example')
    driver.verify_lock=lambda: None
    driver.read_channel=lambda channel: (record,'etag')
    driver.reconcile_views('stable')
    assert decode_json(store.values['active/manifest.json'][0])['head']==record['nativeHead']
    with pytest.raises(ValueError, match='different channel'):
        driver.reconcile_views('preview')
    driver.verify_lock=lambda: (_ for _ in ()).throw(RuntimeError('lock lost'))
    with pytest.raises(RuntimeError,match='lock lost'): driver.reconcile_views('stable')


def test_driver_refuses_view_write_after_channel_record_changes():
    from scripts.releases.simple_native_slots import R2Driver
    record={'nativeHead':head('a',1),'nativeArchive':None,'revision':1,'nextSequence':2}
    class Store:
        def get(self,key): return None
        def put(self,*args): pytest.fail('changed channel must not be written')
    driver=R2Driver(Store(),'https://download.example')
    driver.verify_lock=lambda: None
    reads=iter([(record,'old'),({**record,'revision':2},'new')])
    driver.read_channel=lambda channel: next(reads)
    with pytest.raises(ValueError,match='channel changed'): driver.reconcile_views('stable')


def test_manual_projection_binds_active_build_and_preserves_reusable_setup():
    import hashlib,json
    from scripts.releases.simple_native_slots import project_manual_catalog
    build='b'*32; old='a'*32; origin='https://download.semantier.com'
    root=f'releases/channel-builds/{build}/app.msixbundle';native={'key':root,'size':7,'sha256':'d'*64}
    manifest=json.dumps({'schema':2,'request':{'buildId':build,'sequence':2,'channel':'semantier-acceptance',
              'windowsVersion':'0.0.80.0','identity':{'msixAppIdWithOrg':'Semantier.Agent.Acceptance'}},
              'packages':[{'identity':'Semantier.Agent.Acceptance','publisher':'CN=Semantier Technology, O=Semantier Technology','version':'0.0.80.0','artifact':native}],
              'installation':{'schema':1,'root':root,'objects':[native]}}).encode()
    active={'buildId':build,'sequence':2,'manifestKey':f'releases/channel-builds/{build}/build.json','sha256':hashlib.sha256(manifest).hexdigest()}
    def artifact(b,name): return {'url':f'{origin}/releases/channel-builds/{b}/{name}','size':7,'sha256':'c'*64}
    installer=artifact(old,'semantier-setup-ch2.exe'); archive=artifact(build,'SemantierAgent.zip')
    prior={'version':'0.0.79.0','track':'acceptance','identity':'Semantier.Agent.Acceptance','provenance':'local-signed',
           'publishedAt':'2026-11-02T00:00:00Z','installer':installer,'zip':artifact(old,'SemantierAgent.zip')}
    raw=json.dumps({'schema':1,'releases':[prior]}).encode()
    result=project_manual_catalog(raw,active,manifest,origin,installer,archive,'2026-11-03T00:00:00Z')
    data=json.loads(result);assert data['releases'][0]['nativeBuildId']==build
    assert data['releases'][0]['installer']==installer and prior in data['releases']
    with pytest.raises(ValueError,match='head'):
        project_manual_catalog(raw,{**active,'sha256':'f'*64},manifest,origin,installer,archive,'2026-11-03T00:00:00Z')
    with pytest.raises(ValueError,match='setup'):
        project_manual_catalog(raw,active,manifest,origin,artifact('d'*32,'semantier-setup-ch2.exe'),archive,'2026-11-03T00:00:00Z')


def test_manual_projection_is_fenced_by_channel_head_and_lock(monkeypatch):
    from scripts.releases import simple_native_slots as slots
    class Store:
        def __init__(self): self.body=b'old'; self.writes=[]
        def get(self,key): return (self.body,'catalog-etag') if key=='releases/manual/catalog.json' else (b'manifest','manifest-etag')
        def put(self,key,body,etag): self.writes.append((key,body,etag));self.body=body
    store=Store();driver=slots.R2Driver(store,'https://download.semantier.com')
    head={'buildId':'b'*32,'sequence':2,'manifestKey':'releases/channel-builds/'+'b'*32+'/build.json','sha256':'c'*64}
    record={'nativeHead':head,'revision':4}; reads=[];verified=[];locks=[]
    monkeypatch.setattr(driver,'verify_lock',lambda:locks.append(True))
    monkeypatch.setattr(driver,'read_channel',lambda channel:(reads.append(channel) or record,'head-etag'))
    monkeypatch.setattr(driver,'verify_manual_object',lambda row:verified.append(row))
    monkeypatch.setattr(slots,'project_manual_catalog',lambda *args:b'new')
    rows=[{'url':'setup'}, {'url':'zip'}]
    assert driver.reconcile_manual_catalog('semantier-acceptance',head,b'manifest',*rows,'2026-11-03T00:00:00Z')==b'new'
    assert verified==rows and len(reads)>=3 and len(locks)>=2
    assert store.writes==[('releases/manual/catalog.json',b'new','catalog-etag')]
    store.writes.clear();states=iter([(record,'head-etag'),({**record,'revision':5},'changed')])
    monkeypatch.setattr(driver,'read_channel',lambda channel:next(states))
    with pytest.raises(ValueError,match='changed'):
        driver.reconcile_manual_catalog('semantier-acceptance',head,b'manifest',*rows,'2026-11-03T00:00:00Z')
    assert store.writes==[]


@pytest.mark.parametrize('channel,origin,name',[('channel1','https://download.semantier.cn','semantier-setup-ch1.exe'),('channel2','https://download.semantier.com','semantier-setup-ch2.exe')])
def test_manual_artifacts_bind_signed_setup_and_zip_to_native_graph(tmp_path,channel,origin,name):
    import hashlib,json,zipfile
    from pathlib import Path
    from scripts.releases.simple_native_slots import prepare_manual_artifacts
    build='a'*32;prefix=f'releases/channel-builds/{build}/';dist=tmp_path/'signed/distribution';dist.mkdir(parents=True)
    obj=dist/'SemantierAgent.msixbundle';obj.write_bytes(b'signed bundle fixture')
    row={'key':prefix+obj.name,'size':obj.stat().st_size,'sha256':hashlib.sha256(obj.read_bytes()).hexdigest()}
    graph={'root':row['key'],'objects':[row]};graph_bytes=json.dumps(graph).encode();(dist/'native-graph.json').write_bytes(graph_bytes)
    cert=Path('apps/desktop/scripts/windows-bootstrapper/SemantierAgent-signing.cer').read_bytes()
    archive=tmp_path/'signed'/'release.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr(obj.name,obj.read_bytes());z.writestr('native-graph.json',graph_bytes);z.writestr('SemantierAgent-signing.cer',cert)
    archive_sha=hashlib.sha256(archive.read_bytes()).hexdigest();(archive.parent/(archive.name+'.sha256')).write_text(archive_sha+'  '+archive.name+'\n')
    (tmp_path/'signed/receipt.json').write_text(json.dumps({'archiveName':archive.name}))
    exe=tmp_path/name;exe.write_bytes(b'fixture setup')
    receipt={'channel':channel,'sourceCommit':'b'*40,'sourceTree':'c'*40,'sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),
             'size':exe.stat().st_size,'certificateThumbprint':'A'*40,'signing':'PASS'}
    receipt_file=tmp_path/'setup-receipt.json';receipt_file.write_text(json.dumps(receipt));verified=[]
    manifest={'request':{'buildId':build,'publicBase':origin}}
    files,setup,zip_row=prepare_manual_artifacts(tmp_path,manifest,graph,exe,receipt_file,'A'*40,lambda paths,thumb,facts:verified.append((paths,thumb)),verify_source=lambda commit,tree:None)
    assert setup['url'].endswith(prefix+exe.name) and zip_row['url'].endswith(prefix+archive.name)
    assert len(files)==2 and verified==[([exe],'A'*40)]
    wrong={**receipt,'channel':'channel2' if channel=='channel1' else 'channel1'};receipt_file.write_text(json.dumps(wrong))
    with pytest.raises(ValueError,match='setup receipt'):
        prepare_manual_artifacts(tmp_path,manifest,graph,exe,receipt_file,'A'*40,lambda *_:pytest.fail('wrong channel reached signature'),verify_source=lambda *_:None)
    receipt_file.write_text(json.dumps(receipt))
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr(obj.name,b'X'*row['size']);z.writestr('native-graph.json',graph_bytes);z.writestr('SemantierAgent-signing.cer',cert)
    (archive.parent/(archive.name+'.sha256')).write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  '+archive.name+'\n')
    with pytest.raises(ValueError,match='archive package digest mismatch'):
        prepare_manual_artifacts(tmp_path,manifest,graph,exe,receipt_file,'A'*40,lambda *_:None,verify_source=lambda *_:None)
    exe.write_bytes(b'fixture tamper')
    with pytest.raises(ValueError,match='setup'):
        prepare_manual_artifacts(tmp_path,manifest,graph,exe,receipt_file,'A'*40,lambda *_:pytest.fail('signature check reached'),verify_source=lambda *_:None)
