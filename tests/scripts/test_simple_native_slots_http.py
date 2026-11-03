"""Synthetic loopback R2: validates script transport, never a Windows installation."""
import hashlib
import json
import threading
import pytest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
from xml.sax.saxutils import escape

from scripts.releases import r2
from scripts.releases.channels import R2ChannelStore
from scripts.releases.r2_scope import R2Scope
from scripts.releases.semantier_windows import IDENTITY, acceptance_request
from scripts.releases.simple_native_slots import (ControlStore, R2Driver, cleanup_pending, native_manifest,
                                                   publication_lock, publish_prepared, verify_public)
from test_semantier_windows_channel import native_graph


@contextmanager
def storage():
    objects={}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            parsed=urlsplit(self.path); query=parse_qs(parsed.query,keep_blank_values=True)
            if 'uploads' in query:
                body=b'<ListMultipartUploadsResult><IsTruncated>false</IsTruncated></ListMultipartUploadsResult>'
            elif 'list-type' in query:
                prefix=query.get('prefix',[''])[0]; bucket=parsed.path.strip('/')+'/'
                keys=[k[len(bucket):] for k in objects if k.startswith(bucket+prefix)]
                body=('<ListBucketResult><IsTruncated>false</IsTruncated>'+''.join('<Contents><Key>'+escape(k)+'</Key><Size>'+str(len(objects[bucket+k]))+'</Size></Contents>' for k in sorted(keys))+'</ListBucketResult>').encode()
            else:
                key=parsed.path.lstrip('/')
                if key not in objects: self.send_error(404); return
                body=objects[key]
            self.send_response(200); self.send_header('Content-Length',str(len(body)))
            self.send_header('ETag','"'+hashlib.sha256(body).hexdigest()+'"'); self.end_headers(); self.wfile.write(body)
        def do_HEAD(self):
            key=urlsplit(self.path).path.lstrip('/')
            if key not in objects: self.send_error(404); return
            body=objects[key]; self.send_response(200); self.send_header('Content-Length',str(len(body)))
            self.send_header('ETag','"'+hashlib.sha256(body).hexdigest()+'"'); self.end_headers()
        def do_PUT(self):
            key=urlsplit(self.path).path.lstrip('/'); old=objects.get(key)
            etag='"'+hashlib.sha256(old).hexdigest()+'"' if old is not None else None
            if (self.headers.get('If-None-Match')=='*' and old is not None) or (self.headers.get('If-Match') and self.headers['If-Match']!=etag):
                self.send_error(412); return
            objects[key]=self.rfile.read(int(self.headers.get('Content-Length','0')))
            self.send_response(200); self.end_headers()
        def do_DELETE(self):
            key=urlsplit(self.path).path.lstrip('/'); old=objects.get(key)
            if old is None: self.send_error(404); return
            etag='"'+hashlib.sha256(old).hexdigest()+'"'
            if self.headers.get('If-Match')!=etag: self.send_error(412); return
            del objects[key]; self.send_response(204); self.end_headers()
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try: yield 'http://127.0.0.1:'+str(server.server_port),objects
    finally: server.shutdown();server.server_close();thread.join()


@pytest.mark.parametrize('failure',['none','reuse-setup','stale-head','object-conflict','head-drift'])
def test_native_and_manual_publication_resume_after_lost_catalog_response(tmp_path,monkeypatch,failure):
    from scripts.releases.simple_native_slots import publish_with_manual
    monkeypatch.delenv('R2_DISPOSABLE_RUN',raising=False)
    with storage() as (base,objects):
        creds={'access_key_id':'fixture','secret_key':'fixture'};scope=R2Scope()
        store=R2ChannelStore(creds,base,'public',scope=scope);control=ControlStore(r2,creds,base,'private',scope=scope)
        origin='https://download.semantier.com'
        request=acceptance_request(commit='d'*40,controller='e'*40,version='0.0.59',source_version='0.0.0',base=origin)
        graph=native_graph();build=request['buildId'];graph['root']=graph['root'].replace('a'*32,build);files={}
        for row in graph['objects']:
            row['key']=row['key'].replace('a'*32,build);row['references']=[r.replace('a'*32,build) for r in row['references']]
            content=('fixture '+row['key']).encode();row['size']=len(content);row['sha256']=hashlib.sha256(content).hexdigest()
            path=tmp_path/row['key'].rsplit('/',1)[1];path.write_bytes(content);files[row['key']]={'path':path,'size':len(content),'sha256':row['sha256']}
        signed={'signing':'PASS','verification':'PASS: synthetic','sourceCommit':request['commit'],'packageName':IDENTITY['msixAppIdWithOrg'],
                'packageVersion':request['windowsVersion'],'publisher':'CN=Semantier Technology, O=Semantier Technology',
                'nativeGraph':graph,'signedSha256':graph['objects'][0]['sha256']}
        graph_bytes=json.dumps(graph).encode();manifest,_=native_manifest(request,signed,graph_bytes)
        graph_key=manifest['packages'][0]['native']['graph']['key'];path=tmp_path/'graph.json';path.write_bytes(graph_bytes)
        files[graph_key]={'path':path,'size':len(graph_bytes),'sha256':hashlib.sha256(graph_bytes).hexdigest()}
        prefix='releases/channel-builds/'+build+'/';manual={};descriptors=[]
        for name in ('semantier-setup-ch2.exe','release.zip'):
            path=tmp_path/name;path.write_bytes(('fixture '+name).encode());content=path.read_bytes();key=prefix+name
            manual[key]={'path':path,'size':len(content),'sha256':hashlib.sha256(content).hexdigest()}
            descriptors.append({'url':origin+'/'+key,'size':len(content),'sha256':hashlib.sha256(content).hexdigest()})
        channel={'schema':1,'name':'semantier-acceptance','repository':'chris-han/hermes-agent','policy':'stable-release',
                 'state':'active','revision':1,'nextSequence':request['sequence'],'identity':IDENTITY,'head':None}
        channel_key='public/releases/channels/semantier-acceptance.json';objects[channel_key]=json.dumps(channel).encode()
        if failure=='reuse-setup':
            old='f'*32;old_prefix='releases/channel-builds/'+old+'/'
            new_key=prefix+'semantier-setup-ch2.exe';old_key=old_prefix+'semantier-setup-ch2.exe'
            manual[old_key]=manual.pop(new_key);descriptors[0]['url']=origin+'/'+old_key
            objects['public/'+old_key]=(tmp_path/'semantier-setup-ch2.exe').read_bytes()
            old_zip=b'protected previous archive';old_zip_key=old_prefix+'old.zip';objects['public/'+old_zip_key]=old_zip
            old_row={'version':'0.0.58.0','track':'acceptance','identity':'Semantier.Agent.Acceptance','provenance':'local-signed',
                'publishedAt':'2026-11-02T00:00:00Z','installer':descriptors[0],
                'zip':{'url':origin+'/'+old_zip_key,'size':len(old_zip),'sha256':hashlib.sha256(old_zip).hexdigest()}}
            objects['public/releases/manual/catalog.json']=json.dumps({'schema':1,'releases':[old_row]}).encode()
        if failure=='object-conflict':objects['public/'+prefix+'semantier-setup-ch2.exe']=b'protected conflicting bytes'
        real_signed=r2.signed_request;drifted=[]
        def drift_on_head(method,url,**kwargs):
            try:return real_signed(method,url,**kwargs)
            finally:
                if failure=='head-drift' and method=='HEAD' and '/channel-builds/' in url and not drifted:
                    drifted.append(True);objects[channel_key]=json.dumps({**channel,'revision':2}).encode()
        monkeypatch.setattr(r2,'signed_request',drift_on_head)
        original_put=store.put;lost=[]
        def lose_catalog(key,*args,**kwargs):
            result=original_put(key,*args,**kwargs)
            if key=='releases/manual/catalog.json' and not lost:
                lost.append(True);raise OSError('lost catalog response')
            return result
        monkeypatch.setattr(store,'put',lose_catalog)
        with publication_lock(control) as token:
            driver=R2Driver(store,origin,control,token)
            if failure not in ('none','reuse-setup'):
                stale={'buildId':'a'*32,'sequence':1,'manifestKey':'releases/channel-builds/'+'a'*32+'/build.json','sha256':'b'*64}
                with pytest.raises(ValueError):
                    publish_with_manual(driver,manifest,files,manual,*descriptors,'2026-11-03T00:00:00Z',expected_head=stale if failure=='stale-head' else None)
                assert json.loads(objects[channel_key]).get('nativeHead') is None
                assert 'public/releases/manual/catalog.json' not in objects
                if failure=='object-conflict':assert objects['public/'+prefix+'semantier-setup-ch2.exe']==b'protected conflicting bytes'
                return
            with pytest.raises(OSError,match='lost catalog'):
                publish_with_manual(driver,manifest,files,manual,*descriptors,'2026-11-03T00:00:00Z',expected_head=None)
        committed=json.loads(objects[channel_key]);assert committed['nativeHead']['buildId']==build
        with publication_lock(control) as token:
            driver=R2Driver(store,origin,control,token)
            result=publish_with_manual(driver,manifest,files,manual,*descriptors,'2026-11-03T00:00:00Z',expected_head=None)
        catalog=json.loads(objects['public/releases/manual/catalog.json'])
        assert result['nativeHead']==committed['nativeHead'] and catalog['releases'][0]['nativeBuildId']==build
        assert catalog['releases'][0]['installer']==descriptors[0]
        from scripts.releases import simple_native_slots as script
        from hermes_cli.release_channels import ChannelReader
        monkeypatch.setattr(r2,'credentials',lambda:(creds,base,'public'))
        monkeypatch.setenv('CLOUDFLARE_R2_PUBLIC_URL',origin);monkeypatch.setenv('CLOUDFLARE_R2_CONTROL_BUCKET','private')
        monkeypatch.setenv('CLOUDFLARE_R2_CONTROL_ACCESS_KEY_ID','fixture');monkeypatch.setenv('CLOUDFLARE_R2_CONTROL_SECRET_ACCESS_KEY','fixture')
        monkeypatch.setattr(script,'prepare',lambda *_:(manifest,graph,files))
        monkeypatch.setattr(script,'add_installation_proof',lambda *_:None)
        monkeypatch.setattr(script,'prepare_manual_artifacts',lambda *args,**kwargs:(manual,*descriptors))
        monkeypatch.setattr(script,'verify_public',lambda *args:committed)
        monkeypatch.setattr(ChannelReader,'read_bytes',lambda self,key,*args:objects['public/'+key])
        before=dict(objects)
        script.main(['--output',str(tmp_path),'--thumbprint','A'*40,'--setup-exe',str(tmp_path/'semantier-setup-ch2.exe'),
            '--setup-receipt',str(tmp_path/'fixture-receipt.json'),'--manual-published-at','2026-11-03T00:00:00Z','--expect-empty-head'])
        assert objects==before


@pytest.mark.parametrize('failure', ['none','head','signature','local-digest','remote-digest','immutable-conflict'])
def test_legacy_projection_uses_existing_writer_lock_and_head(tmp_path,monkeypatch,failure):
    monkeypatch.delenv('R2_DISPOSABLE_RUN',raising=False)
    with storage() as (base,objects):
        creds={'access_key_id':'fixture','secret_key':'fixture'};scope=R2Scope()
        store=R2ChannelStore(creds,base,'public',scope=scope);control=ControlStore(r2,creds,base,'private',scope=scope)
        request=acceptance_request(commit='d'*40,controller='e'*40,version='0.0.59',source_version='0.0.0',base=base+'/public')
        graph=native_graph();original='a'*32;build=request['buildId'];graph['root']=graph['root'].replace(original,build)
        for row in graph['objects']:
            row['key']=row['key'].replace(original,build);row['references']=[r.replace(original,build) for r in row['references']]
            payload=(row['key']+' fixture').encode();row['size']=len(payload);row['sha256']=hashlib.sha256(payload).hexdigest()
            (tmp_path/row['key'].rsplit('/',1)[1]).write_bytes(payload);objects['public/'+row['key']]=payload
        signed={'signing':'PASS','verification':'PASS: fixture','sourceCommit':request['commit'],'packageName':IDENTITY['msixAppIdWithOrg'],
                'packageVersion':request['windowsVersion'],'publisher':'CN=Semantier Technology, O=Semantier Technology',
                'nativeGraph':graph,'signedSha256':graph['objects'][0]['sha256']}
        graph_bytes=json.dumps(graph,separators=(',',':')).encode();manifest,_=native_manifest(request,signed,graph_bytes);del manifest['installation']
        raw=json.dumps(manifest,separators=(',',':')).encode();prefix='releases/channel-builds/'+build+'/'
        head={'buildId':build,'sequence':request['sequence'],'manifestKey':prefix+'build.json','sha256':hashlib.sha256(raw).hexdigest()}
        record={'schema':1,'name':'semantier-acceptance','repository':'chris-han/hermes-agent','policy':'stable-release',
                'state':'active','revision':1,'nextSequence':head['sequence']+1,'identity':IDENTITY,'head':None,'nativeHead':head}
        channel_key='public/releases/channels/semantier-acceptance.json';objects[channel_key]=json.dumps(record).encode()
        objects['public/'+head['manifestKey']]=raw;objects['public/'+prefix+'native-graph.json']=graph_bytes
        projection_key='public/'+prefix+'installation.json'
        if failure=='local-digest': (tmp_path/graph['objects'][-1]['key'].rsplit('/',1)[1]).write_bytes(b'bad')
        if failure=='remote-digest': objects['public/'+graph['objects'][-1]['key']]=b'bad'
        if failure=='immutable-conflict':objects[projection_key]=b'protected existing bytes'
        checked=[]
        def sign(_):
            if failure=='head':
                altered={**record,'revision':2};objects[channel_key]=json.dumps(altered).encode()
            return b'fixture-signature'
        def verify(paths,thumb,expected):
            checked.append((paths,thumb,expected))
            if failure=='signature':raise ValueError('signature rejected')
        with publication_lock(control) as token:
            driver=R2Driver(store,'https://download.semantier.com',control,token)
            action=lambda: driver.publish_legacy_projection('semantier-acceptance',head,tmp_path,'A'*40,verify,sign)
            if failure!='none':
                with pytest.raises(ValueError):action()
                if failure=='immutable-conflict':assert objects[projection_key]==b'protected existing bytes'
                else:assert projection_key not in objects
            else:
                key,body=action();assert key==prefix+'installation.json' and objects['public/'+key]==body
                assert objects[channel_key]==json.dumps(record).encode()
        if failure!='local-digest':assert len(checked)==1 and len(checked[0][0])==len(graph['objects'])
        if failure=='none':
            import shutil
            import tempfile
            from pathlib import Path
            from hermes_cli.release_channels import ChannelReader
            from scripts.releases import simple_native_slots as script
            monkeypatch.setattr(r2,'credentials',lambda:(creds,base,'public'))
            monkeypatch.setenv('CLOUDFLARE_R2_PUBLIC_URL','https://download.semantier.com')
            monkeypatch.setenv('CLOUDFLARE_R2_CONTROL_BUCKET','private')
            monkeypatch.setenv('CLOUDFLARE_R2_CONTROL_ACCESS_KEY_ID','fixture');monkeypatch.setenv('CLOUDFLARE_R2_CONTROL_SECRET_ACCESS_KEY','fixture')
            monkeypatch.setattr(script,'verify_windows_signatures',verify)
            monkeypatch.setattr(script,'sign_windows_installation_proof',lambda *_:b'fixture-signature')
            monkeypatch.setattr(script,'verify_public',lambda *args:record)
            monkeypatch.setattr(ChannelReader,'read_bytes',lambda self,key,*args:objects['public/'+key])
            work=Path('/mnt/d/repo/deliverables/work')
            if work.is_dir():
                with tempfile.TemporaryDirectory(prefix='legacy-publisher-test-',dir=work) as directory:
                    for row in graph['objects']:shutil.copy2(tmp_path/row['key'].rsplit('/',1)[1],Path(directory))
                    script.main(['--project-legacy-channel','semantier-acceptance','--legacy-directory',directory,
                        '--thumbprint','A'*40,'--expected-build',build,'--expected-manifest-sha256',head['sha256']])
                assert objects['public/'+prefix+'installation.json']==body


@pytest.mark.parametrize("bootstrap", [False, True])
def test_three_releases_leave_two_and_preserve_other_channel(tmp_path, monkeypatch, bootstrap):
    monkeypatch.delenv('R2_DISPOSABLE_RUN',raising=False)
    with storage() as (base,objects):
        creds={'access_key_id':'fixture','secret_key':'fixture'}; scope=R2Scope()
        store=R2ChannelStore(creds,base,'public',scope=scope)
        control=ControlStore(r2,creds,base,'private',scope=scope)
        channel={'schema':1,'name':'semantier-acceptance','repository':'chris-han/hermes-agent',
                 'policy':'stable-release','state':'active','revision':1,'nextSequence':1,'identity':IDENTITY,'head':None}
        objects['public/releases/channels/semantier-acceptance.json']=json.dumps(channel).encode()
        builds=[]
        for sequence in (1,2,3):
            if sequence==3 and bootstrap:
                key='public/releases/channels/semantier-acceptance.json'
                old=json.loads(objects[key]);old.pop('nativeArchive',None);objects[key]=json.dumps(old).encode()
            request=acceptance_request(commit='d'*40,controller='e'*40,version=f'0.0.{sequence}',source_version='0.0.0',base=base+'/public')
            graph=native_graph(); original='a'*32; graph['root']=graph['root'].replace(original,request['buildId'])
            files={}
            for row in graph['objects']:
                row['key']=row['key'].replace(original,request['buildId']); row['references']=[x.replace(original,request['buildId']) for x in row['references']]
                row['version']=request['windowsVersion']; content=(row['key']+' synthetic').encode(); row['size']=len(content);row['sha256']=hashlib.sha256(content).hexdigest()
                path=tmp_path/row['key'].rsplit('/',1)[-1]; path.write_bytes(content)
                files[row['key']]={'path':path,'size':len(content),'sha256':row['sha256']}
            signed={'signing':'PASS','verification':'PASS: synthetic','sourceCommit':request['commit'],
                    'packageName':IDENTITY['msixAppIdWithOrg'],'packageVersion':request['windowsVersion'],
                    'publisher':'CN=Semantier Technology, O=Semantier Technology','nativeGraph':graph,'signedSha256':graph['objects'][0]['sha256']}
            graph_bytes=json.dumps(graph,sort_keys=True,separators=(',',':')).encode(); manifest,_=native_manifest(request,signed,graph_bytes)
            graph_key=manifest['packages'][0]['native']['graph']['key']; path=tmp_path/'native-graph.json';path.write_bytes(graph_bytes)
            files[graph_key]={'path':path,'size':len(graph_bytes),'sha256':hashlib.sha256(graph_bytes).hexdigest()}
            with publication_lock(control) as token:
                driver=R2Driver(store,base+'/public',control,token)
                cleanup_pending(driver,request['channel']); publish_prepared(driver,manifest,files); cleanup_pending(driver,request['channel']); driver.sweep_old(request['channel'])
            builds.append(request['buildId'])
        final=json.loads(objects['public/releases/channels/semantier-acceptance.json'])
        assert final['nativeHead']['buildId']==builds[2] and final['nativeArchive']['buildId']==builds[1]
        assert not any('/'+builds[0]+'/' in key for key in objects)
        assert all(any('/'+build+'/' in key for key in objects) for build in builds[1:])
        assert 'private/_control/publication.lock' not in objects
        assert verify_public(base+'/public','semantier-acceptance',final['nativeHead'])['nativeHead']==final['nativeHead']


def test_shared_manual_download_prevents_archive_deletion(tmp_path, monkeypatch):
    monkeypatch.delenv('R2_DISPOSABLE_RUN',raising=False)
    with storage() as (base,objects):
        creds={'access_key_id':'fixture','secret_key':'fixture'}; scope=R2Scope()
        store=R2ChannelStore(creds,base,'public',scope=scope);control=ControlStore(r2,creds,base,'private',scope=scope)
        build='a'*32; channel={'schema':1,'name':'semantier-acceptance','repository':'chris-han/hermes-agent',
            'policy':'stable-release','state':'active','revision':1,'nextSequence':4,'identity':IDENTITY,'head':None,
            'nativeHead':{'buildId':'c'*32,'sequence':3,'manifestKey':'releases/channel-builds/'+'c'*32+'/build.json','sha256':'c'*64},
            'nativeArchive':{'buildId':'b'*32,'sequence':2,'manifestKey':'releases/channel-builds/'+'b'*32+'/build.json','sha256':'b'*64},
            'nativeRetirePending':{'buildId':build,'sequence':1,'manifestKey':'releases/channel-builds/'+build+'/build.json','sha256':'a'*64}}
        objects['public/releases/channels/semantier-acceptance.json']=json.dumps(channel).encode()
        objects['public/releases/manual/catalog.json']=json.dumps({'releases':[{'url':'https://example.test/releases/channel-builds/'+build+'/old.zip'}]}).encode()
        key='public/releases/channel-builds/'+build+'/old.zip';objects[key]=b'protected'
        with publication_lock(control) as token:
            driver=R2Driver(store,base+'/public',control,token);cleanup_pending(driver,'semantier-acceptance')
        assert objects[key]==b'protected'
        assert 'nativeRetirePending' not in json.loads(objects['public/releases/channels/semantier-acceptance.json'])


def test_interrupted_delete_is_recovered_from_channel_record(monkeypatch):
    import pytest
    monkeypatch.delenv('R2_DISPOSABLE_RUN',raising=False)
    with storage() as (base,objects):
        creds={'access_key_id':'fixture','secret_key':'fixture'};scope=R2Scope()
        store=R2ChannelStore(creds,base,'public',scope=scope);control=ControlStore(r2,creds,base,'private',scope=scope)
        def h(letter,seq):return {'buildId':letter*32,'sequence':seq,'manifestKey':'releases/channel-builds/'+letter*32+'/build.json','sha256':letter*64}
        channel={'schema':1,'name':'semantier-acceptance','repository':'chris-han/hermes-agent',
                 'policy':'stable-release','state':'active','revision':1,'nextSequence':4,'identity':IDENTITY,'head':None,
                 'nativeHead':h('c',3),'nativeArchive':h('b',2),'nativeRetirePending':h('a',1)}
        channel_key='public/releases/channels/semantier-acceptance.json'
        object_key='public/releases/channel-builds/'+'a'*32+'/old.msix'
        objects[channel_key]=json.dumps(channel).encode();objects[object_key]=b'old'
        real=r2.signed_request
        def fail_delete(method,url,**kwargs):
            if method=='DELETE' and '/channel-builds/' in url: raise OSError('synthetic interruption')
            return real(method,url,**kwargs)
        monkeypatch.setattr(r2,'signed_request',fail_delete)
        with pytest.raises(OSError), publication_lock(control) as token:
            cleanup_pending(R2Driver(store,base+'/public',control,token),'semantier-acceptance')
        assert objects[object_key]==b'old' and json.loads(objects[channel_key])['nativeRetirePending']['buildId']=='a'*32
        monkeypatch.setattr(r2,'signed_request',real)
        with publication_lock(control) as token:
            cleanup_pending(R2Driver(store,base+'/public',control,token),'semantier-acceptance')
        assert object_key not in objects and 'nativeRetirePending' not in json.loads(objects[channel_key])


def test_interrupted_cleanup_leaves_manifest_until_other_objects_are_gone(monkeypatch):
    import pytest
    monkeypatch.delenv('R2_DISPOSABLE_RUN',raising=False)
    with storage() as (base,objects):
        creds={'access_key_id':'fixture','secret_key':'fixture'};scope=R2Scope()
        store=R2ChannelStore(creds,base,'public',scope=scope);control=ControlStore(r2,creds,base,'private',scope=scope)
        build='a'*32;prefix='public/releases/channel-builds/'+build+'/'
        objects[prefix+'build.json']=b'{}';objects[prefix+'asset.msix']=b'asset'
        real=r2.signed_request
        def fail_asset(method,url,**kwargs):
            if method=='DELETE' and url.endswith('/asset.msix'): raise OSError('synthetic')
            return real(method,url,**kwargs)
        monkeypatch.setattr(r2,'signed_request',fail_asset)
        with publication_lock(control) as token:
            with pytest.raises(OSError): R2Driver(store,base+'/public',control,token).delete_build(build)
        assert objects[prefix+'build.json']==b'{}' and objects[prefix+'asset.msix']==b'asset'


def test_command_publishes_with_storage_only(tmp_path, monkeypatch, capsys):
    from scripts.releases import simple_native_slots as script
    monkeypatch.delenv('R2_DISPOSABLE_RUN',raising=False)
    with storage() as (base,objects):
        creds={'access_key_id':'fixture','secret_key':'fixture'}
        request=acceptance_request(commit='d'*40,controller='e'*40,version='0.0.1',source_version='0.0.0',base=base+'/public')
        graph=native_graph();graph['root']=graph['root'].replace('a'*32,request['buildId']);files={}
        for row in graph['objects']:
            row['key']=row['key'].replace('a'*32,request['buildId']);row['references']=[x.replace('a'*32,request['buildId']) for x in row['references']]
            row['version']=request['windowsVersion'];content=('synthetic '+row['key']).encode();row['size']=len(content);row['sha256']=hashlib.sha256(content).hexdigest()
            path=tmp_path/row['key'].rsplit('/',1)[-1];path.write_bytes(content);files[row['key']]={'path':path,'size':len(content),'sha256':row['sha256']}
        signed={'signing':'PASS','verification':'PASS: synthetic','sourceCommit':request['commit'],'packageName':IDENTITY['msixAppIdWithOrg'],
                'packageVersion':request['windowsVersion'],'publisher':'CN=Semantier Technology, O=Semantier Technology',
                'nativeGraph':graph,'signedSha256':graph['objects'][0]['sha256']}
        graph_bytes=json.dumps(graph).encode();manifest,_=native_manifest(request,signed,graph_bytes)
        graph_key=manifest['packages'][0]['native']['graph']['key'];path=tmp_path/'graph.json';path.write_bytes(graph_bytes)
        files[graph_key]={'path':path,'size':len(graph_bytes),'sha256':hashlib.sha256(graph_bytes).hexdigest()}
        channel={'schema':1,'name':'semantier-acceptance','repository':'chris-han/hermes-agent','policy':'stable-release',
                 'state':'active','revision':1,'nextSequence':1,'identity':IDENTITY,'head':None}
        objects['public/releases/channels/semantier-acceptance.json']=json.dumps(channel).encode()
        monkeypatch.setattr(script,'prepare',lambda *_:(manifest,graph,files));monkeypatch.setattr(r2,'credentials',lambda:(creds,base,'public'))
        proof_calls=[]
        monkeypatch.setattr(script,'add_installation_proof',lambda candidate,sign:proof_calls.append(candidate['request']['buildId']))
        monkeypatch.setenv('CLOUDFLARE_R2_PUBLIC_URL',base+'/public');monkeypatch.setenv('CLOUDFLARE_R2_CONTROL_BUCKET','private')
        monkeypatch.setenv('CLOUDFLARE_R2_CONTROL_ACCESS_KEY_ID','fixture');monkeypatch.setenv('CLOUDFLARE_R2_CONTROL_SECRET_ACCESS_KEY','fixture')
        notes=tmp_path/'notes.json';notes.write_text(json.dumps({'en':'Synthetic notes'}))
        args=['--output',str(tmp_path),'--thumbprint','A'*40,'--notes',str(notes)]
        script.main(args)
        assert proof_calls==[request['buildId']]
        result=json.loads(capsys.readouterr().out.splitlines()[-1]);assert result['active']['buildId']==request['buildId']
        published=json.loads(objects['public/'+result['active']['manifestKey']]);assert published['notes']=={'en':'Synthetic notes'}
        prior=dict(objects)
        script.main(args)
        second=json.loads(capsys.readouterr().out.splitlines()[-1])
        assert second==result and objects==prior


def test_retirement_keeps_other_channel_archive(monkeypatch):
    monkeypatch.delenv('R2_DISPOSABLE_RUN',raising=False)
    with storage() as (base,objects):
        creds={'access_key_id':'fixture','secret_key':'fixture'};scope=R2Scope()
        store=R2ChannelStore(creds,base,'public',scope=scope);control=ControlStore(r2,creds,base,'private',scope=scope)
        def h(letter,seq):return {'buildId':letter*32,'sequence':seq,'manifestKey':'releases/channel-builds/'+letter*32+'/build.json','sha256':letter*64}
        common={'schema':1,'repository':'chris-han/hermes-agent','policy':'stable-release','state':'active','revision':1,'nextSequence':4,'identity':IDENTITY,'head':None}
        first={**common,'name':'semantier-acceptance','nativeHead':h('c',3),'nativeArchive':h('b',2),'nativeRetirePending':h('a',1)}
        second={**common,'name':'other','nativeHead':h('c',3),'nativeArchive':h('a',1)}
        objects['public/releases/channels/semantier-acceptance.json']=json.dumps(first).encode();objects['public/releases/channels/other.json']=json.dumps(second).encode()
        key='public/releases/channel-builds/'+'a'*32+'/old.msix';objects[key]=b'other channel archive'
        with publication_lock(control) as token:
            cleanup_pending(R2Driver(store,base+'/public',control,token),'semantier-acceptance')
        assert objects[key]==b'other channel archive'


def test_command_rejects_mismatched_storage_origin_before_mutation(tmp_path,monkeypatch):
    from scripts.releases import simple_native_slots as script
    monkeypatch.delenv('R2_DISPOSABLE_RUN',raising=False)
    with storage() as (base,objects):
        request=acceptance_request(commit='d'*40,controller='e'*40,version='0.0.1',source_version='0.0.0',base=base+'/public')
        monkeypatch.setattr(script,'prepare',lambda *_:({'request':request}, {}, {}))
        monkeypatch.setattr(r2,'credentials',lambda:({'access_key_id':'fixture','secret_key':'fixture'},base,'public'))
        monkeypatch.setenv('CLOUDFLARE_R2_PUBLIC_URL',base+'/another')
        with pytest.raises(ValueError,match='origin'):
            script.main(['--output',str(tmp_path),'--thumbprint','A'*40])
        assert objects=={}


def test_corrupt_local_artifact_is_not_uploaded(tmp_path,monkeypatch):
    monkeypatch.delenv('R2_DISPOSABLE_RUN',raising=False)
    with storage() as (base,objects):
        creds={'access_key_id':'fixture','secret_key':'fixture'};scope=R2Scope()
        store=R2ChannelStore(creds,base,'public',scope=scope);control=ControlStore(r2,creds,base,'private',scope=scope)
        path=tmp_path/'file';path.write_bytes(b'wrong')
        with publication_lock(control) as token:
            driver=R2Driver(store,base+'/public',control,token)
            with pytest.raises(ValueError,match='SHA-256'):
                driver.upload('releases/channel-builds/'+'a'*32+'/file.msix',{'path':path,'size':5,'sha256':'a'*64})
        assert not any('channel-builds' in key for key in objects)


def test_manual_projection_uses_real_fenced_store_and_verified_objects(monkeypatch):
    monkeypatch.delenv('R2_DISPOSABLE_RUN',raising=False)
    with storage() as (base,objects):
        from hermes_cli.release_channels import canonical_json
        creds={'access_key_id':'fixture','secret_key':'fixture'};scope=R2Scope()
        store=R2ChannelStore(creds,base,'public',scope=scope);control=ControlStore(r2,creds,base,'private',scope=scope)
        request=acceptance_request(commit='d'*40,controller='e'*40,version='0.0.80',source_version='0.0.0',base='https://download.semantier.com')
        graph=native_graph();old='a'*32;build=request['buildId'];prefix=f'releases/channel-builds/{build}/'
        graph['root']=graph['root'].replace(old,build)
        for row in graph['objects']:
            row['key']=row['key'].replace(old,build);row['references']=[x.replace(old,build) for x in row['references']];row['version']=request['windowsVersion']
        signed={'signing':'PASS','verification':'PASS: synthetic','sourceCommit':request['commit'],
                'packageName':IDENTITY['msixAppIdWithOrg'],'packageVersion':request['windowsVersion'],
                'publisher':'CN=Semantier Technology, O=Semantier Technology','nativeGraph':graph,'signedSha256':graph['objects'][0]['sha256']}
        manifest,_=native_manifest(request,signed,canonical_json(graph));body=canonical_json(manifest)
        head={'buildId':build,'sequence':request['sequence'],'manifestKey':prefix+'build.json','sha256':hashlib.sha256(body).hexdigest()}
        channel={'schema':1,'name':'semantier-acceptance','repository':'chris-han/hermes-agent','policy':'stable-release','state':'active',
                 'revision':1,'nextSequence':request['sequence']+1,'identity':IDENTITY,'head':None,'nativeHead':head}
        objects['public/releases/channels/semantier-acceptance.json']=canonical_json(channel)
        objects['public/'+head['manifestKey']]=body
        def artifact(name,payload):
            key=prefix+name;objects['public/'+key]=payload
            return {'url':'https://download.semantier.com/'+key,'size':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
        setup=artifact('semantier-setup-ch2.exe',b'signed setup fixture');archive=artifact('release.zip',b'archive fixture')
        with publication_lock(control) as token:
            driver=R2Driver(store,'https://download.semantier.com',control,token)
            projected=driver.reconcile_manual_catalog('semantier-acceptance',head,body,setup,archive,'2026-11-03T00:00:00Z')
            assert json.loads(projected)['releases'][0]['nativeBuildId']==build
            assert objects['public/releases/manual/catalog.json']==projected
            original=projected;objects['public/'+prefix+'release.zip']=b'changed'
            with pytest.raises(ValueError,match='checksum'):
                driver.reconcile_manual_catalog('semantier-acceptance',head,body,setup,archive,'2026-11-03T00:00:00Z')
            assert objects['public/releases/manual/catalog.json']==original

@pytest.mark.parametrize('external_reference', [False, True])
def test_explicit_version_purge_preserves_unrelated_references(monkeypatch, external_reference):
    from scripts.releases.simple_native_slots import purge_explicit_versions
    from hermes_cli.release_channels import canonical_json
    monkeypatch.delenv('R2_DISPOSABLE_RUN', raising=False)
    with storage() as (base, objects):
        creds={'access_key_id':'fixture','secret_key':'fixture'};scope=R2Scope()
        store=R2ChannelStore(creds,base,'public',scope=scope)
        control=ControlStore(r2,creds,base,'private',scope=scope)
        old='a'*32; active='b'*32
        head={'buildId':active,'sequence':79,'manifestKey':'releases/channel-builds/'+active+'/build.json','sha256':'c'*64}
        record={'schema':1,'name':'semantier-acceptance','repository':'chris-han/hermes-agent','policy':'stable-release',
                'state':'active','revision':3,'nextSequence':80,'identity':IDENTITY,'head':None,'nativeHead':head}
        channel_key='public/releases/channels/semantier-acceptance.json'
        objects[channel_key]=canonical_json(record)
        url='https://download.semantier.com/releases/channel-builds/'+old+'/old.exe'
        catalog=canonical_json({'schema':1,'releases':[{'installer':{'url':url},'zip':{'url':url}}]})
        objects['public/releases/manual/catalog.json']=catalog
        for build in (old,active):objects['public/releases/channel-builds/'+build+'/build.json']=b'{}'
        if external_reference:objects['public/releases/channels/other.json']=active.encode()
        with publication_lock(control) as token:
            driver=R2Driver(store,'https://download.semantier.com',control,token)
            if external_reference:
                with pytest.raises(ValueError,match='external'):
                    purge_explicit_versions(driver,'semantier-acceptance',head,hashlib.sha256(catalog).hexdigest(),[old,active])
                assert objects[channel_key]==canonical_json(record)
                assert objects['public/releases/manual/catalog.json']==catalog
            else:
                result=purge_explicit_versions(driver,'semantier-acceptance',head,hashlib.sha256(catalog).hexdigest(),[old,active])
                assert result['nativeHead'] is None and result['nextSequence']==80
                assert all('channel-builds' not in key for key in objects)
                assert json.loads(objects['public/releases/manual/catalog.json'])['releases']==[]

@pytest.mark.parametrize('phase', ['catalog', 'pending'])
def test_explicit_version_purge_resumes_after_lost_write(monkeypatch, phase):
    from scripts.releases.simple_native_slots import purge_explicit_versions
    from hermes_cli.release_channels import canonical_json, decode_json
    monkeypatch.delenv('R2_DISPOSABLE_RUN', raising=False)
    with storage() as (base, objects):
        creds={'access_key_id':'fixture','secret_key':'fixture'};scope=R2Scope()
        store=R2ChannelStore(creds,base,'public',scope=scope)
        control=ControlStore(r2,creds,base,'private',scope=scope)
        old='a'*32;active='b'*32
        head={'buildId':active,'sequence':79,'manifestKey':'releases/channel-builds/'+active+'/build.json','sha256':'c'*64}
        record={'schema':1,'name':'semantier-acceptance','repository':'chris-han/hermes-agent','policy':'stable-release',
                'state':'active','revision':3,'nextSequence':80,'identity':IDENTITY,'head':None,'nativeHead':head}
        channel_key='public/releases/channels/semantier-acceptance.json';objects[channel_key]=canonical_json(record)
        url='https://download.semantier.com/releases/channel-builds/'+old+'/old.exe'
        catalog=canonical_json({'schema':1,'releases':[{'installer':{'url':url},'zip':{'url':url}}]})
        objects['public/releases/manual/catalog.json']=catalog
        for build in (old,active):objects['public/releases/channel-builds/'+build+'/build.json']=b'{}'
        original=store.put;failed=[]
        def lost(key,body,etag=None):
            original(key,body,etag)
            if not failed and ((phase=='catalog' and key.endswith('/catalog.json')) or
                    (phase=='pending' and key.endswith('/semantier-acceptance.json') and decode_json(body).get('nativeRetirePending'))):
                failed.append(True);raise OSError('lost response')
        monkeypatch.setattr(store,'put',lost)
        with publication_lock(control) as token:
            driver=R2Driver(store,'https://download.semantier.com',control,token)
            with pytest.raises(OSError,match='lost response'):
                purge_explicit_versions(driver,'semantier-acceptance',head,hashlib.sha256(catalog).hexdigest(),[old,active])
        with publication_lock(control) as token:
            driver=R2Driver(store,'https://download.semantier.com',control,token)
            result=purge_explicit_versions(driver,'semantier-acceptance',head,hashlib.sha256(catalog).hexdigest(),[old,active])
            assert result['nativeHead'] is None and result['nextSequence']==80
            assert all('channel-builds' not in key for key in objects)
