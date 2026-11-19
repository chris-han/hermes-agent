"""Run the production Framework-compatible Setup resolver on portable JSON fixtures."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_setup_resolves_only_cloud_metadata_and_exact_oss_package(tmp_path):
    pwsh = shutil.which('pwsh')
    if not pwsh:
        pytest.skip('PowerShell compiler required')
    prefix = 'releases/channel-builds/'+'a'*32+'/'
    identity = {'msixAppIdWithOrg': 'Semantier.Agent.Acceptance'}
    row = {'key': prefix+'SemantierAgent.msix', 'size': 1600000000, 'sha256': 'd'*64,
           'identity': identity['msixAppIdWithOrg'], 'publisher': 'CN=Semantier Technology, O=Semantier Technology',
           'version': '0.0.131.0', 'arch': 'x64', 'kind': 'main-package', 'references': [],
           'nativeDigest': base64.b64encode(bytes([1])*32).decode()}
    graph = {'schema': 1, 'transport': 'windows-package-uri', 'root': row['key'],
             'payloadInventorySha256': 'e'*64, 'objects': [row]}
    manifest = {'schema': 2, 'request': {'schema': 1, 'publicBase': 'https://cloud.semantier.cn',
                'repository': 'chris-han/hermes-agent', 'channel': 'semantier-acceptance', 'buildId': 'a'*32,
                'sequence': 131, 'windowsVersion': row['version'], 'commit': 'b'*40, 'identity': identity},
                'packages': [{'platform': 'win32', 'arch': 'x64', 'variant': 'bundled',
                             'identity': row['identity'], 'publisher': row['publisher'], 'version': row['version'],
                             'artifact': {k: row[k] for k in ('key', 'size', 'sha256')},
                             'feed': {'key': prefix+'native-graph.json', 'channel': 'stable'},
                             'native': {'schema': 2, 'packageBase': 'https://assets.semantier.cn',
                                        'api': 'PackageManager.AddPackageByUriAsync',
                                        'digestMethod': 'IAppxDigestProvider.GetDigest', 'digestEncoding': 'base64',
                                        'graph': {'key': prefix+'native-graph.json'}}}]}
    channel = {'schema': 1, 'name': 'semantier-acceptance', 'repository': 'chris-han/hermes-agent',
               'state': 'active', 'policy': 'stable-release', 'identity': identity, 'nextSequence': 132,
               'nativeHead': {'buildId': 'a'*32, 'sequence': 131, 'manifestKey': prefix+'build.json'}}
    mutations = [None,
                 lambda c, m, g: m['packages'][0]['native'].update(packageBase='https://evil.example'),
                 lambda c, m, g: m['request'].update(publicBase='https://download.semantier.cn'),
                 lambda c, m, g: m['request'].update(buildId='f'*32),
                 lambda c, m, g: m['request'].update(commit='not-a-commit'),
                 lambda c, m, g: m['packages'][0]['artifact'].update(key=prefix+'wrong.msix'),
                 lambda c, m, g: g['objects'][0].update(nativeDigest='bad-digest'),
                 lambda c, m, g: g['objects'][0].update(publisher='CN=Other'),
                 lambda c, m, g: c.update(nextSequence=131)]
    for index, mutation in enumerate(mutations):
        c, m, g = copy.deepcopy((channel, manifest, graph))
        if mutation:
            mutation(c, m, g)
        def encode(value):
            return (json.dumps(value, sort_keys=True, separators=(',', ':'))+'\n').encode()
        graph_body = encode(g)
        m['packages'][0]['native']['graph'].update(size=len(graph_body), sha256=hashlib.sha256(graph_body).hexdigest())
        manifest_body = encode(m)
        c['nativeHead']['sha256'] = hashlib.sha256(manifest_body).hexdigest()
        case = tmp_path/str(index);case.mkdir()
        (case/'stamp.json').write_bytes(encode({'dirty': False, 'commit': m['request']['commit'], 'channelBuild': m['request']}))
        for name, body in [('channel', encode(c)), ('manifest', manifest_body), ('graph', graph_body)]:
            (case/(name+'.json')).write_bytes(body)
    staged = tmp_path/'staged';staged.mkdir()
    for name in ('manifest.json','graph.json'):
        shutil.copyfile(tmp_path/'0'/name,staged/name)
    baseline=copy.deepcopy(channel)
    baseline['nativeHead'].update(buildId='f'*32,sequence=130,manifestKey='releases/channel-builds/'+'f'*32+'/build.json',sha256='f'*64)
    baseline['nextSequence']=131
    (staged/'channel.json').write_text(json.dumps(baseline))
    harness = tmp_path/'harness.cs'
    harness.write_text(r'''
public static class ReleaseMetadataTests {
 public static string Run(string directory){
  for(int i=0;i<9;i++){
   string root=System.IO.Path.Combine(directory,i.ToString());
   System.Func<string,byte[]> read=url=>{
    string name;
    if(url==ReleaseSelection.MetadataBase+"/"+ReleaseSelection.ChannelKey)name="channel";
    else if(url==ReleaseSelection.MetadataBase+"/releases/channel-builds/"+new string('a',32)+"/build.json")name="manifest";
    else if(url==ReleaseSelection.MetadataBase+"/releases/channel-builds/"+new string('a',32)+"/native-graph.json")name="graph";
    else throw new System.Exception("Untrusted metadata URI");
    return System.IO.File.ReadAllBytes(System.IO.Path.Combine(root,name+".json"));
   };
   bool refused=false;
   try{
    var result=ReleaseSelection.Resolve(read);
    if(i==0){
     byte[] stamp=System.IO.File.ReadAllBytes(System.IO.Path.Combine(root,"stamp.json"));result.VerifyStamp(stamp);
     string altered=System.Text.Encoding.UTF8.GetString(stamp).Replace(new string('b',40),new string('c',40));
     bool changed=false;try{result.VerifyStamp(System.Text.Encoding.UTF8.GetBytes(altered));}catch(System.IO.InvalidDataException){changed=true;}
     if(!changed)throw new System.Exception("Changed signed source stamp accepted");
    }
    if(i==0 && (result.Uri.AbsoluteUri!="https://assets.semantier.cn/releases/channel-builds/"+new string('a',32)+"/SemantierAgent.msix" || result.Size!=1600000000 || result.Version!="0.0.131.0" || result.Sha256!=new string('d',64)))throw new System.Exception("Wrong selected artifact");
   }catch(System.IO.InvalidDataException){refused=true;}
    catch(System.FormatException){refused=true;}
   if(refused!=(i!=0))throw new System.Exception("Wrong admission for fixture "+i);
  }
  string candidateRoot=System.IO.Path.Combine(directory,"staged");
  System.Func<string,byte[]> nominatedRead=url=>{
   string name=url==ReleaseSelection.MetadataBase+"/"+ReleaseSelection.ChannelKey?"channel":
    url==ReleaseSelection.MetadataBase+"/releases/channel-builds/"+new string('a',32)+"/build.json"?"manifest":
    url==ReleaseSelection.MetadataBase+"/releases/channel-builds/"+new string('a',32)+"/native-graph.json"?"graph":null;
   if(name==null)throw new System.Exception("Untrusted candidate URI");
   return System.IO.File.ReadAllBytes(System.IO.Path.Combine(candidateRoot,name+".json"));
  };
  string pinned;using(var hash=System.Security.Cryptography.SHA256.Create())pinned=System.BitConverter.ToString(hash.ComputeHash(System.IO.File.ReadAllBytes(System.IO.Path.Combine(candidateRoot,"manifest.json")))).Replace("-","").ToLowerInvariant();
  var selected=ReleaseSelection.Resolve(nominatedRead,new string('a',32),pinned,131);
  if(selected.Version!="0.0.131.0" || selected.ManifestSha256!=pinned)throw new System.Exception("Wrong staged selection");
  bool downgrade=false;try{ReleaseSelection.Resolve(nominatedRead,new string('a',32),pinned,130);}catch(System.IO.InvalidDataException){downgrade=true;}
  bool changedPin=false;try{ReleaseSelection.Resolve(nominatedRead,new string('a',32),new string('f',64),131);}catch(System.IO.InvalidDataException){changedPin=true;}
  if(!downgrade || !changedPin)throw new System.Exception("Untrusted staged selection accepted");
  return "PASS:active and pinned staged release;10 negative controls";
 }
}
''')
    script = tmp_path/'run.ps1'
    script.write_text('param($Source,$Harness,$Directory)\n$ErrorActionPreference="Stop"\n'
                      'Add-Type -TypeDefinition ((Get-Content $Source -Raw)+(Get-Content $Harness -Raw))\n'
                      '[ReleaseMetadataTests]::Run($Directory)\n')
    result = subprocess.run([pwsh, '-NoProfile', '-NonInteractive', '-File', str(script),
                             str(ROOT/'win-native-installer/ReleaseMetadata.cs'), str(harness), str(tmp_path)],
                            capture_output=True, text=True, timeout=40, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'PASS:active and pinned staged release;10 negative controls'
