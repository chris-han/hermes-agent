using System;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

static class Probe {
    static string Hex(byte[] b) => Convert.ToHexString(b).ToLowerInvariant();
    static byte[] Manifest(string id, string version) => JsonSerializer.SerializeToUtf8Bytes(new {
        schema=2, request=new {buildId=id, channel="semantier-acceptance", sequence=1, windowsVersion=version,
            identity=new {msixAppIdWithOrg="Semantier.Agent.Acceptance"}},
        packages=new[]{new {platform="win32",arch="x64",variant="native-msix-assets",identity="Semantier.Agent.Acceptance",publisher="CN=Semantier Technology, O=Semantier Technology",version=version,artifact=new {key=$"releases/channel-builds/{id}/SemantierAgent.msixbundle",size=10,sha256=new string('b',64)}}},
        installation=new {schema=1,root=$"releases/channel-builds/{id}/SemantierAgent.msixbundle",objects=new[]{new {key=$"releases/channel-builds/{id}/SemantierAgent.msixbundle",size=10,sha256=new string('b',64)}}}
    });
    static byte[] Record(string id,byte[] m,int revision=1) => JsonSerializer.SerializeToUtf8Bytes(new {schema=1,name="semantier-acceptance",state="active",revision,nativeHead=new {buildId=id,sequence=1,manifestKey=$"releases/channel-builds/{id}/build.json",sha256=Hex(SHA256.HashData(m))}});
    static void Reject(Action action) {try {action();throw new Exception("accepted invalid metadata");} catch(InvalidOperationException) {}}
    public static void Main() {
        var id=new string('a',32);var m=Manifest(id,"0.0.79.0");var r=Record(id,m);
        if(ActiveRelease.Resolve(r,m,r,"https://download.semantier.cn").Version.ToString()!="0.0.79.0")throw new Exception("initial head");
        int calls=0;
        var selected=ActiveRelease.Fetch(uri=>{ calls++; if(uri.Host!="download.semantier.cn")throw new Exception("foreign request"); return calls==2 ? m : r; },"https://download.semantier.cn");
        if(calls!=3 || selected.BuildId!=id)throw new Exception("snapshot request sequence");
        var id2=new string('b',32);var m2=Manifest(id2,"0.0.80.0");var r2=Record(id2,m2,2);
        if(ActiveRelease.Resolve(r2,m2,r2,"https://download.semantier.cn").Version.ToString()!="0.0.80.0")throw new Exception("new head");
        if(ActiveRelease.Resolve(r,m,r,"https://download.semantier.cn").Version.ToString()!="0.0.79.0")throw new Exception("rollback");
        Reject(()=>ActiveRelease.Resolve(r,m,r2,"https://download.semantier.cn"));
        Reject(()=>ActiveRelease.Resolve(r,m2,r,"https://download.semantier.cn"));
        Reject(()=>ActiveRelease.Resolve(r,m,r,"https://evil.example"));
        var foreign=JsonNode.Parse(m); foreign!["installation"]!["objects"]![0]!["key"]=$"releases/channel-builds/{id2}/SemantierAgent.msixbundle";
        var foreignBytes=Encoding.UTF8.GetBytes(foreign.ToJsonString()); var foreignRecord=Record(id,foreignBytes);
        Reject(()=>ActiveRelease.Resolve(foreignRecord,foreignBytes,foreignRecord,"https://download.semantier.cn"));
        var wrong=JsonNode.Parse(m); wrong!["packages"]![0]!["publisher"]="CN=Other";
        var wrongBytes=Encoding.UTF8.GetBytes(wrong.ToJsonString()); var wrongRecord=Record(id,wrongBytes);
        Reject(()=>ActiveRelease.Resolve(wrongRecord,wrongBytes,wrongRecord,"https://download.semantier.cn"));
        Reject(()=>ActiveRelease.Resolve(Encoding.UTF8.GetBytes("{\"schema\":1,\"schema\":1}"),m,r,"https://download.semantier.cn"));
        Console.WriteLine("active resolver fixture PASS");
    }
}
