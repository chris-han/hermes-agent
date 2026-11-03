using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;

static class PackageProbe {
    static ActiveRelease Make(JsonElement graph,bool missing,string version) {
        var objects=graph.GetProperty("objects").EnumerateArray().Select(x=>new {key=x.GetProperty("key").GetString(),size=x.GetProperty("size").GetInt64(),sha256=x.GetProperty("sha256").GetString()}).ToArray();
        if(missing) objects=objects.Take(objects.Length-1).ToArray();
        string root=graph.GetProperty("root").GetString(), id=root.Split('/')[2]; var artifact=objects.Single(x=>x.key==root);
        var m=JsonSerializer.SerializeToUtf8Bytes(new {schema=2,request=new {buildId=id,channel="semantier-acceptance",sequence=1,windowsVersion=version,identity=new {msixAppIdWithOrg="Semantier.Agent.Acceptance"}},
            packages=new[]{new {platform="win32",arch="x64",variant="native-msix-assets",identity="Semantier.Agent.Acceptance",publisher="CN=Semantier Technology, O=Semantier Technology",version,artifact}},
            installation=new {schema=1,root,objects}});
        var r=JsonSerializer.SerializeToUtf8Bytes(new {schema=1,name="semantier-acceptance",state="active",revision=1,nativeHead=new {buildId=id,sequence=1,manifestKey=$"releases/channel-builds/{id}/build.json",sha256=Convert.ToHexString(SHA256.HashData(m)).ToLowerInvariant()}});
        return ActiveRelease.Resolve(r,m,r,"https://download.semantier.cn");
    }
    static void Reject(Action action) {try {action();throw new Exception("invalid package set accepted");} catch(InvalidOperationException) {}}
    public static void Main(string[] args) {
        using var doc=JsonDocument.Parse(File.ReadAllBytes(Path.Combine(args[0],"native-graph.json")));
        string version=doc.RootElement.GetProperty("objects")[0].GetProperty("version").GetString();
        PackageIdentity.Verify(Make(doc.RootElement,false,version),args[0]);
        Reject(()=>PackageIdentity.Verify(Make(doc.RootElement,true,version),args[0]));
        Reject(()=>PackageIdentity.Verify(Make(doc.RootElement,false,version=="0.0.0.0" ? "0.0.0.1" : "0.0.0.0"),args[0]));
        Console.WriteLine("PACKAGE_IDENTITY_AND_CLOSURE_PASS");
    }
}
