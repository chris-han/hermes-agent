using System;
using System.Security.Cryptography;
using System.Text.Json;

// This standalone negative probe has no publisher certificate resource. The
// channel-binary fixture test covers signed active changes and rollback.
static class Probe {
    static void Main() {
        string id=new string('a',32);
        var manifest=JsonSerializer.SerializeToUtf8Bytes(new {schema=2,request=new {buildId=id,channel="semantier-acceptance",sequence=1,windowsVersion="0.0.80.0",publicBase="https://download.semantier.cn",identity=new {msixAppIdWithOrg="Semantier.Agent.Acceptance"}},packages=new[]{new {platform="win32",arch="x64",variant="native-msix-assets",identity="Semantier.Agent.Acceptance",publisher="CN=Semantier Technology, O=Semantier Technology",version="0.0.80.0"}}});
        var record=JsonSerializer.SerializeToUtf8Bytes(new {schema=1,name="semantier-acceptance",state="active",revision=1,nativeHead=new {buildId=id,sequence=1,manifestKey=$"releases/channel-builds/{id}/build.json",sha256=Convert.ToHexString(SHA256.HashData(manifest)).ToLowerInvariant()}});
        try {ActiveRelease.Resolve(record,manifest,record,"https://download.semantier.cn");}
        catch(InvalidOperationException) {Console.WriteLine("unsigned manifest refused");return;}
        throw new Exception("unsigned manifest accepted");
    }
}
