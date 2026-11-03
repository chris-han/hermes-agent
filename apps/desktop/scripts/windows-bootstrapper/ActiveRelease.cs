using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Reflection;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;
using System.Text.Json;
using System.Text.RegularExpressions;

// Immutable, head-bound metadata only. Downloaded package bytes and signatures
// must still be verified independently before Windows deployment.
internal sealed class ActiveRelease {
    internal sealed record Object(string Key, long Size, string Sha256);
    internal Version Version { get; private init; }
    internal string BuildId { get; private init; }
    internal string Root { get; private init; }
    internal IReadOnlyList<Object> Objects { get; private init; }
    const string Identity = "Semantier.Agent.Acceptance";
    const string Publisher = "CN=Semantier Technology, O=Semantier Technology";
    static readonly Regex Id = new("^[a-f0-9]{32}$"), Hash = new("^[a-f0-9]{64}$"),
        File = new("^[A-Za-z0-9_.-]+\\.(msix|msixbundle)$"), Ver = new("^(0|[1-9][0-9]*)(\\.(0|[1-9][0-9]*)){3}$");
    static void Require(bool value) { if (!value) throw new InvalidOperationException("Invalid active release metadata."); }
    static void Unique(JsonElement node) {
        if (node.ValueKind == JsonValueKind.Object) {
            var keys = new HashSet<string>(StringComparer.Ordinal);
            foreach (var p in node.EnumerateObject()) { Require(keys.Add(p.Name)); Unique(p.Value); }
        } else if (node.ValueKind == JsonValueKind.Array) foreach (var child in node.EnumerateArray()) Unique(child);
    }
    static string S(JsonElement obj, string key) { var v=obj.GetProperty(key); Require(v.ValueKind==JsonValueKind.String); return v.GetString(); }
    static long N(JsonElement obj, string key) { var v=obj.GetProperty(key); Require(v.ValueKind==JsonValueKind.Number && v.TryGetInt64(out _)); return v.GetInt64(); }
    static JsonDocument Parse(byte[] bytes) { Require(bytes != null && bytes.Length > 0 && bytes.Length <= 4*1024*1024); var d=JsonDocument.Parse(bytes); Unique(d.RootElement); return d; }
    static JsonElement Head(JsonElement r) {
        Require(N(r,"schema")==1 && S(r,"name")=="semantier-acceptance" && S(r,"state")=="active" && N(r,"revision")>=0);
        var h=r.GetProperty("nativeHead"); Require(h.ValueKind==JsonValueKind.Object);
        string id=S(h,"buildId"); Require(Id.IsMatch(id) && N(h,"sequence")>0 && S(h,"manifestKey")=="releases/channel-builds/"+id+"/build.json" && Hash.IsMatch(S(h,"sha256")));
        return h;
    }
    static bool Same(JsonElement x, JsonElement y) {
        if(x.ValueKind!=y.ValueKind) return false;
        switch(x.ValueKind) {
            case JsonValueKind.Object:
                var a=x.EnumerateObject().ToDictionary(p=>p.Name,p=>p.Value,StringComparer.Ordinal);
                var b=y.EnumerateObject().ToDictionary(p=>p.Name,p=>p.Value,StringComparer.Ordinal);
                return a.Count==b.Count && a.All(p=>b.TryGetValue(p.Key,out var value) && Same(p.Value,value));
            case JsonValueKind.Array:
                return x.GetArrayLength()==y.GetArrayLength() && x.EnumerateArray().Zip(y.EnumerateArray(),Same).All(v=>v);
            case JsonValueKind.String: return x.GetString()==y.GetString();
            case JsonValueKind.Number: return x.TryGetInt64(out var xn) && y.TryGetInt64(out var yn) && xn==yn;
            case JsonValueKind.True: case JsonValueKind.False: case JsonValueKind.Null: return true;
            default: return false;
        }
    }
    static RSA PublisherKey() {
        using var resource=Assembly.GetExecutingAssembly().GetManifestResourceStream("SemantierAgent-signing.cer");
        Require(resource!=null);
        using var bytes=new MemoryStream(); resource.CopyTo(bytes);
        byte[] raw=bytes.ToArray();
        Require(Convert.ToHexString(SHA256.HashData(raw))=="10A6B55E9A03BBC0DCEB8D899C1776BF1FBAD963C82050C16534DA4983C9661D");
        using var cert=new X509Certificate2(raw); var key=cert.GetRSAPublicKey(); Require(key!=null); return key;
    }
    static void VerifyProof(JsonElement root, JsonElement request, JsonElement package, string origin,
                            string id, string version, long sequence) {
        var proof=root.GetProperty("installationProof");
        Require(N(proof,"schema")==1 && S(proof,"algorithm")=="rsa-pkcs1-sha256");
        byte[] payload=Convert.FromBase64String(S(proof,"payload"));
        byte[] signature=Convert.FromBase64String(S(proof,"signature"));
        Require(payload.Length>0 && payload.Length<=4*1024*1024 && signature.Length>0 && signature.Length<=4096);
        using(var key=PublisherKey()) Require(key.VerifyData(payload,signature,HashAlgorithmName.SHA256,RSASignaturePadding.Pkcs1));
        using var data=Parse(payload); var signed=data.RootElement;
        Require(N(signed,"schema")==1 && S(signed,"buildId")==id && S(signed,"channel")=="semantier-acceptance" &&
            N(signed,"sequence")==sequence && S(signed,"origin")==origin && S(signed,"version")==version &&
            S(signed,"identity")==Identity && S(signed,"publisher")==Publisher &&
            S(request,"publicBase")==origin && S(package,"publisher")==Publisher &&
            Same(signed.GetProperty("installation"),root.GetProperty("installation")));
    }
    internal static ActiveRelease Fetch(Func<Uri,byte[]> read, string origin) {
        Require(origin=="https://download.semantier.cn" || origin=="https://download.semantier.com");
        var recordUri=new Uri(origin+"/releases/channels/semantier-acceptance.json");
        byte[] before=read(recordUri); string key;
        using (var record=Parse(before)) key=S(Head(record.RootElement),"manifestKey");
        byte[] manifest=read(new Uri(origin+"/"+key));
        return Resolve(before,manifest,read(recordUri),origin);
    }
    internal static ActiveRelease Resolve(byte[] before, byte[] manifest, byte[] after, string origin) {
        try {
            Require(origin=="https://download.semantier.cn" || origin=="https://download.semantier.com");
            using var b=Parse(before); using var a=Parse(after); using var m=Parse(manifest);
            var h=Head(b.RootElement); var h2=Head(a.RootElement);
            Require(N(b.RootElement,"revision")==N(a.RootElement,"revision") && S(h,"buildId")==S(h2,"buildId") && N(h,"sequence")==N(h2,"sequence") && S(h,"manifestKey")==S(h2,"manifestKey") && S(h,"sha256")==S(h2,"sha256"));
            Require(Convert.ToHexString(SHA256.HashData(manifest)).ToLowerInvariant()==S(h,"sha256"));
            var root=m.RootElement; Require(N(root,"schema")==2); var request=root.GetProperty("request"); string id=S(h,"buildId"), version=S(request,"windowsVersion");
            Require(S(request,"buildId")==id && S(request,"channel")=="semantier-acceptance" && N(request,"sequence")==N(h,"sequence") && S(request.GetProperty("identity"),"msixAppIdWithOrg")==Identity && Ver.IsMatch(version));
            var parts=version.Split('.'); Require(parts.All(x=>ushort.TryParse(x,out _))); var v=new Version(version);
            var packages=root.GetProperty("packages"); Require(packages.ValueKind==JsonValueKind.Array && packages.GetArrayLength()==1); var package=packages[0];
            Require(S(package,"platform")=="win32" && S(package,"arch")=="x64" && S(package,"variant")=="native-msix-assets" && S(package,"identity")==Identity && S(package,"publisher")==Publisher && S(package,"version")==version);
            VerifyProof(root,request,package,origin,id,version,N(h,"sequence"));
            var install=root.GetProperty("installation"); Require(N(install,"schema")==1); var rows=install.GetProperty("objects"); Require(rows.ValueKind==JsonValueKind.Array && rows.GetArrayLength()>0 && rows.GetArrayLength()<=64);
            string prefix="releases/channel-builds/"+id+"/", target=S(install,"root"); var objects=new List<Object>(); var keys=new HashSet<string>(StringComparer.Ordinal);
            foreach (var row in rows.EnumerateArray()) { var key=S(row,"key"); var size=N(row,"size"); var hash=S(row,"sha256"); Require(key.StartsWith(prefix,StringComparison.Ordinal) && File.IsMatch(key.Substring(prefix.Length)) && size>0 && size<=480_000_000 && Hash.IsMatch(hash) && keys.Add(key)); objects.Add(new Object(key,size,hash)); }
            Require(keys.Contains(target) && target.EndsWith(".msixbundle",StringComparison.Ordinal));
            var artifact=package.GetProperty("artifact"); var rootObject=objects.Single(x=>x.Key==target);
            Require(S(artifact,"key")==target && N(artifact,"size")==rootObject.Size && S(artifact,"sha256")==rootObject.Sha256);
            return new ActiveRelease {Version=v,BuildId=id,Root=target,Objects=objects};
        } catch (InvalidOperationException) { throw; } catch (Exception e) when (e is JsonException || e is KeyNotFoundException || e is FormatException || e is ArgumentException || e is OverflowException) { throw new InvalidOperationException("Invalid active release metadata.",e); }
    }
}
