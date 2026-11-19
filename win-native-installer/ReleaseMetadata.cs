// Channel control stays on ECS; only an exact digest-bound OSS object supplies bytes.
internal sealed class ReleaseSelection {
  internal const string MetadataBase="https://cloud.semantier.cn";
  internal const string PackageBase="https://assets.semantier.cn";
  internal const string ChannelKey="releases/channels/semantier-acceptance.json";
  const string Identity="Semantier.Agent.Acceptance", Publisher="CN=Semantier Technology, O=Semantier Technology";
  internal string BuildId,Commit,Version,Sha256,NativeDigest,ManifestSha256;
  internal long Size;
  internal System.Uri Uri;
  static void Require(bool value,string message){if(!value)throw new System.IO.InvalidDataException(message);}
  static System.Xml.Linq.XElement Field(System.Xml.Linq.XElement parent,string name,string type){
    Require(parent!=null && (string)parent.Attribute("type")=="object","Expected metadata object.");
    var rows=System.Linq.Enumerable.ToArray(parent.Elements(name));
    Require(rows.Length==1 && (string)rows[0].Attribute("type")==type,"Invalid metadata field: "+name);
    return rows[0];
  }
  static string Text(System.Xml.Linq.XElement parent,string name){return Field(parent,name,"string").Value;}
  static long Number(System.Xml.Linq.XElement parent,string name){
    long value;Require(long.TryParse(Field(parent,name,"number").Value,System.Globalization.NumberStyles.None,System.Globalization.CultureInfo.InvariantCulture,out value),"Invalid integer: "+name);return value;
  }
  static void Match(string value,string pattern,string label){Require(System.Text.RegularExpressions.Regex.IsMatch(value??"","\\A(?:"+pattern+")\\z",System.Text.RegularExpressions.RegexOptions.CultureInvariant),"Invalid "+label);}
  static System.Xml.Linq.XElement Json(byte[] body){
    Require(body!=null && body.Length>0 && body.Length<=4194304,"Release metadata exceeds limit.");
    var quotas=new System.Xml.XmlDictionaryReaderQuotas();quotas.MaxDepth=64;quotas.MaxStringContentLength=4194304;quotas.MaxArrayLength=4194304;
    using(var reader=System.Runtime.Serialization.Json.JsonReaderWriterFactory.CreateJsonReader(body,quotas))return System.Xml.Linq.XElement.Load(reader);
  }
  static string Hash(byte[] body){using(var hash=System.Security.Cryptography.SHA256.Create())return System.BitConverter.ToString(hash.ComputeHash(body)).Replace("-","").ToLowerInvariant();}
  static System.Xml.Linq.XElement[] Array(System.Xml.Linq.XElement parent,string name){return System.Linq.Enumerable.ToArray(Field(parent,name,"array").Elements());}
  internal void VerifyStamp(byte[] body){
    var stamp=Json(body);var request=Field(stamp,"channelBuild","object");
    Require(Field(stamp,"dirty","boolean").Value=="false" && Text(stamp,"commit")==Commit && Text(request,"commit")==Commit && Text(request,"buildId")==BuildId && Text(request,"publicBase")==MetadataBase && Text(request,"windowsVersion")==Version,"Signed package source stamp mismatch.");
  }
  internal static ReleaseSelection Resolve(System.Func<string,byte[]> read){
    var channel=Json(read(MetadataBase+"/"+ChannelKey));
    Require(Number(channel,"schema")==1 && Text(channel,"name")=="semantier-acceptance" && Text(channel,"repository")=="chris-han/hermes-agent" && Text(channel,"state")=="active" && Text(channel,"policy")=="stable-release","Unexpected release authority.");
    var channelIdentity=Field(channel,"identity","object");Require(Text(channelIdentity,"msixAppIdWithOrg")==Identity,"Unexpected channel identity.");
    var head=Field(channel,"nativeHead","object");
    string build=Text(head,"buildId"), manifestHash=Text(head,"sha256");
    Match(build,"[a-f0-9]{32}","build ID");Match(manifestHash,"[a-f0-9]{64}","manifest hash");
    long sequence=Number(head,"sequence");Require(sequence>=131 && sequence<=65535 && Number(channel,"nextSequence")>sequence,"Invalid release sequence.");
    string prefix="releases/channel-builds/"+build+"/",version="0.0."+sequence.ToString(System.Globalization.CultureInfo.InvariantCulture)+".0";
    Require(Text(head,"manifestKey")==prefix+"build.json","Manifest outside selected build.");
    byte[] manifestBody=read(MetadataBase+"/"+prefix+"build.json");Require(Hash(manifestBody)==manifestHash,"Manifest SHA256 mismatch.");
    var manifest=Json(manifestBody);Require(Number(manifest,"schema")==2,"Expected native release manifest.");
    var request=Field(manifest,"request","object");string commit=Text(request,"commit");Match(commit,"[a-f0-9]{40}","source commit");
    Require(Number(request,"schema")==1 && Text(request,"publicBase")==MetadataBase && Text(request,"repository")=="chris-han/hermes-agent" && Text(request,"channel")=="semantier-acceptance" && Text(request,"buildId")==build && Number(request,"sequence")==sequence && Text(request,"windowsVersion")==version && Text(Field(request,"identity","object"),"msixAppIdWithOrg")==Identity,"Release request binding mismatch.");
    var packages=Array(manifest,"packages");Require(packages.Length==1,"Expected one signed package.");var package=packages[0];
    Require(Text(package,"platform")=="win32" && Text(package,"arch")=="x64" && Text(package,"variant")=="bundled" && Text(package,"identity")==Identity && Text(package,"publisher")==Publisher && Text(package,"version")==version,"Package identity/version mismatch.");
    var artifact=Field(package,"artifact","object");string key=Text(artifact,"key"),sha=Text(artifact,"sha256");long size=Number(artifact,"size");
    Match(sha,"[a-f0-9]{64}","package SHA256");Require(key==prefix+"SemantierAgent.msix" && size>0 && size<=12000000000,"Invalid immutable package.");
    var native=Field(package,"native","object");
    Require(Number(native,"schema")==2 && Text(native,"packageBase")==PackageBase && Text(native,"api")=="PackageManager.AddPackageByUriAsync" && Text(native,"digestMethod")=="IAppxDigestProvider.GetDigest" && Text(native,"digestEncoding")=="base64","Untrusted package origin/digest contract.");
    var descriptor=Field(native,"graph","object");var feed=Field(package,"feed","object");
    Require(Text(descriptor,"key")==prefix+"native-graph.json" && Text(feed,"key")==Text(descriptor,"key") && Text(feed,"channel")=="stable","Graph outside selected build.");
    byte[] graphBody=read(MetadataBase+"/"+prefix+"native-graph.json");
    Require(graphBody.LongLength==Number(descriptor,"size") && Hash(graphBody)==Text(descriptor,"sha256"),"Graph size/SHA256 mismatch.");
    var graph=Json(graphBody);Match(Text(graph,"payloadInventorySha256"),"[a-f0-9]{64}","payload inventory");
    Require(Number(graph,"schema")==1 && Text(graph,"transport")=="windows-package-uri" && Text(graph,"root")==key,"Native graph binding mismatch.");
    var objects=Array(graph,"objects");Require(objects.Length==1,"Unexpected native graph objects.");var row=objects[0];
    Require(Text(row,"key")==key && Number(row,"size")==size && Text(row,"sha256")==sha && Text(row,"identity")==Identity && Text(row,"publisher")==Publisher && Text(row,"version")==version && Text(row,"arch")=="x64" && Text(row,"kind")=="main-package" && Array(row,"references").Length==0,"Native object binding mismatch.");
    string digest=Text(row,"nativeDigest");byte[] decoded=System.Convert.FromBase64String(digest);
    Require(decoded.Length==32 && System.Convert.ToBase64String(decoded)==digest,"Invalid native package digest.");
    return new ReleaseSelection{BuildId=build,Commit=commit,Version=version,Sha256=sha,Size=size,NativeDigest=digest,ManifestSha256=manifestHash,Uri=new System.Uri(PackageBase+"/"+key)};
  }
}
