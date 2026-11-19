// Deterministic native contract tests. Response fixtures are not live OSS evidence.
class OssStagingContractTests {
  static string Digest(byte[] data){using(var h=System.Security.Cryptography.SHA256.Create())return BitConverter.ToString(h.ComputeHash(data)).Replace("-","").ToLowerInvariant();}
  static StagingDownloadGrant Grant(string token){var data=System.Text.Encoding.ASCII.GetBytes("abcdef");string sha=Digest(data);string key="staging/windows-msix/"+new string('a',32)+"/SemantierAgent-"+sha+".msix";return new StagingDownloadGrant{Uri=new Uri("https://assets.semantier.cn/"+key+"?authorization="+token),Key=key,Version="0.0.125.0",Sha256=sha,Size=6};}
  internal static string Run(){
    var first=Grant("old");int renewals=0;var requests=new Requests();
    using(var output=new MemoryStream()){
      BootstrapperWinHttp.DownloadAuthorized(requests,first,output,x=>{},()=>{renewals++;return Grant("new");});
      if(System.Text.Encoding.ASCII.GetString(output.ToArray())!="abcdef" || renewals!=1 || requests.Calls!=4)throw new Exception("Resume/renew/fallback contract failed");
    }
    bool changed=false;
    try{using(var output=new MemoryStream())BootstrapperWinHttp.DownloadAuthorized(new Expired(),Grant("old"),output,x=>{},()=>{var g=Grant("new");g.Version="0.0.126.0";return g;});}catch(InvalidDataException){changed=true;}
    if(!changed)throw new Exception("Renewal switched version");
    bool integrity=false;
    try{using(var output=new MemoryStream())BootstrapperWinHttp.DownloadAuthorized(new Corrupt(),Grant("old"),output,x=>{},()=>Grant("new"));}catch(InvalidDataException){integrity=true;}
    if(!integrity)throw new Exception("Checksum corruption accepted");
    bool forbidden=false;
    try{var bad=Grant("old");bad.Uri=new Uri("https://example.com/evil");bad.Validate();}catch(InvalidDataException){forbidden=true;}
    if(!forbidden)throw new Exception("Foreign host accepted");
    bool unauthorized=false;
    try{using(var output=new MemoryStream())BootstrapperWinHttp.DownloadAuthorized(new Expired(),Grant("old"),output,x=>{},()=>{throw new UnauthorizedAccessException();});}catch(UnauthorizedAccessException){unauthorized=true;}
    if(!unauthorized)throw new Exception("Authorization refusal ignored");
    bool legacy=false;
    try{using(var output=new MemoryStream())BootstrapperWinHttp.Download(new LegacyRange(),new Uri("https://download.semantier.cn/releases/a.msix"),output,6,x=>{});}catch(InvalidDataException){legacy=true;}
    if(!legacy)throw new Exception("200 response to resume accepted");
    return "PASS: full bytes, interruption/resume, H3 failure fallback, 403 renewal, version pin, SHA256, foreign origin, authorization refusal, legacy 200 rejection";
  }
  class Response:IWinHttpResponse {
    public int StatusCode{get;set;}public long ContentLength{get;set;}public string ContentRange{get;set;}public int Protocol{get;set;}public Stream Body{get;set;}
    public void Dispose(){Body.Dispose();}
  }
  static Response Reply(int code,string data,long length,string range){return new Response{StatusCode=code,ContentLength=length,ContentRange=range,Protocol=1,Body=new MemoryStream(System.Text.Encoding.ASCII.GetBytes(data))};}
  class Requests:IWinHttpRequests {
    public int Calls;
    public IWinHttpResponse Open(Uri uri,long offset,bool h3){
      Calls++;
      if(Calls==1){if(!h3)throw new Exception("No H3 attempt");throw new IOException("H3 unavailable");}
      if(Calls==2){if(h3 || offset!=0)throw new Exception("No fallback");return Reply(200,"abc",6,null);}
      if(Calls==3){if(offset!=3)throw new Exception("No resume");return Reply(403,"",0,null);}
      if(offset!=3 || !uri.Query.Contains("new"))throw new Exception("No exact renewal");
      return Reply(206,"def",3,"bytes 3-5/6");
    }
  }
  class Expired:IWinHttpRequests{public IWinHttpResponse Open(Uri u,long o,bool h){return Reply(403,"",0,null);}}
  class Corrupt:IWinHttpRequests{public IWinHttpResponse Open(Uri u,long o,bool h){return Reply(200,"xxxxxx",6,null);}}
  class LegacyRange:IWinHttpRequests{int n;public IWinHttpResponse Open(Uri u,long o,bool h){return ++n==1?Reply(200,"abc",6,null):Reply(200,"def",3,null);}}
}
