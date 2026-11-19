// Opt-in integration harness for the existing downloader. Never installs packages.
public static class OssStagingLive {
    public static object Grant(string url,string key,string version,string sha,long size){return new StagingDownloadGrant{Uri=new Uri(url),Key=key,Version=version,Sha256=sha,Size=size};}
    public sealed class Result {public long bytes;public int requests,http3Failures,renewals,protocol;public long expiredOffset=-1;public bool interrupted,sha256Verified;}
    public static object Download(Func<object> authorize,Func<object> renew,string output,bool interrupt){
        var result=new Result();
        using(var native=new NativeWinHttpRequests(true)){
            var requests=new Measured(native,result,interrupt);
            using(var file=new FileStream(output,FileMode.CreateNew,FileAccess.ReadWrite,FileShare.None)){
                BootstrapperWinHttp.DownloadAuthorized(requests,(StagingDownloadGrant)authorize(),file,x=>{},()=>{result.renewals++;return (StagingDownloadGrant)renew();});
            }
        }
        result.sha256Verified=true;return result;
    }
    public static string Range(object value,string original){
        var grant=(StagingDownloadGrant)value;grant.Validate();
        using(var native=new NativeWinHttpRequests(true)){
            long start=grant.Size/2;
            using(var response=native.Open(grant.Uri,start,false))using(var file=File.OpenRead(original)){
                if(response.StatusCode!=206 || response.ContentLength!=grant.Size-start || response.ContentRange!="bytes "+start+"-"+(grant.Size-1)+"/"+grant.Size)throw new Exception("Range response mismatch");
                file.Position=start;var actual=new byte[65536];int got=0;
                while(got<actual.Length){var chunk=new byte[actual.Length-got];int n=response.Body.Read(chunk,0,chunk.Length);if(n==0)throw new Exception("Range truncated");Buffer.BlockCopy(chunk,0,actual,got,n);got+=n;}
                var expected=new byte[65536];if(file.Read(expected,0,expected.Length)!=expected.Length)throw new Exception("Original truncated");
                for(int i=0;i<actual.Length;i++)if(actual[i]!=expected[i])throw new Exception("Range bytes mismatch");
            }
            using(var response=native.Open(grant.Uri,grant.Size,false))if(response.StatusCode!=416)throw new Exception("Unsatisfiable Range was not 416");
        }
        using(var missing=new NativeWinHttpRequests())using(var response=missing.Open(grant.Uri,1,false))if(response.StatusCode!=403)throw new Exception("Missing signed header accepted");
        return "PASS: native byte-exact 206, unsatisfiable 416, missing bound header 403";
    }
    sealed class Measured:IWinHttpRequests {
        readonly NativeWinHttpRequests native;readonly Result result;readonly bool interrupt;
        public Measured(NativeWinHttpRequests n,Result r,bool i){native=n;result=r;interrupt=i;}
        public IWinHttpResponse Open(Uri uri,long offset,bool h3){
            result.requests++;
            IWinHttpResponse response;
            try{response=native.Open(uri,offset,h3);}catch(IOException){if(h3)result.http3Failures++;throw;}
            result.protocol=response.Protocol;
            if(response.StatusCode==403)result.expiredOffset=offset;
            bool cut=interrupt && !result.interrupted && response.StatusCode==200;
            if(cut)result.interrupted=true;
            return new CountedResponse(response,result,cut);
        }
    }
    sealed class CountedResponse:IWinHttpResponse {
        readonly IWinHttpResponse inner;
        public CountedResponse(IWinHttpResponse i,Result r,bool cut){inner=i;Body=new CountedStream(i.Body,r,cut);}
        public int StatusCode{get{return inner.StatusCode;}}public long ContentLength{get{return inner.ContentLength;}}public string ContentRange{get{return inner.ContentRange;}}public int Protocol{get{return inner.Protocol;}}public Stream Body{get;private set;}
        public void Dispose(){inner.Dispose();}
    }
    sealed class CountedStream:Stream {
        readonly Stream inner;readonly Result result;readonly bool cut;long read;
        public CountedStream(Stream s,Result r,bool c){inner=s;result=r;cut=c;}
        public override int Read(byte[] b,int offset,int count){
            if(cut && read>=8388608){System.Threading.Thread.Sleep(20000);throw new IOException("Qualification interruption");}
            if(cut)count=(int)Math.Min(count,8388608-read);
            int n=inner.Read(b,offset,count);read+=n;result.bytes+=n;return n;
        }
        public override bool CanRead{get{return true;}}public override bool CanWrite{get{return false;}}public override bool CanSeek{get{return false;}}
        public override long Length{get{throw new NotSupportedException();}}public override long Position{get{throw new NotSupportedException();}set{throw new NotSupportedException();}}
        public override void Flush(){}public override long Seek(long a,SeekOrigin b){throw new NotSupportedException();}public override void SetLength(long n){throw new NotSupportedException();}public override void Write(byte[] b,int o,int c){throw new NotSupportedException();}
    }
}
