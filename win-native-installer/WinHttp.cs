interface IWinHttpResponse : IDisposable {
    int StatusCode { get; }
    long ContentLength { get; }
    string ContentRange { get; }
    int Protocol { get; }
    Stream Body { get; }
}
interface IWinHttpRequests { IWinHttpResponse Open(Uri uri,long offset,bool http3); }

static class BootstrapperWinHttp {
    internal static void Download(IWinHttpRequests requests,Uri origin,Stream output,long size,Action<long> progress) {
        if(origin.Scheme!=Uri.UriSchemeHttps || origin.Port!=443 || origin.UserInfo.Length!=0 || origin.Fragment.Length!=0)
            throw new InvalidOperationException("HTTPS port 443 required.");
        if(size<=0 || !output.CanWrite || output.Position!=0)throw new InvalidOperationException("Invalid download target.");
        long total=0;var buffer=new byte[1024*1024];
        for(int attempt=0;attempt<5 && total<size;attempt++) {
            IWinHttpResponse response;
            try { response=requests.Open(origin,total,true); }
            catch(IOException) { try {response=requests.Open(origin,total,false);} catch(IOException) {continue;} }
            using(response) {
                string expected="bytes "+total.ToString(CultureInfo.InvariantCulture)+"-"+(size-1).ToString(CultureInfo.InvariantCulture)+"/"+size.ToString(CultureInfo.InvariantCulture);
                if(response.ContentLength!=size-total || (total==0?response.StatusCode!=200:response.StatusCode!=206 || response.ContentRange!=expected))
                    throw new InvalidDataException("Unexpected download response.");
                while(total<size) {
                    int n;try {n=response.Body.Read(buffer,0,(int)Math.Min(buffer.Length,size-total));}catch(IOException){break;}
                    if(n==0)break;output.Write(buffer,0,n);total+=n;progress(total);
                }
            }
        }
        if(total!=size)throw new IOException("Download incomplete after retries.");
        output.Flush();
    }
}

sealed class NativeWinHttpRequests : IWinHttpRequests, IDisposable {
    readonly IntPtr session;
    public NativeWinHttpRequests() {
        session=NativeWinHttp.Open("SemantierSetup/1",1,null,null,0);
        NativeWinHttp.Check(session!=IntPtr.Zero,"WinHttpOpen");
        try {NativeWinHttp.Check(NativeWinHttp.SetTimeouts(session,15000,15000,15000,30000),"WinHttpSetTimeouts");}
        catch {NativeWinHttp.Close(session);throw;}
    }
    public void Dispose(){NativeWinHttp.Close(session);}
    public IWinHttpResponse Open(Uri uri,long offset,bool http3) { return OpenRequest(uri,offset,http3,"GET"); }
    public IWinHttpResponse OpenHead(Uri uri,bool http3) { return OpenRequest(uri,0,http3,"HEAD"); }
    IWinHttpResponse OpenRequest(Uri uri,long offset,bool http3,string method) {
        IntPtr connection=IntPtr.Zero,request=IntPtr.Zero;
        try {
            connection=NativeWinHttp.Connect(session,uri.IdnHost,443,0);NativeWinHttp.Check(connection!=IntPtr.Zero,"WinHttpConnect");
            request=NativeWinHttp.OpenRequest(connection,method,uri.PathAndQuery,null,null,IntPtr.Zero,0x00800000);
            NativeWinHttp.Check(request!=IntPtr.Zero,"WinHttpOpenRequest");
            int protocols=http3?2:1;NativeWinHttp.Check(NativeWinHttp.SetOption(request,133,ref protocols,4),"WinHttpSetOption protocol");
            int disableRedirect=2;NativeWinHttp.Check(NativeWinHttp.SetOption(request,63,ref disableRedirect,4),"WinHttpSetOption redirect");
            if(http3){int required=1;NativeWinHttp.Check(NativeWinHttp.SetOption(request,145,ref required,4),"WinHttpSetOption required");}
            string headers="Accept-Encoding: identity\r\n"+(offset>0?"Range: bytes="+offset.ToString(CultureInfo.InvariantCulture)+"-\r\n":"");
            NativeWinHttp.Check(NativeWinHttp.SendRequest(request,headers,uint.MaxValue,IntPtr.Zero,0,0,UIntPtr.Zero),"WinHttpSendRequest");
            NativeWinHttp.Check(NativeWinHttp.ReceiveResponse(request,IntPtr.Zero),"WinHttpReceiveResponse");
            int status;long length;
            if(!int.TryParse(NativeWinHttp.Header(request,19,false),NumberStyles.None,CultureInfo.InvariantCulture,out status) ||
               !long.TryParse(NativeWinHttp.Header(request,5,false),NumberStyles.None,CultureInfo.InvariantCulture,out length) || length<0)
                throw new InvalidDataException("Invalid HTTP response headers.");
            string range=NativeWinHttp.Header(request,53,true);
            var response=new NativeWinHttpResponse(connection,request,status,length,range,NativeWinHttp.UsedProtocol(request));connection=IntPtr.Zero;request=IntPtr.Zero;return response;
        } catch(Win32Exception error) {throw new IOException("Windows HTTP transport failed.",error);}
        finally {if(request!=IntPtr.Zero)NativeWinHttp.Close(request);if(connection!=IntPtr.Zero)NativeWinHttp.Close(connection);}
    }
}
sealed class NativeWinHttpResponse : IWinHttpResponse {
    readonly IntPtr connection,request;
    public int StatusCode {get;private set;} public long ContentLength {get;private set;} public string ContentRange {get;private set;} public int Protocol {get;private set;}
    public Stream Body {get;private set;}
    public NativeWinHttpResponse(IntPtr c,IntPtr r,int status,long length,string range,int protocol){Protocol=protocol;connection=c;request=r;StatusCode=status;ContentLength=length;ContentRange=range;Body=new WinHttpReadStream(r);}
    public void Dispose(){Body.Dispose();NativeWinHttp.Close(request);NativeWinHttp.Close(connection);}
}
sealed class WinHttpReadStream : Stream {
    readonly IntPtr request;
    public WinHttpReadStream(IntPtr r){request=r;}
    public override int Read(byte[] buffer,int offset,int count){
        if(offset!=0)throw new ArgumentOutOfRangeException("offset");uint read;
        if(!NativeWinHttp.ReadData(request,buffer,(uint)count,out read))throw new IOException("Windows HTTP read failed.",new Win32Exception(Marshal.GetLastWin32Error()));return (int)read;
    }
    public override bool CanRead{get{return true;}} public override bool CanSeek{get{return false;}} public override bool CanWrite{get{return false;}}
    public override long Length{get{throw new NotSupportedException();}} public override long Position{get{throw new NotSupportedException();}set{throw new NotSupportedException();}}
    public override void Flush(){} public override long Seek(long o,SeekOrigin s){throw new NotSupportedException();} public override void SetLength(long v){throw new NotSupportedException();} public override void Write(byte[] b,int o,int c){throw new NotSupportedException();}
}
static class NativeWinHttp {
    [DllImport("winhttp.dll",EntryPoint="WinHttpOpen",CharSet=CharSet.Unicode,SetLastError=true)] internal static extern IntPtr Open(string a,uint t,string p,string b,uint f);
    [DllImport("winhttp.dll",EntryPoint="WinHttpConnect",CharSet=CharSet.Unicode,SetLastError=true)] internal static extern IntPtr Connect(IntPtr s,string h,ushort p,uint z);
    [DllImport("winhttp.dll",EntryPoint="WinHttpOpenRequest",CharSet=CharSet.Unicode,SetLastError=true)] internal static extern IntPtr OpenRequest(IntPtr c,string v,string o,string ver,string referer,IntPtr accept,uint flags);
    [DllImport("winhttp.dll",EntryPoint="WinHttpSetOption",SetLastError=true)] internal static extern bool SetOption(IntPtr h,uint option,ref int value,uint size);
    [DllImport("winhttp.dll",EntryPoint="WinHttpSetTimeouts",SetLastError=true)] internal static extern bool SetTimeouts(IntPtr h,int resolve,int connect,int send,int receive);
    [DllImport("winhttp.dll",EntryPoint="WinHttpSendRequest",CharSet=CharSet.Unicode,SetLastError=true)] internal static extern bool SendRequest(IntPtr h,string headers,uint length,IntPtr optional,uint optionalLength,uint total,UIntPtr context);
    [DllImport("winhttp.dll",EntryPoint="WinHttpReceiveResponse",SetLastError=true)] internal static extern bool ReceiveResponse(IntPtr h,IntPtr reserved);
    [DllImport("winhttp.dll",EntryPoint="WinHttpQueryHeaders",CharSet=CharSet.Unicode,SetLastError=true)] static extern bool QueryHeaders(IntPtr h,uint level,string name,IntPtr buffer,ref uint length,IntPtr index);
    [DllImport("winhttp.dll",EntryPoint="WinHttpReadData",SetLastError=true)] internal static extern bool ReadData(IntPtr h,byte[] buffer,uint size,out uint read);
    [DllImport("winhttp.dll",EntryPoint="WinHttpCloseHandle",SetLastError=true)] internal static extern bool Close(IntPtr h);
    [DllImport("winhttp.dll",EntryPoint="WinHttpQueryOption",SetLastError=true)] static extern bool QueryOption(IntPtr h,uint option,out int value,ref uint length);
    internal static int UsedProtocol(IntPtr request){int value;uint length=4;Check(QueryOption(request,134,out value,ref length),"WinHttpQueryOption protocol");return value;}
    internal static void Check(bool ok,string action){if(!ok)throw new Win32Exception(Marshal.GetLastWin32Error(),action);}
    internal static string Header(IntPtr h,uint level,bool optional){
        uint length=0;QueryHeaders(h,level,null,IntPtr.Zero,ref length,IntPtr.Zero);int error=Marshal.GetLastWin32Error();
        if(optional && error==12150)return null;if(error!=122 || length>65536)throw new Win32Exception(error,"WinHttpQueryHeaders");
        IntPtr p=Marshal.AllocHGlobal((int)length);try{Check(QueryHeaders(h,level,null,p,ref length,IntPtr.Zero),"WinHttpQueryHeaders");return Marshal.PtrToStringUni(p);}finally{Marshal.FreeHGlobal(p);}
    }
}