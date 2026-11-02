using System;
using System.Runtime.InteropServices;
using System.Text;

// Windows exposes unknown DNS records as wire-format RDATA in DNS_RECORD.Data.
// Accept only the configured origin as the service target; never follow aliases.
static class Http3Dns {
    const string Host = "download.semantier.cn";
    [DllImport("dnsapi.dll", CharSet=CharSet.Unicode)]
    static extern int DnsQuery_W(string name, ushort type, uint options, IntPtr servers, out IntPtr result, IntPtr reserved);
    [DllImport("dnsapi.dll")] static extern void DnsRecordListFree(IntPtr records, int freeType);
    static int U16(byte[] b, int i) { return (b[i] << 8) | b[i+1]; }
    internal static int Parse(byte[] b) {
        if (b.Length < 3 || U16(b,0) == 0) throw new InvalidOperationException("Invalid HTTPS service record.");
        int i=2; var host=new StringBuilder();
        while (true) {
            if(i>=b.Length) throw new InvalidOperationException("Truncated HTTPS target.");
            int n=b[i++]; if(n==0) break;
            if(n>63 || i+n>b.Length || host.Length+n+1>253) throw new InvalidOperationException("Invalid HTTPS target.");
            if(host.Length>0) host.Append('.');
            for(int j=0;j<n;j++) { byte c=b[i++]; if(c>127) throw new InvalidOperationException("Invalid HTTPS target."); host.Append((char)c); }
        }
        if(host.Length>0 && !string.Equals(host.ToString(),Host,StringComparison.OrdinalIgnoreCase)) throw new InvalidOperationException("Unexpected HTTPS target.");
        bool h3=false; int port=443, last=-1; bool portSeen=false; int[] mandatory=null;
        while(i<b.Length) {
            if(i+4>b.Length) throw new InvalidOperationException("Truncated HTTPS parameter.");
            int key=U16(b,i), len=U16(b,i+2); i+=4;
            if(key<=last || i+len>b.Length) throw new InvalidOperationException("Invalid HTTPS parameter.");
            last=key; int end=i+len;
            if(key==0) { if(len==0 || len%2!=0) throw new InvalidOperationException("Invalid mandatory parameter."); mandatory=new int[len/2]; for(int j=0;j<mandatory.Length;j++){mandatory[j]=U16(b,i+2*j); if((mandatory[j]!=1 && mandatory[j]!=3) || (j>0 && mandatory[j]<=mandatory[j-1])) throw new InvalidOperationException("Unsupported mandatory parameter.");} }
            if(key==1) { if(len==0) throw new InvalidOperationException("Empty ALPN."); while(i<end) {int n=b[i++]; if(n==0 || i+n>end) throw new InvalidOperationException("Invalid ALPN."); if(n==2 && b[i]=='h' && b[i+1]=='3') h3=true; i+=n;} }
            if(key==2 && len!=0) throw new InvalidOperationException("Invalid no-default-alpn.");
            if(key==3) {if(len!=2 || (port=U16(b,i))==0) throw new InvalidOperationException("Invalid HTTPS port."); portSeen=true;}
            i=end;
        }
        if(!h3) throw new InvalidOperationException("DNS does not advertise HTTP/3.");
        if(mandatory!=null) foreach(int key in mandatory) if((key==1&&!h3)||(key==3&&!portSeen)) throw new InvalidOperationException("Missing mandatory HTTPS parameter.");
        return port;
    }
    internal static int Port() {
        if(IntPtr.Size!=8) throw new InvalidOperationException("A 64-bit installer is required.");
        IntPtr records; int status=DnsQuery_W(Host,65,0,IntPtr.Zero,out records,IntPtr.Zero);
        if(status!=0 || records==IntPtr.Zero) throw new InvalidOperationException("HTTPS DNS lookup failed: "+status);
        try {
            int? port=null;
            for(IntPtr p=records;p!=IntPtr.Zero;p=Marshal.ReadIntPtr(p)) {
                if((ushort)Marshal.ReadInt16(p,16)!=65) continue;
                string name=Marshal.PtrToStringUni(Marshal.ReadIntPtr(p,8));
                if(!string.Equals(name == null ? null : name.TrimEnd('.'),Host,StringComparison.OrdinalIgnoreCase)) continue;
                int length=(ushort)Marshal.ReadInt16(p,18); var data=new byte[length]; Marshal.Copy(IntPtr.Add(p,32),data,0,length);
                if(port.HasValue) throw new InvalidOperationException("Multiple HTTPS service records are unsupported.");
                port=Parse(data);
            }
            if(!port.HasValue) throw new InvalidOperationException("HTTPS DNS record unavailable.");
            return port.Value;
        } finally { DnsRecordListFree(records,1); }
    }
    #if TESTING
    internal static void SelfTest() {
        byte[] valid={0,1,0,0,1,0,3,2,104,51,0,3,0,2,0x20,0xfc};
        if(Parse(valid)!=8444) throw new Exception("HTTPS port parsing failed.");
        byte[][] bad={new byte[]{0,0,0},new byte[]{0,1,0,0,1,0,3,2,104},new byte[]{0,1,0,0,1,0,3,2,104,50},new byte[]{0,1,0,0,3,0,2,0,0},new byte[]{0,1,1,120,0,0,1,0,3,2,104,51},new byte[]{0,1,0,0,1,0,3,2,104,51,0,1,0,0},new byte[]{0,1,0,0,0,0,2,0,9,0,1,0,3,2,104,51}};
        foreach(var input in bad) {bool rejected=false; try{Parse(input);}catch(InvalidOperationException){rejected=true;} if(!rejected) throw new Exception("Invalid HTTPS record accepted.");}
    }
    #endif
}
