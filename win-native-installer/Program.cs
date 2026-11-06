using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.Reflection;
using System.Net;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Xml.Linq;
[assembly: System.Reflection.AssemblyCompany("Semantier Technology")]
[assembly: System.Reflection.AssemblyProduct("Semantier Setup")]
[assembly: System.Reflection.AssemblyCopyright("Copyright (c) 2026 Semantier Technology")]
internal static class Program {
  internal const string MsixUrl="https://download.semantier.cn/releases/SemantierAgent.msix";
  const string Identity="Semantier.Agent.Acceptance",Publisher="CN=Semantier Technology, O=Semantier Technology";
  const long MaxBytes=12000000000;
  internal static string AppVersionString="latest";
  internal static string LogPath=Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),"Semantier","Setup",DateTime.UtcNow.ToString("yyyyMMdd-HHmmss")+"-"+Guid.NewGuid().ToString("N"),"log.txt");
  static readonly object LogGate=new object();
  internal static void Log(string message){
    lock(LogGate){Directory.CreateDirectory(Path.GetDirectoryName(LogPath));File.AppendAllText(LogPath,DateTime.UtcNow.ToString("o")+" "+message+Environment.NewLine,new UTF8Encoding(false));}
  }
  [STAThread] static int Main(string[] args) {
    try {
#if LOCAL_SETUP
      if(args.Length==2 && args[0]=="--describe-fixed-download"){File.WriteAllText(args[1],"local-adjacent-single-msix",new UTF8Encoding(false));return 0;}
      if(args.Length!=0)return 2;
#else
      if(args.Length==2 && args[0]=="--describe-fixed-download"){File.WriteAllText(args[1],MsixUrl,new UTF8Encoding(false));return 0;}
      if(args.Length==1 && args[0]=="--probe-h3"){
        using(var client=new NativeWinHttpRequests())using(var response=client.Open(new Uri(MsixUrl+".sha256"),0,true)){
          string checksum=ReadChecksum(response);
          if(response.Protocol!=2)throw new IOException("Expected HTTP/3.");
          File.WriteAllText(Path.Combine(Path.GetTempPath(),"semantier-setup-h3-probe.json"),"{\"protocol\":\"h3\",\"url\":\""+MsixUrl+".sha256\",\"sha256\":\""+checksum+"\"}");
        }return 0;
      }
      if(args.Length==2 && args[0]=="--verify-remote"){
        if(args[1].Length!=64 || !args[1].All(Uri.IsHexDigit))throw new ArgumentException("Invalid expected SHA-256.");
        string temp=TemporaryDirectory();try{DownloadPackage(temp,null,args[1]);}finally{Directory.Delete(temp,true);}return 0;
      }
      if(args.Length!=0)return 2;
#endif
      Application.EnableVisualStyles();
      using(var form=new InstallerWindow(report=>{
#if LOCAL_SETUP
        report("Verifying package and publisher…");
        string package=ResolveLocalPackage(AppDomain.CurrentDomain.BaseDirectory);
        VerifyIdentity(package);
        report("Installing for the current Windows user…");Install(package,report);
        report("Semantier Agent "+AppVersionString+" installed.");
#else
        string directory=TemporaryDirectory();
        try{
          report("Downloading signed release…");
          string package=DownloadPackage(directory,(done,total)=>report("Downloading "+(100*done/total).ToString(CultureInfo.InvariantCulture)+"%"),null,report);
          report("Installing for the current Windows user…");Install(package,report);
          report("Semantier Agent "+AppVersionString+" installed.");
        }finally{try{Directory.Delete(directory,true);}catch(IOException){report("Temporary files may remain.");}catch(UnauthorizedAccessException){report("Temporary files may remain.");}}
#endif
      })){Application.Run(form);}
      return 0;
    }catch(Exception error){File.WriteAllText(Path.Combine(Path.GetTempPath(),"semantier-setup-last-error.txt"),error.ToString());return 1;}
  }
  internal static string ResolveLocalPackage(string directory){
    Log("Looking for local MSIX in "+directory);
    string standard=Path.Combine(directory,"SemantierAgent.msix");
    if(File.Exists(standard))return standard;
    string[] files=Directory.GetFiles(directory,"*",SearchOption.TopDirectoryOnly).Where(x=>string.Equals(Path.GetExtension(x),".msix",StringComparison.OrdinalIgnoreCase)).ToArray();
    if(files.Length==0)throw new FileNotFoundException("Place one signed MSIX in the same folder as this Setup EXE.");
    if(files.Length!=1)throw new InvalidDataException("More than one MSIX found. Keep only the package you want to install next to this Setup EXE.");
    return files[0];
  }
  static string TemporaryDirectory(){string path=Path.Combine(Path.GetTempPath(),"semantier-install-"+Guid.NewGuid().ToString("N"));Directory.CreateDirectory(path);return path;}
  static IWinHttpResponse Open(NativeWinHttpRequests client,Uri uri,bool head){
    try{return head?client.OpenHead(uri,true):client.Open(uri,0,true);}
    catch(IOException){return head?client.OpenHead(uri,false):client.Open(uri,0,false);}
  }
  static string ReadChecksum(IWinHttpResponse response){
    if(response.StatusCode!=200 || response.ContentLength<64 || response.ContentLength>4096)throw new InvalidDataException("Invalid checksum response.");
    using(var memory=new MemoryStream()){
      byte[] buffer=new byte[4096];int n;while((n=response.Body.Read(buffer,0,buffer.Length))>0){if(memory.Length+n>4096)throw new InvalidDataException("Checksum response too large.");memory.Write(buffer,0,n);}
      if(memory.Length!=response.ContentLength)throw new InvalidDataException("Incomplete checksum response.");
      string value=Encoding.ASCII.GetString(memory.ToArray()).Split(new[]{' ','\t','\r','\n'},StringSplitOptions.RemoveEmptyEntries).FirstOrDefault()??"";
      if(value.Length!=64 || !value.All(Uri.IsHexDigit))throw new InvalidDataException("Invalid SHA-256 sidecar.");return value.ToLowerInvariant();
    }
  }
  internal static string DownloadPackage(string directory,Action<long,long> progress,string expected,Action<string> report=null){
    using(var client=new NativeWinHttpRequests())for(int attempt=0;attempt<3;attempt++){
      string wanted;using(var response=Open(client,new Uri(MsixUrl+".sha256"),false)){wanted=ReadChecksum(response);}
      if(expected!=null && !string.Equals(wanted,expected,StringComparison.OrdinalIgnoreCase))throw new InvalidDataException("Public release differs from expected SHA-256.");
      long size;using(var head=Open(client,new Uri(MsixUrl),true)){
        if(head.StatusCode!=200 || head.ContentLength<=0 || head.ContentLength>MaxBytes)throw new InvalidDataException("Invalid MSIX response.");size=head.ContentLength;
      }
      string file=Path.Combine(directory,"SemantierAgent.msix");
      using(var output=new FileStream(file,FileMode.Create,FileAccess.Write,FileShare.None)){
        var clock=Stopwatch.StartNew();long last=0;
        BootstrapperWinHttp.Download(client,new Uri(MsixUrl),output,size,done=>{if(progress!=null && (done==size || clock.ElapsedMilliseconds-last>=200)){progress(done,size);last=clock.ElapsedMilliseconds;}});
      }
      if(report!=null)report("Verifying package and publisher…");
      string observed;using(var stream=File.OpenRead(file))using(var sha=SHA256.Create()){observed=BitConverter.ToString(sha.ComputeHash(stream)).Replace("-","").ToLowerInvariant();}
      if(string.Equals(wanted,observed,StringComparison.OrdinalIgnoreCase)){VerifyIdentity(file);return file;}
      if(attempt==2)throw new InvalidDataException("MSIX SHA-256 mismatch; installation stopped.");Thread.Sleep(1000);
    }throw new InvalidDataException("No matching release.");
  }
  internal static void VerifyIdentity(string file){
    using(var zip=ZipFile.OpenRead(file)){
      var entry=zip.GetEntry("AppxManifest.xml");if(entry==null)throw new InvalidDataException("Missing MSIX manifest.");
      using(var stream=entry.Open()){
        var identity=XDocument.Load(stream).Descendants().First(x=>x.Name.LocalName=="Identity");Version version;
        if((string)identity.Attribute("Name")!=Identity || (string)identity.Attribute("Publisher")!=Publisher || (string)identity.Attribute("ProcessorArchitecture")!="x64" || !Version.TryParse((string)identity.Attribute("Version"),out version))throw new InvalidDataException("Unexpected MSIX identity/publisher/architecture."); AppVersionString=version.ToString(3);
      }if(zip.GetEntry("AppxSignature.p7x")==null)throw new InvalidDataException("Unsigned MSIX.");
    }
  }
  internal static X509Certificate2 PublisherCertificate(){
    using(var resource=Assembly.GetExecutingAssembly().GetManifestResourceStream("Semantier.publisher.cer")){
      if(resource==null)throw new InvalidDataException("Certificate resource is missing.");
      using(var bytes=new MemoryStream()){resource.CopyTo(bytes);var cert=new X509Certificate2(bytes.ToArray());ValidatePublisherCertificate(cert);return cert;}
    }
  }
  internal static void ValidatePublisherCertificate(X509Certificate2 cert){
    if(cert.Thumbprint!="57BF15739B4AA447842EAF388856A01B4E5029FF" || cert.Subject!=Publisher || cert.HasPrivateKey)
      throw new InvalidDataException("Certificate does not match the pinned Semantier publisher.");
  }
  static void EnsurePublisherTrust(Action<string> report){
    using(var cert=PublisherCertificate()){
      foreach(StoreName name in new[]{StoreName.TrustedPeople,StoreName.Root})using(var store=new X509Store(name,StoreLocation.LocalMachine)){
        store.Open(OpenFlags.ReadOnly);
        if(store.Certificates.Find(X509FindType.FindByThumbprint,cert.Thumbprint,false).Count>0)return;
      }
      Log("Publisher certificate needs machine trust: "+cert.Thumbprint);
      if(report!=null)report("Waiting for Windows administrator consent…");
      string command="$ErrorActionPreference='Stop'; $c=New-Object Security.Cryptography.X509Certificates.X509Certificate2(,[Convert]::FromBase64String('"+Convert.ToBase64String(cert.RawData)+"')); "+
        "if($c.Thumbprint -cne '57BF15739B4AA447842EAF388856A01B4E5029FF' -or $c.Subject -cne 'CN=Semantier Technology, O=Semantier Technology'){exit 2}; "+
        "$s=New-Object Security.Cryptography.X509Certificates.X509Store('TrustedPeople','LocalMachine'); try{$s.Open('ReadWrite');$s.Add($c)}finally{$s.Close()}";
      var consent=new ProcessStartInfo("powershell.exe","-NoProfile -NonInteractive -EncodedCommand "+Convert.ToBase64String(Encoding.Unicode.GetBytes(command))){UseShellExecute=true,Verb="runas",WindowStyle=ProcessWindowStyle.Hidden};
      using(var process=Process.Start(consent)){
        if(process==null)throw new IOException("Elevation could not start.");process.WaitForExit();
        if(process.ExitCode!=0)throw new IOException("Certificate trust was not completed (exit "+process.ExitCode+"). Administrator approval is required on this PC.");
      }
      using(var store=new X509Store(StoreName.TrustedPeople,StoreLocation.LocalMachine)){
        store.Open(OpenFlags.ReadOnly);
        if(store.Certificates.Find(X509FindType.FindByThumbprint,cert.Thumbprint,false).Count==0)throw new IOException("Certificate trust was not applied.");
      }
      Log("Pinned publisher certificate trusted in LocalMachine/TrustedPeople.");
      if(report!=null)report("Installing for the current Windows user…");
    }
  }
  internal static void Install(string file,Action<string> report=null){
    EnsurePublisherTrust(report);
    Log("Installing "+file+" for "+Environment.UserDomainName+"\\"+Environment.UserName);
    string command="$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; [Console]::OutputEncoding=New-Object Text.UTF8Encoding($false); try { Add-AppxPackage -Path '"+file.Replace("'","''")+"' -ForceApplicationShutdown -ForceUpdateFromAnyVersion } catch { [Console]::Error.WriteLine(($_ | Format-List * -Force | Out-String)); exit 1 }";
    var info=new ProcessStartInfo("powershell.exe","-NoProfile -NonInteractive -OutputFormat Text -EncodedCommand "+Convert.ToBase64String(Encoding.Unicode.GetBytes(command))){UseShellExecute=false,CreateNoWindow=true,RedirectStandardOutput=true,RedirectStandardError=true,StandardOutputEncoding=Encoding.UTF8,StandardErrorEncoding=Encoding.UTF8};
    using(var process=Process.Start(info)){
      if(process==null)throw new IOException("Cannot start Windows installer.");
      var output=process.StandardOutput.ReadToEndAsync();var errors=process.StandardError.ReadToEndAsync();
      process.WaitForExit();Task.WaitAll(output,errors);
      Log("Windows installer exit="+process.ExitCode+"\r\n"+errors.Result+"\r\n"+output.Result);
      if(process.ExitCode!=0)throw new IOException("Windows rejected MSIX: "+process.ExitCode+"\r\n"+errors.Result+"\r\n"+output.Result);
    }
  }
}