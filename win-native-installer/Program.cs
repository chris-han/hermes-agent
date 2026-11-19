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
  internal const string MetadataUrl=ReleaseSelection.MetadataBase+"/"+ReleaseSelection.ChannelKey;
  const string Identity="Semantier.Agent.Acceptance",Publisher="CN=Semantier Technology, O=Semantier Technology";
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
      if(args.Length==2 && args[0]=="--describe-fixed-download"){File.WriteAllText(args[1],MetadataUrl,new UTF8Encoding(false));return 0;}
      if(args.Length==4 && args[0]=="--probe-oss-transport"){
        long size;if(!long.TryParse(args[2],out size) || size<=0 || size>2097152 || args[1].Length!=64 || !args[1].All(Uri.IsHexDigit))throw new ArgumentException("Invalid small EXE probe binding.");
        using(var client=new NativeWinHttpRequests())using(var output=new MemoryStream()){
          BootstrapperWinHttp.Download(client,new Uri(ReleaseSelection.PackageBase+"/releases/installers/semantier-setup-ch1.exe"),output,size,done=>{});
          string observed;using(var hash=SHA256.Create())observed=BitConverter.ToString(hash.ComputeHash(output.ToArray())).Replace("-","").ToLowerInvariant();
          if(observed!=args[1].ToLowerInvariant())throw new InvalidDataException("Small EXE transport SHA256 mismatch.");
          File.WriteAllText(args[3],"{\"status\":\"PASS\",\"scope\":\"SMALL_EXE_TRANSPORT_ONLY_NO_INSTALL\",\"bytes\":"+client.ReceivedBytes+",\"responses\":"+client.OpenedResponses+",\"protocol\":"+client.LastProtocol+",\"sha256\":\""+observed+"\"}",new UTF8Encoding(false));
        }return 0;
      }
      if(args.Length==2 && args[0]=="--describe-release"){
        using(var client=new NativeWinHttpRequests()){
          var selected=ReleaseSelection.Resolve(url=>ReadMetadata(client,url));
          File.WriteAllText(args[1],"{\"scope\":\"SELECTION_ONLY_NO_TRANSFER\",\"buildId\":\""+selected.BuildId+"\",\"version\":\""+selected.Version+"\",\"sha256\":\""+selected.Sha256+"\",\"uri\":\""+selected.Uri.AbsoluteUri+"\",\"manifestSha256\":\""+selected.ManifestSha256+"\"}",new UTF8Encoding(false));
        }return 0;
      }
      if(args.Length==6 && args[0]=="--download-candidate"){
        long sequence;if(!long.TryParse(args[3],NumberStyles.None,CultureInfo.InvariantCulture,out sequence) || args[4].Length!=64 || !args[4].All(Uri.IsHexDigit))throw new ArgumentException("Invalid pinned candidate.");
        Directory.CreateDirectory(args[5]);DownloadPackage(args[5],null,args[4],null,args[1],args[2],sequence);return 0;
      }
      if(args.Length==3 && args[0]=="--download-release"){
        if(args[1].Length!=64 || !args[1].All(Uri.IsHexDigit))throw new ArgumentException("Invalid expected SHA256.");
        Directory.CreateDirectory(args[2]);DownloadPackage(args[2],null,args[1]);return 0;
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
  internal static byte[] ReadMetadata(NativeWinHttpRequests client,string url){
    if(!url.StartsWith(ReleaseSelection.MetadataBase+"/releases/",StringComparison.Ordinal))throw new InvalidDataException("Untrusted metadata origin.");
    using(var response=Open(client,new Uri(url),false))using(var output=new MemoryStream()){
      if(response.StatusCode!=200 || response.ContentLength<=0 || response.ContentLength>4194304)throw new InvalidDataException("Invalid metadata response.");
      var buffer=new byte[65536];int n;
      while((n=response.Body.Read(buffer,0,buffer.Length))>0){if(output.Length+n>response.ContentLength)throw new InvalidDataException("Metadata size mismatch.");output.Write(buffer,0,n);}
      if(output.Length!=response.ContentLength)throw new InvalidDataException("Incomplete metadata.");return output.ToArray();
    }
  }
  internal static string DownloadPackage(string directory,Action<long,long> progress,string expected,Action<string> report=null,string candidateBuild=null,string candidateManifestHash=null,long candidateSequence=0){
    using(var metadata=new NativeWinHttpRequests())using(var client=new NativeWinHttpRequests()){
      var release=ReleaseSelection.Resolve(url=>ReadMetadata(metadata,url),candidateBuild,candidateManifestHash,candidateSequence);
      if(expected!=null && !string.Equals(release.Sha256,expected,StringComparison.OrdinalIgnoreCase))throw new InvalidDataException("Selected release differs from expected SHA256.");
      using(var head=Open(client,release.Uri,true))if(head.StatusCode!=200 || head.ContentLength!=release.Size)throw new InvalidDataException("OSS package size mismatch.");
      string file=Path.Combine(directory,"SemantierAgent.msix");
      // CreateNew prevents an accidental second full download over retained evidence.
      using(var output=new FileStream(file,FileMode.CreateNew,FileAccess.Write,FileShare.None)){
        var clock=Stopwatch.StartNew();long last=0;
        BootstrapperWinHttp.Download(client,release.Uri,output,release.Size,done=>{if(progress!=null && (done==release.Size || clock.ElapsedMilliseconds-last>=200)){progress(done,release.Size);last=clock.ElapsedMilliseconds;}});
      }
      if(report!=null)report("Verifying package and publisher…");
      string observed;using(var stream=File.OpenRead(file))using(var sha=SHA256.Create()){observed=BitConverter.ToString(sha.ComputeHash(stream)).Replace("-","").ToLowerInvariant();}
      if(observed!=release.Sha256)throw new InvalidDataException("MSIX SHA256 mismatch; no full-file retry.");
      VerifyIdentity(file,release);VerifySignature(file);
      var current=ReleaseSelection.Resolve(url=>ReadMetadata(metadata,url),candidateBuild,candidateManifestHash,candidateSequence);
      if(current.ManifestSha256!=release.ManifestSha256 || current.ChannelSha256!=release.ChannelSha256)throw new InvalidDataException("Selected release changed during download.");
      File.WriteAllText(Path.Combine(directory,"download-receipt.json"),"{\"status\":\"DOWNLOADED_VERIFIED_NOT_INSTALLED\",\"selection\":\""+(candidateBuild==null?"ACTIVE_CHANNEL_HEAD":"PINNED_STAGED_CANDIDATE")+"\",\"producedAt\":\""+DateTime.UtcNow.ToString("o")+"\",\"manifestSha256\":\""+release.ManifestSha256+"\",\"channelSha256\":\""+release.ChannelSha256+"\",\"buildId\":\""+release.BuildId+"\",\"version\":\""+release.Version+"\",\"sha256\":\""+observed+"\",\"uri\":\""+release.Uri.AbsoluteUri+"\",\"sourceCommit\":\""+release.Commit+"\",\"nativeDigest\":\""+release.NativeDigest+"\",\"signature\":\"PINNED_SIGNER_VALID\",\"origin\":\""+ReleaseSelection.PackageBase+"\",\"ossResponseBytes\":"+client.ReceivedBytes+",\"responses\":"+client.OpenedResponses+",\"protocol\":"+client.LastProtocol+"}",new UTF8Encoding(false));
      return file;
    }
  }
  internal static void VerifyIdentity(string file,ReleaseSelection selected=null){
    using(var zip=ZipFile.OpenRead(file)){
      var entry=zip.GetEntry("AppxManifest.xml");if(entry==null)throw new InvalidDataException("Missing MSIX manifest.");
      using(var stream=entry.Open()){
        var identity=XDocument.Load(stream).Descendants().First(x=>x.Name.LocalName=="Identity");Version version;
        if((string)identity.Attribute("Name")!=Identity || (string)identity.Attribute("Publisher")!=Publisher || (string)identity.Attribute("ProcessorArchitecture")!="x64" || !Version.TryParse((string)identity.Attribute("Version"),out version))throw new InvalidDataException("Unexpected MSIX identity/publisher/architecture."); if(selected!=null && version.ToString()!=selected.Version)throw new InvalidDataException("Downloaded version differs from selected release.");AppVersionString=version.ToString(3);
      }
      if(selected!=null){
        var stamp=zip.GetEntry("app/resources/install-stamp.json");if(stamp==null)throw new InvalidDataException("Missing package source stamp.");
        using(var input=stamp.Open())using(var bytes=new MemoryStream()){if(stamp.Length>4194304)throw new InvalidDataException("Stamp exceeds limit.");input.CopyTo(bytes);selected.VerifyStamp(bytes.ToArray());}
      }
      if(zip.GetEntry("AppxSignature.p7x")==null)throw new InvalidDataException("Unsigned MSIX.");
    }
  }
  internal static void VerifySignature(string file){
    string command="$ErrorActionPreference='Stop'; $s=Get-AuthenticodeSignature -LiteralPath '"+file.Replace("'","''")+"'; if($s.Status -ne 'Valid' -or $s.SignerCertificate.Subject -cne 'CN=Semantier Technology, O=Semantier Technology' -or $s.SignerCertificate.Thumbprint -cne '57BF15739B4AA447842EAF388856A01B4E5029FF'){exit 1}";
    string shell=Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System),"WindowsPowerShell","v1.0","powershell.exe");
    var info=new ProcessStartInfo(shell,"-NoProfile -NonInteractive -EncodedCommand "+Convert.ToBase64String(Encoding.Unicode.GetBytes(command))){UseShellExecute=false,CreateNoWindow=true};
    info.EnvironmentVariables["PSModulePath"]=Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System),"WindowsPowerShell","v1.0","Modules");
    using(var process=Process.Start(info)){
      if(process==null)throw new IOException("Cannot start Windows signature verifier.");
      if(!process.WaitForExit(600000)){process.Kill();throw new IOException("Windows signature verification timed out.");}
      if(process.ExitCode!=0)throw new InvalidDataException("MSIX signature or pinned publisher verification failed.");
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
    string command="$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; [Console]::OutputEncoding=New-Object Text.UTF8Encoding($false); try { Add-AppxPackage -Path '"+file.Replace("'","''")+"' -ForceApplicationShutdown } catch { [Console]::Error.WriteLine(($_ | Format-List * -Force | Out-String)); exit 1 }";
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