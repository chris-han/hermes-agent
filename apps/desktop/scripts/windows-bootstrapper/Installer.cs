using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.AccessControl;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;
using System.Security.Principal;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Microsoft.Win32.SafeHandles;

#if CHANNEL1
[assembly: AssemblyTitle("Semantier Agent Channel 1 Installer")]
#elif CHANNEL2
[assembly: AssemblyTitle("Semantier Agent Channel 2 Installer")]
#else
#error Select exactly one supported channel
#endif
[assembly: AssemblyProduct("Semantier Agent Installer")]
[assembly: AssemblyCompany("Semantier Technology")]
[assembly: AssemblyVersion("2.0.0.0")]
[assembly: AssemblyFileVersion("2.0.0.0")]

static class Installer {
    #if CHANNEL1
    const string Origin = "https://download.semantier.cn";
    #else
    const string Origin = "https://download.semantier.com";
    #endif
    const string CertHash = "10a6b55e9a03bbc0dceb8d899c1776bf1fbad963c82050c16534da4983c9661d";
    const string Thumbprint = "57BF15739B4AA447842EAF388856A01B4E5029FF";
    const string Identity = "Semantier.Agent.Acceptance";
    static ActiveRelease release;
    static Version Version { get { if(release==null) throw new InvalidOperationException("Active release not resolved."); return release.Version; } }
    static string ExecutablePath { get { return Environment.ProcessPath; } }
    internal static string AppVersionString { get { return release==null ? InstallerText.Choose("current", "当前版本") : release.Version.ToString(); } }
    [StructLayout(LayoutKind.Sequential)] struct FileInformation { public uint Attributes; public System.Runtime.InteropServices.ComTypes.FILETIME Created,Accessed,Written; public uint Volume,SizeHigh,SizeLow,Links,IndexHigh,IndexLow; }
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)] static extern SafeFileHandle CreateFile(string path,uint access,uint share,IntPtr security,uint creation,uint flags,IntPtr template);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool GetFileInformationByHandle(SafeFileHandle handle,out FileInformation info);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool SetFileInformationByHandle(SafeFileHandle handle,int informationClass,ref byte information,uint size);
    static bool SameFile(FileInformation a,FileInformation b) {return a.Volume==b.Volume && a.IndexHigh==b.IndexHigh && a.IndexLow==b.IndexLow;}
    static bool DeleteHeld(string path,SafeFileHandle original,bool directory,Action release) {
        uint flags=directory?0x02200000u:0x00200000u;FileInformation expected,reference,actual;
        if(!GetFileInformationByHandle(original,out expected)){release();return false;}
        // Open a permissive reference while the original still denies rename.
        // Keeping it open prevents the original file ID from being recycled.
        using(var tracker=CreateFile(path,0x80000000,7,IntPtr.Zero,3,flags,IntPtr.Zero)) {
            if(tracker.IsInvalid || !GetFileInformationByHandle(tracker,out reference) || !SameFile(expected,reference)){release();return false;}
            release();
            using(var target=CreateFile(path,0x00010080,3,IntPtr.Zero,3,flags,IntPtr.Zero)) {
                if(target.IsInvalid || !GetFileInformationByHandle(target,out actual) || !SameFile(reference,actual) || (actual.Attributes & 0x400)!=0)return false;
                byte delete=1;return SetFileInformationByHandle(target,4,ref delete,1);
            }
        }
    }
    // A generic-read directory handle without FILE_SHARE_DELETE prevents rename
    // throughout verification/elevation. Never traverse a reparse point.
    static List<SafeFileHandle> HoldDirectories(string path) {
        string full=Path.GetFullPath(path); string root=Path.GetPathRoot(full);
        if(root.StartsWith(@"\\") || new DriveInfo(root).DriveType!=DriveType.Fixed) throw new InvalidOperationException("A local fixed drive is required.");
        var handles=new List<SafeFileHandle>();
        try {
            string current=root;
            var parts=full.Substring(root.Length).Split(new[]{Path.DirectorySeparatorChar},StringSplitOptions.RemoveEmptyEntries);
            foreach(string part in new[]{""}.Concat(parts)) {
                if(part.Length>0) current=Path.Combine(current,part);
                var handle=CreateFile(current,0x80000000,3,IntPtr.Zero,3,0x02200000,IntPtr.Zero);
                if(handle.IsInvalid) {handle.Dispose();throw new System.ComponentModel.Win32Exception();}
                handles.Add(handle); FileInformation info;
                if(!GetFileInformationByHandle(handle,out info) || (info.Attributes & 0x400)!=0 || (info.Attributes & 0x10)==0) throw new InvalidOperationException("Unsafe installation directory.");
            }
            return handles;
        } catch {foreach(var h in handles)h.Dispose();throw;}
    }
    static string Hex(byte[] data) { return BitConverter.ToString(data).Replace("-", "").ToLowerInvariant(); }
    static string Hash(Stream s) { using(var h=SHA256.Create()) return Hex(h.ComputeHash(s)); }
    static byte[] EmbeddedCertificate() {
        using(var s=Assembly.GetExecutingAssembly().GetManifestResourceStream("SemantierAgent-signing.cer")) {
            if(s==null) throw new InvalidOperationException("Missing pinned certificate.");
            using(var m=new MemoryStream()) { s.CopyTo(m); return m.ToArray(); }
        }
    }
    static X509Certificate2 Certificate() {
        var bytes=EmbeddedCertificate(); using(var m=new MemoryStream(bytes)) if(Hash(m)!=CertHash) throw new InvalidOperationException("Certificate hash mismatch.");
        var cert=new X509Certificate2(bytes);
        if(cert.Thumbprint!=Thumbprint || DateTime.UtcNow<cert.NotBefore.ToUniversalTime() || DateTime.UtcNow>=cert.NotAfter.ToUniversalTime()) throw new InvalidOperationException("Certificate identity or validity mismatch.");
        return cert;
    }
    static string TaskBase() {return Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.CommonDocuments),"SemantierInstaller");}
    static string TaskPath(string id) {
        Guid parsed;
        if(!Guid.TryParseExact(id,"N",out parsed) || parsed.ToString("N")!=id) throw new InvalidOperationException("Invalid task identity.");
        return Path.Combine(TaskBase(),id);
    }
    static string PowerShell(string script) {
        string exe=Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System),@"WindowsPowerShell\v1.0\powershell.exe");
        var p=new ProcessStartInfo(exe,"-NoProfile -NonInteractive -EncodedCommand "+Convert.ToBase64String(Encoding.Unicode.GetBytes("$ErrorActionPreference='Stop';"+script))) {
            UseShellExecute=false, CreateNoWindow=true, RedirectStandardOutput=true, RedirectStandardError=true
        };
        using(var process=Process.Start(p)) {
            // PowerShell progress also uses stderr; drain both pipes before waiting.
            var outputRead=process.StandardOutput.ReadToEndAsync();
            var errorRead=process.StandardError.ReadToEndAsync();
            process.WaitForExit();
            string output=outputRead.GetAwaiter().GetResult();
            string error=errorRead.GetAwaiter().GetResult();
            if(process.ExitCode!=0) throw new InvalidOperationException("Windows package operation failed: "+error);
            return output.Trim();
        }
    }
    static void CheckInstalled() {
        string found=PowerShell("@(Get-AppxPackage -Name '"+Identity+"' | ForEach-Object {$_.Version.ToString()}) -join ','");
        if(found.Length==0) return;
        foreach(string item in found.Split(',')) {
            var current=new Version(item.Trim());
            if(current>Version) throw new InvalidOperationException("A newer Semantier Agent is already installed. Downgrade is not performed.");
            if(current==Version) throw new InvalidOperationException("Semantier Agent " + Version + " is already installed for this user.");
        }
    }
    static FileStream CreateOwned(string path,List<FileStream> held) {
        var file=new FileStream(path,FileMode.CreateNew,FileAccess.ReadWrite,FileShare.Read);held.Add(file);return file;
    }
    static FileStream Seal(FileStream file,List<FileStream> held) {
        // A no-delete tracking handle bridges the write-to-read transition.
        // Hash only after the new handle also excludes writers.
        var tracker=new FileStream(file.Name,FileMode.Open,FileAccess.Read,FileShare.ReadWrite);held.Add(tracker);
        file.Dispose();held.Remove(file);
        var strict=new FileStream(tracker.Name,FileMode.Open,FileAccess.Read,FileShare.Read);held.Add(strict);
        tracker.Dispose();held.Remove(tracker);return strict;
    }
    static bool PreferHttp3 => true;
    static byte[] ReadMetadata(System.Net.Http.HttpClient client, Uri uri) {
        using(var response=BootstrapperTransport.Send(client,uri,preferHttp3:PreferHttp3)) {
            if(response.StatusCode!=HttpStatusCode.OK || response.RequestMessage?.RequestUri!=uri ||
                response.Content.Headers.ContentLength is long length && (length<=0 || length>4*1024*1024))
                throw new InvalidOperationException("Unexpected metadata response.");
            using(var input=response.Content.ReadAsStreamAsync().GetAwaiter().GetResult())
            using(var buffer=new MemoryStream()) {
                var chunk=new byte[16384]; int n;
                while((n=input.Read(chunk,0,chunk.Length))>0) { if(buffer.Length+n>4*1024*1024) throw new InvalidOperationException("Metadata exceeds limit."); buffer.Write(chunk,0,n); }
                if(response.Content.Headers.ContentLength is long expected && buffer.Length!=expected) throw new InvalidOperationException("Incomplete metadata response.");
                return buffer.ToArray();
            }
        }
    }
    static FileStream DownloadObject(System.Net.Http.HttpClient client,ActiveRelease.Object item,string dir,Action<string> report,List<FileStream> held) {
        string name=Path.GetFileName(item.Key); var target=CreateOwned(Path.Combine(dir,name),held); int previous=-1;
        BootstrapperTransport.Download(client,new Uri(Origin+"/"+item.Key),target,item.Size,total=>{
            int percent=(int)(total*100/item.Size); if(percent!=previous){report("Downloading "+percent+"%");previous=percent;}
        },preferHttp3:PreferHttp3);
        target.Flush(true); var file=Seal(target,held);
        if(file.Length!=item.Size || Hash(file)!=item.Sha256) throw new InvalidOperationException("Package SHA-256 mismatch.");
        return file;
    }
    static string PrivateDirectory(string id) {
        var sid=WindowsIdentity.GetCurrent().User;
        var security=new DirectorySecurity(); security.SetOwner(sid); security.SetAccessRuleProtection(true,false);
        security.AddAccessRule(new FileSystemAccessRule(sid,FileSystemRights.FullControl,InheritanceFlags.ContainerInherit|InheritanceFlags.ObjectInherit,PropagationFlags.None,AccessControlType.Allow));
        security.AddAccessRule(new FileSystemAccessRule(new SecurityIdentifier(WellKnownSidType.BuiltinAdministratorsSid,null),FileSystemRights.ReadAndExecute,InheritanceFlags.ContainerInherit|InheritanceFlags.ObjectInherit,PropagationFlags.None,AccessControlType.Allow));
        string p=TaskPath(id); if(Directory.Exists(p))throw new InvalidOperationException("Task directory already exists.");
        #if CHANNEL1 || CHANNEL2
        new DirectoryInfo(p).Create(security);
        #else
        Directory.CreateDirectory(p,security);
        #endif
        return p;
    }
    static void Install(string dir) {
        string encoded=Convert.ToBase64String(Encoding.Unicode.GetBytes(Path.Combine(dir,Path.GetFileName(release.Root))));
        #if CHANNEL2
        PowerShell("$p=[Text.Encoding]::Unicode.GetString([Convert]::FromBase64String('"+encoded+"'));"+
            "$installed=@(Get-AppxPackage -Name '"+Identity+"' | Where-Object {!$_.IsBundle});"+
            "if($installed.Count -gt 1){throw 'Ambiguous installed package';}"+
            "if($installed.Count -eq 1){$path=$installed[0].InstallLocation;"+
            "$volumes=@(Get-AppxVolume | Where-Object {!$_.IsOffline -and $path.StartsWith($_.PackageStorePath+'\\',[StringComparison]::OrdinalIgnoreCase)});"+
            "if($volumes.Count -ne 1){throw 'Installed package volume unavailable';}"+
            "Add-AppxPackage -Path $p -Volume $volumes[0]}else{Add-AppxPackage -Path $p}");
        #else
        PowerShell("$p=[Text.Encoding]::Unicode.GetString([Convert]::FromBase64String('"+encoded+"')); Add-AppxPackage -Path $p");
        #endif
        string installed=PowerShell("@(Get-AppxPackage -Name '"+Identity+"' | Where-Object {$_.Version.ToString() -eq '"+Version+"'}).Count");
        if(installed!="1") throw new InvalidOperationException("Windows did not report the expected installed version.");
    }
    static void Execute(Action<string> report) {
        string id=Guid.NewGuid().ToString("N"), dir=null; bool installed=false; var held=new List<FileStream>(); var directories=new List<SafeFileHandle>();
        try {
            using(var handler=new System.Net.Http.SocketsHttpHandler{AllowAutoRedirect=false,UseProxy=false})
            using(var client=new System.Net.Http.HttpClient(handler)) {
                client.Timeout=TimeSpan.FromHours(4);
                client.DefaultRequestHeaders.CacheControl=new System.Net.Http.Headers.CacheControlHeaderValue{NoCache=true,NoStore=true};
                report("Resolving current release…"); release=ActiveRelease.Fetch(uri=>ReadMetadata(client,uri),Origin); CheckInstalled();
                using(var cert=Certificate()) { if(cert.HasPrivateKey) throw new InvalidOperationException("Unexpected private certificate key."); }
                Directory.CreateDirectory(TaskBase()); directories=HoldDirectories(TaskBase());
                dir=PrivateDirectory(id);directories.AddRange(HoldDirectories(dir));
                foreach(var item in release.Objects) DownloadObject(client,item,dir,report,held);
                report("Verifying package and certificate…"); PackageSignature.Verify(dir,release.Objects); PackageIdentity.Verify(release,dir);
                CheckInstalled(); report("Installing for the current Windows user…"); Install(dir); installed=true;
            }
        } finally {
            bool cleaned=true;
            foreach(var h in held) {try {var file=h;string name=file.Name;if(!DeleteHeld(name,file.SafeFileHandle,false,()=>file.Dispose()))cleaned=false;}catch{cleaned=false;h.Dispose();}}
            if(dir!=null && directories.Count>0) {var last=directories[directories.Count-1];try{if(!DeleteHeld(dir,last,true,()=>last.Dispose()))cleaned=false;}catch{cleaned=false;last.Dispose();}}
            foreach(var h in directories)h.Dispose();
            if(dir!=null)report((installed?"Semantier Agent "+Version+" installed. ":"Installation stopped. ")+(cleaned?"Temporary files removed.":"Temporary files may remain in " + dir));
        }
    }
    #if TESTING
    static void SelfTest() {
        if(!PreferHttp3) throw new InvalidOperationException("Both channels must attempt HTTP/3 before TCP fallback.");
        #if CHANNEL1
        #endif
        // More than a pipe buffer on stderr must not prevent stdout completion.
        var noisyChild=System.Threading.Tasks.Task.Run(()=>PowerShell("[Console]::Error.Write(('x' * 262144)); Write-Output 'streams-drained'"));
        if(!noisyChild.Wait(15000) || noisyChild.Result!="streams-drained")
            throw new InvalidOperationException("PowerShell output streams deadlocked or lost stdout.");
        bool childFailure=false;
        try { PowerShell("[Console]::Error.Write(('x' * 262144)); throw 'expected-stream-test-error'"); }
        catch(InvalidOperationException error) { childFailure=error.Message.Contains("expected-stream-test-error"); }
        if(!childFailure)throw new InvalidOperationException("PowerShell failure diagnostics were lost.");
        using(var cert=Certificate()) if(cert.HasPrivateKey) throw new InvalidOperationException("Unexpected private key.");
        string dir=Path.Combine(Path.GetDirectoryName(ExecutablePath),"fixture-"+Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);var locks=HoldDirectories(dir);string path=Path.Combine(dir,"data"),backup=Path.Combine(dir,"original");
        try {
            File.WriteAllText(path,"original");var original=new FileStream(path,FileMode.Open,FileAccess.Read,FileShare.Read);
            bool removed=DeleteHeld(path,original.SafeFileHandle,false,()=>{original.Dispose();File.Move(path,backup);File.WriteAllText(path,"replacement");});
            if(removed || File.ReadAllText(path)!="replacement")throw new InvalidOperationException("Replacement cleanup race was not refused.");
            File.Delete(path);File.Delete(backup);
            File.WriteAllText(path,"owned");var owned=new FileStream(path,FileMode.Open,FileAccess.Read,FileShare.Read);
            if(!DeleteHeld(path,owned.SafeFileHandle,false,()=>owned.Dispose()) || File.Exists(path))throw new InvalidOperationException("Owned cleanup failed.");
            var held=new List<FileStream>();var partial=CreateOwned(path,held);partial.WriteByte(1);partial.Flush(true);
            if(held.Count!=1 || !DeleteHeld(path,partial.SafeFileHandle,false,()=>partial.Dispose()) || File.Exists(path))throw new InvalidOperationException("Interrupted download cleanup failed.");
            held.Clear();var extracting=CreateOwned(path,held);extracting.WriteByte(2);extracting.Flush(true);var sealedFile=Seal(extracting,held);
            if(held.Count!=1 || !DeleteHeld(path,sealedFile.SafeFileHandle,false,()=>sealedFile.Dispose()) || File.Exists(path))throw new InvalidOperationException("Extracted file cleanup failed.");

        } finally {foreach(var h in locks)h.Dispose();Directory.Delete(dir,true);}
    }
    #endif
    [STAThread] static int Main(string[] args) {
        // Read-only fixture inspection exercises this exact compiled resolver; it
        // never downloads, installs, changes trust, or accepts an origin override.
        if((args.Length==4 || args.Length==5) && args[0]=="--inspect-metadata") {
            try {
                byte[] Read(string path) { var info=new FileInfo(path); if(!info.Exists || info.Length<=0 || info.Length>4*1024*1024) throw new InvalidOperationException("Invalid fixture size."); return File.ReadAllBytes(path); }
                int reads=0;
                var snapshot=ActiveRelease.Fetch(_=>Read(args[++reads]),Origin);
                if(reads!=args.Length-1) throw new InvalidOperationException("Invalid fixture read sequence.");
                Console.WriteLine(snapshot.BuildId+" "+snapshot.Version); return 0;
            } catch(Exception e) { Console.Error.WriteLine(e.Message); return 1; }
        }
        Application.EnableVisualStyles(); Application.SetCompatibleTextRenderingDefault(false);
        try {
            var executableDirectories=HoldDirectories(Path.GetDirectoryName(ExecutablePath));
            try {using(var self=new FileStream(ExecutablePath,FileMode.Open,FileAccess.Read,FileShare.Read)) {
                #if TESTING
                if(args.Length==1 && args[0]=="--self-test") {SelfTest(); InstallerWindow.SelfTest(); return 0;}
                if(args.Length==4 && args[0]=="--preview") {InstallerWindow.Preview(args[1],args[2],args[3]); return 0;}
                #endif
                if(args.Length!=0) throw new InvalidOperationException("Unexpected command line.");
                using(var form=new InstallerWindow(Execute)) Application.Run(form);
            }} finally {foreach(var h in executableDirectories)h.Dispose();}
            return 0;
        } catch(Exception error) {
            #if TESTING
            if(args.Length==1 && args[0]=="--self-test") {Console.Error.WriteLine(error);return 1;}
            #endif
            MessageBox.Show(InstallerText.Error(error) + "\n\n" + InstallerText.Choose("Technical details: ","技术详情：") + error.Message,InstallerText.Choose("Semantier setup","Semantier 安装程序"),MessageBoxButtons.OK,MessageBoxIcon.Error);return 1;
        }
    }
}
