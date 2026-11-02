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
[assembly: AssemblyProduct("Semantier Agent Installer")]
[assembly: AssemblyCompany("Semantier Technology")]
[assembly: AssemblyVersion("1.2.0.0")]
[assembly: AssemblyFileVersion("1.2.0.0")]
#elif CHANNEL2
[assembly: AssemblyTitle("Semantier Agent 0.0.77 Channel 2 Installer")]
[assembly: AssemblyProduct("Semantier Agent Installer")]
[assembly: AssemblyCompany("Semantier Technology")]
[assembly: AssemblyVersion("1.3.0.0")]
[assembly: AssemblyFileVersion("1.3.0.0")]
#else
[assembly: AssemblyTitle("Semantier Agent 0.0.64 Installer")]
[assembly: AssemblyProduct("Semantier Agent Installer")]
[assembly: AssemblyCompany("Semantier Technology")]
[assembly: AssemblyVersion("1.1.1.0")]
[assembly: AssemblyFileVersion("1.1.1.0")]

#endif

// Independent, version-specific bootstrapper. It never modifies the release ZIP.
static class Installer {
    #if CHANNEL1
    const string Origin = Channel1Pins.Origin;
    const string ZipUrl = Channel1Pins.ZipUrl;
    const string ZipHash = Channel1Pins.ZipHash;
    const long ZipSize = Channel1Pins.ZipSize;
    const string CertHash = Channel1Pins.CertHash;
    const string Thumbprint = Channel1Pins.Thumbprint;
    const string GraphHash = Channel1Pins.GraphHash;
    const string Identity = Channel1Pins.Identity;
    static readonly Version Version = new Version(Channel1Pins.AppVersion);
    #elif CHANNEL2
    const string Origin = "https://s3.semantier.com";
    const string ZipUrl = Origin + "/releases/channel-builds/9e1ca1e9fad44d35ac9b13a2f09747e5/SemantierAgent-0.0.77-win-x64.zip";
    const string ZipHash = "59ad5f557babc3905835f239c502ea44ad5aa43207603ffff8cbcb757998c229";
    const long ZipSize = 2264240015L;
    const string CertHash = "10a6b55e9a03bbc0dceb8d899c1776bf1fbad963c82050c16534da4983c9661d";
    const string Thumbprint = "57BF15739B4AA447842EAF388856A01B4E5029FF";
    const string BundleHash = "2ad23b092ef7e17fc7517922ab22198902d73fa7a2c52978ef910caf07b33db0";
    const string Identity = "Semantier.Agent.Acceptance";
    static readonly Version Version = new Version(0,0,77,0);
    #else
    const string Origin = "https://s3.semantier.com";
    const string ZipUrl = Origin + "/releases/channel-builds/3d722e4ff1c44b48b908d1ebb606453a/SemantierAgent-0.0.64-win-x64.zip";
    const string ZipHash = "b970fd1b60fd181d8ae29d3eb88f48d217279ef3b55f2c454525683ea3d81e9e";
    const long ZipSize = 2261279211L;
    const string CertHash = "10a6b55e9a03bbc0dceb8d899c1776bf1fbad963c82050c16534da4983c9661d";
    const string Thumbprint = "57BF15739B4AA447842EAF388856A01B4E5029FF";
    const string GraphHash = "cfbf8c7582ecd2818f0c094acbc3177364bb61c74b1c0185ab0585e5eb3ca6d5";
    const string Identity = "Semantier.Agent.Acceptance";
    static readonly Version Version = new Version(0,0,64,0);
    #endif
    internal static string AppVersionString { get { return Version.ToString(); } }
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
    #if CHANNEL1
    static readonly Dictionary<string,string> Expected = Channel1Pins.Expected;
    #elif CHANNEL2
    static readonly Dictionary<string,string> Expected = new Dictionary<string,string>(StringComparer.Ordinal) {
        {"SemantierAgent-signing.cer",CertHash},
        {"SemantierAgent-0.0.77-win-x64.msix","3c0fd642db1b35fa637c356cdd01871e057efcdd54e0b9b94646985d8e7c14a4"},
    };
    #else
    static readonly Dictionary<string,string> Expected = new Dictionary<string,string>(StringComparer.Ordinal) {
        {"native-graph.json",GraphHash}, {"SemantierAgent-signing.cer",CertHash},
        {"SemantierAgent.msixbundle","8fc4d74671d3ffeaf63d7c7a6fbb424e9ba6c14cd31522e7fe09e206e9748604"},
        {"SemantierAgent-main.msix","fdcd1d25ce7ed42ae7267dc19139161f52a9bd3f4b2626b7e5d1d3662fb1f8d7"},
        {"SemantierAgent-runtime0004.msix","5fbd51528bd462d823bef843f65ac60ddfcdf76b7d433ed896ce1086ef5cc5fd"},
        {"SemantierAgent-runtime0003.msix","b3aec2d7dc3dbf066172521527eddff8d01111d98c76062837109a9461f0fa88"},
        {"SemantierAgent-runtime0001.msix","543383d3914c3819178ba718861fd736bbb26908c736240b54b34b75f42dc74a"},
        {"SemantierAgent-runtime0002.msix","acb5bed5a17e01c4a444ade2e3c97243ab01523a3fcebbbe708d198b05723181"},
    };
    #endif
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
    static bool Trusted(X509Certificate2 cert) {
        using(var store=new X509Store(StoreName.TrustedPeople,StoreLocation.LocalMachine)) {
            store.Open(OpenFlags.ReadOnly); foreach(var current in store.Certificates) if(current.Thumbprint==Thumbprint && current.RawData.Length==cert.RawData.Length) {
                using(var s=new MemoryStream(current.RawData)) if(Hash(s)==CertHash) return true;
            }
        }
        return false;
    }
    static string TaskBase() {return Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.CommonDocuments),"SemantierInstaller");}
    static string TaskPath(string id) {
        Guid parsed;
        if(!Guid.TryParseExact(id,"N",out parsed) || parsed.ToString("N")!=id) throw new InvalidOperationException("Invalid task identity.");
        return Path.Combine(TaskBase(),id);
    }
    static List<FileStream> HoldAndVerify(string dir) {
        var held=new List<FileStream>();
        try {
            var zip=new FileStream(Path.Combine(dir,"release.zip"),FileMode.Open,FileAccess.Read,FileShare.Read);held.Add(zip);
            if(zip.Length!=ZipSize || Hash(zip)!=ZipHash) throw new InvalidOperationException("Release ZIP mismatch.");
            foreach(var item in Expected) {
                var file=new FileStream(Path.Combine(dir,item.Key),FileMode.Open,FileAccess.Read,FileShare.Read);held.Add(file);
                if(Hash(file)!=item.Value) throw new InvalidOperationException("Release member mismatch: "+item.Key);
            }
            #if CHANNEL2
            var bundle=new FileStream(Path.Combine(dir,"SemantierAgent.msixbundle"),FileMode.Open,FileAccess.Read,FileShare.Read);held.Add(bundle);
            if(Hash(bundle)!=BundleHash) throw new InvalidOperationException("Bundle mismatch.");
            #endif
            return held;
        } catch {foreach(var h in held)h.Dispose();throw;}
    }
    static void Trust(string id) {
        string dir=TaskPath(id);var directories=HoldDirectories(dir);var files=new List<FileStream>();
        try {
            files=HoldAndVerify(dir); VerifySignatures(dir,false);
            using(var cert=Certificate()) {
                if(MessageBox.Show(InstallerText.Choose("Trust the verified Semantier Technology publisher certificate on this device?","是否在此设备上信任已验证的 Semantier Technology 发布者证书？") + "\n\n" + Thumbprint + "\nLocalMachine / TrustedPeople", InstallerText.Choose("Confirm publisher trust","确认发布者信任"),MessageBoxButtons.YesNo,MessageBoxIcon.Information)!=DialogResult.Yes) throw new OperationCanceledException("Certificate trust cancelled.");
                using(var store=new X509Store(StoreName.TrustedPeople,StoreLocation.LocalMachine)) {store.Open(OpenFlags.ReadWrite);if(!Trusted(cert))store.Add(cert);}
                if(!Trusted(cert))throw new InvalidOperationException("Certificate trust was not confirmed.");
                VerifySignatures(dir,true);
            }
        } finally {foreach(var h in files)h.Dispose();foreach(var h in directories)h.Dispose();}
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
    static FileStream Download(string destination, Action<string> report,List<FileStream> held) {
        ServicePointManager.SecurityProtocol=SecurityProtocolType.Tls12;
        var request=(HttpWebRequest)WebRequest.Create(ZipUrl); request.AllowAutoRedirect=false; request.Timeout=60000; request.ReadWriteTimeout=120000;
        using(var response=(HttpWebResponse)request.GetResponse()) {
            if(response.StatusCode!=HttpStatusCode.OK || response.ResponseUri.AbsoluteUri!=ZipUrl || response.ContentLength!=ZipSize) throw new InvalidOperationException("Unexpected download response.");
            var target=CreateOwned(destination,held);
            using(var source=response.GetResponseStream()) {
                var bytes=new byte[1024*1024]; long total=0; int n; int previous=-1;
                while((n=source.Read(bytes,0,bytes.Length))>0) { target.Write(bytes,0,n); total+=n; if(total>ZipSize) throw new InvalidOperationException("Download exceeds expected size."); int percent=(int)(total*100/ZipSize); if(percent!=previous){report("Downloading " + percent + "%"); previous=percent;} }
                if(total!=ZipSize) throw new InvalidOperationException("Download incomplete."); target.Flush(true);
            }
            return Seal(target,held);
        }
    }
    static string PrivateDirectory(string id) {
        var sid=WindowsIdentity.GetCurrent().User;
        var security=new DirectorySecurity(); security.SetOwner(sid); security.SetAccessRuleProtection(true,false);
        security.AddAccessRule(new FileSystemAccessRule(sid,FileSystemRights.FullControl,InheritanceFlags.ContainerInherit|InheritanceFlags.ObjectInherit,PropagationFlags.None,AccessControlType.Allow));
        security.AddAccessRule(new FileSystemAccessRule(new SecurityIdentifier(WellKnownSidType.BuiltinAdministratorsSid,null),FileSystemRights.ReadAndExecute,InheritanceFlags.ContainerInherit|InheritanceFlags.ObjectInherit,PropagationFlags.None,AccessControlType.Allow));
        string p=TaskPath(id); if(Directory.Exists(p))throw new InvalidOperationException("Task directory already exists.");Directory.CreateDirectory(p,security); return p;
    }
    static void ExtractAndHold(FileStream archive,string dir,List<FileStream> held) {
        var seen=new HashSet<string>(StringComparer.Ordinal);
            using(var z=new ZipArchive(archive,ZipArchiveMode.Read,true)) foreach(var entry in z.Entries) {
                if(!Expected.ContainsKey(entry.FullName)) continue;
                if(!seen.Add(entry.FullName) || entry.FullName!=Path.GetFileName(entry.FullName) || ((entry.ExternalAttributes>>16)&0xF000)==0xA000) throw new InvalidOperationException("Unsafe archive member.");
                string path=Path.Combine(dir,entry.FullName);
                var output=CreateOwned(path,held);
                using(var input=entry.Open()) { input.CopyTo(output); output.Flush(true); }
                var pinned=Seal(output,held);
                if(Hash(pinned)!=Expected[entry.FullName]) throw new InvalidOperationException("Release member SHA-256 mismatch: "+entry.FullName);
            }
            if(seen.Count!=Expected.Count) throw new InvalidOperationException("Release archive is incomplete.");
    }
    static void VerifySignatures(string dir,bool trusted) {
        string encoded=Convert.ToBase64String(Encoding.Unicode.GetBytes(dir));
        string script="$d=[Text.Encoding]::Unicode.GetString([Convert]::FromBase64String('"+encoded+"'));"+
            #if CHANNEL2
            "$names=@('SemantierAgent.msixbundle','SemantierAgent-0.0.77-win-x64.msix');"+
            #else
            "$names=@('SemantierAgent.msixbundle','SemantierAgent-main.msix','SemantierAgent-runtime0001.msix','SemantierAgent-runtime0002.msix','SemantierAgent-runtime0003.msix','SemantierAgent-runtime0004.msix');"+
            #endif
            "foreach($n in $names){$s=Get-AuthenticodeSignature -LiteralPath (Join-Path $d $n);if(!$s.SignerCertificate -or $s.SignerCertificate.Thumbprint -ne '"+Thumbprint+"' -or $s.Status -notin @("+(trusted?"'Valid'":"'Valid','NotTrusted','UnknownError'")+")){throw 'Package signature mismatch'}}";
        PowerShell(script);
    }
    static void Install(string dir) {
        string encoded=Convert.ToBase64String(Encoding.Unicode.GetBytes(Path.Combine(dir,"SemantierAgent.msixbundle")));
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
        CheckInstalled(); string id=Guid.NewGuid().ToString("N"), dir=null; bool installed=false; var held=new List<FileStream>(); var directories=new List<SafeFileHandle>();
        try {
            Directory.CreateDirectory(TaskBase()); directories=HoldDirectories(TaskBase());
            dir=PrivateDirectory(id);directories.AddRange(HoldDirectories(dir));
            string archive=Path.Combine(dir,"release.zip"); report("Downloading signed release…"); var zip=Download(archive,report,held);
            if(zip.Length!=ZipSize || Hash(zip)!=ZipHash) throw new InvalidOperationException("Download SHA-256 mismatch.");zip.Position=0;
            report("Verifying package and certificate…"); ExtractAndHold(zip,dir,held);
            #if CHANNEL2
            using(var embedded=Assembly.GetExecutingAssembly().GetManifestResourceStream("Semantier.channel2.msixbundle")) {
                if(embedded==null) throw new InvalidOperationException("Missing signed bundle.");
                var output=CreateOwned(Path.Combine(dir,"SemantierAgent.msixbundle"),held);
                embedded.CopyTo(output);output.Flush(true);var sealedBundle=Seal(output,held);
                if(Hash(sealedBundle)!=BundleHash) throw new InvalidOperationException("Bundle hash mismatch.");
            }
            #endif
            VerifySignatures(dir,false);
            using(var cert=Certificate()) {
                if(!Trusted(cert)) {
                    var consent=MessageBox.Show(InstallerText.Choose("Windows will request administrator permission to trust the verified Semantier Technology publisher certificate. The application will be installed for your current Windows account. Continue?","Windows 将请求管理员权限，以信任已验证的 Semantier Technology 发布者证书。应用仍将安装到您当前的 Windows 账户。是否继续？"), InstallerText.Choose("Publisher certificate","发布者证书"),MessageBoxButtons.YesNo,MessageBoxIcon.Information);
                    if(consent!=DialogResult.Yes) throw new OperationCanceledException("Certificate consent cancelled.");
                    report("Waiting for Windows administrator consent…");
                    // The administrator may be a different account and unable
                    // to read the initiating user's Downloads directory. Copy
                    // this already-held executable into the task ACL, then pin
                    // its exact bytes and path through elevation.
                    string helper=Path.Combine(dir,"SemantierTrust.exe");
                    var target=CreateOwned(helper,held);
                    using(var source=File.OpenRead(Assembly.GetExecutingAssembly().Location)) {source.CopyTo(target);target.Flush(true);}
                    var helperHold=Seal(target,held);
                    string helperHash=Hash(helperHold);using(var source=File.OpenRead(Assembly.GetExecutingAssembly().Location))if(Hash(source)!=helperHash)throw new InvalidOperationException("Trust helper changed.");
                    var info=new ProcessStartInfo(helper,"--trust "+id+" --language "+InstallerText.Language) {UseShellExecute=true,Verb="runas"};
                    using(var child=Process.Start(info)) { if(child==null) throw new InvalidOperationException("Elevation was not started."); child.WaitForExit(); if(child.ExitCode!=0 || !Trusted(cert)) throw new InvalidOperationException("Certificate trust was not completed."); }
                }
            }
            VerifySignatures(dir,true); CheckInstalled(); report("Installing for the current Windows user…"); Install(dir); installed=true;
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
        // More than a pipe buffer on stderr must not prevent stdout completion.
        var noisyChild=System.Threading.Tasks.Task.Run(()=>PowerShell("[Console]::Error.Write(('x' * 262144)); Write-Output 'streams-drained'"));
        if(!noisyChild.Wait(15000) || noisyChild.Result!="streams-drained")
            throw new InvalidOperationException("PowerShell output streams deadlocked or lost stdout.");
        bool childFailure=false;
        try { PowerShell("[Console]::Error.Write(('x' * 262144)); throw 'expected-stream-test-error'"); }
        catch(InvalidOperationException error) { childFailure=error.Message.Contains("expected-stream-test-error"); }
        if(!childFailure)throw new InvalidOperationException("PowerShell failure diagnostics were lost.");
        using(var cert=Certificate()) if(cert.HasPrivateKey) throw new InvalidOperationException("Unexpected private key.");
        string dir=Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location),"fixture-"+Guid.NewGuid().ToString("N"));
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
            held.Clear();string brokenPath=Path.Combine(dir,"broken.zip");var broken=CreateOwned(brokenPath,held);
            using(var z=new ZipArchive(broken,ZipArchiveMode.Create,true))using(var w=new StreamWriter(z.CreateEntry(Expected.Keys.First()).Open()))w.Write("incorrect member");
            broken.Flush(true);broken=Seal(broken,held);broken.Position=0;bool refused=false;
            try{ExtractAndHold(broken,dir,held);}catch(InvalidOperationException){refused=true;}
            if(!refused || held.Count!=2)throw new InvalidOperationException("Corrupt extraction was not tracked.");
            foreach(var item in held){var file=item;string name=file.Name;if(!DeleteHeld(name,file.SafeFileHandle,false,()=>file.Dispose()))throw new InvalidOperationException("Interrupted extraction cleanup failed.");}
        } finally {foreach(var h in locks)h.Dispose();Directory.Delete(dir,true);}
    }
    #endif
    [STAThread] static int Main(string[] args) {
        Application.EnableVisualStyles(); Application.SetCompatibleTextRenderingDefault(false);
        try {
            var executableDirectories=HoldDirectories(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location));
            try {using(var self=new FileStream(Assembly.GetExecutingAssembly().Location,FileMode.Open,FileAccess.Read,FileShare.Read)) {
                if(args.Length==4 && args[0]=="--trust" && args[2]=="--language") {InstallerText.SetLanguage(args[3]); Trust(args[1]); return 0;}
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
