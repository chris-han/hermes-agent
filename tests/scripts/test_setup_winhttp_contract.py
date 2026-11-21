"""Exercise the embedded Setup against native Framework and local fixtures."""
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest


@pytest.mark.platforms("windows")
def test_native_setup_download_contract_identity_and_resume(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    installer = repo / "win-native-installer"
    source = tmp_path / "Setup.cs"
    source.write_text("\n".join((installer / name).read_text(encoding="utf-8")
                               for name in ("Program.cs", "ReleaseMetadata.cs", "WinHttp.cs", "InstallerWindow.cs")),
                      encoding="utf-8")
    for name, version, publisher in [("old", "0.0.87.0", "Semantier Technology"),
                                     ("new", "9.9.99.0", "Semantier Technology"),
                                     ("wrong", "0.0.87.0", "Other")]:
        with zipfile.ZipFile(tmp_path / f"{name}.msix", "w") as archive:
            archive.writestr("AppxManifest.xml", f'<Package><Identity Name="Semantier.Agent.Acceptance" '
                             f'Publisher="CN={publisher}, O={publisher}" Version="{version}" '
                             'ProcessorArchitecture="x64"/></Package>')
            archive.writestr("AppxSignature.p7x", b"presence fixture; never installed")
    harness = tmp_path / "Tests.cs"
    harness.write_text(r'''
using System;
using System.IO;
using System.Linq;
using System.Text;
class SetupContractTests {
  [STAThread]
  static int Main(string[] args) {
    if(args.Length==4 && args[0]=="--preview"){
      System.Windows.Forms.Application.EnableVisualStyles();InstallerWindow.Preview(args[1],args[2],args[3]);return 0;
    }
    Program.LogPath=Path.Combine(args[0],"log.txt");
    using(var cert=Program.PublisherCertificate()){
      Program.ValidatePublisherCertificate(cert);
      if(cert.HasPrivateKey)throw new Exception("Private key must never ship");
    }
    bool wrongCertificate=false;
    using(var roots=new System.Security.Cryptography.X509Certificates.X509Store(System.Security.Cryptography.X509Certificates.StoreName.Root,System.Security.Cryptography.X509Certificates.StoreLocation.LocalMachine)){
      roots.Open(System.Security.Cryptography.X509Certificates.OpenFlags.ReadOnly);
      var other=roots.Certificates.Cast<System.Security.Cryptography.X509Certificates.X509Certificate2>().First(c=>c.Thumbprint!="57BF15739B4AA447842EAF388856A01B4E5029FF");
      try{Program.ValidatePublisherCertificate(other);}catch(InvalidDataException){wrongCertificate=true;}
    }
    if(!wrongCertificate)throw new Exception("Unpinned certificate accepted");
    InstallerWindow.SelfTest();
    if(Program.MetadataUrl!="https://cloud.semantier.cn/releases/channels/semantier-acceptance.json")throw new Exception("Pinned or wrong URL");
    Program.VerifyIdentity(Path.Combine(args[0],"old.msix"));
    Program.VerifyIdentity(Path.Combine(args[0],"new.msix"));
    bool rejected=false;try{Program.VerifyIdentity(Path.Combine(args[0],"wrong.msix"));}catch(InvalidDataException){rejected=true;}
    if(!rejected)throw new Exception("Wrong publisher accepted");
    string local=Path.Combine(args[0],"local packages");Directory.CreateDirectory(local);
    bool missing=false;try{Program.ResolveLocalPackage(local);}catch(FileNotFoundException){missing=true;}
    if(!missing)throw new Exception("Missing adjacent package accepted");
    string adjacent=Path.Combine(local,"Semantier Agent.MSIX");File.Copy(Path.Combine(args[0],"old.msix"),adjacent);
    File.WriteAllText(Path.Combine(local,"ignore.msix.sha256"),"sidecar");
    if(Program.ResolveLocalPackage(local)!=adjacent)throw new Exception("Adjacent package resolution failed");
    File.Copy(Path.Combine(args[0],"new.msix"),Path.Combine(local,"another.msix"));
    bool ambiguous=false;try{Program.ResolveLocalPackage(local);}catch(InvalidDataException){ambiguous=true;}
    if(!ambiguous)throw new Exception("Ambiguous adjacent packages accepted");
    string standard=Path.Combine(local,"SemantierAgent.msix");File.Copy(Path.Combine(args[0],"old.msix"),standard);
    File.WriteAllText(Path.Combine(local,"unsigned.msix"),"not signed");
    if(Program.ResolveLocalPackage(local)!=standard)throw new Exception("Standard signed-package path did not win over build intermediates");
    Program.VerifyIdentity(Program.ResolveLocalPackage(local));
    var requests=new Requests();
    using(var output=new MemoryStream()){
      BootstrapperWinHttp.Download(requests,new Uri("https://assets.semantier.cn/releases/channel-builds/"+new string('a',32)+"/SemantierAgent.msix"),output,6,total=>{});
      if(Encoding.ASCII.GetString(output.ToArray())!="abcdef" || requests.Calls!=3)throw new Exception("Resume or fallback failed");
    }
    using(var cancelled=new System.Threading.CancellationTokenSource()){
      cancelled.Cancel();bool stopped=false;var noRequests=new Requests();
      try {using(var output=new MemoryStream())BootstrapperWinHttp.Download(noRequests,new Uri("https://assets.semantier.cn/releases/a.msix"),output,6,x=>{},cancelled.Token);}
      catch(OperationCanceledException){stopped=true;}
      if(!stopped || noRequests.Calls!=0)throw new Exception("Cancelled download must not open a network request");
    }
    using(var midTransfer=new System.Threading.CancellationTokenSource()){
      bool stopped=false;var requests2=new Requests();
      try {using(var output=new MemoryStream())BootstrapperWinHttp.Download(requests2,new Uri("https://assets.semantier.cn/releases/a.msix"),output,6,x=>{midTransfer.Cancel();},midTransfer.Token);}
      catch(OperationCanceledException){stopped=true;}
      if(!stopped || requests2.Calls!=2)throw new Exception("Mid-transfer cancellation must not retry or resume");
    }
    if(typeof(Program).Assembly.GetReferencedAssemblies().Any(a=>a.Name=="System.Net.Http" || a.Name=="System.Private.CoreLib"))throw new Exception("Unexpected runtime dependency");
    Console.WriteLine("NATIVE_SETUP_CONTRACT_PASS");return 0;
  }
  class Requests:IWinHttpRequests {
    public int Calls;
    public IWinHttpResponse Open(Uri uri,long offset,bool h3){
      Calls++;
      if(Calls==1){if(!h3)throw new Exception("H3 not preferred");throw new IOException("H3 unavailable");}
      if(Calls==2){if(h3 || offset!=0)throw new Exception("H2 fallback failed");return new Response(true);}
      if(offset!=3)throw new Exception("Wrong resume offset");return new Response(false);
    }
  }
  class Response:IWinHttpResponse {
    readonly bool first;public Response(bool f){first=f;Body=f?(Stream)new Partial():new MemoryStream(Encoding.ASCII.GetBytes("def"));}
    public int StatusCode{get{return first?200:206;}}public long ContentLength{get{return first?6:3;}}
    public string ContentRange{get{return first?null:"bytes 3-5/6";}}public int Protocol{get{return 2;}}
    public Stream Body{get;private set;}public void Dispose(){Body.Dispose();}
  }
  class Partial:MemoryStream {
    public Partial():base(Encoding.ASCII.GetBytes("abc")){}
    public override int Read(byte[] b,int o,int n){if(Position==Length)throw new IOException("interrupted");return base.Read(b,o,n);}
  }
}
''', encoding="utf-8")
    compiler = Path(os.environ["WINDIR"]) / "Microsoft.NET/Framework64/v4.0.30319/csc.exe"
    executable = tmp_path / "Tests.exe"
    references = ["System.Windows.Forms", "System.Drawing", "System.Xml.Linq", "System.Runtime.Serialization",
                  "System.IO.Compression", "System.IO.Compression.FileSystem"]
    subprocess.run([str(compiler), "/nologo", "/target:exe", "/platform:x64", "/define:TESTING", "/main:SetupContractTests",
                    f"/resource:{repo / 'apps/desktop/assets/icon.ico'},Semantier.icon.ico",
                    f"/resource:{installer / 'SemantierPublisher.cer'},Semantier.publisher.cer",
                    f"/resource:{repo / 'apps/desktop/assets/icon.png'},Semantier.logo.png",
                    f"/out:{executable}", *[f"/reference:{ref}.dll" for ref in references],
                    str(source), str(harness)], check=True, timeout=30)
    result = subprocess.run([str(executable), str(tmp_path)], capture_output=True,
                            text=True, check=False, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NATIVE_SETUP_CONTRACT_PASS" in result.stdout
    for language in ("en", "zh-CN"):
        subprocess.run([str(executable), "--preview", language, "ready",
                        str(tmp_path / f"asimov-{language}.png")], check=True, timeout=30)
        assert (tmp_path / f"asimov-{language}.png").is_file()

    local_executable = tmp_path / "LocalSetup.exe"
    subprocess.run([str(compiler), "/nologo", "/target:winexe", "/platform:x64",
                    "/define:LOCAL_SETUP", f"/out:{local_executable}",
                    *[f"/reference:{ref}.dll" for ref in references], str(source)],
                   check=True, timeout=30)
    contract = tmp_path / "local-contract.txt"
    subprocess.run([str(local_executable), "--describe-fixed-download", str(contract)],
                   check=True, timeout=30)
    assert contract.read_text() == "local-adjacent-single-msix"
    rejected = subprocess.run([str(local_executable), "--probe-h3"], check=False, timeout=30)
    assert rejected.returncode == 2


@pytest.mark.platforms("windows")
def test_private_oss_staging_transport_contract(tmp_path):
    repo=Path(__file__).resolve().parents[2]
    source="using System;using System.IO;using System.ComponentModel;using System.Globalization;using System.Runtime.InteropServices;\n"
    source+=(repo/'win-native-installer/WinHttp.cs').read_text()
    source+=(repo/'tests/scripts/oss-staging-download-contract.cs').read_text()
    source+='class Entry {static int Main(){Console.WriteLine(OssStagingContractTests.Run());return 0;}}'
    path=tmp_path/'Test.cs';path.write_text(source)
    compiler=Path(os.environ['WINDIR'])/'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
    exe=tmp_path/'Test.exe'
    subprocess.run([str(compiler),'/nologo','/target:exe','/out:'+str(exe),str(path)],check=True,capture_output=True)
    result=subprocess.run([str(exe)],check=True,capture_output=True,text=True)
    assert result.stdout.startswith('PASS:')

@pytest.mark.parametrize("cancel_after_first_chunk", [False, True])
def test_download_cancellation_cross_platform(tmp_path, cancel_after_first_chunk):
    """Compile/run the same WinHTTP transfer core with a fake stream; no real network."""
    pwsh = shutil.which("pwsh")
    if not pwsh:
        pytest.skip("PowerShell C# compiler unavailable")
    source_file = Path(__file__).resolve().parents[2] / "win-native-installer" / "WinHttp.cs"
    harness = r'''
using System;
using System.IO;
using System.ComponentModel;
using System.Globalization;
using System.Runtime.InteropServices;
public class CancelHarness {
  public static string Run(bool afterChunk) {
    using(var cts=new System.Threading.CancellationTokenSource()){
      var network=new FakeNetwork();
      if(!afterChunk)cts.Cancel();
      bool stopped=false;
      try {
        using(var output=new MemoryStream()){
          BootstrapperWinHttp.Download(network,new Uri("https://assets.semantier.cn/releases/test.msix"),output,3,
            done=>{if(afterChunk)cts.Cancel();},cts.Token);
        }
      } catch(OperationCanceledException){stopped=true;}
      if(!stopped || network.Calls!=(afterChunk?1:0))
        throw new Exception("Cancellation failed to prevent network retry or continuation.");
      return "CANCEL_OK";
    }
  }
  sealed class FakeNetwork:IWinHttpRequests {
    internal int Calls;
    public IWinHttpResponse Open(Uri uri,long offset,bool http3){Calls++;return new FakeResponse();}
  }
  sealed class FakeResponse:IWinHttpResponse {
    public int StatusCode {get{return 200;}}
    public long ContentLength {get{return 3;}}
    public string ContentRange {get{return null;}}
    public int Protocol {get{return 2;}}
    public Stream Body {get;private set;}
    public FakeResponse(){Body=new MemoryStream(new byte[]{1,2,3});}
    public void Dispose(){Body.Dispose();}
  }
}
'''
    combined = tmp_path / "CancelHarness.cs"
    combined.write_text(harness + source_file.read_text(encoding="utf-8"), encoding="utf-8")
    runner = tmp_path / "run.ps1"
    runner.write_text(
        "$ErrorActionPreference='Stop'\n"
        "$code=Get-Content -LiteralPath $args[0] -Raw\n"
        "Add-Type -TypeDefinition $code -IgnoreWarnings -WarningAction SilentlyContinue -ErrorAction Stop\n"
        "[CancelHarness]::Run([bool]::Parse($args[1]))\n", encoding="utf-8")
    result = subprocess.run([pwsh, "-NoProfile", "-NonInteractive", "-File", str(runner),
                             str(combined), str(cancel_after_first_chunk)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CANCEL_OK" in result.stdout

