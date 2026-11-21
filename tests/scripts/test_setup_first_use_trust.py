"""Run real Setup orchestration; substitute only external trust/network/deployment."""
import hashlib
import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.platforms("windows")
def test_first_use_trust_precedes_transfer_and_failure_retains_package(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    installer = repo / "win-native-installer"
    program = (installer / "Program.cs").read_text(encoding="utf-8")
    # Permit the pre-fix entrypoint as a negative control with identical orchestration.
    if "internal static void RunOnlineInstall(" not in program:
        start = program.index('        string directory=TemporaryDirectory();')
        end = program.index('\n#endif', start)
        program += '\n'  # Method inserted within Program below.
        body = program[start:end]
        program = program[:start] + '        RunOnlineInstall(report);' + program[end:]
        pos = program.index('  internal static string ResolveLocalPackage(')
        program = program[:pos] + '  internal static void RunOnlineInstall(Action<string> report){\n' + body + '\n  }\n' + program[pos:]
    byte_preparation = (
        'PreparePackageBytes(SetupTrustTests.Package,34,"'+hashlib.sha256(b"bounded downloaded package fixture").hexdigest()+'",reuseVerified,()=>{SetupTrustTests.Transfers++;File.WriteAllText(SetupTrustTests.Package,"bounded downloaded package fixture");},report);'
        if "internal static void PreparePackageBytes(" in program else
        'if(!File.Exists(SetupTrustTests.Package)){SetupTrustTests.Transfers++;File.WriteAllText(SetupTrustTests.Package,"bounded downloaded package fixture");}'
    )
    start = program.index('  internal static string DownloadPackage(')
    end = program.index('  internal static void VerifyIdentity(', start)
    program = program[:start] + '''
  internal static string DownloadPackage(string directory,Action<long,long> progress,string expected,Action<string> report=null,string candidateBuild=null,string candidateManifestHash=null,long candidateSequence=0,bool reuseVerified=false){
    if(!SetupTrustTests.Trusted)throw new InvalidDataException("Untrusted publisher before transfer");
    SetupTrustTests.Package=Path.Combine(directory,"SemantierAgent.msix");
    BYTE_PREPARATION
    SetupTrustTests.SignatureChecks++;
    if(SetupTrustTests.FailSignature)throw new InvalidDataException("Signature failure fixture");
    return SetupTrustTests.Package;
  }
'''.replace('BYTE_PREPARATION',byte_preparation) + program[end:]
    start = program.index('  static void EnsurePublisherTrust(')
    end = program.index('  internal static void Install(', start)
    program = program[:start] + '''
  static void EnsurePublisherTrust(Action<string> report){
    if(SetupTrustTests.Decline)throw new IOException("Consent declined");
    SetupTrustTests.Trusted=true;
  }
''' + program[end:]
    start = program.index('  internal static void Install(')
    program = program[:start] + '''
  internal static void Install(string file,Action<string> report=null){
    if(!SetupTrustTests.Trusted)throw new Exception("Missing trust");
    SetupTrustTests.Installs++;
    if(SetupTrustTests.FailInstall)throw new IOException("Deployment failure fixture");
  }
}
'''
    harness = r'''
class SetupTrustTests {
  internal static bool Trusted,Decline,FailInstall,FailSignature;
  internal static int Transfers,Installs,SignatureChecks;
  internal static string Package;
  static int Main(string[] args){
    Program.LogPath=System.IO.Path.Combine(args[0],"test.log");
    // Fresh host: trust must precede the very first transfer.
    Program.RunOnlineInstall(s=>{});
    if(!Trusted || Transfers!=1 || Installs!=1 || System.IO.File.Exists(Package))throw new System.Exception("Fresh install or successful cleanup failed");
    Trusted=false;Decline=true;
    try{Program.RunOnlineInstall(s=>{});throw new System.Exception("Consent bypassed");}catch(System.IO.IOException){}
    if(Transfers!=1 || Installs!=1)throw new System.Exception("Declined consent performed network/deployment");
    Decline=false;FailSignature=true;
    for(int attempt=0;attempt<2;attempt++){
      try{Program.RunOnlineInstall(s=>{});throw new System.Exception("Signature failure hidden");}catch(System.IO.InvalidDataException){}
      if(Transfers!=2)throw new System.Exception("Retry redownloaded an integrity-verified package");
    }
    if(SignatureChecks!=3)throw new System.Exception("Retry skipped signature verification");
    FailSignature=false;FailInstall=true;
    try{Program.RunOnlineInstall(s=>{});throw new System.Exception("Deployment failure hidden");}catch(System.IO.IOException){}
    if(Transfers!=2 || !System.IO.File.Exists(Package) || !System.IO.File.ReadAllText(Program.LogPath).Contains(System.IO.Path.GetDirectoryName(Package)))throw new System.Exception("Failure recovery package/path lost");
    FailInstall=false;
    Program.RunOnlineInstall(s=>{});
    if(Transfers!=2 || System.IO.File.Exists(Package))throw new System.Exception("Installation retry redownloaded or cleanup failed");
    System.Console.WriteLine("FIRST_USE_TRUST_ORDER_AND_RECOVERY_PASS");return 0;
  }
}
'''
    if "internal static void PreparePackageBytes(" in program:
        harness = harness.replace('    System.Console.WriteLine("FIRST_USE_TRUST_ORDER_AND_RECOVERY_PASS");', r'''
    string file=System.IO.Path.Combine(args[0],"byte-cache.msix");
    string content="verified fixture";
    string digest;
    using(var sha=System.Security.Cryptography.SHA256.Create())digest=System.BitConverter.ToString(sha.ComputeHash(System.Text.Encoding.UTF8.GetBytes(content))).Replace("-","").ToLowerInvariant();
    int requests=0,reuses=0;
    System.Action fetch=()=>{requests++;System.IO.File.WriteAllText(file,content);};
    System.Action<string> report=s=>{if(s=="Reusing verified download…")reuses++;};
    Program.PreparePackageBytes(file,content.Length,digest,true,fetch,report);
    Program.PreparePackageBytes(file,content.Length,digest,true,fetch,report);
    if(requests!=1 || reuses!=1)throw new System.Exception("Verified retry requested package bytes");
    System.IO.File.WriteAllText(file,new string('x',content.Length));
    Program.PreparePackageBytes(file,content.Length,digest,true,fetch,report);
    if(requests!=2)throw new System.Exception("Same-length corruption reused");
    System.IO.File.WriteAllText(file,"partial");
    Program.PreparePackageBytes(file,content.Length,digest,true,fetch,report);
    if(requests!=3)throw new System.Exception("Partial download reused");
    string other="another revision";
    string otherDigest;
    using(var sha=System.Security.Cryptography.SHA256.Create())otherDigest=System.BitConverter.ToString(sha.ComputeHash(System.Text.Encoding.UTF8.GetBytes(other))).Replace("-","").ToLowerInvariant();
    Program.PreparePackageBytes(file,other.Length,otherDigest,true,()=>{requests++;System.IO.File.WriteAllText(file,other);},report);
    if(requests!=4 || System.IO.File.ReadAllText(file)!=other)throw new System.Exception("Different selected bytes reused");
    bool refused=false;
    try{Program.PreparePackageBytes(file,other.Length,otherDigest,false,()=>{using(var output=new System.IO.FileStream(file,System.IO.FileMode.CreateNew)){}},report);}catch(System.IO.IOException){refused=true;}
    if(!refused || System.IO.File.ReadAllText(file)!=other)throw new System.Exception("Qualification overwrote retained evidence");
    refused=false;
    try{Program.PreparePackageBytes(file,other.Length,otherDigest,true,()=>{},report);}catch(System.IO.InvalidDataException){refused=true;}
    if(refused)throw new System.Exception("Matching cache unexpectedly refused");
    System.IO.File.Delete(file);
    try{Program.PreparePackageBytes(file,other.Length,otherDigest,true,()=>System.IO.File.WriteAllText(file,"bad"),report);}catch(System.IO.InvalidDataException){refused=true;}
    if(!refused)throw new System.Exception("Corrupt new download accepted");
    System.IO.File.Delete(file);
    System.Console.WriteLine("FIRST_USE_TRUST_ORDER_AND_RECOVERY_PASS");''')
    source = tmp_path / "Tests.cs"
    source.write_text(program + "\n" + "\n".join(
        (installer / name).read_text(encoding="utf-8")
        for name in ("ReleaseMetadata.cs", "WinHttp.cs", "InstallerWindow.cs")) + harness,
        encoding="utf-8")
    compiler = Path(os.environ["WINDIR"]) / "Microsoft.NET/Framework64/v4.0.30319/csc.exe"
    executable = tmp_path / "Tests.exe"
    references = ["System.Windows.Forms", "System.Drawing", "System.Xml.Linq",
                  "System.Runtime.Serialization", "System.IO.Compression", "System.IO.Compression.FileSystem"]
    subprocess.run([str(compiler), "/nologo", "/target:exe", "/platform:x64",
                    "/main:SetupTrustTests", f"/out:{executable}",
                    *[f"/reference:{ref}.dll" for ref in references], str(source)],
                   check=True, capture_output=True, text=True, timeout=30)
    result = subprocess.run([str(executable), str(tmp_path)], capture_output=True,
                            text=True, check=False, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FIRST_USE_TRUST_ORDER_AND_RECOVERY_PASS" in result.stdout
