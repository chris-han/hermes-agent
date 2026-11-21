"""Run real Setup orchestration; substitute only external trust/network/deployment."""
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
    start = program.index('  internal static string DownloadPackage(')
    end = program.index('  internal static void VerifyIdentity(', start)
    program = program[:start] + '''
  internal static string DownloadPackage(string directory,Action<long,long> progress,string expected,Action<string> report=null,string candidateBuild=null,string candidateManifestHash=null,long candidateSequence=0){
    SetupTrustTests.Transfers++;
    if(!SetupTrustTests.Trusted)throw new InvalidDataException("Untrusted publisher before transfer");
    SetupTrustTests.Package=Path.Combine(directory,"SemantierAgent.msix");
    File.WriteAllText(SetupTrustTests.Package,"bounded downloaded package fixture");
    return SetupTrustTests.Package;
  }
''' + program[end:]
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
  internal static bool Trusted,Decline,FailInstall;
  internal static int Transfers,Installs;
  internal static string Package;
  static int Main(string[] args){
    Program.LogPath=System.IO.Path.Combine(args[0],"test.log");
    // Fresh host: trust must precede the very first transfer.
    Program.RunOnlineInstall(s=>{});
    if(!Trusted || Transfers!=1 || Installs!=1 || System.IO.File.Exists(Package))throw new System.Exception("Fresh install or successful cleanup failed");
    Trusted=false;Decline=true;
    try{Program.RunOnlineInstall(s=>{});throw new System.Exception("Consent bypassed");}catch(System.IO.IOException){}
    if(Transfers!=1 || Installs!=1)throw new System.Exception("Declined consent performed network/deployment");
    Decline=false;FailInstall=true;
    try{Program.RunOnlineInstall(s=>{});throw new System.Exception("Deployment failure hidden");}catch(System.IO.IOException){}
    if(!System.IO.File.Exists(Package) || !System.IO.File.ReadAllText(Program.LogPath).Contains(System.IO.Path.GetDirectoryName(Package)))throw new System.Exception("Failure recovery package/path lost");
    System.IO.Directory.Delete(System.IO.Path.GetDirectoryName(Package),true);
    System.Console.WriteLine("FIRST_USE_TRUST_ORDER_AND_RECOVERY_PASS");return 0;
  }
}
'''
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
