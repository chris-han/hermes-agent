using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

static class PackageSignature {
    const string Thumbprint="57BF15739B4AA447842EAF388856A01B4E5029FF";
    internal static void Verify(string dir, IReadOnlyList<ActiveRelease.Object> objects) {
        if(objects==null || objects.Count==0 || objects.Any(x=>!System.Text.RegularExpressions.Regex.IsMatch(Path.GetFileName(x.Key), @"^[A-Za-z0-9_.-]+\.(msix|msixbundle)$")))
            throw new InvalidOperationException("Invalid package names.");
        string encoded=Convert.ToBase64String(Encoding.Unicode.GetBytes(dir));
        string names=string.Join(",",objects.Select(x=>"'"+Path.GetFileName(x.Key)+"'"));
        string script="$ErrorActionPreference='Stop';$d=[Text.Encoding]::Unicode.GetString([Convert]::FromBase64String('"+encoded+"'));"+
            "$names=@("+names+");foreach($n in $names){$s=Get-AuthenticodeSignature -LiteralPath (Join-Path $d $n);"+
            "if(!$s.SignerCertificate -or $s.SignerCertificate.Thumbprint -ne '"+Thumbprint+"' -or $s.Status -ne 'Valid'){throw 'Package signature mismatch'}}";
        string exe=Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System),@"WindowsPowerShell\v1.0\powershell.exe");
        var info=new ProcessStartInfo(exe,"-NoProfile -NonInteractive -EncodedCommand "+Convert.ToBase64String(Encoding.Unicode.GetBytes(script))) {
            UseShellExecute=false,CreateNoWindow=true,RedirectStandardOutput=true,RedirectStandardError=true
        };
        using(var process=Process.Start(info)) {
            var stdout=process.StandardOutput.ReadToEndAsync();var stderr=process.StandardError.ReadToEndAsync();
            process.WaitForExit();stdout.GetAwaiter().GetResult();string error=stderr.GetAwaiter().GetResult();
            if(process.ExitCode!=0) throw new InvalidOperationException("Package signature verification failed: "+error);
        }
    }
}
