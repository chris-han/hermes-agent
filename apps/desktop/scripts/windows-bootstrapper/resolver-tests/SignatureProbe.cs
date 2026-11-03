using System;
using System.IO;
using System.Linq;
static class SignatureProbe {
    public static void Main(string[] args) {
        var root=args[0];var name="SemantierAgent.msixbundle";
        var row=new ActiveRelease.Object("releases/channel-builds/"+new string('a',32)+"/"+name,new FileInfo(Path.Combine(root,name)).Length,new string('b',64));
        PackageSignature.Verify(root,new[]{row});
        var temp=Path.Combine(args[1],name);File.Copy(Path.Combine(root,name),temp);
        using(var file=new FileStream(temp,FileMode.Open,FileAccess.ReadWrite)) { file.Position=100;int b=file.ReadByte();file.Position=100;file.WriteByte((byte)(b^1)); }
        bool refused=false;try {PackageSignature.Verify(args[1],new[]{row});} catch(InvalidOperationException){refused=true;}
        if(!refused)throw new Exception("Corrupted signed package was accepted");
        File.Delete(temp);
        File.Copy(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System),"notepad.exe"),temp);
        refused=false;try {PackageSignature.Verify(args[1],new[]{row});} catch(InvalidOperationException){refused=true;}
        if(!refused)throw new Exception("Wrong publisher was accepted");
        Console.WriteLine("SIGNATURE_CORRUPTION_REFUSED");
    }
}
