# Read-only SDK digest for ExpectedDigests. Never substitutes a file SHA256.
function Get-NativePackageDigest([string]$Package) {
  if (-not ('Semantier.NativeDigest' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;
namespace Semantier {
 [ComImport, Guid("beb94909-e451-438b-b5a7-d79e767b75d8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
 interface IAppxFactory {
  void CreatePackageWriter(IntPtr stream, IntPtr settings, out IntPtr writer);
  void CreatePackageReader(IStream stream, [MarshalAs(UnmanagedType.IUnknown)] out object reader);
 }
 [ComImport, Guid("bba65864-965f-4a5f-855f-f074bdbf3a7b"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
 interface IAppxBundleFactory {
  void CreateBundleWriter(IStream stream, ulong version, out IntPtr writer);
  void CreateBundleReader(IStream stream, [MarshalAs(UnmanagedType.IUnknown)] out object reader);
 }
 [ComImport, Guid("9fe2702b-7640-4659-8e6c-349e43c4cdbd"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
 interface IAppxDigestProvider {
  void GetDigest([MarshalAs(UnmanagedType.LPWStr)] out string digest);
 }
 public static class NativeDigest {
  [DllImport("shlwapi.dll", CharSet=CharSet.Unicode, PreserveSig=false)]
  static extern void SHCreateStreamOnFileEx(string file, uint mode, uint attributes, bool create, IntPtr template, out IStream stream);
  public static string Read(string file) {
   IStream stream=null; object factory=null, reader=null;
   try {
    SHCreateStreamOnFileEx(file, 0x20, 0, false, IntPtr.Zero, out stream);
    string extension=System.IO.Path.GetExtension(file);
    if(String.Equals(extension,".msixbundle",StringComparison.OrdinalIgnoreCase) ||
       String.Equals(extension,".appxbundle",StringComparison.OrdinalIgnoreCase)) {
     factory=Activator.CreateInstance(Type.GetTypeFromCLSID(new Guid("378e0446-5384-43b7-8877-e7dbdd883446")));
     ((IAppxBundleFactory)factory).CreateBundleReader(stream,out reader);
    } else if(String.Equals(extension,".msix",StringComparison.OrdinalIgnoreCase) ||
              String.Equals(extension,".appx",StringComparison.OrdinalIgnoreCase)) {
     factory=Activator.CreateInstance(Type.GetTypeFromCLSID(new Guid("5842a140-ff9f-4166-8f5c-62f5b7b0c781")));
     ((IAppxFactory)factory).CreatePackageReader(stream,out reader);
    } else { throw new ArgumentException("Expected a native package or bundle"); }
    string digest; ((IAppxDigestProvider)reader).GetDigest(out digest);
    if (String.IsNullOrWhiteSpace(digest)) throw new InvalidOperationException("Native digest unavailable");
    return digest;
   } finally {
    if(reader!=null) Marshal.FinalReleaseComObject(reader);
    if(factory!=null) Marshal.FinalReleaseComObject(factory);
    if(stream!=null) Marshal.FinalReleaseComObject(stream);
   }
  }
 }
}
'@
  }
  return [Semantier.NativeDigest]::Read((Resolve-Path -LiteralPath $Package).Path)
}
