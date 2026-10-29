# Read-only SDK digest for ExpectedDigests. Never substitutes a file SHA256.
function Initialize-NativePackageInterop {
  if (-not ('Semantier.NativeDigest' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
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
 [ComImport, Guid("b5c49650-99bc-481c-9a34-3d53a4106708"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
 interface IAppxPackageReader {
  void GetBlockMap(out IntPtr value);
  void GetFootprintFile(uint type,out IntPtr value);
  void GetPayloadFile([MarshalAs(UnmanagedType.LPWStr)]string name,out IntPtr value);
  void GetPayloadFiles(out IntPtr value);
  void GetManifest(out IAppxManifestReader value);
 }
 [ComImport, Guid("4e1bd148-55a0-4480-a3d1-15544710637c"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
 interface IAppxManifestReader { void GetPackageId(out IAppxManifestPackageId value); }
 [ComImport, Guid("283ce2d7-7153-4a91-9649-7a0f7240945f"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
 interface IAppxManifestPackageId {
  void GetName([MarshalAs(UnmanagedType.LPWStr)]out string value);
  void GetArchitecture(out uint value);
  void GetPublisher([MarshalAs(UnmanagedType.LPWStr)]out string value);
  void GetVersion(out ulong value);
  void GetResourceId([MarshalAs(UnmanagedType.LPWStr)]out string value);
  void ComparePublisher([MarshalAs(UnmanagedType.LPWStr)]string value,[MarshalAs(UnmanagedType.Bool)]out bool same);
  void GetPackageFullName([MarshalAs(UnmanagedType.LPWStr)]out string value);
  void GetPackageFamilyName([MarshalAs(UnmanagedType.LPWStr)]out string value);
 }
 public class NativeIdentity {
  public string Name,Publisher,Version,ResourceId,FullName,Family;
  public uint Architecture;
 }
 [ComImport, Guid("4c522071-01da-5f85-9d52-29952578a301"), InterfaceType(ComInterfaceType.InterfaceIsIInspectable)]
 interface IUriDigestMap {
  void Lookup(IntPtr key, [MarshalAs(UnmanagedType.HString)] out string value);
  void Size(out uint value);
  void HasKey(IntPtr key, [MarshalAs(UnmanagedType.I1)] out bool found);
  void GetView(out IntPtr view);
  void Insert(IntPtr key, [MarshalAs(UnmanagedType.HString)] string value, [MarshalAs(UnmanagedType.I1)] out bool replaced);
 }
 [ComImport, Guid("44a9796f-723e-4fdf-a218-033e75b0c084"), InterfaceType(ComInterfaceType.InterfaceIsIInspectable)]
 interface IUriFactory {
  // Keep the native pointer: .NET projects a marshalled URI object to System.Uri.
  void CreateUri([MarshalAs(UnmanagedType.HString)] string uri, out IntPtr key);
 }
 public static class NativeDigest {
  [DllImport("combase.dll", PreserveSig=false)]
  static extern void RoGetActivationFactory([MarshalAs(UnmanagedType.HString)] string name, ref Guid iid, out IUriFactory factory);
  public static uint SetExpected(object map, string[] uris, string[] digests) {
   if(uris==null || digests==null || uris.Length==0 || uris.Length>64 || uris.Length!=digests.Length)
    throw new ArgumentException("Incomplete expected native digest set");
   var addresses=new string[uris.Length];var unique=new HashSet<string>(StringComparer.Ordinal);
   for(int i=0;i<uris.Length;i++) {
    var address=new Uri(uris[i],UriKind.Absolute);
    if(address.UserInfo!="" || address.Fragment!="" || address.Query!="" ||
       (address.Scheme!="https" && !(address.Scheme=="http" && address.IsLoopback)))
     throw new ArgumentException("Expected an immutable HTTPS or loopback fixture URI");
    addresses[i]=address.AbsoluteUri;
    if(!unique.Add(addresses[i]))throw new ArgumentException("Duplicate expected native URI");
    var bytes=Convert.FromBase64String(digests[i]);
    if(bytes.Length!=32 || Convert.ToBase64String(bytes)!=digests[i])throw new ArgumentException("Invalid native digest");
   }
   var typed=(IUriDigestMap)map;uint count;typed.Size(out count);
   if(count!=0)throw new ArgumentException("Expected native digest map is already bound");
   IUriFactory factory=null;
   try {
    Guid iid=typeof(IUriFactory).GUID;
    RoGetActivationFactory("Windows.Foundation.Uri",ref iid,out factory);
    for(int i=0;i<addresses.Length;i++) {
     IntPtr key=IntPtr.Zero;
     try {
      factory.CreateUri(addresses[i],out key);
      bool replaced;typed.Insert(key,digests[i],out replaced);
      string actual;typed.Lookup(key,out actual);
      if(replaced || actual!=digests[i])throw new InvalidOperationException("Expected native digest binding failed");
     } finally {if(key!=IntPtr.Zero)Marshal.Release(key);}
    }
    typed.Size(out count);
    if(count!=uris.Length)throw new InvalidOperationException("Expected native digest set incomplete");
    return count;
   } finally {if(factory!=null)Marshal.FinalReleaseComObject(factory);}
  }
  [DllImport("shlwapi.dll", CharSet=CharSet.Unicode, PreserveSig=false)]
  static extern void SHCreateStreamOnFileEx(string file, uint mode, uint attributes, bool create, IntPtr template, out IStream stream);
  public static NativeIdentity Identity(string file) {
   IStream stream=null;object factory=null,reader=null;
   IAppxManifestReader manifest=null;IAppxManifestPackageId id=null;
   try {
    SHCreateStreamOnFileEx(file,0x20,0,false,IntPtr.Zero,out stream);
    factory=Activator.CreateInstance(Type.GetTypeFromCLSID(new Guid("5842a140-ff9f-4166-8f5c-62f5b7b0c781")));
    ((IAppxFactory)factory).CreatePackageReader(stream,out reader);
    ((IAppxPackageReader)reader).GetManifest(out manifest);manifest.GetPackageId(out id);
    var result=new NativeIdentity();ulong version;
    id.GetName(out result.Name);id.GetPublisher(out result.Publisher);id.GetArchitecture(out result.Architecture);
    id.GetResourceId(out result.ResourceId);id.GetPackageFullName(out result.FullName);id.GetPackageFamilyName(out result.Family);
    id.GetVersion(out version);
    result.Version=String.Format(System.Globalization.CultureInfo.InvariantCulture,"{0}.{1}.{2}.{3}",
      (version>>48)&65535,(version>>32)&65535,(version>>16)&65535,version&65535);
    return result;
   } finally {
    if(id!=null)Marshal.ReleaseComObject(id);
    if(manifest!=null)Marshal.ReleaseComObject(manifest);
    if(reader!=null)Marshal.ReleaseComObject(reader);
    if(factory!=null)Marshal.ReleaseComObject(factory);
    if(stream!=null)Marshal.ReleaseComObject(stream);
   }
  }
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
}
function Get-NativePackageDigest([string]$Package) {
  Initialize-NativePackageInterop
  return [Semantier.NativeDigest]::Read((Resolve-Path -LiteralPath $Package).Path)
}
# IMap<Windows.Foundation.Uri,HSTRING> is not an IDictionary in Windows PS5.1.
# This binds options only; it never downloads, stages or registers a package.
function Set-NativeExpectedDigests($Options, [string[]]$Uris, [string[]]$Digests) {
  Initialize-NativePackageInterop
  return [Semantier.NativeDigest]::SetExpected($Options.ExpectedDigests,$Uris,$Digests)
}

function Get-NativePackageIdentity([string]$Package) {
  Initialize-NativePackageInterop
  return [Semantier.NativeDigest]::Identity((Resolve-Path -LiteralPath $Package).Path)
}
