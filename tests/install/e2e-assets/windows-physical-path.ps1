# Shared Windows physical-path and registered package-root boundary.
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
public static class SemantierNativeFacts {
 [DllImport("wtsapi32.dll",CharSet=CharSet.Unicode,SetLastError=true)] public static extern bool WTSQuerySessionInformation(IntPtr s,int id,int info,out IntPtr b,out int n);
 [DllImport("wtsapi32.dll")] public static extern void WTSFreeMemory(IntPtr p);
 [DllImport("kernel32.dll",CharSet=CharSet.Unicode,SetLastError=true)] public static extern SafeFileHandle CreateFile(string n,uint access,uint share,IntPtr sa,uint disposition,uint flags,IntPtr template);
 [DllImport("kernel32.dll",CharSet=CharSet.Unicode,SetLastError=true)] public static extern uint GetFinalPathNameByHandle(SafeFileHandle h,System.Text.StringBuilder p,uint n,uint flags);
 [StructLayout(LayoutKind.Sequential)] public struct Info { public uint attr; public System.Runtime.InteropServices.ComTypes.FILETIME created,accessed,written; public uint volume,high,low,links,indexHigh,indexLow; }
 [DllImport("kernel32.dll",SetLastError=true)] public static extern bool GetFileInformationByHandle(SafeFileHandle h,out Info i);
}
'@

function Physical([string]$path) {
 $item=Get-Item -LiteralPath $path -Force
 $cursor=$item
 while($cursor){if(($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint)-ne 0){throw 'Reparse path refused'};if($cursor -is [IO.DirectoryInfo]){$cursor=$cursor.Parent}else{$cursor=$cursor.Directory}}
 $handle=[SemantierNativeFacts]::CreateFile($item.FullName,[uint32]2147483648,1,[IntPtr]::Zero,3,0x02000000,[IntPtr]::Zero)
 if($handle.IsInvalid){throw ('Cannot hold physical path against replacement: {0}, Win32={1}' -f $path,[Runtime.InteropServices.Marshal]::GetLastWin32Error())}
 $final=[Text.StringBuilder]::new(32768)
 $length=[SemantierNativeFacts]::GetFinalPathNameByHandle($handle,$final,32768,0)
 $canonical=$final.ToString()
 if($canonical.StartsWith('\\?\UNC\')){$canonical='\\'+$canonical.Substring(8)}elseif($canonical.StartsWith('\\?\')){$canonical=$canonical.Substring(4)}
 if($length -eq 0 -or $length -ge 32768 -or $canonical.TrimEnd('\') -ine $item.FullName.TrimEnd('\')){$handle.Dispose();throw 'Opened physical path escaped inspected path'}
 $info=New-Object SemantierNativeFacts+Info
 if(-not [SemantierNativeFacts]::GetFileInformationByHandle($handle,[ref]$info)){$handle.Dispose();throw 'Physical identity unavailable'}
 return @{handle=$handle;id=('{0}:{1}:{2}' -f $info.volume,$info.indexHigh,$info.indexLow);path=$item.FullName}
}
# Only an OS-registered package root may cross a package-volume redirect.
# Fixture/artifact/process paths continue through strict Physical unchanged.
function Registered-Root($package,[switch]$Retain) {
 $logical=[IO.Path]::GetFullPath($package.InstallLocation).TrimEnd('\')
 $full=$package.PackageFullName
 if(-not $full -or [IO.Path]::GetFileName($logical) -cne $full){throw 'Registered package root/name mismatch'}
 $stores=@(Get-AppxVolume|Where-Object{-not $_.IsOffline}|ForEach-Object{[IO.Path]::GetFullPath($_.PackageStorePath).TrimEnd('\')})
 $parent=Split-Path -Parent $logical
 if($parent -notin $stores){throw 'Registered package root outside OS package stores'}
 $cursor=Get-Item -LiteralPath $parent -Force
 while($cursor){if($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Reparse package-store ancestor refused'};$cursor=$cursor.Parent}
 $handle=$null;$physical=$null
 try {
  $handle=[SemantierNativeFacts]::CreateFile($logical,0,1,[IntPtr]::Zero,3,0x02000000,[IntPtr]::Zero)
  if($handle.IsInvalid){throw 'Registered package root unavailable'}
  $final=[Text.StringBuilder]::new(32768)
  $length=[SemantierNativeFacts]::GetFinalPathNameByHandle($handle,$final,32768,0)
  if($length -eq 0 -or $length -ge 32768){throw 'Registered package root resolution failed'}
  $resolved=$final.ToString()
  if($resolved.StartsWith('\\?\')){$resolved=$resolved.Substring(4)}
  $matches=@($stores|ForEach-Object{Join-Path $_ $full}|Where-Object{$_ -ieq $resolved})
  if($matches.Count -ne 1){throw 'Registered package redirect outside exact OS package store'}
  $physical=Physical $matches[0]
  $info=New-Object SemantierNativeFacts+Info
  if(-not [SemantierNativeFacts]::GetFileInformationByHandle($handle,[ref]$info)){throw 'Registered package identity unavailable'}
  if(('{0}:{1}:{2}' -f $info.volume,$info.indexHigh,$info.indexLow) -cne $physical.id){throw 'Registered package redirect changed'}
  if($Retain){$result=$physical;$physical=$null;return $result}
  return $physical.path
 } finally {if($physical){$physical.handle.Dispose()};if($handle){$handle.Dispose()}}
}
