# Actual Windows NTFS behavior; no deployment, networking, elevation or policy changes.
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'native-package-update.ps1')
$sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$temp=[IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\')
$root=Join-Path $temp ('native-journal-acl-unit-'+[guid]::NewGuid().ToString('N'))
$null=[IO.Directory]::CreateDirectory($root)
try{
 # Reproduce the real parent: current-user Modify permits creation but not WRITE_OWNER.
 $acl=[Security.AccessControl.DirectorySecurity]::new();$acl.SetAccessRuleProtection($true,$false)
 foreach($principal in @('S-1-5-18','S-1-5-32-544')){$acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]::new($principal),'FullControl','ContainerInherit,ObjectInherit','None','Allow'))}
 $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]::new($sid),'Modify','ContainerInherit,ObjectInherit','None','Allow'))
 [IO.Directory]::SetAccessControl($root,$acl)
 $journal=Join-Path $root 'new'
 New-NativeUpdateJournalDirectory $journal $sid
 $actual=[IO.Directory]::GetAccessControl($journal)
 if($actual.GetOwner([Security.Principal.SecurityIdentifier]).Value -cne $sid -or !$actual.AreAccessRulesProtected){throw 'Journal owner/isolation mismatch'}
 $rules=@($actual.GetAccessRules($true,$true,[Security.Principal.SecurityIdentifier]))
 $expected=@($sid,'S-1-5-18','S-1-5-32-544')
 if($rules.Count -ne 3){throw 'Unexpected journal access rule'}
 foreach($rule in $rules){if($rule.IdentityReference.Value -cnotin $expected -or $rule.IsInherited -or $rule.AccessControlType -ne 'Allow' -or $rule.FileSystemRights -ne 'FullControl'){throw 'Journal ACL weakened'}}
 [IO.File]::WriteAllText((Join-Path $journal 'intent.json'),'retained')
 $before=$actual.Sddl;$refused=$false
 try{New-NativeUpdateJournalDirectory $journal $sid}catch{$refused=$true}
 if(!$refused -or [IO.Directory]::GetAccessControl($journal).Sddl -cne $before -or [IO.File]::ReadAllText((Join-Path $journal 'intent.json')) -cne 'retained'){throw 'Existing journal retry mutated retained state'}
 $foreign=Join-Path $root 'foreign-actor';$refused=$false
 try{New-NativeUpdateJournalDirectory $foreign 'S-1-5-18'}catch{$refused=$true}
 if(!$refused -or [IO.Directory]::GetAccessControl($foreign).GetOwner([Security.Principal.SecurityIdentifier]).Value -cne $sid){throw 'Unverified owner accepted or changed'}
 @{scope='REAL_WINDOWS_NTFS_ORDINARY_USER';status='PASS';cases=3;owner=$sid;producedAt=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json -Compress
}finally{
 if([IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($root)) -ine $temp -or [IO.Path]::GetFileName($root) -notlike 'native-journal-acl-unit-*'){throw 'Unit fixture cleanup boundary mismatch'}
 Remove-Item -LiteralPath $root -Recurse -Force
}
