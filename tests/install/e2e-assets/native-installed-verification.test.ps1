# Windows fixture with package APIs substituted; no actual deployment or cleanup.
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'windows-bundle-installed.ps1')
$root=Join-Path ([IO.Path]::GetTempPath()) ('semantier-installed-fixture-'+[Guid]::NewGuid().ToString('N'))
$previousKey=[Environment]::GetEnvironmentVariable('SEMANTIER_CLIENT_RECEIPT_KEY','Process')
[Environment]::SetEnvironmentVariable('SEMANTIER_CLIENT_RECEIPT_KEY','fixture-secret','Process')
try {
 $install=Join-Path $root 'install';$resources=Join-Path $install 'app\resources'
 New-Item -ItemType Directory -Path (Join-Path $resources 'agent-payload') -Force|Out-Null
 Set-Content -LiteralPath (Join-Path $install 'app\agent.exe') -Value 'fixture'
 Set-Content -LiteralPath (Join-Path $resources 'install-stamp.json') -Value '{"channelBuild":{"buildId":"fixture"}}'
 $script:fixturePackage=[pscustomobject]@{PackageFullName='Fixture_1.2.3.4_x64__fixture';PackageFamilyName='Fixture_fixture';Name='Fixture';ResourceId='';Publisher='CN=Fixture';Architecture='x64';Version='1.2.3.4';Status='Ok';InstallLocation=$install;SignatureKind='Developer';IsDevelopmentMode=$false}
 $script:fixtureResources=@()
 $script:packageCalls=0
 function Get-AppxPackage { param($Name,$PackageTypeFilter) $script:packageCalls++;if($PackageTypeFilter -eq 'Resource'){return $script:fixtureResources};return $script:fixturePackage }
 function Get-AppxPackageManifest { return @{} }
 function Read-SmokePackageManifest { return @{Version='1.2.3.4';Executable='app\agent.exe'} }
 function Run-Node { if([Environment]::GetEnvironmentVariable('SEMANTIER_CLIENT_RECEIPT_KEY','Process')){throw 'Receipt key reached verifier'};return '"1.2.3.4"' }
 $expected=[pscustomobject]@{msixIdentity='Fixture';publisher='CN=Fixture'}
 $inputMeta=[pscustomobject]@{Version='1.2.3.4';Executable='app\agent.exe'}
 $result=Get-BundleInstalledVerification -Arch 'x64' -Expected $expected -InputMetadata $inputMeta -ChannelRequest 'fixture' -IdentityArgs @() -Metadata 'fixture' -Tag ''
 if($result.identity.packageFullName -cne $script:fixturePackage.PackageFullName){throw 'Exact installed identity not returned'}
 $script:fixturePackage.Version='1.2.3.5'
 $refused=$false
 try { Get-BundleInstalledVerification -Arch 'x64' -Expected $expected -InputMetadata $inputMeta -ChannelRequest 'fixture' -IdentityArgs @() -Metadata 'fixture' -Tag '' | Out-Null } catch {$refused=$true}
 if(!$refused){throw 'Wrong installed version accepted'}
 $status=@{};foreach($key in @('DeploymentInProgress','Servicing','IsPartiallyStaged','VerifyIsOK')){$status[$key]=@{available=$true;value=($key -ceq 'VerifyIsOK')}}
 $nativePkg=[pscustomobject]@{PackageFullName='Fixture_1.2.3.4_x64__fixture';PackageFamilyName='Fixture_fixture';Name='Fixture';Publisher='CN=Fixture';Version='1.2.3.4';Architecture='x64';ResourceId='';InstallLocation=$install}
 $script:snapshot=@{presence='PRESENT';identity=@{full=$nativePkg.PackageFullName;family=$nativePkg.PackageFamilyName;name=$nativePkg.Name;publisher=$nativePkg.Publisher;version=$nativePkg.Version;architecture=$nativePkg.Architecture;resourceId=''};installedLocation=@{state='VALUE';path=$install};status=$status}
 function Get-CIWinRTPackageStatus { return $script:snapshot }
 $null=Assert-BundleWinRTReady $nativePkg 'fixture'
 $script:snapshot.status.IsPartiallyStaged.value=$true
 $refused=$false;try{Assert-BundleWinRTReady $nativePkg 'fixture'|Out-Null}catch{$refused=$true}
 if(!$refused){throw 'Partially staged package accepted'}
 $script:snapshot.status.IsPartiallyStaged.value=$false;$script:snapshot.status.VerifyIsOK.available=$false
 $refused=$false;try{Assert-BundleWinRTReady $nativePkg 'fixture'|Out-Null}catch{$refused=$true}
 if(!$refused){throw 'Unavailable native status accepted'}
 $script:snapshot.status.VerifyIsOK.available=$true;$script:snapshot.installedLocation.path=Join-Path $root 'different'
 $refused=$false;try{Assert-BundleWinRTReady $nativePkg 'fixture'|Out-Null}catch{$refused=$true}
 if(!$refused){throw 'Changed WinRT installed location accepted'}
 $script:snapshot.installedLocation.path=$install
 $script:fixturePackage.Version='1.2.3.4';$script:snapshot.status.VerifyIsOK.available=$true
 $resource=[pscustomobject]@{PackageFullName='Fixture_1.2.3.4_neutral_runtime0001_fixture';PackageFamilyName='Fixture_fixture';Name='Fixture';ResourceId='runtime0001';Publisher='CN=Fixture';Architecture='neutral';Version='1.2.3.4';Status='Ok';InstallLocation=$install;SignatureKind='Developer';IsDevelopmentMode=$false}
 $script:fixtureResources=@($resource)
 function Get-CIWinRTPackageStatus { param($Sid,$FullName)
  $p=if($FullName -ceq $resource.PackageFullName){$resource}else{$script:fixturePackage}
  return @{presence='PRESENT';identity=@{full=$p.PackageFullName;family=$p.PackageFamilyName;name=$p.Name;publisher=$p.Publisher;version=$p.Version;architecture=$p.Architecture;resourceId=$p.ResourceId};installedLocation=@{state='VALUE';path=$install};status=$status}
 }
 $script:nodeCalls=0
 function Run-Node { param([string[]]$Argv) $script:nodeCalls++;if([Environment]::GetEnvironmentVariable('SEMANTIER_CLIENT_RECEIPT_KEY','Process')){throw 'Receipt key reached verifier'};if($Argv -contains 'native-payload'){return '{"filesChecked":1}'};return '"1.2.3.4"' }
 $actualSid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
 $graph=@{objects=@(@{kind='main-bundle'},@{kind='main-package';identity='Fixture';publisher='CN=Fixture';version='1.2.3.4';arch='x64'},@{kind='asset-package';identity='Fixture';publisher='CN=Fixture';resourceId='runtime0001';version='1.2.3.4';arch='neutral'})}
 $verified=Get-BundleInstalledVerification -Arch 'x64' -Expected $expected -InputMetadata $inputMeta -ChannelRequest 'fixture' -IdentityArgs @() -Metadata 'fixture' -Tag '' -ExpectedSid $actualSid -Graph $graph -NativeLayoutPlan 'fixture' -NativeGraph 'fixture' -GraphHash ('a'*64) -LayoutHash ('b'*64)
 if($verified.registration.related.Count -ne 1 -or $verified.folding.filesChecked -ne 1){throw 'Native graph evidence missing'}
 if($verified.registration.ContainsKey('nativeRemoteDigestEnforcement')){throw 'Read-only phase fabricated remote digest proof'}
 $script:rootCalls=0;$script:replaceRoot=$false
 function Registered-Root {param($Package,[switch]$Retain) $script:rootCalls++;$id=$Package.PackageFullName;if($script:replaceRoot -and $script:rootCalls -gt 2){$id='replaced'};return @{id=$id;handle=[IO.MemoryStream]::new()}}
 $bound=Get-CIBoundInstalledVerification -Arch 'x64' -Expected $expected -InputMetadata $inputMeta -ChannelRequest 'fixture' -IdentityArgs @() -Metadata 'fixture' -Tag '' -ExpectedSid $actualSid -Graph $graph -NativeLayoutPlan 'fixture' -NativeGraph 'fixture' -GraphHash ('a'*64) -LayoutHash ('b'*64)
 if($bound.roots.Count -ne 2 -or $bound.verification.folding.filesChecked -ne 1){throw 'Physical-bound installed verification missing'}
 foreach($item in $bound.roots){$item.root.handle.Dispose()}
 $script:rootCalls=0;$refused=$false
 try{Get-CIBoundInstalledVerification -Arch 'x64' -Expected $expected -InputMetadata $inputMeta -ChannelRequest 'fixture' -IdentityArgs @() -Metadata 'fixture' -Tag '' -ExpectedSid 'S-1-5-21-1-2-3-4' -Graph $graph -NativeLayoutPlan 'fixture' -NativeGraph 'fixture'|Out-Null}catch{$refused=$true}
 if(!$refused -or $script:rootCalls -ne 0){throw 'Wrong actor reached physical package roots'}
 $script:rootCalls=0;$script:replaceRoot=$true;$refused=$false
 try{Get-CIBoundInstalledVerification -Arch 'x64' -Expected $expected -InputMetadata $inputMeta -ChannelRequest 'fixture' -IdentityArgs @() -Metadata 'fixture' -Tag '' -ExpectedSid $actualSid -Graph $graph -NativeLayoutPlan 'fixture' -NativeGraph 'fixture'|Out-Null}catch{$refused=$true}
 if(!$refused){throw 'Changed physical package root accepted'}
 $script:replaceRoot=$false
 $graph.objects[1].version='1.2.3.5'
 $reason='';try{Get-BundleInstalledVerification -Arch 'x64' -Expected $expected -InputMetadata $inputMeta -ChannelRequest 'fixture' -IdentityArgs @() -Metadata 'fixture' -Tag '' -ExpectedSid $actualSid -Graph $graph -NativeLayoutPlan 'fixture' -NativeGraph 'fixture'|Out-Null}catch{$reason=$_.Exception.Message}
 if($reason -notlike '*graph main*'){throw 'Changed graph main identity was not refused'}
 $graph.objects[1].version='1.2.3.4'
 $graph.objects[2].publisher='CN=Wrong'
 $reason='';try{Get-BundleInstalledVerification -Arch 'x64' -Expected $expected -InputMetadata $inputMeta -ChannelRequest 'fixture' -IdentityArgs @() -Metadata 'fixture' -Tag '' -ExpectedSid $actualSid -Graph $graph -NativeLayoutPlan 'fixture' -NativeGraph 'fixture'|Out-Null}catch{$reason=$_.Exception.Message}
 if($reason -notlike '*graph asset identity*'){throw 'Changed graph resource publisher was not refused'}
 $graph.objects[2].publisher='CN=Fixture'
 $script:nodeCalls=0;$script:packageCalls=0
 $reason='';try{Get-BundleInstalledVerification -Arch 'x64' -Expected $expected -InputMetadata $inputMeta -ChannelRequest 'fixture' -IdentityArgs @() -Metadata 'fixture' -Tag '' -ExpectedSid 'S-1-5-21-1-2-3-4' -Graph $graph -NativeLayoutPlan 'fixture' -NativeGraph 'fixture'|Out-Null}catch{$reason=$_.Exception.Message}
 if($reason -notlike '*actor*' -or $script:nodeCalls -ne 0 -or $script:packageCalls -ne 0){throw 'Mismatched native actor reached package reads or verifier'}
 $resource.Version='1.2.3.5'
 $refused=$false;try{Get-BundleInstalledVerification -Arch 'x64' -Expected $expected -InputMetadata $inputMeta -ChannelRequest 'fixture' -IdentityArgs @() -Metadata 'fixture' -Tag '' -ExpectedSid $actualSid -Graph $graph -NativeLayoutPlan 'fixture' -NativeGraph 'fixture'|Out-Null}catch{$refused=$true}
 if(!$refused){throw 'Wrong native resource version accepted'}
 $resource.Version='1.2.3.4'
 function Run-Node { param([string[]]$Argv) if($Argv -contains 'native-payload'){$resource.Version='1.2.3.5';return '{"filesChecked":1}'};return '"1.2.3.4"' }
 $refused=$false;try{Get-BundleInstalledVerification -Arch 'x64' -Expected $expected -InputMetadata $inputMeta -ChannelRequest 'fixture' -IdentityArgs @() -Metadata 'fixture' -Tag '' -ExpectedSid $actualSid -Graph $graph -NativeLayoutPlan 'fixture' -NativeGraph 'fixture'|Out-Null}catch{$refused=$true}
 if(!$refused){throw 'Resource change during payload verification accepted'}
 $resource.Version='1.2.3.4'
 function Run-Node { param([string[]]$Argv) if($Argv -contains 'native-payload'){$script:fixturePackage.InstallLocation=Join-Path $root 'replaced';return '{"filesChecked":1}'};return '"1.2.3.4"' }
 $reason='';try{Get-BundleInstalledVerification -Arch 'x64' -Expected $expected -InputMetadata $inputMeta -ChannelRequest 'fixture' -IdentityArgs @() -Metadata 'fixture' -Tag '' -ExpectedSid $actualSid -Graph $graph -NativeLayoutPlan 'fixture' -NativeGraph 'fixture'|Out-Null}catch{$reason=$_.Exception.Message}
 if($reason -notlike '*Installed package graph changed during content verification*'){throw 'Main location change was not refused at the graph recheck'}
 $script:fixturePackage.InstallLocation=$install
 function Run-Node { param([string[]]$Argv) if($Argv -contains 'native-payload'){$resource.InstallLocation=Join-Path $root 'resource-replaced';return '{"filesChecked":1}'};return '"1.2.3.4"' }
 $reason='';try{Get-BundleInstalledVerification -Arch 'x64' -Expected $expected -InputMetadata $inputMeta -ChannelRequest 'fixture' -IdentityArgs @() -Metadata 'fixture' -Tag '' -ExpectedSid $actualSid -Graph $graph -NativeLayoutPlan 'fixture' -NativeGraph 'fixture'|Out-Null}catch{$reason=$_.Exception.Message}
 if($reason -notlike '*Installed native resource changed during content verification*'){throw 'Resource location change was not refused at the graph recheck'}
 $resource.InstallLocation=$install;$script:nodeCalls=0;$script:packageCalls=0
 function Run-Node { param([string[]]$Argv) $script:nodeCalls++;if($Argv -contains 'native-payload'){return '{"filesChecked":1}'};return '"1.2.3.4"' }
 $graph.objects+=@{kind='optional-package';identity='Fixture.Optional'}
 $reason='';try{Get-BundleInstalledVerification -Arch 'x64' -Expected $expected -InputMetadata $inputMeta -ChannelRequest 'fixture' -IdentityArgs @() -Metadata 'fixture' -Tag '' -ExpectedSid $actualSid -Graph $graph -NativeLayoutPlan 'fixture' -NativeGraph 'fixture'|Out-Null}catch{$reason=$_.Exception.Message}
 if($reason -notlike '*applicability*' -or $script:nodeCalls -ne 0 -or $script:packageCalls -ne 0){throw 'Unverified optional applicability reached package reads or verifier'}
 $script:phaseCalls=@()
 function Run-Node { param([string[]]$Argv) if([Environment]::GetEnvironmentVariable('SEMANTIER_CLIENT_RECEIPT_KEY','Process')){throw 'Receipt key reached desktop'};$script:phaseCalls+='desktop' }
 function Test-BundleNativePayload { param($Root,$Package,$Commit,$BuildId,$Out) if([Environment]::GetEnvironmentVariable('SEMANTIER_CLIENT_RECEIPT_KEY','Process')){throw 'Receipt key reached native qualification'};$script:phaseCalls+='native' }
 function Test-BundlePluginInstall { param($Root,$Out,[switch]$IsolatedHome) if([Environment]::GetEnvironmentVariable('SEMANTIER_CLIENT_RECEIPT_KEY','Process')){throw 'Receipt key reached plugin qualification'};$script:phaseCalls+='plugin' }
 Invoke-BundleInstalledQualification -Assets $root -Work $root -Exe 'fixture.exe' -Root $root -HomeDir $root -UserData $root -Out $root -Commit ('a'*40) -NativeGraph 'graph' -Package $script:fixturePackage -Stamp @{channelBuild=@{buildId='fixture'}} -IsolatedHome
 if(($script:phaseCalls -join ',') -cne 'desktop,native,plugin'){throw 'Remaining installed qualification phase order changed'}
 $script:phaseCalls=@()
 Invoke-BundleInstalledQualification -Assets $root -Work $root -Exe 'fixture.exe' -Root $root -HomeDir $root -UserData $root -Out $root -Commit ('a'*40) -Package $script:fixturePackage
 if(($script:phaseCalls -join ',') -cne 'desktop,plugin'){throw 'Legacy qualification phase order changed'}
 function Run-Node { param([string[]]$Argv) throw 'simulated desktop failure' }
 $script:phaseCalls=@();$refused=$false;$beforeLocation=(Get-Location).Path
 try{Invoke-BundleInstalledQualification -Assets $root -Work $root -Exe 'fixture.exe' -Root $root -HomeDir $root -UserData $root -Out $root -Commit ('a'*40) -NativeGraph 'graph' -Package $script:fixturePackage -Stamp @{channelBuild=@{buildId='fixture'}} -IsolatedHome}catch{$refused=$true}
 if(!$refused -or $script:phaseCalls.Count -or (Get-Location).Path -cne $beforeLocation -or [Environment]::GetEnvironmentVariable('SEMANTIER_CLIENT_RECEIPT_KEY','Process') -cne 'fixture-secret'){throw 'Failure did not preserve qualification boundaries'}
 Write-Output 'PASS: installed verification checks main, resource and status; package APIs and payload verifier simulated'
} finally { [Environment]::SetEnvironmentVariable('SEMANTIER_CLIENT_RECEIPT_KEY',$previousKey,'Process');if(Test-Path -LiteralPath $root){Remove-Item -LiteralPath $root -Recurse -Force} }
