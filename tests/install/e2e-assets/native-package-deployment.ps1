# Bounded WD1 native URI proof, called only inside the existing register intent.
# This test adapter never creates CI qualification, publication or update acceptance.
function Invoke-WD1NativeUri($Graph, [string]$Base, [string]$Evidence) {
 $ErrorActionPreference='Stop'
 $context=Get-Variable -Name r -ErrorAction SilentlyContinue
 if(!$context -or $context.Value.purpose -cne 'wd1-native-layout-qualification' -or $context.Value.action -cne 'register' -or !$context.Value.PSObject.Properties['expected_before']){throw 'Existing WD1 register intent context required'}
 if([Security.Principal.WindowsIdentity]::GetCurrent().User.Value -cne 'S-1-5-21-1928370875-4287869425-2092521300-1011' -or -not [Environment]::UserInteractive -or (Get-Process -Id $PID).SessionId -le 0){throw 'Actual authorized baseline desktop required'}
 if($Base -cnotmatch '^http://127\.0\.0\.1:[1-9][0-9]{0,4}$'){throw 'WD1 requires its exact loopback origin'}
 . (Join-Path $PSScriptRoot '..\..\..\apps\desktop\scripts\native-package-digest.ps1')
 Add-Type -AssemblyName System.Runtime.WindowsRuntime
 $null=[Windows.Management.Deployment.PackageManager,Windows.Management.Deployment,ContentType=WindowsRuntime]
 $null=[Windows.Management.Deployment.AddPackageOptions,Windows.Management.Deployment,ContentType=WindowsRuntime]
 $manager=[Windows.Management.Deployment.PackageManager]::new()
 $method=[Windows.Management.Deployment.PackageManager].GetMethod('AddPackageByUriAsync')
 if(!$method){throw 'AddPackageByUriAsync unavailable'}
 $generic=$method.ReturnType.GetGenericArguments()
 $binding=@([System.WindowsRuntimeSystemExtensions].GetMethods()|Where-Object{$_.Name -ceq 'AsTask' -and $_.IsGenericMethodDefinition -and $_.GetGenericArguments().Count -eq 2 -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -ceq 'IAsyncOperationWithProgress`2'})
 if($binding.Count -ne 1){throw 'Actual native asynchronous binding unavailable'}
 $asTask=$binding[0].MakeGenericMethod($generic)
 $uris=@();$digests=@();$rootIndex=-1
 foreach($row in $Graph.objects){
  if($row.key -cnotmatch '^releases/channel-builds/[0-9a-f]{32}/[A-Za-z0-9.-]+\.(msix|msixbundle)$' -or $row.size -gt 480000000){throw 'Invalid bounded native graph'}
  if($row.key -ceq $Graph.root){$rootIndex=$uris.Count}
  $uris+=($Base+'/'+$row.key);$digests+=[string]$row.nativeDigest
 }
 if($rootIndex -lt 0){throw 'Native root absent'}
 function Deploy([string[]]$Expected) {
  $options=[Windows.Management.Deployment.AddPackageOptions]::new()
  $null=Set-NativeExpectedDigests $options $uris $Expected
  $watch=[Diagnostics.Stopwatch]::StartNew()
  try{
   $operation=$manager.AddPackageByUriAsync([Uri]$uris[$rootIndex],$options)
   $task=$asTask.Invoke($null,@($operation))
   if(!$task.Wait([TimeSpan]::FromMinutes(12))){throw [TimeoutException]::new('Native deployment remains unresolved; preserve intent')}
   $deployment=$task.Result
   if($deployment.ExtendedErrorCode){throw $deployment.ExtendedErrorCode}
   return @{producedAt=[DateTime]::UtcNow.ToString('o');status='SUCCESS';hresult='00000000';elapsedSeconds=$watch.Elapsed.TotalSeconds;activityId=[string]$deployment.ActivityId}
  }catch{
   $failureException=$_.Exception.GetBaseException()
   return @{producedAt=[DateTime]::UtcNow.ToString('o');status='FAILURE';hresult=([BitConverter]::ToUInt32([BitConverter]::GetBytes([int]$failureException.HResult),0).ToString('X8'));error=$failureException.Message;elapsedSeconds=$watch.Elapsed.TotalSeconds}
  }finally{$watch.Stop()}
 }
 # Flip one root SDK-digest bit; every URI and every other expected digest stays fixed.
 $wrong=[string[]]$digests.Clone();$bytes=[Convert]::FromBase64String($wrong[$rootIndex]);$bytes[0]=$bytes[0] -bxor 1;$wrong[$rootIndex]=[Convert]::ToBase64String($bytes)
 $refusal=Deploy $wrong
 $refusal|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $Evidence 'wrong-digest.json') -Encoding utf8
 $after=Observe
 if($after.package -or @($after.related).Count){throw 'Wrong-digest call changed registration; preserve actual state and pending intent'}
 # SDK26100 winerror.h: APPX_E_DIGEST_MISMATCH 0x80080219. Other failures do not prove binding.
 if($refusal.status -cne 'FAILURE' -or $refusal.hresult -cne '80080219'){throw 'Exact native digest refusal unproved; preserve failure before positive execution'}
 $positive=Deploy ([string[]]$digests)
 $positive|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $Evidence 'correct-digest.json') -Encoding utf8
 if($positive.status -cne 'SUCCESS'){throw 'Correct native URI deployment failed; preserve pending intent'}
 @{producedAt=[DateTime]::UtcNow.ToString('o');root=$Graph.root;uris=$uris;digests=$digests;refusal=$refusal;positive=$positive;scope='WD1 exact native URI and digest proof only; no migration/differential/cache/relaunch acceptance'}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $Evidence 'native-uri-proof.json') -Encoding utf8
}
