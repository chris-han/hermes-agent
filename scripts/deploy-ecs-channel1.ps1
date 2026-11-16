# Channel 1 metadata publication is composed by deploy-ecs.ps1. No standalone execution.
function Publish-Channel1Artifacts([string]$Signed, [string]$Root, [string]$Stage, [string]$PackageSha256, [string]$Metadata, [string]$Publisher = (Join-Path $PSScriptRoot 'releases/channel1.py')) {
  if($Root -cnotmatch '^[A-Za-z0-9_:/.-]+$' -or $Stage -cnotmatch '^[A-Za-z0-9_:/.-]+$' -or $PackageSha256 -cnotmatch '^[a-f0-9]{64}$') {
    throw 'Invalid Channel 1 publication binding.'
  }
  $protocol=Join-Path (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $Publisher))) 'hermes_cli/release_channels.py'
  Run $BuildPython @('-B',$publisher,'prepare','--package',$Signed,'--output',$Metadata,
    '--expected-sha256',$PackageSha256,'--source-commit',$SourceCommit,'--windows-version',$Version,'--build-id',$BuildId)
  Upload $Signed "$Stage/SemantierAgent.msix"
  Upload $Metadata "$Stage/channel1-metadata.json"
  Upload $publisher "$Stage/channel1.py"
  Upload $protocol "$Stage/channel1-protocol.py"
  return Ssh "python3 -B '$Stage/channel1.py' publish --root '$Root' --stage '$Stage' --metadata-sha256 '$(Hash $Metadata)' --package-sha256 '$PackageSha256' --protocol '$Stage/channel1-protocol.py' --protocol-sha256 '$(Hash $protocol)'"
}
function Get-Channel1ProofCommand([string]$Root, [string]$PackageSha256, [string]$Build) {
  $command=@'
set -eu
python3 - '__ROOT__' '__HASH__' '__BUILD__' <<'CHANNEL1_PROOF'
import hashlib,json,pathlib,sys
root,wanted,build=sys.argv[1:];root=pathlib.Path(root)
assert (root/'releases/installers/semantier-setup-ch1.exe').is_file()
with (root/'releases/SemantierAgent.msix').open('rb') as stream:
    assert hashlib.file_digest(stream,'sha256').hexdigest()==wanted
record=json.loads((root/'releases/channels/semantier-acceptance.json').read_bytes())
head=record['head'];assert head['buildId']==build
body=(root/head['manifestKey']).read_bytes()
assert hashlib.sha256(body).hexdigest()==head['sha256']
manifest=json.loads(body);pkg=manifest['packages'][0]
assert pkg['artifact']['sha256']==wanted
assert (root/pkg['artifact']['key']).samefile(root/'releases/SemantierAgent.msix')
assert (root/pkg['feed']['key']).is_file()
print('CHANNEL1_CURRENT_CLOSURE_VERIFIED')
CHANNEL1_PROOF
'@
  return $command.Replace('__ROOT__',$Root).Replace('__HASH__',$PackageSha256).Replace('__BUILD__',$Build)
}
function Get-Channel1CleanupCommand([string]$Root, [string]$Build) {
  $command=@'
set -eu
root='__ROOT__'
current='__BUILD__'
( cd "$root/releases" && sha256sum -c SemantierAgent.msix.sha256 )
old="$root/releases/channel-builds"
if test -d "$old"; then
  find "$old" -mindepth 2 -maxdepth 5 -type f ! -path "$old/$current/*" \( -name '*.msix' -o -name '*.msixbundle' -o -name '*.zip' -o -name '*.msix.sha256' -o -name '*.msixbundle.sha256' -o -name '*.zip.sha256' \) -print -delete
fi
# The running Landing has been browser-verified to use the fixed Setup link.
# Only now may these legacy discovery pointers be removed.
rm -f "$root/releases/current-release.json" "$root/releases/build.json" \
      "$root/releases/manual/catalog.json"
printf 'OLD_PACKAGE_CLEANUP_PASS\n'
'@
  return $command.Replace('__ROOT__',$Root).Replace('__BUILD__',$Build)
}
