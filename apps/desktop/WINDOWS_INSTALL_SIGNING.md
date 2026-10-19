# Windows MSIX 签名与 ZIP 打包

本文用于每次 Windows x64 构建完成后，重复执行本地签名与打包。
使用自签名证书；安装电脑需要预先信任对应公钥证书。

## 1. 首次创建可复用的签名证书

在构建电脑的普通 PowerShell 中执行一次。
`Subject` 必须与 MSIX 清单的 `Identity Publisher` 完全匹配。
以下值对应 Hermes 当前的非商店 MSIX 发布者：

```powershell
$cert = New-SelfSignedCertificate `
  -Type Custom `
  -Subject 'CN=Nous Research Inc., O=Nous Research Inc., L=Austin, S=Texas, C=US' `
  -KeyUsage DigitalSignature `
  -FriendlyName 'Hermes Local Signing' `
  -CertStoreLocation 'Cert:\CurrentUser\My' `
  -TextExtension @('2.5.29.37={text}1.3.6.1.5.5.7.3.3') `
  -NotAfter (Get-Date).AddYears(2)

$cert.Thumbprint
```

保存输出的证书指纹。证书及私钥保存在当前用户的 Windows 证书存储中；
以后应以同一个 Windows 用户进行签名，不必每次重新创建证书。
证书过期或丢失后，需要换用新证书，并在安装电脑上更新信任。

此前构建中临时生成并已删除私钥的证书，不能仅凭 `.cer` 文件再次签名。
`.cer` 只包含公钥，不包含私钥。

## 2. 每次构建后签名并打包

将下面内容保存为自己的 `.ps1` 脚本，修改顶部变量后运行。
`$msix` 指向新构建的包；每次使用新的 `$dist` 文件夹和 `$zip` 文件名，避免混入旧文件。
`$signtool` 和 `$sevenZip` 指向构建准备阶段提供的工具，示例路径可按实际构建工作目录调整。

```powershell
$ErrorActionPreference = 'Stop'

$thumbprint = 'YOUR_CERTIFICATE_THUMBPRINT'
$msix = 'D:\path\Hermes.msix'
$dist = 'D:\path\distribution-new-build'
$zip = 'D:\path\Hermes-new-build-windows-x64-signed.zip'

$signtool = 'D:\repo\hermes-agent\.build\desktop-main-c277422\packager\winCodeSign\x64\signtool.exe'
$sevenZip = 'D:\repo\hermes-agent\.build\desktop-main-c277422\packager\sevenZip\bin\7za.exe'

if (Test-Path -LiteralPath $dist) { throw '请使用新的分发文件夹。' }
if (Test-Path -LiteralPath $zip) { throw '请使用新的 ZIP 文件名。' }
if (-not (Test-Path -LiteralPath $msix -PathType Leaf)) { throw '找不到 MSIX。' }

$cert = Get-Item -LiteralPath "Cert:\CurrentUser\My\$thumbprint"
if (-not $cert.HasPrivateKey) { throw '签名证书缺少私钥。' }
if ($cert.NotAfter -le (Get-Date)) { throw '签名证书已过期。' }

& $signtool sign /fd SHA256 /sha1 $thumbprint $msix
if ($LASTEXITCODE -ne 0) { throw 'MSIX 签名失败。' }

New-Item -ItemType Directory -Path $dist | Out-Null
Export-Certificate -Cert $cert -FilePath "$dist\Hermes-signing.cer" | Out-Null
Copy-Item -LiteralPath $msix -Destination $dist

Get-ChildItem -LiteralPath $dist -File | Sort-Object Name | ForEach-Object {
    $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  $($_.Name)"
} | Set-Content -LiteralPath "$dist\SHA256SUMS.txt" -Encoding ascii

& $sevenZip a -tzip $zip "$dist\*"
if ($LASTEXITCODE -ne 0) { throw 'ZIP 创建失败。' }

& $sevenZip t $zip
if ($LASTEXITCODE -ne 0) { throw 'ZIP 校验失败。' }

$zipHash = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
Set-Content -LiteralPath ($zip + '.sha256') `
    -Value ($zipHash + '  ' + [IO.Path]::GetFileName($zip)) -Encoding ascii
Get-Item -LiteralPath $zip | Select-Object FullName, Length
Write-Output "SHA256: $zipHash"
```

此流程签署的是 MSIX 包，包签名覆盖包内容；它不会逐个为内部 EXE/DLL 添加 Authenticode 签名。
这里没有配置可信时间戳，因此不要假定证书到期后新安装仍能通过验证。
ZIP 只分发 MSIX、公钥 `.cer` 和校验文件，不要把私钥 `.pfx` 放入分发目录。

## 3. 安装电脑首次信任证书

先通过可信渠道核对 `.cer` 的证书指纹：

```powershell
Get-PfxCertificate -FilePath '.\Hermes-signing.cer' |
    Format-List Subject, Thumbprint, NotBefore, NotAfter
```

确认指纹后，在安装电脑的**管理员 PowerShell**中执行：

```powershell
Import-Certificate -FilePath '.\Hermes-signing.cer' `
  -CertStoreLocation 'Cert:\LocalMachine\TrustedPeople'
```

目标位置是“本地计算机 → 受信任人（Trusted People）”，不是“受信任的根证书颁发机构”。
仅导入当前用户的“个人”证书存储不能代替此机器级信任配置。
每台安装电脑都需要信任一次；后续构建复用同一证书时，无需重复导入。

## 4. 验证签名并安装

信任证书后，可在构建电脑或安装电脑使用 Windows SDK 的 SignTool 验证签名：

```powershell
# 修改为本次安装包和 SignTool 的实际路径。
& 'D:\path\signtool.exe' verify /pa /v '.\Hermes.msix'
if ($LASTEXITCODE -ne 0) { throw '签名验证失败，请查看输出。' }
```

在日常登录用户的**普通 PowerShell**中安装本次 MSIX：

```powershell
Add-AppxPackage -Path '.\Hermes.msix'
```

请将 `Hermes.msix` 替换为实际文件名。
如出现 `0x800B0109` 或无法验证发布者，检查导入位置、证书指纹和有效期。
企业设备还可能有 MSIX 侧载或应用控制策略，需要管理员允许；证书信任不能覆盖这些策略。

自签名证书适用于本地或受控分发。面向公开发布、希望用户无需手动导入证书时，
应使用受信任的正式代码签名方案或 Microsoft Store 分发流程。

## 官方参考

- [Microsoft：创建用于包签名的证书](https://learn.microsoft.com/windows/msix/package/create-certificate-package-signing)
- [Microsoft：MSIX 签名完整指南](https://learn.microsoft.com/windows/msix/package/sign-msix-package-guide)
