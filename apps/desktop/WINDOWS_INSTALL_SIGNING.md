# Windows MSIX 签名与 ZIP 打包

本文用于每次 Windows x64 构建完成后，重复执行本地签名与打包。
使用自签名证书；安装电脑需要预先信任对应公钥证书。

## 一条命令：构建、签名和 ZIP（Semantier / WSL）

在 Semantier 父仓库根目录运行：

```bash
python3 scripts/build_hermes_windows.py --thumbprint EXPLICIT_CERTIFICATE_THUMBPRINT
```

该命令使用 Windows 原生 PowerShell、Git 和依赖，将精确的已提交 Hermes HEAD 通过本地 bundle 传输到独立 Windows 工作树，调用现有 `scripts\build-bundle.ps1`，再调用本文的签名脚本。保留原始 Windows 检出及其未提交工作、专用工作树缓存、未签名 MSIX 和收据。无需远程 push。每次输出使用新目录，失败会留下日志，证书指纹必须明确指定。

`--preflight` 仅检查传输、证书和工具；`--unsigned-receipt 'D:\path\unsigned-build-receipt.json'` 校验已有构建收据及散列后，仅重新签名/打包，不重新构建。完整构建仍沿用原有干净打包流程。详细参数、缓存及清理说明见父仓库 `docs/operational/hermes-windows-build.md`。签名脚本直接读取 MSIX 内的清单，无需解包整个运行时。

本地 Semantier 完整构建的界面名称为 `Semantier Agent <版本>`，可执行文件为 `Semantier Agent.exe`，分发 ZIP 为 `SemantierAgent-<版本>-win-<架构>.zip`。版本从 `0.0.1` 开始，每次正式构建尝试自动递增补丁号，保存在工作目录旁的 `D:\repo\semantier-agent-build-version.json`；失败的正式构建也消耗编号。预检查和仅重新签名不递增。保留此计数器，不因清理构建输出而重置。MSIX 使用对应四段版本（例如 `0.0.1.0`），上游 Python 运行时版本仍独立记录。名称定制不改变包 Identity、ApplicationID、Publisher 或内部状态隔离名称。

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
  -TextExtension @('2.5.29.19={text}', '2.5.29.37={text}1.3.6.1.5.5.7.3.3') `
  -NotAfter (Get-Date).AddYears(2)

$cert.Thumbprint
```

保存输出的证书指纹。证书及私钥保存在当前用户的 Windows 证书存储中；
以后应以同一个 Windows 用户进行签名，不必每次重新创建证书。
证书过期或丢失后，需要换用新证书，并在安装电脑上更新信任。

此前构建中临时生成并已删除私钥的证书，不能仅凭 `.cer` 文件再次签名。
`.cer` 只包含公钥，不包含私钥。

## 2. 每次构建后签名并打包

先检查当前用户已有证书及私钥、有效期、代码签名 EKU 和包清单的 Publisher。优先复用有效且用途合适的证书；有多个候选时必须通过已记录的指纹或既有信任依据明确选择，不能猜测。仅在没有可用证书时创建一次。创建示例的 Subject 只是参考，实际必须与本次 MSIX 清单的 `Identity Publisher` 逐字相同。界面品牌不改变包标识或发布者。

使用仓库提供的脚本（所有路径均填写本次实际值）：

```powershell
& .\apps\desktop\scripts\sign-windows-distribution.ps1 `
  -Msix 'D:\path\SemantierAgent-0.0.1.msix' `
  -Thumbprint 'EXPLICIT_CERTIFICATE_THUMBPRINT' `
  -SourceCommit 'COMPLETE_40_CHARACTER_HERMES_COMMIT' `
  -SignTool 'D:\path\signtool.exe' `
  -SevenZip 'D:\path\7za.exe' `
  -Output 'D:\path\new-distribution'
if ($LASTEXITCODE -ne 0) { throw '分发失败' }
```

脚本保留原始未签名 MSIX 及构建收据，只签署新的分发副本。它先解包清单验证 Publisher/版本/架构，签名后必须在**已有信任**的环境中通过 SignTool `verify /pa /v` 才允许组装 ZIP。不会导入证书或改变机器信任。信任或私钥权限不足时停止并报告具体原因，由用户决定最小必要干预；不能通过自动导入证书使检查变绿。

分发目录包含签名 MSIX、公钥 `.cer`、SHA256SUMS.txt 和 build-metadata.json。元数据绑定完整源提交、包名称/版本/架构、证书指纹/有效期、签名/验证结果、是否使用可信时间戳。所有散列在签名后生成。脚本校验 ZIP 完整性、提取后的文件集合和每项散列，再生成 ZIP 的 SHA-256 边车文件。禁止私钥、PFX、密码或凭据进入分发目录。

可选 `-TimestampUrl` 使用 RFC 3161 时间戳；不提供时明确记录 `trustedTimestamp: false`。包签名覆盖包内容，不等于为每个内部 EXE/DLL 添加 Authenticode 签名。未使用可信时间戳时，不保证证书到期后的安装验证。

## 3. 接收方安装电脑首次信任证书（本地构建流程不执行）

先通过可信渠道核对 `.cer` 的证书指纹：

```powershell
Get-PfxCertificate -FilePath '.\SemantierAgent-signing.cer' |
    Format-List Subject, Thumbprint, NotBefore, NotAfter
```

确认指纹后，在安装电脑的**管理员 PowerShell**中执行：

```powershell
Import-Certificate -FilePath '.\SemantierAgent-signing.cer' `
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
