<#
.SYNOPSIS
Upload CAFAD experiment code using the LAN rclone remote.
.DESCRIPTION
Preserves paths relative to the local project root under the remote cafad root.
Uploads .py/.sh files in scripts, src, test and tests, the Web console static
assets under src/web_console/static, versioned JSON/YAML protocol files under
configs, plus root requirements.txt, .env and main.py.
Uses copy; server datasets and logs are preserved.
Transfer logs and the exact file list are written under PROJECT_ROOT/logs.
The scripts are standalone and work in Windows PowerShell 5.1 and PowerShell 7.
.PARAMETER ProjectRoot
Defaults to the parent of this script's directory.
.PARAMETER DryRun
Ask rclone to preview the upload. This still connects to the configured remote.
.PARAMETER ListOnly
Print the exact file list without contacting the server.
.EXAMPLE
.\scripts\Upload-Code-LAN.ps1
.EXAMPLE
.\scripts\Upload-Code-LAN.ps1 -DryRun
.EXAMPLE
.\scripts\Upload-Code-LAN.ps1 -ListOnly
.NOTES
Remote: lab_rtx5090_server_lan:cafad (the server user's home-relative cafad directory).
The server must be reachable over the LAN at 10.199.161.62:22.
Data, logs, .venv, documents, .ps1 files and nested requirements.txt/.env files
are outside the upload whitelist. The root .env is uploaded to cafad/.env.
Background experiment stdout/stderr and application file logs must be directed
to ~/cafad/logs. Syncthing handles log synchronization separately.
#>
[CmdletBinding()]
param(
    [Parameter()]
    [string]$ProjectRoot,

    [Parameter()]
    [switch]$DryRun,

    [Parameter()]
    [switch]$ListOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
# Handle rclone's exit status explicitly, including when the caller uses PS 7.
$PSNativeCommandUseErrorActionPreference = $false

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}

if ($DryRun -and $ListOnly) {
    throw 'Choose either -DryRun or -ListOnly.'
}
$rootItem = Get-Item -LiteralPath $ProjectRoot
if (-not $rootItem.PSIsContainer -or $rootItem.PSProvider.Name -ne 'FileSystem') {
    throw 'ProjectRoot must be a filesystem directory.'
}
$sourceRoot = $rootItem.FullName.TrimEnd([char[]]@('\', '/'))
$remoteName = 'lab_rtx5090_server_lan'
$destination = $remoteName + ':cafad'
$logRoot = Join-Path $sourceRoot 'logs'
[System.IO.Directory]::CreateDirectory($logRoot) | Out-Null
$runId = (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + '-' + ([guid]::NewGuid().ToString('N').Substring(0, 6))
$runPrefix = Join-Path $logRoot ('upload-code-lan-' + $runId)
$logPath = $runPrefix + '.log'
$fileListPath = $runPrefix + '.files.txt'
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)

function Write-UploadLog {
    param([Parameter(Mandatory = $true)][string]$Message)
    $line = (Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz') + ' ' + $Message
    [System.IO.File]::AppendAllText($logPath, $line + [Environment]::NewLine, $utf8NoBom)
    Write-Host $Message
}

try {
    Write-UploadLog "Source: $sourceRoot"
    Write-UploadLog "Destination: $destination"
    Write-UploadLog "Log: $logPath"

    $requiredRootFiles = @('requirements.txt', '.env', 'main.py')
    foreach ($fileName in $requiredRootFiles) {
        $requiredFile = Join-Path $sourceRoot $fileName
        if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
            throw "Required file is missing: $requiredFile"
        }
        if ((Get-Item -LiteralPath $requiredFile -Force).Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
            throw "$fileName must be a regular file, not a symbolic link."
        }
    }

    $excludedDirectories = @(
        '__pycache__', '.git', '.venv', 'venv', '.pytest_cache',
        '.mypy_cache', '.ruff_cache', '.tox', '.nox', '.ipynb_checkpoints'
    )
    $relativeFiles = [System.Collections.Generic.List[string]]::new()
    foreach ($fileName in $requiredRootFiles) {
        $relativeFiles.Add($fileName)
    }
    $pending = [System.Collections.Generic.Stack[string]]::new()
    foreach ($directoryName in @('scripts', 'src', 'test', 'tests', 'configs')) {
        $directory = Join-Path $sourceRoot $directoryName
        if (Test-Path -LiteralPath $directory -PathType Container) {
            $directoryItem = Get-Item -LiteralPath $directory
            if ($directoryItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
                throw "Source directory must not be a symbolic link: $directory"
            }
            $pending.Push($directory)
        }
    }
    while ($pending.Count -gt 0) {
        $currentDirectory = $pending.Pop()
        foreach ($item in Get-ChildItem -LiteralPath $currentDirectory -Force) {
            if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
                continue
            }
            if ($item.PSIsContainer) {
                if ($item.Name -notin $excludedDirectories) {
                    $pending.Push($item.FullName)
                }
                continue
            }
            $relativePath = $item.FullName.Substring($sourceRoot.Length + 1).Replace('\', '/')
            $isWebStaticAsset = $relativePath -like 'src/web_console/static/*' -and $item.Extension.ToLowerInvariant() -in @('.html', '.js', '.css')
            $isProtocolConfig = $relativePath -like 'configs/*' -and $item.Extension.ToLowerInvariant() -in @('.json', '.yaml', '.yml')
            if ($item.Extension.ToLowerInvariant() -notin @('.py', '.sh') -and -not $isWebStaticAsset -and -not $isProtocolConfig) {
                continue
            }
            if ($relativePath -match '[\r\n]') {
                throw "A source filename contains a newline: $($item.FullName)"
            }
            $relativeFiles.Add($relativePath)
        }
    }

    [string[]]$filesToUpload = @($relativeFiles | Sort-Object -Unique)
    if ($filesToUpload.Count -le $requiredRootFiles.Count) {
        throw 'No experiment .py/.sh files were found; check ProjectRoot.'
    }
    # No BOM and no comment parsing: preserves literal relative paths exactly.
    [System.IO.File]::WriteAllLines($fileListPath, $filesToUpload, $utf8NoBom)
    Write-UploadLog "Selected $($filesToUpload.Count) files. File list: $fileListPath"

    if ($ListOnly) {
        foreach ($relativeFile in $filesToUpload) {
            Write-Output $relativeFile
        }
        Write-UploadLog 'LIST ONLY completed; no server connection was made.'
        return
    }

    $rclone = Get-Command rclone.exe -CommandType Application -ErrorAction Stop
    $rcloneArguments = @(
        'copy', $sourceRoot, $destination,
        '--files-from-raw', $fileListPath,
        '--transfers', '4',
        '--checkers', '8',
        '--contimeout', '15s',
        '--timeout', '60s',
        '--retries', '2',
        '--low-level-retries', '2',
        '--log-file', $logPath,
        '--log-level', 'INFO',
        '--stats', '10s',
        '--stats-one-line'
    )
    if ($DryRun) {
        $rcloneArguments += '--dry-run'
        Write-UploadLog 'DRY RUN: rclone will preview changes.'
    }
    Write-UploadLog 'LAN endpoint: 10.199.161.62:22.'
    & $rclone.Source @rcloneArguments
    $rcloneExitCode = $LASTEXITCODE
    if ($rcloneExitCode -ne 0) {
        throw "rclone failed with exit code $rcloneExitCode. See $logPath"
    }
    if ($DryRun) {
        Write-UploadLog 'DRY RUN completed successfully.'
    } else {
        Write-UploadLog "Upload completed successfully: $destination"
    }
} catch {
    Write-UploadLog ('ERROR: ' + $_.Exception.Message)
    throw
}
