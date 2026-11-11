<#
.SYNOPSIS
Selects the SSH route to the 5090 server without opening an SSH session.

.DESCRIPTION
Probes the server's LAN SSH port. Returns a PowerShell object that later
scripts can inspect. The Cloudflare route requires a client-side cloudflared
ProxyCommand; the hostname is not a plain TCP/22 SSH endpoint.

.EXAMPLE
.\select-ssh-route.ps1

.EXAMPLE
$route = .\select-ssh-route.ps1
if ($route.Route -eq 'LAN') { $route.Target }
#>
[CmdletBinding()]
param(
    [string]$LocalAddress = '10.199.161.62',
    [string]$PublicDomain = 'ssh.jerrycloud.work',
    [ValidateRange(1, 65535)][int]$Port = 22,
    [ValidateRange(100, 30000)][int]$TimeoutMs = 1500
)

function Test-TcpReachable {
    param(
        [string]$Address,
        [int]$Port,
        [int]$TimeoutMs
    )

    $client = [System.Net.Sockets.TcpClient]::new()
    $operation = $null
    try {
        $operation = $client.BeginConnect($Address, $Port, $null, $null)
        if (-not $operation.AsyncWaitHandle.WaitOne($TimeoutMs)) {
            return $false
        }
        $client.EndConnect($operation)
        return $client.Connected
    }
    catch {
        return $false
    }
    finally {
        if ($null -ne $operation) {
            $operation.AsyncWaitHandle.Close()
        }
        $client.Close()
    }
}

$lanAvailable = Test-TcpReachable -Address $LocalAddress -Port $Port -TimeoutMs $TimeoutMs

if ($lanAvailable) {
    [pscustomobject]@{
        Route             = 'LAN'
        Target            = $LocalAddress
        Port              = $Port
        LocalTcpReachable = $true
        ClientMethod      = 'DirectSSH'
    }
}
else {
    [pscustomobject]@{
        Route             = 'CloudflareTunnel'
        Target            = $PublicDomain
        Port              = $Port
        LocalTcpReachable = $false
        ClientMethod      = 'CloudflaredProxyCommand'
    }
}
