# service.ps1 - Manage the InferLite Windows service.
#
# Usage (run from an ELEVATED PowerShell for install/uninstall):
#   powershell -ExecutionPolicy Bypass -File scripts/service.ps1 -Action install `
#       -ModelRepository c:\Test\triton\inferlite\models\sample_model `
#       -HttpPort 8000
#   powershell -ExecutionPolicy Bypass -File scripts/service.ps1 -Action start
#   powershell -ExecutionPolicy Bypass -File scripts/service.ps1 -Action status
#   powershell -ExecutionPolicy Bypass -File scripts/service.ps1 -Action stop
#   powershell -ExecutionPolicy Bypass -File scripts/service.ps1 -Action uninstall
#
# Actions:
#   install     Create the Windows service "InferLite" running inferlite.exe
#               in --service mode. Requires an elevated prompt.
#   uninstall   Delete the service. Requires an elevated prompt.
#   start       Start the service (sc start).
#   stop        Stop the service (sc stop).
#   restart     Stop then start.
#   status      Query the service state and the health endpoint.
#
# The service is auto-start at boot and runs as LocalSystem by default. To run
# under a specific account, pass -ServiceUser and -ServicePassword.
param(
    [ValidateSet("install", "uninstall", "start", "stop", "restart", "status")]
    [string]$Action = "status",
    [string]$ServiceName = "InferLite",
    [string]$ModelRepository,
    [int]$HttpPort = 8100,
    [int]$GrpcPort = 0,
    [string]$HostAddr = "0.0.0.0",
    [switch]$Validated,
    [string]$AuditLog,
    [string]$DiagnosticLog,
    [string]$TlsCert,
    [string]$TlsKey,
    [string]$ServiceUser,
    [string]$ServicePassword,
    [string]$ExePath = ""
)
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $ExePath) { $ExePath = Join-Path $RepoRoot "build\inferlite.exe" }
if (-not (Test-Path $ExePath)) {
    throw "inferlite.exe not found at $ExePath. Build first (scripts/build.ps1 or scripts/build_grpc.ps1)."
}
$ExePath = (Resolve-Path $ExePath).Path

function Invoke-Sc {
    param([string]$ArgsLine)
    $out = sc.exe $ArgsLine 2>&1
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        Write-Host "sc $ArgsLine`n$($out -join "`n")" -ForegroundColor Red
        throw "sc command failed (exit $code)."
    }
    return $out
}

switch ($Action) {
    "install" {
        if (-not $ModelRepository) {
            throw "-ModelRepository is required for install."
        }
        # Build the server argument string (everything after --service).
        $args = @()
        $args += "--model-repository=$ModelRepository"
        $args += "--host=$HostAddr"
        $args += "--http-port=$HttpPort"
        if ($GrpcPort -gt 0) { $args += "--grpc-port=$GrpcPort" }
        if ($Validated) { $args += "--validated-mode" }
        if ($AuditLog)  { $args += "--audit-log=$AuditLog" }
        if ($DiagnosticLog) { $args += "--diagnostic-log=$DiagnosticLog" }
        if ($TlsCert)   { $args += "--tls-cert=$TlsCert" }
        if ($TlsKey)    { $args += "--tls-key=$TlsKey" }
        $argStr = $args -join " "

        $installArgs = @(
            "--install-service",
            "--service-name=$ServiceName",
            "--service-display=InferLite Inference Server"
        ) + $args
        if ($ServiceUser) {
            $installArgs += "--install-service-user=$ServiceUser"
            $installArgs += "--install-service-password=$ServicePassword"
        }
        Write-Host "Installing service '$ServiceName'"
        & $ExePath $installArgs
        if ($LASTEXITCODE -ne 0) { throw "install failed (exit $LASTEXITCODE)." }
        Write-Host "Service '$ServiceName' installed. Start it with: $PSScriptRoot\service.ps1 -Action start"
    }
    "uninstall" {
        Invoke-Sc "delete $ServiceName"
        Write-Host "Service '$ServiceName' uninstalled."
    }
    "start" {
        Invoke-Sc "start $ServiceName"
        Start-Sleep -Seconds 3
        Write-Host "Service '$ServiceName' started."
    }
    "stop" {
        Invoke-Sc "stop $ServiceName"
        Write-Host "Service '$ServiceName' stopped."
    }
    "restart" {
        Invoke-Sc "stop $ServiceName" 2>$null
        Start-Sleep -Seconds 2
        Invoke-Sc "start $ServiceName"
        Start-Sleep -Seconds 3
        Write-Host "Service '$ServiceName' restarted."
    }
    "status" {
        $out = Invoke-Sc "query $ServiceName"
        Write-Host ($out -join "`n")
        # Also probe the health endpoint if the port is reachable.
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:$HttpPort/v2/health/ready" -UseBasicParsing -TimeoutSec 3
            Write-Host ("Health /v2/health/ready -> HTTP " + $r.StatusCode + " " + $r.Content)
        } catch {
            Write-Host "Health /v2/health/ready -> not reachable on port $HttpPort."
        }
    }
}
