# test_service.ps1 - Verify InferLite Windows service support.
#
# Checks:
#   1. Console mode still starts and serves HTTP (regression).
#   2. `--service` launched manually (not under SCM) falls back to a normal
#      foreground console run (so it never silently exits in a cmd window).
#   3. `--install-service` / `--uninstall-service` wired up (requires admin to
#      actually run; skipped if not elevated).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/test_service.ps1
param(
    [string]$ModelRepository,
    [int]$HttpPort = 8801
)
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Exe = Join-Path $RepoRoot "build\inferlite.exe"
if (-not (Test-Path $Exe)) { throw "inferlite.exe not found: $Exe" }

# Default repo: build a clean single-model repo under build/test-repo so the
# test does not depend on the (plugin/ensemble-heavy) models/ tree.
if (-not $ModelRepository) {
    $ModelRepository = Join-Path $RepoRoot "build\test-repo"
    if (-not (Test-Path $ModelRepository)) {
        New-Item -ItemType Directory -Path $ModelRepository -Force | Out-Null
        Copy-Item -Recurse -Path (Join-Path $RepoRoot "models\sample_model") -Destination $ModelRepository
    }
}

function Start-Server {
    param([string[]]$ServerArgs)
    $diag = Join-Path $RepoRoot "build\test_service_diag.log"
    $out = Join-Path $RepoRoot "build\test_service_out.log"
    $err = Join-Path $RepoRoot "build\test_service_err.log"
    $p = Start-Process -FilePath $Exe -ArgumentList $ServerArgs -PassThru -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden
    Start-Sleep -Seconds 8
    if ($p.HasExited) {
        Write-Host "SERVER EXITED EARLY (code $($p.ExitCode)). stderr:"
        Get-Content $err -ErrorAction SilentlyContinue
        return $null
    }
    return $p
}

function Test-Health {
    param([int]$Port)
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/v2/health/ready" -UseBasicParsing -TimeoutSec 5
        Write-Host ("  /v2/health/ready -> HTTP " + $r.StatusCode + " " + $r.Content)
        return ($r.StatusCode -eq 200)
    } catch {
        Write-Host ("  /v2/health/ready -> FAIL: " + $_.Exception.Message)
        return $false
    }
}

$allPass = $true

Write-Host "=== [1] Console mode (regression) ==="
$p = Start-Server @("--model-repository=$ModelRepository", "--http-port=$HttpPort")
if (-not $p) { $allPass = $false }
else {
    $ok = Test-Health -Port $HttpPort
    if (-not $ok) { $allPass = $false }
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
}

Write-Host "=== [2] --service manual fallback (no SCM) ==="
$p2 = Start-Server @("--service", "--model-repository=$ModelRepository", "--http-port=8802")
if (-not $p2) { $allPass = $false }
else {
    $ok = Test-Health -Port 8802
    if (-not $ok) { $allPass = $false }
    Stop-Process -Id $p2.Id -Force -ErrorAction SilentlyContinue
}

Write-Host "=== [3] install/uninstall (admin only) ==="
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if ($isAdmin) {
    Write-Host "  elevated: installing test service (port 8803)..."
    & $Exe "--install-service" "--service-name=InferLiteTest" "--service-display=InferLite Test" "--model-repository=$ModelRepository" "--http-port=8803"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  service installed. starting under SCM (port 8803)..."
        sc.exe start InferLiteTest | Out-Null
        Start-Sleep -Seconds 10
        $svcOk = Test-Health -Port 8803
        if (-not $svcOk) {
            Write-Host "  SCM-run service did not become READY." 
            Write-Host "  --- service event log tail (InferLite) ---"
            Get-WinEvent -LogName System -MaxEvents 20 -ErrorAction SilentlyContinue |
                Where-Object { $_.Message -match "InferLite" } |
                Select-Object -First 5 | ForEach-Object { Write-Host $_.Message }
            $allPass = $false
        } else {
            Write-Host "  stopping service (graceful)..."
            sc.exe stop InferLiteTest | Out-Null
            Start-Sleep -Seconds 4
            $st = (sc.exe query InferLiteTest) -join "`n"
            Write-Host $st
            if ($st -match "STOPPED") { Write-Host "  service stopped cleanly." }
            else { Write-Host "  WARN: service not STOPPED after stop; forcing uninstall may still work." }
        }
        & $Exe "--uninstall-service" "--service-name=InferLiteTest"
        Write-Host "  service uninstalled."
    } else {
        Write-Host "  install returned $LASTEXITCODE (not fatal)."
    }
} else {
    Write-Host "  not elevated; skipping install/uninstall (run as admin to test)."
}

if ($allPass) { Write-Host "`nPASS: console + service fallback OK" }
else { Write-Host "`nFAIL" }
exit ($(if ($allPass) { 0 } else { 1 }))
