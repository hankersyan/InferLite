# test_rate_limiter.ps1 - Triton-style cross-model rate limiter integration test.
#
# Builds a scratch model repository of CPU plugin models that sleep a
# configurable amount per execution (sample_plugin `sleep_ms`), then verifies:
#   1. With --rate-limit, two models that both declare the shared resource
#      "GPU_MEMORY" (count 1, global) are SERIALIZED: two concurrent requests
#      take ~2x the per-execution sleep instead of ~1x.
#   2. Without --rate-limit the same two models run concurrently (~1x sleep).
#   3. Models that declare no rate-limiter resources are never blocked by
#      models that do (they keep running concurrently either way).
#   4. /v2/health/detailed reports the rate_limiter state + live pool usage.
#
# Requires a build with sample_plugin.dll next to inferlite.exe (build/ or
# build-grpc/). Pass -BuildDir to select a build directory.
param(
    [string]$BuildDirName = "build-grpc",
    [int]$Port = 8006
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$build = Join-Path $root $BuildDirName
$exe = "$build\inferlite.exe"
$plugin = "$build\sample_plugin.dll"
if (-not (Test-Path $exe)) { throw "inferlite.exe not found in $build" }
if (-not (Test-Path $plugin)) {
    throw "sample_plugin.dll not found in $build (needed for the plugin test models)"
}
$python = "C:\Apps\anaconda3\envs\test312\python.exe"
if (-not (Test-Path $python)) { throw "python not found: $python (test312 env)" }

$repo = Join-Path $root "temp\rate_limiter_test"
$clientPy = Join-Path $repo "rl_client.py"
Remove-Item $repo -Recurse -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $repo | Out-Null
$pluginPosix = $plugin.Replace('\', '/')

function New-PluginModel {
    param([string]$Name, [int]$SleepMs, [int]$Count = 1, [bool]$DeclareResource = $false)
    $dir = Join-Path $repo $Name
    New-Item -ItemType Directory -Path $dir | Out-Null
    $rl = ""
    if ($DeclareResource) {
        $rl = @"

    rate_limiter {
      priority: 1
      resources { name: "GPU_MEMORY" count: 1 global: true }
    }
"@
    }
    $cfg = "name: `"$Name`"
backend: `"plugin`"
max_batch_size: 0
plugin_library: `"$pluginPosix`"
input {
  name: `"raw`"
  data_type: TYPE_FP32
  dims: [ 1, 4 ]
}
output {
  name: `"out`"
  data_type: TYPE_FP32
  dims: [ 1, 4 ]
}
parameters { key: `"mode`" value { string_value: `"identity`" } }
parameters { key: `"sleep_ms`" value { string_value: `"$SleepMs`" } }
instance_group {
  count: $Count
  kind: KIND_CPU
$rl
}
"
    Set-Content -Path (Join-Path $dir "config.pbtxt") -Value $cfg -Encoding ascii
}

# Two models sharing one global resource unit (serialized when --rate-limit on)
# plus two unconstrained models (must never be serialized).
New-PluginModel -Name "rl_heavy_a" -SleepMs 600 -DeclareResource $true
New-PluginModel -Name "rl_heavy_b" -SleepMs 600 -DeclareResource $true
New-PluginModel -Name "rl_free_c" -SleepMs 250
New-PluginModel -Name "rl_free_d" -SleepMs 250

# Python harness: fire two models concurrently and report wall time + status.
$client = @"
import json, sys, threading, time, urllib.request
base = sys.argv[1]
def infer(model, timeout=20):
    body = json.dumps({"inputs": [{"name": "raw", "shape": [1, 4], "datatype": "FP32",
                                   "data": [1.0, 2.0, 3.0, 4.0]}]}).encode()
    req = urllib.request.Request(base + "/v2/models/" + model + "/infer", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, time.time() - t0
    except Exception as e:
        return 0, time.time() - t0
def pair(a, b):
    out = {}
    def run(n):
        out[n] = infer(n)
    ta = threading.Thread(target=run, args=(a,)); tb = threading.Thread(target=run, args=(b,))
    t0 = time.time(); ta.start(); tb.start(); ta.join(); tb.join()
    wall = time.time() - t0
    ok = all(out[n][0] == 200 for n in (a, b))
    print("%s|%s wall_ms=%.0f a_ms=%.0f b_ms=%.0f ok=%s" % (
        a, b, wall * 1000, out[a][1] * 1000, out[b][1] * 1000, ok))
    return wall, ok
pair("rl_heavy_a", "rl_heavy_b")
pair("rl_free_c", "rl_free_d")
"@
Set-Content -Path $clientPy -Value $client -Encoding utf8

function Run-Server {
    param([string]$RateLimitFlag)
    Get-Process inferlite -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Milliseconds 700
    $argStr = "--model-repository=$repo --http-port=$Port --max-queue-size=200 $RateLimitFlag"
    $p = Start-Process -FilePath $exe -ArgumentList $argStr -PassThru `
        -RedirectStandardOutput "$repo\out.txt" -RedirectStandardError "$repo\err.txt"
    for ($i = 0; $i -lt 30; ++$i) {
        Start-Sleep -Milliseconds 500
        if ($p.HasExited) { break }
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/v2/health/ready" -UseBasicParsing -TimeoutSec 2
            if ($r.StatusCode -eq 200) { return $p }
        } catch { }
    }
    Get-Content "$repo\err.txt" -Encoding utf8 -ErrorAction SilentlyContinue
    throw "server failed to become ready"
}

function Invoke-Rl {
    param([string]$PairName, [int]$MinWallMs, [int]$MaxWallMs)
    $line = (& $python $clientPy "http://127.0.0.1:$Port" 2>&1 | Where-Object { $_ -match $PairName }) -join " "
    if (-not $line) { throw "no result for $PairName" }
    Write-Output $line
    if ($line -match "ok=False") { throw "${PairName}: a request did not return 200" }
    if ($line -match "wall_ms=(\d+)") {
        $wall = [int]$Matches[1]
        if ($wall -lt $MinWallMs) {
            throw "${PairName}: wall ${wall}ms below expected ${MinWallMs}ms"
        }
        if ($wall -gt $MaxWallMs) {
            throw "${PairName}: unexpectedly slow (wall ${wall}ms > ${MaxWallMs}ms)"
        }
    } else {
        throw "could not parse wall time for ${PairName}"
    }
}

Write-Output ""
Write-Output "== A. Rate limiter ENABLED (--rate-limit) =="
$p = Run-Server -RateLimitFlag "--rate-limit"
try {
    $d = (Invoke-WebRequest -Uri "http://127.0.0.1:$Port/v2/health/detailed" -UseBasicParsing -TimeoutSec 5).Content | ConvertFrom-Json
    if (-not $d.rate_limiter.enabled) { throw "rate_limiter.enabled should be true" }
    $pool = $d.rate_limiter.pools | Where-Object { $_.resource -eq "GPU_MEMORY" -and $_.scope -eq "global" }
    if (-not $pool) { throw "GPU_MEMORY global pool missing in /v2/health/detailed" }
    Write-Output ("detailed: rate_limiter.enabled=true, GPU_MEMORY global capacity=" + $pool.capacity + " used=" + $pool.used)
    # Two resource-sharing models must be serialized (~2x sleep), and two
    # unconstrained models must stay concurrent (~1x sleep).
    Invoke-Rl -PairName "rl_heavy_a\|rl_heavy_b" -MinWallMs 900 -MaxWallMs 3000
    Invoke-Rl -PairName "rl_free_c\|rl_free_d" -MinWallMs 0 -MaxWallMs 700
} finally {
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 500
}

Write-Output ""
Write-Output "== B. Rate limiter DISABLED (default) =="
$p = Run-Server -RateLimitFlag ""
try {
    $d = (Invoke-WebRequest -Uri "http://127.0.0.1:$Port/v2/health/detailed" -UseBasicParsing -TimeoutSec 5).Content | ConvertFrom-Json
    if ($d.rate_limiter.enabled) { throw "rate_limiter.enabled should be false without --rate-limit" }
    # Without the flag the two resource-sharing models run concurrently (~1x).
    Invoke-Rl -PairName "rl_heavy_a\|rl_heavy_b" -MinWallMs 0 -MaxWallMs 900
} finally {
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
}

Write-Output ""
Write-Output "RESULT: PASS"
Get-Process inferlite -ErrorAction SilentlyContinue | Stop-Process -Force
