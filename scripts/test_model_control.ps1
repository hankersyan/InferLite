# test_model_control.ps1 - Exercise the Triton-style model-management modes
# (none / poll / explicit) and the repository-control endpoints over HTTP.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File test_model_control.ps1
#
# Requires a gRPC/CPU build (build-grpc\inferlite.exe) as produced by
# build_grpc.ps1. Uses the two OpenVINO CPU models checked into models/
# (sample_model, intel_cpu_model).
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Exe = Join-Path $RepoRoot "build-grpc\inferlite.exe"
$Work = Join-Path $RepoRoot "build-grpc\mmtest"
$Port = 8010

if (-not (Test-Path $Exe)) { throw "not found: $Exe (run scripts\build_grpc.ps1 first)" }

$pass = 0
$fail = 0

function Check($name, $cond) {
    if ($cond) {
        Write-Output "PASS  $name"
        $script:pass++
    } else {
        Write-Output "FAIL  $name"
        $script:fail++
    }
}

function Call-Server($method, $uri, $jsonBody) {
    $req = [System.Net.HttpWebRequest]::Create($uri)
    $req.Method = $method
    $req.ContentType = "application/json"
    $req.Timeout = 20000
    if ($jsonBody) {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($jsonBody)
        $req.ContentLength = $bytes.Length
        $s = $req.GetRequestStream()
        $s.Write($bytes, 0, $bytes.Length)
        $s.Close()
    }
    try {
        $resp = $req.GetResponse()
        $sr = New-Object IO.StreamReader($resp.GetResponseStream())
        return [pscustomobject]@{ Status = [int]$resp.StatusCode; Body = $sr.ReadToEnd() }
    } catch [System.Net.WebException] {
        $er = $_.Exception.Response
        if ($er) {
            $sr = New-Object IO.StreamReader($er.GetResponseStream())
            return [pscustomobject]@{ Status = [int]$er.StatusCode; Body = $sr.ReadToEnd() }
        }
        return [pscustomobject]@{ Status = 0; Body = $_.Exception.Message }
    }
}

function Start-TestServer([string[]]$extraArgs) {
    Get-Process inferlite -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Milliseconds 600
    $dir = Join-Path $RepoRoot "build-grpc"
    $p = Start-Process -FilePath $Exe -WorkingDirectory $dir `
        -ArgumentList (@("--model-repository=$Work\repo", "--http-port=$Port",
                         "--max-queue-size=100", "--http-threads=4") + $extraArgs) `
        -PassThru -RedirectStandardOutput "$Work\srv.out" -RedirectStandardError "$Work\srv.err"
    # Wait for readiness.
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Milliseconds 500
        if ($p.HasExited) {
            Write-Output "SERVER EXITED code=$($p.ExitCode)"
            Get-Content "$Work\srv.err" -ErrorAction SilentlyContinue | Select-Object -Last 20
            throw "server failed to start"
        }
        $r = Call-Server "GET" "http://127.0.0.1:$Port/v2/health/ready"
        if ($r.Status -eq 200) { return $p }
    }
    Stop-Process -Name inferlite -Force
    throw "server not ready in time"
}

function Stop-TestServer {
    Get-Process inferlite -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Milliseconds 500
}

# --- prepare a clean test repository ----------------------------------------
if (Test-Path $Work) { Remove-Item -Recurse -Force $Work }
New-Item -ItemType Directory -Force -Path $Work | Out-Null
New-Item -ItemType Directory -Force -Path "$Work\repo" | Out-Null
Copy-Item -Recurse "$RepoRoot\models\sample_model" "$Work\repo\sample_model"
Copy-Item -Recurse "$RepoRoot\models\intel_cpu_model" "$Work\repo\intel_cpu_model"

$indexUri = "http://127.0.0.1:$Port/v2/repository/index"

# ================= Scenario 1: mode = none ==================================
Write-Output "`n== Scenario 1: model-control-mode=none =="
Start-TestServer @("--model-control-mode=none")
$idx = Call-Server "POST" $indexUri "{}"
Check "none: index lists both models" ($idx.Status -eq 200 -and $idx.Body -match 'sample_model' -and $idx.Body -match 'intel_cpu_model' -and $idx.Body -match '"READY"')
$load = Call-Server "POST" "http://127.0.0.1:$Port/v2/repository/models/intel_cpu_model/load" "{}"
Check "none: load rejected (400)" ($load.Status -eq 400 -and $load.Body -match 'explicit')
$unload = Call-Server "POST" "http://127.0.0.1:$Port/v2/repository/models/sample_model/unload" "{}"
Check "none: unload rejected (400)" ($unload.Status -eq 400)
$inf = Call-Server "POST" "http://127.0.0.1:$Port/v2/models/sample_model/infer" '{"inputs":[{"name":"input","shape":[1,4],"datatype":"FP32","data":[1,2,3,4]}]}'
Check "none: inference works" ($inf.Status -eq 200 -and $inf.Body -match 'outputs')
Stop-TestServer

# ================= Scenario 2: mode = poll ==================================
Write-Output "`n== Scenario 2: model-control-mode=poll (hot add / remove) =="
# Repository starts with only sample_model.
Remove-Item -Recurse -Force "$Work\repo\intel_cpu_model"
Start-TestServer @("--model-control-mode=poll", "--repository-poll-secs=2")
$idx = Call-Server "POST" $indexUri "{}"
Check "poll: startup index has sample_model only" ($idx.Body -match 'sample_model' -and $idx.Body -notmatch 'intel_cpu_model')
$load = Call-Server "POST" "http://127.0.0.1:$Port/v2/repository/models/sample_model/load" "{}"
Check "poll: load rejected (400)" ($load.Status -eq 400)

# Hot-add intel_cpu_model and wait for the poller to pick it up.
Copy-Item -Recurse "$RepoRoot\models\intel_cpu_model" "$Work\repo\intel_cpu_model"
$added = $false
for ($i = 0; $i -lt 15; $i++) {
    Start-Sleep -Seconds 1
    $idx = Call-Server "POST" $indexUri '{"ready":true}'
    if ($idx.Body -match 'intel_cpu_model') { $added = $true; break }
}
Check "poll: hot-added model became READY" $added
$inf = Call-Server "POST" "http://127.0.0.1:$Port/v2/models/intel_cpu_model/infer" '{"inputs":[{"name":"input","shape":[1,4],"datatype":"FP32","data":[5,6,7,8]}]}'
Check "poll: inference on hot-added model" ($inf.Status -eq 200 -and $inf.Body -match 'outputs')

# Hot-remove intel_cpu_model and wait for the poller to drop it.
Remove-Item -Recurse -Force "$Work\repo\intel_cpu_model"
$removed = $false
for ($i = 0; $i -lt 15; $i++) {
    Start-Sleep -Seconds 1
    $idx = Call-Server "POST" $indexUri "{}"
    if ($idx.Body -notmatch 'intel_cpu_model') { $removed = $true; break }
}
Check "poll: removed model gone from index" $removed
$inf = Call-Server "POST" "http://127.0.0.1:$Port/v2/models/intel_cpu_model/infer" '{"inputs":[{"name":"input","shape":[1,4],"datatype":"FP32","data":[1,1,1,1]}]}'
Check "poll: inference on removed model returns 404" ($inf.Status -eq 404)
Stop-TestServer

# ================= Scenario 3: mode = explicit ===============================
Write-Output "`n== Scenario 3: model-control-mode=explicit (load/unload API) =="
Copy-Item -Recurse "$RepoRoot\models\intel_cpu_model" "$Work\repo\intel_cpu_model"
Start-TestServer @("--model-control-mode=explicit")

$idx = Call-Server "POST" $indexUri "{}"
Check "explicit: startup index shows both UNAVAILABLE" ($idx.Body -match 'UNAVAILABLE' -and $idx.Body -match 'sample_model' -and $idx.Body -match 'intel_cpu_model')
$idxReady = Call-Server "POST" $indexUri '{"ready":true}'
Check "explicit: ready-only index empty at startup" ($idxReady.Body.Trim() -eq '[]')
$cfg0 = Call-Server "GET" "http://127.0.0.1:$Port/v2/models/sample_model/config"
Check "explicit: config of unloaded model 404" ($cfg0.Status -eq 404)

$load = Call-Server "POST" "http://127.0.0.1:$Port/v2/repository/models/sample_model/load" "{}"
Check "explicit: load sample_model ok" ($load.Status -eq 200)
$idxReady = Call-Server "POST" $indexUri '{"ready":true}'
Check "explicit: ready-only index has sample_model" ($idxReady.Body -match 'sample_model' -and $idxReady.Body -notmatch 'intel_cpu_model')
$inf = Call-Server "POST" "http://127.0.0.1:$Port/v2/models/sample_model/infer" '{"inputs":[{"name":"input","shape":[1,4],"datatype":"FP32","data":[1,2,3,4]}]}'
Check "explicit: inference on loaded model" ($inf.Status -eq 200)
$cfg = Call-Server "GET" "http://127.0.0.1:$Port/v2/models/sample_model/config"
Check "explicit: config of loaded model 200" ($cfg.Status -eq 200)

# Load with a config override (proto-text) - must still succeed.
$cfgText = 'name: "sample_model"' + "`n" + 'backend: "openvino"' + "`n" + 'max_batch_size: 0' + "`n" +
           'input { name: "input" data_type: TYPE_FP32 dims: [ 1, 4 ] }' + "`n" +
           'output { name: "add" data_type: TYPE_FP32 dims: [ 1, 4 ] }' + "`n" +
           'instance_group { count: 1 kind: KIND_CPU }'
$ovBody = @{ parameters = @{ config = $cfgText } } | ConvertTo-Json -Depth 5
$ov = Call-Server "POST" "http://127.0.0.1:$Port/v2/repository/models/sample_model/load" $ovBody
Check "explicit: load with config override ok" ($ov.Status -eq 200)

$load2 = Call-Server "POST" "http://127.0.0.1:$Port/v2/repository/models/nonexistent/load" "{}"
Check "explicit: load of unknown model 404" ($load2.Status -eq 404)

$unload = Call-Server "POST" "http://127.0.0.1:$Port/v2/repository/models/sample_model/unload" "{}"
Check "explicit: unload sample_model ok" ($unload.Status -eq 200)
$idx = Call-Server "POST" $indexUri "{}"
Check "explicit: sample_model back to UNAVAILABLE" ($idx.Body -match 'sample_model' -and $idx.Body -match 'UNAVAILABLE')
$inf = Call-Server "POST" "http://127.0.0.1:$Port/v2/models/sample_model/infer" '{"inputs":[{"name":"input","shape":[1,4],"datatype":"FP32","data":[1,2,3,4]}]}'
Check "explicit: inference after unload 404" ($inf.Status -eq 404)
$unload2 = Call-Server "POST" "http://127.0.0.1:$Port/v2/repository/models/ghost/unload" "{}"
Check "explicit: unload of unknown model 404" ($unload2.Status -eq 404)
Stop-TestServer

# ================= Scenario 4: explicit + --load-model =======================
Write-Output "`n== Scenario 4: explicit with --load-model=sample_model =="
Start-TestServer @("--model-control-mode=explicit", "--load-model=sample_model")
$idxReady = Call-Server "POST" $indexUri '{"ready":true}'
Check "explicit+load-model: only sample_model READY" ($idxReady.Body -match 'sample_model' -and $idxReady.Body -notmatch 'intel_cpu_model')
Stop-TestServer

Write-Output "`n======================================="
Write-Output "RESULTS: pass=$pass fail=$fail"
if ($fail -gt 0) { exit 1 }
exit 0
