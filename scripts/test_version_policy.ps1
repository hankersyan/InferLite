# test_version_policy.ps1 - Exercise Triton version policies
# (version_policy { specific | latest | all }) and client-requested model
# versions over HTTP and gRPC.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File test_version_policy.ps1
#
# Requires the gRPC/CPU build (build-grpc\inferlite.exe) produced by
# build_grpc.ps1. Reuses the OpenVINO model artifacts under models/sample_model
# to synthesize a two-version model (1/ and 2/) in a scratch repository.
#
# Covered semantics:
#   * Absent policy (Triton default) == latest: the highest numeric version is
#     loaded and served.
#   * The KServe v2 versioned HTTP route /v2/models/<m>/versions/<v>/infer
#     serves only the loaded version and returns 404 for any other version
#     (version-pinning guarantee; never silently served from another version).
#   * Explicit load with a config override pinning version_policy.specific to
#     an older version is the API-driven rollback path.
#   * latest{num_versions:N>1} / all{} select the highest version (InferLite
#     deviation: one version served per model name).
#   * A specific pin to a version that does not exist fails the load (400).
#   * Invalid policies (num_versions < 1, two sub-policies, duplicate/non-
#     positive specific versions) are rejected by validation.
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Exe = Join-Path $RepoRoot "build-grpc\inferlite.exe"
$Work = Join-Path $RepoRoot "build-grpc\vptest"
$Port = 8020
$GrpcPort = 8021

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
    $p = Start-Process -FilePath $Exe -WorkingDirectory (Join-Path $RepoRoot "build-grpc") `
        -ArgumentList (@("--model-repository=$Work\repo", "--http-port=$Port",
                         "--max-queue-size=100", "--http-threads=4") + $extraArgs) `
        -PassThru -RedirectStandardOutput "$Work\srv.out" -RedirectStandardError "$Work\srv.err"
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

# Build a config.pbtxt for a model named $name with the base sample_model I/O
# and, when $policy is non-empty, an appended version_policy block.
function New-ModelConfig($name, $policy) {
    $base = Get-Content (Join-Path $RepoRoot "models\sample_model\config.pbtxt") -Raw
    $cfg = $base -replace 'sample_model', $name
    if ($policy) {
        $cfg = $cfg.TrimEnd() + "`nversion_policy { $policy }`n"
    }
    return $cfg
}

# JSON-escape a config.pbtxt document and wrap it in the load body
# { "parameters": { "config": "..." } }.
function New-LoadBody($cfgText) {
    $esc = $cfgText.Replace('\', '\\').Replace('"', '\"')
    $esc = $esc.Replace("`r`n", '\n').Replace("`n", '\n')
    return '{"parameters":{"config":"' + $esc + '"}}'
}

function New-CleanRepo {
    if (Test-Path $Work) { Remove-Item -Recurse -Force $Work }
    New-Item -ItemType Directory -Force -Path "$Work\repo\multi_model\1" | Out-Null
    New-Item -ItemType Directory -Force -Path "$Work\repo\multi_model\2" | Out-Null
    Copy-Item -Recurse "$RepoRoot\models\sample_model\1\*" "$Work\repo\multi_model\1"
    Copy-Item -Recurse "$RepoRoot\models\sample_model\1\*" "$Work\repo\multi_model\2"
    Copy-Item "$RepoRoot\models\sample_model\metadata.json" "$Work\repo\multi_model\metadata.json"
    # Default config: NO version_policy block (Triton default = latest).
    Set-Content -Path "$Work\repo\multi_model\config.pbtxt" -Value (New-ModelConfig "multi_model" "") -Encoding ascii
}

# Write a config.pbtxt with a version_policy block into the scratch repo.
function Set-RepoPolicy($policy) {
    Set-Content -Path "$Work\repo\multi_model\config.pbtxt" -Value (New-ModelConfig "multi_model" $policy) -Encoding ascii
}

$indexUri = "http://127.0.0.1:$Port/v2/repository/index"

# ================= Scenario 1: default policy (latest) =========================
Write-Output "`n== Scenario 1: no version_policy (Triton default latest) =="
New-CleanRepo
Start-TestServer @("--model-control-mode=none")
$idx = Call-Server "POST" $indexUri "{}"
Check "default: index reports version 2 READY" ($idx.Status -eq 200 -and $idx.Body -match '"name":"multi_model"' -and $idx.Body -match '"version":"2"' -and $idx.Body -match '"READY"')
$inf = Call-Server "POST" "http://127.0.0.1:$Port/v2/models/multi_model/infer" '{"inputs":[{"name":"input","shape":[1,4],"datatype":"FP32","data":[1,2,3,4]}]}'
Check "default: unversioned infer works" ($inf.Status -eq 200 -and $inf.Body -match 'outputs')
$i2 = Call-Server "POST" "http://127.0.0.1:$Port/v2/models/multi_model/versions/2/infer" '{"inputs":[{"name":"input","shape":[1,4],"datatype":"FP32","data":[1,2,3,4]}]}'
Check "default: /versions/2/infer works (200)" ($i2.Status -eq 200 -and $i2.Body -match 'outputs')
$i1 = Call-Server "POST" "http://127.0.0.1:$Port/v2/models/multi_model/versions/1/infer" '{"inputs":[{"name":"input","shape":[1,4],"datatype":"FP32","data":[1,2,3,4]}]}'
Check "default: /versions/1/infer rejected (404)" ($i1.Status -eq 404)
$i9 = Call-Server "POST" "http://127.0.0.1:$Port/v2/models/multi_model/versions/9/infer" '{"inputs":[{"name":"input","shape":[1,4],"datatype":"FP32","data":[1,2,3,4]}]}'
Check "default: /versions/9/infer rejected (404)" ($i9.Status -eq 404)
$c2 = Call-Server "GET" "http://127.0.0.1:$Port/v2/models/multi_model/versions/2/config" $null
Check "default: /versions/2/config works (200)" ($c2.Status -eq 200)
$c1 = Call-Server "GET" "http://127.0.0.1:$Port/v2/models/multi_model/versions/1/config" $null
Check "default: /versions/1/config rejected (404)" ($c1.Status -eq 404)
$bad = Call-Server "POST" "http://127.0.0.1:$Port/v2/models/multi_model/versions/abc/infer" '{"inputs":[{"name":"input","shape":[1,4],"datatype":"FP32","data":[1,2,3,4]}]}'
Check "default: malformed version rejected (400)" ($bad.Status -eq 400)
Stop-TestServer

# ========== Scenario 2: explicit load pins version 1 then 2 (rollback) ========
Write-Output "`n== Scenario 2: explicit load with specific pin (API-driven rollback) =="
New-CleanRepo
Start-TestServer @("--model-control-mode=explicit")
$idx0 = Call-Server "POST" $indexUri "{}"
Check "explicit: initially UNAVAILABLE" ($idx0.Body -match '"name":"multi_model"' -and $idx0.Body -match '"UNAVAILABLE"')
$cfgV1 = New-ModelConfig "multi_model" "specific { versions: 1 }"
$load1 = Call-Server "POST" "http://127.0.0.1:$Port/v2/repository/models/multi_model/load" (New-LoadBody $cfgV1)
Check "explicit: load pin version 1 (200)" ($load1.Status -eq 200)
$idx1 = Call-Server "POST" $indexUri "{}"
Check "explicit: index reports version 1 READY after pin" ($idx1.Body -match '"version":"1"' -and $idx1.Body -match '"READY"')
$i1 = Call-Server "POST" "http://127.0.0.1:$Port/v2/models/multi_model/versions/1/infer" '{"inputs":[{"name":"input","shape":[1,4],"datatype":"FP32","data":[1,2,3,4]}]}'
Check "explicit: pinned /versions/1/infer works (200)" ($i1.Status -eq 200 -and $i1.Body -match 'outputs')
$i2 = Call-Server "POST" "http://127.0.0.1:$Port/v2/models/multi_model/versions/2/infer" '{"inputs":[{"name":"input","shape":[1,4],"datatype":"FP32","data":[1,2,3,4]}]}'
Check "explicit: pinned /versions/2/infer rejected (404)" ($i2.Status -eq 404)
$un = Call-Server "POST" "http://127.0.0.1:$Port/v2/repository/models/multi_model/unload" "{}"
Check "explicit: unload (200)" ($un.Status -eq 200)
$cfgV2 = New-ModelConfig "multi_model" "specific { versions: 2 }"
$load2 = Call-Server "POST" "http://127.0.0.1:$Port/v2/repository/models/multi_model/load" (New-LoadBody $cfgV2)
Check "explicit: load pin version 2 (200)" ($load2.Status -eq 200)
$idx2 = Call-Server "POST" $indexUri "{}"
Check "explicit: index reports version 2 READY after roll-forward" ($idx2.Body -match '"version":"2"' -and $idx2.Body -match '"READY"')
$i2b = Call-Server "POST" "http://127.0.0.1:$Port/v2/models/multi_model/versions/2/infer" '{"inputs":[{"name":"input","shape":[1,4],"datatype":"FP32","data":[1,2,3,4]}]}'
Check "explicit: /versions/2/infer works after roll-forward (200)" ($i2b.Status -eq 200)
$i1b = Call-Server "POST" "http://127.0.0.1:$Port/v2/models/multi_model/versions/1/infer" '{"inputs":[{"name":"input","shape":[1,4],"datatype":"FP32","data":[1,2,3,4]}]}'
Check "explicit: /versions/1/infer rejected after roll-forward (404)" ($i1b.Status -eq 404)
Stop-TestServer

# ========== Scenario 3: specific pin to a nonexistent version fails ===========
Write-Output "`n== Scenario 3: specific pin to a missing version fails load =="
New-CleanRepo
Start-TestServer @("--model-control-mode=explicit")
$cfgBad = New-ModelConfig "multi_model" "specific { versions: 9 }"
$loadBad = Call-Server "POST" "http://127.0.0.1:$Port/v2/repository/models/multi_model/load" (New-LoadBody $cfgBad)
Check "explicit: load pin missing version 9 rejected (400)" ($loadBad.Status -eq 400)
Check "explicit: error names version_policy" ($loadBad.Body -match 'version_policy')
$idxBad = Call-Server "POST" $indexUri "{}"
Check "explicit: model stays UNAVAILABLE after failed pin" ($idxBad.Body -match '"name":"multi_model"' -and $idxBad.Body -match '"UNAVAILABLE"')
Stop-TestServer

# ====== Scenario 4: latest N>1 / all still serve the highest version ==========
Write-Output "`n== Scenario 4: latest{num_versions>1} and all{} load the newest =="
New-CleanRepo
Start-TestServer @("--model-control-mode=explicit")
$cfgLatest = New-ModelConfig "multi_model" "latest { num_versions: 2 }"
$ll = Call-Server "POST" "http://127.0.0.1:$Port/v2/repository/models/multi_model/load" (New-LoadBody $cfgLatest)
Check "explicit: latest{num_versions:2} load (200)" ($ll.Status -eq 200)
$idxL = Call-Server "POST" $indexUri "{}"
Check "explicit: latest{2} serves highest version 2" ($idxL.Body -match '"version":"2"' -and $idxL.Body -match '"READY"')
$un = Call-Server "POST" "http://127.0.0.1:$Port/v2/repository/models/multi_model/unload" "{}"
Check "explicit: unload before all (200)" ($un.Status -eq 200)
$cfgAll = New-ModelConfig "multi_model" "all { }"
$la = Call-Server "POST" "http://127.0.0.1:$Port/v2/repository/models/multi_model/load" (New-LoadBody $cfgAll)
Check "explicit: all{} load (200)" ($la.Status -eq 200)
$idxA = Call-Server "POST" $indexUri "{}"
Check "explicit: all{} serves highest version 2" ($idxA.Body -match '"version":"2"' -and $idxA.Body -match '"READY"')
Stop-TestServer

# ========== Scenario 5: invalid policies rejected by validation ===============
Write-Output "`n== Scenario 5: invalid version_policy blocks rejected =="
New-CleanRepo
Start-TestServer @("--model-control-mode=explicit")
$cfgN0 = New-ModelConfig "multi_model" "latest { num_versions: 0 }"
$ln0 = Call-Server "POST" "http://127.0.0.1:$Port/v2/repository/models/multi_model/load" (New-LoadBody $cfgN0)
Check "invalid: num_versions=0 rejected (400)" ($ln0.Status -eq 400 -and $ln0.Body -match 'num_versions')
$cfgTwo = New-ModelConfig "multi_model" "latest { num_versions: 1 } specific { versions: 1 }"
$ltwo = Call-Server "POST" "http://127.0.0.1:$Port/v2/repository/models/multi_model/load" (New-LoadBody $cfgTwo)
Check "invalid: latest+specific oneof rejected (400)" ($ltwo.Status -eq 400)
$cfgDup = New-ModelConfig "multi_model" "specific { versions: [ 1, 1 ] }"
$ldup = Call-Server "POST" "http://127.0.0.1:$Port/v2/repository/models/multi_model/load" (New-LoadBody $cfgDup)
Check "invalid: duplicate specific versions rejected (400)" ($ldup.Status -eq 400 -and $ldup.Body -match 'more than once')
$cfgNeg = New-ModelConfig "multi_model" "specific { versions: [ -2 ] }"
$lneg = Call-Server "POST" "http://127.0.0.1:$Port/v2/repository/models/multi_model/load" (New-LoadBody $cfgNeg)
Check "invalid: non-positive specific version rejected (400)" ($lneg.Status -eq 400)
Stop-TestServer

# ========== Scenario 6: gRPC client-requested version =========================
Write-Output "`n== Scenario 6: gRPC client-requested version =="
New-CleanRepo
Start-TestServer @("--model-control-mode=none", "--grpc-port=$GrpcPort")
$python = "C:\Apps\anaconda3\envs\test312\python.exe"
if (-not (Test-Path $python)) { Write-Output "SKIP  gRPC python client not found ($python)"; Stop-TestServer }
else {
    $env:PYTHONPATH = "$RepoRoot\generated\py"
    $client = @"
import sys
sys.path.insert(0, r"$RepoRoot\generated\py")
import grpc
import grpc_service_pb2 as pb
import grpc_service_pb2_grpc as g
ch = grpc.insecure_channel("127.0.0.1:$GrpcPort")
stub = g.GRPCInferenceServiceStub(ch)
def show(name, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + name + ("  " + extra if extra else ""))
def infer_req(version):
    req = pb.ModelInferRequest(model_name="multi_model", model_version=version)
    inp = req.inputs.add()
    inp.name = "input"
    inp.datatype = "FP32"
    inp.shape.extend([1, 4])
    inp.contents.fp32_contents.extend([1.0, 2.0, 3.0, 4.0])
    return req
try:
    r = stub.ModelInfer(infer_req("2"), timeout=10)
    show("grpc ModelInfer v2 ok (model_version echoed)", len(r.outputs) > 0 and r.model_version == "2")
except grpc.RpcError as e:
    show("grpc ModelInfer v2 ok (model_version echoed)", False, "code=" + str(e.code()))
try:
    stub.ModelInfer(infer_req("1"), timeout=10)
    show("grpc ModelInfer v1 rejected", False)
except grpc.RpcError as e:
    show("grpc ModelInfer v1 rejected (NOT_FOUND)", e.code() == grpc.StatusCode.NOT_FOUND)
try:
    stub.ModelInfer(infer_req("abc"), timeout=10)
    show("grpc ModelInfer invalid version rejected", False)
except grpc.RpcError as e:
    show("grpc ModelInfer invalid version (INVALID_ARGUMENT)", e.code() == grpc.StatusCode.INVALID_ARGUMENT)
r = stub.ModelReady(pb.ModelReadyRequest(name="multi_model", version="2"), timeout=10)
show("grpc ModelReady v2 ready", r.ready is True)
r = stub.ModelReady(pb.ModelReadyRequest(name="multi_model", version="1"), timeout=10)
show("grpc ModelReady v1 not ready", r.ready is False)
r = stub.ModelMetadata(pb.ModelMetadataRequest(name="multi_model", version="2"), timeout=10)
show("grpc ModelMetadata v2 ok", r.name == "multi_model")
try:
    stub.ModelMetadata(pb.ModelMetadataRequest(name="multi_model", version="1"), timeout=10)
    show("grpc ModelMetadata v1 rejected", False)
except grpc.RpcError as e:
    show("grpc ModelMetadata v1 rejected (NOT_FOUND)", e.code() == grpc.StatusCode.NOT_FOUND)
r = stub.ModelConfig(pb.ModelConfigRequest(name="multi_model", version="2"), timeout=10)
show("grpc ModelConfig v2 ok", "backend: openvino" in r.config)
try:
    stub.ModelConfig(pb.ModelConfigRequest(name="multi_model", version="1"), timeout=10)
    show("grpc ModelConfig v1 rejected", False)
except grpc.RpcError as e:
    show("grpc ModelConfig v1 rejected (NOT_FOUND)", e.code() == grpc.StatusCode.NOT_FOUND)
"@
    Set-Content "$Work\grpc_version_test.py" -Value $client -Encoding utf8
    # Redirect python output to files: PS treats native stderr under
    # $ErrorActionPreference=Stop as a terminating error, so capture both
    # streams explicitly and report them ourselves.
    $pyProc = Start-Process -FilePath $python -ArgumentList "`"$Work\grpc_version_test.py`"" `
        -Wait -PassThru `
        -RedirectStandardOutput "$Work\grpc_version_test.out" `
        -RedirectStandardError "$Work\grpc_version_test.err"
    $res = Get-Content "$Work\grpc_version_test.out" -Raw -ErrorAction SilentlyContinue
    if ($pyProc.ExitCode -ne 0) {
        $res += "`n" + (Get-Content "$Work\grpc_version_test.err" -Raw -ErrorAction SilentlyContinue)
    }
    Write-Output $res.Trim()
    foreach ($line in ($res -split "`r?`n")) {
        if ($line -match '^PASS ') { $script:pass++ }
        elseif ($line -match '^FAIL ') { $script:fail++ }
    }
}
Stop-TestServer

# ===================== summary =================================================
Write-Output "`n=============================="
Write-Output "version-policy tests: PASS=$pass FAIL=$fail"
if ($fail -gt 0) { exit 1 }
Write-Output "ALL VERSION-POLICY TESTS PASSED"
exit 0
