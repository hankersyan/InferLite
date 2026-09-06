# test_warmup.ps1 - Verify Triton-style model_warmup on the InferLite server.
#
# Triton convention exercised here (model_warmup in config.pbtxt):
#   model_warmup [
#     {
#       name: "warmup"
#       inputs { key: "input" value { data_type: TYPE_FP32 dims: [ 1, 4 ] zero_data: true } }
#     }
#   ]
#
# Semantics under test:
#   (1) a model whose warmup request succeeds loads and becomes READY, and the
#       diag log records the warmup execution;
#   (2) a model whose warmup spec is invalid (data_type mismatch) fails to load
#       at startup (kNone mode is fail-fast) with a precise config error.
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$build = Join-Path $RepoRoot "build"
Set-Location $build

Get-Process inferlite -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 500

$modelSrc = Join-Path $RepoRoot "models\sample_model\1"
if (-not (Test-Path (Join-Path $modelSrc "model.xml"))) {
    throw "sample model IR not found under $modelSrc"
}

function New-WarmupRepo($dir, $dataType) {
    if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
    $modelDir = Join-Path $dir "warm_model\1"
    New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
    Copy-Item "$modelSrc\*" $modelDir
    $cfg = @"
name: "warm_model"
backend: "openvino"
max_batch_size: 0
input {
  name: "input"
  data_type: TYPE_FP32
  dims: [ 1, 4 ]
}
output {
  name: "add"
  data_type: TYPE_FP32
  dims: [ 1, 4 ]
}
instance_group {
  count: 1
  kind: KIND_CPU
}
model_warmup [
  {
    name: "warmup"
    inputs {
      key: "input"
      value {
        data_type: $dataType
        dims: [ 1, 4 ]
        zero_data: true
      }
    }
  }
]
"@
    Set-Content -Path (Join-Path $dir "warm_model\config.pbtxt") -Value $cfg -Encoding ascii
}

function Call-Server($method, $uri, $jsonBody) {
    $req = [System.Net.HttpWebRequest]::Create($uri)
    $req.Method = $method
    $req.ContentType = "application/json"
    $req.Timeout = 4000
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
        return "[$([int]$resp.StatusCode)] $($sr.ReadToEnd())"
    } catch [System.Net.WebException] {
        $er = $_.Exception.Response
        if ($er) {
            $sr = New-Object IO.StreamReader($er.GetResponseStream())
            return "[ERR $([int]$er.StatusCode)] $($sr.ReadToEnd())"
        }
        return "[NETERR] $($_.Exception.Message)"
    }
}

$pass = $true

# ---- Phase A: a model with a valid warmup must load and become READY ----
$repoOk = Join-Path $build "warmup_repo_ok"
New-WarmupRepo $repoOk "TYPE_FP32"
$errOk = Join-Path $build "warmup_ok_err.txt"
$outOk = Join-Path $build "warmup_ok_out.txt"
$p = Start-Process -FilePath "$build\inferlite.exe" `
    -ArgumentList "--model-repository=$repoOk --http-port=8003 --diagnostic-log=$build\warmup_ok_diag.txt" `
    -PassThru -RedirectStandardOutput $outOk -RedirectStandardError $errOk

$ready = $false
$lastResp = ""
for ($i = 0; $i -lt 120; $i++) {   # up to 60 s (cold OpenVINO first load can be slow)
    Start-Sleep -Milliseconds 500
    if ($p.HasExited) {
        Write-Output "PHASE A: SERVER EXITED code=$($p.ExitCode)"
        Get-Content $errOk -Encoding utf8
        $pass = $false
        break
    }
    $lastResp = Call-Server "GET" "http://127.0.0.1:8003/v2/health/ready"
    if ($lastResp.StartsWith("[200]")) {
        $ready = $true
        break
    }
}
if (-not $ready -and -not $p.HasExited) {
    Write-Output "PHASE A: server did not become READY (last response: $lastResp)"
    Get-Content $errOk -Encoding utf8
    $pass = $false
}

if ($ready) {
    Write-Output "== A1: health ready =="
    Write-Output (Call-Server "GET" "http://127.0.0.1:8003/v2/health/ready")

    Write-Output ""
    Write-Output "== A2: config reflects model_warmup =="
    $cfgResp = Call-Server "GET" "http://127.0.0.1:8003/v2/models/warm_model/config"
    Write-Output $cfgResp
    if ($cfgResp -notmatch "model_warmup" -or $cfgResp -notmatch "zero_data") {
        Write-Output "PHASE A: /config does not reflect model_warmup"
        $pass = $false
    }

    Write-Output ""
    Write-Output "== A3: diag log records the warmup execution =="
    $diag = Get-Content "$build\warmup_ok_diag.txt" -Encoding utf8 -ErrorAction SilentlyContinue
    $warm = $diag | Where-Object { $_ -match "warmup model 'warm_model' request 'warmup' ok" }
    if ($warm) {
        Write-Output $warm
    } else {
        Write-Output "PHASE A: no warmup diag line found"
        Get-Content $errOk -Encoding utf8
        $pass = $false
    }

    Write-Output ""
    Write-Output "== A4: model still serves inference (2*x+1 over [1,2,3,4]) =="
    $body = '{"inputs":[{"name":"input","shape":[1,4],"datatype":"FP32","data":[1,2,3,4]}]}'
    $inf = Call-Server "POST" "http://127.0.0.1:8003/v2/models/warm_model/infer" $body
    Write-Output $inf
    if ($inf -notmatch '^\[200\]') {
        Write-Output "PHASE A: infer request failed"
        $pass = $false
    } else {
        $json = ($inf -replace '^\[200\] ', '') | ConvertFrom-Json
        $raw = [Convert]::FromBase64String($json.outputs[0].data)
        if ($raw.Length -ne 16) {
            Write-Output "PHASE A: unexpected output byte count $($raw.Length)"
            $pass = $false
        } else {
            $vals = for ($k = 0; $k -lt 4; $k++) { [BitConverter]::ToSingle($raw, $k * 4) }
            $joined = ($vals -join ",")
            Write-Output "PHASE A: decoded outputs = $joined"
            if ($joined -ne "3,5,7,9") {
                Write-Output "PHASE A: unexpected inference result"
                $pass = $false
            }
        }
    }
}
if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
Start-Sleep -Milliseconds 500

# ---- Phase B: an invalid warmup spec must fail-fast at startup (kNone) ----
$repoBad = Join-Path $build "warmup_repo_bad"
New-WarmupRepo $repoBad "TYPE_INT32"   # model input is FP32 -> config error
$errBad = Join-Path $build "warmup_bad_err.txt"
$outBad = Join-Path $build "warmup_bad_out.txt"
$p2 = Start-Process -FilePath "$build\inferlite.exe" `
    -ArgumentList "--model-repository=$repoBad --http-port=8004" `
    -PassThru -RedirectStandardOutput $outBad -RedirectStandardError $errBad

$exited = $false
for ($i = 0; $i -lt 60; $i++) {   # up to 30 s for the config scan to fail
    Start-Sleep -Milliseconds 500
    if ($p2.HasExited) { $exited = $true; break }
}
if ($exited) {
    Write-Output "PHASE B: server exited code=$($p2.ExitCode) (expected fail-fast)"
    $errText = Get-Content $errBad -Encoding utf8 -Raw
    if ($errText -match "model_warmup" -and $errText -match "data_type") {
        Write-Output "PHASE B: config error mentions model_warmup data_type mismatch"
    } else {
        Write-Output "PHASE B: server exited but error does not mention model_warmup:"
        Write-Output $errText
        $pass = $false
    }
} else {
    Write-Output "PHASE B: server did NOT exit (warmup config error was not fail-fast)"
    Stop-Process -Id $p2.Id -Force
    $pass = $false
}

Write-Output ""
if ($pass) {
    Write-Output "ALL WARMUP TESTS PASSED"
    exit 0
}
Write-Output "WARMUP TESTS FAILED"
exit 1
