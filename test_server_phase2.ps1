# test_server_phase2.ps1 - Exercise Phase 2 FDA features against inferlite.
# Validates: model integrity (manifest), audit trail, health/detailed, versions,
# structured error codes, ensemble DAG, plugin backend, and self-test readiness.
$ErrorActionPreference = "Stop"
$build = "c:\Test\triton\inferlite\build"
Set-Location $PSScriptRoot

# Ensure no stale server.
Get-Process inferlite -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 500

Remove-Item "$build\audit_p2.log", "$build\diag_p2.log" -ErrorAction SilentlyContinue

$p = Start-Process -FilePath "$build\inferlite.exe" `
    -ArgumentList "--model-repository=models --http-port=8100 --max-queue-size=100 --http-threads=4 --validated-mode --audit-log=$build\audit_p2.log --diagnostic-log=$build\diag_p2.log" `
    -PassThru -RedirectStandardOutput "$build\out_p2.txt" -RedirectStandardError "$build\err_p2.txt"
Start-Sleep -Seconds 5
if ($p.HasExited) {
    Write-Output "SERVER EXITED code=$($p.ExitCode)"
    if (Test-Path "$build\err_p2.txt") { Write-Output ([IO.File]::ReadAllText("$build\err_p2.txt")) }
    exit 1
}
Write-Output "SERVER RUNNING pid=$($p.Id)"

function Invoke-IL($method, $uri, $jsonBody) {
    $params = @{ Uri = $uri; Method = $method; TimeoutSec = 15 }
    if ($jsonBody) {
        $params.Body = $jsonBody
        $params.ContentType = "application/json"
    }
    try {
        $r = Invoke-WebRequest @params -UseBasicParsing
        return "[$($r.StatusCode)] $($r.Content)"
    } catch {
        $resp = $_.Exception.Response
        if ($resp) {
            $code = $resp.StatusCode
            $bodyText = ""
            try {
                if ($resp.GetResponseStream) {
                    $sr = New-Object IO.StreamReader($resp.GetResponseStream())
                    $bodyText = $sr.ReadToEnd()
                } elseif ($resp.Content) {
                    $bodyText = [string]$resp.Content
                }
            } catch { }
            return "[$([int]$code)] $bodyText"
        }
        return "[NETERR] $($_.Exception.Message)"
    }
}

Write-Output ""
Write-Output "== 1. health/ready (self-test must pass) =="
Write-Output (Invoke-IL "GET" "http://127.0.0.1:8100/v2/health/ready")

Write-Output ""
Write-Output "== 2. health/detailed (per-model hashes + status) =="
Write-Output (Invoke-IL "GET" "http://127.0.0.1:8100/v2/health/detailed")

Write-Output ""
Write-Output "== 3. versions (software + model versions) =="
Write-Output (Invoke-IL "GET" "http://127.0.0.1:8100/v2/versions")

Write-Output ""
Write-Output "== 4. openvino model infer (y=2x+1 -> [3,5,7,9]) =="
Write-Output (Invoke-IL "POST" "http://127.0.0.1:8100/v2/models/sample_model/infer" '{"inputs":[{"name":"input","shape":[1,4],"datatype":"FP32","data":[1,2,3,4]}]}')

Write-Output ""
Write-Output "== 5. plugin backend infer (x*0.5 -> [1,2,3,4]) =="
Write-Output (Invoke-IL "POST" "http://127.0.0.1:8100/v2/models/preprocess_plugin/infer" '{"inputs":[{"name":"raw","shape":[1,4],"datatype":"FP32","data":[2,4,6,8]}]}')

Write-Output ""
Write-Output "== 6. ensemble DAG infer (raw->norm->2x+1->+0.5 -> [2.5,3.5,4.5,5.5]) =="
Write-Output (Invoke-IL "POST" "http://127.0.0.1:8100/v2/models/ensemble_pipeline/infer" '{"inputs":[{"name":"raw","shape":[1,4],"datatype":"FP32","data":[1,2,3,4]}]}')

Write-Output ""
Write-Output "== 7. INVALID_INPUT: wrong shape (expect 400) =="
Write-Output (Invoke-IL "POST" "http://127.0.0.1:8100/v2/models/sample_model/infer" '{"inputs":[{"name":"input","shape":[1,5],"datatype":"FP32","data":[1,2,3,4,5]}]}')

Write-Output ""
Write-Output "== 8. INVALID_INPUT: wrong dtype (expect 400) =="
Write-Output (Invoke-IL "POST" "http://127.0.0.1:8100/v2/models/sample_model/infer" '{"inputs":[{"name":"input","shape":[1,4],"datatype":"INT8","data":[1,2,3,4]}]}')

Write-Output ""
Write-Output "== 9. MODEL_NOT_FOUND (expect 404) =="
Write-Output (Invoke-IL "POST" "http://127.0.0.1:8100/v2/models/nope/infer" '{"inputs":[{"name":"x","shape":[1,1],"datatype":"FP32","data":[1]}]}')

Write-Output ""
Write-Output "== 10. metrics (with config_hash + per-model) =="
Write-Output (Invoke-IL "GET" "http://127.0.0.1:8100/v2/metrics")

Write-Output ""
Write-Output "== 11. model config endpoint (with hashes) =="
Write-Output (Invoke-IL "GET" "http://127.0.0.1:8100/v2/models/ensemble_pipeline/config")

Write-Output ""
Write-Output "== 12. audit log (tamper-evident entries) =="
if (Test-Path "$build\audit_p2.log") {
    Write-Output ([IO.File]::ReadAllText("$build\audit_p2.log"))
} else {
    Write-Output "NO AUDIT LOG"
}

Write-Output ""
Write-Output "Done. Stopping server."
Stop-Process -Name inferlite -Force
