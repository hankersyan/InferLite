# test_phase3_smoke.ps1 - Phase 3 smoke test (CPU-only regression check).
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$build = Join-Path $RepoRoot "build"
Set-Location $build

Get-Process inferlite -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 500

$p = Start-Process -FilePath "$build\inferlite.exe" `
    -ArgumentList "--model-repository=..\models --http-port=8000 --max-queue-size=100 --http-threads=4" `
    -PassThru -RedirectStandardOutput "$build\phase3_out.txt" -RedirectStandardError "$build\phase3_err.txt"
Start-Sleep -Seconds 3

if ($p.HasExited) {
    $t = [System.IO.File]::ReadAllText("$build\phase3_err.txt")
    Write-Output "SERVER EXITED code=$($p.ExitCode)"
    Write-Output $t
    exit 1
}
Write-Output "SERVER RUNNING pid=$($p.Id)"

function Call-Server($method, $uri, $jsonBody) {
    $req = [System.Net.HttpWebRequest]::Create($uri)
    $req.Method = $method
    $req.ContentType = "application/json"
    $req.Timeout = 15000
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
        return "[$($resp.StatusCode)] $($sr.ReadToEnd())"
    } catch [System.Net.WebException] {
        $er = $_.Exception.Response
        if ($er) {
            $sr = New-Object IO.StreamReader($er.GetResponseStream())
            return "[ERR $([int]$er.StatusCode)] $($sr.ReadToEnd())"
        }
        return "[NETERR] $($_.Exception.Message)"
    }
}

Write-Output ""
Write-Output "== health/ready (expect 200 READY) =="
Write-Output (Call-Server "GET" "http://127.0.0.1:8000/v2/health/ready")

Write-Output ""
Write-Output "== health/detailed (expect gpu.enabled=false, device=CPU on models) =="
Write-Output (Call-Server "GET" "http://127.0.0.1:8000/v2/health/detailed")

Write-Output ""
Write-Output "== infer sample_model (expect 2*x+1 = [3,5,7,9]) =="
$body = '{"inputs":[{"name":"input","shape":[1,4],"datatype":"FP32","data":[1,2,3,4]}]}'
Write-Output (Call-Server "POST" "http://127.0.0.1:8000/v2/models/sample_model/infer" $body)

Write-Output ""
Write-Output "== infer ensemble_pipeline (expect [3,5,7,9]) =="
$ebody = '{"inputs":[{"name":"raw","shape":[1,4],"datatype":"FP32","data":[2,4,6,8]}]}'
Write-Output (Call-Server "POST" "http://127.0.0.1:8000/v2/models/ensemble_pipeline/infer" $ebody)

Write-Output ""
Write-Output "== metrics (expect gpu_memory absent or 0 on CPU build) =="
Write-Output (Call-Server "GET" "http://127.0.0.1:8000/v2/metrics")

Write-Output ""
Write-Output "Done. Stopping server."
Stop-Process -Id $p.Id -Force
