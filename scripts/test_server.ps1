# test_server.ps1 - Start inferlite and exercise the REST endpoints.
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$build = Join-Path $RepoRoot "build"
Set-Location $build

# Ensure no stale server.
Get-Process inferlite -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 500

$p = Start-Process -FilePath "$build\inferlite.exe" `
    -ArgumentList "--model-repository=..\models --http-port=8000 --max-queue-size=100 --http-threads=4" `
    -PassThru -RedirectStandardOutput "$build\out.txt" -RedirectStandardError "$build\err.txt"
Start-Sleep -Seconds 3
if ($p.HasExited) {
    Write-Output "SERVER EXITED code=$($p.ExitCode)"
    Write-Output "--- stderr ---"
    Get-Content "$build\err.txt" -Encoding utf8
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
Write-Output "== health/ready =="
Write-Output (Call-Server "GET" "http://127.0.0.1:8000/v2/health/ready")

Write-Output ""
Write-Output "== model config =="
Write-Output (Call-Server "GET" "http://127.0.0.1:8000/v2/models/sample_model/config")

Write-Output ""
Write-Output "== infer (FP32 array, expect 2*x+1 = [3,5,7,9]) =="
$body = '{"inputs":[{"name":"input","shape":[1,4],"datatype":"FP32","data":[1,2,3,4]}]}'
Write-Output (Call-Server "POST" "http://127.0.0.1:8000/v2/models/sample_model/infer" $body)

Write-Output ""
Write-Output "== infer (base64) =="
# base64 of bytes [1,2,3,4] as float32 little-endian
$b64 = [Convert]::ToBase64String([System.BitConverter]::GetBytes([float]1.0)) +
       [Convert]::ToBase64String([System.BitConverter]::GetBytes([float]2.0)) +
       [Convert]::ToBase64String([System.BitConverter]::GetBytes([float]3.0)) +
       [Convert]::ToBase64String([System.BitConverter]::GetBytes([float]4.0))
$body2 = '{"inputs":[{"name":"input","shape":[1,4],"datatype":"FP32","data":"' + $b64 + '"}]}'
Write-Output (Call-Server "POST" "http://127.0.0.1:8000/v2/models/sample_model/infer" $body2)

Write-Output ""
Write-Output "== metrics =="
Write-Output (Call-Server "GET" "http://127.0.0.1:8000/v2/metrics")

Write-Output ""
Write-Output "== bad model (404) =="
Write-Output (Call-Server "GET" "http://127.0.0.1:8000/v2/models/nonexistent/config")

Write-Output ""
Write-Output "== queue full check (concurrency) =="
# fire several concurrent infer requests
$jobs = @()
for ($i = 0; $i -lt 6; $i++) {
    $jobs += Start-Job -ScriptBlock {
        param($u)
        $r = Invoke-WebRequest -Uri $u -Method POST -Body '{"inputs":[{"name":"input","shape":[1,4],"datatype":"FP32","data":[5,6,7,8]}]}' -ContentType "application/json" -UseBasicParsing -TimeoutSec 20
        return "$($r.StatusCode)"
    } -ArgumentList "http://127.0.0.1:8000/v2/models/sample_model/infer"
}
$results = $jobs | Wait-Job | Receive-Job
$jobs | Remove-Job
Write-Output ("concurrent results: " + ($results -join ", "))

Write-Output ""
Write-Output "Done. Stopping server."
Stop-Process -Name inferlite -Force
