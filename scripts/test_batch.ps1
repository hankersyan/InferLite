# test_batch.ps1 - Start inferlite and verify Triton-style batching
# (max_batch_size: 1) on the batched_model.
#
# Triton convention exercised here:
#   config dims are per-request (no batch dim): dims: [ 4 ]
#   client sends shapes with a leading batch dim:  [ 1, 4 ]
#   max_batch_size: 1  => batch dim must be exactly 1.
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$build = Join-Path $RepoRoot "build"
Set-Location $build

Get-Process inferlite -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 500

$p = Start-Process -FilePath "$build\inferlite.exe" `
    -ArgumentList "--model-repository=..\models --http-port=8000 --max-queue-size=100 --http-threads=4" `
    -PassThru -RedirectStandardOutput "$build\out.txt" -RedirectStandardError "$build\err.txt"
Start-Sleep -Seconds 4
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
Write-Output "== model config (expect max_batch_size=1, dims [4]) =="
Write-Output (Call-Server "GET" "http://127.0.0.1:8000/v2/models/batched_model/config")

Write-Output ""
Write-Output "== infer valid batch (shape [1,4], expect 2*x+1 = [3,5,7,9]) =="
$body = '{"inputs":[{"name":"input","shape":[1,4],"datatype":"FP32","data":[1,2,3,4]}]}'
Write-Output (Call-Server "POST" "http://127.0.0.1:8000/v2/models/batched_model/infer" $body)

Write-Output ""
Write-Output "== infer MISSING batch dim (shape [4] -> expect 400) =="
$bad1 = '{"inputs":[{"name":"input","shape":[4],"datatype":"FP32","data":[1,2,3,4]}]}'
Write-Output (Call-Server "POST" "http://127.0.0.1:8000/v2/models/batched_model/infer" $bad1)

Write-Output ""
Write-Output "== infer batch too large (shape [2,4] > max_batch_size=1 -> expect 400) =="
$bad2 = '{"inputs":[{"name":"input","shape":[2,4],"datatype":"FP32","data":[1,2,3,4,5,6,7,8]}]}'
Write-Output (Call-Server "POST" "http://127.0.0.1:8000/v2/models/batched_model/infer" $bad2)

Write-Output ""
Write-Output "Done. Stopping server."
Stop-Process -Name inferlite -Force
