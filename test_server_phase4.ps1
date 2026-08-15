# test_server_phase4.ps1 - Start inferlite and exercise the Phase 4
# multi-device models (Intel CPU / Intel GPU / AUTO).
$ErrorActionPreference = "Stop"
$build = "c:\Test\triton\InferLite-dpsk-f\build"
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

$body = '{"inputs":[{"name":"input","shape":[1,4],"datatype":"FP32","data":[1,2,3,4]}]}'

Write-Output ""
Write-Output "== health/ready =="
Write-Output (Call-Server "GET" "http://127.0.0.1:8000/v2/health/ready")

Write-Output ""
Write-Output "== health/detailed (device field per model) =="
Write-Output (Call-Server "GET" "http://127.0.0.1:8000/v2/health/detailed")

foreach ($m in @("intel_cpu_model","intel_npu_model","intel_auto_model","sample_model")) {
    Write-Output ""
    Write-Output "== config: $m =="
    Write-Output (Call-Server "GET" "http://127.0.0.1:8000/v2/models/$m/config")
    Write-Output "== infer: $m (expect 2*x+1 = [3,5,7,9]) =="
    Write-Output (Call-Server "POST" "http://127.0.0.1:8000/v2/models/$m/infer" $body)
}

Write-Output ""
Write-Output "== metrics (device field) =="
Write-Output (Call-Server "GET" "http://127.0.0.1:8000/v2/metrics")

Write-Output ""
Write-Output "== versions =="
Write-Output (Call-Server "GET" "http://127.0.0.1:8000/v2/versions")

Write-Output ""
Write-Output "Done. Stopping server."
Stop-Process -Name inferlite -Force
