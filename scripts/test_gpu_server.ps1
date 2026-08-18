# test_gpu_server.ps1 - Start the GPU build and verify GPU reporting + CPU regression.
# Builds with build_gpu.ps1 first, then starts the server with CUDA/TensorRT on PATH.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$build = "$root\build-gpu"

# Ensure the GPU build exists.
if (-not (Test-Path "$build\inferlite.exe")) {
    Write-Output "GPU build not found; building..."
    & powershell -NoProfile -ExecutionPolicy Bypass -File "$root\build_gpu.ps1"
}

# Put CUDA + TensorRT runtime DLLs on PATH.
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;" +
            "C:\Tools\nvidia\TensorRT-10.16.1.11.Windows.amd64.cuda-12.9\bin;" + $env:PATH

# Also stage the DLLs next to the exe (robust for DirectX/load-order quirks).
Copy-Item "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\cudart64_12.dll" $build -Force -ErrorAction SilentlyContinue
Copy-Item "C:\Tools\nvidia\TensorRT-10.16.1.11.Windows.amd64.cuda-12.9\bin\nvinfer_10.dll" $build -Force -ErrorAction SilentlyContinue
Copy-Item "C:\Tools\nvidia\TensorRT-10.16.1.11.Windows.amd64.cuda-12.9\bin\nvinfer_plugin_10.dll" $build -Force -ErrorAction SilentlyContinue

Set-Location $build
Get-Process inferlite -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 500

$p = Start-Process -FilePath "$build\inferlite.exe" `
    -ArgumentList "--model-repository=..\models --http-port=8000 --max-queue-size=100 --http-threads=4 --max-gpu-memory-mb=1024" `
    -PassThru -RedirectStandardOutput "$build\gpu_out.txt" -RedirectStandardError "$build\gpu_err.txt"
Start-Sleep -Seconds 3

if ($p.HasExited) {
    Write-Output "SERVER EXITED code=$($p.ExitCode)"
    $t = [System.IO.File]::ReadAllText("$build\gpu_err.txt")
    Write-Output "--- stderr ---"
    Write-Output $t
    exit 1
}
Write-Output "SERVER RUNNING pid=$($p.Id) (GPU build)"

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
Write-Output "== health/detailed (expect gpu.enabled=true) =="
Write-Output (Call-Server "GET" "http://127.0.0.1:8000/v2/health/detailed")

Write-Output ""
Write-Output "== metrics =="
Write-Output (Call-Server "GET" "http://127.0.0.1:8000/v2/metrics")

Write-Output ""
Write-Output "Done. Stopping server."
Stop-Process -Id $p.Id -Force
