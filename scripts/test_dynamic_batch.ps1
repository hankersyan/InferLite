# test_dynamic_batch.ps1 - Verify InferLite batching mode following NVIDIA
# Triton on the dynamic_batch_model (models/dynamic_batch_model).
#
# Triton semantics exercised here:
#   * config dims are per-request (no batch dim): dims: [ 4 ]
#   * a client request carries a leading batch dim B in [1, max_batch_size]
#   * max_batch_size: 8 caps the number of samples per backend execution
#   * dynamic_batching { preferred_batch_size: [8] } => a full batch runs
#     immediately (no artificial delay)
#   * dynamic_batching { max_queue_delay_microseconds: 150000 } => requests
#     that cannot fill a preferred batch wait up to 150ms for late arrivals
#   * concurrent requests are coalesced into one execution and each response
#     is the slice that belongs to it (output split)
$ErrorActionPreference = "Stop"
# Ensure System.Net.Http.HttpClient exists on Windows PowerShell 5.1 as well.
Add-Type -AssemblyName System.Net.Http -ErrorAction SilentlyContinue
# .NET Framework defaults to only 2 connections per host, which would serialize
# the concurrent infer requests and defeat the batching test. Raise it.
[System.Net.ServicePointManager]::DefaultConnectionLimit = 64

$RepoRoot = Split-Path -Parent $PSScriptRoot
$build = Join-Path $RepoRoot "build"
$exe = Join-Path $build "inferlite.exe"
$port = 8012
$base = "http://127.0.0.1:$port"

$repo = Join-Path $build "repo_dynamic_batch"
if (Test-Path $repo) { Remove-Item -Recurse -Force $repo }
New-Item -ItemType Directory -Path $repo | Out-Null
Copy-Item -Recurse (Join-Path $RepoRoot "models\dynamic_batch_model") $repo
Copy-Item -Recurse (Join-Path $RepoRoot "models\sample_model") $repo

$serverLog = Join-Path $build "db_out.txt"
$serverErr = Join-Path $build "db_err.txt"

Get-Process inferlite -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 500
# http-threads bounds how many infer handlers may wait on the scheduler at
# once, and every handler blocks until its request is served. 16 lets all eight
# concurrent requests reach the queue while the batch window is still open.
$p = Start-Process -FilePath $exe `
    -ArgumentList "--model-repository=$repo --http-port=$port --max-queue-size=100 --http-threads=16" `
    -PassThru -RedirectStandardOutput $serverLog -RedirectStandardError $serverErr

$script:failures = 0
function Assert-True($cond, $msg) {
    if (-not $cond) {
        $script:failures++
        Write-Output "FAIL: $msg"
    } else {
        Write-Output "PASS: $msg"
    }
}

try {
    # Wait for readiness (up to 30s).
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        if ($p.HasExited) { break }
        try {
            $r = Invoke-WebRequest -Uri "$base/v2/health/ready" -UseBasicParsing -TimeoutSec 2
            if ($r.StatusCode -eq 200) { $ready = $true; break }
        } catch { }
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) {
        Write-Output "SERVER DID NOT BECOME READY"
        Write-Output "--- stdout ---"; Get-Content $serverLog -ErrorAction SilentlyContinue
        Write-Output "--- stderr ---"; Get-Content $serverErr -ErrorAction SilentlyContinue
        exit 1
    }
    Write-Output "SERVER RUNNING pid=$($p.Id)"

    # ---- HTTP helper (synchronous) ----
    function Call-Server($method, $uri, $jsonBody) {
        $req = [System.Net.HttpWebRequest]::Create($uri)
        $req.Method = $method
        $req.ContentType = "application/json"
        $req.Timeout = 30000
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

    # ---- base64 FP32 decode ----
    function Decode-Fp32($b64) {
        $bytes = [System.Convert]::FromBase64String($b64)
        $vals = @()
        for ($i = 0; $i -lt $bytes.Length; $i += 4) {
            $vals += [System.BitConverter]::ToSingle($bytes, $i)
        }
        return ,$vals
    }

    # Build an infer JSON body: B rows each [v, v+1, v+2, v+3], base offset `start`
    # so different requests carry distinguishable data.
    function New-InferBody($B, $start = 1) {
        $flat = New-Object System.Collections.Generic.List[double]
        for ($r = 0; $r -lt $B; $r++) {
            for ($c = 0; $c -lt 4; $c++) { $flat.Add([double]($start + $r * 4 + $c)) }
        }
        $bytes = New-Object byte[] ($flat.Count * 4)
        for ($i = 0; $i -lt $flat.Count; $i++) {
            [System.BitConverter]::GetBytes([single]$flat[$i]).CopyTo($bytes, $i * 4)
        }
        $b64 = [System.Convert]::ToBase64String($bytes)
        $shape = "[$B,4]"
        return ('{{"inputs":[{{"name":"input","shape":{0},"datatype":"FP32","data":"{1}"}}]}}' -f $shape, $b64)
    }

    # Verify that a success response body encodes y = 2*x + 1 for B rows that
    # started at `start`. The body is either "[200] {json}" (from Call-Server)
    # or "[<status>] {json}" (built around an HttpClient response).
    function Verify-Body($body, $B, $start, $tag) {
        if (-not $body.StartsWith("[200]")) { return $false }
        $json = $body.Substring(6) | ConvertFrom-Json
        $outs = @($json.outputs)
        if ($outs.Count -eq 0) { return $false }
        $out = $outs[0]
        $shape = @($out.shape)
        if ($shape.Count -ne 2) { return $false }
        if ([int64]$shape[0] -ne $B -or [int64]$shape[1] -ne 4) {
            Write-Output "   ($tag) unexpected output shape: $($shape -join ',')"
            return $false
        }
        $vals = Decode-Fp32 $out.data
        if ($vals.Count -ne ($B * 4)) { return $false }
        for ($r = 0; $r -lt $B; $r++) {
            for ($c = 0; $c -lt 4; $c++) {
                $x = $start + $r * 4 + $c
                $expected = 2.0 * $x + 1.0
                if ([Math]::Abs($vals[$r * 4 + $c] - $expected) -gt 1e-3) {
                    Write-Output "   ($tag) row=$r col=$c val=$($vals[$r*4+$c]) expected=$expected"
                    return $false
                }
            }
        }
        return $true
    }

    function Get-Metric($name, $field) {
        $doc = (Invoke-WebRequest -Uri "$base/v2/metrics" -UseBasicParsing -TimeoutSec 10).Content | ConvertFrom-Json
        $row = $doc.models | Where-Object { $_.model_name -eq $name }
        if ($null -eq $row) { return $null }
        $prop = $row.PSObject.Properties[$field]
        if ($null -eq $prop) { return $null }
        return [int64]$prop.Value
    }

    # ---------------- 1. config reports dynamic batching ----------------
    Write-Output ""
    $cfgBody = Call-Server "GET" "$base/v2/models/dynamic_batch_model/config"
    $cfg = ($cfgBody -replace '^\[\d+\] ', '') | ConvertFrom-Json
    Assert-True ($cfg.max_batch_size -eq 8) "config max_batch_size == 8"
    Assert-True ($null -ne $cfg.dynamic_batching) "config exposes dynamic_batching"
    Assert-True ($cfg.dynamic_batching.preferred_batch_size[0] -eq 8) "config preferred_batch_size == [8]"
    Assert-True ($cfg.dynamic_batching.max_queue_delay_microseconds -eq 150000) "config max_queue_delay_microseconds == 150000"

    # ---------------- 2. single request already at a full batch (B=8) ----------------
    Write-Output ""
    $body8 = New-InferBody 8 1
    $r = Call-Server "POST" "$base/v2/models/dynamic_batch_model/infer" $body8
    Assert-True (Verify-Body $r 8 1 "B8") "single full-batch request [8,4] -> output [8,4] y=2x+1"

    # ---------------- 3. oversized request rejected (B=9 > max_batch_size 8) ----------------
    Write-Output ""
    $body9 = New-InferBody 9 1
    $r9 = Call-Server "POST" "$base/v2/models/dynamic_batch_model/infer" $body9
    Assert-True ($r9.StartsWith("[ERR 400]") -or $r9.StartsWith("[400]")) "request with B=9 (>max_batch_size) rejected (400)"

    # ---------------- 4. single sample request still works (waits for the delay window) ----------------
    Write-Output ""
    $r1 = Call-Server "POST" "$base/v2/models/dynamic_batch_model/infer" (New-InferBody 1 100)
    Assert-True (Verify-Body $r1 1 100 "B1") "single-sample request [1,4] -> output [1,4] y=2x+1"

    # ---------------- 5. 8 concurrent single-sample requests coalesce into ONE batch ----------------
    Write-Output ""
    $reqBefore = Get-Metric "dynamic_batch_model" "requests_completed"
    $batchesBefore = Get-Metric "dynamic_batch_model" "batches_executed"
    $samplesBefore = Get-Metric "dynamic_batch_model" "batch_samples"

    $client = New-Object System.Net.Http.HttpClient
    try {
        $tasks = @()
        for ($i = 0; $i -lt 8; $i++) {
            $c = New-Object System.Net.Http.StringContent((New-InferBody 1 (1000 + $i * 4)), [System.Text.Encoding]::UTF8, "application/json")
            $tasks += $client.PostAsync("$base/v2/models/dynamic_batch_model/infer", $c)
        }
        [System.Threading.Tasks.Task]::WaitAll($tasks)
        $allOk = $true
        for ($i = 0; $i -lt $tasks.Count; $i++) {
            $resp = $tasks[$i].Result
            $b = $resp.Content.ReadAsStringAsync().GetAwaiter().GetResult()
            if (-not (Verify-Body "[$([int]$resp.StatusCode)] $b" 1 (1000 + $i * 4) "conc$i")) { $allOk = $false }
        }
        Assert-True $allOk "8 concurrent B=1 requests all return y=2x+1"
    } finally {
        $client.Dispose()
    }
    # Wait for the batch to finish executing.
    Start-Sleep -Milliseconds 300
    $reqAfter = Get-Metric "dynamic_batch_model" "requests_completed"
    $batchesAfter = Get-Metric "dynamic_batch_model" "batches_executed"
    $samplesAfter = Get-Metric "dynamic_batch_model" "batch_samples"
    Assert-True (($reqAfter - $reqBefore) -eq 8) "8 concurrent requests completed"
    Assert-True (($samplesAfter - $samplesBefore) -eq 8) "8 samples served"
    Assert-True (($batchesAfter - $batchesBefore) -eq 1) "8 concurrent requests merged into a single backend execution (batches+1)"

    # ---------------- 6. two concurrent B=4 requests merge into one B=8 execution ----------------
    Write-Output ""
    $reqBefore = Get-Metric "dynamic_batch_model" "requests_completed"
    $batchesBefore = Get-Metric "dynamic_batch_model" "batches_executed"
    $samplesBefore = Get-Metric "dynamic_batch_model" "batch_samples"

    $client = New-Object System.Net.Http.HttpClient
    try {
        $bodies = @((New-InferBody 4 2000), (New-InferBody 4 3000))
        $tasks = @()
        foreach ($b in $bodies) {
            $c = New-Object System.Net.Http.StringContent($b, [System.Text.Encoding]::UTF8, "application/json")
            $tasks += $client.PostAsync("$base/v2/models/dynamic_batch_model/infer", $c)
        }
        [System.Threading.Tasks.Task]::WaitAll($tasks)
        $startOk = $false
        for ($i = 0; $i -lt $tasks.Count; $i++) {
            $resp = $tasks[$i].Result
            $b = $resp.Content.ReadAsStringAsync().GetAwaiter().GetResult()
            $okA = Verify-Body "[$([int]$resp.StatusCode)] $b" 4 2000 "B4a"
            $okB = Verify-Body "[$([int]$resp.StatusCode)] $b" 4 3000 "B4b"
            if ($okA -or $okB) { $startOk = $true }
        }
        Assert-True $startOk "two concurrent B=4 requests each return the correct 4-row slice"
    } finally {
        $client.Dispose()
    }
    Start-Sleep -Milliseconds 300
    $reqAfter = Get-Metric "dynamic_batch_model" "requests_completed"
    $batchesAfter = Get-Metric "dynamic_batch_model" "batches_executed"
    $samplesAfter = Get-Metric "dynamic_batch_model" "batch_samples"
    Assert-True (($reqAfter - $reqBefore) -eq 2) "two B=4 requests completed"
    Assert-True (($samplesAfter - $samplesBefore) -eq 8) "eight samples served across the two requests"
    Assert-True (($batchesAfter - $batchesBefore) -eq 1) "two B=4 requests merged into one B=8 execution (batches+1)"

    # ---------------- 7. regression: non-batching model still serves 1:1 ----------------
    Write-Output ""
    $body1 = '{"inputs":[{"name":"input","shape":[1,4],"datatype":"FP32","data":[1,2,3,4]}]}'
    $reg = Call-Server "POST" "$base/v2/models/sample_model/infer" $body1
    Assert-True (Verify-Body $reg 1 1 "sample_model") "sample_model (no batching) still returns y=2x+1"
} finally {
    Write-Output ""
    if ($script:failures -eq 0) { Write-Output "ALL DYNAMIC-BATCH TESTS PASSED" }
    else { Write-Output "$($script:failures) TEST(S) FAILED" }
    Write-Output "Stopping server."
    Stop-Process -Name inferlite -Force -ErrorAction SilentlyContinue
}
exit $script:failures
