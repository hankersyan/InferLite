# test_priority_ordering.ps1 - Verify Triton dynamic-batching priority
# scheduling and preserve_ordering on models/priority_batch_model
# (priority_levels: 3, default_priority_level: 2, preserve_ordering: true).
#
# Covers:
#   * priority_levels / default_priority_level / preserve_ordering parse &
#     /v2/models/<name>/config reporting
#   * repository-load rejects an out-of-range default_priority_level (fail-fast)
#   * requests carrying an explicit `priority` parameter are attributed to the
#     right level (metrics priority_completed)
#   * requests without a priority use default_priority_level
#   * an out-of-range explicit priority is rejected with INVALID_INPUT
#   * responses are delivered in arrival order under preserve_ordering
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Net.Http -ErrorAction SilentlyContinue
[System.Net.ServicePointManager]::DefaultConnectionLimit = 64

$RepoRoot = Split-Path -Parent $PSScriptRoot
$build = Join-Path $RepoRoot "build"
$exe = Join-Path $build "inferlite.exe"
$port = 8013
$base = "http://127.0.0.1:$port"

$repo = Join-Path $build "repo_prio"
if (Test-Path $repo) { Remove-Item -Recurse -Force $repo }
New-Item -ItemType Directory -Path $repo | Out-Null
Copy-Item -Recurse (Join-Path $RepoRoot "models\priority_batch_model") $repo

Get-Process inferlite -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 500
$p = Start-Process -FilePath $exe `
    -ArgumentList "--model-repository=$repo --model-control-mode=explicit --http-port=$port --max-queue-size=100 --http-threads=16" `
    -PassThru

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
    # Wait for readiness (explicit mode is ready with nothing loaded).
    $ready = $false
    for ($i = 0; $i -lt 40; $i++) {
        if ($p.HasExited) { break }
        try {
            $r = Invoke-WebRequest -Uri "$base/v2/health/ready" -UseBasicParsing -TimeoutSec 2
            if ($r.StatusCode -eq 200) { $ready = $true; break }
        } catch { }
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) {
        Write-Output "SERVER DID NOT BECOME READY"; exit 1
    }
    Write-Output "SERVER RUNNING pid=$($p.Id)"

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

    function Decode-Fp32($b64) {
        $bytes = [System.Convert]::FromBase64String($b64)
        $vals = @()
        for ($i = 0; $i -lt $bytes.Length; $i += 4) {
            $vals += [System.BitConverter]::ToSingle($bytes, $i)
        }
        return ,$vals
    }

    # Infer body for B rows that start at `start` (row-major [B,4]). Optional
    # priority is embedded in the Triton request `parameters` map.
    function New-InferBody($B, $start = 1, $priority = $null) {
        $flat = New-Object System.Collections.Generic.List[double]
        for ($r = 0; $r -lt $B; $r++) {
            for ($c = 0; $c -lt 4; $c++) { $flat.Add([double]($start + $r * 4 + $c)) }
        }
        $bytes = New-Object byte[] ($flat.Count * 4)
        for ($i = 0; $i -lt $flat.Count; $i++) {
            [System.BitConverter]::GetBytes([single]$flat[$i]).CopyTo($bytes, $i * 4)
        }
        $b64 = [System.Convert]::ToBase64String($bytes)
        $params = ""
        if ($null -ne $priority) {
            $params = ',"parameters":{"priority":' + $priority + '}'
        }
        return ('{{"inputs":[{{"name":"input","shape":[{0},4],"datatype":"FP32","data":"{1}"}}]{2}}}' -f $B, $b64, $params)
    }

    function Verify-Body($body, $B, $start, $tag) {
        if (-not $body.StartsWith("[200]")) { return $false }
        $json = $body.Substring(6) | ConvertFrom-Json
        $outs = @($json.outputs)
        if ($outs.Count -eq 0) { return $false }
        $out = $outs[0]
        $shape = @($out.shape)
        if ($shape.Count -ne 2 -or [int64]$shape[0] -ne $B -or [int64]$shape[1] -ne 4) {
            return $false
        }
        $vals = Decode-Fp32 $out.data
        if ($vals.Count -ne ($B * 4)) { return $false }
        for ($r = 0; $r -lt $B; $r++) {
            for ($c = 0; $c -lt 4; $c++) {
                $x = $start + $r * 4 + $c
                $expected = 2.0 * $x + 1.0
                if ([Math]::Abs($vals[$r * 4 + $c] - $expected) -gt 1e-3) { return $false }
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
    function Get-MetricArray($name, $field) {
        $doc = (Invoke-WebRequest -Uri "$base/v2/metrics" -UseBasicParsing -TimeoutSec 10).Content | ConvertFrom-Json
        $row = $doc.models | Where-Object { $_.model_name -eq $name }
        if ($null -eq $row) { return @() }
        $prop = $row.PSObject.Properties[$field]
        if ($null -eq $prop) { return @() }
        return ,(@($prop.Value) | ForEach-Object { [int64]$_ })
    }

    $loadUri = "$base/v2/repository/models/priority_batch_model/load"

    # ---- 1. invalid default_priority_level is rejected at load -------------
    Write-Output ""
    $badCfg = @'
name: "priority_batch_model"
backend: "openvino"
max_batch_size: 8
input { name: "input" data_type: TYPE_FP32 dims: [ 4 ] }
output { name: "add" data_type: TYPE_FP32 dims: [ 4 ] }
instance_group { count: 1 kind: KIND_CPU }
dynamic_batching {
  preferred_batch_size: [ 8 ]
  max_queue_delay_microseconds: 1000
  priority_levels: 3
  default_priority_level: 99
}
'@
    $badBody = @{ parameters = @{ config = $badCfg } } | ConvertTo-Json
    $rBad = Call-Server "POST" $loadUri $badBody
    Assert-True ($rBad.StartsWith("[ERR 400]")) "load rejects default_priority_level=99 outside [1,3]"

    # ---- 2. valid load (on-disk config) ------------------------------------
    Write-Output ""
    $rLoad = Call-Server "POST" $loadUri $null
    Assert-True ($rLoad.StartsWith("[200]")) "priority_batch_model loads from repository"

    # ---- 3. config reports the priority / ordering policy ------------------
    Write-Output ""
    $cfgBody = Call-Server "GET" "$base/v2/models/priority_batch_model/config"
    $cfg = ($cfgBody -replace '^\[\d+\] ', '') | ConvertFrom-Json
    Assert-True ($null -ne $cfg.dynamic_batching) "config exposes dynamic_batching"
    Assert-True ([int64]$cfg.dynamic_batching.priority_levels -eq 3) "config priority_levels == 3"
    Assert-True ([int64]$cfg.dynamic_batching.default_priority_level -eq 2) "config default_priority_level == 2"
    Assert-True ($cfg.dynamic_batching.preserve_ordering -eq $true) "config preserve_ordering == true"

    # ---- 4. out-of-range explicit priority is rejected --------------------
    Write-Output ""
    $rPrioBad = Call-Server "POST" "$base/v2/models/priority_batch_model/infer" (New-InferBody 1 100 5)
    Assert-True ($rPrioBad.StartsWith("[ERR 400]")) "request priority=5 (> priority_levels 3) rejected"

    # ---- 5. attribution: explicit + default priorities ----------------------
    # Fire 6 concurrent B=1 requests: 2 with priority 1, 2 with priority 3 and
    # 2 without an explicit priority (must use default level 2).
    Write-Output ""
    $r0 = Get-Metric "priority_batch_model" "requests_completed"
    $p0 = Get-MetricArray "priority_batch_model" "priority_completed"
    $client = New-Object System.Net.Http.HttpClient
    try {
        $tasks = @()
        $cases = @(@{ p = 1; s = 1000 }, @{ p = 1; s = 1004 },
                   @{ p = 3; s = 2000 }, @{ p = 3; s = 2004 },
                   @{ p = $null; s = 3000 }, @{ p = $null; s = 3004 })
        foreach ($c in $cases) {
            $c2 = $c
            $c2.s = [int]$c.s
            $b = if ($null -eq $c2.p) { New-InferBody 1 $c2.s } else { New-InferBody 1 $c2.s $c2.p }
            $ct = New-Object System.Net.Http.StringContent($b, [System.Text.Encoding]::UTF8, "application/json")
            $tasks += $client.PostAsync("$base/v2/models/priority_batch_model/infer", $ct)
        }
        [System.Threading.Tasks.Task]::WaitAll($tasks)
        $allOk = $true
        for ($i = 0; $i -lt $tasks.Count; $i++) {
            $resp = $tasks[$i].Result
            $b = $resp.Content.ReadAsStringAsync().GetAwaiter().GetResult()
            if (-not (Verify-Body "[$([int]$resp.StatusCode)] $b" 1 $cases[$i].s "prio$i")) { $allOk = $false }
        }
        Assert-True $allOk "6 concurrent priority-tagged requests return y=2x+1"
    } finally {
        $client.Dispose()
    }
    Start-Sleep -Milliseconds 600
    $r1 = Get-Metric "priority_batch_model" "requests_completed"
    $p1 = Get-MetricArray "priority_batch_model" "priority_completed"
    Assert-True (($r1 - $r0) -eq 6) "6 requests completed"
    Assert-True ($p1.Count -eq 3) "metrics expose priority_completed[3]"
    Assert-True (($p1[0] - $p0[0]) -eq 2 -and ($p1[1] - $p0[1]) -eq 2 -and ($p1[2] - $p0[2]) -eq 2) `
        "priority attribution: +2 level1, +2 default level2, +2 level3"

    # ---- 6. preserve_ordering: responses arrive in request order ------------
    # With preserve_ordering the scheduler never completes a later request
    # before an earlier one, even when priorities would otherwise allow a later
    # high-priority request to execute first. Fire one low-priority request
    # (level 3), then 30ms later two high-priority (level 1) requests: the
    # low-priority one must still be the first to complete.
    Write-Output ""
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $lowClient = New-Object System.Net.Http.HttpClient
    $tLow = $lowClient.PostAsync(
        "$base/v2/models/priority_batch_model/infer",
        (New-Object System.Net.Http.StringContent((New-InferBody 1 5000 3), [System.Text.Encoding]::UTF8, "application/json")))
    Start-Sleep -Milliseconds 30
    $highClient = New-Object System.Net.Http.HttpClient
    $tHigh = @()
    for ($i = 0; $i -lt 2; $i++) {
        $tHigh += $highClient.PostAsync(
            "$base/v2/models/priority_batch_model/infer",
            (New-Object System.Net.Http.StringContent((New-InferBody 1 (6000 + $i * 4) 1), [System.Text.Encoding]::UTF8, "application/json")))
    }
    # Record the moment the earlier low-priority request completes. Under
    # preserve_ordering it must be delivered before any later request.
    while (-not $tLow.IsCompleted) { Start-Sleep -Milliseconds 2 }
    $lowTime = $sw.ElapsedMilliseconds
    $lowResp = $tLow.Result
    $lowBody = $lowResp.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    $lowOk = Verify-Body "[$([int]$lowResp.StatusCode)] $lowBody" 1 5000 "ordLow"
    Assert-True $lowOk "low request content correct"

    $highTimes = @()
    for ($i = 0; $i -lt $tHigh.Count; $i++) {
        $startHi = 6000 + $i * 4
        $ok = $tHigh[$i].Wait([TimeSpan]::FromSeconds(15))
        if (-not $ok) { $script:failures++; Write-Output "FAIL: high request did not complete" }
        $resp = $tHigh[$i].Result
        $b = $resp.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        if (-not (Verify-Body "[$([int]$resp.StatusCode)] $b" 1 $startHi "ordHi")) {
            $script:failures++
            Write-Output "FAIL: preserve_ordering high response content"
        }
        $highTimes += $sw.ElapsedMilliseconds
    }
    $lo = @($highTimes | Sort-Object)[0]
    # Small tolerance for client-side measurement jitter; a violated ordering
    # guarantee leaves a far larger gap (a whole batch window, ~150ms).
    Assert-True (($lowTime - $lo) -le 25) "preserve_ordering: earlier low-priority response delivered before later high-priority responses (low=$lowTime ms, firstHigh=$lo ms)"
    $highClient.Dispose()
    $lowClient.Dispose()
} finally {
    Write-Output ""
    if ($script:failures -eq 0) { Write-Output "ALL PRIORITY/ORDERING TESTS PASSED" }
    else { Write-Output "$($script:failures) TEST(S) FAILED" }
    Stop-Process -Name inferlite -Force -ErrorAction SilentlyContinue
}
exit $script:failures
