<#
.SYNOPSIS
Attach to a running inferlite and profile inference latency + throughput across
every READY model, grouped by resolved device.

.DESCRIPTION
profile.ps1 auto-discovers models via GET /v2/health/detailed and profiles every
model whose status is READY, one model at a time (sequential models, concurrent
requests per model). For each model it reports client-side latency percentiles
(min/avg/p50/p90/p95/p99/max) and throughput (req/s), plus a server-side
cross-check computed from the cumulative /v2/metrics counters
(nv_inference_request_success and nv_inference_compute_infer_duration_us) that
yields the average backend execution latency seen by the server.

The output is console-only key=value lines so it can be pasted into a V&V /
latency-evidence record (e.g. docs/VnV-phase-03.md).

Each request carries a distinct numeric payload so every inference is a real
execution (no cross-request response-cache pollution among profiled requests).

Models that cannot be profiled generically are reported as skipped with a
reason, e.g.:
  sequence_batching        - requires control inputs (CORRID/START/END)
  dynamic_dims             - config has a non-static dimension
  unsupported_datatype_*   - input type is not numeric/bool (BYTES/FP16/...)
  input_too_large_elems=N  - synthesized tensor exceeds -MaxInputElements

.PARAMETER BaseUrl
Server base URL, default http://127.0.0.1:8000.

.PARAMETER Concurrency
Concurrent requests per model (workers), default 4.

.PARAMETER RequestsPerWorker
Requests issued by each worker, default 25 (total = Concurrency x RequestsPerWorker).

.PARAMETER Warmup
Sequential warm-up requests per model before measurement, default 10.

.PARAMETER LaunchDelaySec
Seconds between spawning the workers and starting measurement. Keeps worker
process spin-up out of the measured wall clock, default 3.

.PARAMETER TimeoutSec
Per-request HTTP timeout in seconds, default 60.

.PARAMETER MaxInputElements
Skip models whose synthesized input exceeds this many elements, default 100000.

.PARAMETER Models
Optional comma-separated model-name filter. When omitted every READY model is
profiled.

.PARAMETER NoMetrics
Do not read /v2/metrics for the server-side execution-latency cross-check.

.EXAMPLE
scripts/profile.ps1 -BaseUrl http://127.0.0.1:8000

.EXAMPLE
scripts/profile.ps1 -Concurrency 8 -RequestsPerWorker 50 -Warmup 20 -Models sample_model,intel_cpu_model
#>
param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [int]$Concurrency = 4,
    [int]$RequestsPerWorker = 25,
    [int]$Warmup = 10,
    [int]$LaunchDelaySec = 3,
    [int]$TimeoutSec = 60,
    [int64]$MaxInputElements = 100000,
    [string[]]$Models = @(),
    [switch]$NoMetrics
)
$ErrorActionPreference = "Stop"

if ($Concurrency -lt 1) { Write-Output "invalid_concurrency=$Concurrency (must be >= 1)"; exit 1 }
if ($RequestsPerWorker -lt 1) { Write-Output "invalid_requests_per_worker=$RequestsPerWorker (must be >= 1)"; exit 1 }
if ($Models -is [string]) { $Models = @($Models -split ',') }

# ---------------------------------------------------------------- body builder
# Shared source for building a KServe infer body. Every request uses a payload
# derived from its global request id so consecutive requests are distinct.
$buildBodySrc = @'
function Build-BodyJson {
    param($meta, [int64]$rid)
    $ins = @()
    foreach ($inp in $meta) {
        $n = [int64]$inp.n
        $vals = [System.Collections.Generic.List[string]]::new()
        if ([string]$inp.dtype -eq "BOOL") {
            for ($j = 0; $j -lt $n; $j++) { $vals.Add([string](($rid + $j) % 2)) }
        } else {
            for ($j = 0; $j -lt $n; $j++) { $vals.Add([string](1 + (($rid + $j) % 8))) }
        }
        $shape = @($inp.shape | ForEach-Object { "$_" }) -join ","
        $nameEsc = ([string]$inp.name).Replace("\", "\\").Replace('"', '\"')
        $ins += '{"name":"' + $nameEsc + '","shape":[' + $shape +
                '],"datatype":"' + $inp.dtype + '","data":[' + ($vals -join ',') + ']}'
    }
    return '{"inputs":[' + ($ins -join ',') + ']}'
}
'@

# Worker script (in two pieces so `param(...)` stays the first statement when the
# shared body-builder function text is inserted after it): wait for the shared
# start barrier, then fire PerWorker requests.
$workerPrologue = @'
param(
    [string]$Uri,
    [string]$MetaJson,
    [int]$PerWorker,
    [int64]$IdStart,
    [datetime]$StartAtUtc,
    [int]$TimeoutSec
)
$ok = 0
$fail = 0
$lats = New-Object System.Collections.Generic.List[int]
try {
    $meta = @($MetaJson | ConvertFrom-Json)
    while ([DateTime]::UtcNow -lt $StartAtUtc) { Start-Sleep -Milliseconds 25 }
'@

$workerLoop = @'
    for ($k = 0; $k -lt $PerWorker; $k++) {
        $rid = $IdStart + $k
        $body = Build-BodyJson $meta $rid
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        try {
            $r = Invoke-WebRequest -Uri $Uri -Method Post -Body $body `
                -ContentType "application/json" -UseBasicParsing -TimeoutSec $TimeoutSec
            $sw.Stop()
            if ([int]$r.StatusCode -eq 200) {
                $ok++
                $lats.Add([int]$sw.Elapsed.TotalMilliseconds)
            } else {
                $fail++
            }
        } catch {
            $sw.Stop()
            $fail++
        }
    }
} catch {
    $fail = $fail + 1
    return @{ ok = $ok; fail = $fail; lats = @($lats.ToArray()); err = $_.Exception.Message }
}
return @{ ok = $ok; fail = $fail; lats = @($lats.ToArray()) }
'@

# ------------------------------------------------------------- helper functions
function Invoke-GetText {
    param([string]$Path)
    return (Invoke-WebRequest -Uri ($BaseUrl + $Path) -UseBasicParsing `
        -TimeoutSec $TimeoutSec).Content
}

function Test-Ready {
    for ($i = 0; $i -lt 60; $i++) {
        try {
            $c = Invoke-GetText "/v2/health/ready"
            if (($c | ConvertFrom-Json).status -eq "READY") { return $true }
        } catch { }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Get-Percentile {
    param([double[]]$Sorted, [double]$P)
    if ($Sorted.Length -eq 0) { return 0.0 }
    $i = [int][math]::Ceiling(($P / 100.0) * $Sorted.Length) - 1
    if ($i -lt 0) { $i = 0 }
    return $Sorted[$i]
}

function Get-PromU64 {
    param([string]$Text, [string]$Family, [string]$ModelName)
    $rx = '(?m)^' + [regex]::Escape($Family) + '\{model="' +
          [regex]::Escape($ModelName) + '",version="[^"]*"\}\s+([0-9]+)\s*$'
    $m = [regex]::Match($Text, $rx)
    if ($m.Success) { return [uint64]$m.Groups[1].Value }
    return $null
}

function New-InputPlan {
    param($Cfg)
    $res = @{ ok = $false }
    if ($Cfg.PSObject.Properties.Name -contains "sequence_batching") {
        $res.reason = "sequence_batching"
        return $res
    }
    $inputs = @($Cfg.inputs)
    if ($inputs.Count -eq 0) { $res.reason = "no_inputs"; return $res }
    $maxBatch = [int64]0
    if ($null -ne $Cfg.max_batch_size) { $maxBatch = [int64]$Cfg.max_batch_size }
    $plan = @()
    $totalElems = [int64]0
    $numTypes = @("INT8", "INT16", "INT32", "INT64",
                  "UINT8", "UINT16", "UINT32", "UINT64",
                  "FP32", "FP64", "BOOL")
    foreach ($inp in $inputs) {
        $dims = @($inp.dims | ForEach-Object { [int64]$_ })
        if ($dims.Count -eq 0) { $res.reason = "no_dims"; return $res }
        foreach ($d in $dims) {
            if ($d -lt 1) { $res.reason = "dynamic_dims"; return $res }
        }
        $dt = [string]$inp.data_type
        if ($numTypes -notcontains $dt) {
            $res.reason = "unsupported_datatype_$dt"
            return $res
        }
        $shape = @()
        if ($maxBatch -gt 0) { $shape += [int64]1 }   # client prepends the batch dim
        foreach ($d in $dims) { $shape += $d }
        $n = [int64]1
        foreach ($d in $shape) { $n *= $d }
        $totalElems += $n
        $plan += [pscustomobject]@{
            name  = [string]$inp.name
            shape = @($shape)
            dtype = $dt
            n     = $n
        }
    }
    if ($totalElems -gt $MaxInputElements) {
        $res.reason = "input_too_large_elems=$totalElems"
        return $res
    }
    $res.ok = $true
    $res.plan = $plan
    return $res
}

# Make Build-BodyJson available in this scope for warm-up requests.
Invoke-Expression $buildBodySrc

# ------------------------------------------------------------------ header
Write-Output "profile_tool=profile.ps1"
Write-Output "base_url=$BaseUrl"
Write-Output ("concurrency=" + $Concurrency + " requests_per_worker=" + $RequestsPerWorker +
              " total_per_model=" + ($Concurrency * $RequestsPerWorker) + " warmup=" + $Warmup)
Write-Output ("run_started_utc=" + ([DateTime]::UtcNow.ToString("o")))

if (-not (Test-Ready)) {
    Write-Output "server_ready=false"
    Write-Output "error=server not ready (is inferlite running on $BaseUrl ?)"
    exit 1
}

$det = (Invoke-GetText "/v2/health/detailed") | ConvertFrom-Json
Write-Output "server_overall=$($det.overall_status)"
Write-Output "software_version=$($det.software_version)"
Write-Output "openvino_version=$($det.openvino_version)"
if ($null -ne $det.gpu) { Write-Output "gpu_enabled=$($det.gpu.enabled)" }

$ready = @($det.models | Where-Object { $_.status -eq "READY" })
if ($Models.Count -gt 0) {
    $missing = @($Models | Where-Object { $_ -notin @($ready | ForEach-Object { $_.name }) })
    if ($missing.Count -gt 0) {
        Write-Output ("error=requested model(s) not READY: " + ($missing -join ","))
        exit 1
    }
    $ready = @($ready | Where-Object { $Models -contains $_.name })
}
$ready = @($ready | Sort-Object @{ Expression = { "$($_.device)" } }, @{ Expression = { "$($_.name)" } })
Write-Output ("ready_models=" + $ready.Count + " devices=" +
              ((@($ready.device | Sort-Object -Unique)) -join ","))

# ----------------------------------------------------------- profile each model
$profiled = 0
$skipped = 0
$totalOk = 0
$totalFail = 0
$measureSb = [scriptblock]::Create($workerPrologue + "`n" + $buildBodySrc + "`n" + $workerLoop)
$devAgg = @{}

foreach ($m in $ready) {
    $name = [string]$m.name
    $dev = [string]$m.device
    $uri = "$BaseUrl/v2/models/$name/infer"
    Write-Output ""
    Write-Output "model=$name device=$dev"

    # Resolve input plan from the served config.
    try {
        $cfg = (Invoke-GetText "/v2/models/$name/config") | ConvertFrom-Json
    } catch {
        Write-Output "skipped reason=config_error"
        $skipped++
        continue
    }
    $planRes = New-InputPlan $cfg
    if (-not $planRes.ok) {
        Write-Output ("skipped reason=" + $planRes.reason)
        $skipped++
        continue
    }
    $plan = @($planRes.plan)
    $planJson = ConvertTo-Json -Compress -Depth 6 -InputObject @($plan)

    # Warm-up (untimed) and readiness probe for this model.
    $wok = 0; $wfail = 0
    for ($k = 0; $k -lt $Warmup; $k++) {
        $body = Build-BodyJson $plan ([int64](1000000 + $k))   # distinct from measure ids
        try {
            $r = Invoke-WebRequest -Uri $uri -Method Post -Body $body `
                -ContentType "application/json" -UseBasicParsing -TimeoutSec $TimeoutSec
            if ([int]$r.StatusCode -eq 200) { $wok++ } else { $wfail++ }
        } catch { $wfail++ }
    }
    Write-Output ("warmup_ok=$wok warmup_fail=$wfail")
    if ($wfail -gt 0) {
        Write-Output "skipped reason=warmup_failures"
        $skipped++
        continue
    }

    $metricsBefore = $null
    if (-not $NoMetrics) {
        try { $metricsBefore = Invoke-GetText "/v2/metrics" } catch { $metricsBefore = $null }
    }

    # Steady-state measurement: Concurrency worker processes, aligned to $startAtUtc.
    $startAtUtc = [DateTime]::UtcNow.AddSeconds($LaunchDelaySec)
    $jobs = @()
    for ($w = 0; $w -lt $Concurrency; $w++) {
        $idStart = [int64]$Warmup + ([int64]$w * [int64]$RequestsPerWorker)
        $jobs += Start-Job -ScriptBlock $measureSb -ArgumentList `
            $uri, $planJson, $RequestsPerWorker, $idStart, $startAtUtc, $TimeoutSec
    }
    $null = $jobs | Wait-Job
    $wallSec = ([DateTime]::UtcNow - $startAtUtc).TotalSeconds
    if ($wallSec -lt 0) { $wallSec = 0.0 }

    $tok = 0; $tfail = 0; $allLats = @()
    foreach ($j in $jobs) {
        try {
            $jr = Receive-Job $j
            if ($jr) {
                $tok += [int]$jr.ok
                $tfail += [int]$jr.fail
                foreach ($l in @($jr.lats)) { $allLats += [double]$l }
                if ($jr.PSObject.Properties.Name -contains "err" -and $jr.err) {
                    Write-Output ("worker_error=" + $jr.err)
                }
            }
        } catch {
            $tfail += $RequestsPerWorker
            Write-Output ("worker_error=" + $_.Exception.Message)
        }
    }
    $jobs | Remove-Job

    $totalOk += $tok
    $totalFail += $tfail
    $profiled++

    $throughput = 0.0
    if ($wallSec -gt 0) { $throughput = $tok / $wallSec }
    Write-Output ("measure_ok=$tok measure_fail=$tfail measure_expected=" +
                  ($Concurrency * $RequestsPerWorker))
    Write-Output ("elapsed_s=" + [math]::Round($wallSec, 3) +
                  " throughput_req_s=" + [math]::Round($throughput, 2))

    if ($tok -gt 0 -and $allLats.Count -gt 0) {
        $sorted = @($allLats | Sort-Object | ForEach-Object { [double]$_ })
        $arr = [double[]]$sorted
        $min = Get-Percentile $arr 0
        $p50 = Get-Percentile $arr 50
        $p90 = Get-Percentile $arr 90
        $p95 = Get-Percentile $arr 95
        $p99 = Get-Percentile $arr 99
        $max = Get-Percentile $arr 100
        $avg = ($arr | Measure-Object -Average).Average
        Write-Output ("latency_ms_min=" + [math]::Round($min, 3) +
                      " latency_ms_avg=" + [math]::Round($avg, 3) +
                      " latency_ms_p50=" + [math]::Round($p50, 3) +
                      " latency_ms_p90=" + [math]::Round($p90, 3) +
                      " latency_ms_p95=" + [math]::Round($p95, 3) +
                      " latency_ms_p99=" + [math]::Round($p99, 3) +
                      " latency_ms_max=" + [math]::Round($max, 3))

        # Server-side execution-latency cross-check (cumulative counters).
        if (-not $NoMetrics) {
            try {
                $metricsAfter = Invoke-GetText "/v2/metrics"
                $succB = Get-PromU64 $metricsBefore "nv_inference_request_success" $name
                $succA = Get-PromU64 $metricsAfter "nv_inference_request_success" $name
                $execB = Get-PromU64 $metricsBefore "nv_inference_compute_infer_duration_us" $name
                $execA = Get-PromU64 $metricsAfter "nv_inference_compute_infer_duration_us" $name
                if ($null -ne $succB -and $null -ne $succA -and
                    $null -ne $execB -and $null -ne $execA) {
                    $succDelta = [uint64]($succA - $succB)
                    $execDelta = [uint64]($execA - $execB)
                    $srvAvgMs = 0.0
                    if ($succDelta -gt 0) {
                        $srvAvgMs = ($execDelta / 1000.0) / $succDelta
                    }
                    Write-Output ("server_success_delta=$succDelta" +
                                  " server_exec_us_delta=$execDelta" +
                                  " server_avg_exec_ms=" + [math]::Round($srvAvgMs, 3))
                } else {
                    Write-Output "server_metrics_delta=unavailable"
                }
            } catch {
                Write-Output "server_metrics_delta=unavailable"
            }
        }

        if (-not $devAgg.ContainsKey($dev)) {
            $devAgg[$dev] = @{ count = 0; p50Sum = 0.0; tputSum = 0.0; okSum = 0 }
        }
        $d = $devAgg[$dev]
        $d.count++
        $d.p50Sum += $p50
        $d.tputSum += $throughput
        $d.okSum += $tok
    } else {
        Write-Output "latency_ms_available=none"
    }
}

# ---------------------------------------------------------------- device rollup
Write-Output ""
foreach ($dev in @($devAgg.Keys | Sort-Object)) {
    $d = $devAgg[$dev]
    $avgP50 = 0.0
    if ($d.count -gt 0) { $avgP50 = $d.p50Sum / $d.count }
    Write-Output ("device_summary device=$dev models_profiled=" + $d.count +
                  " latency_p50_avg_ms=" + [math]::Round($avgP50, 3) +
                  " throughput_sum_req_s=" + [math]::Round($d.tputSum, 2) +
                  " ok_total=" + $d.okSum)
}

Write-Output ""
Write-Output ("profile_models_profiled=$profiled models_skipped=$skipped" +
              " ok_total=$totalOk fail_total=$totalFail")
if ($totalFail -gt 0) { exit 1 }
