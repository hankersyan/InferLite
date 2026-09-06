# test_response_cache.ps1 - Verify the Triton-style response cache
# (config.pbtxt `response_cache { enable: true }`) on deterministic models.
#
# Triton semantics exercised here:
#   * response_cache is an opt-in per model (config block); models without it
#     never consult or populate a cache.
#   * An identical request (same model generation, same input contents) is
#     answered from the cache without a scheduler/backend execution.
#   * The key is order-independent: two requests carrying the same tensors in a
#     different client order are equivalent (one cache entry, second request is
#     a hit).
#   * A different input is a miss and executes the model.
#   * A reload (even with an unchanged config) starts a fresh, empty cache; a
#     load with a config override (config hash change) also invalidates it.
#   * /v2/metrics (Prometheus text format, mirroring NVIDIA Triton) reports
#     nv_cache_lookup_count / nv_cache_hit_count / nv_cache_num_entries per
#     cached model, along with nv_inference_request_success for executions.
#
# Runs against the HTTP-only build (build\inferlite.exe) in --model-control-
# mode=explicit so the repository load API can be used for the reload test.
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$build = Join-Path $RepoRoot "build"
$exe = Join-Path $build "inferlite.exe"
$port = 8014
$base = "http://127.0.0.1:$port"

if (-not (Test-Path $exe)) { throw "not found: $exe (run scripts\build.ps1 first)" }

$repo = Join-Path $build "repo_cache"
if (Test-Path $repo) { Remove-Item -Recurse -Force $repo }

function New-RepoModel($srcName, $dstName, $enableCache) {
    $dst = Join-Path $repo $dstName
    Copy-Item -Recurse (Join-Path $RepoRoot "models\$srcName") $dst
    $cfgPath = Join-Path $dst "config.pbtxt"
    $cfg = Get-Content $cfgPath -Raw
    $cfg = $cfg -replace [regex]::Escape($srcName), $dstName
    if ($enableCache) {
        $cfg = $cfg.TrimEnd() + "`nresponse_cache {`n  enable: true`n}`n"
    }
    Set-Content -Path $cfgPath -Value $cfg -Encoding ascii
    return $dst
}

New-Item -ItemType Directory -Path $repo | Out-Null
$cachedDir = New-RepoModel "sample_model" "cached_model" $true
$plainDir = New-RepoModel "sample_model" "plain_model" $false
$multiDir = New-RepoModel "multi_io_model" "multi_cached_model" $true

$serverLog = Join-Path $build "rc_out.txt"
$serverErr = Join-Path $build "rc_err.txt"

Get-Process inferlite -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 600
$p = Start-Process -FilePath $exe `
    -ArgumentList "--model-repository=$repo --http-port=$port --max-queue-size=100 --http-threads=4", `
        "--model-control-mode=explicit", `
        "--load-model=cached_model", `
        "--load-model=plain_model", `
        "--load-model=multi_cached_model" `
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
        return [pscustomobject]@{ Status = [int]$resp.StatusCode; Body = $sr.ReadToEnd() }
    } catch [System.Net.WebException] {
        $er = $_.Exception.Response
        if ($er) {
            $sr = New-Object IO.StreamReader($er.GetResponseStream())
            return [pscustomobject]@{ Status = [int]$er.StatusCode; Body = $sr.ReadToEnd() }
        }
        return [pscustomobject]@{ Status = 0; Body = $_.Exception.Message }
    }
}

# ---- /v2/metrics scraping ---------------------------------------------------
# InferLite mirrors NVIDIA Triton: /v2/metrics returns the Prometheus text
# exposition format, one sample per model with {model, version} labels.
# Get-Metric maps the JSON-style field names used by the assertions below to
# the Triton metric family that carries them:
#   requests_completed -> nv_inference_request_success
#   cache_lookups      -> nv_cache_lookup_count
#   cache_hits         -> nv_cache_hit_count
#   cache_entries      -> nv_cache_num_entries
# A model that has no sample for the family returns $null (e.g. plain_model
# has no nv_cache_lookup_count series because it does not enable a cache).
function Get-Metric($name, $field) {
    $metric = switch ($field) {
        "requests_completed" { "nv_inference_request_success" }
        "cache_lookups"      { "nv_cache_lookup_count" }
        "cache_hits"         { "nv_cache_hit_count" }
        "cache_entries"      { "nv_cache_num_entries" }
        default              { $null }
    }
    if (-not $metric) { return $null }
    $text = (Invoke-WebRequest -Uri "$base/v2/metrics" -UseBasicParsing -TimeoutSec 10).Content
    $pattern = '(?m)^' + [regex]::Escape($metric) + '\{model="' + [regex]::Escape($name) + '",version="[^"]*"\} (?<value>[0-9]+)\s*$'
    $m = [regex]::Match($text, $pattern)
    if (-not $m.Success) { return $null }
    return [int64]$m.Groups['value'].Value
}

# Server-level aggregate: sums one metric family over every model sample.
function Get-TopMetric($field) {
    $metric = switch ($field) {
        "cache_lookups" { "nv_cache_lookup_count" }
        default         { $null }
    }
    if (-not $metric) { return $null }
    $text = (Invoke-WebRequest -Uri "$base/v2/metrics" -UseBasicParsing -TimeoutSec 10).Content
    $pattern = '(?m)^' + [regex]::Escape($metric) + '\{[^}]*\} (?<value>[0-9]+)\s*$'
    $total = [int64]0
    foreach ($m in [regex]::Matches($text, $pattern)) {
        $total += [int64]$m.Groups['value'].Value
    }
    return $total
}

# sample_model body: input FP32 [1,4]; output y = 2*x + 1.
function New-Fp32Body($name, [int64[]]$vals) {
    return ('{{"inputs":[{{"name":"input","shape":[1,4],"datatype":"FP32","data":[{0}]}}]}}' -f `
        (($vals | ForEach-Object { "$_" }) -join ","))
}

function Decode-Fp32($b64) {
    $bytes = [System.Convert]::FromBase64String($b64)
    $vals = @()
    for ($i = 0; $i -lt $bytes.Length; $i += 4) {
        $vals += [System.BitConverter]::ToSingle($bytes, $i)
    }
    return ,$vals
}

function Verify-SampleBody($resp, [int64[]]$vals, $tag) {
    if ($resp.Status -ne 200) {
        Write-Output "   ($tag) status $($resp.Status): $($resp.Body)"
        return $false
    }
    $json = $resp.Body | ConvertFrom-Json
    if ($null -eq $json.outputs -or @($json.outputs).Count -eq 0) { return $false }
    $out = @($json.outputs)[0]
    if (@($out.shape).Count -ne 2 -or [int64]@($out.shape)[1] -ne 4) { return $false }
    $got = Decode-Fp32 $out.data
    if ($got.Count -ne 4) { return $false }
    for ($i = 0; $i -lt 4; $i++) {
        $expected = 2.0 * [double]$vals[$i] + 1.0
        if ([Math]::Abs([double]$got[$i] - $expected) -gt 1e-3) {
            Write-Output "   ($tag) i=$i val=$($got[$i]) expected=$expected"
            return $false
        }
    }
    return $true
}

# multi_io_model body with a specific client tensor order. Outputs are compared
# as name -> base64 data maps between responses.
function New-MultiBody([int64[]]$a, [int64[]]$b, $reverse) {
    $aJson = ($a | ForEach-Object { "$_" }) -join ","
    $bJson = ($b | ForEach-Object { "$_" }) -join ","
    $ta = '{{"name":"input_a","shape":[1,4],"datatype":"FP32","data":[{0}]}}' -f $aJson
    $tb = '{{"name":"input_b","shape":[1,4],"datatype":"FP32","data":[{0}]}}' -f $bJson
    if ($reverse) { return '{"inputs":[' + $tb + ',' + $ta + ']}' }
    return '{"inputs":[' + $ta + ',' + $tb + ']}'
}

function Get-OutputMap($resp) {
    $map = @{}
    if ($resp.Status -ne 200) { return $map }
    $json = $resp.Body | ConvertFrom-Json
    foreach ($o in @($json.outputs)) {
        $map[$o.name] = $o.data
    }
    return $map
}

function Same-Outputs($a, $b) {
    if ($a.Count -ne $b.Count) { return $false }
    foreach ($k in $a.Keys) {
        if (-not $b.ContainsKey($k)) { return $false }
        if ($a[$k] -ne $b[$k]) { return $false }
    }
    return $true
}

try {
    # Wait for readiness (up to 60s).
    $ready = $false
    for ($i = 0; $i -lt 120; $i++) {
        if ($p.HasExited) { break }
        $r = Call-Server "GET" "$base/v2/health/ready"
        if ($r.Status -eq 200) { $ready = $true; break }
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) {
        Write-Output "SERVER DID NOT BECOME READY"
        Write-Output "--- stdout ---"; Get-Content $serverLog -ErrorAction SilentlyContinue
        Write-Output "--- stderr ---"; Get-Content $serverErr -ErrorAction SilentlyContinue
        exit 1
    }
    Write-Output "SERVER RUNNING pid=$($p.Id)"

    # ---------------- 1. config exposes the opt-in; plain model has none ------
    Write-Output ""
    $cfgC = (Call-Server "GET" "$base/v2/models/cached_model/config").Body | ConvertFrom-Json
    $cfgP = (Call-Server "GET" "$base/v2/models/plain_model/config").Body | ConvertFrom-Json
    Assert-True ($null -ne $cfgC.response_cache -and $cfgC.response_cache.enable) "config cached_model reports response_cache { enable: true }"
    Assert-True ($null -eq $cfgP.response_cache) "config plain_model reports no response_cache"

    $valsA = @([int64]1, 2, 3, 4)
    $valsB = @([int64]100, 101, 102, 103)

    # ---------------- 2. identical requests are served from the cache --------
    Write-Output ""
    $l0 = Get-Metric "cached_model" "cache_lookups"
    $h0 = Get-Metric "cached_model" "cache_hits"
    $d0 = Get-Metric "cached_model" "requests_completed"
    Assert-True ($l0 -eq 0 -and $h0 -eq 0) "fresh model starts with zero cache lookups/hits"

    $r1 = Call-Server "POST" "$base/v2/models/cached_model/infer" (New-Fp32Body "cached_model" $valsA)
    Assert-True (Verify-SampleBody $r1 $valsA "cached#1") "first request executes and returns y=2x+1 (cache miss)"
    Start-Sleep -Milliseconds 200
    $l1 = Get-Metric "cached_model" "cache_lookups"
    $h1 = Get-Metric "cached_model" "cache_hits"
    $d1 = Get-Metric "cached_model" "requests_completed"
    Assert-True (($l1 - $l0) -eq 1 -and ($h1 - $h0) -eq 0) "first request: lookups+1, hits+0"
    Assert-True (($d1 - $d0) -eq 1) "first request executed the backend once"

    $r2 = Call-Server "POST" "$base/v2/models/cached_model/infer" (New-Fp32Body "cached_model" $valsA)
    Assert-True (Verify-SampleBody $r2 $valsA "cached#2") "identical second request returns y=2x+1"
    Start-Sleep -Milliseconds 200
    $l2 = Get-Metric "cached_model" "cache_lookups"
    $h2 = Get-Metric "cached_model" "cache_hits"
    $d2 = Get-Metric "cached_model" "requests_completed"
    Assert-True (($l2 - $l1) -eq 1 -and ($h2 - $h1) -eq 1) "identical second request: lookups+1, hits+1"
    Assert-True (($d2 - $d1) -eq 0) "identical second request did NOT execute the backend"
    $e2 = Get-Metric "cached_model" "cache_entries"
    Assert-True ($e2 -eq 1) "cache holds one entry after the identical pair"

    # ---------------- 3. a different input is a miss -------------------------
    Write-Output ""
    $r3 = Call-Server "POST" "$base/v2/models/cached_model/infer" (New-Fp32Body "cached_model" $valsB)
    Assert-True (Verify-SampleBody $r3 $valsB "cached#3") "different input returns y=2x+1"
    Start-Sleep -Milliseconds 200
    $l3 = Get-Metric "cached_model" "cache_lookups"
    $h3 = Get-Metric "cached_model" "cache_hits"
    $d3 = Get-Metric "cached_model" "requests_completed"
    Assert-True (($l3 - $l2) -eq 1 -and ($h3 - $h2) -eq 0) "different input: lookups+1, hits+0"
    Assert-True (($d3 - $d2) -eq 1) "different input executed the backend once"
    $e3 = Get-Metric "cached_model" "cache_entries"
    Assert-True ($e3 -eq 2) "cache holds both distinct inputs"

    # A repeat of the very first input is a hit again (LRU keeps both entries).
    $r4 = Call-Server "POST" "$base/v2/models/cached_model/infer" (New-Fp32Body "cached_model" $valsA)
    Assert-True (Verify-SampleBody $r4 $valsA "cached#4") "repeat of input A returns y=2x+1"
    Start-Sleep -Milliseconds 200
    $h4 = Get-Metric "cached_model" "cache_hits"
    $d4 = Get-Metric "cached_model" "requests_completed"
    Assert-True (($h4 - $h3) -eq 1) "repeat of a cached input is a hit"
    Assert-True (($d4 - $d3) -eq 0) "cached hit did not execute the backend"

    # ---------------- 4. cache key is independent of tensor order ------------
    Write-Output ""
    $mA = @([int64]1, 2, 3, 4)
    $mB = @([int64]11, 12, 13, 14)
    $mC = @([int64]21, 22, 23, 24)
    $cl0 = Get-Metric "multi_cached_model" "cache_lookups"
    $ch0 = Get-Metric "multi_cached_model" "cache_hits"
    $cd0 = Get-Metric "multi_cached_model" "requests_completed"

    $m1 = Call-Server "POST" "$base/v2/models/multi_cached_model/infer" (New-MultiBody $mA $mB $false)
    Assert-True ($m1.Status -eq 200) "multi-input request (a then b) returns 200"
    $om1 = Get-OutputMap $m1
    Start-Sleep -Milliseconds 200
    $cl1 = Get-Metric "multi_cached_model" "cache_lookups"
    $ch1 = Get-Metric "multi_cached_model" "cache_hits"
    $cd1 = Get-Metric "multi_cached_model" "requests_completed"
    Assert-True (($cl1 - $cl0) -eq 1 -and ($ch1 - $ch0) -eq 0) "multi-input request: lookups+1, hits+0"
    Assert-True (($cd1 - $cd0) -eq 1) "multi-input request executed the backend once"

    $m2 = Call-Server "POST" "$base/v2/models/multi_cached_model/infer" (New-MultiBody $mA $mB $true)
    $om2 = Get-OutputMap $m2
    Assert-True ($m2.Status -eq 200) "same tensors in reverse order returns 200"
    Assert-True (Same-Outputs $om1 $om2) "reverse-order request returns byte-identical outputs"
    Start-Sleep -Milliseconds 200
    $cl2 = Get-Metric "multi_cached_model" "cache_lookups"
    $ch2 = Get-Metric "multi_cached_model" "cache_hits"
    $cd2 = Get-Metric "multi_cached_model" "requests_completed"
    Assert-True (($cl2 - $cl1) -eq 1 -and ($ch2 - $ch1) -eq 1) "reverse-order request hit the same cache entry"
    Assert-True (($cd2 - $cd1) -eq 0) "reverse-order request did NOT execute the backend"

    $m3 = Call-Server "POST" "$base/v2/models/multi_cached_model/infer" (New-MultiBody $mA $mC $false)
    Assert-True ($m3.Status -eq 200) "changed second input returns 200"
    Start-Sleep -Milliseconds 200
    $cl3 = Get-Metric "multi_cached_model" "cache_lookups"
    $ch3 = Get-Metric "multi_cached_model" "cache_hits"
    $cd3 = Get-Metric "multi_cached_model" "requests_completed"
    Assert-True (($cl3 - $cl2) -eq 1 -and ($ch3 - $ch2) -eq 0) "changed input: lookups+1, hits+0"
    Assert-True (($cd3 - $cd2) -eq 1) "changed input executed the backend once"

    # ---------------- 5. reload invalidates the cache ------------------------
    Write-Output ""
    $rl = Call-Server "POST" "$base/v2/repository/models/cached_model/load"
    Assert-True ($rl.Status -eq 200) "repository load (reload) of cached_model succeeds"
    # The reload swaps in a fresh scheduler AND a fresh (empty) cache.
    $cl = Get-Metric "cached_model" "cache_lookups"
    $ch = Get-Metric "cached_model" "cache_hits"
    $ce = Get-Metric "cached_model" "cache_entries"
    $dn = Get-Metric "cached_model" "requests_completed"
    Assert-True ($cl -eq 0 -and $ch -eq 0 -and $ce -eq 0) "reload starts a fresh, empty response cache"
    $r5 = Call-Server "POST" "$base/v2/models/cached_model/infer" (New-Fp32Body "cached_model" $valsA)
    Assert-True (Verify-SampleBody $r5 $valsA "reload#1") "request after reload returns y=2x+1"
    Start-Sleep -Milliseconds 200
    Assert-True ((Get-Metric "cached_model" "cache_lookups") -eq 1) "post-reload identical request is a miss (lookups=1)"
    Assert-True ((Get-Metric "cached_model" "cache_hits") -eq 0) "post-reload identical request hits+0"
    Assert-True ((Get-Metric "cached_model" "requests_completed") -eq ($dn + 1)) "post-reload request executed the backend"
    $r6 = Call-Server "POST" "$base/v2/models/cached_model/infer" (New-Fp32Body "cached_model" $valsA)
    Assert-True (Verify-SampleBody $r6 $valsA "reload#2") "repeat after reload returns y=2x+1"
    Start-Sleep -Milliseconds 200
    Assert-True ((Get-Metric "cached_model" "cache_hits") -eq 1) "repeat after reload is cached again (hits=1)"

    # ---------------- 6. config override (config hash) invalidates ------------
    # Reload with an override whose text differs only by a trailing comment:
    # same semantics, different config hash -> the cache key changes and the
    # previously stored entry no longer matches.
    Write-Output ""
    $cfgPath = Join-Path $cachedDir "config.pbtxt"
    $cfgText = (Get-Content $cfgPath -Raw).TrimEnd() + "`n# cache-key config-hash marker`n"
    $esc = $cfgText.Replace('\', '\\').Replace('"', '\"')
    $esc = $esc.Replace("`r`n", '\n').Replace("`n", '\n')
    $loadBody = '{"parameters":{"config":"' + $esc + '"}}'
    $rl2 = Call-Server "POST" "$base/v2/repository/models/cached_model/load" $loadBody
    Assert-True ($rl2.Status -eq 200) "repository load with config override succeeds"
    $cl2 = Get-Metric "cached_model" "cache_lookups"
    $ch2 = Get-Metric "cached_model" "cache_hits"
    Assert-True ($cl2 -eq 0 -and $ch2 -eq 0) "config-hash-changing override starts an empty cache"
    $r7 = Call-Server "POST" "$base/v2/models/cached_model/infer" (New-Fp32Body "cached_model" $valsA)
    Assert-True (Verify-SampleBody $r7 $valsA "override#1") "request after config override returns y=2x+1"
    Start-Sleep -Milliseconds 200
    Assert-True ((Get-Metric "cached_model" "cache_hits") -eq 0) "identical input after config override is a miss (config hash in key)"

    # ---------------- 7. models without response_cache never cache -----------
    Write-Output ""
    Assert-True ($null -eq (Get-Metric "plain_model" "cache_lookups")) "plain_model exposes no cache_lookups metric"
    $pd0 = Get-Metric "plain_model" "requests_completed"
    $pa = Call-Server "POST" "$base/v2/models/plain_model/infer" (New-Fp32Body "plain_model" $valsA)
    Assert-True (Verify-SampleBody $pa $valsA "plain#1") "plain_model first request returns y=2x+1"
    $pb = Call-Server "POST" "$base/v2/models/plain_model/infer" (New-Fp32Body "plain_model" $valsA)
    Assert-True (Verify-SampleBody $pb $valsA "plain#2") "plain_model identical second request returns y=2x+1"
    Start-Sleep -Milliseconds 200
    $pd1 = Get-Metric "plain_model" "requests_completed"
    Assert-True (($pd1 - $pd0) -eq 2) "plain_model executed the backend for BOTH identical requests (no cache)"

    # ---------------- 8. server-level aggregate counters exist ---------------
    Write-Output ""
    $agg = Get-TopMetric "cache_lookups"
    Assert-True ($null -ne $agg -and $agg -gt 0) "server-level cache_lookups aggregate is present and non-zero"
} finally {
    Write-Output ""
    if ($script:failures -eq 0) { Write-Output "ALL RESPONSE-CACHE TESTS PASSED" }
    else { Write-Output "$($script:failures) TEST(S) FAILED" }
    Write-Output "Stopping server."
    Stop-Process -Name inferlite -Force -ErrorAction SilentlyContinue
}
exit $script:failures
