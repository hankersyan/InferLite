# test_sequence_batch.ps1 - Verify Triton sequence batching on
# models/sequence_model (a stateful step model).
#
# The model computes output = input + state, state_out = state + 1, where
# `state`/`state_out` are hidden tensors owned by the sequence scheduler:
#   * every request carries START/END/CORRID control tensors (INT32 scalars)
#   * a new sequence starts with START=1 (state reset to zeros)
#   * requests of one sequence run in order and share the single sequence slot
#   * END=1 closes the sequence after the request
#   * an idle sequence is aborted after max_sequence_idle_microseconds and its
#     slot is released (a later sequence starts with fresh state)
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$build = Join-Path $RepoRoot "build"
$exe = Join-Path $build "inferlite.exe"
$port = 8016
$base = "http://127.0.0.1:$port"

$repo = Join-Path $build "repo_seq"
if (Test-Path $repo) { Remove-Item -Recurse -Force $repo }
New-Item -ItemType Directory -Path $repo | Out-Null
Copy-Item -Recurse (Join-Path $RepoRoot "models\sequence_model") $repo

Get-Process inferlite -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 400
$p = Start-Process -FilePath $exe `
    -ArgumentList "--model-repository=$repo --http-port=$port --max-queue-size=100 --http-threads=8" `
    -PassThru -RedirectStandardOutput (Join-Path $build "seq_out.txt") -RedirectStandardError (Join-Path $build "seq_err.txt")

$script:failures = 0
function Assert-True($cond, $msg) {
    if (-not $cond) { $script:failures++; Write-Output "FAIL: $msg" }
    else { Write-Output "PASS: $msg" }
}

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

# Body with the real input x plus the START/END/CORRID control tensors.
function New-SeqBody($x, $start, $end, $corrid) {
    return ('{{"inputs":[{{"name":"input","shape":[1],"datatype":"FP32","data":[{0}]}},' +
            '{{"name":"START","shape":[1],"datatype":"INT32","data":[{1}]}},' +
            '{{"name":"END","shape":[1],"datatype":"INT32","data":[{2}]}},' +
            '{{"name":"CORRID","shape":[1],"datatype":"INT32","data":[{3}]}}]}}') -f $x, $start, $end, $corrid
}

function Get-OutputValue($respText) {
    if (-not $respText.StartsWith("[200]")) { return $null }
    $json = ($respText -replace '^\[\d+\] ', '') | ConvertFrom-Json
    $outs = @($json.outputs)
    if ($outs.Count -ne 1) { return $null }
    $b64 = $outs[0].data
    $bytes = [System.Convert]::FromBase64String($b64)
    return [System.BitConverter]::ToSingle($bytes, 0)
}

try {
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
        Write-Output "SERVER NOT READY"; Get-Content (Join-Path $build "seq_err.txt") -ErrorAction SilentlyContinue
        exit 1
    }
    Write-Output "SERVER RUNNING pid=$($p.Id)"

    # ---- 1. config reports sequence_batching ----
    Write-Output ""
    $cfgBody = Call-Server "GET" "$base/v2/models/sequence_model/config"
    $cfg = ($cfgBody -replace '^\[\d+\] ', '') | ConvertFrom-Json
    Assert-True ($null -ne $cfg.sequence_batching) "config exposes sequence_batching"
    Assert-True ([int64]$cfg.sequence_batching.max_sequence_idle_microseconds -eq 300000) `
        "config idle timeout == 300000us"

    # ---- 2. a full sequence: state accumulates, END closes ----
    Write-Output ""
    $v1 = Get-OutputValue (Call-Server "POST" "$base/v2/models/sequence_model/infer" (New-SeqBody 10 1 0 7))
    $v2 = Get-OutputValue (Call-Server "POST" "$base/v2/models/sequence_model/infer" (New-SeqBody 20 0 0 7))
    $v3 = Get-OutputValue (Call-Server "POST" "$base/v2/models/sequence_model/infer" (New-SeqBody 5 0 1 7))
    Assert-True ($null -ne $v1 -and [Math]::Abs($v1 - 10.0) -lt 1e-3) "START step x=10 -> y=10 (state starts at 0)"
    Assert-True ($null -ne $v2 -and [Math]::Abs($v2 - 21.0) -lt 1e-3) "2nd step x=20 -> y=21 (state advanced once)"
    Assert-True ($null -ne $v3 -and [Math]::Abs($v3 - 7.0) -lt 1e-3) "END step x=5 -> y=7 (state advanced twice)"

    # ---- 3. a new sequence starts with fresh state ----
    Write-Output ""
    $v4 = Get-OutputValue (Call-Server "POST" "$base/v2/models/sequence_model/infer" (New-SeqBody 1 1 0 8))
    Assert-True ($null -ne $v4 -and [Math]::Abs($v4 - 1.0) -lt 1e-3) "new corrid after END starts with state 0"

    # ---- 4. errors: missing START on first request ----
    Write-Output ""
    $rNoStart = Call-Server "POST" "$base/v2/models/sequence_model/infer" (New-SeqBody 1 0 0 9)
    Assert-True ($rNoStart.StartsWith("[ERR 400]")) "first request without START=1 rejected"

    # ---- 5. idle timeout frees the slot with fresh state ----
    Write-Output ""
    $vi = Get-OutputValue (Call-Server "POST" "$base/v2/models/sequence_model/infer" (New-SeqBody 100 1 0 10))
    Assert-True ($null -ne $vi -and [Math]::Abs($vi - 100.0) -lt 1e-3) "idle-start corrid=10 -> y=100"
    Start-Sleep -Milliseconds 700   # > 300ms idle timeout
    $v5 = Get-OutputValue (Call-Server "POST" "$base/v2/models/sequence_model/infer" (New-SeqBody 2 1 0 11))
    Assert-True ($null -ne $v5 -and [Math]::Abs($v5 - 2.0) -lt 1e-3) `
        "idle-aborted corrid=10 freed the slot (corrid=11 starts fresh, y=2)"

    # ---- 6. missing control tensor is rejected ----
    Write-Output ""
    $noCorr = '{"inputs":[{"name":"input","shape":[1],"datatype":"FP32","data":[1]},{"name":"START","shape":[1],"datatype":"INT32","data":[1]},{"name":"END","shape":[1],"datatype":"INT32","data":[0]}]}'
    $rNoCorr = Call-Server "POST" "$base/v2/models/sequence_model/infer" $noCorr
    Assert-True ($rNoCorr.StartsWith("[ERR 400]")) "request missing CORRID control tensor rejected"
} finally {
    Stop-Process -Name inferlite -Force -ErrorAction SilentlyContinue
    Write-Output ""
    if ($script:failures -eq 0) { Write-Output "ALL SEQUENCE-BATCH TESTS PASSED" }
    else { Write-Output "$($script:failures) TEST(S) FAILED" }
}
exit $script:failures
