# test_batch_validated.ps1 - FDA validated-mode compliance for the default
# batching mode with batch size = 1 (models/batched_model, max_batch_size: 1).
#
# Verifies the compliance claims in docs/COMPLIANCE.md for the deterministic
# 1:1 batching posture:
#   * validated mode requires a manifest and self-test-gated readiness
#   * inference is deterministic (identical input+config => identical output)
#   * model integrity: tampering with the IR aborts startup (fail-fast)
#   * input validation is exact: a payload whose byte length does not match the
#     declared shape/type is rejected before reaching the backend
#   * the batch dimension is bounded by max_batch_size (B=2 on a B<=1 model is
#     rejected)
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$build = Join-Path $RepoRoot "build"
$exe = Join-Path $build "inferlite.exe"
$port = 8015
$base = "http://127.0.0.1:$port"

$repo = Join-Path $build "repo_batch_validated"
if (Test-Path $repo) { Remove-Item -Recurse -Force $repo }
New-Item -ItemType Directory -Path (Join-Path $repo "batched_model") | Out-Null
Copy-Item -Recurse (Join-Path $RepoRoot "models\batched_model\*") (Join-Path $repo "batched_model")

# --- manifest: same rule as the server (config_store.cpp / make_manifest.py) ---
function New-ModelManifest($repoDir) {
    $versionDir = Join-Path $repoDir "batched_model\1"
    $parts = @()
    foreach ($f in @("model.xml", "model.bin", "model.plan",
                     "model.npu_blob", "model.gpu_blob")) {
        $p = Join-Path $versionDir $f
        if (Test-Path -LiteralPath $p) {
            $parts += (Get-FileHash -Algorithm SHA256 -LiteralPath $p).Hash.ToLower()
        }
    }
    $combined = ($parts -join "")
    $bytes = [System.Text.Encoding]::ASCII.GetBytes($combined)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $digest = [System.BitConverter]::ToString($sha.ComputeHash($bytes)).Replace("-", "").ToLower()
    $obj = @{ version = "1"; models = @(@{ model_id = "batched_model"; version = "1"; sha256 = $digest }) }
    $text = $obj | ConvertTo-Json -Depth 5
    # No BOM (a UTF-8 BOM would break the server's strict JSON parser).
    [System.IO.File]::WriteAllText((Join-Path $repoDir "manifest.json"), $text,
                                   (New-Object System.Text.UTF8Encoding($false)))
}
New-ModelManifest $repo

$binPath = Join-Path $repo "batched_model\1\model.bin"
$origBin = [System.IO.File]::ReadAllBytes($binPath)

$script:failures = 0
function Assert-True($cond, $msg) {
    if (-not $cond) {
        $script:failures++
        Write-Output "FAIL: $msg"
    } else {
        Write-Output "PASS: $msg"
    }
}

function Start-ValidatedServer {
    $log = Join-Path $build "batchval_out.txt"
    $err = Join-Path $build "batchval_err.txt"
    Get-Process inferlite -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Milliseconds 400
    $p = Start-Process -FilePath $exe `
        -ArgumentList "--model-repository=$repo --validated-mode --http-port=$port --http-threads=4 --audit-log=$(Join-Path $build 'batchval_audit.log') --diagnostic-log=$err" `
        -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
    Start-Sleep -Seconds 1
    return @{ proc = $p; log = $log; err = $err }
}

function Wait-Ready($s, $timeoutSec) {
    for ($i = 0; $i -lt ($timeoutSec * 2); $i++) {
        if ($s.proc.HasExited) { return $false }
        try {
            $r = Invoke-WebRequest -Uri "$base/v2/health/ready" -UseBasicParsing -TimeoutSec 2
            if ($r.StatusCode -eq 200) { return $true }
        } catch { }
        Start-Sleep -Milliseconds 500
    }
    return $false
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
    return ('{{"inputs":[{{"name":"input","shape":[{0},4],"datatype":"FP32","data":"{1}"}}]}}' -f $B, $b64)
}

function Decode-Fp32($b64) {
    $bytes = [System.Convert]::FromBase64String($b64)
    $vals = @()
    for ($i = 0; $i -lt $bytes.Length; $i += 4) { $vals += [System.BitConverter]::ToSingle($bytes, $i) }
    return ,$vals
}

try {
    # ---- 1. validated startup: manifest present + self-test gated ready -------
    Write-Output ""
    $s = Start-ValidatedServer
    $ready = Wait-Ready $s 30
    Assert-True $ready "validated server reports READY after manifest verify + golden self-test"
    if (-not $ready) {
        Write-Output "--- stderr ---"; Get-Content $s.err -ErrorAction SilentlyContinue
        throw "server did not become ready"
    }
    Assert-True (-not $s.proc.HasExited) "server stays running"

    # ---- 2. deterministic inference (identical input => identical output) ----
    Write-Output ""
    $body = New-InferBody 1 1
    $r1 = Call-Server "POST" "$base/v2/models/batched_model/infer" $body
    $r2 = Call-Server "POST" "$base/v2/models/batched_model/infer" $body
    $ok1 = $r1.StartsWith("[200]")
    $ok2 = $r2.StartsWith("[200]")
    $json1 = ($r1 -replace '^\[\d+\] ', '') | ConvertFrom-Json
    $vals = Decode-Fp32 $json1.outputs[0].data
    $expected = @(3.0, 5.0, 7.0, 9.0)
    $match = $true
    for ($i = 0; $i -lt 4; $i++) {
        if ([Math]::Abs($vals[$i] - $expected[$i]) -gt 1e-3) { $match = $false }
    }
    $shapeOk = (@($json1.outputs[0].shape)[0] -eq 1)
    # Determinism: the produced outputs must be identical (the responses differ
    # only in their trace_id / model_name envelope).
    $json2 = ($r2 -replace '^\[\d+\] ', '') | ConvertFrom-Json
    $sameOutput = (($json1.outputs | ConvertTo-Json -Compress -Depth 6) -eq
                   ($json2.outputs | ConvertTo-Json -Compress -Depth 6))
    Assert-True ($ok1 -and $ok2 -and $shapeOk -and $match) "deterministic batch-1 inference returns y=2x+1 = [3,5,7,9]"
    Assert-True $sameOutput "identical request+config yields identical outputs (determinism)"

    # ---- 3. exact input length enforcement (payload shorter than shape) -------
    Write-Output ""
    $shortBody = '{"inputs":[{"name":"input","shape":[1,4],"datatype":"FP32","data":[1.0,2.0,3.0]}]}'
    $rShort = Call-Server "POST" "$base/v2/models/batched_model/infer" $shortBody
    Assert-True ($rShort.StartsWith("[ERR 400]") -and $rShort -match "data length") `
        "short payload (12 bytes for [1,4] FP32) rejected with INVALID_INPUT"

    # ---- 4. batch dimension bounded by max_batch_size = 1 ---------------------
    Write-Output ""
    $rBig = Call-Server "POST" "$base/v2/models/batched_model/infer" (New-InferBody 2 1)
    Assert-True ($rBig.StartsWith("[ERR 400]")) "B=2 request rejected on max_batch_size=1 model"

    # Audit trail written for the served requests.
    $audit = Join-Path $build "batchval_audit.log"
    Assert-True ((Test-Path $audit) -and ((Get-Item $audit).Length -gt 0)) "audit log written"
    Stop-Process -Name inferlite -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 500

    # ---- 5. integrity fail-fast: tamper model.bin, validated start aborts -----
    Write-Output ""
    $tampered = [byte[]]$origBin.Clone()
    $tampered[0] = ($tampered[0] -bxor 0xFF) -band 0xFF
    [System.IO.File]::WriteAllBytes($binPath, $tampered)
    $s2 = Start-ValidatedServer
    $exited = $false
    for ($i = 0; $i -lt 40; $i++) {
        if ($s2.proc.HasExited) { $exited = $true; break }
        Start-Sleep -Milliseconds 500
    }
    if (-not $exited) { Stop-Process -Id $s2.proc.Id -Force -ErrorAction SilentlyContinue }
    Assert-True $exited "server exits after tampering model.bin in validated mode"
    Assert-True (($s2.proc.ExitCode -ne 0)) "exit code is non-zero (fail-fast)"
    $errText = Get-Content $s2.err -Raw -ErrorAction SilentlyContinue
    Assert-True ($errText -match "hash mismatch") "stderr reports model file hash mismatch"
    Stop-Process -Name inferlite -Force -ErrorAction SilentlyContinue

    # Restore the original model file.
    [System.IO.File]::WriteAllBytes($binPath, $origBin)
} finally {
    Stop-Process -Name inferlite -Force -ErrorAction SilentlyContinue
    [System.IO.File]::WriteAllBytes($binPath, $origBin)
    Write-Output ""
    if ($script:failures -eq 0) { Write-Output "ALL BATCH-1 VALIDATED-MODE TESTS PASSED" }
    else { Write-Output "$($script:failures) TEST(S) FAILED" }
}
exit $script:failures
