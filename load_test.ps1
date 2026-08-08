# load_test.ps1 - Sustained concurrent load test against the running inferlite.
param([int]$Concurrency = 16, [int]$PerWorker = 20, [string]$Uri = "http://127.0.0.1:8000/v2/models/sample_model/infer")
$ErrorActionPreference = "Stop"
$body = '{"inputs":[{"name":"input","shape":[1,4],"datatype":"FP32","data":[5,6,7,8]}]}'

$sw = [System.Diagnostics.Stopwatch]::StartNew()
$jobs = @()
for ($w = 0; $w -lt $Concurrency; $w++) {
    $jobs += Start-Job -ScriptBlock {
        param($u, $b, $n)
        $ok = 0; $fail = 0
        for ($i = 0; $i -lt $n; $i++) {
            try {
                $r = Invoke-WebRequest -Uri $u -Method POST -Body $b -ContentType "application/json" -UseBasicParsing -TimeoutSec 60
                if ($r.StatusCode -eq 200) { $ok++ } else { $fail++ }
            } catch { $fail++ }
        }
        return @{ ok = $ok; fail = $fail }
    } -ArgumentList $Uri, $body, $PerWorker
}
$results = $jobs | Wait-Job | Receive-Job
$jobs | Remove-Job
$sw.Stop()
$totalOk = 0; $totalFail = 0
foreach ($r in $results) { $totalOk += [int]$r.ok; $totalFail += [int]$r.fail }
Write-Output ("elapsed_s=" + [math]::Round($sw.Elapsed.TotalSeconds, 2))
Write-Output ("total_ok=" + $totalOk + " total_fail=" + $totalFail)
