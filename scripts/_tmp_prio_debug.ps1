$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$build = Join-Path $RepoRoot "build"
$exe = Join-Path $build "inferlite.exe"
$port = 8014
$base = "http://127.0.0.1:$port"
$repo = Join-Path $build "repo_prio"
if (-not (Test-Path $repo)) {
    New-Item -ItemType Directory -Path $repo | Out-Null
    Copy-Item -Recurse (Join-Path $RepoRoot "models\priority_batch_model") $repo
}
Get-Process inferlite -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 400
$out = Join-Path $build "prio_out.txt"; $err = Join-Path $build "prio_err.txt"
$p = Start-Process -FilePath $exe -ArgumentList "--model-repository=$repo --model-control-mode=explicit --http-port=$port --max-queue-size=100 --http-threads=16" -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
for ($i=0; $i -lt 40; $i++) {
    try { $r = Invoke-WebRequest -Uri "$base/v2/health/ready" -UseBasicParsing -TimeoutSec 2
          if ($r.StatusCode -eq 200) { break } } catch {}
    Start-Sleep -Milliseconds 400
}
Write-Output "ready status=$((Invoke-WebRequest -Uri "$base/v2/health/ready" -UseBasicParsing).StatusCode)"

$uri = "$base/v2/repository/models/priority_batch_model/load"
$req = [System.Net.HttpWebRequest]::Create($uri)
$req.Method = "POST"; $req.ContentLength = 0
try {
  $resp = $req.GetResponse()
  $sr = New-Object IO.StreamReader($resp.GetResponseStream())
  Write-Output "LOAD OK [$([int]$resp.StatusCode)] $($sr.ReadToEnd())"
} catch [System.Net.WebException] {
  $er = $_.Exception.Response
  if ($er) {
    $sr = New-Object IO.StreamReader($er.GetResponseStream())
    Write-Output "LOAD ERR [$([int]$er.StatusCode)] $($sr.ReadToEnd())"
  } else { Write-Output "NETERR $($_.Exception.Message)" }
}
Write-Output "--- server stderr ---"
Get-Content $err -ErrorAction SilentlyContinue
Write-Output "--- server stdout ---"
Get-Content $out -ErrorAction SilentlyContinue
Stop-Process -Name inferlite -Force -ErrorAction SilentlyContinue
