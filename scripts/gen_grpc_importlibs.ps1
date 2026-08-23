# gen_grpc_importlibs.ps1 - Regenerate gRPC/protobuf/absl DLL import libraries.
#
# The Anaconda gRPC package ships import libraries (grpc++.lib, grpc.lib,
# libprotobuf.lib, abseil_dll.lib, ...) that are STALE relative to the actual
# DLLs: e.g. grpc++.lib does not export grpc::Status::OK even though
# grpc++.dll does, which yields LNK2019 at link time. This script rebuilds
# each import library directly from its DLL exports so every exported symbol
# (including internal/static globals used by protoc-generated code) resolves.
#
# Output is written to <repo>/third_party/grpc/importlibs/. Call this once
# after first configure, or CMake will invoke it automatically when it detects
# a mismatch (see CMakeLists.txt).
param([string]$GrpcRoot = "C:\Apps\anaconda3\Library")
$ErrorActionPreference = "Stop"

# Repo root is the parent of this scripts/ directory.
$repoRoot = Split-Path -Parent $PSScriptRoot

# Locate dumpbin + lib from the newest MSVC toolset.
$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$msvcBase = Get-ChildItem (Join-Path $vs "VC\Tools\MSVC") -Directory |
    Sort-Object Name -Descending | Select-Object -First 1
$dumpbin = Join-Path $msvcBase.FullName "bin\Hostx64\x64\dumpbin.exe"
$libexe  = Join-Path $msvcBase.FullName "bin\Hostx64\x64\lib.exe"
if (-not (Test-Path $dumpbin)) { throw "dumpbin not found: $dumpbin" }
if (-not (Test-Path $libexe))  { throw "lib not found: $libexe" }

$bin = Join-Path $GrpcRoot "bin"
$out = Join-Path $repoRoot "third_party\grpc\importlibs"
New-Item -ItemType Directory -Force -Path $out | Out-Null

# DLLs -> (import lib name). Only regenerate ones we actually link against.
$targets = @{
    "grpc++.dll"          = "grpc++.lib"
    "grpc.dll"            = "grpc.lib"
    "gpr.dll"             = "gpr.lib"
    "libprotobuf.dll"     = "libprotobuf.lib"
    "abseil_dll.dll"      = "abseil_dll.lib"
    "re2.dll"             = "re2.lib"
    "cares.dll"           = "cares.lib"
    "zlib.dll"            = "zlib.lib"
}

foreach ($dllName in $targets.Keys) {
    $dll = Join-Path $bin $dllName
    if (-not (Test-Path $dll)) {
        Write-Host "SKIP (no dll): $dllName"
        continue
    }
    $def = Join-Path $out "$dllName.def"
    $outLib = Join-Path $out $targets[$dllName]

    $exports = & $dumpbin /exports $dll
    $names = @()
    foreach ($line in $exports) {
        if ($line -match '^\s*\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+(\S+)\s*$') {
            $names += $matches[1]
        }
    }
    if ($names.Count -eq 0) {
        Write-Host "WARN: no exports parsed from $dllName"
        continue
    }
    $base = [System.IO.Path]::GetFileNameWithoutExtension($dllName)
    $lines = @("LIBRARY $base", "EXPORTS")
    foreach ($n in $names) { $lines += "  $n" }
    [System.IO.File]::WriteAllLines($def, $lines, [System.Text.Encoding]::ASCII)

    & $libexe "/def:$def" "/out:$outLib" "/machine:x64" 2>&1 | Out-Null
    if (-not (Test-Path $outLib)) {
        Write-Host "FAIL: lib generation failed for $dllName"
        continue
    }
    Write-Host "OK: $dllName -> $($targets[$dllName]) ($($names.Count) symbols)"
}

# Stamp used by CMake to know regeneration is up-to-date.
[System.IO.File]::WriteAllText((Join-Path $out ".generated"),
    (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), [System.Text.Encoding]::ASCII)
Write-Host "Import libraries regenerated in $out"
