# build.ps1 - Build InferLite with MSVC (VS2022) + CMake + Ninja.
# Auto-detects the Visual Studio installation via vswhere.
$ErrorActionPreference = "Stop"

# Repo root is the parent of this scripts/ directory.
$RepoRoot = Split-Path -Parent $PSScriptRoot

$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found: $vswhere" }

$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "No Visual Studio with VC tools found." }

$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
$cmake = Join-Path $vs "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ninja = Join-Path $vs "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"

if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found: $vcvars" }
if (-not (Test-Path $cmake)) { throw "cmake not found: $cmake" }
if (-not (Test-Path $ninja)) { throw "ninja not found: $ninja" }

# Build the vcvars-call command. cmd strips the first set of quotes when called
# with /c, so we add an extra layer of quotes.
$buildDir = Join-Path $RepoRoot "build"
$cmd = "`"$vcvars`" >nul 2>&1 && `"$cmake`" -S `"$RepoRoot`" -B `"$buildDir`" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=`"$ninja`""
Write-Host "== Configuring =="
cmd /c $cmd
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

$cmd2 = "`"$vcvars`" >nul 2>&1 && `"$cmake`" --build `"$buildDir`""
Write-Host "== Building =="
cmd /c $cmd2
if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }

Write-Host "Build complete: $buildDir\inferlite.exe"
