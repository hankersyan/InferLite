# build.ps1 - Build InferLite with MSVC (VS2022) + CMake + Ninja.
$ErrorActionPreference = "Stop"

$vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found: $vcvars" }

$cmake = "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ninja = "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"

if (-not (Test-Path $cmake)) { throw "cmake not found: $cmake" }
if (-not (Test-Path $ninja)) { throw "ninja not found: $ninja" }

# Build the vcvars-call command. cmd strips the first set of quotes when called
# with /c, so we add an extra layer of quotes.
$cmd = "`"$vcvars`" >nul 2>&1 && `"$cmake`" -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=`"$ninja`""
Write-Host "== Configuring =="
cmd /c $cmd
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

$cmd2 = "`"$vcvars`" >nul 2>&1 && `"$cmake`" --build build"
Write-Host "== Building =="
cmd /c $cmd2
if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }

Write-Host "Build complete: build\inferlite.exe"
