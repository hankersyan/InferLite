# build_gpu.ps1 - Build InferLite with TensorRT GPU support (MSVC + CMake + Ninja).
# Usage:
#   powershell -ExecutionPolicy Bypass -File build_gpu.ps1
# Adjust TENSORRT_ROOT / OPENVINO_ROOT below as needed.
$ErrorActionPreference = "Stop"

# Repo root is the parent of this scripts/ directory.
$RepoRoot = Split-Path -Parent $PSScriptRoot

# --- Locate VS, cmake, ninja (same as build.ps1) ---
$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found: $vswhere" }
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "No Visual Studio with VC tools found." }
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
$cmake = Join-Path $vs "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ninja = Join-Path $vs "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found: $vcvars" }

# --- Deps ---
$TensorRtRoot = "C:\Tools\nvidia\TensorRT-10.16.1.11.Windows.amd64.cuda-12.9"
$OpenVinoRoot  = "c:\tools\openvino\openvino_toolkit_windows_2025.3.0.19807.44526285f24_x86_64"
$BuildDir      = Join-Path $RepoRoot "build-gpu"

if (-not (Test-Path "$TensorRtRoot\include\NvInfer.h")) {
    throw "TensorRT include not found at $TensorRtRoot"
}

$cmd = "`"$vcvars`" >nul 2>&1 && `"$cmake`" -S `"$RepoRoot`" -B `"$BuildDir`" -G Ninja " +
       "-DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=`"$ninja`" " +
       "-DOPENVINO_ROOT=`"$OpenVinoRoot`" -DTENSORRT_ROOT=`"$TensorRtRoot`""
Write-Host "== Configuring (GPU) =="
cmd /c $cmd
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

$cmd2 = "`"$vcvars`" >nul 2>&1 && `"$cmake`" --build `"$BuildDir`""
Write-Host "== Building =="
cmd /c $cmd2
if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }

Write-Host "GPU build complete: $BuildDir\inferlite.exe"
