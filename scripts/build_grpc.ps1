# build_grpc.ps1 - Build InferLite with gRPC interface (MSVC + CMake + Ninja).
# Usage:
#   powershell -ExecutionPolicy Bypass -File build_grpc.ps1
#   powershell -ExecutionPolicy Bypass -File build_grpc.ps1 -GrpcRoot <sdk> -Gpu
#
# Enables the gRPC interface (INFERLITE_ENABLE_GRPC) against the gRPC C++ SDK
# found via -GrpcRoot (default: the vcpkg source-built installation). The SDK
# must contain include/grpcpp, lib/grpc++.lib, and the protoc / grpc_cpp_plugin
# tools. Prefer a source-built (vcpkg) gRPC built with the same MSVC toolchain;
# a prebuilt DLL stack built with an older MSVC crashes on RPC dispatch (ABI
# mismatch). GPU (TensorRT) is OFF by default; pass -Gpu to also enable it.
param([switch]$Gpu, [string]$GrpcRoot = "C:\Test\vcpkg\vcpkg-2024.12.16\installed\x64-windows")
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
$OpenVinoRoot = "c:\tools\openvino\openvino_toolkit_windows_2025.3.0.19807.44526285f24_x86_64"
$BuildDir     = Join-Path $RepoRoot "build-grpc"

if (-not (Test-Path "$GrpcRoot\include\grpcpp\grpcpp.h")) {
    throw "gRPC headers not found at $GrpcRoot\include\grpcpp"
}
if (-not (Test-Path "$GrpcRoot\lib\grpc++.lib")) {
    throw "grpc++.lib not found at $GrpcRoot\lib"
}

$extra = ""
$label = "gRPC"
if ($Gpu) {
    $trt = "C:\Tools\nvidia\TensorRT-10.16.1.11.Windows.amd64.cuda-12.9"
    if (-not (Test-Path "$trt\include\NvInfer.h")) { throw "TensorRT include not found at $trt" }
    $extra = " -DTENSORRT_ROOT=`"$trt`""
    $label = "gRPC + GPU"
}

$cmd = "`"$vcvars`" >nul 2>&1 && `"$cmake`" -S `"$RepoRoot`" -B `"$BuildDir`" -G Ninja " +
       "-DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=`"$ninja`" " +
       "-DOPENVINO_ROOT=`"$OpenVinoRoot`" -DINFERLITE_ENABLE_GRPC=ON -DGRPC_ROOT=`"$GrpcRoot`"$extra"
Write-Host "== Configuring ($label) with GRPC_ROOT=$GrpcRoot =="
cmd /c $cmd
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

$cmd2 = "`"$vcvars`" >nul 2>&1 && `"$cmake`" --build `"$BuildDir`""
Write-Host "== Building =="
cmd /c $cmd2
if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }

Write-Host "gRPC build complete: $BuildDir\inferlite.exe"
