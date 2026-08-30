# make_release.ps1 - Build the full InferLite vX.Y release package.
#
# Produces a self-contained release under <repo>/dist/v<ver>/ containing:
#   cpu/  - full build (HTTP + gRPC, OpenVINO backends)   inferlite.exe + runtime DLLs
#   gpu/  - full build (HTTP + gRPC, OpenVINO + TensorRT) inferlite.exe + runtime DLLs
#   models/ - ready-to-run sample model repository
#   MANIFEST.json  - per-file SHA-256 integrity manifest
#   README.md      - quick start
#   RELEASE_NOTES.md
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\make_release.ps1 -Version 0.2
param([string]$Version = "0.2", [string]$Channel = "early-access")
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$GrpcRoot = "C:\Test\vcpkg\vcpkg-2024.12.16\installed\x64-windows"

$Ver = $Version -replace '^0\.', 'v0.'           # 0.2 -> v0.2
$VerTag = if ($Version -match '^v') { $Version } else { "v$Version" }
$ReleaseDir = Join-Path $RepoRoot "dist\$VerTag"
$CpuBuild = Join-Path $RepoRoot "build-grpc-cpu"
$GpuBuild = Join-Path $RepoRoot "build-grpc-gpu"
# Curated model repository: ship the runnable device/multi-IO/sample models.
# The combined models/ repo also contains plugin/ensemble demo models; we ship
# only the self-contained device set to keep the release minimal and portable.
$ModelSrc = Join-Path $RepoRoot "models"

Write-Host "== Releasing InferLite $VerTag to $ReleaseDir =="

# --- 1. Build the CPU (gRPC + OpenVINO) and GPU (gRPC + OpenVINO + TensorRT) bundles ---
Write-Host "== Building CPU bundle (HTTP + gRPC + OpenVINO) =="
& (Join-Path $PSScriptRoot "build_grpc.ps1") -GrpcRoot $GrpcRoot -BuildDirName "build-grpc-cpu"
if ($LASTEXITCODE -ne 0) { throw "CPU bundle build failed" }

Write-Host "== Building GPU bundle (HTTP + gRPC + OpenVINO + TensorRT) =="
& (Join-Path $PSScriptRoot "build_grpc.ps1") -GrpcRoot $GrpcRoot -BuildDirName "build-grpc-gpu" -Gpu
if ($LASTEXITCODE -ne 0) { throw "GPU bundle build failed" }

# --- 2. Stage the release tree ---
if (Test-Path $ReleaseDir) { Remove-Item $ReleaseDir -Recurse -Force }
New-Item -ItemType Directory -Path "$ReleaseDir\cpu", "$ReleaseDir\gpu" | Out-Null

# Copy only runtime artifacts from the bundle dirs: the exe and all runtime
# DLLs. Exclude CMake/Ninja build metadata and static-lib artifacts.
foreach ($bundle in @(@{Src=$CpuBuild; Dst="$ReleaseDir\cpu"}, @{Src=$GpuBuild; Dst="$ReleaseDir\gpu"})) {
    Get-ChildItem $bundle.Src -File | ForEach-Object {
        if ($_.Extension -in @('.exe','.dll','.json','.pbtxt','.xml','.bin')) {
            Copy-Item $_.FullName (Join-Path $bundle.Dst $_.Name) -Force
        }
    }
}

# Curated model repository: copy each model directory individually, skipping
# the plugin/ensemble demo models, then regenerate manifest.json for the
# staged subset.
New-Item -ItemType Directory -Path "$ReleaseDir\models" | Out-Null
Get-ChildItem $ModelSrc -Directory | ForEach-Object {
    $name = $_.Name
    if ($name -match 'plugin|ensemble') { return }   # fail-fast at load: skip
    Copy-Item $_.FullName "$ReleaseDir\models\$name" -Recurse -Force
}
& python (Join-Path $RepoRoot "tools\make_manifest.py") --repo "$ReleaseDir\models"
if ($LASTEXITCODE -ne 0) { throw "failed to regenerate release model manifest" }

Write-Host "== Staged bundle contents =="
Write-Host "cpu files: $((Get-ChildItem "$ReleaseDir\cpu" -File).Count)"
Write-Host "gpu files: $((Get-ChildItem "$ReleaseDir\gpu" -File).Count)"

# --- 3. Write README.md and RELEASE_NOTES.md ---
$readme = @"
# InferLite $VerTag - Quick Start

InferLite is a Triton-compatible inference server with CPU (Intel/OpenVINO),
Intel NPU (OpenVINO), NVIDIA GPU (TensorRT), and a gRPC interface
(KServe/Triton v2 ``GRPCInferenceService``). This package is self-contained:
every runtime DLL is bundled next to ``inferlite.exe``.

## 1. Choose a bundle
- ``cpu/`` - OpenVINO only (CPU, Intel NPU, Intel iGPU, AUTO) + HTTP + gRPC.
- ``gpu/`` - OpenVINO **+ TensorRT** (adds NVIDIA GPU backend) + HTTP + gRPC.

Both contain an ``inferlite.exe``; use the folder matching your host.

## 2. Run
````powershell
cd cpu        # or: cd gpu
.\inferlite.exe --model-repository=..\models --http-port=8100 --grpc-port=8101
````

## 3. Check status
````powershell
Invoke-WebRequest http://127.0.0.1:8100/v2/health/ready          # 200 when READY
Invoke-WebRequest http://127.0.0.1:8100/v2/health/detailed       # cpu/gpu.enabled + per-model device
# gRPC: grpcurl or the Python test client in the repo (scripts/test_grpc_server.ps1)
````

## 4. Infer (HTTP)
````powershell
Invoke-WebRequest -Uri http://127.0.0.1:8100/v2/models/sample_model/infer `
  -Method POST -ContentType application/json `
  -Body '{"inputs":[{"name":"input","shape":[1,4],"datatype":"FP32","data":[1,2,3,4]}]}'
# expects [3,5,7,9]  (y = 2*x + 1)
````

## 5. Infer (gRPC)
The gRPC interface exposes ``ServerLive``, ``ServerReady``, ``ServerMetadata``,
``ModelReady``, ``ModelMetadata``, ``ModelConfig`` and ``ModelInfer``. The repo's
``scripts/test_grpc_server.ps1`` is a ready-to-use client (requires Python with
grpcio + the generated stubs).

## 6. Run as a Windows service
The same binary runs under the Service Control Manager. From an **elevated**
PowerShell:

````powershell
# install (auto-start, LocalSystem) and start
# IMPORTANT: the model repository MUST be an ABSOLUTE path. A Windows service
# starts with CWD = C:\Windows\System32, so a relative path (..\models) will
# not resolve and the service will fail to load models at start.
.\inferlite.exe --install-service --model-repository=C:\full\path\to\dist\$VerTag\models --http-port=8100 --grpc-port=8101
sc start InferLite
# or the convenience manager:
powershell -ExecutionPolicy Bypass -File ..\scripts\service.ps1 -Action install `
    -ModelRepository C:\full\path\to\dist\$VerTag\models -HttpPort 8100 -GrpcPort 8101
powershell -ExecutionPolicy Bypass -File ..\scripts\service.ps1 -Action start

# manage / remove
sc stop InferLite      # graceful stop
sc delete InferLite    # uninstall (elevated)
````

``--service`` runs under the SCM and falls back to a foreground console run when
launched manually; ``--service-name`` selects the service name (default
``InferLite``). A stop request triggers a graceful shutdown (listeners close,
audit log finalized).

## Notes
- Sample models are in ``models/`` (OpenVINO device variants of ``y = 2*x + 1``,
  ``multi_io_model`` with two inputs / two outputs, and ``batched_model`` showing
  Triton-style ``max_batch_size`` with config ``dims: [4]`` and client shape
  ``[1, 4]``).
- Verify integrity with the SHA-256 values in ``MANIFEST.json``.
- **TensorRT end-to-end execution requires a GPU with compute capability >= 7.5**
  (TensorRT 10.x dropped Pascal/SM 6.1 support). The ``gpu/`` bundle still runs
  and reports ``gpu.enabled:true``; it just cannot build an engine on older GPUs.
- See ``RELEASE_NOTES.md`` for full details, supported devices, and known issues.
"@
$readme | Set-Content (Join-Path $ReleaseDir "README.md") -Encoding utf8

$notes = @"
# InferLite $VerTag - Release Notes

**Release:** $VerTag (channel: $Channel)
**Release date:** $(Get-Date -Format "yyyy-MM-dd")
**Branch:** ``dev-0.2-grpc``
**Software version reported by server:** ``InferLite 2.0.0``
**Package:** ``dist/$VerTag`` - self-contained runtime bundles (CPU + GPU) with
all dependencies, sample model repository, and release notes.

---

## What's new in $VerTag

**v0.2.3 (this release):**
- **Windows service support.** ``inferlite.exe`` now runs under the Service
  Control Manager as well as from a console/cmd window:
  - ``--install-service`` / ``--uninstall-service`` (elevated) register/remove a
    Windows service that reproduces the install-time arguments from the
    registry (auto-start, LocalSystem by default, or a custom account).
  - ``--service`` runs under the SCM and **falls back to a normal foreground
    console run** when launched manually, so it is safe to type in a cmd window.
  - ``--service-name`` / ``--service-display`` / ``--install-service-user`` /
    ``--install-service-password`` control the service identity and account.
  - ``scripts/service.ps1`` manages the service (install/start/stop/restart/
    status/uninstall); ``scripts/test_service.ps1`` verifies console + service
    modes incl. graceful ``sc stop`` shutdown (SCM control handler -> graceful
    stop -> clean ``STOPPED``).
- **Shared CLI argument grammar.** The parser was extracted into a single
  ``parseServerOptions`` used by both the console entry point and the service
  worker, so both modes accept identical options.
- **gRPC human-pose test.** ``test_human_pose_estimation.py`` gained ``--grpc`` /
  ``--grpc-server``: the same keypoint-detection end-to-end test now runs over
  the gRPC ``ModelInfer`` RPC (large FP32 tensor), and ``test_grpc_server.ps1``
  includes it. Client-setup time is reported separately so HTTP vs gRPC timing
  is comparable.

**v0.2.2 (previous):**
- **Triton-style batching (``max_batch_size``)** implemented. Shapes now follow
  Triton's convention: ``max_batch_size: 0`` disables batching (tensor shapes
  match ``dims`` exactly); ``max_batch_size > 0`` enables a leading batch
  dimension on request/output tensors, where config ``dims`` are per-request and
  the accepted batch is ``1 <= B <= max_batch_size``.
- **``batched_model``** added (``max_batch_size: 1``): config ``dims: [4]`` while
  clients send/receive ``[1, 4]``. A shape missing the batch dim (``[4]``) or
  exceeding ``max_batch_size`` (``[2, 4]``) is rejected with ``INVALID_INPUT``.
- **``tools/make_batched_model.py``** generates the model; **``scripts/test_batch.ps1``**
  verifies the batch convention end-to-end.

**v0.2.1 (previous):**
- **Triton array-of-message ``input``/``output`` syntax** now supported in
  ``config.pbtxt``, e.g. ``input: [ { name: "x" dims: [1,4] }, ... ]`` (in
  addition to the repeated-message form ``input { ... }``).
- **Multi-input / multi-output model** ``multi_io_model`` added (2 inputs, 2
  outputs) to exercise the array syntax end-to-end.
- **Model repository consolidated** into a single ``models/`` (the duplicate
  ``models_verify/`` was removed; only representative models kept - device
  variants, sample, multi-IO, and one plugin/ensemble pipeline each).

**v0.2 (previous):**
- **gRPC interface** (Phase 5): a Triton/KServe v2-compatible
  ``GRPCInferenceService`` (``ServerLive``/``ServerReady``/``ServerMetadata``/
  ``ModelReady``/``ModelMetadata``/``ModelConfig``/``ModelInfer``). The gRPC and
  HTTP paths share one inference core (``InferLite::runInference``).
- Built against a **vcpkg source-built gRPC** (MSVC 19.44-matching ABI); the
  earlier prebuilt-DLL ABI crash is resolved.
- Repository reorganization: build/check/test scripts under ``scripts/``,
  third-party dependencies under ``third_party/``, releases under ``dist/``.

## What's in this package

````
dist/$VerTag/
|-- RELEASE_NOTES.md            # this file
|-- MANIFEST.json               # artifact list + SHA-256 checksums
|-- README.md                   # quick start
|-- cpu/                        # OpenVINO + HTTP + gRPC runtime (console + Windows service)
|   `-- inferlite.exe + *.dll (OpenVINO 2025.3 + TBB + gRPC runtimes)
|-- gpu/                        # OpenVINO + TensorRT + HTTP + gRPC runtime (console + Windows service)
|   `-- inferlite.exe + *.dll (OpenVINO + TensorRT 10.16 + CUDA 12.6 + gRPC)
`-- models/                     # ready-to-run sample model repository
    |-- sample_model/  intel_cpu_model/  intel_npu_model/
    |-- intel_gpu_model/  intel_auto_model/  multi_io_model/
    |-- batched_model/  (Triton max_batch_size: 1 demo)
    `-- manifest.json
````

Both ``cpu/`` and ``gpu/`` are **fully self-contained** - all required runtime
DLLs are bundled next to ``inferlite.exe``.

---

## Supported devices

| Backend | Config ``kind`` | Status in $VerTag |
|---------|---------------|----------------|
| CPU (Intel/OpenVINO) | ``KIND_CPU`` | [x] verified (inference correct) |
| Intel NPU (OpenVINO) | ``KIND_NPU`` | [x] code path & server support; falls back to CPU when no NPU hardware |
| Intel iGPU (OpenVINO) | ``KIND_GPU_INTEL`` | [x] code path; falls back to CPU when device can't compile |
| Intel AUTO (OpenVINO) | ``KIND_AUTO`` | [x] verified |
| NVIDIA GPU (TensorRT) | ``KIND_GPU`` | [x] build + server enablement verified; E2E execution requires SM >= 7.5 |

### Verified environment (this build)
- Windows 11 x64, MSVC / Visual Studio 2022, CMake + Ninja.
- OpenVINO 2025.3.0, CUDA v12.6, TensorRT 10.16.1.11 (cuda-12.9 build).
- gRPC 1.67 (vcpkg source-built, static gRPC core + DLL deps).
- Build-host GPU: NVIDIA GTX 1070 (SM 6.1 / Pascal).

### Known hardware limitation (NVIDIA GPU / TensorRT)
TensorRT 10.16 **dropped support for Pascal (SM < 7.5)**. On a GTX 1070 engine
building fails. The GPU bundle still **runs and reports ``gpu.enabled:true``**;
end-to-end TRT inference requires a GPU with compute capability **>= 7.5**.

---

## Quick start

````powershell
cd dist\$VerTag\cpu          # or: \gpu
.\inferlite.exe --model-repository=..\models --http-port=8100 --grpc-port=8101
````

````powershell
Invoke-WebRequest http://127.0.0.1:8100/v2/health/ready
Invoke-WebRequest http://127.0.0.1:8100/v2/health/detailed
# gRPC: scripts/test_grpc_server.ps1 (Python + grpcio + generated stubs)
````

---

## Runtime dependencies bundled

**``cpu/``** - OpenVINO 2025.3.0 runtime + plugins and TBB, plus the gRPC runtime
DLLs (protobuf, abseil, re2, c-ares, zlib, OpenSSL).

**``gpu/``** - everything in ``cpu/`` **plus** the TensorRT 10.16 runtime
(``nvinfer_10.dll``, ``nvinfer_lean_10.dll``, ``nvinfer_dispatch_10.dll``,
``nvinfer_vc_plugin_10.dll``, ``nvinfer_plugin_10.dll``, ``nvonnxparser_10.dll``,
``nvinfer_builder_resource_*.dll``) + CUDA 12.6 runtimes (``cudart64_12.dll``,
``nvrtc64_120_0.dll``, ``nvrtc-builtins64_126.dll``).

---

## Fixes / improvements since v0.1
- **gRPC interface added** (opt-in at build; bundled in this release).
- **GPU link fix:** ``gpu_memory_manager.cpp`` is now compiled when the GPU
  backend is enabled (previously it was missing from the source list, causing
  unresolved ``GpuMemoryManager`` symbols).
- **Self-contained GPU bundle:** TensorRT runtime DLLs are copied next to the
  exe so the server runs without TensorRT on ``PATH``.
- **Windows service support (v0.2.3):** run as a managed service under the SCM
  or as a normal console process; ``--install-service`` / ``--uninstall-service``
  / ``--service`` + ``scripts/service.ps1``.

---

## Known issues / not yet validated
- End-to-end TensorRT inference requires SM >= 7.5 (see above).
- gRPC streaming is not implemented (unary RPCs only).
- The shipped ``models/`` contains only the self-contained device/sample/multi-IO
  models; the repo's plugin/ensemble demo models are excluded for portability.

---

*InferLite $VerTag - bundled CPU / Intel-NPU / NVIDIA-GPU support with a
KServe/Triton v2 gRPC interface.*
"@
$notes | Set-Content (Join-Path $ReleaseDir "RELEASE_NOTES.md") -Encoding utf8

# --- 4. Generate MANIFEST.json (SHA-256 for every artifact, incl. docs) ---
Write-Host "== Generating MANIFEST.json =="
$manifest = [ordered]@{
    version = $Version
    release = "InferLite"
    release_date = Get-Date -Format "yyyy-MM-dd"
    artifacts = @()
}
Get-ChildItem $ReleaseDir -Recurse -File | Sort-Object FullName | ForEach-Object {
    $rel = $_.FullName.Substring($ReleaseDir.Length + 1)
    $hash = (Get-FileHash $_.FullName -Algorithm SHA256).Hash
    $manifest.artifacts += [ordered]@{
        path = $rel
        size_bytes = $_.Length
        sha256 = $hash
    }
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $ReleaseDir "MANIFEST.json") -Encoding utf8

Write-Host "Release complete: $ReleaseDir"
