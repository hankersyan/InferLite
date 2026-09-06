# make_release.ps1 - Build the full InferLite vX.Y release package.
#
# Produces a release under <repo>/dist/v<ver>/ containing:
#   cpu/      - self-contained bundle (HTTP + gRPC + OpenVINO): inferlite.exe
#               + every runtime DLL, so it runs with no extra installs.
#   gpu/      - GPU bundle (HTTP + gRPC + OpenVINO + TensorRT). The oversized
#               TensorRT runtime is NOT bundled here: gpu/ is self-supplied
#               (see README.md). Copy the DLLs from tensorrt/ (or a TensorRT
#               10.x bin/) next to inferlite.exe before running it.
#   tensorrt/ - optional TensorRT 10.x runtime DLLs (nvinfer_*.dll,
#               nvonnxparser_*.dll) so a gpu/ user can supply the runtime
#               straight from this package.
#   models/   - ready-to-run curated sample model repository (regenerated
#               manifest.json for the staged subset)
#   MANIFEST.json   - per-file SHA-256 integrity manifest
#   README.md       - quick start
#   RELEASE_NOTES.md
#   InferLite-<ver>-intel.zip  /  -gpu.zip  (cpu or gpu bundle + models + docs)
#
# Layout follows dist/v0.2.3: the gpu/ bundle is "self-supplied" - TensorRT
# DLLs are not placed inside gpu/ but under tensorrt/ for the user to copy.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\make_release.ps1 -Version 0.4
param([string]$Version = "0.4", [string]$Channel = "release", [string]$Branch = "")
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$GrpcRoot = "C:\Test\vcpkg\vcpkg-2024.12.16\installed\x64-windows"

if (-not $Branch) { $Branch = git -C $RepoRoot rev-parse --abbrev-ref HEAD }

$VerTag = if ($Version -match '^v') { $Version } else { "v$Version" }
$ReleaseDir = Join-Path $RepoRoot "dist\$VerTag"
$CpuBuild = Join-Path $RepoRoot "build-grpc-cpu"
$GpuBuild = Join-Path $RepoRoot "build-grpc-gpu"
# Curated model repository: ship the runnable device/demo models. The combined
# models/ repo also contains plugin/ensemble demo models (which need a real
# sample_plugin.dll and fail-fast at load when the shipped DLL is a stub) and
# the large human-pose model (kept out to stay lean; README shows how to add
# it). Exclude all three groups so the release models load on a clean host.
$ModelSrc = Join-Path $RepoRoot "models"
$ModelSkip = 'plugin|ensemble|human-pose'

Write-Host "== Releasing InferLite $VerTag to $ReleaseDir (branch: $Branch) =="

# --- 1. Build the CPU (gRPC + OpenVINO) and GPU (gRPC + OpenVINO + TensorRT) bundles ---
Write-Host "== Building CPU bundle (HTTP + gRPC + OpenVINO) =="
& (Join-Path $PSScriptRoot "build_grpc.ps1") -GrpcRoot $GrpcRoot -BuildDirName "build-grpc-cpu"
if ($LASTEXITCODE -ne 0) { throw "CPU bundle build failed" }

Write-Host "== Building GPU bundle (HTTP + gRPC + OpenVINO + TensorRT) =="
& (Join-Path $PSScriptRoot "build_grpc.ps1") -GrpcRoot $GrpcRoot -BuildDirName "build-grpc-gpu" -Gpu
if ($LASTEXITCODE -ne 0) { throw "GPU bundle build failed" }

# --- 2. Stage the release tree ---
if (Test-Path $ReleaseDir) { Remove-Item $ReleaseDir -Recurse -Force }
New-Item -ItemType Directory -Path "$ReleaseDir\cpu", "$ReleaseDir\gpu", "$ReleaseDir\tensorrt" | Out-Null

# Copy runtime artifacts (exe + DLLs) from each bundle dir.
Get-ChildItem $CpuBuild -File | Where-Object { $_.Extension -in @('.exe','.dll') } | ForEach-Object {
    Copy-Item $_.FullName (Join-Path "$ReleaseDir\cpu" $_.Name) -Force
}
# GPU bundle: everything except the TensorRT runtime DLLs goes to gpu/; the
# TensorRT DLLs go to tensorrt/ (self-supplied, following dist/v0.2.3).
Get-ChildItem $GpuBuild -File | Where-Object { $_.Extension -in @('.exe','.dll') } | ForEach-Object {
    if ($_.Name -match '^nvinfer.*\.dll$' -or $_.Name -match '^nvonnxparser.*\.dll$') {
        Copy-Item $_.FullName (Join-Path "$ReleaseDir\tensorrt" $_.Name) -Force
    } else {
        Copy-Item $_.FullName (Join-Path "$ReleaseDir\gpu" $_.Name) -Force
    }
}

Write-Host "== Staged bundle contents =="
Write-Host "cpu files:      $((Get-ChildItem "$ReleaseDir\cpu" -File).Count)"
Write-Host "gpu files:      $((Get-ChildItem "$ReleaseDir\gpu" -File).Count)"
Write-Host "tensorrt files: $((Get-ChildItem "$ReleaseDir\tensorrt" -File).Count)"

# Curated model repository: copy each model directory individually, skipping
# plugin/ensemble/human-pose, then regenerate manifest.json for the staged
# subset.
New-Item -ItemType Directory -Path "$ReleaseDir\models" | Out-Null
Get-ChildItem $ModelSrc -Directory | ForEach-Object {
    if ($_.Name -match $ModelSkip) { return }
    Copy-Item $_.FullName "$ReleaseDir\models\$($_.Name)" -Recurse -Force
}
& python (Join-Path $RepoRoot "tools\make_manifest.py") --repo "$ReleaseDir\models"
if ($LASTEXITCODE -ne 0) { throw "failed to regenerate release model manifest" }
Write-Host "shipped models: $((Get-ChildItem "$ReleaseDir\models" -Directory).Count)"

# --- 3. Write README.md and RELEASE_NOTES.md ---
$readme = @"
# InferLite $VerTag - Quick Start

InferLite is a Triton-compatible inference server with CPU (Intel/OpenVINO),
Intel NPU (OpenVINO), NVIDIA GPU (TensorRT), and a gRPC interface
(KServe/Triton v2 ``GRPCInferenceService``). The ``cpu/`` bundle is fully
self-contained - every runtime DLL is bundled next to ``inferlite.exe``. The
``gpu/`` bundle is **self-supplied**: because the TensorRT runtime is oversized,
its DLLs are not included inside ``gpu/``. They are provided in ``tensorrt/``
(or from your own TensorRT 10.x install) and you make them visible to
``inferlite.exe`` before launching the GPU bundle.

## 1. Choose a bundle
- ``cpu/`` - OpenVINO only (CPU, Intel NPU, Intel iGPU, AUTO) + HTTP + gRPC.
  **Self-contained** - no extra installs; run ``inferlite.exe`` directly.
- ``gpu/`` - OpenVINO **+ TensorRT** (adds NVIDIA GPU backend) + HTTP + gRPC.
  **Self-supplied TensorRT runtime** - the oversized TensorRT DLLs are not
  bundled inside ``gpu/``. Before the GPU bundle can start you must supply a
  TensorRT 10.x runtime (see section 2).

Both contain an ``inferlite.exe``; use the folder matching your host.

## 2. Run

**GPU prerequisites (``gpu/`` only).** The ``gpu/inferlite.exe`` binary is
linked against the TensorRT runtime, so it does not start until the runtime
DLLs are visible. Supply them from this package (``tensorrt/``) or from an
installed TensorRT 10.x:

```powershell
# Option A - copy the supplied TensorRT runtime next to the exe (recommended):
Copy-Item ..\tensorrt\*.dll .\      # run this from inside the gpu/ folder
# Option B - install TensorRT 10.x (verified against 10.16.1.11, cuda-12.9
#            build) and add its bin/ directory to PATH
```

- The following TensorRT runtime libraries are required:
  - ``nvinfer_10.dll``
  - ``nvinfer_plugin_10.dll``
  - ``nvinfer_lean_10.dll``
  - ``nvinfer_dispatch_10.dll``
  - ``nvinfer_vc_plugin_10.dll``
  - ``nvonnxparser_10.dll``
  - ``nvinfer_builder_resource_ptx_10.dll`` and the
    ``nvinfer_builder_resource_sm*_10.dll`` architecture resources
- The **CUDA 12** runtime (``cudart64_12.dll``) must also be reachable via
  ``PATH`` or next to the exe (a CUDA 12 runtime install provides it).
- TensorRT 10.x needs a GPU with **compute capability >= 7.5** (Pascal/SM 6.1
  is not supported). OpenVINO CPU/NPU/Intel-GPU models run fine without
  TensorRT - use the ``cpu/`` bundle for those.

```powershell
cd cpu        # or: cd gpu   (after supplying the TensorRT runtime)
.\inferlite.exe --model-repository=..\models --http-port=8100 --grpc-port=8101
```

## 3. Check status
```powershell
Invoke-WebRequest http://127.0.0.1:8100/v2/health/ready          # 200 when READY
Invoke-WebRequest http://127.0.0.1:8100/v2/health/detailed       # cpu/gpu.enabled + per-model device
# gRPC: grpcurl or the Python test client in the repo (scripts/test_grpc_server.ps1)
```

## 4. Infer (HTTP)
```powershell
Invoke-WebRequest -Uri http://127.0.0.1:8100/v2/models/sample_model/infer `
  -Method POST -ContentType application/json `
  -Body '{"inputs":[{"name":"input","shape":[1,4],"datatype":"FP32","data":[1,2,3,4]}]}'
# expects [3,5,7,9]  (y = 2*x + 1)
```

## 5. Infer (gRPC)
The gRPC interface exposes ``ServerLive``, ``ServerReady``, ``ServerMetadata``,
``ModelReady``, ``ModelMetadata``, ``ModelConfig``, ``ModelInfer`` and the
repository RPCs (``RepositoryIndex``/``RepositoryModelLoad``/
``RepositoryModelUnload``). The repo's ``scripts/test_grpc_server.ps1`` is a
ready-to-use client (requires Python with grpcio + the generated stubs).

## 6. Run as a Windows service
The same binary runs under the Service Control Manager. From an **elevated**
PowerShell:

```powershell
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
```

``--service`` runs under the SCM and falls back to a foreground console run when
launched manually; ``--service-name`` selects the service name (default
``InferLite``). A stop request triggers a graceful shutdown (listeners close,
audit log finalized).

## Notes
- Sample models are in ``models/``:
  - ``sample_model`` / ``multi_io_model`` - ``y = 2*x + 1``, single and
    multi-input/multi-output.
  - ``intel_cpu_model`` / ``intel_npu_model`` / ``intel_gpu_model`` /
    ``intel_auto_model`` - OpenVINO device variants (``KIND_CPU`` /
    ``KIND_NPU`` / ``KIND_GPU_INTEL`` / ``KIND_AUTO``).
  - ``batched_model`` - Triton-style ``max_batch_size: 1`` (config ``dims: [4]``,
    client shape ``[1, 4]``).
  - ``dynamic_batch_model`` - Triton ``dynamic_batching {}`` (``max_batch_size:
    8``, ``preferred_batch_size: [8]``): the scheduler merges concurrent
    requests into one execution.
  - ``priority_batch_model`` - priority levels + ``preserve_ordering`` on top of
    dynamic batching (``priority_levels: 3``, ``default_priority_level: 2``).
  - ``sequence_model`` - Triton ``sequence_batching {}`` stateful model
    (``output = input + state``; ``CORRID``/``START``/``END`` control tensors).
- Verify integrity with the SHA-256 values in ``MANIFEST.json``.
- **``gpu/`` is self-supplied:** the oversized TensorRT runtime is not bundled
  inside it - copy the DLLs from ``tensorrt/`` (or provide a TensorRT 10.x
  runtime, e.g. 10.16.1.11 cuda-12.9 build) before running ``gpu/inferlite.exe``.
- **TensorRT end-to-end execution requires a GPU with compute capability >= 7.5**
  (TensorRT 10.x dropped Pascal/SM 6.1 support). The server still runs and
  reports ``gpu.enabled:true``; it just cannot build an engine on older GPUs.
- See ``RELEASE_NOTES.md`` for full details, supported devices, and known issues.
"@
$readme | Set-Content (Join-Path $ReleaseDir "README.md") -Encoding utf8

$notes = @"
# InferLite $VerTag - Release Notes

**Release:** $VerTag (channel: $Channel)
**Release date:** $(Get-Date -Format "yyyy-MM-dd")
**Branch:** ``$Branch``
**Software version reported by server:** ``InferLite 2.0.0``
**Package:** ``dist/$VerTag`` - self-contained ``cpu/`` and self-supplied
``gpu/`` runtime bundles (HTTP + gRPC), optional ``tensorrt/`` runtime,
curated sample model repository, and release notes.

---

## What's new in $VerTag

**v0.4.0 (this release):**
- **Model management (Triton model-control modes).** ``--model-control-mode``
  selects the repository policy: ``none`` (default - load all models at
  startup, fail-fast on any invalid model), ``poll`` (startup and periodic
  ``--repository-poll-secs`` reload; hot-add/hot-remove/hot-update; a broken
  model is reported ``UNAVAILABLE``, never fatal) and ``explicit`` (load only
  ``--load-model`` names, then driven through the control API). The Triton
  repository-control endpoints are implemented over HTTP and gRPC:
  ``POST /v2/repository/index``, ``/v2/repository/models/<name>/load``
  (optional ``config`` proto-text override) and ``/unload`` (with
  ``unload_dependents``). Ensembles that depend on a reloaded/unloaded model
  are reloaded/unloaded with it. See ``scripts/test_model_control.ps1``.
- **Dynamic batching (request-combining).** A Triton ``dynamic_batching {}``
  block lets the scheduler coalesce queued concurrent requests into a single
  backend execution (total batch ``<= max_batch_size``) and slice the merged
  output back per request; ``preferred_batch_size`` and
  ``max_queue_delay_microseconds`` control batching latency. Only
  ``openvino`` models on ``KIND_CPU``/``KIND_AUTO`` are eligible. Demo model:
  ``dynamic_batch_model`` (``max_batch_size: 8``). See
  ``scripts/test_dynamic_batch.ps1``.
- **Priority scheduling.** ``dynamic_batching { priority_levels: N
  default_priority_level: D }`` schedules requests by priority (1 highest),
  with arrival order preserved inside a level. Clients set a priority via the
  Triton ``priority`` request parameter (HTTP ``parameters.priority``; gRPC
  ``parameters["priority"]``). ``preserve_ordering: true`` returns responses in
  request arrival order even when execution reorders them. Demo model:
  ``priority_batch_model``. See ``scripts/test_priority_ordering.ps1``.
- **Sequence batching (stateful models).** A Triton ``sequence_batching {}``
  block routes every request of one sequence (``CORRID`` control tensor) to the
  model's single sequence slot in arrival order; ``START``/``END`` mark
  boundaries and ``max_sequence_idle_microseconds`` frees stalled slots. Hidden
  ``state`` tensors (``input_name``/``output_name``) are owned by the scheduler
  and never sent/received by clients. Demo model: ``sequence_model``
  (``output = input + state``). See ``scripts/test_sequence_batch.ps1``.
- **Version policies.** Triton ``version_policy {}`` in ``config.pbtxt``:
  ``specific { versions: [...] }`` pins a version (fail-fast if it is not on
  disk), ``latest { num_versions: N }`` (default) and ``all {}`` select the
  highest version present. Clients can pin a version per request over HTTP
  (``/v2/models/<name>/versions/<v>/...``) and gRPC (``model_version``);
  InferLite serves one version per model name, so any other version is
  rejected. See ``scripts/test_version_policy.ps1``.
- **Model warmup.** A Triton ``model_warmup`` block runs zero-filled sample
  requests through the real scheduler at load time, before the model is marked
  ready (kernel compilation, first-touch allocations, plugin caches). A failed
  warmup leaves the model ``UNAVAILABLE`` (poll/explicit) or aborts startup
  fail-fast (none). See ``scripts/test_warmup.ps1``.
- **Cross-model rate limiter.** Models declare shared resources in
  ``instance_group { rate_limiter { ... } }``; with ``--rate-limit`` the server
  serializes executions that would over-subscribe the same resource (protects a
  single GPU from OOM/memory contention). ``--rate-limit-resource=<name>:<n>``
  raises a pool's capacity. See ``docs/RATE_LIMITER.md``.
- **Response cache.** A ``response_cache { enable: true }`` block answers
  identical requests from a bounded per-model LRU instead of executing the
  model again. The key covers model name, version, config hash, artifact hash
  and canonicalized inputs. ``--response-cache-max-entries`` /
  ``--response-cache-max-bytes`` bound the cache. See
  ``scripts/test_response_cache.ps1``.
- **Prometheus metrics.** ``GET /v2/metrics`` returns the Prometheus text
  exposition format (NVIDIA Triton ``nv_*`` metric names) for every loaded
  model: request success/failure, inference/execution counts, pending-request
  gauge, cumulative inference duration, per-priority completions, and
  response-cache counters.
- **Profiling tool.** ``scripts/profile.ps1`` attaches to a running server,
  auto-discovers every READY model and reports per-model latency percentiles
  (min/avg/p50/p90/p95/p99/max), throughput and a server-side execution-latency
  cross-check from ``/v2/metrics`` - console ``key=value`` output suitable as
  V&V latency evidence.
- **Protocol parity hygiene (P0).** KServe v2 route/error-code parity pass over
  HTTP and gRPC (version-pinned routes, repository RPCs, parameter handling).
- **New demo models + tools** in ``models/``: ``dynamic_batch_model``,
  ``priority_batch_model``, ``sequence_model`` (generated by
  ``tools/make_dynamic_batch_model.py`` / ``tools/make_sequence_model.py``).
- **Packaging (this release):** the GPU bundle is now **self-supplied** for
  TensorRT, following ``dist/v0.2.3``: the oversized TensorRT runtime is not
  placed inside ``gpu/``; the required DLLs ship separately in ``tensorrt/``
  for you to copy next to ``gpu/inferlite.exe`` (or supply from your own
  TensorRT 10.x install). ``cpu/`` remains fully self-contained.

**v0.2.3 (previous release):**
- **Windows service support.** ``inferlite.exe`` runs under the Service Control
  Manager as well as from a console/cmd window (``--install-service`` /
  ``--uninstall-service`` / ``--service`` / ``--service-name`` +
  ``scripts/service.ps1``).
- **Shared CLI argument grammar** between the console entry point and the
  service worker; **gRPC human-pose test** over ``ModelInfer``.

## What's in this package

````
dist/$VerTag/
|-- RELEASE_NOTES.md            # this file
|-- MANIFEST.json               # artifact list + SHA-256 checksums
|-- README.md                   # quick start
|-- cpu/                        # OpenVINO + HTTP + gRPC runtime (self-contained)
|   `-- inferlite.exe + *.dll (OpenVINO 2025.3 + TBB + gRPC runtimes)
|-- gpu/                        # OpenVINO + TensorRT + HTTP + gRPC runtime
|   `-- inferlite.exe + *.dll (OpenVINO + gRPC; TRT self-supplied)
|-- tensorrt/                   # optional TensorRT 10.x runtime DLLs
|   `-- nvinfer_*.dll, nvonnxparser_*.dll   (copy into gpu/ to enable TRT)
|-- models/                     # ready-to-run curated model repository
|   |-- sample_model/  intel_cpu_model/  intel_npu_model/
|   |-- intel_gpu_model/  intel_auto_model/  multi_io_model/
|   |-- batched_model/  dynamic_batch_model/  priority_batch_model/
|   |-- sequence_model/
|   `-- manifest.json
|-- InferLite-$VerTag-intel.zip # cpu/ + models/ + docs
`-- InferLite-$VerTag-gpu.zip   # gpu/ + models/ + docs
````

``cpu/`` is **fully self-contained** - all required runtime DLLs are bundled
next to ``inferlite.exe``. ``gpu/`` is **self-supplied** for TensorRT: it runs
after you make the ``tensorrt/`` DLLs (or a TensorRT 10.x runtime) visible to
``gpu/inferlite.exe``.

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
building fails. The GPU bundle still **runs and reports ``gpu.enabled:true``**
once the TensorRT runtime is supplied; end-to-end TRT inference requires a GPU
with compute capability **>= 7.5**.

---

## Quick start

````powershell
cd dist\$VerTag\cpu          # or: \gpu  (after copying tensorrt\*.dll into gpu\)
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

**``gpu/``** - everything in ``cpu/`` plus the GPU-enabled ``inferlite.exe``.
The TensorRT runtime is **not** bundled inside ``gpu/`` (self-supplied): copy
``tensorrt/*.dll`` (``nvinfer_10.dll``, ``nvinfer_lean_10.dll``,
``nvinfer_dispatch_10.dll``, ``nvinfer_vc_plugin_10.dll``,
``nvinfer_plugin_10.dll``, ``nvonnxparser_10.dll`` and the
``nvinfer_builder_resource_*_10.dll`` architecture resources) next to the exe,
or install TensorRT 10.x and put its ``bin/`` on ``PATH``. The CUDA 12 runtime
(``cudart64_12.dll``) must also be reachable via ``PATH`` or next to the exe.

---

## Fixes / improvements since v0.1
- **gRPC interface added** (opt-in at build; bundled since v0.2).
- **GPU link fix:** ``gpu_memory_manager.cpp`` is now compiled when the GPU
  backend is enabled (previously it was missing from the source list, causing
  unresolved ``GpuMemoryManager`` symbols).
- **Windows service support (v0.2.3):** run as a managed service under the SCM
  or as a normal console process.
- **Triton parity (v0.3/v0.4):** model-control modes + repository-control API,
  dynamic batching, priority levels, sequence batching, version policies,
  model warmup, cross-model rate limiter, response cache, Prometheus metrics.
- **Self-supplied GPU bundle (v0.4.0):** TensorRT runtime is not placed inside
  ``gpu/``; it ships separately in ``tensorrt/`` so you control which TensorRT
  runtime is used.

---

## Known issues / not yet validated
- End-to-end TensorRT inference requires SM >= 7.5 (see above).
- gRPC streaming is not implemented (unary RPCs only).
- The shipped ``models/`` contains only the self-contained device/demo models;
  the repo's plugin/ensemble demo models and the large human-pose model are
  excluded for portability (the README shows how to add the pose model).

---

*InferLite $VerTag - bundled CPU / Intel-NPU / NVIDIA-GPU support with a
KServe/Triton v2 gRPC interface and Triton-style scheduling (batching,
priorities, sequences, rate limiting, response cache).*
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

# --- 5. Convenience zips (cpu or gpu bundle + models + docs) ---
Write-Host "== Creating convenience zips =="
$docPaths = @(
    (Join-Path $ReleaseDir "README.md"),
    (Join-Path $ReleaseDir "RELEASE_NOTES.md"),
    (Join-Path $ReleaseDir "MANIFEST.json")
)
$intelZip = Join-Path $ReleaseDir "InferLite-$VerTag-intel.zip"
Compress-Archive -Path (@((Join-Path $ReleaseDir "cpu"), (Join-Path $ReleaseDir "models")) + $docPaths) `
    -DestinationPath $intelZip -CompressionLevel Optimal
$gpuZip = Join-Path $ReleaseDir "InferLite-$VerTag-gpu.zip"
Compress-Archive -Path (@((Join-Path $ReleaseDir "gpu"), (Join-Path $ReleaseDir "models")) + $docPaths) `
    -DestinationPath $gpuZip -CompressionLevel Optimal

Write-Host "Release complete: $ReleaseDir"
