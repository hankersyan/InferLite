# InferLite

A lightweight, deterministic, single-node inference server for a fixed
industrial workstation. It adopts the proven, reference-aligned architecture of
a production serving framework — model repository, backend abstraction, bounded
scheduling, and reusable memory — while dropping all cloud-scale, multi-tenant,
and dynamic operations that add unnecessary complexity on the factory floor.

This repository implements **Phase 1** (see `docs/PRD-phase-01.md`),
**Phase 2** (see `docs/PRD-phase-02.md`), and **Phase 4** (see
`docs/PRD-phase-04.md`, Intel CPU / NPU / GPU / AUTO multi-device execution);
`docs/PRD-all.md` describes the full product direction. See
[`docs/COMPLIANCE.md`](docs/COMPLIANCE.md) for the FDA / medical-device
compliance posture.

## Motivations
- Nvidia phased out the triton inference server's windows support.

## Vision (from `docs/PRD-all.md`)

- **Reference-aligned architecture** — model repository layout, model
  configuration, backend interface, instance groups, and tensor sharing.
- **Multi-backend support** — native backends for GPU acceleration and CPU.
- **C++ plugin system** — custom pre-/post-processing nodes running inside the
  pipeline graph as if they were backends.
- **Concurrent execution** — multiple model instances and plugin nodes run in
  parallel.
- **Zero-copy pipelines** — data is passed by reference between adjacent steps
  on the same device.
- **Deterministic, low-latency serving** — static graph, no dynamic batching,
  no hot reloading, minimal scheduling overhead.
- **Standard interface** — a compatible, synchronous subset of the reference
  server's request API plus health and JSON metrics.

## Medical Equipment Level & FDA Compliance

InferLite is architected as a **Class C software component of a regulated
medical device** (IEC 62304), with model-integrity verification, deterministic
execution, input/output validation, fault isolation, resource limits, and a
tamper-evident audit trail. Full details — the regulatory framework, safety
mechanisms, validated (locked) mode, required documentation package, and
security/privacy notes — live in **[`docs/COMPLIANCE.md`](docs/COMPLIANCE.md)**.

## Features

### Completed

- **Model repository** — parses model configuration files, picks the highest
  numeric version directory for each model.
- **Scheduler** — bounded FIFO request queue with configurable depth and request
  timeout; at most `instance_group.count` requests run concurrently per model.
- **Memory management** — pool of reusable host buffers, pinned host buffers, and
  OpenCL device-buffer bookkeeping for device data staging.
- **CPU backend** — wraps the framework's compiled-model object.
- **Interface** — inference, readiness health, model config, and JSON metrics.
- **Fail-fast startup** — unsupported configurations abort the server.
- **Model integrity & traceability** — `manifest.json` with SHA-256 hashes
  (including precompiled NPU/GPU blobs); verified at startup; mismatches cause
  fail-fast refusal to start. `metadata.json` carries `model_id`, `version`,
  `intended_use`, `approval_status`.
- **Deterministic resource limits** — input/output size caps (50 MB default),
  per-request inference time limit (5000 ms), bounded queue.
- **Input/output validation** — strict tensor shape/type/size checks; output
  NaN/Inf/range detection; structured error codes (`INVALID_INPUT`,
  `OUTPUT_VALIDATION_FAILED`, `RESOURCE_EXHAUSTED`, `TIMEOUT`, ...).
- **Fault isolation** — every backend call is exception-contained and returns a
  `BackendResult`; no silent failures.
- **Tamper-evident audit log** — append-only, hash-chained JSON entries
  (`trace_id`, model/config hashes, software version, duration, device).
- **Health & startup self-test** — golden-input verification per model; server
  reports `READY` only if all self-tests pass. `GET /v2/health/detailed` gives
  per-model status.
- **Config & version management** — config/manifest hashing, software + OpenVINO
  version reporting via `GET /v2/versions`.
- **CPU ensemble DAG executor** — static `ensemble_scheduling` steps with
  zero-copy host memory passthrough; a step failure cancels the whole pipeline.
- **C++ plugin backend** — shared libraries implementing the InferLite plugin
  ABI, loaded at startup, hash-verified against the manifest, and executed on
  the CPU thread pool.
- **Metrics** — queue depth, per-model latency, and a configuration hash.
- **Intel CPU execution** — compiles the IR (`model.xml`/`model.bin`) on the
  OpenVINO CPU plugin; thread/stream tuning applied only where the plugin
  accepts it.
- **Intel NPU execution** — loads a precompiled `model.npu_blob` via
  `ov::Core::import_model`.
- **Intel GPU execution** — loads a precompiled `model.gpu_blob` via
  `ov::Core::import_model`.
- **Intel AUTO execution** — lets OpenVINO select the best available Intel device
  (NPU > GPU > CPU); imports an existing blob for that device or compiles the IR.
- **Triton-compatible device selection** — `instance_group.kind` chooses the
  execution device: `KIND_CPU`, `KIND_NPU`, `KIND_GPU_INTEL`, or `KIND_AUTO`.
- **Device reporting** — health/detailed, metrics, and audit logs report the
  resolved execution device per model (`CPU`, `NPU`, `INTEL_GPU`, `AUTO`).
- **Device model tooling** — `tools/make_device_models.py` generates CPU/NPU/GPU/
  AUTO sample models, exporting precompiled blobs when the corresponding Intel
  hardware is present; reference configs live in `tools/examples/`.

### Todo

- **NVIDIA GPU (TensorRT) execution** — a dedicated TensorRT backend for
  `KIND_GPU` (NVIDIA) models.
- **GPU ensembles** — zero-copy DAG pipelines executing across device memory.
- **Dynamic batching** — combining concurrent requests into a single inference.
- **Live model updates** — reloading or hot-swapping models at runtime.
- **Profiling tool** — latency and throughput profiling across devices.
- **Batch API** — Triton-compatible `batching` and batch inference endpoints.
- **gRPC interface** — Triton-compatible gRPC inference, health, and model
  endpoints (currently HTTP only).

## Layout

```
models/
  <model_name>/
    config.pbtxt
    metadata.json         # FDA model metadata (optional)
    manifest.json         # repository-level approved-model manifest (validated mode)
    <version>/            # highest numeric version is used
      model.xml
      model.bin
src/
  main.cpp                 # CLI + fail-fast startup
  infer_lite.*             # app wiring + routing + audit/validation orchestration
  http_server.*            # HTTP/1.1 server (thread pool)
  json.*                   # minimal JSON parser/serializer
  pbtxt.*                  # model configuration parser
  model_repository.*       # repository scan + validation
  backend.hpp              # abstract backend interface (BackendResult)
  openvino_backend.*       # OpenVINO backend (CPU / NPU / Intel GPU / AUTO)
  plugin_backend.*         # C++ plugin backend (shared-library ABI)
  plugin_api.hpp           # plugin ABI (inferlite_plugin_*)
  ensemble_executor.*      # CPU ensemble DAG executor (zero-copy host memory)
  scheduler.*              # bounded FIFO scheduler (with inference time limit)
  memory_manager.*         # host + pinned + device-buffer memory pools
  audit_log.*              # tamper-evident hash-chained audit log
  config_store.*           # manifest/metadata/self-test/hash management
  validation.*             # input/output validation + structured error codes
  sha256.*                 # SHA-256 hashing
  tensor.hpp               # tensor/data-type definitions
  diagnostics.*            # engineer-facing diagnostic log
tools/
  make_sample_model.py     # generates models/sample_model (IR + config + metadata)
  make_device_models.py    # generates CPU/NPU/GPU/AUTO device sample models
  make_manifest.py         # generates models/manifest.json with SHA-256 hashes
  examples/                # reference configs (kind: KIND_NPU / KIND_GPU_INTEL / KIND_AUTO)
  sample_plugin/           # example CPU plugin source (sample_plugin.dll)
```

## Build

Prerequisites:

- Microsoft Visual Studio 2022 (MSVC C++ x64 toolset)
- CMake + Ninja (shipped with VS)
- OpenVINO 2025.3 Windows C++ runtime (extracted under `c:\tools\openvino`, or
  point CMake at it with `-DOPENVINO_ROOT=...`)

```
powershell -ExecutionPolicy Bypass -File build.ps1
```

The build also produces `build\sample_plugin.dll` (example plugin). To use the
plugin/ensemble demo models, copy the DLL into each plugin model directory:

```
Copy-Item build\sample_plugin.dll models\preprocess_plugin\
Copy-Item build\sample_plugin.dll models\postprocess_plugin\
```

The build copies the required runtime libraries next to `build\inferlite.exe`.

## Run

```
build\inferlite.exe --model-repository=models --http-port=8000 \
    --max-queue-size=100 --http-threads=4
```

Validated mode (requires `models\manifest.json`; enables self-test readiness):

```
build\inferlite.exe --model-repository=models --http-port=8000 --validated-mode \
    --audit-log=build\audit.log --diagnostic-log=build\diag.log
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--model-repository=<path>` | (required) | Model repository root |
| `--host=<addr>` | `0.0.0.0` | Listen address |
| `--http-port=<port>` | `8000` | HTTP port |
| `--max-queue-size=<n>` | `100` | Max queued requests (0 = unbounded) |
| `--request-timeout-ms=<n>` | `30000` | Per-request queue timeout |
| `--http-threads=<n>` | `4` | HTTP worker threads |
| `--validated-mode` | off | Require manifest; self-test gates readiness |
| `--audit-log=<path>` | off | Tamper-evident audit log file |
| `--diagnostic-log=<path>` | off | Engineer-facing diagnostic log file |
| `--max-input-size-bytes=<n>` | `52428800` | Input size limit |
| `--max-output-size-bytes=<n>` | `52428800` | Output size limit |
| `--max-inference-time-ms=<n>` | `5000` | Per-request inference time limit |
| `--tls-cert=<path>` / `--tls-key=<path>` | – | TLS cert/key (validated deployments front the server with a TLS 1.2+ reverse proxy) |
| `--software-version=<s>` | `InferLite 2.0.0` | Reported software version |

## Interface

### Readiness
```
GET /v2/health/ready      -> 200 {"status":"READY"}  (only if self-tests passed)
GET /v2/health/detailed   -> per-model status + hashes + versions
GET /v2/versions          -> software + OpenVINO + model versions
```

### Model config
```
GET /v2/models/<model_name>/config   -> parsed config + config/model hashes
```

### Inference
```
POST /v2/models/<model_name>/infer
Content-Type: application/json
{
  "inputs": [
    {"name": "input", "shape": [1, 4], "datatype": "FP32",
     "data": [1.0, 2.0, 3.0, 4.0]}
  ]
}
```
`data` may be a JSON number array (serialized per `datatype`) or a base64
string. The response contains `outputs` (base64 `data`) and a `trace_id`.

### Metrics
```
GET /v2/metrics    -> requests counts, average latency, queue depth, config hash
```

## Testing

- `tools/make_sample_model.py` generates the sample OpenVINO model (`y = 2x + 1`).
- `tools/make_device_models.py` generates the Phase 4 CPU/NPU/GPU/AUTO device
  sample models (exporting precompiled blobs when the corresponding Intel
  hardware is present).
- `tools/make_manifest.py` generates `models\manifest.json` with SHA-256 hashes.
- `test_server_phase2.ps1` starts the server in validated mode and exercises
  integrity, validation, ensemble, plugin, audit log, and metrics.
- `test_server_phase4.ps1` starts the server and exercises the Phase 4
  multi-device models (CPU, NPU, AUTO), verifying device reporting, config
  `kind`, inference, and metrics.
- `load_test.ps1 -Concurrency <n> -PerWorker <m>` runs a sustained concurrent
  load test.

The demo ensemble (`ensemble_pipeline`) chains:
preprocess (`×0.5`) → sample_model (`2x+1`) → postprocess (`+0.5`).
For input `[1,2,3,4]` it returns `[2.5,3.5,4.5,5.5]`.

## Acknowledgements
- deepseek-v4-flash
