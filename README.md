# InferLite

A lightweight, deterministic, single-node inference server for a fixed
industrial workstation. It adopts the proven, reference-aligned architecture of
a production serving framework — model repository, backend abstraction, bounded
scheduling, and reusable memory — while dropping all cloud-scale, multi-tenant,
and dynamic operations that add unnecessary complexity on the factory floor.

This repository implements **Phase 1** (see `docs/PRD-phase-01.md`),
**Phase 2** (see `docs/PRD-phase-02.md`), and **Phase 3** (see
`docs/PRD-phase-03.md`); `docs/PRD-all.md` describes the full product direction.

Phase 3 adds an opt-in **TensorRT GPU backend** on top of the validated CPU
runtime. GPU support is compiled only when a TensorRT SDK is available at CMake
configure time (`-DTENSORRT_ROOT=...`); without it the server builds and runs
CPU-only with no regression. OpenVINO models, plugins, and ensembles remain
CPU-only.

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

InferLite is designed as a **software component of a regulated medical device**.
It is architected to meet the FDA's software lifecycle and cybersecurity
expectations, and can serve as the inference runtime layer for a medical AI
product (e.g., a chest X-ray collimation or clinical image-analysis workflow).

> **Important:** InferLite is a *software-only runtime*. Achieving FDA clearance
> (510(k) / De Novo / PMA) requires the full device context — including clinical
> validation of the AI function, the hosting system, usability, and the
> manufacturer's Quality System (QMS). This section documents the compliance
> posture of the inference runtime itself, which is the prerequisite foundation.

### Regulatory Framework

The runtime is built against the following frameworks and guidance:

| Framework | Applicability |
|-----------|---------------|
| FDA 21 CFR Part 820 (QMSR / ISO 13485) | Design controls, traceability, records |
| IEC 62304:2006 + AMD1:2015 | Software life-cycle (safety class C posture) |
| ISO 14971:2019 | Risk management (hazard & risk controls) |
| FDA Software Guidance (2023) | Premarket software documentation |
| FDA Cybersecurity Guidance (2026) | Secure by design, trust boundaries |

### Medical Equipment Level (Software Classification)

InferLite is developed and intended for use as **Class C software** in the
IEC 62304 sense — software whose failure could contribute to patient harm
through a wrong inference output, a hung request, or resource exhaustion. To
earn that classification the runtime implements the following **safety
mechanisms** (traceable to risk controls):

- **Deterministic execution** — static model set, no runtime model loading, no
  dynamic batching; identical input + configuration yields identical output.
- **Model integrity** — approved-model manifest with SHA-256 verification; a
  tampered model file aborts startup (fail-fast), preventing a wrong-model
  scenario.
- **Input/output validation** — strict tensor shape/type/size checks and output
  NaN/Inf/range detection prevent malformed or unsafe data from propagating.
- **Fault isolation** — every backend/plugin call is exception-contained and
  returns a structured `ErrorCode`; no silent failures, no corrupted outputs.
- **Resource limits** — input/output size caps and per-request inference time
  limits bound worst-case resource consumption (ISO 14971 risk control).
- **Safe failure** — self-test failure or instance unavailability makes the
  server report `NOT_READY` (503) rather than serving degraded results.

### FDA Compliance Features (Implemented)

Each Phase 2 feature maps to a regulatory requirement:

| Compliance Objective | Implementation | Reference |
|----------------------|----------------|-----------|
| Model integrity & traceability | `manifest.json` SHA-256 hashes, `metadata.json`, config hashing | IEC 62304 §5.2, 5.8 |
| Deterministic execution | Hard resource limits, per-request timeouts, validated-config mode | ISO 14971; IEC 62304 §5.3.4 |
| Input/output validation | Shape/dtype/size checks; output NaN/Inf/range checks | ISO 14971; IEC 62304 §5.3.4 |
| Fault isolation & safe failure | Exception containment, structured error codes, no silent failures | IEC 62304 §5.3.4 |
| Audit trail | Tamper-evident, hash-chained inference audit log | 21 CFR 820.30 |
| Health & self-test | Startup golden-input self-test; detailed health endpoint | IEC 62304 §5.6 |
| Secure communication | Validated mode + TLS 1.2+ termination (reverse proxy) | FDA Cybersecurity (2026) |
| Config & version management | Hashed config/manifest, locked runtime, version reporting | IEC 62304 §5.8 |
| Software identification | Server + OpenVINO + model version reporting | FDA Software Guidance §4 |
| CPU ensembles & plugins | DAG executor + C++ plugin backend with full isolation | IEC 62304 §5.3 |

### Validated (Locked) Mode

Run with `--validated-mode` to enforce the FDA baseline:

- **Manifest required** — the server refuses to start unless `manifest.json` is
  present and every model/plugin hash matches.
- **Self-test gated readiness** — `/v2/health/ready` returns `200` only after
  all models pass their golden-input self-tests.
- **Audit trail** — every inference is recorded in a tamper-evident, hash-chained
  log (enable with `--audit-log=<path>`).
- **No admin APIs** — only inference and health endpoints are exposed.
- **TLS** — validated deployments must front the server with a TLS 1.2+ reverse
  proxy (see the TLS note below).

### Required Documentation Package

To use InferLite in a regulatory submission, the following artifacts (defined in
`docs/PRD-phase-02.md` §7) must be produced and reviewed:

- Software Development Plan (SDP)
- Software Requirements Specification (SRS)
- Architecture & Detailed Design
- Risk Management File (ISO 14971)
- Requirements Traceability Matrix (Requirement → Design → Risk → Test)
- Verification & Validation Plan
- OTS Software Validation Report (OpenVINO, HTTP lib, OS)
- Cybersecurity Documentation (threat model, TLS, manifest verification)

### Security & Privacy Notes

- **No patient data is stored.** The audit log records tensor *shapes*, hashes,
  and timing — never the tensor payloads.
- **Minimal attack surface.** Only inference and health endpoints are exposed;
  there is no admin or management API.
- **Minimal privileges.** Deploy the server under a restricted OS account and
  place it behind a network boundary.

## What Is Implemented

### Phase 1 — Foundation

- **Model repository** — parses model configuration files, picks the highest
  numeric version directory for each model.
- **Scheduler** — bounded FIFO request queue with configurable depth and request
  timeout; at most `instance_group.count` requests run concurrently per model.
- **Memory management** — a pool of reusable host buffers recycled after each
  response; no per-request allocations, no device memory.
- **Backend** — a CPU backend wrapping the framework's compiled-model object.
- **Interface** — inference, readiness health, model config, and JSON metrics.
- **Fail-fast startup** — unsupported configurations abort the server.

### Phase 2 — FDA-Compliant CPU Runtime

Adds P0 FDA-critical controls on the CPU-only foundation (all GPU features
deferred to Phase 3):

- **Model integrity & traceability** — `manifest.json` with SHA-256 hashes;
  verified at startup; mismatches cause fail-fast refusal to start. `metadata.json`
  carries `model_id`, `version`, `intended_use`, `approval_status`.
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
- **Metrics** — expanded with queue depth, per-model latency, and a configuration
  hash.

### Phase 3 — TensorRT GPU Acceleration (Opt-in)

Adds a validated, deterministic TensorRT GPU backend under the same FDA safety
boundary as the CPU runtime:

- **TensorRT backend** — deserializes approved `model.plan` engine files; each
  instance owns a CUDA stream; `execute()` enqueues on the stream and
  synchronizes via a CUDA event.
- **GPU memory manager** — a reusable CUDA device buffer pool plus a pinned host
  pool for efficient host↔device transfers.
- **Instance groups** — `kind: KIND_GPU` with a `count` for TensorRT models;
  multiple GPU instances run concurrently on separate streams and alongside CPU
  models.
- **Unified scheduler** — manages both CPU and GPU instances with busy/free
  tracking and quarantine of a CUDA-faulted instance (fault isolation).
- **GPU-aware ensembles** — DAGs may mix CPU and TensorRT nodes; cross-device
  edges use explicit pinned copies, and a step failure cancels the ensemble.
- **FDA controls extended to GPU** — `.plan` files are SHA-256 hashed and listed
  in the manifest, outputs are validated after the device→host copy, the audit
  log records `device: "GPU"`, and `MAX_GPU_MEMORY_MB` /
  `MAX_INFERENCE_TIME_MS` bound GPU execution.

### Deferred to later phases (not in Phase 1/2/3)

OpenVINO GPU plugin, dynamic/static batching, live model updates, a profiling
tool, multi-GPU, and gRPC/streaming.

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
  openvino_backend.*       # OpenVINO CPU backend
  plugin_backend.*         # C++ plugin backend (shared-library ABI)
  plugin_api.hpp           # plugin ABI (inferlite_plugin_*)
  ensemble_executor.*      # CPU ensemble DAG executor (zero-copy host memory)
  scheduler.*              # bounded FIFO scheduler (with inference time limit)
  memory_manager.*         # host memory pool
  audit_log.*              # tamper-evident hash-chained audit log
  config_store.*           # manifest/metadata/self-test/hash management
  validation.*             # input/output validation + structured error codes
  sha256.*                 # SHA-256 hashing
  tensor.hpp               # tensor/data-type definitions
  diagnostics.*            # engineer-facing diagnostic log
tools/
  make_sample_model.py     # generates models/sample_model (IR + config + metadata)
  make_manifest.py         # generates models/manifest.json with SHA-256 hashes
  sample_plugin/           # example CPU plugin source (sample_plugin.dll)
```

## Build

Prerequisites:

- Microsoft Visual Studio 2022 (MSVC C++ x64 toolset)
- CMake + Ninja (shipped with VS)
- OpenVINO 2025.3 Windows C++ runtime (extracted under `c:\tools\openvino`, or
  point CMake at it with `-DOPENVINO_ROOT=...`)
- (Optional, for the GPU backend) NVIDIA TensorRT SDK + CUDA; set `TENSORRT_ROOT`
  to the TensorRT install directory at CMake configure time.

```
powershell -ExecutionPolicy Bypass -File build.ps1
```

CPU-only by default. To enable the GPU backend:

```
cmake -S . -B build-gpu -DOPENVINO_ROOT=c:\tools\openvino -DTENSORRT_ROOT=C:\TensorRT
cmake --build build-gpu --config Release
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
| `--max-gpu-memory-mb=<n>` | `2048` | Per-model GPU memory cap (TensorRT) |
| `--max-concurrent-gpu-instances=<n>` | `4` | Max concurrent GPU instances |
| `--gpu-device=<n>` | `0` | CUDA device index (single GPU only) |

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
- `tools/make_manifest.py` generates `models\manifest.json` with SHA-256 hashes.
- `test_server_phase2.ps1` starts the server in validated mode and exercises
  integrity, validation, ensemble, plugin, audit log, and metrics.
- `load_test.ps1 -Concurrency <n> -PerWorker <m>` runs a sustained concurrent
  load test.

The demo ensemble (`ensemble_pipeline`) chains:
preprocess (`×0.5`) → sample_model (`2x+1`) → postprocess (`+0.5`).
For input `[1,2,3,4]` it returns `[2.5,3.5,4.5,5.5]`.

## Acknowledgements
- deepseek-v4-flash
