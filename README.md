# InferLite

A lightweight, deterministic, single-node inference server for a fixed
industrial workstation. It adopts the proven, reference-aligned architecture of
a production serving framework — model repository, backend abstraction, bounded
scheduling, and reusable memory — while dropping all cloud-scale, multi-tenant,
and dynamic operations that add unnecessary complexity on the factory floor.

This repository currently implements **Phase 1** (see `docs/PRD-phase-01.md`);
`docs/PRD-all.md` describes the full product direction.

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

## Phase 1 — What Is Implemented Today

The foundation of the above vision: a single-node server that scans a model
repository, loads models, schedules requests, and serves synchronous inference.

- **Model repository** — parses model configuration files, picks the highest
  numeric version directory for each model.
- **Scheduler** — bounded FIFO request queue with configurable depth and request
  timeout; at most `instance_group.count` requests run concurrently per model.
- **Memory management** — a pool of reusable host buffers recycled after each
  response; no per-request allocations, no device memory.
- **Backend** — a CPU backend wrapping the framework's compiled-model object.
- **Interface** — inference, readiness health, model config, and JSON metrics.
- **Fail-fast startup** — unsupported configurations (batching, GPU instance
  groups) abort the server immediately.

### Deferred to later phases (not in Phase 1)

GPU backends, ensemble/DAG scheduling, plugin backends, batching, live model
updates, and cross-backend zero-copy.

## Layout

```
models/
  <model_name>/
    config.pbtxt
    <version>/            # highest numeric version is used
      model.xml
      model.bin
src/
  main.cpp                 # CLI + fail-fast startup
  infer_lite.*             # app wiring + routing
  http_server.*            # HTTP/1.1 server (thread pool)
  json.*                   # minimal JSON parser/serializer
  pbtxt.*                  # model configuration parser
  model_repository.*       # repository scan + validation
  openvino_backend.*       # current CPU backend implementation
  scheduler.*              # bounded FIFO scheduler
  memory_manager.*         # host memory pool
  tensor.hpp               # tensor/data-type definitions
tools/
  make_sample_model.py     # generates models/sample_model (IR + config.pbtxt)
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

The build copies the required runtime libraries (including third-party
dependencies such as TBB) next to `build\inferlite.exe`.

## Run

```
build\inferlite.exe --model-repository=models --http-port=8000 \
    --max-queue-size=100 --http-threads=4
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--model-repository=<path>` | (required) | Model repository root |
| `--host=<addr>` | `0.0.0.0` | Listen address |
| `--http-port=<port>` | `8000` | HTTP port |
| `--max-queue-size=<n>` | `100` | Max queued requests (0 = unbounded) |
| `--request-timeout-ms=<n>` | `30000` | Per-request timeout |
| `--http-threads=<n>` | `4` | HTTP worker threads |

## Interface

### Readiness

```
GET /v2/health/ready
```
`200 {}` when ready, `503 {}` otherwise.

### Model config

```
GET /v2/models/<model_name>/config
```
Returns the parsed model configuration as JSON.

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
string. The response contains `outputs` with base64 `data`.

### Metrics

```
GET /v2/metrics
```
Returns JSON with `requests_completed`, `requests_failed`,
`requests_timed_out`, `average_inference_latency_us`, and `queue_depth`.

## Testing

- `tools/make_sample_model.py` generates a sample model computing `y = 2x + 1`.
- `test_server.ps1` starts the server and exercises every endpoint.
- `load_test.ps1 -Concurrency <n> -PerWorker <m>` runs a sustained concurrent
  load test and prints success/failure counts.

For input `[1,2,3,4]`, the sample model returns `[3,5,7,9]` (base64 in the
`outputs[].data` field).


## Acknowledgements
- deepseek-v4-flash