## Composite Product Requirements Document  
### Lightweight Multi‑Backend Inference Server for Industrial Edge

---

### 1. Overview

**Project Name:** Edge Inference Server  
**Objective:** Build a lightweight, deterministic, single‑node inference server for a single‑GPU, single‑CPU industrial workstation. The architecture closely mirrors a well‑known production inference serving framework—its model repository layout, backend abstraction, ensemble DAG scheduling, and zero‑copy memory management—while removing all cloud‑scale, multi‑tenant, and dynamic operations that are unnecessary in a fixed industrial setting.

---

### 2. Product Goals

- **Reference‑aligned architecture** – model repository structure, `config.pbtxt`‑style model configuration, backend interface, ensemble graphs, instance groups, and zero‑copy tensor sharing.
- **Multi‑backend support** – first‑party backends for TensorRT (LLMs) and OpenVINO (CPU/GPU).
- **Native C++ plugin system** – custom pre‑ and post‑processing nodes that run inside the ensemble graph as if they were backends.
- **Concurrent execution** – multiple model instances (CPU and GPU) and plugin nodes run in parallel using CUDA streams and a CPU thread pool.
- **Zero‑copy DAG** – ensemble pipelines pass tensor data by reference (device or host pointer) without copying between adjacent nodes on the same device.
- **Deterministic, low‑latency serving** – fixed static graph, no dynamic batching, no hot model reloading, minimal scheduling overhead.
- **Standard interface** – HTTP/REST API that is compatible with the inference protocol of the reference server (simplified subset), plus health and basic JSON metrics.

---

### 3. Out of Scope (Non‑Goals)

- Multi‑node, multi‑GPU scaling, Kubernetes, MIG.
- Dynamic batching, priority queues, rate limiting (a simple bounded request queue is acceptable).
- Live model updates / hot‑swap – model changes require a full server restart.
- Full streaming, sequence batching, stateful models, or gRPC streaming.
- Python backends or business logic scripting – all custom logic must be C++.
- Prometheus endpoint – JSON‑format metrics are sufficient.
- Multi‑model concurrency across different GPU contexts (single CUDA context used).

---

### 4. System Architecture

The server adopts the layered design of the reference production framework, adapted for a single‑node, single‑process deployment.

```
┌─────────────────────────────────────────────────────────┐
│                    HTTP Endpoint                        │
│               (REST, /v2/models/.../infer)              │
├─────────────────────────────────────────────────────────┤
│                   Scheduler                             │
│         (request queue → model instance dispatch)       │
├─────────────────────────────────────────────────────────┤
│                 Ensemble Scheduler                      │
│   (DAG executor, zero‑copy tensor passthrough)          │
├──────────────┬──────────────┬───────────────────────────┤
│  Backend 1   │  Backend 2   │   Plugin Backend          │
│  (TensorRT)  │ (OpenVINO)   │  (C++ custom nodes)       │
├──────────────┴──────────────┴───────────────────────────┤
│                Memory Manager                           │
│      (CUDA device pointers, pinned host, arena)         │
├─────────────────────────────────────────────────────────┤
│              Model Repository                           │
│   (directory structure + config.pbtxt parser)           │
└─────────────────────────────────────────────────────────┘
```

**Key simplifications from the reference server:**  
- Single process, no separate control plane.  
- Ensemble graph is static, defined at startup.  
- Backends are compiled into the server, not loaded dynamically.  
- No model version directory traversal for live updates – only the highest version number is loaded at start.

---

### 5. Core Components

#### 5.1 Model Repository

- **Layout:** Identical to the reference server’s model repository:
  ```
  models/
    encoder/
      1/
        model.plan (TensorRT) / model.xml, model.bin (OpenVINO)
      config.pbtxt
    decoder_llm/
      1/
        model.plan
      config.pbtxt
    preprocessing/
      1/
        config.pbtxt (plugin backend)
    ensemble_pipeline/
      1/
        config.pbtxt (ensemble definition)
  ```
- **Version policy:** Only the latest version (e.g., `1`) is loaded at startup. The server reads the highest numeric directory name and ignores all others.

#### 5.2 Model Configuration (`config.pbtxt`)

The server uses the same protobuf‑like text format as the reference server for each model, plugin, or ensemble. Supported fields:

- `name`, `backend` (valid values: `tensorrt`, `openvino`, `plugin`, `ensemble`)
- `max_batch_size` — Triton‑style batching: `0` disables batching (no batch
  dimension); `>0` enables batching where request tensors carry a leading batch
  dimension `B` (`1 <= B <= max_batch_size`) and `dims` are per‑request (without
  the batch dimension). With `max_batch_size: 1` the batch dimension is always
  `1`, so a model whose IR accepts `[1, 4]` declares `dims: [4]` and clients send
  shape `[1, 4]`.
- `input` / `output` tensor definitions (name, data type, dims)
- `instance_group` – `count`, `kind` (KIND_GPU, KIND_CPU), and optionally `gpus` (only device 0)
- `ensemble_scheduling` (for ensemble models): a `step` list with `model_name`, `input_map`, `output_map`
- `parameters` (for plugin models): a Triton-style `parameters { key: "..." value { string_value: "..." } }` block, repeated per key. The key/value pairs are passed to the plugin's `inferlite_plugin_create`, so each pipeline owns its pre/post-processing behavior (e.g. a per-pipeline `scale`, `clamp_min`/`clamp_max`, `offset`) without code changes. See `tools/sample_plugin/sample_plugin.cpp` for the keys understood by the sample plugin.

**Example: plugin backend model**
```
name: "preprocessing"
backend: "plugin"
max_batch_size: 0
input [{ name: "raw_image", data_type: TYPE_UINT8, dims: [-1, -1, 3] }]
output [{ name: "normalized_tensor", data_type: TYPE_FP32, dims: [3, 224, 224] }]
instance_group [{ kind: KIND_CPU, count: 2 }]
plugin_library: "libpreprocess_plugin.so"
parameters {
  key: "scale"
  value { string_value: "0.5" }
}
```

Multiple pipelines can each reference their own plugin models (e.g. `preprocess_pipeline_a` / `preprocess_pipeline_b`) inside their `ensemble_scheduling`; each plugin model carries its own `parameters`, so each pipeline's preprocessing and postprocessing are owned and configured independently.

#### 5.3 Inference Server Core

- **Model Manager:** Scans the model repository at startup, parses all `config.pbtxt` files, instantiates backends and model instances, and holds them ready. No model loading/unloading occurs after startup.
- **Scheduler:** Maintains a bounded FIFO request queue. Each incoming request is matched to a model (or ensemble) and dispatched to an available instance. If all instances are busy, the request waits. A configurable maximum queue size and request timeout prevent resource exhaustion.
- **Concurrency:**
  - *GPU instances:* Each model instance may use its own CUDA stream. Multiple instances of different models run concurrently on the GPU, with no stream blocking unless data dependencies require it.
  - *CPU instances:* A dedicated thread pool executes OpenVINO CPU models and plugin nodes.
  - The scheduler respects `instance_group` count and device placement.

#### 5.4 Backend Interface

A minimal backend API closely modeled after the reference server’s backend API, but reduced to essential functions:

- `BackendModel::load(config, model_path)` – loads model into device memory.
- `BackendModelInstance::execute(inputs, outputs, stream)` – runs inference on a specified CUDA stream (GPU backends) or a CPU thread (CPU backends).
- `BackendModel::unload()` – releases resources.

**Implemented backends:**
- **TensorRT Backend:** Wraps `nvinfer1::ICudaEngine` and `IExecutionContext`. Accepts device pointers for inputs/outputs, enqueues kernels on the provided stream.
- **OpenVINO Backend:** Wraps `ov::CompiledModel`. When `kind: KIND_GPU` is requested, it uses the OpenVINO GPU plugin (which operates via OpenCL). To enable zero‑copy with TensorRT, a device‑to‑device copy between CUDA and OpenCL contexts may be required, so the graph is designed to minimize such cross‑backend GPU transitions. For CPU instances, host memory is used; pinned memory ensures fast host‑to‑device transfers when needed.
- **Plugin Backend:** Loads a shared library (`.so`) that implements a `PluginNode` C++ interface. Plugins have access to raw tensor pointers and can optionally launch CUDA kernels on the given stream.

#### 5.5 Ensemble Scheduler (DAG Executor)

Ensembles are defined with `backend: "ensemble"` in `config.pbtxt`. The server internally builds a static directed acyclic graph from the steps.

- **Zero‑copy execution:** A per‑request tensor arena allocates buffers. When an edge connects two nodes on the same device, the downstream node receives the exact same memory pointer—no copy occurs. For GPU nodes, tensors remain as CUDA device pointers; for CPU, they stay in host memory.
- **Concurrent execution of independent nodes:** Nodes with no dependencies fire simultaneously. GPU nodes are launched on separate CUDA streams; a consumer node that needs a GPU output inserts a CUDA event wait to synchronize with the producer stream. CPU nodes are dispatched to the thread pool.
- **Device‑aware scheduling:** The scheduler tracks tensor location (host or device). When a tensor produced on one device must be consumed on another, a single copy node is automatically inserted. This makes cross‑device transitions explicit while keeping the overall copy count minimal.
- **Static graph only:** No dynamic branching or conditional execution (BLS). The entire pipeline is predetermined at configuration time.

#### 5.6 Memory Manager

- Maintains a pool of reusable buffers (GPU device memory and pinned host memory) sized according to the largest expected tensor dimensions.
- Each inference request acquires tensors from the pool; after the response is sent, they are returned.
- Zero‑copy is achieved by passing raw pointers between backends. The arena avoids per‑request allocations and fragmentation.

#### 5.7 HTTP API

A lightweight HTTP server (e.g., using `libmicrohttpd` or `cpp-httplib`) exposes a subset of the industry‑standard inference REST API:

- `POST /v2/models/<model_name>/infer`  
  Request: JSON body matching the known inference request schema (model name, inputs with name, shape, datatype, data).  
  Response: JSON with outputs.  
  For efficiency, binary tensor data can be sent as base64‑encoded strings, or (optionally) as a custom multipart binary format.
- `GET /v2/health/ready` → `200 OK` when the server is ready to serve.
- `GET /v2/models/<model_name>/config` → returns the model’s parsed configuration.
- `GET /v2/metrics` → returns a JSON object containing request counts, average latency, queue depth, GPU memory usage, etc. (Prometheus format is not required.)

This API intentionally mirrors the reference server’s endpoint structure to simplify client integration, but only a minimal, synchronous subset is implemented.

#### 5.8 Metrics & Observability

- Per‑model: request count, success/error count, average inference time, queue wait time.
- System: CPU usage, GPU memory, current queue depth.
- Exposed as a simple JSON endpoint; the server may also periodically log these values to a file.
- A health endpoint enables integration with external watchdog services.

---

### 6. Request Lifecycle (Ensemble Example)

1. Client sends `POST /v2/models/ensemble_pipeline/infer` with raw sensor data.
2. HTTP handler deserializes the request and enqueues it in the bounded request queue.
3. Scheduler picks the request when an ensemble instance is available.
4. Ensemble executor allocates a tensor arena and maps input tensors to the first step(s).
5. **Step 1 – preprocessing plugin (CPU):** Dispatched to thread pool. Output tensor (normalized float) stored in pinned host memory.
6. **Step 2 – TensorRT encoder (GPU):** Input is copied once from host to GPU. TensorRT instance launched on its CUDA stream, output remains as device pointer.
7. **Step 3 – OpenVINO classifier (GPU, same device):** Receives the same device pointer, no copy. Launched on a separate CUDA stream; synchronization with the encoder is handled by a CUDA event wait when the classifier’s input is ready.
8. **Step 4 – postprocessing plugin (CPU):** The executor sees that step 3’s output is on GPU and step 4 is CPU, so it inserts a device‑to‑host copy. After copy, the CPU plugin runs in the thread pool, producing final result tensors in host memory.
9. Final tensors are serialized into the HTTP response and returned. All arena buffers are released.

---

### 7. Configuration and Startup

- Server binary command line:  
  `--model-repository=/path/to/models --http-port=8000 --max-queue-size=100 --gpu-device=0`
- On startup, the server scans the model repository, validates configurations, loads backends, creates model instances, builds ensemble DAGs, and starts the HTTP listener.
- Any configuration error or backend failure aborts the server immediately (fail‑fast).

---

### 8. Design Decisions & Constraints

| Feature | Implementation | Rationale |
|---------|---------------|-----------|
| Model Repository & `config.pbtxt` | Fully adopted | Ensures compatibility with existing tooling and operational familiarity |
| Backend API | Adapted (built‑in backends) | Removes dynamic loading complexity; only TensorRT, OpenVINO, and plugin backends needed |
| Instance Groups | Adopted | Controls per‑model concurrency via `count` and `kind` |
| Ensemble Scheduling (DAG) | Adopted | Zero‑copy, concurrent execution of pipeline steps; static graph only |
| Triton Batch Dimension (`max_batch_size`) | Adopted | `max_batch_size` shapes follow Triton: config `dims` are per‑request; clients prepend the batch dim. Request combining across the queue is deferred; `max_batch_size` bounds the accepted batch dim |
| Live Model Update | Omitted | Server restart only, guaranteeing deterministic behaviour |
| Shared Memory / IPC | Omitted | Single process can share pointers directly |
| Priority / Rate Limiting | Omitted | Bounded queue provides sufficient back‑pressure control |
| gRPC / Streaming | Omitted | HTTP/REST is simpler and adequate for the use case |
| Model Analyzer / Autotune | Omitted | Manual tuning via `instance_group` and stream counts |
| Prometheus Metrics | Omitted | JSON endpoint is sufficient for industrial diagnostics |

---

### 9. Success Criteria

- **Performance:** End‑to‑end latency overhead (HTTP + scheduling) ≤ 5% compared to direct C++ inference.
- **Correctness:** Ensemble zero‑copy execution produces identical numerical outputs to a sequential pipeline with explicit copies.
- **Stability:** Server runs 24/7 without memory leaks or GPU memory fragmentation under sustained load.
- **Deployability:** Single binary, no external dependencies, drop‑in model repository compatible with the reference server’s format.

---

This PRD defines a minimal, robust inference server that inherits the proven architecture of the industry’s leading serving framework while staying strictly within the boundaries of a single‑GPU, single‑CPU industrial workstation. It delivers the required features – multi‑backend support, zero‑copy DAG ensembles, concurrent execution, and C++ plugins – without any of the complexity or overhead that would compromise reliability or determinism on the factory floor.