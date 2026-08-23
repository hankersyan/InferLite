# PRD Lite: Phase 1 - OpenVINO CPU-Only HTTP Inference Server

## 1. Overview
**Objective:** Build the foundational, single-node inference server restricted to CPU execution. This phase establishes the core architecture (model repository, scheduler, memory manager) and enables synchronous inference via the OpenVINO CPU backend over an HTTP/REST API, without any batching logic.

## 2. Scope
**In Scope:**
- HTTP/REST API (inference, health, config, metrics).
- OpenVINO backend integration (strictly CPU).
- Model repository parsing (`config.pbtxt` format).
- Bounded FIFO request scheduler with CPU instance groups.
- Basic host memory management for single requests.

**Out of Scope (Deferred to Later Phases):**
- GPU support (OpenVINO GPU plugin, TensorRT, CUDA streams).
- Ensemble scheduling, DAG execution, and C++ plugins.
- Dynamic batching and static batching (all requests are processed strictly 1:1).
- Live model updates and cross-backend zero-copy.

---

## 3. System Architecture (Phase 1)

```text
┌─────────────────────────────────────────────────────────┐
│                    HTTP Endpoint                        │
│               (REST, /v2/models/.../infer)              │
├─────────────────────────────────────────────────────────┤
│                   Scheduler                             │
│         (FIFO request queue → CPU instance dispatch)    │
├─────────────────────────────────────────────────────────┤
│                 OpenVINO Backend (CPU)                  │
│      (Wraps ov::CompiledModel for CPU instances)        │
├─────────────────────────────────────────────────────────┤
│                Memory Manager                           │
│      (Host memory pool, no device allocations)          │
├─────────────────────────────────────────────────────────┤
│              Model Repository                           │
│   (directory structure + config.pbtxt parser)           │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Core Components

### 4.1 Model Repository & Configuration
- **Layout:** Follows the standard directory structure.
  ```text
  models/
    openvino_model/
      1/
        model.xml
        model.bin
      config.pbtxt
  ```
- **Version Policy:** Server reads the highest numeric directory name at startup and ignores others.
- **Configuration (`config.pbtxt`):** Supports a minimal subset of fields:
  - `name`, `backend` (must be `openvino`)
  - `max_batch_size` (Triton‑style: `0` disables batching; `>0` enables a
    leading batch dimension on request tensors, with config `dims` per‑request).
  - `input` / `output` tensor definitions (name, data_type, dims)
  - `instance_group` (`count`, `kind`: strictly `KIND_CPU`).

### 4.2 Inference Server Core
- **Model Manager:** Scans repository at startup, parses `config.pbtxt`, loads OpenVINO models, and creates CPU instances based on the `instance_group` count. No loading/unloading occurs after startup.
- **Scheduler:** Maintains a bounded FIFO request queue. Dispatches single requests to available CPU instances. If all instances are busy, the request waits. Configurable max queue size and request timeout apply.

### 4.3 OpenVINO Backend
- Wraps `ov::CompiledModel` using the OpenVINO CPU plugin.
- Executes inference via a dedicated CPU thread pool.
- **Backend API:** Implements `load(config, path)`, `execute(inputs, outputs)`, and `unload()`.
- Assumes 1:1 execution (one input tensor yields one output tensor per request).

### 4.4 Memory Manager
- Maintains a pool of reusable host memory buffers sized for the expected tensor dimensions.
- Each request acquires tensors from the pool; buffers are returned to the pool after the HTTP response is sent.
- No GPU/device memory or pinned memory is required in this phase.

### 4.5 HTTP API
A lightweight HTTP server exposes a minimal, synchronous subset of the standard inference REST API:
- `POST /v2/models/<model_name>/infer`: Accepts JSON body (model name, inputs with name, shape, datatype, data). Returns JSON with outputs. (Binary data via base64).
- `GET /v2/health/ready`: Returns `200 OK` when models are loaded and server is ready.
- `GET /v2/models/<model_name>/config`: Returns the parsed `config.pbtxt` as JSON.
- `GET /v2/metrics`: Returns JSON object with request counts, average latency, queue depth.

---

## 5. Request Lifecycle (Phase 1)

1. Client sends `POST /v2/models/openvino_model/infer` with a single input payload.
2. HTTP handler deserializes the JSON request and enqueues it in the bounded request queue.
3. Scheduler picks the request when a CPU OpenVINO instance is available.
4. Memory manager allocates required input/output host buffers.
5. Input data is copied directly into the allocated host buffer.
6. OpenVINO CPU backend executes inference on a worker thread.
7. Output tensors are serialized into a JSON HTTP response and returned to the client.
8. Host buffers are released back to the pool.

---

## 6. Configuration & Startup

- **Command Line:** 
  `--model-repository=/path/to/models --http-port=8000 --max-queue-size=100`
- **Startup Sequence:** Scan repo → Validate configs (reject `max_batch_size < 0` or invalid `KIND_*`) → Load OpenVINO CPU backends → Create instances → Start HTTP listener.
- **Fail-fast:** Any configuration error or OpenVINO backend failure aborts the server immediately.

---

## 7. Success Criteria for Phase 1

- **Functionality:** Server successfully loads OpenVINO models and serves single-request inference via HTTP REST API.
- **Performance:** HTTP + scheduling overhead is ≤ 5% compared to direct C++ OpenVINO CPU inference.
- **Concurrency:** Server correctly handles concurrent requests up to the configured `instance_group` count without crashing.
- **Stability:** Runs sustained load without memory leaks or crashes.