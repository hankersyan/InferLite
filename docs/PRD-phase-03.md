# PRD: Phase 3 – TensorRT GPU Acceleration (FDA‑Controlled)

**Builds on:** Phase 2 FDA‑Compliant CPU Inference Server  
**GPU Scope:** TensorRT backend only; OpenVINO remains CPU‑only.  
**Regulatory framework:** Same as Phase 2—IEC 62304 Class C, ISO 14971, FDA software/cybersecurity guidance. All Phase 3 changes are made under formal design controls and require updates to the risk file, V&V, and documentation.

---

## 1. Objective

Add a **validated, deterministic TensorRT GPU backend** to the existing CPU inference runtime. This allows selected models to run on GPU, while OpenVINO models, CPU plugins, and ensemble DAGs continue to operate with full traceability and safety. The GPU execution is fully integrated into the safety boundary: model integrity, resource limits, fault isolation, audit logging, and output validation are extended to TensorRT.

---

## 2. What Changes (Additions Only)

### 2.1 TensorRT Backend (New, GPU)
- **Engine loading:** Deserialize `.plan` files (TensorRT serialized engines) from the model repository.
- **Model integrity:** `.plan` files are hashed and listed in the approved manifest, same as OpenVINO models.
- **GPU memory manager:** Add a pool of reusable CUDA device memory buffers, sized for max input/output per model.
- **Instance groups:** Support `instance_group` with `kind: KIND_GPU` and a `count` for TensorRT models. Each instance gets its own CUDA stream for concurrent execution.
- **Execution:** Backend’s `execute()` receives device pointers and a CUDA stream; it enqueues work and returns immediately. The caller synchronizes via CUDA events.
- **Resource limits:** `MAX_GPU_MEMORY_MB` is enforced; allocation failure returns `RESOURCE_EXHAUSTED`.
- **Error handling:** CUDA errors (e.g., misaligned address, launch failure) are caught, the offending instance is quarantined, and a structured error code is returned.

### 2.2 Unified Scheduler (Extended)
- The scheduler now manages **both CPU instances (OpenVINO, plugins) and GPU instances (TensorRT)**.
- GPU instances are tracked by a busy/free flag with associated CUDA stream; when a request is assigned to a GPU instance, it is marked busy until the CUDA event signals completion.
- Concurrent execution: multiple TensorRT instances (if configured) can run in parallel on separate streams, and they can run concurrently with CPU models.

### 2.3 GPU‑Aware Ensemble DAG Executor
- The existing static DAG executor is extended to handle nodes that run on GPU (TensorRT) and CPU (OpenVINO/plugins).
- **Zero‑copy on same device:** Tensor outputs produced by a GPU node are passed as device pointers to the next GPU node without copy.
- **Cross‑device transitions:** When a GPU tensor is needed by a CPU node (or vice versa), the executor inserts a single **pinned‑memory copy** (host↔device). This copy is treated as an explicit step, and its overhead is part of the logged latency.
- All steps are still isolated; a TensorRT failure cancels the ensemble and frees GPU resources.

### 2.4 FDA Controls Extended to GPU

| Control | GPU Implementation |
|---------|--------------------|
| **Model integrity** | SHA‑256 of `.plan` file verified against manifest at startup. |
| **Deterministic resources** | `MAX_GPU_MEMORY_MB`, `MAX_CONCURRENT_GPU_INSTANCES`, and `MAX_INFERENCE_TIME_MS` (covers GPU execution). |
| **Input validation** | Unchanged (CPU‑side check before GPU copy). |
| **Output validation** | Applied after GPU→host copy (same range/NaN checks). |
| **Fault isolation** | CUDA errors per request; failed GPU instance quarantined; server process remains stable. |
| **Audit log** | Each inference records `device: "GPU"`, GPU memory used, and model hash. |
| **Startup self‑test** | Golden input test runs for each TensorRT model, exactly like CPU models. |
| **TLS & no admin APIs** | Unchanged. |

### 2.5 Memory Manager (Extended)
- Adds a **CUDA device memory pool** for TensorRT input/output buffers.
- Maintains a **pinned host memory pool** for efficient host‑device transfers.
- The existing host pool remains for CPU‑only models and plugins.

### 2.6 Model Repository & Configuration
- TensorRT models use `backend: "tensorrt"` in `config.pbtxt`.
- `instance_group` may specify `kind: KIND_GPU`.
- The approved manifest now includes `.plan` file hashes.
- Ensemble graphs may mix CPU and TensorRT nodes; the scheduler handles device placement automatically.

---

## 3. What Is Explicitly NOT in Phase 3

- **No OpenVINO GPU plugin.** OpenVINO models stay CPU‑only. This avoids OpenCL/CUDA interoperability complexity and keeps the regulatory surface smaller.
- **No request‑combining batching.** Inference is single‑request; a model may opt into Triton‑style batch‑dimension shapes via `max_batch_size` (config `dims` per‑request, clients prepend the batch dim).
- **No live model updates or runtime model loading.**
- **No profiling tool.**
- **No multi‑GPU support** (only a single physical GPU, device 0).
- **No gRPC or streaming.**

These remain for future phases or are permanently out of scope.

---

## 4. Architecture Diagram (Phase 3)

```
┌──────────────────────────────────────────────────────────────┐
│                  Medical Device                              │
│  ┌──────────────────────────────────────────────────────────┐│
│  │ Clinical Application (X‑ray workflow)                    ││
│  └───────────────────────┬──────────────────────────────────┘│
│                          │                                   │
│  ┌───────────────────────▼──────────────────────────────────┐│
│  │            SAFETY BOUNDARY (unchanged)                   ││
│  │  Input validation, resource limits, fault isolation      ││
│  └───────────────────────┬──────────────────────────────────┘│
│                          │                                   │
│  ┌───────────────────────▼──────────────────────────────────┐│
│  │       DETERMINISTIC AI RUNTIME (CPU + TensorRT GPU)      ││
│  │                                                          ││
│  │  ┌──────────────────────┐  ┌───────────────────────────┐ ││
│  │  │ CPU Instance Pool    │  │ GPU Instance Pool         │ ││
│  │  │ (OpenVINO, plugins)  │  │ (TensorRT, CUDA streams)  │ ││
│  │  └──────────────────────┘  └───────────────────────────┘ ││
│  │                                                          ││
│  │  Ensemble DAG Executor (zero‑copy GPU, cross‑device copy) ││
│  │  Output Validator                                        ││
│  │  Model Lifecycle Manager (manifest covers .plan files)   ││
│  │                                                          ││
│  │  Memory Manager: host pool + device pool + pinned pool   ││
│  └───────────────────────┬──────────────────────────────────┘│
│                          │                                   │
│  ┌───────────────────────▼──────────────────────────────────┐│
│  │        AUDIT & DIAGNOSTIC LAYER (extended for GPU)       ││
│  │  Audit log records device=GPU, GPU memory                ││
│  │  Self‑test covers TensorRT models                        ││
│  │  Health endpoint reports GPU status                      ││
│  └──────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
```

---

## 5. Regulatory Impact

- **Risk file update:** New hazards added (GPU memory corruption, CUDA driver crash, TensorRT engine mis‑compilation). Controls are the same pattern: hash verification, instance quarantine, timeout, and deterministic error codes.
- **OTS validation extended:** TensorRT and CUDA libraries are added to the validated OTS list. Full test suite must pass with the chosen versions.
- **Documentation update:** SRS, architecture design, traceability matrix, and V&V plan are revised to include TensorRT‑related requirements and tests.
- **No change to the fundamental regulatory strategy:** The server remains a controlled, validated runtime; GPU acceleration is simply another approved backend within the same safety framework.

---

## 6. Phase 3 Success Criteria

- **TensorRT execution:** Approved TensorRT models load, self‑test, and serve inference requests with latency overhead ≤5% vs. direct TensorRT API call.
- **Concurrent CPU+GPU:** Ensembles mixing CPU (OpenVINO) and GPU (TensorRT) steps execute correctly, with appropriate cross‑device copies and no deadlocks.
- **FDA controls hold for GPU:** Model integrity (hash), output validation, audit logging (GPU device identifier), fault isolation, and self‑test all work identically to CPU models.
- **No regression:** All Phase 2 CPU‑only functionality continues to pass tests.
- **Documentation updated:** Risk file, SRS, and V&V reports reflect the new GPU scope.

---

This Phase 3 gives you GPU acceleration exactly where it matters—TensorRT for LLMs or high‑throughput models—while keeping the regulatory burden low by deferring OpenVINO GPU and all batching/streaming features. The result is a still‑deterministic, fully traceable, medical‑grade AI runtime that can leverage a single GPU without compromising FDA compliance.