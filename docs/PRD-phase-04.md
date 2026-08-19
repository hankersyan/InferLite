# PRD: Phase 4 – Intel Multi‑Device Acceleration (CPU + NPU + GPU) via OpenVINO

**Builds on:** Phase 3 TensorRT GPU Acceleration (FDA‑Controlled)  
**New accelerator scope:** Intel NPU and Intel GPU (via OpenVINO GPU plugin), unified Intel device configuration including “auto” mode  
**Regulatory framework:** Same as Phases 2/3 — IEC 62304 Class C, ISO 14971, FDA software/cybersecurity guidance. All Phase 4 changes are made under formal design controls and require updates to the risk file, V&V, and documentation.

---

## 1. Objective

Extend the inference server to support **all three Intel execution devices** through the OpenVINO backend:

- **Intel CPU** (existing, now unified under a device field)
- **Intel NPU** (new, OpenVINO NPU plugin)
- **Intel GPU** (new, OpenVINO GPU plugin – Intel Arc / iGPU)

Additionally, introduce an **Intel automatic device selection mode** (“auto”) that lets OpenVINO choose the best available device (CPU, NPU, or GPU) at model load time. NVIDIA GPU (TensorRT) support from Phase 3 remains unchanged.

The server now manages **four accelerator domains**:

| Device | Backend / Execution | Status |
|--------|---------------------|--------|
| Intel CPU | OpenVINO CPU | Phase 1–2: released; unified in Phase 4 |
| NVIDIA GPU | TensorRT (CUDA) | Phase 3: released; unchanged |
| Intel NPU | OpenVINO NPU | Phase 4: new |
| Intel GPU | OpenVINO GPU | Phase 4: new |

All accelerators run inside the same validated safety boundary, scheduler, ensemble DAG, and audit framework.

---

## 2. Backend Roadmap Alignment

| Backend / Accelerator | Status |
|----------------------|--------|
| Intel CPU (OpenVINO) | Phase 1–2: released; config updated in Phase 4 |
| NVIDIA GPU (TensorRT) | Phase 3: released (no changes in Phase 4) |
| **Intel NPU (OpenVINO NPU)** | **Phase 4: new** |
| **Intel GPU (OpenVINO GPU)** | **Phase 4: new** |
| Intel Auto mode (CPU/NPU/GPU) | **Phase 4: new** |
| Moore Threads GPU | Hold-on – no Phase 4 work |

---

## 3. Key Configuration Changes

### 3.1 Unified Device Specification in `config.pbtxt`

To support multiple Intel devices without proliferating backend names, the existing `instance_group` is extended with a new **`device`** field. The existing `kind` field is retained for backward compatibility, but for OpenVINO models the `device` field takes precedence.

**Supported `device` values (Phase 4):**

| `device` value | Meaning | Backend |
|----------------|---------|---------|
| `"cpu"` | Intel CPU (OpenVINO CPU plugin) | `openvino` |
| `"npu"` | Intel NPU (OpenVINO NPU plugin) | `openvino` |
| `"gpui"` | Intel GPU (OpenVINO GPU plugin) | `openvino` |
| `"auto"` | Intel automatic device selection (CPU/NPU/GPU) | `openvino` |

For NVIDIA GPU models, the existing `backend: "tensorrt"` and `kind: KIND_GPU` remain unchanged. The `device` field is **not used** for TensorRT models.

**Example: Intel NPU model**
```
name: "npu_model"
backend: "openvino"
max_batch_size: 0
input [{ name: "input", data_type: TYPE_FP32, dims: [1, 3, 224, 224] }]
output [{ name: "output", data_type: TYPE_FP32, dims: [1, 1000] }]
instance_group [{ count: 2, device: "npu" }]
```

**Example: Intel GPU model**
```
name: "gpu_intel_model"
backend: "openvino"
max_batch_size: 0
input [{ name: "input", data_type: TYPE_FP32, dims: [1, 3, 224, 224] }]
output [{ name: "output", data_type: TYPE_FP32, dims: [1, 1000] }]
instance_group [{ count: 1, device: "gpui" }]
```

**Example: Intel auto mode**
```
name: "auto_model"
backend: "openvino"
max_batch_size: 0
input [{ name: "input", data_type: TYPE_FP32, dims: [1, 3, 224, 224] }]
output [{ name: "output", data_type: TYPE_FP32, dims: [1, 1000] }]
instance_group [{ count: 2, device: "auto" }]
```

- In auto mode, OpenVINO’s `AUTO` plugin selects the best available device at load time, with a preference order (configurable, default NPU > GPU > CPU).
- If the selected device is not available or the model cannot be compiled for it, startup fails (fail‑fast) unless fallback is explicitly allowed.

**Backward compatibility:** For existing OpenVINO CPU models, if `device` is absent, the server defaults to `"cpu"`. If `kind: KIND_CPU` is present without `device`, it also maps to CPU. The new `device` field is the preferred mechanism going forward.

### 3.2 Model Repository Layout

```
models/
  intel_cpu_model/
    1/
      model.xml
      model.bin
    config.pbtxt                     # device: "cpu"
  intel_npu_model/
    1/
      model.xml
      model.bin
      model.npu_blob                 # Precompiled NPU blob (required for NPU)
    config.pbtxt                     # device: "npu"
  intel_gpu_model/
    1/
      model.xml
      model.bin
      model.gpu_blob                 # Precompiled GPU blob (required for Intel GPU)
    config.pbtxt                     # device: "gpui"
  nvidia_tensorrt_model/
    1/
      model.plan                     # TensorRT engine (Phase 3, unchanged)
    config.pbtxt                     # backend: "tensorrt", kind: KIND_GPU
  ensemble_mixed/
    1/
      config.pbtxt                   # ensemble referencing CPU, NPU, GPU, and TensorRT nodes
```

- **NPU models** require a precompiled `model.npu_blob`.
- **Intel GPU models** require a precompiled `model.gpu_blob`.
- **Auto models** may include any of these blobs; OpenVINO AUTO selects the device and uses the corresponding blob.
- In validated mode, all blobs are hashed and listed in the approved manifest.

---

## 4. What Changes (Additions Only)

### 4.1 OpenVINO Backend Extended for NPU, Intel GPU, and AUTO

- **Device selection:** The OpenVINO backend now accepts a `device` parameter (`cpu`, `npu`, `gpui`, `auto`). It uses the corresponding OpenVINO plugin (`CPU`, `NPU`, `GPU`, or `AUTO`).
- **Precompiled blobs:** For `npu` and `gpui`, the backend loads precompiled blobs via `ov::Core::import_model(blob_path, device)`. For `auto`, it either imports blobs for all available devices or lets OpenVINO AUTO compile from IR; in validated mode, blobs are mandatory for deterministic behavior.
- **Instance groups:** Each `instance_group` with a given `device` creates a pool of `ov::InferRequest` objects for that device.
- **Execution:** CPU and NPU instances run in the CPU thread pool; Intel GPU instances run in the CPU thread pool but drive OpenVINO GPU inference (which internally manages OpenCL). All inference calls are synchronous from the server’s perspective.
- **Error handling:** All OpenVINO exceptions (including NPU/GPU driver errors) are caught, the offending `InferRequest` is reset or recreated, and a structured error code is returned. After repeated failures, the instance is quarantined.

### 4.2 Memory Manager Extensions

- **Host memory pool (CPU, NPU):** Existing pool reused for CPU and NPU tensors. No copy between CPU and NPU steps.
- **Pinned host memory pool:** Added for efficient transfers between host (CPU/NPU) and Intel GPU (OpenCL). This pool is page‑aligned and used for GPU input/output staging.
- **Intel GPU device memory pool:** A new OpenCL device memory pool is introduced to hold tensors that reside on Intel GPU. OpenVINO GPU models expect tensors in OpenCL buffers; the memory manager manages these buffers and provides zero‑copy pointer passing for consecutive GPU steps.
- **Tensor location tracking:** The memory manager tracks whether a tensor is in host memory, OpenCL device memory (Intel GPU), or CUDA device memory (NVIDIA GPU). This is used by the ensemble executor to decide when copies are needed.

### 4.3 Unified Scheduler Extensions

The scheduler now manages **four instance pools**:

| Instance Kind | Backend | Execution Context |
|---------------|---------|-------------------|
| `KIND_CPU` (or `device: "cpu"`) | OpenVINO CPU, plugins | CPU thread pool |
| `KIND_GPU` (NVIDIA) | TensorRT | CUDA streams |
| `KIND_NPU` (or `device: "npu"`) | OpenVINO NPU | CPU thread pool + NPU InferRequest |
| `KIND_GPU_INTEL` (or `device: "gpui"`) | OpenVINO GPU | CPU thread pool + OpenCL InferRequest |
| `KIND_AUTO` (or `device: "auto"`) | OpenVINO AUTO | Device selected by OpenVINO |

Each instance is represented by a busy/free flag and an `InferRequest` (or TensorRT context). The FIFO request queue and per‑model instance counts apply across all device types.

### 4.4 Ensemble DAG Executor Extensions

The static DAG executor now handles nodes on **CPU, NPU, Intel GPU, and NVIDIA GPU**.

- **Zero‑copy domains:**
  - CPU ↔ NPU: same host memory pointer, no copy.
  - Intel GPU ↔ Intel GPU: same OpenCL buffer pointer, no copy.
  - NVIDIA GPU ↔ NVIDIA GPU: same CUDA device pointer, no copy.
- **Cross‑device transitions (single copy only):**
  - Intel GPU ↔ CPU/NPU: pinned host memory copy (host ↔ OpenCL buffer).
  - NVIDIA GPU ↔ CPU/NPU/Intel GPU: CUDA device ↔ host copy (via pinned memory); if needed, Intel GPU ↔ NVIDIA GPU requires two copies (CUDA→host→OpenCL).
- The executor tracks tensor location with a domain enum: `HOST`, `OPENCL_DEVICE`, `CUDA_DEVICE`. Zero‑copy is applied only within the same domain.
- All steps remain isolated; any failure cancels the ensemble and frees resources.

### 4.5 FDA Controls Extended to Intel NPU and Intel GPU

| Control | Intel NPU Implementation | Intel GPU Implementation |
|---------|--------------------------|--------------------------|
| **Model integrity** | SHA‑256 of `model.npu_blob` verified against manifest at startup. | SHA‑256 of `model.gpu_blob` verified against manifest at startup. |
| **Deterministic resources** | `MAX_NPU_INFERENCE_TIME_MS`, `MAX_NPU_HOST_MEMORY_MB`, per‑model NPU instance count. | `MAX_GPU_INTEL_INFERENCE_TIME_MS`, `MAX_GPU_INTEL_MEMORY_MB`, per‑model Intel GPU instance count. |
| **Input validation** | Unchanged (CPU‑side check before execution). | Unchanged. |
| **Output validation** | Applied after NPU inference. | Applied after Intel GPU inference (requires GPU→host copy for validation). |
| **Fault isolation** | NPU driver errors caught; InferRequest reset/recreated; quarantine after repeated failures. | OpenCL/GPU errors caught; InferRequest reset/recreated; quarantine after repeated failures. |
| **Audit log** | Records `device: "NPU"`, NPU model hash, driver version. | Records `device: "INTEL_GPU"`, GPU model hash, driver version. |
| **Startup self‑test** | Golden input test for each NPU model. | Golden input test for each Intel GPU model. |
| **TLS & no admin APIs** | Unchanged. | Unchanged. |
| **Software identification** | Version reporting includes OpenVINO NPU driver version. | Version reporting includes OpenVINO GPU driver/OpenCL version. |

---

## 5. What Is Explicitly NOT in Phase 4

- **No changes to NVIDIA GPU (TensorRT) implementation** – already released in Phase 3.
- **No Moore Threads GPU** – remains on hold.
- **No request‑combining batching.** A model may opt into Triton‑style batch‑dimension shapes via `max_batch_size` (config `dims` per‑request, clients prepend the batch dim).
- **No live model updates or runtime model loading.**
- **No multi‑device per model** (each model instance group is tied to a single device; ensembles can combine multiple devices).
- **No gRPC or streaming.**

---

## 6. Architecture Diagram (Phase 4)

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
│  │       DETERMINISTIC AI RUNTIME                           ││
│  │       (CPU + NPU + Intel GPU + NVIDIA GPU)               ││
│  │                                                          ││
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ││
│  │  │ CPU       │ │ NPU       │ │ Intel GPU │ │ NVIDIA GPU│ ││
│  │  │ Pool      │ │ Pool      │ │ Pool      │ │ Pool      │ ││
│  │  │ (OpenVINO │ │ (OpenVINO │ │ (OpenVINO │ │ (TensorRT)│ ││
│  │  │ CPU)      │ │ NPU)      │ │ GPU)      │ │           │ ││
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘ ││
│  │                                                          ││
│  │  Ensemble DAG Executor                                   ││
│  │  (zero‑copy within CPU↔NPU, Intel GPU, NVIDIA GPU;       ││
│  │   single copy across host/OpenCL/CUDA boundaries)        ││
│  │  Output Validator                                        ││
│  │  Model Lifecycle Manager (manifest covers .npu_blob,     ││
│  │                            .gpu_blob, .plan)             ││
│  │                                                          ││
│  │  Memory Manager:                                         ││
│  │    host pool (CPU/NPU) + OpenCL pool (Intel GPU) +       ││
│  │    CUDA pool (NVIDIA) + pinned host pool                 ││
│  └───────────────────────┬──────────────────────────────────┘│
│                          │                                   │
│  ┌───────────────────────▼──────────────────────────────────┐│
│  │        AUDIT & DIAGNOSTIC LAYER (extended for all)       ││
│  │  Audit log records device, model hash, driver versions   ││
│  │  Self‑test covers all backends                           ││
│  │  Health endpoint reports all device statuses             ││
│  └──────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
```

---

## 7. Regulatory Impact

- **Risk file update:** New hazards added for Intel NPU and Intel GPU:
  - NPU driver crash/hang, compiled blob mismatch, silent computation corruption.
  - Intel GPU driver/OpenCL crash, kernel mis‑compilation, memory corruption, thermal throttling.
  Controls follow the same pattern: hash verification, instance quarantine, timeout, deterministic error codes, startup self‑test.

- **OTS validation extended:** OpenVINO NPU plugin, Intel NPU driver, Level Zero, OpenVINO GPU plugin, Intel GPU driver, OpenCL ICD, and all related libraries are added to the validated OTS list.

- **Documentation update:** SRS, architecture design, traceability matrix, and V&V plan are revised to include NPU, Intel GPU, and auto mode requirements/tests.

---

## 8. Phase 4 Success Criteria

- **Intel CPU:** Existing CPU models continue to work with the new `device` field; no regressions.
- **Intel NPU:** Approved NPU models load from precompiled blob, pass startup self‑test, and serve inference requests with latency overhead ≤5% vs. direct OpenVINO NPU API call.
- **Intel GPU:** Approved Intel GPU models load from precompiled blob, pass self‑test, and serve inference with latency overhead ≤5% vs. direct OpenVINO GPU API call.
- **Auto mode:** Models configured with `device: "auto"` are deployed on the best available Intel device at startup; behavior is deterministic and logged.
- **Concurrent multi‑device ensembles:** Ensembles mixing CPU, NPU, Intel GPU, and NVIDIA GPU execute correctly with zero‑copy within domains and single copies across domains; no deadlocks.
- **FDA controls hold for all devices:** Model integrity, output validation, audit logging (device identifier), fault isolation, and self‑test work identically across CPU, NPU, Intel GPU, and TensorRT.
- **No regression:** All Phase 2 (CPU) and Phase 3 (TensorRT) functionality continues to pass tests.
- **Documentation updated:** Risk file, SRS, and V&V reports reflect the new multi‑device Intel scope.

---

This Phase 4 elevates the server to a true **multi‑accelerator inference runtime** covering all Intel execution units and NVIDIA GPU, while maintaining the deterministic, traceable, and medical‑grade foundation established in previous phases.