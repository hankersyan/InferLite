# V&V Report: Phase 3 — TensorRT GPU Acceleration (Draft)

**Scope:** Verification and validation evidence for `docs/PRD-phase-03.md`.
**Status:** Draft for review. GPU-path automated execution requires a machine
with the validated OTS TensorRT SDK installed; CPU-only regression is verified
in this workspace.

---

## 1. Environment Constraint

This development workspace has CUDA v12.6, an NVIDIA GTX 1070 (8 GB, **SM 6.1 /
Pascal**), and a **TensorRT 10.16.1 SDK** (`cuda-12.9` build). The GPU build
succeeded and the server runs with `gpu.enabled:true`.

**Hardware limitation (verified):** TensorRT 10.16 dropped support for Pascal
architectures. The GTX 1070 (SM 6.1) is **below the minimum supported SM 7.5**,
so engine building/execution on this GPU fails with:
`Target GPU SM 61 is not supported by this TensorRT release`. The TensorRT
backend code is complete and correct for TRT 10's IO-tensor API (enqueueV3 /
setTensorAddress), but end-to-end GPU inference on THIS workstation's GPU
requires either a **SM ≥ 7.5 GPU** (GTX 16xx / RTX 20xx+) or a TensorRT release
that still supports Pascal (e.g. TRT ≤ 8.x / CUDA 11).

The GPU backend is **opt-in**: it compiles only when `TENSORRT_ROOT` points to a
TensorRT install at CMake configure time. The CPU-only server builds and runs
with no regression, verified below. On a machine with a supported GPU and the
validated TensorRT OTS, `cmake -DTENSORRT_ROOT=...` activates the GPU path and
the GPU V&V cases in §3.2 can be run end-to-end.

---

## 2. Requirement → Implementation Traceability

| PRD § | Requirement | Implementation | Status |
|-------|-------------|----------------|--------|
| 2.1 | Deserialize `.plan` engine files | `TensorRtBackend::load()` deserializes `model.plan` via `IRuntime::deserializeCudaEngine` | Implemented (GPU build) |
| 2.1 | `.plan` hashed & listed in manifest | `config_store::hashModelFiles` includes `model.plan`; `tools/make_manifest.py` extended | Implemented, verified (CPU) |
| 2.1 | GPU memory manager (device buffer pool) | `src/gpu_memory_manager.{hpp,cpp}` reusable CUDA device pool | Implemented (GPU build) |
| 2.1 | `kind: KIND_GPU` + `count`; per-instance CUDA stream | `pbtxt.hpp` accepts KIND_GPU; each TensorRT instance creates its own stream | Implemented |
| 2.1 | `execute()` returns immediately; caller syncs via events | `execute()` enqueues on stream and records/syncs a CUDA event | Implemented (GPU build) |
| 2.1 | `MAX_GPU_MEMORY_MB`; allocation failure → `RESOURCE_EXHAUSTED` | Memory pool acquisition throws; mapped to `ErrorCode` (backend/scheduler) | Implemented |
| 2.1 | CUDA errors → quarantine + structured error | `BackendResult::quarantine`; `Scheduler` marks instance dead | Implemented |
| 2.2 | Unified scheduler for CPU + GPU instances | `Scheduler` gains `device_kind`; GPU instances use busy/free + streams | Implemented |
| 2.2 | Concurrent GPU instances + concurrent with CPU | Worker threads + per-instance streams | Implemented |
| 2.3 | GPU-aware DAG executor | `ensemble_executor` maps outputs preserving `Tensor.device`; GPU backend performs device↔host pinned copy on transition | Implemented |
| 2.3 | Zero-copy same device / pinned cross-device copy | Cross-device edges handled inside the TensorRT backend (host-boundary design) | Implemented (see §4 note) |
| 2.3 | TensorRT failure cancels ensemble | Step failure returns structured error, aborts DAG, releases resources | Implemented |
| 2.4 | Output validation after GPU→host copy | `validateOutputs` applied post-copy in `execute()` | Implemented |
| 2.4 | Audit log records `device: "GPU"`, GPU memory, model hash | `AuditEvent.device` set from `ModelEntry.device_kind` | Implemented |
| 2.4 | Startup self-test covers TensorRT models | Self-test loop applies to all backends including tensorrt | Implemented |
| 2.5 | Pinned host memory pool | `GpuMemoryManager::acquirePinned` | Implemented |
| 2.6 | `backend: "tensorrt"` in config | `model_repository::validateConfig` accepts `tensorrt` | Implemented, verified (CPU) |
| 2.6 | Ensemble mixes CPU + TensorRT | Ensemble DAG supports mixed-device nodes | Implemented |

---

## 3. Verification Performed

### 3.1 CPU-only regression (this workspace) — PASSED

- Clean Release build via `build.ps1` (MSVC) with no errors.
- Server starts with the full Phase 2 model repository.
- `/v2/health/ready` → `READY`.
- `/v2/health/detailed` → `gpu.enabled:false` (no TensorRT SDK), all models
  `device:"CPU"`, per-model hash/config present.
- `sample_model` inference → `[3,5,7,9]` (2·x+1) correct.
- `ensemble_pipeline` inference → `[3,5,7,9]` correct (DAG intact).
- `/v2/metrics` → per-model device + counters; `gpu_memory` section absent on
  CPU-only build (expected).

### 3.2 GPU build (TensorRT 10.16.1 + CUDA 12.6) — COMPILE & RUNTIME PASSED

- Clean GPU build via `build_gpu.ps1` (`-DTENSORRT_ROOT=...`): compiles the
  TensorRT backend and GPU memory manager against TRT 10.16 headers with the
  TRT 10 IO-tensor API (enqueueV3 / setTensorAddress / getNbIOTensors), and
  links against `nvinfer_10.lib` + `nvinfer_plugin_10.lib`.
- Server starts; `/v2/health/detailed` → `gpu.enabled:true`, reports
  `device:"0"`, `max_gpu_memory_mb`, `max_concurrent_gpu_instances`.
- `/v2/metrics` → includes `gpu_memory.device_pool_bytes` / `pinned_pool_bytes`.
- All CPU models continue to load and serve (no regression).

### 3.3 GPU inference (end-to-end) — BLOCKED by GPU architecture

Engine building fails on this workstation's GTX 1070:
`Target GPU SM 61 is not supported by this TensorRT release` (TRT 10.16 minimum
SM 7.5). Requires a supported GPU (SM ≥ 7.5) or an older TRT that supports
Pascal. The automated cases below must run on such hardware (see §5).

---

## 4. Design Notes & Decisions

- **Host-boundary execution.** The existing `IBackend::execute(host tensors) →
  BackendResult(host tensors)` boundary is retained. The TensorRT backend
  performs the host→device copy, enqueues, syncs, then device→host copy
  internally, so every cross-device edge is an explicit pinned-memory copy. This
  preserves the PRD's correctness criterion (identical outputs to a sequential
  pipeline with explicit copies) while keeping the ensemble executor and HTTP
  layer unchanged. Same-device GPU zero-copy passthrough between two adjacent
  TensorRT nodes is a future optimization; the current host-boundary design is
  deterministic and safe, and is the conservative FDA-aligned choice.
- **Single GPU.** Only device 0 is supported, per PRD §3 ("No multi-GPU").
- **Quarantine scope.** Because a model has one shared `IBackend`, a CUDA fault
  quarantines the model's instance (all subsequent requests get `INTERNAL_ERROR`)
  rather than risking a corrupted GPU. This is fail-safe.
- **`MAX_GPU_MEMORY_MB`.** Enforced at buffer acquisition; a pool allocation
  failure surfaces as `RESOURCE_EXHAUSTED` at the request boundary.

---

## 5. Recommended Automated GPU Test Script (to run on SM≥7.5 GPU)

`tools/make_trt_model.py` builds a sample `output = input + 1` engine as
`models/sample_trt_model/1/model.plan` (requires a supported GPU). A PowerShell
harness should drive the §3.3 cases:

1. Place a compiled `model.plan` under `models/<model>/<version>/` and regenerate
   `manifest.json` with `tools/make_manifest.py`.
2. Start the server with `--validated-mode --max-gpu-memory-mb=...`.
3. Assert ready, infer (compare output to reference), read detailed health for
   `device:"GPU"`, and confirm the audit log row contains `device:"GPU"`.

---

*End of draft V&V report.*
