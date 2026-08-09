# V&V Report: Phase 3 — TensorRT GPU Acceleration (Draft)

**Scope:** Verification and validation evidence for `docs/PRD-phase-03.md`.
**Status:** Draft for review. GPU-path automated execution requires a machine
with the validated OTS TensorRT SDK installed; CPU-only regression is verified
in this workspace.

---

## 1. Environment Constraint

This development workspace has CUDA v12.6 and an NVIDIA GTX 1070 (8 GB) GPU
present, but **no TensorRT SDK installed**. The TensorRT GPU backend is
therefore implemented as **opt-in**: it compiles only when `TENSORRT_ROOT`
points to a TensorRT install at CMake configure time. The CPU-only server
builds and runs with no regression, verified below.

> This is the same pattern used for OpenVINO detection. On a machine with the
> validated TensorRT OTS installed, `cmake -DTENSORRT_ROOT=...` activates the
> GPU path and the automated GPU V&V cases in §3 can be run.

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

### 3.2 GPU-path — PENDING on TensorRT OTS

Requires a machine with TensorRT installed. Automated cases to run (see §5):

1. TensorRT model loads from `model.plan` and passes golden self-test.
2. TensorRT inference output equals a CPU reference within FP32 tolerance.
3. Latency overhead ≤5% vs. direct TensorRT API call.
4. Mixed CPU+GPU ensemble executes with correct cross-device copies.
5. GPU instance quarantine on injected CUDA fault; server stays stable.
6. `RESOURCE_EXHAUSTED` when GPU memory cap is exceeded.
7. Manifest mismatch on a modified `.plan` → fail-fast startup.

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

## 5. Recommended Automated GPU Test Script (to run on TensorRT OTS)

A PowerShell harness `test_gpu.ps1` is recommended to drive §3.2. It should:

1. Place a compiled `model.plan` under `models/<model>/<version>/` and regenerate
   `manifest.json` with `tools/make_manifest.py`.
2. Start the server with `--validated-mode --max-gpu-memory-mb=...`.
3. Assert ready, infer (compare output to reference), read detailed health for
   `device:"GPU"`, and confirm the audit log row contains `device:"GPU"`.

---

*End of draft V&V report.*
