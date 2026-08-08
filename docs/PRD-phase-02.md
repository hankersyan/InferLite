# PRD: Phase 2 – FDA‑Compliant CPU Inference Server  
**(Regulatory‑Ready Foundation – GPU Deferred to Phase 3)**

**Builds on:** Phase 1 OpenVINO CPU‑Only HTTP Server  
**Target Regulatory Framework:**  
- FDA 21 CFR Part 820 (QMSR / ISO 13485)  
- IEC 62304:2006+AMD1:2015 (Software Life Cycle)  
- ISO 14971:2019 (Risk Management)  
- FDA Guidance: “Content of Premarket Submissions for Device Software Functions” (2023)  
- FDA Guidance: “Cybersecurity in Medical Devices” (2026)  

---

## 1. Phase 2 Objective

Transform the Phase 1 CPU‑only inference server into a **deterministic, traceable, and verifiable medical‑device AI runtime**. This phase implements all P0 FDA‑critical features on the existing OpenVINO CPU foundation. GPU acceleration, TensorRT, and GPU‑enabled ensembles are **excluded** from Phase 2 and will be added in Phase 3 once the regulatory baseline is fully established and auditable.

The server remains a **single‑node, CPU‑only** execution engine. The HTTP API, model repository, and basic scheduler are retained and hardened. Ensemble DAGs and custom C++ plugins are introduced as CPU‑only capabilities, subject to the same strict controls.

---

## 2. Regulatory Highlights at a Glance (Phase 2)

| Compliance Objective | Implemented Feature (CPU‑Only) | Regulatory Reference |
|----------------------|--------------------------------|----------------------|
| **Model integrity & traceability** | Approved model manifest, SHA‑256 hash verification, model metadata package | IEC 62304 §5.2, 5.8; FDA Software Guidance §4 |
| **Deterministic execution** | Hard resource limits, timeout, validated‑configuration mode | ISO 14971; IEC 62304 §5.3.4 |
| **Input/output validation** | Strict tensor shape/dtype/size checks; output NaN/Inf/range validation | ISO 14971 risk control; IEC 62304 §5.3.4 |
| **Fault isolation & safe failure** | Request‑level exception containment, structured error codes, no silent failures | IEC 62304 §5.3.4; FDA Cybersecurity Guidance |
| **Audit trail** | Tamper‑evident, hash‑chained inference audit log | FDA Software Guidance §4; 21 CFR 820.30 |
| **Health & self‑test** | Startup functional self‑test with golden input, detailed health endpoint | IEC 62304 §5.6; ISO 14971 |
| **Secure communication** | TLS 1.2+ in validated mode, no admin APIs | FDA Cybersecurity Guidance (2026) |
| **Configuration versioning** | Hashed configuration & manifest, locked runtime | IEC 62304 §5.8 |
| **Software identification** | Version reporting of server, OpenVINO, and models | FDA Software Guidance §4 |
| **CPU ensembles & plugins** | Directed acyclic graph executor (CPU), C++ plugin backend (CPU) with full isolation | IEC 62304 §5.3 (architectural segregation) |

All features are **traceable** to requirements and risk controls.

---

## 3. System Architecture (CPU‑Only, FDA‑Annotated)

```
┌──────────────────────────────────────────────────────────────┐
│                  Medical Device                              │
│  ┌──────────────────────────────────────────────────────────┐│
│  │ Clinical Application (e.g., X‑ray workflow)              ││
│  └───────────────────────┬──────────────────────────────────┘│
│                          │                                   │
│  ┌───────────────────────▼──────────────────────────────────┐│
│  │            SAFETY BOUNDARY (FDA Cybersecurity)           ││
│  │                                                          ││
│  │  • Input Validation        [ISO 14971 Risk Control]      ││
│  │  • Resource Limits         [IEC 62304 §5.3.4]            ││
│  │  • Fault Isolation         [IEC 62304 §5.3.4]            ││
│  │  • Safe Failure/Fallback   [ISO 14971]                   ││
│  └───────────────────────┬──────────────────────────────────┘│
│                          │                                   │
│  ┌───────────────────────▼──────────────────────────────────┐│
│  │       DETERMINISTIC CPU AI RUNTIME (IEC 62304 Class C)   ││
│  │                                                          ││
│  │  • Scheduler (CPU instance pool, FIFO queue)             ││
│  │  • Model Lifecycle Manager (manifest‑driven)             ││
│  │  • Model Integrity Verifier (SHA‑256)                    ││
│  │  • CPU Ensemble DAG Executor (zero‑copy host memory)     ││
│  │  • Output Validator (per‑model checks)                   ││
│  │  • C++ Plugin Backend (CPU only, sandboxed)              ││
│  │                                                          ││
│  │  Backend: OpenVINO CPU                                   ││
│  │  Memory Manager: Host memory pool only                   ││
│  └───────────────────────┬──────────────────────────────────┘│
│                          │                                   │
│  ┌───────────────────────▼──────────────────────────────────┐│
│  │        DIAGNOSTIC & AUDIT LAYER (FDA Traceability)       ││
│  │                                                          ││
│  │  • Tamper‑evident Audit Log (hash‑chained entries)       ││
│  │  • Diagnostic Log (engineer‑facing)                      ││
│  │  • Health/Self‑Test (golden input verification)          ││
│  │  • Version & Configuration Reporting                     ││
│  └──────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
```

All components run on CPU. No CUDA, no TensorRT, no GPU memory pool.

---

## 4. Detailed FDA‑Critical Features (No GPU)

### 4.1 Model Lifecycle & Integrity

- **Manifest (`manifest.json`):** Lists every approved model/version with its SHA‑256 hash. The manifest itself is hashed and verified at startup against a known good value.
- **File hash verification:** All model files (`.xml`, `.bin`) are hashed and compared to the manifest. Mismatch → server refuses to start.
- **Model metadata (`metadata.json`):** `model_id`, `version`, `intended_use`, `training_dataset_id`, `approval_status`. Logged into audit trail.
- **Immutable model set:** No runtime loading/unloading; model changes require a controlled device software update.

### 4.2 Deterministic Resource Management (CPU‑Only)

Hard limits enforced before and during inference:

| Limit | Default | Enforcement |
|-------|---------|-------------|
| `MAX_CONCURRENT_REQUESTS` | Number of CPU instances | Scheduler queue depth |
| `MAX_QUEUE_DEPTH` | 100 | Reject with 503 |
| `MAX_INPUT_SIZE_BYTES` | 50 MB | Input validation |
| `MAX_OUTPUT_SIZE_BYTES` | 50 MB | Output validation |
| `MAX_INFERENCE_TIME_MS` | 5000 | Per‑request timer |
| `MAX_CPU_THREADS` | 4 | Thread pool cap |

Exceeding any limit results in a structured error response; no unbounded resource consumption.

### 4.3 Input & Output Validation

- **Input validation:** Tensor names, data types, shapes, and byte sizes checked against model spec. Invalid → `INVALID_INPUT` error.
- **Output validation (per‑model):**
  - NaN/Inf detection.
  - Range checks (min/max values).
  - Shape conformity.
  - Optional confidence threshold.
  - Failure → `OUTPUT_VALIDATION_FAILED` error; no result returned.

### 4.4 Fault Isolation & Safe Failure

- **Exception containment:** All OpenVINO inference and plugin execution wrapped in try/catch; any exception caught, logged, and returned as `INTERNAL_ERROR`.
- **Structured error codes:** Every failure mode (timeout, memory, validation, backend error) mapped to a unique error code.
- **No silent failures:** Any unhandled error results in a safe error response, never a corrupted output or server crash.
- **Graceful degradation:** If an OpenVINO instance becomes unrecoverable, it is marked dead; subsequent requests may be queued for healthy instances or rejected.

### 4.5 Traceability & Audit Logging

- **Tamper‑evident audit log:** Append‑only file; each entry contains:
  ```json
  {
    "trace_id": "uuid",
    "timestamp": "ISO8601",
    "model_id": "ChestPA_Collimation",
    "model_version": "2.3.1",
    "model_hash": "abc123...",
    "software_version": "InferenceServer 2.0.0",
    "config_hash": "def456...",
    "input_shape": [1,3,224,224],
    "inference_status": "SUCCESS",
    "error_code": "",
    "duration_ms": 120,
    "device": "CPU"
  }
  ```
  Entries are hash‑chained to detect tampering. No patient data is stored.
- **Diagnostic log:** Separate, human‑readable log for operational events (queue depth, CPU usage, errors).

### 4.6 Health Monitoring & Startup Self‑Test

- **Functional self‑test:** On startup, for each approved model, a known golden input is run; output is compared to expected tensor (bit‑for‑bit or within epsilon). Only if all models pass does the server signal `READY`.
- **Health endpoint:** `GET /v2/health/ready` returns `200` only if self‑tests passed and resources are healthy. `GET /v2/health/detailed` gives per‑model status.

### 4.7 Security Baseline (CPU‑Only)

- **TLS 1.2+** enforced in validated mode.
- **No admin APIs:** Only inference and health endpoints exposed.
- **Minimal privileges:** Server runs under a restricted OS account.
- **Manifest verification** ensures only approved models are loaded.

### 4.8 Configuration & Version Management

- **Configuration hashing:** All `config.pbtxt` and manifest hashed at startup; hash is logged and reported.
- **Software version reporting:** Server version, OpenVINO version, model versions available via log/endpoint.
- **No runtime configuration changes** allowed.

---

## 5. Technical Additions (CPU‑Only, Controlled)

The following **new technical capabilities** are added on top of Phase 1, but remain entirely CPU‑bound and subject to all FDA controls:

### 5.1 Ensemble DAG Executor (CPU Only)

- **Static DAG** defined in `config.pbtxt` with `backend: "ensemble"`.
- Steps reference other CPU models or plugins.
- **Zero‑copy host memory** passing: intermediate tensors stay in the memory manager’s host pool; pointers are forwarded between steps without copying.
- Steps run sequentially or concurrently (dependent on DAG topology) using the CPU thread pool.
- Each step is isolated: a failure in one step cancels the entire ensemble with a clear error code, and all resources are released.

### 5.2 C++ Plugin Backend (CPU Only)

- Plugins are compiled as shared libraries (`.so`), loaded at startup, and verified against a hash in the manifest.
- Each plugin implements a `PluginNode::execute(host_inputs, host_outputs)` method.
- Plugins run on the CPU thread pool, subject to the same timeout and resource limits.
- Plugins are treated exactly like model nodes in ensembles and standalone pipelines.

### 5.3 HTTP API & Metrics (Unchanged)

- Same `/v2/models/.../infer` interface; now works with ensemble and plugin models transparently.
- Metrics endpoint expanded with queue depth, per‑model latency, and configuration hash.

---

## 6. What Is NOT in Phase 2 (Deferred to Phase 3)

- **No GPU backends** (TensorRT, OpenVINO GPU).
- **No CUDA streams, GPU memory manager, or GPU‑specific zero‑copy.**
- **No GPU ensembles** (all DAGs are CPU‑only).
- **No batching** (dynamic or static).
- **No live model updates** or A/B versions.
- **No profiling tool.**

These items will be added in Phase 3, which will start from the validated CPU‑only baseline and incrementally introduce GPU acceleration under the same design controls.

---

## 7. Regulatory Documentation Package (Produced in Phase 2)

| Document | Content |
|----------|---------|
| Software Development Plan (SDP) | Lifecycle, roles, tools, standards |
| Software Requirements Specification (SRS) | 80+ requirements covering all P0 features |
| Architecture & Detailed Design | CPU‑only architecture, DAG executor, plugin interface, error handling |
| Risk Management File (ISO 14971) | Hazard analysis for CPU inference (wrong output, hang, resource exhaustion, model tampering) |
| Traceability Matrix | Requirement → Design → Risk → Test |
| Verification & Validation Plan | Unit, integration, system tests, stress tests |
| OTS Software Validation Report | OpenVINO, HTTP lib, OS dependencies |
| Cybersecurity Documentation | Threat model, TLS, manifest verification |

All documentation is maintained under version control and reviewed at Phase 2 completion.

---

## 8. Request Lifecycle (CPU Ensemble with FDA Controls)

1. **Client (X‑ray app) sends HTTPS POST** → input validated (shape/dtype/size).
2. **Scheduler** checks queue depth and instance availability. Over limit → `RESOURCE_EXHAUSTED`.
3. **Audit trace** created with `trace_id`.
4. **Timeout** started.
5. **Ensemble execution:**
   - Preprocess plugin (CPU) → isolated.
   - OpenVINO model A (CPU) → output in host pool.
   - OpenVINO model B (CPU) → receives pointer, no copy.
   - Postprocess plugin (CPU) → output validation applied.
6. **Output validation** → passes → result returned with trace ID.
7. **Audit log entry** finalized and hash‑chained.
8. **Any failure** → error code returned, resources freed, audit log updated.

---

## 9. Success Criteria for Phase 2

- **Compliance baseline:** All P0 features (integrity, resource limits, validation, audit log, self‑test, fault isolation) implemented and verified.
- **Ensemble & plugins:** DAG executor correctly handles CPU‑only chains; plugins load and execute within bounds.
- **Determinism:** Same input/configuration yields identical output.
- **Fault handling:** Injected faults (corrupted model, OpenVINO exception, memory pressure) produce defined error codes; server remains stable.
- **Traceability:** Every inference captured in tamper‑evident audit log with full software/model version chain.
- **Documentation:** Complete SRS, risk file, architecture, traceability matrix aligned with the implemented feature set.
- **Regulatory readiness:** The server and documentation can be used as the software component for a 510(k) or PMA submission (clinical AI function validation is separate).

---

This Phase 2 delivers a **medical‑grade, CPU‑only AI inference runtime** with all necessary FDA controls. GPU acceleration is deliberately postponed to Phase 3, where it can be added on top of a fully validated and auditable foundation, significantly reducing regulatory risk.