# InferLite — Medical Equipment Level & FDA Compliance

InferLite is designed as a **software component of a regulated medical device**.
It is architected to meet the FDA's software lifecycle and cybersecurity
expectations, and can serve as the inference runtime layer for a medical AI
product (e.g., a chest X-ray collimation or clinical image-analysis workflow).

> **Important:** InferLite is a *software-only runtime*. Achieving FDA clearance
> (510(k) / De Novo / PMA) requires the full device context — including clinical
> validation of the AI function, the hosting system, usability, and the
> manufacturer's Quality System (QMS). This section documents the compliance
> posture of the inference runtime itself, which is the prerequisite foundation.

## Regulatory Framework

The runtime is built against the following frameworks and guidance:

| Framework | Applicability |
|-----------|---------------|
| FDA 21 CFR Part 820 (QMSR / ISO 13485) | Design controls, traceability, records |
| IEC 62304:2006 + AMD1:2015 | Software life-cycle (safety class C posture) |
| ISO 14971:2019 | Risk management (hazard & risk controls) |
| FDA Software Guidance (2023) | Premarket software documentation |
| FDA Cybersecurity Guidance (2026) | Secure by design, trust boundaries |

## Medical Equipment Level (Software Classification)

InferLite is developed and intended for use as **Class C software** in the
IEC 62304 sense — software whose failure could contribute to patient harm
through a wrong inference output, a hung request, or resource exhaustion. To
earn that classification the runtime implements the following **safety
mechanisms** (traceable to risk controls):

- **Deterministic execution** — static model set, no runtime model loading, no
  request‑combining batching (identical input + configuration yields identical
  output); `max_batch_size` only adds a Triton‑style batch dimension to shapes,
  so inference remains a deterministic 1:1 request→output mapping.
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

## FDA Compliance Features (Implemented)

Each feature maps to a regulatory requirement:

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

## Validated (Locked) Mode

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

## Required Documentation Package

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

## Security & Privacy Notes

- **No patient data is stored.** The audit log records tensor *shapes*, hashes,
  and timing — never the tensor payloads.
- **Minimal attack surface.** Only inference and health endpoints are exposed;
  there is no admin or management API.
- **Minimal privileges.** Deploy the server under a restricted OS account and
  place it behind a network boundary.
