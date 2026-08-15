## Concise Triton Feature List — Single Node Workstation

### Hardware & Backends
- Single NVIDIA GPU, CPU-only, or Intel xPU
- Backends: TensorRT, ONNX Runtime, PyTorch, TensorFlow, OpenVINO, Python, FIL, DALI
- Mix CPU and GPU models on same node

### Model Repository & Management
- Local filesystem model repository
- Multiple models per server
- Multiple versions per model
- Version policies: specific / latest / all
- Explicit load/unload at runtime
- Local only — no cloud storage required

### Scheduling & Performance
- Dynamic batching (optional)
- Multiple model instances (optional)
- Sequence batching for stateful models
- Cross-model rate limiter
- Model warmup
- Request priority and bounded queues
- Response cache (optional)

### Pipelines & Multi-Model
- Static ensemble model — DAG of models served as one endpoint
- Business Logic Scripting (BLS) — Python-based dynamic pipelines
- Mixed backends in one pipeline
- Direct multi-model serving without ensembles

### Protocols & Data Transfer
- HTTP/REST
- gRPC unary
- gRPC streaming
- In-process C++/Python API
- System shared memory
- CUDA shared memory

### Observability & Operations
- Health/readiness endpoints
- Prometheus metrics
- Request tracing
- Runtime logging
- Per-model statistics

### Performance Tools
- `perf_analyzer` — latency/throughput benchmarking
- `model_analyzer` — config tuning
- `genai-perf` — LLM benchmarking

### Not Included / Not Needed
- Kubernetes autoscaling
- Cloud object storage
- Multi-node distributed inference
- Built-in authentication
- High availability/failover

For deterministic use: disable dynamic batching, use 1 instance, fixed shapes, specific version, explicit control, model warmup.