# NVIDIA Triton Inference Server — Combined Master Feature List

This document combines the most useful details from three previous Triton feature summaries into one detailed, operational reference. It includes current context, core capabilities, backend-dependent features, and deployment caveats.

> **Note on naming and direction:**  
> Since March 2025, NVIDIA has positioned Triton as **Dynamo-Triton** within the broader **NVIDIA Dynamo Platform**.  
> - **Triton / Dynamo-Triton** remains the general-purpose inference server for vision, speech, classical ML, ensembles, and single-node generative AI.  
> - **NVIDIA Dynamo** is the newer platform aimed at large-scale, multi-node, disaggregated LLM serving with KV-aware routing, prefill/decode splitting, and multi-tier KV-cache management.  
>
> Unless otherwise stated, the features below refer to the core Triton Inference Server and its standard backends.

---

## 1. Multi-Framework and Backend Support

Triton serves models from many frameworks through a backend architecture.

### Built-in / commonly available backends

- **TensorRT** — NVIDIA-optimized inference
- **TensorRT-LLM** — optimized LLM serving with continuous batching, paged KV-cache, quantization, tensor/pipeline parallelism
- **PyTorch** — TorchScript / LibTorch
- **ONNX Runtime** — CPU/GPU execution, optionally using TensorRT execution provider
- **TensorFlow** — SavedModel / GraphDef  
  ⚠️ Availability may require a separate backend build or container in recent Triton releases.
- **OpenVINO** — Intel CPU/GPU/VPU
- **Python backend** — arbitrary Python code, Hugging Face pipelines, custom logic
- **vLLM backend** — serve models through vLLM’s PagedAttention engine
- **FIL / Forest Inference Library** — classical ML models:
  - XGBoost
  - LightGBM
  - scikit-learn RandomForest
  - cuML forest models
- **DALI** — NVIDIA Data Loading Library for GPU-accelerated preprocessing and augmentation
- **Custom C/C++ backends** — proprietary frameworks, accelerators, non-neural algorithms

### Extensibility

- Custom C/C++ backend API
- Python backend for rapid development and orchestration
- Repository agents for load-time model preparation, decryption, conversion, and validation
- Support for framework-specific custom operators when required libraries are supplied

A single Triton process can simultaneously serve models from multiple frameworks/backends.

---

## 2. Hardware and Platform Support

Triton can run across multiple hardware targets:

- NVIDIA GPUs
- x86 CPUs
- ARM / SBSA CPUs
- NVIDIA Jetson and other edge devices
- Cloud, data center, embedded environments
- AWS Inferentia indirectly through Python backend / AWS Neuron integration

### Deployment forms

- Prebuilt NVIDIA NGC containers
- Custom Docker images
- Build from source
- CPU-only builds
- Standalone server process
- **In-process Triton library** embedded in C/C++, Python, or Java applications

Backend and platform support are not uniform. For example, TensorRT requires NVIDIA GPUs, while ONNX Runtime can execute on CPU or GPU. Always check the official platform/backend support matrix.

---

## 3. Model Repository and Storage

Triton loads models from one or more **model repositories**.

### Repository structure

```text
model_repository/
└── model_name/
    ├── config.pbtxt
    ├── 1/
    │   └── model_file
    ├── 2/
    │   └── model_file
    └── labels.txt
```

### Repository features

- Multiple models per repository
- Multiple repositories attached to one server
- Multiple model versions
- Per-model configuration
- Label files for classification models
- Model-specific backend libraries and artifacts
- Version-selection policies
- Additional named model configurations

### Supported repository locations

- Local filesystem
- Mounted network filesystem
- Amazon S3
- Google Cloud Storage
- Microsoft Azure Blob Storage
- S3-compatible object storage

Remote repositories are copied/mounted locally for loading, depending on configuration.

---

## 4. Model Configuration

Each model can have a `config.pbtxt` describing runtime behavior.

### Configurable items

- Input/output tensor names
- Tensor data types
- Fixed and variable tensor dimensions
- Maximum batch size
- Dynamic batching settings
- Sequence batching settings
- Model instance count
- CPU/GPU placement
- GPU device selection
- Model version policy
- Queue policy and request priorities
- Model warmup
- Response caching
- Rate-limiter resources
- Backend-specific parameters
- Tensor reshaping
- Ragged batching
- Optimization settings
- Ensemble scheduling
- Custom metric controls

### Auto-completed configuration

Triton can infer minimal configuration for:

- TensorRT
- ONNX Runtime
- OpenVINO

Python models can implement `auto_complete_config()` to supply metadata programmatically. Auto-completion can be disabled for strict control.

---

## 5. Model Versioning

Triton supports multiple versions of a model:

```text
my_model/
├── 1/
├── 2/
└── 3/
```

### Version policies

A model can expose:

- Latest version
- N most recent versions
- Specific version numbers
- All available versions

Clients may request a specific version or let Triton choose the default according to policy.

This enables:

- Controlled rollouts
- A/B testing
- Backward compatibility
- Canary deployments
- Fast rollback

---

## 6. Model Lifecycle and Management

Triton supports three model-control modes.

### `NONE`

- Default mode
- Loads all available models at startup
- Repository changes ignored while running
- Load/unload APIs disabled

### `EXPLICIT`

- Models loaded/unloaded through API
- Server can start with selected models or no models
- Good for controlled production deployments
- Useful when GPU memory cannot hold all models

### `POLL`

- Triton periodically checks the repository
- Models loaded/reloaded/unloaded on file changes
- Convenient for development
- Not recommended for production because repository may be observed partially updated

Model management operations are available over HTTP, gRPC, and in-process APIs.

---

## 7. Concurrent Model Execution

Triton can run:

- Different models concurrently
- Multiple instances of the same model concurrently
- Instances across multiple GPUs
- CPU and GPU model instances simultaneously

### Instance groups

`instance_group` controls:

- Number of model instances
- CPU or GPU execution
- Which GPUs host instances
- Parallel execution capacity

Example:

```protobuf
instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [0, 1]
  }
]
```

Concurrent execution improves throughput when one instance does not fully utilize the underlying hardware.

---

## 8. Scheduling and Performance Features

### Dynamic Batching

Triton can combine individual requests into larger batches before model execution.

Example:

```text
Request A: batch 1
Request B: batch 1
Request C: batch 1
Request D: batch 1
        ↓
One model execution with batch 4
```

Dynamic batching controls:

- Maximum batch size
- Preferred batch sizes
- Maximum queue delay
- Priority levels
- Per-priority queue policies
- Request timeouts
- Maximum queue size
- Preserve response ordering
- Custom batching strategies
- Ragged batching

Larger GPU batches usually improve throughput, but queuing to form batches can add latency.

### Ragged Batching

Allows variable-sized inputs to be batched without padding all tensors to the same shape.

Useful for:

- Variable-length text
- Variable-length audio
- Variable-sized feature arrays
- Recommendation-system features

Triton can generate additional batch-input tensors containing element counts or offsets. The model must understand the concatenated representation.

### Sequence Batching and Stateful Inference

Supports requests belonging to ordered sequences that must maintain state or routing affinity.

Sequence features:

- Sequence/correlation IDs
- Start and end indicators
- Ready controls
- Configurable idle timeout
- Routing all requests in a sequence to the same model instance
- Direct and oldest scheduling strategies
- Dynamic batching across active sequences
- Implicit state management

#### Implicit state management

Triton can manage state tensors between requests. A model can output a state tensor, and Triton stores it and supplies it as input for the next request in the same sequence. This allows otherwise stateless models to participate in stateful workflows.

### Queue Management and Priorities

Each model has its own scheduler.

Scheduler types:

- Default scheduler
- Dynamic batch scheduler
- Sequence scheduler
- Ensemble scheduler

Queue controls:

- Priority levels
- Default request priority
- Request timeout
- Maximum queue size
- Different timeout behavior per priority
- Rejecting requests when limits are reached
- Preserving response order

### Cross-Model Rate Limiter

The rate limiter controls when prepared inference executions may run on model instances.

It can coordinate scheduling across **all loaded models**.

Rate-limiter capabilities:

- Cross-model resource constraints
- Model-instance priorities
- Global or device-specific resources
- Limiting execution based on logical resources
- Preventing memory-intensive models from running simultaneously
- Prioritizing important models

Logical resources may represent GPU workspace, accelerator memory, CPU threads, or application-defined concurrency limits.

### Model Warmup

Model warmup runs predefined sample requests while a model is loading.

Used to initialize:

- CUDA kernels
- Lazy backend initialization
- TensorRT execution resources
- Memory allocations
- Framework caches
- Shape-specific optimization paths

A model is not marked ready until configured warmup requests complete. This avoids first-inference latency for production traffic.

---

## 9. Model Pipelines and Composition

### Ensemble Models

An ensemble model connects multiple Triton models into a directed acyclic graph (DAG).

Example:

```text
Image bytes
    ↓
DALI preprocessing
    ↓
TensorRT classifier
    ↓
Python postprocessing
    ↓
Classification result
```

Ensemble characteristics:

- Tensor-based routing between models
- Models may use different backends
- Intermediate tensors passed in memory without serialization
- One client request triggers the entire pipeline
- Components remain independently configurable and reusable

Ensembles are best for relatively static tensor-processing graphs.

### Business Logic Scripting (BLS)

A Python backend model can call other Triton models programmatically.

Supports:

- Conditional execution
- Loops
- Branching
- Calling multiple models
- Combining responses
- Error handling
- Dynamic pipeline selection
- Asynchronous model calls
- Recursive composition with constraints

BLS is more flexible than ensembles when the workflow cannot be expressed as a fixed DAG.

---

## 10. Decoupled Models and Streaming Responses

Standard execution produces one response per request. Decoupled model API allows:

- Zero responses
- One response
- Multiple responses over time

Important for:

- LLM token streaming
- Streaming ASR
- Iterative algorithms
- Progressive image generation
- Event detection

Decoupled behavior requires backend support and is commonly used with gRPC streaming.

---

## 11. Request Cancellation

Clients can cancel long-running or no-longer-needed requests.

Use cases:

- LLM generation
- Large backlogs
- Streaming inference
- User-aborted requests
- Application deadlines

Triton checks cancellation at key scheduling and pipeline points. Cancellation support varies by API and client, and may also depend on backend cooperation.

---

## 12. Inference Protocols and APIs

Triton implements KServe-compatible v2 protocols over:

- HTTP/REST
- gRPC
- Bidirectional gRPC streaming

### Protocol operations

- Server metadata
- Model metadata
- Model configuration
- Server liveness
- Server readiness
- Model readiness
- Inference
- Model statistics
- Repository index
- Model load
- Model unload
- Trace configuration
- Logging configuration

### Protocol extensions

Triton provides extensions for:

- Binary tensor data
- Classification results
- Request scheduling policy
- Sequence information
- System shared memory
- CUDA shared memory
- Model configuration
- Model repository management
- Inference statistics
- Tracing
- Logging
- Request parameters

---

## 13. Shared Memory Data Transfer

Clients can send or receive tensor data through:

- System shared memory
- CUDA shared memory

Instead of serializing large tensors into HTTP/gRPC messages, the client registers a memory region and tells Triton where the tensor resides.

Benefits:

- Reduced serialization overhead
- Reduced copying
- Lower latency for large tensors
- Efficient same-host inference
- Direct GPU-resident data use with CUDA shared memory

Shared-memory access must be enabled on the server.

---

## 14. Client Libraries

Official client support:

- Python HTTP client
- Python gRPC client
- C++ HTTP client
- C++ gRPC client
- Java client
- Generated gRPC clients for other languages

Client features may include:

- Synchronous inference
- Asynchronous inference
- gRPC streaming
- Python `asyncio`
- HTTP/gRPC compression
- SSL/TLS configuration
- Shared-memory utilities
- Custom request headers
- Request cancellation
- Model metadata and statistics APIs

Exact features differ between HTTP, gRPC, Python, C++, and Java clients.

---

## 15. In-Process Server APIs

Triton can be embedded directly into an application as a shared library.

Available in-process interfaces:

- C/C++
- Python
- Java

Useful for:

- Edge applications
- Embedded systems
- Avoiding network overhead
- Custom service processes
- Direct lifecycle control

The in-process C API exposes model management, inference, server options, metrics, and cancellation.

---

## 16. Observability

### Prometheus Metrics

Triton exposes Prometheus-format metrics from a dedicated endpoint.

#### Request metrics

- Successful/failed request count
- Inference count
- Model execution count
- Pending request count
- Rejected requests
- Cancelled requests
- Backend failures

#### Latency breakdown

- End-to-end request time
- Queue time
- Backend input-processing time
- Model compute time
- Backend output-processing time
- Time to first response for streaming workloads

#### Hardware metrics

- GPU utilization
- GPU memory usage
- GPU power usage
- GPU power limit
- GPU energy consumption
- CPU utilization
- CPU memory usage
- Pinned-memory pool usage

#### Cache metrics

- Cache hits
- Cache misses
- Cache-hit duration
- Cache-miss and insertion duration

Backends can register custom metrics through Triton’s metrics API.

### Tracing

Triton can generate detailed per-request traces.

Trace events include:

- Request reception
- Queueing
- Model execution
- Tensor processing
- Ensemble steps
- Response generation
- Child requests from BLS

Output modes:

- Triton-native trace output
- OpenTelemetry integration

Configuration options:

- Sampling rate
- Trace count
- Trace level
- Output files
- OpenTelemetry exporters/resource settings

### Health and Readiness Endpoints

Provides:

- Server liveness
- Server readiness
- Individual model readiness
- Server metadata
- Model metadata

Supports Kubernetes liveness/readiness probes, load balancers, and autoscaling systems. A server can be live but not ready if required models are still loading/warming up.

### Logging

Runtime-adjustable verbose logging and log format controls.

---

## 17. Response Caching

Triton can cache inference responses to avoid recomputing identical requests.

Cache key includes:

- Model name
- Model version
- Input tensor names
- Shapes and data types
- Input tensor contents

On a cache hit, Triton returns stored output without executing the model.

Cache capabilities:

- Per-server cache configuration
- Per-model cache enablement
- Built-in and pluggable cache implementations
- Cache hit/miss metrics
- Cache lookup/insertion latency metrics
- Custom cache implementations via cache API

Caching is most effective for deterministic models with many duplicate requests. It may add overhead for workloads with mostly unique requests.

---

## 18. Extensibility and Customization

### Custom C/C++ Backends

Implement custom execution engines for:

- Proprietary frameworks
- Custom preprocessing/postprocessing
- Specialized accelerator integration
- Non-neural ML algorithms
- Database/feature-store access
- Custom batching logic
- Decoupled responses
- Backend-specific metrics
- Memory-management optimizations

### Python Backend

Serve arbitrary Python logic without converting to TensorRT/ONNX/TorchScript.

Typical uses:

- Hugging Face models
- NumPy/SciPy processing
- Pre/postprocessing
- Custom business logic
- External library calls
- BLS orchestration
- vLLM integration
- AWS Neuron invocation
- Rapid prototyping

Capabilities:

- Multiple Python model instances
- Custom execution environments
- Request cancellation checks
- Decoupled responses
- Model auto-configuration
- Shared-memory communication with Triton Core

### Repository Agents

Execute logic during model load/unload.

Uses:

- Model decryption
- Authentication/authorization against external systems
- Model conversion
- Artifact validation
- Downloading supplementary files
- Transforming model artifacts
- Cleanup on unload

### Custom Operators

Models containing framework-specific custom operations can be served when required libraries are supplied.

Supported for:

- TensorRT
- PyTorch
- ONNX Runtime

---

## 19. Generative AI and LLM Serving

Generative AI support is primarily provided by **TensorRT-LLM**, **vLLM**, and **Python-based backends**. Features vary by backend.

Depending on backend, capabilities may include:

- Streaming token generation
- Continuous / inflight batching
- Tensor parallelism
- Pipeline parallelism
- Multi-GPU / multi-node execution
- KV-cache management
- Quantized inference (FP8, INT8, INT4)
- LoRA adapters, including multiple adapters
- Top-k / top-p sampling
- Beam search
- Chunked context
- Speculative decoding
- Medusa / EAGLE accelerated decoding
- Constrained decoding
- Function calling
- OpenAI-compatible APIs
- Multimodal model serving
- Embedding and ranking models

These are **not all generic Triton Core features**; they depend on the selected generative backend and its configuration.

---

## 20. Performance Testing and Optimization Tools

### Performance Analyzer (`perf_analyzer`)

Generates inference load and measures:

- Throughput
- Latency
- Concurrency behavior
- Request-rate behavior
- Batch-size effects
- HTTP vs gRPC performance
- Streaming performance
- Shared-memory performance

### Model Analyzer

Automatically searches deployment configurations:

- Model instance count
- Batch size
- Dynamic batching parameters
- GPU placement
- Multi-model configurations

Produces reports showing throughput, latency, memory use, and configuration trade-offs.

### GenAI-Perf

Designed for generative-AI workloads:

- Time to first token
- Inter-token latency
- Token throughput
- End-to-end request latency
- Concurrency behavior
- Embedding/ranking performance
- Visual-language model performance
- Multi-LoRA workloads

### Model Navigator

Automates model export, conversion, backend selection, and optimization for Triton deployment.

---

## 21. Security and Networking Considerations

Triton clients support TLS/SSL and custom headers. However, Triton itself is primarily an inference server, not a complete API security gateway.

For production deployments, authentication, authorization, quotas, and traffic policies are commonly handled using:

- Kubernetes ingress
- API gateways
- Service meshes
- Reverse proxies
- Cloud load balancers
- Custom authorization layers

The Triton Python client plugin API can add headers such as authorization tokens, but the server does not have a built-in authentication mechanism that consumes them.

---

## 22. Deployment and Integration

### Containerization

- Official Docker images on NGC
- Monthly-updated images
- Lightweight and full images with different framework combinations
- Custom images with unused backends removed

### Kubernetes

- Helm charts
- NVIDIA GPU Operator integration
- KServe integration
- Prometheus-driven autoscaling
- MIG (Multi-Instance GPU) support

### Cloud Platforms

Triton is integrated or deployable on:

- Amazon SageMaker
- Azure ML
- Google Vertex AI
- Alibaba Cloud ML platforms

### Edge

Runs on Jetson/JetPack and other ARM64 devices, including embedded deployments.

### MLOps Ecosystem

Works with Seldon, KServe, Kubeflow, and other serving/orchestration frameworks.

---

## 23. Enterprise Features

Available through NVIDIA AI Enterprise:

- Security patching
- CVE management
- API stability guarantees
- Long-term support branches
- Access to NVIDIA experts
- Validated, certified builds
- Predictable release cadence

Useful for regulated or mission-critical deployments.

---

## Summary Table

| Area | Main capabilities |
|---|---|
| Frameworks | TensorRT, TensorRT-LLM, PyTorch, ONNX Runtime, TensorFlow, OpenVINO, Python, vLLM, FIL, DALI |
| Hardware | NVIDIA GPU, x86 CPU, ARM, Jetson, indirect AWS Inferentia |
| Model storage | Local, S3, GCS, Azure, multiple repositories |
| Model management | Versioning, load/unload, repository polling, explicit control |
| Scheduling | Dynamic batching, ragged batching, sequence batching, rate limiter, priorities, concurrency |
| Pipelines | Ensembles, BLS, preprocessing and postprocessing |
| APIs | HTTP/REST, gRPC, streaming gRPC, in-process APIs |
| Data transfer | Binary tensors, system shared memory, CUDA shared memory |
| Reliability | Health checks, readiness, warmup, cancellation, queue controls |
| Observability | Prometheus metrics, statistics, tracing, OpenTelemetry, logging |
| Extensibility | C/C++ backends, Python backend, repository agents, custom operators |
| GenAI | Streaming, inflight batching, LoRA, speculative decoding, OpenAI APIs |
| Optimization tools | Performance Analyzer, Model Analyzer, GenAI-Perf, Model Navigator |

---

## Key Caveats

- **Backend/container-specific:** TensorFlow, PyTorch, OpenVINO, and some generative features may require specific containers or backend builds.
- **Platform-specific:** Not all backends work on all hardware.
- **LLM features:** Many generative AI capabilities come from TensorRT-LLM, vLLM, or Python backends, not from Triton Core.
- **Security:** Triton does not provide built-in authentication/authorization; use a proxy/gateway/service mesh.
- **Naming:** Triton is now part of NVIDIA Dynamo-Triton. For massive multi-node LLM serving, NVIDIA directs users to the broader NVIDIA Dynamo platform.