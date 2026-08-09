// tensorrt_backend.hpp - TensorRT GPU backend (Phase 3).
//
// Wraps nvinfer1::ICudaEngine and IExecutionContext for a single approved
// model. A model.plan file (serialized TensorRT engine) is deserialized at
// startup. Each backend instance owns a dedicated CUDA stream. execute() copies
// host inputs to device buffers, enqueues inference on the stream, synchronizes
// via a CUDA event, and copies outputs back to host.
//
// GPU support is opt-in (INFERLITE_ENABLE_GPU, set by CMake when a TensorRT SDK
// is present). Without it, the class is a stub that returns a structured error,
// so the CPU-only server builds and runs with no regression.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "backend.hpp"
#include "tensor.hpp"

namespace inferlite {

struct ModelConfig;

#ifdef INFERLITE_ENABLE_GPU

class GpuMemoryManager;

class TensorRtBackend : public IBackend {
public:
    explicit TensorRtBackend(std::shared_ptr<GpuMemoryManager> gpu_memory);
    ~TensorRtBackend() override;

    TensorRtBackend(const TensorRtBackend&) = delete;
    TensorRtBackend& operator=(const TensorRtBackend&) = delete;

    // Deserialize model.plan and create the execution context + stream.
    // `plan_path` is the absolute path to the .plan file. Throws std::runtime_error
    // on any failure (fail-fast startup).
    void load(const ModelConfig& config, const std::string& plan_path,
              size_t max_gpu_memory_mb);

    BackendResult execute(const std::vector<Tensor>& inputs) override;

    void unload();

    // Bytes of GPU device memory currently bound for this instance's I/O.
    size_t boundGpuBytes() const { return bound_gpu_bytes_; }

    // CUDA stream for this instance (used by the GPU-aware ensemble executor).
    void* stream() const { return stream_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    std::shared_ptr<GpuMemoryManager> gpu_memory_;
    std::shared_ptr<const ModelConfig> config_;
    void* stream_ = nullptr;         // cudaStream_t
    size_t bound_gpu_bytes_ = 0;
};

#else  // !INFERLITE_ENABLE_GPU

// Stub used when GPU support is not compiled in.
class TensorRtBackend : public IBackend {
public:
    TensorRtBackend() = default;
    void load(const ModelConfig&, const std::string&, size_t) {
        // GPU support is not compiled into this binary.
    }
    BackendResult execute(const std::vector<Tensor>&) override {
        BackendResult r;
        r.ok = false;
        r.error_code = ErrorCode::kInternalError;
        r.error = "TensorRT backend requested but GPU support is not compiled in "
                  "(rebuild with a TensorRT SDK).";
        return r;
    }
    void unload() {}
    size_t boundGpuBytes() const { return 0; }
    void* stream() const { return nullptr; }
};

#endif  // INFERLITE_ENABLE_GPU

}  // namespace inferlite
