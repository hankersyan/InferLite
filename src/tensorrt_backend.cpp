// tensorrt_backend.cpp - TensorRT GPU backend (Phase 3).
#include "tensorrt_backend.hpp"

#include <fstream>
#include <stdexcept>
#include <vector>

#include "pbtxt.hpp"
#include "validation.hpp"

namespace inferlite {

#ifdef INFERLITE_ENABLE_GPU

#include <cuda_runtime.h>
#include <NvInfer.h>

#include "gpu_memory_manager.hpp"

namespace {

using nvinfer1::Dims;
using nvinfer1::ICudaEngine;
using nvinfer1::IExecutionContext;
using nvinfer1::IRuntime;

// Our DataType -> TensorRT DataType.
nvinfer1::DataType toTrtType(inferlite::DataType t) {
    switch (t) {
        case inferlite::DataType::kInt8: return nvinfer1::DataType::kINT8;
        case inferlite::DataType::kUint8: return nvinfer1::DataType::kUINT8;
        case inferlite::DataType::kInt32: return nvinfer1::DataType::kINT32;
        case inferlite::DataType::kFloat16: return nvinfer1::DataType::kHALF;
        case inferlite::DataType::kFloat32: return nvinfer1::DataType::kFLOAT;
        case inferlite::DataType::kFloat64: return nvinfer1::DataType::kDOUBLE;
        case inferlite::DataType::kBool: return nvinfer1::DataType::kBOOL;
        default: return nvinfer1::DataType::kFLOAT;
    }
}

inferlite::DataType fromTrtType(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kINT8: return inferlite::DataType::kInt8;
        case nvinfer1::DataType::kUINT8: return inferlite::DataType::kUint8;
        case nvinfer1::DataType::kINT32: return inferlite::DataType::kInt32;
        case nvinfer1::DataType::kHALF: return inferlite::DataType::kFloat16;
        case nvinfer1::DataType::kFLOAT: return inferlite::DataType::kFloat32;
        case nvinfer1::DataType::kDOUBLE: return inferlite::DataType::kFloat64;
        case nvinfer1::DataType::kBOOL: return inferlite::DataType::kBool;
        default: return inferlite::DataType::kInvalid;
    }
}

// A no-op logger that suppresses the verbose TensorRT output.
class SilentLogger : public nvinfer1::ILogger {
public:
    void log(Severity, const char*) noexcept override {}
};

std::vector<char> readFileBytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open .plan file: " + path);
    std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<char> buf(static_cast<size_t>(size));
    if (size > 0 && !in.read(buf.data(), size)) {
        throw std::runtime_error("failed to read .plan file: " + path);
    }
    return buf;
}

}  // namespace

// PIMPL keeps the heavy TensorRT types out of the header.
struct TensorRtBackend::Impl {
    std::shared_ptr<IRuntime> runtime;
    std::shared_ptr<ICudaEngine> engine;
    std::shared_ptr<IExecutionContext> context;
    std::vector<std::vector<int64_t>> input_shapes;   // declared input shapes
    std::vector<std::vector<int64_t>> output_shapes;  // computed output shapes
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
};

TensorRtBackend::TensorRtBackend(std::shared_ptr<GpuMemoryManager> gpu_memory)
    : gpu_memory_(std::move(gpu_memory)) {}

TensorRtBackend::~TensorRtBackend() {
    unload();
}

void TensorRtBackend::load(const ModelConfig& config, const std::string& plan_path,
                           size_t max_gpu_memory_mb) {
    if (!gpu_memory_) {
        throw std::runtime_error("TensorRtBackend::load: no GPU memory manager");
    }
    config_ = std::make_shared<ModelConfig>(config);

    impl_ = std::make_unique<Impl>();

    // Set the device context to GPU 0 (single-GPU only, per PRD).
    cudaError_t cerr = cudaSetDevice(0);
    if (cerr != cudaSuccess) {
        throw std::runtime_error(std::string("TensorRT: cudaSetDevice(0) failed: ") +
                                 cudaGetErrorString(cerr));
    }

    try {
        static SilentLogger logger;
        impl_->runtime = std::shared_ptr<IRuntime>(
            nvinfer1::createInferRuntime(logger), [](IRuntime* r) {
                if (r) r->destroy();
            });
        if (!impl_->runtime) {
            throw std::runtime_error("TensorRT: failed to create runtime");
        }

        std::vector<char> blob = readFileBytes(plan_path);
        impl_->engine = std::shared_ptr<ICudaEngine>(
            impl_->runtime->deserializeCudaEngine(blob.data(), blob.size()),
            [](ICudaEngine* e) {
                if (e) e->destroy();
            });
        if (!impl_->engine) {
            throw std::runtime_error("TensorRT: failed to deserialize engine from " +
                                     plan_path);
        }

        impl_->context = std::shared_ptr<IExecutionContext>(
            impl_->engine->createExecutionContext(),
            [](IExecutionContext* c) {
                if (c) c->destroy();
            });
        if (!impl_->context) {
            throw std::runtime_error("TensorRT: failed to create execution context");
        }

        // Record input/output binding names in a deterministic order. The config
        // declares input/output specs; we match bindings to them by name.
        int nb = impl_->engine->getNbBindings();
        impl_->input_names.clear();
        impl_->output_names.clear();
        for (int i = 0; i < nb; ++i) {
            const char* name = impl_->engine->getBindingName(i);
            if (impl_->engine->bindingIsInput(i)) {
                impl_->input_names.push_back(name);
            } else {
                impl_->output_names.push_back(name);
            }
        }

        // Create the per-instance CUDA stream.
        cerr = cudaStreamCreate(reinterpret_cast<cudaStream_t*>(&stream_));
        if (cerr != cudaSuccess) {
            throw std::runtime_error(std::string("TensorRT: cudaStreamCreate failed: ") +
                                     cudaGetErrorString(cerr));
        }
    } catch (const std::exception&) {
        unload();
        throw;
    }
}

void TensorRtBackend::unload() {
    if (stream_) {
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_));
        cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream_));
        stream_ = nullptr;
    }
    impl_.reset();
    config_.reset();
    bound_gpu_bytes_ = 0;
}

BackendResult TensorRtBackend::execute(const std::vector<Tensor>& inputs) {
    BackendResult result;
    result.device = "GPU";

    if (!impl_ || !impl_->engine || !impl_->context || !stream_) {
        result.ok = false;
        result.error_code = ErrorCode::kInternalError;
        result.error = "TensorRtBackend::execute called before load()";
        return result;
    }

    // Input validation (CPU-side, before any GPU copy) - ISO 14971 control.
    size_t max_input = 50u * 1024u * 1024u;
    ErrorCode ec = ErrorCode::kNone;
    std::string em;
    if (!validateInputs(*config_, inputs, max_input, ec, em)) {
        result.ok = false;
        result.error_code = ec;
        result.error = em;
        return result;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);
    cudaEvent_t done = nullptr;
    try {
        if (cudaEventCreateWithFlags(&done, cudaEventDisableTiming) != cudaSuccess) {
            throw std::runtime_error("cudaEventCreate failed");
        }

        // Bind device buffers. Bind the input bindings from the host inputs and
        // allocate device memory for the outputs (sized from the config specs).
        const int nb = impl_->engine->getNbBindings();
        std::vector<void*> bindings(static_cast<size_t>(nb), nullptr);
        size_t gpu_bytes = 0;

        for (int i = 0; i < nb; ++i) {
            const char* name = impl_->engine->getBindingName(i);
            bool is_input = impl_->engine->bindingIsInput(i);

            if (is_input) {
                // Find the matching host input by binding name.
                const Tensor* host_in = nullptr;
                for (const auto& t : inputs) {
                    if (t.name == name) { host_in = &t; break; }
                }
                if (!host_in) {
                    result.ok = false;
                    result.error_code = ErrorCode::kInvalidInput;
                    result.error = "TensorRT model missing input binding '" +
                                   std::string(name) + "'";
                    return result;
                }
                DeviceBuffer dev = gpu_memory_->acquireDevice(host_in->data.size());
                gpu_bytes += dev.capacity();
                cudaMemcpyAsync(dev.data(), host_in->data.data(), host_in->data.size(),
                                cudaMemcpyHostToDevice, stream);
                bindings[static_cast<size_t>(i)] = dev.data();
            } else {
                // Output: size from the config's declared output spec (by name).
                const TensorSpec* spec = nullptr;
                for (const auto& o : config_->outputs) {
                    if (o.name == name) { spec = &o; break; }
                }
                size_t nbytes = 0;
                if (spec) {
                    nbytes = tensorByteSize(spec->dims, spec->data_type);
                } else {
                    nbytes = 1u * 1024u * 1024u;  // fallback bound
                }
                DeviceBuffer dev = gpu_memory_->acquireDevice(nbytes);
                gpu_bytes += dev.capacity();
                bindings[static_cast<size_t>(i)] = dev.data();
            }
        }

        bound_gpu_bytes_ = gpu_bytes;
        result.gpu_memory_bytes = gpu_bytes;

        // Enqueue inference and record a completion event. The caller (ensemble
        // executor or scheduler) synchronizes via the event / stream.
        const void* const* raw_bindings = bindings.data();
        if (!impl_->context->enqueueV2(raw_bindings, stream, nullptr)) {
            result.ok = false;
            result.error_code = ErrorCode::kInternalError;
            result.error = "TensorRT enqueueV2 failed";
            return result;
        }
        if (cudaEventRecord(done, stream) != cudaSuccess) {
            throw std::runtime_error("cudaEventRecord failed");
        }
        if (cudaEventSynchronize(done) != cudaSuccess) {
            throw std::runtime_error("cudaEventSynchronize failed");
        }

        // Copy outputs back to host and build result tensors.
        result.outputs.clear();
        for (int i = 0; i < nb; ++i) {
            const char* name = impl_->engine->getBindingName(i);
            if (impl_->engine->bindingIsInput(i)) continue;

            TensorSpec spec;
            spec.name = name;
            spec.data_type = fromTrtType(impl_->engine->getBindingDataType(i));
            Dims d = impl_->engine->getBindingDimensions(i);
            spec.dims.assign(d.d, d.d + d.nbDims);

            size_t nbytes = tensorByteSize(spec.dims, spec.data_type);
            if (nbytes == 0) nbytes = 1u * 1024u * 1024u;

            // Copy from the output device binding back to host.
            Tensor out;
            out.name = spec.name;
            out.type = spec.data_type;
            out.shape = spec.dims;
            out.data.resize(nbytes);
            cudaMemcpyAsync(out.data.data(), bindings[static_cast<size_t>(i)], nbytes,
                            cudaMemcpyDeviceToHost, stream);
            result.outputs.push_back(std::move(out));
        }

        // Synchronize the copy stream so host data is complete before validation.
        cudaStreamSynchronize(stream);

        // Output validation (ISO 14971) applied after GPU->host copy.
        size_t max_output = 50u * 1024u * 1024u;
        if (!validateOutputs(*config_, result.outputs, max_output, ec, em)) {
            result.ok = false;
            result.error_code = ec;
            result.error = em;
            result.outputs.clear();
            return result;
        }

        result.ok = true;
        result.error_code = ErrorCode::kNone;
    } catch (const std::exception& e) {
        // A CUDA/TensorRT failure indicates possible device corruption; the
        // offending instance must be quarantined (fault isolation).
        result.ok = false;
        result.error_code = ErrorCode::kInternalError;
        result.error = std::string("TensorRT inference failed: ") + e.what();
        result.quarantine = true;
    }

    if (done) cudaEventDestroy(done);
    return result;
}

#else  // !INFERLITE_ENABLE_GPU

// GPU support disabled: stub implementation in the header is used.

#endif  // INFERLITE_ENABLE_GPU

}  // namespace inferlite
