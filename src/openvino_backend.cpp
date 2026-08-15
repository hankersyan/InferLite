#include "openvino_backend.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "openvino/openvino.hpp"
#include "pbtxt.hpp"
#include "validation.hpp"

namespace inferlite {

namespace {
// Map our DataType to an OpenVINO element type.
ov::element::Type toOvType(DataType t) {
    switch (t) {
        case DataType::kInt8: return ov::element::i8;
        case DataType::kUint8: return ov::element::u8;
        case DataType::kInt16: return ov::element::i16;
        case DataType::kUint16: return ov::element::u16;
        case DataType::kInt32: return ov::element::i32;
        case DataType::kUint32: return ov::element::u32;
        case DataType::kInt64: return ov::element::i64;
        case DataType::kUint64: return ov::element::u64;
        case DataType::kFloat16: return ov::element::f16;
        case DataType::kFloat32: return ov::element::f32;
        case DataType::kFloat64: return ov::element::f64;
        case DataType::kBool: return ov::element::boolean;
        default:
            throw std::runtime_error("unsupported data type in toOvType");
    }
}
}  // namespace

OpenVinoBackend::OpenVinoBackend() : core_(std::make_shared<ov::Core>()) {}

OpenVinoBackend::~OpenVinoBackend() {
    unload();
}

namespace {
// Build a path helper: join `dir` with `file`.
std::string joinPath(const std::string& dir, const std::string& file) {
    std::string out = dir;
    if (!out.empty() && out.back() != '/' && out.back() != '\\') out += "/";
    out += file;
    return out;
}
}  // namespace

void OpenVinoBackend::load(const ModelConfig& config, const std::string& model_path) {
    if (core_ == nullptr) core_ = std::make_shared<ov::Core>();

    config_ = std::make_shared<ModelConfig>(config);
    device_kind_ = config.instance_group.device_kind;
    if (device_kind_ == DeviceKind::kInvalid || device_kind_ == DeviceKind::kNvidiaGpu) {
        // NVIDIA GPU (TensorRT) is not an OpenVINO device; default to CPU for a
        // defensive fallback (validation rejects this earlier in practice).
        device_kind_ = DeviceKind::kCpu;
    }
    switch (device_kind_) {
        case DeviceKind::kNpu: device_label_ = "NPU"; break;
        case DeviceKind::kGpuIntel: device_label_ = "INTEL_GPU"; break;
        case DeviceKind::kAuto: device_label_ = "AUTO"; break;
        default: device_label_ = "CPU"; break;
    }

    const std::string ov_device = ovDeviceName(device_kind_);

    ov::AnyMap props;
    // Apply plugin-specific tuning only where the plugin accepts it. CPU accepts
    // INFERENCE_NUM_THREADS and NUM_STREAMS. The GPU/NPU plugins reject the
    // CPU-only INFERENCE_NUM_THREADS and interpret NUM_STREAMS with a streams
    // enum; to keep deterministic one-instance behavior we rely on their
    // defaults rather than passing a raw integer they may reject.
    if (device_kind_ == DeviceKind::kCpu) {
        props["INFERENCE_NUM_THREADS"] = 0;  // let OpenVINO pick based on CPU
        props["NUM_STREAMS"] = 1;            // one instance => single stream
    }

    try {
        // Read a blob file fully into memory and import from an istringstream.
        // This matches how the OpenVINO Python API consumes an in-memory stream
        // and avoids the GPU/NPU plugins misreading a file-backed stream.
        auto readBlob = [](const std::string& path) -> std::string {
            std::ifstream in(path, std::ios::binary);
            if (!in.good()) {
                throw std::runtime_error(std::string("precompiled blob missing: ") + path);
            }
            std::ostringstream ss;
            ss << in.rdbuf();
            return ss.str();
        };

        if (device_kind_ == DeviceKind::kNpu || device_kind_ == DeviceKind::kGpuIntel) {
            // Precompiled blobs are mandatory and verified by the manifest in
            // validated mode. Load via import_model from the in-memory blob.
            const std::string blob_file =
                joinPath(model_path, device_kind_ == DeviceKind::kNpu ? "model.npu_blob"
                                                                      : "model.gpu_blob");
            std::istringstream blob_stream(readBlob(blob_file), std::ios::binary);
            compiled_ = std::make_shared<ov::CompiledModel>(
                core_->import_model(blob_stream, ov_device, props));
        } else if (device_kind_ == DeviceKind::kAuto) {
            // AUTO: prefer importing an existing blob for the best available
            // device; otherwise fall back to compiling from IR on the device
            // OpenVINO selects. In validated mode the manifest mandates the blob.
            bool imported = false;
            for (const char* blob : {"model.npu_blob", "model.gpu_blob", "model.blob"}) {
                const std::string bf = joinPath(model_path, blob);
                if (std::ifstream(bf, std::ios::binary).good()) {
                    std::istringstream blob_stream(readBlob(bf), std::ios::binary);
                    compiled_ = std::make_shared<ov::CompiledModel>(
                        core_->import_model(blob_stream, ov_device, props));
                    imported = true;
                    break;
                }
            }
            if (!imported) {
                const std::string ir_file = joinPath(model_path, "model.xml");
                compiled_ = std::make_shared<ov::CompiledModel>(
                    core_->compile_model(ir_file, ov_device, props));
            }
        } else {
            // CPU: compile from IR on the explicit CPU plugin (no fallback).
            const std::string ir_file = joinPath(model_path, "model.xml");
            compiled_ = std::make_shared<ov::CompiledModel>(
                core_->compile_model(ir_file, ov_device, props));
        }
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("OpenVINO failed to load model '") +
                                 config.name + "' on device '" + ov_device + "': " + e.what());
    }
}

std::string OpenVinoBackend::ovDeviceName(DeviceKind k) {
    switch (k) {
        case DeviceKind::kNpu: return "NPU";
        case DeviceKind::kGpuIntel: return "GPU";
        case DeviceKind::kAuto: return "AUTO";
        default: return "CPU";
    }
}

size_t OpenVinoBackend::streams() const {
    if (!compiled_) return 0;
    return compiled_->get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS").as<size_t>();
}

BackendResult OpenVinoBackend::execute(const std::vector<Tensor>& inputs) {
    BackendResult result;

    if (!compiled_) {
        result.ok = false;
        result.error_code = ErrorCode::kInternalError;
        result.error = "OpenVinoBackend::execute called before load()";
        return result;
    }

    // Input validation against the model spec (shape/type/size).
    size_t max_input = limits_ ? limits_->max_input_size_bytes : (50u * 1024u * 1024u);
    ErrorCode ec = ErrorCode::kNone;
    std::string em;
    if (!validateInputs(*config_, inputs, max_input, ec, em)) {
        result.ok = false;
        result.error_code = ec;
        result.error = em;
        return result;
    }

    try {
        ov::InferRequest req = compiled_->create_infer_request();

        for (const auto& in : inputs) {
            ov::Shape ov_shape(in.shape.begin(), in.shape.end());
            void* host_ptr = const_cast<uint8_t*>(in.data.data());
            ov::Tensor t(toOvType(in.type), ov_shape, host_ptr);
            req.set_tensor(in.name, t);
        }

        req.infer();

        const auto& outs = compiled_->outputs();
        result.outputs.clear();
        result.outputs.reserve(outs.size());
        for (size_t oi = 0; oi < outs.size(); ++oi) {
            const auto& out = outs[oi];
            ov::Tensor t = req.get_tensor(out);
            Tensor r;
            // Name the output per the config's declared output spec (by
            // position) so ensemble mapping and responses are deterministic and
            // match the declared names, regardless of the IR result name.
            r.name = (oi < config_->outputs.size()) ? config_->outputs[oi].name
                                                    : (out.get_names().empty()
                                                           ? out.get_any_name()
                                                           : *out.get_names().begin());
            auto shape = t.get_shape();
            r.shape.assign(shape.begin(), shape.end());
            r.type = mapToDataType(t.get_element_type());
            void* raw = t.data();
            size_t nbytes = tensorByteSize(r.shape, r.type);
            r.data.assign(static_cast<uint8_t*>(raw), static_cast<uint8_t*>(raw) + nbytes);
            result.outputs.push_back(std::move(r));
        }
    } catch (const std::exception& e) {
        result.ok = false;
        result.error_code = ErrorCode::kInternalError;
        result.error = std::string("OpenVINO inference failed on device '") + device_label_ +
                      "': " + e.what();
        return result;
    }

    // Output validation against the model spec (shape, NaN/Inf, range, size).
    size_t max_output = limits_ ? limits_->max_output_size_bytes : (50u * 1024u * 1024u);
    if (!validateOutputs(*config_, result.outputs, max_output, ec, em)) {
        result.ok = false;
        result.error_code = ec;
        result.error = em;
        result.outputs.clear();
        return result;
    }

    result.ok = true;
    result.error_code = ErrorCode::kNone;
    return result;
}

void OpenVinoBackend::unload() {
    compiled_.reset();
    config_.reset();
}

DataType OpenVinoBackend::mapToDataType(const ov::element::Type& t) const {
    if (t == ov::element::i8) return DataType::kInt8;
    if (t == ov::element::u8) return DataType::kUint8;
    if (t == ov::element::i16) return DataType::kInt16;
    if (t == ov::element::u16) return DataType::kUint16;
    if (t == ov::element::i32) return DataType::kInt32;
    if (t == ov::element::u32) return DataType::kUint32;
    if (t == ov::element::i64) return DataType::kInt64;
    if (t == ov::element::u64) return DataType::kUint64;
    if (t == ov::element::f16) return DataType::kFloat16;
    if (t == ov::element::f32) return DataType::kFloat32;
    if (t == ov::element::f64) return DataType::kFloat64;
    if (t == ov::element::boolean) return DataType::kBool;
    return DataType::kInvalid;
}

}  // namespace inferlite
