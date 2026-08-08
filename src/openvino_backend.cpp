#include "openvino_backend.hpp"

#include <cstddef>
#include <cstdint>
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

void OpenVinoBackend::load(const ModelConfig& config, const std::string& model_path) {
    if (core_ == nullptr) core_ = std::make_shared<ov::Core>();

    config_ = std::make_shared<ModelConfig>(config);

    // Strictly CPU plugin. We pass an explicit device "CPU" so any accidental
    // GPU fallback is impossible.
    ov::AnyMap props;
    props["INFERENCE_NUM_THREADS"] = 0;  // let OpenVINO pick based on CPU
    props["NUM_STREAMS"] = 1;            // one instance => single stream

    // model_path is a directory containing model.xml / model.bin. Build the
    // explicit IR file path for compile_model.
    std::string ir_file = model_path;
    if (!ir_file.empty() && ir_file.back() != '/' && ir_file.back() != '\\') {
        ir_file += "/";
    }
    ir_file += "model.xml";

    try {
        compiled_ = std::make_shared<ov::CompiledModel>(
            core_->compile_model(ir_file, "CPU", props));
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("OpenVINO failed to compile model '") +
                                 config.name + "': " + e.what());
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
        result.error = std::string("OpenVINO inference failed: ") + e.what();
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
