#include "validation.hpp"

#include <cmath>
#include <cstring>
#include <set>
#include <sstream>

namespace inferlite {

namespace {
// "[1,4,3]" for diagnostics (declared in the anonymous namespace so it does not
// leak into the public validation API).
std::string shapeToString(const std::vector<int64_t>& shape) {
    std::ostringstream ss;
    ss << '[';
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) ss << ',';
        ss << shape[i];
    }
    ss << ']';
    return ss.str();
}
}  // namespace

bool dimsConform(const std::vector<int64_t>& spec_dims, const std::vector<int64_t>& actual) {
    if (spec_dims.size() != actual.size()) return false;
    for (size_t i = 0; i < spec_dims.size(); ++i) {
        int64_t s = spec_dims[i];
        if (s == -1) continue;  // dynamic dim accepts any
        if (s != actual[i]) return false;
    }
    return true;
}

bool resolveBatch(int64_t max_batch_size, const std::vector<int64_t>& spec_dims,
                  const std::vector<int64_t>& actual, int64_t& batch) {
    if (max_batch_size <= 0) {
        // Batching disabled: no batch dimension. The tensor shape must conform
        // to the spec dims directly (a single request with no leading batch).
        if (!dimsConform(spec_dims, actual)) return false;
        batch = 1;
        return true;
    }
    // Triton batching enabled: the tensor carries a leading batch dimension B
    // (the number of requests combined), and `actual[1:]` is the per-request
    // shape which must conform to the spec dims.
    if (actual.size() != spec_dims.size() + 1) return false;
    batch = actual[0];
    if (batch < 1 || batch > max_batch_size) return false;
    std::vector<int64_t> rest(actual.begin() + 1, actual.end());
    return dimsConform(spec_dims, rest);
}

bool validateInputTensor(const TensorSpec& spec, const Tensor& input, size_t max_input_bytes,
                         int64_t max_batch_size, ErrorCode& err_code, std::string& err_msg) {
    err_code = ErrorCode::kInvalidInput;
    if (input.name != spec.name) {
        err_msg = "input tensor name '" + input.name + "' does not match spec '" +
                  spec.name + "'";
        return false;
    }
    if (input.type != spec.data_type) {
        err_msg = "input '" + input.name + "' data type mismatch: got " +
                  std::string(dataTypeToString(input.type)) + ", expected " +
                  std::string(dataTypeToString(spec.data_type));
        return false;
    }
    int64_t batch = 0;
    if (!resolveBatch(max_batch_size, spec.dims, input.shape, batch)) {
        std::ostringstream ss;
        ss << "input '" << input.name << "' shape mismatch: got [";
        for (size_t i = 0; i < input.shape.size(); ++i) {
            if (i) ss << ',';
            ss << input.shape[i];
        }
        ss << "], spec ";
        if (max_batch_size > 0) {
            // The expected client shape is [B, ...spec.dims] with 1<=B<=max_batch_size.
            ss << "[B, ...[" ;
            for (size_t i = 0; i < spec.dims.size(); ++i) {
                if (i) ss << ',';
                ss << spec.dims[i];
            }
            ss << "]] where 1<=B<=" << max_batch_size;
        } else {
            ss << "[";
            for (size_t i = 0; i < spec.dims.size(); ++i) {
                if (i) ss << ',';
                ss << spec.dims[i];
            }
            ss << "]";
        }
        err_msg = ss.str();
        return false;
    }
    // Deterministic input validation: the payload byte length must exactly match
    // the declared shape and data type (ISO 14971 risk control). A short or
    // oversized payload must never reach the backend -- a truncated buffer could
    // otherwise be read out of bounds inside the runtime.
    const size_t expected_bytes = tensorByteSize(input.shape, input.type);
    if (input.data.size() != expected_bytes) {
        err_msg = "input '" + input.name + "' data length " +
                  std::to_string(input.data.size()) + " bytes does not match shape [" +
                  shapeToString(input.shape) + "] " + dataTypeToString(input.type) +
                  " (expected " + std::to_string(expected_bytes) + " bytes)";
        return false;
    }
    // Enforce deterministic input-size limit.
    if (input.data.size() > max_input_bytes) {
        err_msg = "input '" + input.name + "' size " + std::to_string(input.data.size()) +
                  " bytes exceeds limit " + std::to_string(max_input_bytes);
        return false;
    }
    err_code = ErrorCode::kNone;
    return true;
}

bool validateInputs(const ModelConfig& cfg, const std::vector<Tensor>& inputs,
                    size_t max_input_bytes, ErrorCode& err_code, std::string& err_msg) {
    // Every declared input must be present.
    for (const auto& spec : cfg.inputs) {
        bool found = false;
        for (const auto& in : inputs) {
            if (in.name == spec.name) {
                if (!validateInputTensor(spec, in, max_input_bytes, cfg.max_batch_size,
                                         err_code, err_msg)) {
                    return false;
                }
                found = true;
                break;
            }
        }
        if (!found) {
            err_code = ErrorCode::kInvalidInput;
            err_msg = "missing required input '" + spec.name + "' for model '" + cfg.name + "'";
            return false;
        }
    }
    // No unknown extra inputs.
    for (const auto& in : inputs) {
        bool known = false;
        for (const auto& spec : cfg.inputs) {
            if (in.name == spec.name) { known = true; break; }
        }
        if (!known) {
            err_code = ErrorCode::kInvalidInput;
            err_msg = "unexpected input tensor '" + in.name + "' for model '" + cfg.name + "'";
            return false;
        }
    }
    err_code = ErrorCode::kNone;
    return true;
}

bool validateClientInputs(const ModelConfig& cfg, const std::vector<Tensor>& inputs,
                          size_t max_input_bytes, ErrorCode& err_code,
                          std::string& err_msg) {
    if (!cfg.sequence.enabled) {
        return validateInputs(cfg, inputs, max_input_bytes, err_code, err_msg);
    }
    std::set<std::string> state_in, controls;
    for (const auto& st : cfg.sequence.states) state_in.insert(st.input_name);
    for (const auto& ci : cfg.sequence.control_input) controls.insert(ci.name);

    // Every declared non-state input must be present and valid.
    for (const auto& spec : cfg.inputs) {
        if (state_in.count(spec.name)) continue;  // scheduler-owned state
        bool found = false;
        for (const auto& in : inputs) {
            if (in.name == spec.name) {
                if (!validateInputTensor(spec, in, max_input_bytes, cfg.max_batch_size,
                                         err_code, err_msg)) {
                    return false;
                }
                found = true;
                break;
            }
        }
        if (!found) {
            err_code = ErrorCode::kInvalidInput;
            err_msg = "missing required input '" + spec.name + "' for model '" + cfg.name + "'";
            return false;
        }
    }
    // Reject unknown tensors; accept control tensors (validated below) only.
    for (const auto& in : inputs) {
        if (controls.count(in.name)) continue;
        bool declared = false;
        for (const auto& spec : cfg.inputs) {
            if (in.name == spec.name) { declared = true; break; }
        }
        if (state_in.count(in.name)) {
            err_code = ErrorCode::kInvalidInput;
            err_msg = "client must not send the sequence state tensor '" + in.name +
                      "' for model '" + cfg.name + "' (the scheduler owns it)";
            return false;
        }
        if (!declared) {
            err_code = ErrorCode::kInvalidInput;
            err_msg = "unexpected input tensor '" + in.name + "' for model '" + cfg.name + "'";
            return false;
        }
    }
    // Basic control tensor checks: declared type, at least one scalar, size cap.
    for (const auto& ci : cfg.sequence.control_input) {
        const Tensor* t = nullptr;
        for (const auto& in : inputs) {
            if (in.name == ci.name) { t = &in; break; }
        }
        if (t == nullptr || ci.controls.empty()) {
            err_code = ErrorCode::kInvalidInput;
            err_msg = "missing sequence control input '" + ci.name + "' for model '" +
                      cfg.name + "'";
            return false;
        }
        if (t->type != ci.controls[0].data_type) {
            err_code = ErrorCode::kInvalidInput;
            err_msg = "sequence control input '" + ci.name + "' has datatype " +
                      std::string(dataTypeToString(t->type)) + ", expected " +
                      dataTypeToString(ci.controls[0].data_type);
            return false;
        }
        if (t->data.size() < dataTypeSize(t->type)) {
            err_code = ErrorCode::kInvalidInput;
            err_msg = "sequence control input '" + ci.name + "' is empty";
            return false;
        }
        if (t->data.size() > max_input_bytes) {
            err_code = ErrorCode::kInvalidInput;
            err_msg = "sequence control input '" + ci.name + "' exceeds the input size limit";
            return false;
        }
    }
    err_code = ErrorCode::kNone;
    return true;
}

// Interpret a tensor's raw bytes as a double, honoring its data type.
// Returns false if the value cannot be represented (e.g. NaN via IEEE).
bool readTensorScalar(const Tensor& t, size_t elem_idx, double& out) {
    size_t elem = dataTypeSize(t.type);
    if (elem == 0 || t.data.size() < (elem_idx + 1) * elem) return false;
    const uint8_t* p = t.data.data() + elem_idx * elem;
    switch (t.type) {
        case DataType::kInt8: out = static_cast<int8_t>(p[0]); return true;
        case DataType::kUint8: out = p[0]; return true;
        case DataType::kInt16: {
            int16_t v;
            std::memcpy(&v, p, 2);
            out = v;
            return true;
        }
        case DataType::kUint16: {
            uint16_t v;
            std::memcpy(&v, p, 2);
            out = v;
            return true;
        }
        case DataType::kInt32: {
            int32_t v;
            std::memcpy(&v, p, 4);
            out = v;
            return true;
        }
        case DataType::kUint32: {
            uint32_t v;
            std::memcpy(&v, p, 4);
            out = v;
            return true;
        }
        case DataType::kInt64: {
            int64_t v;
            std::memcpy(&v, p, 8);
            out = static_cast<double>(v);
            return true;
        }
        case DataType::kUint64: {
            uint64_t v;
            std::memcpy(&v, p, 8);
            out = static_cast<double>(v);
            return true;
        }
        case DataType::kFloat16: {
            uint16_t h;
            std::memcpy(&h, p, 2);
            // half -> float
            uint32_t sign = (h >> 15) & 1u;
            uint32_t exp = (h >> 10) & 0x1Fu;
            uint32_t frac = h & 0x3FFu;
            uint32_t fbits;
            if (exp == 0 && frac == 0) {
                fbits = sign << 31;
            } else if (exp == 31) {
                fbits = (sign << 31) | 0x7F800000u | (frac << 13);
            } else if (exp == 0) {
                // subnormal: normalize
                uint32_t e = 127 - 15 + 1;
                while ((frac & 0x400u) == 0) { frac <<= 1; --e; }
                frac &= 0x3FFu;
                fbits = (sign << 31) | (e << 23) | (frac << 13);
            } else {
                fbits = (sign << 31) | ((exp - 15 + 127) << 23) | (frac << 13);
            }
            float f;
            std::memcpy(&f, &fbits, 4);
            out = f;
            return true;
        }
        case DataType::kFloat32: {
            float f;
            std::memcpy(&f, p, 4);
            out = f;
            return true;
        }
        case DataType::kFloat64: {
            double d;
            std::memcpy(&d, p, 8);
            out = d;
            return true;
        }
        case DataType::kBool:
            out = p[0] != 0 ? 1.0 : 0.0;
            return true;
        default:
            return false;
    }
}

bool validateOutput(const TensorSpec& spec, const OutputValidation& rules, const Tensor& output,
                    int64_t max_batch_size, ErrorCode& err_code, std::string& err_msg) {
    err_code = ErrorCode::kOutputValidationFailed;
    // Note: the caller matches outputs to specs by position, so we do not
    // require the tensor name to equal the spec name here.
    if (output.type != spec.data_type) {
        err_msg = "output '" + output.name + "' data type mismatch: got " +
                  std::string(dataTypeToString(output.type)) + ", expected " +
                  std::string(dataTypeToString(spec.data_type));
        return false;
    }
    if (rules.check_shape) {
        int64_t batch = 0;
        if (!resolveBatch(max_batch_size, spec.dims, output.shape, batch)) {
            std::ostringstream ss;
            ss << "output '" << output.name << "' shape does not conform to spec";
            err_msg = ss.str();
            return false;
        }
    }
    // Deterministic output validation: the produced byte length must exactly
    // match the declared shape/type before any element is inspected or returned.
    if (output.data.size() != tensorByteSize(output.shape, output.type)) {
        err_msg = "output '" + output.name + "' data length " +
                  std::to_string(output.data.size()) + " bytes does not match shape [" +
                  shapeToString(output.shape) + "] " + dataTypeToString(output.type) +
                  " (expected " +
                  std::to_string(tensorByteSize(output.shape, output.type)) + " bytes)";
        return false;
    }
    // Scan every element for NaN / Inf and range.
    size_t count = static_cast<size_t>(shapeElementCount(output.shape));
    bool is_float = output.type == DataType::kFloat16 || output.type == DataType::kFloat32 ||
                    output.type == DataType::kFloat64;
    bool has_range = rules.check_range;
    for (size_t i = 0; i < count; ++i) {
        double v = 0.0;
        if (!readTensorScalar(output, i, v)) {
            err_msg = "output '" + output.name + "' contains an unreadable value";
            return false;
        }
        if (is_float && rules.detect_nan_inf && (std::isnan(v) || std::isinf(v))) {
            err_msg = "output '" + output.name + "' contains NaN/Inf at index " +
                      std::to_string(i);
            return false;
        }
        if (has_range && (v < rules.min_value || v > rules.max_value)) {
            err_msg = "output '" + output.name + "' value " + std::to_string(v) +
                      " out of range [" + std::to_string(rules.min_value) + ", " +
                      std::to_string(rules.max_value) + "]";
            return false;
        }
        // Optional confidence gate: the first (or max) element is a probability.
        if (rules.confidence_threshold && i == 0 && v < rules.min_confidence) {
            err_msg = "output '" + output.name + "' confidence " + std::to_string(v) +
                      " below threshold " + std::to_string(rules.min_confidence);
            return false;
        }
    }
    err_code = ErrorCode::kNone;
    return true;
}

bool validateOutputs(const ModelConfig& cfg, const std::vector<Tensor>& outputs,
                     size_t max_output_bytes, ErrorCode& err_code, std::string& err_msg) {
    // Match declared outputs to produced outputs by position (the i-th declared
    // output corresponds to the i-th produced output). Runtime tensor names
    // (e.g. OpenVINO IR result names) may differ from the config names.
    if (outputs.size() < cfg.outputs.size()) {
        err_code = ErrorCode::kOutputValidationFailed;
        err_msg = "model '" + cfg.name + "' produced " + std::to_string(outputs.size()) +
                  " outputs but declares " + std::to_string(cfg.outputs.size());
        return false;
    }
    for (size_t i = 0; i < cfg.outputs.size(); ++i) {
        const Tensor& out = outputs[i];
        if (out.data.size() > max_output_bytes) {
            err_code = ErrorCode::kOutputValidationFailed;
            err_msg = "output '" + out.name + "' size " +
                      std::to_string(out.data.size()) + " bytes exceeds limit " +
                      std::to_string(max_output_bytes);
            return false;
        }
        if (!validateOutput(cfg.outputs[i], cfg.output_validation, out, cfg.max_batch_size,
                            err_code, err_msg)) {
            return false;
        }
    }
    err_code = ErrorCode::kNone;
    return true;
}

}  // namespace inferlite
