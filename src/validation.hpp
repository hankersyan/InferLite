// validation.hpp - Input/output validation and deterministic resource limits
// (ISO 14971 risk controls, IEC 62304 §5.3.4). Strict checks reject malformed
// or unsafe requests with a structured error code before any inference runs,
// and validate outputs before a result is returned.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "pbtxt.hpp"
#include "tensor.hpp"

namespace inferlite {

// Deterministic resource limits enforced before and during inference.
struct ResourceLimits {
    size_t max_input_size_bytes = 50u * 1024u * 1024u;   // 50 MB
    size_t max_output_size_bytes = 50u * 1024u * 1024u;  // 50 MB
    int64_t max_inference_time_ms = 5000;                // per-request timer
    size_t max_queue_depth = 100;                        // scheduler queue cap
};

// Check an input tensor against its model spec (name/type/shape) and enforce
// the input byte-size limit. On failure sets `err_code` and returns false.
bool validateInputTensor(const TensorSpec& spec, const Tensor& input,
                         size_t max_input_bytes, ErrorCode& err_code, std::string& err_msg);

// Validate the full set of request inputs against a model config. All declared
// inputs must be present and valid; no extra tensors allowed.
bool validateInputs(const ModelConfig& cfg, const std::vector<Tensor>& inputs,
                    size_t max_input_bytes, ErrorCode& err_code, std::string& err_msg);

// Validate a produced output tensor against the model's output spec and the
// configured output-validation rules (shape, NaN/Inf, range, confidence).
bool validateOutput(const TensorSpec& spec, const OutputValidation& rules,
                    const Tensor& output, ErrorCode& err_code, std::string& err_msg);

// Validate all outputs of a request against the model config.
bool validateOutputs(const ModelConfig& cfg, const std::vector<Tensor>& outputs,
                     size_t max_output_bytes, ErrorCode& err_code, std::string& err_msg);

// Shape-dimension conformance helper (supports -1 dynamic dims in the spec).
bool dimsConform(const std::vector<int64_t>& spec_dims, const std::vector<int64_t>& actual);

// Read the scalar at `elem_idx` of a dense tensor as a double, honoring its data
// type. Returns false if the index/type is invalid.
bool readTensorScalar(const Tensor& t, size_t elem_idx, double& out);

}  // namespace inferlite
