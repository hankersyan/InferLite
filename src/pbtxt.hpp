// pbtxt.hpp - Minimal parser for the subset of proto-text used in model
// config.pbtxt files. Supports the fields required by Phase 1.
#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensor.hpp"

namespace inferlite {

// A single input/output tensor definition from config.pbtxt.
struct TensorSpec {
    std::string name;
    DataType data_type = DataType::kInvalid;
    std::vector<int64_t> dims;  // negative value == -1 means a dynamic dim
};

struct InstanceGroup {
    int count = 1;
    // Kind must be KIND_CPU. KIND_GPU is rejected during validation.
    std::string kind = "KIND_CPU";
};

// Parsed representation of one model's config.pbtxt.
struct ModelConfig {
    std::string name;
    std::string backend;        // must be "openvino"
    int64_t max_batch_size = 0; // must be 0 (batching disabled)
    std::vector<TensorSpec> inputs;
    std::vector<TensorSpec> outputs;
    InstanceGroup instance_group;
};

// Thrown on parse errors.
struct PbtxtError : public std::runtime_error {
    explicit PbtxtError(const std::string& msg) : std::runtime_error(msg) {}
};

// Parse a config.pbtxt string into a ModelConfig. Throws PbtxtError on failure.
ModelConfig parseConfigPbtxt(const std::string& text);

}  // namespace inferlite
