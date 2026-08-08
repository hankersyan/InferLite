// backend.hpp - Abstract backend interface implemented by OpenVINO, plugin,
// and ensemble executors. The scheduler dispatches requests to any backend
// uniformly (IEC 62304 architectural segregation).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "tensor.hpp"

namespace inferlite {

// Result of one backend execution, carrying a structured error code so the
// scheduler can map failures to the FDA-defined error taxonomy.
struct BackendResult {
    bool ok = true;
    ErrorCode error_code = ErrorCode::kNone;
    std::string error;  // human-readable diagnostic
    std::vector<Tensor> outputs;
};

class IBackend {
public:
    virtual ~IBackend() = default;

    // Run inference/processing on host tensors. Implementations must catch all
    // internal exceptions and return a BackendResult rather than letting an
    // exception escape (fault containment).
    virtual BackendResult execute(const std::vector<Tensor>& inputs) = 0;
};

// A factory/instance wrapper so one model config yields one schedulable unit.
// The scheduler holds a shared IBackend per model.
using BackendPtr = std::shared_ptr<IBackend>;

}  // namespace inferlite
