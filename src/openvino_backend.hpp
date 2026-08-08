// openvino_backend.hpp - OpenVINO CPU backend.
//
// Wraps ov::CompiledModel using the CPU plugin. Implements the backend API:
//   load(config, model_path)
//   execute(inputs) -> BackendResult (1:1, synchronous, fault-contained)
//   unload()
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "backend.hpp"
#include "tensor.hpp"

namespace ov {
class Core;
class CompiledModel;
namespace element {
class Type;
}  // namespace element
}  // namespace ov

namespace inferlite {

struct ModelConfig;
struct ResourceLimits;

class OpenVinoBackend : public IBackend {
public:
    OpenVinoBackend();
    ~OpenVinoBackend() override;

    OpenVinoBackend(const OpenVinoBackend&) = delete;
    OpenVinoBackend& operator=(const OpenVinoBackend&) = delete;

    // Load a model from an OpenVINO IR directory. `model_path` points at the
    // folder containing model.xml / model.bin. Throws std::runtime_error on
    // failure. Must be called before execute.
    void load(const ModelConfig& config, const std::string& model_path);

    // Run synchronous CPU inference and return a fault-contained result.
    BackendResult execute(const std::vector<Tensor>& inputs) override;

    // Release the compiled model and plugin resources.
    void unload();

    bool isLoaded() const { return compiled_ != nullptr; }

    // Number of CPU streams configured on this instance's compiled model.
    size_t streams() const;

    // Attach resource limits used for output-size checks. Called at load.
    void setResourceLimits(const ResourceLimits* limits) { limits_ = limits; }

private:
    DataType mapToDataType(const ov::element::Type& t) const;

    std::shared_ptr<ov::Core> core_;
    std::shared_ptr<ov::CompiledModel> compiled_;
    std::shared_ptr<const ModelConfig> config_;
    const ResourceLimits* limits_ = nullptr;
};

}  // namespace inferlite
