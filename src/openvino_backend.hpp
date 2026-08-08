// openvino_backend.hpp - OpenVINO CPU backend.
//
// Wraps ov::CompiledModel using the CPU plugin. Implements the backend API:
//   load(config, model_path)
//   execute(inputs, outputs)   (1:1, synchronous)
//   unload()
#pragma once

#include <memory>
#include <string>
#include <vector>

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

class OpenVinoBackend {
public:
    OpenVinoBackend();
    ~OpenVinoBackend();

    OpenVinoBackend(const OpenVinoBackend&) = delete;
    OpenVinoBackend& operator=(const OpenVinoBackend&) = delete;

    // Load a model from an OpenVINO IR directory. `model_path` points at the
    // folder containing model.xml / model.bin. Throws std::runtime_error on
    // failure. Must be called before execute.
    void load(const ModelConfig& config, const std::string& model_path);

    // Run synchronous CPU inference. `inputs` must match the model's expected
    // input tensors (name/type/shape). Populates `outputs`. This is a 1:1
    // request->response path; no batching.
    void execute(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs);

    // Release the compiled model and plugin resources.
    void unload();

    bool isLoaded() const { return compiled_ != nullptr; }

    // Number of CPU streams configured on this instance's compiled model.
    size_t streams() const;

private:
    DataType mapToDataType(const ov::element::Type& t) const;

    std::shared_ptr<ov::Core> core_;
    std::shared_ptr<ov::CompiledModel> compiled_;
};

}  // namespace inferlite
