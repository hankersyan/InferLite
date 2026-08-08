// plugin_backend.hpp - C++ plugin backend (CPU only).
//
// Plugins are shared libraries compiled against plugin_api.hpp and loaded at
// startup. Each plugin exposes a small C ABI factory that returns a PluginNode.
// Plugins run on the CPU thread pool under the same timeout/resource limits as
// OpenVINO models, and are scheduled like model nodes (also inside ensembles).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "tensor.hpp"

namespace inferlite {

struct ModelConfig;
struct TensorSpec;

// A loaded plugin instance bound to one model config. execute() is called from
// a scheduler worker thread with a bounded timeout.
class PluginBackend {
public:
    PluginBackend();
    ~PluginBackend();

    PluginBackend(const PluginBackend&) = delete;
    PluginBackend& operator=(const PluginBackend&) = delete;

    // Load the plugin shared library and create one node instance. `lib_path`
    // is the resolved absolute path to the .so/.dll; `config` carries the
    // model's input/output specs. Throws std::runtime_error on failure.
    void load(const ModelConfig& config, const std::string& lib_path);

    // Run the plugin on host tensors. `inputs` and `outputs` reference host
    // memory. Populates `outputs`. Throws on failure.
    void execute(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs);

    void unload();
    bool isLoaded() const { return handle_ != nullptr; }

private:
    std::vector<TensorSpec> output_specs_;  // declared output specs (for sizing)
    void* handle_ = nullptr;
    void* node_ = nullptr;
};

}  // namespace inferlite
