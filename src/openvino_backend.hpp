// openvino_backend.hpp - OpenVINO multi-device backend (Phase 4).
//
// Wraps ov::CompiledModel across the Intel execution devices:
//   CPU  (OpenVINO CPU plugin)
//   NPU  (OpenVINO NPU plugin, precompiled .npu_blob)
//   gpui (OpenVINO GPU plugin, precompiled .gpu_blob)
//   auto (OpenVINO AUTO plugin; imports available blobs or compiles from IR)
//
// Implements the backend API:
//   load(config, model_path)
//   execute(inputs) -> BackendResult (1:1, synchronous, fault-contained)
//   unload()
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "backend.hpp"
#include "pbtxt.hpp"
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

    // Load a model from an OpenVINO IR directory or a precompiled blob.
    // `model_path` points at the version folder containing model.xml/model.bin
    // and, for NPU/GPU, the corresponding .npu_blob/.gpu_blob. The target device
    // is taken from `config.instance_group.device_kind`. Throws
    // std::runtime_error on failure. Must be called before execute.
    void load(const ModelConfig& config, const std::string& model_path);

    // Run synchronous inference on the configured device and return a
    // fault-contained result.
    BackendResult execute(const std::vector<Tensor>& inputs) override;

    // Release the compiled model and plugin resources.
    void unload();

    bool isLoaded() const { return compiled_ != nullptr; }

    // Number of streams configured on this instance's compiled model.
    size_t streams() const;

    // Resolved device kind (cpu/npu/gpui/auto). Valid after load().
    DeviceKind deviceKind() const { return device_kind_; }

    // Human-readable device identifier for audit logs: "CPU", "NPU",
    // "INTEL_GPU", or "AUTO".
    const std::string& deviceLabel() const { return device_label_; }

    // Attach resource limits used for output-size checks. Called at load.
    void setResourceLimits(const ResourceLimits* limits) { limits_ = limits; }

private:
    DataType mapToDataType(const ov::element::Type& t) const;
    // Map a resolved device kind to an OpenVINO plugin name ("CPU"/"NPU"/"GPU"/
    // "AUTO").
    static std::string ovDeviceName(DeviceKind k);

    std::shared_ptr<ov::Core> core_;
    std::shared_ptr<ov::CompiledModel> compiled_;
    std::shared_ptr<const ModelConfig> config_;
    const ResourceLimits* limits_ = nullptr;
    DeviceKind device_kind_ = DeviceKind::kCpu;
    std::string device_label_ = "CPU";
};

}  // namespace inferlite
