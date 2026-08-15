// pbtxt.hpp - Minimal parser for the subset of proto-text used in model
// config.pbtxt files. Supports the fields required by Phases 1 and 2.
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

// Execution device kinds managed by the scheduler (Phase 4: multi-device).
// NVIDIA GPU (TensorRT) is unchanged from Phase 3; Intel CPU/NPU/GPU and
// Intel AUTO are OpenVINO device targets.
enum class DeviceKind : int {
    kCpu,        // OpenVINO CPU plugin
    kNvidiaGpu,  // NVIDIA GPU (TensorRT / CUDA)
    kNpu,        // OpenVINO NPU plugin
    kGpuIntel,   // OpenVINO GPU plugin (Intel Arc / iGPU)
    kAuto,       // OpenVINO AUTO device selection (CPU/NPU/GPU)
    kInvalid,
};

// Map a `kind` string from instance_group (KIND_CPU / KIND_NPU /
// KIND_GPU_INTEL / KIND_AUTO / KIND_GPU) to a DeviceKind. Returns kInvalid for
// unknown values. NVIDIA GPU (KIND_GPU) is mapped to kNvidiaGpu for
// completeness though it is not used by the OpenVINO backend.
DeviceKind deviceKindFromString(const std::string& s);
inline const char* deviceKindToString(DeviceKind k) {
    switch (k) {
        case DeviceKind::kCpu: return "cpu";
        case DeviceKind::kNvidiaGpu: return "cuda";
        case DeviceKind::kNpu: return "npu";
        case DeviceKind::kGpuIntel: return "gpui";
        case DeviceKind::kAuto: return "auto";
        default: return "invalid";
    }
}

struct InstanceGroup {
    int count = 1;
    // Triton-style kind (KIND_CPU / KIND_GPU / KIND_NPU / KIND_GPU_INTEL /
    // KIND_AUTO). KIND_GPU maps to NVIDIA TensorRT on device 0; the remaining
    // kinds select the OpenVINO execution device (CPU/NPU/Intel GPU/AUTO).
    // For OpenVINO models this selects the execution device.
    std::string kind = "KIND_CPU";
    // Resolved device kind used by the scheduler/backend. Derived from `kind`.
    // Defaults to CPU for OpenVINO.
    DeviceKind device_kind = DeviceKind::kCpu;
};

// One step of an ensemble (backend: "ensemble").
struct EnsembleStep {
    std::string model_name;                 // reference to another model/plugin/ensemble
    std::vector<std::string> input_map_from;  // tensor names supplied to the step
    std::vector<std::string> input_map_to;    // step's input parameter names
    std::vector<std::string> output_map_from; // step output names
    std::vector<std::string> output_map_to;   // names in the parent scope
};

// FDA model metadata (metadata.json or inline in config.pbtxt).
struct ModelMetadata {
    std::string model_id;
    std::string version;
    std::string intended_use;
    std::string training_dataset_id;
    std::string approval_status;
};

// Output validation controls (ISO 14971 risk controls).
struct OutputValidation {
    bool detect_nan_inf = true;      // reject NaN / Inf outputs
    bool check_range = false;        // enforce min/max
    double min_value = 0.0;
    double max_value = 0.0;
    bool check_shape = true;         // enforce declared output dims
    bool confidence_threshold = false; // optional confidence gate
    double min_confidence = 0.0;
};

// Startup functional self-test: a known golden input and the expected output.
struct GoldenTest {
    bool enabled = false;
    // A single golden input tensor (name -> JSON-encoded number array).
    std::vector<Tensor> input;
    // Optional expected output (name -> numbers) for bit/epsilon comparison.
    std::vector<Tensor> expected_output;
    double epsilon = 0.0;  // 0 => bit-for-bit
};

// Parsed representation of one model's config.pbtxt.
struct ModelConfig {
    std::string name;
    std::string backend;  // "openvino", "plugin", "ensemble"
    int64_t max_batch_size = 0;  // must be 0 (batching disabled)
    std::vector<TensorSpec> inputs;
    std::vector<TensorSpec> outputs;
    InstanceGroup instance_group;

    // --- Phase 2 additions ---
    // Plugin backend: shared library name (e.g. "libpreprocess_plugin.so").
    std::string plugin_library;
    // Ensemble backend: ordered steps that form the DAG.
    std::vector<EnsembleStep> ensemble_steps;
    // FDA model metadata.
    ModelMetadata metadata;
    // Output validation rules.
    OutputValidation output_validation;
    // Startup self-test golden input.
    GoldenTest self_test;
    // Optional per-model overrides of global resource limits (ms / bytes).
    int64_t max_inference_time_ms = 0;  // 0 => use global default
    size_t max_input_size_bytes = 0;    // 0 => use global default
    size_t max_output_size_bytes = 0;   // 0 => use global default
    // Absolute path to this model's repository directory (set by scanRepository).
    std::string model_path;
    // SHA-256 hex of the model files (from manifest), populated at load.
    std::string manifest_hash;
};

// Thrown on parse errors.
struct PbtxtError : public std::runtime_error {
    explicit PbtxtError(const std::string& msg) : std::runtime_error(msg) {}
};

// Parse a config.pbtxt string into a ModelConfig. Throws PbtxtError on failure.
ModelConfig parseConfigPbtxt(const std::string& text);

// Parse a metadata.json string into ModelMetadata. Throws PbtxtError on failure.
ModelMetadata parseMetadataJson(const std::string& text);

}  // namespace inferlite
