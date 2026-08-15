#include "model_repository.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "sha256.hpp"

namespace inferlite {

namespace fs = std::filesystem;

namespace {

bool isAllDigits(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s) {
        if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    }
    return true;
}

std::string readTextFile(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) {
        throw RepositoryError("cannot open file: " + p.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Find the highest numeric subdirectory (the "version" directory). Ignores
// non-numeric names. Returns false if none found.
bool highestVersionDir(const fs::path& model_dir, fs::path& out, int64_t& version) {
    int64_t best = -1;
    bool found = false;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(model_dir, ec)) {
        if (ec) break;
        if (!entry.is_directory()) continue;
        std::string name = entry.path().filename().string();
        if (!isAllDigits(name)) continue;
        int64_t v = std::stoll(name);
        if (v > best) {
            best = v;
            out = entry.path();
            found = true;
        }
    }
    version = best;
    return found;
}

// Validate a config. Throws RepositoryError on any violation (fail-fast
// startup). Supports openvino, plugin, and ensemble backends (all CPU-only).
void validateConfig(const ModelConfig& cfg, const fs::path& model_dir) {
    if (cfg.backend != "openvino" && cfg.backend != "plugin" && cfg.backend != "ensemble") {
        throw RepositoryError("model '" + cfg.name + "' uses unsupported backend '" +
                              cfg.backend +
                              "' (only 'openvino', 'plugin', 'ensemble' are supported)");
    }
    if (cfg.max_batch_size != 0) {
        throw RepositoryError("model '" + cfg.name +
                              "' has max_batch_size=" + std::to_string(cfg.max_batch_size) +
                              "; batching must be disabled (max_batch_size=0)");
    }
    if (cfg.instance_group.count <= 0) {
        throw RepositoryError("model '" + cfg.name + "' has invalid instance_group count " +
                              std::to_string(cfg.instance_group.count));
    }

    // Phase 4: validate the resolved device kind. For OpenVINO models the
    // accepted kinds are KIND_CPU / KIND_NPU / KIND_GPU_INTEL / KIND_AUTO.
    // NVIDIA GPU (KIND_GPU) is handled by a separate backend and not OpenVINO.
    const DeviceKind dk = cfg.instance_group.device_kind;
    if (dk == DeviceKind::kInvalid) {
        throw RepositoryError("model '" + cfg.name + "' has invalid instance_group kind '" +
                              cfg.instance_group.kind +
                              "'; expected KIND_CPU|KIND_NPU|KIND_GPU_INTEL|KIND_AUTO");
    }
    // KIND_GPU (NVIDIA) is not an OpenVINO target; it is only valid for the
    // TensorRT backend which is outside Phase 4 scope.
    if (dk == DeviceKind::kNvidiaGpu && cfg.backend == "openvino") {
        throw RepositoryError("model '" + cfg.name +
                              "' requests NVIDIA GPU (KIND_GPU) with the "
                              "'openvino' backend; use the 'tensorrt' backend instead");
    }

    if (cfg.backend == "plugin") {
        if (cfg.plugin_library.empty()) {
            throw RepositoryError("plugin model '" + cfg.name +
                                  "' must set 'plugin_library'");
        }
        if (cfg.inputs.empty() || cfg.outputs.empty()) {
            throw RepositoryError("plugin model '" + cfg.name +
                                  "' must declare inputs and outputs");
        }
        return;
    }

    if (cfg.backend == "ensemble") {
        if (cfg.ensemble_steps.empty()) {
            throw RepositoryError("ensemble model '" + cfg.name +
                                  "' must define 'ensemble_scheduling' steps");
        }
        return;
    }

    // openvino backend: the version directory must contain the model artifacts
    // required for the configured device. CPU/AUTO need model.xml (AUTO may use
    // a blob); NPU needs model.npu_blob; Intel GPU needs model.gpu_blob.
    fs::path version_dir;
    int64_t version = -1;
    if (!highestVersionDir(model_dir, version_dir, version)) {
        throw RepositoryError("model '" + cfg.name + "' has no numeric version directory");
    }
    if (dk == DeviceKind::kNpu) {
        if (!fs::exists(version_dir / "model.npu_blob")) {
            throw RepositoryError("NPU model '" + cfg.name +
                                  "' missing model.npu_blob in version dir '" +
                                  version_dir.string() + "'");
        }
    } else if (dk == DeviceKind::kGpuIntel) {
        if (!fs::exists(version_dir / "model.gpu_blob")) {
            throw RepositoryError("Intel GPU model '" + cfg.name +
                                  "' missing model.gpu_blob in version dir '" +
                                  version_dir.string() + "'");
        }
    } else {
        // CPU / AUTO: require model.xml (compile-from-IR fallback).
        if (!fs::exists(version_dir / "model.xml")) {
            throw RepositoryError("model '" + cfg.name + "' missing model.xml in version dir '" +
                                  version_dir.string() + "'");
        }
    }
}

}  // namespace

std::vector<LoadedModel> scanRepository(const std::string& root) {
    std::error_code ec;
    if (!fs::exists(root) || !fs::is_directory(root, ec)) {
        throw RepositoryError("model repository does not exist or is not a directory: " + root);
    }

    std::vector<LoadedModel> models;
    for (const auto& entry : fs::directory_iterator(root, ec)) {
        if (ec) break;
        if (!entry.is_directory()) continue;

        const fs::path model_dir = entry.path();
        const std::string model_name = model_dir.filename().string();
        const fs::path config_file = model_dir / "config.pbtxt";
        if (!fs::exists(config_file)) {
            // A folder without config.pbtxt is not a model; skip silently.
            continue;
        }

        std::string text = readTextFile(config_file);
        ModelConfig cfg = parseConfigPbtxt(text);

        // Model name from config must match the directory name.
        if (cfg.name != model_name) {
            throw RepositoryError("config name '" + cfg.name + "' does not match directory '" +
                                  model_name + "'");
        }

        validateConfig(cfg, model_dir);

        fs::path version_dir;
        int64_t version = -1;
        highestVersionDir(model_dir, version_dir, version);

        // Load optional metadata.json (FDA model metadata) if present.
        fs::path meta_file = model_dir / "metadata.json";
        if (fs::exists(meta_file)) {
            cfg.metadata = parseMetadataJson(readTextFile(meta_file));
        }

        // Store the model's repository directory path.
        cfg.model_path = model_dir.string();

        LoadedModel lm;
        lm.config = std::make_shared<ModelConfig>(std::move(cfg));
        lm.version_path = version_dir.string();
        lm.version = version;
        lm.config_hash = hexEncode(sha256(text));
        models.push_back(std::move(lm));
    }

    if (models.empty()) {
        throw RepositoryError("no valid models found in repository: " + root);
    }
    return models;
}

}  // namespace inferlite
