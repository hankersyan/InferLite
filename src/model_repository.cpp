#include "model_repository.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <map>
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
// startup). Supports openvino, tensorrt, plugin, and ensemble backends.
// OpenVINO/plugins/ensembles are CPU-only; TensorRT uses KIND_GPU instances.
void validateConfig(const ModelConfig& cfg, const fs::path& model_dir) {
    if (cfg.backend != "openvino" && cfg.backend != "tensorrt" &&
        cfg.backend != "plugin" && cfg.backend != "ensemble") {
        throw RepositoryError("model '" + cfg.name + "' uses unsupported backend '" +
                              cfg.backend + "' (only 'openvino', 'tensorrt', 'plugin', "
                              "'ensemble' are supported)");
    }
    // max_batch_size follows Triton's convention: 0 disables batching; >0
    // enables Triton-style batching where request tensors carry a leading batch
    // dimension (1 <= B <= max_batch_size) and config dims are per-request.
    // Only a negative value is rejected.
    if (cfg.max_batch_size < 0) {
        throw RepositoryError("model '" + cfg.name +
                              "' has invalid max_batch_size=" +
                              std::to_string(cfg.max_batch_size) +
                              "; must be >= 0");
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

    // OpenVINO backend: the version directory must contain the model artifacts
    // required for the configured device. CPU/AUTO need model.xml (AUTO may use
    // a blob); NPU needs model.npu_blob; Intel GPU needs model.gpu_blob.
    // TensorRT backend: the version directory must contain model.plan.
    fs::path version_dir;
    int64_t version = -1;
    if (!highestVersionDir(model_dir, version_dir, version)) {
        throw RepositoryError("model '" + cfg.name + "' has no numeric version directory");
    }
    if (cfg.backend == "tensorrt") {
        if (!fs::exists(version_dir / "model.plan")) {
            throw RepositoryError("model '" + cfg.name + "' missing model.plan in version dir '" +
                                  version_dir.string() + "'");
        }
    } else if (dk == DeviceKind::kNpu) {
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

std::vector<std::string> listRepositoryModelNames(const std::string& root) {
    std::vector<std::string> names;
    std::error_code ec;
    if (!fs::exists(root) || !fs::is_directory(root, ec)) return names;
    for (const auto& entry : fs::directory_iterator(root, ec)) {
        if (ec) break;
        if (!entry.is_directory()) continue;
        // A folder without config.pbtxt is not a model; skip silently.
        if (!fs::exists(entry.path() / "config.pbtxt")) continue;
        names.push_back(entry.path().filename().string());
    }
    return names;
}

LoadedModel loadModelConfig(const std::string& root, const std::string& model_name,
                            const std::string& override_text) {
    const fs::path model_dir = fs::path(root) / model_name;
    const fs::path config_file = model_dir / "config.pbtxt";
    if (override_text.empty() && !fs::exists(config_file)) {
        throw RepositoryError("model '" + model_name + "' not found (no config.pbtxt under " +
                              model_dir.string() + ")");
    }

    // With an override, the caller-supplied document replaces config.pbtxt; the
    // on-disk version directory and metadata are still authoritative.
    std::string text = override_text.empty() ? readTextFile(config_file) : override_text;
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
    return lm;
}

std::vector<LoadedModel> scanRepository(const std::string& root) {
    std::error_code ec;
    if (!fs::exists(root) || !fs::is_directory(root, ec)) {
        throw RepositoryError("model repository does not exist or is not a directory: " + root);
    }

    std::vector<LoadedModel> models;
    for (const auto& name : listRepositoryModelNames(root)) {
        models.push_back(loadModelConfig(root, name));
    }

    if (models.empty()) {
        throw RepositoryError("no valid models found in repository: " + root);
    }
    return models;
}

std::string highestModelVersionString(const std::string& root, const std::string& model_name) {
    fs::path version_dir;
    int64_t version = -1;
    highestVersionDir(fs::path(root) / model_name, version_dir, version);
    return version < 0 ? std::string() : std::to_string(version);
}

std::string fingerprintModelDirectory(const std::string& root, const std::string& model_name) {
    const fs::path dir = fs::path(root) / model_name;
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return "";

    // Collect (relative path -> "<mtime>:<size>") for every regular file. The
    // map keys sort entries so the fingerprint is deterministic across runs.
    std::map<std::string, std::string> parts;
    fs::recursive_directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec);
    const fs::recursive_directory_iterator end;
    for (; it != end; it.increment(ec)) {
        if (ec) {
            ec.clear();
            continue;
        }
        std::error_code fec;
        if (!it->is_regular_file(fec)) continue;
        std::error_code mec;
        auto ft = it->last_write_time(mec);
        std::string stamp = mec ? std::string("?") : std::to_string(ft.time_since_epoch().count());
        std::error_code sec;
        uintmax_t size = it->file_size(sec);
        parts[fs::relative(it->path(), dir).generic_string()] =
            stamp + ":" + (sec ? std::to_string(0) : std::to_string(size));
    }

    if (parts.empty()) return "";
    std::string combined;
    for (const auto& kv : parts) {
        combined += kv.first;
        combined += '|';
        combined += kv.second;
        combined += '\n';
    }
    return hexEncode(sha256(combined));
}

}  // namespace inferlite
