#include "model_repository.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
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

// All numeric version subdirectories of a model, newest first. Ignores
// non-numeric directory names.
std::vector<std::pair<int64_t, fs::path>> numericVersionDirs(const fs::path& model_dir) {
    std::vector<std::pair<int64_t, fs::path>> out;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(model_dir, ec)) {
        if (ec) break;
        if (!entry.is_directory()) continue;
        const std::string name = entry.path().filename().string();
        if (!isAllDigits(name)) continue;
        out.emplace_back(std::stoll(name), entry.path());
    }
    std::sort(out.begin(), out.end(),
              [](const std::pair<int64_t, fs::path>& a,
                 const std::pair<int64_t, fs::path>& b) { return a.first > b.first; });
    return out;
}

// Resolve the single version directory a model loads, honoring the model's
// Triton `version_policy` (see pbtxt.hpp). InferLite serves one version per
// model name at a time (documented deviation from Triton, which can keep
// several versions ready):
//   latest / all (and the implicit default) -> the highest numeric version
//     present. For `latest`, Triton's num_versions > 1 would keep N versions
//     ready; only the newest eligible one is loaded here.
//   specific -> the highest listed version that exists on disk. When none of
//     the listed versions is present, false is returned so loading fails fast
//     and a stale pin is never silently ignored (the operator must update the
//     config or the repository).
// Returns false when no eligible version directory exists.
bool resolveVersionDir(const ModelConfig& cfg, const fs::path& model_dir, fs::path& out,
                       int64_t& version) {
    const auto dirs = numericVersionDirs(model_dir);
    const VersionPolicy& vp = cfg.version_policy;
    if (vp.kind == VersionPolicyKind::kSpecific) {
        for (const auto& dv : dirs) {
            if (std::find(vp.versions.begin(), vp.versions.end(), dv.first) !=
                vp.versions.end()) {
                out = dv.second;
                version = dv.first;
                return true;
            }
        }
        return false;
    }
    // latest / all (or an absent block): the highest numeric version.
    if (dirs.empty()) return false;
    out = dirs.front().second;
    version = dirs.front().first;
    return true;
}

// Comma-joined decimal list helper for diagnostics.
std::string joinNumbers(const std::vector<int64_t>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(v[i]);
    }
    return s;
}

// Validate the Triton version_policy block of a parsed config: syntax and
// Triton constraints, independent of which versions exist on disk (existence is
// checked against the resolved directory below). Throws RepositoryError on any
// violation (fail-fast startup).
void validateVersionPolicy(const ModelConfig& cfg) {
    const VersionPolicy& vp = cfg.version_policy;
    if (!vp.configured) return;  // implicit Triton default: latest { num_versions: 1 }
    switch (vp.kind) {
        case VersionPolicyKind::kLatest:
            if (vp.num_versions < 1) {
                throw RepositoryError("model '" + cfg.name +
                                      "' has version_policy.latest num_versions=" +
                                      std::to_string(vp.num_versions) +
                                      "; Triton requires num_versions >= 1");
            }
            break;
        case VersionPolicyKind::kSpecific: {
            if (vp.versions.empty()) {
                throw RepositoryError("model '" + cfg.name +
                                      "' has version_policy.specific without any 'versions'");
            }
            std::set<int64_t> seen;
            for (int64_t v : vp.versions) {
                if (v <= 0) {
                    throw RepositoryError("model '" + cfg.name +
                                          "' has version_policy.specific version " +
                                          std::to_string(v) +
                                          "; versions must be positive integers");
                }
                if (!seen.insert(v).second) {
                    throw RepositoryError("model '" + cfg.name +
                                          "' lists version " + std::to_string(v) +
                                          " more than once in version_policy.specific");
                }
            }
            break;
        }
        case VersionPolicyKind::kAll:
            break;
        default:
            throw RepositoryError("model '" + cfg.name +
                                  "' has an invalid version_policy (internal error)");
    }
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

    // Triton version_policy constraints apply to every backend (even
    // plugin/ensemble configs, whose version directories are not artifacts).
    validateVersionPolicy(cfg);
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

    // Triton model_warmup: sample requests executed through the real scheduler
    // at load time (before the model is marked ready). Constraints validated
    // here so a bad config fails fast:
    //   - not supported together with sequence_batching (a warmup cannot open a
    //     sequence slot or drive the sequence control tensors)
    //   - every warmup input must name a declared model input, use the model
    //     input's data type, and resolve to concrete (non-dynamic) dims
    //   - warmups only support zero-filled payloads (zero_data: true);
    //     Triton's input_data_file is not implemented
    //   - batch_size must fit the model's max_batch_size when batching is on,
    //     and must be 0/1 otherwise
    if (!cfg.warmups.empty()) {
        if (cfg.sequence.enabled) {
            throw RepositoryError("model '" + cfg.name +
                                  "' configures 'model_warmup' with 'sequence_batching'; "
                                  "warmup of sequence models is not supported (control "
                                  "tensors cannot be driven from zero-filled warmup inputs)");
        }
        std::map<std::string, const TensorSpec*> spec_by_name;
        for (const auto& in : cfg.inputs) spec_by_name[in.name] = &in;
        for (const auto& wu : cfg.warmups) {
            if (wu.name.empty()) {
                throw RepositoryError("model '" + cfg.name +
                                      "' has a model_warmup entry without a 'name'");
            }
            if (wu.batch_size < 0) {
                throw RepositoryError("model '" + cfg.name + "' model_warmup '" + wu.name +
                                      "' has negative batch_size=" +
                                      std::to_string(wu.batch_size));
            }
            if (wu.batch_size > 1 && cfg.max_batch_size <= 0) {
                throw RepositoryError("model '" + cfg.name + "' model_warmup '" + wu.name +
                                      "' sets batch_size=" + std::to_string(wu.batch_size) +
                                      " but the model does not batch (max_batch_size=0; "
                                      "warmup batch_size must be 0 or 1)");
            }
            if (cfg.max_batch_size > 0 && wu.batch_size > cfg.max_batch_size) {
                throw RepositoryError("model '" + cfg.name + "' model_warmup '" + wu.name +
                                      "' has batch_size=" + std::to_string(wu.batch_size) +
                                      " greater than max_batch_size=" +
                                      std::to_string(cfg.max_batch_size));
            }
            if (wu.inputs.empty()) {
                throw RepositoryError("model '" + cfg.name + "' model_warmup '" + wu.name +
                                      "' declares no inputs");
            }
            for (const auto& wi : wu.inputs) {
                auto it = spec_by_name.find(wi.name);
                if (it == spec_by_name.end()) {
                    throw RepositoryError("model '" + cfg.name + "' model_warmup '" + wu.name +
                                          "' references unknown input '" + wi.name + "'");
                }
                if (wi.has_type && wi.data_type != it->second->data_type) {
                    throw RepositoryError("model '" + cfg.name + "' model_warmup '" + wu.name +
                                          "' input '" + wi.name + "' declares data_type " +
                                          dataTypeToString(wi.data_type) + " which does not "
                                          "match the model input type " +
                                          dataTypeToString(it->second->data_type));
                }
                if (!wi.zero_data) {
                    throw RepositoryError("model '" + cfg.name + "' model_warmup '" + wu.name +
                                          "' input '" + wi.name +
                                          "' must set zero_data: true (InferLite warmup "
                                          "fills inputs with zeros; Triton input_data_file "
                                          "is not implemented)");
                }
                if (wi.has_shape) {
                    for (int64_t d : wi.shape) {
                        if (d <= 0) {
                            throw RepositoryError("model '" + cfg.name + "' model_warmup '" +
                                                  wu.name + "' input '" + wi.name +
                                                  "' has non-positive shape entry " +
                                                  std::to_string(d));
                        }
                    }
                } else {
                    if (!wi.has_dims) {
                        bool dynamic = false;
                        for (int64_t d : it->second->dims) {
                            if (d <= 0) dynamic = true;
                        }
                        if (dynamic) {
                            throw RepositoryError("model '" + cfg.name + "' model_warmup '" +
                                                  wu.name + "' input '" + wi.name +
                                                  "' inherits dynamic dims from the model "
                                                  "config; warmup must declare explicit "
                                                  "'dims'");
                        }
                    }
                    const std::vector<int64_t>& dims =
                        wi.has_dims ? wi.dims : it->second->dims;
                    for (int64_t d : dims) {
                        if (d <= 0) {
                            throw RepositoryError("model '" + cfg.name + "' model_warmup '" +
                                                  wu.name + "' input '" + wi.name +
                                                  "' has non-positive dims entry " +
                                                  std::to_string(d));
                        }
                    }
                }
            }
        }
    }

    // Phase 7 (batching mode): Triton dynamic-batching policy constraints.
    // Mirror NVIDIA Triton's validation of ModelConfig.dynamic_batching:
    //   - requires max_batch_size > 0 (batching must be enabled)
    //   - preferred_batch_size entries must be in [1, max_batch_size]
    //   - max_queue_delay_microseconds must be >= 0
    // InferLite additionally restricts the batch scheduler to OpenVINO models
    // running on a CPU (or AUTO) instance because cross-request batch merging
    // needs a model that accepts a dynamic batch dimension at runtime
    // (precompiled NPU/GPU blobs and TensorRT plans are shape-locked here).
    if (cfg.batching.enabled) {
        if (cfg.max_batch_size <= 0) {
            throw RepositoryError("model '" + cfg.name +
                                  "' enables 'dynamic_batching' but max_batch_size=" +
                                  std::to_string(cfg.max_batch_size) +
                                  "; Triton requires max_batch_size > 0 to batch requests");
        }
        if (cfg.backend != "openvino") {
            throw RepositoryError("model '" + cfg.name +
                                  "' enables 'dynamic_batching' with backend '" +
                                  cfg.backend +
                                  "'; only the 'openvino' backend supports dynamic batching");
        }
        const DeviceKind dk = cfg.instance_group.device_kind;
        if (dk != DeviceKind::kCpu && dk != DeviceKind::kAuto) {
            throw RepositoryError("model '" + cfg.name +
                                  "' enables 'dynamic_batching' with instance kind '" +
                                  cfg.instance_group.kind +
                                  "'; only KIND_CPU / KIND_AUTO instances support it");
        }
        for (int64_t b : cfg.batching.preferred_batch_size) {
            if (b <= 0 || b > cfg.max_batch_size) {
                throw RepositoryError("model '" + cfg.name +
                                      "' has preferred_batch_size=" + std::to_string(b) +
                                      " outside [1, max_batch_size=" +
                                      std::to_string(cfg.max_batch_size) + "]");
            }
        }
        if (cfg.batching.max_queue_delay_us < 0) {
            throw RepositoryError("model '" + cfg.name +
                                  "' has negative max_queue_delay_microseconds=" +
                                  std::to_string(cfg.batching.max_queue_delay_us));
        }
        // Priority scheduling (mirrors Triton): priority levels start at 1 with
        // 1 the highest priority; default_priority_level must be in
        // [1, priority_levels] when priorities are enabled.
        if (cfg.batching.priority_levels < 0) {
            throw RepositoryError("model '" + cfg.name + "' has negative priority_levels=" +
                                  std::to_string(cfg.batching.priority_levels));
        }
        if (cfg.batching.priority_levels > 0) {
            if (cfg.batching.default_priority_level < 1 ||
                cfg.batching.default_priority_level > cfg.batching.priority_levels) {
                throw RepositoryError("model '" + cfg.name +
                                      "' has default_priority_level=" +
                                      std::to_string(cfg.batching.default_priority_level) +
                                      " outside [1, priority_levels=" +
                                      std::to_string(cfg.batching.priority_levels) + "]");
            }
        } else if (cfg.batching.default_priority_level > 1) {
            // A default without any priority level is only meaningful when
            // priority scheduling is configured.
            throw RepositoryError("model '" + cfg.name +
                                  "' sets default_priority_level=" +
                                  std::to_string(cfg.batching.default_priority_level) +
                                  " but priority_levels is not > 0");
        }
    }

    // Triton sequence batching (stateful models). Mutually exclusive with
    // dynamic batching; a sequence model must run on a stable backend with one
    // instance, no client-visible batch dimension, and its hidden state tensors
    // declared in the model I/O.
    if (cfg.sequence.enabled) {
        if (cfg.batching.enabled) {
            throw RepositoryError("model '" + cfg.name +
                                  "' configures both 'dynamic_batching' and "
                                  "'sequence_batching'; Triton allows one scheduler per model");
        }
        if (cfg.max_batch_size != 0) {
            throw RepositoryError("model '" + cfg.name +
                                  "' enables 'sequence_batching' but max_batch_size=" +
                                  std::to_string(cfg.max_batch_size) +
                                  " is not 0; sequence batching requires tensors without a "
                                  "client-visible batch dimension");
        }
        if (cfg.backend != "openvino") {
            throw RepositoryError("model '" + cfg.name +
                                  "' enables 'sequence_batching' with backend '" +
                                  cfg.backend + "'; only the 'openvino' backend supports it");
        }
        const DeviceKind sdk = cfg.instance_group.device_kind;
        if (sdk != DeviceKind::kCpu && sdk != DeviceKind::kAuto) {
            throw RepositoryError("model '" + cfg.name +
                                  "' enables 'sequence_batching' with instance kind '" +
                                  cfg.instance_group.kind +
                                  "'; only KIND_CPU / KIND_AUTO instances support it");
        }
        if (cfg.instance_group.count != 1) {
            throw RepositoryError("model '" + cfg.name +
                                  "' enables 'sequence_batching' with instance_group count " +
                                  std::to_string(cfg.instance_group.count) +
                                  "; InferLite supports one sequence slot per model "
                                  "(Triton binds one sequence per instance)");
        }
        if (cfg.sequence.max_sequence_idle_us < 0) {
            throw RepositoryError("model '" + cfg.name +
                                  "' has negative max_sequence_idle_microseconds");
        }
        if (cfg.sequence.control_input.empty()) {
            throw RepositoryError("model '" + cfg.name +
                                  "' enables 'sequence_batching' without any control_input");
        }
        std::set<std::string> control_names;
        bool have_start = false, have_corrid = false;
        for (const auto& ci : cfg.sequence.control_input) {
            if (ci.name.empty() || !control_names.insert(ci.name).second) {
                throw RepositoryError("model '" + cfg.name +
                                      "' has an empty or duplicate control_input name '" +
                                      ci.name + "'");
            }
            for (const auto& ctl : ci.controls) {
                if (ctl.kind == SequenceControlKind::kInvalid ||
                    ctl.data_type == DataType::kInvalid) {
                    throw RepositoryError("model '" + cfg.name + "' control_input '" +
                                          ci.name +
                                          "' has an invalid control kind or data type");
                }
                if (ctl.kind == SequenceControlKind::kSequenceStart) have_start = true;
                if (ctl.kind == SequenceControlKind::kSequenceCorrId) have_corrid = true;
            }
        }
        if (!have_start || !have_corrid) {
            throw RepositoryError("model '" + cfg.name +
                                  "' sequence_batching must configure START and CORRID "
                                  "control inputs (END is optional)");
        }
        // Hidden state tensors must be part of the model I/O (clients never
        // send/receive them) and must not collide with control tensors.
        for (const auto& st : cfg.sequence.states) {
            bool in_ok = false, out_ok = false;
            for (const auto& in : cfg.inputs) {
                if (in.name == st.input_name && in.data_type == st.data_type) in_ok = true;
            }
            for (const auto& out : cfg.outputs) {
                if (out.name == st.output_name && out.data_type == st.data_type) out_ok = true;
            }
            if (!in_ok) {
                throw RepositoryError("model '" + cfg.name + "' sequence state input '" +
                                      st.input_name +
                                      "' is not declared as a model input of the same type");
            }
            if (!out_ok) {
                throw RepositoryError("model '" + cfg.name + "' sequence state output '" +
                                      st.output_name +
                                      "' is not declared as a model output of the same type");
            }
            if (control_names.count(st.input_name) || control_names.count(st.output_name)) {
                throw RepositoryError("model '" + cfg.name +
                                      "' sequence state tensor collides with a control_input");
            }
        }
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

    // OpenVINO backend: the version directory selected by version_policy must
    // contain the model artifacts required for the configured device. CPU/AUTO
    // need model.xml (AUTO may use a blob); NPU needs model.npu_blob; Intel GPU
    // needs model.gpu_blob. TensorRT backend: the selected version directory
    // must contain model.plan.
    fs::path version_dir;
    int64_t version = -1;
    if (!resolveVersionDir(cfg, model_dir, version_dir, version)) {
        std::string avail;
        for (const auto& dv : numericVersionDirs(model_dir)) {
            if (!avail.empty()) avail += ", ";
            avail += std::to_string(dv.first);
        }
        if (cfg.version_policy.configured &&
            cfg.version_policy.kind == VersionPolicyKind::kSpecific) {
            throw RepositoryError("model '" + cfg.name +
                                  "' version_policy.specific requests versions [" +
                                  joinNumbers(cfg.version_policy.versions) +
                                  "] but none is present in the repository" +
                                  (avail.empty() ? " (no numeric version directory)"
                                                 : " (available versions: " + avail + ")"));
        }
        throw RepositoryError("model '" + cfg.name + "' has no numeric version directory" +
                              (avail.empty() ? "" : " (available versions: " + avail + ")"));
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

    // Version selection honors the model's version_policy (specific pins an
    // exact version; latest/all use the highest numeric version). Plugin and
    // ensemble configs have no per-version artifacts, so they keep the
    // historical "highest numeric directory" choice for reporting; validateConfig
    // already threw above for an openvino/tensorrt model with no eligible
    // version directory, so this is only a defensive error.
    fs::path version_dir;
    int64_t version = -1;
    if (cfg.backend == "plugin" || cfg.backend == "ensemble") {
        highestVersionDir(model_dir, version_dir, version);
    } else if (!resolveVersionDir(cfg, model_dir, version_dir, version)) {
        throw RepositoryError("model '" + model_name +
                              "' has no version directory matching its version_policy");
    }

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
