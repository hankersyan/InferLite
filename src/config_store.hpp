// config_store.hpp - FDA configuration & integrity management.
//
// Owns the approved model manifest, computes SHA-256 hashes of model files and
// configuration, holds the software/OpenVINO version strings, and runs the
// startup functional self-test (golden input verification) for each model.
#pragma once

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "pbtxt.hpp"
#include "tensor.hpp"

namespace inferlite {

// One entry of the approved-model manifest (manifest.json).
struct ManifestEntry {
    std::string model_id;    // matches config name
    std::string version;     // numeric version dir
    std::string sha256;      // SHA-256 of the concatenated model files (xml+bin)
    std::string plugin_sha256;  // SHA-256 of the plugin library (if any)
};

// The whole manifest: a map from model_id -> entry plus the manifest's own hash.
struct Manifest {
    std::map<std::string, ManifestEntry> models;
    std::string manifest_hash;  // SHA-256 of the manifest.json file
    bool enabled = false;       // false if no manifest.json present (non-validated mode)
};

struct SelfTestResult {
    std::string model_id;
    bool passed = false;
    std::string detail;  // error detail if failed
};

class ConfigStore {
public:
    // `repository_root`: model repository root (for manifest.json/metadata.json).
    // `enforce_manifest`: if true, a missing manifest is a fatal error.
    ConfigStore(std::string repository_root, bool enforce_manifest);

    // Load manifest.json and metadata.json for every model. Computes file hashes
    // and config hashes. Throws std::runtime_error on integrity failures.
    void load();

    // Register and verify a model's IR files against the manifest. Computes the
    // model file hash and stores it for reporting. If the manifest is enabled
    // and the model is listed, a mismatch throws. Returns the stored model hash.
    std::string registerModelFiles(const std::string& model_id, const std::string& version_dir);

    // Record the SHA-256 of a model's config.pbtxt for integrity reporting.
    void registerConfigHash(const std::string& model_id, const std::string& config_text);

    // Software/OpenVINO version strings (reported to clients and audit log).
    void setSoftwareVersion(std::string v) { software_version_ = std::move(v); }
    void setOpenvinoVersion(std::string v) { openvino_version_ = std::move(v); }
    const std::string& softwareVersion() const { return software_version_; }
    const std::string& openvinoVersion() const { return openvino_version_; }

    // SHA-256 hex of a config.pbtxt text (per model). Populated after load.
    const std::string& configHash(const std::string& model_id) const;

    // Approved manifest hash for a model (from manifest). Empty if none.
    const std::string& manifestHash(const std::string& model_id) const;

    // Model file hash for a model (computed). Empty if not found.
    const std::string& modelHash(const std::string& model_id) const;

    // Combined config+model integrity hash reported in metrics.
    std::string configStoreHash() const;

    // Startup functional self-test: for each loaded model with a golden input,
    // run it through a provided executor and compare. `exec` runs one request
    // and returns success + outputs. Returns per-model results.
    std::vector<SelfTestResult> runSelfTest(
        const std::vector<std::shared_ptr<const ModelConfig>>& configs,
        const std::function<bool(const std::shared_ptr<const ModelConfig>&,
                                 const std::vector<Tensor>&, std::vector<Tensor>&)>& exec);

    // True if a manifest was present and verified.
    bool manifestEnabled() const { return manifest_.enabled; }

private:
    // Guards the per-model hash maps below. Hashes are registered during model
    // loads (which, with runtime model management, can happen after the HTTP /
    // gRPC servers are already serving) and read concurrently by metrics and
    // model-reporting handlers.
    mutable std::mutex mu_;

    std::string repository_root_;
    bool enforce_manifest_;
    Manifest manifest_;
    std::string software_version_ = "InferLite 2.0.0";
    std::string openvino_version_;
    std::map<std::string, std::string> config_hashes_;   // model -> config hash
    std::map<std::string, std::string> model_hashes_;    // model -> file hash
    std::map<std::string, ModelMetadata> metadata_;      // model -> metadata
};

}  // namespace inferlite
