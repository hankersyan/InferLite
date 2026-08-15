#include "config_store.hpp"

#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "json.hpp"
#include "sha256.hpp"
#include "validation.hpp"

namespace inferlite {

namespace fs = std::filesystem;

namespace {

std::string readTextFile(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open file: " + p.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Compute the SHA-256 of a model's artifact files. Phase 4 includes the
// precompiled blobs (.npu_blob / .gpu_blob) which carry the compiled model
// weights for NPU / Intel GPU, alongside the IR files (model.xml / model.bin).
std::string hashModelFiles(const std::string& version_dir) {
    fs::path dir(version_dir);
    std::vector<std::string> parts;
    for (const char* f : {"model.xml", "model.bin", "model.npu_blob", "model.gpu_blob"}) {
        fs::path fp = dir / f;
        if (fs::exists(fp)) {
            parts.push_back(sha256FileHex(fp.string()));
        }
    }
    if (parts.empty()) return "";
    // Hash the concatenation of the per-file hashes to form a stable model hash.
    std::string combined;
    for (const auto& p : parts) combined += p;
    return hexEncode(sha256(combined));
}

// Verify expected == actual, case-insensitively (hex strings).
bool hashEqual(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(a[i])) !=
            std::tolower(static_cast<unsigned char>(b[i]))) {
            return false;
        }
    }
    return true;
}

}  // namespace

ConfigStore::ConfigStore(std::string repository_root, bool enforce_manifest)
    : repository_root_(std::move(repository_root)), enforce_manifest_(enforce_manifest) {}

void ConfigStore::load() {
    fs::path root(repository_root_);
    fs::path manifest_path = root / "manifest.json";

    if (fs::exists(manifest_path)) {
        std::string manifest_text = readTextFile(manifest_path);
        manifest_.manifest_hash = hexEncode(sha256(manifest_text));
        json::Value doc = json::parse(manifest_text);

        const json::Value* models_node = doc.find("models");
        if (models_node && models_node->isArray()) {
            for (const auto& m : models_node->asArray()) {
                ManifestEntry e;
                if (const json::Value* v = m.find("model_id")) e.model_id = v->asString();
                if (const json::Value* v = m.find("version")) e.version = v->asString();
                if (const json::Value* v = m.find("sha256")) e.sha256 = v->asString();
                if (const json::Value* v = m.find("plugin_sha256")) e.plugin_sha256 = v->asString();
                if (!e.model_id.empty()) {
                    manifest_.models[e.model_id] = std::move(e);
                }
            }
        }
        manifest_.enabled = true;
    } else if (enforce_manifest_) {
        throw std::runtime_error("validated mode requires manifest.json in repository: " +
                                 repository_root_);
    }
}

std::string ConfigStore::registerModelFiles(const std::string& model_id,
                                            const std::string& version_dir) {
    std::string actual = hashModelFiles(version_dir);
    model_hashes_[model_id] = actual;

    if (manifest_.enabled) {
        auto it = manifest_.models.find(model_id);
        if (it != manifest_.models.end()) {
            if (!hashEqual(it->second.sha256, actual)) {
                throw std::runtime_error("model '" + model_id +
                                         "' file hash mismatch: manifest says " +
                                         it->second.sha256 + " but computed " + actual);
            }
        }
    }
    return actual;
}

void ConfigStore::registerConfigHash(const std::string& model_id,
                                     const std::string& config_hash_hex) {
    config_hashes_[model_id] = config_hash_hex;
}

const std::string& ConfigStore::configHash(const std::string& model_id) const {
    static const std::string kEmpty;
    auto it = config_hashes_.find(model_id);
    return it == config_hashes_.end() ? kEmpty : it->second;
}

const std::string& ConfigStore::manifestHash(const std::string& model_id) const {
    static const std::string kEmpty;
    auto it = manifest_.models.find(model_id);
    if (it == manifest_.models.end()) return kEmpty;
    return it->second.sha256;
}

const std::string& ConfigStore::modelHash(const std::string& model_id) const {
    static const std::string kEmpty;
    auto it = model_hashes_.find(model_id);
    return it == model_hashes_.end() ? kEmpty : it->second;
}

std::string ConfigStore::configStoreHash() const {
    std::string combined;
    for (const auto& kv : config_hashes_) combined += kv.second;
    for (const auto& kv : model_hashes_) combined += kv.second;
    return combined.empty() ? std::string() : hexEncode(sha256(combined));
}

std::vector<SelfTestResult> ConfigStore::runSelfTest(
    const std::vector<std::shared_ptr<const ModelConfig>>& configs,
    const std::function<bool(const std::shared_ptr<const ModelConfig>&,
                             const std::vector<Tensor>&, std::vector<Tensor>&)>& exec) {
    std::vector<SelfTestResult> results;
    for (const auto& cfg : configs) {
        SelfTestResult r;
        r.model_id = cfg->name;
        if (!cfg->self_test.enabled || cfg->self_test.input.empty()) {
            // No golden input configured: report as not-applicable (passed).
            r.passed = true;
            r.detail = "no_golden_input";
            results.push_back(std::move(r));
            continue;
        }

        std::vector<Tensor> outputs;
        try {
            bool ok = exec(cfg, cfg->self_test.input, outputs);
            if (!ok) {
                r.passed = false;
                r.detail = "self-test execution failed";
            } else if (!cfg->self_test.expected_output.empty()) {
                // Compare outputs to expected values within epsilon.
                double eps = cfg->self_test.epsilon;
                bool match = true;
                for (const auto& exp : cfg->self_test.expected_output) {
                    const Tensor* got = nullptr;
                    for (const auto& o : outputs) {
                        if (o.name == exp.name) { got = &o; break; }
                    }
                    if (!got) { match = false; break; }
                    if (got->shape != exp.shape) { match = false; break; }
                    size_t n = got->data.size() / dataTypeSize(got->type);
                    for (size_t i = 0; i < n; ++i) {
                        double g = 0, e = 0;
                        if (!readTensorScalar(*got, i, g) || !readTensorScalar(exp, i, e)) {
                            match = false;
                            break;
                        }
                        if (eps > 0.0) {
                            if (std::fabs(g - e) > eps) { match = false; break; }
                        } else if (g != e) {
                            match = false;
                            break;
                        }
                    }
                    if (!match) break;
                }
                r.passed = match;
                r.detail = match ? "ok" : "output mismatch vs golden";
            } else {
                // No expected output: just require a successful run.
                r.passed = true;
                r.detail = "ok";
            }
        } catch (const std::exception& ex) {
            r.passed = false;
            r.detail = std::string("self-test threw: ") + ex.what();
        }
        results.push_back(std::move(r));
    }
    return results;
}

}  // namespace inferlite
