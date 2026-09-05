#include "infer_lite.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "openvino/openvino.hpp"

#include "ensemble_executor.hpp"
#include "json.hpp"
#include "model_repository.hpp"
#include "openvino_backend.hpp"
#include "pbtxt.hpp"
#include "plugin_backend.hpp"
#include "sha256.hpp"
#include "tensorrt_backend.hpp"
#include "validation.hpp"

#ifdef INFERLITE_ENABLE_GPU
#include "gpu_memory_manager.hpp"
#endif

#ifdef INFERLITE_ENABLE_GRPC
#include "grpc_server.hpp"
#endif

namespace inferlite {

namespace fs = std::filesystem;

namespace {

// ---- base64 decoding (for binary tensor payloads) -------------------------
int base64Val(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

std::vector<uint8_t> base64Decode(const std::string& in) {
    std::vector<uint8_t> out;
    int val = 0, bits = -8;
    for (unsigned char c : in) {
        if (c == '=') break;
        int v = base64Val(c);
        if (v < 0) continue;
        val = (val << 6) | v;
        bits += 6;
        if (bits >= 0) {
            out.push_back(static_cast<uint8_t>((val >> bits) & 0xFF));
            bits -= 8;
        }
    }
    return out;
}

std::string base64Encode(const uint8_t* data, size_t len) {
    static const char* tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    size_t i = 0;
    while (i + 3 <= len) {
        uint32_t n = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2];
        out += tbl[(n >> 18) & 63];
        out += tbl[(n >> 12) & 63];
        out += tbl[(n >> 6) & 63];
        out += tbl[n & 63];
        i += 3;
    }
    if (i + 1 == len) {
        uint32_t n = data[i] << 16;
        out += tbl[(n >> 18) & 63];
        out += tbl[(n >> 12) & 63];
        out += "==";
    } else if (i + 2 == len) {
        uint32_t n = (data[i] << 16) | (data[i + 1] << 8);
        out += tbl[(n >> 18) & 63];
        out += tbl[(n >> 12) & 63];
        out += tbl[(n >> 6) & 63];
        out += "=";
    }
    return out;
}

// Extract model name from a path like /v2/models/<name>/infer.
bool extractModelName(const std::string& path, std::string& name) {
    const std::string prefix = "/v2/models/";
    if (path.rfind(prefix, 0) != 0) return false;
    std::string rest = path.substr(prefix.size());
    size_t slash = rest.find('/');
    name = rest.substr(0, slash);
    return !name.empty();
}

bool endsWith(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Generate a UUID v4-ish trace id (RFC 4122) for audit trail correlation.
std::string generateTraceId() {
    static std::mt19937_64 rng{std::random_device{}()};
    std::ostringstream os;
    os << std::hex;
    uint64_t a = rng(), b = rng();
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%08x-%04x-4%03x-%04x-%012llx",
                  static_cast<unsigned>(a >> 32),
                  static_cast<unsigned>((a >> 16) & 0xFFFF),
                  static_cast<unsigned>(a & 0x0FFF),
                  static_cast<unsigned>((b >> 48) & 0xFFFF),
                  static_cast<unsigned long long>(b & 0xFFFFFFFFFFFFULL));
    os << buf;
    return os.str();
}

// Map an ErrorCode to an HTTP status.
int errorCodeToStatus(ErrorCode c) {
    switch (c) {
        case ErrorCode::kInvalidInput: return 400;
        case ErrorCode::kModelNotFound: return 404;
        case ErrorCode::kResourceExhausted: return 503;
        case ErrorCode::kTimeout: return 504;
        case ErrorCode::kSelfTestFailed: return 503;
        default: return 500;
    }
}

std::string joinWith(const std::vector<std::string>& v, const char* sep) {
    std::string out;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) out += sep;
        out += v[i];
    }
    return out;
}

HttpResponse jsonError(int status, const std::string& msg) {
    HttpResponse r;
    r.status = status;
    r.body = std::string("{\"error\":\"") + msg + "\"}";
    return r;
}

}  // namespace

// ---- model-control-mode conversions ----------------------------------------

std::string modelControlModeToString(ModelControlMode m) {
    switch (m) {
        case ModelControlMode::kNone: return "none";
        case ModelControlMode::kPoll: return "poll";
        case ModelControlMode::kExplicit: return "explicit";
    }
    return "none";
}

ModelControlMode modelControlModeFromString(const std::string& s) {
    if (s == "none") return ModelControlMode::kNone;
    if (s == "poll") return ModelControlMode::kPoll;
    if (s == "explicit") return ModelControlMode::kExplicit;
    throw std::runtime_error("invalid --model-control-mode '" + s +
                             "' (expected 'none', 'poll', or 'explicit')");
}

// ---- construction -----------------------------------------------------------

InferLite::InferLite(const ServerOptions& opts) : opts_(opts) {
    memory_ = std::make_shared<MemoryManager>();
#ifdef INFERLITE_ENABLE_GPU
    gpu_memory_ = std::make_shared<GpuMemoryManager>();
#endif
    diag_ = std::make_unique<Diagnostics>(opts_.diagnostic_log_path);
    audit_ = std::make_unique<AuditLog>(opts_.audit_log_path);

    limits_.max_input_size_bytes = opts_.max_input_size_bytes;
    limits_.max_output_size_bytes = opts_.max_output_size_bytes;
    limits_.max_inference_time_ms = opts_.max_inference_time_ms;
    limits_.max_queue_depth = opts_.max_queue_size;

    config_store_ = std::make_unique<ConfigStore>(opts_.model_repository,
                                                  opts_.validated_mode);
    config_store_->setSoftwareVersion(opts_.software_version);
    try {
        ov::Version v = ov::get_openvino_version();
        config_store_->setOpenvinoVersion(std::string(v.buildNumber));
    } catch (...) {
        config_store_->setOpenvinoVersion("unknown");
    }
    config_store_->load();
    diag_->info("manifest loaded (enabled=" +
                std::string(config_store_->manifestEnabled() ? "true" : "false") + ")");
    if (opts_.validated_mode && opts_.tls_cert_file.empty()) {
        diag_->warn("validated-mode is enabled without TLS; deployments MUST front this "
                    "server with a TLS 1.2+ reverse proxy (FDA Cybersecurity 2026).");
    }

    // All modes fail fast when the repository root itself is missing.
    {
        std::error_code ec;
        if (!fs::is_directory(opts_.model_repository, ec)) {
            throw RepositoryError("model repository does not exist or is not a directory: " +
                                  opts_.model_repository);
        }
    }
    if (opts_.model_control_mode == ModelControlMode::kPoll &&
        opts_.repository_poll_secs == 0) {
        throw std::runtime_error("model-control-mode 'poll' requires a non-zero "
                                 "--repository-poll-secs");
    }
    if (opts_.model_control_mode != ModelControlMode::kExplicit &&
        !opts_.load_models.empty()) {
        bool star = false;
        for (const auto& m : opts_.load_models) star = star || (m == "*");
        if (star) {
            throw std::runtime_error("--load-model=* is only valid with "
                                     "--model-control-mode=explicit");
        }
        diag_->warn("--load-model is only honored in 'explicit' model-control mode");
    }

    switch (opts_.model_control_mode) {
        case ModelControlMode::kNone: {
            // Legacy (default) behavior: load and self-verify every model in the
            // repository; any config/backend/integrity error aborts startup.
            std::vector<LoadedModel> loaded = scanRepository(opts_.model_repository);
            std::vector<LoadedModel> pending_ensembles;
            for (auto& lm : loaded) {
                if (lm.config->backend == "ensemble") {
                    pending_ensembles.push_back(std::move(lm));
                    continue;
                }
                commitEntry(buildEntry(lm));
            }
            // Topological ensemble construction: an ensemble may only be built
            // once every model it references is committed.
            bool progress = true;
            while (!pending_ensembles.empty() && progress) {
                progress = false;
                for (auto it = pending_ensembles.begin(); it != pending_ensembles.end();) {
                    auto e = newEntryShell(*it);
                    std::vector<std::string> unresolved;
                    try {
                        attachEnsembleBackend(e, unresolved);
                    } catch (const std::exception& ex) {
                        unresolved.push_back(std::string("build failed: ") + ex.what());
                    }
                    if (!unresolved.empty()) {
                        ++it;
                        continue;
                    }
                    attachScheduler(e);
                    e->state = ModelEntry::State::kReady;
                    commitEntry(e);
                    it = pending_ensembles.erase(it);
                    progress = true;
                }
            }
            if (!pending_ensembles.empty()) {
                std::vector<std::string> names;
                for (const auto& lm : pending_ensembles) names.push_back(lm.config->name);
                throw std::runtime_error("cannot construct ensembles with unsatisfied "
                                         "dependencies: " + joinWith(names, ", "));
            }
            break;
        }

        case ModelControlMode::kPoll: {
            // Load every discoverable model at startup, but never let one broken
            // model abort the server: failures are recorded as UNAVAILABLE so the
            // poller can retry once the repository is fixed.
            refreshKnownModelsFromDisk();
            std::vector<std::string> names = listRepositoryModelNames(opts_.model_repository);
            // Seed the fingerprint map so the first poll tick only reacts to
            // real changes (a dependency loaded transitively through an
            // ensemble must not be reloaded pointlessly on the next poll).
            for (const auto& name : names) {
                repo_fingerprints_[name] = fingerprintModelDirectory(opts_.model_repository, name);
            }
            for (const auto& name : names) {
                {
                    std::lock_guard<std::mutex> lock(models_mu_);
                    auto e = findEntryLocked(name);
                    if (e && e->loaded()) continue;
                }
                std::lock_guard<std::mutex> lock(lifecycle_mu_);
                ControlStatus st = loadModelInternal(name, "");
                if (!st.ok) {
                    diag_->warn("startup load of model '" + name + "' deferred: " + st.error);
                }
            }
            break;
        }

        case ModelControlMode::kExplicit: {
            // Discover the repository for the index but load only what the
            // operator asked for (--load-model) or, with '*', everything.
            refreshKnownModelsFromDisk();
            std::vector<std::string> names = listRepositoryModelNames(opts_.model_repository);

            std::vector<std::string> to_load;
            for (const auto& m : opts_.load_models) {
                if (m == "*") {
                    if (opts_.load_models.size() > 1) {
                        throw std::runtime_error("--load-model=* cannot be combined with "
                                                 "explicit model names");
                    }
                    to_load = names;
                    break;
                }
                to_load.push_back(m);
            }
            for (const auto& name : to_load) {
                bool found = std::find(names.begin(), names.end(), name) != names.end();
                if (!found) {
                    throw std::runtime_error("--load-model '" + name + "' not found in "
                                             "repository '" + opts_.model_repository + "'");
                }
            }
            for (const auto& name : to_load) {
                std::lock_guard<std::mutex> lock(lifecycle_mu_);
                ControlStatus st = loadModelInternal(name, "");
                if (!st.ok) {
                    throw std::runtime_error("startup load of model '" + name +
                                             "' failed: " + st.error);
                }
            }
            break;
        }
    }

    // Run the startup functional self-test for every model loaded at startup
    // (each with a golden input, when configured).
    runStartupSelfTest();
}

InferLite::~InferLite() {
    stop();
}

// ---- lifecycle helpers -------------------------------------------------------

std::shared_ptr<InferLite::ModelEntry> InferLite::newEntryShell(const LoadedModel& lm) {
    auto entry = std::make_shared<ModelEntry>();
    entry->name = lm.config->name;
    entry->config = lm.config;
    entry->version_path = lm.version_path;
    entry->version = lm.version;
    entry->config_hash = lm.config_hash;
    entry->state = ModelEntry::State::kLoading;
    entry->reason.clear();

    // Register config hash and verify model file integrity against the
    // manifest (if enabled). Ensembles/plugins have no artifact files to
    // verify; OpenVINO (model.xml/.bin) and TensorRT (model.plan) do.
    config_store_->registerConfigHash(entry->name, lm.config_hash);
    if ((lm.config->backend == "openvino" || lm.config->backend == "tensorrt") &&
        !lm.version_path.empty()) {
        config_store_->registerModelFiles(entry->name, lm.version_path);
    }
    entry->device_label =
        (lm.config->backend == "tensorrt" && lm.config->instance_group.kind == "KIND_GPU")
            ? "GPU"
            : "CPU";
    return entry;
}

void InferLite::attachScheduler(const std::shared_ptr<ModelEntry>& e) {
    size_t instance_count =
        static_cast<size_t>(std::max<int>(1, e->config->instance_group.count));
    int64_t per_req_timeout = opts_.request_timeout_ms;
    if (e->config->max_inference_time_ms > 0) {
        per_req_timeout = e->config->max_inference_time_ms;
    }
    e->scheduler = std::make_shared<Scheduler>(e->backend, e->config, instance_count,
                                               opts_.max_queue_size, per_req_timeout,
                                               limits_.max_inference_time_ms, memory_,
                                               e->device_label);
}

BackendPtr InferLite::makeBackend(const LoadedModel& lm) {
    const auto& cfg = *lm.config;

    if (cfg.backend == "plugin") {
        // Resolve the plugin library path: relative to the model's repository
        // directory, or absolute.
        std::filesystem::path lib = cfg.plugin_library;
        if (!lib.is_absolute()) {
            lib = std::filesystem::path(cfg.model_path) / cfg.plugin_library;
        }
        // Verify the plugin library hash against the manifest (if enabled).
        if (config_store_->manifestEnabled()) {
            const std::string& expected = config_store_->manifestHash(cfg.name);
            if (!expected.empty()) {
                std::string actual = sha256FileHex(lib.string());
                if (actual != expected) {
                    throw std::runtime_error("plugin library '" + cfg.name +
                                             "' hash mismatch: expected " + expected +
                                             " got " + actual);
                }
            }
        }
        auto plugin_backend = std::make_shared<PluginBackend>();
        plugin_backend->load(cfg, lib.string());
        struct PluginWrapper : IBackend {
            std::shared_ptr<PluginBackend> pb;
            explicit PluginWrapper(std::shared_ptr<PluginBackend> p) : pb(std::move(p)) {}
            BackendResult execute(const std::vector<Tensor>& inputs) override {
                BackendResult r;
                try {
                    std::vector<Tensor> outputs;
                    pb->execute(inputs, outputs);
                    r.ok = true;
                    r.outputs = std::move(outputs);
                } catch (const std::exception& e) {
                    r.ok = false;
                    r.error_code = ErrorCode::kInternalError;
                    r.error = e.what();
                }
                return r;
            }
        };
        return std::make_shared<PluginWrapper>(plugin_backend);
    }

    if (cfg.backend == "ensemble") {
        throw std::runtime_error("ensemble '" + cfg.name +
                                 "' must be built in the ensemble pass");
    }

    if (cfg.backend == "tensorrt") {
#ifdef INFERLITE_ENABLE_GPU
        if (!gpu_memory_) {
            throw std::runtime_error("TensorRT backend '" + cfg.name +
                                     "' requested but GPU memory manager is unavailable");
        }
        std::string plan_path = lm.version_path;
        if (!plan_path.empty() && plan_path.back() != '/' && plan_path.back() != '\\') {
            plan_path += "/";
        }
        plan_path += "model.plan";
        auto trt = std::make_shared<TensorRtBackend>(gpu_memory_);
        trt->load(cfg, plan_path, opts_.max_gpu_memory_mb);
        return trt;
#else
        throw std::runtime_error("TensorRT backend '" + cfg.name +
                                 "' requested but GPU support is not compiled in "
                                 "(rebuild with a TensorRT SDK / TENSORRT_ROOT)");
#endif
    }

    // openvino backend.
    auto backend = std::make_shared<OpenVinoBackend>();
    backend->setResourceLimits(&limits_);
    backend->load(cfg, lm.version_path);
    return backend;
}

std::shared_ptr<InferLite::ModelEntry> InferLite::buildEntry(const LoadedModel& lm) {
    if (lm.config->backend == "ensemble") {
        throw std::runtime_error("ensemble '" + lm.config->name +
                                 "' must be built once its dependencies are committed");
    }
    auto entry = newEntryShell(lm);
    entry->backend = makeBackend(lm);
    // Derive the execution device label for audit/health reporting. OpenVINO
    // backends report their resolved device (CPU/NPU/INTEL_GPU/AUTO); the
    // TensorRT backend reports "GPU"; everything else (plugin) is CPU.
    if (auto ovb = std::dynamic_pointer_cast<OpenVinoBackend>(entry->backend)) {
        entry->device_label = ovb->deviceLabel();
    } else if (entry->device_label != "GPU") {
        entry->device_label = "CPU";
    }
    attachScheduler(entry);
    entry->state = ModelEntry::State::kReady;
    return entry;
}

BackendPtr InferLite::backendByName(const std::string& name) {
    std::lock_guard<std::mutex> lock(models_mu_);
    auto e = findEntryLocked(name);
    return (e && e->loaded()) ? e->backend : nullptr;
}

void InferLite::attachEnsembleBackend(const std::shared_ptr<ModelEntry>& e,
                                      std::vector<std::string>& unresolved) {
    std::set<std::string> missing;
    for (const auto& step : e->config->ensemble_steps) {
        if (!backendByName(step.model_name)) missing.insert(step.model_name);
    }
    if (!missing.empty()) {
        for (const auto& m : missing) unresolved.push_back(m);
        return;
    }
    auto provider = [this](const std::string& name) -> BackendPtr {
        return backendByName(name);
    };
    e->backend = std::make_shared<EnsembleExecutor>(e->config, provider);
}

bool InferLite::runEntrySelfTest(const std::shared_ptr<ModelEntry>& entry) {
    if (!entry->scheduler) return false;
    std::vector<std::shared_ptr<const ModelConfig>> configs{entry->config};
    auto exec = [&entry, this](const std::shared_ptr<const ModelConfig>& cfg,
                               const std::vector<Tensor>& in,
                               std::vector<Tensor>& out) -> bool {
        if (cfg->name != entry->name || !entry->scheduler) return false;
        auto req = std::make_shared<InferenceRequest>();
        req->inputs = in;
        req->timeout_ms = opts_.request_timeout_ms;
        auto res = entry->scheduler->submit(req);
        if (res && res->ok) {
            out = res->outputs;
            return true;
        }
        return false;
    };
    auto results = config_store_->runSelfTest(configs, exec);
    if (results.empty()) return true;
    if (!results[0].passed) {
        diag_->error("self-test model '" + entry->name + "' failed detail=" + results[0].detail);
    }
    return results[0].passed;
}

void InferLite::runStartupSelfTest() {
    std::map<std::string, std::shared_ptr<Scheduler>> schedulers;
    std::vector<std::shared_ptr<const ModelConfig>> configs;
    {
        std::lock_guard<std::mutex> lock(models_mu_);
        for (const auto& ep : models_) {
            if (!ep->loaded()) continue;
            configs.push_back(ep->config);
            schedulers[ep->name] = ep->scheduler;
        }
    }

    auto exec = [&schedulers, this](const std::shared_ptr<const ModelConfig>& cfg,
                                    const std::vector<Tensor>& in,
                                    std::vector<Tensor>& out) -> bool {
        auto it = schedulers.find(cfg->name);
        if (it == schedulers.end()) return false;
        auto req = std::make_shared<InferenceRequest>();
        req->inputs = in;
        req->timeout_ms = opts_.request_timeout_ms;
        auto res = it->second->submit(req);
        if (res && res->ok) {
            out = res->outputs;
            return true;
        }
        return false;
    };

    auto results = config_store_->runSelfTest(configs, exec);
    bool all_passed = true;
    for (const auto& r : results) {
        if (!r.passed) all_passed = false;
        diag_->info("self-test model '" + r.model_id + "' passed=" +
                    std::string(r.passed ? "true" : "false") +
                    (r.detail.empty() ? "" : " detail=" + r.detail));
    }
    self_test_passed_ = all_passed;
    if (!all_passed) {
        diag_->error("startup self-test FAILED; server will report NOT READY");
    }
}

// ---- server start / stop ----------------------------------------------------

void InferLite::start() {
    auto handler = [this](const HttpRequest& req) { return handleRequest(req); };
    http_ = std::make_unique<HttpServer>(opts_.host, opts_.http_port, handler,
                                         opts_.http_threads);
    http_->start();
#ifdef INFERLITE_ENABLE_GRPC
    if (opts_.grpc_port > 0) {
        grpc_ = std::make_unique<GrpcServer>(this, opts_.host, opts_.grpc_port);
        grpc_->start();
    }
#endif
    running_ = true;

    if (opts_.model_control_mode == ModelControlMode::kPoll) {
        std::lock_guard<std::mutex> lock(poll_mu_);
        poll_stop_ = false;
        poll_thread_ = std::thread([this]() { pollLoop(); });
        diag_->info("repository poller started (interval " +
                    std::to_string(opts_.repository_poll_secs) + "s)");
    }
}

void InferLite::stop() {
    // Stop the poller first so no model load/unload races with server teardown.
    {
        std::lock_guard<std::mutex> lock(poll_mu_);
        poll_stop_ = true;
    }
    poll_cv_.notify_all();
    if (poll_thread_.joinable()) poll_thread_.join();

#ifdef INFERLITE_ENABLE_GRPC
    if (grpc_) {
        grpc_->stop();
        grpc_.reset();
    }
#endif
    if (http_) {
        http_->stop();
        http_.reset();
    }
    running_ = false;
}

void InferLite::waitForShutdown() {
    while (running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

// ---- model table accessors ---------------------------------------------------

std::string InferLite::stateString(const ModelEntry& e) const {
    switch (e.state) {
        case ModelEntry::State::kReady: return "READY";
        case ModelEntry::State::kLoading: return "LOADING";
        case ModelEntry::State::kUnloading: return "UNLOADING";
        default: return "UNAVAILABLE";
    }
}

std::shared_ptr<InferLite::ModelEntry> InferLite::findEntryLocked(const std::string& name) const {
    for (const auto& ep : models_) {
        if (ep->name == name) return ep;
    }
    return nullptr;
}

bool InferLite::modelExists(const std::string& name) const {
    std::lock_guard<std::mutex> lock(models_mu_);
    auto e = findEntryLocked(name);
    return e && e->loaded();
}

std::string InferLite::modelDevice(const std::string& name) const {
    std::lock_guard<std::mutex> lock(models_mu_);
    auto e = findEntryLocked(name);
    return (e && e->loaded()) ? e->device_label : std::string();
}

std::vector<std::string> InferLite::modelNames() const {
    std::vector<std::string> names;
    std::lock_guard<std::mutex> lock(models_mu_);
    for (const auto& ep : models_) {
        if (ep->loaded()) names.push_back(ep->name);
    }
    return names;
}

std::string InferLite::newTraceId() {
    return generateTraceId();
}

std::vector<ModelInfo> InferLite::modelInfo() const {
    std::vector<ModelInfo> out;
    std::lock_guard<std::mutex> lock(models_mu_);
    for (const auto& ep : models_) {
        ModelInfo mi;
        mi.name = ep->name;
        mi.state = stateString(*ep);
        mi.reason = ep->reason;
        mi.version = ep->version;
        mi.version_path = ep->version_path;
        mi.ready = ep->loaded();
        if (ep->config) {
            mi.backend = ep->config->backend;
            mi.device_label = ep->device_label;
            mi.config_hash = ep->config_hash;
            mi.model_hash = config_store_->modelHash(ep->name);
            mi.config = ep->config;
        }
        out.push_back(std::move(mi));
    }
    return out;
}

// ---- lifecycle mutation (assumes lifecycle_mu_ is held by the caller) ---------

void InferLite::commitEntry(const std::shared_ptr<ModelEntry>& fresh) {
    std::lock_guard<std::mutex> lock(models_mu_);
    auto existing = findEntryLocked(fresh->name);
    if (!existing) {
        models_.push_back(fresh);
    } else {
        // Keep the same entry object so any reader that copied the shared_ptr
        // observes consistent fields; old scheduler/backend references are
        // dropped here and live until the last in-flight request releases them.
        existing->config = fresh->config;
        existing->config_hash = fresh->config_hash;
        existing->backend = fresh->backend;
        existing->scheduler = fresh->scheduler;
        existing->version_path = fresh->version_path;
        existing->version = fresh->version;
        existing->device_label = fresh->device_label;
        existing->state = fresh->state;
        existing->reason = fresh->reason;
    }
    if (fresh->device_label == "GPU" &&
        gpu_usage_bytes_.find(fresh->name) == gpu_usage_bytes_.end()) {
        gpu_usage_bytes_[fresh->name].store(0, std::memory_order_relaxed);
    }
}

void InferLite::markUnavailableLocked(const std::string& name, const std::string& reason) {
    auto e = findEntryLocked(name);
    if (e && !e->loaded()) {
        e->state = ModelEntry::State::kUnavailable;
        e->reason = reason;
    }
}

std::vector<std::string> InferLite::loadedDependents(const std::string& name) const {
    std::vector<std::string> out;
    std::lock_guard<std::mutex> lock(models_mu_);
    for (const auto& ep : models_) {
        if (!ep->loaded() || ep->name == name || !ep->config) continue;
        bool references = false;
        for (const auto& step : ep->config->ensemble_steps) {
            if (step.model_name == name) {
                references = true;
                break;
            }
        }
        if (references) out.push_back(ep->name);
    }
    return out;
}

void InferLite::refreshKnownModelsFromDisk() {
    std::error_code ec;
    if (!fs::is_directory(opts_.model_repository, ec)) return;
    std::vector<std::string> names = listRepositoryModelNames(opts_.model_repository);
    std::set<std::string> on_disk(names.begin(), names.end());

    std::lock_guard<std::mutex> lock(models_mu_);
    // Add placeholders for models not yet tracked.
    for (const auto& name : names) {
        if (findEntryLocked(name)) continue;
        auto e = std::make_shared<ModelEntry>();
        e->name = name;
        e->state = ModelEntry::State::kUnavailable;
        e->reason = "model not loaded";
        std::string v = highestModelVersionString(opts_.model_repository, name);
        if (!v.empty()) e->version = std::stoll(v);
        models_.push_back(std::move(e));
    }
    // Drop untracked UNAVAILABLE placeholders whose directory has disappeared
    // (a model that was loaded, or whose config was parsed, is never dropped
    // here: in explicit mode the model stays known after its directory is
    // removed, and unloads are driven explicitly).
    for (auto it = models_.begin(); it != models_.end();) {
        const auto& ep = *it;
        if (!ep->loaded() && !ep->config && on_disk.count(ep->name) == 0) {
            it = models_.erase(it);
        } else {
            ++it;
        }
    }
}

// Load one model. `config_override` replaces the repository config.pbtxt.
// Ensemble dependencies are loaded (from the repository) first. On success the
// entry is atomically committed so in-flight requests against a previous
// instance drain on the old scheduler. When an already-loaded model is reloaded
// successfully, loaded models that reference it (ensembles) are reloaded too.
ControlStatus InferLite::loadModelInternal(const std::string& name,
                                           const std::string& config_override) {
    std::error_code ec;
    const bool dir_exists = fs::is_directory(fs::path(opts_.model_repository) / name, ec);

    bool already_loaded = false;
    {
        std::lock_guard<std::mutex> lock(models_mu_);
        auto e = findEntryLocked(name);
        already_loaded = e && e->loaded();
    }

    if (!dir_exists) {
        return ControlStatus{false, 404,
                             "model '" + name + "' not found in repository '" +
                                 opts_.model_repository + "'"};
    }

    // 1. Resolve and validate the model configuration (disk or override).
    LoadedModel lm;
    try {
        lm = loadModelConfig(opts_.model_repository, name, config_override);
    } catch (const std::exception& ex) {
        std::lock_guard<std::mutex> lock(models_mu_);
        markUnavailableLocked(name, ex.what());
        return ControlStatus{false, 400,
                             "failed to load model '" + name + "': " + ex.what()};
    }

    // 2. Ensembles require every referenced model to be loaded first.
    if (lm.config->backend == "ensemble") {
        for (const auto& step : lm.config->ensemble_steps) {
            bool dep_loaded = false;
            {
                std::lock_guard<std::mutex> lock(models_mu_);
                auto d = findEntryLocked(step.model_name);
                dep_loaded = d && d->loaded();
            }
            if (dep_loaded) continue;
            ControlStatus dep_st = loadModelInternal(step.model_name, "");
            if (!dep_st.ok) {
                return ControlStatus{false, 400,
                                     "cannot load ensemble '" + name + "': dependency '" +
                                         step.model_name + "' failed to load: " + dep_st.error};
            }
        }
    }

    // 3. Build the new instance (backend + scheduler). No model lock is held
    //    while the backend compiles/loads, so serving continues during reload.
    std::shared_ptr<ModelEntry> fresh = newEntryShell(lm);
    try {
        if (lm.config->backend == "ensemble") {
            std::vector<std::string> unresolved;
            attachEnsembleBackend(fresh, unresolved);
            if (!unresolved.empty()) {
                return ControlStatus{false, 400,
                                     "cannot load ensemble '" + name +
                                         "': unresolved dependencies: " +
                                         joinWith(unresolved, ", ")};
            }
        } else {
            fresh->backend = makeBackend(lm);
        }
        // Derive the resolved device label (OpenVINO reports its device).
        if (auto ovb = std::dynamic_pointer_cast<OpenVinoBackend>(fresh->backend)) {
            fresh->device_label = ovb->deviceLabel();
        } else if (fresh->device_label != "GPU") {
            fresh->device_label = "CPU";
        }
        attachScheduler(fresh);
    } catch (const std::exception& ex) {
        std::lock_guard<std::mutex> lock(models_mu_);
        markUnavailableLocked(name, ex.what());
        return ControlStatus{false, 400,
                             "failed to load model '" + name + "': " + ex.what()};
    }

    // 4. Golden-input functional self-test before the model is made available.
    if (!runEntrySelfTest(fresh)) {
        std::lock_guard<std::mutex> lock(models_mu_);
        if (!already_loaded) {
            markUnavailableLocked(name, "model failed its functional self-test");
        }
        return ControlStatus{false, 500,
                             "model '" + name + "' failed its functional self-test"};
    }

    // 5. Commit the new instance (atomic swap under the model lock).
    fresh->state = ModelEntry::State::kReady;
    fresh->reason.clear();
    commitEntry(fresh);
    diag_->info("model '" + name + "' loaded (device=" + fresh->device_label +
                ", version=" + (fresh->version >= 0 ? std::to_string(fresh->version) : "n/a") +
                ")");

    // 6. A reload of an ensemble dependency must propagate to the loaded
    //    ensembles that reference it so their DAGs observe the new instance.
    if (already_loaded) {
        for (const auto& dep : loadedDependents(name)) {
            if (dep == name) continue;
            ControlStatus st = loadModelInternal(dep, "");
            if (!st.ok) {
                diag_->error("reload of model '" + name + "' cascaded to dependent '" + dep +
                             "' failed: " + st.error);
            }
        }
    }
    return ControlStatus{true, 200, ""};
}

ControlStatus InferLite::unloadOne(const std::string& name, bool unload_dependents,
                                   std::vector<std::string>& visited) {
    if (std::find(visited.begin(), visited.end(), name) != visited.end()) {
        return ControlStatus{true, 200, ""};
    }
    visited.push_back(name);

    bool known = false;
    bool loaded = false;
    {
        std::lock_guard<std::mutex> lock(models_mu_);
        auto e = findEntryLocked(name);
        if (e) {
            known = true;
            loaded = e->loaded();
        }
    }
    if (!known) {
        return ControlStatus{false, 404,
                             "model '" + name + "' not found in repository '" +
                                 opts_.model_repository + "'"};
    }

    if (loaded) {
        // Reject unloading a model that loaded ensembles depend on, unless the
        // caller asks to take the dependents down too.
        std::vector<std::string> deps = loadedDependents(name);
        if (!deps.empty() && !unload_dependents) {
            return ControlStatus{false, 409,
                                 "model '" + name + "' is referenced by loaded model(s): " +
                                     joinWith(deps, ", ") +
                                     "; unload them first or set unload_dependents=true"};
        }
        for (const auto& d : deps) {
            if (d == name) continue;
            ControlStatus st = unloadOne(d, true, visited);
            if (!st.ok) return st;
        }
    }

    // Unload the model itself.
    std::error_code ec;
    const bool on_disk = fs::is_directory(fs::path(opts_.model_repository) / name, ec);
    std::lock_guard<std::mutex> lock(models_mu_);
    auto e = findEntryLocked(name);
    if (!e) return ControlStatus{true, 200, ""};  // already gone
    if (!loaded) {
        // Never-loaded placeholder: unload is a no-op success (Triton keeps the
        // model visible as UNAVAILABLE). When the poller observed removal, the
        // placeholder is erased entirely.
        if (!on_disk && !e->config) {
            models_.erase(
                std::remove_if(models_.begin(), models_.end(),
                               [&](const std::shared_ptr<ModelEntry>& p) { return p == e; }),
                models_.end());
        }
        return ControlStatus{true, 200, ""};
    }

    diag_->info("model '" + name + "' unloaded");
    if (on_disk) {
        // Explicit unload: the model stays known to the repository (UNAVAILABLE)
        // so the index can still list it.
        e->state = ModelEntry::State::kUnavailable;
        e->reason = "model unloaded";
        e->backend.reset();
        e->scheduler.reset();
        // config/version_path are retained for reporting.
    } else {
        // Poll detected removal: forget the model entirely.
        models_.erase(
            std::remove_if(models_.begin(), models_.end(),
                           [&](const std::shared_ptr<ModelEntry>& p) { return p == e; }),
            models_.end());
    }
    return ControlStatus{true, 200, ""};
}

ControlStatus InferLite::unloadModelInternal(const std::string& name, bool unload_dependents) {
    std::vector<std::string> visited;
    return unloadOne(name, unload_dependents, visited);
}

// ---- public repository-control API ------------------------------------------

std::vector<ModelIndexEntry> InferLite::repositoryIndex(bool ready_only) {
    if (opts_.model_control_mode != ModelControlMode::kNone) {
        refreshKnownModelsFromDisk();
    }
    std::vector<ModelIndexEntry> out;
    std::lock_guard<std::mutex> lock(models_mu_);
    for (const auto& ep : models_) {
        ModelIndexEntry ie;
        ie.name = ep->name;
        ie.version = ep->version >= 0 ? std::to_string(ep->version)
                                      : highestModelVersionString(opts_.model_repository,
                                                                  ep->name);
        ie.state = stateString(*ep);
        ie.reason = ep->reason;
        if (ready_only && ie.state != "READY") continue;
        out.push_back(std::move(ie));
    }
    return out;
}

ControlStatus InferLite::repositoryLoad(const std::string& model_name,
                                        const std::string& config_text_override) {
    if (opts_.model_control_mode != ModelControlMode::kExplicit) {
        return ControlStatus{false, 400,
                             "repository load requests require --model-control-mode=explicit "
                             "(current mode: " + modelControlModeName() + ")"};
    }
    if (model_name.empty()) {
        return ControlStatus{false, 400, "model name must not be empty"};
    }
    // Make sure a freshly created model directory is visible to the loader.
    refreshKnownModelsFromDisk();
    std::lock_guard<std::mutex> lock(lifecycle_mu_);
    return loadModelInternal(model_name, config_text_override);
}

ControlStatus InferLite::repositoryUnload(const std::string& model_name,
                                          bool unload_dependents) {
    if (opts_.model_control_mode != ModelControlMode::kExplicit) {
        return ControlStatus{false, 400,
                             "repository unload requests require --model-control-mode=explicit "
                             "(current mode: " + modelControlModeName() + ")"};
    }
    if (model_name.empty()) {
        return ControlStatus{false, 400, "model name must not be empty"};
    }
    refreshKnownModelsFromDisk();
    std::lock_guard<std::mutex> lock(lifecycle_mu_);
    return unloadModelInternal(model_name, unload_dependents);
}

// ---- poll mode ----------------------------------------------------------------

void InferLite::pollLoop() {
    std::unique_lock<std::mutex> lock(poll_mu_);
    while (!poll_stop_) {
        poll_cv_.wait_for(lock, std::chrono::seconds(opts_.repository_poll_secs),
                          [this]() { return poll_stop_; });
        if (poll_stop_) break;
        lock.unlock();
        try {
            pollRepositoryOnce();
        } catch (const std::exception& ex) {
            diag_->error(std::string("repository poll failed: ") + ex.what());
        }
        lock.lock();
    }
}

void InferLite::pollRepositoryOnce() {
    std::error_code ec;
    if (!fs::is_directory(opts_.model_repository, ec)) return;
    std::vector<std::string> names = listRepositoryModelNames(opts_.model_repository);
    std::set<std::string> name_set(names.begin(), names.end());

    // Fingerprint the current state of every model directory.
    std::map<std::string, std::string> disk_fp;
    for (const auto& n : names) {
        try {
            disk_fp[n] = fingerprintModelDirectory(opts_.model_repository, n);
        } catch (const std::exception&) {
            disk_fp[n] = "";
        }
    }

    // Diff against the last polled state.
    std::vector<std::string> changed;
    std::vector<std::string> removed;
    for (const auto& n : names) {
        auto it = repo_fingerprints_.find(n);
        if (it == repo_fingerprints_.end() || it->second != disk_fp[n]) {
            changed.push_back(n);
        }
    }
    for (const auto& kv : repo_fingerprints_) {
        if (name_set.count(kv.first) == 0) removed.push_back(kv.first);
    }

    // Apply changes under the lifecycle lock, one model at a time.
    std::lock_guard<std::mutex> lock(lifecycle_mu_);
    for (const auto& n : changed) {
        bool was_loaded = false;
        {
            std::lock_guard<std::mutex> mlock(models_mu_);
            auto e = findEntryLocked(n);
            was_loaded = e && e->loaded();
        }
        ControlStatus st = loadModelInternal(n, "");
        if (!st.ok) {
            diag_->warn(std::string("poll ") + (was_loaded ? "reload" : "load") +
                        " of model '" + n + "' failed: " + st.error);
        } else {
            diag_->info(std::string("poll ") + (was_loaded ? "reloaded" : "loaded") +
                        " model '" + n + "'");
        }
        auto fpit = disk_fp.find(n);
        if (fpit != disk_fp.end()) repo_fingerprints_[n] = fpit->second;
    }
    for (const auto& n : removed) {
        ControlStatus st = unloadModelInternal(n, /*unload_dependents=*/true);
        repo_fingerprints_.erase(n);
        if (!st.ok) {
            diag_->warn("poll removal of model '" + n + "' failed: " + st.error);
        } else {
            diag_->info("poll removed model '" + n + "'");
        }
    }
}

// ---- inference ---------------------------------------------------------------

InferenceOutcome InferLite::runInference(const std::string& model_name,
                                         std::vector<Tensor> inputs,
                                         std::string trace_id) {
    InferenceOutcome outcome;

    // Locate the model and grab stable references under the lock; the entry may
    // be reloaded/unloaded between requests, but a held scheduler/config keeps
    // serving and is destroyed only after the last reference is dropped.
    std::shared_ptr<const ModelConfig> cfg;
    std::shared_ptr<Scheduler> sched;
    std::string entry_name, device_label;
    std::string cfg_hash, model_hash, meta_version;
    {
        std::lock_guard<std::mutex> lock(models_mu_);
        for (const auto& ep : models_) {
            if (ep->name != model_name || !ep->loaded()) continue;
            cfg = ep->config;
            sched = ep->scheduler;
            entry_name = ep->name;
            device_label = ep->device_label;
            meta_version = ep->config ? ep->config->metadata.version : std::string();
            cfg_hash = config_store_->configHash(model_name);
            model_hash = config_store_->modelHash(model_name);
            break;
        }
    }
    if (!cfg || !sched) {
        outcome.error_code = ErrorCode::kModelNotFound;
        outcome.error = "MODEL_NOT_FOUND";
        return outcome;
    }

    auto fail = [&outcome](ErrorCode code, const std::string& msg) {
        outcome.ok = false;
        outcome.error_code = code;
        outcome.error = msg;
        return outcome;
    };

    if (trace_id.empty()) trace_id = generateTraceId();
    outcome.trace_id = trace_id;

    // Structured input validation against the model spec (shape/type/size).
    ErrorCode vc = ErrorCode::kNone;
    std::string vm;
    if (!validateInputs(*cfg, inputs, limits_.max_input_size_bytes, vc, vm)) {
        return fail(vc, vm);
    }

    // Build the audit input-shape summary.
    std::string input_shape_str;
    {
        std::ostringstream ss;
        ss << '[';
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (i) ss << ',';
            for (size_t d = 0; d < inputs[i].shape.size(); ++d) {
                if (d) ss << ',';
                ss << inputs[i].shape[d];
            }
        }
        ss << ']';
        input_shape_str = ss.str();
    }

    auto ireq = std::make_shared<InferenceRequest>();
    ireq->inputs = std::move(inputs);
    ireq->timeout_ms = opts_.request_timeout_ms;

    auto start = std::chrono::steady_clock::now();
    std::shared_ptr<InferenceResult> result;
    try {
        result = sched->submit(ireq);
    } catch (const std::exception& e) {
        result = std::make_shared<InferenceResult>();
        result->ok = false;
        result->error_code = ErrorCode::kInternalError;
        result->error = e.what();
    }
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - start)
                           .count();

    // Finalize the audit entry.
    if (audit_) {
        AuditEntry ae;
        ae.trace_id = trace_id;
        ae.model_id = entry_name;
        ae.model_version = meta_version;
        ae.model_hash = model_hash;
        ae.software_version = config_store_->softwareVersion();
        ae.config_hash = cfg_hash;
        ae.duration_ms = duration_ms;
        ae.device = device_label;
        try {
            json::Value parsed_shape = json::parse(input_shape_str);
            if (parsed_shape.isArray()) {
                for (const auto& d : parsed_shape.asArray()) {
                    ae.input_shape.push_back(static_cast<int64_t>(d.asDouble()));
                }
            }
        } catch (...) {
        }
        ae.inference_status = result->ok ? "SUCCESS" : "FAILURE";
        ae.error_code = result->ok ? "" : std::string(errorCodeToString(result->error_code));
        try {
            audit_->write(ae);
        } catch (const std::exception& e) {
            diag_->error(std::string("audit log write failed: ") + e.what());
        }
    }

    if (!result->ok) {
        outcome.error_code = result->error_code;
        outcome.error = result->error;
        return outcome;
    }

    outcome.ok = true;
    outcome.device = device_label;
    outcome.outputs = std::move(result->outputs);
    return outcome;
}

// ---- HTTP dispatch -----------------------------------------------------------

HttpResponse InferLite::handleRequest(const HttpRequest& req) {
    if (req.path == "/v2/health/ready" || req.path == "/v2/health/live") {
        return handleHealthReady();
    }
    if (req.path == "/v2/health/detailed") return handleHealthDetailed();
    if (req.path == "/v2/metrics") return handleMetrics();
    if (req.path == "/v2/versions") return handleVersions();

    // --- Phase 6: repository control routes (POST only) ---
    if (req.path == "/v2/repository/index") {
        if (req.method != "POST") return jsonError(405, "Method Not Allowed");
        return handleRepositoryIndex(req);
    }
    const std::string repo_models_prefix = "/v2/repository/models/";
    if (req.path.rfind(repo_models_prefix, 0) == 0) {
        std::string rest = req.path.substr(repo_models_prefix.size());
        if (endsWith(rest, "/load")) {
            if (req.method != "POST") return jsonError(405, "Method Not Allowed");
            std::string name = rest.substr(0, rest.size() - std::string("/load").size());
            return name.empty() ? jsonError(404, "Not Found")
                                : handleRepositoryLoad(req, name);
        }
        if (endsWith(rest, "/unload")) {
            if (req.method != "POST") return jsonError(405, "Method Not Allowed");
            std::string name = rest.substr(0, rest.size() - std::string("/unload").size());
            return name.empty() ? jsonError(404, "Not Found")
                                : handleRepositoryUnload(req, name);
        }
    }

    {
        std::string name;
        if (req.path.rfind("/config") == req.path.size() - 7 &&
            extractModelName(req.path.substr(0, req.path.size() - 7), name)) {
            return handleConfig(name);
        }
    }

    if (req.method == "POST") {
        std::string name;
        if (req.path.rfind("/infer") == req.path.size() - 6 &&
            extractModelName(req.path.substr(0, req.path.size() - 6), name)) {
            return handleInfer(req, name);
        }
    }

    return jsonError(404, "Not Found");
}

HttpResponse InferLite::handleRepositoryIndex(const HttpRequest& req) {
    // Optional body: { "ready": <bool> }. Default returns all models.
    bool ready_only = false;
    if (!req.body.empty()) {
        json::Value doc;
        try {
            doc = json::parse(req.body);
        } catch (const std::exception& e) {
            return jsonError(400, std::string("invalid JSON: ") + e.what());
        }
        if (!doc.isObject()) return jsonError(400, "request body must be a JSON object");
        if (const json::Value* ready = doc.find("ready")) {
            if (!ready->isBool()) return jsonError(400, "'ready' must be a boolean");
            ready_only = ready->asBool();
        }
    }

    std::vector<ModelIndexEntry> entries = repositoryIndex(ready_only);
    json::Value arr = json::Value(json::Value::Array());
    for (const auto& e : entries) {
        json::Value obj = json::Value::Object();
        obj.asObject()["name"] = json::Value(e.name);
        obj.asObject()["version"] = json::Value(e.version);
        obj.asObject()["state"] = json::Value(e.state);
        obj.asObject()["reason"] = json::Value(e.reason);
        arr.asArray().push_back(std::move(obj));
    }
    HttpResponse resp;
    resp.status = 200;
    resp.body = arr.dump();
    return resp;
}

HttpResponse InferLite::handleRepositoryLoad(const HttpRequest& req,
                                             const std::string& model_name) {
    std::string config_override;
    if (!req.body.empty()) {
        json::Value doc;
        try {
            doc = json::parse(req.body);
        } catch (const std::exception& e) {
            return jsonError(400, std::string("invalid JSON: ") + e.what());
        }
        if (!doc.isObject()) return jsonError(400, "request body must be a JSON object");
        const json::Value* params = doc.find("parameters");
        if (params) {
            if (!params->isObject()) return jsonError(400, "'parameters' must be an object");
            if (const json::Value* cfg = params->find("config")) {
                if (!cfg->isString()) return jsonError(400, "parameter 'config' must be a string");
                config_override = cfg->asString();
            }
        }
    }
    ControlStatus st = repositoryLoad(model_name, config_override);
    HttpResponse resp;
    resp.status = st.ok ? 200 : st.http_status;
    resp.body = st.ok ? "" : jsonError(st.http_status, st.error).body;
    return resp;
}

HttpResponse InferLite::handleRepositoryUnload(const HttpRequest& req,
                                               const std::string& model_name) {
    bool unload_dependents = false;
    if (!req.body.empty()) {
        json::Value doc;
        try {
            doc = json::parse(req.body);
        } catch (const std::exception& e) {
            return jsonError(400, std::string("invalid JSON: ") + e.what());
        }
        if (!doc.isObject()) return jsonError(400, "request body must be a JSON object");
        const json::Value* params = doc.find("parameters");
        if (params) {
            if (!params->isObject()) return jsonError(400, "'parameters' must be an object");
            if (const json::Value* ud = params->find("unload_dependents")) {
                if (!ud->isBool()) {
                    return jsonError(400, "parameter 'unload_dependents' must be a boolean");
                }
                unload_dependents = ud->asBool();
            }
        }
    }
    ControlStatus st = repositoryUnload(model_name, unload_dependents);
    HttpResponse resp;
    resp.status = st.ok ? 200 : st.http_status;
    resp.body = st.ok ? "" : jsonError(st.http_status, st.error).body;
    return resp;
}

HttpResponse InferLite::handleHealthReady() {
    HttpResponse resp;
    resp.status = ready() ? 200 : 503;
    json::Value o = json::Value::Object();
    o.asObject()["status"] = json::Value(ready() ? "READY" : "NOT_READY");
    resp.body = o.dump();
    return resp;
}

HttpResponse InferLite::handleHealthDetailed() {
    HttpResponse resp;
    resp.status = 200;
    json::Value o = json::Value::Object();
    o.asObject()["overall_status"] =
        json::Value(self_test_passed_.load() ? "READY" : "DEGRADED");
    o.asObject()["software_version"] = json::Value(config_store_->softwareVersion());
    o.asObject()["openvino_version"] = json::Value(config_store_->openvinoVersion());
    o.asObject()["manifest_enabled"] = json::Value(config_store_->manifestEnabled());
    o.asObject()["model_control_mode"] = json::Value(modelControlModeName());
    o.asObject()["repository_poll_secs"] =
        json::Value(static_cast<int64_t>(opts_.repository_poll_secs));
#ifdef INFERLITE_ENABLE_GPU
    o.asObject()["gpu"] = json::Value(json::Value::Object());
    {
        json::Value& g = o.asObject()["gpu"];
        g.asObject()["enabled"] = json::Value(true);
        g.asObject()["device"] = json::Value(opts_.gpu_device);
        g.asObject()["max_gpu_memory_mb"] = json::Value(static_cast<int64_t>(opts_.max_gpu_memory_mb));
        g.asObject()["max_concurrent_gpu_instances"] =
            json::Value(static_cast<int64_t>(opts_.max_concurrent_gpu_instances));
        g.asObject()["device_pool_bytes"] = json::Value(static_cast<int64_t>(gpu_memory_->devicePoolBytes()));
        g.asObject()["pinned_pool_bytes"] = json::Value(static_cast<int64_t>(gpu_memory_->pinnedPoolBytes()));
    }
#else
    o.asObject()["gpu"] = json::Value(json::Value::Object());
    o.asObject()["gpu"].asObject()["enabled"] = json::Value(false);
    o.asObject()["gpu"].asObject()["reason"] = json::Value("not compiled in (no TensorRT SDK)");
#endif
    json::Value models = json::Value(json::Value::Array());
    {
        std::lock_guard<std::mutex> lock(models_mu_);
        for (const auto& ep : models_) {
            const auto& e = *ep;
            json::Value m = json::Value::Object();
            m.asObject()["name"] = json::Value(e.name);
            m.asObject()["state"] = json::Value(stateString(e));
            if (e.config) {
                m.asObject()["backend"] = json::Value(e.config->backend);
            }
            if (e.loaded()) {
                m.asObject()["device"] = json::Value(e.device_label);
                m.asObject()["model_hash"] = json::Value(config_store_->modelHash(e.name));
                m.asObject()["config_hash"] = json::Value(config_store_->configHash(e.name));
                m.asObject()["status"] = json::Value("READY");
            } else {
                m.asObject()["reason"] = json::Value(e.reason);
                m.asObject()["status"] = json::Value(stateString(e));
            }
            models.asArray().push_back(std::move(m));
        }
    }
    o.asObject()["models"] = std::move(models);
    resp.body = o.dump();
    return resp;
}

HttpResponse InferLite::handleVersions() {
    HttpResponse resp;
    resp.status = 200;
    json::Value o = json::Value::Object();
    o.asObject()["software_version"] = json::Value(config_store_->softwareVersion());
    o.asObject()["openvino_version"] = json::Value(config_store_->openvinoVersion());
    o.asObject()["manifest_hash"] = json::Value(
        config_store_->manifestEnabled() ? std::string("present") : std::string("absent"));
    json::Value models = json::Value(json::Value::Array());
    {
        std::lock_guard<std::mutex> lock(models_mu_);
        for (const auto& ep : models_) {
            if (!ep->loaded() || !ep->config) continue;
            json::Value m = json::Value::Object();
            m.asObject()["name"] = json::Value(ep->name);
            m.asObject()["version"] = json::Value(ep->config->metadata.version.empty()
                                                      ? std::string("unknown")
                                                      : ep->config->metadata.version);
            m.asObject()["model_hash"] = json::Value(config_store_->modelHash(ep->name));
            models.asArray().push_back(std::move(m));
        }
    }
    o.asObject()["models"] = std::move(models);
    resp.body = o.dump();
    return resp;
}

HttpResponse InferLite::handleConfig(const std::string& name) {
    std::shared_ptr<const ModelConfig> cfg;
    {
        std::lock_guard<std::mutex> lock(models_mu_);
        auto e = findEntryLocked(name);
        if (e && e->loaded() && e->config) cfg = e->config;
    }
    if (!cfg) {
        HttpResponse resp;
        resp.status = 404;
        resp.body = "{\"error\":\"MODEL_NOT_FOUND\"}";
        return resp;
    }
    const auto& c = *cfg;
    json::Value obj = json::Value::Object();
    obj.asObject()["name"] = json::Value(c.name);
    obj.asObject()["backend"] = json::Value(c.backend);
    obj.asObject()["max_batch_size"] = json::Value(c.max_batch_size);
    if (c.batching.enabled) {
        json::Value db = json::Value::Object();
        json::Value prefs = json::Value(json::Value::Array());
        for (int64_t p : c.batching.preferred_batch_size) {
            prefs.asArray().push_back(json::Value(p));
        }
        db.asObject()["preferred_batch_size"] = std::move(prefs);
        db.asObject()["max_queue_delay_microseconds"] =
            json::Value(c.batching.max_queue_delay_us);
        obj.asObject()["dynamic_batching"] = std::move(db);
    }
    json::Value ig = json::Value::Object();
    ig.asObject()["count"] = json::Value(static_cast<int64_t>(c.instance_group.count));
    ig.asObject()["kind"] = json::Value(c.instance_group.kind);
    obj.asObject()["instance_group"] = std::move(ig);
    obj.asObject()["inputs"] = json::Value(json::Value::Array());
    obj.asObject()["outputs"] = json::Value(json::Value::Array());
    auto addSpec = [](json::Value& arr, const TensorSpec& spec) {
        json::Value s = json::Value::Object();
        s.asObject()["name"] = json::Value(spec.name);
        s.asObject()["data_type"] = json::Value(dataTypeToString(spec.data_type));
        json::Value dims = json::Value(json::Value::Array());
        for (int64_t d : spec.dims) dims.asArray().push_back(json::Value(d));
        s.asObject()["dims"] = std::move(dims);
        arr.asArray().push_back(std::move(s));
    };
    for (const auto& in : c.inputs) addSpec(obj.asObject()["inputs"], in);
    for (const auto& out : c.outputs) addSpec(obj.asObject()["outputs"], out);
    if (!c.plugin_library.empty()) {
        obj.asObject()["plugin_library"] = json::Value(c.plugin_library);
    }
    if (c.backend == "ensemble") {
        json::Value steps = json::Value(json::Value::Array());
        for (const auto& st : c.ensemble_steps) {
            json::Value s = json::Value::Object();
            s.asObject()["model_name"] = json::Value(st.model_name);
            steps.asArray().push_back(std::move(s));
        }
        obj.asObject()["ensemble_steps"] = std::move(steps);
    }
    obj.asObject()["config_hash"] = json::Value(config_store_->configHash(name));
    obj.asObject()["model_hash"] = json::Value(config_store_->modelHash(name));
    HttpResponse resp;
    resp.status = 200;
    resp.body = obj.dump();
    return resp;
}

HttpResponse InferLite::handleMetrics() {
    HttpResponse resp;
    json::Value obj = json::Value::Object();
    json::Value models_arr = json::Value(json::Value::Array());
    uint64_t total_completed = 0, total_failed = 0, total_timeout = 0, total_us = 0;
    size_t total_queue = 0;

    {
        // Build the per-model and aggregate metrics under the model lock so a
        // concurrent load/unload cannot tear field reads.
        std::lock_guard<std::mutex> lock(models_mu_);
        for (const auto& ep : models_) {
            if (!ep->loaded()) continue;
            const auto& m = *ep;
            const auto& st = m.scheduler->stats();
            total_completed += st.requests_completed.load();
            total_failed += st.requests_failed.load();
            total_timeout += st.requests_timed_out.load();
            total_us += st.total_exec_us.load();
            total_queue += m.scheduler->queueDepth();

            json::Value mm = json::Value::Object();
            mm.asObject()["model_name"] = json::Value(m.name);
            mm.asObject()["device"] = json::Value(m.device_label);
            mm.asObject()["requests_completed"] = json::Value(static_cast<int64_t>(st.requests_completed.load()));
            mm.asObject()["requests_failed"] = json::Value(static_cast<int64_t>(st.requests_failed.load()));
            mm.asObject()["requests_timed_out"] = json::Value(static_cast<int64_t>(st.requests_timed_out.load()));
            mm.asObject()["average_latency_us"] = json::Value(m.scheduler->averageLatencyUs());
            mm.asObject()["queue_depth"] = json::Value(static_cast<int64_t>(m.scheduler->queueDepth()));
            if (m.scheduler->batchingEnabled()) {
                mm.asObject()["batching_enabled"] = json::Value(true);
                mm.asObject()["max_batch_size"] = json::Value(m.config ? m.config->max_batch_size
                                                                       : 0);
                mm.asObject()["batches_executed"] =
                    json::Value(static_cast<int64_t>(m.scheduler->batchesCompleted()));
                mm.asObject()["batch_samples"] =
                    json::Value(static_cast<int64_t>(m.scheduler->samplesCompleted()));
                mm.asObject()["average_batch_size"] =
                    json::Value(m.scheduler->averageBatchSize());
            }
            if (m.device_label == "GPU") {
                auto it = gpu_usage_bytes_.find(m.name);
                mm.asObject()["gpu_memory_bytes"] = json::Value(static_cast<int64_t>(
                    it != gpu_usage_bytes_.end() ? it->second.load(std::memory_order_relaxed) : 0));
            }
            models_arr.asArray().push_back(std::move(mm));
        }
    }
    double avg_us = total_completed > 0
                        ? static_cast<double>(total_us) / static_cast<double>(total_completed)
                        : 0.0;
    obj.asObject()["requests_completed"] = json::Value(static_cast<int64_t>(total_completed));
    obj.asObject()["requests_failed"] = json::Value(static_cast<int64_t>(total_failed));
    obj.asObject()["requests_timed_out"] = json::Value(static_cast<int64_t>(total_timeout));
    obj.asObject()["average_inference_latency_us"] = json::Value(avg_us);
    obj.asObject()["queue_depth"] = json::Value(static_cast<int64_t>(total_queue));
    obj.asObject()["config_hash"] = json::Value(config_store_->configStoreHash());
    obj.asObject()["models"] = std::move(models_arr);
#ifdef INFERLITE_ENABLE_GPU
    obj.asObject()["gpu_memory"] = json::Value(json::Value::Object());
    {
        json::Value& g = obj.asObject()["gpu_memory"];
        g.asObject()["device_pool_bytes"] = json::Value(static_cast<int64_t>(gpu_memory_->devicePoolBytes()));
        g.asObject()["pinned_pool_bytes"] = json::Value(static_cast<int64_t>(gpu_memory_->pinnedPoolBytes()));
    }
#endif
    resp.status = 200;
    resp.body = obj.dump();
    return resp;
}

HttpResponse InferLite::handleInfer(const HttpRequest& req, const std::string& model_name) {
    HttpResponse resp;
    resp.content_type = "application/json";

    auto makeError = [](ErrorCode code, const std::string& msg) -> HttpResponse {
        HttpResponse r;
        r.status = errorCodeToStatus(code);
        r.body = std::string("{\"error\":\"") + errorCodeToString(code) +
                 "\",\"message\":\"" + msg + "\"}";
        return r;
    };

    // Parse JSON body.
    json::Value doc;
    try {
        doc = json::parse(req.body);
    } catch (const std::exception& e) {
        return makeError(ErrorCode::kInvalidInput, std::string("invalid JSON: ") + e.what());
    }

    const json::Value* inputs_node = doc.find("inputs");
    if (!inputs_node || !inputs_node->isArray() || inputs_node->asArray().empty()) {
        return makeError(ErrorCode::kInvalidInput, "missing or empty 'inputs' array");
    }

    std::vector<Tensor> input_tensors;
    input_tensors.reserve(inputs_node->asArray().size());
    for (const auto& in : inputs_node->asArray()) {
        if (!in.isObject()) return makeError(ErrorCode::kInvalidInput, "each input must be an object");
        auto name_it = in.asObject().find("name");
        if (name_it == in.asObject().end() || !name_it->second.isString()) {
            return makeError(ErrorCode::kInvalidInput, "input missing 'name'");
        }
        const std::string& iname = name_it->second.asString();
        const json::Value* dtype_node = in.find("datatype");
        if (!dtype_node || !dtype_node->isString()) {
            return makeError(ErrorCode::kInvalidInput, "input '" + iname + "' missing 'datatype'");
        }
        DataType dt = dataTypeFromString(dtype_node->asString());
        if (dt == DataType::kInvalid) {
            return makeError(ErrorCode::kInvalidInput, "unsupported datatype: " + dtype_node->asString());
        }
        std::vector<int64_t> shape;
        const json::Value* shape_node = in.find("shape");
        if (shape_node) {
            if (shape_node->isArray()) {
                for (const auto& d : shape_node->asArray()) {
                    shape.push_back(static_cast<int64_t>(d.asDouble()));
                }
            } else if (shape_node->isNumber()) {
                shape.push_back(static_cast<int64_t>(shape_node->asDouble()));
            } else {
                return makeError(ErrorCode::kInvalidInput, "invalid shape for '" + iname + "'");
            }
        }
        std::vector<uint8_t> payload;
        const json::Value* data_node = in.find("data");
        if (data_node && data_node->isString()) {
            payload = base64Decode(data_node->asString());
        } else if (data_node && data_node->isArray()) {
            size_t elem = dataTypeSize(dt);
            if (elem == 0) return makeError(ErrorCode::kInvalidInput, "invalid datatype for array data");
            const auto& vals = data_node->asArray();
            payload.resize(vals.size() * elem);
            uint8_t* out = payload.data();
            for (size_t i = 0; i < vals.size(); ++i) {
                writeTensorScalar(out + i * elem, dt, vals[i].asDouble());
            }
        } else {
            return makeError(ErrorCode::kInvalidInput, "input '" + iname + "' missing 'data'");
        }
        Tensor t;
        t.name = iname;
        t.type = dt;
        t.shape = shape;
        t.data = std::move(payload);
        input_tensors.push_back(std::move(t));
    }

    // Shared inference core: validation + scheduling + audit + outputs.
    InferenceOutcome outcome = runInference(model_name, std::move(input_tensors),
                                            generateTraceId());
    if (!outcome.ok) {
        return makeError(outcome.error_code, outcome.error);
    }

    // Build response: outputs with name/shape/datatype/data (base64).
    json::Value resp_obj = json::Value::Object();
    json::Value outputs_arr = json::Value(json::Value::Array());
    for (const auto& out : outcome.outputs) {
        json::Value o = json::Value::Object();
        o.asObject()["name"] = json::Value(out.name);
        json::Value shape = json::Value(json::Value::Array());
        for (int64_t d : out.shape) shape.asArray().push_back(json::Value(d));
        o.asObject()["shape"] = std::move(shape);
        o.asObject()["datatype"] = json::Value(dataTypeToString(out.type));
        o.asObject()["data"] = json::Value(base64Encode(out.data.data(), out.data.size()));
        outputs_arr.asArray().push_back(std::move(o));
    }
    resp_obj.asObject()["model_name"] = json::Value(model_name);
    resp_obj.asObject()["trace_id"] = json::Value(outcome.trace_id);
    resp_obj.asObject()["outputs"] = std::move(outputs_arr);
    resp.status = 200;
    resp.body = resp_obj.dump();
    return resp;
}

}  // namespace inferlite
