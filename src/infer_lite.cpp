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
#include "validation.hpp"

namespace inferlite {

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

}  // namespace

InferLite::InferLite(const ServerOptions& opts) : opts_(opts) {
    memory_ = std::make_shared<MemoryManager>();
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

    // Scan the repository (fail-fast on any error).
    std::vector<LoadedModel> loaded = scanRepository(opts_.model_repository);

    // Pass 1: instantiate non-ensemble backends (openvino, plugin). Ensembles
    // are built in pass 2 after all referenced backends exist.
    std::map<std::string, BackendPtr> backend_by_name;
    std::vector<ModelEntry> entries;

    for (auto& lm : loaded) {
        ModelEntry entry;
        entry.name = lm.config->name;
        entry.config = lm.config;
        entry.version_path = lm.version_path;

        // Register config hash and verify model file integrity against the
        // manifest (if enabled). Ensembles/plugins have no IR files to verify.
        config_store_->registerConfigHash(entry.name, lm.config_hash);
        if (lm.config->backend == "openvino") {
            config_store_->registerModelFiles(entry.name, lm.version_path);
        }

        if (lm.config->backend == "ensemble") {
            // Defer; resolved in pass 2.
            entries.push_back(std::move(entry));
            continue;
        }

        entry.backend = makeBackend(lm);
        // Phase 4: derive the execution device label for audit/health reporting.
        if (auto ovb = std::dynamic_pointer_cast<OpenVinoBackend>(entry.backend)) {
            entry.device_label = ovb->deviceLabel();
        } else {
            // Plugin / other backends run on CPU.
            entry.device_label = "CPU";
        }
        backend_by_name[entry.name] = entry.backend;
        entries.push_back(std::move(entry));
    }

    // Pass 2: build ensemble executors (provider resolves referenced backends).
    for (size_t i = 0; i < entries.size(); ++i) {
        if (entries[i].config->backend == "ensemble") {
            auto provider = [&backend_by_name](const std::string& name) -> BackendPtr {
                auto it = backend_by_name.find(name);
                return it == backend_by_name.end() ? nullptr : it->second;
            };
            entries[i].backend =
                std::make_shared<EnsembleExecutor>(entries[i].config, provider);
            backend_by_name[entries[i].name] = entries[i].backend;
        }
    }

    // Create schedulers for every model.
    for (auto& e : entries) {
        size_t instance_count =
            static_cast<size_t>(std::max<int>(1, e.config->instance_group.count));
        int64_t per_req_timeout = opts_.request_timeout_ms;
        if (e.config->max_inference_time_ms > 0) {
            per_req_timeout = e.config->max_inference_time_ms;
        }
        e.scheduler = std::make_shared<Scheduler>(e.backend, e.config, instance_count,
                                                  opts_.max_queue_size, per_req_timeout,
                                                  limits_.max_inference_time_ms, memory_);
        models_.push_back(std::move(e));
    }

    // Run the startup functional self-test for each model with a golden input.
    runStartupSelfTest();
}

InferLite::~InferLite() {
    stop();
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

    // openvino backend.
    auto backend = std::make_shared<OpenVinoBackend>();
    backend->setResourceLimits(&limits_);
    backend->load(cfg, lm.version_path);
    return backend;
}

void InferLite::runStartupSelfTest() {
    std::vector<std::shared_ptr<const ModelConfig>> configs;
    for (const auto& e : models_) configs.push_back(e.config);

    auto exec = [this](const std::shared_ptr<const ModelConfig>& cfg,
                       const std::vector<Tensor>& in,
                       std::vector<Tensor>& out) -> bool {
        for (auto& e : models_) {
            if (e.name == cfg->name) {
                auto req = std::make_shared<InferenceRequest>();
                req->inputs = in;
                req->timeout_ms = opts_.request_timeout_ms;
                auto res = e.scheduler->submit(req);
                if (res && res->ok) {
                    out = res->outputs;
                    return true;
                }
                return false;
            }
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

void InferLite::start() {
    auto handler = [this](const HttpRequest& req) { return handleRequest(req); };
    http_ = std::make_unique<HttpServer>(opts_.host, opts_.http_port, handler,
                                         opts_.http_threads);
    http_->start();
    running_ = true;
}

void InferLite::stop() {
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

HttpResponse InferLite::handleRequest(const HttpRequest& req) {
    if (req.path == "/v2/health/ready" || req.path == "/v2/health/live") {
        return handleHealthReady();
    }
    if (req.path == "/v2/health/detailed") return handleHealthDetailed();
    if (req.path == "/v2/metrics") return handleMetrics();
    if (req.path == "/v2/versions") return handleVersions();

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

    HttpResponse resp;
    resp.status = 404;
    resp.body = "{\"error\":\"Not Found\"}";
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
    json::Value models = json::Value(json::Value::Array());
    for (const auto& e : models_) {
        json::Value m = json::Value::Object();
        m.asObject()["name"] = json::Value(e.name);
        m.asObject()["backend"] = json::Value(e.config->backend);
        m.asObject()["device"] = json::Value(e.device_label);
        m.asObject()["model_hash"] = json::Value(config_store_->modelHash(e.name));
        m.asObject()["config_hash"] = json::Value(config_store_->configHash(e.name));
        m.asObject()["status"] = json::Value("READY");
        models.asArray().push_back(std::move(m));
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
    for (const auto& e : models_) {
        json::Value m = json::Value::Object();
        m.asObject()["name"] = json::Value(e.name);
        m.asObject()["version"] = json::Value(e.config->metadata.version.empty()
                                                  ? std::string("unknown")
                                                  : e.config->metadata.version);
        m.asObject()["model_hash"] = json::Value(config_store_->modelHash(e.name));
        models.asArray().push_back(std::move(m));
    }
    o.asObject()["models"] = std::move(models);
    resp.body = o.dump();
    return resp;
}

HttpResponse InferLite::handleConfig(const std::string& name) {
    HttpResponse resp;
    for (const auto& m : models_) {
        if (m.name != name) continue;
        const auto& c = m.config;
        json::Value obj = json::Value::Object();
        obj.asObject()["name"] = json::Value(c->name);
        obj.asObject()["backend"] = json::Value(c->backend);
        obj.asObject()["max_batch_size"] = json::Value(c->max_batch_size);
        json::Value ig = json::Value::Object();
        ig.asObject()["count"] = json::Value(static_cast<int64_t>(c->instance_group.count));
        ig.asObject()["kind"] = json::Value(c->instance_group.kind);
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
        for (const auto& in : c->inputs) addSpec(obj.asObject()["inputs"], in);
        for (const auto& out : c->outputs) addSpec(obj.asObject()["outputs"], out);
        if (!c->plugin_library.empty()) {
            obj.asObject()["plugin_library"] = json::Value(c->plugin_library);
        }
        if (c->backend == "ensemble") {
            json::Value steps = json::Value(json::Value::Array());
            for (const auto& st : c->ensemble_steps) {
                json::Value s = json::Value::Object();
                s.asObject()["model_name"] = json::Value(st.model_name);
                steps.asArray().push_back(std::move(s));
            }
            obj.asObject()["ensemble_steps"] = std::move(steps);
        }
        obj.asObject()["config_hash"] = json::Value(config_store_->configHash(name));
        obj.asObject()["model_hash"] = json::Value(config_store_->modelHash(name));
        resp.status = 200;
        resp.body = obj.dump();
        return resp;
    }
    resp.status = 404;
    resp.body = "{\"error\":\"MODEL_NOT_FOUND\"}";
    return resp;
}

HttpResponse InferLite::handleMetrics() {
    HttpResponse resp;
    json::Value obj = json::Value::Object();
    json::Value models_arr = json::Value(json::Value::Array());
    uint64_t total_completed = 0, total_failed = 0, total_timeout = 0, total_us = 0;
    size_t total_queue = 0;
    for (const auto& m : models_) {
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
        models_arr.asArray().push_back(std::move(mm));
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
    resp.status = 200;
    resp.body = obj.dump();
    return resp;
}

HttpResponse InferLite::handleInfer(const HttpRequest& req, const std::string& model_name) {
    HttpResponse resp;
    resp.content_type = "application/json";

    // Locate the model.
    const ModelEntry* entry = nullptr;
    for (const auto& m : models_) {
        if (m.name == model_name) {
            entry = &m;
            break;
        }
    }
    if (!entry) {
        resp.status = 404;
        resp.body = "{\"error\":\"MODEL_NOT_FOUND\"}";
        return resp;
    }

    auto makeError = [&](ErrorCode code, const std::string& msg) -> HttpResponse {
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

    // Structured input validation against the model spec (shape/type/size).
    ErrorCode vc = ErrorCode::kNone;
    std::string vm;
    if (!validateInputs(*entry->config, input_tensors, limits_.max_input_size_bytes, vc, vm)) {
        return makeError(vc, vm);
    }

    // Create the audit entry (trace_id + model + config hashes).
    std::string trace_id = generateTraceId();
    std::string input_shape_str;
    {
        std::ostringstream ss;
        ss << '[';
        for (size_t i = 0; i < input_tensors.size(); ++i) {
            if (i) ss << ',';
            for (size_t d = 0; d < input_tensors[i].shape.size(); ++d) {
                if (d) ss << ',';
                ss << input_tensors[i].shape[d];
            }
        }
        ss << ']';
        input_shape_str = ss.str();
    }

    auto ireq = std::make_shared<InferenceRequest>();
    ireq->inputs = std::move(input_tensors);
    ireq->timeout_ms = opts_.request_timeout_ms;

    auto start = std::chrono::steady_clock::now();
    std::shared_ptr<InferenceResult> result;
    try {
        result = entry->scheduler->submit(ireq);
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
        ae.model_id = entry->name;
        ae.model_version = entry->config->metadata.version;
        ae.model_hash = config_store_->modelHash(entry->name);
        ae.software_version = config_store_->softwareVersion();
        ae.config_hash = config_store_->configHash(entry->name);
        ae.duration_ms = duration_ms;
        // Phase 4: report the resolved execution device (CPU/NPU/INTEL_GPU/AUTO).
        ae.device = entry->device_label;
        // input_shape is serialized by the audit helper; pass a parsed shape.
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
        // Map result error code to a structured error response.
        ErrorCode code = result->error_code;
        return makeError(code, result->error);
    }

    // Build response: outputs with name/shape/datatype/data (base64).
    json::Value resp_obj = json::Value::Object();
    json::Value outputs_arr = json::Value(json::Value::Array());
    for (const auto& out : result->outputs) {
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
    resp_obj.asObject()["trace_id"] = json::Value(trace_id);
    resp_obj.asObject()["outputs"] = std::move(outputs_arr);
    resp.status = 200;
    resp.body = resp_obj.dump();
    return resp;
}

}  // namespace inferlite
