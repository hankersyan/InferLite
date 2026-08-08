#include "infer_lite.hpp"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstring>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "json.hpp"
#include "model_repository.hpp"
#include "openvino_backend.hpp"
#include "pbtxt.hpp"

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
        if (v < 0) continue;  // ignore whitespace / invalid chars
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

// Write a single scalar value (from a JSON double) into the tensor's
// little-endian binary layout, per the declared DataType.
void writeScalar(uint8_t* dst, DataType dt, double v) {
    auto putInt = [&](int64_t x) {
        uint64_t uv = static_cast<uint64_t>(x);
        for (size_t b = 0; b < dataTypeSize(dt); ++b) {
            dst[b] = static_cast<uint8_t>((uv >> (8 * b)) & 0xFF);
        }
    };
    switch (dt) {
        case DataType::kInt8: putInt(static_cast<int64_t>(v)); break;
        case DataType::kUint8: putInt(static_cast<int64_t>(v)); break;
        case DataType::kInt16: putInt(static_cast<int64_t>(v)); break;
        case DataType::kUint16: putInt(static_cast<int64_t>(v)); break;
        case DataType::kInt32: putInt(static_cast<int64_t>(v)); break;
        case DataType::kUint32: putInt(static_cast<int64_t>(v)); break;
        case DataType::kInt64: putInt(static_cast<int64_t>(v)); break;
        case DataType::kUint64: putInt(static_cast<int64_t>(v)); break;
        case DataType::kFloat16: {
            // Convert double -> half (round-to-nearest via float).
            float fv = static_cast<float>(v);
            uint32_t fbits;
            std::memcpy(&fbits, &fv, sizeof(fbits));
            uint16_t h = static_cast<uint16_t>((fbits + 0x1000) >> 13);
            dst[0] = static_cast<uint8_t>(h & 0xFF);
            dst[1] = static_cast<uint8_t>((h >> 8) & 0xFF);
            break;
        }
        case DataType::kFloat32: {
            float fv = static_cast<float>(v);
            uint32_t bits;
            std::memcpy(&bits, &fv, sizeof(bits));
            dst[0] = static_cast<uint8_t>(bits & 0xFF);
            dst[1] = static_cast<uint8_t>((bits >> 8) & 0xFF);
            dst[2] = static_cast<uint8_t>((bits >> 16) & 0xFF);
            dst[3] = static_cast<uint8_t>((bits >> 24) & 0xFF);
            break;
        }
        case DataType::kFloat64: {
            uint64_t bits;
            std::memcpy(&bits, &v, sizeof(bits));
            for (size_t b = 0; b < 8; ++b) {
                dst[b] = static_cast<uint8_t>((bits >> (8 * b)) & 0xFF);
            }
            break;
        }
        case DataType::kBool:
            dst[0] = v != 0.0 ? 1 : 0;
            break;
        default:
            break;
    }
}

}  // namespace

InferLite::InferLite(const ServerOptions& opts) : opts_(opts) {
    memory_ = std::make_shared<MemoryManager>();

    // Scan the repository (fail-fast on any error).
    std::vector<LoadedModel> loaded = scanRepository(opts_.model_repository);

    for (auto& lm : loaded) {
        ModelEntry entry;
        entry.name = lm.config->name;
        entry.config = lm.config;

        auto backend = std::make_shared<OpenVinoBackend>();
        backend->load(*lm.config, lm.version_path);

        entry.backend = backend;

        size_t instance_count =
            static_cast<size_t>(std::max<int>(1, lm.config->instance_group.count));
        auto scheduler = std::make_shared<Scheduler>(backend, lm.config, instance_count,
                                                     opts_.max_queue_size,
                                                     opts_.request_timeout_ms, memory_);
        entry.scheduler = scheduler;
        models_.push_back(std::move(entry));
    }
}

InferLite::~InferLite() {
    stop();
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
    // Schedulers are stopped by their destructors when models_ is destroyed.
}

void InferLite::waitForShutdown() {
    // Simple blocking loop; interrupted by Ctrl+C via the console handler.
    // We just sleep; on Windows the process is terminated on Ctrl+C anyway.
    while (running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

HttpResponse InferLite::handleRequest(const HttpRequest& req) {
    // Health endpoint.
    if (req.path == "/v2/health/ready") return handleHealthReady();
    if (req.path == "/v2/health/live") return handleHealthReady();

    // Metrics endpoint.
    if (req.path == "/v2/metrics") return handleMetrics();

    // Config endpoint: /v2/models/<name>/config
    {
        std::string name;
        if (req.path.rfind("/config") == req.path.size() - 7 &&
            extractModelName(req.path.substr(0, req.path.size() - 7), name)) {
            return handleConfig(name);
        }
    }

    // Inference endpoint: POST /v2/models/<name>/infer
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
    resp.status = running_ && !models_.empty() ? 200 : 503;
    resp.body = "{}";
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
        resp.status = 200;
        resp.body = obj.dump();
        return resp;
    }
    resp.status = 404;
    resp.body = "{\"error\":\"model not found\"}";
    return resp;
}

HttpResponse InferLite::handleMetrics() {
    HttpResponse resp;
    json::Value obj = json::Value::Object();
    obj.asObject()["requests_completed"] = json::Value(int64_t(0));
    obj.asObject()["requests_failed"] = json::Value(int64_t(0));
    obj.asObject()["requests_timed_out"] = json::Value(int64_t(0));
    obj.asObject()["average_inference_latency_us"] = json::Value(int64_t(0));
    obj.asObject()["queue_depth"] = json::Value(int64_t(0));
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
    }
    double avg_us = total_completed > 0
                        ? static_cast<double>(total_us) / static_cast<double>(total_completed)
                        : 0.0;
    obj.asObject()["requests_completed"] = json::Value(static_cast<int64_t>(total_completed));
    obj.asObject()["requests_failed"] = json::Value(static_cast<int64_t>(total_failed));
    obj.asObject()["requests_timed_out"] = json::Value(static_cast<int64_t>(total_timeout));
    obj.asObject()["average_inference_latency_us"] = json::Value(avg_us);
    obj.asObject()["queue_depth"] = json::Value(static_cast<int64_t>(total_queue));
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
        resp.body = "{\"error\":\"model not found\"}";
        return resp;
    }

    // Parse JSON body.
    json::Value doc;
    try {
        doc = json::parse(req.body);
    } catch (const std::exception& e) {
        resp.status = 400;
        resp.body = std::string("{\"error\":\"invalid JSON: ") + e.what() + "\"}";
        return resp;
    }

    auto makeError = [&](const std::string& msg) -> HttpResponse {
        HttpResponse r;
        r.status = 400;
        r.body = "{\"error\":\"" + msg + "\"}";
        return r;
    };

    // Parse "inputs": array of {name, shape, datatype, data}
    const json::Value* inputs_node = doc.find("inputs");
    if (!inputs_node || !inputs_node->isArray() || inputs_node->asArray().empty()) {
        return makeError("missing or empty 'inputs' array");
    }

    std::vector<Tensor> input_tensors;
    input_tensors.reserve(inputs_node->asArray().size());
    for (const auto& in : inputs_node->asArray()) {
        if (!in.isObject()) return makeError("each input must be an object");
        const std::string* iname = nullptr;
        const json::Value* shape_node = in.find("shape");
        const json::Value* dtype_node = in.find("datatype");
        const json::Value* data_node = in.find("data");
        auto name_it = in.asObject().find("name");
        if (name_it == in.asObject().end() || !name_it->second.isString()) {
            return makeError("input missing 'name'");
        }
        iname = &name_it->second.asString();
        if (!dtype_node || !dtype_node->isString()) return makeError("input missing 'datatype'");
        DataType dt = dataTypeFromString(dtype_node->asString());
        if (dt == DataType::kInvalid) {
            return makeError("unsupported datatype: " + dtype_node->asString());
        }
        std::vector<int64_t> shape;
        if (shape_node) {
            if (shape_node->isArray()) {
                for (const auto& d : shape_node->asArray()) {
                    shape.push_back(static_cast<int64_t>(d.asDouble()));
                }
            } else if (shape_node->isNumber()) {
                shape.push_back(static_cast<int64_t>(shape_node->asDouble()));
            } else {
                return makeError("invalid shape");
            }
        }
        // Data may be a base64 string or a JSON array of numbers.
        std::vector<uint8_t> payload;
        if (data_node && data_node->isString()) {
            payload = base64Decode(data_node->asString());
        } else if (data_node && data_node->isArray()) {
            // Serialize each JSON number into the tensor's binary layout
            // (little-endian), respecting the declared datatype.
            size_t elem = dataTypeSize(dt);
            if (elem == 0) return makeError("invalid datatype for array data");
            const auto& vals = data_node->asArray();
            payload.resize(vals.size() * elem);
            uint8_t* out = payload.data();
            for (size_t i = 0; i < vals.size(); ++i) {
                double dv = vals[i].asDouble();
                writeScalar(out + i * elem, dt, dv);
            }
        } else {
            return makeError("input missing 'data'");
        }
        Tensor t;
        t.name = *iname;
        t.type = dt;
        t.shape = shape;
        t.data = std::move(payload);
        input_tensors.push_back(std::move(t));
    }

    // Enqueue and wait.
    auto ireq = std::make_shared<InferenceRequest>();
    ireq->inputs = std::move(input_tensors);
    ireq->timeout_ms = opts_.request_timeout_ms;

    std::shared_ptr<InferenceResult> result;
    try {
        result = entry->scheduler->submit(ireq);
    } catch (const std::exception& e) {
        resp.status = 503;
        resp.body = "{\"error\":\"" + std::string(e.what()) + "\"}";
        return resp;
    }

    if (!result->ok) {
        resp.status = 500;
        resp.body = "{\"error\":\"" + result->error + "\"}";
        return resp;
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
    resp_obj.asObject()["outputs"] = std::move(outputs_arr);
    resp.status = 200;
    resp.body = resp_obj.dump();
    return resp;
}

}  // namespace inferlite
