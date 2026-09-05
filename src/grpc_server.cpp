// grpc_server.cpp - KServe / Triton v2-compatible gRPC interface.
#include "grpc_server.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "infer_lite.hpp"
#include "tensor.hpp"

namespace inferlite {

namespace {

// Map an internal ErrorCode to a gRPC status code (kept parallel to the HTTP
// status mapping in infer_lite.cpp so both protocols agree on error taxonomy).
::grpc::StatusCode errorCodeToGrpc(ErrorCode c) {
    switch (c) {
        case ErrorCode::kInvalidInput: return ::grpc::StatusCode::INVALID_ARGUMENT;
        case ErrorCode::kModelNotFound: return ::grpc::StatusCode::NOT_FOUND;
        case ErrorCode::kResourceExhausted: return ::grpc::StatusCode::RESOURCE_EXHAUSTED;
        case ErrorCode::kTimeout: return ::grpc::StatusCode::DEADLINE_EXCEEDED;
        case ErrorCode::kSelfTestFailed: return ::grpc::StatusCode::UNAVAILABLE;
        default: return ::grpc::StatusCode::INTERNAL;
    }
}

// Map a repository-control result to a gRPC status (kept parallel to the HTTP
// status mapping in infer_lite.cpp so both protocols agree on error taxonomy).
::grpc::Status controlStatusToGrpc(const ControlStatus& st) {
    if (st.ok) return ::grpc::Status::OK;
    switch (st.http_status) {
        case 404: return ::grpc::Status(::grpc::StatusCode::NOT_FOUND, st.error);
        case 409: return ::grpc::Status(::grpc::StatusCode::FAILED_PRECONDITION, st.error);
        case 500: return ::grpc::Status(::grpc::StatusCode::INTERNAL, st.error);
        default: return ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT, st.error);
    }
}

// Convert the typed `contents` of a KServe InferInputTensor into a raw
// little-endian byte payload for the given DataType. Returns false and sets
// `err` if the contents are missing or inconsistent with the datatype.
bool contentsToBytes(const inference::InferTensorContents& c, DataType dt,
                     std::vector<uint8_t>& out, std::string& err) {
    size_t elem = dataTypeSize(dt);
    if (elem == 0) {
        err = "unsupported datatype";
        return false;
    }
    // Determine how many scalars are present (exactly one repeated field may
    // be populated for a given datatype).
    size_t n = 0;
    const void* src = nullptr;
    if (dt == DataType::kBool) {
        n = c.bool_contents_size();
        src = c.bool_contents().data();
    } else if (dt == DataType::kInt8 || dt == DataType::kInt16 || dt == DataType::kInt32) {
        n = c.int_contents_size();
        src = c.int_contents().data();
    } else if (dt == DataType::kInt64) {
        n = c.int64_contents_size();
        src = c.int64_contents().data();
    } else if (dt == DataType::kUint8 || dt == DataType::kUint16 || dt == DataType::kUint32) {
        n = c.uint_contents_size();
        src = c.uint_contents().data();
    } else if (dt == DataType::kUint64) {
        n = c.uint64_contents_size();
        src = c.uint64_contents().data();
    } else if (dt == DataType::kFloat32) {
        n = c.fp32_contents_size();
        src = c.fp32_contents().data();
    } else if (dt == DataType::kFloat64) {
        n = c.fp64_contents_size();
        src = c.fp64_contents().data();
    } else {
        err = "datatype requires raw binary payload (unsupported by contents)";
        return false;
    }
    if (n == 0 || src == nullptr) {
        err = "input contents empty";
        return false;
    }
    out.resize(n * elem);
    if (dt == DataType::kBool) {
        const bool* p = static_cast<const bool*>(src);
        for (size_t i = 0; i < n; ++i) out[i] = p[i] ? 1 : 0;
    } else if (dt == DataType::kInt8 || dt == DataType::kInt16 || dt == DataType::kInt32) {
        const int32_t* p = static_cast<const int32_t*>(src);
        for (size_t i = 0; i < n; ++i) writeTensorScalar(out.data() + i * elem, dt, p[i]);
    } else if (dt == DataType::kInt64) {
        const int64_t* p = static_cast<const int64_t*>(src);
        for (size_t i = 0; i < n; ++i) writeTensorScalar(out.data() + i * elem, dt,
                                                         static_cast<double>(p[i]));
    } else if (dt == DataType::kUint8 || dt == DataType::kUint16 || dt == DataType::kUint32) {
        const uint32_t* p = static_cast<const uint32_t*>(src);
        for (size_t i = 0; i < n; ++i) writeTensorScalar(out.data() + i * elem, dt, p[i]);
    } else if (dt == DataType::kUint64) {
        const uint64_t* p = static_cast<const uint64_t*>(src);
        for (size_t i = 0; i < n; ++i) writeTensorScalar(out.data() + i * elem, dt,
                                                         static_cast<double>(p[i]));
    } else if (dt == DataType::kFloat32) {
        const float* p = static_cast<const float*>(src);
        for (size_t i = 0; i < n; ++i) writeTensorScalar(out.data() + i * elem, dt, p[i]);
    } else if (dt == DataType::kFloat64) {
        const double* p = static_cast<const double*>(src);
        for (size_t i = 0; i < n; ++i) writeTensorScalar(out.data() + i * elem, dt, p[i]);
    }
    return true;
}

// Fill the typed `contents` of a KServe InferOutputTensor from a raw Tensor.
void bytesToContents(const Tensor& t, inference::InferTensorContents& c) {
    const uint8_t* data = t.data.data();
    size_t elem = dataTypeSize(t.type);
    size_t n = elem ? t.data.size() / elem : 0;
    auto read = [&](size_t i) {
        double v = 0.0;
        // readTensorScalar is in validation.hpp; inline a tiny reader here to
        // avoid a heavyweight include. All InferLite types are IEEE/little-endian.
        const uint8_t* p = data + i * elem;
        switch (t.type) {
            case DataType::kInt8:
            case DataType::kUint8:
            case DataType::kBool: v = static_cast<double>(*p); break;
            case DataType::kInt16: case DataType::kUint16: {
                uint16_t u;
                std::memcpy(&u, p, 2);
                v = static_cast<double>(static_cast<int16_t>(u));
                break;
            }
            case DataType::kInt32: case DataType::kUint32: {
                uint32_t u;
                std::memcpy(&u, p, 4);
                v = static_cast<double>(static_cast<int32_t>(u));
                break;
            }
            case DataType::kInt64: case DataType::kUint64: {
                uint64_t u;
                std::memcpy(&u, p, 8);
                v = static_cast<double>(static_cast<int64_t>(u));
                break;
            }
            case DataType::kFloat32: {
                float f;
                std::memcpy(&f, p, 4);
                v = static_cast<double>(f);
                break;
            }
            case DataType::kFloat64: {
                double d;
                std::memcpy(&d, p, 8);
                v = d;
                break;
            }
            default: break;
        }
        return v;
    };
    switch (t.type) {
        case DataType::kBool:
            for (size_t i = 0; i < n; ++i) c.add_bool_contents(read(i) != 0.0);
            break;
        case DataType::kInt8:
        case DataType::kInt16:
        case DataType::kInt32:
            for (size_t i = 0; i < n; ++i) c.add_int_contents(static_cast<int32_t>(read(i)));
            break;
        case DataType::kInt64:
            for (size_t i = 0; i < n; ++i) c.add_int64_contents(static_cast<int64_t>(read(i)));
            break;
        case DataType::kUint8:
        case DataType::kUint16:
        case DataType::kUint32:
            for (size_t i = 0; i < n; ++i) c.add_uint_contents(static_cast<uint32_t>(read(i)));
            break;
        case DataType::kUint64:
            for (size_t i = 0; i < n; ++i) c.add_uint64_contents(static_cast<uint64_t>(read(i)));
            break;
        case DataType::kFloat32:
            for (size_t i = 0; i < n; ++i) c.add_fp32_contents(static_cast<float>(read(i)));
            break;
        case DataType::kFloat64:
            for (size_t i = 0; i < n; ++i) c.add_fp64_contents(read(i));
            break;
        default:
            break;
    }
}

}  // namespace

GrpcServer::GrpcServer(InferLite* owner, std::string host, int port)
    : owner_(owner), host_(std::move(host)), port_(port) {}

GrpcServer::~GrpcServer() {
    stop();
}

void GrpcServer::start() {
    grpc::ServerBuilder builder;
    // 64 MiB receive cap: headroom above the 50 MB default input-size limit so
    // any request accepted by the inference core fits on the wire.
    builder.SetMaxReceiveMessageSize(64 * 1024 * 1024);
    const std::string addr = (host_.empty() ? std::string("0.0.0.0") : host_) + ":" +
                             std::to_string(port_);
    builder.AddListeningPort(addr, grpc::InsecureServerCredentials(), &bound_port_);
    builder.RegisterService(this);
    server_ = builder.BuildAndStart();
    if (!server_) {
        throw std::runtime_error("gRPC server failed to start on port " +
                                 std::to_string(port_));
    }
}

void GrpcServer::stop() {
    if (server_) {
        server_->Shutdown();
        server_->Wait();
        server_.reset();
    }
}

int GrpcServer::port() const {
    return bound_port_ > 0 ? bound_port_ : port_;
}

::grpc::Status GrpcServer::ServerLive(::grpc::ServerContext*,
                                      const inference::ServerLiveRequest*,
                                      inference::ServerLiveResponse* response) {
    // The server is "live" whenever the process is running (accepting RPCs).
    response->set_live(true);
    return ::grpc::Status::OK;
}

::grpc::Status GrpcServer::ServerReady(::grpc::ServerContext*,
                                       const inference::ServerReadyRequest*,
                                       inference::ServerReadyResponse* response) {
    response->set_ready(owner_->ready());
    return ::grpc::Status::OK;
}

::grpc::Status GrpcServer::ServerMetadata(::grpc::ServerContext*,
                                          const inference::ServerMetadataRequest*,
                                          inference::ServerMetadataResponse* response) {
    response->set_name("InferLite");
    response->set_version(owner_->options().software_version);
    // Advertised protocol extensions this build supports.
    response->add_extensions("binary_tensor_data");
    response->add_extensions("model_repository");
    response->add_extensions("model_repository(unload_dependents)");
    return ::grpc::Status::OK;
}

::grpc::Status GrpcServer::ModelReady(::grpc::ServerContext*,
                                      const inference::ModelReadyRequest* request,
                                      inference::ModelReadyResponse* response) {
    const std::string& name = request->name();
    bool found = false;
    bool ready = false;
    for (const auto& mi : owner_->modelInfo()) {
        if (mi.name != name) continue;
        found = true;
        ready = mi.ready;
        break;
    }
    if (!found) {
        return ::grpc::Status(::grpc::StatusCode::NOT_FOUND,
                              "model '" + name + "' not found in the repository");
    }
    response->set_ready(ready);
    return ::grpc::Status::OK;
}

::grpc::Status GrpcServer::ModelMetadata(::grpc::ServerContext*,
                                         const inference::ModelMetadataRequest* request,
                                         inference::ModelMetadataResponse* response) {
    const std::string& name = request->name();
    for (const auto& mi : owner_->modelInfo()) {
        if (mi.name != name) continue;
        if (!mi.ready || !mi.config) {
            return ::grpc::Status(::grpc::StatusCode::NOT_FOUND,
                                  "model '" + name + "' is not loaded");
        }
        const auto& cfg = *mi.config;
        response->set_name(name);
        response->set_versions(cfg.metadata.version.empty() ? "unknown" : cfg.metadata.version);
        response->set_platform(cfg.backend);
        auto add_spec = [&](const TensorSpec& spec, bool is_input) {
            inference::ModelMetadataResponse_TensorMetadata* md =
                is_input ? response->add_inputs() : response->add_outputs();
            md->set_name(spec.name);
            md->set_datatype(dataTypeToString(spec.data_type));
            for (int64_t d : spec.dims) md->add_shape(d);
        };
        for (const auto& in : cfg.inputs) add_spec(in, true);
        for (const auto& out : cfg.outputs) add_spec(out, false);
        return ::grpc::Status::OK;
    }
    return ::grpc::Status(::grpc::StatusCode::NOT_FOUND,
                          "model '" + name + "' is not loaded");
}

::grpc::Status GrpcServer::ModelConfig(::grpc::ServerContext*,
                                       const inference::ModelConfigRequest* request,
                                       inference::ModelConfigResponse* response) {
    const std::string& name = request->name();
    for (const auto& mi : owner_->modelInfo()) {
        if (mi.name != name) continue;
        if (!mi.ready || !mi.config) {
            return ::grpc::Status(::grpc::StatusCode::NOT_FOUND,
                                  "model '" + name + "' is not loaded");
        }
        const auto& cfg = *mi.config;
        response->set_name(name);
        response->set_version(cfg.metadata.version.empty() ? "unknown" : cfg.metadata.version);
        // A compact human-readable summary of the model configuration.
        std::string s = "name: " + cfg.name + "\nbackend: " + cfg.backend +
                        "\nmax_batch_size: " + std::to_string(cfg.max_batch_size) +
                        "\ninstance_group: { count: " +
                        std::to_string(cfg.instance_group.count) +
                        ", kind: " + cfg.instance_group.kind + " }\n";
        for (const auto& in : cfg.inputs) {
            s += "input: { name: " + in.name + ", datatype: " +
                 dataTypeToString(in.data_type) + " }\n";
        }
        for (const auto& out : cfg.outputs) {
            s += "output: { name: " + out.name + ", datatype: " +
                 dataTypeToString(out.data_type) + " }\n";
        }
        s += "config_hash: " + owner_->configStore().configHash(name) + "\n";
        s += "model_hash: " + owner_->configStore().modelHash(name) + "\n";
        response->set_config(s);
        return ::grpc::Status::OK;
    }
    return ::grpc::Status(::grpc::StatusCode::NOT_FOUND,
                          "model '" + name + "' is not loaded");
}

::grpc::Status GrpcServer::ModelInfer(::grpc::ServerContext*,
                                      const inference::ModelInferRequest* request,
                                      inference::ModelInferResponse* response) {
    const std::string& model_name = request->model_name();
    if (!owner_->modelExists(model_name)) {
        return ::grpc::Status(::grpc::StatusCode::NOT_FOUND,
                              "model '" + model_name + "' is not loaded");
    }

    // Convert KServe InferInputTensor -> internal Tensor (raw byte payload).
    std::vector<Tensor> inputs;
    inputs.reserve(request->inputs_size());
    for (const auto& in : request->inputs()) {
        Tensor t;
        t.name = in.name();
        t.type = dataTypeFromString(in.datatype());
        if (t.type == DataType::kInvalid) {
            return ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT,
                                  "input '" + in.name() + "' unsupported datatype: " +
                                      in.datatype());
        }
        for (int64_t d : in.shape()) t.shape.push_back(d);
        std::string err;
        if (!contentsToBytes(in.contents(), t.type, t.data, err)) {
            return ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT,
                                  "input '" + in.name() + "': " + err);
        }
        inputs.push_back(std::move(t));
    }

    // Shared core: validation + scheduling + audit + outputs.
    InferenceOutcome outcome =
        owner_->runInference(model_name, std::move(inputs), request->id());

    if (!outcome.ok) {
        return ::grpc::Status(errorCodeToGrpc(outcome.error_code), outcome.error);
    }

    // Serialize outputs into KServe InferOutputTensor.
    response->set_model_name(model_name);
    response->set_id(outcome.trace_id);
    for (const auto& out : outcome.outputs) {
        inference::ModelInferResponse_InferOutputTensor* o = response->add_outputs();
        o->set_name(out.name);
        o->set_datatype(dataTypeToString(out.type));
        for (int64_t d : out.shape) o->add_shape(d);
        bytesToContents(out, *o->mutable_contents());
    }
    return ::grpc::Status::OK;
}

::grpc::Status GrpcServer::RepositoryIndex(::grpc::ServerContext*,
                                           const inference::RepositoryIndexRequest* request,
                                           inference::RepositoryIndexResponse* response) {
    (void)request->repository_name();  // single-repository server
    for (const auto& e : owner_->repositoryIndex(request->ready())) {
        auto* mi = response->add_models();
        mi->set_name(e.name);
        mi->set_version(e.version);
        mi->set_state(e.state);
        mi->set_reason(e.reason);
    }
    return ::grpc::Status::OK;
}

::grpc::Status GrpcServer::RepositoryModelLoad(::grpc::ServerContext*,
                                               const inference::RepositoryModelLoadRequest* request,
                                               inference::RepositoryModelLoadResponse*) {
    const std::string& name = request->model_name();
    if (name.empty()) {
        return ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT,
                              "model name must not be empty");
    }
    std::string config_override;
    for (const auto& kv : request->parameters()) {
        if (kv.first == "config") {
            if (!kv.second.has_string_param()) {
                return ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT,
                                      "parameter 'config' must be a string");
            }
            config_override = kv.second.string_param();
        } else if (kv.first.rfind("file:", 0) == 0) {
            return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED,
                                  "inline file overrides ('" + kv.first +
                                      "') are not supported by InferLite");
        }
        // Unknown parameters are ignored (matches Triton's forward-compatible
        // handling of unsupported load parameters).
    }
    return controlStatusToGrpc(owner_->repositoryLoad(name, config_override));
}

::grpc::Status GrpcServer::RepositoryModelUnload(::grpc::ServerContext*,
                                                 const inference::RepositoryModelUnloadRequest* request,
                                                 inference::RepositoryModelUnloadResponse*) {
    const std::string& name = request->model_name();
    if (name.empty()) {
        return ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT,
                              "model name must not be empty");
    }
    bool unload_dependents = false;
    for (const auto& kv : request->parameters()) {
        if (kv.first == "unload_dependents") {
            if (!kv.second.has_bool_param()) {
                return ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT,
                                      "parameter 'unload_dependents' must be a boolean");
            }
            unload_dependents = kv.second.bool_param();
        }
    }
    return controlStatusToGrpc(owner_->repositoryUnload(name, unload_dependents));
}

}  // namespace inferlite
