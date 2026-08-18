// grpc_server.hpp - KServe / Triton v2-compatible gRPC interface.
//
// Provides the GRPCInferenceService (see proto/grpc_service.proto) as a
// synchronous unary service. Implements liveness/readiness, server metadata,
// model readiness/metadata/config, and model inference. It reuses the shared
// InferLite::runInference() core so both HTTP and gRPC apply identical
// validation, scheduling, audit, and error semantics.
//
// This file is only compiled when INFERLITE_ENABLE_GRPC is defined (see
// CMakeLists.txt); the build is opt-in against a prebuilt gRPC C++ SDK,
// mirroring the TensorRT backend pattern.
#pragma once

#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>

#include "generated/grpc_service.grpc.pb.h"
#include "generated/grpc_service.pb.h"

namespace inferlite {

class InferLite;

// Owns a gRPC server that exposes inference.GRPCInferenceService. Each RPC
// dispatches to the shared inference core. `server_` is created in start();
// the sync service methods are invoked on gRPC's internal thread pool.
class GrpcServer final : public inference::GRPCInferenceService::Service {
public:
    // `owner`: the InferLite instance backing this service (must outlive the
    // server). `port`: TCP port; 0 means an ephemeral port (use port() after
    // start()).
    explicit GrpcServer(InferLite* owner, int port);
    ~GrpcServer() override;

    GrpcServer(const GrpcServer&) = delete;
    GrpcServer& operator=(const GrpcServer&) = delete;

    // Start listening and spawn gRPC threads (non-blocking). Throws
    // std::runtime_error on bind/setup failure.
    void start();
    // Shut down the listener and wait for in-flight RPCs to drain.
    void stop();

    // Actual bound port (valid after start()).
    int port() const;

    // --- GRPCInferenceService RPC implementations -------------------------
    ::grpc::Status ServerLive(::grpc::ServerContext* context,
                              const inference::ServerLiveRequest* request,
                              inference::ServerLiveResponse* response) override;
    ::grpc::Status ServerReady(::grpc::ServerContext* context,
                               const inference::ServerReadyRequest* request,
                               inference::ServerReadyResponse* response) override;
    ::grpc::Status ServerMetadata(::grpc::ServerContext* context,
                                  const inference::ServerMetadataRequest* request,
                                  inference::ServerMetadataResponse* response) override;
    ::grpc::Status ModelReady(::grpc::ServerContext* context,
                              const inference::ModelReadyRequest* request,
                              inference::ModelReadyResponse* response) override;
    ::grpc::Status ModelMetadata(::grpc::ServerContext* context,
                                 const inference::ModelMetadataRequest* request,
                                 inference::ModelMetadataResponse* response) override;
    ::grpc::Status ModelConfig(::grpc::ServerContext* context,
                               const inference::ModelConfigRequest* request,
                               inference::ModelConfigResponse* response) override;
    ::grpc::Status ModelInfer(::grpc::ServerContext* context,
                              const inference::ModelInferRequest* request,
                              inference::ModelInferResponse* response) override;

private:
    InferLite* owner_;
    int port_;
    std::unique_ptr<grpc::Server> server_;
    int bound_port_ = 0;
};

}  // namespace inferlite
