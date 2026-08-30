// infer_lite.hpp - Top-level application: wires together the model repository,
// backends (OpenVINO / plugin / ensemble), memory manager, scheduler, audit log,
// diagnostics, config store, and HTTP server.
#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "audit_log.hpp"
#include "backend.hpp"
#include "config_store.hpp"
#include "diagnostics.hpp"
#include "http_server.hpp"
#include "memory_manager.hpp"
#include "model_repository.hpp"
#include "scheduler.hpp"
#include "validation.hpp"

#ifdef INFERLITE_ENABLE_GPU
#include "gpu_memory_manager.hpp"
#endif

namespace inferlite {

class OpenVinoBackend;
class PluginBackend;
class TensorRtBackend;
class GrpcServer;  // implemented in grpc_server.hpp (only when gRPC is enabled)

// Result of one model inference, returned by InferLite::runInference. Used by
// both the HTTP handler and (when enabled) the gRPC service so every protocol
// shares the same validation / scheduling / audit / error path (IEC 62304).
struct InferenceOutcome {
    bool ok = false;
    ErrorCode error_code = ErrorCode::kNone;
    std::string error;       // human-readable diagnostic (empty on success)
    std::vector<Tensor> outputs;
    std::string trace_id;    // audit-trail correlation id
    std::string device;      // resolved execution device (CPU / NPU / ...)
};

// Parse InferLite command-line tokens (e.g. "--model-repository=...").
// Shared by main.cpp (console) and the Windows service worker so both modes
// use an identical argument grammar. Throws std::runtime_error on bad input.
// Implemented in main.cpp.
struct ServerOptions;
ServerOptions parseServerOptions(const std::vector<std::string>& tokens);

struct ServerOptions {
    std::string model_repository;
    std::string host = "0.0.0.0";
    int http_port = 8000;
    // gRPC port. 0 (or negative) disables the gRPC listener. Only meaningful
    // when the server was built with gRPC support (INFERLITE_ENABLE_GRPC).
    int grpc_port = 0;
    size_t max_queue_size = 100;
    int64_t request_timeout_ms = 30000;
    size_t http_threads = 4;

    // --- Phase 2 (FDA) options ---
    bool validated_mode = false;        // require manifest + TLS
    std::string audit_log_path;         // empty => audit log disabled
    std::string diagnostic_log_path;    // empty => stderr only
    size_t max_input_size_bytes = 50u * 1024u * 1024u;
    size_t max_output_size_bytes = 50u * 1024u * 1024u;
    int64_t max_inference_time_ms = 5000;
    std::string tls_cert_file;          // validated mode certificate (PEM)
    std::string tls_key_file;           // validated mode key (PEM)
    std::string software_version = "InferLite 2.0.0";

    // --- Phase 3 (GPU / TensorRT) options ---
    // Per-model cap on GPU device memory (bytes). Enforced by the TensorRT
    // backend; allocation failure returns RESOURCE_EXHAUSTED.
    size_t max_gpu_memory_mb = 2048;
    // Maximum number of concurrent GPU instances (across all TensorRT models).
    size_t max_concurrent_gpu_instances = 4;
    // Optional absolute path to a TensorRT engine directory for GPU models.
    std::string gpu_device = "0";  // only device 0 supported (single GPU)
};

class InferLite {
public:
    // Scans the repository, verifies integrity, loads all backends, builds
    // ensemble DAGs, runs the startup self-test, and creates schedulers.
    // Throws std::runtime_error on any failure (fail-fast).
    explicit InferLite(const ServerOptions& opts);
    ~InferLite();

    InferLite(const InferLite&) = delete;
    InferLite& operator=(const InferLite&) = delete;

    // Start the HTTP listener (and gRPC listener if enabled) (non-blocking).
    // Throws on bind failure.
    void start();
    // Block until the process is signalled to stop (SIGINT / Ctrl+C / a
    // Windows-service stop request).
    void waitForShutdown();
    // Request a graceful shutdown. Thread-safe; waitForShutdown() returns after
    // this is called. Used by the Windows service control handler to stop the
    // server from an SCM callback thread.
    void requestStop() { running_ = false; }
    // Stop the server and clean up.
    void stop();

    // True only if startup self-tests passed and resources are healthy.
    bool ready() const {
        return running_ && !models_.empty() && self_test_passed_;
    }

    // Shared inference entry point used by the HTTP handler and the gRPC
    // service. Validates inputs against the model spec, schedules the request,
    // records the audit entry, and returns structured outputs/error. `inputs`
    // must already be parsed from the wire format; `trace_id` correlates the
    // audit entry (generate with InferLite::newTraceId() if unset).
    InferenceOutcome runInference(const std::string& model_name,
                                  std::vector<Tensor> inputs,
                                  std::string trace_id);

    // Whether the model with `name` exists and is ready to serve.
    bool modelExists(const std::string& name) const;
    // Resolved device label for a model ("CPU", "NPU", "INTEL_GPU", "AUTO",
    // "GPU"), or empty if the model is unknown.
    std::string modelDevice(const std::string& name) const;
    // All loaded model names (for repository-style enumeration).
    std::vector<std::string> modelNames() const;
    // Generate a UUID-v4-ish trace id for audit correlation.
    static std::string newTraceId();

private:
    friend class GrpcServer;  // gRPC service accesses models_/config_store_/audit_

    struct ModelEntry;  // forward declaration (full definition below)

    // Accessors for the gRPC service / tests (private; GrpcServer is a friend).
    const std::vector<ModelEntry>& models() const { return models_; }
    const ConfigStore& configStore() const { return *config_store_; }
    const ServerOptions& options() const { return opts_; }

    HttpResponse handleRequest(const HttpRequest& req);
    HttpResponse handleInfer(const HttpRequest& req, const std::string& model_name);
    HttpResponse handleConfig(const std::string& model_name);
    HttpResponse handleHealthReady();
    HttpResponse handleHealthDetailed();
    HttpResponse handleMetrics();
    HttpResponse handleVersions();

    BackendPtr makeBackend(const LoadedModel& lm);
    void runStartupSelfTest();
    // Find the scheduler for a model name, or nullptr. Thread-safe (read-only).
    const Scheduler* findScheduler(const std::string& name) const;

    ServerOptions opts_;
    std::shared_ptr<MemoryManager> memory_;
#ifdef INFERLITE_ENABLE_GPU
    std::shared_ptr<GpuMemoryManager> gpu_memory_;
#endif
    std::unique_ptr<ConfigStore> config_store_;
    std::unique_ptr<AuditLog> audit_;
    std::unique_ptr<Diagnostics> diag_;
    ResourceLimits limits_;

    struct ModelEntry {
        std::string name;
        std::shared_ptr<const ModelConfig> config;
        BackendPtr backend;
        std::shared_ptr<Scheduler> scheduler;
        std::string version_path;
        // Resolved device label for the model's backend, e.g. "CPU", "NPU",
        // "INTEL_GPU", "GPU" (TensorRT), or "AUTO". Used for audit logging and
        // health reporting.
        std::string device_label = "CPU";
    };
    std::vector<ModelEntry> models_;

    // Per-model GPU resource accounting (Phase 3): model name -> bound bytes.
    std::map<std::string, std::atomic<uint64_t>> gpu_usage_bytes_;

    std::unique_ptr<HttpServer> http_;
#ifdef INFERLITE_ENABLE_GRPC
    std::unique_ptr<GrpcServer> grpc_;
#endif
    std::atomic<bool> running_{false};
    std::atomic<bool> self_test_passed_{false};
};

}  // namespace inferlite
