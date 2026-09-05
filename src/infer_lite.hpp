// infer_lite.hpp - Top-level application: wires together the model repository,
// backends (OpenVINO / plugin / ensemble), memory manager, scheduler, audit log,
// diagnostics, config store, model management, and HTTP/gRPC servers.
#pragma once

#include <atomic>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
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

// --- Triton-style model control mode ----------------------------------------
// Mirrors NVIDIA Triton's --model-control-mode. See docs/PRD-all.md and the
// Triton "Model Management" documentation for the reference semantics.
enum class ModelControlMode {
    kNone,     // load everything at startup; runtime load/unload disabled
    kPoll,     // load everything at startup; poll the repository and hot-load /
               // reload / unload on config/artifact changes
    kExplicit  // load only what --load-model names (or nothing); drive the rest
               // through the repository-control endpoints
};

// Display/parse helpers. modelControlModeFromString throws std::runtime_error
// for anything other than "none" | "poll" | "explicit".
std::string modelControlModeToString(ModelControlMode m);
ModelControlMode modelControlModeFromString(const std::string& s);

// One row of the Triton repository-index response.
struct ModelIndexEntry {
    std::string name;
    std::string version;   // highest available version as decimal; "" if none
    std::string state;     // "READY" | "UNAVAILABLE" (| "LOADING" transiently)
    std::string reason;    // empty when READY
};

// Result of a repository load/unload control request. http_status follows the
// Triton HTTP mapping (200 OK, 400 invalid/not-supported, 404 not found,
// 409 conflict).
struct ControlStatus {
    bool ok = false;
    int http_status = 400;
    std::string error;  // empty on success
};

// Read-only snapshot of one model known to the repository (loaded or not) used
// by report handlers and the gRPC service. `config` is null when the model is
// not loaded (or its config could not be parsed).
struct ModelInfo {
    std::string name;
    std::string state;       // READY / UNAVAILABLE / LOADING / UNLOADING
    std::string reason;
    std::string backend;     // parsed config backend, or "" when unloadable
    std::string device_label;
    std::string version_path;
    int64_t version = -1;
    std::string config_hash;
    std::string model_hash;
    bool ready = false;
    std::shared_ptr<const ModelConfig> config;
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

    // --- Phase 6 (Triton model management) options ---
    // Model-control mode (default "none" preserves the legacy fail-fast
    // load-everything-at-startup behavior).
    ModelControlMode model_control_mode = ModelControlMode::kNone;
    // Repository poll interval (seconds). Only meaningful with kPoll; 0 is
    // rejected when the mode is poll (a poll interval is required).
    size_t repository_poll_secs = 15;
    // Models to load at startup under kExplicit. May repeat the flag; the
    // special value "*" loads every model in the repository (and cannot be
    // combined with explicit names, matching Triton). Ignored otherwise.
    std::vector<std::string> load_models;
};

class InferLite {
public:
    // Scans the repository, verifies integrity, loads the models required by
    // the configured model-control mode, runs the startup self-test, and
    // creates schedulers. Throws std::runtime_error on any failure (fail-fast).
    explicit InferLite(const ServerOptions& opts);
    ~InferLite();

    InferLite(const InferLite&) = delete;
    InferLite& operator=(const InferLite&) = delete;

    // Start the HTTP listener (and gRPC listener if enabled) (non-blocking).
    // Also starts the repository poller when model-control mode is "poll".
    // Throws on bind failure.
    void start();
    // Block until the process is signalled to stop (SIGINT / Ctrl+C / a
    // Windows-service stop request).
    void waitForShutdown();
    // Request a graceful shutdown. Thread-safe; waitForShutdown() returns after
    // this is called. Used by the Windows service control handler to stop the
    // server from an SCM callback thread.
    void requestStop() { running_ = false; }
    // Stop the server (poller, gRPC, HTTP) and clean up.
    void stop();

    // True only if the server finished startup cleanly: the process is running
    // and every model that was required to load at startup passed its golden
    // self-test. Models loaded later through repository-control requests do
    // not affect this flag (their readiness is reported per model).
    bool ready() const { return running_ && self_test_passed_; }

    // Shared inference entry point used by the HTTP handler and the gRPC
    // service. Validates inputs against the model spec, schedules the request,
    // records the audit entry, and returns structured outputs/error. `inputs`
    // must already be parsed from the wire format; `trace_id` correlates the
    // audit entry (generate with InferLite::newTraceId() if unset).
    InferenceOutcome runInference(const std::string& model_name,
                                  std::vector<Tensor> inputs,
                                  std::string trace_id,
                                  int64_t priority = 0);

    // Whether the model with `name` exists and is currently ready to serve.
    bool modelExists(const std::string& name) const;
    // Resolved device label for a loaded model ("CPU", "NPU", "INTEL_GPU",
    // "AUTO", "GPU"), or empty if the model is unknown / not loaded.
    std::string modelDevice(const std::string& name) const;
    // All currently-loaded model names.
    std::vector<std::string> modelNames() const;
    // Generate a UUID-v4-ish trace id for audit correlation.
    static std::string newTraceId();

    // --- Phase 6: Triton-style model repository control ----------------------
    // The active model-control mode ("none" / "poll" / "explicit").
    ModelControlMode modelControlMode() const { return opts_.model_control_mode; }
    std::string modelControlModeName() const { return modelControlModeToString(opts_.model_control_mode); }

    // Repository index (Triton RepositoryIndex). Returns one entry per model
    // known to the repository (loaded or not). `ready_only` filters to models
    // currently in the READY state.
    std::vector<ModelIndexEntry> repositoryIndex(bool ready_only);

    // Repository load (Triton RepositoryModelLoad). Loads (or, when already
    // loaded, reloads) a model from the repository. `config_text_override`, when
    // non-empty, is a config.pbtxt document used instead of the repository
    // config.pbtxt. In explicit mode, dependencies of an ensemble are loaded
    // first automatically.
    ControlStatus repositoryLoad(const std::string& model_name,
                                 const std::string& config_text_override = std::string());

    // Repository unload (Triton RepositoryModelUnload). Unloads a model and, if
    // `unload_dependents` is true, also unloads loaded models that reference it
    // (e.g. ensembles composed of it).
    ControlStatus repositoryUnload(const std::string& model_name,
                                   bool unload_dependents = false);

    // Snapshot of every model known to the repository (for reporting / gRPC).
    std::vector<ModelInfo> modelInfo() const;

private:
    friend class GrpcServer;  // gRPC service uses options()/configStore()/...

    struct ModelEntry;  // forward declaration (full definition below)

    // Accessors for the gRPC service / tests (private; GrpcServer is a friend).
    const ConfigStore& configStore() const { return *config_store_; }
    const ServerOptions& options() const { return opts_; }

    HttpResponse handleRequest(const HttpRequest& req);
    HttpResponse handleInfer(const HttpRequest& req, const std::string& model_name);
    HttpResponse handleConfig(const std::string& model_name);
    HttpResponse handleHealthReady();
    HttpResponse handleHealthDetailed();
    HttpResponse handleMetrics();
    HttpResponse handleVersions();
    HttpResponse handleRepositoryIndex(const HttpRequest& req);
    HttpResponse handleRepositoryLoad(const HttpRequest& req, const std::string& model_name);
    HttpResponse handleRepositoryUnload(const HttpRequest& req, const std::string& model_name);

    // --- model lifecycle -----------------------------------------------------
    // Create a ModelEntry shell for a scanned model: copies config/version
    // fields, registers config/model hashes, and seeds the default device
    // label. Does not create the backend or scheduler.
    std::shared_ptr<ModelEntry> newEntryShell(const LoadedModel& lm);
    // Build a fully initialized ModelEntry (backend + scheduler) for a scanned
    // model. Registers config/model hashes and accounts GPU memory. Throws
    // std::runtime_error when the backend cannot be instantiated. Ensemble
    // configs are rejected here (they need committed dependencies).
    std::shared_ptr<ModelEntry> buildEntry(const LoadedModel& lm);
    // Run the golden-input functional self-test for one freshly built model.
    // Returns true when it passes (or no golden input is configured).
    bool runEntrySelfTest(const std::shared_ptr<ModelEntry>& e);
    // Load (or reload) one model. `config_override` overrides config.pbtxt.
    // Ensemble dependencies are resolved first. Never holds the model lock
    // while a backend loads, so serving continues during a reload.
    ControlStatus loadModelInternal(const std::string& model_name,
                                    const std::string& config_override);
    // Unload one model (and, with `unload_dependents`, everything referencing
    // it). When `erase_when_gone` is true (poll saw the directory disappear)
    // the model is forgotten entirely.
    ControlStatus unloadModelInternal(const std::string& model_name, bool unload_dependents);
    // Bring the known-model table in line with the repository directory: add
    // placeholders (UNAVAILABLE) for newly appearing models. Called by the
    // index/load paths in poll and explicit modes.
    void refreshKnownModelsFromDisk();

    // Poll-mode internals.
    void pollLoop();
    void pollRepositoryOnce();
    void setPollRunning(bool on);
    std::map<std::string, std::string> repo_fingerprints_;  // poll bookkeeping

    // Backend construction for the plugin backend (with manifest verification).
    BackendPtr makeBackend(const LoadedModel& lm);
    void runStartupSelfTest();

    // --- lifecycle helpers (called from buildEntry / load paths) ------------
    // Create the per-instance Scheduler for an entry (after its backend exists).
    void attachScheduler(const std::shared_ptr<ModelEntry>& e);
    // Build the ensemble executor for an entry whose referenced models are
    // already committed/loaded. On success sets e->backend; on failure appends
    // the unresolved dependency names to `unresolved`.
    void attachEnsembleBackend(const std::shared_ptr<ModelEntry>& e,
                               std::vector<std::string>& unresolved);
    BackendPtr backendByName(const std::string& name);
    // Names of loaded models whose config references `name` (ensemble steps).
    std::vector<std::string> loadedDependents(const std::string& name) const;
    // Commit `fresh` into the model table (replace or insert) under models_mu_.
    void commitEntry(const std::shared_ptr<ModelEntry>& fresh);
    // Recursive unload used by unloadModelInternal (lifecycle_mu_ held).
    ControlStatus unloadOne(const std::string& name, bool unload_dependents,
                            std::vector<std::string>& visited);
    // Force model `name` back to the unavailable state (used on load failure)
    // without touching an existing loaded instance when one is present.
    void markUnavailableLocked(const std::string& name, const std::string& reason);

    // Lookup helpers. Entry fields are guarded by models_mu_; findEntryLocked
    // and stateString must only be called while models_mu_ is held.
    std::shared_ptr<ModelEntry> findEntryLocked(const std::string& name) const;
    std::string stateString(const ModelEntry& e) const;

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
        int64_t version = -1;
        std::string config_hash;   // SHA-256 of the applied config.pbtxt text
        // Resolved device label for the model's backend, e.g. "CPU", "NPU",
        // "INTEL_GPU", "GPU" (TensorRT), or "AUTO". Used for audit logging and
        // health reporting. Empty when the model is not loaded.
        std::string device_label;
        // Phase 6: lifecycle state used by the repository index / control API.
        enum class State : int { kUnavailable, kLoading, kReady, kUnloading };
        State state = State::kUnavailable;
        std::string reason;  // why the model is UNAVAILABLE ("" when READY)

        bool loaded() const { return state == State::kReady && scheduler != nullptr; }
    };
    // All models known to the repository. Each entry is owned by a shared_ptr
    // so request handlers may keep a copy across an unload/reload (the old
    // scheduler/backend is destroyed only once the last in-flight request
    // drops its reference).
    mutable std::mutex models_mu_;
    std::vector<std::shared_ptr<ModelEntry>> models_;

    // Per-model GPU resource accounting (Phase 3): model name -> bound bytes.
    std::map<std::string, std::atomic<uint64_t>> gpu_usage_bytes_;

    // Serializes load/unload/reload transitions (including poll-driven ones) so
    // two control operations on the same model cannot interleave.
    std::mutex lifecycle_mu_;

    // Repository poller (only when model_control_mode == kPoll).
    std::mutex poll_mu_;
    std::condition_variable poll_cv_;
    bool poll_stop_ = false;
    std::thread poll_thread_;

    std::unique_ptr<HttpServer> http_;
#ifdef INFERLITE_ENABLE_GRPC
    std::unique_ptr<GrpcServer> grpc_;
#endif
    std::atomic<bool> running_{false};
    std::atomic<bool> self_test_passed_{false};
};

}  // namespace inferlite
