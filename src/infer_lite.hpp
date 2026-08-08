// infer_lite.hpp - Top-level application: wires together the model repository,
// OpenVINO CPU backends, memory manager, scheduler, and HTTP server.
#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "http_server.hpp"
#include "memory_manager.hpp"
#include "model_repository.hpp"
#include "scheduler.hpp"

namespace inferlite {

class OpenVinoBackend;

struct ServerOptions {
    std::string model_repository;
    std::string host = "0.0.0.0";
    int http_port = 8000;
    size_t max_queue_size = 100;
    int64_t request_timeout_ms = 30000;
    size_t http_threads = 4;
};

class InferLite {
public:
    // Scans the repository, loads all OpenVINO models, and creates schedulers.
    // Throws std::runtime_error on any failure (fail-fast).
    explicit InferLite(const ServerOptions& opts);
    ~InferLite();

    InferLite(const InferLite&) = delete;
    InferLite& operator=(const InferLite&) = delete;

    // Start the HTTP listener (non-blocking). Throws on bind failure.
    void start();
    // Block until the process is signalled to stop (SIGINT / Ctrl+C).
    void waitForShutdown();
    // Stop the server and clean up.
    void stop();

private:
    HttpResponse handleRequest(const HttpRequest& req);
    HttpResponse handleInfer(const HttpRequest& req, const std::string& model_name);
    HttpResponse handleConfig(const std::string& model_name);
    HttpResponse handleHealthReady();
    HttpResponse handleMetrics();

    ServerOptions opts_;
    std::shared_ptr<MemoryManager> memory_;

    struct ModelEntry {
        std::string name;
        std::shared_ptr<const ModelConfig> config;
        std::shared_ptr<OpenVinoBackend> backend;
        std::shared_ptr<Scheduler> scheduler;
    };
    std::vector<ModelEntry> models_;

    std::unique_ptr<HttpServer> http_;
    std::atomic<bool> running_{false};
};

}  // namespace inferlite
