// scheduler.hpp - Bounded FIFO request scheduler with CPU instance groups.
//
// Maintains a bounded FIFO queue of inference requests. One worker thread per
// CPU instance consumes requests from the queue, so at most `instance_count`
// requests are in-flight simultaneously. Requests beyond the current in-flight
// capacity wait in the queue. A configurable request timeout aborts a request
// that has waited too long in the queue. A hard per-request inference time
// limit (MAX_INFERENCE_TIME_MS) aborts a request that runs too long.
#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "backend.hpp"
#include "memory_manager.hpp"
#include "tensor.hpp"

namespace inferlite {

struct ModelConfig;

struct InferenceResult {
    bool ok = false;
    ErrorCode error_code = ErrorCode::kNone;
    std::string error;
    std::vector<Tensor> outputs;
    int64_t inference_us = 0;
};

struct SchedulerStats {
    std::atomic<uint64_t> requests_completed{0};
    std::atomic<uint64_t> requests_failed{0};
    std::atomic<uint64_t> requests_timed_out{0};
    // Total inference time (excluding queue wait) in microseconds.
    std::atomic<uint64_t> total_exec_us{0};
};

// A single request enqueued by an HTTP handler. The handler blocks on the
// completion promise until a worker produces the result.
struct InferenceRequest {
    std::vector<Tensor> inputs;
    // How long (in milliseconds) this request may wait in the queue before the
    // scheduler rejects it with a timeout.
    int64_t timeout_ms = 30000;

    // Completion channel.
    std::mutex m;
    std::condition_variable cv;
    bool done = false;
    InferenceResult result;
};

class Scheduler {
public:
    // max_queue_size: hard cap on the number of requests waiting for an
    // instance. 0 disables the bound (unbounded).
    Scheduler(BackendPtr backend, std::shared_ptr<const ModelConfig> config,
              size_t instance_count, size_t max_queue_size, int64_t default_timeout_ms,
              int64_t max_inference_time_ms, std::shared_ptr<MemoryManager> memory);
    ~Scheduler();

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    // Enqueue a request and block until it completes or times out. Returns a
    // shared InferenceResult. Throws std::runtime_error if the queue is full.
    std::shared_ptr<InferenceResult> submit(std::shared_ptr<InferenceRequest> req);

    // Current number of requests waiting in the queue.
    size_t queueDepth() const;

    const SchedulerStats& stats() const { return stats_; }
    size_t instanceCount() const { return workers_.size(); }

    // Per-model latency (microseconds, average). Used by metrics.
    double averageLatencyUs() const {
        uint64_t n = stats_.requests_completed.load();
        if (n == 0) return 0.0;
        return static_cast<double>(stats_.total_exec_us.load()) / static_cast<double>(n);
    }

private:
    void workerLoop(size_t idx);
    void processOne(std::shared_ptr<InferenceRequest> req);

    BackendPtr backend_;
    std::shared_ptr<const ModelConfig> config_;
    std::shared_ptr<MemoryManager> memory_;

    size_t max_queue_size_;
    int64_t default_timeout_ms_;
    int64_t max_inference_time_ms_;

    mutable std::mutex queue_mu_;
    std::condition_variable queue_cv_;
    std::deque<std::shared_ptr<InferenceRequest>> queue_;
    bool stop_ = false;

    std::vector<std::thread> workers_;
    SchedulerStats stats_;

    // Semaphore-like counter limiting the number of requests dequeued but not
    // yet completed. Bound = instance_count.
    std::mutex inflight_mu_;
    std::condition_variable inflight_cv_;
    size_t inflight_ = 0;
};

}  // namespace inferlite
