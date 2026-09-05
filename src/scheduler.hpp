// scheduler.hpp - Bounded FIFO request scheduler with CPU instance groups.
//
// Maintains a bounded FIFO queue of inference requests. One worker thread per
// CPU instance consumes requests from the queue, so at most `instance_count`
// requests are in-flight simultaneously. Requests beyond the current in-flight
// capacity wait in the queue. A configurable request timeout aborts a request
// that has waited too long in the queue. A hard per-request inference time
// limit (MAX_INFERENCE_TIME_MS) aborts a request that runs too long.
//
// Batching mode (Phase 7, follows NVIDIA Triton): when the model config enables
// dynamic batching (`max_batch_size > 0` plus a `dynamic_batching {}` policy),
// a worker coalesces multiple queued requests into one backend execution whose
// leading batch dimension is the sum of the requests' batch dimensions (never
// exceeding max_batch_size). Execution happens as soon as a preferred batch
// size is reached, the queue can contribute no more fitting requests, or the
// oldest request in the batch has waited max_queue_delay_microseconds.
// Outputs of the merged execution are sliced back to each request.
//
// Two additional Triton dynamic-batching policies are honored:
//   * priority_levels / default_priority_level - requests are scheduled by
//     priority (1 is highest); requests at the same level keep arrival order.
//     Requests may carry an explicit priority (see InferLite::runInference).
//   * preserve_ordering - responses are returned in the order the requests
//     arrived at the scheduler, even when execution reorders them (e.g. a
//     higher-priority request may execute first but its response still waits
//     for every earlier request to complete first).
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "backend.hpp"
#include "memory_manager.hpp"
#include "tensor.hpp"

#ifdef INFERLITE_ENABLE_GPU
#include "gpu_memory_manager.hpp"
#endif

namespace inferlite {

struct ModelConfig;

struct InferenceResult {
    bool ok = false;
    ErrorCode error_code = ErrorCode::kNone;
    std::string error;
    std::vector<Tensor> outputs;
    int64_t inference_us = 0;
};

struct InferenceRequest;

// Outcome of one request within a finished merged batch, queued for delivery.
// Delivery is immediate unless the model enables preserve_ordering, in which
// case it happens strictly in `seq` order.
struct BatchFinish {
    uint64_t seq = 0;
    std::shared_ptr<InferenceRequest> req;
    bool ok = false;
    ErrorCode ec = ErrorCode::kNone;
    std::string err;
    std::vector<Tensor> outputs;
    int64_t us = 0;
};

struct SchedulerStats {
    std::atomic<uint64_t> requests_completed{0};
    std::atomic<uint64_t> requests_failed{0};
    std::atomic<uint64_t> requests_timed_out{0};
    // Total inference time (excluding queue wait) in microseconds.
    std::atomic<uint64_t> total_exec_us{0};
    // --- Batching mode (Phase 7) ---
    // Successful merged backend executions (a "batch"). Counted only for models
    // with dynamic batching enabled.
    std::atomic<uint64_t> batches_completed{0};
    // Sum of request batch dimensions served by successful executions (the
    // number of samples in every request). averageBatchSize() =
    // samples_completed / batches_completed.
    std::atomic<uint64_t> samples_completed{0};
};

// A single request enqueued by an HTTP handler. The handler blocks on the
// completion promise until a worker produces the result.
struct InferenceRequest {
    std::vector<Tensor> inputs;
    // How long (in milliseconds) this request may wait in the queue before the
    // scheduler rejects it with a timeout.
    int64_t timeout_ms = 30000;
    // Batching mode: number of samples this request carries (the leading batch
    // dimension of its tensors when the model batches; 1 otherwise). Filled by
    // Scheduler::submit so batching workers never re-derive it.
    int64_t batch = 1;
    // Time the request entered the scheduler queue. The dynamic-batch delay
    // window is measured from here (the oldest request of a batch).
    std::chrono::steady_clock::time_point enqueued_at{};
    // Batching mode (priority scheduling): explicit request priority, or 0 when
    // the caller did not provide one (the scheduler substitutes the model's
    // default_priority_level). Lower values are higher priority; 1 is highest.
    int64_t priority = 0;
    // Scheduler-assigned monotonically increasing enqueue sequence. Used as the
    // arrival order for priority FIFO stability and for preserve_ordering.
    uint64_t seq = 0;

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
    // `device_kind`: "CPU" spawns a worker thread per instance; "GPU" tracks a
    // pool of busy/free GPU instances (each mapped to a CUDA stream) driven by
    // the same worker threads, enabling concurrent execution across streams.
    Scheduler(BackendPtr backend, std::shared_ptr<const ModelConfig> config,
              size_t instance_count, size_t max_queue_size, int64_t default_timeout_ms,
              int64_t max_inference_time_ms, std::shared_ptr<MemoryManager> memory,
              std::string device_kind = "CPU");
    ~Scheduler();

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    // Enqueue a request and block until it completes or times out. Returns a
    // shared InferenceResult; when the queue is at capacity the returned result
    // carries ErrorCode::kResourceExhausted.
    std::shared_ptr<InferenceResult> submit(std::shared_ptr<InferenceRequest> req);

    // Current number of requests waiting in the queue.
    size_t queueDepth() const;

    const SchedulerStats& stats() const { return stats_; }
    size_t instanceCount() const { return worker_count_; }
    const std::string& deviceKind() const { return device_kind_; }

    // Per-model latency (microseconds, average). Used by metrics.
    double averageLatencyUs() const {
        uint64_t n = stats_.requests_completed.load();
        if (n == 0) return 0.0;
        return static_cast<double>(stats_.total_exec_us.load()) / static_cast<double>(n);
    }

    // --- Batching mode (Phase 7) ---
    // True when this scheduler coalesces requests across concurrent inference
    // calls (config `max_batch_size > 0` and a `dynamic_batching {}` policy).
    bool batchingEnabled() const { return max_batch_size_ > 0 && batch_mode_; }
    // Number of successful backend executions (merged batches).
    uint64_t batchesCompleted() const { return stats_.batches_completed.load(); }
    // Total samples served by successful executions.
    uint64_t samplesCompleted() const { return stats_.samples_completed.load(); }
    // Average number of samples per successful execution (batch size).
    double averageBatchSize() const {
        uint64_t n = batchesCompleted();
        if (n == 0) return 0.0;
        return static_cast<double>(samplesCompleted()) / static_cast<double>(n);
    }

    // Priority scheduling: number of successfully completed requests per
    // priority level (index 0 == priority 1). Empty when priority scheduling
    // is not enabled for this model.
    std::vector<uint64_t> priorityServed() const;

private:
    void workerLoop(size_t idx);
    // One worker iteration. Returns false when the worker must exit (shutdown
    // requested and the queue is drained).
    bool runSingleCycle();
    bool runBatchCycle();
    void processOne(std::shared_ptr<InferenceRequest> req);

    // Execute one merged batch (requests already removed from the queue and
    // counted in inflight_). Slices the merged outputs back to each request.
    void executeBatch(std::vector<std::shared_ptr<InferenceRequest>>& group,
                      int64_t samples);

    // Batching helpers (Phase 7).
    bool isPreferredBatchSize(int64_t samples) const;
    // Whether the batch is "full enough" to dispatch: a preferred size was
    // reached, or (no preferred sizes) the model's max_batch_size was reached.
    bool atBatchTarget(int64_t samples) const;
    // Priority-aware queue insertion; queue_mu_ must be held by the caller.
    void enqueueLocked(const std::shared_ptr<InferenceRequest>& req);

    // Deliver one batch outcome to its request (fills result, stats, notify).
    void deliverBatchFinish(const BatchFinish& spec);
    // Complete a finished batch; honors preserve_ordering when enabled.
    void finishBatchMembers(std::vector<BatchFinish>& specs);
    // preserve_ordering delivery gate: a request may only be delivered (marked
    // done) once every request that arrived earlier has been delivered.
    // `task` performs the delivery (deliverBatchFinish); it is invoked strictly
    // in enqueue-sequence order. Never call with a request lock held.
    void orderedNoteFinished(uint64_t seq, std::function<void()> task);
    void orderedNoteSkipped(uint64_t seq);
    void orderedRelease();

    BackendPtr backend_;
    std::shared_ptr<const ModelConfig> config_;
    std::shared_ptr<MemoryManager> memory_;
    std::string device_kind_;

    size_t max_queue_size_;
    int64_t default_timeout_ms_;
    int64_t max_inference_time_ms_;

    // --- Batching mode (Phase 7) ---
    int64_t max_batch_size_ = 0;      // model's Triton max_batch_size (>0 batching-capable)
    bool batch_mode_ = false;         // dynamic_batching {} policy present
    int64_t batch_delay_us_ = 0;      // max_queue_delay_microseconds
    std::vector<int64_t> preferred_batch_size_;  // sorted ascending
    // Triton priority scheduling (dynamic_batching.priority_levels).
    bool priority_active_ = false;
    int64_t priority_levels_ = 0;
    int64_t default_priority_level_ = 1;
    // Successful completions per priority level (index 0 == priority 1).
    // unique_ptr keeps the (non-copyable) atomics stable in the container.
    std::vector<std::unique_ptr<std::atomic<uint64_t>>> priority_served_;
    // Triton preserve_ordering: deliver responses in enqueue order.
    bool preserve_ordering_ = false;
    // Monotonic enqueue sequence; guarded by queue_mu_.
    uint64_t next_seq_ = 1;

    // preserve_ordering delivery gate.
    mutable std::mutex order_mu_;
    uint64_t order_next_ = 1;                 // next seq allowed to be delivered
    std::map<uint64_t, std::function<void()>> order_pending_;   // finished, awaiting delivery
    std::set<uint64_t> order_skipped_;        // seqs resolved out-of-band (timeouts/drops)

    mutable std::mutex queue_mu_;
    std::condition_variable queue_cv_;
    std::deque<std::shared_ptr<InferenceRequest>> queue_;
    bool stop_ = false;
    // Set by the worker currently assembling a dynamic batch so a second worker
    // cannot steal requests out of the batch being accumulated.
    bool assembling_ = false;

    std::vector<std::thread> workers_;
    size_t worker_count_ = 0;
    SchedulerStats stats_;

    // Phase 3: true once a backend instance quarantines itself (CUDA fault).
    // A quarantined scheduler rejects further work with INTERNAL_ERROR.
    std::atomic<bool> quarantined_{false};

    // Semaphore-like counter limiting the number of requests dequeued but not
    // yet completed. Bound = instance_count.
    std::mutex inflight_mu_;
    std::condition_variable inflight_cv_;
    size_t inflight_ = 0;
};

}  // namespace inferlite
