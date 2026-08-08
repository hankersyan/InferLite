#include "scheduler.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

#include "openvino_backend.hpp"
#include "pbtxt.hpp"

namespace inferlite {

namespace {
using clock = std::chrono::steady_clock;
}

Scheduler::Scheduler(std::shared_ptr<OpenVinoBackend> backend,
                     std::shared_ptr<const ModelConfig> config,
                     size_t instance_count, size_t max_queue_size,
                     int64_t default_timeout_ms,
                     std::shared_ptr<MemoryManager> memory)
    : backend_(std::move(backend)),
      config_(std::move(config)),
      memory_(std::move(memory)),
      max_queue_size_(max_queue_size),
      default_timeout_ms_(default_timeout_ms) {
    // Spawn one worker thread per CPU instance.
    for (size_t i = 0; i < instance_count; ++i) {
        workers_.emplace_back([this, i]() { workerLoop(i); });
    }
}

Scheduler::~Scheduler() {
    {
        std::lock_guard<std::mutex> lock(queue_mu_);
        stop_ = true;
    }
    queue_cv_.notify_all();
    inflight_cv_.notify_all();
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
}

size_t Scheduler::queueDepth() const {
    std::lock_guard<std::mutex> lock(queue_mu_);
    return queue_.size();
}

std::shared_ptr<InferenceResult> Scheduler::submit(std::shared_ptr<InferenceRequest> req) {
    // Bound the queue: reject if at capacity (requests in queue + in-flight).
    {
        std::lock_guard<std::mutex> qlock(queue_mu_);
        std::lock_guard<std::mutex> ilock(inflight_mu_);
        size_t pending = queue_.size() + inflight_;
        if (max_queue_size_ > 0 && pending >= max_queue_size_) {
            throw std::runtime_error("request queue is full (max_queue_size=" +
                                     std::to_string(max_queue_size_) + ")");
        }
        queue_.push_back(req);
    }
    queue_cv_.notify_one();

    // Block until a worker completes this request (or the queue-timeout fires).
    std::unique_lock<std::mutex> lock(req->m);
    if (!req->cv.wait_for(lock, std::chrono::milliseconds(req->timeout_ms),
                          [&]() { return req->done; })) {
        // Timeout while waiting for execution. Mark the request as failed.
        // The worker may still be processing it; it will observe the flag.
        req->result.ok = false;
        req->result.error = "request timed out after " +
                            std::to_string(req->timeout_ms) + "ms";
        req->done = true;
        // NOTE: the worker may concurrently write req->result; the mutex is
        // held here so we are safe to touch the fields.
        req->cv.notify_all();
        stats_.requests_timed_out.fetch_add(1, std::memory_order_relaxed);
        return std::make_shared<InferenceResult>(req->result);
    }
    return std::make_shared<InferenceResult>(req->result);
}

void Scheduler::workerLoop(size_t idx) {
    (void)idx;
    while (true) {
        std::shared_ptr<InferenceRequest> req;
        {
            std::unique_lock<std::mutex> lock(queue_mu_);
            queue_cv_.wait(lock, [this]() { return stop_ || !queue_.empty(); });
            if (stop_ && queue_.empty()) return;
            req = queue_.front();
            queue_.pop_front();
        }
        {
            std::lock_guard<std::mutex> ilock(inflight_mu_);
            ++inflight_;
        }
        processOne(req);
        {
            std::lock_guard<std::mutex> ilock(inflight_mu_);
            --inflight_;
        }
        inflight_cv_.notify_all();
    }
}

void Scheduler::processOne(std::shared_ptr<InferenceRequest> req) {
    // If the request already timed out (result marked), skip execution.
    {
        std::lock_guard<std::mutex> lock(req->m);
        if (req->done) return;  // timed out while queued
    }

    auto start = clock::now();
    try {
        std::vector<Tensor> outputs;
        backend_->execute(req->inputs, outputs);
        std::lock_guard<std::mutex> lock(req->m);
        if (req->done) return;  // client already gave up
        req->result.ok = true;
        req->result.outputs = std::move(outputs);
        req->result.error.clear();
        req->done = true;
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(req->m);
        if (req->done) return;
        req->result.ok = false;
        req->result.error = e.what();
        req->done = true;
    }
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - start).count();
    if (req->result.ok) {
        stats_.requests_completed.fetch_add(1, std::memory_order_relaxed);
        stats_.total_exec_us.fetch_add(static_cast<uint64_t>(us), std::memory_order_relaxed);
    } else {
        stats_.requests_failed.fetch_add(1, std::memory_order_relaxed);
    }
    req->cv.notify_all();
}

}  // namespace inferlite
