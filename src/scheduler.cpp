#include "scheduler.hpp"

#include <algorithm>
#include <chrono>

#include "pbtxt.hpp"

namespace inferlite {

namespace {
using clock = std::chrono::steady_clock;

// One request inside a dynamic batch. `req` has been removed from the queue and
// counted against inflight_; `batch` is its leading batch dimension.
struct BatchMember {
    std::shared_ptr<InferenceRequest> req;
    int64_t batch = 1;
};
}  // namespace

Scheduler::Scheduler(BackendPtr backend, std::shared_ptr<const ModelConfig> config,
                     size_t instance_count, size_t max_queue_size,
                     int64_t default_timeout_ms, int64_t max_inference_time_ms,
                     std::shared_ptr<MemoryManager> memory, std::string device_kind)
    : backend_(std::move(backend)),
      config_(std::move(config)),
      memory_(std::move(memory)),
      device_kind_(std::move(device_kind)),
      max_queue_size_(max_queue_size),
      default_timeout_ms_(default_timeout_ms),
      max_inference_time_ms_(max_inference_time_ms > 0 ? max_inference_time_ms : 5000) {
    // Phase 7 (batching mode): derive the Triton batching parameters from the
    // model config. Dynamic (cross-request) batching is active only when the
    // model declares max_batch_size > 0 AND a `dynamic_batching {}` policy.
    max_batch_size_ = (config_ && config_->max_batch_size > 0) ? config_->max_batch_size : 0;
    batch_mode_ = max_batch_size_ > 0 && config_ && config_->batching.enabled;
    batch_delay_us_ = (config_ && config_->batching.enabled) ? config_->batching.max_queue_delay_us
                                                             : 0;
    if (config_) {
        preferred_batch_size_ = config_->batching.preferred_batch_size;
    }
    std::sort(preferred_batch_size_.begin(), preferred_batch_size_.end());

    worker_count_ = instance_count;
    // Spawn one worker thread per instance. For GPU instances the backend
    // executes on its own CUDA stream; workers simply drive the blocking
    // execute() call (which the backend synchronizes via CUDA events), so the
    // same worker model covers both CPU and GPU concurrently.
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
    // Phase 7: capture the request's own batch dimension (leading tensor dim)
    // once at enqueue time; batching workers use it for capacity accounting.
    if (req->batch < 1) req->batch = 1;
    if (max_batch_size_ > 0 && !req->inputs.empty()) {
        const auto& s = req->inputs.front().shape;
        if (!s.empty() && s[0] >= 1) req->batch = s[0];
    }
    req->enqueued_at = clock::now();

    // Bound the queue: reject if at capacity (requests in queue + in-flight).
    {
        std::lock_guard<std::mutex> qlock(queue_mu_);
        std::lock_guard<std::mutex> ilock(inflight_mu_);
        size_t pending = queue_.size() + inflight_;
        if (max_queue_size_ > 0 && pending >= max_queue_size_) {
            auto res = std::make_shared<InferenceResult>();
            res->ok = false;
            res->error_code = ErrorCode::kResourceExhausted;
            res->error = "request queue is full (max_queue_size=" +
                         std::to_string(max_queue_size_) + ")";
            return res;
        }
        queue_.push_back(req);
    }
    // notify_all (rather than notify_one) so the worker that is mid-batch and
    // waiting for more requests to join also sees the arrival.
    queue_cv_.notify_all();

    // Block until a worker completes this request (or the queue-timeout fires).
    std::unique_lock<std::mutex> lock(req->m);
    if (!req->cv.wait_for(lock, std::chrono::milliseconds(req->timeout_ms),
                          [&]() { return req->done; })) {
        // Timeout while waiting for execution. Mark the request as failed.
        req->result.ok = false;
        req->result.error_code = ErrorCode::kTimeout;
        req->result.error = "request timed out after " +
                            std::to_string(req->timeout_ms) + "ms";
        req->done = true;
        req->cv.notify_all();
        stats_.requests_timed_out.fetch_add(1, std::memory_order_relaxed);
        return std::make_shared<InferenceResult>(req->result);
    }
    return std::make_shared<InferenceResult>(req->result);
}

void Scheduler::workerLoop(size_t idx) {
    (void)idx;
    while (true) {
        const bool cont = batchingEnabled() ? runBatchCycle() : runSingleCycle();
        if (!cont) return;
    }
}

// Non-batching path (or a batching-capable model without a dynamic_batching
// policy): pop one request and execute it standalone.
bool Scheduler::runSingleCycle() {
    std::shared_ptr<InferenceRequest> req;
    {
        std::unique_lock<std::mutex> lock(queue_mu_);
        queue_cv_.wait(lock, [this]() { return stop_ || !queue_.empty(); });
        if (stop_ && queue_.empty()) return false;
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
    return true;
}

// Batching path: assemble one merged batch from the head of the queue and run
// it with a single backend execute() call (NVIDIA Triton dynamic batching).
bool Scheduler::runBatchCycle() {
    std::vector<std::shared_ptr<InferenceRequest>> group;
    int64_t samples = 0;
    {
        std::unique_lock<std::mutex> lock(queue_mu_);
        // Only one worker assembles a batch at a time so requests that arrive
        // during the wait window stay in the queue for this batch.
        queue_cv_.wait(lock, [this]() { return stop_ || (!queue_.empty() && !assembling_); });
        if (stop_ && queue_.empty()) return false;
        assembling_ = true;

        // Pop every queued request that fits the remaining capacity, stopping as
        // soon as a preferred batch size (or max_batch_size) is reached.
        auto absorb = [&]() {
            while (!queue_.empty() && !atBatchTarget(samples)) {
                const int64_t need = queue_.front()->batch;
                if (samples + need > max_batch_size_) break;  // never overfill
                {
                    std::lock_guard<std::mutex> ilock(inflight_mu_);
                    ++inflight_;
                }
                group.push_back(queue_.front());
                queue_.pop_front();
                samples += need;
            }
        };
        absorb();

        // Wait-window: keep this batch open until the oldest request's delay
        // deadline so late arrivals can join, unless the batch is already at a
        // target size or the queue head cannot fit the remaining capacity.
        if (!stop_ && batch_delay_us_ > 0 && !atBatchTarget(samples)) {
            const auto deadline =
                group.front()->enqueued_at + std::chrono::microseconds(batch_delay_us_);
            while (!stop_ && !atBatchTarget(samples)) {
                if (clock::now() >= deadline) break;
                if (!queue_.empty()) {
                    const int64_t need = queue_.front()->batch;
                    if (samples + need > max_batch_size_) break;
                }
                queue_cv_.wait_until(lock, deadline,
                                     [this]() { return stop_ || !queue_.empty(); });
                absorb();
            }
        }
        assembling_ = false;
    }
    // Announce the batch slot is free again so another worker may assemble the
    // next batch while this one executes.
    queue_cv_.notify_all();

    executeBatch(group, samples);
    return true;
}

void Scheduler::processOne(std::shared_ptr<InferenceRequest> req) {
    // If the request already timed out (result marked), skip execution.
    {
        std::lock_guard<std::mutex> lock(req->m);
        if (req->done) return;  // timed out while queued
    }

    // Phase 3 fault isolation: a quarantined (CUDA-faulted) instance refuses
    // further work with a structured error code rather than risking corruption.
    if (quarantined_.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(req->m);
        if (req->done) return;
        req->result.ok = false;
        req->result.error_code = ErrorCode::kInternalError;
        req->result.error = "model instance is quarantined after a CUDA fault";
        req->done = true;
        stats_.requests_failed.fetch_add(1, std::memory_order_relaxed);
        req->cv.notify_all();
        return;
    }

    auto start = clock::now();
    try {
        // Backends are CPU-bound and block synchronously. We record the elapsed
        // time and, if it exceeds the per-request inference limit, report a
        // TIMEOUT. Backend calls are fault-contained (return BackendResult), so
        // no exception escapes. (True thread cancellation is unsafe on CPU.)
        BackendResult bres = backend_->execute(req->inputs);

        auto us = std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - start).count();

        std::lock_guard<std::mutex> rlock(req->m);
        if (req->done) return;  // client already gave up
        req->result.inference_us = us;

        if (us > max_inference_time_ms_ * 1000) {
            req->result.ok = false;
            req->result.error_code = ErrorCode::kTimeout;
            req->result.error = "inference exceeded time limit (" +
                                std::to_string(max_inference_time_ms_) + "ms)";
        } else {
            req->result.ok = bres.ok;
            req->result.error_code = bres.error_code;
            req->result.error = bres.error;
            req->result.outputs = std::move(bres.outputs);

            // Phase 3 fault isolation: if the backend quarantined itself (e.g.
            // a CUDA error), mark the model instance dead so no further work is
            // assigned. A dead instance yields a structured INTERNAL_ERROR.
            if (bres.quarantine) {
                quarantined_.store(true, std::memory_order_release);
            }
        }
        req->done = true;
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(req->m);
        if (req->done) return;
        req->result.ok = false;
        req->result.error_code = ErrorCode::kInternalError;
        req->result.error = e.what();
        req->done = true;
    }

    auto us = std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - start).count();
    if (req->result.ok) {
        stats_.requests_completed.fetch_add(1, std::memory_order_relaxed);
        stats_.total_exec_us.fetch_add(static_cast<uint64_t>(us), std::memory_order_relaxed);
    } else if (req->result.error_code == ErrorCode::kTimeout) {
        stats_.requests_timed_out.fetch_add(1, std::memory_order_relaxed);
    } else {
        stats_.requests_failed.fetch_add(1, std::memory_order_relaxed);
    }
    req->cv.notify_all();
}

bool Scheduler::isPreferredBatchSize(int64_t samples) const {
    if (preferred_batch_size_.empty()) return false;
    return std::binary_search(preferred_batch_size_.begin(), preferred_batch_size_.end(), samples);
}

bool Scheduler::atBatchTarget(int64_t samples) const {
    if (samples >= max_batch_size_) return true;
    return isPreferredBatchSize(samples);
}

// Execute one merged batch. `group` holds the requests popped by the assembler
// (already counted against inflight_); `samples` is their total batch dimension
// as accumulated. Requests whose submit() already timed out are dropped here.
void Scheduler::executeBatch(std::vector<std::shared_ptr<InferenceRequest>>& group,
                             int64_t samples) {
    (void)samples;  // recomputed from the surviving members below
    if (group.empty()) return;

    // Drop requests the client already gave up on (submit() timed out while the
    // request waited) and release their in-flight slots. They were never
    // executed, so their error result (set by submit) is final.
    std::vector<BatchMember> members;
    members.reserve(group.size());
    for (auto& req : group) {
        {
            std::lock_guard<std::mutex> lock(req->m);
            if (req->done) {
                {
                    std::lock_guard<std::mutex> ilock(inflight_mu_);
                    --inflight_;
                }
                continue;
            }
        }
        int64_t b = req->batch < 1 ? 1 : req->batch;
        if (b > max_batch_size_) b = max_batch_size_;  // defensive clamp
        members.push_back(BatchMember{req, b});
    }
    if (members.empty()) return;

    int64_t total = 0;
    for (const auto& m : members) total += m.batch;

    // Every surviving member holds one in-flight slot (incremented when the
    // assembler popped it). It must be released exactly once no matter which
    // path below finishes the batch -- including members whose submit() gives
    // up while the merged inference is still running.
    const size_t slot_count = members.size();
    auto releaseSlots = [&]() {
        std::lock_guard<std::mutex> ilock(inflight_mu_);
        inflight_ -= slot_count;
    };

    // Fail-everything helper: mark each surviving request failed (skip any that
    // time out mid-execution, whose submit() already counted the timeout) and
    // count only the requests this batch actually failed.
    auto failAll = [&](ErrorCode code, const std::string& msg, bool as_timeout,
                       int64_t us) {
        uint64_t n = 0;
        for (auto& m : members) {
            std::lock_guard<std::mutex> lock(m.req->m);
            if (m.req->done) continue;
            m.req->result.inference_us = us;
            m.req->result.ok = false;
            m.req->result.error_code = code;
            m.req->result.error = msg;
            m.req->done = true;
            m.req->cv.notify_all();
            ++n;
        }
        if (n > 0) {
            if (as_timeout) {
                stats_.requests_timed_out.fetch_add(n, std::memory_order_relaxed);
            } else {
                stats_.requests_failed.fetch_add(n, std::memory_order_relaxed);
            }
        }
    };

    // Phase 3 fault isolation (same gate as processOne).
    if (quarantined_.load(std::memory_order_acquire)) {
        failAll(ErrorCode::kInternalError, "model instance is quarantined after a CUDA fault",
                false, 0);
        releaseSlots();
        return;
    }

    // Merge every member's inputs into one batch: config declares the
    // per-request tensor specs (no batch dim), so the merged tensor shape is
    // [total] ++ spec.dims and its payload is the row-major concatenation.
    std::vector<Tensor> merged;
    if (!config_) {
        failAll(ErrorCode::kInternalError, "scheduler has no model config", false, 0);
        releaseSlots();
        return;
    }
    for (const auto& spec : config_->inputs) {
        Tensor m;
        m.name = spec.name;
        m.type = spec.data_type;
        std::vector<int64_t> rest;
        bool first = true;
        for (auto& mem : members) {
            const Tensor* t = nullptr;
            for (const auto& in : mem.req->inputs) {
                if (in.name == spec.name) {
                    t = &in;
                    break;
                }
            }
            if (t == nullptr || t->shape.size() < 1) {
                failAll(ErrorCode::kInternalError,
                        "batched request is missing input tensor '" + spec.name +
                            "' or carries no batch dimension",
                        false, 0);
                releaseSlots();
                return;
            }
            if (first) {
                rest.assign(t->shape.begin() + 1, t->shape.end());
                m.type = t->type;
                first = false;
            } else {
                // Every member must share the same per-sample shape and type so
                // the row-major concatenation stays well-formed.
                if (t->type != m.type ||
                    !std::equal(t->shape.begin() + 1, t->shape.end(), rest.begin(),
                                rest.end())) {
                    failAll(ErrorCode::kInternalError,
                            "batched requests carry inconsistent per-sample shapes for input '" +
                                spec.name + "'",
                            false, 0);
                    releaseSlots();
                    return;
                }
            }
            m.data.insert(m.data.end(), t->data.begin(), t->data.end());
        }
        if (first) {  // no member contributed (defensive; members is non-empty)
            failAll(ErrorCode::kInternalError, "internal batching error", false, 0);
            releaseSlots();
            return;
        }
        m.shape.reserve(rest.size() + 1);
        m.shape.push_back(total);
        m.shape.insert(m.shape.end(), rest.begin(), rest.end());
        merged.push_back(std::move(m));
    }

    auto start = clock::now();
    BackendResult bres;
    try {
        bres = backend_->execute(merged);
    } catch (const std::exception& e) {
        failAll(ErrorCode::kInternalError, e.what(), false, 0);
        releaseSlots();
        return;
    }
    const int64_t us =
        std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - start).count();

    if (bres.quarantine) {
        quarantined_.store(true, std::memory_order_release);
    }

    // Hard inference-time limit: every request in the batch is a TIMEOUT.
    if (us > max_inference_time_ms_ * 1000) {
        failAll(ErrorCode::kTimeout,
                "inference exceeded time limit (" + std::to_string(max_inference_time_ms_) +
                    "ms)",
                true, us);
        releaseSlots();
        return;
    }

    if (!bres.ok) {
        failAll(bres.error_code, bres.error, false, us);
        releaseSlots();
        return;
    }

    // Slice the merged outputs back to each member in queue order.
    std::vector<std::vector<Tensor>> per_req(members.size());
    {
        bool slice_ok = true;
        for (const auto& out : bres.outputs) {
            if (out.shape.empty()) {
                slice_ok = false;
                break;
            }
            const size_t rows = static_cast<size_t>(out.shape[0]);
            if (rows != static_cast<size_t>(total)) {
                slice_ok = false;
                break;
            }
            const size_t row_bytes = rows > 0 ? out.data.size() / rows : 0;
            size_t offset = 0;
            for (size_t i = 0; i < members.size(); ++i) {
                const size_t cnt = static_cast<size_t>(members[i].batch) * row_bytes;
                if (offset + cnt > out.data.size()) {
                    slice_ok = false;
                    break;
                }
                Tensor t;
                t.name = out.name;
                t.type = out.type;
                t.shape.assign(out.shape.begin() + 1, out.shape.end());
                t.shape.insert(t.shape.begin(), members[i].batch);
                t.data.assign(out.data.begin() + static_cast<std::ptrdiff_t>(offset),
                              out.data.begin() + static_cast<std::ptrdiff_t>(offset + cnt));
                per_req[i].push_back(std::move(t));
                offset += cnt;
            }
            if (!slice_ok) break;
        }
        if (!slice_ok) {
            // The model returned a shape that cannot be sliced back into the
            // per-request tensors. This is the classic symptom of a model whose
            // IR was built with a static batch dimension: dynamic batching
            // requires a model that accepts any batch <= max_batch_size.
            failAll(ErrorCode::kInternalError,
                    "merged output cannot be sliced back into per-request batches "
                    "(the model IR must accept a dynamic batch dimension up to " +
                        std::to_string(max_batch_size_) + ")",
                    false, us);
            releaseSlots();
            return;
        }
    }

    // Success: complete every member with its slice.
    stats_.batches_completed.fetch_add(1, std::memory_order_relaxed);
    for (size_t i = 0; i < members.size(); ++i) {
        std::lock_guard<std::mutex> lock(members[i].req->m);
        if (members[i].req->done) continue;  // client gave up during execution
        members[i].req->result.inference_us = us;
        members[i].req->result.ok = true;
        members[i].req->result.error_code = ErrorCode::kNone;
        members[i].req->result.outputs = std::move(per_req[i]);
        members[i].req->done = true;
        members[i].req->cv.notify_all();
        stats_.requests_completed.fetch_add(1, std::memory_order_relaxed);
        stats_.samples_completed.fetch_add(static_cast<uint64_t>(members[i].batch),
                                           std::memory_order_relaxed);
        stats_.total_exec_us.fetch_add(static_cast<uint64_t>(us), std::memory_order_relaxed);
    }
    releaseSlots();
}

}  // namespace inferlite
