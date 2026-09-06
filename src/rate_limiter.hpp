// rate_limiter.hpp - Triton-style cross-model resource rate limiter.
//
// Coordinates backend executions ACROSS all loaded models. A model declares,
// in its `instance_group { rate_limiter { ... } }`, the shared resources one
// execution consumes (e.g. GPU memory / device copies). Before a scheduler
// worker runs a backend execution it must reserve those units; if they are not
// available the execution is postponed and a waiting model with a higher
// `rate_limiter.priority` is preferred when capacity frees. This serializes
// memory-heavy models so several loaded models can never over-subscribe the
// one GPU concurrently (see docs/RATE_LIMITER.md).
//
// Semantics mirror NVIDIA Triton's rate limiter:
//   * a `global: true` resource is a single system-wide pool;
//   * a `global: false` resource is a pool per execution device;
//   * the available units of a pool default to the largest single requirement
//     declared by a loaded model (guaranteeing any one model can run) and can
//     be raised with the server flag `--rate-limit-resource=<name>:<count>`;
//   * models that declare no resources are unrestricted.
#pragma once

#include <atomic>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "pbtxt.hpp"  // RateLimiterResource

namespace inferlite {

class RateLimiter {
public:
    // `enabled` is set by the server flag `--rate-limit`. When false the
    // limiter is inert: acquire() always grants immediately and no waiting
    // occurs (mirrors Triton's `--rate-limit=off`, the default).
    explicit RateLimiter(bool enabled);
    ~RateLimiter() = default;

    RateLimiter(const RateLimiter&) = delete;
    RateLimiter& operator=(const RateLimiter&) = delete;

    bool enabled() const { return enabled_; }

    // Raise (or cap) the available units of a resource pool. Applies to every
    // pool that backs `name` (global and per-device). Values below the largest
    // declared requirement are legal but may serialize models that would
    // otherwise fit (mirrors `--rate-limit-resource=<name>:<count>`).
    void overrideResourceCapacity(const std::string& name, int64_t capacity);

    // Declare the resources one execution of the model scheduler identified by
    // `owner` consumes. `owner` must be unique per scheduler instance (InferLite
    // uses "<model>#<seq>") so a reload does not cancel a draining predecessor.
    // `device` is the resolved execution-device label used to key non-global
    // pools. Recomputes affected pool capacities (default = max over
    // declarations). Call once per scheduler; call unregisterModel() before the
    // scheduler is torn down.
    void registerModel(const std::string& owner, int64_t priority,
                       const std::vector<RateLimiterResource>& resources,
                       const std::string& device);
    void unregisterModel(const std::string& owner);

    // An admission: the caller may run one backend execution while holding it.
    // Releasing returns the reserved units to their pools and admits the next
    // waiting model. granted() is false only when the acquisition was aborted
    // (`cancel` became true / the model was unregistered while waiting).
    class Lease {
    public:
        Lease() = default;
        ~Lease();
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept;
        Lease& operator=(Lease&& other) noexcept;

        bool granted() const { return granted_; }
        // Return the reserved units (idempotent; safe on a default lease).
        void release();

    private:
        friend class RateLimiter;
        RateLimiter* rate_limiter_ = nullptr;
        std::vector<std::pair<std::string, int64_t>> credits_;  // pool key -> units
        bool granted_ = false;
    };

    // Block until the resources declared by `owner` are reserved (or the wait
    // is aborted). `cancel` is an optional atomic the owner sets true to wake
    // and abort the wait (used at shutdown). Never blocks when the limiter is
    // disabled or the owner declares no resources.
    Lease acquire(const std::string& owner, const std::atomic<bool>* cancel = nullptr);

    // Read-only snapshot for diagnostics / metrics.
    struct PoolInfo {
        std::string resource_name;  // the declared resource name (e.g. "GPU_MEMORY")
        bool global = true;
        std::string device;         // device key of a per-device pool
        int64_t capacity = 0;       // available units
        int64_t used = 0;           // units currently reserved
    };
    std::vector<PoolInfo> pools() const;

    struct Stats {
        uint64_t admissions = 0;  // granted executions (enabled builds only)
        uint64_t waits = 0;       // times an execution had to wait for capacity
        uint64_t waiters_now = 0; // models currently blocked on capacity
    };
    Stats stats() const;

private:
    struct Declared {
        int64_t priority = 0;
        std::string device;
        std::vector<RateLimiterResource> resources;
    };
    struct Pool {
        bool global = true;
        std::string resource_name;
        std::string device;
        int64_t capacity = 0;
        int64_t used = 0;
        // model name -> units that model declares for this pool (capacity is
        // derived as the max over contributors unless overridden).
        std::vector<std::pair<std::string, int64_t>> contributors;
    };
    struct Waiter {
        uint64_t seq = 0;
        std::string model;
        const std::atomic<bool>* cancel = nullptr;
        bool canceled = false;
        bool granted = false;
        std::vector<std::pair<std::string, int64_t>> credits;
    };

    static std::string poolKey(const RateLimiterResource& r, const std::string& device);
    void recomputeCapacityLocked(Pool& p);
    // Reserve `need` for `d` when every required pool has spare capacity.
    bool canReserveLocked(const Declared& d,
                          std::vector<std::pair<std::string, int64_t>>& credits) const;
    void reserveLocked(const Declared& d, std::vector<std::pair<std::string, int64_t>>& credits);
    void releaseLocked(const std::vector<std::pair<std::string, int64_t>>& credits);
    void grantLocked();          // admit the best waiting model(s) that fit now
    void eraseWaiterLocked(const std::shared_ptr<Waiter>& w);

    bool enabled_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::map<std::string, Declared> models_;
    std::map<std::string, Pool> pools_;          // pool key -> pool
    std::map<std::string, int64_t> overrides_;   // resource name -> capacity
    std::vector<std::shared_ptr<Waiter>> waiters_;
    uint64_t next_waiter_seq_ = 1;

    std::atomic<uint64_t> admissions_{0};
    std::atomic<uint64_t> waits_{0};
};

}  // namespace inferlite
