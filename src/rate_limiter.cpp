#include "rate_limiter.hpp"

#include <algorithm>
#include <cstdint>

namespace inferlite {

namespace {
// Prefixes used to namespace global vs per-device pools so the same resource
// name can coexist as a global pool and a device pool without aliasing.
std::string globalPrefix() { return "global/"; }
std::string devicePrefix() { return "device/"; }
}  // namespace

RateLimiter::RateLimiter(bool enabled) : enabled_(enabled) {}

void RateLimiter::overrideResourceCapacity(const std::string& name, int64_t capacity) {
    if (name.empty() || capacity < 1) return;  // ignored: not a usable override
    std::lock_guard<std::mutex> lock(mu_);
    overrides_[name] = capacity;
    // Recompute every pool that backs this resource name (the override applies
    // to the resource's pools regardless of global/per-device scope).
    for (auto& kv : pools_) {
        if (kv.second.resource_name == name) recomputeCapacityLocked(kv.second);
    }
    grantLocked();
    cv_.notify_all();
}

void RateLimiter::registerModel(const std::string& model, int64_t priority,
                                const std::vector<RateLimiterResource>& resources,
                                const std::string& device) {
    std::lock_guard<std::mutex> lock(mu_);
    // Re-registration (a reload replaces the scheduler) must first drop the old
    // declaration so contributions/capacities are not double counted.
    auto it = models_.find(model);
    if (it != models_.end()) {
        for (const auto& r : it->second.resources) {
            auto pk = pools_.find(poolKey(r, it->second.device));
            if (pk == pools_.end()) continue;
            auto& contribs = pk->second.contributors;
            contribs.erase(std::remove_if(contribs.begin(), contribs.end(),
                                          [&](const auto& c) { return c.first == model; }),
                           contribs.end());
            if (contribs.empty()) {
                pools_.erase(pk);
            } else {
                recomputeCapacityLocked(pk->second);
            }
        }
        models_.erase(it);
    }

    Declared d;
    d.priority = priority;
    d.device = device;
    d.resources = resources;
    models_.emplace(model, std::move(d));

    for (const auto& r : resources) {
        const std::string key = poolKey(r, device);
        auto& p = pools_[key];
        p.global = r.global;
        p.resource_name = r.name;
        p.device = device;
        bool found = false;
        for (auto& c : p.contributors) {
            if (c.first == model) {
                c.second = r.count;  // defensive: single contributor per model
                found = true;
                break;
            }
        }
        if (!found) p.contributors.emplace_back(model, r.count);
        recomputeCapacityLocked(p);
    }
    grantLocked();
    cv_.notify_all();
}

void RateLimiter::unregisterModel(const std::string& model) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = models_.find(model);
    if (it == models_.end()) return;

    for (const auto& r : it->second.resources) {
        auto pk = pools_.find(poolKey(r, it->second.device));
        if (pk == pools_.end()) continue;
        auto& contribs = pk->second.contributors;
        contribs.erase(std::remove_if(contribs.begin(), contribs.end(),
                                      [&](const auto& c) { return c.first == model; }),
                       contribs.end());
        if (contribs.empty()) {
            pools_.erase(pk);
        } else {
            recomputeCapacityLocked(pk->second);
        }
    }
    models_.erase(it);

    // Abort every waiter of the model (its scheduler is being torn down).
    for (auto& w : waiters_) {
        if (w->model == model) w->canceled = true;
    }
    grantLocked();
    cv_.notify_all();
}

std::string RateLimiter::poolKey(const RateLimiterResource& r, const std::string& device) {
    if (r.global) return globalPrefix() + r.name;
    return devicePrefix() + (device.empty() ? std::string("unknown") : device) + "/" + r.name;
}

void RateLimiter::recomputeCapacityLocked(Pool& p) {
    int64_t need = 0;
    for (const auto& c : p.contributors) need = std::max(need, c.second);
    auto ov = overrides_.find(p.resource_name);
    // An override applies to every pool backing the resource name.
    if (ov != overrides_.end()) {
        p.capacity = ov->second;
    } else {
        p.capacity = need;
    }
}

bool RateLimiter::canReserveLocked(
    const Declared& d, std::vector<std::pair<std::string, int64_t>>& credits) const {
    credits.clear();
    for (const auto& r : d.resources) {
        const std::string key = poolKey(r, d.device);
        auto it = pools_.find(key);
        // A missing pool cannot happen for a registered model (registration
        // creates one); treat it as unconstrained rather than deadlock.
        if (it != pools_.end()) {
            if (it->second.used + r.count > it->second.capacity) return false;
            credits.emplace_back(key, r.count);
        }
    }
    return true;
}

void RateLimiter::reserveLocked(const Declared& d,
                                std::vector<std::pair<std::string, int64_t>>& credits) {
    credits.clear();
    for (const auto& r : d.resources) {
        auto it = pools_.find(poolKey(r, d.device));
        if (it != pools_.end()) {
            it->second.used += r.count;
            credits.emplace_back(it->first, r.count);
        }
    }
}

void RateLimiter::releaseLocked(const std::vector<std::pair<std::string, int64_t>>& credits) {
    for (const auto& c : credits) {
        auto it = pools_.find(c.first);
        if (it != pools_.end()) {
            it->second.used = std::max<int64_t>(0, it->second.used - c.second);
        }
    }
    grantLocked();
}

void RateLimiter::grantLocked() {
    // Repeatedly admit the best waiter that can FULLY reserve its resources
    // right now. "Best" = highest rate_limiter.priority, then oldest wait.
    // Considering only currently-satisfiable waiters avoids a high-priority
    // multi-resource waiter convoying others that only need a free resource.
    while (true) {
        std::shared_ptr<Waiter> chosen;
        for (const auto& w : waiters_) {
            if (w->granted || w->canceled) continue;
            if (w->cancel && w->cancel->load()) {
                w->canceled = true;  // owner asked to abort (shutdown)
                continue;
            }
            auto mit = models_.find(w->model);
            if (mit == models_.end()) {
                w->canceled = true;  // model vanished while waiting
                continue;
            }
            std::vector<std::pair<std::string, int64_t>> scratch;
            if (!canReserveLocked(mit->second, scratch)) continue;  // not eligible yet
            if (!chosen) {
                chosen = w;
                continue;
            }
            auto cit = models_.find(chosen->model);
            const int64_t chosen_prio = cit == models_.end() ? 0 : cit->second.priority;
            if (mit->second.priority > chosen_prio ||
                (mit->second.priority == chosen_prio && w->seq < chosen->seq)) {
                chosen = w;
            }
        }
        if (!chosen) break;

        auto mit = models_.find(chosen->model);
        std::vector<std::pair<std::string, int64_t>> credits;
        if (mit == models_.end() || !canReserveLocked(mit->second, credits)) break;
        reserveLocked(mit->second, credits);
        chosen->granted = true;
        chosen->credits = std::move(credits);
    }
}

void RateLimiter::eraseWaiterLocked(const std::shared_ptr<Waiter>& w) {
    for (auto it = waiters_.begin(); it != waiters_.end(); ++it) {
        if (*it == w) {
            waiters_.erase(it);
            return;
        }
    }
}

RateLimiter::Lease RateLimiter::acquire(const std::string& model,
                                        const std::atomic<bool>* cancel) {
    // Disabled limiter: immediate grant, no reservation (no-op lease).
    Lease lease;
    if (!enabled_) {
        lease.granted_ = true;
        return lease;
    }
    std::unique_lock<std::mutex> lock(mu_);
    auto it = models_.find(model);
    // Unknown model (not registered) or a model with no declared resources is
    // unrestricted: grant immediately.
    if (it == models_.end() || it->second.resources.empty()) {
        lease.granted_ = true;
        return lease;
    }
    if (cancel && cancel->load()) return lease;  // aborted before waiting

    std::vector<std::pair<std::string, int64_t>> credits;
    if (canReserveLocked(it->second, credits)) {
        reserveLocked(it->second, credits);
        lease.rate_limiter_ = this;
        lease.credits_ = std::move(credits);
        lease.granted_ = true;
        admissions_.fetch_add(1, std::memory_order_relaxed);
        return lease;
    }

    auto w = std::make_shared<Waiter>();
    w->seq = next_waiter_seq_++;
    w->model = model;
    w->cancel = cancel;
    waiters_.push_back(w);
    waits_.fetch_add(1, std::memory_order_relaxed);

    // Waiters all block on the shared condition variable; grantLocked(),
    // unregisterModel() and Lease::release() call cv_.notify_all() whenever
    // capacity/state changes. Each loop iteration re-evaluates this waiter.
    for (;;) {
        cv_.wait(lock, [&]() {
            return w->granted || w->canceled || (cancel && cancel->load());
        });
        if (w->granted) {
            eraseWaiterLocked(w);
            lease.rate_limiter_ = this;
            lease.credits_ = std::move(w->credits);
            lease.granted_ = true;
            admissions_.fetch_add(1, std::memory_order_relaxed);
            return lease;
        }
        if (w->canceled || (cancel && cancel->load())) {
            eraseWaiterLocked(w);
            return lease;  // granted() == false: caller must not execute
        }
    }
}

RateLimiter::Lease::~Lease() {
    release();
}

RateLimiter::Lease::Lease(Lease&& other) noexcept {
    rate_limiter_ = other.rate_limiter_;
    credits_ = std::move(other.credits_);
    granted_ = other.granted_;
    other.rate_limiter_ = nullptr;
    other.granted_ = false;
}

RateLimiter::Lease& RateLimiter::Lease::operator=(Lease&& other) noexcept {
    if (this != &other) {
        release();
        rate_limiter_ = other.rate_limiter_;
        credits_ = std::move(other.credits_);
        granted_ = other.granted_;
        other.rate_limiter_ = nullptr;
        other.granted_ = false;
    }
    return *this;
}

void RateLimiter::Lease::release() {
    if (!granted_ || rate_limiter_ == nullptr) return;
    RateLimiter* rl = rate_limiter_;
    rate_limiter_ = nullptr;
    granted_ = false;
    {
        std::lock_guard<std::mutex> lock(rl->mu_);
        rl->releaseLocked(credits_);
    }
    rl->cv_.notify_all();
}

std::vector<RateLimiter::PoolInfo> RateLimiter::pools() const {
    std::vector<PoolInfo> out;
    std::lock_guard<std::mutex> lock(mu_);
    out.reserve(pools_.size());
    for (const auto& kv : pools_) {
        PoolInfo info;
        info.resource_name = kv.second.resource_name;
        info.global = kv.second.global;
        info.device = kv.second.device;
        info.capacity = kv.second.capacity;
        info.used = kv.second.used;
        out.push_back(std::move(info));
    }
    return out;
}

RateLimiter::Stats RateLimiter::stats() const {
    Stats s;
    s.admissions = admissions_.load(std::memory_order_relaxed);
    s.waits = waits_.load(std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(mu_);
        s.waiters_now = waiters_.size();
    }
    return s;
}

}  // namespace inferlite
