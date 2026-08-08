#include "memory_manager.hpp"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <stdexcept>

namespace inferlite {

MemoryManager::MemoryManager(size_t min_granularity)
    : min_granularity_(std::max<size_t>(1, min_granularity)) {}

PooledBuffer::PooledBuffer(uint8_t* ptr, size_t capacity,
                           std::shared_ptr<MemoryManager> mgr)
    : data_(ptr), capacity_(capacity), mgr_(std::move(mgr)) {}

PooledBuffer::PooledBuffer(PooledBuffer&& other) noexcept {
    data_ = other.data_;
    capacity_ = other.capacity_;
    mgr_ = std::move(other.mgr_);
    other.data_ = nullptr;
    other.capacity_ = 0;
}

PooledBuffer& PooledBuffer::operator=(PooledBuffer&& other) noexcept {
    if (this != &other) {
        // Return any currently held buffer first.
        if (data_ && mgr_) mgr_->release(data_, capacity_);
        data_ = other.data_;
        capacity_ = other.capacity_;
        mgr_ = std::move(other.mgr_);
        other.data_ = nullptr;
        other.capacity_ = 0;
    }
    return *this;
}

PooledBuffer::~PooledBuffer() {
    if (data_ && mgr_) mgr_->release(data_, capacity_);
}

PooledBuffer MemoryManager::acquire(size_t requested) {
    // Round up to the pool granularity so small tensors map to shared buckets.
    size_t rounded = ((requested + min_granularity_ - 1) / min_granularity_) * min_granularity_;

    std::lock_guard<std::mutex> lock(mu_);
    auto it = free_.find(rounded);
    if (it != free_.end() && !it->second.empty()) {
        uint8_t* ptr = it->second.back();
        it->second.pop_back();
        if (it->second.empty()) free_.erase(it);
        return PooledBuffer(ptr, rounded, shared_from_this());
    }
    // No pooled buffer of this size available; allocate fresh.
    // Use raw new[] so we own the allocation explicitly.
    void* mem = std::malloc(rounded);
    if (!mem) {
        throw std::bad_alloc();
    }
    pool_bytes_ += rounded;
    return PooledBuffer(static_cast<uint8_t*>(mem), rounded, shared_from_this());
}

void MemoryManager::release(uint8_t* ptr, size_t capacity) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mu_);
    free_[capacity].push_back(ptr);
}

}  // namespace inferlite
