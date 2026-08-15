#include "memory_manager.hpp"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <stdexcept>

namespace inferlite {

namespace {
// Page-aligned allocation for pinned host memory (usable as staging buffers
// for OpenCL/CUDA transfers). Windows provides _aligned_malloc with power-of-2
// alignment; elsewhere we use posix_memalign/aligned_alloc where available.
void* alignedAlloc(size_t bytes, size_t alignment) {
#ifdef _WIN32
    return _aligned_malloc(bytes, alignment);
#elif defined(__APPLE__)
    void* p = nullptr;
    if (posix_memalign(&p, alignment, bytes) != 0) return nullptr;
    return p;
#else
    return std::aligned_alloc(alignment, bytes);
#endif
}

void alignedFree(void* p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
}
}  // namespace

size_t MemoryManager::pageSize() {
#ifdef _WIN32
    static const size_t kPage = 4096;  // standard x64 page size
    return kPage;
#else
    static const size_t kPage = 4096;
    return kPage;
#endif
}

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

// --- PinnedBuffer RAII ------------------------------------------------------
PinnedBuffer::PinnedBuffer(uint8_t* ptr, size_t capacity, std::shared_ptr<MemoryManager> mgr)
    : data_(ptr), capacity_(capacity), mgr_(std::move(mgr)) {}

PinnedBuffer::PinnedBuffer(PinnedBuffer&& other) noexcept {
    data_ = other.data_;
    capacity_ = other.capacity_;
    mgr_ = std::move(other.mgr_);
    other.data_ = nullptr;
    other.capacity_ = 0;
}

PinnedBuffer& PinnedBuffer::operator=(PinnedBuffer&& other) noexcept {
    if (this != &other) {
        if (data_ && mgr_) mgr_->releasePinned(data_, capacity_);
        data_ = other.data_;
        capacity_ = other.capacity_;
        mgr_ = std::move(other.mgr_);
        other.data_ = nullptr;
        other.capacity_ = 0;
    }
    return *this;
}

PinnedBuffer::~PinnedBuffer() {
    if (data_ && mgr_) mgr_->releasePinned(data_, capacity_);
}

PinnedBuffer MemoryManager::acquirePinned(size_t requested) {
    // Round up to page size so pinned buffers are page-aligned and page-count
    // based, matching OpenCL/CUDA staging requirements.
    const size_t page = pageSize();
    size_t rounded = ((requested + page - 1) / page) * page;

    std::lock_guard<std::mutex> lock(mu_);
    auto it = free_pinned_.find(rounded);
    if (it != free_pinned_.end() && !it->second.empty()) {
        uint8_t* ptr = it->second.back();
        it->second.pop_back();
        if (it->second.empty()) free_pinned_.erase(it);
        return PinnedBuffer(ptr, rounded, shared_from_this());
    }
    void* mem = alignedAlloc(rounded, page);
    if (!mem) {
        throw std::bad_alloc();
    }
    pinned_bytes_ += rounded;
    return PinnedBuffer(static_cast<uint8_t*>(mem), rounded, shared_from_this());
}

void MemoryManager::releasePinned(uint8_t* ptr, size_t capacity) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mu_);
    free_pinned_[capacity].push_back(ptr);
}

// --- Device (OpenCL) buffer bookkeeping ------------------------------------
void MemoryManager::registerDeviceBuffer(void* dev_ptr, size_t bytes, MemoryDomain domain) {
    if (!dev_ptr) return;
    std::lock_guard<std::mutex> lock(mu_);
    device_buffers_[dev_ptr] = {bytes, domain};
}

void MemoryManager::unregisterDeviceBuffer(void* dev_ptr, size_t bytes) {
    if (!dev_ptr) return;
    std::lock_guard<std::mutex> lock(mu_);
    auto it = device_buffers_.find(dev_ptr);
    if (it != device_buffers_.end() && it->second.first == bytes) {
        device_buffers_.erase(it);
    }
}

MemoryDomain MemoryManager::deviceBufferDomain(void* dev_ptr) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = device_buffers_.find(dev_ptr);
    return it == device_buffers_.end() ? MemoryDomain::kHost : it->second.second;
}

}  // namespace inferlite
