// gpu_memory_manager.cpp - CUDA device + pinned host memory pools (Phase 3).
#include "gpu_memory_manager.hpp"

#include <algorithm>
#include <string>

namespace inferlite {

#ifdef INFERLITE_ENABLE_GPU

#include "cuda_runtime.h"

namespace {

// Throw a runtime_error describing the last CUDA error, prefixed with `what`.
[[noreturn]] void throwCuda(const char* what, cudaError_t err) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

}  // namespace

DeviceBuffer::DeviceBuffer(void* ptr, size_t capacity, std::shared_ptr<void> mgr)
    : ptr_(ptr), capacity_(capacity), mgr_(std::move(mgr)) {}

DeviceBuffer::~DeviceBuffer() {
    if (ptr_ && mgr_) {
        auto* g = static_cast<GpuMemoryManager*>(mgr_.get());
        g->releaseDevice(ptr_, capacity_);
    }
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept {
    ptr_ = other.ptr_;
    capacity_ = other.capacity_;
    mgr_ = std::move(other.mgr_);
    other.ptr_ = nullptr;
    other.capacity_ = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        if (ptr_ && mgr_) {
            static_cast<GpuMemoryManager*>(mgr_.get())->releaseDevice(ptr_, capacity_);
        }
        ptr_ = other.ptr_;
        capacity_ = other.capacity_;
        mgr_ = std::move(other.mgr_);
        other.ptr_ = nullptr;
        other.capacity_ = 0;
    }
    return *this;
}

PinnedBuffer::PinnedBuffer(void* ptr, size_t capacity, std::shared_ptr<void> mgr)
    : ptr_(ptr), capacity_(capacity), mgr_(std::move(mgr)) {}

PinnedBuffer::~PinnedBuffer() {
    if (ptr_ && mgr_) {
        static_cast<GpuMemoryManager*>(mgr_.get())->releasePinned(ptr_, capacity_);
    }
}

PinnedBuffer::PinnedBuffer(PinnedBuffer&& other) noexcept {
    ptr_ = other.ptr_;
    capacity_ = other.capacity_;
    mgr_ = std::move(other.mgr_);
    other.ptr_ = nullptr;
    other.capacity_ = 0;
}

PinnedBuffer& PinnedBuffer::operator=(PinnedBuffer&& other) noexcept {
    if (this != &other) {
        if (ptr_ && mgr_) {
            static_cast<GpuMemoryManager*>(mgr_.get())->releasePinned(ptr_, capacity_);
        }
        ptr_ = other.ptr_;
        capacity_ = other.capacity_;
        mgr_ = std::move(other.mgr_);
        other.ptr_ = nullptr;
        other.capacity_ = 0;
    }
    return *this;
}

GpuMemoryManager::GpuMemoryManager(size_t min_granularity)
    : min_granularity_(std::max<size_t>(1, min_granularity)) {}

DeviceBuffer GpuMemoryManager::acquireDevice(size_t requested) {
    size_t rounded = ((requested + min_granularity_ - 1) / min_granularity_) * min_granularity_;
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = device_free_.find(rounded);
        if (it != device_free_.end() && !it->second.empty()) {
            void* ptr = it->second.back();
            it->second.pop_back();
            if (it->second.empty()) device_free_.erase(it);
            return DeviceBuffer(ptr, rounded, shared_from_this());
        }
    }
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, rounded);
    if (err != cudaSuccess || ptr == nullptr) {
        throwCuda("cudaMalloc (device memory pool)", err);
    }
    {
        std::lock_guard<std::mutex> lock(mu_);
        device_pool_bytes_ += rounded;
    }
    return DeviceBuffer(ptr, rounded, shared_from_this());
}

PinnedBuffer GpuMemoryManager::acquirePinned(size_t requested) {
    size_t rounded = ((requested + min_granularity_ - 1) / min_granularity_) * min_granularity_;
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = pinned_free_.find(rounded);
        if (it != pinned_free_.end() && !it->second.empty()) {
            void* ptr = it->second.back();
            it->second.pop_back();
            if (it->second.empty()) pinned_free_.erase(it);
            return PinnedBuffer(ptr, rounded, shared_from_this());
        }
    }
    void* ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, rounded);
    if (err != cudaSuccess || ptr == nullptr) {
        throwCuda("cudaMallocHost (pinned memory pool)", err);
    }
    {
        std::lock_guard<std::mutex> lock(mu_);
        pinned_pool_bytes_ += rounded;
    }
    return PinnedBuffer(ptr, rounded, shared_from_this());
}

void GpuMemoryManager::releaseDevice(void* ptr, size_t capacity) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mu_);
    device_free_[capacity].push_back(ptr);
}

void GpuMemoryManager::releasePinned(void* ptr, size_t capacity) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mu_);
    pinned_free_[capacity].push_back(ptr);
}

size_t GpuMemoryManager::devicePoolBytes() const {
    std::lock_guard<std::mutex> lock(mu_);
    return device_pool_bytes_;
}

size_t GpuMemoryManager::pinnedPoolBytes() const {
    std::lock_guard<std::mutex> lock(mu_);
    return pinned_pool_bytes_;
}

#else  // !INFERLITE_ENABLE_GPU

// GPU support disabled: nothing to compile.

#endif  // INFERLITE_ENABLE_GPU

}  // namespace inferlite
