// gpu_memory_manager.hpp - CUDA device + pinned host memory pools (Phase 3).
//
// Manages reusable CUDA device memory buffers (for TensorRT input/output) and
// a pinned host memory pool (for efficient host<->device transfers). CPU-only
// models and plugins keep using the plain host MemoryManager.
//
// GPU support is opt-in: it is compiled only when INFERLITE_ENABLE_GPU is
// defined (set by CMake when a TensorRT SDK is available). When GPU support is
// disabled, all methods either return an error or are no-ops, so the CPU-only
// server builds and runs with no regression.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace inferlite {

#ifdef INFERLITE_ENABLE_GPU

// A device buffer acquired from the GPU memory pool. On destruction it is
// returned to the pool. Owns a CUDA device pointer.
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(void* ptr, size_t capacity, std::shared_ptr<void> mgr);
    ~DeviceBuffer();
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    void* data() const { return ptr_; }
    size_t capacity() const { return capacity_; }

private:
    void* ptr_ = nullptr;
    size_t capacity_ = 0;
    std::shared_ptr<void> mgr_;  // keeps the owning pool alive
};

// A pinned host buffer acquired from the pool. Used for host<->device copies.
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    PinnedBuffer(void* ptr, size_t capacity, std::shared_ptr<void> mgr);
    ~PinnedBuffer();
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    PinnedBuffer(PinnedBuffer&& other) noexcept;
    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept;

    void* data() const { return ptr_; }
    size_t capacity() const { return capacity_; }

private:
    void* ptr_ = nullptr;
    size_t capacity_ = 0;
    std::shared_ptr<void> mgr_;
};

// Pool of reusable CUDA device memory and pinned host memory.
class GpuMemoryManager : public std::enable_shared_from_this<GpuMemoryManager> {
public:
    explicit GpuMemoryManager(size_t min_granularity = 4096);

    // Acquire a device buffer of at least `requested` bytes.
    // Throws std::runtime_error with RESOURCE_EXHAUSTED semantics on failure.
    DeviceBuffer acquireDevice(size_t requested);

    // Acquire a pinned host buffer of at least `requested` bytes.
    PinnedBuffer acquirePinned(size_t requested);

    void releaseDevice(void* ptr, size_t capacity);
    void releasePinned(void* ptr, size_t capacity);

    // Total device memory currently pooled (bytes).
    size_t devicePoolBytes() const;
    // Total pinned host memory currently pooled (bytes).
    size_t pinnedPoolBytes() const;

private:
    size_t min_granularity_;
    mutable std::mutex mu_;
    std::unordered_map<size_t, std::vector<void*>> device_free_;
    std::unordered_map<size_t, std::vector<void*>> pinned_free_;
    size_t device_pool_bytes_ = 0;
    size_t pinned_pool_bytes_ = 0;
};

#else  // !INFERLITE_ENABLE_GPU

// Stub: GPU support not compiled in. These types exist so callers can still
// reference them, but any operation throws a clear error.
class DeviceBuffer {
public:
    void* data() const { return nullptr; }
    size_t capacity() const { return 0; }
};

class PinnedBuffer {
public:
    void* data() const { return nullptr; }
    size_t capacity() const { return 0; }
};

class GpuMemoryManager {
public:
    size_t devicePoolBytes() const { return 0; }
    size_t pinnedPoolBytes() const { return 0; }
};

#endif  // INFERLITE_ENABLE_GPU

}  // namespace inferlite
