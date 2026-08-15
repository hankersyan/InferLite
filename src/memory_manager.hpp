// memory_manager.hpp - Host / pinned / device memory pools (Phase 4).
//
// Phase 1 required only host (CPU) memory. Phase 4 adds:
//   - a page-aligned pinned host pool (for staging between host/CPU/NPU and
//     Intel GPU via OpenCL),
//   - an opaque Intel GPU (OpenCL) device-memory pool,
//   - tensor location tracking so the ensemble executor can decide when a copy
//     across a device boundary is required.
//
// The OpenCL device-memory pool is represented by an opaque, backend-owned
// allocation handle (the OpenVINO GPU backend creates the actual cl_mem via its
// own OpenCL context). This module provides the bookkeeping, location enum, and
// pinned host allocations so the scheduler/ensemble can reason about copies
// without embedding OpenCL headers here.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace inferlite {

// Where a tensor's bytes physically reside. The ensemble executor uses this to
// decide whether a zero-copy (same domain) or a copy (different domain) is
// needed when moving data between steps.
enum class MemoryDomain : int {
    kHost,          // CPU / NPU host memory (normal or pinned host)
    kOpenClDevice,  // Intel GPU device memory (OpenCL buffer)
    kCudaDevice,    // NVIDIA GPU device memory (CUDA buffer)
};

class MemoryManager;

// A host buffer acquired from the pool. On destruction it is automatically
// returned to the pool it came from.
class PooledBuffer {
public:
    PooledBuffer() = default;
    PooledBuffer(uint8_t* ptr, size_t capacity, std::shared_ptr<MemoryManager> mgr);
    ~PooledBuffer();

    PooledBuffer(const PooledBuffer&) = delete;
    PooledBuffer& operator=(const PooledBuffer&) = delete;

    PooledBuffer(PooledBuffer&& other) noexcept;
    PooledBuffer& operator=(PooledBuffer&& other) noexcept;

    uint8_t* data() const { return data_; }
    size_t capacity() const { return capacity_; }

private:
    uint8_t* data_ = nullptr;
    size_t capacity_ = 0;
    std::shared_ptr<MemoryManager> mgr_;
};

// A page-aligned (pinned) host buffer used to stage transfers between host
// memory and device memory (Intel GPU / NVIDIA GPU). Same RAII recycle pattern
// as PooledBuffer but allocated with page alignment and reclaimed on release.
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    PinnedBuffer(uint8_t* ptr, size_t capacity, std::shared_ptr<MemoryManager> mgr);
    ~PinnedBuffer();

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    PinnedBuffer(PinnedBuffer&& other) noexcept;
    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept;

    uint8_t* data() const { return data_; }
    size_t capacity() const { return capacity_; }

private:
    uint8_t* data_ = nullptr;
    size_t capacity_ = 0;
    std::shared_ptr<MemoryManager> mgr_;
};

class MemoryManager : public std::enable_shared_from_this<MemoryManager> {
public:
    explicit MemoryManager(size_t min_granularity = 4096);

    // --- Host pool (CPU / NPU) ---
    // Acquire a reusable host buffer with at least `requested` bytes.
    PooledBuffer acquire(size_t requested);
    // Return a host buffer to the pool. Called by PooledBuffer on destruction.
    void release(uint8_t* ptr, size_t capacity);

    // --- Pinned host pool (page-aligned, for device staging) ---
    // Acquire a page-aligned pinned buffer with at least `requested` bytes.
    PinnedBuffer acquirePinned(size_t requested);
    // Return a pinned buffer to the pool.
    void releasePinned(uint8_t* ptr, size_t capacity);

    size_t poolBytes() const {
        std::lock_guard<std::mutex> lock(mu_);
        return pool_bytes_;
    }

    size_t pinnedBytes() const {
        std::lock_guard<std::mutex> lock(mu_);
        return pinned_bytes_;
    }

    // --- Intel GPU (OpenCL) device-memory pool (bookkeeping only) ---
    // Allocate a device-side buffer. `dev_ptr` is an opaque backend-owned
    // pointer (e.g. a cl_mem cast to void*) created by the OpenCL context owner.
    // The manager only tracks lifetime/location; the caller owns the context.
    void registerDeviceBuffer(void* dev_ptr, size_t bytes, MemoryDomain domain);
    // Return a device buffer. `dev_ptr` and `bytes` must match a registered
    // buffer. The manager does NOT free OpenCL memory (the owner does); it only
    // releases bookkeeping so zero-copy bookkeeping stays consistent.
    void unregisterDeviceBuffer(void* dev_ptr, size_t bytes);
    // Query the domain (location) a device buffer was registered under.
    MemoryDomain deviceBufferDomain(void* dev_ptr) const;

    // Page size used for pinned allocations (Windows page granularity).
    static size_t pageSize();

private:
    void doRelease(uint8_t* ptr, size_t capacity);

    size_t min_granularity_;
    mutable std::mutex mu_;
    // Free host buffers grouped by capacity bucket (capacity -> list of ptrs).
    std::unordered_map<size_t, std::vector<uint8_t*>> free_;
    // Free pinned buffers grouped by capacity bucket.
    std::unordered_map<size_t, std::vector<uint8_t*>> free_pinned_;
    // Registered device buffers: ptr -> {bytes, domain}.
    std::unordered_map<void*, std::pair<size_t, MemoryDomain>> device_buffers_;
    size_t pool_bytes_ = 0;
    size_t pinned_bytes_ = 0;
};

}  // namespace inferlite
