// memory_manager.hpp - Host memory pool for reusable inference buffers.
//
// Phase 1 only requires host (CPU) memory. Buffers are sized to the expected
// tensor dimensions (rounded up to a minimum granularity) and recycled after
// the HTTP response is sent. No GPU/device memory and no pinned memory.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace inferlite {

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

class MemoryManager : public std::enable_shared_from_this<MemoryManager> {
public:
    explicit MemoryManager(size_t min_granularity = 4096);

    // Acquire a reusable buffer with at least `requested` bytes. Reuses an
    // existing pooled buffer of sufficient capacity if available, otherwise
    // allocates new memory.
    PooledBuffer acquire(size_t requested);

    // Return a buffer to the pool. Called by PooledBuffer on destruction.
    void release(uint8_t* ptr, size_t capacity);

    size_t poolBytes() const {
        std::lock_guard<std::mutex> lock(mu_);
        return pool_bytes_;
    }

private:
    void doRelease(uint8_t* ptr, size_t capacity);

    size_t min_granularity_;
    mutable std::mutex mu_;
    // Free buffers grouped by capacity bucket (capacity -> list of raw ptrs).
    std::unordered_map<size_t, std::vector<uint8_t*>> free_;
    size_t pool_bytes_ = 0;
};

}  // namespace inferlite
