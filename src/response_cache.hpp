// response_cache.hpp - Triton-style per-model response cache.
//
// Mirrors NVIDIA Triton's "response cache" for deterministic models. When a
// model config opts in (`response_cache { enable: true }`), requests whose
// inputs were already answered are served from a bounded LRU of previously
// computed responses instead of queueing and executing the model again.
//
// Safety: caching is correct ONLY for deterministic models, so enabling it is
// an operator assertion of determinism (the pipeline is audited/deployed as
// such). Stateful models (sequence_batching) are rejected at config
// validation; nothing here re-checks statefulness.
//
// Triton keys cached responses on model name + version + input tensors; we add
// the model's config hash and file hash to the key, so any change to the
// config (including a load-time override) or to the model artifacts yields a
// cache miss. On top of that each successful load builds a brand-new cache, so
// entries can never outlive the exact model generation that produced them even
// when a reload leaves a stale reference in flight.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.hpp"

namespace inferlite {

// Cache accounting, exposed per model through /v2/metrics (the local analogs of
// Triton's nv_cache_lookup_count / nv_cache_hit_count counters).
struct ResponseCacheStats {
    std::atomic<uint64_t> lookups{0};     // requests checked against the cache
    std::atomic<uint64_t> hits{0};        // lookups answered from the cache
    std::atomic<uint64_t> insertions{0};  // successful stores (cache misses)
    std::atomic<uint64_t> evictions{0};   // LRU / capacity evictions
};

// A bounded LRU keyed by a response-cache key string (see
// makeResponseCacheKey). Entries hold deep copies of the response tensors so a
// stored response is independent of the request that produced it. Thread-safe:
// concurrent lookups/inserts from the HTTP and gRPC paths are serialized.
class ResponseCache {
public:
    // max_entries / max_bytes bound the cache. A value of 0 in either
    // dimension means that dimension is unbounded.
    ResponseCache(size_t max_entries, size_t max_bytes);
    ~ResponseCache() = default;

    ResponseCache(const ResponseCache&) = delete;
    ResponseCache& operator=(const ResponseCache&) = delete;

    // Look up `key`. On a hit, copies the stored outputs into `out`, refreshes
    // the entry as most-recently-used and returns true; otherwise returns
    // false and leaves `out` untouched. Never throws.
    bool lookup(const std::string& key, std::vector<Tensor>& out);

    // Store a deep copy of `outputs` under `key`. Host-resident (CPU) tensors
    // are stored; a response containing a device tensor that has no host copy
    // is not cached at all. Least-recently-used entries are evicted when a
    // capacity bound would be exceeded.
    void insert(const std::string& key, const std::vector<Tensor>& outputs);

    // Drop every entry and free the stored payloads. Used when a model is
    // unloaded or a cache is being retired.
    void clear();

    const ResponseCacheStats& stats() const { return stats_; }
    size_t entries() const;  // entries currently held
    size_t bytes() const;    // payload bytes currently held

private:
    struct Entry {
        std::string key;
        std::vector<Tensor> outputs;
        size_t bytes = 0;
    };

    mutable std::mutex mu_;
    const size_t max_entries_;  // 0 = unbounded
    const size_t max_bytes_;    // 0 = unbounded
    // Front = most recently used. Entries live in the list and are referenced
    // by index_ for O(1) lookup and MRU promotion.
    std::list<Entry> items_;
    std::unordered_map<std::string, std::list<Entry>::iterator> index_;
    size_t bytes_ = 0;
    ResponseCacheStats stats_;
};

// Build a stable cache key for one inference request. Canonicalizes inputs by
// name so two requests carrying the same tensors in a different order are
// equivalent (Triton likewise hashes inputs independent of client ordering).
//
// The key is the lowercase hex SHA-256 of:
//   model name | loaded version | config hash | model file hash
//   | per-input { name | data type | shape dims | byte length | contents }
// with fixed-width length prefixes, so it is deterministic across processes
// serving the same model generation and collision-resistant (256-bit digest).
std::string makeResponseCacheKey(const std::string& model_name, int64_t version,
                                 const std::string& config_hash,
                                 const std::string& model_hash,
                                 const std::vector<Tensor>& inputs);

}  // namespace inferlite
