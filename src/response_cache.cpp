// response_cache.cpp - implementation of the Triton-style response cache.
#include "response_cache.hpp"

#include <algorithm>

#include "sha256.hpp"

namespace inferlite {
namespace {

// Little-endian fixed-width encoders keep the key stable across platforms
// (the key is hashed, so the exact byte order only has to be self-consistent,
// but the fixed width prevents ambiguous framing of concatenated fields).
void appendU64(std::string& buf, uint64_t v) {
    for (int b = 0; b < 8; ++b) buf.push_back(static_cast<char>((v >> (8 * b)) & 0xFF));
}

void appendString(std::string& buf, const std::string& s) {
    appendU64(buf, static_cast<uint64_t>(s.size()));
    buf.append(s.data(), s.size());
}

void appendBytes(std::string& buf, const uint8_t* p, size_t n) {
    appendU64(buf, static_cast<uint64_t>(n));
    if (n) buf.append(reinterpret_cast<const char*>(p), n);
}

}  // namespace

ResponseCache::ResponseCache(size_t max_entries, size_t max_bytes)
    : max_entries_(max_entries), max_bytes_(max_bytes) {}

bool ResponseCache::lookup(const std::string& key, std::vector<Tensor>& out) {
    stats_.lookups.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(mu_);
    auto it = index_.find(key);
    if (it == index_.end()) return false;
    // MRU promotion: splice the hit to the front of the list.
    items_.splice(items_.begin(), items_, it->second);
    out = it->second->outputs;  // copy the stored tensors (payloads included)
    stats_.hits.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void ResponseCache::insert(const std::string& key, const std::vector<Tensor>& outputs) {
    // Only host-resident payloads are cacheable: serializers (HTTP base64,
    // gRPC) read tensor .data directly. A device-resident output without a
    // host copy would be cached as empty bytes, so refuse to store it.
    size_t bytes = 0;
    for (const auto& t : outputs) {
        if (t.device != TensorDevice::kCpu) return;
        bytes += t.data.size();
    }

    std::lock_guard<std::mutex> lock(mu_);

    // Refresh in place when the key is already present (a duplicate request
    // racing its own miss) so we do not leak capacity on re-inserts.
    auto it = index_.find(key);
    if (it != index_.end()) {
        bytes_ -= it->second->bytes;
        bytes_ += bytes;
        it->second->outputs = outputs;
        it->second->bytes = bytes;
        items_.splice(items_.begin(), items_, it->second);
        stats_.insertions.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    // Evict least-recently-used entries from the tail until the entry and byte
    // caps admit the new entry. A single payload larger than max_bytes_ still
    // gets admitted once the cache is empty (it simply crowds out its peers on
    // the next insert), matching Triton's "always admit the miss" behavior.
    while (!items_.empty() &&
           ((max_entries_ > 0 && items_.size() >= max_entries_) ||
            (max_bytes_ > 0 && bytes_ + bytes > max_bytes_))) {
        Entry& victim = items_.back();
        bytes_ -= victim.bytes;
        index_.erase(victim.key);
        items_.pop_back();
        stats_.evictions.fetch_add(1, std::memory_order_relaxed);
    }

    items_.emplace_front(Entry{key, outputs, bytes});
    index_[key] = items_.begin();
    bytes_ += bytes;
    stats_.insertions.fetch_add(1, std::memory_order_relaxed);
}

void ResponseCache::clear() {
    std::lock_guard<std::mutex> lock(mu_);
    items_.clear();
    index_.clear();
    bytes_ = 0;
}

size_t ResponseCache::entries() const {
    std::lock_guard<std::mutex> lock(mu_);
    return items_.size();
}

size_t ResponseCache::bytes() const {
    std::lock_guard<std::mutex> lock(mu_);
    return bytes_;
}

std::string makeResponseCacheKey(const std::string& model_name, int64_t version,
                                 const std::string& config_hash,
                                 const std::string& model_hash,
                                 const std::vector<Tensor>& inputs) {
    std::string buf;
    appendString(buf, model_name);
    appendU64(buf, static_cast<uint64_t>(version));
    appendString(buf, config_hash);
    appendString(buf, model_hash);

    // Canonical input order: inputs were validated upstream to be exactly the
    // declared model input set, so sorting by name makes the key independent
    // of client tensor ordering while remaining unambiguous.
    std::vector<size_t> order(inputs.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return inputs[a].name < inputs[b].name;
    });
    appendU64(buf, static_cast<uint64_t>(order.size()));
    for (size_t idx : order) {
        const Tensor& t = inputs[idx];
        appendString(buf, t.name);
        appendU64(buf, static_cast<uint64_t>(t.type));
        appendU64(buf, static_cast<uint64_t>(t.shape.size()));
        for (int64_t d : t.shape) appendU64(buf, static_cast<uint64_t>(d));
        appendBytes(buf, t.data.data(), t.data.size());
    }

    return hexEncode(
        sha256(reinterpret_cast<const uint8_t*>(buf.data()), buf.size()));
}

}  // namespace inferlite
