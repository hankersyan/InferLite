// audit_log.hpp - Tamper-evident, append-only inference audit log (FDA traceability).
//
// Each entry is a single-line JSON object with a `prev_hash` field chaining it to
// the previous entry, plus a `chain_hash` covering the entry's own content and the
// previous chain hash. Tampering with any entry breaks the chain. No patient data
// is ever stored. Logging is synchronous and serialized under a mutex so entries
// are always written in a total order.
#pragma once

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"

namespace inferlite {

struct AuditEntry {
    std::string trace_id;
    std::string timestamp;  // ISO8601 UTC
    std::string model_id;
    std::string model_version;
    std::string model_hash;
    std::string software_version;
    std::string config_hash;
    std::vector<int64_t> input_shape;
    std::string inference_status;  // SUCCESS / FAILURE / TIMEOUT / REJECTED
    std::string error_code;        // empty on success
    int64_t duration_ms = 0;
    std::string device = "CPU";
};

// Thrown on audit log I/O failures (fail-fast when unable to open the log).
struct AuditError : public std::runtime_error {
    explicit AuditError(const std::string& msg) : std::runtime_error(msg) {}
};

class AuditLog {
public:
    // Opens (appends to) the log file at `path` and computes the chain starting
    // hash from the last entry present, so the chain continues across restarts.
    explicit AuditLog(const std::string& path);

    // Write one entry (hash-chained). Throws AuditError on I/O failure.
    void write(const AuditEntry& entry);

    // Verify the entire chain from the file. Returns true if every entry's
    // prev_hash/chain_hash is consistent. Used by health/detailed and tests.
    bool verifyChain() const;

    // Current chain tail hash (empty if no entries yet).
    std::string currentHash() const;

private:
    std::string path_;
    mutable std::mutex mu_;
    std::string tail_hash_;  // chain hash of the last written entry (hex)
    int64_t sequence_ = 0;   // strictly increasing per-entry sequence
};

// Compute the chain hash over (sequence + entry content + prev_hash).
std::string computeChainHash(int64_t sequence, const std::string& entry_body,
                             const std::string& prev_hash);

// Build the entry body (the fields inside the "entry" object) as compact JSON.
std::string auditEntryBodyOnly(const AuditEntry& entry);

// Build a full single-line log record: {seq, prev_hash, entry, chain_hash}.
std::string auditEntryJson(int64_t seq, const std::string& body,
                           const std::string& prev, const std::string& chain);

}  // namespace inferlite
