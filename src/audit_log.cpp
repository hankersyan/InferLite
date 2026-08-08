#include "audit_log.hpp"

#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "sha256.hpp"

namespace inferlite {

namespace {

std::string iso8601Utc() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream os;
    os << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return os.str();
}

std::string shapeToJson(const std::vector<int64_t>& shape) {
    std::ostringstream os;
    os << '[';
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) os << ',';
        os << shape[i];
    }
    os << ']';
    return os.str();
}

std::string extractField(const std::string& line, const std::string& key) {
    const std::string k = "\"" + key + "\":\"";
    size_t pos = line.find(k);
    if (pos == std::string::npos) return "";
    pos += k.size();
    size_t end = line.find('"', pos);
    if (end == std::string::npos) return "";
    return line.substr(pos, end - pos);
}

std::string extractBody(const std::string& line) {
    const std::string k = "\"entry\":";
    size_t pos = line.find(k);
    if (pos == std::string::npos) return "";
    pos += k.size();
    int depth = 0;
    bool in_str = false;
    size_t i = pos;
    for (; i < line.size(); ++i) {
        char c = line[i];
        if (in_str) {
            if (c == '\\') { ++i; continue; }
            if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') { in_str = true; continue; }
        if (c == '{') ++depth;
        else if (c == '}') {
            --depth;
            if (depth == 0) { ++i; break; }
        }
    }
    return line.substr(pos, i - pos);
}

}  // namespace

std::string computeChainHash(int64_t sequence, const std::string& entry_body,
                             const std::string& prev_hash) {
    std::string raw = std::to_string(sequence) + "|" + prev_hash + "|" + entry_body;
    return hexEncode(sha256(raw));
}

std::string auditEntryBodyOnly(const AuditEntry& e) {
    std::ostringstream body;
    body << "{\"trace_id\":\"" << e.trace_id << "\""
         << ",\"timestamp\":\"" << e.timestamp << "\""
         << ",\"model_id\":\"" << e.model_id << "\""
         << ",\"model_version\":\"" << e.model_version << "\""
         << ",\"model_hash\":\"" << e.model_hash << "\""
         << ",\"software_version\":\"" << e.software_version << "\""
         << ",\"config_hash\":\"" << e.config_hash << "\""
         << ",\"input_shape\":" << shapeToJson(e.input_shape)
         << ",\"inference_status\":\"" << e.inference_status << "\""
         << ",\"error_code\":\"" << e.error_code << "\""
         << ",\"duration_ms\":" << e.duration_ms
         << ",\"device\":\"" << e.device << "\"}";
    return body.str();
}

std::string auditEntryJson(int64_t seq, const std::string& body, const std::string& prev,
                           const std::string& chain) {
    std::ostringstream os;
    os << "{\"seq\":" << seq
       << ",\"prev_hash\":\"" << prev << "\""
       << ",\"entry\":" << body
       << ",\"chain_hash\":\"" << chain << "\"}\n";
    return os.str();
}

AuditLog::AuditLog(const std::string& path) : path_(path) {
    // If the file exists, scan to the last entry to recover the tail hash and
    // sequence number so the chain continues across restarts.
    std::ifstream in(path_, std::ios::binary);
    std::string last_line;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        last_line = line;
        ++sequence_;
    }
    in.close();

    if (!last_line.empty()) {
        std::string tail = extractField(last_line, "chain_hash");
        if (tail.empty()) {
            throw AuditError("audit log tail entry is malformed: " + path_);
        }
        tail_hash_ = tail;
    }
}

void AuditLog::write(const AuditEntry& e) {
    std::lock_guard<std::mutex> lock(mu_);
    std::ofstream out(path_, std::ios::binary | std::ios::app);
    if (!out) {
        throw AuditError("cannot open audit log for append: " + path_);
    }

    int64_t seq = ++sequence_;
    AuditEntry copy = e;
    if (copy.timestamp.empty()) copy.timestamp = iso8601Utc();
    std::string body = auditEntryBodyOnly(copy);
    std::string chain = computeChainHash(seq, body, tail_hash_);
    std::string line = auditEntryJson(seq, body, tail_hash_, chain);
    out.write(line.data(), static_cast<std::streamsize>(line.size()));
    out.flush();
    if (!out) {
        throw AuditError("failed to write audit log entry: " + path_);
    }
    tail_hash_ = chain;
}

std::string AuditLog::currentHash() const {
    std::lock_guard<std::mutex> lock(mu_);
    return tail_hash_;
}

bool AuditLog::verifyChain() const {
    std::lock_guard<std::mutex> lock(mu_);
    std::ifstream in(path_, std::ios::binary);
    if (!in) return false;
    std::string prev;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        const std::string skey = "\"seq\":";
        size_t sp = line.find(skey);
        if (sp == std::string::npos) return false;
        sp += skey.size();
        size_t se = line.find(',', sp);
        if (se == std::string::npos) return false;
        int64_t seq = std::stoll(line.substr(sp, se - sp));

        std::string body = extractBody(line);
        std::string stored_prev = extractField(line, "prev_hash");
        std::string stored_chain = extractField(line, "chain_hash");
        if (stored_prev != prev) return false;  // chain broken

        std::string recomputed = computeChainHash(seq, body, prev);
        if (recomputed != stored_chain) return false;
        prev = stored_chain;
    }
    return true;
}

}  // namespace inferlite
