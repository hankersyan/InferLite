// diagnostics.hpp - Engineer-facing diagnostic log (separate from the audit
// trail). Human-readable operational events: queue depth, startup, errors.
// Thread-safe, synchronous appends.
#pragma once

#include <mutex>
#include <string>

namespace inferlite {

class Diagnostics {
public:
    // `path` is the diagnostic log file; empty disables file logging. Events
    // are also echoed to stderr.
    explicit Diagnostics(const std::string& path);
    ~Diagnostics();

    Diagnostics(const Diagnostics&) = delete;
    Diagnostics& operator=(const Diagnostics&) = delete;

    void log(const std::string& level, const std::string& message);

    void info(const std::string& m) { log("INFO", m); }
    void warn(const std::string& m) { log("WARN", m); }
    void error(const std::string& m) { log("ERROR", m); }

    // Reset counters / reopen if needed.
    void setEnabled(bool e) {
        std::lock_guard<std::mutex> lock(mu_);
        enabled_ = e;
    }

private:
    std::string path_;
    mutable std::mutex mu_;
    bool enabled_ = true;
};

}  // namespace inferlite
