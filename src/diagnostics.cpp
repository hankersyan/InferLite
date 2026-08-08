#include "diagnostics.hpp"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace inferlite {

namespace {

std::string timestamp() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream os;
    os << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return os.str();
}

}  // namespace

Diagnostics::Diagnostics(const std::string& path) : path_(path) {}

Diagnostics::~Diagnostics() = default;

void Diagnostics::log(const std::string& level, const std::string& message) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!enabled_) return;
    std::string line = "[" + timestamp() + "] [" + level + "] " + message + "\n";
    std::cerr << line << std::flush;
    if (!path_.empty()) {
        std::ofstream out(path_, std::ios::binary | std::ios::app);
        if (out) {
            out.write(line.data(), static_cast<std::streamsize>(line.size()));
            out.flush();
        }
    }
}

}  // namespace inferlite
