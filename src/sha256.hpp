// sha256.hpp - FIPS 180-4 SHA-256 for model integrity verification and
// tamper-evident audit chaining. Self-contained, no external crypto dependency.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace inferlite {

// Compute the SHA-256 digest of the given bytes. Returns the 32-byte digest.
std::vector<uint8_t> sha256(const uint8_t* data, size_t len);

// Convenience overloads.
inline std::vector<uint8_t> sha256(const std::string& s) {
    return sha256(reinterpret_cast<const uint8_t*>(s.data()), s.size());
}
inline std::vector<uint8_t> sha256(const std::vector<uint8_t>& v) {
    return sha256(v.data(), v.size());
}

// Hex-encode a byte range (lowercase). Used for manifest hashes and the audit
// chain values that are reported/logged.
std::string hexEncode(const uint8_t* data, size_t len);
inline std::string hexEncode(const std::vector<uint8_t>& v) {
    return hexEncode(v.data(), v.size());
}

// SHA-256 of a file on disk. Throws std::runtime_error if the file cannot be
// opened or read.
std::string sha256FileHex(const std::string& path);

}  // namespace inferlite
