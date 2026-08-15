// tensor.hpp - Shared tensor and data-type definitions.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace inferlite {

// Minimal supported set of tensor data types (mirrors a subset of Triton's
// data_type enum). All are host CPU types in Phase 1/2; Phase 3 adds device
// tensors for the TensorRT GPU backend.
enum class DataType : int {
    kInt8 = 0,
    kUint8,
    kInt16,
    kUint16,
    kInt32,
    kUint32,
    kInt64,
    kUint64,
    kFloat16,
    kFloat32,
    kFloat64,
    kBool,
    kInvalid,
};

inline const char* dataTypeToString(DataType t) {
    switch (t) {
        case DataType::kInt8: return "INT8";
        case DataType::kUint8: return "UINT8";
        case DataType::kInt16: return "INT16";
        case DataType::kUint16: return "UINT16";
        case DataType::kInt32: return "INT32";
        case DataType::kUint32: return "UINT32";
        case DataType::kInt64: return "INT64";
        case DataType::kUint64: return "UINT64";
        case DataType::kFloat16: return "FP16";
        case DataType::kFloat32: return "FP32";
        case DataType::kFloat64: return "FP64";
        case DataType::kBool: return "BOOL";
        default: return "INVALID";
    }
}

inline DataType dataTypeFromString(const std::string& s) {
    if (s == "INT8") return DataType::kInt8;
    if (s == "UINT8") return DataType::kUint8;
    if (s == "INT16") return DataType::kInt16;
    if (s == "UINT16") return DataType::kUint16;
    if (s == "INT32") return DataType::kInt32;
    if (s == "UINT32") return DataType::kUint32;
    if (s == "INT64") return DataType::kInt64;
    if (s == "UINT64") return DataType::kUint64;
    if (s == "FP16") return DataType::kFloat16;
    if (s == "FLOAT" || s == "FP32") return DataType::kFloat32;
    if (s == "FP64") return DataType::kFloat64;
    if (s == "BOOL") return DataType::kBool;
    return DataType::kInvalid;
}

// Size in bytes of one scalar element of a data type.
inline size_t dataTypeSize(DataType t) {
    switch (t) {
        case DataType::kInt8:
        case DataType::kUint8:
        case DataType::kBool: return 1;
        case DataType::kInt16:
        case DataType::kUint16:
        case DataType::kFloat16: return 2;
        case DataType::kInt32:
        case DataType::kUint32:
        case DataType::kFloat32: return 4;
        case DataType::kInt64:
        case DataType::kUint64:
        case DataType::kFloat64: return 8;
        default: return 0;
    }
}

// Number of scalar elements described by a shape (empty shape == scalar == 1).
inline int64_t shapeElementCount(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

// Total bytes of a dense tensor with the given shape and type.
inline size_t tensorByteSize(const std::vector<int64_t>& shape, DataType type) {
    int64_t n = shapeElementCount(shape);
    return static_cast<size_t>(n) * dataTypeSize(type);
}

// Where a tensor's payload lives. CPU tensors hold bytes in `data` (host
// memory). GPU tensors hold a pointer to CUDA device memory in `device_ptr`
// (Phase 3 TensorRT backend); `data` is empty for device tensors unless a
// host copy has been materialized.
enum class TensorDevice : int { kCpu = 0, kGpu };

inline const char* tensorDeviceToString(TensorDevice d) {
    return d == TensorDevice::kGpu ? "GPU" : "CPU";
}

struct Tensor {
    std::string name;
    DataType type = DataType::kInvalid;
    std::vector<int64_t> shape;
    // Raw byte payload (host memory, little-endian native layout). For GPU
    // tensors this is empty (the data lives on the device).
    std::vector<uint8_t> data;
    // Device placement. kCpu: `data` is authoritative. kGpu: `device_ptr` is
    // the CUDA device pointer and `device_bytes` is the byte length.
    TensorDevice device = TensorDevice::kCpu;
    void* device_ptr = nullptr;   // CUDA device memory (GPU tensors only)
    size_t device_bytes = 0;      // byte length of the device buffer
    // Byte length of the logical tensor payload (== data.size() for CPU).
    size_t byteLength() const {
        return device == TensorDevice::kGpu ? device_bytes : data.size();
    }
};

// Write a single scalar value (from a double) into a tensor's little-endian
// binary layout, per the declared DataType. Used to serialize JSON number
// arrays and golden-test inputs. dst must point to a buffer >= dataTypeSize(dt).
inline void writeTensorScalar(uint8_t* dst, DataType dt, double v) {
    auto putInt = [&](int64_t x) {
        uint64_t uv = static_cast<uint64_t>(x);
        for (size_t b = 0; b < dataTypeSize(dt); ++b) {
            dst[b] = static_cast<uint8_t>((uv >> (8 * b)) & 0xFF);
        }
    };
    switch (dt) {
        case DataType::kInt8:
        case DataType::kUint8:
        case DataType::kInt16:
        case DataType::kUint16:
        case DataType::kInt32:
        case DataType::kUint32:
        case DataType::kInt64:
        case DataType::kUint64:
            putInt(static_cast<int64_t>(v));
            break;
        case DataType::kFloat16: {
            float fv = static_cast<float>(v);
            uint32_t fbits;
            std::memcpy(&fbits, &fv, sizeof(fbits));
            uint16_t h = static_cast<uint16_t>((fbits + 0x1000) >> 13);
            dst[0] = static_cast<uint8_t>(h & 0xFF);
            dst[1] = static_cast<uint8_t>((h >> 8) & 0xFF);
            break;
        }
        case DataType::kFloat32: {
            float fv = static_cast<float>(v);
            uint32_t bits;
            std::memcpy(&bits, &fv, sizeof(bits));
            dst[0] = static_cast<uint8_t>(bits & 0xFF);
            dst[1] = static_cast<uint8_t>((bits >> 8) & 0xFF);
            dst[2] = static_cast<uint8_t>((bits >> 16) & 0xFF);
            dst[3] = static_cast<uint8_t>((bits >> 24) & 0xFF);
            break;
        }
        case DataType::kFloat64: {
            uint64_t bits;
            std::memcpy(&bits, &v, sizeof(bits));
            for (size_t b = 0; b < 8; ++b) {
                dst[b] = static_cast<uint8_t>((bits >> (8 * b)) & 0xFF);
            }
            break;
        }
        case DataType::kBool:
            dst[0] = v != 0.0 ? 1 : 0;
            break;
        default:
            break;
    }
}

// Structured, stable error codes returned to clients (ISO 14971 risk control,
// no silent failures). Every failure mode maps to exactly one code.
enum class ErrorCode : int {
    kNone = 0,
    kInvalidInput,           // 400 input name/type/shape/size invalid
    kOutputValidationFailed, // 500 output failed NaN/Inf/range/shape checks
    kResourceExhausted,      // 503 queue at capacity
    kTimeout,                // 504 request exceeded allowed time
    kInternalError,          // 500 any other backend/plugin/runtime failure
    kModelNotFound,          // 404 unknown model
    kSelfTestFailed,         // 503 startup self-test did not pass
};

inline const char* errorCodeToString(ErrorCode c) {
    switch (c) {
        case ErrorCode::kNone: return "";
        case ErrorCode::kInvalidInput: return "INVALID_INPUT";
        case ErrorCode::kOutputValidationFailed: return "OUTPUT_VALIDATION_FAILED";
        case ErrorCode::kResourceExhausted: return "RESOURCE_EXHAUSTED";
        case ErrorCode::kTimeout: return "TIMEOUT";
        case ErrorCode::kInternalError: return "INTERNAL_ERROR";
        case ErrorCode::kModelNotFound: return "MODEL_NOT_FOUND";
        case ErrorCode::kSelfTestFailed: return "SELF_TEST_FAILED";
        default: return "UNKNOWN";
    }
}

}  // namespace inferlite
