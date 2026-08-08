// tensor.hpp - Shared tensor and data-type definitions.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace inferlite {

// Minimal supported set of tensor data types (mirrors a subset of Triton's
// data_type enum). All are host CPU types; no device memory in phase 1.
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

struct Tensor {
    std::string name;
    DataType type = DataType::kInvalid;
    std::vector<int64_t> shape;
    // Raw byte payload (host memory, little-endian native layout).
    std::vector<uint8_t> data;
};

}  // namespace inferlite
