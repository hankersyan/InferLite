// plugin_api.hpp - Stable C ABI between InferLite and CPU plugin shared
// libraries. Plugins are compiled separately and loaded at startup; the server
// verifies each plugin library hash against the manifest before loading.
//
// A plugin library must export exactly one entry point:
//
//   extern "C" PluginNodeHandle inferlite_plugin_create(const PluginNodeInfo* info,
//                                                       char* errbuf, size_t errbuf_size);
//
// It may optionally export inferlite_plugin_version() to report a version.
#pragma once

#include <cstddef>
#include <cstdint>

#ifdef _WIN32
#define INFERLITE_PLUGIN_API __declspec(dllexport)
#else
#define INFERLITE_PLUGIN_API __attribute__((visibility("default")))
#endif

// Tensor description passed to a plugin node (read-only metadata).
typedef struct InferliteTensorDesc {
    const char* name;
    int32_t data_type;      // matches inferlite::DataType
    const int64_t* dims;    // shape
    int32_t rank;
    const uint8_t* data;    // host pointer
    size_t byte_size;
} InferliteTensorDesc;

// Configuration supplied at node creation.
typedef struct PluginNodeInfo {
    const char* model_name;
    const char* plugin_library;
    const InferliteTensorDesc* inputs;  // declared input specs (data may be null)
    int32_t input_count;
    const InferliteTensorDesc* outputs;  // declared output specs
    int32_t output_count;
} PluginNodeInfo;

// Opaque handle to a plugin node instance.
typedef void* PluginNodeHandle;

// Optional: report a version string. Return a static string. If absent, the
// server treats the version as "unknown".
typedef const char* (*inferlite_plugin_version_fn)(void);

#ifdef __cplusplus
extern "C" {
#endif

// Create a plugin node. On success return a non-null handle. On failure return
// NULL and write a NUL-terminated message into errbuf.
typedef PluginNodeHandle (*inferlite_plugin_create_fn)(const PluginNodeInfo* info, char* errbuf,
                                                       size_t errbuf_size);

// Execute the node on host tensors. `inputs`/`input_count` and
// `outputs`/`output_count` describe actual tensors. Outputs must be filled in
// place (data pointers to caller-allocated host buffers sized per declared
// output specs). Returns 0 on success, non-zero on failure.
typedef int (*inferlite_plugin_execute_fn)(PluginNodeHandle node, const InferliteTensorDesc* inputs,
                                           int32_t input_count,
                                           const InferliteTensorDesc* outputs,
                                           int32_t output_count, char* errbuf,
                                           size_t errbuf_size);

// Destroy the node.
typedef void (*inferlite_plugin_destroy_fn)(PluginNodeHandle node);

#ifdef __cplusplus
}
#endif
