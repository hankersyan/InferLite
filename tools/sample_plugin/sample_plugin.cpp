// sample_plugin.cpp - Example CPU plugin implementing the InferLite plugin ABI.
//
// Behavior is configured per model via the config.pbtxt `parameters` block, so
// each pipeline owns its own pre/post-processing. Supported keys:
//   mode:       "preprocess" | "postprocess" | "identity"
//               (default: inferred from the model name substring)
//   scale:      multiplier applied to every element (preprocess)
//   offset:     value added after clamping (postprocess)
//   clamp_min:  lower clamp bound (postprocess; disabled if absent)
//   clamp_max:  upper clamp bound (postprocess; disabled if absent)
//
// Mode defaults (used when no parameters are given) preserve the Phase 2
// behavior for the stock preprocess/postprocess plugin models:
//   preprocess  -> scale 0.5
//   postprocess -> clamp [0, 100], offset 0.5
//
// Compile as a shared library:
//   cl /LD sample_plugin.cpp /Fe:sample_plugin.dll
//
// The plugin runs on host tensors only (CPU). It writes results in place into
// the caller-provided output buffers, which are sized per the declared output
// specs (static shapes only in Phase 2).
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "../../src/plugin_api.hpp"

namespace {

struct SampleNode {
    std::string mode = "identity";
    double scale = 1.0;
    double offset = 0.0;
    bool clamp_min_set = false;
    bool clamp_max_set = false;
    double clamp_min = 0.0;
    double clamp_max = 0.0;
    int32_t out_dtype = 0;
    int64_t out_elem_count = 0;
};

// Locate a parameter value by key (case-sensitive, Triton-style).
const char* findParam(const PluginNodeInfo* info, const char* key) {
    if (!info || !info->parameter_keys || !info->parameter_values) return nullptr;
    for (int32_t i = 0; i < info->parameter_count; ++i) {
        if (info->parameter_keys[i] && std::strcmp(info->parameter_keys[i], key) == 0) {
            return info->parameter_values[i];
        }
    }
    return nullptr;
}

bool parseDouble(const char* s, double& out) {
    if (!s || !*s) return false;
    char* end = nullptr;
    double v = std::strtod(s, &end);
    if (end == s) return false;
    out = v;
    return true;
}

// Write a float value into a tensor buffer of the given data type.
void writeElement(uint8_t* dst, int32_t dtype, double v) {
    switch (dtype) {
        case 9: {  // kFloat32
            float f = static_cast<float>(v);
            std::memcpy(dst, &f, 4);
            break;
        }
        case 8: {  // kFloat16
            float f = static_cast<float>(v);
            uint32_t bits;
            std::memcpy(&bits, &f, 4);
            uint16_t h = static_cast<uint16_t>((bits + 0x1000) >> 13);
            std::memcpy(dst, &h, 2);
            break;
        }
        default:
            break;
    }
}

double readElement(const uint8_t* src, int32_t dtype) {
    switch (dtype) {
        case 9: {  // kFloat32
            float f;
            std::memcpy(&f, src, 4);
            return f;
        }
        default:
            return 0.0;
    }
}

}  // namespace

extern "C" INFERLITE_PLUGIN_API PluginNodeHandle inferlite_plugin_create(
    const PluginNodeInfo* info, char* errbuf, size_t errbuf_size) {
    (void)errbuf_size;
    SampleNode* node = new SampleNode();
    if (info && info->model_name) {
        std::string name = info->model_name;
        if (name.find("preprocess") != std::string::npos) node->mode = "preprocess";
        else if (name.find("postprocess") != std::string::npos) node->mode = "postprocess";
    }

    // Mode defaults (Phase 2 parity) before parameter overrides.
    if (node->mode == "preprocess") {
        node->scale = 0.5;
    } else if (node->mode == "postprocess") {
        node->clamp_min_set = true;
        node->clamp_min = 0.0;
        node->clamp_max_set = true;
        node->clamp_max = 100.0;
        node->offset = 0.5;
    }

    // Per-model parameters: each pipeline owns its pre/post behavior.
    const char* v = nullptr;
    if ((v = findParam(info, "mode")) && *v) node->mode = v;
    if ((v = findParam(info, "scale")) && parseDouble(v, node->scale)) {}
    if ((v = findParam(info, "offset")) && parseDouble(v, node->offset)) {}
    if ((v = findParam(info, "clamp_min")) && parseDouble(v, node->clamp_min)) {
        node->clamp_min_set = true;
    }
    if ((v = findParam(info, "clamp_max")) && parseDouble(v, node->clamp_max)) {
        node->clamp_max_set = true;
    }

    if (info && info->outputs && info->output_count > 0) {
        node->out_dtype = info->outputs[0].data_type;
        node->out_elem_count = 1;
        for (int i = 0; i < info->outputs[0].rank; ++i) {
            node->out_elem_count *= info->outputs[0].dims[i];
        }
    }
    return node;
}

extern "C" INFERLITE_PLUGIN_API const char* inferlite_plugin_version() {
    return "sample_plugin-1.1.0";
}

extern "C" INFERLITE_PLUGIN_API int inferlite_plugin_execute(
    PluginNodeHandle h, const InferliteTensorDesc* inputs, int32_t input_count,
    const InferliteTensorDesc* outputs, int32_t output_count, char* errbuf,
    size_t errbuf_size) {
    (void)errbuf_size;
    SampleNode* node = static_cast<SampleNode*>(h);
    if (!node || output_count < 1 || input_count < 1) {
        if (errbuf) std::snprintf(errbuf, errbuf_size, "bad args");
        return 1;
    }

    const InferliteTensorDesc& in = inputs[0];
    const InferliteTensorDesc& out = outputs[0];

    if (!in.data || !out.data) {
        if (errbuf) std::snprintf(errbuf, errbuf_size, "null tensor data");
        return 1;
    }

    // The ABI exposes output buffers as const pointers; the server owns the
    // writable buffers, so cast away const for in-place writing.
    uint8_t* out_data = const_cast<uint8_t*>(out.data);

    int64_t n = 1;
    for (int i = 0; i < in.rank; ++i) n *= in.dims[i];
    size_t elem = in.byte_size / (n > 0 ? n : 1);

    for (int64_t i = 0; i < n; ++i) {
        double v = readElement(in.data + i * elem, in.data_type);
        if (node->mode == "preprocess") {
            v = v * node->scale;
        } else if (node->mode == "postprocess") {
            if (node->clamp_min_set && v < node->clamp_min) v = node->clamp_min;
            if (node->clamp_max_set && v > node->clamp_max) v = node->clamp_max;
            v = v + node->offset;
        }  // "identity": pass through
        writeElement(out_data + i * elem, out.data_type, v);
    }
    return 0;
}

extern "C" INFERLITE_PLUGIN_API void inferlite_plugin_destroy(PluginNodeHandle h) {
    delete static_cast<SampleNode*>(h);
}
