#include "plugin_backend.hpp"

#include <cstring>
#include <stdexcept>

#include "plugin_api.hpp"
#include "pbtxt.hpp"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace inferlite {

namespace {

void* loadLibrary(const std::string& path) {
#if defined(_WIN32)
    HMODULE h = ::LoadLibraryA(path.c_str());
    if (!h) {
        throw std::runtime_error("failed to load plugin library '" + path +
                                 "': Win32 error " + std::to_string(::GetLastError()));
    }
    return reinterpret_cast<void*>(h);
#else
    void* h = ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        throw std::runtime_error("failed to load plugin library '" + path + "': " +
                                 std::string(::dlerror()));
    }
    return h;
#endif
}

void* resolveSymbol(void* handle, const char* name) {
#if defined(_WIN32)
    void* p = reinterpret_cast<void*>(::GetProcAddress(
        reinterpret_cast<HMODULE>(handle), name));
    if (!p) {
        throw std::runtime_error("plugin library missing required symbol '" +
                                 std::string(name) + "'");
    }
    return p;
#else
    void* p = ::dlsym(handle, name);
    if (!p) {
        throw std::runtime_error("plugin library missing required symbol '" +
                                 std::string(name) + "'");
    }
    return p;
#endif
}

}  // namespace

PluginBackend::PluginBackend() = default;

PluginBackend::~PluginBackend() {
    unload();
}

void PluginBackend::load(const ModelConfig& config, const std::string& lib_path) {
    if (config.plugin_library.empty()) {
        throw std::runtime_error("plugin backend model '" + config.name +
                                 "' has no 'plugin_library' configured");
    }

    handle_ = loadLibrary(lib_path);

    try {
        // Locate the factory.
        auto create = reinterpret_cast<inferlite_plugin_create_fn>(
            resolveSymbol(handle_, "inferlite_plugin_create"));

        // Build the PluginNodeInfo from the declared config specs.
        std::vector<InferliteTensorDesc> in_specs(config.inputs.size());
        std::vector<std::vector<int64_t>> in_dims(config.inputs.size());
        for (size_t i = 0; i < config.inputs.size(); ++i) {
            const auto& s = config.inputs[i];
            in_dims[i] = s.dims;
            in_specs[i].name = s.name.c_str();
            in_specs[i].data_type = static_cast<int32_t>(s.data_type);
            in_specs[i].dims = in_dims[i].data();
            in_specs[i].rank = static_cast<int32_t>(s.dims.size());
            in_specs[i].data = nullptr;
            in_specs[i].byte_size = 0;
        }
        std::vector<InferliteTensorDesc> out_specs(config.outputs.size());
        std::vector<std::vector<int64_t>> out_dims(config.outputs.size());
        for (size_t i = 0; i < config.outputs.size(); ++i) {
            const auto& s = config.outputs[i];
            out_dims[i] = s.dims;
            out_specs[i].name = s.name.c_str();
            out_specs[i].data_type = static_cast<int32_t>(s.data_type);
            out_specs[i].dims = out_dims[i].data();
            out_specs[i].rank = static_cast<int32_t>(s.dims.size());
            out_specs[i].data = nullptr;
            out_specs[i].byte_size = 0;
        }

        PluginNodeInfo info{};
        info.model_name = config.name.c_str();
        info.plugin_library = config.plugin_library.c_str();
        info.inputs = in_specs.empty() ? nullptr : in_specs.data();
        info.input_count = static_cast<int32_t>(in_specs.size());
        info.outputs = out_specs.empty() ? nullptr : out_specs.data();
        info.output_count = static_cast<int32_t>(out_specs.size());

        // Per-model parameters (Triton `parameters` block): each pipeline's
        // plugin model carries its own pre/post-processing configuration.
        param_keys_.clear();
        param_values_.clear();
        param_key_ptrs_.clear();
        param_value_ptrs_.clear();
        for (const auto& p : config.parameters) {
            param_keys_.push_back(p.key);
            param_values_.push_back(p.value);
        }
        param_key_ptrs_.reserve(param_keys_.size());
        param_value_ptrs_.reserve(param_values_.size());
        for (size_t i = 0; i < param_keys_.size(); ++i) {
            param_key_ptrs_.push_back(param_keys_[i].c_str());
            param_value_ptrs_.push_back(param_values_[i].c_str());
        }
        info.parameter_keys = param_key_ptrs_.empty() ? nullptr : param_key_ptrs_.data();
        info.parameter_values = param_value_ptrs_.empty() ? nullptr : param_value_ptrs_.data();
        info.parameter_count = static_cast<int32_t>(param_keys_.size());

        char errbuf[512] = {0};
        node_ = create(&info, errbuf, sizeof(errbuf));
        if (!node_) {
            throw std::runtime_error("plugin create failed for '" + config.name + "': " +
                                     std::string(errbuf[0] ? errbuf : "unknown error"));
        }
        output_specs_ = config.outputs;
    } catch (...) {
        unload();
        throw;
    }
}

void PluginBackend::execute(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
    if (!handle_ || !node_) {
        throw std::runtime_error("PluginBackend::execute called before load()");
    }
    auto exec = reinterpret_cast<inferlite_plugin_execute_fn>(
        resolveSymbol(handle_, "inferlite_plugin_execute"));

    std::vector<InferliteTensorDesc> in_tensors(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        in_tensors[i].name = inputs[i].name.c_str();
        in_tensors[i].data_type = static_cast<int32_t>(inputs[i].type);
        in_tensors[i].dims = inputs[i].shape.data();
        in_tensors[i].rank = static_cast<int32_t>(inputs[i].shape.size());
        in_tensors[i].data = inputs[i].data.data();
        in_tensors[i].byte_size = inputs[i].data.size();
    }

    // Outputs: build tensors sized per the declared output specs so the plugin
    // has writable host buffers to fill in place. Static shapes only in Phase 2.
    std::vector<InferliteTensorDesc> out_tensors(output_specs_.size());
    outputs.clear();
    outputs.reserve(output_specs_.size());
    std::vector<Tensor> built(output_specs_.size());
    for (size_t i = 0; i < output_specs_.size(); ++i) {
        const TensorSpec& spec = output_specs_[i];
        built[i].name = spec.name;
        built[i].type = spec.data_type;
        built[i].shape = spec.dims;
        built[i].data.resize(tensorByteSize(spec.dims, spec.data_type));
        out_tensors[i].name = built[i].name.c_str();
        out_tensors[i].data_type = static_cast<int32_t>(built[i].type);
        out_tensors[i].dims = built[i].shape.data();
        out_tensors[i].rank = static_cast<int32_t>(built[i].shape.size());
        out_tensors[i].data = built[i].data.data();
        out_tensors[i].byte_size = built[i].data.size();
    }

    char errbuf[512] = {0};
    int rc = exec(node_, in_tensors.data(), static_cast<int32_t>(in_tensors.size()),
                  out_tensors.data(), static_cast<int32_t>(out_tensors.size()),
                  errbuf, sizeof(errbuf));
    if (rc != 0) {
        throw std::runtime_error("plugin execute failed: " +
                                 std::string(errbuf[0] ? errbuf : "unknown error"));
    }
    outputs = std::move(built);
}

void PluginBackend::unload() {
    if (handle_ && node_) {
#if defined(_WIN32)
        auto destroy = reinterpret_cast<inferlite_plugin_destroy_fn>(
            ::GetProcAddress(reinterpret_cast<HMODULE>(handle_), "inferlite_plugin_destroy"));
#else
        auto destroy = reinterpret_cast<inferlite_plugin_destroy_fn>(
            ::dlsym(handle_, "inferlite_plugin_destroy"));
#endif
        if (destroy) destroy(node_);
        node_ = nullptr;
    }
    if (handle_) {
#if defined(_WIN32)
        ::FreeLibrary(reinterpret_cast<HMODULE>(handle_));
#else
        ::dlclose(handle_);
#endif
        handle_ = nullptr;
    }
}

}  // namespace inferlite
