// ensemble_executor.hpp - Device-aware ensemble DAG executor (CPU + GPU).
//
// Builds a static directed acyclic graph from the ensemble's steps at startup
// and executes it per request. Steps may run on CPU (OpenVINO, plugins) or GPU
// (TensorRT). Intermediate tensors are passed between steps; a TensorRT step
// transitions host<->device internally (a pinned-memory copy), so cross-device
// edges are explicit and correctness (identical outputs to a sequential
// pipeline with explicit copies) is preserved. A failure in any step cancels the
// whole ensemble with a structured error code and releases all resources.
#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "backend.hpp"
#include "pbtxt.hpp"
#include "tensor.hpp"

namespace inferlite {

// Registry used to resolve an ensemble step's referenced model/plugin/ensemble
// to its executable backend.
using BackendProvider = std::function<BackendPtr(const std::string& model_name)>;

class EnsembleExecutor : public IBackend {
public:
    // `config` is the ensemble's ModelConfig (backend == "ensemble").
    // `provider` resolves step model_name -> backend. Throws std::runtime_error
    // if the DAG is invalid (cycle, missing step, unknown reference).
    EnsembleExecutor(std::shared_ptr<const ModelConfig> config, BackendProvider provider);
    ~EnsembleExecutor() override = default;

    BackendResult execute(const std::vector<Tensor>& inputs) override;

private:
    struct Node {
        EnsembleStep step;
        BackendPtr backend;
        std::vector<size_t> deps;  // indices of nodes that must run first
    };

    void buildDag(const BackendProvider& provider);

    std::shared_ptr<const ModelConfig> config_;
    std::vector<Node> nodes_;
    // name -> node index
    std::map<std::string, size_t> node_index_;
};

}  // namespace inferlite
