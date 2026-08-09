#include "ensemble_executor.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>

#include "validation.hpp"

namespace inferlite {

namespace {

// Find a tensor by name; return its index or -1.
int64_t findTensor(const std::vector<Tensor>& tensors, const std::string& name) {
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i].name == name) return static_cast<int64_t>(i);
    }
    return -1;
}

// Move/copy a named tensor into a vector (copy semantics so each step owns the
// bytes it needs; the source stays valid for later steps / output collection).
void upsertTensor(std::vector<Tensor>& out, const Tensor& t) {
    int64_t idx = findTensor(out, t.name);
    if (idx >= 0) {
        out[static_cast<size_t>(idx)] = t;
    } else {
        out.push_back(t);
    }
}

}  // namespace

EnsembleExecutor::EnsembleExecutor(std::shared_ptr<const ModelConfig> config,
                                   BackendProvider provider)
    : config_(std::move(config)) {
    buildDag(provider);
}

void EnsembleExecutor::buildDag(const BackendProvider& provider) {
    if (config_->ensemble_steps.empty()) {
        throw std::runtime_error("ensemble '" + config_->name +
                                 "' has no steps (empty ensemble_scheduling)");
    }

    // Resolve each step's backend and record declared dependencies: a step
    // depends on any earlier step that produces one of its inputs.
    for (size_t i = 0; i < config_->ensemble_steps.size(); ++i) {
        const EnsembleStep& st = config_->ensemble_steps[i];
        BackendPtr backend = provider(st.model_name);
        if (!backend) {
            throw std::runtime_error("ensemble '" + config_->name + "' step references unknown " +
                                     "model '" + st.model_name + "'");
        }
        Node n;
        n.step = st;
        n.backend = std::move(backend);
        nodes_.push_back(std::move(n));
        node_index_[st.model_name] = i;
    }

    // Compute dependencies: for step i, find all earlier steps j whose output
    // maps to a tensor that step i consumes via its input_map.
    for (size_t i = 0; i < nodes_.size(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            // If any input_to of step i equals any output_to of step j, then j
            // must precede i.
            for (const auto& in_to : nodes_[i].step.input_map_to) {
                for (const auto& out_to : nodes_[j].step.output_map_to) {
                    if (in_to == out_to) {
                        nodes_[i].deps.push_back(j);
                    }
                }
            }
        }
        // De-duplicate deps.
        std::sort(nodes_[i].deps.begin(), nodes_[i].deps.end());
        nodes_[i].deps.erase(std::unique(nodes_[i].deps.begin(), nodes_[i].deps.end()),
                             nodes_[i].deps.end());
    }
}

BackendResult EnsembleExecutor::execute(const std::vector<Tensor>& inputs) {
    BackendResult result;

    // workspace holds the current named tensors produced so far, starting with
    // the ensemble's request inputs.
    std::vector<Tensor> workspace = inputs;

    // The ensemble's declared inputs must be satisfied by the request inputs.
    for (const auto& spec : config_->inputs) {
        if (findTensor(workspace, spec.name) < 0) {
            result.ok = false;
            result.error_code = ErrorCode::kInvalidInput;
            result.error = "ensemble '" + config_->name + "' is missing input '" +
                           spec.name + "'";
            return result;
        }
    }

    // Topological execution. Because deps are strictly from earlier to later
    // steps (j < i), the declared order is already a valid topological order.
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const Node& node = nodes_[i];

        // Gather this step's inputs from the workspace using its input_map.
        std::vector<Tensor> step_inputs;
        for (size_t k = 0; k < node.step.input_map_to.size(); ++k) {
            const std::string& from = node.step.input_map_from[k];
            int64_t idx = findTensor(workspace, from);
            if (idx < 0) {
                result.ok = false;
                result.error_code = ErrorCode::kInternalError;
                result.error = "ensemble '" + config_->name + "' step '" +
                               node.step.model_name + "' input '" + from +
                               "' was not produced";
                return result;
            }
            // Rename to the step's parameter name.
            Tensor t = workspace[static_cast<size_t>(idx)];
            t.name = node.step.input_map_to[k];
            step_inputs.push_back(std::move(t));
        }

        // Execute the step (backend internally validates and contains faults).
        BackendResult step_result = node.backend->execute(step_inputs);
        if (!step_result.ok) {
            result.ok = false;
            result.error_code = step_result.error_code;
            result.error = "ensemble '" + config_->name + "' step '" +
                           node.step.model_name + "' failed: " + step_result.error;
            return result;
        }

        // Map the step's outputs back into the workspace scope, preserving each
        // tensor's device placement so downstream steps know where it lives and
        // the executor can make correct cross-device copy decisions.
        for (const auto& out : step_result.outputs) {
            // Find the output_map entry that names this output.
            const std::string* target = nullptr;
            for (size_t k = 0; k < node.step.output_map_from.size(); ++k) {
                if (node.step.output_map_from[k] == out.name) {
                    target = &node.step.output_map_to[k];
                    break;
                }
            }
            if (target) {
                Tensor t = out;
                t.name = *target;
                upsertTensor(workspace, t);
            }
            // Outputs not in the map are ignored for the workspace but may be
            // exposed as the ensemble's final outputs if declared.
        }
    }

    // Phase 3: materialize any remaining device (GPU) tensors into host memory
    // before exposing them as ensemble outputs. Each GPU backend already returns
    // host-resident output tensors (it performs the device->host pinned copy),
    // so this is a defensive guarantee that ensemble outputs are always readable
    // host data for the HTTP layer and output validation.
    for (auto& t : workspace) {
        if (t.device == DeviceKind::kGpu) {
            // In this host-boundary design the backend returned host data; clear
            // the device marker so consumers treat it as host memory.
            t.device = DeviceKind::kCpu;
            t.device_ptr = nullptr;
            t.device_bytes = 0;
        }
    }

    // Build the ensemble outputs from the declared output specs.
    for (const auto& spec : config_->outputs) {
        int64_t idx = findTensor(workspace, spec.name);
        if (idx < 0) {
            result.ok = false;
            result.error_code = ErrorCode::kOutputValidationFailed;
            result.error = "ensemble '" + config_->name + "' did not produce declared output '" +
                           spec.name + "'";
            return result;
        }
        result.outputs.push_back(workspace[static_cast<size_t>(idx)]);
    }

    result.ok = true;
    result.error_code = ErrorCode::kNone;
    return result;
}

}  // namespace inferlite
