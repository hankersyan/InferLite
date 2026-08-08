// model_repository.hpp - Scans the model repository directory and parses
// config.pbtxt files into ModelConfig objects.
#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "pbtxt.hpp"

namespace inferlite {

// Locate and load all valid model configs under a repository root.
//
// Layout expected:
//   <root>/<model_name>/<version>/model.xml
//   <root>/<model_name>/config.pbtxt
//
// For each model, the highest numeric version subdirectory is selected; the
// path to that version folder is returned so the backend can load model.xml.
struct LoadedModel {
    std::shared_ptr<ModelConfig> config;
    std::string version_path;  // absolute path to the selected version folder
    int64_t version = -1;
};

// Thrown on repository scanning/validation errors (fail-fast).
struct RepositoryError : public std::runtime_error {
    explicit RepositoryError(const std::string& msg) : std::runtime_error(msg) {}
};

// Scan the repository and validate every model. Throws RepositoryError if any
// model config is malformed, uses an unsupported backend, enables batching, or
// requests a non-CPU instance group.
std::vector<LoadedModel> scanRepository(const std::string& root);

}  // namespace inferlite
