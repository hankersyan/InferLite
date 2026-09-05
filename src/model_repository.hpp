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
    std::string config_hash;   // SHA-256 hex of the config.pbtxt text
};

// Thrown on repository scanning/validation errors (fail-fast).
struct RepositoryError : public std::runtime_error {
    explicit RepositoryError(const std::string& msg) : std::runtime_error(msg) {}
};

// Scan the repository and validate every model. Throws RepositoryError if any
// model config is malformed, uses an unsupported backend, enables batching, or
// requests a non-CPU instance group.
std::vector<LoadedModel> scanRepository(const std::string& root);

// --- Model-management helpers (Phase 6 / Triton model-control modes) -------

// Names of all candidate models under `root`: one per top-level subdirectory
// that contains a config.pbtxt. Per-model contents are NOT parsed or validated,
// so a broken model directory never aborts discovery. Returns an empty vector
// (not an error) when `root` does not exist or is not a directory.
std::vector<std::string> listRepositoryModelNames(const std::string& root);

// Load a single model from the repository: read <root>/<model_name>/config.pbtxt,
// require the config `name` to equal `model_name`, validate the config
// (backend/instance-group/artifacts), select the highest numeric version
// directory, read optional metadata.json, and set config.model_path.
// `override_text`, when non-empty, is parsed and validated in place of the
// config.pbtxt on disk (the version directory and metadata are still taken from
// disk). Throws RepositoryError when the model is missing or invalid.
LoadedModel loadModelConfig(const std::string& root, const std::string& model_name,
                            const std::string& override_text = std::string());

// Highest numeric version directory name for a model, as a decimal string.
// Returns "" when the model has no numeric version subdirectory.
std::string highestModelVersionString(const std::string& root, const std::string& model_name);

// Stable fingerprint of a model directory's contents (recursive). Poll-based
// model management compares fingerprints to detect config/artifact changes.
// The fingerprint is deterministic (sorted by relative path) and combines the
// relative path, file size, and last-write time of every regular file under
// <root>/<model_name>. Returns "" when the model directory is absent.
std::string fingerprintModelDirectory(const std::string& root, const std::string& model_name);

}  // namespace inferlite
