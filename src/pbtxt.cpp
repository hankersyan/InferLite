#include "pbtxt.hpp"

#include <cctype>
#include <cstdint>
#include <sstream>
#include <stdexcept>

#include "json.hpp"

namespace inferlite {

namespace {

// Simple tokenizer / parser for the proto-text subset:
//   name: "foo"
//   backend: "openvino"
//   max_batch_size: 0
//   input { name: "x" data_type: TYPE_FP32 dims: [ 1, 3, 224, 224 ] }
//   instance_group { count: 2 kind: KIND_CPU }
//
// Grammar handled:
//   scalar fields  -> ident ':' value
//   message fields -> ident '{' ... '}'
//   arrays         -> dims: [ 1, 2, -1 ]  (or single value dims: 4)
// Values are double-quoted strings, integer literals (may be negative), or
// enum identifiers (unquoted words). Comments (# ...) are skipped.

class Lexer {
public:
    explicit Lexer(const std::string& s) : s_(s) {}

    char peek() const {
        skipTrivia();
        if (pos_ < s_.size()) return s_[pos_];
        return '\0';
    }

    bool eof() const {
        skipTrivia();
        return pos_ >= s_.size();
    }

    char get() {
        skipTrivia();
        if (pos_ < s_.size()) return s_[pos_++];
        return '\0';
    }

    // Read the next token as a string. Token types: '{', '}', '[', ']', ':',
    // ',', quoted string, identifier, or integer literal.
    std::string next() {
        skipTrivia();
        if (pos_ >= s_.size()) return {};
        char c = s_[pos_];
        if (c == '{' || c == '}' || c == '[' || c == ']' || c == ':' || c == ',') {
            ++pos_;
            return std::string(1, c);
        }
        if (c == '"') {
            return readQuoted();
        }
        return readWord();
    }

private:
    void skipTrivia() const {
        while (pos_ < s_.size()) {
            char c = s_[pos_];
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
                ++pos_;
            } else if (c == '#') {
                while (pos_ < s_.size() && s_[pos_] != '\n') ++pos_;
            } else {
                break;
            }
        }
    }

    std::string readQuoted() {
        // assumes s_[pos_] == '"'
        ++pos_;
        std::string out;
        while (pos_ < s_.size()) {
            char c = s_[pos_++];
            if (c == '"') break;
            if (c == '\\' && pos_ < s_.size()) {
                char e = s_[pos_++];
                switch (e) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    default: out += e; break;
                }
            } else {
                out += c;
            }
        }
        return out;
    }

    std::string readWord() {
        std::string out;
        while (pos_ < s_.size()) {
            char c = s_[pos_];
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '#' ||
                c == '{' || c == '}' || c == '[' || c == ']' || c == ':' || c == ',') {
                break;
            }
            out += c;
            ++pos_;
        }
        return out;
    }

    const std::string& s_;
    mutable size_t pos_ = 0;
};

int64_t parseInteger(const std::string& tok) {
    try {
        size_t idx = 0;
        long long v = std::stoll(tok, &idx, 10);
        if (idx != tok.size()) throw std::invalid_argument("trailing");
        return static_cast<int64_t>(v);
    } catch (...) {
        throw PbtxtError("invalid integer literal: '" + tok + "'");
    }
}

std::vector<int64_t> parseDims(Lexer& lex) {
    std::vector<int64_t> dims;
    std::string tok = lex.next();
    if (tok == "[") {
        while (true) {
            std::string t = lex.next();
            if (t == "]") break;
            if (t == ",") continue;  // skip element separators
            dims.push_back(parseInteger(t));
        }
    } else {
        dims.push_back(parseInteger(tok));
    }
    return dims;
}

// "TYPE_FP32" -> "FP32"
std::string stripTypePrefix(const std::string& v) {
    const std::string prefix = "TYPE_";
    if (v.rfind(prefix, 0) == 0) return v.substr(prefix.size());
    return v;
}

// Upper-case a copy of a string.
std::string toUpper(const std::string& s) {
    std::string out = s;
    for (char& c : out) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return out;
}

// Read the next token and require it to equal `expected`.
void requireToken(Lexer& lex, const std::string& expected, const std::string& what) {
    std::string tok = lex.next();
    if (tok != expected) {
        throw PbtxtError("expected '" + expected + "' for " + what + ", got '" + tok + "'");
    }
}

// Parse a repeated scalar string list, e.g. `names: [ "a", "b" ]` or the
// single-name form `name: "a"`. Returns the list of tokens.
std::vector<std::string> parseStringList(Lexer& lex) {
    std::vector<std::string> out;
    std::string tok = lex.next();
    if (tok == "[") {
        while (true) {
            std::string t = lex.next();
            if (t == "]") break;
            if (t == ",") continue;
            out.push_back(t);
        }
    } else {
        out.push_back(tok);
    }
    return out;
}

// Parse a `self_test` golden-input tensor block:
//   self_test {
//     input {
//       name: "input"
//       data_type: TYPE_FP32
//       dims: [ 1, 4 ]
//       data: [ 1.0, 2.0, 3.0, 4.0 ]   # JSON array of numbers
//     }
//     expected_output { ... }           # optional
//     epsilon: 0.0001                   # optional
//   }
Tensor parseSelfTestTensor(Lexer& lex, const char* what) {
    requireToken(lex, "{", std::string(what) + " block");
    TensorSpec spec;
    std::vector<double> data_vals;
    bool has_data = false;
    while (true) {
        std::string f = lex.next();
        if (f == "}") break;
        if (f.empty()) throw PbtxtError("unexpected EOF in " + std::string(what));
        if (f == "name") {
            requireToken(lex, ":", "name");
            spec.name = lex.next();
        } else if (f == "data_type") {
            requireToken(lex, ":", "data_type");
            std::string v = stripTypePrefix(lex.next());
            spec.data_type = dataTypeFromString(v);
            if (spec.data_type == DataType::kInvalid) {
                throw PbtxtError("unsupported data_type in " + std::string(what) + ": " + v);
            }
        } else if (f == "dims") {
            requireToken(lex, ":", "dims");
            spec.dims = parseDims(lex);
        } else if (f == "data") {
            requireToken(lex, ":", "data");
            // data is a JSON array of numbers.
            std::string t = lex.next();
            if (t != "[") throw PbtxtError("expected '[' for data in " + std::string(what));
            while (true) {
                std::string n = lex.next();
                if (n == "]") break;
                if (n == ",") continue;
                data_vals.push_back(std::stod(n));
            }
            has_data = true;
        } else {
            std::string t = lex.next();
            if (t == "{") {
                int depth = 1;
                while (depth > 0) {
                    std::string inner = lex.next();
                    if (inner == "{") ++depth;
                    else if (inner == "}") --depth;
                    else if (inner.empty()) throw PbtxtError("unbalanced braces");
                }
            }
        }
    }
    if (spec.name.empty()) throw PbtxtError(std::string(what) + " missing 'name'");
    if (spec.data_type == DataType::kInvalid) {
        throw PbtxtError(std::string(what) + " missing valid 'data_type'");
    }
    Tensor t;
    t.name = spec.name;
    t.type = spec.data_type;
    t.shape = spec.dims;
    if (has_data) {
        size_t elem = dataTypeSize(spec.data_type);
        t.data.resize(data_vals.size() * elem);
        uint8_t* out = t.data.data();
        for (size_t i = 0; i < data_vals.size(); ++i) {
            writeTensorScalar(out + i * elem, spec.data_type, data_vals[i]);
        }
    }
    return t;
}

// Resolve a DeviceKind from an instance_group's `kind` string.
DeviceKind kindFromStringInternal(const std::string& s) {
    std::string v = toUpper(s);
    if (v == "CPU" || v == "KIND_CPU") return DeviceKind::kCpu;
    if (v == "NPU" || v == "KIND_NPU") return DeviceKind::kNpu;
    if (v == "GPUI" || v == "GPU_INTEL" || v == "KIND_GPU_INTEL") return DeviceKind::kGpuIntel;
    if (v == "AUTO" || v == "KIND_AUTO") return DeviceKind::kAuto;
    if (v == "CUDA" || v == "GPU" || v == "TENSORRT" || v == "KIND_GPU") return DeviceKind::kNvidiaGpu;
    return DeviceKind::kInvalid;
}

}  // namespace

DeviceKind deviceKindFromString(const std::string& s) {
    return kindFromStringInternal(s);
}

ModelConfig parseConfigPbtxt(const std::string& text) {
    ModelConfig cfg;
    Lexer lex(text);

    auto require = [&](const std::string& expected, const std::string& what) {
        std::string tok = lex.next();
        if (tok != expected) {
            throw PbtxtError("expected '" + expected + "' for " + what + ", got '" + tok + "'");
        }
    };

    while (!lex.eof()) {
        std::string field = lex.next();
        if (field.empty()) break;

        if (field == "name") {
            require(":", "name");
            cfg.name = lex.next();
        } else if (field == "backend") {
            require(":", "backend");
            cfg.backend = lex.next();
        } else if (field == "max_batch_size") {
            require(":", "max_batch_size");
            cfg.max_batch_size = parseInteger(lex.next());
        } else if (field == "input") {
            require("{", "input");
            TensorSpec spec;
            while (true) {
                std::string f = lex.next();
                if (f == "}") break;
                if (f.empty()) throw PbtxtError("unexpected EOF in input{}");
                if (f == "name") {
                    require(":", "input.name");
                    spec.name = lex.next();
                } else if (f == "data_type") {
                    require(":", "input.data_type");
                    std::string v = stripTypePrefix(lex.next());
                    spec.data_type = dataTypeFromString(v);
                    if (spec.data_type == DataType::kInvalid) {
                        throw PbtxtError("unsupported data_type in input: " + v);
                    }
                } else if (f == "dims") {
                    require(":", "input.dims");
                    spec.dims = parseDims(lex);
                } else {
                    // Skip unknown scalar/message fields inside input.
                    std::string t = lex.next();
                    if (t == "{") {
                        int depth = 1;
                        while (depth > 0) {
                            std::string inner = lex.next();
                            if (inner == "{") ++depth;
                            else if (inner == "}") --depth;
                            else if (inner.empty()) throw PbtxtError("unbalanced braces");
                        }
                    }
                }
            }
            cfg.inputs.push_back(std::move(spec));
        } else if (field == "output") {
            require("{", "output");
            TensorSpec spec;
            while (true) {
                std::string f = lex.next();
                if (f == "}") break;
                if (f.empty()) throw PbtxtError("unexpected EOF in output{}");
                if (f == "name") {
                    require(":", "output.name");
                    spec.name = lex.next();
                } else if (f == "data_type") {
                    require(":", "output.data_type");
                    std::string v = stripTypePrefix(lex.next());
                    spec.data_type = dataTypeFromString(v);
                    if (spec.data_type == DataType::kInvalid) {
                        throw PbtxtError("unsupported data_type in output: " + v);
                    }
                } else if (f == "dims") {
                    require(":", "output.dims");
                    spec.dims = parseDims(lex);
                } else {
                    std::string t = lex.next();
                    if (t == "{") {
                        int depth = 1;
                        while (depth > 0) {
                            std::string inner = lex.next();
                            if (inner == "{") ++depth;
                            else if (inner == "}") --depth;
                            else if (inner.empty()) throw PbtxtError("unbalanced braces");
                        }
                    }
                }
            }
            cfg.outputs.push_back(std::move(spec));
        } else if (field == "instance_group") {
            require("{", "instance_group");
            while (true) {
                std::string f = lex.next();
                if (f == "}") break;
                if (f.empty()) throw PbtxtError("unexpected EOF in instance_group{}");
                if (f == "count") {
                    require(":", "instance_group.count");
                    cfg.instance_group.count = static_cast<int>(parseInteger(lex.next()));
                } else if (f == "kind") {
                    require(":", "instance_group.kind");
                    cfg.instance_group.kind = lex.next();
                } else {
                    std::string t = lex.next();
                    if (t == "{") {
                        int depth = 1;
                        while (depth > 0) {
                            std::string inner = lex.next();
                            if (inner == "{") ++depth;
                            else if (inner == "}") --depth;
                            else if (inner.empty()) throw PbtxtError("unbalanced braces");
                        }
                    }
                }
            }
        } else if (field == "plugin_library") {
            require(":", "plugin_library");
            cfg.plugin_library = lex.next();
        } else if (field == "ensemble_scheduling") {
            require("{", "ensemble_scheduling");
            while (true) {
                std::string f = lex.next();
                if (f == "}") break;
                if (f.empty()) throw PbtxtError("unexpected EOF in ensemble_scheduling{}");
                if (f == "step") {
                    require("{", "step");
                    EnsembleStep st;
                    while (true) {
                        std::string sf = lex.next();
                        if (sf == "}") break;
                        if (sf.empty()) throw PbtxtError("unexpected EOF in step{}");
                        if (sf == "model_name") {
                            require(":", "step.model_name");
                            st.model_name = lex.next();
                        } else if (sf == "input_map") {
                            require("{", "step.input_map");
                            while (true) {
                                std::string k = lex.next();
                                if (k == "}") break;
                                require(":", "input_map key");
                                st.input_map_to.push_back(k);
                                st.input_map_from.push_back(lex.next());
                            }
                        } else if (sf == "output_map") {
                            require("{", "step.output_map");
                            while (true) {
                                std::string k = lex.next();
                                if (k == "}") break;
                                require(":", "output_map key");
                                // key = step output name (from), value = parent
                                // scope name (to).
                                st.output_map_from.push_back(k);
                                st.output_map_to.push_back(lex.next());
                            }
                        } else {
                            std::string t = lex.next();
                            if (t == "{") {
                                int depth = 1;
                                while (depth > 0) {
                                    std::string inner = lex.next();
                                    if (inner == "{") ++depth;
                                    else if (inner == "}") --depth;
                                    else if (inner.empty()) throw PbtxtError("unbalanced braces");
                                }
                            }
                        }
                    }
                    if (st.model_name.empty()) {
                        throw PbtxtError("ensemble step missing 'model_name'");
                    }
                    cfg.ensemble_steps.push_back(std::move(st));
                } else {
                    std::string t = lex.next();
                    if (t == "{") {
                        int depth = 1;
                        while (depth > 0) {
                            std::string inner = lex.next();
                            if (inner == "{") ++depth;
                            else if (inner == "}") --depth;
                            else if (inner.empty()) throw PbtxtError("unbalanced braces");
                        }
                    }
                }
            }
        } else if (field == "metadata") {
            require("{", "metadata");
            while (true) {
                std::string f = lex.next();
                if (f == "}") break;
                if (f.empty()) throw PbtxtError("unexpected EOF in metadata{}");
                if (f == "model_id") {
                    require(":", "metadata.model_id");
                    cfg.metadata.model_id = lex.next();
                } else if (f == "version") {
                    require(":", "metadata.version");
                    cfg.metadata.version = lex.next();
                } else if (f == "intended_use") {
                    require(":", "metadata.intended_use");
                    cfg.metadata.intended_use = lex.next();
                } else if (f == "training_dataset_id") {
                    require(":", "metadata.training_dataset_id");
                    cfg.metadata.training_dataset_id = lex.next();
                } else if (f == "approval_status") {
                    require(":", "metadata.approval_status");
                    cfg.metadata.approval_status = lex.next();
                } else {
                    std::string t = lex.next();
                    if (t == "{") {
                        int depth = 1;
                        while (depth > 0) {
                            std::string inner = lex.next();
                            if (inner == "{") ++depth;
                            else if (inner == "}") --depth;
                            else if (inner.empty()) throw PbtxtError("unbalanced braces");
                        }
                    }
                }
            }
        } else if (field == "output_validation" || field == "validate") {
            require("{", "output_validation");
            while (true) {
                std::string f = lex.next();
                if (f == "}") break;
                if (f.empty()) throw PbtxtError("unexpected EOF in output_validation{}");
                if (f == "detect_nan_inf") {
                    require(":", "detect_nan_inf");
                    cfg.output_validation.detect_nan_inf = parseInteger(lex.next()) != 0;
                } else if (f == "check_range") {
                    require(":", "check_range");
                    cfg.output_validation.check_range = parseInteger(lex.next()) != 0;
                } else if (f == "min_value") {
                    require(":", "min_value");
                    cfg.output_validation.min_value = std::stod(lex.next());
                } else if (f == "max_value") {
                    require(":", "max_value");
                    cfg.output_validation.max_value = std::stod(lex.next());
                } else if (f == "check_shape") {
                    require(":", "check_shape");
                    cfg.output_validation.check_shape = parseInteger(lex.next()) != 0;
                } else if (f == "confidence_threshold") {
                    require(":", "confidence_threshold");
                    cfg.output_validation.confidence_threshold = parseInteger(lex.next()) != 0;
                } else if (f == "min_confidence") {
                    require(":", "min_confidence");
                    cfg.output_validation.min_confidence = std::stod(lex.next());
                } else {
                    std::string t = lex.next();
                    if (t == "{") {
                        int depth = 1;
                        while (depth > 0) {
                            std::string inner = lex.next();
                            if (inner == "{") ++depth;
                            else if (inner == "}") --depth;
                            else if (inner.empty()) throw PbtxtError("unbalanced braces");
                        }
                    }
                }
            }
        } else if (field == "self_test") {
            require("{", "self_test");
            while (true) {
                std::string f = lex.next();
                if (f == "}") break;
                if (f.empty()) throw PbtxtError("unexpected EOF in self_test{}");
                if (f == "input") {
                    cfg.self_test.enabled = true;
                    cfg.self_test.input.push_back(parseSelfTestTensor(lex, "self_test.input"));
                } else if (f == "expected_output") {
                    cfg.self_test.expected_output.push_back(
                        parseSelfTestTensor(lex, "self_test.expected_output"));
                } else if (f == "epsilon") {
                    require(":", "epsilon");
                    cfg.self_test.epsilon = std::stod(lex.next());
                } else {
                    std::string t = lex.next();
                    if (t == "{") {
                        int depth = 1;
                        while (depth > 0) {
                            std::string inner = lex.next();
                            if (inner == "{") ++depth;
                            else if (inner == "}") --depth;
                            else if (inner.empty()) throw PbtxtError("unbalanced braces");
                        }
                    }
                }
            }
        } else if (field == "max_inference_time_ms") {
            require(":", "max_inference_time_ms");
            cfg.max_inference_time_ms = parseInteger(lex.next());
        } else if (field == "max_input_size_bytes") {
            require(":", "max_input_size_bytes");
            cfg.max_input_size_bytes = static_cast<size_t>(parseInteger(lex.next()));
        } else if (field == "max_output_size_bytes") {
            require(":", "max_output_size_bytes");
            cfg.max_output_size_bytes = static_cast<size_t>(parseInteger(lex.next()));
        } else {
            // Unknown top-level field: skip scalar or message.
            std::string t = lex.next();
            if (t == "{") {
                int depth = 1;
                while (depth > 0) {
                    std::string inner = lex.next();
                    if (inner == "{") ++depth;
                    else if (inner == "}") --depth;
                    else if (inner.empty()) throw PbtxtError("unbalanced braces");
                }
            }
        }
    }

    if (cfg.name.empty()) throw PbtxtError("config missing required field 'name'");

    // Phase 4: resolve the effective device kind from `kind`
    // (KIND_CPU/KIND_NPU/KIND_GPU_INTEL/KIND_AUTO), Triton-style. Unknown values
    // are left as kInvalid so validation can fail fast with a precise message.
    cfg.instance_group.device_kind = deviceKindFromString(cfg.instance_group.kind);
    return cfg;
}

ModelMetadata parseMetadataJson(const std::string& text) {
    json::Value doc = json::parse(text);
    ModelMetadata md;
    auto get = [&](const char* key, std::string& out) {
        const json::Value* v = doc.find(key);
        if (v && v->isString()) out = v->asString();
    };
    get("model_id", md.model_id);
    get("version", md.version);
    get("intended_use", md.intended_use);
    get("training_dataset_id", md.training_dataset_id);
    get("approval_status", md.approval_status);
    return md;
}

}  // namespace inferlite
