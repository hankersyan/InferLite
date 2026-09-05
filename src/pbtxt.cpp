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

// Parse a list of numbers / bools: `[ 0, 1 ]` or a single token `1`.
std::vector<double> parseNumberArray(Lexer& lex) {
    std::vector<double> out;
    std::string tok = lex.next();
    if (tok == "[") {
        while (true) {
            std::string t = lex.next();
            if (t == "]") break;
            if (t == ",") continue;
            if (t == "true" || t == "True") {
                out.push_back(1.0);
            } else if (t == "false" || t == "False") {
                out.push_back(0.0);
            } else {
                out.push_back(std::stod(t));
            }
        }
    } else {
        if (tok == "true" || tok == "True") {
            out.push_back(1.0);
        } else if (tok == "false" || tok == "False") {
            out.push_back(0.0);
        } else {
            out.push_back(std::stod(tok));
        }
    }
    return out;
}

// Defined below (after stripTypePrefix) in this anonymous namespace.
std::string toUpper(const std::string& s);

// Map a CONTROL_SEQUENCE_* kind string to the enum.
SequenceControlKind sequenceKindFromString(const std::string& s) {
    std::string v = toUpper(s);
    if (v == "START" || v == "CONTROL_SEQUENCE_START") return SequenceControlKind::kSequenceStart;
    if (v == "END" || v == "CONTROL_SEQUENCE_END") return SequenceControlKind::kSequenceEnd;
    if (v == "READY" || v == "CONTROL_SEQUENCE_READY") return SequenceControlKind::kSequenceReady;
    if (v == "CORRID" || v == "CONTROL_SEQUENCE_CORRID" ||
        v == "CORRELATIONID" || v == "CONTROL_SEQUENCE_CORRELATIONID") {
        return SequenceControlKind::kSequenceCorrId;
    }
    return SequenceControlKind::kInvalid;
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

// Parse the body of a single tensor-spec message used by the `input`/`output`
// fields, e.g.:
//   { name: "x" data_type: TYPE_FP32 dims: [ 1, 4 ] }
// The caller must have already consumed the leading '{'; this reads fields
// until the matching '}'. `what` is used only in error messages.
TensorSpec parseTensorSpecBody(Lexer& lex, const char* what) {
    TensorSpec spec;
    while (true) {
        std::string f = lex.next();
        if (f == "}") break;
        if (f.empty()) throw PbtxtError("unexpected EOF in " + std::string(what) + " block");
        if (f == "name") {
            requireToken(lex, ":", "input.name");
            spec.name = lex.next();
        } else if (f == "data_type") {
            requireToken(lex, ":", "input.data_type");
            std::string v = stripTypePrefix(lex.next());
            spec.data_type = dataTypeFromString(v);
            if (spec.data_type == DataType::kInvalid) {
                throw PbtxtError("unsupported data_type in " + std::string(what) + ": " + v);
            }
        } else if (f == "dims") {
            requireToken(lex, ":", "input.dims");
            spec.dims = parseDims(lex);
        } else {
            // Skip unknown scalar/message fields inside the block.
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
    return spec;
}

// Parse the `input`/`output` field. Proto-text forms accepted:
//   repeated-message form (already consumed 'input'):
//     input { name: "x" ... }     -> caller passed 'tok' == "{"
//   array-of-message form (Triton unified IO schema):
//     input: [ { name: "x" ... }, { name: "y" ... } ]
//       -> caller passed 'tok' == ":"
//     input  [ { name: "x" ... }, { name: "y" ... } ]
//       -> caller passed 'tok' == "["  (colon optional, per proto text format)
std::vector<TensorSpec> parseIOField(Lexer& lex, const char* what, const std::string& tok) {
    std::vector<TensorSpec> specs;
    if (tok == ":" || tok == "[") {
        // Array-of-message form: input: [ {...}, {...} ]  or  input [ {...}, {...} ]
        std::string t = tok == "[" ? tok : lex.next();
        if (t != "[") {
            throw PbtxtError("expected '[' for " + std::string(what) + " array, got '" + t + "'");
        }
        while (true) {
            std::string n = lex.next();
            if (n == "]") break;
            if (n == ",") continue;
            if (n != "{") {
                throw PbtxtError("expected '{' in " + std::string(what) + " array, got '" + n + "'");
            }
            specs.push_back(parseTensorSpecBody(lex, what));
        }
    } else {
        // Repeated-message form: input { ... }
        if (tok != "{") {
            throw PbtxtError("expected '{' for " + std::string(what) + ", got '" + tok + "'");
        }
        specs.push_back(parseTensorSpecBody(lex, what));
    }
    return specs;
}

// ---- Triton model_warmup parsing (config `model_warmup { ... }`) ----

// Parse one ModelWarmup.Input value-message body (leading '{' consumed):
//   { data_type: TYPE_FP32 dims: [ 4 ] zero_data: true }
void parseWarmupInputBody(Lexer& lex, WarmupInput& out) {
    while (true) {
        std::string f = lex.next();
        if (f == "}") break;
        if (f.empty()) throw PbtxtError("unexpected EOF in model_warmup input block");
        if (f == "data_type") {
            requireToken(lex, ":", "warmup input data_type");
            std::string v = stripTypePrefix(lex.next());
            out.data_type = dataTypeFromString(v);
            if (out.data_type == DataType::kInvalid) {
                throw PbtxtError("unsupported data_type in model_warmup input: " + v);
            }
            out.has_type = true;
        } else if (f == "dims") {
            requireToken(lex, ":", "warmup input dims");
            out.dims = parseDims(lex);
            out.has_dims = true;
        } else if (f == "shape") {
            requireToken(lex, ":", "warmup input shape");
            out.shape = parseDims(lex);
            out.has_shape = true;
        } else if (f == "zero_data") {
            requireToken(lex, ":", "warmup input zero_data");
            std::string bt = lex.next();
            if (bt == "true") {
                out.zero_data = true;
            } else if (bt == "false") {
                out.zero_data = false;
            } else {
                out.zero_data = parseInteger(bt) != 0;
            }
        } else {
            // Skip unknown scalar/message fields (input_data_file is parsed but
            // rejected later in validateConfig with a precise error).
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
}

// Parse the proto-text map form `inputs { key: "X" value { ... } }` (leading
// '{' consumed). A `value` requires a preceding `key`, whose name is attached
// to the parsed input.
void parseWarmupInputsMap(Lexer& lex, ModelWarmup& wu) {
    while (true) {
        std::string f = lex.next();
        if (f == "}") break;
        if (f.empty()) throw PbtxtError("unexpected EOF in model_warmup.inputs map");
        if (f == "key") {
            requireToken(lex, ":", "model_warmup.inputs key");
            if (wu.inputs.empty() || !wu.inputs.back().name.empty()) {
                // A fresh key/entry unless the previous entry never received a value.
                wu.inputs.emplace_back();
            }
            wu.inputs.back().name = lex.next();
        } else if (f == "value") {
            std::string t = lex.next();  // optional ':' before '{'
            if (t == ":") t = lex.next();
            if (t != "{") throw PbtxtError("expected '{' for model_warmup input value");
            if (wu.inputs.empty() || wu.inputs.back().name.empty()) {
                throw PbtxtError("model_warmup.inputs 'value' without a preceding 'key'");
            }
            parseWarmupInputBody(lex, wu.inputs.back());
        } else {
            // Skip unknown scalar/message fields.
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
}

// Parse one warmup-request message (leading '{' consumed):
//   { name: "warmup" batch_size: 1 inputs { key: "INPUT" value { ... } } }
ModelWarmup parseModelWarmupBody(Lexer& lex) {
    ModelWarmup wu;
    while (true) {
        std::string f = lex.next();
        if (f == "}") break;
        if (f.empty()) throw PbtxtError("unexpected EOF in model_warmup block");
        if (f == "name") {
            requireToken(lex, ":", "model_warmup.name");
            wu.name = lex.next();
        } else if (f == "batch_size") {
            requireToken(lex, ":", "model_warmup.batch_size");
            wu.batch_size = parseInteger(lex.next());
        } else if (f == "inputs") {
            std::string t = lex.next();
            if (t == ":") t = lex.next();
            if (t != "{") throw PbtxtError("expected '{' for model_warmup.inputs");
            parseWarmupInputsMap(lex, wu);
        } else {
            // Skip unknown scalar/message fields (Triton also defines `count`,
            // which InferLite does not implement: each warmup request runs once).
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
    return wu;
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

// Map a control value-field name (int32_false_true / fp32_false_true / ...) to
// the data type it implies. Returns kInvalid for unknown field names.
DataType controlValueFieldType(const std::string& f) {
    std::string v = toUpper(f);
    if (v.find("INT32") != std::string::npos) return DataType::kInt32;
    if (v.find("INT64") != std::string::npos) return DataType::kInt64;
    if (v.find("FP32") != std::string::npos || v.find("FLOAT") != std::string::npos) {
        return DataType::kFloat32;
    }
    if (v.find("FP64") != std::string::npos || v.find("DOUBLE") != std::string::npos) {
        return DataType::kFloat64;
    }
    if (v.find("BOOL") != std::string::npos) return DataType::kBool;
    if (v.find("UINT32") != std::string::npos) return DataType::kUint32;
    if (v.find("UINT64") != std::string::npos) return DataType::kUint64;
    return DataType::kInvalid;
}

// Parse one `control { ... }` entry (the caller consumed the leading '{').
SequenceControlSpec parseControlBody(Lexer& lex) {
    SequenceControlSpec c;
    while (true) {
        std::string f = lex.next();
        if (f == "}") break;
        if (f.empty()) throw PbtxtError("unexpected EOF in control{}");
        if (f == "kind") {
            requireToken(lex, ":", "control.kind");
            c.kind = sequenceKindFromString(lex.next());
            if (c.kind == SequenceControlKind::kInvalid) {
                throw PbtxtError("unknown sequence control kind");
            }
        } else if (controlValueFieldType(f) != DataType::kInvalid) {
            // int32_false_true / fp32_false_true / bool_false_true: [false, true].
            requireToken(lex, ":", "control values");
            auto vals = parseNumberArray(lex);
            c.data_type = controlValueFieldType(f);
            if (!vals.empty()) c.false_value = vals[0];
            if (vals.size() > 1) c.true_value = vals[1];
            else if (vals.size() == 1) c.true_value = vals[0];
        } else {
            // Skip unknown scalar/message fields.
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
    if (c.kind == SequenceControlKind::kInvalid) {
        throw PbtxtError("control is missing a valid 'kind'");
    }
    return c;
}

// Parse one `control_input { ... }` entry (the caller consumed the leading '{').
SequenceControlInputSpec parseControlInputBody(Lexer& lex) {
    SequenceControlInputSpec in;
    while (true) {
        std::string f = lex.next();
        if (f == "}") break;
        if (f.empty()) throw PbtxtError("unexpected EOF in control_input{}");
        if (f == "name") {
            requireToken(lex, ":", "control_input.name");
            in.name = lex.next();
        } else if (f == "control") {
            std::string t = lex.next();
            if (t != "{") {
                // Also accept `control: { ... }`.
                if (t == ":") t = lex.next();
            }
            if (t != "{") {
                throw PbtxtError("expected '{' for control_input.control, got '" + t + "'");
            }
            in.controls.push_back(parseControlBody(lex));
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
    if (in.name.empty()) throw PbtxtError("control_input missing 'name'");
    return in;
}

// Parse one `state { ... }` entry (the caller consumed the leading '{').
SequenceStateSpec parseSequenceStateBody(Lexer& lex) {
    SequenceStateSpec st;
    while (true) {
        std::string f = lex.next();
        if (f == "}") break;
        if (f.empty()) throw PbtxtError("unexpected EOF in sequence state{}");
        if (f == "input_name") {
            requireToken(lex, ":", "state.input_name");
            st.input_name = lex.next();
        } else if (f == "output_name") {
            requireToken(lex, ":", "state.output_name");
            st.output_name = lex.next();
        } else if (f == "data_type") {
            requireToken(lex, ":", "state.data_type");
            std::string v = stripTypePrefix(lex.next());
            st.data_type = dataTypeFromString(v);
            if (st.data_type == DataType::kInvalid) {
                throw PbtxtError("unsupported state data_type: " + v);
            }
        } else if (f == "dims") {
            requireToken(lex, ":", "state.dims");
            st.dims = parseDims(lex);
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
    if (st.input_name.empty() || st.output_name.empty()) {
        throw PbtxtError("sequence state requires input_name and output_name");
    }
    if (st.data_type == DataType::kInvalid) {
        throw PbtxtError("sequence state requires a valid data_type");
    }
    if (st.dims.empty()) {
        throw PbtxtError("sequence state requires dims");
    }
    return st;
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
        } else if (field == "backend" || field == "platform") {
            // Triton accepts both `backend` and the legacy `platform` field.
            require(":", field);
            cfg.backend = lex.next();
        } else if (field == "max_batch_size") {
            require(":", "max_batch_size");
            cfg.max_batch_size = parseInteger(lex.next());
        } else if (field == "input") {
            // Accept both `input { ... }` and `input: [ { ... }, ... ]`.
            std::string tok = lex.next();
            auto specs = parseIOField(lex, "input", tok);
            cfg.inputs.insert(cfg.inputs.end(), specs.begin(), specs.end());
        } else if (field == "output") {
            // Accept both `output { ... }` and `output: [ { ... }, ... ]`.
            std::string tok = lex.next();
            auto specs = parseIOField(lex, "output", tok);
            cfg.outputs.insert(cfg.outputs.end(), specs.begin(), specs.end());
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
        } else if (field == "parameters") {
            require("{", "parameters");
            while (true) {
                std::string f = lex.next();
                if (f == "}") break;
                if (f.empty()) throw PbtxtError("unexpected EOF in parameters{}");
                if (f == "key") {
                    require(":", "parameters.key");
                    PluginParameter p;
                    p.key = lex.next();
                    // Triton-style value wrapper:
                    //   value { string_value: "..." | int64_value: N | bool_value: B }
                    std::string t = lex.next();
                    if (t == "value") t = lex.next();
                    if (t != "{") {
                        throw PbtxtError("expected 'value { ... }' for parameters key '" +
                                         p.key + "'");
                    }
                    int depth = 1;
                    while (depth > 0) {
                        std::string inner = lex.next();
                        if (inner == "{") { ++depth; continue; }
                        if (inner == "}") { --depth; continue; }
                        if (inner.empty()) throw PbtxtError("unbalanced braces in parameters.value");
                        if (inner == "string_value" || inner == "int64_value" ||
                            inner == "bool_value" || inner == "uint64_value" ||
                            inner == "double_value") {
                            require(":", "parameters.value." + inner);
                            p.value = lex.next();
                        } else {
                            // Unknown scalar inside value{}: consume ':' + value.
                            require(":", "parameters.value field");
                            lex.next();
                        }
                    }
                    cfg.parameters.push_back(std::move(p));
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
        } else if (field == "dynamic_batching") {
            require("{", "dynamic_batching");
            cfg.batching.enabled = true;
            while (true) {
                std::string f = lex.next();
                if (f == "}") break;
                if (f.empty()) throw PbtxtError("unexpected EOF in dynamic_batching{}");
                if (f == "preferred_batch_size") {
                    // Repeated field; accept both `[ 4, 8 ]` and repeated scalar
                    // occurrences `preferred_batch_size: 4`.
                    require(":", "dynamic_batching.preferred_batch_size");
                    auto vals = parseDims(lex);
                    cfg.batching.preferred_batch_size.insert(
                        cfg.batching.preferred_batch_size.end(), vals.begin(), vals.end());
                } else if (f == "max_queue_delay_microseconds") {
                    require(":", "dynamic_batching.max_queue_delay_microseconds");
                    cfg.batching.max_queue_delay_us = parseInteger(lex.next());
                } else if (f == "priority_levels") {
                    require(":", "dynamic_batching.priority_levels");
                    cfg.batching.priority_levels = parseInteger(lex.next());
                } else if (f == "default_priority_level") {
                    require(":", "dynamic_batching.default_priority_level");
                    cfg.batching.default_priority_level = parseInteger(lex.next());
                } else if (f == "preserve_ordering") {
                    require(":", "dynamic_batching.preserve_ordering");
                    // proto-text booleans: accept true/false and 1/0.
                    std::string bt = lex.next();
                    if (bt == "true") {
                        cfg.batching.preserve_ordering = true;
                    } else if (bt == "false") {
                        cfg.batching.preserve_ordering = false;
                    } else {
                        cfg.batching.preserve_ordering = parseInteger(bt) != 0;
                    }
                } else {
                    // Skip unknown scalar/message fields (Triton also defines
                    // queue policies default_queue_policy /
                    // priority_queue_policy, which InferLite does not implement
                    // yet).
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
        } else if (field == "sequence_batching") {
            require("{", "sequence_batching");
            cfg.sequence.enabled = true;
            while (true) {
                std::string f = lex.next();
                if (f == "}") break;
                if (f.empty()) throw PbtxtError("unexpected EOF in sequence_batching{}");
                if (f == "max_sequence_idle_microseconds") {
                    require(":", "sequence_batching.max_sequence_idle_microseconds");
                    cfg.sequence.max_sequence_idle_us = parseInteger(lex.next());
                } else if (f == "control_input") {
                    std::string t = lex.next();
                    if (t == ":") t = lex.next();
                    if (t == "[") {
                        while (true) {
                            std::string n = lex.next();
                            if (n == "]") break;
                            if (n == ",") continue;
                            if (n != "{") {
                                throw PbtxtError("expected '{' in control_input array");
                            }
                            cfg.sequence.control_input.push_back(parseControlInputBody(lex));
                        }
                    } else if (t == "{") {
                        cfg.sequence.control_input.push_back(parseControlInputBody(lex));
                    } else {
                        throw PbtxtError("expected '{' or '[' for control_input");
                    }
                } else if (f == "state") {
                    std::string t = lex.next();
                    if (t == ":") t = lex.next();
                    if (t == "[") {
                        while (true) {
                            std::string n = lex.next();
                            if (n == "]") break;
                            if (n == ",") continue;
                            if (n != "{") {
                                throw PbtxtError("expected '{' in state array");
                            }
                            cfg.sequence.states.push_back(parseSequenceStateBody(lex));
                        }
                    } else if (t == "{") {
                        cfg.sequence.states.push_back(parseSequenceStateBody(lex));
                    } else {
                        throw PbtxtError("expected '{' or '[' for sequence state");
                    }
                } else {
                    // Skip unknown scalar/message fields.
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
        } else if (field == "model_warmup") {
            // Triton repeated message. Accept both the array form
            //   model_warmup [ { ... }, { ... } ]
            // and the repeated-message form `model_warmup { ... }`.
            std::string t = lex.next();
            if (t == ":") t = lex.next();
            if (t == "[") {
                while (true) {
                    std::string n = lex.next();
                    if (n == "]") break;
                    if (n == ",") continue;
                    if (n != "{") {
                        throw PbtxtError("expected '{' in model_warmup array");
                    }
                    cfg.warmups.push_back(parseModelWarmupBody(lex));
                }
            } else if (t == "{") {
                cfg.warmups.push_back(parseModelWarmupBody(lex));
            } else {
                throw PbtxtError("expected '{' or '[' for model_warmup");
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
