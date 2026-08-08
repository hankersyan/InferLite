#include "pbtxt.hpp"

#include <cctype>
#include <sstream>

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

}  // namespace

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
    if (cfg.inputs.empty()) throw PbtxtError("config for '" + cfg.name + "' has no inputs");
    if (cfg.outputs.empty()) throw PbtxtError("config for '" + cfg.name + "' has no outputs");
    return cfg;
}

}  // namespace inferlite
