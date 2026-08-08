// json.hpp - Minimal JSON parser/serializer for the REST API. Self-contained,
// no external dependencies. Not a fully conformant JSON library, but covers the
// Triton-style request/response payloads used in Phase 1.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace inferlite {
namespace json {

struct JsonError : public std::runtime_error {
    explicit JsonError(const std::string& msg) : std::runtime_error(msg) {}
};

class Value;

// JSON value type using a discriminated union.
class Value {
public:
    enum class Type { Null, Bool, Int, Double, String, Array, Object };

    using Array = std::vector<Value>;
    using Object = std::map<std::string, Value>;

    Value() : v_(nullptr) {}
    Value(std::nullptr_t) : v_(nullptr) {}
    Value(bool b) : v_(b) {}
    Value(int v) : v_(static_cast<int64_t>(v)) {}
    Value(int64_t v) : v_(v) {}
    Value(uint64_t v) : v_(static_cast<int64_t>(v)) {}
    Value(double d) : v_(d) {}
    Value(const char* s) : v_(std::string(s)) {}
    Value(std::string s) : v_(std::move(s)) {}
    Value(Array a) : v_(std::move(a)) {}
    Value(Object o) : v_(std::move(o)) {}

    Type type() const { return static_cast<Type>(v_.index()); }

    bool isNull() const { return type() == Type::Null; }
    bool isBool() const { return type() == Type::Bool; }
    bool isNumber() const { return type() == Type::Int || type() == Type::Double; }
    bool isString() const { return type() == Type::String; }
    bool isArray() const { return type() == Type::Array; }
    bool isObject() const { return type() == Type::Object; }

    bool asBool() const { return std::get<bool>(v_); }
    int64_t asInt() const { return std::get<int64_t>(v_); }
    double asDouble() const {
        if (isNumber()) {
            if (type() == Type::Int) return static_cast<double>(std::get<int64_t>(v_));
            return std::get<double>(v_);
        }
        return 0.0;
    }
    const std::string& asString() const { return std::get<std::string>(v_); }
    const Array& asArray() const { return std::get<Array>(v_); }
    Array& asArray() { return std::get<Array>(v_); }
    const Object& asObject() const { return std::get<Object>(v_); }
    Object& asObject() { return std::get<Object>(v_); }

    bool has(const std::string& key) const {
        return isObject() && asObject().count(key) > 0;
    }
    const Value* find(const std::string& key) const {
        if (!isObject()) return nullptr;
        auto it = asObject().find(key);
        return it == asObject().end() ? nullptr : &it->second;
    }

    // Serialize to JSON string.
    std::string dump() const;

private:
    std::variant<std::nullptr_t, bool, int64_t, double, std::string, Array, Object> v_;
};

// Parse a JSON string into a Value. Throws JsonError on failure.
Value parse(const std::string& text);

}  // namespace json
}  // namespace inferlite
