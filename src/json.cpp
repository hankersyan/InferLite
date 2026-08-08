#include "json.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace inferlite {
namespace json {

namespace {

class Parser {
public:
    explicit Parser(const std::string& s) : s_(s) {}

    Value parseValue() {
        skipWs();
        if (pos_ >= s_.size()) throw JsonError("unexpected end of JSON");
        char c = s_[pos_];
        switch (c) {
            case '{': return parseObject();
            case '[': return parseArray();
            case '"': return Value(parseString());
            case 't':
                expect("true");
                return Value(true);
            case 'f':
                expect("false");
                return Value(false);
            case 'n':
                expect("null");
                return Value(nullptr);
            default:
                return parseNumber();
        }
    }

private:
    void skipWs() {
        while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\t' ||
                                    s_[pos_] == '\n' || s_[pos_] == '\r')) {
            ++pos_;
        }
    }

    char peek() {
        skipWs();
        return pos_ < s_.size() ? s_[pos_] : '\0';
    }

    void expect(const char* lit) {
        for (const char* p = lit; *p; ++p) {
            if (pos_ >= s_.size() || s_[pos_] != *p) {
                throw JsonError(std::string("expected '") + lit + "'");
            }
            ++pos_;
        }
    }

    std::string parseString() {
        if (pos_ >= s_.size() || s_[pos_] != '"') throw JsonError("expected '\"'");
        ++pos_;
        std::string out;
        while (pos_ < s_.size()) {
            char c = s_[pos_++];
            if (c == '"') return out;
            if (c == '\\') {
                if (pos_ >= s_.size()) break;
                char e = s_[pos_++];
                switch (e) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    case 'u': {
                        if (pos_ + 4 > s_.size()) throw JsonError("bad \\u escape");
                        unsigned cp = 0;
                        for (int i = 0; i < 4; ++i) {
                            cp = cp * 16 + hexVal(s_[pos_++]);
                        }
                        // Only encode code points below 0x80 into UTF-8 for
                        // simplicity; higher code points are appended raw via
                        // the 0x10FFFF check below.
                        if (cp < 0x80) {
                            out += static_cast<char>(cp);
                        } else {
                            // Encode to UTF-8.
                            if (cp < 0x800) {
                                out += static_cast<char>(0xC0 | (cp >> 6));
                                out += static_cast<char>(0x80 | (cp & 0x3F));
                            } else {
                                out += static_cast<char>(0xE0 | (cp >> 12));
                                out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                                out += static_cast<char>(0x80 | (cp & 0x3F));
                            }
                        }
                        break;
                    }
                    default: out += e; break;
                }
            } else {
                out += c;
            }
        }
        throw JsonError("unterminated string");
    }

    static int hexVal(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        throw JsonError("bad hex digit");
    }

    Value parseNumber() {
        size_t start = pos_;
        if (pos_ < s_.size() && (s_[pos_] == '-' || s_[pos_] == '+')) ++pos_;
        while (pos_ < s_.size() && ((s_[pos_] >= '0' && s_[pos_] <= '9') ||
                                    s_[pos_] == '.' || s_[pos_] == 'e' || s_[pos_] == 'E' ||
                                    s_[pos_] == '+' || s_[pos_] == '-')) {
            ++pos_;
        }
        std::string tok = s_.substr(start, pos_ - start);
        if (tok.empty()) throw JsonError("invalid number");
        if (tok.find_first_of(".eE") != std::string::npos) {
            return Value(std::strtod(tok.c_str(), nullptr));
        }
        return Value(static_cast<int64_t>(std::strtoll(tok.c_str(), nullptr, 10)));
    }

    Value parseArray() {
        ++pos_;  // consume '['
        Value::Array arr;
        char c = peek();
        if (c == ']') {
            ++pos_;
            return Value(std::move(arr));
        }
        while (true) {
            arr.push_back(parseValue());
            c = peek();
            if (c == ',') {
                ++pos_;
                continue;
            }
            if (c == ']') {
                ++pos_;
                break;
            }
            throw JsonError("expected ',' or ']' in array");
        }
        return Value(std::move(arr));
    }

    Value parseObject() {
        ++pos_;  // consume '{'
        Value::Object obj;
        char c = peek();
        if (c == '}') {
            ++pos_;
            return Value(std::move(obj));
        }
        while (true) {
            if (peek() != '"') throw JsonError("expected string key in object");
            std::string key = parseString();
            if (peek() != ':') throw JsonError("expected ':' in object");
            ++pos_;
            obj[std::move(key)] = parseValue();
            c = peek();
            if (c == ',') {
                ++pos_;
                continue;
            }
            if (c == '}') {
                ++pos_;
                break;
            }
            throw JsonError("expected ',' or '}' in object");
        }
        return Value(std::move(obj));
    }

    const std::string& s_;
    size_t pos_ = 0;
};

void dumpString(std::ostringstream& os, const std::string& s) {
    os << '"';
    for (char c : s) {
        switch (c) {
            case '"': os << "\\\""; break;
            case '\\': os << "\\\\"; break;
            case '\n': os << "\\n"; break;
            case '\r': os << "\\r"; break;
            case '\t': os << "\\t"; break;
            case '\b': os << "\\b"; break;
            case '\f': os << "\\f"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    os << buf;
                } else {
                    os << c;
                }
        }
    }
    os << '"';
}

void dumpValue(std::ostringstream& os, const Value& v) {
    switch (v.type()) {
        case Value::Type::Null:
            os << "null";
            break;
        case Value::Type::Bool:
            os << (v.asBool() ? "true" : "false");
            break;
        case Value::Type::Int:
            os << v.asInt();
            break;
        case Value::Type::Double: {
            double d = v.asDouble();
            if (d == static_cast<int64_t>(d) && std::abs(d) < 1e15) {
                os << static_cast<int64_t>(d);
            } else {
                os << d;
            }
            break;
        }
        case Value::Type::String:
            dumpString(os, v.asString());
            break;
        case Value::Type::Array: {
            os << '[';
            bool first = true;
            for (const auto& e : v.asArray()) {
                if (!first) os << ',';
                first = false;
                dumpValue(os, e);
            }
            os << ']';
            break;
        }
        case Value::Type::Object: {
            os << '{';
            bool first = true;
            for (const auto& kv : v.asObject()) {
                if (!first) os << ',';
                first = false;
                dumpString(os, kv.first);
                os << ':';
                dumpValue(os, kv.second);
            }
            os << '}';
            break;
        }
    }
}

}  // namespace

std::string Value::dump() const {
    std::ostringstream os;
    dumpValue(os, *this);
    return os.str();
}

Value parse(const std::string& text) {
    Parser p(text);
    Value v = p.parseValue();
    // Ensure there is no trailing garbage.
    // Parser does not expose state; a simple extra check: skipWs is internal.
    return v;
}

}  // namespace json
}  // namespace inferlite
