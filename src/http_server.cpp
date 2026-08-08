#include "http_server.hpp"

#include <winsock2.h>
#include <ws2tcpip.h>

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <sstream>
#include <stdexcept>

#pragma comment(lib, "ws2_32.lib")

namespace inferlite {

namespace {
constexpr size_t kMaxRequestHeaderBytes = 64 * 1024;
constexpr size_t kReadBufferSize = 64 * 1024;

std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

std::string toLower(std::string s) {
    for (auto& c : s) c = static_cast<char>(::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string urlDecode(const std::string& s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size()) {
            auto hex = [](char c) -> int {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                return -1;
            };
            int hi = hex(s[i + 1]);
            int lo = hex(s[i + 2]);
            if (hi >= 0 && lo >= 0) {
                out += static_cast<char>((hi << 4) | lo);
                i += 2;
                continue;
            }
        }
        out += s[i];
    }
    return out;
}

void parseQuery(const std::string& qs, std::map<std::string, std::string>& query) {
    size_t start = 0;
    while (start <= qs.size()) {
        size_t amp = qs.find('&', start);
        std::string pair =
            qs.substr(start, amp == std::string::npos ? std::string::npos : amp - start);
        size_t eq = pair.find('=');
        if (eq != std::string::npos) {
            query[urlDecode(pair.substr(0, eq))] = urlDecode(pair.substr(eq + 1));
        } else if (!pair.empty()) {
            query[urlDecode(pair)] = "";
        }
        if (amp == std::string::npos) break;
        start = amp + 1;
    }
}

class Socket {
public:
    explicit Socket(int fd = -1) : fd_(fd) {}
    ~Socket() { close(); }
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;
    Socket(Socket&& o) noexcept : fd_(o.fd_) { o.fd_ = -1; }
    Socket& operator=(Socket&& o) noexcept {
        if (this != &o) {
            close();
            fd_ = o.fd_;
            o.fd_ = -1;
        }
        return *this;
    }
    int get() const { return fd_; }
    void close() {
        if (fd_ != -1) {
            ::closesocket(fd_);
            fd_ = -1;
        }
    }

private:
    int fd_;
};

const char* statusText(int status) {
    switch (status) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 409: return "Conflict";
        case 500: return "Internal Server Error";
        case 503: return "Service Unavailable";
        default: return "OK";
    }
}

bool readRequest(Socket& sock, HttpRequest& req, std::string& err) {
    std::string buf;
    buf.reserve(8192);
    char tmp[kReadBufferSize];

    size_t header_end = std::string::npos;
    while (header_end == std::string::npos) {
        int n = ::recv(sock.get(), tmp, sizeof(tmp), 0);
        if (n == 0) return false;  // connection closed
        if (n < 0) {
            err = "recv failed";
            return false;
        }
        buf.append(tmp, static_cast<size_t>(n));
        if (buf.size() > kMaxRequestHeaderBytes) {
            err = "request header too large";
            return false;
        }
        header_end = buf.find("\r\n\r\n");
    }

    std::string head = buf.substr(0, header_end);
    size_t body_start = header_end + 4;

    size_t line_end = head.find("\r\n");
    std::string request_line = line_end == std::string::npos ? head : head.substr(0, line_end);
    std::istringstream line_ss(request_line);
    std::string method, target;
    line_ss >> method >> target;
    if (method.empty() || target.empty()) {
        err = "malformed request line";
        return false;
    }
    req.method = method;

    size_t qpos = target.find('?');
    req.path = urlDecode(qpos == std::string::npos ? target : target.substr(0, qpos));
    if (qpos != std::string::npos) {
        parseQuery(target.substr(qpos + 1), req.query);
    }

    size_t hstart = line_end == std::string::npos ? head.size() : line_end + 2;
    size_t pos = hstart;
    while (pos < head.size()) {
        size_t e = head.find("\r\n", pos);
        std::string hline = head.substr(pos, e == std::string::npos ? std::string::npos : e - pos);
        size_t colon = hline.find(':');
        if (colon != std::string::npos) {
            req.headers[toLower(trim(hline.substr(0, colon)))] = trim(hline.substr(colon + 1));
        }
        if (e == std::string::npos) break;
        pos = e + 2;
    }

    size_t content_length = 0;
    auto it = req.headers.find("content-length");
    if (it != req.headers.end()) {
        try {
            content_length = static_cast<size_t>(std::stoull(it->second));
        } catch (...) {
            err = "invalid content-length";
            return false;
        }
    }

    size_t have = buf.size() - body_start;
    req.body = buf.substr(body_start, have);
    while (req.body.size() < content_length) {
        int n = ::recv(sock.get(), tmp, sizeof(tmp), 0);
        if (n <= 0) {
            err = "connection lost while reading body";
            return false;
        }
        req.body.append(tmp, static_cast<size_t>(n));
    }
    return true;
}

void sendResponse(Socket& sock, const HttpResponse& resp) {
    std::ostringstream head;
    head << "HTTP/1.1 " << resp.status << " " << statusText(resp.status) << "\r\n";
    head << "Content-Type: " << resp.content_type << "\r\n";
    head << "Content-Length: " << resp.body.size() << "\r\n";
    head << "Connection: close\r\n";
    head << "Access-Control-Allow-Origin: *\r\n";
    head << "Cache-Control: no-store\r\n";
    head << "\r\n";
    std::string out = head.str() + resp.body;
    size_t sent = 0;
    while (sent < out.size()) {
        int n = ::send(sock.get(), out.data() + sent, static_cast<int>(out.size() - sent), 0);
        if (n <= 0) return;
        sent += static_cast<size_t>(n);
    }
}

}  // namespace

HttpServer::HttpServer(std::string host, int port, HttpHandler handler, size_t thread_count)
    : host_(std::move(host)),
      port_(port),
      handler_(std::move(handler)),
      thread_count_(thread_count > 0 ? thread_count : 1) {}

HttpServer::~HttpServer() {
    stop();
}

void HttpServer::start() {
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        throw std::runtime_error("WSAStartup failed");
    }

    listen_sock_ = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listen_sock_ == -1) {
        WSACleanup();
        throw std::runtime_error("socket() failed: WSA error " +
                                 std::to_string(WSAGetLastError()));
    }

    int opt = 1;
    ::setsockopt(listen_sock_, SOL_SOCKET, SO_REUSEADDR,
                 reinterpret_cast<const char*>(&opt), sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    if (host_.empty() || host_ == "0.0.0.0" || host_ == "*") {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        ::inet_pton(AF_INET, host_.c_str(), &addr.sin_addr);
    }
    addr.sin_port = htons(static_cast<u_short>(port_));

    if (::bind(listen_sock_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        int e = WSAGetLastError();
        ::closesocket(listen_sock_);
        listen_sock_ = -1;
        WSACleanup();
        throw std::runtime_error("bind() failed on port " + std::to_string(port_) +
                                 ": WSA error " + std::to_string(e));
    }

    if (::listen(listen_sock_, SOMAXCONN) != 0) {
        int e = WSAGetLastError();
        ::closesocket(listen_sock_);
        listen_sock_ = -1;
        WSACleanup();
        throw std::runtime_error("listen() failed: WSA error " + std::to_string(e));
    }

    running_ = true;
    accept_thread_ = std::thread([this]() { acceptLoop(); });
    for (size_t i = 0; i < thread_count_; ++i) {
        workers_.emplace_back([this]() { workerLoop(); });
    }
}

void HttpServer::stop() {
    if (!running_ && accept_thread_.get_id() == std::thread::id()) return;
    running_ = false;
    if (listen_sock_ != -1) {
        ::closesocket(listen_sock_);
        listen_sock_ = -1;
    }
    // Wake any workers blocked waiting for a socket.
    {
        std::lock_guard<std::mutex> lock(sock_mu_);
        sock_cv_.notify_all();
    }
    if (accept_thread_.joinable()) accept_thread_.join();
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
    WSACleanup();
}

void HttpServer::acceptLoop() {
    while (running_) {
        sockaddr_in client{};
        int len = sizeof(client);
        int cs = ::accept(listen_sock_, reinterpret_cast<sockaddr*>(&client), &len);
        if (!running_) break;
        if (cs == -1) {
            if (WSAGetLastError() == WSAEINTR || WSAGetLastError() == WSAENETDOWN) continue;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        {
            std::lock_guard<std::mutex> lock(sock_mu_);
            pending_.push_back(cs);
        }
        sock_cv_.notify_one();
    }
}

void HttpServer::workerLoop() {
    while (running_) {
        int fd = -1;
        {
            std::unique_lock<std::mutex> lock(sock_mu_);
            sock_cv_.wait(lock, [this]() { return !running_ || !pending_.empty(); });
            if (!running_ && pending_.empty()) return;
            fd = pending_.front();
            pending_.pop_front();
        }
        Socket sock(fd);
        HttpRequest req;
        std::string err;
        if (!readRequest(sock, req, err)) continue;
        HttpResponse resp;
        try {
            resp = handler_(req);
        } catch (const std::exception& e) {
            resp.status = 500;
            resp.body = std::string("{\"error\":\"") + e.what() + "\"}";
        }
        sendResponse(sock, resp);
    }
}

}  // namespace inferlite
