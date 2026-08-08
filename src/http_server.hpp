// http_server.hpp - Lightweight synchronous HTTP/1.1 server on Windows
// (Winsock2). Exposes a fixed set of REST routes and dispatches to a callback.
#pragma once

#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace inferlite {

struct HttpRequest {
    std::string method;     // GET / POST
    std::string path;       // request target, e.g. /v2/health/ready
    std::string body;       // request body (for POST)
    std::map<std::string, std::string> headers;
    std::map<std::string, std::string> query;  // parsed query string
};

struct HttpResponse {
    int status = 200;
    std::string content_type = "application/json";
    std::string body;
};

// Handler signature: (request) -> response.
using HttpHandler = std::function<HttpResponse(const HttpRequest&)>;

class HttpServer {
public:
    // handler: receives every request. Returns nullopt to respond 404.
    HttpServer(std::string host, int port, HttpHandler handler, size_t thread_count);
    ~HttpServer();

    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    // Start listening and spawn worker threads. Returns immediately.
    // Throws std::runtime_error on socket setup failure.
    void start();
    // Stop the listener and join workers.
    void stop();

    int port() const { return port_; }

private:
    void acceptLoop();
    void workerLoop();
    void handleConnection(void* clientSocketPtr);

    std::string host_;
    int port_;
    HttpHandler handler_;
    size_t thread_count_;

    int listen_sock_ = -1;
    bool running_ = false;
    std::thread accept_thread_;
    std::vector<std::thread> workers_;

    // Queue of accepted client sockets shared with worker threads.
    std::mutex sock_mu_;
    std::condition_variable sock_cv_;
    std::deque<int> pending_;
};

}  // namespace inferlite
