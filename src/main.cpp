// main.cpp - InferLite Phase 1 entry point.
//
// Usage:
//   inferlite --model-repository=/path/to/models --http-port=8000 \
//             --max-queue-size=100 [--host=0.0.0.0] [--request-timeout-ms=30000]
//
// Fail-fast: any repository/config/backend error aborts startup immediately.
#include <iostream>
#include <stdexcept>
#include <string>

#include "infer_lite.hpp"

namespace {

void printUsage(const char* prog) {
    std::cerr
        << "InferLite - OpenVINO CPU-only HTTP inference server (Phase 1)\n\n"
        << "Usage: " << prog << " [options]\n\n"
        << "Options:\n"
        << "  --model-repository=<path>   Path to the model repository (required)\n"
        << "  --host=<addr>               Listen host (default: 0.0.0.0)\n"
        << "  --http-port=<port>          HTTP port (default: 8000)\n"
        << "  --max-queue-size=<n>        Max queued requests (default: 100, 0=unbounded)\n"
        << "  --request-timeout-ms=<n>    Per-request timeout in ms (default: 30000)\n"
        << "  --http-threads=<n>          HTTP worker threads (default: 4)\n"
        << "  --help                      Show this help\n";
}

inferlite::ServerOptions parseArgs(int argc, char** argv) {
    inferlite::ServerOptions opts;
    bool has_repo = false;

    auto requireValue = [](const std::string& arg, const std::string& flag) -> std::string {
        size_t eq = arg.find('=');
        if (eq == std::string::npos || eq + 1 >= arg.size()) {
            throw std::runtime_error("missing value for " + flag);
        }
        return arg.substr(eq + 1);
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        } else if (arg.rfind("--model-repository=", 0) == 0) {
            opts.model_repository = requireValue(arg, "--model-repository");
            has_repo = true;
        } else if (arg.rfind("--host=", 0) == 0) {
            opts.host = requireValue(arg, "--host");
        } else if (arg.rfind("--http-port=", 0) == 0) {
            opts.http_port = std::stoi(requireValue(arg, "--http-port"));
        } else if (arg.rfind("--max-queue-size=", 0) == 0) {
            opts.max_queue_size = static_cast<size_t>(
                std::max(0, std::stoi(requireValue(arg, "--max-queue-size"))));
        } else if (arg.rfind("--request-timeout-ms=", 0) == 0) {
            opts.request_timeout_ms = std::stoll(requireValue(arg, "--request-timeout-ms"));
        } else if (arg.rfind("--http-threads=", 0) == 0) {
            opts.http_threads = static_cast<size_t>(
                std::max(1, std::stoi(requireValue(arg, "--http-threads"))));
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }

    if (!has_repo) {
        throw std::runtime_error("missing required option --model-repository=<path>");
    }
    return opts;
}

}  // namespace

int main(int argc, char** argv) {
    inferlite::ServerOptions opts;
    try {
        opts = parseArgs(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n\n";
        printUsage(argv[0]);
        return 2;
    }

    try {
        // Fail-fast startup: repository scan + model load happen here and any
        // error aborts the server immediately.
        std::cerr << "[boot] constructing server...\n" << std::flush;
        inferlite::InferLite server(opts);
        std::cerr << "[boot] server constructed\n" << std::flush;
        server.start();
        std::cerr << "[boot] server started\n" << std::flush;
        std::cout << "InferLite ready: listening on " << opts.host << ":"
                  << opts.http_port << " (model repo: " << opts.model_repository << ")\n"
                  << std::flush;
        std::cout << "Press Ctrl+C to stop.\n" << std::flush;
        server.waitForShutdown();
    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n" << std::flush;
        return 1;
    }
    return 0;
}
