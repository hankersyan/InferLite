// main.cpp - InferLite entry point.
//
// Usage:
//   inferlite --model-repository=/path/to/models --http-port=8000 \
//             --max-queue-size=100 [--host=0.0.0.0] [--request-timeout-ms=30000]
//             [--validated-mode] [--audit-log=path] [--diagnostic-log=path]
//             [--max-input-size-bytes=N] [--max-output-size-bytes=N]
//             [--max-inference-time-ms=N] [--tls-cert=path] [--tls-key=path]
//             [--max-gpu-memory-mb=N] [--max-concurrent-gpu-instances=N]
//             [--gpu-device=N]
//
// Fail-fast: any repository/config/backend/integrity error aborts startup.
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

#include "infer_lite.hpp"

namespace {

void printUsage(const char* prog) {
    std::cerr
        << "InferLite - FDA-compliant CPU inference server (Phase 2)\n\n"
        << "Usage: " << prog << " [options]\n\n"
        << "Options:\n"
        << "  --model-repository=<path>   Path to the model repository (required)\n"
        << "  --host=<addr>               Listen host (default: 0.0.0.0)\n"
        << "  --http-port=<port>          HTTP port (default: 8000)\n"
        << "  --grpc-port=<port>          gRPC port (default: 0=disabled; built with gRPC)\n"
        << "  --max-queue-size=<n>        Max queued requests (default: 100, 0=unbounded)\n"
        << "  --request-timeout-ms=<n>    Per-request queue timeout in ms (default: 30000)\n"
        << "  --http-threads=<n>          HTTP worker threads (default: 4)\n"
        << "  --validated-mode            Require manifest.json + report readiness via self-test\n"
        << "  --audit-log=<path>          Tamper-evident audit log file (recommended in validated mode)\n"
        << "  --diagnostic-log=<path>     Engineer-facing diagnostic log file\n"
        << "  --max-input-size-bytes=<n>  Input size limit (default: 52428800)\n"
        << "  --max-output-size-bytes=<n> Output size limit (default: 52428800)\n"
        << "  --max-inference-time-ms=<n> Per-request inference time limit (default: 5000)\n"
        << "  --tls-cert=<path>           TLS certificate file (PEM) for validated mode\n"
        << "  --tls-key=<path>            TLS private key file (PEM)\n"
        << "  --software-version=<s>      Software version reported (default: InferLite 2.0.0)\n"
        << "  --max-gpu-memory-mb=<n>     Per-model GPU memory cap in MiB (default: 2048)\n"
        << "  --max-concurrent-gpu-instances=<n>  Max concurrent GPU instances (default: 4)\n"
        << "  --gpu-device=<n>            CUDA device index (default: 0, single GPU only)\n"
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
        } else if (arg.rfind("--grpc-port=", 0) == 0) {
            opts.grpc_port = std::stoi(requireValue(arg, "--grpc-port"));
        } else if (arg.rfind("--max-queue-size=", 0) == 0) {
            opts.max_queue_size = static_cast<size_t>(
                std::max(0, std::stoi(requireValue(arg, "--max-queue-size"))));
        } else if (arg.rfind("--request-timeout-ms=", 0) == 0) {
            opts.request_timeout_ms = std::stoll(requireValue(arg, "--request-timeout-ms"));
        } else if (arg.rfind("--http-threads=", 0) == 0) {
            opts.http_threads = static_cast<size_t>(
                std::max(1, std::stoi(requireValue(arg, "--http-threads"))));
        } else if (arg == "--validated-mode") {
            opts.validated_mode = true;
        } else if (arg.rfind("--audit-log=", 0) == 0) {
            opts.audit_log_path = requireValue(arg, "--audit-log");
        } else if (arg.rfind("--diagnostic-log=", 0) == 0) {
            opts.diagnostic_log_path = requireValue(arg, "--diagnostic-log");
        } else if (arg.rfind("--max-input-size-bytes=", 0) == 0) {
            opts.max_input_size_bytes = static_cast<size_t>(
                std::stoll(requireValue(arg, "--max-input-size-bytes")));
        } else if (arg.rfind("--max-output-size-bytes=", 0) == 0) {
            opts.max_output_size_bytes = static_cast<size_t>(
                std::stoll(requireValue(arg, "--max-output-size-bytes")));
        } else if (arg.rfind("--max-inference-time-ms=", 0) == 0) {
            opts.max_inference_time_ms = std::stoll(requireValue(arg, "--max-inference-time-ms"));
        } else if (arg.rfind("--tls-cert=", 0) == 0) {
            opts.tls_cert_file = requireValue(arg, "--tls-cert");
        } else if (arg.rfind("--tls-key=", 0) == 0) {
            opts.tls_key_file = requireValue(arg, "--tls-key");
        } else if (arg.rfind("--software-version=", 0) == 0) {
            opts.software_version = requireValue(arg, "--software-version");
        } else if (arg.rfind("--max-gpu-memory-mb=", 0) == 0) {
            opts.max_gpu_memory_mb = static_cast<size_t>(
                std::max(0, std::stoi(requireValue(arg, "--max-gpu-memory-mb"))));
        } else if (arg.rfind("--max-concurrent-gpu-instances=", 0) == 0) {
            opts.max_concurrent_gpu_instances = static_cast<size_t>(
                std::max(1, std::stoi(requireValue(arg, "--max-concurrent-gpu-instances"))));
        } else if (arg.rfind("--gpu-device=", 0) == 0) {
            opts.gpu_device = requireValue(arg, "--gpu-device");
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
        std::cerr << "[boot] constructing server...\n" << std::flush;
        inferlite::InferLite server(opts);
        std::cerr << "[boot] server constructed\n" << std::flush;
        server.start();
        std::cerr << "[boot] server started\n" << std::flush;
        std::cout << "InferLite ready: HTTP listening on " << opts.host << ":"
                  << opts.http_port << " (model repo: " << opts.model_repository << ")\n"
                  << std::flush;
#ifdef INFERLITE_ENABLE_GRPC
        if (opts.grpc_port > 0) {
            std::cout << "InferLite ready: gRPC listening on " << opts.host << ":"
                      << opts.grpc_port << "\n"
                      << std::flush;
        }
#endif
        std::cout << "Press Ctrl+C to stop.\n" << std::flush;
        server.waitForShutdown();
    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n" << std::flush;
        return 1;
    }
    return 0;
}
