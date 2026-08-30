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
//   Windows service modes (requires an elevated prompt to install/uninstall):
//     --install-service        Register this exe as a Windows service
//     --uninstall-service      Remove the registered Windows service
//     --service                Run under the SCM (falls back to console when
//                              launched manually)
//     --service-name=<name>    Service name (default: InferLite)
//
// Fail-fast: any repository/config/backend/integrity error aborts startup.
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "infer_lite.hpp"

#ifdef _WIN32
#include "service_support.hpp"
#endif

namespace inferlite {
namespace {

std::string requireValue(const std::string& arg, const std::string& flag) {
    size_t eq = arg.find('=');
    if (eq == std::string::npos || eq + 1 >= arg.size()) {
        throw std::runtime_error("missing value for " + flag);
    }
    return arg.substr(eq + 1);
}

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
#ifdef _WIN32
        << "  --install-service           Install InferLite as a Windows service (admin)\n"
        << "  --uninstall-service         Remove the InferLite Windows service (admin)\n"
        << "  --service                   Run under the Windows Service Control Manager\n"
        << "  --service-name=<name>       Service name to install/run (default: InferLite)\n"
#endif
        << "  --help                      Show this help\n";
}

}  // namespace
}  // namespace inferlite

// Shared, header-declared argument parser. Called by both the console entry
// point (main) and the Windows service worker so both use identical grammar.
inferlite::ServerOptions inferlite::parseServerOptions(const std::vector<std::string>& tokens) {
    ServerOptions opts;
    bool has_repo = false;

    for (const auto& arg : tokens) {
        if (arg == "--help" || arg == "-h") {
            throw std::runtime_error("--help");
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

namespace {

std::vector<std::string> argvToVector(int argc, char** argv) {
    std::vector<std::string> out;
    out.reserve(static_cast<size_t>(argc > 0 ? argc : 0));
    for (int i = 1; i < argc; ++i) out.emplace_back(argv[i]);
    return out;
}

int runServer(const inferlite::ServerOptions& opts) {
    using namespace inferlite;
    InferLite server(opts);
    server.start();
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
    server.stop();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace inferlite;
    std::vector<std::string> tokens = argvToVector(argc, argv);

    // --help / -h anywhere on the command line prints usage and exits.
    for (const auto& t : tokens) {
        if (t == "--help" || t == "-h") {
            printUsage(argv[0]);
            return 0;
        }
    }

    // --- Windows service control (install / uninstall) ----------------------
#ifdef _WIN32
    {
        bool do_install = false, do_uninstall = false;
        std::string svc_name = "InferLite";
        std::string svc_display = "InferLite Inference Server";
        std::string svc_user, svc_password;
        std::vector<std::string> server_args;
        for (const auto& t : tokens) {
            if (t == "--install-service") {
                do_install = true;
            } else if (t == "--uninstall-service") {
                do_uninstall = true;
            } else if (t.rfind("--service-name=", 0) == 0) {
                svc_name = t.substr(std::string("--service-name=").size());
            } else if (t.rfind("--service-display=", 0) == 0) {
                svc_display = t.substr(std::string("--service-display=").size());
            } else if (t.rfind("--install-service-user=", 0) == 0) {
                svc_user = t.substr(std::string("--install-service-user=").size());
            } else if (t.rfind("--install-service-password=", 0) == 0) {
                svc_password = t.substr(std::string("--install-service-password=").size());
            } else {
                server_args.push_back(t);
            }
        }

        if (do_install || do_uninstall) {
            try {
                if (do_uninstall) {
                    uninstallService(svc_name);
                    std::cout << "Uninstalled service '" << svc_name << "'.\n";
                }
                if (do_install) {
                    // Build the argument string the service should run with.
                    std::string args_str;
                    for (const auto& a : server_args) {
                        if (!args_str.empty()) args_str += " ";
                        args_str += a;
                    }
                    installService(svc_name, svc_display, args_str, svc_user, svc_password);
                    std::cout << "Installed service '" << svc_name << "'.\n"
                              << "Start it with: sc start " << svc_name << "\n"
                              << "or: scripts\\service.ps1 -Action start\n";
                }
            } catch (const std::exception& e) {
                std::cerr << "error: " << e.what() << "\n";
                return 2;
            }
            return 0;
        }
    }
#endif

    // --- Normal / service run -------------------------------------------------
    // Build the effective run tokens: drop --service / --service-name so they
    // are never fed to parseServerOptions (which would reject them as unknown).
    std::vector<std::string> run_tokens;
    for (const auto& x : tokens) {
        if (x == "--service" || x.rfind("--service-name=", 0) == 0) continue;
        run_tokens.push_back(x);
    }

    try {
        ServerOptions opts = parseServerOptions(run_tokens);

#ifdef _WIN32
        // Detect --service. The service worker hands control to the SCM and
        // blocks until the service is stopped. When the binary is launched
        // manually (not by the SCM) runAsService returns -1 and we fall
        // through to a normal foreground console run, so `--service` is safe
        // to type in a cmd window.
        for (const auto& t : tokens) {
            if (t == "--service") {
                int rc = runAsService(tokens, opts);
                if (rc != -1) return rc;   // ran under SCM until stopped
                break;                     // manual launch -> foreground console
            }
        }
#endif

        std::cerr << "[boot] constructing server...\n" << std::flush;
        return runServer(opts);
    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n" << std::flush;
        return 1;
    }
    return 0;
}
