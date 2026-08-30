// service_support.cpp - Windows service integration for InferLite.
//
// Implements install/uninstall/run for the SCM. The service stores its
// command-line arguments in the registry (Parameters\ConfigArgs) at install
// time; at start the SCM invokes ServiceMain, which re-reads those arguments
// so the service reproduces the exact command line configured at install.
//
// The heavy lifting (InferLite construction, HTTP/gRPC start, graceful stop)
// stays in main.cpp / infer_lite.cpp; this file only wraps the SCM protocol
// (status reporting + control handling) around the already-existing
// InferLite::requestStop() / start() / stop() / waitForShutdown() surface.
#ifdef _WIN32

#include "service_support.hpp"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsvc.h>

#include <shellapi.h>

#include <atomic>
#include <cstdio>
#include <stdexcept>

#include "infer_lite.hpp"

#pragma comment(lib, "advapi32.lib")
#pragma comment(lib, "shell32.lib")

namespace inferlite {
namespace {

constexpr DWORD kStopTimeoutMs = 15000;

// The server instance currently running under the service. Set in ServiceMain
// before starting and cleared after stop; accessed by the SCM control handler
// on a separate thread.
InferLite* g_server = nullptr;
std::atomic<bool> g_service_running{false};
std::atomic<bool> g_stop_requested{false};

SERVICE_STATUS g_status{};
SERVICE_STATUS_HANDLE g_status_handle = nullptr;

void reportStatus(DWORD state, DWORD wait_hint_ms = 0, DWORD exit_code = NO_ERROR) {
    g_status.dwServiceType = SERVICE_WIN32_OWN_PROCESS;
    g_status.dwCurrentState = state;
    g_status.dwControlsAccepted =
        (state == SERVICE_RUNNING) ? (SERVICE_ACCEPT_STOP | SERVICE_ACCEPT_SHUTDOWN) : 0;
    g_status.dwWin32ExitCode = exit_code;
    g_status.dwServiceSpecificExitCode = 0;
    g_status.dwCheckPoint = 0;
    g_status.dwWaitHint = wait_hint_ms;
    if (g_status_handle) {
        SetServiceStatus(g_status_handle, &g_status);
    }
}

// Parse a UTF-8 argument string into UTF-8 tokens, honoring Windows quoting
// rules via CommandLineToArgvW.
std::vector<std::string> splitArgsUtf8(const std::string& args) {
    std::vector<std::string> out;
    if (args.empty()) return out;
    // Convert UTF-8 -> UTF-16 for the Win32 parser.
    int wide_len = MultiByteToWideChar(CP_UTF8, 0, args.c_str(), -1, nullptr, 0);
    std::wstring wide(static_cast<size_t>(wide_len), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, args.c_str(), -1, wide.data(), wide_len);

    int argc = 0;
    LPWSTR* argvw = CommandLineToArgvW(wide.c_str(), &argc);
    if (!argvw) return out;
    for (int i = 0; i < argc; ++i) {
        int len = WideCharToMultiByte(CP_UTF8, 0, argvw[i], -1, nullptr, 0, nullptr, nullptr);
        std::string s(static_cast<size_t>(len), '\0');
        WideCharToMultiByte(CP_UTF8, 0, argvw[i], -1, s.data(), len, nullptr, nullptr);
        if (!s.empty() && s.back() == '\0') s.pop_back();
        out.push_back(std::move(s));
    }
    LocalFree(argvw);
    return out;
}

std::string toUtf8(const std::wstring& w) {
    int len = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string s(static_cast<size_t>(len), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, s.data(), len, nullptr, nullptr);
    if (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}

std::wstring toWide(const std::string& s) {
    int len = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w(static_cast<size_t>(len), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), len);
    if (!w.empty() && w.back() == L'\0') w.pop_back();
    return w;
}

// Quote an argument for a command line (CommandLineToArgvW-compatible).
std::string quoteArg(const std::string& s) {
    if (s.find_first_of(" \t\"") == std::string::npos && !s.empty()) return s;
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') {
            out += "\\\"";
        } else {
            out += c;
        }
    }
    out += "\"";
    return out;
}

// Read the ConfigArgs value from the service Parameters registry key.
std::string readConfigArgs(const std::string& name) {
    HKEY key = nullptr;
    std::wstring subkey = L"SYSTEM\\CurrentControlSet\\Services\\" + toWide(name) + L"\\Parameters";
    if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, subkey.c_str(), 0, KEY_READ, &key) != ERROR_SUCCESS) {
        throw std::runtime_error("failed to open service Parameters key for '" + name + "'");
    }
    DWORD type = 0, size = 0;
    if (RegQueryValueExW(key, L"ConfigArgs", nullptr, &type, nullptr, &size) != ERROR_SUCCESS) {
        RegCloseKey(key);
        throw std::runtime_error("ConfigArgs not found for service '" + name + "'");
    }
    std::wstring w(static_cast<size_t>(size) / sizeof(wchar_t), L'\0');
    LONG rc = RegQueryValueExW(key, L"ConfigArgs", nullptr, &type,
                               reinterpret_cast<LPBYTE>(w.data()), &size);
    RegCloseKey(key);
    if (rc != ERROR_SUCCESS) {
        throw std::runtime_error("failed to read ConfigArgs for service '" + name + "'");
    }
    while (!w.empty() && w.back() == L'\0') w.pop_back();
    return toUtf8(w);
}

}  // namespace

bool installService(const std::string& name,
                    const std::string& display_name,
                    const std::string& args,
                    const std::string& user,
                    const std::string& password) {
    // Full path of the running executable.
    wchar_t exe_path[MAX_PATH + 4] = {0};
    DWORD n = GetModuleFileNameW(nullptr, exe_path, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) {
        throw std::runtime_error("failed to resolve executable path");
    }
    std::string quoted_exe = quoteArg(toUtf8(exe_path));

    // Build the service binary path: "<exe>" --service --service-name=<name> [args...]
    // --service-name is embedded so that when the SCM launches this binary,
    // StartServiceCtrlDispatcher is handed the exact registered service name.
    std::string cmdline = quoted_exe + " --service --service-name=" + name;
    if (!args.empty()) cmdline += " " + args;

    SC_HANDLE scm = OpenSCManagerW(nullptr, nullptr, SC_MANAGER_ALL_ACCESS);
    if (!scm) {
        DWORD e = GetLastError();
        throw std::runtime_error("OpenSCManager failed (error " + std::to_string(e) +
                                 "). Install must be run from an elevated prompt.");
    }

    // Delete an existing service with the same name first (idempotent install).
    SC_HANDLE existing = OpenServiceW(scm, toWide(name).c_str(), DELETE);
    if (existing) {
        DeleteService(existing);
        CloseServiceHandle(existing);
    }

    std::wstring user_w = toWide(user);
    std::wstring pass_w = toWide(password);

    // SERVICE_WIN32_OWN_PROCESS, running as LocalSystem unless a service
    // account was supplied via -ServiceUser/-ServicePassword.
    SC_HANDLE svc = CreateServiceW(
        scm,
        toWide(name).c_str(),
        toWide(display_name.empty() ? name : display_name).c_str(),
        SERVICE_ALL_ACCESS,
        SERVICE_WIN32_OWN_PROCESS,
        SERVICE_AUTO_START,
        SERVICE_ERROR_NORMAL,
        toWide(cmdline).c_str(),
        nullptr, nullptr, nullptr,
        user.empty() ? nullptr : user_w.c_str(),
        password.empty() ? nullptr : pass_w.c_str());
    if (!svc) {
        DWORD e = GetLastError();
        CloseServiceHandle(scm);
        throw std::runtime_error("CreateService failed (error " + std::to_string(e) +
                                 "). Run from an elevated prompt.");
    }

    // Store the argument list so the service can reproduce it at start.
    HKEY key = nullptr;
    std::wstring subkey = L"SYSTEM\\CurrentControlSet\\Services\\" + toWide(name) + L"\\Parameters";
    if (RegCreateKeyExW(HKEY_LOCAL_MACHINE, subkey.c_str(), 0, nullptr, 0,
                        KEY_WRITE, nullptr, &key, nullptr) == ERROR_SUCCESS) {
        std::wstring args_w = toWide(args);
        RegSetValueExW(key, L"ConfigArgs", 0, REG_SZ,
                       reinterpret_cast<const BYTE*>(args_w.c_str()),
                       static_cast<DWORD>((args_w.size() + 1) * sizeof(wchar_t)));
        RegCloseKey(key);
    }

    // Set failure recovery: restart on crash (best-effort, ignore errors).
    SC_ACTION actions[3] = {
        {SC_ACTION_RESTART, 1000},
        {SC_ACTION_RESTART, 5000},
        {SC_ACTION_RESTART, 10000},
    };
    SERVICE_FAILURE_ACTIONS fa{};
    fa.dwResetPeriod = 86400;
    fa.cActions = 3;
    fa.lpsaActions = actions;
    ChangeServiceConfig2W(svc, SERVICE_CONFIG_FAILURE_ACTIONS, &fa);

    CloseServiceHandle(svc);
    CloseServiceHandle(scm);
    return true;
}

void uninstallService(const std::string& name) {
    SC_HANDLE scm = OpenSCManagerW(nullptr, nullptr, SC_MANAGER_ALL_ACCESS);
    if (!scm) {
        DWORD e = GetLastError();
        throw std::runtime_error("OpenSCManager failed (error " + std::to_string(e) +
                                 "). Uninstall must be run from an elevated prompt.");
    }
    SC_HANDLE svc = OpenServiceW(scm, toWide(name).c_str(), DELETE);
    if (!svc) {
        DWORD e = GetLastError();
        CloseServiceHandle(scm);
        throw std::runtime_error("service '" + name + "' not found (error " +
                                 std::to_string(e) + ").");
    }
    if (!DeleteService(svc)) {
        DWORD e = GetLastError();
        CloseServiceHandle(svc);
        CloseServiceHandle(scm);
        throw std::runtime_error("DeleteService failed for '" + name + "' (error " +
                                 std::to_string(e) + ").");
    }
    CloseServiceHandle(svc);
    CloseServiceHandle(scm);
}

namespace {

// The SCM control handler runs on a dedicated thread provided by the SCM. It
// must respond quickly: for STOP/SHUTDOWN we request a graceful shutdown and
// let ServiceMain report the stopped state once the server has actually ended.
void WINAPI ServiceControlHandler(DWORD control) {
    switch (control) {
        case SERVICE_CONTROL_STOP:
        case SERVICE_CONTROL_SHUTDOWN:
            if (g_status.dwCurrentState != SERVICE_STOPPED) {
                g_stop_requested = true;
                reportStatus(SERVICE_STOP_PENDING, kStopTimeoutMs);
                if (g_server) g_server->requestStop();
            }
            break;
        case SERVICE_CONTROL_INTERROGATE:
            SetServiceStatus(g_status_handle, &g_status);
            break;
        default:
            break;
    }
}

// Parse the server args (stored at install time) and run the server. This is
// the worker thread body started from ServiceMain; it blocks until the server
// stops, then reports SERVICE_STOPPED.
DWORD WINAPI ServiceWorker(LPVOID param) {
    std::string* args_ptr = static_cast<std::string*>(param);
    std::string args = *args_ptr;
    delete args_ptr;

    try {
        reportStatus(SERVICE_START_PENDING, 5000);

        std::vector<std::string> tokens = splitArgsUtf8(args);
        // Reuse main.cpp's arg parser via a small shim: build a temporary
        // argc/argv in the parser's expected form.
        ServerOptions opts = parseServerOptions(tokens);

        InferLite server(opts);
        g_server = &server;
        server.start();
        reportStatus(SERVICE_RUNNING);
        g_service_running = true;

        server.waitForShutdown();
        server.stop();
        g_server = nullptr;
        g_service_running = false;
        reportStatus(SERVICE_STOPPED);
        return 0;
    } catch (const std::exception& e) {
        g_server = nullptr;
        g_service_running = false;
        std::fprintf(stderr, "InferLite service FATAL: %s\n", e.what());
        reportStatus(SERVICE_STOPPED, 0, static_cast<DWORD>(SERVICE_ERROR_CRITICAL));
        return 1;
    }
}

}  // namespace

void WINAPI ServiceMain(DWORD argc, wchar_t** argv) {
    // The service name is the first (and only) SCM-supplied token.
    std::string name = (argc > 0) ? toUtf8(argv[0]) : "InferLite";

    g_status_handle = RegisterServiceCtrlHandlerW(
        toWide(name).c_str(), ServiceControlHandler);
    if (!g_status_handle) return;

    g_status.dwServiceType = SERVICE_WIN32_OWN_PROCESS;
    reportStatus(SERVICE_START_PENDING, 5000);

    std::string args;
    try {
        args = readConfigArgs(name);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "InferLite service: %s\n", e.what());
        reportStatus(SERVICE_STOPPED, 0, static_cast<DWORD>(SERVICE_ERROR_CRITICAL));
        return;
    }

    // Run the server on a worker thread so ServiceMain can keep servicing SCM
    // control events (SetServiceStatus) and respond to STOP promptly.
    HANDLE worker = CreateThread(nullptr, 0, ServiceWorker,
                                 new std::string(std::move(args)), 0, nullptr);
    if (!worker) {
        reportStatus(SERVICE_STOPPED, 0, static_cast<DWORD>(SERVICE_ERROR_CRITICAL));
        return;
    }
    WaitForSingleObject(worker, INFINITE);
    CloseHandle(worker);
}

int runAsService(const std::vector<std::string>& args,
                 const ServerOptions& options) {
    // When launched by SCM, StartServiceCtrlDispatcher returns ERROR_SUCCESS.
    // When launched manually it returns FALSE with ERROR_FAILED_SERVICE_CONTROLLER_CONNECT.
    std::wstring svc_name = L"InferLite";
    for (const auto& a : args) {
        if (a.rfind("--service-name=", 0) == 0) {
            svc_name = toWide(a.substr(std::string("--service-name=").size()));
        }
    }

    SERVICE_TABLE_ENTRYW table[] = {
        {const_cast<LPWSTR>(svc_name.c_str()), ServiceMain},
        {nullptr, nullptr},
    };
    if (!StartServiceCtrlDispatcherW(table)) {
        DWORD e = GetLastError();
        if (e == ERROR_FAILED_SERVICE_CONTROLLER_CONNECT) {
            // Not under SCM (run manually from a cmd window): fall back to a
            // normal foreground console run so --service is safe to type.
            return -1;
        }
        throw std::runtime_error("StartServiceCtrlDispatcher failed (error " +
                                 std::to_string(e) + ").");
    }
    return 0;
}

}  // namespace inferlite

#endif  // _WIN32
