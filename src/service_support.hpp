// service_support.hpp - Windows service integration for InferLite.
//
// Lets the inferlite.exe binary run under the Windows Service Control Manager
// (SCM) as well as from a plain console / cmd window. Three operational modes
// are exposed via command-line switches (see main.cpp):
//
//   --install-service   Create + configure the Windows service (requires admin).
//   --uninstall-service Delete the Windows service (requires admin).
//   --service           Run under the SCM as a service. When the binary is
//                       launched manually (not by SCM) this falls back to
//                       console mode so it is safe to run from a cmd window.
//
// Without any of these switches the server runs in normal console/foreground
// mode, exactly as before. This module is Windows-only (guarded by _WIN32).
#pragma once

#include <string>
#include <vector>

namespace inferlite {

// Install the Windows service "name" that runs this executable with the given
// command-line arguments (everything after the exe path). Returns true on
// success; throws std::runtime_error with a descriptive message on failure
// (e.g. insufficient privileges). Requires an elevated process.
bool installService(const std::string& name,
                    const std::string& display_name,
                    const std::string& args,
                    const std::string& user = std::string(),
                    const std::string& password = std::string());

// Delete the Windows service "name". Returns true on success; throws on
// failure (e.g. the service does not exist, or insufficient privileges).
void uninstallService(const std::string& name);

// Run the service. argv holds the parsed command-line tokens (excluding the
// service-name token SCM passes as argv[0]); options is the already-parsed
// ServerOptions. This function blocks until the service is stopped. Returns 0
// on a clean stop. When the binary is not started by SCM (launched manually),
// it runs the server in the foreground and returns -1 so the caller can fall
// back to normal console mode.
int runAsService(const std::vector<std::string>& args,
                 const struct ServerOptions& options);

}  // namespace inferlite
