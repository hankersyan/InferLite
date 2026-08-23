# gen_grpc_py.ps1 - Generate Python gRPC stubs for the test client.
$ErrorActionPreference = "Stop"
# Repo root is the parent of this scripts/ directory.
$RepoRoot = Split-Path -Parent $PSScriptRoot
$lib = "C:\Apps\anaconda3\Library"
$protoc = "$lib\bin\protoc.exe"
$plugin = "$lib\bin\grpc_python_plugin.exe"
$protoDir = Join-Path $RepoRoot "proto"
$out = Join-Path $RepoRoot "generated\py"
if (-not (Test-Path $plugin)) {
  # vcpkg's grpc ships grpc_python_plugin under tools; try it.
  $vcpkgPlugin = "C:\Test\vcpkg\vcpkg-2024.12.16\installed\x64-windows\tools\grpc\grpc_python_plugin.exe"
  if (Test-Path $vcpkgPlugin) { $plugin = $vcpkgPlugin }
}
if (-not (Test-Path $plugin)) { throw "grpc_python_plugin not found" }
New-Item -ItemType Directory -Force -Path $out | Out-Null
& $protoc --proto_path=$protoDir --python_out=$out --grpc_python_out=$out --plugin=protoc-gen-grpc_python=$plugin "$protoDir\grpc_service.proto" 2>&1
if ($LASTEXITCODE -ne 0) { throw "python stub generation failed" }
Get-ChildItem $out | Select-Object -ExpandProperty Name
