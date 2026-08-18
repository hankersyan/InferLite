# gen_grpc.ps1 - Regenerate gRPC/protobuf C++ code from proto/grpc_service.proto.
# Uses the Anaconda protoc + grpc_cpp_plugin. Output goes to generated/.
$ErrorActionPreference = "Stop"
# Repo root is the parent of this scripts/ directory.
$RepoRoot = Split-Path -Parent $PSScriptRoot
$lib = "C:\Apps\anaconda3\Library"
$protoc = "$lib\bin\protoc.exe"
$plugin = "$lib\bin\grpc_cpp_plugin.exe"
if (-not (Test-Path $protoc)) { throw "protoc not found: $protoc" }
if (-not (Test-Path $plugin)) { throw "grpc_cpp_plugin not found: $plugin" }

$protoDir = Join-Path $RepoRoot "proto"
$outDir = Join-Path $RepoRoot "generated"
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

Write-Host "== Generating C++ from $protoDir\grpc_service.proto =="
& $protoc --proto_path=$protoDir `
          --cpp_out=$outDir `
          --grpc_out=$outDir `
          --plugin=protoc-gen-grpc=$plugin `
          "$protoDir\grpc_service.proto"
if ($LASTEXITCODE -ne 0) { throw "protoc generation failed" }

Write-Host "Generated:"
Get-ChildItem $outDir -Filter "grpc_service*" | Select-Object -ExpandProperty Name
