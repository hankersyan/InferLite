cmake_minimum_required(VERSION 3.15)
project(InferLite LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------------------
# OpenVINO toolkit. Prefer the env var OPENVINO_INSTALL_DIR; otherwise use the
# well-known location under c:\tools. The user can override via -DOPENVINO_ROOT=...
# ---------------------------------------------------------------------------
if(DEFINED OPENVINO_ROOT)
    set(OV_INSTALL ${OPENVINO_ROOT})
elseif(DEFINED ENV{OPENVINO_INSTALL_DIR})
    set(OV_INSTALL $ENV{OPENVINO_INSTALL_DIR})
else()
    set(OV_INSTALL "c:/tools/openvino/openvino_toolkit_windows_2025.3.0.19807.44526285f24_x86_64")
endif()

set(OV_RUNTIME_DIR ${OV_INSTALL}/runtime)
set(OV_INCLUDE_DIR ${OV_RUNTIME_DIR}/include)
set(OV_LIB_DIR ${OV_RUNTIME_DIR}/lib/intel64/Release)
set(OV_BIN_DIR ${OV_RUNTIME_DIR}/bin/intel64/Release)

if(NOT EXISTS ${OV_INCLUDE_DIR}/openvino/openvino.hpp)
    message(FATAL_ERROR "OpenVINO include dir not found at ${OV_INCLUDE_DIR}. "
        "Set OPENVINO_ROOT to your OpenVINO toolkit install.")
endif()

# ---------------------------------------------------------------------------
# TensorRT (Phase 3, GPU). Opt-in: the GPU backend is compiled only when a
# TensorRT SDK is available. Locate it via -DTENSORRT_ROOT=<dir> (the directory
# containing include/ and lib/) or the env var TENSORRT_ROOT. When absent, the
# server builds and runs CPU-only with no regression.
# ---------------------------------------------------------------------------
set(INFERLITE_ENABLE_GPU OFF)
if(DEFINED TENSORRT_ROOT)
    set(TRT_ROOT ${TENSORRT_ROOT})
elseif(DEFINED ENV{TENSORRT_ROOT})
    set(TRT_ROOT $ENV{TENSORRT_ROOT})
endif()
if(TRT_ROOT AND EXISTS ${TRT_ROOT}/include/NvInfer.h)
    set(INFERLITE_ENABLE_GPU ON)
    message(STATUS "TensorRT SDK found at ${TRT_ROOT}; GPU backend enabled.")
else()
    message(STATUS "TensorRT SDK not found; building CPU-only (set TENSORRT_ROOT to enable GPU).")
endif()

# ---------------------------------------------------------------------------
# gRPC (Phase 5, protocol). Opt-in: the gRPC interface is compiled only when the
# user explicitly enables it with -DINFERLITE_ENABLE_GRPC=ON and provides a
# gRPC C++ SDK via -DGRPC_ROOT=<dir> (the dir containing include/ and lib/) or
# the env var GRPC_ROOT. When disabled, the server builds and runs HTTP-only
# with no regression.
#
# Two SDK layouts are supported:
#   * Source-built (vcpkg / system MSVC):  grpc++.lib/grpc.lib are real static
#     libraries compiled with the local toolchain. Tools live under tools/
#     (tools/protobuf/protoc.exe, tools/grpc/grpc_cpp_plugin.exe) and runtime
#     DLLs (libprotobuf.dll, abseil_dll.dll, ...) under bin/. This layout has a
#     matching STL/CRT ABI and runs correctly.
#   * Prebuilt DLL stack (e.g. Anaconda):  grpc++.lib etc. are DLL import
#     libraries that are stale vs. the shipped DLLs. They must be regenerated
#     from the DLLs (gen_grpc_importlibs.ps1). This layout is built with an
#     older MSVC and crashes on RPC dispatch (ABI mismatch); rebuild gRPC from
#     source with the matching toolchain instead.
# ---------------------------------------------------------------------------
# Respect -DINFERLITE_ENABLE_GRPC=ON from the command line / cache; default OFF.
if(NOT DEFINED INFERLITE_ENABLE_GRPC)
    set(INFERLITE_ENABLE_GRPC OFF)
endif()
if(INFERLITE_ENABLE_GRPC)
    if(DEFINED GRPC_ROOT)
        set(GRPC_ROOT_DIR ${GRPC_ROOT})
    elseif(DEFINED ENV{GRPC_ROOT})
        set(GRPC_ROOT_DIR $ENV{GRPC_ROOT})
    else()
        set(GRPC_ROOT_DIR "C:/Apps/anaconda3/Library")
        message(STATUS "GRPC_ROOT not set; probing well-known path ${GRPC_ROOT_DIR}")
    endif()

    if(NOT EXISTS ${GRPC_ROOT_DIR}/include/grpcpp/grpcpp.h
       OR NOT EXISTS ${GRPC_ROOT_DIR}/lib/grpc++.lib)
        message(FATAL_ERROR
            "INFERLITE_ENABLE_GRPC=ON but no gRPC C++ SDK at ${GRPC_ROOT_DIR}. "
            "Set GRPC_ROOT to a directory containing include/grpcpp/grpcpp.h and "
            "lib/grpc++.lib (e.g. a vcpkg installed/x64-windows).")
    endif()
    set(GRPC_INCLUDE_DIR ${GRPC_ROOT_DIR}/include)
    set(GRPC_LIB_DIR ${GRPC_ROOT_DIR}/lib)
    set(GRPC_BIN_DIR ${GRPC_ROOT_DIR}/bin)
    # Locate protoc / grpc_cpp_plugin: bin/ (Anaconda) or tools/* (vcpkg).
    find_program(PROTOC_EXECUTABLE protoc
        PATHS ${GRPC_BIN_DIR} ${GRPC_ROOT_DIR}
        PATH_SUFFIXES tools/protobuf
        NO_DEFAULT_PATH)
    find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin
        PATHS ${GRPC_BIN_DIR} ${GRPC_ROOT_DIR}
        PATH_SUFFIXES tools/grpc
        NO_DEFAULT_PATH)
    if(NOT PROTOC_EXECUTABLE OR NOT GRPC_CPP_PLUGIN)
        message(FATAL_ERROR
            "protoc / grpc_cpp_plugin not found under ${GRPC_ROOT_DIR} (looked in "
            "bin/ and tools/). Install the grpc & protobuf vcpkg ports.")
    endif()
    # Decide the SDK flavor: a DLL stack ships grpc++.dll next to grpc++.lib;
    # a source-built (static) gRPC does not, and its .lib files are used as-is.
    if(EXISTS ${GRPC_BIN_DIR}/grpc++.dll)
        set(GRPC_DLL_STACK ON)
        message(STATUS "gRPC: DLL import-lib stack detected at ${GRPC_ROOT_DIR} "
                       "(regenerating import libs; ABI mismatch risk).")
    else()
        set(GRPC_DLL_STACK OFF)
        message(STATUS "gRPC: source-built (static) SDK detected at ${GRPC_ROOT_DIR}.")
    endif()
    message(STATUS "gRPC SDK found: includes=${GRPC_INCLUDE_DIR} "
                   "protoc=${PROTOC_EXECUTABLE} plugin=${GRPC_CPP_PLUGIN}")
else()
    message(STATUS "gRPC interface DISABLED (set -DINFERLITE_ENABLE_GRPC=ON and GRPC_ROOT to enable).")
endif()

# Generate gRPC/protobuf C++ from proto/grpc_service.proto when gRPC is on.
set(GRPC_GENERATED_DIR ${CMAKE_CURRENT_SOURCE_DIR}/generated)
if(INFERLITE_ENABLE_GRPC)
    set(GRPC_SRC ${CMAKE_CURRENT_SOURCE_DIR}/proto/grpc_service.proto)
    set(GRPC_GEN_SRCS
        ${GRPC_GENERATED_DIR}/grpc_service.pb.cc
        ${GRPC_GENERATED_DIR}/grpc_service.grpc.pb.cc
    )
    add_custom_command(
        OUTPUT ${GRPC_GEN_SRCS}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${GRPC_GENERATED_DIR}
        COMMAND ${PROTOC_EXECUTABLE} --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/proto
                --cpp_out=${GRPC_GENERATED_DIR}
                --grpc_out=${GRPC_GENERATED_DIR}
                --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
                ${GRPC_SRC}
        DEPENDS ${GRPC_SRC}
        VERBATIM
    )
endif()

add_executable(inferlite
    src/main.cpp
    src/infer_lite.cpp
    src/audit_log.cpp
    src/config_store.cpp
    src/diagnostics.cpp
    src/ensemble_executor.cpp
    src/http_server.cpp
    src/json.cpp
    src/memory_manager.cpp
    src/model_repository.cpp
    src/openvino_backend.cpp
    src/pbtxt.cpp
    src/plugin_backend.cpp
    src/scheduler.cpp
    src/sha256.cpp
    src/validation.cpp
    src/tensorrt_backend.cpp
    src/service_support.cpp
)

# --- Example CPU plugin shared library (for Phase 2 plugin backend demo) -----
add_library(sample_plugin SHARED tools/sample_plugin/sample_plugin.cpp)
target_include_directories(sample_plugin PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_include_directories(inferlite PRIVATE ${OV_INCLUDE_DIR})
target_link_libraries(inferlite PRIVATE ${OV_LIB_DIR}/openvino.lib)

# Windows service (SCM) support: advapi32 for OpenSCManager/CreateService/...
# and shell32 for CommandLineToArgvW. Harmless on non-Windows toolchains
# (the service source is guarded by _WIN32 and the libs are Windows-only).
if(WIN32)
    target_link_libraries(inferlite PRIVATE advapi32 shell32)
endif()

# --- Phase 5: gRPC interface when a gRPC SDK is available --------------------
if(INFERLITE_ENABLE_GRPC)
    target_compile_definitions(inferlite PRIVATE
        INFERLITE_ENABLE_GRPC=1
        # The Anaconda gRPC stack ships protobuf/absl as DLLs; the headers must
        # see the corresponding *_USE_DLLS macros or they select the static
        # library code path and fail to link against the DLL import libs.
        PROTOBUF_USE_DLLS
    )
    # Root is on the include path so `#include "generated/grpc_service.pb.h"`
    # (as referenced by grpc_server.hpp) resolves to ${GRPC_GENERATED_DIR}.
    target_include_directories(inferlite PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${GRPC_INCLUDE_DIR}
    )
    set_source_files_properties(${GRPC_GEN_SRCS} PROPERTIES GENERATED TRUE)
    target_sources(inferlite PRIVATE src/grpc_server.cpp ${GRPC_GEN_SRCS})

    if(GRPC_DLL_STACK)
        # The prebuilt DLL stack ships *DLL* import libraries (grpc++.lib,
        # grpc.lib, libprotobuf.lib, abseil_dll.lib, ...) that are stale
        # relative to the shipped DLLs: e.g. grpc++.lib omits grpc::Status::OK,
        # and libprotobuf.lib omits protobuf internal globals used by
        # protoc-generated code. Regenerate the import libs directly from the
        # DLLs (see scripts/gen_grpc_importlibs.ps1) and link those. The output
        # is deterministic and cached in third_party/grpc/importlibs/.
        set(GRPC_IMPLIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/grpc/importlibs)
        set(GRPC_IMPLIB_STAMP ${GRPC_IMPLIB_DIR}/.generated)
        add_custom_command(
            OUTPUT ${GRPC_IMPLIB_STAMP}
            COMMAND ${CMAKE_COMMAND} -E env
                    powershell -ExecutionPolicy Bypass -File
                    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/gen_grpc_importlibs.ps1
                    -GrpcRoot ${GRPC_ROOT_DIR}
            COMMENT "Regenerating gRPC DLL import libraries from DLLs"
            VERBATIM
        )
        set(_grpc_libs "")
        foreach(_l grpc++ grpc gpr libprotobuf abseil_dll re2 cares zlib)
            set(_reg ${GRPC_IMPLIB_DIR}/${_l}.lib)
            if(EXISTS ${_reg})
                list(APPEND _grpc_libs ${_reg})
            else()
                list(APPEND _grpc_libs ${GRPC_LIB_DIR}/${_l}.lib)
            endif()
        endforeach()
        add_custom_target(regenerate_grpc_implibs DEPENDS ${GRPC_IMPLIB_STAMP})
        add_dependencies(inferlite regenerate_grpc_implibs)
    else()
        # Source-built gRPC (vcpkg): vcpkg exports proper CMake targets whose
        # INTERFACE_LINK_LIBRARIES carry the full transitive dependency set
        # (gRPC++ wrapper, gRPC core, upb, address_sorting, re2, zlib, c-ares,
        # OpenSSL, and every needed Abseil component). Prefer find_package over
        # hand-listing the static libs, which is fragile (vcpkg builds the gRPC
        # core statically while Abseil/protobuf/re2/cares/zlib/OpenSSL are DLLs,
        # so the correct link line mixes static libs and DLL import libs).
        # CMP0144: our project variable GRPC_ROOT is unrelated to the package
        # var gRPC_ROOT that find_package would consult; we use CMAKE_PREFIX_PATH.
        if(POLICY CMP0144)
            cmake_policy(SET CMP0144 NEW)
        endif()
        list(APPEND CMAKE_PREFIX_PATH ${GRPC_ROOT_DIR})
        find_package(gRPC CONFIG REQUIRED)
        set(_grpc_libs gRPC::grpc++)
    endif()
    # Link order: gRPC++ wrapper first, then core, then transitive deps.
    target_link_libraries(inferlite PRIVATE ${_grpc_libs})
    # The gRPC/protobuf runtime DLLs (plus transitive deps) must sit next to
    # inferlite.exe so it runs without setting PATH. For the source-built
    # (static gRPC++) flavor only the protobuf/absl/re2/cares/zlib DLLs exist;
    # for the DLL stack the gRPC core DLLs are present too. Globbing only what
    # actually exists keeps this correct for both layouts.
    file(GLOB GRPC_RUNTIME_DLLS
        ${GRPC_BIN_DIR}/grpc++.dll
        ${GRPC_BIN_DIR}/grpc.dll
        ${GRPC_BIN_DIR}/grpc++_unsecure.dll
        ${GRPC_BIN_DIR}/grpc_unsecure.dll
        ${GRPC_BIN_DIR}/libprotobuf.dll
        ${GRPC_BIN_DIR}/libprotobuf-lite.dll
        ${GRPC_BIN_DIR}/abseil_dll.dll
        ${GRPC_BIN_DIR}/re2.dll
        ${GRPC_BIN_DIR}/cares.dll
        ${GRPC_BIN_DIR}/zlib.dll
        ${GRPC_BIN_DIR}/zlib1.dll
        ${GRPC_BIN_DIR}/zlib-ng2.dll
        ${GRPC_BIN_DIR}/pcre2-8.dll
        ${GRPC_BIN_DIR}/libssl-3-x64.dll
        ${GRPC_BIN_DIR}/libcrypto-3-x64.dll
        ${GRPC_BIN_DIR}/pthreadVC3.dll
        ${GRPC_BIN_DIR}/pthreadVCE3.dll
        ${GRPC_BIN_DIR}/pthreadVSE3.dll
    )
    add_custom_command(TARGET inferlite POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${GRPC_RUNTIME_DLLS}
            $<TARGET_FILE_DIR:inferlite>
    )
    message(STATUS "gRPC enabled: includes=${GRPC_INCLUDE_DIR} libs=${GRPC_LIB_DIR}")
endif()

# --- Phase 3: TensorRT / CUDA when GPU is enabled ---------------------------
if(INFERLITE_ENABLE_GPU)
    target_compile_definitions(inferlite PRIVATE INFERLITE_ENABLE_GPU=1)
    target_include_directories(inferlite PRIVATE ${TRT_ROOT}/include)
    # GPU memory manager (CUDA pinned + device buffers) is compiled only when
    # the GPU backend is enabled; tensorrt_backend.cpp references its symbols.
    target_sources(inferlite PRIVATE src/gpu_memory_manager.cpp)
    # CUDA toolkit (find the CUDA runtime for linking).
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        target_include_directories(inferlite PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(inferlite PRIVATE CUDA::cudart)
    else()
        message(WARNING "CUDA toolkit not found; using default CUDA include path.")
        target_include_directories(inferlite PRIVATE
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include")
        target_link_libraries(inferlite PRIVATE
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/lib/x64/cudart.lib")
    endif()
    # TensorRT libraries. TensorRT 10+ uses versioned import libraries
    # (nvinfer_10.lib, nvinfer_plugin_10.lib); older releases use nvinfer.lib.
    set(TRT_LIB_DIR ${TRT_ROOT}/lib)
    set(TRT_NAMES nvinfer_10 nvinfer)
    set(TRT_PLUGIN_NAMES nvinfer_plugin_10 nvinfer_plugin)
    find_library(TRT_NVINFER NAMES ${TRT_NAMES} PATHS ${TRT_LIB_DIR} NO_DEFAULT_PATH)
    find_library(TRT_NVINFER_PLUGIN NAMES ${TRT_PLUGIN_NAMES} PATHS ${TRT_LIB_DIR} NO_DEFAULT_PATH)
    if(NOT TRT_NVINFER)
        find_library(TRT_NVINFER NAMES ${TRT_NAMES}
            PATHS ${TRT_ROOT}/lib ${TRT_ROOT}/targets/x86_64-linux-gnu/lib)
    endif()
    if(TRT_NVINFER)
        target_link_libraries(inferlite PRIVATE ${TRT_NVINFER})
        if(TRT_NVINFER_PLUGIN)
            target_link_libraries(inferlite PRIVATE ${TRT_NVINFER_PLUGIN})
        endif()
    else()
        # fallback: rely on the import library next to the headers
        file(GLOB TRT_LIBS ${TRT_LIB_DIR}/*.lib)
        target_link_libraries(inferlite PRIVATE ${TRT_LIBS})
    endif()
    message(STATUS "GPU enabled: includes=${TRT_ROOT}/include libs=${TRT_LIB_DIR}")
    # Copy the TensorRT runtime DLLs next to the executable so it runs without
    # needing TensorRT's bin/ on PATH. TensorRT ships its runtime DLLs under
    # bin/ (e.g. nvinfer_10.dll, nvinfer_plugin_10.dll). cudart is usually on
    # the system PATH; we also copy it when it can be located.
    file(GLOB TRT_RUNTIME_DLLS ${TRT_ROOT}/bin/nvinfer*.dll ${TRT_ROOT}/bin/nvonnxparser*.dll)
    find_file(CUDA_CUDART_DLL
        NAMES cudart64_12.dll cudart64_*.dll
        PATHS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"
        PATH_SUFFIXES bin
        NO_DEFAULT_PATH)
    if(CUDA_CUDART_DLL)
        list(APPEND TRT_RUNTIME_DLLS ${CUDA_CUDART_DLL})
    endif()
    add_custom_command(TARGET inferlite POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TRT_RUNTIME_DLLS}
            $<TARGET_FILE_DIR:inferlite>
        COMMAND_EXPAND_LISTS
    )
endif()

# Copy OpenVINO DLLs next to the executable so it can run without setting PATH.
# This includes the runtime plugins plus third-party libs (TBB, etc.).
file(GLOB OV_ALL_DLLS ${OV_BIN_DIR}/*.dll)
file(GLOB OV_3RDPARTY_DLLS
    ${OV_INSTALL}/runtime/3rdparty/*/bin/*.dll
)
add_custom_command(TARGET inferlite POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${OV_ALL_DLLS} ${OV_3RDPARTY_DLLS}
        $<TARGET_FILE_DIR:inferlite>
)
