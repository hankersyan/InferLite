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
)

# --- Example CPU plugin shared library (for Phase 2 plugin backend demo) -----
add_library(sample_plugin SHARED tools/sample_plugin/sample_plugin.cpp)
target_include_directories(sample_plugin PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_include_directories(inferlite PRIVATE ${OV_INCLUDE_DIR})
target_link_libraries(inferlite PRIVATE ${OV_LIB_DIR}/openvino.lib)

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
