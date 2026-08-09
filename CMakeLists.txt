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
    src/gpu_memory_manager.cpp
)

# --- Example CPU plugin shared library (for Phase 2 plugin backend demo) -----
add_library(sample_plugin SHARED tools/sample_plugin/sample_plugin.cpp)
target_include_directories(sample_plugin PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_include_directories(inferlite PRIVATE ${OV_INCLUDE_DIR})
target_link_libraries(inferlite PRIVATE ${OV_LIB_DIR}/openvino.lib)

# --- Phase 3: TensorRT / CUDA when GPU is enabled ---------------------------
if(INFERLITE_ENABLE_GPU)
    target_compile_definitions(inferlite PRIVATE INFERLITE_ENABLE_GPU=1)
    target_include_directories(inferlite PRIVATE ${TRT_ROOT}/include)
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
