cmake_minimum_required(VERSION 3.16)
project(flow_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

# ═══════════════════════════════════════════════════════════════
# 第三方库路径 (相对于项目根目录)
# ═══════════════════════════════════════════════════════════════
# 自动检测: 从 CMakeLists.txt 所在目录向上搜索 third_party/
if(NOT THIRD_PARTY_ROOT)
    set(_search "${CMAKE_CURRENT_SOURCE_DIR}")
    foreach(_i RANGE 1 6)
        if(EXISTS "${_search}/third_party")
            set(THIRD_PARTY_ROOT "${_search}/third_party")
            break()
        endif()
        get_filename_component(_search "${_search}" DIRECTORY)
    endforeach()
endif()
if(NOT THIRD_PARTY_ROOT)
    set(THIRD_PARTY_ROOT "" CACHE PATH "Root of third_party/ directory (auto-detection failed)")
    message(WARNING "Could not auto-detect third_party/. "
                    "Set -DTHIRD_PARTY_ROOT=/absolute/path/to/third_party")
endif()

# ── OpenCV ────────────────────────────────────────────────────
# 优先使用显式路径 (third_party/opencv/windows), 回退到 find_package
set(OpenCV_HINT "${THIRD_PARTY_ROOT}/opencv/windows")
if(EXISTS "${OpenCV_HINT}/include")
    message(STATUS "OpenCV: using third_party path ${OpenCV_HINT}")
    set(OpenCV_INCLUDE_DIRS "${OpenCV_HINT}/include")
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(OpenCV_LIB_DIR "${OpenCV_HINT}/lib")
    else()
        set(OpenCV_LIB_DIR "${OpenCV_HINT}/lib")
    endif()
    file(GLOB OpenCV_LIBS_RELEASE "${OpenCV_LIB_DIR}/opencv_world[0-9]*.lib")
    file(GLOB OpenCV_LIBS_DEBUG   "${OpenCV_LIB_DIR}/opencv_world[0-9]*d.lib")
    # 排除 debug lib 出现在 release 列表中
    list(FILTER OpenCV_LIBS_RELEASE EXCLUDE REGEX ".*d\\.lib$")
    if(OpenCV_LIBS_RELEASE)
        list(GET OpenCV_LIBS_RELEASE 0 OpenCV_LIB_RELEASE)
        message(STATUS "  release lib: ${OpenCV_LIB_RELEASE}")
    endif()
    if(OpenCV_LIBS_DEBUG)
        list(GET OpenCV_LIBS_DEBUG 0 OpenCV_LIB_DEBUG)
        message(STATUS "  debug lib:   ${OpenCV_LIB_DEBUG}")
    else()
        set(OpenCV_LIB_DEBUG "${OpenCV_LIB_RELEASE}")
    endif()
    set(OpenCV_LIBS optimized "${OpenCV_LIB_RELEASE}" debug "${OpenCV_LIB_DEBUG}")
    set(OpenCV_FOUND TRUE)
else()
    find_package(OpenCV REQUIRED)
endif()

# ── ONNX Runtime ──────────────────────────────────────────────
set(ONNXRUNTIME_ROOT "${THIRD_PARTY_ROOT}/onnxruntime/windows"
    CACHE PATH "Path to ONNX Runtime SDK")
if(EXISTS "${ONNXRUNTIME_ROOT}/include")
    message(STATUS "ONNX Runtime: ${ONNXRUNTIME_ROOT}")
    include_directories("${ONNXRUNTIME_ROOT}/include")
    link_directories("${ONNXRUNTIME_ROOT}/lib")
else()
    message(WARNING "ONNX Runtime not found at ${ONNXRUNTIME_ROOT}")
    message(WARNING "  Expected structure: third_party/onnxruntime/windows/{include,lib,bin}/")
endif()
find_library(ONNXRUNTIME_LIB onnxruntime HINTS "${ONNXRUNTIME_ROOT}/lib")
if(NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "onnxruntime.lib not found. Check ONNXRUNTIME_ROOT path.")
endif()

add_executable(flow_inference
    main.cpp
    ParticleInferencer.cpp
    FlowDynamics.cpp
    TiledParticleInferencer.cpp
    ParticleInferencer.h
    FlowDynamics.h
    TiledParticleInferencer.h
    Particle.h
    ParticleIdManager.h
)

target_include_directories(flow_inference PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(flow_inference PRIVATE "${THIRD_PARTY_ROOT}/nlohmann")
target_link_libraries(flow_inference ${OpenCV_LIBS} ${ONNXRUNTIME_LIB})

# ── FlowDynamics 对齐测试 (P0-2) ──
add_executable(flow_dynamics_align
    flow_dynamics_align.cpp
    FlowDynamics.cpp
    FlowDynamics.h
    Particle.h
    ParticleIdManager.h
)
target_include_directories(flow_dynamics_align PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(flow_dynamics_align ${OpenCV_LIBS})

# ── 运行时 DLL 部署 (P0-F3: 复制到输出目录，避免 0xC0000135) ──
set(ONNX_BIN_DIR "${ONNXRUNTIME_ROOT}/bin")
set(OpenCV_BIN_DIR "${OpenCV_HINT}/bin")

# ONNX Runtime DLLs
if(EXISTS "${ONNX_BIN_DIR}")
    file(GLOB ONNX_DLLS "${ONNX_BIN_DIR}/*.dll")
    foreach(_dll ${ONNX_DLLS})
        get_filename_component(_dll_name "${_dll}" NAME)
        add_custom_command(TARGET flow_inference POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_dll}"
                "$<TARGET_FILE_DIR:flow_inference>"
            COMMENT "Copying ONNX Runtime DLL: ${_dll_name}"
        )
    endforeach()
else()
    message(WARNING "ONNX Runtime bin/ not found at ${ONNX_BIN_DIR}. "
                    "flow_inference.exe will fail to start without onnxruntime.dll.")
endif()

# OpenCV DLLs (for flow_inference)
if(EXISTS "${OpenCV_BIN_DIR}")
    file(GLOB OpenCV_DLLS "${OpenCV_BIN_DIR}/*.dll")
    foreach(_dll ${OpenCV_DLLS})
        get_filename_component(_dll_name "${_dll}" NAME)
        add_custom_command(TARGET flow_inference POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_dll}"
                "$<TARGET_FILE_DIR:flow_inference>"
            COMMENT "Copying OpenCV DLL (inference): ${_dll_name}"
        )
        # flow_dynamics_align 也需要 OpenCV DLL
        add_custom_command(TARGET flow_dynamics_align POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_dll}"
                "$<TARGET_FILE_DIR:flow_dynamics_align>"
            COMMENT "Copying OpenCV DLL (align): ${_dll_name}"
        )
    endforeach()
else()
    message(WARNING "OpenCV bin/ not found at ${OpenCV_BIN_DIR}. "
                    "Executables will fail to start without opencv_world*.dll.")
endif()
