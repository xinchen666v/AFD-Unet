cmake_minimum_required(VERSION 3.16)
project(afd_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---- ONNX Runtime ----
set(ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-win-x64-1.31.1")
if(NOT EXISTS "${ORT_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR
        "未找到 ONNX Runtime，请把官方包解压到: ${ORT_ROOT}\n"
        "下载: https://github.com/microsoft/onnxruntime/releases")
endif()

add_executable(afd_demo src/main.cpp src/afd_inference.cpp)
target_include_directories(afd_demo PRIVATE include "${ORT_ROOT}/include")

if(MSVC)
    # MSVC：直接链官方 .lib
    target_link_libraries(afd_demo PRIVATE "${ORT_ROOT}/lib/onnxruntime.lib")
    target_compile_options(afd_demo PRIVATE /O2)
else()
    # MinGW：可直接链接 ORT 的 MSVC dll（C ABI 兼容）
    target_link_libraries(afd_demo PRIVATE "${ORT_ROOT}/lib/onnxruntime.dll")
    # 静态链接 GCC 运行时，避免依赖 libstdc++-6.dll / libgcc_s_seh-1.dll
    target_link_options(afd_demo PRIVATE -static-libgcc -static-libstdc++)
    target_compile_options(afd_demo PRIVATE -O2)
endif()

# 运行时 dll 必须与 exe 同目录
add_custom_command(TARGET afd_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ORT_ROOT}/lib/onnxruntime.dll"
        "$<TARGET_FILE_DIR:afd_demo>/"
    COMMENT "Copy onnxruntime.dll to output dir"
)
