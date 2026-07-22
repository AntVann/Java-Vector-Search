cmake_minimum_required(VERSION 3.20)

project(vectorforge_jni LANGUAGES CXX)

option(VECTORFORGE_ENABLE_CUDA "Enable the experimental CUDA backend" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED OUTPUT_DIR)
    set(OUTPUT_DIR "${CMAKE_BINARY_DIR}/native-lib")
endif()

if(DEFINED JAVA_HOME)
    set(ENV{JAVA_HOME} "${JAVA_HOME}")
endif()

find_package(JNI REQUIRED)

set(VECTORFORGE_NATIVE_SOURCES
    src/main/cpp/native_bruteforce_jni.cpp
    src/main/cpp/native_cpu_index.cpp
    src/main/cpp/native_cuda_index.cpp
)

add_library(vectorforge_jni SHARED ${VECTORFORGE_NATIVE_SOURCES})

target_include_directories(vectorforge_jni PRIVATE
    ${JNI_INCLUDE_DIRS}
)

target_compile_definitions(vectorforge_jni PRIVATE
    NOMINMAX
)

if(VECTORFORGE_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_include_directories(vectorforge_jni PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(vectorforge_jni PRIVATE VECTORFORGE_ENABLE_CUDA=1)
endif()

if(MINGW)
    target_link_options(vectorforge_jni PRIVATE
        -static-libgcc
        -static-libstdc++
    )
endif()

set_target_properties(vectorforge_jni PROPERTIES
    OUTPUT_NAME "vectorforge_jni"
    PREFIX ""
    ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
)
