cmake_minimum_required(VERSION 3.20)

project(vectorforge_jni LANGUAGES CXX)

option(VECTORFORGE_ENABLE_CUDA "Enable the experimental CUDA backend" OFF)
option(VECTORFORGE_ENABLE_CUVS "Enable the optional cuVS backend" OFF)

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
    src/main/cpp/native_cuvs_index.cpp
)

add_library(vectorforge_jni SHARED ${VECTORFORGE_NATIVE_SOURCES})

target_include_directories(vectorforge_jni PRIVATE
    ${JNI_INCLUDE_DIRS}
)

target_compile_definitions(vectorforge_jni PRIVATE
    NOMINMAX
)

target_link_libraries(vectorforge_jni PRIVATE
    ${CMAKE_DL_LIBS}
)

if(VECTORFORGE_ENABLE_CUDA OR VECTORFORGE_ENABLE_CUVS)
    find_package(CUDAToolkit REQUIRED)
endif()

if(VECTORFORGE_ENABLE_CUDA)
    target_include_directories(vectorforge_jni PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(vectorforge_jni PRIVATE VECTORFORGE_ENABLE_CUDA=1)
endif()

if(VECTORFORGE_ENABLE_CUVS)
    find_package(cuvs CONFIG REQUIRED COMPONENTS c_api)
    target_compile_definitions(vectorforge_jni PRIVATE VECTORFORGE_ENABLE_CUVS=1)
    target_link_libraries(vectorforge_jni PRIVATE
        cuvs::c_api
        CUDA::cudart
    )
endif()

if(MINGW)
    target_link_options(vectorforge_jni PRIVATE
        -static-libgcc
        -static-libstdc++
    )
endif()

set_target_properties(vectorforge_jni PROPERTIES
    OUTPUT_NAME "vectorforge_jni"
    ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIR}"
)

if(WIN32)
    set_target_properties(vectorforge_jni PROPERTIES PREFIX "")
endif()
