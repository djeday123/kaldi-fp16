cmake_minimum_required(VERSION 3.18)
project(KaldiFP16 VERSION 1.0.0 LANGUAGES CXX CUDA)

# C++ Standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build options
option(ENABLE_TENSOR_CORES "Enable Tensor Core operations" ON)
option(ENABLE_FP16 "Enable FP16 support" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build unit tests" ON)

# CUDA Architecture - Auto-detect or use provided
if(NOT DEFINED CUDA_ARCH)
    set(CUDA_ARCH "70;75;80;86;89;90" CACHE STRING "CUDA architectures to build for")
endif()

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

# CUDA compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0")

# Enable Tensor Core specific optimizations
if(ENABLE_TENSOR_CORES)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DENABLE_TENSOR_CORES")
    # Use tensor core operations in cuBLAS
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCUBLAS_TENSOR_OP_MATH")
endif()

# Enable FP16 support
if(ENABLE_FP16)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DENABLE_FP16")
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Source files
file(GLOB_RECURSE KALDI_FP16_SOURCES
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/*.cu"
)

# Create library
add_library(kaldi-fp16 SHARED ${KALDI_FP16_SOURCES})

# Link libraries
target_link_libraries(kaldi-fp16
    CUDA::cudart
    CUDA::cublas
)

# Set library properties
set_target_properties(kaldi-fp16 PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER "include/kaldi-fp16/matrix-fp16.h;include/kaldi-fp16/tensor-ops.h"
)

# Install targets
install(TARGETS kaldi-fp16
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include/kaldi-fp16
)

# Build examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Build tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "Kaldi-FP16 Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Tensor Cores: ${ENABLE_TENSOR_CORES}")
message(STATUS "  FP16 Support: ${ENABLE_FP16}")
message(STATUS "  Build Examples: ${BUILD_EXAMPLES}")
message(STATUS "  Build Tests: ${BUILD_TESTS}")
message(STATUS "")
