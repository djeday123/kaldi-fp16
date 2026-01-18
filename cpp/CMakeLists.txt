cmake_minimum_required(VERSION 3.18)
project(kaldi_fp16 LANGUAGES CXX CUDA)

# ==============================================================================
# Build options
# ==============================================================================

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)

# CUDA architecture - default to Ampere (RTX 30xx, A100)
set(CUDA_ARCH "86" CACHE STRING "CUDA architecture (e.g., 70 for V100, 80 for A100, 86 for RTX 30xx)")

# ==============================================================================
# C++ settings
# ==============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Position independent code for shared library
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimization
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG")

# ==============================================================================
# CUDA settings
# ==============================================================================

find_package(CUDAToolkit REQUIRED)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g -G -DDEBUG")

# Enable separable compilation for device linking
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# ==============================================================================
# Dependencies
# ==============================================================================

# cuBLAS (for FP16 GEMM with Tensor Cores)
find_library(CUBLAS_LIBRARY cublas
    HINTS ${CUDAToolkit_LIBRARY_DIR}
    REQUIRED
)

# cuDNN (optional, for more ops)
find_library(CUDNN_LIBRARY cudnn
    HINTS ${CUDAToolkit_LIBRARY_DIR}
)

# ==============================================================================
# Main library
# ==============================================================================

add_library(kaldi_fp16
    src/tensor_fp16.cpp
    cuda/kernels.cu
)

target_include_directories(kaldi_fp16 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(kaldi_fp16 PUBLIC
    CUDA::cudart
    CUDA::cublas
)

if(CUDNN_LIBRARY)
    target_compile_definitions(kaldi_fp16 PRIVATE HAVE_CUDNN)
    target_link_libraries(kaldi_fp16 PUBLIC ${CUDNN_LIBRARY})
endif()

# ==============================================================================
# C interface for CGO
# ==============================================================================

add_library(kaldi_fp16_cgo SHARED
    src/cgo_interface.cpp
)

target_link_libraries(kaldi_fp16_cgo PRIVATE kaldi_fp16)

# ==============================================================================
# Tests
# ==============================================================================

if(BUILD_TESTS)
    enable_testing()
    
    add_executable(test_tensor_fp16 tests/test_tensor.cpp)
    target_link_libraries(test_tensor_fp16 PRIVATE kaldi_fp16)
    add_test(NAME TensorFP16Test COMMAND test_tensor_fp16)
    
    add_executable(test_gemm tests/test_gemm.cu)
    target_link_libraries(test_gemm PRIVATE kaldi_fp16)
    add_test(NAME GEMMTest COMMAND test_gemm)
endif()

# ==============================================================================
# Examples
# ==============================================================================

if(BUILD_EXAMPLES)
    add_executable(example_matmul examples/matmul_benchmark.cu)
    target_link_libraries(example_matmul PRIVATE kaldi_fp16)
    
    add_executable(example_dnn examples/simple_dnn.cpp)
    target_link_libraries(example_dnn PRIVATE kaldi_fp16)
endif()

# ==============================================================================
# Installation
# ==============================================================================

install(TARGETS kaldi_fp16 kaldi_fp16_cgo
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include/kaldi_fp16
)

# ==============================================================================
# Info
# ==============================================================================

message(STATUS "")
message(STATUS "Kaldi-FP16 Configuration:")
message(STATUS "  CUDA Architecture: ${CUDA_ARCH}")
message(STATUS "  CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "  cuBLAS: ${CUBLAS_LIBRARY}")
message(STATUS "  cuDNN: ${CUDNN_LIBRARY}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
