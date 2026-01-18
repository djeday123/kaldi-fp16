#include "kaldi-fp16/tensor-ops.h"
#include <stdexcept>
#include <cuda_runtime.h>

namespace kaldi {
namespace fp16 {

// CUDA error checking
#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      throw std::runtime_error(std::string("CUDA error: ") + \
                               cudaGetErrorString(err)); \
    } \
  } while(0)

#define CUBLAS_CHECK(call) \
  do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
      throw std::runtime_error("cuBLAS error"); \
    } \
  } while(0)

void InitTensorCoreOps() {
  // Initialize CUDA runtime by calling a lightweight CUDA API
  // cudaFree(0) is a common idiom to trigger runtime initialization
  // without actually freeing any memory (null pointer is safe)
  cudaFree(0);
  
  // Query device properties
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  
  // Check for Tensor Core support (compute capability >= 7.0)
  if (prop.major < 7) {
    throw std::runtime_error(
        "Tensor Cores require compute capability 7.0 or higher. "
        "Current device: " + std::string(prop.name) + 
        " (compute " + std::to_string(prop.major) + "." + 
        std::to_string(prop.minor) + ")");
  }
}

bool TensorCoresAvailable() {
  int device;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) {
    return false;
  }
  
  cudaDeviceProp prop;
  err = cudaGetDeviceProperties(&prop, device);
  if (err != cudaSuccess) {
    return false;
  }
  
  // Tensor Cores available on compute capability >= 7.0
  // 7.0: Volta (V100)
  // 7.5: Turing (RTX 20xx)
  // 8.0: Ampere (A100, RTX 30xx)
  // 8.6: Ampere (RTX 3060/3050)
  // 8.9: Ada Lovelace (RTX 40xx)
  // 9.0: Hopper (H100)
  return prop.major >= 7;
}

int GetTensorCoreComputeCapability() {
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  
  return prop.major * 10 + prop.minor;
}

void TensorCoreGemm(cublasHandle_t handle,
                   int m, int n, int k,
                   const float* alpha,
                   const half* A, int lda,
                   const half* B, int ldb,
                   const float* beta,
                   half* C, int ldc,
                   half* D, int ldd) {
  // Use cublasGemmEx for Tensor Core acceleration
  // This uses FP16 input/output with FP32 accumulation for best accuracy
  CUBLAS_CHECK(cublasGemmEx(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      m, n, k,
      alpha,
      A, CUDA_R_16F, lda,
      B, CUDA_R_16F, ldb,
      beta,
      C, CUDA_R_16F, ldc,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  
  // Copy result to D if different from C
  if (D != C) {
    // Use cudaMemcpy2D for efficient strided copy
    CUDA_CHECK(cudaMemcpy2D(
        D, ldd * sizeof(half),  // destination and pitch
        C, ldc * sizeof(half),  // source and pitch
        m * sizeof(half),       // width in bytes (number of elements in a row)
        n,                      // height (number of rows)
        cudaMemcpyDeviceToDevice));
  }
}

void EnableTensorCoreMath(cublasHandle_t handle) {
#ifdef ENABLE_TENSOR_CORES
  // Set math mode to allow Tensor Core operations
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
#else
  // Use default math mode (CUDA cores only)
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
#endif
}

void DisableTensorCoreMath(cublasHandle_t handle) {
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
}

float BenchmarkMatMul(int m, int n, int k, 
                     bool use_tensor_cores,
                     int num_iterations) {
  // Create cuBLAS handle
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  
  if (use_tensor_cores) {
    EnableTensorCoreMath(handle);
  } else {
    DisableTensorCoreMath(handle);
  }
  
  // Allocate matrices
  half *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, m * k * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_B, k * n * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_C, m * n * sizeof(half)));
  
  // Initialize with dummy values
  CUDA_CHECK(cudaMemset(d_A, 0, m * k * sizeof(half)));
  CUDA_CHECK(cudaMemset(d_B, 0, k * n * sizeof(half)));
  CUDA_CHECK(cudaMemset(d_C, 0, m * n * sizeof(half)));
  
  float alpha = 1.0f, beta = 0.0f;
  
  // Warm-up
  for (int i = 0; i < 10; ++i) {
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,
        &alpha,
        d_A, CUDA_R_16F, m,
        d_B, CUDA_R_16F, k,
        &beta,
        d_C, CUDA_R_16F, m,
        CUBLAS_COMPUTE_32F,
        use_tensor_cores ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  
  // Benchmark
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < num_iterations; ++i) {
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,
        &alpha,
        d_A, CUDA_R_16F, m,
        d_B, CUDA_R_16F, k,
        &beta,
        d_C, CUDA_R_16F, m,
        CUBLAS_COMPUTE_32F,
        use_tensor_cores ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  
  float total_time_ms;
  CUDA_CHECK(cudaEventElapsedTime(&total_time_ms, start, stop));
  float avg_time_ms = total_time_ms / num_iterations;
  
  // Cleanup
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUBLAS_CHECK(cublasDestroy(handle));
  
  return avg_time_ms;
}

float GetTFLOPS(int m, int n, int k, float time_ms) {
  // Matrix multiplication performs 2*m*n*k FLOPs
  double flops = 2.0 * m * n * k;
  // Convert to TFLOPS
  double tflops = (flops / (time_ms / 1000.0)) / 1e12;
  return static_cast<float>(tflops);
}

}  // namespace fp16
}  // namespace kaldi
