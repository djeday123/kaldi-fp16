#ifndef KALDI_FP16_TENSOR_OPS_H_
#define KALDI_FP16_TENSOR_OPS_H_

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

namespace kaldi {
namespace fp16 {

/// @file tensor-ops.h
/// @brief Tensor Core optimized operations for FP16 matrices
///
/// This file contains functions that specifically leverage NVIDIA Tensor Cores
/// for accelerated matrix operations. These operations are optimized for
/// Volta, Turing, Ampere, Ada Lovelace, and Hopper GPU architectures.

/// Initialize Tensor Core operations
/// Should be called once at program startup
void InitTensorCoreOps();

/// Check if Tensor Cores are available on current GPU
/// @return true if Tensor Cores are supported
bool TensorCoresAvailable();

/// Get Tensor Core compute capability
/// @return Compute capability version (e.g., 70 for Volta, 80 for Ampere)
int GetTensorCoreComputeCapability();

/// Tensor Core optimized matrix multiplication
/// Performs: D = alpha * (A @ B) + beta * C
/// @param handle cuBLAS handle
/// @param m Number of rows in A and C
/// @param n Number of columns in B and C
/// @param k Number of columns in A and rows in B
/// @param alpha Scalar multiplier for A*B
/// @param A Device pointer to matrix A (m x k)
/// @param lda Leading dimension of A
/// @param B Device pointer to matrix B (k x n)
/// @param ldb Leading dimension of B
/// @param beta Scalar multiplier for C
/// @param C Device pointer to matrix C (m x n)
/// @param ldc Leading dimension of C
/// @param D Device pointer to matrix D (m x n), can be same as C
/// @param ldd Leading dimension of D
void TensorCoreGemm(cublasHandle_t handle,
                   int m, int n, int k,
                   const float* alpha,
                   const half* A, int lda,
                   const half* B, int ldb,
                   const float* beta,
                   half* C, int ldc,
                   half* D, int ldd);

/// Set cuBLAS to use Tensor Core math mode
/// @param handle cuBLAS handle
void EnableTensorCoreMath(cublasHandle_t handle);

/// Disable Tensor Core math mode (use CUDA cores only)
/// @param handle cuBLAS handle
void DisableTensorCoreMath(cublasHandle_t handle);

/// Performance benchmarking utilities

/// Benchmark matrix multiplication performance
/// @param m Matrix dimension M
/// @param n Matrix dimension N
/// @param k Matrix dimension K
/// @param use_tensor_cores Whether to use Tensor Cores
/// @param num_iterations Number of iterations for benchmarking
/// @return Average time in milliseconds
float BenchmarkMatMul(int m, int n, int k, 
                     bool use_tensor_cores,
                     int num_iterations = 100);

/// Calculate TFLOPS from matrix multiplication parameters
/// @param m Matrix dimension M
/// @param n Matrix dimension N
/// @param k Matrix dimension K
/// @param time_ms Time in milliseconds
/// @return Performance in TFLOPS (trillion floating-point operations per second) as a float
float GetTFLOPS(int m, int n, int k, float time_ms);

}  // namespace fp16
}  // namespace kaldi

#endif  // KALDI_FP16_TENSOR_OPS_H_
