#include "kaldi-fp16/matrix-fp16.h"
#include "kaldi-fp16/tensor-ops.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace kaldi::fp16;

// Simple FP32 matrix multiplication for reference
void MatMulFP32(const std::vector<float>& A, 
                const std::vector<float>& B,
                std::vector<float>& C,
                int M, int K, int N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

float ComputeRelativeError(const std::vector<float>& reference,
                          const std::vector<float>& result) {
  float max_diff = 0.0f;
  float max_ref = 0.0f;
  
  for (size_t i = 0; i < reference.size(); ++i) {
    float diff = std::abs(reference[i] - result[i]);
    max_diff = std::max(max_diff, diff);
    max_ref = std::max(max_ref, std::abs(reference[i]));
  }
  
  return max_diff / (max_ref + 1e-6f);
}

int main() {
  std::cout << "=== Kaldi-FP16 Precision Comparison ===" << std::endl;
  std::cout << std::endl;
  
  if (!TensorCoresAvailable()) {
    std::cout << "Tensor Cores not available on this device" << std::endl;
    return 1;
  }
  
  // Use small matrices for CPU comparison
  const int M = 64;
  const int K = 64;
  const int N = 64;
  
  std::cout << "Comparing FP32 (CPU) vs FP16 Tensor Cores (GPU)" << std::endl;
  std::cout << "Matrix dimensions: " << M << " x " << K << " x " << N << std::endl;
  std::cout << std::endl;
  
  try {
    // Generate random input matrices
    std::vector<float> A_host(M * K);
    std::vector<float> B_host(K * N);
    std::vector<float> C_fp32(M * N);
    std::vector<float> C_fp16(M * N);
    
    // Initialize with small random values
    for (int i = 0; i < M * K; ++i) {
      A_host[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }
    for (int i = 0; i < K * N; ++i) {
      B_host[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }
    
    // Compute reference result with FP32 on CPU
    std::cout << "Computing FP32 reference on CPU..." << std::endl;
    MatMulFP32(A_host, B_host, C_fp32, M, K, N);
    std::cout << "✓ FP32 computation complete" << std::endl;
    std::cout << std::endl;
    
    // Compute with FP16 on GPU
    std::cout << "Computing FP16 result on GPU with Tensor Cores..." << std::endl;
    MatrixFP16 A_gpu(M, K);
    MatrixFP16 B_gpu(K, N);
    MatrixFP16 C_gpu(M, N);
    
    A_gpu.CopyFromHost(A_host.data());
    B_gpu.CopyFromHost(B_host.data());
    C_gpu.SetZero();
    
    C_gpu.AddMatMat(1.0f, A_gpu, kNoTrans, B_gpu, kNoTrans, 0.0f);
    
    C_gpu.CopyToHost(C_fp16.data());
    std::cout << "✓ FP16 computation complete" << std::endl;
    std::cout << std::endl;
    
    // Compare results
    float rel_error = ComputeRelativeError(C_fp32, C_fp16);
    
    std::cout << "Results:" << std::endl;
    std::cout << "  Relative Error: " << std::scientific << std::setprecision(6) 
              << rel_error << std::endl;
    std::cout << std::endl;
    
    // Print sample values
    std::cout << "Sample values (first 4x4 block):" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    
    std::cout << "FP32 (CPU):" << std::endl;
    for (int i = 0; i < std::min(4, M); ++i) {
      std::cout << "  ";
      for (int j = 0; j < std::min(4, N); ++j) {
        std::cout << std::setw(12) << C_fp32[i * N + j] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "FP16 (GPU Tensor Cores):" << std::endl;
    for (int i = 0; i < std::min(4, M); ++i) {
      std::cout << "  ";
      for (int j = 0; j < std::min(4, N); ++j) {
        std::cout << std::setw(12) << C_fp16[i * N + j] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "Difference:" << std::endl;
    for (int i = 0; i < std::min(4, M); ++i) {
      std::cout << "  ";
      for (int j = 0; j < std::min(4, N); ++j) {
        float diff = C_fp32[i * N + j] - C_fp16[i * N + j];
        std::cout << std::setw(12) << diff << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "Analysis:" << std::endl;
    if (rel_error < 1e-2) {
      std::cout << "  ✓ Excellent agreement between FP32 and FP16" << std::endl;
      std::cout << "    FP16 with Tensor Cores provides accurate results" << std::endl;
      std::cout << "    with significant performance benefits" << std::endl;
    } else if (rel_error < 1e-1) {
      std::cout << "  ✓ Good agreement between FP32 and FP16" << std::endl;
      std::cout << "    Small precision loss acceptable for most applications" << std::endl;
    } else {
      std::cout << "  ⚠ Notable difference between FP32 and FP16" << std::endl;
      std::cout << "    Consider mixed precision for critical applications" << std::endl;
    }
    
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  
  return 0;
}
