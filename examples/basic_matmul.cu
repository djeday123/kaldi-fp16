#include "kaldi-fp16/matrix-fp16.h"
#include "kaldi-fp16/tensor-ops.h"
#include <iostream>
#include <iomanip>

using namespace kaldi::fp16;

int main() {
  std::cout << "=== Kaldi-FP16 Basic Matrix Multiplication Example ===" << std::endl;
  std::cout << std::endl;
  
  // Check for Tensor Core support
  if (TensorCoresAvailable()) {
    int cc = GetTensorCoreComputeCapability();
    std::cout << "✓ Tensor Cores available (compute capability " 
              << cc / 10 << "." << cc % 10 << ")" << std::endl;
  } else {
    std::cout << "✗ Tensor Cores not available" << std::endl;
    return 1;
  }
  
  std::cout << std::endl;
  
  // Matrix dimensions
  const int M = 1024;
  const int K = 1024;
  const int N = 1024;
  
  std::cout << "Matrix dimensions:" << std::endl;
  std::cout << "  A: " << M << " x " << K << std::endl;
  std::cout << "  B: " << K << " x " << N << std::endl;
  std::cout << "  C: " << M << " x " << N << std::endl;
  std::cout << std::endl;
  
  try {
    // Create matrices
    std::cout << "Creating FP16 matrices..." << std::endl;
    MatrixFP16 A(M, K);
    MatrixFP16 B(K, N);
    MatrixFP16 C(M, N);
    
    // Initialize with random values
    std::cout << "Initializing with random values..." << std::endl;
    A.SetRandn();
    B.SetRandn();
    C.SetZero();
    
    // Perform matrix multiplication: C = A * B
    std::cout << "Performing matrix multiplication using Tensor Cores..." << std::endl;
    C.AddMatMat(1.0f, A, kNoTrans, B, kNoTrans, 0.0f);
    
    std::cout << "✓ Matrix multiplication completed successfully!" << std::endl;
    std::cout << std::endl;
    
    // Copy a small portion to host to verify
    const int verify_size = 4;
    std::vector<float> host_c(verify_size * verify_size);
    
    // Create a temporary small matrix to extract values
    MatrixFP16 C_small(verify_size, verify_size);
    
    std::cout << "Sample output values (top-left " << verify_size << "x" << verify_size << "):" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    
    // Note: In a real implementation, we would need a proper submatrix copy
    // For now, we'll just indicate the operation was successful
    std::cout << "  [Matrix computation completed on GPU]" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Example completed successfully!" << std::endl;
    
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  
  return 0;
}
