#include "kaldi-fp16/matrix-fp16.h"
#include "kaldi-fp16/tensor-ops.h"
#include <iostream>
#include <iomanip>
#include <vector>

using namespace kaldi::fp16;

void PrintResults(const std::string& name, int m, int n, int k, 
                 float time_ms, bool use_tensor_cores) {
  float tflops = GetTFLOPS(m, n, k, time_ms);
  
  std::cout << std::left << std::setw(30) << name
            << std::right << std::setw(10) << std::fixed << std::setprecision(3) 
            << time_ms << " ms"
            << std::setw(12) << std::fixed << std::setprecision(2)
            << tflops << " TFLOPS"
            << (use_tensor_cores ? "  [Tensor Cores]" : "  [CUDA Cores]")
            << std::endl;
}

int main() {
  std::cout << "=== Kaldi-FP16 Performance Benchmark ===" << std::endl;
  std::cout << std::endl;
  
  // Check device info
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  
  std::cout << "GPU Information:" << std::endl;
  std::cout << "  Device: " << prop.name << std::endl;
  std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
  std::cout << "  Memory: " << prop.totalGlobalMem / (1024*1024*1024) << " GB" << std::endl;
  
  if (TensorCoresAvailable()) {
    std::cout << "  Tensor Cores: ✓ Available" << std::endl;
  } else {
    std::cout << "  Tensor Cores: ✗ Not available" << std::endl;
    return 1;
  }
  
  std::cout << std::endl;
  
  // Test different matrix sizes
  std::vector<std::tuple<int, int, int>> sizes = {
    {512, 512, 512},
    {1024, 1024, 1024},
    {2048, 2048, 2048},
    {4096, 4096, 4096},
    {8192, 8192, 8192}
  };
  
  const int num_iterations = 100;
  
  std::cout << "Benchmarking matrix multiplication (average over " 
            << num_iterations << " iterations):" << std::endl;
  std::cout << std::string(80, '=') << std::endl;
  std::cout << std::left << std::setw(30) << "Matrix Size (M x N x K)"
            << std::right << std::setw(10) << "Time"
            << std::setw(12) << "Performance" << std::endl;
  std::cout << std::string(80, '-') << std::endl;
  
  for (const auto& size : sizes) {
    int m, n, k;
    std::tie(m, n, k) = size;
    
    std::string size_str = std::to_string(m) + "x" + 
                          std::to_string(n) + "x" + 
                          std::to_string(k);
    
    try {
      // Benchmark with Tensor Cores
      float time_tensor = BenchmarkMatMul(m, n, k, true, num_iterations);
      PrintResults(size_str, m, n, k, time_tensor, true);
      
      // Benchmark without Tensor Cores (only for smaller sizes)
      if (m <= 2048) {
        float time_cuda = BenchmarkMatMul(m, n, k, false, num_iterations);
        std::string cuda_str = size_str + " (CUDA only)";
        PrintResults(cuda_str, m, n, k, time_cuda, false);
        
        // Calculate speedup
        float speedup = time_cuda / time_tensor;
        std::cout << "  → Speedup: " << std::fixed << std::setprecision(2) 
                  << speedup << "x" << std::endl;
      }
      
      std::cout << std::endl;
      
    } catch (const std::exception& e) {
      std::cerr << "  Error for size " << size_str << ": " << e.what() << std::endl;
    }
  }
  
  std::cout << std::string(80, '=') << std::endl;
  std::cout << std::endl;
  std::cout << "Benchmark completed!" << std::endl;
  std::cout << std::endl;
  std::cout << "Note: Tensor Cores provide significant speedup for FP16 operations" << std::endl;
  std::cout << "      on supported NVIDIA GPUs (Volta, Turing, Ampere, Ada, Hopper)" << std::endl;
  
  return 0;
}
