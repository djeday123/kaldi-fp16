#include "kaldi-fp16/matrix-fp16.h"
#include "kaldi-fp16/tensor-ops.h"
#include <stdexcept>
#include <cmath>
#include <random>
#include <curand.h>

namespace kaldi {
namespace fp16 {

// CUDA error checking macro
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

// Global cuBLAS handle (singleton)
static cublasHandle_t g_cublas_handle = nullptr;

cublasHandle_t MatrixFP16::GetCublasHandle() {
  if (g_cublas_handle == nullptr) {
    CUBLAS_CHECK(cublasCreate(&g_cublas_handle));
    // Enable Tensor Core operations
    EnableTensorCoreMath(g_cublas_handle);
  }
  return g_cublas_handle;
}

// CUDA kernel for FP32 to FP16 conversion
__global__ void ConvertFP32ToFP16Kernel(half* dst, const float* src, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    dst[idx] = __float2half(src[idx]);
  }
}

// CUDA kernel for FP16 to FP32 conversion
__global__ void ConvertFP16ToFP32Kernel(float* dst, const half* src, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    dst[idx] = __half2float(src[idx]);
  }
}

void ConvertFP32ToFP16(half* dst, const float* src, size_t size) {
  int blockSize = 256;
  int numBlocks = (size + blockSize - 1) / blockSize;
  ConvertFP32ToFP16Kernel<<<numBlocks, blockSize>>>(dst, src, size);
  CUDA_CHECK(cudaGetLastError());
}

void ConvertFP16ToFP32(float* dst, const half* src, size_t size) {
  int blockSize = 256;
  int numBlocks = (size + blockSize - 1) / blockSize;
  ConvertFP16ToFP32Kernel<<<numBlocks, blockSize>>>(dst, src, size);
  CUDA_CHECK(cudaGetLastError());
}

MatrixFP16::MatrixFP16()
    : num_rows_(0), num_cols_(0), stride_(0), data_(nullptr) {}

MatrixFP16::MatrixFP16(size_t rows, size_t cols)
    : num_rows_(rows), num_cols_(cols), stride_(cols), data_(nullptr) {
  if (rows > 0 && cols > 0) {
    Allocate();
  }
}

MatrixFP16::~MatrixFP16() {
  Deallocate();
}

MatrixFP16::MatrixFP16(const MatrixFP16& other)
    : num_rows_(other.num_rows_),
      num_cols_(other.num_cols_),
      stride_(other.stride_),
      data_(nullptr) {
  if (other.data_ != nullptr) {
    Allocate();
    CUDA_CHECK(cudaMemcpy(data_, other.data_,
                          num_rows_ * stride_ * sizeof(half),
                          cudaMemcpyDeviceToDevice));
  }
}

MatrixFP16::MatrixFP16(MatrixFP16&& other) noexcept
    : num_rows_(other.num_rows_),
      num_cols_(other.num_cols_),
      stride_(other.stride_),
      data_(other.data_) {
  other.num_rows_ = 0;
  other.num_cols_ = 0;
  other.stride_ = 0;
  other.data_ = nullptr;
}

MatrixFP16& MatrixFP16::operator=(const MatrixFP16& other) {
  if (this != &other) {
    Deallocate();
    num_rows_ = other.num_rows_;
    num_cols_ = other.num_cols_;
    stride_ = other.stride_;
    if (other.data_ != nullptr) {
      Allocate();
      CUDA_CHECK(cudaMemcpy(data_, other.data_,
                            num_rows_ * stride_ * sizeof(half),
                            cudaMemcpyDeviceToDevice));
    }
  }
  return *this;
}

MatrixFP16& MatrixFP16::operator=(MatrixFP16&& other) noexcept {
  if (this != &other) {
    Deallocate();
    num_rows_ = other.num_rows_;
    num_cols_ = other.num_cols_;
    stride_ = other.stride_;
    data_ = other.data_;
    other.num_rows_ = 0;
    other.num_cols_ = 0;
    other.stride_ = 0;
    other.data_ = nullptr;
  }
  return *this;
}

void MatrixFP16::Allocate() {
  // Align stride to 8 for optimal memory access with Tensor Cores
  stride_ = (num_cols_ + 7) & ~7;
  CUDA_CHECK(cudaMalloc(&data_, num_rows_ * stride_ * sizeof(half)));
}

void MatrixFP16::Deallocate() {
  if (data_ != nullptr) {
    cudaFree(data_);
    data_ = nullptr;
  }
}

void MatrixFP16::Resize(size_t rows, size_t cols) {
  if (rows != num_rows_ || cols != num_cols_) {
    Deallocate();
    num_rows_ = rows;
    num_cols_ = cols;
    if (rows > 0 && cols > 0) {
      Allocate();
    }
  }
}

void MatrixFP16::SetZero() {
  if (data_ != nullptr) {
    CUDA_CHECK(cudaMemset(data_, 0, num_rows_ * stride_ * sizeof(half)));
  }
}

void MatrixFP16::SetRandn() {
  if (data_ == nullptr) return;
  
  // Generate random FP32 data on host
  std::vector<float> host_data(num_rows_ * num_cols_);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(0.0f, 1.0f);
  
  for (size_t i = 0; i < host_data.size(); ++i) {
    host_data[i] = dist(gen);
  }
  
  CopyFromHost(host_data.data());
}

void MatrixFP16::CopyFromHost(const float* data) {
  if (data_ == nullptr) return;
  
  // Allocate temporary device memory for FP32 data
  float* d_fp32;
  CUDA_CHECK(cudaMalloc(&d_fp32, num_rows_ * num_cols_ * sizeof(float)));
  
  // Copy FP32 data to device
  CUDA_CHECK(cudaMemcpy(d_fp32, data,
                        num_rows_ * num_cols_ * sizeof(float),
                        cudaMemcpyHostToDevice));
  
  // Convert to FP16 with proper stride handling
  for (size_t row = 0; row < num_rows_; ++row) {
    ConvertFP32ToFP16(data_ + row * stride_,
                     d_fp32 + row * num_cols_,
                     num_cols_);
  }
  
  CUDA_CHECK(cudaFree(d_fp32));
  CUDA_CHECK(cudaDeviceSynchronize());
}

void MatrixFP16::CopyToHost(float* data) const {
  if (data_ == nullptr) return;
  
  // Allocate temporary device memory for FP32 data
  float* d_fp32;
  CUDA_CHECK(cudaMalloc(&d_fp32, num_rows_ * num_cols_ * sizeof(float)));
  
  // Convert from FP16 to FP32 with proper stride handling
  for (size_t row = 0; row < num_rows_; ++row) {
    ConvertFP16ToFP32(d_fp32 + row * num_cols_,
                     data_ + row * stride_,
                     num_cols_);
  }
  
  // Copy FP32 data to host
  CUDA_CHECK(cudaMemcpy(data, d_fp32,
                        num_rows_ * num_cols_ * sizeof(float),
                        cudaMemcpyDeviceToHost));
  
  CUDA_CHECK(cudaFree(d_fp32));
  CUDA_CHECK(cudaDeviceSynchronize());
}

void MatrixFP16::CopyFromHostFP16(const half* data) {
  if (data_ == nullptr) return;
  
  // Handle stride properly
  if (stride_ == num_cols_) {
    CUDA_CHECK(cudaMemcpy(data_, data,
                          num_rows_ * num_cols_ * sizeof(half),
                          cudaMemcpyHostToDevice));
  } else {
    for (size_t row = 0; row < num_rows_; ++row) {
      CUDA_CHECK(cudaMemcpy(data_ + row * stride_,
                            data + row * num_cols_,
                            num_cols_ * sizeof(half),
                            cudaMemcpyHostToDevice));
    }
  }
}

void MatrixFP16::CopyToHostFP16(half* data) const {
  if (data_ == nullptr) return;
  
  // Handle stride properly
  if (stride_ == num_cols_) {
    CUDA_CHECK(cudaMemcpy(data, data_,
                          num_rows_ * num_cols_ * sizeof(half),
                          cudaMemcpyDeviceToHost));
  } else {
    for (size_t row = 0; row < num_rows_; ++row) {
      CUDA_CHECK(cudaMemcpy(data + row * num_cols_,
                            data_ + row * stride_,
                            num_cols_ * sizeof(half),
                            cudaMemcpyDeviceToHost));
    }
  }
}

void MatrixFP16::AddMatMat(float alpha,
                          const MatrixFP16& A, MatrixTransposeType transA,
                          const MatrixFP16& B, MatrixTransposeType transB,
                          float beta) {
  // Get dimensions after transpose
  int m = transA == kNoTrans ? A.num_rows_ : A.num_cols_;
  int n = transB == kNoTrans ? B.num_cols_ : B.num_rows_;
  int k = transA == kNoTrans ? A.num_cols_ : A.num_rows_;
  
  // Verify dimensions
  int k_check = transB == kNoTrans ? B.num_rows_ : B.num_cols_;
  if (k != k_check) {
    throw std::runtime_error("Matrix dimensions don't match for multiplication");
  }
  
  // Ensure this matrix has correct dimensions
  if (num_rows_ != static_cast<size_t>(m) || num_cols_ != static_cast<size_t>(n)) {
    throw std::runtime_error("Output matrix has incorrect dimensions");
  }
  
  cublasHandle_t handle = GetCublasHandle();
  
  cublasOperation_t opA = transA == kNoTrans ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t opB = transB == kNoTrans ? CUBLAS_OP_N : CUBLAS_OP_T;
  
  // Use Tensor Core accelerated GEMM with FP16 compute and FP32 accumulation
  const half h_alpha = __float2half(alpha);
  const half h_beta = __float2half(beta);
  
  // cuBLAS uses column-major, so we swap A and B and transpose operations
  CUBLAS_CHECK(cublasGemmEx(
      handle,
      opB, opA,
      n, m, k,
      &alpha,
      B.data_, CUDA_R_16F, B.stride_,
      A.data_, CUDA_R_16F, A.stride_,
      &beta,
      data_, CUDA_R_16F, stride_,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  
  CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace fp16
}  // namespace kaldi
