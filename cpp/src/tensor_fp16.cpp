#include "tensor_fp16.h"
#include <cstring>
#include <algorithm>

namespace kaldi_fp16 {

// ============================================================================
// CuBLASHandle Implementation
// ============================================================================

CuBLASHandle::CuBLASHandle() {
    CUBLAS_CHECK(cublasCreate(&handle_));
    enableTensorCores();
}

CuBLASHandle::~CuBLASHandle() {
    if (handle_) {
        cublasDestroy(handle_);
    }
}

void CuBLASHandle::enableTensorCores() {
    // Enable Tensor Core math for FP16 operations
    // This uses CUBLAS_TENSOR_OP_MATH which automatically uses
    // Tensor Cores when available (Volta+)
    CUBLAS_CHECK(cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH));
}

void CuBLASHandle::setStream(cudaStream_t stream) {
    CUBLAS_CHECK(cublasSetStream(handle_, stream));
}

// ============================================================================
// TensorFP16 Implementation
// ============================================================================

TensorFP16::TensorFP16() = default;

TensorFP16::TensorFP16(int rows, int cols, bool allocate) 
    : rows_(rows), cols_(cols) {
    if (allocate && rows > 0 && cols > 0) {
        this->allocate(rows, cols);
    }
}

TensorFP16::TensorFP16(const std::vector<int>& shape) {
    if (shape.size() == 1) {
        rows_ = 1;
        cols_ = shape[0];
    } else if (shape.size() == 2) {
        rows_ = shape[0];
        cols_ = shape[1];
    } else {
        throw std::runtime_error("TensorFP16 only supports 1D and 2D tensors");
    }
    allocate(rows_, cols_);
}

TensorFP16::~TensorFP16() {
    free();
}

TensorFP16::TensorFP16(TensorFP16&& other) noexcept
    : data_(other.data_), rows_(other.rows_), cols_(other.cols_), 
      owns_memory_(other.owns_memory_) {
    other.data_ = nullptr;
    other.rows_ = 0;
    other.cols_ = 0;
    other.owns_memory_ = false;
}

TensorFP16& TensorFP16::operator=(TensorFP16&& other) noexcept {
    if (this != &other) {
        free();
        data_ = other.data_;
        rows_ = other.rows_;
        cols_ = other.cols_;
        owns_memory_ = other.owns_memory_;
        other.data_ = nullptr;
        other.rows_ = 0;
        other.cols_ = 0;
        other.owns_memory_ = false;
    }
    return *this;
}

void TensorFP16::allocate(int rows, int cols) {
    free();
    rows_ = rows;
    cols_ = cols;
    if (rows > 0 && cols > 0) {
        CUDA_CHECK(cudaMalloc(&data_, bytes()));
        owns_memory_ = true;
    }
}

void TensorFP16::free() {
    if (data_ && owns_memory_) {
        cudaFree(data_);
    }
    data_ = nullptr;
    rows_ = 0;
    cols_ = 0;
    owns_memory_ = true;
}

void TensorFP16::zero() {
    if (data_ && size() > 0) {
        CUDA_CHECK(cudaMemset(data_, 0, bytes()));
    }
}

void TensorFP16::copyFromHost(const float* host_data, size_t count) {
    if (count > size()) {
        throw std::runtime_error("copyFromHost: count exceeds tensor size");
    }
    
    // Allocate temporary FP16 buffer on host
    std::vector<__half> host_fp16(count);
    for (size_t i = 0; i < count; ++i) {
        host_fp16[i] = __float2half(host_data[i]);
    }
    
    CUDA_CHECK(cudaMemcpy(data_, host_fp16.data(), count * sizeof(__half), 
                          cudaMemcpyHostToDevice));
}

void TensorFP16::copyFromHostFP16(const __half* host_data, size_t count) {
    if (count > size()) {
        throw std::runtime_error("copyFromHostFP16: count exceeds tensor size");
    }
    CUDA_CHECK(cudaMemcpy(data_, host_data, count * sizeof(__half), 
                          cudaMemcpyHostToDevice));
}

void TensorFP16::copyToHost(float* host_data, size_t count) const {
    if (count > size()) {
        throw std::runtime_error("copyToHost: count exceeds tensor size");
    }
    
    // Allocate temporary FP16 buffer on host
    std::vector<__half> host_fp16(count);
    CUDA_CHECK(cudaMemcpy(host_fp16.data(), data_, count * sizeof(__half), 
                          cudaMemcpyDeviceToHost));
    
    for (size_t i = 0; i < count; ++i) {
        host_data[i] = __half2float(host_fp16[i]);
    }
}

void TensorFP16::copyToHostFP16(__half* host_data, size_t count) const {
    if (count > size()) {
        throw std::runtime_error("copyToHostFP16: count exceeds tensor size");
    }
    CUDA_CHECK(cudaMemcpy(host_data, data_, count * sizeof(__half), 
                          cudaMemcpyDeviceToHost));
}

TensorFP16 TensorFP16::zeros(int rows, int cols) {
    TensorFP16 t(rows, cols);
    t.zero();
    return t;
}

TensorFP16 TensorFP16::ones(int rows, int cols) {
    TensorFP16 t(rows, cols);
    std::vector<__half> ones_data(t.size(), __float2half(1.0f));
    t.copyFromHostFP16(ones_data.data(), t.size());
    return t;
}

TensorFP16 TensorFP16::fromFP32(const float* host_data, int rows, int cols) {
    TensorFP16 t(rows, cols);
    t.copyFromHost(host_data, t.size());
    return t;
}

// ============================================================================
// FP16 GEMM Operations (Tensor Cores)
// ============================================================================

namespace ops {

void gemm(const CuBLASHandle& handle,
          const TensorFP16& A,
          const TensorFP16& B,
          TensorFP16& C,
          float alpha,
          float beta,
          bool transA,
          bool transB) {
    
    // cuBLAS uses column-major, so we compute C^T = B^T * A^T
    // which gives us C in row-major (what we want)
    
    int M = transA ? A.cols() : A.rows();
    int N = transB ? B.rows() : B.cols();
    int K = transA ? A.rows() : A.cols();
    
    int lda = A.cols();  // Leading dimension for row-major A
    int ldb = B.cols();  // Leading dimension for row-major B
    int ldc = C.cols();  // Leading dimension for row-major C
    
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    
    // For row-major: C = A * B becomes C^T = B^T * A^T in column-major
    // So we swap A and B, swap M and N
    CUBLAS_CHECK(cublasGemmEx(
        handle.get(),
        transB ? CUBLAS_OP_T : CUBLAS_OP_N,  // opA for B^T
        transA ? CUBLAS_OP_T : CUBLAS_OP_N,  // opB for A^T
        N, M, K,                              // Swapped M and N
        &alpha,
        B.data(), CUDA_R_16F, ldb,           // B is "A" in cuBLAS
        A.data(), CUDA_R_16F, lda,           // A is "B" in cuBLAS
        &beta,
        C.data(), CUDA_R_16F, ldc,
        CUBLAS_COMPUTE_16F,                  // FP16 compute
        CUBLAS_GEMM_DEFAULT_TENSOR_OP        // Use Tensor Cores!
    ));
}

void gemmBatched(const CuBLASHandle& handle,
                 const std::vector<const TensorFP16*>& As,
                 const std::vector<const TensorFP16*>& Bs,
                 std::vector<TensorFP16*>& Cs,
                 float alpha,
                 float beta) {
    
    int batchCount = static_cast<int>(As.size());
    if (batchCount == 0) return;
    
    // Collect device pointers
    std::vector<const __half*> Aarray(batchCount);
    std::vector<const __half*> Barray(batchCount);
    std::vector<__half*> Carray(batchCount);
    
    for (int i = 0; i < batchCount; ++i) {
        Aarray[i] = As[i]->data();
        Barray[i] = Bs[i]->data();
        Carray[i] = Cs[i]->data();
    }
    
    // Copy pointer arrays to device
    const __half** d_Aarray;
    const __half** d_Barray;
    __half** d_Carray;
    
    CUDA_CHECK(cudaMalloc(&d_Aarray, batchCount * sizeof(__half*)));
    CUDA_CHECK(cudaMalloc(&d_Barray, batchCount * sizeof(__half*)));
    CUDA_CHECK(cudaMalloc(&d_Carray, batchCount * sizeof(__half*)));
    
    CUDA_CHECK(cudaMemcpy(d_Aarray, Aarray.data(), 
                          batchCount * sizeof(__half*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Barray, Barray.data(), 
                          batchCount * sizeof(__half*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Carray, Carray.data(), 
                          batchCount * sizeof(__half*), cudaMemcpyHostToDevice));
    
    int M = As[0]->rows();
    int N = Bs[0]->cols();
    int K = As[0]->cols();
    
    CUBLAS_CHECK(cublasHgemmBatched(
        handle.get(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        reinterpret_cast<const __half*>(&alpha),
        d_Barray, N,
        d_Aarray, K,
        reinterpret_cast<const __half*>(&beta),
        d_Carray, N,
        batchCount
    ));
    
    cudaFree(d_Aarray);
    cudaFree(d_Barray);
    cudaFree(d_Carray);
}

void gemmStridedBatched(const CuBLASHandle& handle,
                        const TensorFP16& A,
                        const TensorFP16& B,
                        TensorFP16& C,
                        int batchCount,
                        long long strideA,
                        long long strideB,
                        long long strideC,
                        float alpha,
                        float beta) {
    
    // Assuming each batch is M x K and K x N
    int M = static_cast<int>(strideC / C.cols());
    int N = C.cols();
    int K = static_cast<int>(strideA / M);
    
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        handle.get(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B.data(), CUDA_R_16F, N, strideB,
        A.data(), CUDA_R_16F, K, strideA,
        &beta,
        C.data(), CUDA_R_16F, N, strideC,
        batchCount,
        CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

} // namespace ops

// ============================================================================
// Loss Scaler for Mixed Precision Training
// ============================================================================

LossScaler::LossScaler(float initial_scale, float growth_factor, 
                       float backoff_factor, int growth_interval)
    : scale_(initial_scale), growth_factor_(growth_factor),
      backoff_factor_(backoff_factor), growth_interval_(growth_interval) {}

void LossScaler::update(bool overflow_detected) {
    if (overflow_detected) {
        scale_ *= backoff_factor_;
        steps_since_growth_ = 0;
    } else {
        steps_since_growth_++;
        if (steps_since_growth_ >= growth_interval_) {
            scale_ *= growth_factor_;
            steps_since_growth_ = 0;
        }
    }
}

// ============================================================================
// Affine Layer Implementation
// ============================================================================

AffineLayer::AffineLayer(int input_dim, int output_dim)
    : weights_(input_dim, output_dim),
      bias_(1, output_dim),
      grad_weights_(input_dim, output_dim),
      grad_bias_(1, output_dim),
      handle_(std::make_shared<CuBLASHandle>()) {
    
    // Initialize with Xavier/Glorot initialization
    float scale = std::sqrt(2.0f / (input_dim + output_dim));
    std::vector<float> init_weights(weights_.size());
    for (auto& w : init_weights) {
        w = scale * (2.0f * (rand() / (float)RAND_MAX) - 1.0f);
    }
    weights_.copyFromHost(init_weights.data(), weights_.size());
    bias_.zero();
    grad_weights_.zero();
    grad_bias_.zero();
}

void AffineLayer::forward(const TensorFP16& input, TensorFP16& output) {
    // Cache input for backward pass
    input_cache_.allocate(input.rows(), input.cols());
    CUDA_CHECK(cudaMemcpy(input_cache_.data(), input.data(), 
                          input.bytes(), cudaMemcpyDeviceToDevice));
    
    // output = input * weights
    ops::gemm(*handle_, input, weights_, output);
    
    // Add bias (broadcast across batch dimension)
    // TODO: Implement CUDA kernel for bias addition
}

void AffineLayer::backward(const TensorFP16& grad_output, TensorFP16& grad_input) {
    // grad_weights = input^T * grad_output
    ops::gemm(*handle_, input_cache_, grad_output, grad_weights_, 
              1.0f, 1.0f, true, false);  // Accumulate gradients
    
    // grad_input = grad_output * weights^T
    ops::gemm(*handle_, grad_output, weights_, grad_input,
              1.0f, 0.0f, false, true);
    
    // TODO: Accumulate grad_bias
}

void AffineLayer::updateWeights(float learning_rate) {
    // Simple SGD update: weights -= lr * grad_weights
    // TODO: Implement scale kernel
}

} // namespace kaldi_fp16
