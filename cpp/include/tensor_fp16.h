#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <memory>
#include <vector>
#include <stdexcept>

namespace kaldi_fp16 {

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error("cuBLAS error: " + std::to_string(status)); \
        } \
    } while(0)

// Forward declarations
class CuBLASHandle;
class TensorFP16;

/**
 * @brief RAII wrapper for cuBLAS handle
 * Thread-safe, one handle per GPU stream
 */
class CuBLASHandle {
public:
    CuBLASHandle();
    ~CuBLASHandle();
    
    cublasHandle_t get() const { return handle_; }
    
    // Enable Tensor Cores for FP16
    void enableTensorCores();
    
    // Set stream for async operations
    void setStream(cudaStream_t stream);
    
private:
    cublasHandle_t handle_;
    CuBLASHandle(const CuBLASHandle&) = delete;
    CuBLASHandle& operator=(const CuBLASHandle&) = delete;
};

/**
 * @brief FP16 Tensor on GPU
 * 
 * Core data structure for all FP16 operations.
 * Supports automatic memory management and
 * seamless integration with cuBLAS Tensor Cores.
 */
class TensorFP16 {
public:
    // Constructors
    TensorFP16();
    TensorFP16(int rows, int cols, bool allocate = true);
    TensorFP16(const std::vector<int>& shape);
    ~TensorFP16();
    
    // Move semantics (no copy for GPU memory)
    TensorFP16(TensorFP16&& other) noexcept;
    TensorFP16& operator=(TensorFP16&& other) noexcept;
    
    // Disable copy
    TensorFP16(const TensorFP16&) = delete;
    TensorFP16& operator=(const TensorFP16&) = delete;
    
    // Accessors
    __half* data() { return data_; }
    const __half* data() const { return data_; }
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    size_t size() const { return static_cast<size_t>(rows_) * cols_; }
    size_t bytes() const { return size() * sizeof(__half); }
    
    // Memory operations
    void allocate(int rows, int cols);
    void free();
    void zero();
    
    // Host <-> Device transfers
    void copyFromHost(const float* host_data, size_t count);
    void copyFromHostFP16(const __half* host_data, size_t count);
    void copyToHost(float* host_data, size_t count) const;
    void copyToHostFP16(__half* host_data, size_t count) const;
    
    // FP32 conversion utilities
    static void convertFP32ToFP16(const float* src, __half* dst, size_t count, cudaStream_t stream = 0);
    static void convertFP16ToFP32(const __half* src, float* dst, size_t count, cudaStream_t stream = 0);
    
    // Factory methods
    static TensorFP16 zeros(int rows, int cols);
    static TensorFP16 ones(int rows, int cols);
    static TensorFP16 fromFP32(const float* host_data, int rows, int cols);
    
private:
    __half* data_ = nullptr;
    int rows_ = 0;
    int cols_ = 0;
    bool owns_memory_ = true;
};

/**
 * @brief FP16 Matrix Operations using Tensor Cores
 * 
 * All operations use cuBLAS with CUBLAS_TENSOR_OP_MATH
 * for automatic Tensor Core acceleration on supported GPUs.
 */
namespace ops {

/**
 * @brief GEMM: C = alpha * A * B + beta * C
 * Uses Tensor Cores when available (Volta+)
 * 
 * @param handle cuBLAS handle with Tensor Cores enabled
 * @param A Input matrix A (M x K)
 * @param B Input matrix B (K x N)
 * @param C Output matrix C (M x N)
 * @param alpha Scalar multiplier for A*B
 * @param beta Scalar multiplier for C
 * @param transA Transpose A
 * @param transB Transpose B
 */
void gemm(const CuBLASHandle& handle,
          const TensorFP16& A,
          const TensorFP16& B,
          TensorFP16& C,
          float alpha = 1.0f,
          float beta = 0.0f,
          bool transA = false,
          bool transB = false);

/**
 * @brief Batched GEMM for multiple matrix multiplications
 * Efficient for small matrices (batch processing in Kaldi)
 */
void gemmBatched(const CuBLASHandle& handle,
                 const std::vector<const TensorFP16*>& As,
                 const std::vector<const TensorFP16*>& Bs,
                 std::vector<TensorFP16*>& Cs,
                 float alpha = 1.0f,
                 float beta = 0.0f);

/**
 * @brief Strided batched GEMM (more efficient memory layout)
 */
void gemmStridedBatched(const CuBLASHandle& handle,
                        const TensorFP16& A,
                        const TensorFP16& B,
                        TensorFP16& C,
                        int batchCount,
                        long long strideA,
                        long long strideB,
                        long long strideC,
                        float alpha = 1.0f,
                        float beta = 0.0f);

// Element-wise operations (CUDA kernels)
void relu(TensorFP16& x, cudaStream_t stream = 0);
void sigmoid(TensorFP16& x, cudaStream_t stream = 0);
void tanh(TensorFP16& x, cudaStream_t stream = 0);
void softmax(TensorFP16& x, int axis = -1, cudaStream_t stream = 0);

// Loss gradients
void reluBackward(const TensorFP16& x, TensorFP16& grad, cudaStream_t stream = 0);
void sigmoidBackward(const TensorFP16& output, TensorFP16& grad, cudaStream_t stream = 0);

// Vector operations
void add(TensorFP16& a, const TensorFP16& b, cudaStream_t stream = 0);
void scale(TensorFP16& x, float alpha, cudaStream_t stream = 0);

} // namespace ops

/**
 * @brief Mixed Precision Training Utilities
 * 
 * Loss scaling for stable FP16 training
 */
class LossScaler {
public:
    LossScaler(float initial_scale = 65536.0f, 
               float growth_factor = 2.0f,
               float backoff_factor = 0.5f,
               int growth_interval = 2000);
    
    float getScale() const { return scale_; }
    void update(bool overflow_detected);
    
    // Scale gradients before backward
    void scaleGradients(TensorFP16& grads);
    
    // Unscale gradients after backward
    void unscaleGradients(TensorFP16& grads);
    
    // Check for overflow/underflow
    bool checkOverflow(const TensorFP16& grads);
    
private:
    float scale_;
    float growth_factor_;
    float backoff_factor_;
    int growth_interval_;
    int steps_since_growth_ = 0;
};

/**
 * @brief DNN Layer base class for Kaldi-style networks
 */
class Layer {
public:
    virtual ~Layer() = default;
    
    virtual void forward(const TensorFP16& input, TensorFP16& output) = 0;
    virtual void backward(const TensorFP16& grad_output, TensorFP16& grad_input) = 0;
    
    virtual void updateWeights(float learning_rate) {}
    virtual size_t parameterCount() const { return 0; }
};

/**
 * @brief Affine (fully connected) layer with FP16 weights
 */
class AffineLayer : public Layer {
public:
    AffineLayer(int input_dim, int output_dim);
    
    void forward(const TensorFP16& input, TensorFP16& output) override;
    void backward(const TensorFP16& grad_output, TensorFP16& grad_input) override;
    void updateWeights(float learning_rate) override;
    
    size_t parameterCount() const override { 
        return weights_.size() + bias_.size(); 
    }
    
    // Access for initialization
    TensorFP16& weights() { return weights_; }
    TensorFP16& bias() { return bias_; }
    
private:
    TensorFP16 weights_;      // [input_dim x output_dim]
    TensorFP16 bias_;         // [1 x output_dim]
    TensorFP16 grad_weights_;
    TensorFP16 grad_bias_;
    TensorFP16 input_cache_;  // For backward pass
    
    std::shared_ptr<CuBLASHandle> handle_;
};

} // namespace kaldi_fp16
