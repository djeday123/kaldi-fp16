// CGO Interface - C wrapper for Go bindings
// Links Go kaldibridge package to CUDA kernels

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstring>
#include <string>
#include <vector>

// Thread-local error message
static thread_local std::string g_last_error;

extern "C" {

// ============================================================================
// Error Handling
// ============================================================================

const char* kaldi_get_last_error() {
    if (g_last_error.empty()) {
        return nullptr;
    }
    return g_last_error.c_str();
}

void kaldi_clear_error() {
    g_last_error.clear();
}

static void set_error(const char* msg) {
    g_last_error = msg;
}

static void set_cuda_error(cudaError_t err) {
    g_last_error = cudaGetErrorString(err);
}

static void set_cublas_error(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_NOT_INITIALIZED: g_last_error = "cuBLAS not initialized"; break;
        case CUBLAS_STATUS_ALLOC_FAILED: g_last_error = "cuBLAS allocation failed"; break;
        case CUBLAS_STATUS_INVALID_VALUE: g_last_error = "cuBLAS invalid value"; break;
        case CUBLAS_STATUS_ARCH_MISMATCH: g_last_error = "cuBLAS arch mismatch"; break;
        case CUBLAS_STATUS_EXECUTION_FAILED: g_last_error = "cuBLAS execution failed"; break;
        default: g_last_error = "cuBLAS unknown error";
    }
}

// ============================================================================
// cuBLAS Handle Management
// ============================================================================

void* kaldi_cublas_create() {
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        set_cublas_error(status);
        return nullptr;
    }
    return (void*)handle;
}

void kaldi_cublas_destroy(void* handle) {
    if (handle) {
        cublasDestroy((cublasHandle_t)handle);
    }
}

void kaldi_cublas_enable_tensor_cores(void* handle) {
    if (handle) {
        // Enable Tensor Core math
        cublasSetMathMode((cublasHandle_t)handle, CUBLAS_TENSOR_OP_MATH);
    }
}

// ============================================================================
// Tensor Management
// ============================================================================

struct TensorFP16 {
    __half* data;
    int rows;
    int cols;
    size_t size;
};

void* kaldi_tensor_create(int rows, int cols) {
    TensorFP16* t = new TensorFP16();
    t->rows = rows;
    t->cols = cols;
    t->size = (size_t)rows * cols;
    
    cudaError_t err = cudaMalloc(&t->data, t->size * sizeof(__half));
    if (err != cudaSuccess) {
        set_cuda_error(err);
        delete t;
        return nullptr;
    }
    
    return t;
}

void* kaldi_tensor_zeros(int rows, int cols) {
    TensorFP16* t = (TensorFP16*)kaldi_tensor_create(rows, cols);
    if (t) {
        cudaMemset(t->data, 0, t->size * sizeof(__half));
    }
    return t;
}

void* kaldi_tensor_ones(int rows, int cols) {
    TensorFP16* t = (TensorFP16*)kaldi_tensor_create(rows, cols);
    if (t) {
        // Fill with 1.0 in FP16
        std::vector<__half> ones(t->size, __float2half(1.0f));
        cudaMemcpy(t->data, ones.data(), t->size * sizeof(__half), cudaMemcpyHostToDevice);
    }
    return t;
}

void kaldi_tensor_free(void* tensor) {
    if (tensor) {
        TensorFP16* t = (TensorFP16*)tensor;
        if (t->data) {
            cudaFree(t->data);
        }
        delete t;
    }
}

int kaldi_tensor_rows(void* tensor) {
    return tensor ? ((TensorFP16*)tensor)->rows : 0;
}

int kaldi_tensor_cols(void* tensor) {
    return tensor ? ((TensorFP16*)tensor)->cols : 0;
}

size_t kaldi_tensor_size(void* tensor) {
    return tensor ? ((TensorFP16*)tensor)->size : 0;
}

// ============================================================================
// Data Transfer (FP32 <-> FP16)
// ============================================================================

// CUDA kernel for FP32 to FP16 conversion
__global__ void fp32_to_fp16_kernel(const float* src, __half* dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

// CUDA kernel for FP16 to FP32 conversion
__global__ void fp16_to_fp32_kernel(const __half* src, float* dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __half2float(src[idx]);
    }
}

void kaldi_tensor_copy_from_host_fp32(void* tensor, const float* data, size_t count) {
    if (!tensor || !data) return;
    
    TensorFP16* t = (TensorFP16*)tensor;
    if (count > t->size) count = t->size;
    
    // Allocate temp GPU buffer for FP32
    float* d_fp32;
    cudaMalloc(&d_fp32, count * sizeof(float));
    cudaMemcpy(d_fp32, data, count * sizeof(float), cudaMemcpyHostToDevice);
    
    // Convert FP32 -> FP16 on GPU
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    fp32_to_fp16_kernel<<<blocks, threads>>>(d_fp32, t->data, count);
    
    cudaFree(d_fp32);
}

void kaldi_tensor_copy_to_host_fp32(void* tensor, float* data, size_t count) {
    if (!tensor || !data) return;
    
    TensorFP16* t = (TensorFP16*)tensor;
    if (count > t->size) count = t->size;
    
    // Allocate temp GPU buffer for FP32
    float* d_fp32;
    cudaMalloc(&d_fp32, count * sizeof(float));
    
    // Convert FP16 -> FP32 on GPU
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    fp16_to_fp32_kernel<<<blocks, threads>>>(t->data, d_fp32, count);
    
    cudaMemcpy(data, d_fp32, count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_fp32);
}

// ============================================================================
// GEMM with Tensor Cores
// ============================================================================

void kaldi_gemm(void* handle, void* A, void* B, void* C,
                float alpha, float beta, int transA, int transB) {
    if (!handle || !A || !B || !C) {
        set_error("null pointer in GEMM");
        return;
    }
    
    cublasHandle_t h = (cublasHandle_t)handle;
    TensorFP16* tA = (TensorFP16*)A;
    TensorFP16* tB = (TensorFP16*)B;
    TensorFP16* tC = (TensorFP16*)C;
    
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    
    int m = transA ? tA->cols : tA->rows;
    int k = transA ? tA->rows : tA->cols;
    int n = transB ? tB->rows : tB->cols;
    
    // Use FP16 GEMM with Tensor Cores
    // Note: cuBLAS uses column-major, so we compute B^T @ A^T = (A @ B)^T
    __half h_alpha = __float2half(alpha);
    __half h_beta = __float2half(beta);
    
    cublasStatus_t status = cublasHgemm(h,
        opB, opA,
        n, m, k,
        &h_alpha,
        tB->data, tB->cols,
        tA->data, tA->cols,
        &h_beta,
        tC->data, tC->cols
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        set_cublas_error(status);
    }
}

// ============================================================================
// Activation Functions (in-place)
// ============================================================================

__global__ void relu_kernel(__half* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __half2float(data[idx]);
        data[idx] = __float2half(val > 0.0f ? val : 0.0f);
    }
}

__global__ void sigmoid_kernel(__half* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __half2float(data[idx]);
        data[idx] = __float2half(1.0f / (1.0f + expf(-val)));
    }
}

__global__ void tanh_kernel(__half* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __half2float(data[idx]);
        data[idx] = __float2half(tanhf(val));
    }
}

void kaldi_relu(void* tensor) {
    if (!tensor) return;
    TensorFP16* t = (TensorFP16*)tensor;
    int threads = 256;
    int blocks = (t->size + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(t->data, t->size);
}

void kaldi_sigmoid(void* tensor) {
    if (!tensor) return;
    TensorFP16* t = (TensorFP16*)tensor;
    int threads = 256;
    int blocks = (t->size + threads - 1) / threads;
    sigmoid_kernel<<<blocks, threads>>>(t->data, t->size);
}

void kaldi_tanh(void* tensor) {
    if (!tensor) return;
    TensorFP16* t = (TensorFP16*)tensor;
    int threads = 256;
    int blocks = (t->size + threads - 1) / threads;
    tanh_kernel<<<blocks, threads>>>(t->data, t->size);
}

// ============================================================================
// Softmax
// ============================================================================

__global__ void softmax_kernel(__half* data, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    extern __shared__ float shared[];
    
    // Find max
    float max_val = -1e10f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float val = __half2float(data[row * cols + c]);
        if (val > max_val) max_val = val;
    }
    shared[threadIdx.x] = max_val;
    __syncthreads();
    
    // Reduce max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && shared[threadIdx.x + s] > shared[threadIdx.x]) {
            shared[threadIdx.x] = shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    max_val = shared[0];
    __syncthreads();
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float val = expf(__half2float(data[row * cols + c]) - max_val);
        data[row * cols + c] = __float2half(val);
        sum += val;
    }
    shared[threadIdx.x] = sum;
    __syncthreads();
    
    // Reduce sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    sum = shared[0];
    
    // Normalize
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float val = __half2float(data[row * cols + c]) / sum;
        data[row * cols + c] = __float2half(val);
    }
}

void kaldi_softmax(void* tensor) {
    if (!tensor) return;
    TensorFP16* t = (TensorFP16*)tensor;
    
    int threads = 256;
    if (t->cols < threads) threads = t->cols;
    
    softmax_kernel<<<t->rows, threads, threads * sizeof(float)>>>(
        t->data, t->rows, t->cols
    );
}

// ============================================================================
// Element-wise Operations
// ============================================================================

__global__ void add_kernel(__half* a, const __half* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = __float2half(__half2float(a[idx]) + __half2float(b[idx]));
    }
}

__global__ void scale_kernel(__half* data, float alpha, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = __float2half(__half2float(data[idx]) * alpha);
    }
}

void kaldi_add(void* a, void* b) {
    if (!a || !b) return;
    TensorFP16* tA = (TensorFP16*)a;
    TensorFP16* tB = (TensorFP16*)b;
    
    int threads = 256;
    int blocks = (tA->size + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(tA->data, tB->data, tA->size);
}

void kaldi_scale(void* tensor, float alpha) {
    if (!tensor) return;
    TensorFP16* t = (TensorFP16*)tensor;
    
    int threads = 256;
    int blocks = (t->size + threads - 1) / threads;
    scale_kernel<<<blocks, threads>>>(t->data, alpha, t->size);
}

// ============================================================================
// Loss Scaler for Mixed Precision
// ============================================================================

struct LossScaler {
    float scale;
    float growth_factor;
    float backoff_factor;
    int growth_interval;
    int steps_since_growth;
};

void* kaldi_loss_scaler_create(float initial_scale) {
    LossScaler* ls = new LossScaler();
    ls->scale = initial_scale;
    ls->growth_factor = 2.0f;
    ls->backoff_factor = 0.5f;
    ls->growth_interval = 2000;
    ls->steps_since_growth = 0;
    return ls;
}

void kaldi_loss_scaler_free(void* scaler) {
    if (scaler) {
        delete (LossScaler*)scaler;
    }
}

float kaldi_loss_scaler_get_scale(void* scaler) {
    return scaler ? ((LossScaler*)scaler)->scale : 1.0f;
}

void kaldi_loss_scaler_update(void* scaler, int overflow) {
    if (!scaler) return;
    LossScaler* ls = (LossScaler*)scaler;
    
    if (overflow) {
        ls->scale *= ls->backoff_factor;
        ls->steps_since_growth = 0;
    } else {
        ls->steps_since_growth++;
        if (ls->steps_since_growth >= ls->growth_interval) {
            ls->scale *= ls->growth_factor;
            ls->steps_since_growth = 0;
        }
    }
    
    // Clamp scale
    if (ls->scale < 1.0f) ls->scale = 1.0f;
    if (ls->scale > 65536.0f) ls->scale = 65536.0f;
}

}  // extern "C"
