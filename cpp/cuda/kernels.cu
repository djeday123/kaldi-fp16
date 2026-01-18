/**
 * @file kernels.cu
 * @brief CUDA kernels for FP16 element-wise operations
 * 
 * Optimized for modern GPUs with Tensor Cores (Volta+)
 * Uses __half2 for 2x throughput where possible
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

namespace kaldi_fp16 {
namespace kernels {

// Thread block size for 1D kernels
constexpr int BLOCK_SIZE = 256;

// ============================================================================
// Helper functions for FP16 math
// ============================================================================

__device__ __forceinline__ __half hexp(__half x) {
    return __float2half(expf(__half2float(x)));
}

__device__ __forceinline__ __half hlog(__half x) {
    return __float2half(logf(__half2float(x)));
}

__device__ __forceinline__ __half htanh(__half x) {
    return __float2half(tanhf(__half2float(x)));
}

__device__ __forceinline__ __half hsigmoid(__half x) {
    float fx = __half2float(x);
    return __float2half(1.0f / (1.0f + expf(-fx)));
}

// ============================================================================
// ReLU
// ============================================================================

__global__ void relu_kernel(__half* x, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        __half val = x[idx];
        x[idx] = __hgt(val, __float2half(0.0f)) ? val : __float2half(0.0f);
    }
}

// Vectorized ReLU using half2 (2x throughput)
__global__ void relu_kernel_vec(__half2* x, size_t n2) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) {
        __half2 val = x[idx];
        __half2 zero = __float2half2_rn(0.0f);
        
        // Compare each half and mask
        __half2 mask;
        mask.x = __hgt(val.x, zero.x) ? __float2half(1.0f) : __float2half(0.0f);
        mask.y = __hgt(val.y, zero.y) ? __float2half(1.0f) : __float2half(0.0f);
        
        x[idx] = __hmul2(val, mask);
    }
}

__global__ void relu_backward_kernel(const __half* x, __half* grad, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        __half zero = __float2half(0.0f);
        grad[idx] = __hgt(x[idx], zero) ? grad[idx] : zero;
    }
}

// ============================================================================
// Sigmoid
// ============================================================================

__global__ void sigmoid_kernel(__half* x, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = hsigmoid(x[idx]);
    }
}

__global__ void sigmoid_backward_kernel(const __half* output, __half* grad, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // grad = grad * output * (1 - output)
        __half out = output[idx];
        __half one = __float2half(1.0f);
        __half one_minus_out = __hsub(one, out);
        grad[idx] = __hmul(__hmul(grad[idx], out), one_minus_out);
    }
}

// ============================================================================
// Tanh
// ============================================================================

__global__ void tanh_kernel(__half* x, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = htanh(x[idx]);
    }
}

__global__ void tanh_backward_kernel(const __half* output, __half* grad, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // grad = grad * (1 - output^2)
        __half out = output[idx];
        __half one = __float2half(1.0f);
        __half out_sq = __hmul(out, out);
        grad[idx] = __hmul(grad[idx], __hsub(one, out_sq));
    }
}

// ============================================================================
// Softmax (row-wise)
// ============================================================================

// Stable softmax: subtract max, then exp, then normalize
__global__ void softmax_kernel(__half* x, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    __half* row_data = x + row * cols;
    
    // Find max (for numerical stability)
    __half max_val = row_data[0];
    for (int i = 1; i < cols; ++i) {
        if (__hgt(row_data[i], max_val)) {
            max_val = row_data[i];
        }
    }
    
    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < cols; ++i) {
        float val = expf(__half2float(__hsub(row_data[i], max_val)));
        row_data[i] = __float2half(val);
        sum += val;
    }
    
    // Normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < cols; ++i) {
        row_data[i] = __float2half(__half2float(row_data[i]) * inv_sum);
    }
}

// More efficient softmax using shared memory
__global__ void softmax_kernel_shared(__half* x, int rows, int cols) {
    extern __shared__ float sdata[];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row >= rows) return;
    
    __half* row_data = x + row * cols;
    
    // Each thread handles multiple elements
    int items_per_thread = (cols + blockDim.x - 1) / blockDim.x;
    
    // Find local max
    float local_max = -INFINITY;
    for (int i = 0; i < items_per_thread; ++i) {
        int idx = tid + i * blockDim.x;
        if (idx < cols) {
            float val = __half2float(row_data[idx]);
            local_max = fmaxf(local_max, val);
        }
    }
    
    // Reduce to find global max
    sdata[tid] = local_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    float global_max = sdata[0];
    __syncthreads();
    
    // Compute exp and local sum
    float local_sum = 0.0f;
    for (int i = 0; i < items_per_thread; ++i) {
        int idx = tid + i * blockDim.x;
        if (idx < cols) {
            float val = expf(__half2float(row_data[idx]) - global_max);
            row_data[idx] = __float2half(val);  // Temporary storage
            local_sum += val;
        }
    }
    
    // Reduce to find global sum
    sdata[tid] = local_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    float global_sum = sdata[0];
    float inv_sum = 1.0f / global_sum;
    
    // Normalize
    for (int i = 0; i < items_per_thread; ++i) {
        int idx = tid + i * blockDim.x;
        if (idx < cols) {
            row_data[idx] = __float2half(__half2float(row_data[idx]) * inv_sum);
        }
    }
}

// ============================================================================
// Vector Operations
// ============================================================================

__global__ void add_kernel(__half* a, const __half* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = __hadd(a[idx], b[idx]);
    }
}

__global__ void add_kernel_vec(__half2* a, const __half2* b, size_t n2) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) {
        a[idx] = __hadd2(a[idx], b[idx]);
    }
}

__global__ void scale_kernel(__half* x, float alpha, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = __float2half(__half2float(x[idx]) * alpha);
    }
}

__global__ void scale_kernel_vec(__half2* x, float alpha, size_t n2) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) {
        __half2 alpha2 = __float2half2_rn(alpha);
        x[idx] = __hmul2(x[idx], alpha2);
    }
}

// Fused multiply-add: x = alpha * x + beta * y
__global__ void axpby_kernel(__half* x, const __half* y, float alpha, float beta, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fx = __half2float(x[idx]);
        float fy = __half2float(y[idx]);
        x[idx] = __float2half(alpha * fx + beta * fy);
    }
}

// ============================================================================
// Bias Operations
// ============================================================================

// Add bias vector to each row: out[i,j] = x[i,j] + bias[j]
__global__ void add_bias_kernel(__half* x, const __half* bias, int rows, int cols) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        x[idx] = __hadd(x[idx], bias[col]);
    }
}

// ============================================================================
// FP32 <-> FP16 Conversion
// ============================================================================

__global__ void convert_fp32_to_fp16_kernel(const float* src, __half* dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

__global__ void convert_fp16_to_fp32_kernel(const __half* src, float* dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __half2float(src[idx]);
    }
}

// ============================================================================
// Loss Functions
// ============================================================================

// Cross-entropy loss gradient (for softmax output)
// grad = output - target
__global__ void cross_entropy_grad_kernel(__half* grad, const __half* output, 
                                          const __half* target, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad[idx] = __hsub(output[idx], target[idx]);
    }
}

// Frame-level cross-entropy for Kaldi (sparse labels)
__global__ void frame_ce_grad_kernel(__half* grad, const __half* output,
                                     const int* labels, int batch_size, int num_classes) {
    int frame = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= batch_size) return;
    
    int label = labels[frame];
    __half* frame_grad = grad + frame * num_classes;
    const __half* frame_out = output + frame * num_classes;
    
    for (int c = 0; c < num_classes; ++c) {
        __half target = (c == label) ? __float2half(1.0f) : __float2half(0.0f);
        frame_grad[c] = __hsub(frame_out[c], target);
    }
}

// ============================================================================
// Gradient Clipping
// ============================================================================

// Check for inf/nan in gradients (for loss scaling)
__global__ void check_overflow_kernel(const __half* x, int* overflow_flag, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __half2float(x[idx]);
        if (isinf(val) || isnan(val)) {
            atomicOr(overflow_flag, 1);
        }
    }
}

// Clip gradients by norm
__global__ void clip_grad_norm_kernel(__half* grad, float max_norm, float current_norm, size_t n) {
    if (current_norm <= max_norm) return;
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float scale = max_norm / current_norm;
        grad[idx] = __float2half(__half2float(grad[idx]) * scale);
    }
}

} // namespace kernels

// ============================================================================
// Host-callable wrappers
// ============================================================================

namespace ops {

void relu(__half* x, size_t n, cudaStream_t stream) {
    int blocks = (n + kernels::BLOCK_SIZE - 1) / kernels::BLOCK_SIZE;
    kernels::relu_kernel<<<blocks, kernels::BLOCK_SIZE, 0, stream>>>(x, n);
}

void sigmoid(__half* x, size_t n, cudaStream_t stream) {
    int blocks = (n + kernels::BLOCK_SIZE - 1) / kernels::BLOCK_SIZE;
    kernels::sigmoid_kernel<<<blocks, kernels::BLOCK_SIZE, 0, stream>>>(x, n);
}

void tanh(__half* x, size_t n, cudaStream_t stream) {
    int blocks = (n + kernels::BLOCK_SIZE - 1) / kernels::BLOCK_SIZE;
    kernels::tanh_kernel<<<blocks, kernels::BLOCK_SIZE, 0, stream>>>(x, n);
}

void softmax(__half* x, int rows, int cols, cudaStream_t stream) {
    if (cols <= 1024) {
        // Use shared memory version for reasonable sizes
        int shared_mem_size = kernels::BLOCK_SIZE * sizeof(float);
        kernels::softmax_kernel_shared<<<rows, kernels::BLOCK_SIZE, 
                                         shared_mem_size, stream>>>(x, rows, cols);
    } else {
        // Fall back to simple version for very wide rows
        kernels::softmax_kernel<<<rows, 1, 0, stream>>>(x, rows, cols);
    }
}

void add(__half* a, const __half* b, size_t n, cudaStream_t stream) {
    // Use vectorized version if aligned
    if (n % 2 == 0) {
        int blocks = (n/2 + kernels::BLOCK_SIZE - 1) / kernels::BLOCK_SIZE;
        kernels::add_kernel_vec<<<blocks, kernels::BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<__half2*>(a),
            reinterpret_cast<const __half2*>(b),
            n / 2
        );
    } else {
        int blocks = (n + kernels::BLOCK_SIZE - 1) / kernels::BLOCK_SIZE;
        kernels::add_kernel<<<blocks, kernels::BLOCK_SIZE, 0, stream>>>(a, b, n);
    }
}

void scale(__half* x, float alpha, size_t n, cudaStream_t stream) {
    if (n % 2 == 0) {
        int blocks = (n/2 + kernels::BLOCK_SIZE - 1) / kernels::BLOCK_SIZE;
        kernels::scale_kernel_vec<<<blocks, kernels::BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<__half2*>(x), alpha, n / 2
        );
    } else {
        int blocks = (n + kernels::BLOCK_SIZE - 1) / kernels::BLOCK_SIZE;
        kernels::scale_kernel<<<blocks, kernels::BLOCK_SIZE, 0, stream>>>(x, alpha, n);
    }
}

void addBias(__half* x, const __half* bias, int rows, int cols, cudaStream_t stream) {
    dim3 block(kernels::BLOCK_SIZE);
    dim3 grid((cols + block.x - 1) / block.x, rows);
    kernels::add_bias_kernel<<<grid, block, 0, stream>>>(x, bias, rows, cols);
}

void convertFP32ToFP16(const float* src, __half* dst, size_t n, cudaStream_t stream) {
    int blocks = (n + kernels::BLOCK_SIZE - 1) / kernels::BLOCK_SIZE;
    kernels::convert_fp32_to_fp16_kernel<<<blocks, kernels::BLOCK_SIZE, 0, stream>>>(src, dst, n);
}

void convertFP16ToFP32(const __half* src, float* dst, size_t n, cudaStream_t stream) {
    int blocks = (n + kernels::BLOCK_SIZE - 1) / kernels::BLOCK_SIZE;
    kernels::convert_fp16_to_fp32_kernel<<<blocks, kernels::BLOCK_SIZE, 0, stream>>>(src, dst, n);
}

bool checkOverflow(const __half* x, size_t n, cudaStream_t stream) {
    int* d_flag;
    cudaMalloc(&d_flag, sizeof(int));
    cudaMemset(d_flag, 0, sizeof(int));
    
    int blocks = (n + kernels::BLOCK_SIZE - 1) / kernels::BLOCK_SIZE;
    kernels::check_overflow_kernel<<<blocks, kernels::BLOCK_SIZE, 0, stream>>>(x, d_flag, n);
    
    int h_flag;
    cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_flag);
    
    return h_flag != 0;
}

} // namespace ops
} // namespace kaldi_fp16
