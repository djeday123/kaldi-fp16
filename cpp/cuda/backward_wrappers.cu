/**
 * @file backward_wrappers.cu
 * @brief C API wrappers for backward pass operations
 *
 * Self-contained file — includes its own kernels because
 * kernels.cu uses C++ namespaces (kaldi_fp16::kernels)
 * which aren't accessible from a separate translation unit.
 *
 * Provides:
 *   ops_relu_backward     — grad *= (x > 0)
 *   ops_sigmoid_backward  — grad *= out * (1 - out)
 *   ops_tanh_backward     — grad *= (1 - out^2)
 *   ops_transpose         — dst[N×M] = src[M×N]^T
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdarg>

// ============================================================
// Error handling (same pattern as ops.cu)
// ============================================================

static __thread char bw_error_buf[256];
static __thread bool bw_has_error = false;

static void bw_set_error(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vsnprintf(bw_error_buf, sizeof(bw_error_buf), fmt, args);
    va_end(args);
    bw_has_error = true;
}

// ============================================================
// Kernels
// ============================================================

__global__ void bw_relu_backward_kernel(const __half *x, __half *grad, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        __half zero = __float2half(0.0f);
        grad[idx] = __hgt(x[idx], zero) ? grad[idx] : zero;
    }
}

__global__ void bw_sigmoid_backward_kernel(const __half *output, __half *grad, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        __half out = output[idx];
        __half one = __float2half(1.0f);
        __half one_minus_out = __hsub(one, out);
        grad[idx] = __hmul(__hmul(grad[idx], out), one_minus_out);
    }
}

__global__ void bw_tanh_backward_kernel(const __half *output, __half *grad, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        __half out = output[idx];
        __half one = __float2half(1.0f);
        __half out_sq = __hmul(out, out);
        grad[idx] = __hmul(grad[idx], __hsub(one, out_sq));
    }
}

__global__ void bw_transpose_kernel(const __half *src, __half *dst, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total)
        return;

    int r = idx / cols;
    int c = idx % cols;
    dst[c * rows + r] = src[r * cols + c];
}

__global__ void bw_maxpool1d_backward_kernel(const __half *grad_output, const int *indices, __half *grad_input, int batch_size, int time_in, int time_out, int channels)
{
    int b = blockIdx.z;
    int t_out = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (b >= batch_size || t_out >= time_out || c >= channels)
        return;

    int out_idx = b * time_out * channels + t_out * channels + c;
    int max_t = indices[out_idx];
    int in_idx = b * time_in * channels + max_t * channels + c;

    atomicAdd(reinterpret_cast<float *>(&grad_input[in_idx]),
              __half2float(grad_output[out_idx]));
}

__global__ void bw_batchnorm_backward_kernel(const __half *grad_out, __half *grad_in, const float *gamma, const float *variance, float eps, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total)
        return;

    int d = idx % cols;
    float scale = gamma[d] / sqrtf(variance[d] + eps);
    float g = __half2float(grad_out[idx]);
    grad_in[idx] = __float2half(g * scale);
}

// FP16 → FP32 conversion kernel
__global__ void bw_fp16_to_fp32_kernel(const __half *src, float *dst, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        dst[idx] = __half2float(src[idx]);
}

// SGD update kernel:
//   velocity = momentum * velocity + grad
//   w_fp32 -= lr * velocity
//   w_fp16 = fp16(w_fp32)
__global__ void bw_sgd_update_kernel(float *w_fp32, __half *w_fp16, const __half *grad_fp16, float *velocity, float lr, float momentum, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    float g = __half2float(grad_fp16[idx]);
    float v = momentum * velocity[idx] + g;
    velocity[idx] = v;

    float w = w_fp32[idx] - lr * v;
    w_fp32[idx] = w;
    w_fp16[idx] = __float2half(w);
}

// ============================================================
// C API
// ============================================================

extern "C"
{

    int ops_relu_backward(const void *x, void *grad, int count)
    {
        int threads = 256;
        int blocks = (count + threads - 1) / threads;
        bw_relu_backward_kernel<<<blocks, threads>>>(
            (const __half *)x, (__half *)grad, count);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            bw_set_error("relu_backward: %s", cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

    int ops_sigmoid_backward(const void *output, void *grad, int count)
    {
        int threads = 256;
        int blocks = (count + threads - 1) / threads;
        bw_sigmoid_backward_kernel<<<blocks, threads>>>(
            (const __half *)output, (__half *)grad, count);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            bw_set_error("sigmoid_backward: %s", cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

    int ops_tanh_backward(const void *output, void *grad, int count)
    {
        int threads = 256;
        int blocks = (count + threads - 1) / threads;
        bw_tanh_backward_kernel<<<blocks, threads>>>(
            (const __half *)output, (__half *)grad, count);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            bw_set_error("tanh_backward: %s", cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

    int ops_transpose(const void *src, void *dst, int M, int N)
    {
        int total = M * N;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        bw_transpose_kernel<<<blocks, threads>>>(
            (const __half *)src, (__half *)dst, M, N);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            bw_set_error("transpose: %s", cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

    void launch_maxpool1d_backward_fp16(const void *grad_output, const void *indices, void *grad_input, int batch_size, int time_in, int time_out, int channels, void *stream)
    {
        dim3 block(16, 16);
        dim3 grid(
            (time_out + block.x - 1) / block.x,
            (channels + block.y - 1) / block.y,
            batch_size);

        bw_maxpool1d_backward_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
            (const __half *)grad_output,
            (const int *)indices,
            (__half *)grad_input,
            batch_size, time_in, time_out, channels);
    }

    int ops_batchnorm_backward(
        const void *grad_out,
        void *grad_in,
        const float *gamma,
        const float *variance,
        float eps,
        int rows,
        int cols)
    {
        int total = rows * cols;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        bw_batchnorm_backward_kernel<<<blocks, threads>>>(
            (const __half *)grad_out,
            (__half *)grad_in,
            gamma,
            variance,
            eps,
            rows, cols);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            bw_set_error("batchnorm_backward: %s", cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

    int ops_fp16_to_fp32(const void *src, float *dst, int count)
    {
        int threads = 256;
        int blocks = (count + threads - 1) / threads;
        bw_fp16_to_fp32_kernel<<<blocks, threads>>>(
            (const __half *)src, dst, count);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            bw_set_error("fp16_to_fp32: %s", cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

    int ops_sgd_update(
        float *w_fp32,
        void *w_fp16,
        const void *grad_fp16,
        float *velocity,
        float lr,
        float momentum,
        int count)
    {
        int threads = 256;
        int blocks = (count + threads - 1) / threads;
        bw_sgd_update_kernel<<<blocks, threads>>>(
            w_fp32, (__half *)w_fp16, (const __half *)grad_fp16,
            velocity, lr, momentum, count);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            bw_set_error("sgd_update: %s", cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

} // extern "C"
