// Forward pass compute operations: GEMM, activations, batchnorm, etc.
// Uses cuBLAS for GEMM with FP16 Tensor Cores, custom kernels for element-wise ops

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>

static __thread char g_ops_error[512] = {0};

static void ops_set_error(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_ops_error, sizeof(g_ops_error), fmt, args);
    va_end(args);
}

// ============================================================
// CUDA kernels (must be outside extern "C")
// ============================================================

__global__ void kernel_relu(__half *data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        __half val = data[idx];
        if (__hlt(val, __float2half(0.0f)))
        {
            data[idx] = __float2half(0.0f);
        }
    }
}

__global__ void kernel_sigmoid(__half *data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float x = __half2float(data[idx]);
        data[idx] = __float2half(1.0f / (1.0f + expf(-x)));
    }
}

__global__ void kernel_tanh(__half *data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float x = __half2float(data[idx]);
        data[idx] = __float2half(tanhf(x));
    }
}

__global__ void kernel_clipped_relu(__half *data, int n, float ceiling)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float x = __half2float(data[idx]);
        x = fmaxf(0.0f, fminf(x, ceiling));
        data[idx] = __float2half(x);
    }
}

__global__ void kernel_softmax(__half *data, int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows)
        return;

    __half *rowData = data + row * cols;

    // Find max (for numerical stability)
    float maxVal = -1e30f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
    {
        float v = __half2float(rowData[j]);
        if (v > maxVal)
            maxVal = v;
    }

    // Warp reduce to find global max
    __shared__ float shared_max;
    if (threadIdx.x == 0)
        shared_max = -1e30f;
    __syncthreads();
    atomicMax((int *)&shared_max, __float_as_int(maxVal));
    __syncthreads();
    maxVal = shared_max;

    // Compute exp and sum
    float localSum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
    {
        float v = expf(__half2float(rowData[j]) - maxVal);
        rowData[j] = __float2half(v);
        localSum += v;
    }

    __shared__ float shared_sum;
    if (threadIdx.x == 0)
        shared_sum = 0.0f;
    __syncthreads();
    atomicAdd(&shared_sum, localSum);
    __syncthreads();

    // Normalize
    float invSum = 1.0f / shared_sum;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
    {
        rowData[j] = __float2half(__half2float(rowData[j]) * invSum);
    }
}

__global__ void kernel_log_softmax(__half *data, int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows)
        return;

    __half *rowData = data + row * cols;

    // Find max
    float maxVal = -1e30f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
    {
        float v = __half2float(rowData[j]);
        if (v > maxVal)
            maxVal = v;
    }

    __shared__ float shared_max;
    if (threadIdx.x == 0)
        shared_max = -1e30f;
    __syncthreads();
    atomicMax((int *)&shared_max, __float_as_int(maxVal));
    __syncthreads();
    maxVal = shared_max;

    // Sum of exp
    float localSum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
    {
        localSum += expf(__half2float(rowData[j]) - maxVal);
    }

    __shared__ float shared_sum;
    if (threadIdx.x == 0)
        shared_sum = 0.0f;
    __syncthreads();
    atomicAdd(&shared_sum, localSum);
    __syncthreads();

    float logSumExp = maxVal + logf(shared_sum);

    // x - log_sum_exp
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
    {
        rowData[j] = __float2half(__half2float(rowData[j]) - logSumExp);
    }
}

// BatchNorm forward (inference mode)
// For each element: y = gamma * (x - mean) / sqrt(var + eps) + beta
// x: [T x D] FP16, mean/var/gamma/beta: [D] FP32
__global__ void kernel_batchnorm_forward(
    __half *x, int T, int D,
    const float *mean, const float *var,
    const float *gamma, const float *beta,
    float epsilon)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * D;
    if (idx >= total)
        return;

    int d = idx % D;
    float val = __half2float(x[idx]);
    float norm = (val - mean[d]) / sqrtf(var[d] + epsilon);
    val = gamma[d] * norm + beta[d];
    x[idx] = __float2half(val);
}

// Kaldi-style BatchNorm with target_rms
// Normalizes to zero mean, unit variance, then scales to target_rms
__global__ void kernel_batchnorm_rms(
    __half *x, int T, int D,
    const float *mean, const float *var,
    float target_rms, float epsilon)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * D)
        return;

    int d = idx % D;
    float val = __half2float(x[idx]);
    float norm = (val - mean[d]) / sqrtf(var[d] + epsilon);
    x[idx] = __float2half(norm * target_rms);
}

// dst = alpha * src + beta * dst
__global__ void kernel_add_scaled(__half *dst, const __half *src, int n,
                                  float alpha, float beta)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float d = __half2float(dst[idx]);
        float s = __half2float(src[idx]);
        dst[idx] = __float2half(alpha * s + beta * d);
    }
}

__global__ void kernel_add(__half *dst, const __half *src, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float d = __half2float(dst[idx]);
        float s = __half2float(src[idx]);
        dst[idx] = __float2half(d + s);
    }
}

__global__ void kernel_fill(__half *data, int n, float val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        data[idx] = __float2half(val);
    }
}

// Concat cols: copy src columns into dst at offset
// dst[t, dst_col_offset : dst_col_offset + src_cols] = src[t, :]
__global__ void kernel_concat_cols(
    __half *dst, int T, int dst_cols,
    const __half *src, int src_cols,
    int dst_col_offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * src_cols;
    if (idx >= total)
        return;

    int t = idx / src_cols;
    int c = idx % src_cols;
    dst[t * dst_cols + dst_col_offset + c] = src[t * src_cols + c];
}

// Combine feature maps
// Reorder [T x (H*F1 + H*F2)] -> [T x H*(F1+F2)]
__global__ void kernel_combine_feature_maps(
    __half *data, const __half *temp, int T, int total_dim,
    int height, int nf1, int nf2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * total_dim;
    if (idx >= total)
        return;

    int t = idx / total_dim;
    int d = idx % total_dim;

    int total_filters = nf1 + nf2;
    int h = d / total_filters;
    int f = d % total_filters;

    int src_idx;
    if (f < nf1)
    {
        // From first input: stored as [height * nf1] block
        src_idx = t * total_dim + h * nf1 + f;
    }
    else
    {
        // From second input: stored after first block
        src_idx = t * total_dim + height * nf1 + h * nf2 + (f - nf1);
    }

    data[idx] = temp[src_idx];
}

// Subsample rows: copy every stride-th row starting at row_offset
__global__ void kernel_subsample_rows(
    __half *dst, const __half *src,
    int out_rows, int cols, int stride, int row_offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_rows * cols;
    if (idx >= total)
        return;

    int out_row = idx / cols;
    int col = idx % cols;
    int in_row = row_offset + out_row * stride;

    dst[out_row * cols + col] = src[in_row * cols + col];
}

// ============================================================
// C interface
// ============================================================

extern "C"
{

    const char *ops_last_error() { return g_ops_error[0] ? g_ops_error : NULL; }
    void ops_clear_error() { g_ops_error[0] = 0; }

    // ============================================================
    // cuBLAS handle
    // ============================================================

    void *ops_cublas_create()
    {
        cublasHandle_t handle;
        cublasStatus_t status = cublasCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            ops_set_error("cublasCreate failed: %d", (int)status);
            return NULL;
        }
        // Enable Tensor Cores
        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH); // includes tensor ops on Ada
        return (void *)handle;
    }

    void ops_cublas_destroy(void *handle)
    {
        if (handle)
            cublasDestroy((cublasHandle_t)handle);
    }

    // ============================================================
    // GEMM: C = alpha * A * B + beta * C
    //
    // Row-major layout: Kaldi/Go store matrices row-major
    // cuBLAS expects column-major, so we compute: C^T = B^T * A^T
    // which gives us C in row-major.
    //
    // A: [M x K], B: [K x N], C: [M x N]
    // ============================================================

    int ops_gemm(void *handle,
                 int M, int N, int K,
                 float alpha,
                 const void *A, int lda,
                 const void *B, int ldb,
                 float beta,
                 void *C, int ldc)
    {

        cublasHandle_t h = (cublasHandle_t)handle;

        // Row-major -> column-major trick:
        // C_rm(M,N) = A_rm(M,K) * B_rm(K,N)
        // becomes: C_cm(N,M) = B_cm(N,K) * A_cm(K,M)

        cublasStatus_t status = cublasGemmEx(
            h,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, // swapped M,N for row-major
            &alpha,
            B, CUDA_R_16F, N, // B^T as column-major: (N, K) with ld=N
            A, CUDA_R_16F, K, // A^T as column-major: (K, M) with ld=K
            &beta,
            C, CUDA_R_16F, N,             // C^T as column-major: (N, M) with ld=N
            CUBLAS_COMPUTE_32F,           // FP32 accumulation
            CUBLAS_GEMM_DEFAULT_TENSOR_OP // Use Tensor Cores
        );

        if (status != CUBLAS_STATUS_SUCCESS)
        {
            ops_set_error("cublasGemmEx failed: %d (M=%d N=%d K=%d)", (int)status, M, N, K);
            return -1;
        }
        return 0;
    }

    int ops_gemm_strided(void *handle,
                         int M, int N, int K,
                         float alpha,
                         const void *A, int lda, int64_t strideA,
                         const void *B, int ldb, int64_t strideB,
                         float beta,
                         void *C, int ldc, int64_t strideC,
                         int batch_count)
    {

        cublasHandle_t h = (cublasHandle_t)handle;

        cublasStatus_t status = cublasGemmStridedBatchedEx(
            h,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            B, CUDA_R_16F, N, strideB,
            A, CUDA_R_16F, K, strideA,
            &beta,
            C, CUDA_R_16F, N, strideC,
            batch_count,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        if (status != CUBLAS_STATUS_SUCCESS)
        {
            ops_set_error("cublasGemmStridedBatchedEx failed: %d", (int)status);
            return -1;
        }
        return 0;
    }

    // ============================================================
    // Activations
    // ============================================================

    int ops_relu(void *data, int count)
    {
        int threads = 256;
        int blocks = (count + threads - 1) / threads;
        kernel_relu<<<blocks, threads>>>((__half *)data, count);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            ops_set_error("relu kernel: %s", cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

    int ops_sigmoid(void *data, int count)
    {
        int threads = 256;
        int blocks = (count + threads - 1) / threads;
        kernel_sigmoid<<<blocks, threads>>>((__half *)data, count);
        return cudaGetLastError() == cudaSuccess ? 0 : -1;
    }

    int ops_tanh_act(void *data, int count)
    {
        int threads = 256;
        int blocks = (count + threads - 1) / threads;
        kernel_tanh<<<blocks, threads>>>((__half *)data, count);
        return cudaGetLastError() == cudaSuccess ? 0 : -1;
    }

    int ops_clipped_relu(void *data, int count, float ceiling)
    {
        int threads = 256;
        int blocks = (count + threads - 1) / threads;
        kernel_clipped_relu<<<blocks, threads>>>((__half *)data, count, ceiling);
        return cudaGetLastError() == cudaSuccess ? 0 : -1;
    }

    // ============================================================
    // Softmax / Log-softmax (per-row)
    // ============================================================

    int ops_softmax(void *data, int rows, int cols)
    {
        int threads = min(cols, 256);
        kernel_softmax<<<rows, threads>>>((__half *)data, rows, cols);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            ops_set_error("softmax kernel: %s", cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

    int ops_log_softmax(void *data, int rows, int cols)
    {
        int threads = min(cols, 256);
        kernel_log_softmax<<<rows, threads>>>((__half *)data, rows, cols);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            ops_set_error("log_softmax kernel: %s", cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

    // ============================================================
    // BatchNorm
    // ============================================================

    int ops_batchnorm_forward(void *x, int T, int D,
                              const float *mean, const float *var,
                              const float *gamma, const float *beta,
                              float epsilon)
    {
        int total = T * D;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        kernel_batchnorm_forward<<<blocks, threads>>>(
            (__half *)x, T, D, mean, var, gamma, beta, epsilon);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            ops_set_error("batchnorm kernel: %s", cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

    int ops_batchnorm_forward_rms(void *x, int T, int D,
                                  const float *mean, const float *var,
                                  float target_rms, float epsilon)
    {
        int total = T * D;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        kernel_batchnorm_rms<<<blocks, threads>>>(
            (__half *)x, T, D, mean, var, target_rms, epsilon);
        return cudaGetLastError() == cudaSuccess ? 0 : -1;
    }

    // ============================================================
    // Element-wise operations
    // ============================================================

    int ops_add_scaled(void *dst, const void *src, int count, float alpha, float beta)
    {
        int threads = 256;
        int blocks = (count + threads - 1) / threads;
        kernel_add_scaled<<<blocks, threads>>>((__half *)dst, (const __half *)src, count, alpha, beta);
        return cudaGetLastError() == cudaSuccess ? 0 : -1;
    }

    int ops_add(void *dst, const void *src, int count)
    {
        int threads = 256;
        int blocks = (count + threads - 1) / threads;
        kernel_add<<<blocks, threads>>>((__half *)dst, (const __half *)src, count);
        return cudaGetLastError() == cudaSuccess ? 0 : -1;
    }

    int ops_copy(void *dst, const void *src, int count)
    {
        cudaError_t err = cudaMemcpy(dst, src, count * sizeof(__half), cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess)
        {
            ops_set_error("copy: %s", cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

    int ops_fill(void *dst, int count, float val)
    {
        int threads = 256;
        int blocks = (count + threads - 1) / threads;
        kernel_fill<<<blocks, threads>>>((__half *)dst, count, val);
        return cudaGetLastError() == cudaSuccess ? 0 : -1;
    }

    // ============================================================
    // Concat cols
    // ============================================================

    int ops_concat_cols(void *dst, int T, int dst_cols,
                        const void *src, int src_cols,
                        int dst_col_offset)
    {
        int total = T * src_cols;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        kernel_concat_cols<<<blocks, threads>>>(
            (__half *)dst, T, dst_cols, (const __half *)src, src_cols, dst_col_offset);
        return cudaGetLastError() == cudaSuccess ? 0 : -1;
    }

    // ============================================================
    // Combine feature maps
    // ============================================================

    int ops_combine_feature_maps(void *data, int T, int total_dim,
                                 int height, int nf1, int nf2)
    {
        int total = T * total_dim;

        // Need temp buffer for the reorder
        __half *temp;
        cudaError_t err = cudaMalloc(&temp, total * sizeof(__half));
        if (err != cudaSuccess)
        {
            ops_set_error("combine alloc: %s", cudaGetErrorString(err));
            return -1;
        }

        // Copy data to temp
        cudaMemcpy(temp, data, total * sizeof(__half), cudaMemcpyDeviceToDevice);

        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        kernel_combine_feature_maps<<<blocks, threads>>>(
            (__half *)data, temp, T, total_dim, height, nf1, nf2);

        cudaFree(temp);

        return cudaGetLastError() == cudaSuccess ? 0 : -1;
    }

    // ============================================================
    // Subsample rows
    // ============================================================

    void ops_subsample_rows(void *dst, const void *src,
                            int in_rows, int cols,
                            int stride, int row_offset)
    {
        int out_rows = (in_rows - row_offset + stride - 1) / stride;
        int total = out_rows * cols;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        kernel_subsample_rows<<<blocks, threads>>>(
            (__half *)dst, (const __half *)src,
            out_rows, cols, stride, row_offset);
    }

} // extern "C"
