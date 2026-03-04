// Bridge CUDA functions: direct FP16 transfer, CSR, pinned memory, combined batch
// Supplements existing cgo_interface.cu with DataLoader integration

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

static __thread char g_bridge_error[512] = {0};

static void bridge_set_error(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_bridge_error, sizeof(g_bridge_error), fmt, args);
    va_end(args);
}

extern "C"
{

    const char *bridge_last_error()
    {
        return g_bridge_error[0] ? g_bridge_error : NULL;
    }

    void bridge_clear_error()
    {
        g_bridge_error[0] = 0;
    }

    // ============================================================================
    // GPU Init / Info
    // ============================================================================

    int bridge_gpu_init(int device_id)
    {
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess)
        {
            bridge_set_error("cudaSetDevice(%d): %s", device_id, cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

    int bridge_gpu_get_free_memory(size_t *free_bytes, size_t *total_bytes)
    {
        cudaError_t err = cudaMemGetInfo(free_bytes, total_bytes);
        if (err != cudaSuccess)
        {
            bridge_set_error("cudaMemGetInfo: %s", cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

    int bridge_gpu_sync()
    {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            bridge_set_error("cudaDeviceSynchronize: %s", cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

    // ============================================================================
    // Memory Allocation
    // ============================================================================

    void *bridge_gpu_malloc(size_t bytes)
    {
        void *ptr = NULL;
        cudaError_t err = cudaMalloc(&ptr, bytes);
        if (err != cudaSuccess)
        {
            bridge_set_error("cudaMalloc(%zu): %s", bytes, cudaGetErrorString(err));
            return NULL;
        }
        return ptr;
    }

    void bridge_gpu_free(void *ptr)
    {
        if (ptr)
            cudaFree(ptr);
    }

    // Pinned (page-locked) memory - 2x faster PCIe transfers
    void *bridge_host_alloc(size_t bytes)
    {
        void *ptr = NULL;
        cudaError_t err = cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
        if (err != cudaSuccess)
        {
            bridge_set_error("cudaHostAlloc(%zu): %s", bytes, cudaGetErrorString(err));
            return NULL;
        }
        return ptr;
    }

    void bridge_host_free(void *ptr)
    {
        if (ptr)
            cudaFreeHost(ptr);
    }

    // ============================================================================
    // Direct FP16 Transfer (CPU uint16 → GPU half)
    // No conversion needed - uint16 and __half have same bit layout
    // ============================================================================

    int bridge_transfer_fp16(void *dst_device, const uint16_t *src_host, size_t count)
    {
        size_t bytes = count * sizeof(uint16_t); // == sizeof(__half)
        cudaError_t err = cudaMemcpy(dst_device, src_host, bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            bridge_set_error("cudaMemcpy FP16 H2D (%zu): %s", count, cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

    // Read FP16 from GPU back to CPU
    int bridge_read_fp16(uint16_t *dst_host, const void *src_device, size_t count)
    {
        size_t bytes = count * sizeof(uint16_t);
        cudaError_t err = cudaMemcpy(dst_host, src_device, bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            bridge_set_error("cudaMemcpy FP16 D2H (%zu): %s", count, cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

    // Transfer int32 data (CSR indices)
    int bridge_transfer_int32(void *dst_device, const int32_t *src_host, size_t count)
    {
        size_t bytes = count * sizeof(int32_t);
        cudaError_t err = cudaMemcpy(dst_device, src_host, bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            bridge_set_error("cudaMemcpy int32 H2D (%zu): %s", count, cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

    // Transfer float32 data (CSR weights)
    int bridge_transfer_float32(void *dst_device, const float *src_host, size_t count)
    {
        size_t bytes = count * sizeof(float);
        cudaError_t err = cudaMemcpy(dst_device, src_host, bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            bridge_set_error("cudaMemcpy float32 H2D (%zu): %s", count, cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

    // ============================================================================
    // Combined Batch Transfer
    // One big cudaMalloc + one big cudaMemcpy instead of many small ones
    // ============================================================================

    // Layout of combined GPU buffer:
    // [features_fp16 | ivectors_fp16 | csr_row_ptr | csr_col_idx | csr_labels | csr_weights]
    // All aligned to 256 bytes for GPU efficiency

    typedef struct
    {
        // Device pointers into the combined buffer
        void *d_features;    // __half*, [total_frames * feat_dim]
        void *d_ivectors;    // __half*, [batch_size * ivec_dim]
        void *d_csr_row_ptr; // int32_t*, [num_states + 1]
        void *d_csr_col_idx; // int32_t*, [num_arcs]
        void *d_csr_labels;  // int32_t*, [num_arcs]
        void *d_csr_weights; // float*, [num_arcs]

        // The combined buffer
        void *d_buffer; // single allocation
        size_t total_bytes;

        // Sizes for each section
        size_t features_bytes;
        size_t ivectors_bytes;
        size_t csr_rowptr_bytes;
        size_t csr_colidx_bytes;
        size_t csr_labels_bytes;
        size_t csr_weights_bytes;
    } GPUBatchPtrs;

    static size_t align256(size_t bytes)
    {
        return (bytes + 255) & ~255;
    }

    // Allocate combined GPU buffer and set up pointers
    int bridge_batch_alloc(
        int total_frames, int feat_dim,
        int batch_size, int ivec_dim,
        int num_states, int num_arcs,
        GPUBatchPtrs *out)
    {
        memset(out, 0, sizeof(GPUBatchPtrs));

        // Calculate aligned sizes
        out->features_bytes = align256(total_frames * feat_dim * sizeof(uint16_t));
        out->ivectors_bytes = align256(batch_size * ivec_dim * sizeof(uint16_t));
        out->csr_rowptr_bytes = align256((num_states + 1) * sizeof(int32_t));
        out->csr_colidx_bytes = align256(num_arcs * sizeof(int32_t));
        out->csr_labels_bytes = align256(num_arcs * sizeof(int32_t));
        out->csr_weights_bytes = align256(num_arcs * sizeof(float));

        out->total_bytes = out->features_bytes + out->ivectors_bytes +
                           out->csr_rowptr_bytes + out->csr_colidx_bytes +
                           out->csr_labels_bytes + out->csr_weights_bytes;

        // Single allocation
        cudaError_t err = cudaMalloc(&out->d_buffer, out->total_bytes);
        if (err != cudaSuccess)
        {
            bridge_set_error("cudaMalloc combined (%zu bytes): %s",
                             out->total_bytes, cudaGetErrorString(err));
            return -1;
        }

        // Set up pointers
        char *base = (char *)out->d_buffer;
        size_t offset = 0;

        out->d_features = base + offset;
        offset += out->features_bytes;
        out->d_ivectors = base + offset;
        offset += out->ivectors_bytes;
        out->d_csr_row_ptr = base + offset;
        offset += out->csr_rowptr_bytes;
        out->d_csr_col_idx = base + offset;
        offset += out->csr_colidx_bytes;
        out->d_csr_labels = base + offset;
        offset += out->csr_labels_bytes;
        out->d_csr_weights = base + offset;

        return 0;
    }

    // Transfer all batch data in combined calls
    // host_buf should contain: [features_fp16 | ivectors_fp16 | rowptr | colidx | labels | weights]
    // packed with the same alignment as GPUBatchPtrs
    int bridge_batch_transfer(const GPUBatchPtrs *ptrs, const void *host_buf, size_t total_bytes)
    {
        cudaError_t err = cudaMemcpy(ptrs->d_buffer, host_buf, total_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            bridge_set_error("cudaMemcpy batch (%zu bytes): %s",
                             total_bytes, cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

    // Free combined buffer
    void bridge_batch_free(GPUBatchPtrs *ptrs)
    {
        if (ptrs && ptrs->d_buffer)
        {
            cudaFree(ptrs->d_buffer);
            memset(ptrs, 0, sizeof(GPUBatchPtrs));
        }
    }

    extern "C" void bridge_gpu_memset(void *ptr, int value, size_t bytes)
    {
        cudaMemset(ptr, value, bytes);
    }

    // ============================================================================
    // FP16 ↔ FP32 conversion kernels on GPU
    // ============================================================================

    __global__ void kernel_fp16_to_fp32(float *dst, const __half *src, size_t n)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
        {
            dst[idx] = __half2float(src[idx]);
        }
    }

    __global__ void kernel_fp32_to_fp16(__half *dst, const float *src, size_t n)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
        {
            dst[idx] = __float2half(src[idx]);
        }
    }

    int bridge_fp16_to_fp32_gpu(float *dst_device, const void *src_device, size_t count)
    {
        int threads = 256;
        int blocks = (count + threads - 1) / threads;
        kernel_fp16_to_fp32<<<blocks, threads>>>(dst_device, (const __half *)src_device, count);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            bridge_set_error("fp16_to_fp32 kernel: %s", cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

    int bridge_fp32_to_fp16_gpu(void *dst_device, const float *src_device, size_t count)
    {
        int threads = 256;
        int blocks = (count + threads - 1) / threads;
        kernel_fp32_to_fp16<<<blocks, threads>>>((__half *)dst_device, src_device, count);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            bridge_set_error("fp32_to_fp16 kernel: %s", cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

} // extern "C"
