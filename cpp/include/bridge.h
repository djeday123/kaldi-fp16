#ifndef KALDI_FP16_BRIDGE_H
#define KALDI_FP16_BRIDGE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // Error handling
    const char *bridge_last_error(void);
    void bridge_clear_error(void);

    // GPU init
    int bridge_gpu_init(int device_id);
    int bridge_gpu_get_free_memory(size_t *free_bytes, size_t *total_bytes);
    int bridge_gpu_sync(void);

    // Memory
    void *bridge_gpu_malloc(size_t bytes);
    void bridge_gpu_free(void *ptr);
    void *bridge_host_alloc(size_t bytes); // pinned memory
    void bridge_host_free(void *ptr);

    // Direct transfers
    int bridge_transfer_fp16(void *dst_device, const uint16_t *src_host, size_t count);
    int bridge_read_fp16(uint16_t *dst_host, const void *src_device, size_t count);
    int bridge_transfer_int32(void *dst_device, const int32_t *src_host, size_t count);
    int bridge_transfer_float32(void *dst_device, const float *src_host, size_t count);

    // Combined batch transfer
    typedef struct
    {
        void *d_features;
        void *d_ivectors;
        void *d_csr_row_ptr;
        void *d_csr_col_idx;
        void *d_csr_labels;
        void *d_csr_weights;
        void *d_buffer;
        size_t total_bytes;
        size_t features_bytes;
        size_t ivectors_bytes;
        size_t csr_rowptr_bytes;
        size_t csr_colidx_bytes;
        size_t csr_labels_bytes;
        size_t csr_weights_bytes;
    } GPUBatchPtrs;

    int bridge_batch_alloc(int total_frames, int feat_dim, int batch_size, int ivec_dim,
                           int num_states, int num_arcs, GPUBatchPtrs *out);
    int bridge_batch_transfer(const GPUBatchPtrs *ptrs, const void *host_buf, size_t total_bytes);
    void bridge_batch_free(GPUBatchPtrs *ptrs);
    void bridge_gpu_memset(void *ptr, int value, size_t bytes);

    // GPU conversion kernels
    int bridge_fp16_to_fp32_gpu(float *dst_device, const void *src_device, size_t count);
    int bridge_fp32_to_fp16_gpu(void *dst_device, const float *src_device, size_t count);

#ifdef __cplusplus
}
#endif

#endif // KALDI_FP16_BRIDGE_H