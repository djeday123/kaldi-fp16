#ifndef KALDI_FP16_OPS_H
#define KALDI_FP16_OPS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // Subsample rows: copy every stride-th row starting at row_offset
    // dst: [out_rows x cols] FP16
    // src: [in_rows x cols] FP16
    // out_rows = (in_rows - row_offset + stride - 1) / stride
    void ops_subsample_rows(void *dst, const void *src,
                            int in_rows, int cols,
                            int stride, int row_offset);

    // ============================================================
    // cuBLAS handle
    // ============================================================

    void *ops_cublas_create(void);
    void ops_cublas_destroy(void *handle);

    // ============================================================
    // GEMM: C = alpha * A * B + beta * C
    // All FP16 storage, FP32 accumulation (Tensor Cores)
    //
    // A: [M x K], B: [K x N], C: [M x N]  (row-major)
    // ============================================================

    int ops_gemm(void *handle,
                 int M, int N, int K,
                 float alpha,
                 const void *A, int lda,
                 const void *B, int ldb,
                 float beta,
                 void *C, int ldc);

    // Batched: C[i] = alpha * A[i] * B + beta * C[i]
    // Same B for all batches (weight sharing)
    // A_array: array of pointers, or strided
    int ops_gemm_strided(void *handle,
                         int M, int N, int K,
                         float alpha,
                         const void *A, int lda, int64_t strideA,
                         const void *B, int ldb, int64_t strideB,
                         float beta,
                         void *C, int ldc, int64_t strideC,
                         int batch_count);

    // ============================================================
    // Activations (in-place on FP16 data)
    // ============================================================

    int ops_relu(void *data, int count);
    int ops_sigmoid(void *data, int count);
    int ops_tanh_act(void *data, int count);
    int ops_clipped_relu(void *data, int count, float ceiling); // min(max(0,x), ceiling)

    // ============================================================
    // Softmax / Log-softmax
    // data: [rows x cols], applies per-row
    // ============================================================

    int ops_softmax(void *data, int rows, int cols);
    int ops_log_softmax(void *data, int rows, int cols);

    // ============================================================
    // BatchNorm forward (inference mode)
    //
    // x: [T x D] FP16 in/out
    // mean, var: [D] FP32 running stats
    // gamma, beta: [D] FP32 (scale, offset)
    // epsilon: small constant
    // ============================================================

    int ops_batchnorm_forward(void *x, int T, int D,
                              const float *mean, const float *var,
                              const float *gamma, const float *beta,
                              float epsilon);

    // BatchNorm with target_rms (Kaldi-style: normalize then scale to target_rms)
    int ops_batchnorm_forward_rms(void *x, int T, int D,
                                  const float *mean, const float *var,
                                  float target_rms, float epsilon);

    // ============================================================
    // Element-wise operations on FP16
    // ============================================================

    // dst = alpha * src + beta * dst
    int ops_add_scaled(void *dst, const void *src, int count,
                       float alpha, float beta);

    // dst += src (simple add)
    int ops_add(void *dst, const void *src, int count);

    // dst = src (copy)
    int ops_copy(void *dst, const void *src, int count);

    // dst = val (fill)
    int ops_fill(void *dst, int count, float val);

    // ============================================================
    // Concat / Append (for multi-branch merge)
    // Appends along columns: [T x D1] + [T x D2] → [T x (D1+D2)]
    // ============================================================

    int ops_concat_cols(void *dst, int T, int dst_cols,
                        const void *src, int src_cols,
                        int dst_col_offset);

    // ============================================================
    // TDNN-F specific: time-stride splicing
    // Creates spliced input by selecting frames at stride offsets
    //
    // src: [T x D], dst: [T x D] (same size, but shifted frames)
    // For stride=3: frame t of dst gets frame t*3 of src (or nearest)
    // ============================================================

    // Not needed if we handle time context in the affine layer
    // (Kaldi does this via descriptor, not a separate operation)

    // ============================================================
    // Combine feature maps (reorder dimensions)
    // Interleaves features from two concatenated inputs
    //
    // input: [T x (H*F1 + H*F2)] where F1, F2 are filter counts
    // output: [T x H*(F1+F2)] reordered so each height position
    //         has all its filters together
    // ============================================================

    int ops_combine_feature_maps(void *data, int T, int total_dim,
                                 int height, int num_filters1, int num_filters2);

    // ============================================================
    // Error handling
    // ============================================================

    const char *ops_last_error(void);
    void ops_clear_error(void);

    // ============================================================
    // Backward activations (add to cpp/include/ops.h)
    // Insert BEFORE the closing #ifdef __cplusplus / } / #endif
    // ============================================================

    // ============================================================
    // Activation backward passes
    //
    // All modify grad in-place: grad *= d(activation)/d(input)
    // ============================================================

    // ReLU backward: grad[i] = x[i] > 0 ? grad[i] : 0
    // x: [count] FP16 pre-activation (saved from forward)
    // grad: [count] FP16 in/out
    int ops_relu_backward(const void *x, void *grad, int count);

    // Sigmoid backward: grad[i] *= output[i] * (1 - output[i])
    // output: [count] FP16 post-activation (saved from forward)
    // grad: [count] FP16 in/out
    int ops_sigmoid_backward(const void *output, void *grad, int count);

    // Tanh backward: grad[i] *= (1 - output[i]^2)
    // output: [count] FP16 post-activation (saved from forward)
    // grad: [count] FP16 in/out
    int ops_tanh_backward(const void *output, void *grad, int count);

    // Transpose: dst[N×M] = src[M×N]^T
    int ops_transpose(const void *src, void *dst, int M, int N);

    // BatchNorm backward (inference): gradIn = gradOut * gamma / sqrt(var + eps)
    int ops_batchnorm_backward(const void *grad_out, void *grad_in,
                               const float *gamma, const float *variance,
                               float eps, int rows, int cols);

    // FP16 → FP32 conversion
    int ops_fp16_to_fp32(const void *src, float *dst, int count);

    // SGD update with FP32 master weights
    int ops_sgd_update(float *w_fp32, void *w_fp16, const void *grad_fp16,
                       float *velocity, float lr, float momentum, int count);

    int ops_slice_cols(const void *src, int T, int src_cols,
                       void *dst, int dst_cols, int src_col_offset);

#ifdef __cplusplus
}
#endif

#endif // KALDI_FP16_OPS_H