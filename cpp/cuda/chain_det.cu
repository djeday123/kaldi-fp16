// chain_det.cu — Deterministic chain forward-backward (no atomics)
//
// For scientific comparison with Kaldi's CPU implementation.
// Uses sequential LogAdd in fixed arc order instead of atomic CAS,
// producing bit-identical results regardless of thread scheduling.
//
// Tradeoff: parallelism over arcs -> parallelism over states (forward)
// or sequential per-state (backward/posteriors). Fine for small
// numerator FSTs (~100-200 states, ~200 arcs).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chain.h"

static const float kLogZeroDet = -1.0e+30f;

// ============================================================
// Helper: sequential LogAdd (deterministic)
// ============================================================

__device__ float logadd_det(float a, float b)
{
    if (a <= kLogZeroDet)
        return b;
    if (b <= kLogZeroDet)
        return a;
    float max_v = fmaxf(a, b);
    float min_v = fminf(a, b);
    return max_v + log1pf(expf(min_v - max_v));
}

// ============================================================
// FP32 -> FP16 conversion (local copy, not shared across TUs)
// ============================================================

__global__ void kernel_f32_to_f16_det(const float *in, __half *out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        out[idx] = __float2half(in[idx]);
}

// ============================================================
// Forward pass — one thread per destination state
//
// Uses reverse CSR index: for each dst, iterates incoming arcs
// in sorted arc_idx order. No atomics.
// ============================================================

__global__ void kernel_chain_forward_det(
    float *alpha,
    const __half *nnet_output,
    const int32_t *labels,
    const float *weights,
    const int32_t *rev_row_ptr,
    const int32_t *rev_arc_idx,
    const int32_t *rev_src,
    int S, int P,
    int t)
{
    int dst = blockIdx.x * blockDim.x + threadIdx.x;
    if (dst >= S)
        return;

    float val = kLogZeroDet;

    int start = rev_row_ptr[dst];
    int end = rev_row_ptr[dst + 1];

    for (int i = start; i < end; i++)
    {
        int arc = rev_arc_idx[i];
        int src = rev_src[i];
        int pdf = labels[arc];
        float w = weights[arc];

        if (pdf <= 0 || pdf > P)
            continue;

        float src_alpha = alpha[t * S + src];
        if (src_alpha <= kLogZeroDet)
            continue;

        float nnet_val = __half2float(nnet_output[t * P + (pdf - 1)]);
        float arc_val = src_alpha + nnet_val + w;

        val = logadd_det(val, arc_val);
    }

    alpha[(t + 1) * S + dst] = val;
}

// ============================================================
// Backward pass — one thread per source state
//
// CSR already groups arcs by src, so iterate row_ptr[src]..row_ptr[src+1]
// sequentially. No atomics.
// ============================================================

__global__ void kernel_chain_backward_det(
    float *beta,
    const __half *nnet_output,
    const int32_t *row_ptr,
    const int32_t *col_idx,
    const int32_t *labels,
    const float *weights,
    int S, int P,
    int t)
{
    int src = blockIdx.x * blockDim.x + threadIdx.x;
    if (src >= S)
        return;

    float val = kLogZeroDet;

    for (int a = row_ptr[src]; a < row_ptr[src + 1]; a++)
    {
        int dst = col_idx[a];
        int pdf = labels[a];
        float w = weights[a];

        if (pdf <= 0 || pdf > P)
            continue;

        float dst_beta = beta[(t + 1) * S + dst];
        if (dst_beta <= kLogZeroDet)
            continue;

        float nnet_val = __half2float(nnet_output[t * P + (pdf - 1)]);
        float arc_val = dst_beta + nnet_val + w;

        val = logadd_det(val, arc_val);
    }

    beta[t * S + src] = val;
}

// ============================================================
// Posteriors — single thread, iterates all arcs sequentially
// ============================================================

__global__ void kernel_chain_posteriors_det(
    float *posteriors,
    const float *alpha,
    const float *beta,
    const __half *nnet_output,
    const int32_t *row_ptr,
    const int32_t *col_idx,
    const int32_t *labels,
    const float *weights,
    int S, int P,
    int t,
    float total_logprob)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    for (int src = 0; src < S; src++)
    {
        float a = alpha[t * S + src];
        if (a <= kLogZeroDet)
            continue;

        for (int arc = row_ptr[src]; arc < row_ptr[src + 1]; arc++)
        {
            int dst = col_idx[arc];
            int pdf = labels[arc];
            float w = weights[arc];

            if (pdf <= 0 || pdf > P)
                continue;

            float b = beta[(t + 1) * S + dst];
            if (b <= kLogZeroDet)
                continue;

            float nnet_val = __half2float(nnet_output[t * P + (pdf - 1)]);
            float log_post = a + nnet_val + w + b - total_logprob;

            if (log_post > 0.0f)
                log_post = 0.0f;
            float post = expf(log_post);

            posteriors[t * P + (pdf - 1)] += post;
        }
    }
}

// ============================================================
// Helper kernels
// ============================================================

__global__ void kernel_fill_logzero_det(float *data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] = kLogZeroDet;
}

__global__ void kernel_set_start_det(float *alpha, int S, int start_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
        alpha[start_state] = 0.0f;
}

__global__ void kernel_set_finals_det(
    float *beta, int S, int T,
    const int32_t *final_states, const float *final_weights, int num_final)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_final)
    {
        beta[T * S + final_states[idx]] = final_weights[idx];
    }
}

__global__ void kernel_total_logprob_det(
    const float *alpha, int S, int T,
    const int32_t *final_states, const float *final_weights, int num_final,
    float *result)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    float total = kLogZeroDet;
    for (int i = 0; i < num_final; i++)
    {
        float val = alpha[T * S + final_states[i]] + final_weights[i];
        total = logadd_det(total, val);
    }
    *result = total;
}

// ============================================================
// Build reverse CSR on CPU
// ============================================================

static void build_reverse_csr(
    const int32_t *h_row_ptr, const int32_t *h_col_idx,
    int S, int A,
    int32_t **out_rev_row_ptr, int32_t **out_rev_arc_idx, int32_t **out_rev_src)
{
    int32_t *count = (int32_t *)calloc(S, sizeof(int32_t));
    for (int a = 0; a < A; a++)
    {
        int dst = h_col_idx[a];
        if (dst >= 0 && dst < S)
            count[dst]++;
    }

    int32_t *rev_row_ptr = (int32_t *)malloc((S + 1) * sizeof(int32_t));
    rev_row_ptr[0] = 0;
    for (int s = 0; s < S; s++)
        rev_row_ptr[s + 1] = rev_row_ptr[s] + count[s];

    int32_t *rev_arc_idx = (int32_t *)malloc(A * sizeof(int32_t));
    int32_t *rev_src = (int32_t *)malloc(A * sizeof(int32_t));
    int32_t *pos = (int32_t *)calloc(S, sizeof(int32_t));

    for (int src = 0; src < S; src++)
    {
        for (int a = h_row_ptr[src]; a < h_row_ptr[src + 1]; a++)
        {
            int dst = h_col_idx[a];
            if (dst < 0 || dst >= S)
                continue;
            int slot = rev_row_ptr[dst] + pos[dst];
            rev_arc_idx[slot] = a;
            rev_src[slot] = src;
            pos[dst]++;
        }
    }

    free(count);
    free(pos);
    *out_rev_row_ptr = rev_row_ptr;
    *out_rev_arc_idx = rev_arc_idx;
    *out_rev_src = rev_src;
}

// ============================================================
// C API
// ============================================================

extern "C"
{

    int chain_forward_backward_det(
        const void *nnet_output,
        const ChainFstGPU *fst,
        int T, int num_pdfs,
        float *alpha, float *beta,
        float *total_logprob)
    {
        int S = fst->num_states;
        int A = fst->num_arcs;
        int threads = 256;

        // Build reverse CSR
        int32_t *h_row_ptr = (int32_t *)malloc((S + 1) * sizeof(int32_t));
        int32_t *h_col_idx = (int32_t *)malloc(A * sizeof(int32_t));
        cudaMemcpy(h_row_ptr, fst->row_ptr, (S + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_col_idx, fst->col_idx, A * sizeof(int32_t), cudaMemcpyDeviceToHost);

        int32_t *h_rev_row_ptr, *h_rev_arc_idx, *h_rev_src;
        build_reverse_csr(h_row_ptr, h_col_idx, S, A,
                          &h_rev_row_ptr, &h_rev_arc_idx, &h_rev_src);
        free(h_row_ptr);
        free(h_col_idx);

        int32_t *d_rev_row_ptr, *d_rev_arc_idx, *d_rev_src;
        cudaMalloc(&d_rev_row_ptr, (S + 1) * sizeof(int32_t));
        cudaMalloc(&d_rev_arc_idx, A * sizeof(int32_t));
        cudaMalloc(&d_rev_src, A * sizeof(int32_t));
        cudaMemcpy(d_rev_row_ptr, h_rev_row_ptr, (S + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rev_arc_idx, h_rev_arc_idx, A * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rev_src, h_rev_src, A * sizeof(int32_t), cudaMemcpyHostToDevice);
        free(h_rev_row_ptr);
        free(h_rev_arc_idx);
        free(h_rev_src);

        // Init alpha
        int alpha_size = (T + 1) * S;
        int blocks_init = (alpha_size + threads - 1) / threads;
        kernel_fill_logzero_det<<<blocks_init, threads>>>(alpha, alpha_size);
        kernel_set_start_det<<<1, 1>>>(alpha, S, fst->start_state);

        // Forward: one thread per dst
        int blocks_state = (S + threads - 1) / threads;
        for (int t = 0; t < T; t++)
        {
            kernel_chain_forward_det<<<blocks_state, threads>>>(
                alpha, (const __half *)nnet_output,
                fst->labels, fst->weights,
                d_rev_row_ptr, d_rev_arc_idx, d_rev_src,
                S, num_pdfs, t);
        }

        // Total logprob
        float *d_total;
        cudaMalloc(&d_total, sizeof(float));
        kernel_total_logprob_det<<<1, 1>>>(
            alpha, S, T,
            fst->final_states, fst->final_weights, fst->num_final, d_total);
        cudaMemcpy(total_logprob, d_total, sizeof(float), cudaMemcpyDeviceToHost);

        // Init beta
        int beta_size = (T + 1) * S;
        kernel_fill_logzero_det<<<(beta_size + threads - 1) / threads, threads>>>(beta, beta_size);
        int blocks_final = (fst->num_final + threads - 1) / threads;
        if (blocks_final < 1)
            blocks_final = 1;
        kernel_set_finals_det<<<blocks_final, threads>>>(
            beta, S, T, fst->final_states, fst->final_weights, fst->num_final);

        // Backward: one thread per src
        for (int t = T - 1; t >= 0; t--)
        {
            kernel_chain_backward_det<<<blocks_state, threads>>>(
                beta, (const __half *)nnet_output,
                fst->row_ptr, fst->col_idx, fst->labels, fst->weights,
                S, num_pdfs, t);
        }

        cudaFree(d_rev_row_ptr);
        cudaFree(d_rev_arc_idx);
        cudaFree(d_rev_src);
        cudaFree(d_total);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "chain_forward_backward_det: %s\n", cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

    int chain_compute_posteriors_det(
        const void *nnet_output,
        const ChainFstGPU *fst,
        int T, int num_pdfs,
        const float *alpha, const float *beta,
        float total_logprob,
        float *posteriors)
    {
        cudaMemset(posteriors, 0, T * num_pdfs * sizeof(float));

        for (int t = 0; t < T; t++)
        {
            kernel_chain_posteriors_det<<<1, 1>>>(
                posteriors, alpha, beta,
                (const __half *)nnet_output,
                fst->row_ptr, fst->col_idx, fst->labels, fst->weights,
                fst->num_states, num_pdfs, t, total_logprob);
        }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "chain_compute_posteriors_det: %s\n", cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

    float chain_num_forward_backward_det(
        const int *fst_row_ptr,
        const int *fst_col_idx,
        const float *fst_weights,
        const int *fst_pdf_ids,
        const int *fst_final_states,
        const float *fst_final_weights,
        int num_states,
        int num_arcs,
        int num_final,
        const float *nnet_output,
        float *num_post,
        int T,
        int num_pdfs,
        void *stream)
    {
        (void)stream;

        ChainFstGPU fst;
        fst.row_ptr = (int32_t *)fst_row_ptr;
        fst.col_idx = (int32_t *)fst_col_idx;
        fst.labels = (int32_t *)fst_pdf_ids;
        fst.weights = (float *)fst_weights;
        fst.final_states = (int32_t *)fst_final_states;
        fst.final_weights = (float *)fst_final_weights;
        fst.num_states = num_states;
        fst.num_arcs = num_arcs;
        fst.num_final = num_final;
        fst.start_state = 0;

        // FP32 -> FP16
        int total = T * num_pdfs;
        __half *d_nnet_fp16;
        cudaMalloc(&d_nnet_fp16, total * sizeof(__half));
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        kernel_f32_to_f16_det<<<blocks, threads>>>(nnet_output, d_nnet_fp16, total);

        // Workspace
        size_t ws_bytes = (size_t)(T + 1) * num_states * sizeof(float);
        float *alpha, *beta;
        cudaMalloc(&alpha, ws_bytes);
        cudaMalloc(&beta, ws_bytes);

        float total_logprob;
        int ret = chain_forward_backward_det(d_nnet_fp16, &fst, T, num_pdfs,
                                             alpha, beta, &total_logprob);
        if (ret != 0)
        {
            cudaFree(d_nnet_fp16);
            cudaFree(alpha);
            cudaFree(beta);
            return -1e30f;
        }

        ret = chain_compute_posteriors_det(d_nnet_fp16, &fst, T, num_pdfs,
                                           alpha, beta, total_logprob, num_post);

        cudaFree(d_nnet_fp16);
        cudaFree(alpha);
        cudaFree(beta);

        if (ret != 0)
            return -1e30f;
        return total_logprob;
    }

} // extern "C"
