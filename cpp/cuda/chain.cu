// Chain LF-MMI loss CUDA implementation
//
// Forward-backward algorithm on FSTs in log-semiring
// Parallelized over arcs at each time step
//
// References:
//   - Kaldi: src/chain/chain-kernels.cu
//   - Povey et al., "Purely Sequence-Trained Neural Networks for ASR"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>
#include <float.h>
#include <math.h>

#include "chain.h"

static const float kLogZero = -1.0e+30f;

static __thread char g_chain_error[512] = {0};

static void chain_set_error(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_chain_error, sizeof(g_chain_error), fmt, args);
    va_end(args);
}

extern "C"
{
    const char *chain_last_error() { return g_chain_error[0] ? g_chain_error : NULL; }
    void chain_clear_error() { g_chain_error[0] = 0; }
}

// ============================================================
// Log-domain atomic add (CAS loop)
// log(exp(addr) + exp(val))
// ============================================================

__device__ void atomicLogAdd(float *addr, float val)
{
    if (val <= kLogZero)
        return; // adding zero in log-domain

    int *addr_as_int = (int *)addr;
    int old = *addr_as_int;
    int assumed;
    do
    {
        assumed = old;
        float old_val = __int_as_float(assumed);
        float new_val;
        if (old_val <= kLogZero)
        {
            new_val = val;
        }
        else
        {
            float max_v = fmaxf(old_val, val);
            float min_v = fminf(old_val, val);
            new_val = max_v + log1pf(expf(min_v - max_v));
        }
        old = atomicCAS(addr_as_int, assumed, __float_as_int(new_val));
    } while (assumed != old);
}

// ============================================================
// Forward pass kernel
//
// For each arc (src → dst, pdf, weight), at time t:
//   alpha[t+1][dst] = log_add(alpha[t+1][dst],
//                              alpha[t][src] + nnet[t][pdf-1] + weight)
//
// Parallelized over arcs. Each thread handles one arc.
// ============================================================

__global__ void kernel_chain_forward(
    float *alpha,              // [(T+1) x S]
    const __half *nnet_output, // [T x P]
    const int32_t *row_ptr,    // [S + 1]
    const int32_t *col_idx,    // [A] — destination state
    const int32_t *labels,     // [A] — pdf-id (1-indexed)
    const float *weights,      // [A] — log-weight
    int S, int P, int A,       // num_states, num_pdfs, num_arcs
    int t                      // current time step
)
{
    int arc_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (arc_idx >= A)
        return;

    // Find source state for this arc
    // Binary search in row_ptr to find state s where row_ptr[s] <= arc_idx < row_ptr[s+1]
    int lo = 0, hi = S - 1;
    while (lo < hi)
    {
        int mid = (lo + hi + 1) / 2;
        if (row_ptr[mid] <= arc_idx)
            lo = mid;
        else
            hi = mid - 1;
    }
    int src = lo;

    // Check arc belongs to src
    if (arc_idx < row_ptr[src] || arc_idx >= row_ptr[src + 1])
        return;

    int dst = col_idx[arc_idx];
    int pdf = labels[arc_idx]; // 1-indexed
    float w = weights[arc_idx];

    // Skip epsilon arcs for now (pdf == 0)
    // In Kaldi chain, all arcs should have pdf labels
    if (pdf <= 0 || pdf > P)
        return;

    float src_alpha = alpha[t * S + src];
    if (src_alpha <= kLogZero)
        return;

    // nnet output for this pdf at time t
    float nnet_val = __half2float(nnet_output[t * P + (pdf - 1)]);

    float arc_val = src_alpha + nnet_val + w;

    // Accumulate into alpha[t+1][dst]
    atomicLogAdd(&alpha[(t + 1) * S + dst], arc_val);
}

// ============================================================
// Backward pass kernel
//
// For each arc (src → dst, pdf, weight), at time t:
//   beta[t][src] = log_add(beta[t][src],
//                           beta[t+1][dst] + nnet[t][pdf-1] + weight)
// ============================================================

__global__ void kernel_chain_backward(
    float *beta,
    const __half *nnet_output,
    const int32_t *row_ptr,
    const int32_t *col_idx,
    const int32_t *labels,
    const float *weights,
    int S, int P, int A,
    int t)
{
    int arc_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (arc_idx >= A)
        return;

    // Find source state
    int lo = 0, hi = S - 1;
    while (lo < hi)
    {
        int mid = (lo + hi + 1) / 2;
        if (row_ptr[mid] <= arc_idx)
            lo = mid;
        else
            hi = mid - 1;
    }
    int src = lo;
    if (arc_idx < row_ptr[src] || arc_idx >= row_ptr[src + 1])
        return;

    int dst = col_idx[arc_idx];
    int pdf = labels[arc_idx];
    float w = weights[arc_idx];

    if (pdf <= 0 || pdf > P)
        return;

    float dst_beta = beta[(t + 1) * S + dst];
    if (dst_beta <= kLogZero)
        return;

    float nnet_val = __half2float(nnet_output[t * P + (pdf - 1)]);
    float arc_val = dst_beta + nnet_val + w;

    atomicLogAdd(&beta[t * S + src], arc_val);
}

// ============================================================
// Initialize alpha/beta to kLogZero
// ============================================================

__global__ void kernel_fill_logzero(float *data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        data[idx] = kLogZero;
    }
}

// ============================================================
// Set alpha[0][start_state] = 0
// ============================================================

__global__ void kernel_set_start(float *alpha, int S, int start_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        alpha[start_state] = 0.0f; // log(1) = 0
    }
}

// ============================================================
// Set beta[T][final_states] = final_weights
// ============================================================

__global__ void kernel_set_finals(
    float *beta, int S, int T,
    const int32_t *final_states, const float *final_weights, int num_final)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_final)
    {
        int s = final_states[idx];
        float w = final_weights[idx];
        beta[T * S + s] = w;
    }
}

// ============================================================
// Compute total log-prob from alpha[T] and final states
// ============================================================

__global__ void kernel_total_logprob(
    const float *alpha, int S, int T,
    const int32_t *final_states, const float *final_weights, int num_final,
    float *result)
{
    // Single-threaded for simplicity (num_final is typically small)
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        float total = kLogZero;
        for (int i = 0; i < num_final; i++)
        {
            int s = final_states[i];
            float val = alpha[T * S + s] + final_weights[i];
            if (total <= kLogZero)
            {
                total = val;
            }
            else if (val > kLogZero)
            {
                float max_v = fmaxf(total, val);
                float min_v = fminf(total, val);
                total = max_v + log1pf(expf(min_v - max_v));
            }
        }
        *result = total;
    }
}

// ============================================================
// Compute posteriors
//
// For each arc at time t with label=pdf:
//   post[t][pdf-1] += exp(alpha[t][src] + nnet[t][pdf-1] + weight
//                         + beta[t+1][dst] - total_logprob)
// ============================================================

__global__ void kernel_chain_posteriors(
    float *posteriors, // [T x P] FP32 output
    const float *alpha,
    const float *beta,
    const __half *nnet_output,
    const int32_t *row_ptr,
    const int32_t *col_idx,
    const int32_t *labels,
    const float *weights,
    int S, int P, int A,
    int t,
    float total_logprob)
{
    int arc_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (arc_idx >= A)
        return;

    // Find source state
    int lo = 0, hi = S - 1;
    while (lo < hi)
    {
        int mid = (lo + hi + 1) / 2;
        if (row_ptr[mid] <= arc_idx)
            lo = mid;
        else
            hi = mid - 1;
    }
    int src = lo;
    if (arc_idx < row_ptr[src] || arc_idx >= row_ptr[src + 1])
        return;

    int dst = col_idx[arc_idx];
    int pdf = labels[arc_idx];
    float w = weights[arc_idx];

    if (pdf <= 0 || pdf > P)
        return;

    float a = alpha[t * S + src];
    float b = beta[(t + 1) * S + dst];
    if (a <= kLogZero || b <= kLogZero)
        return;

    float nnet_val = __half2float(nnet_output[t * P + (pdf - 1)]);
    float log_post = a + nnet_val + w + b - total_logprob;

    // Clamp to avoid exp overflow
    if (log_post > 0.0f)
        log_post = 0.0f;

    float post = expf(log_post);

    // Accumulate posterior
    atomicAdd(&posteriors[t * P + (pdf - 1)], post);
}

// ============================================================
// Compute gradient: grad = -(num_post - den_post)
// And convert to FP16
// ============================================================

__global__ void kernel_chain_gradient(
    __half *grad_output,
    const float *num_post,
    const float *den_post,
    int total_elements,
    float supervision_weight // typically 1.0
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements)
        return;

    float num = num_post[idx];
    float den = den_post[idx];

    // grad = -(num - den) * weight = (den - num) * weight
    float grad = (den - num) * supervision_weight;

    // Clamp gradient for numerical stability
    grad = fmaxf(-30.0f, fminf(30.0f, grad));

    grad_output[idx] = __float2half(grad);
}

// ============================================================
// C interface implementations
// ============================================================

extern "C"
{

    size_t chain_workspace_bytes(int T, int num_states)
    {
        // alpha: (T+1) * num_states * sizeof(float)
        // beta:  (T+1) * num_states * sizeof(float)
        return 2 * (size_t)(T + 1) * num_states * sizeof(float);
    }

    int chain_forward_backward(
        const void *nnet_output,
        const ChainFstGPU *fst,
        int T, int num_pdfs,
        float *alpha, float *beta,
        float *total_logprob)
    {
        int S = fst->num_states;
        int A = fst->num_arcs;
        int threads = 256;

        // Initialize alpha to -inf
        int alpha_size = (T + 1) * S;
        int blocks_init = (alpha_size + threads - 1) / threads;
        kernel_fill_logzero<<<blocks_init, threads>>>(alpha, alpha_size);

        // Set start state
        kernel_set_start<<<1, 1>>>(alpha, S, fst->start_state);

        // Forward pass: t = 0 .. T-1
        int blocks_arc = (A + threads - 1) / threads;
        for (int t = 0; t < T; t++)
        {
            // Initialize alpha[t+1] row to -inf (already done above for all)
            kernel_chain_forward<<<blocks_arc, threads>>>(
                alpha, (const __half *)nnet_output,
                fst->row_ptr, fst->col_idx, fst->labels, fst->weights,
                S, num_pdfs, A, t);
        }

        // Compute total log-prob from forward
        float *d_total;
        cudaMalloc(&d_total, sizeof(float));
        kernel_total_logprob<<<1, 1>>>(
            alpha, S, T,
            fst->final_states, fst->final_weights, fst->num_final,
            d_total);
        cudaMemcpy(total_logprob, d_total, sizeof(float), cudaMemcpyDeviceToHost);

        // Initialize beta to -inf
        int beta_size = (T + 1) * S;
        int blocks_beta = (beta_size + threads - 1) / threads;
        kernel_fill_logzero<<<blocks_beta, threads>>>(beta, beta_size);

        // Set final states
        int blocks_final = (fst->num_final + threads - 1) / threads;
        if (blocks_final < 1)
            blocks_final = 1;
        kernel_set_finals<<<blocks_final, threads>>>(
            beta, S, T,
            fst->final_states, fst->final_weights, fst->num_final);

        // Backward pass: t = T-1 .. 0
        for (int t = T - 1; t >= 0; t--)
        {
            kernel_chain_backward<<<blocks_arc, threads>>>(
                beta, (const __half *)nnet_output,
                fst->row_ptr, fst->col_idx, fst->labels, fst->weights,
                S, num_pdfs, A, t);
        }

        cudaFree(d_total);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            chain_set_error("chain_forward_backward: %s", cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

    int chain_compute_posteriors(
        const void *nnet_output,
        const ChainFstGPU *fst,
        int T, int num_pdfs,
        const float *alpha, const float *beta,
        float total_logprob,
        float *posteriors)
    {
        int S = fst->num_states;
        int A = fst->num_arcs;
        int threads = 256;

        // Zero posteriors
        cudaMemset(posteriors, 0, T * num_pdfs * sizeof(float));

        // Accumulate posteriors per time step
        int blocks_arc = (A + threads - 1) / threads;
        for (int t = 0; t < T; t++)
        {
            kernel_chain_posteriors<<<blocks_arc, threads>>>(
                posteriors, alpha, beta,
                (const __half *)nnet_output,
                fst->row_ptr, fst->col_idx, fst->labels, fst->weights,
                S, num_pdfs, A, t, total_logprob);
        }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            chain_set_error("chain_compute_posteriors: %s", cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

    int chain_compute_loss(
        const void *nnet_output,
        const ChainFstGPU *num_fst,
        const ChainFstGPU *den_fst,
        int T, int num_pdfs,
        void *grad_output,
        ChainLossResult *result)
    {
        int threads = 256;

        // ---- Numerator ----
        int num_S = num_fst->num_states;
        size_t num_ws = (size_t)(T + 1) * num_S * sizeof(float);

        float *num_alpha, *num_beta;
        cudaError_t err;
        err = cudaMalloc(&num_alpha, num_ws);
        if (err != cudaSuccess)
        {
            chain_set_error("alloc num_alpha: %s", cudaGetErrorString(err));
            return -1;
        }
        err = cudaMalloc(&num_beta, num_ws);
        if (err != cudaSuccess)
        {
            cudaFree(num_alpha);
            chain_set_error("alloc num_beta: %s", cudaGetErrorString(err));
            return -1;
        }

        float num_logprob;
        int ret = chain_forward_backward(nnet_output, num_fst, T, num_pdfs,
                                         num_alpha, num_beta, &num_logprob);
        if (ret != 0)
        {
            cudaFree(num_alpha);
            cudaFree(num_beta);
            return -1;
        }

        // ---- Denominator ----
        int den_S = den_fst->num_states;
        size_t den_ws = (size_t)(T + 1) * den_S * sizeof(float);

        float *den_alpha, *den_beta;
        err = cudaMalloc(&den_alpha, den_ws);
        if (err != cudaSuccess)
        {
            cudaFree(num_alpha);
            cudaFree(num_beta);
            chain_set_error("alloc den_alpha: %s", cudaGetErrorString(err));
            return -1;
        }
        err = cudaMalloc(&den_beta, den_ws);
        if (err != cudaSuccess)
        {
            cudaFree(num_alpha);
            cudaFree(num_beta);
            cudaFree(den_alpha);
            chain_set_error("alloc den_beta: %s", cudaGetErrorString(err));
            return -1;
        }

        float den_logprob;
        ret = chain_forward_backward(nnet_output, den_fst, T, num_pdfs,
                                     den_alpha, den_beta, &den_logprob);
        if (ret != 0)
        {
            cudaFree(num_alpha);
            cudaFree(num_beta);
            cudaFree(den_alpha);
            cudaFree(den_beta);
            return -1;
        }

        // ---- Loss ----
        result->num_logprob = num_logprob;
        result->den_logprob = den_logprob;
        result->loss = -(num_logprob - den_logprob);

        // ---- Gradient (if requested) ----
        if (grad_output)
        {
            size_t post_bytes = (size_t)T * num_pdfs * sizeof(float);
            float *num_post, *den_post;

            err = cudaMalloc(&num_post, post_bytes);
            if (err != cudaSuccess)
                goto cleanup;
            err = cudaMalloc(&den_post, post_bytes);
            if (err != cudaSuccess)
            {
                cudaFree(num_post);
                goto cleanup;
            }

            ret = chain_compute_posteriors(nnet_output, num_fst, T, num_pdfs,
                                           num_alpha, num_beta, num_logprob, num_post);
            if (ret != 0)
            {
                cudaFree(num_post);
                cudaFree(den_post);
                goto cleanup;
            }

            ret = chain_compute_posteriors(nnet_output, den_fst, T, num_pdfs,
                                           den_alpha, den_beta, den_logprob, den_post);
            if (ret != 0)
            {
                cudaFree(num_post);
                cudaFree(den_post);
                goto cleanup;
            }

            // grad = (den_post - num_post), convert to FP16
            int total = T * num_pdfs;
            int blocks = (total + threads - 1) / threads;
            kernel_chain_gradient<<<blocks, threads>>>(
                (__half *)grad_output, num_post, den_post, total, 1.0f);

            cudaFree(num_post);
            cudaFree(den_post);
        }

    cleanup:
        cudaFree(num_alpha);
        cudaFree(num_beta);
        cudaFree(den_alpha);
        cudaFree(den_beta);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            chain_set_error("chain_compute_loss: %s", cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

} // extern "C"
