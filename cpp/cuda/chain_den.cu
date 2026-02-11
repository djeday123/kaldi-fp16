// Native Denominator for Chain LF-MMI — Probability Space
//
// Replaces Kaldi wrapper (libkaldi_den.so) with native CUDA.
// Implements all 6 Kaldi-specific features:
//   1. Probability space (exp(nnet_output)), not log-domain
//   2. Initial probs from 100-iter HMM warmup
//   3. Leaky HMM (1e-05 coefficient)
//   4. Arbitrary scaling (divide by sum each frame)
//   5. All states are final (weight = 1.0)
//   6. transition_prob = exp(-arc.weight)
//
// Reference: Kaldi src/chain/chain-denominator.cc
//            Kaldi src/chain/chain-den-graph.cc

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <float.h>

#include "chain_den.h"

// ============================================================
// Error handling
// ============================================================

static __thread char g_den_error[512] = {0};

static void den_set_error(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_den_error, sizeof(g_den_error), fmt, args);
    va_end(args);
}

extern "C"
{
    const char *den_last_error() { return g_den_error[0] ? g_den_error : NULL; }
    void den_clear_error() { g_den_error[0] = 0; }
}

// ============================================================
// GPU Sum Reduction
// ============================================================

__global__ void kernel_sum_reduce(const float *data, float *partial_sums, int n)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + tid;

    float val = 0.0f;
    if (idx < n)
        val += data[idx];
    if (idx + blockDim.x < n)
        val += data[idx + blockDim.x];
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        partial_sums[blockIdx.x] = sdata[0];
}

__global__ void kernel_final_reduce(float *partial_sums, float *result, int n)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    sdata[tid] = (tid < n) ? partial_sums[tid] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        *result = sdata[0];
}

// Host helper: sum GPU array -> host float
static float gpu_sum(const float *d_data, int n,
                     float *d_partial, float *d_result)
{
    int threads = 256;
    int blocks = (n + threads * 2 - 1) / (threads * 2);
    if (blocks < 1)
        blocks = 1;

    kernel_sum_reduce<<<blocks, threads, threads * sizeof(float)>>>(
        d_data, d_partial, n);

    int final_threads = 1;
    while (final_threads < blocks)
        final_threads *= 2;
    if (final_threads > 1024)
        final_threads = 1024;

    kernel_final_reduce<<<1, final_threads, final_threads * sizeof(float)>>>(
        d_partial, d_result, blocks);

    float result;
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}

// ============================================================
// Kernels
// ============================================================

// exp(nnet_output) with clamp [-30, 30] (matches Kaldi ApplyExpLimited)
__global__ void kernel_apply_exp(float *out, const float *in, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    float val = in[idx];
    val = fmaxf(-30.0f, fminf(30.0f, val));
    out[idx] = expf(val);
}

// alpha[s] = initial_probs[s]
__global__ void kernel_init_alpha(float *alpha, const float *initial_probs, int S)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < S)
        alpha[s] = initial_probs[s];
}

// Forward leaky HMM (AlphaDash):
//   alpha'[s] = alpha[s] + tot_alpha * leaky * init[s]
// where tot_alpha = sum(alpha) and init[s] is per-state
__global__ void kernel_alpha_dash(
    float *alpha_dash,
    const float *alpha,
    const float *initial_probs,
    float alpha_sum,
    float leaky_coeff,
    int S)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < S)
        alpha_dash[s] = alpha[s] + alpha_sum * leaky_coeff * initial_probs[s];
}

// Forward transition propagation (one timestep)
// alpha_next[dst] += alpha_dash[src] * trans_prob * exp_nnet[pdf]
__global__ void kernel_den_forward_transitions(
    float *alpha_next,
    const float *alpha_dash,
    const float *exp_nnet_t,
    const int32_t *src_states,
    const int32_t *dst_states,
    const int32_t *pdf_ids,
    const float *transition_probs,
    int num_trans, int P)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_trans)
        return;

    int src = src_states[idx];
    int dst = dst_states[idx];
    int pdf = pdf_ids[idx];
    float tp = transition_probs[idx];

    float src_val = alpha_dash[src];
    if (src_val <= 0.0f)
        return;

    float nnet_val = (pdf >= 0 && pdf < P) ? exp_nnet_t[pdf] : 0.0f;
    float contribution = src_val * tp * nnet_val;

    if (contribution > 0.0f)
        atomicAdd(&alpha_next[dst], contribution);
}

// Scale array by constant: data[i] *= scale
__global__ void kernel_scale(float *data, float scale, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] *= scale;
}

// Fill array with constant value
__global__ void kernel_fill(float *data, float val, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] = val;
}

// Element-wise multiply: out[i] = a[i] * b[i]
__global__ void kernel_elementwise_mul(float *out, const float *a, const float *b, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        out[idx] = a[idx] * b[idx];
}

// Add scalar to all elements: data[i] += val
__global__ void kernel_add_scalar(float *data, float val, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] += val;
}

// Backward transition propagation (one timestep)
// beta_dash[src] += beta[t+1][dst] * trans_prob * exp_nnet[pdf]
__global__ void kernel_den_backward_transitions(
    float *beta_dash,
    const float *beta_next,
    const float *exp_nnet_t,
    const int32_t *src_states,
    const int32_t *dst_states,
    const int32_t *pdf_ids,
    const float *transition_probs,
    int num_trans, int P)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_trans)
        return;

    int src = src_states[idx];
    int dst = dst_states[idx];
    int pdf = pdf_ids[idx];
    float tp = transition_probs[idx];

    float dst_val = beta_next[dst];
    if (dst_val <= 0.0f)
        return;

    float nnet_val = (pdf >= 0 && pdf < P) ? exp_nnet_t[pdf] : 0.0f;
    float contribution = dst_val * tp * nnet_val;

    if (contribution > 0.0f)
        atomicAdd(&beta_dash[src], contribution);
}

// Posteriors: grad[pdf] += alpha_dash[src] * tp * exp_nnet[pdf] * beta[t+1][dst]
__global__ void kernel_den_posteriors(
    float *grad_t,
    const float *alpha_dash,
    const float *beta_next,
    const float *exp_nnet_t,
    const int32_t *src_states,
    const int32_t *dst_states,
    const int32_t *pdf_ids,
    const float *transition_probs,
    int num_trans, int P)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_trans)
        return;

    int src = src_states[idx];
    int dst = dst_states[idx];
    int pdf = pdf_ids[idx];
    float tp = transition_probs[idx];

    if (pdf < 0 || pdf >= P)
        return;

    float a = alpha_dash[src];
    float b = beta_next[dst];
    if (a <= 0.0f || b <= 0.0f)
        return;

    float nnet_val = exp_nnet_t[pdf];
    float posterior = a * tp * nnet_val * b;

    if (posterior > 0.0f)
        atomicAdd(&grad_t[pdf], posterior);
}

// ============================================================
// C Interface
// ============================================================

extern "C"
{

    int den_fst_upload(
        DenFstGPU *fst,
        const int32_t *src, const int32_t *dst,
        const int32_t *pdf, const float *trans_probs,
        int num_trans, int num_states, int num_pdfs)
    {
        fst->num_transitions = num_trans;
        fst->num_states = num_states;
        fst->num_pdfs = num_pdfs;

        size_t int_bytes = (size_t)num_trans * sizeof(int32_t);
        size_t flt_bytes = (size_t)num_trans * sizeof(float);

        cudaError_t err;

        err = cudaMalloc(&fst->src_states, int_bytes);
        if (err != cudaSuccess)
        {
            den_set_error("malloc src: %s", cudaGetErrorString(err));
            return -1;
        }
        err = cudaMalloc(&fst->dst_states, int_bytes);
        if (err != cudaSuccess)
        {
            den_set_error("malloc dst: %s", cudaGetErrorString(err));
            return -1;
        }
        err = cudaMalloc(&fst->pdf_ids, int_bytes);
        if (err != cudaSuccess)
        {
            den_set_error("malloc pdf: %s", cudaGetErrorString(err));
            return -1;
        }
        err = cudaMalloc(&fst->transition_probs, flt_bytes);
        if (err != cudaSuccess)
        {
            den_set_error("malloc tp: %s", cudaGetErrorString(err));
            return -1;
        }

        cudaMemcpy(fst->src_states, src, int_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(fst->dst_states, dst, int_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(fst->pdf_ids, pdf, int_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(fst->transition_probs, trans_probs, flt_bytes, cudaMemcpyHostToDevice);

        return 0;
    }

    void den_fst_free(DenFstGPU *fst)
    {
        if (fst->src_states)
        {
            cudaFree(fst->src_states);
            fst->src_states = NULL;
        }
        if (fst->dst_states)
        {
            cudaFree(fst->dst_states);
            fst->dst_states = NULL;
        }
        if (fst->pdf_ids)
        {
            cudaFree(fst->pdf_ids);
            fst->pdf_ids = NULL;
        }
        if (fst->transition_probs)
        {
            cudaFree(fst->transition_probs);
            fst->transition_probs = NULL;
        }
    }

    // ============================================================
    // Forward only
    // ============================================================

    float den_forward(
        const DenFstGPU *fst,
        const float *h_nnet_output,
        const float *h_initial_probs,
        int T,
        float leaky_hmm_coeff)
    {
        int S = fst->num_states;
        int P = fst->num_pdfs;
        int A = fst->num_transitions;
        int threads = 256;

        float *d_exp_nnet, *d_initial;
        float *d_alpha_cur, *d_alpha_next, *d_alpha_dash;
        float *d_partial, *d_result;

        size_t nnet_bytes = (size_t)T * P * sizeof(float);
        size_t state_bytes = (size_t)S * sizeof(float);

        cudaMalloc(&d_exp_nnet, nnet_bytes);
        cudaMalloc(&d_initial, state_bytes);
        cudaMalloc(&d_alpha_cur, state_bytes);
        cudaMalloc(&d_alpha_next, state_bytes);
        cudaMalloc(&d_alpha_dash, state_bytes);

        int max_reduce_blocks = (S + 511) / 512 + 1;
        cudaMalloc(&d_partial, max_reduce_blocks * sizeof(float));
        cudaMalloc(&d_result, sizeof(float));

        // Upload + exp(nnet_output)
        float *d_nnet_raw;
        cudaMalloc(&d_nnet_raw, nnet_bytes);
        cudaMemcpy(d_nnet_raw, h_nnet_output, nnet_bytes, cudaMemcpyHostToDevice);
        int nnet_total = T * P;
        int blocks_exp = (nnet_total + threads - 1) / threads;
        kernel_apply_exp<<<blocks_exp, threads>>>(d_exp_nnet, d_nnet_raw, nnet_total);
        cudaFree(d_nnet_raw);

        cudaMemcpy(d_initial, h_initial_probs, state_bytes, cudaMemcpyHostToDevice);

        int blocks_s = (S + threads - 1) / threads;
        int blocks_a = (A + threads - 1) / threads;
        if (blocks_a < 1)
            blocks_a = 1;

        // === Kaldi Forward ===
        // AlphaFirstFrame: alpha[0] = initial_probs
        kernel_init_alpha<<<blocks_s, threads>>>(d_alpha_cur, d_initial, S);

        // AlphaDash(0): alpha_sum = sum(alpha[0]), alpha' = alpha + leaky*init*alpha_sum
        float alpha_sum = gpu_sum(d_alpha_cur, S, d_partial, d_result);
        kernel_alpha_dash<<<blocks_s, threads>>>(
            d_alpha_dash, d_alpha_cur, d_initial,
            alpha_sum, leaky_hmm_coeff, S);

        double log_correction = 0.0;

        // Forward: t = 1..T
        for (int t = 1; t <= T; t++)
        {
            // AlphaGeneralFrame: alpha[t] = transitions(alpha'[t-1], prob[t-1]) / alpha_sum[t-1]
            cudaMemset(d_alpha_next, 0, state_bytes);

            kernel_den_forward_transitions<<<blocks_a, threads>>>(
                d_alpha_next, d_alpha_dash,
                d_exp_nnet + (size_t)(t - 1) * P,
                fst->src_states, fst->dst_states,
                fst->pdf_ids, fst->transition_probs,
                A, P);

            if (alpha_sum > 0.0f)
            {
                kernel_scale<<<blocks_s, threads>>>(d_alpha_next, 1.0f / alpha_sum, S);
                log_correction += log((double)alpha_sum);
            }

            // AlphaDash(t)
            alpha_sum = gpu_sum(d_alpha_next, S, d_partial, d_result);
            kernel_alpha_dash<<<blocks_s, threads>>>(
                d_alpha_dash, d_alpha_next, d_initial,
                alpha_sum, leaky_hmm_coeff, S);

            // Swap
            float *tmp = d_alpha_cur;
            d_alpha_cur = d_alpha_next;
            d_alpha_next = tmp;
        }

        // total_prob = sum(alpha'[T])
        float total_prob = gpu_sum(d_alpha_dash, S, d_partial, d_result);
        float log_prob = (float)(log((double)total_prob) + log_correction);

        cudaFree(d_exp_nnet);
        cudaFree(d_initial);
        cudaFree(d_alpha_cur);
        cudaFree(d_alpha_next);
        cudaFree(d_alpha_dash);
        cudaFree(d_partial);
        cudaFree(d_result);

        return log_prob;
    }

    // ============================================================
    // Forward-Backward with gradients
    //
    // Follows Kaldi chain-denominator.cc exactly:
    //
    // Forward:
    //   alpha[0] = init
    //   alpha_sum[t] = sum(alpha[t])
    //   alpha'[t] = alpha[t] + leaky * init * alpha_sum[t]
    //   alpha[t+1] = transitions(alpha'[t], prob[t]) / alpha_sum[t]
    //
    // Backward:
    //   beta'[T] = 1 / total_prob  (all states, uniform)
    //   tot_beta[t] = leaky * dot(init, beta'[t])  <-- SCALAR
    //   beta[t] = beta'[t] + tot_beta[t]           <-- add scalar to all
    //   beta'[t] = transitions_bwd(beta[t+1], prob[t]) / alpha_sum[t]
    //
    // Posteriors:
    //   gamma[t][pdf] += alpha'[t][src] * beta[t+1][dst] * tp * prob[t][pdf] / alpha_sum[t]
    //
    // ============================================================

    float den_forward_backward(
        const DenFstGPU *fst,
        const float *h_nnet_output,
        const float *h_initial_probs,
        int T,
        float leaky_hmm_coeff,
        float *h_grad_output)
    {
        int S = fst->num_states;
        int P = fst->num_pdfs;
        int A = fst->num_transitions;
        int threads = 256;

        size_t nnet_bytes = (size_t)T * P * sizeof(float);
        size_t state_bytes = (size_t)S * sizeof(float);

        // --- GPU allocations ---
        float *d_exp_nnet, *d_initial;
        float *d_alpha_dash_all;  // [(T+1) x S] store alpha' for all frames
        float *d_alpha_dash_temp; // [S] current alpha' (working buffer)
        float *d_alpha_temp;      // [S] current raw alpha
        float *d_beta_dash;       // [S] beta' (before leaky)
        float *d_beta;            // [S] beta (after leaky)
        float *d_temp;            // [S] temp for dot product
        float *d_grad;            // [T x P]
        float *d_partial, *d_result;

        cudaMalloc(&d_exp_nnet, nnet_bytes);
        cudaMalloc(&d_initial, state_bytes);
        cudaMalloc(&d_alpha_dash_all, (size_t)(T + 1) * S * sizeof(float));
        cudaMalloc(&d_alpha_dash_temp, state_bytes);
        cudaMalloc(&d_alpha_temp, state_bytes);
        cudaMalloc(&d_beta_dash, state_bytes);
        cudaMalloc(&d_beta, state_bytes);
        cudaMalloc(&d_temp, state_bytes);
        cudaMalloc(&d_grad, nnet_bytes);
        cudaMemset(d_grad, 0, nnet_bytes);

        int max_reduce_blocks = (S + 511) / 512 + 1;
        cudaMalloc(&d_partial, max_reduce_blocks * sizeof(float));
        cudaMalloc(&d_result, sizeof(float));

        // Upload + exp
        float *d_nnet_raw;
        cudaMalloc(&d_nnet_raw, nnet_bytes);
        cudaMemcpy(d_nnet_raw, h_nnet_output, nnet_bytes, cudaMemcpyHostToDevice);
        int nnet_total = T * P;
        int blocks_exp = (nnet_total + threads - 1) / threads;
        kernel_apply_exp<<<blocks_exp, threads>>>(d_exp_nnet, d_nnet_raw, nnet_total);
        cudaFree(d_nnet_raw);

        cudaMemcpy(d_initial, h_initial_probs, state_bytes, cudaMemcpyHostToDevice);

        int blocks_s = (S + threads - 1) / threads;
        int blocks_a = (A + threads - 1) / threads;
        int blocks_p = (P + threads - 1) / threads;
        if (blocks_a < 1)
            blocks_a = 1;

        // alpha_sum[t] stored on CPU for backward
        float *h_alpha_sum = (float *)malloc((T + 1) * sizeof(float));

        // ============================================================
        // FORWARD
        // ============================================================

        // AlphaFirstFrame: alpha[0] = initial_probs
        kernel_init_alpha<<<blocks_s, threads>>>(d_alpha_temp, d_initial, S);

        // AlphaDash(0)
        h_alpha_sum[0] = gpu_sum(d_alpha_temp, S, d_partial, d_result);
        kernel_alpha_dash<<<blocks_s, threads>>>(
            d_alpha_dash_temp, d_alpha_temp, d_initial,
            h_alpha_sum[0], leaky_hmm_coeff, S);

        // Save alpha'[0]
        cudaMemcpy(d_alpha_dash_all, d_alpha_dash_temp, state_bytes, cudaMemcpyDeviceToDevice);

        double log_correction = 0.0;

        for (int t = 1; t <= T; t++)
        {
            // AlphaGeneralFrame(t): alpha[t] = transitions(alpha'[t-1], prob[t-1]) / alpha_sum[t-1]
            cudaMemset(d_alpha_temp, 0, state_bytes);

            kernel_den_forward_transitions<<<blocks_a, threads>>>(
                d_alpha_temp, d_alpha_dash_temp,
                d_exp_nnet + (size_t)(t - 1) * P,
                fst->src_states, fst->dst_states,
                fst->pdf_ids, fst->transition_probs,
                A, P);

            if (h_alpha_sum[t - 1] > 0.0f)
            {
                kernel_scale<<<blocks_s, threads>>>(d_alpha_temp, 1.0f / h_alpha_sum[t - 1], S);
                log_correction += log((double)h_alpha_sum[t - 1]);
            }

            // AlphaDash(t)
            h_alpha_sum[t] = gpu_sum(d_alpha_temp, S, d_partial, d_result);
            kernel_alpha_dash<<<blocks_s, threads>>>(
                d_alpha_dash_temp, d_alpha_temp, d_initial,
                h_alpha_sum[t], leaky_hmm_coeff, S);

            // Save alpha'[t]
            cudaMemcpy(d_alpha_dash_all + (size_t)t * S, d_alpha_dash_temp,
                       state_bytes, cudaMemcpyDeviceToDevice);
        }

        // total_prob = sum(alpha'[T])
        float total_prob = gpu_sum(d_alpha_dash_temp, S, d_partial, d_result);
        float log_prob = (float)(log((double)total_prob) + log_correction);

        // ============================================================
        // BACKWARD
        //
        // Kaldi convention:
        //   beta'  = "beta_dash" = before leaky
        //   beta   = after leaky = used in transition propagation
        //
        // Key formulas from Kaldi header comment:
        //   beta'(T, i) = 1 / total_prob
        //   tot_beta(t) = leaky * sum_i(init(i) * beta'(t,i))  <-- scalar
        //   beta(t, i)  = beta'(t, i) + tot_beta(t)            <-- add scalar
        //   beta'(t, i) = sum_j(beta(t+1,j) * p * x(t,n) / alpha_sum(t))
        //   gamma(t, n) += alpha'(t,i) * beta(t+1,j) * p * x(t,n) / alpha_sum(t)
        // ============================================================

        // BetaDashLastFrame: beta'[T] = 1 / total_prob
        float inv_tot_prob = (total_prob > 0.0f) ? 1.0f / total_prob : 0.0f;
        kernel_fill<<<blocks_s, threads>>>(d_beta_dash, inv_tot_prob, S);

        // Beta(T): tot_beta = leaky * dot(init, beta'[T]), beta = beta' + tot_beta
        kernel_elementwise_mul<<<blocks_s, threads>>>(d_temp, d_initial, d_beta_dash, S);
        float tot_beta_scalar = leaky_hmm_coeff * gpu_sum(d_temp, S, d_partial, d_result);
        cudaMemcpy(d_beta, d_beta_dash, state_bytes, cudaMemcpyDeviceToDevice);
        kernel_add_scalar<<<blocks_s, threads>>>(d_beta, tot_beta_scalar, S);

        for (int t = T - 1; t >= 0; t--)
        {
            // BetaDashGeneralFrame(t):
            //   beta'[t][src] = sum_arcs(beta[t+1][dst] * tp * prob[t][pdf]) / alpha_sum[t]
            //   gamma[t][pdf] += alpha'[t][src] * beta[t+1][dst] * tp * prob[t][pdf] / alpha_sum[t]

            // 1. Compute beta'[t] from transitions (using beta[t+1])
            cudaMemset(d_beta_dash, 0, state_bytes);

            kernel_den_backward_transitions<<<blocks_a, threads>>>(
                d_beta_dash, d_beta,
                d_exp_nnet + (size_t)t * P,
                fst->src_states, fst->dst_states,
                fst->pdf_ids, fst->transition_probs,
                A, P);

            // Scale by 1/alpha_sum[t]
            if (h_alpha_sum[t] > 0.0f)
                kernel_scale<<<blocks_s, threads>>>(d_beta_dash, 1.0f / h_alpha_sum[t], S);

            // 2. Compute posteriors: alpha'[t] * beta[t+1] * tp * prob / alpha_sum[t]
            float *saved_alpha_dash = d_alpha_dash_all + (size_t)t * S;

            kernel_den_posteriors<<<blocks_a, threads>>>(
                d_grad + (size_t)t * P,
                saved_alpha_dash, d_beta,
                d_exp_nnet + (size_t)t * P,
                fst->src_states, fst->dst_states,
                fst->pdf_ids, fst->transition_probs,
                A, P);

            // Scale posteriors by 1/alpha_sum[t]
            if (h_alpha_sum[t] > 0.0f)
                kernel_scale<<<blocks_p, threads>>>(
                    d_grad + (size_t)t * P,
                    1.0f / h_alpha_sum[t], P);

            // 3. Beta(t): convert beta'[t] to beta[t]
            //    tot_beta = leaky * dot(init, beta'[t])
            //    beta[t] = beta'[t] + tot_beta
            kernel_elementwise_mul<<<blocks_s, threads>>>(d_temp, d_initial, d_beta_dash, S);
            tot_beta_scalar = leaky_hmm_coeff * gpu_sum(d_temp, S, d_partial, d_result);
            cudaMemcpy(d_beta, d_beta_dash, state_bytes, cudaMemcpyDeviceToDevice);
            kernel_add_scalar<<<blocks_s, threads>>>(d_beta, tot_beta_scalar, S);
        }

        // Copy gradients to host
        if (h_grad_output)
            cudaMemcpy(h_grad_output, d_grad, nnet_bytes, cudaMemcpyDeviceToHost);

        // Cleanup
        cudaFree(d_exp_nnet);
        cudaFree(d_initial);
        cudaFree(d_alpha_dash_all);
        cudaFree(d_alpha_dash_temp);
        cudaFree(d_alpha_temp);
        cudaFree(d_beta_dash);
        cudaFree(d_beta);
        cudaFree(d_temp);
        cudaFree(d_grad);
        cudaFree(d_partial);
        cudaFree(d_result);
        free(h_alpha_sum);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            den_set_error("den_forward_backward: %s", cudaGetErrorString(err));
            return -1e30f;
        }

        return log_prob;
    }

} // extern "C"
