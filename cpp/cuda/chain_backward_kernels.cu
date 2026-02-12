// =============================================================================
// chain_backward_kernels.cu — CUDA kernels for chain loss backward pass
//
// Add these to cpp/cuda/chain.cu (or create as separate file)
// These implement the missing components identified in the architecture review:
//   - PenalizeOutOfRange (from Kaldi chain-training.cc)
//   - Gradient combination (num_post - den_post)
//   - L2 regularization
//   - FP32→FP16 gradient conversion
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>

// Forward declaration (already in chain.cu)
// extern void chain_forward_backward(...);

// ---------------------------------------------------------------------------
// Kernel: PenalizeOutOfRange
// Matches Kaldi's PenalizeOutOfRange() in chain-training.cc
//
// Kaldi applies this to ~50% of frames for efficiency.
// We use a simple hash-based selection: apply to even-indexed rows.
// ---------------------------------------------------------------------------
__global__ void kernel_penalize_out_of_range(
    const float *nnet_output, // [T × num_pdfs]
    float *grad_output,       // [T × num_pdfs] — modified in-place
    float limit,              // 30.0
    float scale,              // 2 * out_of_range_regularize
    int T,
    int num_pdfs,
    int *out_of_range_count) // atomic counter
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * num_pdfs;
    if (idx >= total)
        return;

    int t = idx / num_pdfs;

    // Apply to ~50% of frames (even time indices), matching Kaldi's approach
    // Kaldi uses: if (RandInt(0,1) == 0) continue;
    // We use deterministic: every other frame
    if (t % 2 != 0)
        return;

    float val = nnet_output[idx];

    if (val < -limit)
    {
        // deriv += (-limit - val) * scale
        // This pushes val back toward -limit
        grad_output[idx] += (-limit - val) * scale;
        if (out_of_range_count)
            atomicAdd(out_of_range_count, 1);
    }
    else if (val > limit)
    {
        // deriv += (limit - val) * scale
        // This pushes val back toward +limit
        grad_output[idx] += (limit - val) * scale;
        if (out_of_range_count)
            atomicAdd(out_of_range_count, 1);
    }
}

// ---------------------------------------------------------------------------
// Kernel: Combine gradient = weight * (num_post - den_post)
// Output as FP32 (conversion to FP16 done separately if needed)
// ---------------------------------------------------------------------------
__global__ void kernel_combine_gradient_fp32(
    const float *num_post,
    const float *den_post,
    float weight,
    float *grad_output,
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements)
        return;

    grad_output[idx] = weight * (num_post[idx] - den_post[idx]);
}

// ---------------------------------------------------------------------------
// Kernel: Combine gradient and convert directly to FP16
// grad[i] = (half)(weight * (num_post[i] - den_post[i]))
// ---------------------------------------------------------------------------
__global__ void kernel_combine_gradient_fp16(
    const float *num_post,
    const float *den_post,
    float weight,
    __half *grad_output,
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements)
        return;

    float g = weight * (num_post[idx] - den_post[idx]);
    grad_output[idx] = __float2half(g);
}

// ---------------------------------------------------------------------------
// Kernel: L2 regularization
// grad[i] -= l2_scale * nnet_output[i]
// Also computes partial sum of nnet_output[i]^2 for l2_term
// ---------------------------------------------------------------------------
__global__ void kernel_l2_regularize(
    const float *nnet_output,
    float *grad_output,
    float l2_scale,
    int total_elements,
    float *partial_sq_sum) // [gridDim.x] partial sums
{
    extern __shared__ float sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float sq = 0.0f;
    if (idx < total_elements)
    {
        float val = nnet_output[idx];
        grad_output[idx] -= l2_scale * val;
        sq = val * val;
    }

    // Reduction for l2_term computation
    sdata[tid] = sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0 && partial_sq_sum)
    {
        partial_sq_sum[blockIdx.x] = sdata[0];
    }
}

// ---------------------------------------------------------------------------
// Kernel: FP32 → FP16 conversion
// ---------------------------------------------------------------------------
__global__ void kernel_fp32_to_fp16(
    const float *input,
    __half *output,
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements)
        return;
    output[idx] = __float2half(input[idx]);
}

// ---------------------------------------------------------------------------
// Kernel: Add posterior gradient (for use when grad already has values)
// grad[i] += weight * (num_post[i] - den_post[i])
// ---------------------------------------------------------------------------
__global__ void kernel_add_posterior_gradient(
    const float *num_post,
    const float *den_post,
    float *grad,
    float weight,
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements)
        return;

    grad[idx] += weight * (num_post[idx] - den_post[idx]);
}

// ===========================================================================
// C API implementations
// ===========================================================================

extern "C"
{

    int chain_combine_gradient(
        const float *num_post,
        const float *den_post,
        float weight,
        int T,
        int num_pdfs,
        void *grad_output)
    {
        int total = T * num_pdfs;
        int blockSize = 256;
        int gridSize = (total + blockSize - 1) / blockSize;

        kernel_combine_gradient_fp16<<<gridSize, blockSize>>>(
            num_post, den_post, weight,
            (__half *)grad_output, total);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "chain_combine_gradient error: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

    int chain_penalize_out_of_range(
        const float *nnet_output,
        float *grad_output,
        float limit,
        float scale,
        int T,
        int num_pdfs)
    {
        int total = T * num_pdfs;
        int blockSize = 256;
        int gridSize = (total + blockSize - 1) / blockSize;

        // Allocate counter on device
        int *d_count;
        cudaMalloc(&d_count, sizeof(int));
        cudaMemset(d_count, 0, sizeof(int));

        kernel_penalize_out_of_range<<<gridSize, blockSize>>>(
            nnet_output, grad_output, limit, scale, T, num_pdfs, d_count);

        int h_count = 0;
        cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_count);

        return h_count;
    }

    float chain_l2_regularize(
        const float *nnet_output,
        float *grad_output,
        float l2_scale,
        int total_elements)
    {
        int blockSize = 256;
        int gridSize = (total_elements + blockSize - 1) / blockSize;

        // Allocate partial sums
        float *d_partial;
        cudaMalloc(&d_partial, gridSize * sizeof(float));

        kernel_l2_regularize<<<gridSize, blockSize, blockSize * sizeof(float)>>>(
            nnet_output, grad_output, l2_scale, total_elements, d_partial);

        // Copy partial sums to host and reduce
        float *h_partial = (float *)malloc(gridSize * sizeof(float));
        cudaMemcpy(h_partial, d_partial, gridSize * sizeof(float),
                   cudaMemcpyDeviceToHost);

        double sq_sum = 0.0;
        for (int i = 0; i < gridSize; i++)
        {
            sq_sum += h_partial[i];
        }

        free(h_partial);
        cudaFree(d_partial);

        // l2_term = -0.5 * l2_scale * sum(nnet_output^2)
        return (float)(-0.5 * l2_scale * sq_sum);
    }

    int chain_grad_fp32_to_fp16(
        const float *grad_fp32,
        void *grad_fp16,
        int total_elements)
    {
        int blockSize = 256;
        int gridSize = (total_elements + blockSize - 1) / blockSize;

        kernel_fp32_to_fp16<<<gridSize, blockSize>>>(
            grad_fp32, (__half *)grad_fp16, total_elements);

        return 0;
    }

    // grad[i] += weight * (num_post[i] - den_post[i])
    // Used when grad already has PenalizeOutOfRange values
    int chain_add_posterior_gradient(
        const float *num_post,
        const float *den_post,
        float *grad,
        float weight,
        int total_elements)
    {
        int blockSize = 256;
        int gridSize = (total_elements + blockSize - 1) / blockSize;

        kernel_add_posterior_gradient<<<gridSize, blockSize>>>(
            num_post, den_post, grad, weight, total_elements);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "chain_add_posterior_gradient error: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

} // extern "C"

// ===========================================================================
// chain_num_forward_backward — wrapper for backward.go
//
// Takes raw CSR pointers + final states + FP32 nnet_output.
// Internally converts to FP16, builds ChainFstGPU, runs fwd-bwd + posteriors.
//
// Final states are passed explicitly from Go (parsed from supervision FST).
// Previously this hardcoded all states as final — WRONG for numerator FSTs
// which typically have a single final state (e.g. state 120 with weight 0).
// ===========================================================================

#include "chain.h"

// FP32→FP16 conversion kernel (local, not exported)
__global__ void kernel_f32_to_f16_local(const float *in, __half *out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        out[idx] = __float2half(in[idx]);
}

extern "C"
{

    float chain_num_forward_backward(
        const int *fst_row_ptr,
        const int *fst_col_idx,
        const float *fst_weights,
        const int *fst_pdf_ids,
        const int *fst_final_states,    // [num_final] on GPU
        const float *fst_final_weights, // [num_final] on GPU
        int num_states,
        int num_arcs,
        int num_final,
        const float *nnet_output, // FP32 [T × num_pdfs] on GPU
        float *num_post,          // FP32 [T × num_pdfs] on GPU — output
        int T,
        int num_pdfs,
        cudaStream_t stream)
    {
        (void)stream; // TODO: use stream for all ops

        // 1. Build ChainFstGPU from raw pointers — all already on GPU
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

        // 2. Convert FP32 nnet_output → FP16 (chain.cu kernels expect FP16)
        int total = T * num_pdfs;
        __half *d_nnet_fp16;
        cudaMalloc(&d_nnet_fp16, total * sizeof(__half));

        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        kernel_f32_to_f16_local<<<blocks, threads>>>(nnet_output, d_nnet_fp16, total);

        // 3. Forward-backward
        size_t ws_bytes = (size_t)(T + 1) * num_states * sizeof(float);
        float *alpha, *beta;
        cudaMalloc(&alpha, ws_bytes);
        cudaMalloc(&beta, ws_bytes);

        float total_logprob;
        int ret = chain_forward_backward(d_nnet_fp16, &fst, T, num_pdfs,
                                         alpha, beta, &total_logprob);
        if (ret != 0)
        {
            cudaFree(d_nnet_fp16);
            cudaFree(alpha);
            cudaFree(beta);
            return -1e30f;
        }

        // 4. Compute posteriors
        ret = chain_compute_posteriors(d_nnet_fp16, &fst, T, num_pdfs,
                                       alpha, beta, total_logprob, num_post);

        // Cleanup — only temp buffers, NOT fst pointers (owned by Go)
        cudaFree(d_nnet_fp16);
        cudaFree(alpha);
        cudaFree(beta);

        if (ret != 0)
            return -1e30f;
        return total_logprob;
    }

} // extern "C"
