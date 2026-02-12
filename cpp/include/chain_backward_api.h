// =============================================================================
// chain_backward_api.h — C API for chain loss backward pass components
//
// These functions split chain_compute_loss into separate components:
//   1. chain_num_forward_backward  — numerator fwd-bwd (wraps chain.cu)
//   2. den_forward / den_backward  — prob-domain (chain_den.cu, already exists)
//   3. chain_combine_gradient      — num_post - den_post + penalties
// =============================================================================

#ifndef CHAIN_BACKWARD_API_H
#define CHAIN_BACKWARD_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // ---------------------------------------------------------------------------
    // 1. Numerator forward-backward
    //
    // Takes raw CSR pointers + FP32 nnet_output.
    // Internally converts to FP16, builds ChainFstGPU, runs fwd-bwd + posteriors.
    //
    // Args:
    //   fst_row_ptr:       [num_states+1] CSR row pointers (device, int32)
    //   fst_col_idx:       [num_arcs] destination states (device, int32)
    //   fst_weights:       [num_arcs] log-weights (device, float)
    //   fst_pdf_ids:       [num_arcs] pdf-ids, 1-indexed (device, int32)
    //   fst_final_states:  [num_final] indices of final states (device, int32)
    //   fst_final_weights: [num_final] log-weights of final states (device, float)
    //                      (negated tropical weights: store 0.0 for weight=0)
    //   num_states:        number of FST states
    //   num_arcs:          number of FST arcs
    //   num_final:         number of final states
    //   nnet_output:       [T x num_pdfs] network output (device, FP32)
    //   num_post:          [T x num_pdfs] OUTPUT: numerator posteriors (device, FP32)
    //   T:                 number of time frames
    //   num_pdfs:          number of output classes
    //   stream:            CUDA stream (pass NULL for default)
    //
    // Returns: total log-probability, or -1e30 on error
    // ---------------------------------------------------------------------------
    float chain_num_forward_backward(
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
        void *stream);

    // ---------------------------------------------------------------------------
    // 2. Combine gradients: grad = weight * (num_post - den_post)
    //    Output as FP16.
    //
    // Args:
    //   num_post:     [T x num_pdfs] numerator posteriors (device, FP32)
    //   den_post:     [T x num_pdfs] denominator posteriors (device, FP32)
    //   weight:       supervision weight (scalar)
    //   T:            number of time frames
    //   num_pdfs:     number of output classes
    //   grad_output:  [T x num_pdfs] OUTPUT: gradient (device, FP16 as void*)
    //
    // Returns: 0 on success
    // ---------------------------------------------------------------------------
    int chain_combine_gradient(
        const float *num_post,
        const float *den_post,
        float weight,
        int T,
        int num_pdfs,
        void *grad_output);

    // ---------------------------------------------------------------------------
    // 3. PenalizeOutOfRange: add gradient penalty for values outside [-limit, limit]
    //    Matches Kaldi's chain-training.cc PenalizeOutOfRange().
    //
    //    For ~50% of frames (even time indices):
    //      if val < -limit:  deriv += (-limit - val) * scale
    //      if val >  limit:  deriv += ( limit - val) * scale
    //
    // Args:
    //   nnet_output:  [T x num_pdfs] network output (device, FP32)
    //   grad_output:  [T x num_pdfs] gradient (device, FP32) — modified in-place
    //   limit:        threshold (default: 30.0)
    //   scale:        penalty scale = 2.0 * out_of_range_regularize
    //   T:            number of time frames
    //   num_pdfs:     number of output classes
    //
    // Returns: number of out-of-range values found
    // ---------------------------------------------------------------------------
    int chain_penalize_out_of_range(
        const float *nnet_output,
        float *grad_output,
        float limit,
        float scale,
        int T,
        int num_pdfs);

    // ---------------------------------------------------------------------------
    // 4. L2 regularization: grad -= l2_scale * nnet_output
    //
    // Args:
    //   nnet_output:    [T x num_pdfs] network output (device, FP32)
    //   grad_output:    [T x num_pdfs] gradient (device, FP32) — modified in-place
    //   l2_scale:       = weight * l2_regularize
    //   total_elements: T * num_pdfs
    //
    // Returns: l2_term = -0.5 * l2_scale * sum(nnet_output^2)
    // ---------------------------------------------------------------------------
    float chain_l2_regularize(
        const float *nnet_output,
        float *grad_output,
        float l2_scale,
        int total_elements);

    // ---------------------------------------------------------------------------
    // 5. Convert FP32 gradient to FP16
    // ---------------------------------------------------------------------------
    int chain_grad_fp32_to_fp16(
        const float *grad_fp32,
        void *grad_fp16,
        int total_elements);

    // ---------------------------------------------------------------------------
    // 6. Add posterior gradient: grad[i] += weight * (num_post[i] - den_post[i])
    //    Used when grad already contains PenalizeOutOfRange values.
    // ---------------------------------------------------------------------------
    int chain_add_posterior_gradient(
        const float *num_post,
        const float *den_post,
        float *grad,
        float weight,
        int total_elements);

#ifdef __cplusplus
}
#endif

#endif // CHAIN_BACKWARD_API_H