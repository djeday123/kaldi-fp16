#ifndef KALDI_FP16_CHAIN_DEN_H
#define KALDI_FP16_CHAIN_DEN_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // ============================================================
    // Native Denominator for Chain LF-MMI
    //
    // Implements Kaldi's DenominatorComputation in probability space:
    //   1. exp(nnet_output) — work in probability, not log domain
    //   2. initial_probs from 100-iter HMM warmup (computed on CPU)
    //   3. Leaky HMM: alpha += tot * leaky_coeff * initial_probs
    //   4. Arbitrary scaling: alpha /= sum(alpha) each frame
    //   5. ALL states are final (weight = 1.0)
    //   6. transition_prob = exp(-arc.weight) from tropical semiring
    //
    // Reference: Kaldi src/chain/chain-denominator.cc
    //            Kaldi src/chain/chain-den-graph.cc
    // ============================================================

    // Transition data stored as Structure-of-Arrays on GPU
    typedef struct
    {
        int32_t *src_states;     // [num_transitions]
        int32_t *dst_states;     // [num_transitions]
        int32_t *pdf_ids;        // [num_transitions] 0-indexed
        float *transition_probs; // [num_transitions] = exp(-arc.weight)
        int num_transitions;
        int num_states;
        int num_pdfs;
    } DenFstGPU;

    // ---- Upload / Free ----

    // Upload transition data to GPU (SoA format)
    // src, dst, pdf: [num_trans] int32 on host
    // trans_probs: [num_trans] float32 on host (already exp(-weight))
    // Returns 0 on success
    int den_fst_upload(
        DenFstGPU *fst,
        const int32_t *src, const int32_t *dst,
        const int32_t *pdf, const float *trans_probs,
        int num_trans, int num_states, int num_pdfs);

    void den_fst_free(DenFstGPU *fst);

    // ---- Forward only ----

    // nnet_output: [T x num_pdfs] float32 on HOST
    // initial_probs: [num_states] float32 on HOST
    // Returns log_prob
    float den_forward(
        const DenFstGPU *fst,
        const float *nnet_output,
        const float *initial_probs,
        int T,
        float leaky_hmm_coeff);

    // ---- Forward-Backward with gradients ----

    // grad_output: [T x num_pdfs] float32 on HOST (output, denominator posteriors)
    // Returns log_prob
    float den_forward_backward(
        const DenFstGPU *fst,
        const float *nnet_output,
        const float *initial_probs,
        int T,
        float leaky_hmm_coeff,
        float *grad_output);

    // ---- Error handling ----
    const char *den_last_error();
    void den_clear_error();

#ifdef __cplusplus
}
#endif

#endif // KALDI_FP16_CHAIN_DEN_H