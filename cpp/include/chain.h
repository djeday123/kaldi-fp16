#ifndef KALDI_FP16_CHAIN_H
#define KALDI_FP16_CHAIN_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // ============================================================
    // Chain LF-MMI Loss
    //
    // Loss = -(log P_num - log P_den)
    // Grad = -(posterior_num - posterior_den)
    //
    // Both numerator and denominator use forward-backward on FST
    // in log-semiring on GPU.
    // ============================================================

    // ChainFstGPU — FST on GPU in CSR format
    typedef struct
    {
        int32_t *row_ptr;      // [num_states + 1]
        int32_t *col_idx;      // [num_arcs] — destination states
        int32_t *labels;       // [num_arcs] — pdf-ids (1-indexed, 0=epsilon)
        float *weights;        // [num_arcs] — log-weights
        int32_t *final_states; // indices of final states
        float *final_weights;  // log-weights of final states
        int num_states;
        int num_arcs;
        int num_final;
        int start_state;
    } ChainFstGPU;

    // ChainLossResult — output from chain loss computation
    typedef struct
    {
        float num_logprob; // numerator log-probability
        float den_logprob; // denominator log-probability
        float loss;        // -(num_logprob - den_logprob)
        // grad: [T x num_pdfs] FP32 on GPU (caller provides buffer)
    } ChainLossResult;

    // ============================================================
    // Core computation
    // ============================================================

    // chain_forward_backward: Run forward-backward on an FST
    //
    // nnet_output:  [T x num_pdfs] FP16 on GPU (network output, NOT log-softmax)
    // fst:          FST in CSR format on GPU
    // T:            number of frames
    // num_pdfs:     number of PDFs (output dimension)
    // alpha:        [T+1 x num_states] FP32 workspace (caller allocates)
    // beta:         [T+1 x num_states] FP32 workspace (caller allocates)
    // total_logprob: output — total log-probability
    //
    // Returns 0 on success, -1 on error
    int chain_forward_backward(
        const void *nnet_output, // FP16 [T x num_pdfs]
        const ChainFstGPU *fst,
        int T,
        int num_pdfs,
        float *alpha,        // workspace [T+1 x S]
        float *beta,         // workspace [T+1 x S]
        float *total_logprob // output scalar
    );

    // chain_compute_posteriors: Compute per-frame posteriors from alpha/beta
    //
    // After forward-backward, posterior[t][p] = sum over arcs with label=p:
    //   exp(alpha[t][s] + nnet[t][p-1] + weight + beta[t+1][d] - total_logprob)
    //
    // posteriors: [T x num_pdfs] FP32 on GPU (output)
    int chain_compute_posteriors(
        const void *nnet_output, // FP16 [T x num_pdfs]
        const ChainFstGPU *fst,
        int T,
        int num_pdfs,
        const float *alpha,
        const float *beta,
        float total_logprob,
        float *posteriors // output [T x num_pdfs] FP32
    );

    // chain_compute_loss: Full chain loss computation
    //
    // Computes:
    //   1. Forward-backward on numerator FST → num_logprob, num_posteriors
    //   2. Forward-backward on denominator FST → den_logprob, den_posteriors
    //   3. loss = -(num_logprob - den_logprob)
    //   4. grad = -(num_posteriors - den_posteriors)  (written as FP16)
    //
    // nnet_output:    [T x num_pdfs] FP16 (raw network output)
    // num_fst:        numerator FST (per-utterance supervision)
    // den_fst:        denominator FST (shared, phone LM graph)
    // grad_output:    [T x num_pdfs] FP16 (output gradient, can be NULL)
    // result:         loss values
    int chain_compute_loss(
        const void *nnet_output,
        const ChainFstGPU *num_fst,
        const ChainFstGPU *den_fst,
        int T,
        int num_pdfs,
        void *grad_output, // FP16 [T x num_pdfs], NULL to skip
        ChainLossResult *result);

    // ============================================================
    // Workspace allocation helper
    // ============================================================

    // chain_workspace_bytes: Returns bytes needed for alpha/beta workspace
    // for a given num_states and T
    size_t chain_workspace_bytes(int T, int num_states);

    // ============================================================
    // Error handling
    // ============================================================

    const char *chain_last_error(void);
    void chain_clear_error(void);

#ifdef __cplusplus
}
#endif

#endif // KALDI_FP16_CHAIN_H