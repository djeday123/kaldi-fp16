#ifndef KALDI_DEN_WRAPPER_H
#define KALDI_DEN_WRAPPER_H

#ifdef __cplusplus
extern "C"
{
#endif

    // Initialize: load den.fst, compute initial probs
    // Returns opaque handle, NULL on error
    void *kaldi_den_init(const char *den_fst_path, int num_pdfs);

    // Forward pass: compute denominator log-prob
    // nnet_output: [T * num_sequences, num_pdfs] row-major float32
    // Returns total log-prob (sum over sequences)
    float kaldi_den_forward(void *handle, const float *nnet_output,
                            int num_rows, int num_sequences,
                            float leaky_hmm_coeff);

    // Forward+Backward: compute log-prob and gradients
    // grad_output: [num_rows, num_pdfs] output gradients, pre-allocated
    // Returns total log-prob
    float kaldi_den_forward_backward(void *handle, const float *nnet_output,
                                     int num_rows, int num_sequences,
                                     float leaky_hmm_coeff, float deriv_weight,
                                     float *grad_output);

    // Free handle
    void kaldi_den_free(void *handle);

#ifdef __cplusplus
}
#endif
#endif
