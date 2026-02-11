#include "kaldi_den_wrapper.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main()
{
    int num_pdfs = 3080;
    int T = 34; // frames_per_seq for seq 0
    int num_seq = 1;
    int num_rows = T * num_seq;

    // Init
    void *h = kaldi_den_init(
        "/opt/kaldi/egs/work_3/s5/exp/chain_cnn_v2/cnn_tdnn1d_v2_sp/den.fst",
        num_pdfs);
    if (!h)
    {
        fprintf(stderr, "init failed\n");
        return 1;
    }

    // Zero nnet output
    float *nnet_output = (float *)calloc(num_rows * num_pdfs, sizeof(float));

    float logprob = kaldi_den_forward(h, nnet_output, num_rows, num_seq, 1e-05);
    printf("den_logprob = %.6f\n", logprob);
    printf("den_logprob_per_frame = %.6f\n", logprob / T);

    free(nnet_output);
    kaldi_den_free(h);
    return 0;
}