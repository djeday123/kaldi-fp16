#include "kaldi_den_wrapper.h"
#include "chain/chain-training.h"
#include "chain/chain-den-graph.h"
#include "chain/chain-denominator.h"
#include "fstext/fstext-lib.h"
#include "cudamatrix/cu-matrix.h"

using namespace kaldi;
using namespace kaldi::chain;

struct KaldiDenHandle
{
    DenominatorGraph *den_graph;
    int num_pdfs;
};

extern "C"
{

    void *kaldi_den_init(const char *den_fst_path, int num_pdfs)
    {
        try
        {
            fst::StdVectorFst *fst = fst::ReadFstKaldi(std::string(den_fst_path));
            KaldiDenHandle *h = new KaldiDenHandle;
            h->den_graph = new DenominatorGraph(*fst, num_pdfs);
            h->num_pdfs = num_pdfs;
            delete fst;
            KALDI_LOG << "DenominatorGraph loaded: " << h->den_graph->NumStates()
                      << " states, " << num_pdfs << " pdfs";
            return h;
        }
        catch (std::exception &e)
        {
            KALDI_ERR << "kaldi_den_init failed: " << e.what();
            return NULL;
        }
    }

    float kaldi_den_forward(void *handle, const float *nnet_output,
                            int num_rows, int num_sequences,
                            float leaky_hmm_coeff)
    {
        KaldiDenHandle *h = (KaldiDenHandle *)handle;
        try
        {
            // Copy nnet_output to CuMatrix
            Matrix<BaseFloat> cpu_mat(num_rows, h->num_pdfs);
            memcpy(cpu_mat.Data(), nnet_output,
                   num_rows * h->num_pdfs * sizeof(float));
            // Note: Kaldi Matrix is row-major with stride, need row-by-row copy
            // Actually Matrix stores data contiguously if stride == num_cols
            // Let's do it properly:
            CuMatrix<BaseFloat> gpu_mat(num_rows, h->num_pdfs, kUndefined);
            // copy row by row to handle stride
            {
                Matrix<BaseFloat> tmp(num_rows, h->num_pdfs);
                for (int r = 0; r < num_rows; r++)
                    memcpy(tmp.RowData(r), nnet_output + r * h->num_pdfs,
                           h->num_pdfs * sizeof(float));
                gpu_mat.CopyFromMat(tmp);
            }

            ChainTrainingOptions opts;
            opts.leaky_hmm_coefficient = leaky_hmm_coeff;

            DenominatorComputation den(opts, *(h->den_graph),
                                       num_sequences, gpu_mat);
            BaseFloat logprob = den.Forward();
            return logprob;
        }
        catch (std::exception &e)
        {
            KALDI_ERR << "kaldi_den_forward failed: " << e.what();
            return -1e30;
        }
    }

    float kaldi_den_forward_backward(void *handle, const float *nnet_output,
                                     int num_rows, int num_sequences,
                                     float leaky_hmm_coeff, float deriv_weight,
                                     float *grad_output)
    {
        KaldiDenHandle *h = (KaldiDenHandle *)handle;
        try
        {
            Matrix<BaseFloat> tmp(num_rows, h->num_pdfs);
            for (int r = 0; r < num_rows; r++)
                memcpy(tmp.RowData(r), nnet_output + r * h->num_pdfs,
                       h->num_pdfs * sizeof(float));
            CuMatrix<BaseFloat> gpu_mat(num_rows, h->num_pdfs, kUndefined);
            gpu_mat.CopyFromMat(tmp);

            ChainTrainingOptions opts;
            opts.leaky_hmm_coefficient = leaky_hmm_coeff;

            DenominatorComputation den(opts, *(h->den_graph),
                                       num_sequences, gpu_mat);
            BaseFloat logprob = den.Forward();

            CuMatrix<BaseFloat> gpu_deriv(num_rows, h->num_pdfs, kSetZero);
            den.Backward(deriv_weight, &gpu_deriv);

            // Copy back
            Matrix<BaseFloat> cpu_deriv(num_rows, h->num_pdfs);
            gpu_deriv.CopyToMat(&cpu_deriv);
            for (int r = 0; r < num_rows; r++)
                memcpy(grad_output + r * h->num_pdfs, cpu_deriv.RowData(r),
                       h->num_pdfs * sizeof(float));

            return logprob;
        }
        catch (std::exception &e)
        {
            KALDI_ERR << "kaldi_den_forward_backward failed: " << e.what();
            return -1e30;
        }
    }

    void kaldi_den_free(void *handle)
    {
        if (handle)
        {
            KaldiDenHandle *h = (KaldiDenHandle *)handle;
            delete h->den_graph;
            delete h;
        }
    }

} // extern "C"
