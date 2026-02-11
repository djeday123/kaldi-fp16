#include "chain/chain-training.h"
#include "chain/chain-den-graph.h"
#include "chain/chain-denominator.h"
#include "chain/chain-numerator.h"
#include "nnet3/nnet-chain-example.h"
#include "util/common-utils.h"

int main(int argc, char *argv[])
{
    using namespace kaldi;
    using namespace kaldi::chain;
    using namespace kaldi::nnet3;

    const char *usage = "Usage: chain-verify <den-fst> <chain-egs-rspecifier>\n";
    ParseOptions po(usage);
    po.Read(argc, argv);
    if (po.NumArgs() != 2)
    {
        po.PrintUsage();
        return 1;
    }

    fst::StdVectorFst *den_fst = fst::ReadFstKaldi(po.GetArg(1));

    SequentialNnetChainExampleReader reader(po.GetArg(2));
    KALDI_ASSERT(!reader.Done());
    const NnetChainExample &eg = reader.Value();
    const NnetChainSupervision &sup = eg.outputs[0];

    int32 fps = sup.supervision.frames_per_sequence;
    int32 num_seq = sup.supervision.num_sequences;
    int32 tot_frames = fps * num_seq;
    int32 num_pdfs = sup.supervision.label_dim;

    KALDI_LOG << "fps=" << fps << " num_seq=" << num_seq
              << " num_pdfs=" << num_pdfs << " tot_frames=" << tot_frames;

    DenominatorGraph den_graph(*den_fst, num_pdfs);

    // Zero nnet output
    CuMatrix<BaseFloat> nnet_output(tot_frames, num_pdfs, kSetZero);

    // --- Numerator (non-e2e) ---
    NumeratorComputation numerator(sup.supervision, nnet_output);
    BaseFloat num_logprob = numerator.Forward();
    KALDI_LOG << "num_logprob_total=" << num_logprob
              << " per_frame=" << num_logprob / tot_frames;

    // --- Denominator ---
    ChainTrainingOptions opts;
    opts.leaky_hmm_coefficient = 1e-05;
    DenominatorComputation denominator(opts, den_graph, num_seq, nnet_output);
    BaseFloat den_logprob = denominator.Forward();
    KALDI_LOG << "den_logprob_total=" << den_logprob
              << " per_frame=" << den_logprob / tot_frames;

    KALDI_LOG << "objf_total=" << (num_logprob - den_logprob)
              << " per_frame=" << (num_logprob - den_logprob) / tot_frames;

    // Without leaky HMM
    ChainTrainingOptions opts2;
    opts2.leaky_hmm_coefficient = 0;
    DenominatorComputation denom2(opts2, den_graph, num_seq, nnet_output);
    BaseFloat den2 = denom2.Forward();
    KALDI_LOG << "den_NO_LEAKY=" << den2 << " per_frame=" << den2 / tot_frames;
    KALDI_LOG << "objf_NO_LEAKY per_frame=" << (num_logprob - den2) / tot_frames;

    delete den_fst;
    return 0;
}
