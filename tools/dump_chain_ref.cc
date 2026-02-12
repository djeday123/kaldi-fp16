// dump_chain_ref.cc — Dump Kaldi chain training reference values
//
// Reads one cegs example, creates deterministic nnet_output,
// runs numerator and denominator separately, dumps all intermediates.
//
// Usage:
//   dump_chain_ref <den.fst> <cegs-rspecifier> <output-dir>
//
// Example:
//   dump_chain_ref \
//     /opt/kaldi/egs/work_3/s5/exp/chain_cnn_v2/cnn_tdnn1d_v2_sp/den.fst \
//     "ark:/opt/kaldi/egs/work_3/s5/exp/chain_cnn_v2/cnn_tdnn1d_v2_sp/egs/cegs.15.ark" \
//     /tmp/chain_ref
//
// Outputs:
//   meta.txt           — dimensions, logprobs, weight
//   nnet_output.bin    — [T × D] FP32 raw binary
//   num_deriv.bin      — [T × D] FP32 numerator posteriors
//   den_deriv.bin      — [T × D] FP32 denominator posteriors
//   total_deriv.bin    — [T × D] FP32 combined gradient (num - den)
//   supervision.txt    — FST info (states, arcs, etc.)

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet3/nnet-chain-example.h"
#include "chain/chain-supervision.h"
#include "chain/chain-training.h"
#include "chain/chain-den-graph.h"
#include "chain/chain-numerator.h"
#include "chain/chain-denominator.h"
#include "cudamatrix/cu-device.h"

#include <fstream>
#include <sys/stat.h>
#include <iomanip>

using namespace kaldi;
using namespace kaldi::chain;
using namespace kaldi::nnet3;

// Write raw FP32 matrix to binary file
void WriteBinaryMatrix(const std::string &filename,
                       const CuMatrix<BaseFloat> &cu_mat)
{
    Matrix<BaseFloat> mat(cu_mat);
    std::ofstream out(filename, std::ios::binary);
    KALDI_ASSERT(out.is_open());
    int32 rows = mat.NumRows(), cols = mat.NumCols();
    // Write dimensions header
    out.write(reinterpret_cast<const char *>(&rows), sizeof(int32));
    out.write(reinterpret_cast<const char *>(&cols), sizeof(int32));
    // Write data row by row (matrix may have stride != cols)
    for (int32 r = 0; r < rows; r++)
    {
        out.write(reinterpret_cast<const char *>(mat.RowData(r)),
                  cols * sizeof(BaseFloat));
    }
    out.close();
    KALDI_LOG << "Wrote " << filename << " [" << rows << " x " << cols << "]";
}

// Write FST supervision details
void WriteFstInfo(const std::string &filename,
                  const Supervision &sup)
{
    std::ofstream out(filename);
    out << "weight=" << sup.weight << "\n";
    out << "num_sequences=" << sup.num_sequences << "\n";
    out << "frames_per_sequence=" << sup.frames_per_sequence << "\n";
    out << "label_dim=" << sup.label_dim << "\n";
    out << "e2e=" << (sup.e2e_fsts.empty() ? "false" : "true") << "\n";

    // FST stats
    if (!sup.e2e_fsts.empty())
    {
        out << "fst_type=e2e\n";
        out << "num_e2e_fsts=" << sup.e2e_fsts.size() << "\n";
    }
    else
    {
        int32 num_states = 0, num_arcs = 0;
        for (fst::StateIterator<fst::StdVectorFst> siter(sup.fst);
             !siter.Done(); siter.Next())
        {
            num_states++;
            for (fst::ArcIterator<fst::StdVectorFst> aiter(sup.fst, siter.Value());
                 !aiter.Done(); aiter.Next())
            {
                num_arcs++;
            }
        }
        out << "fst_type=regular\n";
        out << "fst_num_states=" << num_states << "\n";
        out << "fst_num_arcs=" << num_arcs << "\n";
        out << "fst_start=" << sup.fst.Start() << "\n";

        // Dump FST arcs for detailed comparison
        out << "\n# FST arcs: src_state dst_state label weight\n";
        for (fst::StateIterator<fst::StdVectorFst> siter(sup.fst);
             !siter.Done(); siter.Next())
        {
            int32 s = siter.Value();
            for (fst::ArcIterator<fst::StdVectorFst> aiter(sup.fst, s);
                 !aiter.Done(); aiter.Next())
            {
                const fst::StdArc &arc = aiter.Value();
                out << s << " " << arc.nextstate << " "
                    << arc.ilabel << " " << arc.weight.Value() << "\n";
            }
            // Final weight
            fst::StdFst::Weight fw = sup.fst.Final(s);
            if (fw != fst::StdFst::Weight::Zero())
            {
                out << s << " final " << fw.Value() << "\n";
            }
        }
    }
    out.close();
    KALDI_LOG << "Wrote " << filename;
}

// Dump DenominatorGraph (den.fst) info
void WriteDenFstInfo(const std::string &filename,
                     const DenominatorGraph &den_graph)
{
    std::ofstream out(filename);
    out << "num_pdfs=" << den_graph.NumPdfs() << "\n";
    out << "num_states=" << den_graph.NumStates() << "\n";
    out.close();
    KALDI_LOG << "Wrote " << filename;
}

int main(int argc, char *argv[])
{
    try
    {
        const char *usage =
            "Dump Kaldi chain training reference values for verification.\n"
            "\n"
            "Usage: dump_chain_ref <den-fst> <cegs-rspecifier> <output-dir>\n"
            "\n"
            "e.g.: dump_chain_ref den.fst ark:cegs.15.ark /tmp/chain_ref\n";

        ParseOptions po(usage);

        // Chain training options — we disable regularization for clean comparison
        ChainTrainingOptions chain_opts;
        chain_opts.l2_regularize = 0.0;
        chain_opts.out_of_range_regularize = 0.0;
        chain_opts.leaky_hmm_coefficient = 1.0e-05; // standard value

        BaseFloat nnet_output_scale = 0.1; // scale for random nnet output
        int32 seed = 42;

        po.Register("leaky-hmm-coefficient", &chain_opts.leaky_hmm_coefficient,
                    "Leaky HMM coefficient for denominator");
        po.Register("nnet-output-scale", &nnet_output_scale,
                    "Scale for random nnet output values");
        po.Register("seed", &seed, "Random seed for nnet output");

        po.Read(argc, argv);

        if (po.NumArgs() != 3)
        {
            po.PrintUsage();
            exit(1);
        }

        std::string den_fst_rxfilename = po.GetArg(1),
                    cegs_rspecifier = po.GetArg(2),
                    output_dir = po.GetArg(3);

        // Create output directory
        mkdir(output_dir.c_str(), 0755);

        // Initialize CUDA
#if HAVE_CUDA == 1
        CuDevice::Instantiate().SelectGpuId("yes");
#endif

        // 1. Load denominator FST
        KALDI_LOG << "Loading den.fst from " << den_fst_rxfilename;
        fst::StdVectorFst den_fst;
        ReadFstKaldi(den_fst_rxfilename, &den_fst);

        // Find max label in den_fst to determine num_pdfs
        int32 num_pdfs = -1;
        for (fst::StateIterator<fst::StdVectorFst> siter(den_fst);
             !siter.Done(); siter.Next())
        {
            for (fst::ArcIterator<fst::StdVectorFst> aiter(den_fst, siter.Value());
                 !aiter.Done(); aiter.Next())
            {
                if (aiter.Value().ilabel > num_pdfs)
                    num_pdfs = aiter.Value().ilabel;
            }
        }
        KALDI_ASSERT(num_pdfs > 0);
        KALDI_LOG << "Den FST: num_pdfs = " << num_pdfs;

        DenominatorGraph den_graph(den_fst, num_pdfs);

        // 2. Read first cegs example
        KALDI_LOG << "Reading cegs from " << cegs_rspecifier;
        SequentialNnetChainExampleReader reader(cegs_rspecifier);
        KALDI_ASSERT(!reader.Done());

        const NnetChainExample &eg = reader.Value();
        std::string key = reader.Key();
        KALDI_LOG << "Processing example: " << key;

        // Get supervision
        KALDI_ASSERT(eg.outputs.size() >= 1);
        const NnetChainSupervision &ncs = eg.outputs[0];
        const Supervision &sup = ncs.supervision;

        KALDI_LOG << "Supervision: weight=" << sup.weight
                  << " num_sequences=" << sup.num_sequences
                  << " frames_per_sequence=" << sup.frames_per_sequence
                  << " label_dim=" << sup.label_dim
                  << " e2e=" << (sup.e2e_fsts.empty() ? "no" : "yes");

        KALDI_ASSERT(sup.e2e_fsts.empty() &&
                     "Only regular (non-e2e) supervision supported");

        int32 T = sup.num_sequences * sup.frames_per_sequence;
        int32 D = num_pdfs; // use den_graph's num_pdfs

        KALDI_LOG << "Matrix dimensions: T=" << T << " D=" << D
                  << " (num_sequences=" << sup.num_sequences
                  << " frames_per_sequence=" << sup.frames_per_sequence << ")";

        // 3. Create deterministic nnet_output
        // Use seeded random values scaled small so exp() doesn't overflow
        srand(seed);
        Matrix<BaseFloat> nnet_output_cpu(T, D);
        for (int32 r = 0; r < T; r++)
        {
            for (int32 c = 0; c < D; c++)
            {
                // Deterministic pseudo-random: small values in [-scale, scale]
                float val = nnet_output_scale *
                            (2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f);
                nnet_output_cpu(r, c) = val;
            }
        }

        CuMatrix<BaseFloat> nnet_output(nnet_output_cpu);

        // 4. Run DENOMINATOR
        KALDI_LOG << "=== Running Denominator ===";
        CuMatrix<BaseFloat> den_deriv(T, D, kSetZero);
        BaseFloat den_logprob;
        {
            DenominatorComputation denominator(chain_opts, den_graph,
                                               sup.num_sequences, nnet_output);
            den_logprob = denominator.Forward();
            bool ok = denominator.Backward(-1.0, &den_deriv);
            // den_deriv now contains -1 * den_posteriors
            // So actual den_posteriors = -den_deriv
            KALDI_LOG << "Denominator logprob (unweighted): " << den_logprob
                      << " backward_ok=" << std::boolalpha << ok;
        }

        // 5. Run NUMERATOR
        KALDI_LOG << "=== Running Numerator ===";
        CuMatrix<BaseFloat> num_deriv(T, D, kSetZero);
        BaseFloat num_logprob;
        {
            NumeratorComputation numerator(sup, nnet_output);
            num_logprob = numerator.Forward();
            // Note: Forward() returns logprob * weight
            numerator.Backward(&num_deriv);
            // num_deriv now contains num_posteriors * weight
            KALDI_LOG << "Numerator logprob (weighted): " << num_logprob;
        }

        // 6. Compute combined gradient (as Kaldi does)
        // total_deriv = num_deriv + den_deriv
        // where num_deriv = weight * num_posteriors
        // and den_deriv = -weight * den_posteriors (from Backward(-weight, ...))
        // Actually Kaldi passes -supervision.weight to denominator.Backward
        // So: total = weight * num_post + (-weight) * den_post
        //           = weight * (num_post - den_post)
        CuMatrix<BaseFloat> total_deriv(T, D, kSetZero);
        total_deriv.AddMat(1.0, num_deriv); // + weight * num_post
        total_deriv.AddMat(1.0, den_deriv); // + (-weight * den_post)

        BaseFloat objf = num_logprob - sup.weight * den_logprob;
        BaseFloat weight = sup.weight * sup.num_sequences * sup.frames_per_sequence;

        KALDI_LOG << "=== Results ===";
        KALDI_LOG << "num_logprob (weighted) = " << num_logprob;
        KALDI_LOG << "den_logprob (unweighted) = " << den_logprob;
        KALDI_LOG << "den_logprob (weighted) = " << sup.weight * den_logprob;
        KALDI_LOG << "objf = " << objf;
        KALDI_LOG << "objf/frame = " << objf / weight;
        KALDI_LOG << "weight = " << weight;

        // 7. Dump everything
        WriteBinaryMatrix(output_dir + "/nnet_output.bin", nnet_output);
        WriteBinaryMatrix(output_dir + "/num_deriv.bin", num_deriv);
        WriteBinaryMatrix(output_dir + "/den_deriv.bin", den_deriv);
        WriteBinaryMatrix(output_dir + "/total_deriv.bin", total_deriv);
        WriteFstInfo(output_dir + "/supervision.txt", sup);
        WriteDenFstInfo(output_dir + "/den_info.txt", den_graph);

        // Write meta file
        {
            std::ofstream meta(output_dir + "/meta.txt");
            meta << "key=" << key << "\n";
            meta << "T=" << T << "\n";
            meta << "D=" << D << "\n";
            meta << "num_sequences=" << sup.num_sequences << "\n";
            meta << "frames_per_sequence=" << sup.frames_per_sequence << "\n";
            meta << "weight=" << sup.weight << "\n";
            meta << "seed=" << seed << "\n";
            meta << "nnet_output_scale=" << nnet_output_scale << "\n";
            meta << "leaky_hmm_coefficient=" << chain_opts.leaky_hmm_coefficient << "\n";
            meta << "l2_regularize=" << chain_opts.l2_regularize << "\n";
            meta << "out_of_range_regularize=" << chain_opts.out_of_range_regularize << "\n";
            meta << "num_logprob_weighted=" << std::setprecision(15) << num_logprob << "\n";
            meta << "den_logprob_unweighted=" << std::setprecision(15) << den_logprob << "\n";
            meta << "den_logprob_weighted=" << std::setprecision(15) << sup.weight * den_logprob << "\n";
            meta << "objf=" << std::setprecision(15) << objf << "\n";
            meta << "objf_per_frame=" << std::setprecision(15) << objf / weight << "\n";
            meta << "label_dim=" << sup.label_dim << "\n";
            meta.close();
            KALDI_LOG << "Wrote " << output_dir << "/meta.txt";
        }

        // Also dump nnet_output as text (first few rows for quick debugging)
        {
            Matrix<BaseFloat> mat(nnet_output_cpu);
            std::ofstream out(output_dir + "/nnet_output_sample.txt");
            int32 max_rows = std::min(T, (int32)5);
            int32 max_cols = std::min(D, (int32)20);
            out << "# First " << max_rows << " rows, " << max_cols << " cols\n";
            for (int32 r = 0; r < max_rows; r++)
            {
                for (int32 c = 0; c < max_cols; c++)
                {
                    out << mat(r, c);
                    if (c < max_cols - 1)
                        out << " ";
                }
                out << "\n";
            }
            out.close();
        }

        // Dump deriv stats
        {
            Matrix<BaseFloat> num_d(num_deriv), den_d(den_deriv), tot_d(total_deriv);
            BaseFloat num_sum = 0, den_sum = 0, tot_sum = 0;
            BaseFloat num_absmax = 0, den_absmax = 0, tot_absmax = 0;
            for (int32 r = 0; r < T; r++)
            {
                for (int32 c = 0; c < D; c++)
                {
                    num_sum += num_d(r, c);
                    den_sum += den_d(r, c);
                    tot_sum += tot_d(r, c);
                    num_absmax = std::max(num_absmax, std::abs(num_d(r, c)));
                    den_absmax = std::max(den_absmax, std::abs(den_d(r, c)));
                    tot_absmax = std::max(tot_absmax, std::abs(tot_d(r, c)));
                }
            }
            KALDI_LOG << "Deriv stats:";
            KALDI_LOG << "  num_deriv: sum=" << num_sum << " absmax=" << num_absmax;
            KALDI_LOG << "  den_deriv: sum=" << den_sum << " absmax=" << den_absmax;
            KALDI_LOG << "  tot_deriv: sum=" << tot_sum << " absmax=" << tot_absmax;
        }

        KALDI_LOG << "All reference data dumped to " << output_dir;

#if HAVE_CUDA == 1
        CuDevice::Instantiate().PrintProfile();
#endif

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return -1;
    }
}