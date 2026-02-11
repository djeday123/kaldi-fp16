package main

import (
	"bufio"
	"fmt"
	"log"
	"os"

	"kaldi-fp16/internal/gpu"
	"kaldi-fp16/internal/loader"
	"kaldi-fp16/internal/nnet"
	"kaldi-fp16/internal/parser"
	"kaldi-fp16/internal/sparse"
)

func main() {
	gpu.Init(0)

	dl, err := loader.NewDataLoaderFromPaths(
		[]string{"/opt/kaldi/egs/work_3/s5/exp/chain_cnn_v2/cnn_tdnn1d_v2_sp/egs/cegs.1.ark"},
		8, false)
	if err != nil {
		log.Fatal(err)
	}
	tb, err := dl.NextBatch()
	if err != nil {
		log.Fatal(err)
	}

	fps := tb.FramesPerSeq[0]
	numPdfs := 3080
	fmt.Printf("Seq 0: fps=%d, numPdfs=%d\n", fps, numPdfs)

	// Zero nnet output
	zeroOutput, err := gpu.ZeroTensor(fps, numPdfs)
	if err != nil {
		log.Fatal(err)
	}
	defer zeroOutput.Free()

	// === NUMERATOR (our CUDA kernels) ===
	seq0FstGPU, err := nnet.NewChainFstGPU(tb.PerSeqCSRs[0])
	if err != nil {
		log.Fatal("num fst:", err)
	}
	defer seq0FstGPU.Free()

	numLP, err := nnet.ForwardBackward(zeroOutput, seq0FstGPU)
	if err != nil {
		log.Fatal("num FB:", err)
	}
	fmt.Printf("num_logprob = %.6f  (Kaldi ref: -28.349600)\n", numLP)

	// === DENOMINATOR (Kaldi wrapper) ===
	kaldiDen, err := nnet.NewKaldiDenominator(
		"/opt/kaldi/egs/work_3/s5/exp/chain_cnn_v2/cnn_tdnn1d_v2_sp/den.fst",
		numPdfs)
	if err != nil {
		log.Fatal("kaldi den:", err)
	}
	defer kaldiDen.Free()

	// Zero CPU buffer for Kaldi
	zeroCPU := make([]float32, fps*numPdfs)
	denLP, err := kaldiDen.Forward(zeroCPU, fps, 1)
	if err != nil {
		log.Fatal("den forward:", err)
	}
	fmt.Printf("den_logprob = %.6f  (Kaldi ref: -0.313154)\n", denLP)

	objf := numLP - denLP
	fmt.Printf("objf = %.6f\n", objf)
	fmt.Printf("objf_per_frame = %.6f  (Kaldi ref: -0.824602)\n", float64(objf)/float64(fps))
}

func loadDenFst(path string) (*sparse.CSR, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	reader := bufio.NewReader(f)
	fst := parser.ReadFst(reader)
	if fst == nil {
		return nil, fmt.Errorf("unsupported FST format")
	}
	csr, err := sparse.FstToCSR(fst)
	if err != nil {
		return nil, fmt.Errorf("to CSR: %w", err)
	}
	return csr, nil
}
