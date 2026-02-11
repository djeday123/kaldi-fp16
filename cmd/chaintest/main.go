package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"os"
	"time"

	"kaldi-fp16/internal/gpu"
	"kaldi-fp16/internal/loader"
	"kaldi-fp16/internal/nnet"
	"kaldi-fp16/internal/parser"
	"kaldi-fp16/internal/sparse"
)

func main() {
	xconfigPath := "/opt/kaldi/egs/work_3/s5/exp/chain_cnn_v2/cnn_tdnn1d_v2_sp/configs/network.xconfig"
	arkPattern := "/opt/kaldi/egs/work_3/s5/exp/chain_cnn_v2/cnn_tdnn1d_v2_sp/egs/cegs.*.ark"
	denFstPath := "/opt/kaldi/egs/work_3/s5/exp/chain_cnn_v2/cnn_tdnn1d_v2_sp/den.fst"
	batchSize := 8
	deviceID := 0

	if len(os.Args) > 1 {
		denFstPath = os.Args[1]
	}

	fmt.Println("=== Chain LF-MMI Loss Test (per-sequence) ===")
	fmt.Println()

	// 1. Parse model
	model, err := nnet.BuildModel(xconfigPath)
	if err != nil {
		log.Fatalf("BuildModel: %v", err)
	}
	numPdfs := model.ChainOutput().OutputDim
	fmt.Printf("Model: %d PDFs, %.1fM params\n", numPdfs, float64(model.NumParams)/1e6)

	// 2. Init GPU
	if err := gpu.Init(deviceID); err != nil {
		log.Fatalf("GPU init: %v", err)
	}
	handle, err := gpu.NewHandle()
	if err != nil {
		log.Fatalf("cuBLAS: %v", err)
	}
	defer handle.Destroy()
	freeBefore, _, _ := gpu.MemoryInfo()

	// 3. Create network
	network, err := nnet.NewNetwork(model, handle)
	if err != nil {
		log.Fatalf("NewNetwork: %v", err)
	}
	defer network.Free()

	// 4. Load denominator FST
	fmt.Printf("Loading den.fst: %s\n", denFstPath)
	denCSR, err := loadDenFst(denFstPath)
	if err != nil {
		fmt.Printf("  Could not load den.fst: %v\n", err)
		fmt.Println("  Will use per-sequence numerator as denominator (structure test)")
		denCSR = nil
	} else {
		fmt.Printf("  Den FST: %d states, %d arcs, %d final\n",
			denCSR.NumStates, denCSR.NumArcs, len(denCSR.FinalStates))
	}

	// 5. Load batch
	fmt.Printf("\nLoading batch (size=%d)...\n", batchSize)
	dl, err := loader.NewDataLoader(loader.DataLoaderConfig{
		Pattern:   arkPattern,
		BatchSize: batchSize,
		DropLast:  true,
	})
	if err != nil {
		log.Fatalf("DataLoader: %v", err)
	}

	tb, err := dl.NextBatch()
	if err != nil {
		log.Fatalf("NextBatch: %v", err)
	}
	fmt.Printf("Batch: %d seqs, %d total frames\n", tb.BatchSize, tb.Features.Rows)
	fmt.Printf("Per-sequence FSTs: %d\n", len(tb.PerSeqCSRs))
	for i, csr := range tb.PerSeqCSRs {
		fmt.Printf("  Seq %d: frames=%d, fps=%d, states=%d, arcs=%d, final=%d, start=%d\n",
			i, tb.NumFrames[i], tb.FramesPerSeq[i], csr.NumStates, csr.NumArcs,
			len(csr.FinalStates), csr.StartState)
	}

	// 6. Transfer to GPU
	gpuBatch, err := gpu.TransferBatch(tb)
	if err != nil {
		log.Fatalf("TransferBatch: %v", err)
	}
	defer gpuBatch.Free()

	features := &gpu.Tensor{
		Ptr:  gpuBatch.Features(),
		Rows: gpuBatch.TotalFrames,
		Cols: gpuBatch.FeatDim,
	}
	var ivectors *gpu.Tensor
	if gpuBatch.IvecDim > 0 {
		ivectors = &gpu.Tensor{
			Ptr:  gpuBatch.Ivectors(),
			Rows: gpuBatch.BatchSize,
			Cols: gpuBatch.IvecDim,
		}
	}

	// 7. Forward pass
	fmt.Println("\nRunning forward pass...")
	t0 := time.Now()
	state, err := network.Forward(features, ivectors)
	if err != nil {
		log.Fatalf("Forward: %v", err)
	}
	gpu.Sync()
	fmt.Printf("Forward: %v\n", time.Since(t0).Round(time.Microsecond))
	defer state.Free()

	nnetOutput := state.Output
	fmt.Printf("Output: [%d x %d]\n", nnetOutput.Rows, nnetOutput.Cols)

	// 8. Test single-sequence forward-backward (with subsampling)
	fmt.Println("\n--- Single-sequence forward-backward ---")
	{
		seq0 := tb.PerSeqCSRs[0]
		seq0Frames := tb.NumFrames[0]
		seq0Fps := tb.FramesPerSeq[0]
		seq0Offset := tb.FrameOffsets[0]
		leftContext := 30
		subsamplingFactor := 3

		fmt.Printf("Seq 0: offset=%d, frames=%d, fps=%d, states=%d, arcs=%d\n",
			seq0Offset, seq0Frames, seq0Fps, seq0.NumStates, seq0.NumArcs)

		// Subsample
		effectiveInRows := leftContext + seq0Fps*subsamplingFactor
		if effectiveInRows > seq0Frames {
			effectiveInRows = seq0Frames
		}
		seqView := nnetOutput.View(seq0Offset, effectiveInRows, nnetOutput.Cols)
		subsampled, err := gpu.SubsampleRows(seqView, subsamplingFactor, leftContext)
		if err != nil {
			log.Fatalf("SubsampleRows: %v", err)
		}
		fmt.Printf("Subsampled: [%d x %d]\n", subsampled.Rows, subsampled.Cols)

		seq0FstGPU, err := nnet.NewChainFstGPU(seq0)
		if err != nil {
			log.Fatalf("Upload seq0 FST: %v", err)
		}

		t1 := time.Now()
		logprob, err := nnet.ForwardBackward(subsampled, seq0FstGPU)
		if err != nil {
			log.Fatalf("ForwardBackward: %v", err)
		}
		gpu.Sync()
		fmt.Printf("  Time: %v\n", time.Since(t1).Round(time.Microsecond))
		fmt.Printf("  LogProb: %.6f\n", logprob)

		if logprob < -1e20 {
			fmt.Println("  WARNING: LogProb ~ -inf (no valid paths?)")
		} else {
			fmt.Println("  OK: valid logprob")
		}

		subsampled.Free()

		// 9. Upload denominator FST (or use seq0 as fallback)
		var denFstGPU *nnet.ChainFstGPU
		if denCSR != nil {
			denFstGPU, err = nnet.NewChainFstGPU(denCSR)
			if err != nil {
				log.Fatalf("Upload den FST: %v", err)
			}
			defer denFstGPU.Free()
		} else {
			denFstGPU = seq0FstGPU
			fmt.Println("  (using seq0 num FST as denominator for testing)")
		}

		// 10. Full batch chain loss
		fmt.Println("\n--- Batch Chain LF-MMI Loss ---")

		// Compute total subsampled frames for gradient tensor
		totalSubsampled := 0
		for _, fps := range tb.FramesPerSeq {
			totalSubsampled += fps
		}
		fmt.Printf("Total subsampled frames: %d\n", totalSubsampled)

		grad, err := gpu.NewTensor(totalSubsampled, nnetOutput.Cols)
		if err != nil {
			log.Fatalf("alloc gradient: %v", err)
		}
		defer grad.Free()

		t2 := time.Now()
		result, err := nnet.ComputeChainLossBatch(
			nnetOutput,
			tb.PerSeqCSRs,
			denFstGPU,
			tb.FrameOffsets,
			tb.NumFrames,
			tb.FramesPerSeq, 3, 30,
			grad,
		)
		if err != nil {
			log.Fatalf("ComputeChainLossBatch: %v", err)
		}
		gpu.Sync()
		lossTime := time.Since(t2)

		fmt.Printf("  Time: %v\n", lossTime.Round(time.Microsecond))
		fmt.Printf("  Avg num logprob: %.6f\n", result.NumLogprob)
		fmt.Printf("  Avg den logprob: %.6f\n", result.DenLogprob)
		fmt.Printf("  Avg loss:        %.6f\n", result.Loss)

		// 11. Gradient check
		fmt.Println("\n--- Gradient Check ---")
		gradData, err := grad.ToFP32()
		if err != nil {
			log.Fatalf("Read gradient: %v", err)
		}

		nans, infs, zeros := 0, 0, 0
		var gradAbsSum float64
		var gradMin, gradMax float32 = 1e30, -1e30
		for _, v := range gradData {
			if v != v {
				nans++
				continue
			}
			if v > 1e15 || v < -1e15 {
				infs++
				continue
			}
			gradAbsSum += math.Abs(float64(v))
			if v < gradMin {
				gradMin = v
			}
			if v > gradMax {
				gradMax = v
			}
			if v == 0 {
				zeros++
			}
		}

		total := len(gradData)
		nonzero := total - zeros - nans - infs
		fmt.Printf("  Elements:  %d\n", total)
		fmt.Printf("  NaN:       %d\n", nans)
		fmt.Printf("  Inf:       %d\n", infs)
		fmt.Printf("  Zeros:     %d (%.1f%%)\n", zeros, 100*float64(zeros)/float64(total))
		fmt.Printf("  Non-zero:  %d (%.1f%%)\n", nonzero, 100*float64(nonzero)/float64(total))
		if nonzero > 0 {
			fmt.Printf("  Range:     [%.6f, %.6f]\n", gradMin, gradMax)
			fmt.Printf("  Abs mean:  %.6f\n", gradAbsSum/float64(total))
		}

		// 12. Memory
		freeAfter, _, _ := gpu.MemoryInfo()
		fmt.Printf("\nGPU memory used: %.1f MB\n", float64(freeBefore-freeAfter)/(1024*1024))

		// 13. Verdict
		fmt.Println()
		ok := true
		if nans > 0 {
			fmt.Println("WARNING: NaN in gradients")
			ok = false
		}
		if infs > 0 {
			fmt.Println("WARNING: Inf in gradients")
			ok = false
		}
		if zeros == total {
			fmt.Println("WARNING: All gradients zero")
			ok = false
		}
		if math.IsNaN(float64(result.Loss)) {
			fmt.Println("WARNING: NaN loss")
			ok = false
		}
		if ok {
			fmt.Println("Chain LF-MMI loss works!")
		}

		seq0FstGPU.Free()
	}
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
		return nil, fmt.Errorf("unsupported FST format (need compact_acceptor)")
	}

	csr, err := sparse.FstToCSR(fst)
	if err != nil {
		return nil, fmt.Errorf("to CSR: %w", err)
	}
	return csr, nil
}
