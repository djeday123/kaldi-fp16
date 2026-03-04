package main

import (
	"bufio"
	"fmt"
	"kaldi-fp16/internal/gpu"
	"kaldi-fp16/internal/loader"
	"kaldi-fp16/internal/nnet"
	"kaldi-fp16/internal/parser"
	"kaldi-fp16/internal/sparse"
	"math"
	"os"
)

const (
	xconfig    = "/opt/kaldi/egs/work_3/s5/exp/chain_cnn_v2/cnn_tdnn1d_v2_sp/configs/network.xconfig"
	modelMdl   = "/opt/kaldi/egs/work_3/s5/exp/chain_cnn_v2/cnn_tdnn1d_v2_sp/17820.mdl"
	egsArk     = "/opt/kaldi/egs/work_3/s5/exp/chain_cnn_v2/cnn_tdnn1d_v2_sp/egs/cegs.92.ark"
	denFstPath = "/opt/kaldi/egs/work_3/s5/exp/chain_cnn_v2/cnn_tdnn1d_v2_sp/den.fst"
)

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

func main() {
	fmt.Println("=== Parse model ===")
	model, err := nnet.BuildModel(xconfig)
	fmt.Println("=== Execution order ===")
	for _, layer := range model.ExecutionOrder() {
		inputs := ""
		for _, n := range layer.InputNames {
			inputs += n + " "
		}
		fmt.Printf("  %-20s [%-12s] in=%-5d out=%-5d inputs=[%s]\n",
			layer.Name, layer.Type, layer.InputDim, layer.OutputDim, inputs)
	}

	if err != nil {
		panic(err)
	}

	fmt.Println("=== Parse weights ===")
	text, err := nnet.ExportModelText(modelMdl)
	if err != nil {
		panic(err)
	}
	components, err := nnet.ParseNnet3Text(text)
	if err != nil {
		panic(err)
	}
	text = ""
	fmt.Printf("Components: %d\n", len(components))

	for _, name := range []string{"attention1.affine", "attention1.attention", "attention1.relu", "attention1.batchnorm"} {
		if c, ok := components[name]; ok {
			fmt.Printf("  comp %s: type=%s linear=%dx%d bias=%d mean=%d var=%d\n",
				name, c.Type, c.LinearRows, c.LinearCols, len(c.BiasParams),
				len(c.StatsMean), len(c.StatsVar))
		} else {
			fmt.Printf("  comp %s: NOT FOUND\n", name)
		}
	}

	fmt.Println("=== Load 1 batch from egs ===")
	for _, name := range []string{
		"tdnnf7.linear", "tdnnf7.affine",
		"tdnnf8.linear", "tdnnf8.affine",
		"tdnnf16.linear", "tdnnf16.affine",
		"tdnnf17.linear", "tdnnf17.affine",
	} {
		if c, ok := components[name]; ok {
			fmt.Printf("  %s: %dx%d\n", name, c.LinearRows, c.LinearCols)
		}
	}
	dl, err := loader.NewDataLoaderFromPaths([]string{egsArk}, 1, false)
	if err != nil {
		panic(err)
	}
	batch, err := dl.NextBatch()
	if err != nil {
		panic(err)
	}
	fmt.Printf("Batch: %d seqs, %d frames, feat=%d, ivec=%d\n",
		batch.BatchSize, batch.Features.Rows, batch.Features.Cols, batch.Ivectors.Cols)
	fmt.Printf("PerSeqCSRs: %d, FrameOffsets: %v, NumFrames: %v, FramesPerSeq: %v\n",
		len(batch.PerSeqCSRs), batch.FrameOffsets, batch.NumFrames, batch.FramesPerSeq)

	fmt.Println("=== Load den.fst ===")
	denCSR, err := loadDenFst(denFstPath)
	if err != nil {
		panic(fmt.Sprintf("loadDenFst: %v", err))
	}
	fmt.Printf("den.fst: %d states, %d arcs\n", denCSR.NumStates, denCSR.NumArcs)

	fmt.Println("=== GPU init ===")
	if err := gpu.Init(0); err != nil {
		panic(err)
	}
	handle, err := gpu.NewHandle()
	if err != nil {
		panic(err)
	}
	defer handle.Destroy()

	free, _, _ := gpu.MemoryInfo()
	fmt.Printf("GPU: %.0f MB free\n", float64(free)/1e6)

	net, err := nnet.NewNetworkFromKaldi(model, handle, components)
	if err != nil {
		panic(fmt.Sprintf("NewNetworkFromKaldi: %v", err))
	}
	defer net.Free()

	denFst, err := nnet.NewChainFstGPU(denCSR)
	if err != nil {
		panic(fmt.Sprintf("denFst GPU: %v", err))
	}
	defer denFst.Free()

	gpuBatch, err := gpu.TransferBatch(batch)
	if err != nil {
		panic(fmt.Sprintf("TransferBatch: %v", err))
	}
	defer gpuBatch.Free()

	features := &gpu.Tensor{
		Ptr:  gpuBatch.Features(),
		Rows: batch.Features.Rows,
		Cols: batch.Features.Cols,
	}
	var ivectors *gpu.Tensor
	if batch.Ivectors != nil && batch.Ivectors.Cols > 0 {
		ivectors = &gpu.Tensor{
			Ptr:  gpuBatch.Ivectors(),
			Rows: batch.BatchSize,
			Cols: batch.Ivectors.Cols,
		}
	}

	fmt.Println("=== Forward pass ===")
	state, err := net.Forward(features, ivectors)
	if err != nil {
		panic(fmt.Sprintf("Forward: %v", err))
	}
	defer state.Free()

	fmt.Printf("Output: %d x %d\n", state.Output.Rows, state.Output.Cols)

	outputData, err := state.Output.ToFP32()
	if err != nil {
		panic(fmt.Sprintf("ToFP32: %v", err))
	}
	minVal, maxVal := float32(math.MaxFloat32), float32(-math.MaxFloat32)
	sum := float64(0)
	nans := 0
	for _, v := range outputData {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			nans++
			continue
		}
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
		sum += float64(v)
	}
	mean := sum / float64(len(outputData)-nans)
	fmt.Printf("Stats: min=%.4f max=%.4f mean=%.6f nans=%d\n", minVal, maxVal, mean, nans)

	// Chain loss
	fmt.Println("=== Chain Loss ===")
	P := state.Output.Cols
	var totalLoss float64
	var totalFrames int

	for i := 0; i < batch.BatchSize; i++ {
		if i >= len(batch.PerSeqCSRs) {
			break
		}
		csr := batch.PerSeqCSRs[i]
		T := batch.NumFrames[i]
		offset := batch.FrameOffsets[i]
		fps := batch.FramesPerSeq[i]

		fmt.Printf("  seq %d: T=%d offset=%d fps=%d fst_states=%d fst_arcs=%d\n",
			i, T, offset, fps, csr.NumStates, csr.NumArcs)

		seqOutput := state.Output.View(offset, T, P)

		// Subsample: every 3rd frame → ~fps frames
		subsampled, err := gpu.SubsampleRows(seqOutput, 3, 0)
		if err != nil {
			fmt.Printf("  seq %d: subsample failed: %v\n", i, err)
			continue
		}
		fmt.Printf("  seq %d: subsampled %d → %d frames (need %d)\n",
			i, T, subsampled.Rows, fps)

		// Use View to trim to exact fps if subsampled has more rows
		var lossInput *gpu.Tensor
		if subsampled.Rows >= fps {
			lossInput = subsampled.View(0, fps, P)
		} else {
			lossInput = subsampled
		}

		numFst, err := nnet.NewChainFstGPU(csr)
		if err != nil {
			subsampled.Free()
			fmt.Printf("  seq %d: num FST upload failed: %v\n", i, err)
			continue
		}

		result, err := nnet.ComputeChainLoss(lossInput, numFst, denFst, nil)
		numFst.Free()
		subsampled.Free()
		if err != nil {
			fmt.Printf("  seq %d: chain loss failed: %v\n", i, err)
			continue
		}

		perFrame := result.Loss / float32(fps)
		fmt.Printf("  seq %d: num=%.4f den=%.4f loss=%.4f loss/frame=%.6f\n",
			i, result.NumLogprob, result.DenLogprob, result.Loss, perFrame)

		totalLoss += float64(result.Loss)
		totalFrames += fps
	}

	if totalFrames > 0 {
		fmt.Printf("\nTotal: loss/frame=%.6f over %d frames\n", totalLoss/float64(totalFrames), totalFrames)
		fmt.Printf("Kaldi reference: -0.0601 per frame over 115 frames\n")
	}

	fmt.Println("=== DONE ===")
}
