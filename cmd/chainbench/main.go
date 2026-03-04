package main

/*
#cgo CFLAGS: -I/usr/local/cuda-12.8/include
#cgo LDFLAGS: -L${SRCDIR}/../../cpp/build -lkaldi_fp16 -L/usr/local/cuda-12.8/targets/x86_64-linux/lib -lcudart -lcublas
#include <cuda_runtime.h>
*/
import "C"

import (
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"time"
	"unsafe"

	"kaldi-fp16/internal/gpu"
	"kaldi-fp16/internal/loader"
	"kaldi-fp16/internal/nnet"
)

func main() {
	numArks := flag.Int("arks", 10, "number of ark files to process")
	flag.Parse()

	fmt.Printf("=== Chain Atomic vs Deterministic Benchmark ===\n")
	fmt.Printf("Processing %d ark files\n\n", *numArks)

	gpu.Init(0)

	// Find ark files
	egsDir := "/opt/kaldi/egs/work_3/s5/exp/chain_cnn_v2/cnn_tdnn1d_v2_sp/egs"
	pattern := filepath.Join(egsDir, "cegs.*.ark")
	arkFiles, err := filepath.Glob(pattern)
	if err != nil {
		log.Fatal(err)
	}
	sort.Strings(arkFiles)

	if *numArks > len(arkFiles) {
		*numArks = len(arkFiles)
	}
	arkFiles = arkFiles[:*numArks]
	fmt.Printf("Using %d ark files\n", len(arkFiles))

	numPdfs := 3080

	// Load denominator
	kaldiDen, err := nnet.NewKaldiDenominator(
		"/opt/kaldi/egs/work_3/s5/exp/chain_cnn_v2/cnn_tdnn1d_v2_sp/den.fst",
		numPdfs)
	if err != nil {
		log.Fatal("kaldi den:", err)
	}
	defer kaldiDen.Free()

	// Stats
	var totalExamples int
	var totalFrames int64
	var totalObjfAtomic, totalObjfDet float64
	var totalNumLPAtomic, totalNumLPDet float64
	var totalDenLP float64
	var diffs []float64 // per-example |atomic - det| for num_logprob

	startTime := time.Now()

	for arkIdx, arkFile := range arkFiles {
		arkStart := time.Now()
		arkExamples := 0
		arkObjfAtomic := 0.0
		arkObjfDet := 0.0

		dl, err := loader.NewDataLoaderFromPaths([]string{arkFile}, 1, false)
		if err != nil {
			log.Printf("WARN: skip %s: %v", arkFile, err)
			continue
		}

		for {
			tb, err := dl.NextBatch()
			if tb == nil && err == nil {
				break
			}
			if err != nil {
				break // end of ark
			}
			if len(tb.PerSeqCSRs) == 0 || len(tb.FramesPerSeq) == 0 {
				break
			}

			fps := tb.FramesPerSeq[0]
			csr := tb.PerSeqCSRs[0]

			if fps <= 0 || csr.NumStates <= 0 {
				continue
			}

			// Create random-ish nnet_output (use zeros for speed —
			// the point is comparing atomic vs det, not absolute values)
			// Actually we need non-zero to see real differences
			nnetData := make([]float32, fps*numPdfs)
			// Simple deterministic fill based on example index
			seed := uint32(totalExamples*7 + 13)
			for i := range nnetData {
				seed = seed*1103515245 + 12345
				nnetData[i] = float32(int32(seed>>16)&0x7FFF-16384) / 163840.0 // [-0.1, 0.1]
			}

			// Upload to GPU as FP16 tensor (for atomic version)
			nnetTensor, err := gpu.TensorFromFP32(nnetData, fps, numPdfs)
			if err != nil {
				log.Printf("WARN: TensorFromFP32: %v", err)
				continue
			}

			// Upload num FST
			fstGPU, err := nnet.NewChainFstGPU(csr)
			if err != nil {
				nnetTensor.Free()
				log.Printf("WARN: NewChainFstGPU: %v", err)
				continue
			}

			// === Atomic forward-backward ===
			atomicLP, err := nnet.ForwardBackward(nnetTensor, fstGPU)
			if err != nil {
				fstGPU.Free()
				nnetTensor.Free()
				continue
			}

			// === Deterministic forward-backward ===
			// Upload FP32 to GPU
			nnetFP32Size := fps * numPdfs * 4
			var nnetFP32GPU unsafe.Pointer
			C.cudaMalloc(&nnetFP32GPU, C.size_t(nnetFP32Size))
			C.cudaMemcpy(nnetFP32GPU, unsafe.Pointer(&nnetData[0]),
				C.size_t(nnetFP32Size), C.cudaMemcpyHostToDevice)

			var numPostGPU unsafe.Pointer
			C.cudaMalloc(&numPostGPU, C.size_t(nnetFP32Size))

			csrGPU := nnet.GetCSROnGPU(fstGPU)
			detLP := nnet.ForwardBackwardDet(csrGPU, nnetFP32GPU, numPostGPU, fps, numPdfs)

			C.cudaFree(nnetFP32GPU)
			C.cudaFree(numPostGPU)

			// === Denominator ===
			denLP, err := kaldiDen.Forward(nnetData, fps, 1)
			if err != nil {
				fstGPU.Free()
				nnetTensor.Free()
				continue
			}

			fstGPU.Free()
			nnetTensor.Free()

			// Accumulate
			objfAtomic := float64(atomicLP) - float64(denLP)
			objfDet := detLP - float64(denLP)
			diff := math.Abs(float64(atomicLP) - detLP)

			totalNumLPAtomic += float64(atomicLP)
			totalNumLPDet += detLP
			totalDenLP += float64(denLP)
			totalObjfAtomic += objfAtomic
			totalObjfDet += objfDet
			diffs = append(diffs, diff)
			totalFrames += int64(fps)
			totalExamples++
			arkExamples++
			arkObjfAtomic += objfAtomic
			arkObjfDet += objfDet
		}

		elapsed := time.Since(arkStart)
		fmt.Printf("ark %3d: %4d examples, objf_atomic=%.2f objf_det=%.2f |diff|=%.4f  [%v]\n",
			arkIdx+1, arkExamples,
			arkObjfAtomic/float64(arkExamples),
			arkObjfDet/float64(arkExamples),
			math.Abs(arkObjfAtomic-arkObjfDet)/float64(arkExamples),
			elapsed.Round(time.Second))
	}

	totalElapsed := time.Since(startTime)

	// Compute stats
	var sumDiff, sumDiffSq, maxDiff float64
	for _, d := range diffs {
		sumDiff += d
		sumDiffSq += d * d
		if d > maxDiff {
			maxDiff = d
		}
	}
	meanDiff := sumDiff / float64(len(diffs))
	variance := sumDiffSq/float64(len(diffs)) - meanDiff*meanDiff
	stdDiff := math.Sqrt(math.Max(0, variance))

	// Sort for percentiles
	sort.Float64s(diffs)
	p50 := diffs[len(diffs)/2]
	p95 := diffs[int(float64(len(diffs))*0.95)]
	p99 := diffs[int(float64(len(diffs))*0.99)]

	fmt.Println()
	fmt.Println("========================================")
	fmt.Printf("Total examples:  %d\n", totalExamples)
	fmt.Printf("Total frames:    %d\n", totalFrames)
	fmt.Printf("Total time:      %v\n", totalElapsed.Round(time.Second))
	fmt.Printf("Examples/sec:    %.1f\n", float64(totalExamples)/totalElapsed.Seconds())
	fmt.Println()
	fmt.Println("--- Accumulated objective ---")
	fmt.Printf("total_objf_atomic:  %.6f  (per_frame: %.6f)\n",
		totalObjfAtomic, totalObjfAtomic/float64(totalFrames))
	fmt.Printf("total_objf_det:     %.6f  (per_frame: %.6f)\n",
		totalObjfDet, totalObjfDet/float64(totalFrames))
	fmt.Printf("total_objf_diff:    %.6f  (per_frame: %.2e)\n",
		totalObjfAtomic-totalObjfDet,
		(totalObjfAtomic-totalObjfDet)/float64(totalFrames))
	fmt.Println()
	fmt.Println("--- Numerator logprob ---")
	fmt.Printf("total_num_lp_atomic: %.6f\n", totalNumLPAtomic)
	fmt.Printf("total_num_lp_det:    %.6f\n", totalNumLPDet)
	fmt.Printf("total_num_lp_diff:   %.6f\n", totalNumLPAtomic-totalNumLPDet)
	fmt.Println()
	fmt.Println("--- Per-example |atomic - det| num_logprob ---")
	fmt.Printf("mean:   %.2e\n", meanDiff)
	fmt.Printf("std:    %.2e\n", stdDiff)
	fmt.Printf("max:    %.2e\n", maxDiff)
	fmt.Printf("p50:    %.2e\n", p50)
	fmt.Printf("p95:    %.2e\n", p95)
	fmt.Printf("p99:    %.2e\n", p99)

	// Write CSV for analysis
	csvPath := "/tmp/chain_atomic_vs_det.csv"
	csvFile, err := os.Create(csvPath)
	if err == nil {
		fmt.Fprintf(csvFile, "example,diff_num_logprob\n")
		for i, d := range diffs {
			fmt.Fprintf(csvFile, "%d,%.10e\n", i, d)
		}
		csvFile.Close()
		fmt.Printf("\nCSV saved: %s\n", csvPath)
	}
}
