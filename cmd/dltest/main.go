package main

import (
	"fmt"
	"log"
	"math"
	"time"

	"kaldi-fp16/internal/fp16"
	"kaldi-fp16/internal/loader"
)

const egsPath = "/opt/kaldi/egs/work_3/s5/exp/chain_cnn_v2/cnn_tdnn1d_v2_sp/egs/cegs.?.ark"

func main() {
	fmt.Println("=== DataLoader Real Data Test (ALL FILES) ===")
	fmt.Printf("Pattern: %s\n\n", egsPath)

	dl, err := loader.NewDataLoader(loader.DataLoaderConfig{
		Pattern:   egsPath,
		BatchSize: 64,
		Shuffle:   false,
		Verbose:   false,
	})
	if err != nil {
		log.Fatalf("Failed to create DataLoader: %v", err)
	}
	defer dl.Close()

	batchNum := 0
	totalExamples := 0
	totalFrames := 0
	totalStates := 0
	totalArcs := 0
	totalFP16Time := time.Duration(0)
	csrErrors := 0

	// Aggregate FP16 stats across ALL batches
	var globalMaxAbsErr float32
	var globalMaxRelErr float32
	var globalSumAbsErr float64
	var globalTotalValues int
	var globalExactCount int
	var globalOverflowCount int  // values that became Inf
	var globalUnderflowCount int // values that became 0

	// Ivector stats
	var ivMaxAbsErr float32
	var ivMaxRelErr float32
	var ivSumAbsErr float64
	var ivTotalValues int

	start := time.Now()

	for {
		tb, err := dl.NextBatch()
		if err != nil {
			log.Printf("Batch error: %v", err)
			continue
		}
		if tb == nil {
			break
		}

		batchNum++
		totalExamples += tb.BatchSize
		totalFrames += tb.Features.Rows
		totalStates += tb.FstCSR.NumStates
		totalArcs += tb.FstCSR.NumArcs

		// CSR validation on every batch
		if err := tb.FstCSR.Validate(); err != nil {
			csrErrors++
			if csrErrors <= 5 {
				fmt.Printf("  ⚠️  Batch %d CSR FAILED: %v\n", batchNum, err)
			}
		}

		// FP16 analysis on every batch
		featStats := fp16.AnalyzeConversion(tb.Features.Data)
		globalTotalValues += featStats.Count
		globalExactCount += featStats.NumExact
		globalOverflowCount += featStats.NumInf
		globalUnderflowCount += featStats.NumZero
		globalSumAbsErr += float64(featStats.AvgAbsErr) * float64(featStats.Count)
		if featStats.MaxAbsErr > globalMaxAbsErr {
			globalMaxAbsErr = featStats.MaxAbsErr
		}
		if featStats.MaxRelErr > globalMaxRelErr {
			fmt.Printf("  🔍 Batch %d: NEW maxRelErr=%.6f (%.2f%%)  FP32=%.10f → FP16=%.10f  absErr=%.10f\n",
				batchNum, featStats.MaxRelErr, featStats.MaxRelErr*100,
				featStats.WorstOriginal, featStats.WorstConverted,
				featStats.WorstOriginal-featStats.WorstConverted)
			globalMaxRelErr = featStats.MaxRelErr
		}

		// Ivector stats
		if tb.Ivectors != nil {
			ivStats := fp16.AnalyzeConversion(tb.Ivectors.Data)
			ivTotalValues += ivStats.Count
			ivSumAbsErr += float64(ivStats.AvgAbsErr) * float64(ivStats.Count)
			if ivStats.MaxAbsErr > ivMaxAbsErr {
				ivMaxAbsErr = ivStats.MaxAbsErr
			}
			if ivStats.MaxRelErr > ivMaxRelErr {
				fmt.Printf("  🔍 Batch %d: NEW iv maxRelErr=%.6f (%.2f%%)  FP32=%.10f → FP16=%.10f  absErr=%.10f\n",
					batchNum, ivStats.MaxRelErr, ivStats.MaxRelErr*100,
					ivStats.WorstOriginal, ivStats.WorstConverted,
					ivStats.WorstOriginal-ivStats.WorstConverted)
				ivMaxRelErr = ivStats.MaxRelErr
			}
		}

		// Print first 3 batches detailed
		if batchNum <= 3 {
			fmt.Printf("Batch %d:\n", batchNum)
			fmt.Printf("  Examples:     %d\n", tb.BatchSize)
			fmt.Printf("  Features:     %d × %d\n", tb.Features.Rows, tb.Features.Cols)
			fmt.Printf("  FST:          %d states, %d arcs\n", tb.FstCSR.NumStates, tb.FstCSR.NumArcs)
			fmt.Printf("  FP16 feat:    maxAbsErr=%.6f maxRelErr=%.6f\n", featStats.MaxAbsErr, featStats.MaxRelErr)
			fmt.Println()
		}

		if batchNum%500 == 0 {
			elapsed := time.Since(start)
			avgAbsErr := float64(0)
			if globalTotalValues > 0 {
				avgAbsErr = globalSumAbsErr / float64(globalTotalValues)
			}
			fmt.Printf("  ... batch %d: %d examples, %.1f batch/s, maxRelErr=%.6f avgAbsErr=%.6f\n",
				batchNum, totalExamples, float64(batchNum)/elapsed.Seconds(),
				globalMaxRelErr, avgAbsErr)
		}

		// FP16 conversion timing
		fp16Start := time.Now()
		fp16.ConvertFloat32ToFloat16(tb.Features.Data)
		if tb.Ivectors != nil {
			fp16.ConvertFloat32ToFloat16(tb.Ivectors.Data)
		}
		totalFP16Time += time.Since(fp16Start)
	}

	elapsed := time.Since(start)
	globalAvgAbsErr := float32(0)
	if globalTotalValues > 0 {
		globalAvgAbsErr = float32(globalSumAbsErr / float64(globalTotalValues))
	}
	ivAvgAbsErr := float32(0)
	if ivTotalValues > 0 {
		ivAvgAbsErr = float32(ivSumAbsErr / float64(ivTotalValues))
	}

	fmt.Println("\n=== Summary ===")
	fmt.Printf("Total batches:    %d\n", batchNum)
	fmt.Printf("Total examples:   %d\n", totalExamples)
	fmt.Printf("Total frames:     %d\n", totalFrames)
	fmt.Printf("Total FST states: %d\n", totalStates)
	fmt.Printf("Total FST arcs:   %d\n", totalArcs)
	fmt.Printf("CSR errors:       %d\n", csrErrors)

	fmt.Println("\n=== FP16 Precision (Features) ===")
	fmt.Printf("Total values:     %d\n", globalTotalValues)
	fmt.Printf("Max abs error:    %.6f\n", globalMaxAbsErr)
	fmt.Printf("Avg abs error:    %.6f\n", globalAvgAbsErr)
	fmt.Printf("Max rel error:    %.6f (%.4f%%)\n", globalMaxRelErr, globalMaxRelErr*100)
	fmt.Printf("Exact (bit-perfect): %d / %d (%.2f%%)\n",
		globalExactCount, globalTotalValues,
		100.0*float64(globalExactCount)/math.Max(float64(globalTotalValues), 1))
	fmt.Printf("Overflow (→Inf):  %d\n", globalOverflowCount)
	fmt.Printf("Underflow (→0):   %d\n", globalUnderflowCount)
	fmt.Printf("Memory:           %d MB → %d MB (2x)\n",
		globalTotalValues*4/1024/1024, globalTotalValues*2/1024/1024)

	fmt.Println("\n=== FP16 Precision (Ivectors) ===")
	fmt.Printf("Total values:     %d\n", ivTotalValues)
	fmt.Printf("Max abs error:    %.6f\n", ivMaxAbsErr)
	fmt.Printf("Avg abs error:    %.6f\n", ivAvgAbsErr)
	fmt.Printf("Max rel error:    %.6f (%.4f%%)\n", ivMaxRelErr, ivMaxRelErr*100)

	fmt.Println("\n=== Performance ===")
	fmt.Printf("Total time:       %.2fs\n", elapsed.Seconds())
	fmt.Printf("FP16 time:        %.2fs (%.1f%%)\n", totalFP16Time.Seconds(),
		100.0*totalFP16Time.Seconds()/elapsed.Seconds())
	if batchNum > 0 {
		fmt.Printf("Avg batch time:   %.1f ms\n", float64(elapsed.Milliseconds())/float64(batchNum))
		fmt.Printf("Throughput:       %.0f examples/sec\n", float64(totalExamples)/elapsed.Seconds())
	}
}
