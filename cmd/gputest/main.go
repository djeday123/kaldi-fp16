package main

import (
	"fmt"
	"log"
	"math"
	"os"
	"time"

	"kaldi-fp16/internal/gpu"
	"kaldi-fp16/internal/loader"
)

func main() {
	// Configuration
	arkPattern := "/opt/kaldi/egs/work_3/s5/exp/cnn_tdnn_50h_bn192_l2_001/egs/cegs.*.ark"
	batchSize := 64
	deviceID := 0
	maxBatches := 20

	if len(os.Args) > 1 {
		arkPattern = os.Args[1]
	}

	// 1. Initialize GPU
	fmt.Println("=== GPU Bridge Test ===")
	fmt.Println()

	if err := gpu.Init(deviceID); err != nil {
		log.Fatalf("GPU init failed: %v", err)
	}

	free, total, err := gpu.MemoryInfo()
	if err != nil {
		log.Fatalf("GPU memory info failed: %v", err)
	}
	fmt.Printf("GPU %d: %.1f GB free / %.1f GB total\n", deviceID,
		float64(free)/(1<<30), float64(total)/(1<<30))

	// 2. Create DataLoader
	dl, err := loader.NewDataLoader(loader.DataLoaderConfig{
		Pattern:   arkPattern,
		BatchSize: batchSize,
		Shuffle:   false,
		DropLast:  true,
		Verbose:   false,
	})
	if err != nil {
		log.Fatalf("DataLoader failed: %v", err)
	}

	// 3. Allocate pinned buffer (reused across batches)
	pinned, err := gpu.NewPinnedBuffer(32 * 1024 * 1024) // 32 MB initial
	if err != nil {
		log.Printf("Pinned alloc failed, using regular memory: %v", err)
		pinned = nil
	} else {
		defer pinned.Free()
	}

	// 4. Transfer batches and verify
	fmt.Printf("\nTransferring up to %d batches (batch_size=%d)...\n\n", maxBatches, batchSize)

	var (
		totalBatches    int
		totalFrames     int
		totalBytes      uint64
		totalTransferMs float64
		totalParseMs    float64
		maxFeatErr      float32
		maxIvecErr      float32
	)

	for i := 0; i < maxBatches; i++ {
		// Parse batch on CPU
		t0 := time.Now()
		tb, err := dl.NextBatch()
		if err != nil {
			log.Fatalf("NextBatch failed: %v", err)
		}
		if tb == nil {
			fmt.Printf("  EOF after %d batches\n", i)
			break
		}
		parseTime := time.Since(t0)

		// Transfer to GPU
		t1 := time.Now()
		var gb *gpu.GPUBatch
		if pinned != nil {
			gb, err = gpu.TransferBatchPinned(tb, pinned)
		} else {
			gb, err = gpu.TransferBatch(tb)
		}
		if err != nil {
			log.Fatalf("Transfer failed batch %d: %v", i, err)
		}
		transferTime := time.Since(t1)

		// Verify: read back and compare
		featBack, err := gb.ReadFeatures()
		if err != nil {
			log.Fatalf("ReadFeatures failed: %v", err)
		}

		// Compare with FP16 round-trip precision
		featErr := maxAbsErr(tb.Features.Data, featBack)
		if featErr > maxFeatErr {
			maxFeatErr = featErr
		}

		ivecErr := float32(0)
		if tb.Ivectors != nil {
			ivecBack, err := gb.ReadIvectors()
			if err != nil {
				log.Fatalf("ReadIvectors failed: %v", err)
			}
			ivecErr = maxAbsErr(tb.Ivectors.Data, ivecBack)
			if ivecErr > maxIvecErr {
				maxIvecErr = ivecErr
			}
		}

		if i < 5 || i == maxBatches-1 {
			fmt.Printf("  batch %2d: %3d examples, %5d frames, %5d states, %6d arcs | "+
				"parse %.1fms, transfer %.1fms (%.1f MB) | featErr=%.2e ivecErr=%.2e\n",
				i, tb.BatchSize, tb.Features.Rows,
				tb.FstCSR.NumStates, tb.FstCSR.NumArcs,
				float64(parseTime.Microseconds())/1000.0,
				float64(transferTime.Microseconds())/1000.0,
				float64(gb.TotalBytes())/(1024*1024),
				featErr, ivecErr,
			)
		} else if i == 5 {
			fmt.Println("  ...")
		}

		totalBatches++
		totalFrames += tb.Features.Rows
		totalBytes += gb.TotalBytes()
		totalTransferMs += float64(transferTime.Microseconds()) / 1000.0
		totalParseMs += float64(parseTime.Microseconds()) / 1000.0

		gb.Free()
	}

	// 5. Summary
	fmt.Println()
	fmt.Println("=== Summary ===")
	fmt.Printf("Batches:      %d\n", totalBatches)
	fmt.Printf("Total frames: %d\n", totalFrames)
	fmt.Printf("GPU memory:   %.1f MB total transferred\n", float64(totalBytes)/(1024*1024))
	fmt.Printf("Parse time:   %.1f ms total (%.2f ms/batch)\n",
		totalParseMs, totalParseMs/float64(totalBatches))
	fmt.Printf("Transfer time: %.1f ms total (%.2f ms/batch)\n",
		totalTransferMs, totalTransferMs/float64(totalBatches))
	fmt.Printf("Throughput:   %.0f frames/sec (transfer only)\n",
		float64(totalFrames)/(totalTransferMs/1000.0))
	fmt.Printf("Max feat error (FP16 round-trip): %.2e\n", maxFeatErr)
	fmt.Printf("Max ivec error (FP16 round-trip): %.2e\n", maxIvecErr)

	if pinned != nil {
		fmt.Println("Transfer mode: PINNED memory (DMA)")
	} else {
		fmt.Println("Transfer mode: regular memory")
	}

	fmt.Println("\n✓ DataLoader → FP16 → GPU pipeline working!")
}

// maxAbsErr computes max |a - b| where b has FP16 round-trip error
func maxAbsErr(original, roundTripped []float32) float32 {
	if len(original) != len(roundTripped) {
		return float32(math.MaxFloat32)
	}
	var maxErr float32
	for i := range original {
		err := float32(math.Abs(float64(original[i] - roundTripped[i])))
		if err > maxErr {
			maxErr = err
		}
	}
	return maxErr
}
