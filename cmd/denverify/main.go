package main

// Verification: NativeDenominator vs KaldiDenominator (wrapper)
//
// Test with zero nnet output (all PDFs equal probability).
// Both systems should produce identical log-prob.
//
// Usage:
//   go run cmd/denverify/main.go \
//     -den /opt/kaldi/egs/work_3/s5/exp/chain_cnn_v2/cnn_tdnn1d_v2_sp/den.fst \
//     -egs /opt/kaldi/egs/work_3/s5/exp/chain_cnn_v2/cnn_tdnn1d_v2_sp/egs/cegs.1.ark \
//     -pdfs 3080

import (
	"flag"
	"fmt"
	"math"
	"os"

	"kaldi-fp16/internal/loader"
	"kaldi-fp16/internal/nnet"
)

func main() {
	denFst := flag.String("den", "", "Path to den.fst")
	egsPath := flag.String("egs", "", "Path to cegs.*.ark (for getting T from real data)")
	numPdfs := flag.Int("pdfs", 3080, "Number of PDFs")
	T := flag.Int("T", 34, "Number of frames (used if no egs path)")
	flag.Parse()

	if *denFst == "" {
		fmt.Fprintln(os.Stderr, "Usage: denverify -den <den.fst> [-egs <cegs.ark>] [-pdfs 3080] [-T 34]")
		os.Exit(1)
	}

	// ============================================================
	// Determine T (number of output frames)
	// ============================================================
	framesPerSeq := *T
	if *egsPath != "" {
		fmt.Printf("Loading first example from %s to get FramesPerSeq...\n", *egsPath)
		dl, err := loader.NewDataLoaderFromPaths([]string{*egsPath}, 1, false)
		if err != nil {
			fmt.Fprintf(os.Stderr, "DataLoader error: %v\n", err)
			os.Exit(1)
		}
		batch, err := dl.NextBatch()
		if err != nil {
			fmt.Fprintf(os.Stderr, "NextBatch error: %v\n", err)
			os.Exit(1)
		}
		if len(batch.FramesPerSeq) > 0 {
			framesPerSeq = batch.FramesPerSeq[0]
		}
		fmt.Printf("Using FramesPerSeq=%d from first example\n", framesPerSeq)
	}

	fmt.Printf("\n=== Verification: T=%d, P=%d ===\n\n", framesPerSeq, *numPdfs)

	// Create zero nnet output [T × P]
	nnetOutput := make([]float32, framesPerSeq*(*numPdfs))
	// All zeros → exp(0) = 1.0 for all PDFs

	// ============================================================
	// 1. Native Denominator
	// ============================================================
	fmt.Println("--- Native Denominator ---")
	native, err := nnet.NewNativeDenominator(*denFst, *numPdfs)
	if err != nil {
		fmt.Fprintf(os.Stderr, "NativeDenominator init error: %v\n", err)
		os.Exit(1)
	}
	defer native.Free()

	nativeLogprob, err := native.Forward(nnetOutput, framesPerSeq, 1)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Native forward error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("native  log_prob = %.6f  (per_frame = %.6f)\n",
		nativeLogprob, nativeLogprob/float32(framesPerSeq))

	// ============================================================
	// 2. Kaldi Wrapper Denominator
	// ============================================================
	fmt.Println("\n--- Kaldi Wrapper Denominator ---")
	kaldi, err := nnet.NewKaldiDenominator(*denFst, *numPdfs)
	if err != nil {
		fmt.Fprintf(os.Stderr, "KaldiDenominator init error: %v\n", err)
		fmt.Println("(Kaldi wrapper not available — skip comparison)")
		fmt.Printf("\nNative-only result: log_prob = %.6f\n", nativeLogprob)
		return
	}
	defer kaldi.Free()

	kaldiLogprob, err := kaldi.Forward(nnetOutput, framesPerSeq, 1)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Kaldi forward error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("kaldi   log_prob = %.6f  (per_frame = %.6f)\n",
		kaldiLogprob, kaldiLogprob/float32(framesPerSeq))

	// ============================================================
	// 3. Compare
	// ============================================================
	diff := math.Abs(float64(nativeLogprob - kaldiLogprob))
	relDiff := diff / math.Abs(float64(kaldiLogprob))

	fmt.Printf("\n=== Comparison ===\n")
	fmt.Printf("abs_diff  = %.6e\n", diff)
	fmt.Printf("rel_diff  = %.6e\n", relDiff)

	if diff < 0.01 {
		fmt.Printf("\n✅ MATCH (diff < 0.01)\n")
	} else if diff < 0.1 {
		fmt.Printf("\n⚠️  CLOSE (diff < 0.1) — check precision\n")
	} else {
		fmt.Printf("\n❌ MISMATCH (diff = %.4f)\n", diff)
		os.Exit(1)
	}

	// ============================================================
	// 4. Test with gradients
	// ============================================================
	fmt.Println("\n--- Forward-Backward with gradients ---")

	nativeLP, nativeGrad, err := native.ForwardBackward(nnetOutput, framesPerSeq, 1, 1.0)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Native FB error: %v\n", err)
		os.Exit(1)
	}

	kaldiLP, kaldiGrad, err := kaldi.ForwardBackward(nnetOutput, framesPerSeq, 1, 1.0)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Kaldi FB error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("native  FB log_prob = %.6f\n", nativeLP)
	fmt.Printf("kaldi   FB log_prob = %.6f\n", kaldiLP)

	// Compare gradients
	maxGradDiff := 0.0
	sumGradDiff := 0.0
	nGrad := len(nativeGrad)
	if len(kaldiGrad) < nGrad {
		nGrad = len(kaldiGrad)
	}
	for i := 0; i < nGrad; i++ {
		d := math.Abs(float64(nativeGrad[i] - kaldiGrad[i]))
		sumGradDiff += d
		if d > maxGradDiff {
			maxGradDiff = d
		}
	}
	avgGradDiff := sumGradDiff / float64(nGrad)

	fmt.Printf("grad max_diff = %.6e\n", maxGradDiff)
	fmt.Printf("grad avg_diff = %.6e\n", avgGradDiff)

	if maxGradDiff < 0.001 {
		fmt.Printf("\n✅ Gradients MATCH\n")
	} else {
		fmt.Printf("\n⚠️  Gradient diff > 0.001, may need investigation\n")
	}

	// Gradient histogram
	buckets := map[string]int{
		"<1e-5": 0, "1e-5..1e-4": 0, "1e-4..1e-3": 0,
		"1e-3..1e-2": 0, ">1e-2": 0,
	}
	for i := 0; i < nGrad; i++ {
		d := math.Abs(float64(nativeGrad[i] - kaldiGrad[i]))
		switch {
		case d < 1e-5:
			buckets["<1e-5"]++
		case d < 1e-4:
			buckets["1e-5..1e-4"]++
		case d < 1e-3:
			buckets["1e-4..1e-3"]++
		case d < 1e-2:
			buckets["1e-3..1e-2"]++
		default:
			buckets[">1e-2"]++
		}
	}
	fmt.Printf("\nGradient diff histogram:\n")
	for _, k := range []string{"<1e-5", "1e-5..1e-4", "1e-4..1e-3", "1e-3..1e-2", ">1e-2"} {
		pct := 100.0 * float64(buckets[k]) / float64(nGrad)
		fmt.Printf("  %12s: %6d (%5.1f%%)\n", k, buckets[k], pct)
	}
}
