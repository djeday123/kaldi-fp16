// backward_test.go — Numerical gradient verification for chain backward pass
//
// Verifies analytical gradient against finite differences:
//   numerical_grad[t][p] ≈ (loss(x + ε*e_tp) - loss(x - ε*e_tp)) / (2ε)
//
// This should match:
//   analytical_grad[t][p] = sup_weight * (num_post[t][p] - den_post[t][p])

package nnet

import (
	"fmt"
	"math"
	"testing"
)

// TestChainGradientNumerical verifies backward pass using finite differences
//
// Algorithm:
//
//	For a subset of (t, pdf) positions:
//	  1. Compute loss with nnet_output[t][pdf] + epsilon
//	  2. Compute loss with nnet_output[t][pdf] - epsilon
//	  3. numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
//	  4. Compare with analytical gradient from backward pass
//
// This is the gold standard for gradient verification.
func TestChainGradientNumerical(t *testing.T) {
	// Test parameters
	const (
		T          = 10   // frames (small for testing)
		numPdfs    = 20   // pdf-ids (small for testing)
		epsilon    = 1e-4 // finite difference step
		tolerance  = 1e-3 // relative tolerance
		supWeight  = 1.0
		numSamples = 50 // how many (t, pdf) positions to check
	)

	t.Log("=== Numerical Gradient Check for Chain Backward Pass ===")
	t.Logf("T=%d, numPdfs=%d, epsilon=%e, tolerance=%e", T, numPdfs, epsilon, tolerance)

	// 1. Create a small synthetic numerator FST
	//    (simple linear chain: state0 → state1 → ... → stateT)
	numStates := T + 1
	numArcs := T
	// Each arc: src=t, dst=t+1, pdf=t%numPdfs, weight=0.0
	rowPtr := make([]int32, numStates+1)
	colIdx := make([]int32, numArcs)
	weights := make([]float32, numArcs)
	pdfIds := make([]int32, numArcs)

	for i := 0; i < T; i++ {
		rowPtr[i] = int32(i)
		colIdx[i] = int32(i + 1)
		weights[i] = 0.0 // log-prob = 0 → prob = 1
		pdfIds[i] = int32(i % numPdfs)
	}
	rowPtr[T] = int32(T) // last state has no outgoing arcs

	// 2. Create random nnet_output [T × numPdfs]
	nnetOutput := make([]float32, T*numPdfs)
	for i := range nnetOutput {
		// Small random values (deterministic for reproducibility)
		nnetOutput[i] = float32(math.Sin(float64(i)*0.1)) * 0.5
	}

	// 3. Compute analytical gradient via backward pass
	analyticalGrad := make([]float32, T*numPdfs)
	numPost := make([]float32, T*numPdfs)
	denPost := make([]float32, T*numPdfs)

	// CPU reference: compute numerator and denominator posteriors
	// (This would call the actual GPU functions in the real test)
	baseLoss := computeChainLossCPU(nnetOutput, rowPtr, colIdx, weights, pdfIds,
		numPost, denPost, T, numPdfs, numStates)

	// Analytical gradient: sup_weight * (num_post - den_post)
	for i := 0; i < T*numPdfs; i++ {
		analyticalGrad[i] = supWeight * (numPost[i] - denPost[i])
	}

	// 4. Numerical gradient check
	maxRelError := 0.0
	maxAbsError := 0.0
	numChecked := 0

	// Check a subset of positions
	step := T * numPdfs / numSamples
	if step < 1 {
		step = 1
	}

	for idx := 0; idx < T*numPdfs && numChecked < numSamples; idx += step {
		frame := idx / numPdfs
		pdf := idx % numPdfs

		// loss(x + ε)
		nnetOutput[idx] += float32(epsilon)
		lossPlus := computeChainLossCPU(nnetOutput, rowPtr, colIdx, weights, pdfIds,
			nil, nil, T, numPdfs, numStates)

		// loss(x - ε) (subtract 2ε from the already +ε value)
		nnetOutput[idx] -= 2 * float32(epsilon)
		lossMinus := computeChainLossCPU(nnetOutput, rowPtr, colIdx, weights, pdfIds,
			nil, nil, T, numPdfs, numStates)

		// Restore original value
		nnetOutput[idx] += float32(epsilon)

		// Numerical gradient
		numGrad := (lossPlus - lossMinus) / (2 * epsilon)
		anaGrad := float64(analyticalGrad[idx])

		absErr := math.Abs(numGrad - anaGrad)
		relErr := absErr / math.Max(math.Max(math.Abs(numGrad), math.Abs(anaGrad)), 1e-10)

		if absErr > maxAbsError {
			maxAbsError = absErr
		}
		if relErr > maxRelError {
			maxRelError = relErr
		}

		if relErr > tolerance && absErr > 1e-6 {
			t.Errorf("Gradient mismatch at [t=%d, pdf=%d]: numerical=%.6e, analytical=%.6e, relErr=%.6e",
				frame, pdf, numGrad, anaGrad, relErr)
		}

		numChecked++
	}

	t.Logf("Base loss: %.6f", baseLoss)
	t.Logf("Checked %d positions", numChecked)
	t.Logf("Max relative error: %.6e", maxRelError)
	t.Logf("Max absolute error: %.6e", maxAbsError)

	if maxRelError < tolerance {
		t.Logf("✅ PASS: Gradient check passed (max_rel_err=%.2e < %.2e)", maxRelError, tolerance)
	}
}

// computeChainLossCPU computes chain loss on CPU for gradient verification.
// This is a simplified reference implementation.
//
// For the numerator: forward-backward on the linear FST
// For the denominator: simple uniform model (for testing only)
//
// Returns: objf = num_logprob - den_logprob
//
// If numPost/denPost are non-nil, also computes posteriors.
func computeChainLossCPU(
	nnetOutput []float32,
	rowPtr, colIdx []int32,
	weights []float32,
	pdfIds []int32,
	numPost, denPost []float32,
	T, numPdfs, numStates int,
) float64 {

	// --- Numerator: Forward-backward on FST (log-domain) ---
	alpha := make([]float64, (T+1)*numStates) // [T+1][numStates]
	beta := make([]float64, (T+1)*numStates)

	// Initialize
	for i := range alpha {
		alpha[i] = math.Inf(-1)
		beta[i] = math.Inf(-1)
	}
	alpha[0] = 0.0 // start state

	// Forward pass
	for t := 0; t < T; t++ {
		for s := 0; s < numStates; s++ {
			if alpha[t*numStates+s] == math.Inf(-1) {
				continue
			}
			start := int(rowPtr[s])
			end := int(rowPtr[s+1])
			for a := start; a < end; a++ {
				dst := int(colIdx[a])
				pdf := int(pdfIds[a])
				w := float64(weights[a])
				logProb := float64(nnetOutput[t*numPdfs+pdf]) + w
				alpha[(t+1)*numStates+dst] = logAdd(
					alpha[(t+1)*numStates+dst],
					alpha[t*numStates+s]+logProb,
				)
			}
		}
	}

	// Total probability (sum over all final states at time T)
	numTotalLogprob := math.Inf(-1)
	for s := 0; s < numStates; s++ {
		numTotalLogprob = logAdd(numTotalLogprob, alpha[T*numStates+s])
	}

	// Backward pass
	for s := 0; s < numStates; s++ {
		beta[T*numStates+s] = 0.0 // all states are final
	}

	for t := T - 1; t >= 0; t-- {
		for s := 0; s < numStates; s++ {
			start := int(rowPtr[s])
			end := int(rowPtr[s+1])
			for a := start; a < end; a++ {
				dst := int(colIdx[a])
				pdf := int(pdfIds[a])
				w := float64(weights[a])
				logProb := float64(nnetOutput[t*numPdfs+pdf]) + w
				beta[t*numStates+s] = logAdd(
					beta[t*numStates+s],
					beta[(t+1)*numStates+dst]+logProb,
				)
			}
		}
	}

	// Compute numerator posteriors
	if numPost != nil {
		for i := range numPost {
			numPost[i] = 0.0
		}
		for t := 0; t < T; t++ {
			for s := 0; s < numStates; s++ {
				if alpha[t*numStates+s] == math.Inf(-1) {
					continue
				}
				start := int(rowPtr[s])
				end := int(rowPtr[s+1])
				for a := start; a < end; a++ {
					dst := int(colIdx[a])
					pdf := int(pdfIds[a])
					w := float64(weights[a])
					logProb := float64(nnetOutput[t*numPdfs+pdf]) + w
					// posterior = exp(alpha[t][s] + logProb + beta[t+1][dst] - total)
					logPosterior := alpha[t*numStates+s] + logProb +
						beta[(t+1)*numStates+dst] - numTotalLogprob
					numPost[t*numPdfs+pdf] += float32(math.Exp(logPosterior))
				}
			}
		}
	}

	// --- Denominator: Simple uniform model for testing ---
	// den_logprob = T * log(1/numPdfs) = -T * log(numPdfs)
	// In reality this uses the full HMM, but for gradient checking
	// we use a simple uniform model where den_post[t][p] = 1/numPdfs
	denTotalLogprob := -float64(T) * math.Log(float64(numPdfs))

	if denPost != nil {
		uniformPost := float32(1.0 / float64(numPdfs))
		for i := range denPost {
			denPost[i] = uniformPost
		}
	}

	return numTotalLogprob - denTotalLogprob
}

// logAdd computes log(exp(a) + exp(b)) numerically stable
func logAdd(a, b float64) float64 {
	if a == math.Inf(-1) {
		return b
	}
	if b == math.Inf(-1) {
		return a
	}
	if a > b {
		return a + math.Log1p(math.Exp(b-a))
	}
	return b + math.Log1p(math.Exp(a-b))
}

// PrintGradientSummary prints a diagnostic summary of gradient statistics
func PrintGradientSummary(grad []float32, T, numPdfs int) {
	total := T * numPdfs
	if total == 0 {
		fmt.Println("Empty gradient")
		return
	}

	var sumAbs, maxAbs float64
	nonZero := 0

	for _, g := range grad {
		abs := math.Abs(float64(g))
		sumAbs += abs
		if abs > maxAbs {
			maxAbs = abs
		}
		if abs > 1e-10 {
			nonZero++
		}
	}

	fmt.Printf("Gradient stats: total=%d, non_zero=%d (%.1f%%), mean_abs=%.6e, max_abs=%.6e\n",
		total, nonZero, 100*float64(nonZero)/float64(total),
		sumAbs/float64(total), maxAbs)
}
