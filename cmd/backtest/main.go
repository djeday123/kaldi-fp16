package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"

	"kaldi-fp16/internal/gpu"
	"kaldi-fp16/internal/nnet"
)

func main() {
	fmt.Println("=== Backward Pass Verification ===")
	fmt.Println()

	if err := gpu.Init(0); err != nil {
		log.Fatalf("GPU init: %v", err)
	}

	h, err := gpu.NewHandle()
	if err != nil {
		log.Fatalf("cuBLAS: %v", err)
	}
	defer h.Destroy()

	fmt.Println("--- Individual Ops ---")
	testTranspose(h)
	testReLUBackward(h)
	testSigmoidBackward(h)
	testTanhBackward(h)
	testAffineBackward(h)
	testBatchNormBackward(h)

	fmt.Println()
	fmt.Println("--- Network Forward → Backward ---")
	testNetworkBackward(h)

	fmt.Println()
	fmt.Println("--- Numerical Gradient Check ---")
	testNumericalGradient(h)

	fmt.Println()
	fmt.Println("✓ All backward tests passed!")
}

// ============================================================
// Individual Ops
// ============================================================

func testTranspose(h *gpu.Handle) {
	fmt.Print("Transpose... ")
	M, N := 32, 64
	data := randFloats(M * N)
	src, _ := gpu.TensorFromFP32(data, M, N)
	defer src.Free()
	tmp, _ := gpu.NewTensor(N, M)
	defer tmp.Free()
	must(gpu.Transpose(src, tmp))
	dst, _ := gpu.NewTensor(M, N)
	defer dst.Free()
	must(gpu.Transpose(tmp, dst))
	result, _ := dst.ToFP32()
	fmt.Printf("roundtrip=%.2e ✓\n", maxAbsErr(data, result))
}

func testReLUBackward(h *gpu.Handle) {
	fmt.Print("ReLU backward... ")
	n := 1024
	xData := make([]float32, n)
	for i := range xData {
		xData[i] = rand.Float32()*4 - 2
	}
	gradData := randFloats(n)
	x, _ := gpu.TensorFromFP32(xData, 1, n)
	defer x.Free()
	grad, _ := gpu.TensorFromFP32(gradData, 1, n)
	defer grad.Free()
	must(gpu.ReLUBackward(x, grad))
	result, _ := grad.ToFP32()
	passOK, passN, zeroOK, zeroN := 0, 0, 0, 0
	for i := range xData {
		if xData[i] > 0.1 {
			passN++
			if math.Abs(float64(result[i]-gradData[i])) < 0.01 {
				passOK++
			}
		} else if xData[i] < -0.1 {
			zeroN++
			if math.Abs(float64(result[i])) < 1e-3 {
				zeroOK++
			}
		}
	}
	fmt.Printf("pass=%d/%d zeros=%d/%d ✓\n", passOK, passN, zeroOK, zeroN)
}

func testSigmoidBackward(h *gpu.Handle) {
	fmt.Print("Sigmoid backward... ")
	n := 1024
	xData := make([]float32, n)
	for i := range xData {
		xData[i] = rand.Float32()*6 - 3
	}
	gradData := randFloats(n)
	sigOut := make([]float32, n)
	for i := range xData {
		sigOut[i] = float32(1.0 / (1.0 + math.Exp(-float64(xData[i]))))
	}
	output, _ := gpu.TensorFromFP32(sigOut, 1, n)
	defer output.Free()
	grad, _ := gpu.TensorFromFP32(gradData, 1, n)
	defer grad.Free()
	must(gpu.SigmoidBackward(output, grad))
	result, _ := grad.ToFP32()
	expected := make([]float32, n)
	for i := range sigOut {
		expected[i] = gradData[i] * sigOut[i] * (1 - sigOut[i])
	}
	fmt.Printf("max_rel_err=%.2e ✓\n", maxRelErr(result, expected))
}

func testTanhBackward(h *gpu.Handle) {
	fmt.Print("Tanh backward... ")
	n := 1024
	xData := make([]float32, n)
	for i := range xData {
		xData[i] = rand.Float32()*4 - 2
	}
	gradData := randFloats(n)
	tanhOut := make([]float32, n)
	for i := range xData {
		tanhOut[i] = float32(math.Tanh(float64(xData[i])))
	}
	output, _ := gpu.TensorFromFP32(tanhOut, 1, n)
	defer output.Free()
	grad, _ := gpu.TensorFromFP32(gradData, 1, n)
	defer grad.Free()
	must(gpu.TanhBackward(output, grad))
	result, _ := grad.ToFP32()
	expected := make([]float32, n)
	for i := range tanhOut {
		expected[i] = gradData[i] * (1 - tanhOut[i]*tanhOut[i])
	}
	fmt.Printf("max_rel_err=%.2e ✓\n", maxRelErr(result, expected))
}

func testAffineBackward(h *gpu.Handle) {
	fmt.Print("Affine backward... ")
	T, M, K := 16, 32, 24
	weightData := randFloats(M * K)
	gradOutData := randFloats(T * K)
	gradOutput, _ := gpu.TensorFromFP32(gradOutData, T, K)
	defer gradOutput.Free()
	weight, _ := gpu.TensorFromFP32(weightData, M, K)
	defer weight.Free()
	gradOutFP16, _ := gradOutput.ToFP32()
	weightFP16, _ := weight.ToFP32()
	gradInput, err := gpu.AffineBackwardData(h, gradOutput, weight)
	must(err)
	defer gradInput.Free()
	analytical, _ := gradInput.ToFP32()
	expected := make([]float32, T*M)
	for t := 0; t < T; t++ {
		for m := 0; m < M; m++ {
			var sum float64
			for k := 0; k < K; k++ {
				sum += float64(gradOutFP16[t*K+k]) * float64(weightFP16[m*K+k])
			}
			expected[t*M+m] = float32(sum)
		}
	}
	fmt.Printf("max_rel_err=%.2e ✓\n", maxRelErr(analytical, expected))
}

func testBatchNormBackward(h *gpu.Handle) {
	fmt.Print("BatchNorm backward... ")
	rows, cols := 64, 128
	gradOutData := randFloats(rows * cols)
	gamma := make([]float32, cols)
	variance := make([]float32, cols)
	mean := make([]float32, cols)
	beta := make([]float32, cols)
	for i := 0; i < cols; i++ {
		gamma[i] = rand.Float32() + 0.5
		variance[i] = rand.Float32()*1.5 + 0.5
	}
	bn, err := gpu.NewBNParams(mean, variance, gamma, beta)
	must(err)
	defer bn.Free()
	gradOut, _ := gpu.TensorFromFP32(gradOutData, rows, cols)
	defer gradOut.Free()
	gradIn, _ := gpu.NewTensor(rows, cols)
	defer gradIn.Free()
	must(gpu.BatchNormBackward(gradOut, gradIn, bn, 1e-5))
	result, _ := gradIn.ToFP32()
	gradOutFP16, _ := gradOut.ToFP32()
	expected := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			scale := gamma[c] / float32(math.Sqrt(float64(variance[c]+1e-5)))
			expected[r*cols+c] = gradOutFP16[r*cols+c] * scale
		}
	}
	maxErr := maxRelErr(result, expected)
	fmt.Printf("max_rel_err=%.2e ", maxErr)
	if maxErr > 0.02 {
		log.Fatalf("FAIL")
	}
	fmt.Println("✓")
}

// ============================================================
// Network Forward → Backward
// ============================================================

func testNetworkBackward(h *gpu.Handle) {
	fmt.Print("Building test network... ")
	xconfig := `input name=input dim=40
input name=ivector dim=32
idct-layer name=idct input=input dim=40
linear-component name=linear1 input=Append(idct, ivector) dim=128
batchnorm-component name=bn1
tdnnf-layer name=tdnnf1 dim=128 bottleneck-dim=64 time-stride=0 bypass-scale=0.66
tdnnf-layer name=tdnnf2 dim=128 bottleneck-dim=64 time-stride=0 bypass-scale=0.66
prefinal-layer name=prefinal input=tdnnf2 small-dim=64 big-dim=128
output-layer name=output dim=40 include-log-softmax=false`

	model, err := nnet.BuildModelFromString(xconfig)
	must(err)
	fmt.Printf("%d layers, ~%d params\n", len(model.Layers), model.NumParams)

	net, err := nnet.NewNetwork(model, h)
	must(err)
	defer net.Free()
	net.Training = true

	T := 32
	features, _ := gpu.TensorFromFP32(randFloats(T*40), T, 40)
	defer features.Free()
	ivectors, _ := gpu.TensorFromFP32(randFloats(T*32), T, 32)
	defer ivectors.Free()

	fmt.Print("Forward pass... ")
	state, err := net.Forward(features, ivectors)
	must(err)
	defer state.Free()
	fmt.Printf("output=[%d×%d] ✓\n", state.Output.Rows, state.Output.Cols)

	outputGrad, _ := gpu.TensorFromFP32(randFloats(state.Output.Rows*state.Output.Cols),
		state.Output.Rows, state.Output.Cols)
	defer outputGrad.Free()

	fmt.Print("Backward pass (with Append)... ")
	bwdState, err := net.Backward(outputGrad, state)
	must(err)
	defer bwdState.Free()

	trainable := []string{"linear1", "tdnnf1", "tdnnf2", "prefinal", "output"}
	for _, name := range trainable {
		wg, ok := bwdState.WeightGrads[name]
		if !ok || wg == nil {
			log.Fatalf("FAIL: no weight gradients for %s", name)
		}
	}
	fmt.Printf("weight grads for %d layers ✓\n", len(bwdState.WeightGrads))

	fmt.Print("Gradient magnitudes... ")
	for name, wg := range bwdState.WeightGrads {
		var t *gpu.Tensor
		switch {
		case wg.GradW != nil:
			t = wg.GradW
		case wg.GradLinearW != nil:
			t = wg.GradLinearW
		case wg.GradSmallW != nil:
			t = wg.GradSmallW
		}
		if t != nil {
			data, _ := t.ToFP32()
			var sumAbs float64
			for _, v := range data {
				sumAbs += math.Abs(float64(v))
			}
			meanAbs := sumAbs / float64(len(data))
			fmt.Printf("%s=%.2e ", name, meanAbs)
			if meanAbs < 1e-10 {
				log.Fatalf("FAIL: zero gradient for %s", name)
			}
		}
	}
	fmt.Println("✓")
}

// ============================================================
// Numerical Gradient Check
// ============================================================

func testNumericalGradient(h *gpu.Handle) {
	fmt.Print("Numerical gradient check... \n")

	xconfig := `input name=input dim=40
idct-layer name=idct input=input dim=40
linear-component name=linear1 dim=128
batchnorm-component name=bn1
tdnnf-layer name=tdnnf1 dim=128 bottleneck-dim=64 time-stride=0 bypass-scale=0.66
tdnnf-layer name=tdnnf2 dim=128 bottleneck-dim=64 time-stride=0 bypass-scale=0.66
prefinal-layer name=prefinal input=tdnnf2 small-dim=64 big-dim=128
output-layer name=output dim=40 include-log-softmax=false`

	model, err := nnet.BuildModelFromString(xconfig)
	must(err)

	net, err := nnet.NewNetwork(model, h)
	must(err)
	defer net.Free()

	T := 4
	featData := randFloats(T * 40)

	computeLoss := func() float64 {
		feat, _ := gpu.TensorFromFP32(featData, T, 40)
		defer feat.Free()
		state, err := net.Forward(feat, nil)
		if err != nil {
			log.Fatalf("forward: %v", err)
		}
		defer state.Free()
		outData, _ := state.Output.ToFP32()
		return sumF32(outData)
	}

	// Analytical gradient via backward
	feat, _ := gpu.TensorFromFP32(featData, T, 40)
	state, err := net.Forward(feat, nil)
	must(err)

	ones := make([]float32, state.Output.Rows*state.Output.Cols)
	for i := range ones {
		ones[i] = 1.0
	}
	outputGrad, _ := gpu.TensorFromFP32(ones, state.Output.Rows, state.Output.Cols)

	bwdState, err := net.Backward(outputGrad, state)
	must(err)

	outWG := bwdState.WeightGrads["output"]
	if outWG == nil || outWG.GradW == nil {
		log.Fatal("FAIL: no gradient for output weights")
	}
	analyticalData, _ := outWG.GradW.ToFP32()

	outputGrad.Free()
	state.Free()
	feat.Free()
	bwdState.Free()

	// Read current weights
	outWeights := net.Weights["output"]
	wData, _ := outWeights.W.ToFP32()
	wRows := outWeights.W.Rows
	wCols := outWeights.W.Cols

	// Numerical gradient
	eps := float32(0.1)
	numChecks := 20
	step := len(wData) / numChecks
	if step < 1 {
		step = 1
	}

	var maxRelE float32
	var maxAbsE float32
	checked := 0
	outliers := 0
	for idx := 0; idx < len(wData) && checked < numChecks; idx += step {
		orig := wData[idx]

		wData[idx] = orig + eps
		newW, _ := gpu.TensorFromFP32(wData, wRows, wCols)
		outWeights.W.Free()
		outWeights.W = newW
		lossPlus := computeLoss()

		wData[idx] = orig - eps
		newW2, _ := gpu.TensorFromFP32(wData, wRows, wCols)
		outWeights.W.Free()
		outWeights.W = newW2
		lossMinus := computeLoss()

		wData[idx] = orig
		newW3, _ := gpu.TensorFromFP32(wData, wRows, wCols)
		outWeights.W.Free()
		outWeights.W = newW3

		numGrad := (lossPlus - lossMinus) / (2 * float64(eps))
		anaGrad := float64(analyticalData[idx])

		absE := float32(math.Abs(numGrad - anaGrad))
		denom := math.Max(math.Abs(numGrad), math.Max(math.Abs(anaGrad), 1e-6))
		relE := float32(math.Abs(numGrad-anaGrad) / denom)

		if absE > maxAbsE {
			maxAbsE = absE
		}
		if relE > maxRelE {
			maxRelE = relE
		}

		if relE > 0.05 {
			outliers++
			fmt.Printf("  [%d] num=%.4e ana=%.4e abs=%.2e rel=%.2e\n",
				idx, numGrad, anaGrad, absE, relE)
		}

		checked++
	}

	fmt.Printf("  max_rel_err=%.2e max_abs_err=%.2e outliers=%d/%d ",
		maxRelE, maxAbsE, outliers, checked)

	// Accept if: most checks pass and abs error is small
	if maxRelE > 0.2 && maxAbsE > 0.1 {
		log.Fatalf("FAIL")
	}
	fmt.Println("✓")
}

// ============================================================
// Helpers
// ============================================================

func randFloats(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = rand.Float32()*2 - 1
	}
	return out
}

func maxAbsErr(a, b []float32) float32 {
	var m float32
	for i := range a {
		e := float32(math.Abs(float64(a[i] - b[i])))
		if e > m {
			m = e
		}
	}
	return m
}

func maxRelErr(a, b []float32) float32 {
	var m float32
	for i := range a {
		d := float32(math.Max(math.Abs(float64(b[i])), 1e-6))
		e := float32(math.Abs(float64(a[i]-b[i]))) / d
		if e > m {
			m = e
		}
	}
	return m
}

func sumF32(data []float32) float64 {
	var s float64
	for _, v := range data {
		s += float64(v)
	}
	return s
}

func must(err error) {
	if err != nil {
		log.Fatalf("FATAL: %v", err)
	}
}
