// cmd/sgdtest/main.go — Verify SGD optimizer correctness
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
	fmt.Println("=== SGD Optimizer Test ===")
	fmt.Println()

	if err := gpu.Init(0); err != nil {
		log.Fatalf("GPU init: %v", err)
	}

	h, err := gpu.NewHandle()
	if err != nil {
		log.Fatalf("cuBLAS: %v", err)
	}
	defer h.Destroy()

	testSGDBasic(h)
	testSGDMomentum(h)
	testSGDMasterWeights(h)
	testSGDNetwork(h)

	fmt.Println()
	fmt.Println("✓ All SGD tests passed!")
}

// Test 1: Basic SGD (no momentum) — w -= lr * grad
func testSGDBasic(h *gpu.Handle) {
	fmt.Print("SGD basic (no momentum)... ")

	lr := float32(0.01)
	n := 256

	wData := randFloats(n)
	gradData := randFloats(n)

	w, _ := gpu.TensorFromFP32(wData, 1, n)
	defer w.Free()
	grad, _ := gpu.TensorFromFP32(gradData, 1, n)
	defer grad.Free()

	opt := gpu.NewSGDOptimizer(lr, 0.0) // no momentum
	defer opt.Free()
	must(opt.RegisterParam("test", "w", w))

	must(opt.Update("test", "w", w, grad))

	result, _ := w.ToFP32()

	// Read back FP16-rounded inputs for reference
	wFP16 := fp16Round(wData)
	gFP16 := fp16Round(gradData)

	expected := make([]float32, n)
	for i := range wFP16 {
		expected[i] = wFP16[i] - lr*gFP16[i]
	}

	maxErr := maxRelErr(result, expected)
	fmt.Printf("max_rel_err=%.2e ", maxErr)
	if maxErr > 0.01 {
		log.Fatalf("FAIL")
	}
	fmt.Println("✓")
}

// Test 2: SGD with momentum — velocity accumulates
func testSGDMomentum(h *gpu.Handle) {
	fmt.Print("SGD with momentum... ")

	lr := float32(0.01)
	momentum := float32(0.9)
	n := 256

	wData := randFloats(n)
	w, _ := gpu.TensorFromFP32(wData, 1, n)
	defer w.Free()

	opt := gpu.NewSGDOptimizer(lr, momentum)
	defer opt.Free()
	must(opt.RegisterParam("test", "w", w))

	// Two steps with different gradients
	grad1Data := randFloats(n)
	grad2Data := randFloats(n)

	grad1, _ := gpu.TensorFromFP32(grad1Data, 1, n)
	defer grad1.Free()
	grad2, _ := gpu.TensorFromFP32(grad2Data, 1, n)
	defer grad2.Free()

	// Step 1
	must(opt.Update("test", "w", w, grad1))

	// Step 2
	must(opt.Update("test", "w", w, grad2))

	result, _ := w.ToFP32()

	// CPU reference
	wFP16 := fp16Round(wData)
	g1FP16 := fp16Round(grad1Data)
	g2FP16 := fp16Round(grad2Data)

	vel := make([]float32, n)
	wCPU := make([]float32, n)
	copy(wCPU, wFP16)

	// Step 1: v = 0*momentum + g1; w -= lr*v
	for i := range vel {
		vel[i] = g1FP16[i]
		wCPU[i] -= lr * vel[i]
	}
	// Step 2: v = momentum*v + g2; w -= lr*v
	for i := range vel {
		vel[i] = momentum*vel[i] + g2FP16[i]
		wCPU[i] -= lr * vel[i]
	}

	// FP16 round the final result for comparison
	expected := fp16Round(wCPU)

	maxErr := maxRelErr(result, expected)
	fmt.Printf("max_rel_err=%.2e ", maxErr)
	if maxErr > 0.02 {
		log.Fatalf("FAIL")
	}
	fmt.Println("✓")
}

// Test 3: FP32 master weights preserve precision
func testSGDMasterWeights(h *gpu.Handle) {
	fmt.Print("FP32 master weights precision... ")

	lr := float32(0.0001) // very small lr
	n := 256

	// Start with value 1.0 — small updates should accumulate
	wData := make([]float32, n)
	for i := range wData {
		wData[i] = 1.0
	}

	w, _ := gpu.TensorFromFP32(wData, 1, n)
	defer w.Free()

	opt := gpu.NewSGDOptimizer(lr, 0.0)
	defer opt.Free()
	must(opt.RegisterParam("test", "w", w))

	// Apply 100 steps with grad=1.0
	// Expected: w = 1.0 - 100 * 0.0001 * 1.0 = 0.99
	onesData := make([]float32, n)
	for i := range onesData {
		onesData[i] = 1.0
	}

	for step := 0; step < 100; step++ {
		grad, _ := gpu.TensorFromFP32(onesData, 1, n)
		must(opt.Update("test", "w", w, grad))
		grad.Free()
	}

	result, _ := w.ToFP32()
	expected := float32(0.99) // 1.0 - 100 * 0.0001

	maxErr := float32(0)
	for _, v := range result {
		e := float32(math.Abs(float64(v - expected)))
		if e > maxErr {
			maxErr = e
		}
	}

	fmt.Printf("w[0]=%.6f expected=%.6f err=%.2e ", result[0], expected, maxErr)

	// Without FP32 master: 0.0001 < FP16 epsilon for 1.0, updates would be lost
	// FP16 eps at 1.0 = 2^-10 ≈ 0.001, so 0.0001 update would be rounded to 0
	if maxErr > 0.002 {
		log.Fatalf("FAIL — master weights not working")
	}
	fmt.Println("✓")
}

// Test 4: Full network forward → backward → SGD update → loss decreases
func testSGDNetwork(h *gpu.Handle) {
	fmt.Print("Network training loop (loss should decrease)... \n")

	xconfig := `input name=input dim=40
linear-component name=linear1 dim=128
batchnorm-component name=bn1
prefinal-layer name=prefinal small-dim=64 big-dim=128
output-layer name=output dim=40 include-log-softmax=false`

	model, err := nnet.BuildModelFromString(xconfig)
	must(err)

	net, err := nnet.NewNetwork(model, h)
	must(err)
	defer net.Free()
	net.Training = true

	// Register all weights with optimizer
	opt := gpu.NewSGDOptimizer(0.001, 0.9)
	defer opt.Free()

	for name, lw := range net.Weights {
		if lw.W != nil {
			must(opt.RegisterParam(name, "W", lw.W))
		}
		if lw.Bias != nil {
			must(opt.RegisterParam(name, "Bias", lw.Bias))
		}
		if lw.LinearW != nil {
			must(opt.RegisterParam(name, "LinearW", lw.LinearW))
		}
		if lw.AffineW != nil {
			must(opt.RegisterParam(name, "AffineW", lw.AffineW))
		}
		if lw.AffineBias != nil {
			must(opt.RegisterParam(name, "AffineBias", lw.AffineBias))
		}
		if lw.SmallW != nil {
			must(opt.RegisterParam(name, "SmallW", lw.SmallW))
		}
		if lw.BigW != nil {
			must(opt.RegisterParam(name, "BigW", lw.BigW))
		}
		if lw.BigBias != nil {
			must(opt.RegisterParam(name, "BigBias", lw.BigBias))
		}
	}

	T := 32
	featDim := 40

	// Fixed target: we want output to be zeros
	// Loss = sum(output^2) / 2, grad = output
	featData := randFloats(T * featDim)

	var losses []float64
	for step := 0; step < 20; step++ {
		features, _ := gpu.TensorFromFP32(featData, T, featDim)

		state, err := net.Forward(features, nil)
		must(err)

		// Loss = 0.5 * sum(output^2)
		outData, _ := state.Output.ToFP32()
		var loss float64
		for _, v := range outData {
			loss += float64(v) * float64(v)
		}
		loss *= 0.5

		// Grad = output (dL/dOutput = output for L = 0.5*||output||^2)
		outputGrad, _ := gpu.TensorFromFP32(outData, state.Output.Rows, state.Output.Cols)

		bwdState, err := net.Backward(outputGrad, state)
		must(err)

		// Apply SGD updates
		for name, wg := range bwdState.WeightGrads {
			lw := net.Weights[name]
			if wg.GradW != nil && lw.W != nil {
				opt.Update(name, "W", lw.W, wg.GradW)
			}
			if wg.GradBias != nil && lw.Bias != nil {
				opt.Update(name, "Bias", lw.Bias, wg.GradBias)
			}
			if wg.GradLinearW != nil && lw.LinearW != nil {
				opt.Update(name, "LinearW", lw.LinearW, wg.GradLinearW)
			}
			if wg.GradAffineW != nil && lw.AffineW != nil {
				opt.Update(name, "AffineW", lw.AffineW, wg.GradAffineW)
			}
			if wg.GradAffineBias != nil && lw.AffineBias != nil {
				opt.Update(name, "AffineBias", lw.AffineBias, wg.GradAffineBias)
			}
			if wg.GradSmallW != nil && lw.SmallW != nil {
				opt.Update(name, "SmallW", lw.SmallW, wg.GradSmallW)
			}
			if wg.GradBigW != nil && lw.BigW != nil {
				opt.Update(name, "BigW", lw.BigW, wg.GradBigW)
			}
			if wg.GradBigBias != nil && lw.BigBias != nil {
				opt.Update(name, "BigBias", lw.BigBias, wg.GradBigBias)
			}
		}

		bwdState.Free()
		outputGrad.Free()
		state.Free()
		features.Free()

		losses = append(losses, loss)
		if step < 5 || step == 19 {
			fmt.Printf("  step %2d: loss=%.4f\n", step, loss)
		} else if step == 5 {
			fmt.Println("  ...")
		}
	}

	// Check loss decreased
	if losses[19] < losses[0] {
		ratio := losses[19] / losses[0]
		fmt.Printf("  loss ratio: %.4f (%.1f%% reduction) ✓\n", ratio, (1-ratio)*100)
	} else {
		log.Fatalf("FAIL: loss did not decrease: %.4f → %.4f", losses[0], losses[19])
	}
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

// fp16Round simulates FP16 rounding via GPU roundtrip
func fp16Round(data []float32) []float32 {
	t, _ := gpu.TensorFromFP32(data, 1, len(data))
	defer t.Free()
	result, _ := t.ToFP32()
	return result
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

func must(err error) {
	if err != nil {
		log.Fatalf("FATAL: %v", err)
	}
}
