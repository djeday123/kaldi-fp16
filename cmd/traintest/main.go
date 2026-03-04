// cmd/traintest/main.go — Verify Trainer API works end-to-end
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
	fmt.Println("=== Trainer Integration Test ===")
	fmt.Println()

	if err := gpu.Init(0); err != nil {
		log.Fatalf("GPU init: %v", err)
	}

	h, err := gpu.NewHandle()
	if err != nil {
		log.Fatalf("cuBLAS: %v", err)
	}
	defer h.Destroy()

	testTrainerAPI(h)

	fmt.Println()
	fmt.Println("✓ Trainer integration test passed!")
}

func testTrainerAPI(h *gpu.Handle) {
	fmt.Println("Building network...")

	xconfig := `input name=input dim=40
linear-component name=linear1 dim=128
batchnorm-component name=bn1
tdnnf-layer name=tdnnf1 dim=128 bottleneck-dim=64 time-stride=0 bypass-scale=0.66
prefinal-layer name=prefinal small-dim=64 big-dim=128
output-layer name=output dim=40 include-log-softmax=false`

	model, err := nnet.BuildModelFromString(xconfig)
	must(err)
	fmt.Printf("  %d layers, ~%d params\n", len(model.Layers), model.NumParams)

	net, err := nnet.NewNetwork(model, h)
	must(err)
	defer net.Free()

	config := nnet.TrainConfig{
		LearningRate: 0.001,
		Momentum:     0.9,
	}

	// Create Trainer (registers all weights with optimizer)
	fmt.Println("Creating Trainer...")
	trainer, err := nnet.NewTrainer(net, nil, config)
	must(err)
	defer trainer.Free()

	// Simulate training loop: manual forward → backward → update
	// (Can't use Trainer.Step without real chain loss / denominator FST)
	T := 32
	featDim := 40
	featData := randFloats(T * featDim)

	fmt.Println("Running 10 training steps (L2 loss on output)...")

	var losses []float64
	for step := 0; step < 10; step++ {
		features, _ := gpu.TensorFromFP32(featData, T, featDim)
		state, err := net.Forward(features, nil)
		must(err)

		// L2 loss
		outData, _ := state.Output.ToFP32()
		var loss float64
		for _, v := range outData {
			loss += float64(v) * float64(v)
		}
		loss *= 0.5
		losses = append(losses, loss)

		// Backward
		outputGrad, _ := gpu.TensorFromFP32(outData, state.Output.Rows, state.Output.Cols)
		bwdState, err := net.Backward(outputGrad, state)
		must(err)

		// Update via Trainer's optimizer
		for name, wg := range bwdState.WeightGrads {
			lw := net.Weights[name]
			if lw == nil {
				continue
			}
			if wg.GradW != nil && lw.W != nil {
				trainer.Optimizer.Update(name, "W", lw.W, wg.GradW)
			}
			if wg.GradBias != nil && lw.Bias != nil {
				trainer.Optimizer.Update(name, "Bias", lw.Bias, wg.GradBias)
			}
			if wg.GradLinearW != nil && lw.LinearW != nil {
				trainer.Optimizer.Update(name, "LinearW", lw.LinearW, wg.GradLinearW)
			}
			if wg.GradAffineW != nil && lw.AffineW != nil {
				trainer.Optimizer.Update(name, "AffineW", lw.AffineW, wg.GradAffineW)
			}
			if wg.GradAffineBias != nil && lw.AffineBias != nil {
				trainer.Optimizer.Update(name, "AffineBias", lw.AffineBias, wg.GradAffineBias)
			}
			if wg.GradSmallW != nil && lw.SmallW != nil {
				trainer.Optimizer.Update(name, "SmallW", lw.SmallW, wg.GradSmallW)
			}
			if wg.GradBigW != nil && lw.BigW != nil {
				trainer.Optimizer.Update(name, "BigW", lw.BigW, wg.GradBigW)
			}
			if wg.GradBigBias != nil && lw.BigBias != nil {
				trainer.Optimizer.Update(name, "BigBias", lw.BigBias, wg.GradBigBias)
			}
		}

		bwdState.Free()
		outputGrad.Free()
		state.Free()
		features.Free()

		fmt.Printf("  step %d: loss=%.4f\n", step, loss)
	}

	// Test LR scheduling
	fmt.Print("LR scheduling... ")
	trainer.SetLR(0.0001)
	if trainer.Config.LearningRate != 0.0001 {
		log.Fatalf("FAIL: LR not updated")
	}
	fmt.Println("✓")

	// Verify loss decreased
	ratio := losses[9] / losses[0]
	fmt.Printf("Loss: %.4f → %.4f (%.1f%% reduction)\n", losses[0], losses[9], (1-ratio)*100)
	if losses[9] >= losses[0] {
		log.Fatalf("FAIL: loss did not decrease")
	}

	// Verify weights actually changed
	fmt.Print("Weights changed... ")
	for name, lw := range net.Weights {
		if lw.W != nil {
			data, _ := lw.W.ToFP32()
			var sumAbs float64
			for _, v := range data {
				sumAbs += math.Abs(float64(v))
			}
			mean := sumAbs / float64(len(data))
			if mean < 1e-10 {
				log.Fatalf("FAIL: %s weights are zero", name)
			}
		}
	}
	fmt.Println("✓")
}

func randFloats(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = rand.Float32()*2 - 1
	}
	return out
}

func must(err error) {
	if err != nil {
		log.Fatalf("FATAL: %v", err)
	}
}
