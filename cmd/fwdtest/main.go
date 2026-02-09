package main

import (
	"fmt"
	"log"
	"os"
	"time"

	"kaldi-fp16/internal/gpu"
	"kaldi-fp16/internal/loader"
	"kaldi-fp16/internal/nnet"
)

func main() {
	xconfigPath := "/opt/kaldi/egs/work_3/s5/exp/chain_cnn_v2/cnn_tdnn1d_v2_sp/configs/network.xconfig"
	arkPattern := "/opt/kaldi/egs/work_3/s5/exp/chain_cnn_v2/cnn_tdnn1d_v2_sp/egs/cegs.*.ark"
	batchSize := 64
	deviceID := 0

	if len(os.Args) > 1 {
		xconfigPath = os.Args[1]
	}

	fmt.Println("=== Forward Pass Test ===")
	fmt.Println()

	// 1. Parse xconfig
	fmt.Printf("Parsing: %s\n", xconfigPath)
	model, err := nnet.BuildModel(xconfigPath)
	if err != nil {
		log.Fatalf("BuildModel: %v", err)
	}
	fmt.Printf("Model: %d layers, ~%.1fM params\n", len(model.Layers), float64(model.NumParams)/1e6)
	fmt.Printf("Chain output: dim=%d\n", model.ChainOutput().OutputDim)
	fmt.Println()

	// 2. Init GPU
	if err := gpu.Init(deviceID); err != nil {
		log.Fatalf("GPU init: %v", err)
	}
	free, total, _ := gpu.MemoryInfo()
	fmt.Printf("GPU %d: %.1f / %.1f GB\n", deviceID, float64(free)/(1<<30), float64(total)/(1<<30))

	// 3. Create network on GPU (random weights for now)
	fmt.Println("Allocating network weights on GPU...")
	t0 := time.Now()
	handle, err := gpu.NewHandle()
	if err != nil {
		log.Fatalf("cuBLAS handle: %v", err)
	}
	defer handle.Destroy()

	network, err := nnet.NewNetwork(model, handle)
	if err != nil {
		log.Fatalf("NewNetwork: %v", err)
	}
	defer network.Free()

	free2, _, _ := gpu.MemoryInfo()
	fmt.Printf("Weights allocated in %v (GPU mem used: %.1f MB)\n",
		time.Since(t0).Round(time.Millisecond),
		float64(free-free2)/(1024*1024))
	fmt.Println()

	// 4. Load one batch
	fmt.Printf("Loading batch (size=%d)...\n", batchSize)
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
	if tb == nil {
		log.Fatal("No data")
	}
	ivecDim := 0
	if tb.Ivectors != nil {
		ivecDim = tb.Ivectors.Cols
	}
	fmt.Printf("Batch: %d examples, %d frames, feat_dim=%d, ivec_dim=%d\n",
		tb.BatchSize, tb.Features.Rows, tb.Features.Cols, ivecDim)
	fmt.Printf("FST: %d states, %d arcs, label_dim=%d\n",
		tb.FstCSR.NumStates, tb.FstCSR.NumArcs, tb.LabelDim)

	// 5. Transfer to GPU
	fmt.Println("\nTransferring to GPU...")
	gpuBatch, err := gpu.TransferBatch(tb)
	if err != nil {
		log.Fatalf("TransferBatch: %v", err)
	}
	defer gpuBatch.Free()

	// Create FP16 tensors from GPU batch for forward pass
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

	fmt.Printf("GPU features: %s\n", features)
	if ivectors != nil {
		fmt.Printf("GPU ivectors: %s\n", ivectors)
	}

	// 6. Forward pass
	fmt.Println("\nRunning forward pass...")
	t1 := time.Now()
	state, err := network.Forward(features, ivectors)
	if err != nil {
		log.Fatalf("Forward: %v", err)
	}
	gpu.Sync()
	forwardTime := time.Since(t1)
	defer state.Free()

	fmt.Printf("Forward pass: %v\n", forwardTime.Round(time.Microsecond))

	// 7. Check output
	fmt.Println("\n=== Output Verification ===")

	if state.Output != nil {
		fmt.Printf("Chain output: [%d × %d]", state.Output.Rows, state.Output.Cols)
		expectedDim := model.ChainOutput().OutputDim
		if state.Output.Cols == expectedDim {
			fmt.Println(" ✓")
		} else {
			fmt.Printf(" ✗ (expected cols=%d)\n", expectedDim)
		}

		// Read back a few values
		data, err := state.Output.ToFP32()
		if err != nil {
			fmt.Printf("  ReadBack error: %v\n", err)
		} else {
			fmt.Printf("  First 5 values: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
				data[0], data[1], data[2], data[3], data[4])
			// Check for NaN/Inf
			nans, infs := 0, 0
			for _, v := range data {
				if v != v {
					nans++
				}
				if v > 1e30 || v < -1e30 {
					infs++
				}
			}
			fmt.Printf("  NaN: %d, Inf: %d / %d total\n", nans, infs, len(data))
		}
	} else {
		fmt.Println("Chain output: nil ✗")
	}

	if state.OutputXent != nil {
		fmt.Printf("Xent output:  [%d × %d]", state.OutputXent.Rows, state.OutputXent.Cols)
		expectedDim := model.XentOutput().OutputDim
		if state.OutputXent.Cols == expectedDim {
			fmt.Println(" ✓")
		} else {
			fmt.Printf(" ✗ (expected cols=%d)\n", expectedDim)
		}
	}

	// 8. Activation dimensions through the network
	fmt.Println("\nActivation shapes:")
	for _, layer := range model.ExecutionOrder() {
		act, ok := state.Activations[layer.Name]
		if !ok {
			fmt.Printf("  %-30s MISSING\n", layer.Name)
			continue
		}
		check := "✓"
		if act.Cols != layer.OutputDim {
			check = fmt.Sprintf("✗ expected %d", layer.OutputDim)
		}
		fmt.Printf("  %-30s [%6d × %4d] %s\n", layer.Name, act.Rows, act.Cols, check)
	}

	// 9. Performance summary
	fmt.Printf("\n=== Performance ===\n")
	fmt.Printf("Frames:       %d\n", features.Rows)
	fmt.Printf("Forward time: %v\n", forwardTime.Round(time.Microsecond))
	fps := float64(features.Rows) / forwardTime.Seconds()
	fmt.Printf("Throughput:   %.0f frames/sec\n", fps)
	rtf := forwardTime.Seconds() / (float64(features.Rows) * 0.01) // 10ms per frame
	fmt.Printf("RTF:          %.4f (%.0fx realtime)\n", rtf, 1.0/rtf)

	fmt.Println("\n✓ Forward pass complete!")
}
