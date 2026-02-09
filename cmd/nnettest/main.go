package main

import (
	"fmt"
	"log"
	"os"

	"kaldi-fp16/internal/nnet"
)

func main() {
	xconfigPath := "/opt/kaldi/egs/work_3/s5/exp/chain_cnn_v2/cnn_tdnn1d_v2_sp/configs/network.xconfig"
	if len(os.Args) > 1 {
		xconfigPath = os.Args[1]
	}

	fmt.Printf("Parsing: %s\n\n", xconfigPath)

	// Parse xconfig
	model, err := nnet.BuildModel(xconfigPath)
	if err != nil {
		log.Fatalf("Failed to build model: %v", err)
	}

	// Print summary
	fmt.Print(model.Summary())

	// Print execution order
	fmt.Println("\nExecution order:")
	for i, l := range model.ExecutionOrder() {
		inputs := ""
		if len(l.InputNames) > 0 {
			inputs = " ← " + fmt.Sprintf("%v", l.InputNames)
		}
		fmt.Printf("  %2d. %-30s [%d → %d]%s\n", i+1, l.Name, l.InputDim, l.OutputDim, inputs)
	}

	// Chain output info
	fmt.Println()
	if out := model.ChainOutput(); out != nil {
		fmt.Printf("Chain output: %s (dim=%d)\n", out.Name, out.OutputDim)
	}
	if xent := model.XentOutput(); xent != nil {
		fmt.Printf("Xent output:  %s (dim=%d)\n", xent.Name, xent.OutputDim)
	}
}
