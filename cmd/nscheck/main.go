package main

import (
	"fmt"
	"kaldi-fp16/internal/loader"
	"log"
)

func main() {
	batches, err := loader.LoadBatches("/opt/kaldi/egs/work_3/s5/exp/chain_cnn_v2/cnn_tdnn1d_v2_sp/egs/cegs.1.ark", 4)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Loaded %d batches\n", len(batches))
	if len(batches) > 0 {
		b := batches[0]
		fmt.Printf("BatchSize=%d NumSequences=%d Weight=%.2f LabelDim=%d\n",
			b.BatchSize, b.NumSequences, b.Weight, b.LabelDim)
		for i := 0; i < b.BatchSize; i++ {
			fmt.Printf("  seq %d: key=%s frames=%d framesPerSeq=%d\n",
				i, b.Keys[i], b.NumFrames[i], b.FramesPerSeq[i])
		}
		fmt.Printf("Features: %dx%d\n", b.Features.Rows, b.Features.Cols)
		if b.Ivectors != nil {
			fmt.Printf("Ivectors: %dx%d\n", b.Ivectors.Rows, b.Ivectors.Cols)
		}
		fmt.Printf("FST: %d states, %d arcs\n", b.FstCSR.NumStates, b.FstCSR.NumArcs)
	}
}
