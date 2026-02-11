package main

import (
	"fmt"
	"kaldi-fp16/internal/loader"
)

func main() {
	batches, _ := loader.LoadBatches("/opt/kaldi/egs/work_3/s5/exp/chain_cnn_v2/cnn_tdnn1d_v2_sp/egs/cegs.1.ark", 4)
	if len(batches) > 0 {
		b := batches[0]
		fmt.Printf("FramesPerSeq=%d\n", b.FramesPerSeq)
		for i, ex := range b.Examples {
			fmt.Printf("  ex %d: NumSequences=%d FramesPerSeq=%d frames=%d\n",
				i, ex.Supervision.NumSequences, ex.Supervision.FramesPerSeq,
				ex.NumRows)
		}
	}
}
