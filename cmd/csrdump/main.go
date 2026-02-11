package main

import (
	"fmt"
	"kaldi-fp16/internal/parser"
	"log"
)

func main() {
	r, err := parser.NewReader("/opt/kaldi/egs/work_3/s5/exp/chain_cnn_v2/cnn_tdnn1d_v2_sp/egs/cegs.1.ark")
	if err != nil {
		log.Fatal(err)
	}
	defer r.Close()

	for i := 0; i < 8; i++ {
		ex, err := r.ReadExample()
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("ex %d [%s]: NumSeq=%d FramesPerSeq=%d rows=%d fst_states=%d\n",
			i, ex.Key, ex.Supervision.NumSequences, ex.Supervision.FramesPerSeq,
			len(ex.Inputs[0].Indexes), ex.Supervision.Fst.NumStates)
	}
}
