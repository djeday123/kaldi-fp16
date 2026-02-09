package egsreader

import (
	"fmt"
	"testing"
)

func TestReader(t *testing.T) {
	arkPath := "/data/kaldi-data/exp/chain_cnn_v2/cnn_tdnn_50h_bn192_l2_001/egs/cegs.1.ark"

	reader, err := Open(arkPath)
	if err != nil {
		t.Fatalf("Failed to open: %v", err)
	}
	defer reader.Close()

	count := 0
	valid := 0
	usable := 0
	frameSizes := make(map[int]int)

	for {
		ex, err := reader.Next()
		if err != nil {
			t.Fatalf("Read error at %d: %v", count, err)
		}
		if ex == nil {
			break
		}
		count++

		if err := ex.Validate(); err != nil {
			if count <= 5 {
				t.Logf("Example %d (%s): %v", count, ex.Key, err)
			}
			continue
		}
		valid++

		if ex.IsUsable() {
			usable++
		}

		if ex.Input != nil && ex.Input.Data != nil {
			frameSizes[ex.Input.Data.Rows]++
		}

		if count <= 3 {
			fmt.Printf("Example %d: key=%s, frames=%d, ivector=%dx%d\n",
				count, ex.Key, ex.Input.Data.Rows,
				ex.Ivector.Data.Rows, ex.Ivector.Data.Cols)
		}
	}

	fmt.Printf("\nTotal: %d, Valid: %d (%.1f%%), Usable: %d\n",
		count, valid, float64(valid)/float64(count)*100, usable)
	fmt.Printf("Frame sizes: %v\n", frameSizes)

	if valid != count {
		t.Errorf("Not all examples valid: %d/%d", valid, count)
	}
}
