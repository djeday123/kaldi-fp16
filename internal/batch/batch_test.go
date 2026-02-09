package batch

import (
	"testing"

	"kaldi-fp16/internal/parser"
)

// helper: create a fake Example with given frames and feat_dim
func fakeExample(key string, frames, featDim, ivectorDim int) *parser.Example {
	// Features data: fill with frame index for easy verification
	featData := make([]float32, frames*featDim)
	for i := range featData {
		featData[i] = float32(i)
	}

	ex := &parser.Example{
		Key:        key,
		NumInputs:  2,
		NumOutputs: 1,
		Inputs: []parser.IoBlock{
			{
				Name: "input",
				Size: frames,
				Matrix: parser.MatrixInfo{
					Type: "FM",
					Rows: frames,
					Cols: featDim,
					Data: featData,
				},
			},
		},
	}

	if ivectorDim > 0 {
		ivData := make([]float32, ivectorDim)
		for i := range ivData {
			ivData[i] = float32(i) * 0.1
		}
		ex.Inputs = append(ex.Inputs, parser.IoBlock{
			Name: "ivector",
			Size: 1,
			Matrix: parser.MatrixInfo{
				Type: "FM",
				Rows: 1,
				Cols: ivectorDim,
				Data: ivData,
			},
		})
	}

	return ex
}

// ============================================================
// Test: NewBatch — basic
// ============================================================
func TestNewBatch_Basic(t *testing.T) {
	examples := []*parser.Example{
		fakeExample("ex1", 10, 40, 100),
		fakeExample("ex2", 15, 40, 100),
		fakeExample("ex3", 20, 40, 100),
	}

	b, err := NewBatch(examples)
	if err != nil {
		t.Fatal(err)
	}

	if b.BatchSize != 3 {
		t.Errorf("BatchSize = %d, expected 3", b.BatchSize)
	}
	if b.TotalFrames() != 45 {
		t.Errorf("TotalFrames = %d, expected 45 (10+15+20)", b.TotalFrames())
	}
	if b.FeatDim() != 40 {
		t.Errorf("FeatDim = %d, expected 40", b.FeatDim())
	}
	if b.IvectorDim() != 100 {
		t.Errorf("IvectorDim = %d, expected 100", b.IvectorDim())
	}
}

// ============================================================
// Test: NewBatch — frame offsets
// ============================================================
func TestNewBatch_FrameOffsets(t *testing.T) {
	examples := []*parser.Example{
		fakeExample("ex1", 10, 40, 100),
		fakeExample("ex2", 15, 40, 100),
		fakeExample("ex3", 20, 40, 100),
	}

	b, err := NewBatch(examples)
	if err != nil {
		t.Fatal(err)
	}

	expectedOffsets := []int{0, 10, 25}
	expectedFrames := []int{10, 15, 20}

	for i := 0; i < 3; i++ {
		if b.FrameOffsets[i] != expectedOffsets[i] {
			t.Errorf("FrameOffsets[%d] = %d, expected %d", i, b.FrameOffsets[i], expectedOffsets[i])
		}
		if b.NumFrames[i] != expectedFrames[i] {
			t.Errorf("NumFrames[%d] = %d, expected %d", i, b.NumFrames[i], expectedFrames[i])
		}
	}
}

// ============================================================
// Test: NewBatch — features merged correctly
// ============================================================
func TestNewBatch_FeaturesMerged(t *testing.T) {
	ex1 := fakeExample("ex1", 2, 3, 0) // 2 frames × 3 dims, no ivector
	ex2 := fakeExample("ex2", 2, 3, 0)

	// Set known values
	ex1.Inputs[0].Matrix.Data = []float32{1, 2, 3, 4, 5, 6}
	ex2.Inputs[0].Matrix.Data = []float32{7, 8, 9, 10, 11, 12}

	b, err := NewBatch([]*parser.Example{ex1, ex2})
	if err != nil {
		t.Fatal(err)
	}

	expected := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	if len(b.Features.Data) != len(expected) {
		t.Fatalf("features len = %d, expected %d", len(b.Features.Data), len(expected))
	}
	for i, v := range expected {
		if b.Features.Data[i] != v {
			t.Errorf("Features.Data[%d] = %f, expected %f", i, b.Features.Data[i], v)
		}
	}

	// Check At access
	if b.Features.At(2, 0) != 7 { // row 2 = first row of ex2
		t.Errorf("Features.At(2,0) = %f, expected 7", b.Features.At(2, 0))
	}
}

// ============================================================
// Test: NewBatch — ivectors merged correctly
// ============================================================
func TestNewBatch_IvectorsMerged(t *testing.T) {
	ex1 := fakeExample("ex1", 5, 40, 3) // ivector dim=3
	ex2 := fakeExample("ex2", 5, 40, 3)

	ex1.Inputs[1].Matrix.Data = []float32{0.1, 0.2, 0.3}
	ex2.Inputs[1].Matrix.Data = []float32{0.4, 0.5, 0.6}

	b, err := NewBatch([]*parser.Example{ex1, ex2})
	if err != nil {
		t.Fatal(err)
	}

	if b.Ivectors.Rows != 2 || b.Ivectors.Cols != 3 {
		t.Fatalf("Ivectors dims = %dx%d, expected 2x3", b.Ivectors.Rows, b.Ivectors.Cols)
	}

	row0 := b.Ivectors.Row(0)
	if row0[0] != 0.1 || row0[1] != 0.2 || row0[2] != 0.3 {
		t.Errorf("Ivectors row 0 = %v, expected [0.1, 0.2, 0.3]", row0)
	}
	row1 := b.Ivectors.Row(1)
	if row1[0] != 0.4 || row1[1] != 0.5 || row1[2] != 0.6 {
		t.Errorf("Ivectors row 1 = %v, expected [0.4, 0.5, 0.6]", row1)
	}
}

// ============================================================
// Test: NewBatch — no ivectors
// ============================================================
func TestNewBatch_NoIvectors(t *testing.T) {
	examples := []*parser.Example{
		fakeExample("ex1", 10, 40, 0),
		fakeExample("ex2", 10, 40, 0),
	}

	b, err := NewBatch(examples)
	if err != nil {
		t.Fatal(err)
	}

	if b.Ivectors != nil {
		t.Error("Ivectors should be nil when no ivectors")
	}
	if b.IvectorDim() != 0 {
		t.Errorf("IvectorDim = %d, expected 0", b.IvectorDim())
	}
}

// ============================================================
// Test: NewBatch — empty list → error
// ============================================================
func TestNewBatch_Empty(t *testing.T) {
	_, err := NewBatch([]*parser.Example{})
	if err == nil {
		t.Error("expected error for empty list")
	}
}

// ============================================================
// Test: NewBatch — no inputs → error
// ============================================================
func TestNewBatch_NoInputs(t *testing.T) {
	ex := &parser.Example{Key: "bad", Inputs: []parser.IoBlock{}}
	_, err := NewBatch([]*parser.Example{ex})
	if err == nil {
		t.Error("expected error for example with no inputs")
	}
}

// ============================================================
// Test: NewBatch — mismatched feat dim → error
// ============================================================
func TestNewBatch_MismatchedFeatDim(t *testing.T) {
	ex1 := fakeExample("ex1", 10, 40, 0)
	ex2 := fakeExample("ex2", 10, 80, 0) // different feat dim

	_, err := NewBatch([]*parser.Example{ex1, ex2})
	if err == nil {
		t.Error("expected error for mismatched feat dimensions")
	}
}

// ============================================================
// Test: NewBatch — wrong first input name → error
// ============================================================
func TestNewBatch_WrongInputName(t *testing.T) {
	ex := fakeExample("ex1", 10, 40, 0)
	ex.Inputs[0].Name = "something_else"

	_, err := NewBatch([]*parser.Example{ex})
	if err == nil {
		t.Error("expected error when first input is not 'input'")
	}
}

// ============================================================
// Test: MergedMatrix helpers
// ============================================================
func TestMergedMatrix_At(t *testing.T) {
	m := &MergedMatrix{Rows: 2, Cols: 3, Data: []float32{1, 2, 3, 4, 5, 6}}
	if m.At(1, 1) != 5 {
		t.Errorf("At(1,1) = %f, expected 5", m.At(1, 1))
	}
}

func TestMergedMatrix_Row(t *testing.T) {
	m := &MergedMatrix{Rows: 2, Cols: 3, Data: []float32{1, 2, 3, 4, 5, 6}}
	row := m.Row(1)
	if len(row) != 3 || row[0] != 4 || row[1] != 5 || row[2] != 6 {
		t.Errorf("Row(1) = %v, expected [4, 5, 6]", row)
	}
}
