package loader

import (
	"math"
	"testing"

	"kaldi-fp16/internal/parser"
	"kaldi-fp16/internal/sparse"
)

// helper: create a realistic training example
func makeTrainingExample(key string, frames int) *parser.Example {
	inf := float32(math.Inf(1))

	// Features: frames × 40
	featData := make([]float32, frames*40)
	for i := range featData {
		featData[i] = float32(i) * 0.01
	}

	// Ivector: 1 × 100
	ivData := make([]float32, 100)
	for i := range ivData {
		ivData[i] = float32(i) * 0.1
	}

	// FST: simple 4-state graph
	fst := &parser.Fst{
		Start:     0,
		NumStates: 4,
		NumArcs:   4,
		States: []parser.FstState{
			{Arcs: []parser.FstArc{{Label: 42, Weight: 0, NextState: 1}}, Final: inf},
			{Arcs: []parser.FstArc{{Label: 116, Weight: 0, NextState: 2}, {Label: 534, Weight: 0, NextState: 3}}, Final: inf},
			{Arcs: []parser.FstArc{{Label: 327, Weight: 0, NextState: 3}}, Final: inf},
			{Arcs: []parser.FstArc{}, Final: 0.0},
		},
	}

	return &parser.Example{
		Key:       key,
		NumInputs: 2,
		Inputs: []parser.IoBlock{
			{
				Name: "input",
				Size: frames,
				Matrix: parser.MatrixInfo{
					Type: "CM",
					Rows: frames,
					Cols: 40,
					Data: featData,
				},
			},
			{
				Name: "ivector",
				Size: 1,
				Matrix: parser.MatrixInfo{
					Type: "CM2",
					Rows: 1,
					Cols: 100,
					Data: ivData,
				},
			},
		},
		Supervision: parser.SupervisionBlock{
			Name:         "output",
			Weight:       1.0,
			NumSequences: 1,
			FramesPerSeq: frames,
			LabelDim:     3080,
			Fst:          fst,
		},
	}
}

// ============================================================
// validateExample
// ============================================================

func TestValidateExample_Valid(t *testing.T) {
	ex := makeTrainingExample("test1", 150)
	if err := validateExample(ex); err != nil {
		t.Errorf("valid example rejected: %v", err)
	}
}

func TestValidateExample_NoInputs(t *testing.T) {
	ex := &parser.Example{Key: "bad", Inputs: []parser.IoBlock{}}
	if err := validateExample(ex); err == nil {
		t.Error("expected error for no inputs")
	}
}

func TestValidateExample_WrongInputName(t *testing.T) {
	ex := makeTrainingExample("test", 10)
	ex.Inputs[0].Name = "wrong"
	if err := validateExample(ex); err == nil {
		t.Error("expected error for wrong input name")
	}
}

func TestValidateExample_NilFst(t *testing.T) {
	ex := makeTrainingExample("test", 10)
	ex.Supervision.Fst = nil
	if err := validateExample(ex); err == nil {
		t.Error("expected error for nil FST")
	}
}

func TestValidateExample_ZeroWeight(t *testing.T) {
	ex := makeTrainingExample("test", 10)
	ex.Supervision.Weight = 0
	if err := validateExample(ex); err == nil {
		t.Error("expected error for zero weight")
	}
}

func TestValidateExample_NilData(t *testing.T) {
	ex := makeTrainingExample("test", 10)
	ex.Inputs[0].Matrix.Data = nil
	if err := validateExample(ex); err == nil {
		t.Error("expected error for nil data")
	}
}

// ============================================================
// mergeFSTs
// ============================================================

func TestMergeFSTs_Basic(t *testing.T) {
	examples := []*parser.Example{
		makeTrainingExample("ex1", 150),
		makeTrainingExample("ex2", 150),
		makeTrainingExample("ex3", 150),
	}

	csr, _, offsets, err := mergeFSTs(examples)
	if err != nil {
		t.Fatal(err)
	}

	// 3 FSTs × 4 states = 12
	if csr.NumStates != 12 {
		t.Errorf("NumStates = %d, expected 12", csr.NumStates)
	}
	// 3 FSTs × 4 arcs = 12
	if csr.NumArcs != 12 {
		t.Errorf("NumArcs = %d, expected 12", csr.NumArcs)
	}
	// Offsets: 0, 4, 8
	if offsets[0] != 0 || offsets[1] != 4 || offsets[2] != 8 {
		t.Errorf("offsets = %v, expected [0, 4, 8]", offsets)
	}
	// Validate
	if err := csr.Validate(); err != nil {
		t.Errorf("validation failed: %v", err)
	}
}

func TestMergeFSTs_LabelDim(t *testing.T) {
	examples := []*parser.Example{
		makeTrainingExample("ex1", 150),
	}

	csr, _, _, err := mergeFSTs(examples)
	if err != nil {
		t.Fatal(err)
	}

	// Max label in our test FST = 534, so LabelDim = 535
	if csr.LabelDim() != 535 {
		t.Errorf("LabelDim = %d, expected 535", csr.LabelDim())
	}
}

// ============================================================
// TrainingBatch assembly (full pipeline)
// ============================================================

func TestTrainingBatch_FullPipeline(t *testing.T) {
	examples := []*parser.Example{
		makeTrainingExample("utt1", 150),
		makeTrainingExample("utt2", 120),
		makeTrainingExample("utt3", 180),
	}

	// Manually assemble what DataLoader.NextBatch does
	// (can't use real ark files in unit test)

	// Validate
	for _, ex := range examples {
		if err := validateExample(ex); err != nil {
			t.Fatalf("validate failed: %v", err)
		}
	}

	// Merge FSTs
	csr, _, offsets, err := mergeFSTs(examples)
	if err != nil {
		t.Fatal(err)
	}

	// Check dimensions
	if csr.NumStates != 12 { // 3 × 4
		t.Errorf("NumStates = %d", csr.NumStates)
	}

	// Check offsets allow indexing into merged FST
	for i, off := range offsets {
		if off < 0 || int(off) >= csr.NumStates {
			t.Errorf("offset[%d] = %d out of range", i, off)
		}
	}

	// Verify CSR is valid
	if err := csr.Validate(); err != nil {
		t.Fatal(err)
	}

	// Convert back to COO to verify arcs
	coo := sparse.CSRToCOO(csr)

	// First FST: arcs at rows 0-3
	// Second FST: arcs at rows 4-7
	// Third FST: arcs at rows 8-11
	foundArc := false
	for j := 0; j < coo.NumArcs; j++ {
		// Second FST, first arc: row=4, col=5, label=42
		if coo.Rows[j] == 4 && coo.Cols[j] == 5 && coo.Labels[j] == 42 {
			foundArc = true
		}
	}
	if !foundArc {
		t.Error("expected to find arc (row=4, col=5, label=42) from second FST")
	}
}

// ============================================================
// DataLoader config validation
// ============================================================

func TestNewDataLoader_BadBatchSize(t *testing.T) {
	_, err := NewDataLoader(DataLoaderConfig{
		Pattern:   "/nonexistent/*.ark",
		BatchSize: 0,
	})
	if err == nil {
		t.Error("expected error for batch size 0")
	}
}

func TestNewDataLoader_NoFiles(t *testing.T) {
	_, err := NewDataLoader(DataLoaderConfig{
		Pattern:   "/nonexistent/path/*.ark",
		BatchSize: 64,
	})
	if err == nil {
		t.Error("expected error for no matching files")
	}
}

func TestNewDataLoaderFromPaths_BadBatchSize(t *testing.T) {
	_, err := NewDataLoaderFromPaths([]string{"a.ark"}, -1, false)
	if err == nil {
		t.Error("expected error for negative batch size")
	}
}

// ============================================================
// DataLoaderStats
// ============================================================

func TestDataLoaderStats_String(t *testing.T) {
	s := DataLoaderStats{
		BatchesServed: 100,
		ExamplesRead:  6400,
	}
	str := s.String()
	if len(str) == 0 {
		t.Error("Stats.String() returned empty")
	}
}
