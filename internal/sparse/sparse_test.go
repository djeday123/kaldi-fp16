package sparse

import (
	"math"
	"testing"

	"kaldi-fp16/internal/parser"
)

// makeFst builds a test FST:
//
//	state0 --[label=42, w=0.0]--> state1
//	state1 --[label=116, w=0.0]--> state2
//	state1 --[label=534, w=0.0]--> state3
//	state2 --[label=327, w=0.0]--> state3
//	state3 is final (weight=0.0)
//
// 4 states, 4 arcs, start=0
func makeFst() *parser.Fst {
	inf := float32(math.Inf(1))
	return &parser.Fst{
		Start:     0,
		NumStates: 4,
		NumArcs:   4,
		States: []parser.FstState{
			{Arcs: []parser.FstArc{{Label: 42, Weight: 0.0, NextState: 1}}, Final: inf},
			{Arcs: []parser.FstArc{{Label: 116, Weight: 0.0, NextState: 2}, {Label: 534, Weight: 0.0, NextState: 3}}, Final: inf},
			{Arcs: []parser.FstArc{{Label: 327, Weight: 0.0, NextState: 3}}, Final: inf},
			{Arcs: []parser.FstArc{}, Final: 0.0}, // final state
		},
	}
}

// ============================================================
// FstToCSR
// ============================================================

func TestFstToCSR_Basic(t *testing.T) {
	fst := makeFst()
	csr, err := FstToCSR(fst)
	if err != nil {
		t.Fatal(err)
	}

	if csr.NumStates != 4 {
		t.Errorf("NumStates = %d, expected 4", csr.NumStates)
	}
	if csr.NumArcs != 4 {
		t.Errorf("NumArcs = %d, expected 4", csr.NumArcs)
	}
	if csr.StartState != 0 {
		t.Errorf("StartState = %d, expected 0", csr.StartState)
	}
}

func TestFstToCSR_RowPtr(t *testing.T) {
	fst := makeFst()
	csr, _ := FstToCSR(fst)

	// state0: 1 arc, state1: 2 arcs, state2: 1 arc, state3: 0 arcs
	expectedRowPtr := []int32{0, 1, 3, 4, 4}
	for i, v := range expectedRowPtr {
		if csr.RowPtr[i] != v {
			t.Errorf("RowPtr[%d] = %d, expected %d", i, csr.RowPtr[i], v)
		}
	}
}

func TestFstToCSR_Arcs(t *testing.T) {
	fst := makeFst()
	csr, _ := FstToCSR(fst)

	// Arc 0: state0 → state1, label=42
	if csr.ColIdx[0] != 1 || csr.Labels[0] != 42 {
		t.Errorf("arc 0: col=%d label=%d, expected col=1 label=42", csr.ColIdx[0], csr.Labels[0])
	}
	// Arc 1: state1 → state2, label=116
	if csr.ColIdx[1] != 2 || csr.Labels[1] != 116 {
		t.Errorf("arc 1: col=%d label=%d, expected col=2 label=116", csr.ColIdx[1], csr.Labels[1])
	}
	// Arc 2: state1 → state3, label=534
	if csr.ColIdx[2] != 3 || csr.Labels[2] != 534 {
		t.Errorf("arc 2: col=%d label=%d, expected col=3 label=534", csr.ColIdx[2], csr.Labels[2])
	}
	// Arc 3: state2 → state3, label=327
	if csr.ColIdx[3] != 3 || csr.Labels[3] != 327 {
		t.Errorf("arc 3: col=%d label=%d, expected col=3 label=327", csr.ColIdx[3], csr.Labels[3])
	}
}

func TestFstToCSR_FinalStates(t *testing.T) {
	fst := makeFst()
	csr, _ := FstToCSR(fst)

	if len(csr.FinalStates) != 1 {
		t.Fatalf("FinalStates count = %d, expected 1", len(csr.FinalStates))
	}
	if csr.FinalStates[0] != 3 {
		t.Errorf("FinalStates[0] = %d, expected 3", csr.FinalStates[0])
	}
	if csr.FinalWeights[0] != 0.0 {
		t.Errorf("FinalWeights[0] = %f, expected 0.0", csr.FinalWeights[0])
	}
}

func TestFstToCSR_Validate(t *testing.T) {
	fst := makeFst()
	csr, _ := FstToCSR(fst)

	if err := csr.Validate(); err != nil {
		t.Errorf("Validate failed: %v", err)
	}
}

func TestFstToCSR_LabelDim(t *testing.T) {
	fst := makeFst()
	csr, _ := FstToCSR(fst)

	// max label = 534, so LabelDim = 535
	if csr.LabelDim() != 535 {
		t.Errorf("LabelDim = %d, expected 535", csr.LabelDim())
	}
}

func TestFstToCSR_NilFst(t *testing.T) {
	_, err := FstToCSR(nil)
	if err == nil {
		t.Error("expected error for nil FST")
	}
}

func TestFstToCSR_EmptyFst(t *testing.T) {
	fst := &parser.Fst{NumStates: 0}
	_, err := FstToCSR(fst)
	if err == nil {
		t.Error("expected error for empty FST")
	}
}

// ============================================================
// FstToCOO
// ============================================================

func TestFstToCOO_Basic(t *testing.T) {
	fst := makeFst()
	coo, err := FstToCOO(fst)
	if err != nil {
		t.Fatal(err)
	}

	if coo.NumStates != 4 || coo.NumArcs != 4 {
		t.Errorf("dims = %d states, %d arcs; expected 4, 4", coo.NumStates, coo.NumArcs)
	}

	// Check arc 0: row=0, col=1, label=42
	if coo.Rows[0] != 0 || coo.Cols[0] != 1 || coo.Labels[0] != 42 {
		t.Errorf("arc 0: row=%d col=%d label=%d", coo.Rows[0], coo.Cols[0], coo.Labels[0])
	}

	// Check arc 2: row=1, col=3, label=534
	if coo.Rows[2] != 1 || coo.Cols[2] != 3 || coo.Labels[2] != 534 {
		t.Errorf("arc 2: row=%d col=%d label=%d", coo.Rows[2], coo.Cols[2], coo.Labels[2])
	}
}

func TestFstToCOO_NilFst(t *testing.T) {
	_, err := FstToCOO(nil)
	if err == nil {
		t.Error("expected error for nil FST")
	}
}

// ============================================================
// CSR ↔ COO round-trip
// ============================================================

func TestCSRToCOO_RoundTrip(t *testing.T) {
	fst := makeFst()
	csr, _ := FstToCSR(fst)

	// CSR → COO
	coo := CSRToCOO(csr)

	if coo.NumStates != csr.NumStates || coo.NumArcs != csr.NumArcs {
		t.Fatalf("dims mismatch after CSR→COO")
	}

	// Verify all arcs match
	for j := 0; j < coo.NumArcs; j++ {
		if coo.Cols[j] != csr.ColIdx[j] {
			t.Errorf("arc %d: COO col=%d, CSR col=%d", j, coo.Cols[j], csr.ColIdx[j])
		}
		if coo.Labels[j] != csr.Labels[j] {
			t.Errorf("arc %d: COO label=%d, CSR label=%d", j, coo.Labels[j], csr.Labels[j])
		}
	}

	// Verify row expansion is correct
	// Arc 0 should be row 0 (state0 has 1 arc)
	if coo.Rows[0] != 0 {
		t.Errorf("Rows[0] = %d, expected 0", coo.Rows[0])
	}
	// Arcs 1,2 should be row 1 (state1 has 2 arcs)
	if coo.Rows[1] != 1 || coo.Rows[2] != 1 {
		t.Errorf("Rows[1]=%d, Rows[2]=%d, expected 1,1", coo.Rows[1], coo.Rows[2])
	}
}

func TestCOOToCSR_RoundTrip(t *testing.T) {
	fst := makeFst()
	origCSR, _ := FstToCSR(fst)

	// CSR → COO → CSR
	coo := CSRToCOO(origCSR)
	newCSR := COOToCSR(coo)

	// Verify RowPtr matches
	for i := 0; i <= newCSR.NumStates; i++ {
		if newCSR.RowPtr[i] != origCSR.RowPtr[i] {
			t.Errorf("RowPtr[%d]: new=%d, orig=%d", i, newCSR.RowPtr[i], origCSR.RowPtr[i])
		}
	}

	// Verify arcs match
	for j := 0; j < newCSR.NumArcs; j++ {
		if newCSR.ColIdx[j] != origCSR.ColIdx[j] || newCSR.Labels[j] != origCSR.Labels[j] {
			t.Errorf("arc %d mismatch", j)
		}
	}

	if err := newCSR.Validate(); err != nil {
		t.Errorf("round-trip CSR validation failed: %v", err)
	}
}

// ============================================================
// MergeCOO (batching)
// ============================================================

func TestMergeCOO_Basic(t *testing.T) {
	fst := makeFst()
	coo1, _ := FstToCOO(fst)
	coo2, _ := FstToCOO(fst)

	merged, offsets, err := MergeCOO([]*COO{coo1, coo2})
	if err != nil {
		t.Fatal(err)
	}

	// 2 FSTs × 4 states = 8 states
	if merged.NumStates != 8 {
		t.Errorf("merged NumStates = %d, expected 8", merged.NumStates)
	}
	// 2 FSTs × 4 arcs = 8 arcs
	if merged.NumArcs != 8 {
		t.Errorf("merged NumArcs = %d, expected 8", merged.NumArcs)
	}

	// Offsets
	if offsets[0] != 0 || offsets[1] != 4 {
		t.Errorf("offsets = %v, expected [0, 4]", offsets)
	}

	// First FST arcs: rows 0-3 (unchanged)
	if merged.Rows[0] != 0 || merged.Cols[0] != 1 {
		t.Errorf("fst1 arc 0: row=%d col=%d, expected 0,1", merged.Rows[0], merged.Cols[0])
	}

	// Second FST arcs: rows offset by 4
	// Original arc 0: row=0, col=1 → merged: row=4, col=5
	if merged.Rows[4] != 4 || merged.Cols[4] != 5 {
		t.Errorf("fst2 arc 0: row=%d col=%d, expected 4,5", merged.Rows[4], merged.Cols[4])
	}
}

func TestMergeCOO_FinalStates(t *testing.T) {
	fst := makeFst()
	coo1, _ := FstToCOO(fst)
	coo2, _ := FstToCOO(fst)

	merged, _, _ := MergeCOO([]*COO{coo1, coo2})

	// Each FST has 1 final state (state 3)
	// Merged: state 3 and state 7
	if len(merged.FinalStates) != 2 {
		t.Fatalf("FinalStates count = %d, expected 2", len(merged.FinalStates))
	}
	if merged.FinalStates[0] != 3 || merged.FinalStates[1] != 7 {
		t.Errorf("FinalStates = %v, expected [3, 7]", merged.FinalStates)
	}
}

func TestMergeCOO_Empty(t *testing.T) {
	_, _, err := MergeCOO([]*COO{})
	if err == nil {
		t.Error("expected error for empty list")
	}
}

func TestMergeCOO_ToCSR(t *testing.T) {
	fst := makeFst()
	coo1, _ := FstToCOO(fst)
	coo2, _ := FstToCOO(fst)

	merged, _, _ := MergeCOO([]*COO{coo1, coo2})

	// Convert merged COO to CSR for GPU
	csr := COOToCSR(merged)

	if err := csr.Validate(); err != nil {
		t.Errorf("merged CSR validation failed: %v", err)
	}

	if csr.NumStates != 8 || csr.NumArcs != 8 {
		t.Errorf("merged CSR: %d states, %d arcs; expected 8, 8", csr.NumStates, csr.NumArcs)
	}
}

// ============================================================
// Multiple final states
// ============================================================

func TestFstToCSR_MultipleFinals(t *testing.T) {
	inf := float32(math.Inf(1))
	fst := &parser.Fst{
		Start:     0,
		NumStates: 3,
		NumArcs:   2,
		States: []parser.FstState{
			{Arcs: []parser.FstArc{{Label: 1, Weight: 0, NextState: 1}}, Final: inf},
			{Arcs: []parser.FstArc{{Label: 2, Weight: 0, NextState: 2}}, Final: 1.5},
			{Arcs: []parser.FstArc{}, Final: 0.0},
		},
	}

	csr, err := FstToCSR(fst)
	if err != nil {
		t.Fatal(err)
	}

	if len(csr.FinalStates) != 2 {
		t.Fatalf("FinalStates count = %d, expected 2", len(csr.FinalStates))
	}
	if csr.FinalStates[0] != 1 || csr.FinalWeights[0] != 1.5 {
		t.Errorf("final 0: state=%d weight=%f, expected 1, 1.5", csr.FinalStates[0], csr.FinalWeights[0])
	}
	if csr.FinalStates[1] != 2 || csr.FinalWeights[1] != 0.0 {
		t.Errorf("final 1: state=%d weight=%f, expected 2, 0.0", csr.FinalStates[1], csr.FinalWeights[1])
	}
}

// ============================================================
// Single-state FST
// ============================================================

func TestFstToCSR_SingleState(t *testing.T) {
	fst := &parser.Fst{
		Start:     0,
		NumStates: 1,
		NumArcs:   0,
		States: []parser.FstState{
			{Arcs: []parser.FstArc{}, Final: 0.0},
		},
	}

	csr, err := FstToCSR(fst)
	if err != nil {
		t.Fatal(err)
	}

	if csr.NumStates != 1 || csr.NumArcs != 0 {
		t.Errorf("dims: %d states, %d arcs", csr.NumStates, csr.NumArcs)
	}
	if len(csr.FinalStates) != 1 {
		t.Errorf("FinalStates = %d, expected 1", len(csr.FinalStates))
	}
	if err := csr.Validate(); err != nil {
		t.Errorf("Validate failed: %v", err)
	}
}
