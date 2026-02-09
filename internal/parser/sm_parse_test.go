package parser

import (
	"encoding/binary"
	"math"
	"testing"
)

// Helper: encode WriteBasicType format for SM (space + size(4) + value)
func smBasicInt32(v int32) []byte {
	out := []byte{' ', 4}
	buf := make([]byte, 4)
	binary.LittleEndian.PutUint32(buf, uint32(v))
	out = append(out, buf...)
	return out
}

func smBasicFloat32(v float32) []byte {
	out := []byte{' ', 4}
	buf := make([]byte, 4)
	binary.LittleEndian.PutUint32(buf, math.Float32bits(v))
	out = append(out, buf...)
	return out
}

// ============================================================
// Test: ReadSparseMatrix — valid 2 rows
// Row 0: dim=5, 2 elements: {idx=1, val=0.8}, {idx=3, val=0.2}
// Row 1: dim=5, 1 element:  {idx=4, val=1.0}
// ============================================================
func TestReadSparseMatrix_Valid(t *testing.T) {
	var data []byte

	// num_rows = 2 (WriteBasicType<int32>)
	data = append(data, smBasicInt32(2)...)

	// Row 0: "SV" + dim=5 + num_elems=2 + pairs
	data = append(data, 'S', 'V')
	data = append(data, smBasicInt32(5)...)     // dim
	data = append(data, smBasicInt32(2)...)     // num_elems
	data = append(data, smBasicInt32(1)...)     // index=1
	data = append(data, smBasicFloat32(0.8)...) // value=0.8
	data = append(data, smBasicInt32(3)...)     // index=3
	data = append(data, smBasicFloat32(0.2)...) // value=0.2

	// Row 1: "SV" + dim=5 + num_elems=1 + pairs
	data = append(data, 'S', 'V')
	data = append(data, smBasicInt32(5)...)     // dim
	data = append(data, smBasicInt32(1)...)     // num_elems
	data = append(data, smBasicInt32(4)...)     // index=4
	data = append(data, smBasicFloat32(1.0)...) // value=1.0

	reader := readerFromBytes(data)
	sm := ReadSparseMatrix(reader)

	if sm == nil {
		t.Fatal("expected valid SparseMatrix, got nil")
	}
	if len(sm.Rows) != 2 {
		t.Fatalf("expected 2 rows, got %d", len(sm.Rows))
	}

	// Row 0
	if sm.Rows[0].Dim != 5 {
		t.Errorf("row 0 dim = %d, expected 5", sm.Rows[0].Dim)
	}
	if len(sm.Rows[0].Pairs) != 2 {
		t.Fatalf("row 0 pairs = %d, expected 2", len(sm.Rows[0].Pairs))
	}
	if sm.Rows[0].Pairs[0].Index != 1 || sm.Rows[0].Pairs[0].Value != 0.8 {
		t.Errorf("row 0 pair 0 = {%d, %f}, expected {1, 0.8}", sm.Rows[0].Pairs[0].Index, sm.Rows[0].Pairs[0].Value)
	}
	if sm.Rows[0].Pairs[1].Index != 3 || sm.Rows[0].Pairs[1].Value != 0.2 {
		t.Errorf("row 0 pair 1 = {%d, %f}, expected {3, 0.2}", sm.Rows[0].Pairs[1].Index, sm.Rows[0].Pairs[1].Value)
	}

	// Row 1
	if sm.Rows[1].Dim != 5 {
		t.Errorf("row 1 dim = %d, expected 5", sm.Rows[1].Dim)
	}
	if len(sm.Rows[1].Pairs) != 1 {
		t.Fatalf("row 1 pairs = %d, expected 1", len(sm.Rows[1].Pairs))
	}
	if sm.Rows[1].Pairs[0].Index != 4 || sm.Rows[1].Pairs[0].Value != 1.0 {
		t.Errorf("row 1 pair 0 = {%d, %f}, expected {4, 1.0}", sm.Rows[1].Pairs[0].Index, sm.Rows[1].Pairs[0].Value)
	}
}

// ============================================================
// Test: ReadSparseMatrix — empty row (0 elements)
// ============================================================
func TestReadSparseMatrix_EmptyRow(t *testing.T) {
	var data []byte

	data = append(data, smBasicInt32(1)...) // 1 row

	data = append(data, 'S', 'V')
	data = append(data, smBasicInt32(10)...) // dim=10
	data = append(data, smBasicInt32(0)...)  // num_elems=0

	reader := readerFromBytes(data)
	sm := ReadSparseMatrix(reader)

	if sm == nil {
		t.Fatal("expected valid SparseMatrix, got nil")
	}
	if len(sm.Rows) != 1 {
		t.Fatalf("expected 1 row, got %d", len(sm.Rows))
	}
	if sm.Rows[0].Dim != 10 {
		t.Errorf("dim = %d, expected 10", sm.Rows[0].Dim)
	}
	if len(sm.Rows[0].Pairs) != 0 {
		t.Errorf("pairs = %d, expected 0", len(sm.Rows[0].Pairs))
	}
}

// ============================================================
// Test: ReadSparseMatrix — invalid num_rows
// ============================================================
func TestReadSparseMatrix_InvalidNumRows(t *testing.T) {
	var data []byte
	data = append(data, smBasicInt32(-1)...) // negative

	reader := readerFromBytes(data)
	sm := ReadSparseMatrix(reader)

	if sm != nil {
		t.Error("expected nil for negative num_rows")
	}
}

// ============================================================
// Test: ReadSparseMatrix — bad SV token
// ============================================================
func TestReadSparseMatrix_BadSVToken(t *testing.T) {
	var data []byte
	data = append(data, smBasicInt32(1)...) // 1 row
	data = append(data, 'X', 'Y')           // not "SV"

	reader := readerFromBytes(data)
	sm := ReadSparseMatrix(reader)

	if sm != nil {
		t.Error("expected nil for bad SV token")
	}
}
