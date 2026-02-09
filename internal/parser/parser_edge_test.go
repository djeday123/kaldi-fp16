package parser

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// === Helpers ===

func readerFromBytes(data []byte) *bufio.Reader {
	return bufio.NewReader(bytes.NewReader(data))
}

func memReader(data []byte) *Reader {
	return &Reader{
		reader: bufio.NewReader(bytes.NewReader(data)),
	}
}

func leInt32(v int32) []byte {
	buf := make([]byte, 4)
	binary.LittleEndian.PutUint32(buf, uint32(v))
	return buf
}

func basicInt32(v int32) []byte {
	out := []byte{' ', 4}
	out = append(out, leInt32(v)...)
	return out
}

func leUint32(v uint32) []byte {
	buf := make([]byte, 4)
	binary.LittleEndian.PutUint32(buf, v)
	return buf
}

func leFloat32(v float32) []byte {
	buf := make([]byte, 4)
	binary.LittleEndian.PutUint32(buf, math.Float32bits(v))
	return buf
}

func writeTempFile(t *testing.T, data []byte) string {
	t.Helper()
	tmp := filepath.Join(t.TempDir(), "test.ark")
	if err := os.WriteFile(tmp, data, 0644); err != nil {
		t.Fatal(err)
	}
	return tmp
}

// ============================================================
// Test 1: readIndexVector count <= 0 returns error
// ============================================================
func TestReadIndexVector_ZeroCount(t *testing.T) {
	r := memReader([]byte{})

	result, err := r.readIndexVector(0)
	if result != nil || err == nil {
		t.Errorf("expected nil+error for count=0, got %v, %v", result, err)
	}

	result, err = r.readIndexVector(-1)
	if result != nil || err == nil {
		t.Errorf("expected nil+error for count=-1, got %v, %v", result, err)
	}
}

// ============================================================
// Test 2: readIndexVector normal delta encoding
// ============================================================
func TestReadIndexVector_NormalDelta(t *testing.T) {
	data := []byte{0, 1, 0}
	r := memReader(data)
	result, err := r.readIndexVector(3)
	if err != nil {
		t.Fatal(err)
	}
	if len(result) != 3 {
		t.Fatalf("expected 3 indexes, got %d", len(result))
	}
	if result[0].N != 0 || result[0].T != 0 || result[0].X != 0 {
		t.Errorf("idx[0] = %+v, expected {0,0,0}", result[0])
	}
	if result[1].T != 1 {
		t.Errorf("idx[1].T = %d, expected 1", result[1].T)
	}
	if result[2].T != 1 {
		t.Errorf("idx[2].T = %d, expected 1", result[2].T)
	}
}

// ============================================================
// Test 3: readIndexVector long format (byte == 127)
// ============================================================
func TestReadIndexVector_LongFormat(t *testing.T) {
	var data []byte
	data = append(data, 127)
	data = append(data, basicInt32(2)...)
	data = append(data, basicInt32(10)...)
	data = append(data, basicInt32(3)...)

	r := memReader(data)
	result, err := r.readIndexVector(1)
	if err != nil {
		t.Fatal(err)
	}
	if len(result) != 1 {
		t.Fatalf("expected 1 index, got %d", len(result))
	}
	if result[0].N != 2 || result[0].T != 10 || result[0].X != 3 {
		t.Errorf("idx[0] = %+v, expected {2,10,3}", result[0])
	}
}

// ============================================================
// Test 4: readIndexVector unexpected byte -128
// ============================================================
func TestReadIndexVector_UnexpectedByte128(t *testing.T) {
	data := []byte{0x80}
	r := memReader(data)
	result, err := r.readIndexVector(1)
	if err != nil {
		t.Fatal(err)
	}
	if len(result) != 1 {
		t.Fatalf("expected 1 index, got %d", len(result))
	}
	if result[0].T != -128 {
		t.Errorf("idx[0].T = %d, expected -128", result[0].T)
	}
}

// ============================================================
// Test 5: readIndexVector byte 125 and 126
// ============================================================
func TestReadIndexVector_UnexpectedByte125_126(t *testing.T) {
	data := []byte{0, 125, 126}
	r := memReader(data)
	result, err := r.readIndexVector(3)
	if err != nil {
		t.Fatal(err)
	}
	if len(result) != 3 {
		t.Fatalf("expected 3 indexes, got %d", len(result))
	}
	if result[1].T != 125 {
		t.Errorf("idx[1].T = %d, expected 125", result[1].T)
	}
	if result[2].T != 125+126 {
		t.Errorf("idx[2].T = %d, expected %d", result[2].T, 125+126)
	}
}

// ============================================================
// Test 6: readIndexVector n != 0 (merged egs warning)
// ============================================================
func TestReadIndexVector_MergedEgs_N_NotZero(t *testing.T) {
	var data []byte
	data = append(data, 127)
	data = append(data, basicInt32(1)...)
	data = append(data, basicInt32(0)...)
	data = append(data, basicInt32(0)...)

	r := memReader(data)
	result, err := r.readIndexVector(1)
	if err != nil {
		t.Fatal(err)
	}
	if len(result) != 1 {
		t.Fatalf("expected 1 index, got %d", len(result))
	}
	if result[0].N != 1 {
		t.Errorf("idx[0].N = %d, expected 1", result[0].N)
	}
	t.Log("Check stdout for [WARN] about merged egs")
}

// ============================================================
// Test 7: readIndexVector x != 0 (extra dim warning)
// ============================================================
func TestReadIndexVector_ExtraDim_X_NotZero(t *testing.T) {
	var data []byte
	data = append(data, 127)
	data = append(data, basicInt32(0)...)
	data = append(data, basicInt32(0)...)
	data = append(data, basicInt32(5)...)

	r := memReader(data)
	result, err := r.readIndexVector(1)
	if err != nil {
		t.Fatal(err)
	}
	if len(result) != 1 {
		t.Fatalf("expected 1 index, got %d", len(result))
	}
	if result[0].X != 5 {
		t.Errorf("idx[0].X = %d, expected 5", result[0].X)
	}
	t.Log("Check stdout for [WARN] about extra dimension")
}

// ============================================================
// Test 8: readIndexVector EOF mid-read returns error
// ============================================================
func TestReadIndexVector_PartialEOF(t *testing.T) {
	data := []byte{0, 1}
	r := memReader(data)
	result, err := r.readIndexVector(5)
	if err == nil {
		t.Error("expected EOF error")
	}
	if len(result) != 2 {
		t.Errorf("expected 2 indexes (partial), got %d", len(result))
	}
}

// ============================================================
// Test 9: DetectFormat — binary ark
// ============================================================
func TestDetectFormat_BinaryArk(t *testing.T) {
	data := []byte("somekey ")
	data = append(data, 0x00, 'B')
	data = append(data, make([]byte, 50)...)

	tmpFile := writeTempFile(t, data)
	err := DetectFormat(tmpFile)
	if err != nil {
		t.Errorf("binary ark should pass: %v", err)
	}
}

// ============================================================
// Test 10: DetectFormat — text ark
// ============================================================
func TestDetectFormat_TextArk(t *testing.T) {
	data := []byte("somekey ark,t:-\n<Nnet3ChainEg>\n")

	tmpFile := writeTempFile(t, data)
	err := DetectFormat(tmpFile)
	if err == nil {
		t.Error("text ark should return error")
	}
	if !strings.Contains(err.Error(), "text ark") {
		t.Errorf("error should mention 'text ark', got: %v", err)
	}
}

// ============================================================
// Test 11: DetectFormat — tiny file
// ============================================================
func TestDetectFormat_TinyFile(t *testing.T) {
	data := []byte("hi")

	tmpFile := writeTempFile(t, data)
	err := DetectFormat(tmpFile)
	if err == nil {
		t.Error("tiny file should return error")
	}
}

// ============================================================
// Test 12: ReadFst bad magic → nil
// ============================================================
func TestReadFst_BadMagic(t *testing.T) {
	data := leInt32(0x12345678)
	reader := readerFromBytes(data)
	fst := ReadFst(reader)
	if fst != nil {
		t.Error("expected nil FST for bad magic number")
	}
}

// ============================================================
// Test 13: ReadFst wrong type → nil
// ============================================================
func TestReadFst_WrongFstType(t *testing.T) {
	var data []byte
	buf := make([]byte, 4)
	binary.LittleEndian.PutUint32(buf, 0x7eb2fdd6)
	data = append(data, buf...)
	data = append(data, leInt32(6)...)
	data = append(data, []byte("vector")...)
	data = append(data, leInt32(8)...)
	data = append(data, []byte("standard")...)

	reader := readerFromBytes(data)
	fst := ReadFst(reader)
	if fst != nil {
		t.Error("expected nil FST for wrong fst type 'vector'")
	}
}

// ============================================================
// Test 14: ReadFst valid minimal
// ============================================================
func TestReadFst_ValidMinimal(t *testing.T) {
	var data []byte

	buf4 := make([]byte, 4)
	binary.LittleEndian.PutUint32(buf4, 0x7eb2fdd6)
	data = append(data, buf4...)
	data = append(data, leInt32(16)...)
	data = append(data, []byte("compact_acceptor")...)
	data = append(data, leInt32(8)...)
	data = append(data, []byte("standard")...)
	data = append(data, leInt32(1)...) // version
	data = append(data, leInt32(0)...) // flags
	props := make([]byte, 8)
	data = append(data, props...) // properties
	start := make([]byte, 8)
	data = append(data, start...) // start
	ns := make([]byte, 8)
	binary.LittleEndian.PutUint64(ns, 2)
	data = append(data, ns...) // numStates=2
	na := make([]byte, 8)
	binary.LittleEndian.PutUint64(na, 1)
	data = append(data, na...) // numArcs=1

	data = append(data, leUint32(0)...) // state 0 start
	data = append(data, leUint32(1)...) // state 1 start
	data = append(data, leUint32(2)...) // ncompacts

	// compact[0]: arc state 0 → state 1, label=42
	data = append(data, leInt32(42)...)
	data = append(data, leFloat32(0.0)...)
	data = append(data, leInt32(1)...)

	// compact[1]: final weight state 1
	data = append(data, leInt32(0)...)
	data = append(data, leFloat32(0.0)...)
	data = append(data, leInt32(-1)...)

	reader := readerFromBytes(data)
	fst := ReadFst(reader)
	if fst == nil {
		t.Fatal("expected valid FST, got nil")
	}
	if fst.NumStates != 2 {
		t.Errorf("numStates = %d, expected 2", fst.NumStates)
	}
	if len(fst.States[0].Arcs) != 1 {
		t.Errorf("state 0 arcs = %d, expected 1", len(fst.States[0].Arcs))
	}
	if fst.States[0].Arcs[0].Label != 42 {
		t.Errorf("arc label = %d, expected 42", fst.States[0].Arcs[0].Label)
	}
	if fst.States[0].Arcs[0].NextState != 1 {
		t.Errorf("arc nextState = %d, expected 1", fst.States[0].Arcs[0].NextState)
	}
	if fst.States[1].Final != 0.0 {
		t.Errorf("state 1 final = %f, expected 0.0", fst.States[1].Final)
	}
	if !math.IsInf(float64(fst.States[0].Final), 1) {
		t.Errorf("state 0 should NOT be final (expected +Inf)")
	}
}
