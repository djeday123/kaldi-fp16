package parser

import (
	"compress/gzip"
	"os"
	"path/filepath"
	"testing"
)

// ============================================================
// Matrix helpers: At, Set, Row
// ============================================================

func TestMatrixInfo_At(t *testing.T) {
	m := &MatrixInfo{
		Rows: 2,
		Cols: 3,
		Data: []float32{1, 2, 3, 4, 5, 6},
	}

	if m.At(0, 0) != 1 {
		t.Errorf("At(0,0) = %f, expected 1", m.At(0, 0))
	}
	if m.At(0, 2) != 3 {
		t.Errorf("At(0,2) = %f, expected 3", m.At(0, 2))
	}
	if m.At(1, 0) != 4 {
		t.Errorf("At(1,0) = %f, expected 4", m.At(1, 0))
	}
	if m.At(1, 2) != 6 {
		t.Errorf("At(1,2) = %f, expected 6", m.At(1, 2))
	}
}

func TestMatrixInfo_Set(t *testing.T) {
	m := &MatrixInfo{
		Rows: 2,
		Cols: 2,
		Data: []float32{0, 0, 0, 0},
	}

	m.Set(0, 1, 3.14)
	m.Set(1, 0, 2.71)

	if m.At(0, 1) != 3.14 {
		t.Errorf("At(0,1) = %f, expected 3.14", m.At(0, 1))
	}
	if m.At(1, 0) != 2.71 {
		t.Errorf("At(1,0) = %f, expected 2.71", m.At(1, 0))
	}
}

func TestMatrixInfo_Row(t *testing.T) {
	m := &MatrixInfo{
		Rows: 3,
		Cols: 2,
		Data: []float32{1, 2, 3, 4, 5, 6},
	}

	row0 := m.Row(0)
	if len(row0) != 2 || row0[0] != 1 || row0[1] != 2 {
		t.Errorf("Row(0) = %v, expected [1, 2]", row0)
	}

	row2 := m.Row(2)
	if len(row2) != 2 || row2[0] != 5 || row2[1] != 6 {
		t.Errorf("Row(2) = %v, expected [5, 6]", row2)
	}

	// Zero-copy check: modify through Row slice
	row0[0] = 99
	if m.Data[0] != 99 {
		t.Error("Row() should return zero-copy slice")
	}
}

// ============================================================
// DetectFormat + NewReader integration
// ============================================================

func TestNewReader_BinaryArk(t *testing.T) {
	// Create a minimal binary ark file
	data := []byte("somekey ")
	data = append(data, 0x00, 'B')
	data = append(data, make([]byte, 50)...)

	tmp := filepath.Join(t.TempDir(), "test.ark")
	os.WriteFile(tmp, data, 0644)

	r, err := NewReader(tmp)
	if err != nil {
		t.Fatalf("NewReader failed: %v", err)
	}
	defer r.Close()
}

func TestNewReader_TextArkRejected(t *testing.T) {
	data := []byte("somekey ark,t:-\n<Nnet3ChainEg>\n")

	tmp := filepath.Join(t.TempDir(), "test.ark")
	os.WriteFile(tmp, data, 0644)

	_, err := NewReader(tmp)
	if err == nil {
		t.Error("NewReader should reject text ark")
	}
}

func TestNewReader_NonexistentFile(t *testing.T) {
	_, err := NewReader("/nonexistent/path.ark")
	if err == nil {
		t.Error("NewReader should fail for nonexistent file")
	}
}

// ============================================================
// Gzip support
// ============================================================

func TestNewReader_GzipArk(t *testing.T) {
	// Create a gzipped binary ark
	data := []byte("somekey ")
	data = append(data, 0x00, 'B')
	data = append(data, make([]byte, 50)...)

	tmp := filepath.Join(t.TempDir(), "test.ark.gz")
	f, err := os.Create(tmp)
	if err != nil {
		t.Fatal(err)
	}
	gz := gzip.NewWriter(f)
	gz.Write(data)
	gz.Close()
	f.Close()

	r, err := NewReader(tmp)
	if err != nil {
		t.Fatalf("NewReader gzip failed: %v", err)
	}
	defer r.Close()

	// Should be able to read (will get nil since our mini ark isn't a full example)
	ex, _ := r.ReadExample()
	// Key should be found even in gzip
	_ = ex
}

func TestNewReader_BadGzip(t *testing.T) {
	// Not actually gzipped
	data := []byte("this is not gzip data")
	tmp := filepath.Join(t.TempDir(), "bad.ark.gz")
	os.WriteFile(tmp, data, 0644)

	_, err := NewReader(tmp)
	if err == nil {
		t.Error("NewReader should fail for invalid gzip")
	}
}

func TestNewReader_GzipSkipsDetectFormat(t *testing.T) {
	// .gz files can't be checked by DetectFormat (compressed)
	// So even a text ark inside gzip should open (DetectFormat is skipped)
	data := []byte("somekey ark,t:-\n")

	tmp := filepath.Join(t.TempDir(), "text.ark.gz")
	f, _ := os.Create(tmp)
	gz := gzip.NewWriter(f)
	gz.Write(data)
	gz.Close()
	f.Close()

	// Should open successfully (DetectFormat skipped for .gz)
	r, err := NewReader(tmp)
	if err != nil {
		t.Fatalf("gzip NewReader should not run DetectFormat: %v", err)
	}
	r.Close()
}
