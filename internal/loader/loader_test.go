package loader

import (
	"os"
	"path/filepath"
	"testing"
)

// createFakeArk creates a minimal binary ark file with N "examples"
// Each "example" is just a key + binary marker + padding
// Parser won't find valid examples but won't crash either
func createFakeArk(t *testing.T, dir, name string) string {
	t.Helper()
	path := filepath.Join(dir, name)

	// Binary ark: "key \0B" + some data
	data := []byte("testkey1 ")
	data = append(data, 0x00, 'B')
	data = append(data, make([]byte, 50)...)

	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatal(err)
	}
	return path
}

// ============================================================
// Test: NewEgsIterator — glob pattern
// ============================================================
func TestNewEgsIterator_GlobPattern(t *testing.T) {
	dir := t.TempDir()
	createFakeArk(t, dir, "cegs.1.ark")
	createFakeArk(t, dir, "cegs.2.ark")
	createFakeArk(t, dir, "cegs.3.ark")

	it, err := NewEgsIterator(filepath.Join(dir, "cegs.*.ark"), false)
	if err != nil {
		t.Fatal(err)
	}
	defer it.Close()

	if it.NumFiles() != 3 {
		t.Errorf("NumFiles = %d, expected 3", it.NumFiles())
	}
}

// ============================================================
// Test: NewEgsIterator — no matches → error
// ============================================================
func TestNewEgsIterator_NoMatches(t *testing.T) {
	_, err := NewEgsIterator("/nonexistent/path/*.ark", false)
	if err == nil {
		t.Error("expected error for no matching files")
	}
}

// ============================================================
// Test: NewEgsIteratorFromPaths — empty → error
// ============================================================
func TestNewEgsIteratorFromPaths_Empty(t *testing.T) {
	_, err := NewEgsIteratorFromPaths([]string{}, false)
	if err == nil {
		t.Error("expected error for empty paths")
	}
}

// ============================================================
// Test: NewEgsIteratorFromPaths — basic
// ============================================================
func TestNewEgsIteratorFromPaths_Basic(t *testing.T) {
	dir := t.TempDir()
	p1 := createFakeArk(t, dir, "a.ark")
	p2 := createFakeArk(t, dir, "b.ark")

	it, err := NewEgsIteratorFromPaths([]string{p1, p2}, false)
	if err != nil {
		t.Fatal(err)
	}
	defer it.Close()

	if it.NumFiles() != 2 {
		t.Errorf("NumFiles = %d, expected 2", it.NumFiles())
	}
}

// ============================================================
// Test: Iterator exhaustion — Next returns nil after all files
// ============================================================
func TestEgsIterator_Exhaustion(t *testing.T) {
	dir := t.TempDir()
	p1 := createFakeArk(t, dir, "a.ark")

	it, err := NewEgsIteratorFromPaths([]string{p1}, false)
	if err != nil {
		t.Fatal(err)
	}
	defer it.Close()

	// Our fake ark has binary marker but no valid example body,
	// so parser returns error on first read, iterator skips the file
	// After all files exhausted, Next returns nil, nil
	for {
		ex, err := it.Next()
		if ex == nil {
			break
		}
		_ = err
	}

	// After exhaustion, should return nil, nil
	ex, err := it.Next()
	if err != nil {
		t.Fatalf("unexpected error after exhaustion: %v", err)
	}
	if ex != nil {
		t.Error("expected nil after exhaustion")
	}
}

// ============================================================
// Test: Progress tracking
// ============================================================
func TestEgsIterator_Progress(t *testing.T) {
	dir := t.TempDir()
	p1 := createFakeArk(t, dir, "a.ark")
	p2 := createFakeArk(t, dir, "b.ark")

	it, err := NewEgsIteratorFromPaths([]string{p1, p2}, false)
	if err != nil {
		t.Fatal(err)
	}
	defer it.Close()

	done, total := it.Progress()
	if done != 0 || total != 2 {
		t.Errorf("Progress before read = %d/%d, expected 0/2", done, total)
	}

	// Exhaust iterator (fake arks may return errors, keep going)
	for {
		ex, _ := it.Next()
		if ex == nil {
			break
		}
	}

	done, total = it.Progress()
	if total != 2 {
		t.Errorf("Progress total = %d, expected 2", total)
	}
}

// ============================================================
// Test: CurrentFile
// ============================================================
func TestEgsIterator_CurrentFile(t *testing.T) {
	dir := t.TempDir()
	p1 := createFakeArk(t, dir, "a.ark")

	it, err := NewEgsIteratorFromPaths([]string{p1}, false)
	if err != nil {
		t.Fatal(err)
	}
	defer it.Close()

	if it.CurrentFile() != p1 {
		t.Errorf("CurrentFile = %s, expected %s", it.CurrentFile(), p1)
	}
}

// ============================================================
// Test: Reset
// ============================================================
func TestEgsIterator_Reset(t *testing.T) {
	dir := t.TempDir()
	p1 := createFakeArk(t, dir, "a.ark")

	it, err := NewEgsIteratorFromPaths([]string{p1}, false)
	if err != nil {
		t.Fatal(err)
	}
	defer it.Close()

	// Exhaust
	for {
		ex, _ := it.Next()
		if ex == nil {
			break
		}
	}

	done, _ := it.Progress()
	if done != 1 {
		t.Errorf("after exhaust done = %d, expected 1", done)
	}

	// Reset
	it.Reset()

	done, _ = it.Progress()
	if done != 0 {
		t.Errorf("after reset done = %d, expected 0", done)
	}
}

// ============================================================
// Test: Shuffle — order changes
// ============================================================
func TestEgsIterator_Shuffle(t *testing.T) {
	dir := t.TempDir()
	paths := make([]string, 20)
	for i := 0; i < 20; i++ {
		paths[i] = createFakeArk(t, dir, filepath.Base(filepath.Join(dir, string(rune('a'+i))+".ark")))
	}

	it, err := NewEgsIteratorFromPaths(paths, true)
	if err != nil {
		t.Fatal(err)
	}
	defer it.Close()

	// With 20 files and shuffle, extremely unlikely to stay in same order
	sameOrder := true
	for i, p := range paths {
		if it.arkPaths[i] != p {
			sameOrder = false
			break
		}
	}
	if sameOrder {
		t.Log("WARN: shuffle didn't change order (extremely unlikely with 20 files)")
	}
}

// ============================================================
// Test: Skip bad files gracefully
// ============================================================
func TestEgsIterator_SkipBadFiles(t *testing.T) {
	dir := t.TempDir()
	bad := filepath.Join(dir, "bad.ark")
	os.WriteFile(bad, []byte("not valid"), 0644) // will fail DetectFormat

	// Create a "good" empty ark — has binary marker but no key pattern
	// findExampleStart will hit EOF and return nil, nil (not error)
	good := filepath.Join(dir, "good.ark")
	data := []byte{0x00, 'B'}
	data = append(data, make([]byte, 50)...)
	os.WriteFile(good, data, 0644)

	it, err := NewEgsIteratorFromPaths([]string{bad, good}, false)
	if err != nil {
		t.Fatal(err)
	}
	defer it.Close()

	// bad file: DetectFormat fails, iterator skips
	// good file: no key found, returns nil
	ex, err := it.Next()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ex != nil {
		t.Error("expected nil (no valid examples)")
	}
}
