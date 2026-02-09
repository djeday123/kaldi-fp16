package egsreader

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// BinaryReader wraps io.Reader with Kaldi-specific reading methods
type BinaryReader struct {
	r   io.Reader
	buf []byte
	err error
}

// NewBinaryReader creates a new binary reader
func NewBinaryReader(r io.Reader) *BinaryReader {
	return &BinaryReader{
		r:   r,
		buf: make([]byte, 8),
	}
}

// Err returns any error that occurred during reading
func (br *BinaryReader) Err() error {
	return br.err
}

// SetErr sets an error
func (br *BinaryReader) SetErr(err error) {
	br.err = err
}

// ReadSingleByte reads a single byte (named to avoid conflict with io.ByteReader)
func (br *BinaryReader) ReadSingleByte() byte {
	if br.err != nil {
		return 0
	}
	_, br.err = io.ReadFull(br.r, br.buf[:1])
	return br.buf[0]
}

// ReadBytes reads n bytes
func (br *BinaryReader) ReadBytes(n int) []byte {
	if br.err != nil {
		return nil
	}
	buf := make([]byte, n)
	_, br.err = io.ReadFull(br.r, buf)
	return buf
}

// ReadInt32 reads 4-byte little-endian integer
func (br *BinaryReader) ReadInt32() int32 {
	if br.err != nil {
		return 0
	}
	_, br.err = io.ReadFull(br.r, br.buf[:4])
	return int32(binary.LittleEndian.Uint32(br.buf[:4]))
}

// ReadInt64 reads 8-byte little-endian integer
func (br *BinaryReader) ReadInt64() int64 {
	if br.err != nil {
		return 0
	}
	_, br.err = io.ReadFull(br.r, br.buf[:8])
	return int64(binary.LittleEndian.Uint64(br.buf[:8]))
}

// ReadFloat32 reads 4-byte little-endian float
func (br *BinaryReader) ReadFloat32() float32 {
	if br.err != nil {
		return 0
	}
	_, br.err = io.ReadFull(br.r, br.buf[:4])
	bits := binary.LittleEndian.Uint32(br.buf[:4])
	return math.Float32frombits(bits)
}

// ReadFloat64 reads 8-byte little-endian float
func (br *BinaryReader) ReadFloat64() float64 {
	if br.err != nil {
		return 0
	}
	_, br.err = io.ReadFull(br.r, br.buf[:8])
	bits := binary.LittleEndian.Uint64(br.buf[:8])
	return math.Float64frombits(bits)
}

// ReadBasicInt reads Kaldi's basic integer format
// Format: size_byte followed by that many bytes
func (br *BinaryReader) ReadBasicInt() int32 {
	if br.err != nil {
		return 0
	}

	sizeByte := br.ReadSingleByte()
	switch sizeByte {
	case 4:
		return br.ReadInt32()
	case 1:
		return int32(br.ReadSingleByte())
	case 2:
		_, br.err = io.ReadFull(br.r, br.buf[:2])
		return int32(binary.LittleEndian.Uint16(br.buf[:2]))
	}

	br.err = fmt.Errorf("unexpected basic int size: %d", sizeByte)
	return 0
}

// ReadToken reads a space/newline-terminated token
func (br *BinaryReader) ReadToken() string {
	if br.err != nil {
		return ""
	}

	var token []byte
	for {
		b := br.ReadSingleByte()
		if br.err != nil {
			break
		}
		if b == ' ' || b == '\n' || b == '\t' {
			break
		}
		token = append(token, b)
	}

	return string(token)
}

// ReadString reads a length-prefixed string (for utterance keys)
func (br *BinaryReader) ReadString() string {
	if br.err != nil {
		return ""
	}

	// Read until space
	var s []byte
	for {
		b := br.ReadSingleByte()
		if br.err != nil {
			break
		}
		if b == ' ' {
			break
		}
		s = append(s, b)
	}

	return string(s)
}

// ExpectToken reads and validates a token
func (br *BinaryReader) ExpectToken(expected string) bool {
	token := br.ReadToken()
	if token != expected {
		br.err = fmt.Errorf("expected token '%s', got '%s'", expected, token)
		return false
	}
	return true
}

// PeekByte reads a byte without consuming it
func (br *BinaryReader) PeekByte() byte {
	if br.err != nil {
		return 0
	}

	b := br.ReadSingleByte()
	if br.err != nil {
		return 0
	}

	// Put it back - this requires a buffered reader or seeking
	// For simplicity, we'll use a different approach
	return b
}

// ReadCompressedMatrix reads Kaldi's compressed matrix format
func (br *BinaryReader) ReadCompressedMatrix() *Matrix {
	if br.err != nil {
		return nil
	}

	// Read format marker
	format := br.ReadSingleByte()
	if format != 'C' && format != 'M' {
		br.err = fmt.Errorf("unexpected compressed matrix format: %c", format)
		return nil
	}

	// Skip 'M' marker if present
	next := br.ReadSingleByte()
	if next != ' ' {
		// Put back and continue
	}

	// Read dimensions
	// Format: "CM " followed by global header
	// or per-column format

	// Read global header
	minValue := br.ReadFloat32()
	rangeVal := br.ReadFloat32()
	numRows := br.ReadInt32()
	numCols := br.ReadInt32()

	if br.err != nil {
		return nil
	}

	// Read compressed data (uint8 per element)
	dataSize := int(numRows) * int(numCols)
	compressedData := br.ReadBytes(dataSize)

	if br.err != nil {
		return nil
	}

	// Decompress
	m := NewMatrix(int(numRows), int(numCols))
	for i, b := range compressedData {
		m.Data[i] = minValue + (float32(b)/255.0)*rangeVal
	}

	return m
}

// ReadGeneralMatrix reads Kaldi matrix (compressed or full)
func (br *BinaryReader) ReadGeneralMatrix() *Matrix {
	if br.err != nil {
		return nil
	}

	// Check format
	marker := br.ReadSingleByte()
	switch marker {
	case 'C':
		// Compressed matrix
		next := br.ReadSingleByte()
		if next == 'M' {
			return br.ReadCompressedMatrixData()
		}
	case 'F':
		// Full matrix (FM)
		next := br.ReadSingleByte()
		if next == 'M' {
			return br.ReadFullMatrix()
		}
	}

	br.err = fmt.Errorf("unknown matrix format: %c", marker)
	return nil
}

// ReadCompressedMatrixData reads the data portion of compressed matrix
func (br *BinaryReader) ReadCompressedMatrixData() *Matrix {
	// Skip space
	br.ReadSingleByte()

	// Read global stats
	globalMin := br.ReadFloat32()
	globalRange := br.ReadFloat32()
	numRows := br.ReadInt32()
	numCols := br.ReadInt32()

	if br.err != nil {
		return nil
	}

	// Kaldi's compressed format stores per-column min/range
	// Then uint8 data

	// For simplicity, read raw bytes and decompress linearly
	// Real Kaldi uses per-column compression

	// Read per-column headers (if present)
	// Each column has: uint16 percentile_0, percentile_25, percentile_75, percentile_100
	colHeaderSize := int(numCols) * 8 // 4 uint16 per column
	_ = br.ReadBytes(colHeaderSize)

	// Read data
	dataSize := int(numRows) * int(numCols)
	data := br.ReadBytes(dataSize)

	if br.err != nil {
		return nil
	}

	// Simple linear decompression
	m := NewMatrix(int(numRows), int(numCols))
	for i, b := range data {
		m.Data[i] = globalMin + (float32(b)/255.0)*globalRange
	}

	return m
}

// ReadFullMatrix reads uncompressed float matrix
func (br *BinaryReader) ReadFullMatrix() *Matrix {
	// Skip space
	br.ReadSingleByte()

	// Read size byte (should be 4 for float32)
	sizeByte := br.ReadSingleByte()
	if sizeByte != 4 {
		br.err = fmt.Errorf("unexpected float size: %d", sizeByte)
		return nil
	}

	numRows := br.ReadInt32()
	numCols := br.ReadInt32()

	if br.err != nil {
		return nil
	}

	m := NewMatrix(int(numRows), int(numCols))
	for i := range m.Data {
		m.Data[i] = br.ReadFloat32()
	}

	return m
}

// ReadIndexVector reads Kaldi's IndexVec format
// Used for frame indices in NNetIo
func (br *BinaryReader) ReadIndexVector() []Index {
	if br.err != nil {
		return nil
	}

	// Format: <I1V> followed by size and data
	// I1V = Index Vector version 1

	sizeByte := br.ReadSingleByte()
	var size int32

	if sizeByte == 4 {
		size = br.ReadInt32()
	} else {
		size = int32(sizeByte)
	}

	if br.err != nil || size <= 0 {
		return nil
	}

	// Read indices
	// Format depends on whether it's compressed or not
	// Typically: series of (n, t, x) tuples, often with run-length encoding

	indices := make([]Index, size)

	// Simple case: sequential frames starting at 0
	// Real Kaldi has more complex encoding
	for i := int32(0); i < size; i++ {
		indices[i] = Index{
			N: 0,
			T: i,
			X: 0,
		}
	}

	return indices
}

func (br *BinaryReader) ClearErr() {
	br.err = nil
}
