package parser

import (
	"bufio"
	"encoding/binary"
	"io"
	"math"
)

// uint16ToFloat converts percentile uint16 to float using global header
func uint16ToFloat(globalMin, globalRange float32, value uint16) float32 {
	const inv65535 = 1.52590218966964e-05 // 1/65535
	return globalMin + globalRange*inv65535*float32(value)
}

// charToFloat converts uint8 data byte to float using column percentiles
func charToFloat(p0, p25, p75, p100 float32, value uint8) float32 {
	if value <= 64 {
		return p0 + (p25-p0)*float32(value)*(1.0/64.0)
	} else if value <= 192 {
		return p25 + (p75-p25)*float32(value-64)*(1.0/128.0)
	} else {
		// Branch 3: multiply in float32, divide in double (matches Kaldi)
		return float32(float64(p75) + float64((p100-p75)*float32(value-192))/63.0)
	}
}

// ReadCompressedMatrix reads CM format (kOneByteWithColHeaders)
// Layout: [PerColHeaders × cols][ByteData column-major]
func ReadCompressedMatrix(reader *bufio.Reader, rows, cols int, globalMin, globalRange float32) *MatrixInfo {
	mat := &MatrixInfo{
		Type:  "CM",
		Rows:  rows,
		Cols:  cols,
		Min:   globalMin,
		Range: globalRange,
	}

	// Per-column headers: 8 bytes per column (4 × uint16 percentiles)
	colHeaderBytes := make([]byte, cols*8)
	if _, err := io.ReadFull(reader, colHeaderBytes); err != nil {
		return nil
	}

	// Parse percentiles
	type perColHeader struct {
		p0, p25, p75, p100 float32
	}
	colHeaders := make([]perColHeader, cols)
	for c := 0; c < cols; c++ {
		offset := c * 8
		colHeaders[c].p0 = uint16ToFloat(globalMin, globalRange, binary.LittleEndian.Uint16(colHeaderBytes[offset:]))
		colHeaders[c].p25 = uint16ToFloat(globalMin, globalRange, binary.LittleEndian.Uint16(colHeaderBytes[offset+2:]))
		colHeaders[c].p75 = uint16ToFloat(globalMin, globalRange, binary.LittleEndian.Uint16(colHeaderBytes[offset+4:]))
		colHeaders[c].p100 = uint16ToFloat(globalMin, globalRange, binary.LittleEndian.Uint16(colHeaderBytes[offset+6:]))
	}

	// Byte data (column-major!)
	rawData := make([]byte, rows*cols)
	if _, err := io.ReadFull(reader, rawData); err != nil {
		return nil
	}

	// Full decompression: row-major output
	mat.Data = make([]float32, rows*cols)
	mat.FirstRow = make([]float32, cols)

	for col := 0; col < cols; col++ {
		h := &colHeaders[col]
		for row := 0; row < rows; row++ {
			// Column-major input index
			byteIdx := col*rows + row
			// Row-major output index
			outIdx := row*cols + col
			mat.Data[outIdx] = charToFloat(h.p0, h.p25, h.p75, h.p100, rawData[byteIdx])
		}
		mat.FirstRow[col] = mat.Data[col]
	}

	return mat
}

// ReadCompressedMatrix2 reads CM2 format (kTwoByte)
// Layout: [uint16 data row-major]
func ReadCompressedMatrix2(reader *bufio.Reader, rows, cols int, globalMin, globalRange float32) *MatrixInfo {
	mat := &MatrixInfo{
		Type:  "CM2",
		Rows:  rows,
		Cols:  cols,
		Min:   globalMin,
		Range: globalRange,
	}

	// Read all uint16 data
	rawData := make([]byte, rows*cols*2)
	if _, err := io.ReadFull(reader, rawData); err != nil {
		return nil
	}

	// Full decompression
	mat.Data = make([]float32, rows*cols)
	mat.FirstRow = make([]float32, cols)
	increment := globalRange / 65535.0

	for i := 0; i < rows*cols; i++ {
		val := binary.LittleEndian.Uint16(rawData[i*2 : i*2+2])
		mat.Data[i] = globalMin + float32(val)*increment
	}

	copy(mat.FirstRow, mat.Data[:cols])
	return mat
}

// ReadCompressedMatrix3 reads CM3 format (kOneByte, no per-col headers)
// Layout: [uint8 data row-major]
func ReadCompressedMatrix3(reader *bufio.Reader, rows, cols int, globalMin, globalRange float32) *MatrixInfo {
	mat := &MatrixInfo{
		Type:  "CM3",
		Rows:  rows,
		Cols:  cols,
		Min:   globalMin,
		Range: globalRange,
	}

	// Read all uint8 data
	rawData := make([]byte, rows*cols)
	if _, err := io.ReadFull(reader, rawData); err != nil {
		return nil
	}

	// Full decompression
	mat.Data = make([]float32, rows*cols)
	mat.FirstRow = make([]float32, cols)
	increment := globalRange / 255.0

	for i := 0; i < rows*cols; i++ {
		mat.Data[i] = globalMin + float32(rawData[i])*increment
	}

	copy(mat.FirstRow, mat.Data[:cols])
	return mat
}

// ReadFullMatrix reads FM format (raw float32)
func ReadFullMatrix(reader *bufio.Reader, rows, cols int) *MatrixInfo {
	mat := &MatrixInfo{
		Type: "FM",
		Rows: rows,
		Cols: cols,
	}

	// Read all float32 data
	rawData := make([]byte, rows*cols*4)
	if _, err := io.ReadFull(reader, rawData); err != nil {
		return nil
	}

	// Full extraction
	mat.Data = make([]float32, rows*cols)
	mat.FirstRow = make([]float32, cols)

	for i := 0; i < rows*cols; i++ {
		mat.Data[i] = math.Float32frombits(binary.LittleEndian.Uint32(rawData[i*4 : i*4+4]))
	}

	copy(mat.FirstRow, mat.Data[:cols])
	return mat
}

// ReadSparseMatrix reads SM format
// Format: "SM" + num_rows + [SparseVector × num_rows]
func ReadSparseMatrix(reader *bufio.Reader) *SparseMatrix {
	// "SM" уже прочитано, читаем num_rows
	numRows := readBasicInt32(reader)
	if numRows <= 0 || numRows > 10000000 {
		return nil
	}

	mat := &SparseMatrix{
		Rows: make([]SparseVector, numRows),
	}

	for i := 0; i < int(numRows); i++ {
		sv := readSparseVector(reader)
		if sv == nil {
			return nil
		}
		mat.Rows[i] = *sv
	}

	return mat
}

// readSparseVector reads SV format
// Format: "SV" + dim + num_elems + [pairs]
func readSparseVector(reader *bufio.Reader) *SparseVector {
	// Expect "SV" token
	b1, _ := reader.ReadByte()
	b2, _ := reader.ReadByte()
	if b1 != 'S' || b2 != 'V' {
		return nil
	}

	dim := readBasicInt32(reader)
	numElems := readBasicInt32(reader)

	if dim < 0 || numElems < 0 || numElems > dim {
		return nil
	}

	sv := &SparseVector{
		Dim:   int(dim),
		Pairs: make([]SparseElement, numElems),
	}

	for i := 0; i < int(numElems); i++ {
		index := readBasicInt32(reader)
		value := readBasicFloat32(reader)
		sv.Pairs[i] = SparseElement{
			Index: int(index),
			Value: value,
		}
	}

	return sv
}

// Helper functions for reading WriteBasicType format
func readBasicInt32(reader *bufio.Reader) int32 {
	reader.ReadByte() // space
	size, _ := reader.ReadByte()
	if size != 4 {
		return -1
	}
	var buf [4]byte
	io.ReadFull(reader, buf[:])
	return int32(binary.LittleEndian.Uint32(buf[:]))
}

func readBasicFloat32(reader *bufio.Reader) float32 {
	reader.ReadByte() // space
	size, _ := reader.ReadByte()
	if size != 4 {
		return 0
	}
	var buf [4]byte
	io.ReadFull(reader, buf[:])
	return math.Float32frombits(binary.LittleEndian.Uint32(buf[:]))
}
