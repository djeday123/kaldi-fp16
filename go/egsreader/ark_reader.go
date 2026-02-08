package egsreader

import (
	"bufio"
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

type ArkReader struct {
	file     *os.File
	reader   *bufio.Reader
	gzReader *gzip.Reader
	binary   *BinaryReader
	atEOF    bool
}

func OpenArk(path string) (*ArkReader, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open ark: %w", err)
	}

	ar := &ArkReader{file: file}

	if strings.HasSuffix(path, ".gz") {
		ar.gzReader, err = gzip.NewReader(file)
		if err != nil {
			file.Close()
			return nil, fmt.Errorf("failed to create gzip reader: %w", err)
		}
		ar.reader = bufio.NewReader(ar.gzReader)
	} else {
		ar.reader = bufio.NewReader(file)
	}

	ar.binary = NewBinaryReader(ar.reader)
	return ar, nil
}

func (ar *ArkReader) Close() error {
	if ar.gzReader != nil {
		ar.gzReader.Close()
	}
	return ar.file.Close()
}

func (ar *ArkReader) Next() (*NNet3ChainExample, error) {
	if ar.atEOF {
		return nil, nil
	}

	var keyBuf []byte
	inKey := false

	for {
		b, err := ar.reader.ReadByte()
		if err == io.EOF {
			ar.atEOF = true
			return nil, nil
		}
		if err != nil {
			return nil, err
		}

		if !inKey {
			if (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') {
				inKey = true
				keyBuf = []byte{b}
			}
			continue
		}

		if (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') ||
			(b >= '0' && b <= '9') || b == '-' || b == '_' {
			keyBuf = append(keyBuf, b)
			continue
		}

		if b == ' ' && len(keyBuf) >= 3 {
			b2, _ := ar.reader.ReadByte()
			if b2 == 0 {
				b3, _ := ar.reader.ReadByte()
				if b3 == 'B' {
					eg, err := ar.readChainExample()
					if err != nil {
						return nil, fmt.Errorf("failed to read '%s': %w", string(keyBuf), err)
					}
					eg.Key = string(keyBuf)
					return eg, nil
				}
			}
		}

		inKey = false
		keyBuf = nil
	}
}

func (ar *ArkReader) readChainExample() (*NNet3ChainExample, error) {
	ar.binary.ClearErr()

	eg := &NNet3ChainExample{}
	eg.Supervision = &ChainSupervision{}

	for {
		b, err := ar.reader.ReadByte()
		if err != nil {
			return eg, nil
		}

		// Matrix markers: CM, CM2, CM3, FM
		if b == 'C' || b == 'F' {
			b2, err := ar.reader.ReadByte()
			if err != nil {
				return eg, nil
			}

			var mat *Matrix
			if b == 'C' && b2 == 'M' {
				b3, err := ar.reader.ReadByte()
				if err != nil {
					return eg, nil
				}
				switch b3 {
				case '2':
					mat = ar.readCompressedMatrix2()
				case '3':
					mat = ar.readCompressedMatrix3()
				case ' ':
					ar.reader.UnreadByte()
					mat = ar.readCompressedMatrix()
				default:
					ar.reader.UnreadByte()
					mat = ar.readCompressedMatrix()
				}
			} else if b == 'F' && b2 == 'M' {
				mat = ar.readFullMatrix()
			} else {
				ar.reader.UnreadByte()
				continue
			}

			if mat != nil {
				switch mat.Cols {
				case 40:
					eg.Input = &NNetIo{Name: "input", Data: mat}
				case 100:
					eg.Ivector = &NNetIo{Name: "ivector", Data: mat}
				}
			}
			continue
		}

		// Tag markers
		if b == '<' {
			tag := ar.readTagContent()
			switch tag {
			case "Weight":
				ar.reader.ReadByte() // skip space
				ar.reader.ReadByte() // skip size byte (0x04)
				eg.Supervision.Weight = ar.binary.ReadFloat32()
			case "NumSequences":
				ar.reader.ReadByte() // skip space
				eg.Supervision.NumSequences = ar.binary.ReadBasicInt()
			case "FramesPerSeq":
				ar.reader.ReadByte() // skip space
				eg.Supervision.FramesPerSeq = ar.binary.ReadBasicInt()
			case "LabelDim":
				ar.reader.ReadByte() // skip space
				eg.Supervision.LabelDim = ar.binary.ReadBasicInt()
			case "/Nnet3ChainEg":
				return eg, nil
			}
		}
	}
}

func (ar *ArkReader) readTagContent() string {
	var content []byte
	for {
		b, err := ar.reader.ReadByte()
		if err != nil || b == '>' || b == ' ' {
			if b == ' ' {
				ar.reader.UnreadByte()
			}
			break
		}
		content = append(content, b)
	}
	return string(content)
}

// CM format: uint8 per value
func (ar *ArkReader) readCompressedMatrix() *Matrix {
	ar.binary.ReadSingleByte() // space

	globalMin := ar.binary.ReadFloat32()
	globalRange := ar.binary.ReadFloat32()
	numRows := ar.binary.ReadInt32()
	numCols := ar.binary.ReadInt32()

	// fmt.Printf("[DEBUG CM] min=%.2f range=%.2f rows=%d cols=%d err=%v\n",
	// globalMin, globalRange, numRows, numCols, ar.binary.Err())

	if numRows <= 0 || numCols <= 0 || numRows > 100000 || numCols > 10000 {
		// fmt.Printf("[DEBUG CM] Invalid dims, returning nil\n")
		return nil
	}

	// Per-column headers (8 bytes per column)
	colHeaderSize := int(numCols) * 8
	io.CopyN(io.Discard, ar.reader, int64(colHeaderSize))

	// Compressed data (uint8)
	dataSize := int(numRows) * int(numCols)
	compressedData := make([]byte, dataSize)
	io.ReadFull(ar.reader, compressedData)
	// if err != nil {
	// 	fmt.Printf("[DEBUG CM - Error] Read %d/%d bytes, err=%v\n", n, dataSize, err)
	// 	return nil
	// }
	// fmt.Printf("[DEBUG CM] Read %d/%d bytes, err=%v\n", n, dataSize, err)

	m := NewMatrix(int(numRows), int(numCols))
	for i, b := range compressedData {
		m.Data[i] = globalMin + (float32(b)/255.0)*globalRange
	}

	return m
}

// CM2 format: uint16 per value, NO per-column headers
func (ar *ArkReader) readCompressedMatrix2() *Matrix {
	ar.binary.ReadSingleByte() // space

	globalMin := ar.binary.ReadFloat32()
	globalRange := ar.binary.ReadFloat32()
	numRows := ar.binary.ReadInt32()
	numCols := ar.binary.ReadInt32()

	if numRows <= 0 || numCols <= 0 || numRows > 100000 || numCols > 10000 {
		return nil
	}

	// CM2: uint16 per value, no column headers
	dataSize := int(numRows) * int(numCols) * 2
	compressedData := make([]byte, dataSize)
	io.ReadFull(ar.reader, compressedData)

	m := NewMatrix(int(numRows), int(numCols))
	for i := 0; i < int(numRows)*int(numCols); i++ {
		val := binary.LittleEndian.Uint16(compressedData[i*2 : i*2+2])
		m.Data[i] = globalMin + (float32(val)/65535.0)*globalRange
	}

	return m
}

// CM3 format: similar to CM2 but different header
func (ar *ArkReader) readCompressedMatrix3() *Matrix {
	// Same as CM2 for now
	return ar.readCompressedMatrix2()
}

func (ar *ArkReader) readFullMatrix() *Matrix {
	ar.binary.ReadSingleByte() // space

	sizeByte := ar.binary.ReadSingleByte()
	if sizeByte != 4 {
		return nil
	}

	numRows := ar.binary.ReadInt32()
	numCols := ar.binary.ReadInt32()

	if numRows <= 0 || numCols <= 0 {
		return nil
	}

	m := NewMatrix(int(numRows), int(numCols))
	for i := range m.Data {
		m.Data[i] = ar.binary.ReadFloat32()
	}

	return m
}

// Peek reads a byte without consuming
// func (r *bufio.Reader) Peek(n int) ([]byte, error) {
// 	return r.Peek(n)
// }

// --------------------------------------------------------------------
// EgsIterator
// --------------------------------------------------------------------

type EgsIterator struct {
	arkPaths []string
	current  int
	reader   *ArkReader
	shuffle  bool
}

func NewEgsIterator(pattern string, shuffle bool) (*EgsIterator, error) {
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return nil, err
	}
	if len(matches) == 0 {
		return nil, fmt.Errorf("no files match pattern: %s", pattern)
	}
	return &EgsIterator{arkPaths: matches, shuffle: shuffle}, nil
}

func (it *EgsIterator) Next() (*NNet3ChainExample, error) {
	for {
		if it.reader == nil {
			if it.current >= len(it.arkPaths) {
				return nil, nil
			}
			var err error
			it.reader, err = OpenArk(it.arkPaths[it.current])
			if err != nil {
				it.current++
				continue
			}
		}

		eg, err := it.reader.Next()
		if err != nil {
			it.reader.Close()
			it.reader = nil
			it.current++
			continue
		}

		if eg == nil {
			it.reader.Close()
			it.reader = nil
			it.current++
			continue
		}

		return eg, nil
	}
}

func (it *EgsIterator) Close() error {
	if it.reader != nil {
		return it.reader.Close()
	}
	return nil
}

func (it *EgsIterator) Reset() {
	if it.reader != nil {
		it.reader.Close()
		it.reader = nil
	}
	it.current = 0
}

// --------------------------------------------------------------------
// Debugging versions of functions with additional logging
// --------------------------------------------------------------------

func OpenArkDebug(path string) (*ArkReader, error) {
	return OpenArk(path)
}

func (ar *ArkReader) NextDebug() (*NNet3ChainExample, error) {
	if ar.atEOF {
		return nil, nil
	}

	var keyBuf []byte
	inKey := false
	scanned := 0

	for {
		b, err := ar.reader.ReadByte()
		scanned++
		if err == io.EOF {
			ar.atEOF = true
			return nil, nil
		}
		if err != nil {
			return nil, err
		}

		if !inKey {
			if (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') {
				inKey = true
				keyBuf = []byte{b}
			}
			continue
		}

		if (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') ||
			(b >= '0' && b <= '9') || b == '-' || b == '_' {
			keyBuf = append(keyBuf, b)
			continue
		}

		if b == ' ' && len(keyBuf) >= 3 {
			b2, _ := ar.reader.ReadByte()
			if b2 == 0 {
				b3, _ := ar.reader.ReadByte()
				if b3 == 'B' {
					fmt.Printf("[DEBUG] Found key '%s' after scanning %d bytes\n", string(keyBuf), scanned)
					eg, err := ar.readChainExampleDebug()
					if err != nil {
						return nil, fmt.Errorf("failed to read '%s': %w", string(keyBuf), err)
					}
					eg.Key = string(keyBuf)
					return eg, nil
				}
			}
		}

		inKey = false
		keyBuf = nil
	}
}

func (ar *ArkReader) readChainExampleDebug() (*NNet3ChainExample, error) {
	ar.binary.ClearErr() // Сбросить ошибку от предыдущего примера

	eg := &NNet3ChainExample{}
	eg.Supervision = &ChainSupervision{}

	bytesRead := 0
	matricesFound := 0

	for {
		b, err := ar.reader.ReadByte()
		bytesRead++
		if err != nil {
			fmt.Printf("[DEBUG] EOF after %d bytes, %d matrices\n", bytesRead, matricesFound)
			return eg, nil
		}

		// Matrix markers: CM, CM2, CM3, FM
		if b == 'C' || b == 'F' {
			b2, err := ar.reader.ReadByte()
			bytesRead++
			if err != nil {
				return eg, nil
			}

			var mat *Matrix
			if b == 'C' && b2 == 'M' {
				b3, err := ar.reader.ReadByte()
				bytesRead++
				if err != nil {
					return eg, nil
				}
				switch b3 {
				case '2':
					fmt.Printf("[DEBUG] Found CM2 at byte %d\n", bytesRead)
					mat = ar.readCompressedMatrix2()
					matricesFound++
				case '3':
					fmt.Printf("[DEBUG] Found CM3 at byte %d\n", bytesRead)
					mat = ar.readCompressedMatrix3()
					matricesFound++
				case ' ':
					ar.reader.UnreadByte()
					fmt.Printf("[DEBUG] Found CM at byte %d\n", bytesRead)
					mat = ar.readCompressedMatrix()
					matricesFound++
				default:
					ar.reader.UnreadByte()
					fmt.Printf("[DEBUG] Found CM (default) at byte %d, b3=0x%02x\n", bytesRead, b3)
					mat = ar.readCompressedMatrix()
					matricesFound++
				}
			} else if b == 'F' && b2 == 'M' {
				fmt.Printf("[DEBUG] Found FM at byte %d\n", bytesRead)
				mat = ar.readFullMatrix()
				matricesFound++
			} else {
				ar.reader.UnreadByte()
				continue
			}

			if mat != nil {
				fmt.Printf("[DEBUG] Matrix: %dx%d\n", mat.Rows, mat.Cols)
				switch mat.Cols {
				case 40:
					eg.Input = &NNetIo{Name: "input", Data: mat}
				case 100:
					eg.Ivector = &NNetIo{Name: "ivector", Data: mat}
				}
			} else {
				fmt.Printf("[DEBUG] Matrix is nil!\n")
			}
			continue
		}

		// Tag markers
		if b == '<' {
			tag := ar.readTagContent()
			switch tag {
			case "Weight":
				eg.Supervision.Weight = ar.binary.ReadFloat32()
			case "NumSequences":
				eg.Supervision.NumSequences = ar.binary.ReadBasicInt()
			case "FramesPerSeq":
				eg.Supervision.FramesPerSeq = ar.binary.ReadBasicInt()
			case "LabelDim":
				eg.Supervision.LabelDim = ar.binary.ReadBasicInt()
			case "/Nnet3ChainEg":
				fmt.Printf("[DEBUG] End tag at byte %d, %d matrices found\n", bytesRead, matricesFound)
				return eg, nil
			}
		}
	}
}
