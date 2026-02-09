package parser

import (
	"bufio"
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
)

// DebugLevel controls verbosity:
//
//	0 = off
//	1 = summary (first row values for first example only)
//	2 = all matrices (header info for every matrix)
//	3 = full (all tags and parsing details)
var DebugLevel = 0
var debugExampleCount = 0

// Debug helpers
func debugSummary() bool { return DebugLevel >= 1 }
func debugMatrix() bool  { return DebugLevel >= 2 }
func debugFull() bool    { return DebugLevel >= 3 }

// Reader читает примеры из ark файла
type Reader struct {
	file     *os.File
	gzReader *gzip.Reader
	reader   *bufio.Reader
}

// NewReader создаёт новый Reader
// Поддерживает .ark и .ark.gz файлы
// Автоматически проверяет формат (бинарный vs текстовый)
func NewReader(path string) (*Reader, error) {
	// gzip файлы нельзя проверить DetectFormat — они сжатые
	if !strings.HasSuffix(path, ".gz") {
		if err := DetectFormat(path); err != nil {
			return nil, fmt.Errorf("format check failed for %s: %w", path, err)
		}
	}

	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	r := &Reader{file: file}

	if strings.HasSuffix(path, ".gz") {
		r.gzReader, err = gzip.NewReader(file)
		if err != nil {
			file.Close()
			return nil, fmt.Errorf("gzip reader failed for %s: %w", path, err)
		}
		r.reader = bufio.NewReader(r.gzReader)
	} else {
		r.reader = bufio.NewReader(file)
	}

	return r, nil
}

// Close закрывает файл
func (r *Reader) Close() error {
	if r.gzReader != nil {
		r.gzReader.Close()
	}
	return r.file.Close()
}

// DetectFormat проверяет что файл бинарный ark, не текстовый
// Ищет паттерн \0B (binary marker) в первых 256 байтах
func DetectFormat(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	buf := make([]byte, 256)
	n, err := f.Read(buf)
	if err != nil || n < 10 {
		return fmt.Errorf("file too small or unreadable (%d bytes)", n)
	}

	// Binary ark: "key \0B..." — ищем \0B
	for i := 0; i < n-1; i++ {
		if buf[i] == 0x00 && buf[i+1] == 'B' {
			return nil // binary format confirmed
		}
	}

	// Не нашли \0B — проверяем на текстовые признаки
	hasNewline := false
	for i := 0; i < n; i++ {
		if buf[i] == '\n' {
			hasNewline = true
			break
		}
	}
	if hasNewline {
		return fmt.Errorf("text ark format detected (no binary \\0B marker found), only binary ark files supported")
	}

	return fmt.Errorf("unknown format (no binary \\0B marker found in first %d bytes)", n)
}

// ReadExample читает следующий пример
func (r *Reader) ReadExample() (*Example, error) {
	key, found := r.findExampleStart()
	if !found {
		return nil, nil // EOF
	}

	ex, err := r.parseExample()
	if err != nil {
		return nil, err
	}
	ex.Key = key
	return ex, nil
}

// findExampleStart ищет начало следующего примера
func (r *Reader) findExampleStart() (string, bool) {
	var keyBuf []byte
	inKey := false

	for {
		b, err := r.reader.ReadByte()
		if err != nil {
			return "", false
		}
		if !inKey {
			if isLetter(b) {
				inKey = true
				keyBuf = []byte{b}
			}
			continue
		}
		if isLetter(b) || isDigit(b) || b == '-' || b == '_' || b == '.' {
			keyBuf = append(keyBuf, b)
			continue
		}
		if b == ' ' && len(keyBuf) >= 3 {
			b2, _ := r.reader.ReadByte()
			if b2 == 0 {
				b3, _ := r.reader.ReadByte()
				if b3 == 'B' {
					return string(keyBuf), true
				}
			}
		}
		inKey = false
		keyBuf = nil
	}
}

// parseExample парсит структуру примера
func (r *Reader) parseExample() (*Example, error) {
	ex := &Example{}
	var currentIoName string
	var currentIoSize int
	var currentIndexes []Index

	for {
		b, err := r.reader.ReadByte()
		if err != nil {
			return ex, fmt.Errorf("unexpected EOF")
		}

		// Матрица: CM, CM2, CM3, FM - ТОЛЬКО после всех <I1> прочитаны
		if (b == 'C' || b == 'F') && currentIoName != "" {
			b2, err := r.reader.ReadByte()
			if err != nil {
				continue
			}

			var mat *MatrixInfo
			if b == 'C' && b2 == 'M' {
				b3, err := r.reader.ReadByte()
				if err != nil {
					continue
				}
				switch b3 {
				case '2':
					r.reader.ReadByte() // space
					mat = r.readCM2()
				case '3':
					r.reader.ReadByte() // space
					mat = r.readCM3()
				case ' ':
					mat = r.readCM()
				default:
					r.reader.UnreadByte()
					continue
				}
			} else if b == 'F' && b2 == 'M' {
				b3, _ := r.reader.ReadByte()
				if b3 == ' ' {
					mat = r.readFM()
				} else {
					r.reader.UnreadByte()
					continue
				}
			} else {
				r.reader.UnreadByte()
				continue
			}

			if mat != nil {
				io := IoBlock{
					Name:    currentIoName,
					Size:    currentIoSize,
					Indexes: currentIndexes,
					Matrix:  *mat,
				}
				ex.Inputs = append(ex.Inputs, io)
				if debugMatrix() {
					fmt.Printf("[DEBUG] Added input: name=%s, rows=%d, cols=%d\n",
						currentIoName, mat.Rows, mat.Cols)
				}
				currentIoName = ""
				currentIoSize = 0
			}
			continue
		}

		// Тег
		if b == '<' {
			tag, valid := r.tryReadTag()
			if !valid {
				continue
			}
			if debugFull() {
				fmt.Printf("[DEBUG] Tag: <%s>\n", tag)
			}
			switch tag {
			case "NumInputs":
				ex.NumInputs = int(r.readBasicIntValue())
			case "NumOutputs":
				ex.NumOutputs = int(r.readBasicIntValue())
			case "NnetIo":
				currentIoName = r.readName()
				if debugFull() {
					fmt.Printf("[DEBUG] NnetIo: name=%s\n", currentIoName)
				}
			case "I1V":
				count := int(r.readBasicIntValue())
				indexes, err := r.readIndexVector(count)
				if err != nil {
					return ex, fmt.Errorf("I1V read error (name=%s): %w", currentIoName, err)
				}
				if currentIoName != "" {
					currentIoSize = count
					currentIndexes = indexes
				} else if ex.Supervision.Name != "" {
					ex.Supervision.Size = count
					ex.Supervision.Indexes = indexes
				}
			case "/NnetIo":
				currentIoName = ""
				currentIoSize = 0
			case "NnetChainSup":
				ex.Supervision.Name = r.readName()
			case "Weight":
				r.reader.ReadByte() // space
				r.reader.ReadByte() // size byte
				ex.Supervision.Weight = r.readFloat32()
			case "NumSequences":
				ex.Supervision.NumSequences = int(r.readBasicIntValue())
			case "FramesPerSeq":
				ex.Supervision.FramesPerSeq = int(r.readBasicIntValue())
			case "LabelDim":
				ex.Supervision.LabelDim = int(r.readBasicIntValue())
			case "End2End":
				r.reader.ReadByte() // space
				e2e, _ := r.reader.ReadByte()
				ex.Supervision.End2End = (e2e == 'T')
				if !ex.Supervision.End2End {
					// Read FST
					fst := ReadFst(r.reader)
					if fst == nil {
						return ex, fmt.Errorf("failed to read FST for example %s", ex.Key)
					}
					ex.Supervision.Fst = fst
				}
			case "DW", "DW2":
				ex.Supervision.DerivWeights = readDerivWeights(r.reader, tag)
			case "/Nnet3ChainEg":
				debugExampleCount++
				if debugFull() {
					fmt.Printf("[DEBUG] End of example, inputs=%d\n", len(ex.Inputs))
				}
				return ex, nil
			}
		}
	}
}

// readCM reads CM format - calls matrix.go function
func (r *Reader) readCM() *MatrixInfo {
	globalMin := r.readFloat32()
	globalRange := r.readFloat32()
	numRows := r.readInt32()
	numCols := r.readInt32()

	if debugMatrix() {
		fmt.Printf("[DEBUG] CM: min=%.2f, range=%.2f, rows=%d, cols=%d\n",
			globalMin, globalRange, numRows, numCols)
	}

	if numRows <= 0 || numCols <= 0 || numRows > 100000 || numCols > 10000 {
		return nil
	}

	mat := ReadCompressedMatrix(r.reader, int(numRows), int(numCols), globalMin, globalRange)

	if debugSummary() && debugExampleCount == 0 && mat != nil && len(mat.FirstRow) >= 3 {
		fmt.Printf("[DEBUG] FirstRow[0:3]: [%.2f, %.2f, %.2f]\n",
			mat.FirstRow[0], mat.FirstRow[1], mat.FirstRow[2])
	}

	return mat
}

// readCM2 reads CM2 format - calls matrix.go function
func (r *Reader) readCM2() *MatrixInfo {
	globalMin := r.readFloat32()
	globalRange := r.readFloat32()
	numRows := r.readInt32()
	numCols := r.readInt32()

	if debugMatrix() {
		fmt.Printf("[DEBUG] CM2: min=%.2f, range=%.2f, rows=%d, cols=%d\n",
			globalMin, globalRange, numRows, numCols)
	}

	if numRows <= 0 || numCols <= 0 || numRows > 100000 || numCols > 10000 {
		return nil
	}

	return ReadCompressedMatrix2(r.reader, int(numRows), int(numCols), globalMin, globalRange)
}

// readCM3 reads CM3 format - calls matrix.go function
func (r *Reader) readCM3() *MatrixInfo {
	globalMin := r.readFloat32()
	globalRange := r.readFloat32()
	numRows := r.readInt32()
	numCols := r.readInt32()

	if debugMatrix() {
		fmt.Printf("[DEBUG] CM3: min=%.2f, range=%.2f, rows=%d, cols=%d\n",
			globalMin, globalRange, numRows, numCols)
	}

	if numRows <= 0 || numCols <= 0 || numRows > 100000 || numCols > 10000 {
		return nil
	}

	return ReadCompressedMatrix3(r.reader, int(numRows), int(numCols), globalMin, globalRange)
}

// readFM reads FM format - calls matrix.go function
func (r *Reader) readFM() *MatrixInfo {
	sizeByte, _ := r.reader.ReadByte()
	if sizeByte != 4 {
		return nil
	}

	numRows := r.readInt32()
	numCols := r.readInt32()

	if numRows <= 0 || numCols <= 0 {
		return nil
	}

	return ReadFullMatrix(r.reader, int(numRows), int(numCols))
}

// Helper functions for reading

func (r *Reader) tryReadTag() (string, bool) {
	var tagBytes []byte
	for {
		b, err := r.reader.ReadByte()
		if err != nil {
			return "", false
		}
		if b == '>' {
			break
		}
		if b == ' ' {
			r.reader.UnreadByte()
			break
		}
		if !isLetter(b) && !isDigit(b) && b != '/' && b != '_' {
			return "", false
		}
		tagBytes = append(tagBytes, b)
		if len(tagBytes) > 30 {
			return "", false
		}
	}
	if len(tagBytes) < 2 {
		return "", false
	}
	return string(tagBytes), true
}

func (r *Reader) readName() string {
	b, _ := r.reader.ReadByte()
	if b != ' ' {
		r.reader.UnreadByte()
	}
	var name []byte
	for {
		b, err := r.reader.ReadByte()
		if err != nil || b == ' ' || b == '<' {
			if b == '<' {
				r.reader.UnreadByte()
			}
			break
		}
		name = append(name, b)
	}
	return string(name)
}

func (r *Reader) readBasicIntValue() int32 {
	r.reader.ReadByte() // space
	size, _ := r.reader.ReadByte()
	switch size {
	case 1:
		b, _ := r.reader.ReadByte()
		return int32(b)
	case 4:
		return r.readInt32()
	}
	return 0
}

func (r *Reader) readFloat32() float32 {
	var buf [4]byte
	io.ReadFull(r.reader, buf[:])
	return math.Float32frombits(binary.LittleEndian.Uint32(buf[:]))
}

func (r *Reader) readInt32() int32 {
	var buf [4]byte
	io.ReadFull(r.reader, buf[:])
	return int32(binary.LittleEndian.Uint32(buf[:]))
}

func isLetter(b byte) bool { return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') }
func isDigit(b byte) bool  { return b >= '0' && b <= '9' }

// Validate проверяет корректность примера
func (ex *Example) Validate() bool {
	if ex.NumInputs != 2 || ex.NumOutputs != 1 || len(ex.Inputs) != 2 {
		return false
	}
	if ex.Inputs[0].Name != "input" || ex.Inputs[0].Matrix.Cols != 40 {
		return false
	}
	if ex.Inputs[1].Name != "ivector" || ex.Inputs[1].Matrix.Rows != 1 || ex.Inputs[1].Matrix.Cols != 100 {
		return false
	}
	return true
}

// IsUsable проверяет, можно ли использовать пример для тренировки
func (ex *Example) IsUsable() bool {
	return ex.Validate() && ex.Supervision.Weight > 0 && ex.Supervision.LabelDim == 3080
}

// readIndexVector reads index vector after <I1V> count
// Format: delta encoding
// Kaldi writes deltas in range [-124, 124], 127 = long format
func (r *Reader) readIndexVector(count int) ([]Index, error) {
	if count <= 0 {
		return nil, fmt.Errorf("invalid index vector count: %d", count)
	}
	indexes := make([]Index, count)

	for i := 0; i < count; i++ {
		b, err := r.reader.ReadByte()
		if err != nil {
			return indexes[:i], fmt.Errorf("EOF after %d/%d indexes", i, count)
		}

		c := int8(b)

		if debugFull() && i < 5 {
			fmt.Printf("[DEBUG] readIndex[%d]: byte=%d (0x%02x)\n", i, c, b)
		}

		if c == 127 {
			// Long format: n, t, x as WriteBasicType
			indexes[i] = Index{
				N: int(r.readBasicIntValue()),
				T: int(r.readBasicIntValue()),
				X: int(r.readBasicIntValue()),
			}
			if debugFull() {
				fmt.Printf("[DEBUG] readIndex[%d]: long format n=%d t=%d x=%d\n",
					i, indexes[i].N, indexes[i].T, indexes[i].X)
			}
		} else if c == -128 || (c >= -127 && c <= -125) || c == 125 || c == 126 {
			// Values outside Kaldi's valid delta range [-124, 124]
			fmt.Printf("[WARN] readIndex[%d]: unexpected delta byte %d (0x%02x), possible data corruption\n", i, c, b)
			if i == 0 {
				indexes[i] = Index{N: 0, T: int(c), X: 0}
			} else {
				last := indexes[i-1]
				indexes[i] = Index{N: last.N, T: last.T + int(c), X: last.X}
			}
		} else {
			// Delta format: t_delta in [-124, 124]
			if i == 0 {
				indexes[i] = Index{N: 0, T: int(c), X: 0}
			} else {
				last := indexes[i-1]
				indexes[i] = Index{N: last.N, T: last.T + int(c), X: last.X}
			}
		}
	}

	// Validate: our dataset uses only n=0, x=0
	for i := range indexes {
		if indexes[i].N != 0 {
			fmt.Printf("[WARN] index[%d]: n=%d (merged egs detected, not supported in our pipeline)\n", i, indexes[i].N)
			break // warn once
		}
	}
	for i := range indexes {
		if indexes[i].X != 0 {
			fmt.Printf("[WARN] index[%d]: x=%d (extra dimension detected, not supported in our pipeline)\n", i, indexes[i].X)
			break // warn once
		}
	}

	return indexes, nil
}
