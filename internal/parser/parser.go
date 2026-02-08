package parser

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
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
	file   *os.File
	reader *bufio.Reader
}

// NewReader создаёт новый Reader
func NewReader(path string) (*Reader, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	return &Reader{
		file:   file,
		reader: bufio.NewReader(file),
	}, nil
}

// Close закрывает файл
func (r *Reader) Close() error {
	return r.file.Close()
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
	//var pendingI1Count int // сколько <I1> тегов ещё ожидаем

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
					Name:   currentIoName,
					Size:   currentIoSize,
					Matrix: *mat,
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
				currentIoSize = count
				r.skipIndexVector(count)
				// После skipIndexVector готовы читать матрицу
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
				ex.Supervision.End2End = r.readBasicIntValue() != 0
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

// skipIndexVector пропускает бинарные индексы после <I1V>
// Формат: дельта-кодирование, каждый индекс 1 байт (если |delta| < 125)
// или 1 + 12 байт (если байт == 127)
func (r *Reader) skipIndexVector(count int) {
	if debugFull() {
		fmt.Printf("[DEBUG] skipIndexVector: count=%d\n", count)
	}
	for i := 0; i < count; i++ {
		b, err := r.reader.ReadByte()
		if err != nil {
			return
		}
		if debugFull() && i < 5 {
			fmt.Printf("[DEBUG] skipIndex[%d]: byte=%d (0x%02x)\n", i, int8(b), b)
		}
		if b == 127 {
			if debugFull() {
				fmt.Printf("[DEBUG] skipIndex[%d]: long format, skipping 15 bytes\n", i)
			}
			for j := 0; j < 15; j++ {
				r.reader.ReadByte()
			}
		}
	}
}
