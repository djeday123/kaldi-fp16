// Package egsreader reads Kaldi NNet3 chain examples (cegs)
package egsreader

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

// Reader читает примеры из ark файла
type Reader struct {
	file   *os.File
	reader *bufio.Reader
}

// Open открывает ark файл для чтения
func Open(path string) (*Reader, error) {
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

// Next читает следующий пример. Возвращает nil, nil при EOF
func (r *Reader) Next() (*NNet3ChainExample, error) {
	key, found := r.findExampleStart()
	if !found {
		return nil, nil
	}

	ex, err := r.parseExample()
	if err != nil {
		return nil, err
	}
	ex.Key = key
	return ex, nil
}

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

		if isLetter(b) || isDigit(b) || b == '-' || b == '_' {
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

func (r *Reader) parseExample() (*NNet3ChainExample, error) {
	ex := &NNet3ChainExample{
		Supervision: &ChainSupervision{},
	}

	var currentIoName string
	var currentIoSize int
	inSupervision := false

	for {
		b, err := r.reader.ReadByte()
		if err != nil {
			return ex, fmt.Errorf("unexpected EOF")
		}

		// Матрица: CM, CM2, CM3, FM
		if b == 'C' || b == 'F' {
			mat := r.tryParseMatrix(b)
			if mat != nil {
				io := &NNetIo{
					Name: currentIoName,
					Data: mat,
				}
				// Заполняем индексы
				for t := 0; t < currentIoSize; t++ {
					io.Indexes = append(io.Indexes, Index{N: 0, T: int32(t), X: 0})
				}

				switch currentIoName {
				case "input":
					ex.Input = io
				case "ivector":
					ex.Ivector = io
				}
				currentIoName = ""
			}
			continue
		}

		// Тег
		if b == '<' {
			tag, valid := r.tryReadTag()
			if !valid {
				continue
			}

			switch tag {
			case "NnetIo":
				currentIoName = r.readName()
			case "I1V":
				currentIoSize = int(r.readBasicInt())
			case "NnetChainSup":
				ex.Supervision.Name = r.readName()
			case "Supervision":
				inSupervision = true
				_ = inSupervision // suppress unused warning
			case "Weight":
				r.reader.ReadByte() // space
				r.reader.ReadByte() // size
				ex.Supervision.Weight = r.readFloat32()
			case "NumSequences":
				ex.Supervision.NumSequences = r.readBasicInt()
			case "FramesPerSeq":
				ex.Supervision.FramesPerSeq = r.readBasicInt()
			case "LabelDim":
				ex.Supervision.LabelDim = r.readBasicInt()
			case "End2End":
				ex.Supervision.E2ESupervision = r.readBasicInt() != 0
			case "/Nnet3ChainEg":
				return ex, nil
			}
		}
	}
}

func (r *Reader) tryParseMatrix(firstByte byte) *Matrix {
	// Peek для проверки без изменения позиции
	// GlobalHeader = 20 bytes (format + min + range + rows + cols)
	// Нужно peek: marker(1-2) + space(1) + header(20) = 23 bytes max
	peek, err := r.reader.Peek(23)
	if err != nil || len(peek) < 21 {
		return nil
	}

	if peek[0] != 'M' {
		return nil
	}

	matType := "CM"
	if firstByte == 'F' {
		matType = "FM"
	}

	headerOffset := 2 // после "M "

	if peek[1] == '2' {
		if peek[2] != ' ' {
			return nil
		}
		matType = "CM2"
		headerOffset = 3
	} else if peek[1] == '3' {
		if peek[2] != ' ' {
			return nil
		}
		matType = "CM3"
		headerOffset = 3
	} else if peek[1] != ' ' {
		return nil
	}

	// Читаем полный GlobalHeader (20 bytes)
	headerBytes := peek[headerOffset : headerOffset+20]
	format := int32(binary.LittleEndian.Uint32(headerBytes[0:4]))
	globalMin := math.Float32frombits(binary.LittleEndian.Uint32(headerBytes[4:8]))
	globalRange := math.Float32frombits(binary.LittleEndian.Uint32(headerBytes[8:12]))
	rows := int32(binary.LittleEndian.Uint32(headerBytes[12:16]))
	cols := int32(binary.LittleEndian.Uint32(headerBytes[16:20]))

	// Валидация
	if rows <= 0 || cols <= 0 || rows > 100000 || cols > 10000 {
		return nil
	}
	if globalRange < 0 || globalRange > 10000 {
		return nil
	}
	// Проверка format
	if format < 1 || format > 3 {
		return nil
	}

	// Всё валидно - читаем (discard marker + space + header)
	r.reader.Discard(headerOffset + 20)

	// Читаем данные матрицы
	var data []float32
	switch matType {
	case "CM":
		// format должен быть 1 (kOneByteWithColHeaders)
		data = r.readCompressedMatrixCM(int(rows), int(cols), globalMin, globalRange)
	case "CM2":
		// format должен быть 2 (kTwoByte)
		data = r.readCompressedMatrixCM2(int(rows), int(cols), globalMin, globalRange)
	case "CM3":
		// format должен быть 3 (kOneByte)
		data = r.readCompressedMatrixCM3(int(rows), int(cols), globalMin, globalRange)
	case "FM":
		data = r.readFullMatrix(int(rows), int(cols))
	}

	return &Matrix{
		Rows: int(rows),
		Cols: int(cols),
		Data: data,
	}
}

// PerColHeader holds percentile values for one column (8 bytes)
type PerColHeader struct {
	Percentile0   uint16 // min
	Percentile25  uint16 // 25th percentile
	Percentile75  uint16 // 75th percentile
	Percentile100 uint16 // max
}

// uint16ToFloat converts percentile uint16 to float using global header
// Formula: min_value + range * (1/65535) * value
func uint16ToFloat(globalMin, globalRange float32, value uint16) float32 {
	const inv65535 = 1.52590218966964e-05 // 1/65535
	return globalMin + globalRange*inv65535*float32(value)
}

// charToFloat converts uint8 data byte to float using column percentiles
// Uses piecewise linear interpolation in 3 ranges:
//
//	[0-64]    → linear from p0 to p25
//	[64-192]  → linear from p25 to p75
//	[192-255] → linear from p75 to p100
func charToFloat(p0, p25, p75, p100 float32, value uint8) float32 {
	if value <= 64 {
		return p0 + (p25-p0)*float32(value)*(1.0/64.0)
	} else if value <= 192 {
		return p25 + (p75-p25)*float32(value-64)*(1.0/128.0)
	} else {
		return p75 + (p100-p75)*float32(value-192)*(1.0/63.0)
	}
}

// readCompressedMatrixCM reads CM format (kOneByteWithColHeaders)
// Layout: [PerColHeaders × cols][ByteData column-major]
func (r *Reader) readCompressedMatrixCM(rows, cols int, globalMin, globalRange float32) []float32 {
	// Читаем per-column headers (8 bytes per col)
	colHeaders := make([]PerColHeader, cols)
	for c := 0; c < cols; c++ {
		binary.Read(r.reader, binary.LittleEndian, &colHeaders[c].Percentile0)
		binary.Read(r.reader, binary.LittleEndian, &colHeaders[c].Percentile25)
		binary.Read(r.reader, binary.LittleEndian, &colHeaders[c].Percentile75)
		binary.Read(r.reader, binary.LittleEndian, &colHeaders[c].Percentile100)
	}

	// Читаем byte данные (COLUMN-MAJOR!)
	rawData := make([]byte, rows*cols)
	io.ReadFull(r.reader, rawData)

	// Распаковываем в row-major output
	data := make([]float32, rows*cols)

	for col := 0; col < cols; col++ {
		// Конвертируем percentiles в float для этой колонки
		p0 := uint16ToFloat(globalMin, globalRange, colHeaders[col].Percentile0)
		p25 := uint16ToFloat(globalMin, globalRange, colHeaders[col].Percentile25)
		p75 := uint16ToFloat(globalMin, globalRange, colHeaders[col].Percentile75)
		p100 := uint16ToFloat(globalMin, globalRange, colHeaders[col].Percentile100)

		// Декомпрессия колонки
		for row := 0; row < rows; row++ {
			// Input: column-major index
			byteIdx := col*rows + row
			// Output: row-major index
			outIdx := row*cols + col
			data[outIdx] = charToFloat(p0, p25, p75, p100, rawData[byteIdx])
		}
	}

	return data
}

// readCompressedMatrixCM2 reads CM2 format (kTwoByte)
// Layout: [uint16 data row-major]
func (r *Reader) readCompressedMatrixCM2(rows, cols int, globalMin, globalRange float32) []float32 {
	// CM2: uint16 данные без per-column headers, row-major
	rawData := make([]uint16, rows*cols)
	binary.Read(r.reader, binary.LittleEndian, rawData)

	data := make([]float32, rows*cols)
	increment := globalRange / 65535.0

	for i := 0; i < len(rawData); i++ {
		data[i] = globalMin + float32(rawData[i])*increment
	}

	return data
}

// readCompressedMatrixCM3 reads CM3 format (kOneByte)
// Layout: [uint8 data row-major]
func (r *Reader) readCompressedMatrixCM3(rows, cols int, globalMin, globalRange float32) []float32 {
	// CM3: uint8 данные без per-column headers, row-major
	rawData := make([]byte, rows*cols)
	io.ReadFull(r.reader, rawData)

	data := make([]float32, rows*cols)
	increment := globalRange / 255.0

	for i := 0; i < len(rawData); i++ {
		data[i] = globalMin + float32(rawData[i])*increment
	}

	return data
}

// readCompressedMatrix - старая функция, оставлена для совместимости
// DEPRECATED: используйте readCompressedMatrixCM
func (r *Reader) readCompressedMatrix(rows, cols int) []float32 {
	// Читаем per-column headers (8 bytes per col: percentile0, percentile25, percentile75, percentile100)
	colHeaders := make([]struct{ min, range_ float32 }, cols)
	for c := 0; c < cols; c++ {
		var p0, p25, p75, p100 uint16
		binary.Read(r.reader, binary.LittleEndian, &p0)
		binary.Read(r.reader, binary.LittleEndian, &p25)
		binary.Read(r.reader, binary.LittleEndian, &p75)
		binary.Read(r.reader, binary.LittleEndian, &p100)
		// Simplified: use p0 as min, (p100-p0) as range
		colHeaders[c].min = float32(p0)
		colHeaders[c].range_ = float32(p100) - float32(p0)
	}

	// Читаем uint8 данные
	rawData := make([]byte, rows*cols)
	io.ReadFull(r.reader, rawData)

	// Распаковываем
	data := make([]float32, rows*cols)
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			idx := row*cols + col
			val := float32(rawData[idx]) / 255.0
			data[idx] = colHeaders[col].min + val*colHeaders[col].range_
		}
	}

	return data
}

// readCompressedMatrix2 - старая функция для CM2/CM3, оставлена для совместимости
// DEPRECATED: используйте readCompressedMatrixCM2 или readCompressedMatrixCM3
func (r *Reader) readCompressedMatrix2(rows, cols int, globalMin, globalRange float32) []float32 {
	// CM2: uint16 данные без per-column headers
	rawData := make([]uint16, rows*cols)
	binary.Read(r.reader, binary.LittleEndian, rawData)

	data := make([]float32, rows*cols)
	for i := 0; i < len(rawData); i++ {
		val := float32(rawData[i]) / 65535.0
		data[i] = globalMin + val*globalRange
	}

	return data
}

func (r *Reader) readFullMatrix(rows, cols int) []float32 {
	data := make([]float32, rows*cols)
	binary.Read(r.reader, binary.LittleEndian, data)
	return data
}

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

	if len(tagBytes) < 3 {
		return "", false
	}

	if !isLetter(tagBytes[0]) && tagBytes[0] != '/' {
		return "", false
	}

	if tagBytes[0] == '/' {
		if len(tagBytes) < 4 || !isLetter(tagBytes[1]) {
			return "", false
		}
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

func (r *Reader) readBasicInt() int32 {
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
	r.reader.Read(buf[:])
	bits := binary.LittleEndian.Uint32(buf[:])
	return math.Float32frombits(bits)
}

func (r *Reader) readInt32() int32 {
	var buf [4]byte
	r.reader.Read(buf[:])
	return int32(binary.LittleEndian.Uint32(buf[:]))
}

func isLetter(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')
}

func isDigit(b byte) bool {
	return b >= '0' && b <= '9'
}

// Validate проверяет структуру примера
func (ex *NNet3ChainExample) Validate() error {
	if ex.Input == nil {
		return fmt.Errorf("missing input")
	}
	if ex.Input.Data == nil || ex.Input.Data.Cols != 40 {
		return fmt.Errorf("invalid input: cols=%d, expected 40", ex.Input.Data.Cols)
	}
	if ex.Ivector == nil {
		return fmt.Errorf("missing ivector")
	}
	if ex.Ivector.Data == nil || ex.Ivector.Data.Rows != 1 || ex.Ivector.Data.Cols != 100 {
		return fmt.Errorf("invalid ivector: rows=%d, cols=%d", ex.Ivector.Data.Rows, ex.Ivector.Data.Cols)
	}
	return nil
}

// IsUsable возвращает true если пример можно использовать для тренировки
func (ex *NNet3ChainExample) IsUsable() bool {
	return ex.Supervision != nil && ex.Supervision.Weight > 0 && ex.Supervision.LabelDim == 3080
}
