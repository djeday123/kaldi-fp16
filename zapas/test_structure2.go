package main

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// Структура для парсинга
type Example struct {
	Key         string
	NumInputs   int
	NumOutputs  int
	Inputs      []IoBlock
	Supervision SupervisionBlock
}

type IoBlock struct {
	Name   string
	Size   int
	Matrix MatrixInfo
}

type MatrixInfo struct {
	Type  string // CM, CM2, CM3, FM
	Rows  int
	Cols  int
	Min   float32
	Range float32
}

type SupervisionBlock struct {
	Name         string
	Size         int
	Weight       float32
	NumSequences int
	FramesPerSeq int
	LabelDim     int
	End2End      bool
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: test_structure2 <ark_path>")
		os.Exit(1)
	}

	file, _ := os.Open(os.Args[1])
	defer file.Close()
	reader := bufio.NewReader(file)

	showDetails := 3
	count := 0
	valid := 0
	invalid := 0
	zeroWeight := 0
	unusualLabelDim := 0

	// Статистика
	frameSizes := make(map[int]int)

	for {
		key, found := findExampleStart(reader)
		if !found {
			break
		}
		count++

		ex, err := parseExampleStruct(reader)
		if err != nil {
			invalid++
			if count <= 10 {
				fmt.Printf("Example %d (%s): ERROR - %v\n", count, key, err)
			}
			continue
		}
		ex.Key = key

		// Валидация структуры
		if !validateExample(ex) {
			invalid++
			if invalid <= 10 {
				fmt.Printf("Example %d (%s): INVALID structure\n", count, key)
			}
			continue
		}

		valid++

		// Подсчёт особых случаев (не ошибки парсера, а реальные данные)
		if ex.Supervision.Weight == 0 {
			zeroWeight++
		}
		if ex.Supervision.LabelDim != 3080 {
			unusualLabelDim++
		}

		// Статистика
		if len(ex.Inputs) > 0 {
			frameSizes[ex.Inputs[0].Matrix.Rows]++
		}

		// Вывод первых N
		if count <= showDetails {
			jsonBytes, _ := json.MarshalIndent(ex, "", "  ")
			fmt.Printf("\n========== EXAMPLE %d ==========\n", count)
			fmt.Println(string(jsonBytes))
		}

		if count%2000 == 0 {
			fmt.Printf("Processed %d examples (%d valid, %d invalid)...\n", count, valid, invalid)
		}
	}

	fmt.Printf("\n========================================\n")
	fmt.Printf("TOTAL: %d examples\n", count)
	fmt.Printf("VALID: %d (%.2f%%)\n", valid, float64(valid)/float64(count)*100)
	fmt.Printf("INVALID: %d (%.2f%%)\n", invalid, float64(invalid)/float64(count)*100)

	fmt.Printf("\nSpecial cases (valid but unusual):\n")
	fmt.Printf("  Weight=0 (skipped in training): %d\n", zeroWeight)
	fmt.Printf("  LabelDim!=3080: %d\n", unusualLabelDim)
	fmt.Printf("  Usable for training: %d\n", valid-zeroWeight-unusualLabelDim)

	fmt.Printf("\nFrame size distribution:\n")
	for size, cnt := range frameSizes {
		fmt.Printf("  %d frames: %d examples (%.1f%%)\n", size, cnt, float64(cnt)/float64(count)*100)
	}
}

func findExampleStart(reader *bufio.Reader) (string, bool) {
	var keyBuf []byte
	inKey := false

	for {
		b, err := reader.ReadByte()
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
			b2, _ := reader.ReadByte()
			if b2 == 0 {
				b3, _ := reader.ReadByte()
				if b3 == 'B' {
					return string(keyBuf), true
				}
			}
		}

		inKey = false
		keyBuf = nil
	}
}

func parseExampleStruct(reader *bufio.Reader) (*Example, error) {
	ex := &Example{}

	var currentIo *IoBlock
	inSupervision := false

	for {
		b, err := reader.ReadByte()
		if err != nil {
			return ex, fmt.Errorf("unexpected EOF")
		}

		// Матрица: CM, CM2, CM3, FM
		if b == 'C' || b == 'F' {
			// Peek для проверки без изменения позиции
			peek, err := reader.Peek(19) // "M2 " + 16 bytes header
			if err != nil || len(peek) < 19 {
				continue
			}

			if peek[0] != 'M' {
				continue
			}

			matType := "CM"
			if b == 'F' {
				matType = "FM"
			}

			headerOffset := 2 // после "M "

			if peek[1] == '2' {
				if peek[2] != ' ' {
					continue // ложный CM2
				}
				matType = "CM2"
				headerOffset = 3 // после "M2 "
			} else if peek[1] == '3' {
				if peek[2] != ' ' {
					continue
				}
				matType = "CM3"
				headerOffset = 3
			} else if peek[1] != ' ' {
				continue // CM/FM без пробела
			}

			// Проверяем header в peek
			headerBytes := peek[headerOffset : headerOffset+16]
			globalMin := math.Float32frombits(binary.LittleEndian.Uint32(headerBytes[0:4]))
			globalRange := math.Float32frombits(binary.LittleEndian.Uint32(headerBytes[4:8]))
			rows := int32(binary.LittleEndian.Uint32(headerBytes[8:12]))
			cols := int32(binary.LittleEndian.Uint32(headerBytes[12:16]))

			// Строгая валидация
			if rows <= 0 || cols <= 0 || rows > 100000 || cols > 10000 {
				continue
			}
			if globalRange < 0 || globalRange > 10000 {
				continue
			}

			// Всё валидно - теперь реально читаем
			reader.Discard(headerOffset + 16)

			// Пропускаем данные матрицы
			dataSize := 0
			switch matType {
			case "CM":
				dataSize = int(cols)*8 + int(rows)*int(cols)
			case "CM2", "CM3":
				dataSize = int(rows) * int(cols) * 2
			case "FM":
				dataSize = int(rows) * int(cols) * 4
			}
			skipBytes(reader, dataSize)

			mat := &MatrixInfo{
				Type:  matType,
				Rows:  int(rows),
				Cols:  int(cols),
				Min:   globalMin,
				Range: globalRange,
			}

			if currentIo != nil {
				currentIo.Matrix = *mat
				ex.Inputs = append(ex.Inputs, *currentIo)
				currentIo = nil
			}
			continue
		}

		// Тег
		if b == '<' {
			tag, valid := tryReadTag(reader)
			if !valid {
				continue
			}

			switch tag {
			case "NumInputs":
				ex.NumInputs = int(readBasicIntValue(reader))
			case "NumOutputs":
				ex.NumOutputs = int(readBasicIntValue(reader))
			case "NnetIo":
				name := readName(reader)
				currentIo = &IoBlock{Name: name}
			case "I1V":
				size := int(readBasicIntValue(reader))
				if currentIo != nil {
					currentIo.Size = size
				} else if inSupervision {
					ex.Supervision.Size = size
				}
			case "NnetChainSup":
				ex.Supervision.Name = readName(reader)
			case "Supervision":
				inSupervision = true
			case "Weight":
				reader.ReadByte() // space
				reader.ReadByte() // size
				ex.Supervision.Weight = readFloat32(reader)
			case "NumSequences":
				ex.Supervision.NumSequences = int(readBasicIntValue(reader))
			case "FramesPerSeq":
				ex.Supervision.FramesPerSeq = int(readBasicIntValue(reader))
			case "LabelDim":
				ex.Supervision.LabelDim = int(readBasicIntValue(reader))
			case "End2End":
				ex.Supervision.End2End = readBasicIntValue(reader) != 0
			case "/Nnet3ChainEg":
				return ex, nil
			}
		}
	}
}

func validateExample(ex *Example) bool {
	// Проверяем структуру
	if ex.NumInputs != 2 {
		fmt.Printf("  -> NumInputs=%d (expected 2)\n", ex.NumInputs)
		return false
	}
	if ex.NumOutputs != 1 {
		fmt.Printf("  -> NumOutputs=%d (expected 1)\n", ex.NumOutputs)
		return false
	}
	if len(ex.Inputs) != 2 {
		fmt.Printf("  -> len(Inputs)=%d (expected 2)\n", len(ex.Inputs))
		return false
	}

	// Input должен быть [N, 40]
	if ex.Inputs[0].Name != "input" || ex.Inputs[0].Matrix.Cols != 40 {
		fmt.Printf("  -> Input[0]: name=%s, cols=%d (expected input, 40)\n",
			ex.Inputs[0].Name, ex.Inputs[0].Matrix.Cols)
		return false
	}

	// Ivector должен быть [1, 100]
	if ex.Inputs[1].Name != "ivector" || ex.Inputs[1].Matrix.Rows != 1 || ex.Inputs[1].Matrix.Cols != 100 {
		fmt.Printf("  -> Input[1]: name=%s, rows=%d, cols=%d (expected ivector, 1, 100)\n",
			ex.Inputs[1].Name, ex.Inputs[1].Matrix.Rows, ex.Inputs[1].Matrix.Cols)
		return false
	}

	return true
}

func tryReadTag(reader *bufio.Reader) (string, bool) {
	var tagBytes []byte

	for {
		b, err := reader.ReadByte()
		if err != nil {
			return "", false
		}

		if b == '>' {
			break
		}

		if b == ' ' {
			reader.UnreadByte()
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

func readName(reader *bufio.Reader) string {
	b, _ := reader.ReadByte()
	if b != ' ' {
		reader.UnreadByte()
	}

	var name []byte
	for {
		b, err := reader.ReadByte()
		if err != nil || b == ' ' || b == '<' {
			if b == '<' {
				reader.UnreadByte()
			}
			break
		}
		name = append(name, b)
	}
	return string(name)
}

func readBasicIntValue(reader *bufio.Reader) int32 {
	reader.ReadByte() // space
	size, _ := reader.ReadByte()

	switch size {
	case 1:
		b, _ := reader.ReadByte()
		return int32(b)
	case 4:
		return readInt32(reader)
	}
	return 0
}

func readFloat32(reader *bufio.Reader) float32 {
	var buf [4]byte
	reader.Read(buf[:])
	bits := binary.LittleEndian.Uint32(buf[:])
	return math.Float32frombits(bits)
}

func readInt32(reader *bufio.Reader) int32 {
	var buf [4]byte
	reader.Read(buf[:])
	return int32(binary.LittleEndian.Uint32(buf[:]))
}

func skipBytes(reader *bufio.Reader, n int) {
	buf := make([]byte, 4096)
	for n > 0 {
		toRead := n
		if toRead > 4096 {
			toRead = 4096
		}
		reader.Read(buf[:toRead])
		n -= toRead
	}
}

func isLetter(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')
}

func isDigit(b byte) bool {
	return b >= '0' && b <= '9'
}
