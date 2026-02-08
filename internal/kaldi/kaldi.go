package kaldi

import (
	"bufio"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

// TextExample - пример из текстового вывода Kaldi
type TextExample struct {
	Key         string
	NumInputs   int
	InputName   string
	InputFrames int
	InputDim    int
	FirstRow    []float32   // для обратной совместимости
	AllRows     [][]float32 // все строки матрицы
}

// GetTextOutput запускает nnet3-chain-copy-egs и возвращает текстовый вывод
func GetTextOutput(arkFile string, maxExamples int) ([]TextExample, error) {
	// Каждый пример ~300 строк (матрица 224 строки + supervision)
	// Берём с запасом: maxExamples * 800 строк
	lineLimit := maxExamples * 800
	if lineLimit < 20000 {
		lineLimit = 20000
	}

	cmd := exec.Command("bash", "-c",
		fmt.Sprintf("nnet3-chain-copy-egs 'ark:%s' ark,t:- 2>/dev/null | head -n %d", arkFile, lineLimit))
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("nnet3-chain-copy-egs failed: %v", err)
	}
	return ParseTextOutput(string(output), maxExamples)
}

// ParseTextOutput парсит текстовый вывод Kaldi
func ParseTextOutput(text string, maxExamples int) ([]TextExample, error) {
	var examples []TextExample
	var current *TextExample
	inInputMatrix := false
	inputMatrixDone := false // Флаг что input матрица полностью прочитана

	scanner := bufio.NewScanner(strings.NewReader(text))
	buf := make([]byte, 0, 1024*1024)
	scanner.Buffer(buf, 10*1024*1024)

	for scanner.Scan() {
		line := scanner.Text()

		// Проверяем на конец примера (может быть в любом месте строки)
		if strings.Contains(line, "</Nnet3ChainEg>") {
			if current != nil {
				// Устанавливаем FirstRow для совместимости
				if len(current.AllRows) > 0 {
					current.FirstRow = current.AllRows[0]
				}
				examples = append(examples, *current)
				if len(examples) >= maxExamples {
					return examples, nil
				}
			}
			current = nil
			inInputMatrix = false
			inputMatrixDone = false
		}

		// Новый пример - ищем <Nnet3ChainEg>
		idx := strings.Index(line, "<Nnet3ChainEg>")
		if idx >= 0 {
			if idx == 0 || line[idx-1] != '/' {
				current = &TextExample{}
				inInputMatrix = false
				inputMatrixDone = false

				beforeTag := line[:idx]
				parts := strings.Fields(beforeTag)
				if len(parts) > 0 {
					lastPart := parts[len(parts)-1]
					if lastPart != "</Nnet3ChainEg>" {
						current.Key = lastPart
					}
				}

				afterTag := line[idx:]
				tagParts := strings.Fields(afterTag)
				for i, p := range tagParts {
					if p == "<NumInputs>" && i+1 < len(tagParts) {
						current.NumInputs, _ = strconv.Atoi(tagParts[i+1])
					}
				}
				continue
			}
		}

		if current == nil {
			continue
		}

		// Конец NnetIo блока
		if strings.Contains(line, "</NnetIo>") {
			if inInputMatrix {
				inputMatrixDone = true
			}
			inInputMatrix = false
			continue
		}

		// Input header: "<NnetIo> input <I1V> 224 ..."
		if strings.Contains(line, "<NnetIo>") {
			parts := strings.Fields(line)
			for i, p := range parts {
				if p == "<NnetIo>" && i+1 < len(parts) {
					ioName := parts[i+1]
					if current.InputName == "" && ioName == "input" && !inputMatrixDone {
						current.InputName = ioName
						inInputMatrix = true
					}
				}
				if p == "<I1V>" && i+1 < len(parts) && current.InputName == "input" && current.InputFrames == 0 {
					current.InputFrames, _ = strconv.Atoi(parts[i+1])
				}
			}
			continue
		}

		// Строки матрицы - читаем все строки input матрицы
		if inInputMatrix && !inputMatrixDone {
			trimmed := strings.TrimSpace(line)
			if len(trimmed) == 0 {
				continue
			}

			firstChar := trimmed[0]
			if firstChar == '[' || firstChar == '-' || (firstChar >= '0' && firstChar <= '9') {
				// Убираем [ и ] если есть
				trimmed = strings.TrimPrefix(trimmed, "[")
				isLastRow := strings.HasSuffix(trimmed, "]")
				trimmed = strings.TrimSuffix(trimmed, "]")
				trimmed = strings.TrimSpace(trimmed)

				if len(trimmed) > 0 {
					parts := strings.Fields(trimmed)

					// Устанавливаем размерность по первой строке
					if current.InputDim == 0 {
						current.InputDim = len(parts)
					}

					row := make([]float32, 0, len(parts))
					for _, p := range parts {
						if v, err := strconv.ParseFloat(p, 32); err == nil {
							row = append(row, float32(v))
						}
					}
					current.AllRows = append(current.AllRows, row)

					if isLastRow {
						inputMatrixDone = true
						inInputMatrix = false
					}
				}
			}
		}
	}

	// Последний пример
	if current != nil && len(examples) < maxExamples {
		if len(current.AllRows) > 0 {
			current.FirstRow = current.AllRows[0]
		}
		examples = append(examples, *current)
	}

	return examples, nil
}
