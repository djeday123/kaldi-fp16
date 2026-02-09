package batch

import (
	"fmt"

	"kaldi-fp16/internal/parser"
)

// Batch представляет мини-батч для тренировки
type Batch struct {
	Examples []*parser.Example

	// Объединённые данные
	Features *MergedMatrix // [total_frames, feat_dim] — все input фреймы
	Ivectors *MergedMatrix // [batch_size, ivector_dim] — один ivector на пример

	// Метаданные
	FrameOffsets []int // начало фреймов каждого примера в Features
	NumFrames    []int // количество фреймов каждого примера
	BatchSize    int
}

// MergedMatrix — объединённая матрица из нескольких примеров
type MergedMatrix struct {
	Rows int
	Cols int
	Data []float32 // row-major
}

// At returns element at (row, col)
func (m *MergedMatrix) At(row, col int) float32 {
	return m.Data[row*m.Cols+col]
}

// Row returns a slice of the row data (zero-copy)
func (m *MergedMatrix) Row(row int) []float32 {
	start := row * m.Cols
	return m.Data[start : start+m.Cols]
}

// NewBatch создаёт батч из списка примеров
// Объединяет features и ivectors в непрерывные массивы
func NewBatch(examples []*parser.Example) (*Batch, error) {
	if len(examples) == 0 {
		return nil, fmt.Errorf("empty examples list")
	}

	b := &Batch{
		Examples:     examples,
		FrameOffsets: make([]int, len(examples)),
		NumFrames:    make([]int, len(examples)),
		BatchSize:    len(examples),
	}

	// Подсчёт размеров
	totalFrames := 0
	featDim := 0
	ivectorDim := 0

	for i, ex := range examples {
		if len(ex.Inputs) < 1 {
			return nil, fmt.Errorf("example %d (%s): no inputs", i, ex.Key)
		}

		// input — первый блок (features)
		input := &ex.Inputs[0]
		if input.Name != "input" {
			return nil, fmt.Errorf("example %d (%s): first input is '%s', expected 'input'", i, ex.Key, input.Name)
		}

		b.FrameOffsets[i] = totalFrames
		b.NumFrames[i] = input.Matrix.Rows
		totalFrames += input.Matrix.Rows

		if featDim == 0 {
			featDim = input.Matrix.Cols
		} else if input.Matrix.Cols != featDim {
			return nil, fmt.Errorf("example %d (%s): feat_dim=%d, expected %d", i, ex.Key, input.Matrix.Cols, featDim)
		}

		// ivector — второй блок (если есть)
		if len(ex.Inputs) >= 2 {
			iv := &ex.Inputs[1]
			if iv.Name == "ivector" {
				if ivectorDim == 0 {
					ivectorDim = iv.Matrix.Cols
				}
			}
		}
	}

	// Merge features
	b.Features = &MergedMatrix{
		Rows: totalFrames,
		Cols: featDim,
		Data: make([]float32, totalFrames*featDim),
	}

	offset := 0
	for _, ex := range examples {
		input := &ex.Inputs[0]
		n := input.Matrix.Rows * input.Matrix.Cols
		copy(b.Features.Data[offset:offset+n], input.Matrix.Data)
		offset += n
	}

	// Merge ivectors (один на пример)
	if ivectorDim > 0 {
		b.Ivectors = &MergedMatrix{
			Rows: len(examples),
			Cols: ivectorDim,
			Data: make([]float32, len(examples)*ivectorDim),
		}

		for i, ex := range examples {
			if len(ex.Inputs) >= 2 && ex.Inputs[1].Name == "ivector" {
				copy(b.Ivectors.Row(i), ex.Inputs[1].Matrix.Data)
			}
		}
	}

	return b, nil
}

// TotalFrames возвращает общее число фреймов в батче
func (b *Batch) TotalFrames() int {
	return b.Features.Rows
}

// FeatDim возвращает размерность features
func (b *Batch) FeatDim() int {
	return b.Features.Cols
}

// IvectorDim возвращает размерность ivectors (0 если нет)
func (b *Batch) IvectorDim() int {
	if b.Ivectors == nil {
		return 0
	}
	return b.Ivectors.Cols
}
