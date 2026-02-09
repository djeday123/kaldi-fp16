package loader

import (
	"fmt"
	"log"
	"time"

	"kaldi-fp16/internal/batch"
	"kaldi-fp16/internal/parser"
	"kaldi-fp16/internal/sparse"
)

// TrainingBatch — полный батч готовый для GPU
// Содержит объединённые features, ivectors, и FST в sparse формате
type TrainingBatch struct {
	// Данные из batch.Batch
	Features *batch.MergedMatrix // [total_frames, feat_dim]
	Ivectors *batch.MergedMatrix // [batch_size, ivector_dim]

	// FST supervision в sparse формате
	FstCSR       *sparse.CSR // объединённый FST для всего батча
	StateOffsets []int32     // смещение состояний каждого примера в merged FST

	// Метаданные
	FrameOffsets []int // начало фреймов каждого примера
	NumFrames    []int // количество фреймов каждого примера
	BatchSize    int   // количество примеров
	LabelDim     int   // число PDF (макс label + 1)

	// Supervision info (одинаковые для всех примеров в батче)
	FramesPerSeq int
	Weight       float32

	// Ключи (для отладки)
	Keys []string
}

// DataLoader загружает данные из ark файлов и формирует TrainingBatch
type DataLoader struct {
	iterator  *EgsIterator
	batchSize int
	dropLast  bool // отбросить последний неполный батч
	verbose   bool

	// Статистика
	batchesServed int
	examplesRead  int
	totalTime     time.Duration
}

// DataLoaderConfig — настройки DataLoader
type DataLoaderConfig struct {
	Pattern   string // glob паттерн ark файлов
	BatchSize int
	Shuffle   bool // перемешать порядок файлов
	DropLast  bool // отбросить последний неполный батч
	Verbose   bool // печатать статистику
}

// NewDataLoader создаёт DataLoader
func NewDataLoader(cfg DataLoaderConfig) (*DataLoader, error) {
	if cfg.BatchSize <= 0 {
		return nil, fmt.Errorf("batch size must be > 0, got %d", cfg.BatchSize)
	}

	it, err := NewEgsIterator(cfg.Pattern, cfg.Shuffle)
	if err != nil {
		return nil, fmt.Errorf("failed to create iterator: %w", err)
	}

	return &DataLoader{
		iterator:  it,
		batchSize: cfg.BatchSize,
		dropLast:  cfg.DropLast,
		verbose:   cfg.Verbose,
	}, nil
}

// NewDataLoaderFromPaths создаёт DataLoader из списка путей
func NewDataLoaderFromPaths(paths []string, batchSize int, shuffle bool) (*DataLoader, error) {
	if batchSize <= 0 {
		return nil, fmt.Errorf("batch size must be > 0, got %d", batchSize)
	}

	it, err := NewEgsIteratorFromPaths(paths, shuffle)
	if err != nil {
		return nil, fmt.Errorf("failed to create iterator: %w", err)
	}

	return &DataLoader{
		iterator:  it,
		batchSize: batchSize,
	}, nil
}

// NextBatch возвращает следующий TrainingBatch
// Возвращает nil, nil когда данные закончились
func (dl *DataLoader) NextBatch() (*TrainingBatch, error) {
	start := time.Now()

	// 1. Собираем batchSize примеров
	examples := make([]*parser.Example, 0, dl.batchSize)
	for len(examples) < dl.batchSize {
		ex, err := dl.iterator.Next()
		if err != nil {
			if dl.verbose {
				log.Printf("[DataLoader] skip error: %v", err)
			}
			continue
		}
		if ex == nil {
			break // EOF
		}

		// Проверяем что пример пригоден
		if err := validateExample(ex); err != nil {
			if dl.verbose {
				log.Printf("[DataLoader] skip %s: %v", ex.Key, err)
			}
			continue
		}

		examples = append(examples, ex)
		dl.examplesRead++
	}

	// Нет данных
	if len(examples) == 0 {
		return nil, nil
	}

	// Неполный батч
	if len(examples) < dl.batchSize && dl.dropLast {
		return nil, nil
	}

	// 2. Собираем batch (features + ivectors)
	b, err := batch.NewBatch(examples)
	if err != nil {
		return nil, fmt.Errorf("batch assembly failed: %w", err)
	}

	// 3. Конвертируем FST → sparse и мержим
	fstCSR, stateOffsets, err := mergeFSTs(examples)
	if err != nil {
		return nil, fmt.Errorf("FST merge failed: %w", err)
	}

	// 4. Собираем TrainingBatch
	tb := &TrainingBatch{
		Features:     b.Features,
		Ivectors:     b.Ivectors,
		FstCSR:       fstCSR,
		StateOffsets: stateOffsets,
		FrameOffsets: b.FrameOffsets,
		NumFrames:    b.NumFrames,
		BatchSize:    len(examples),
		LabelDim:     fstCSR.LabelDim(),
		FramesPerSeq: examples[0].Supervision.FramesPerSeq,
		Weight:       examples[0].Supervision.Weight,
		Keys:         make([]string, len(examples)),
	}

	for i, ex := range examples {
		tb.Keys[i] = ex.Key
	}

	dl.batchesServed++
	dl.totalTime += time.Since(start)

	if dl.verbose && dl.batchesServed%100 == 0 {
		log.Printf("[DataLoader] batch %d: %d examples, %d frames, %d states, %d arcs (%.1f ms)",
			dl.batchesServed, tb.BatchSize, tb.Features.Rows,
			fstCSR.NumStates, fstCSR.NumArcs,
			float64(time.Since(start).Microseconds())/1000.0)
	}

	return tb, nil
}

// Reset сбрасывает DataLoader на начало (для новой эпохи)
func (dl *DataLoader) Reset() {
	dl.iterator.Reset()
	dl.batchesServed = 0
	dl.examplesRead = 0
	dl.totalTime = 0
}

// Close закрывает DataLoader
func (dl *DataLoader) Close() error {
	return dl.iterator.Close()
}

// Stats возвращает статистику
func (dl *DataLoader) Stats() DataLoaderStats {
	avgTime := time.Duration(0)
	if dl.batchesServed > 0 {
		avgTime = dl.totalTime / time.Duration(dl.batchesServed)
	}
	return DataLoaderStats{
		BatchesServed: dl.batchesServed,
		ExamplesRead:  dl.examplesRead,
		TotalTime:     dl.totalTime,
		AvgBatchTime:  avgTime,
	}
}

// DataLoaderStats — статистика работы DataLoader
type DataLoaderStats struct {
	BatchesServed int
	ExamplesRead  int
	TotalTime     time.Duration
	AvgBatchTime  time.Duration
}

func (s DataLoaderStats) String() string {
	return fmt.Sprintf("batches=%d examples=%d total=%.1fs avg=%.1fms",
		s.BatchesServed, s.ExamplesRead,
		s.TotalTime.Seconds(),
		float64(s.AvgBatchTime.Microseconds())/1000.0)
}

// validateExample проверяет что пример пригоден для тренировки
func validateExample(ex *parser.Example) error {
	if len(ex.Inputs) < 1 {
		return fmt.Errorf("no inputs")
	}
	if ex.Inputs[0].Name != "input" {
		return fmt.Errorf("first input is '%s', expected 'input'", ex.Inputs[0].Name)
	}
	if ex.Inputs[0].Matrix.Rows <= 0 || ex.Inputs[0].Matrix.Cols <= 0 {
		return fmt.Errorf("invalid input matrix: %dx%d", ex.Inputs[0].Matrix.Rows, ex.Inputs[0].Matrix.Cols)
	}
	if ex.Inputs[0].Matrix.Data == nil {
		return fmt.Errorf("input matrix data is nil")
	}
	if ex.Supervision.Fst == nil {
		return fmt.Errorf("supervision FST is nil")
	}
	if ex.Supervision.Weight <= 0 {
		return fmt.Errorf("zero or negative weight: %f", ex.Supervision.Weight)
	}
	return nil
}

// mergeFSTs конвертирует и мержит FST из всех примеров
func mergeFSTs(examples []*parser.Example) (*sparse.CSR, []int32, error) {
	coos := make([]*sparse.COO, len(examples))

	for i, ex := range examples {
		coo, err := sparse.FstToCOO(ex.Supervision.Fst)
		if err != nil {
			return nil, nil, fmt.Errorf("example %d (%s): %w", i, ex.Key, err)
		}
		coos[i] = coo
	}

	merged, offsets, err := sparse.MergeCOO(coos)
	if err != nil {
		return nil, nil, err
	}

	csr := sparse.COOToCSR(merged)

	if err := csr.Validate(); err != nil {
		return nil, nil, fmt.Errorf("merged CSR validation failed: %w", err)
	}

	return csr, offsets, nil
}
