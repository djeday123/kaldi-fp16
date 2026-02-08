// Package egsreader reads Kaldi NNet3 chain examples (cegs)
// Binary format compatible with Kaldi's nnet3-chain-copy-egs
package egsreader

// NNet3ChainExample represents a single training example
// Contains features, ivectors, and supervision (alignment + FST)
type NNet3ChainExample struct {
	Key string // Utterance ID (e.g., "lbi-1000922-20254112-0000-162")

	// Inputs
	Input   *NNetIo // Main features [frames, feat_dim]
	Ivector *NNetIo // I-vectors [1, ivector_dim] (optional)

	// Supervision for chain training
	Supervision *ChainSupervision
}

// NNetIo represents input/output for nnet3
type NNetIo struct {
	Name    string  // "input" or "ivector"
	Indexes []Index // Frame indices
	Data    *Matrix // Feature data
}

// Index represents (n, t, x) tuple for nnet3
type Index struct {
	N int32 // Minibatch index (usually 0 for single example)
	T int32 // Time index
	X int32 // Extra index (usually 0)
}

// Matrix holds feature data
type Matrix struct {
	Rows int
	Cols int
	Data []float32 // Row-major: Data[row*Cols + col]
}

// NewMatrix creates a new matrix
func NewMatrix(rows, cols int) *Matrix {
	return &Matrix{
		Rows: rows,
		Cols: cols,
		Data: make([]float32, rows*cols),
	}
}

// At returns element at (row, col)
func (m *Matrix) At(row, col int) float32 {
	return m.Data[row*m.Cols+col]
}

// Set sets element at (row, col)
func (m *Matrix) Set(row, col int, val float32) {
	m.Data[row*m.Cols+col] = val
}

// Row returns a slice of the row data
func (m *Matrix) Row(row int) []float32 {
	start := row * m.Cols
	return m.Data[start : start+m.Cols]
}

// ChainSupervision contains alignment and FST for LF-MMI
type ChainSupervision struct {
	Name           string
	Weight         float32
	NumSequences   int32
	FramesPerSeq   int32
	LabelDim       int32 // Number of PDFs
	Fst            *Fst  // Numerator FST (supervision graph)
	E2ESupervision bool  // End-to-end supervision flag
}

// Fst represents a finite state transducer (simplified)
type Fst struct {
	NumStates  int32
	StartState int32
	Arcs       []Arc
	Finals     []FinalState
}

// Arc represents an FST arc
type Arc struct {
	FromState int32
	ToState   int32
	ILabel    int32 // Input label (pdf-id + 1, 0 is epsilon)
	OLabel    int32 // Output label
	Weight    float32
}

// FinalState represents a final state with weight
type FinalState struct {
	State  int32
	Weight float32
}

// CompressedMatrix holds Kaldi's compressed matrix format
type CompressedMatrix struct {
	Format   byte // 'C' for global header, 'M' for per-column
	NumRows  int32
	NumCols  int32
	MinValue float32
	Range    float32
	Data     []byte // Compressed data (uint8 or uint16)
}

// Decompress converts compressed matrix to float32
func (cm *CompressedMatrix) Decompress() *Matrix {
	m := NewMatrix(int(cm.NumRows), int(cm.NumCols))

	// Simple linear decompression: value = min + (data / 255) * range
	for i, b := range cm.Data {
		if i < len(m.Data) {
			m.Data[i] = cm.MinValue + (float32(b)/255.0)*cm.Range
		}
	}

	return m
}

// Batch represents a minibatch of examples for training
type Batch struct {
	Examples []*NNet3ChainExample

	// Merged data for GPU
	Features *Matrix // [total_frames, feat_dim]
	Ivectors *Matrix // [batch_size, ivector_dim]

	// Frame info
	FrameOffsets []int // Start frame for each example
	NumFrames    []int // Number of frames per example
}

// NewBatch creates a batch from examples
func NewBatch(examples []*NNet3ChainExample) *Batch {
	if len(examples) == 0 {
		return nil
	}

	batch := &Batch{
		Examples:     examples,
		FrameOffsets: make([]int, len(examples)),
		NumFrames:    make([]int, len(examples)),
	}

	// Calculate total frames and dimensions
	totalFrames := 0
	featDim := 0
	ivectorDim := 0

	for i, eg := range examples {
		if eg.Input != nil && eg.Input.Data != nil {
			batch.FrameOffsets[i] = totalFrames
			batch.NumFrames[i] = eg.Input.Data.Rows
			totalFrames += eg.Input.Data.Rows
			featDim = eg.Input.Data.Cols
		}
		if eg.Ivector != nil && eg.Ivector.Data != nil {
			ivectorDim = eg.Ivector.Data.Cols
		}
	}

	// Merge features
	if totalFrames > 0 && featDim > 0 {
		batch.Features = NewMatrix(totalFrames, featDim)
		offset := 0
		for _, eg := range examples {
			if eg.Input != nil && eg.Input.Data != nil {
				copy(batch.Features.Data[offset:], eg.Input.Data.Data)
				offset += len(eg.Input.Data.Data)
			}
		}
	}

	// Merge ivectors (one per example)
	if ivectorDim > 0 {
		batch.Ivectors = NewMatrix(len(examples), ivectorDim)
		for i, eg := range examples {
			if eg.Ivector != nil && eg.Ivector.Data != nil {
				copy(batch.Ivectors.Row(i), eg.Ivector.Data.Data)
			}
		}
	}

	return batch
}
