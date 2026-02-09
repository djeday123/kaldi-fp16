package parser

type Example struct {
	Key         string
	NumInputs   int
	NumOutputs  int
	Inputs      []IoBlock
	Supervision SupervisionBlock
}

// Index represents a single index (n, t, x) in Kaldi
type Index struct {
	N int // sequence number
	T int // time
	X int // extra
}

type IoBlock struct {
	Name    string
	Size    int
	Indexes []Index // Index vector
	Matrix  MatrixInfo
}

type MatrixInfo struct {
	Type     string // CM, CM2, CM3, FM
	Rows     int
	Cols     int
	Min      float32
	Range    float32
	Data     []float32 // полные данные (rows * cols), row-major
	FirstRow []float32 // только первая строка (для быстрого сравнения)
}

// At returns element at (row, col)
func (m *MatrixInfo) At(row, col int) float32 {
	return m.Data[row*m.Cols+col]
}

// Set sets element at (row, col)
func (m *MatrixInfo) Set(row, col int, val float32) {
	m.Data[row*m.Cols+col] = val
}

// Row returns a slice of the row data (zero-copy)
func (m *MatrixInfo) Row(row int) []float32 {
	start := row * m.Cols
	return m.Data[start : start+m.Cols]
}

type SupervisionBlock struct {
	Name         string
	Size         int
	Indexes      []Index // Index vector for output
	Weight       float32
	NumSequences int
	FramesPerSeq int
	LabelDim     int
	End2End      bool
	Fst          *Fst      // FST граф
	DerivWeights []float32 // веса для backprop
}

// // Supervision contains chain supervision data
// type Supervision struct {
// 	Weight       float32
// 	NumSequences int32
// 	FramesPerSeq int32
// 	LabelDim     int32
// 	End2End      bool
// 	Fst          *Fst
// 	DerivWeights []float32
// }

type FileStats struct {
	Path            string
	Total           int
	Valid           int
	Invalid         int
	ZeroWeight      int
	UnusualLabelDim int
	Usable          int
	FrameSizes      map[int]int
}

// SparseVector - разреженный вектор
type SparseVector struct {
	Dim   int
	Pairs []SparseElement
}

// SparseElement - элемент разреженного вектора
type SparseElement struct {
	Index int
	Value float32
}

// SparseMatrix - разреженная матрица (массив SparseVector)
type SparseMatrix struct {
	Rows []SparseVector
}

// FST types for chain supervision

// FstArc represents a single arc in the FST
type FstArc struct {
	Label     int32
	Weight    float32
	NextState int32
}

// FstState represents a state with its outgoing arcs
type FstState struct {
	Arcs  []FstArc
	Final float32 // Final weight (inf = not final)
}

// Fst represents a finite state transducer (acceptor)
type Fst struct {
	Start      int64
	NumStates  int64
	NumArcs    int64
	Properties uint64
	States     []FstState
}
