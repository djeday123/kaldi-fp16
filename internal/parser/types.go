package parser

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
	Type     string // CM, CM2, CM3, FM
	Rows     int
	Cols     int
	Min      float32
	Range    float32
	Data     []float32 // полные данные (rows * cols)
	FirstRow []float32 // только первая строка (для быстрого сравнения)
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
