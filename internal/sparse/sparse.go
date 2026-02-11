package sparse

import (
	"fmt"
	"math"
	"sort"

	"kaldi-fp16/internal/parser"
)

// CSR — Compressed Sparse Row format for GPU
// Used for chain forward/backward pass on GPU
//
// For FST with N states and M arcs:
//
//	RowPtr[i]..RowPtr[i+1] = range of arcs leaving state i
//	ColIdx[j] = destination state of arc j
//	Labels[j] = pdf-id of arc j (0 = epsilon)
//	Weights[j] = weight of arc j (log-probability)
type CSR struct {
	NumStates int       // number of states (rows)
	NumArcs   int       // number of arcs (non-zeros)
	RowPtr    []int32   // length = NumStates + 1
	ColIdx    []int32   // length = NumArcs
	Labels    []int32   // length = NumArcs (pdf-id, 1-indexed in Kaldi)
	Weights   []float32 // length = NumArcs (log-space)

	// Final state info
	FinalStates  []int32   // indices of final states
	FinalWeights []float32 // weights of final states

	// Start state
	StartState int32
}

// COO — Coordinate format (useful for batching/merging)
//
// Each arc stored as (row, col, label, weight) tuple
// Easy to concatenate multiple FSTs with state offset
type COO struct {
	NumStates int
	NumArcs   int
	Rows      []int32   // source state
	Cols      []int32   // destination state
	Labels    []int32   // pdf-id
	Weights   []float32 // log-weight

	FinalStates  []int32
	FinalWeights []float32
	StartState   int32
}

// FstToCSR converts parsed FST to CSR format
func FstToCSR(fst *parser.Fst) (*CSR, error) {
	if fst == nil {
		return nil, fmt.Errorf("nil FST")
	}
	if fst.NumStates <= 0 {
		return nil, fmt.Errorf("FST has no states")
	}

	numStates := int(fst.NumStates)
	numArcs := int(fst.NumArcs)

	csr := &CSR{
		NumStates:  numStates,
		NumArcs:    numArcs,
		RowPtr:     make([]int32, numStates+1),
		ColIdx:     make([]int32, 0, numArcs),
		Labels:     make([]int32, 0, numArcs),
		Weights:    make([]float32, 0, numArcs),
		StartState: int32(fst.Start),
	}

	// Build row pointers and arc data
	arcIdx := int32(0)
	for i := 0; i < numStates; i++ {
		csr.RowPtr[i] = arcIdx
		state := &fst.States[i]

		for _, arc := range state.Arcs {
			csr.ColIdx = append(csr.ColIdx, arc.NextState)
			csr.Labels = append(csr.Labels, arc.Label)
			csr.Weights = append(csr.Weights, -arc.Weight) // negate: tropical → log-prob
			arcIdx++
		}

		// Collect final states
		if !math.IsInf(float64(state.Final), 1) {
			csr.FinalStates = append(csr.FinalStates, int32(i))
			csr.FinalWeights = append(csr.FinalWeights, -state.Final) // negate: tropical → log-prob
		}
	}
	csr.RowPtr[numStates] = arcIdx

	// Verify
	if int(arcIdx) != numArcs {
		return nil, fmt.Errorf("arc count mismatch: counted %d, expected %d", arcIdx, numArcs)
	}

	return csr, nil
}

// FstToCOO converts parsed FST to COO format
func FstToCOO(fst *parser.Fst) (*COO, error) {
	if fst == nil {
		return nil, fmt.Errorf("nil FST")
	}
	if fst.NumStates <= 0 {
		return nil, fmt.Errorf("FST has no states")
	}

	numStates := int(fst.NumStates)
	numArcs := int(fst.NumArcs)

	coo := &COO{
		NumStates:  numStates,
		NumArcs:    numArcs,
		Rows:       make([]int32, 0, numArcs),
		Cols:       make([]int32, 0, numArcs),
		Labels:     make([]int32, 0, numArcs),
		Weights:    make([]float32, 0, numArcs),
		StartState: int32(fst.Start),
	}

	for i := 0; i < numStates; i++ {
		state := &fst.States[i]
		for _, arc := range state.Arcs {
			coo.Rows = append(coo.Rows, int32(i))
			coo.Cols = append(coo.Cols, arc.NextState)
			coo.Labels = append(coo.Labels, arc.Label)
			coo.Weights = append(coo.Weights, -arc.Weight) // negate: tropical → log-prob
		}

		if !math.IsInf(float64(state.Final), 1) {
			coo.FinalStates = append(coo.FinalStates, int32(i))
			coo.FinalWeights = append(coo.FinalWeights, -state.Final) // negate: tropical → log-prob
		}
	}

	return coo, nil
}

// CSRToCOO converts CSR to COO
func CSRToCOO(csr *CSR) *COO {
	coo := &COO{
		NumStates:    csr.NumStates,
		NumArcs:      csr.NumArcs,
		Rows:         make([]int32, csr.NumArcs),
		Cols:         make([]int32, csr.NumArcs),
		Labels:       make([]int32, csr.NumArcs),
		Weights:      make([]float32, csr.NumArcs),
		FinalStates:  csr.FinalStates,
		FinalWeights: csr.FinalWeights,
		StartState:   csr.StartState,
	}

	copy(coo.Cols, csr.ColIdx)
	copy(coo.Labels, csr.Labels)
	copy(coo.Weights, csr.Weights)

	// Expand row pointers to row indices
	for i := 0; i < csr.NumStates; i++ {
		for j := csr.RowPtr[i]; j < csr.RowPtr[i+1]; j++ {
			coo.Rows[j] = int32(i)
		}
	}

	return coo
}

// COOToCSR converts COO to CSR (sorts by row if needed)
func COOToCSR(coo *COO) *CSR {
	csr := &CSR{
		NumStates:    coo.NumStates,
		NumArcs:      coo.NumArcs,
		RowPtr:       make([]int32, coo.NumStates+1),
		ColIdx:       make([]int32, coo.NumArcs),
		Labels:       make([]int32, coo.NumArcs),
		Weights:      make([]float32, coo.NumArcs),
		FinalStates:  coo.FinalStates,
		FinalWeights: coo.FinalWeights,
		StartState:   coo.StartState,
	}

	// Sort arcs by row (stable sort preserves column order within row)
	indices := make([]int, coo.NumArcs)
	for i := range indices {
		indices[i] = i
	}
	sort.SliceStable(indices, func(a, b int) bool {
		return coo.Rows[indices[a]] < coo.Rows[indices[b]]
	})

	// Fill sorted data
	for j, idx := range indices {
		csr.ColIdx[j] = coo.Cols[idx]
		csr.Labels[j] = coo.Labels[idx]
		csr.Weights[j] = coo.Weights[idx]
	}

	// Build row pointers
	for _, idx := range indices {
		csr.RowPtr[coo.Rows[idx]+1]++
	}
	// Cumulative sum
	for i := 1; i <= coo.NumStates; i++ {
		csr.RowPtr[i] += csr.RowPtr[i-1]
	}

	return csr
}

// MergeCOO merges multiple COO FSTs into one (for batching)
// Each FST gets a state offset so states don't collide
// Returns merged COO + per-example state offsets
func MergeCOO(fsts []*COO) (*COO, []int32, error) {
	if len(fsts) == 0 {
		return nil, nil, fmt.Errorf("empty FST list")
	}

	// Calculate total sizes and offsets
	totalStates := 0
	totalArcs := 0
	offsets := make([]int32, len(fsts))

	for i, f := range fsts {
		offsets[i] = int32(totalStates)
		totalStates += f.NumStates
		totalArcs += f.NumArcs
	}

	merged := &COO{
		NumStates: totalStates,
		NumArcs:   totalArcs,
		Rows:      make([]int32, 0, totalArcs),
		Cols:      make([]int32, 0, totalArcs),
		Labels:    make([]int32, 0, totalArcs),
		Weights:   make([]float32, 0, totalArcs),
	}

	for i, f := range fsts {
		offset := offsets[i]

		// Add arcs with offset
		for j := 0; j < f.NumArcs; j++ {
			merged.Rows = append(merged.Rows, f.Rows[j]+offset)
			merged.Cols = append(merged.Cols, f.Cols[j]+offset)
			merged.Labels = append(merged.Labels, f.Labels[j])
			merged.Weights = append(merged.Weights, f.Weights[j])
		}

		// Add final states with offset
		for j := range f.FinalStates {
			merged.FinalStates = append(merged.FinalStates, f.FinalStates[j]+offset)
			merged.FinalWeights = append(merged.FinalWeights, f.FinalWeights[j])
		}
	}

	return merged, offsets, nil
}

// LabelDim returns the maximum label + 1 (number of PDFs)
func (csr *CSR) LabelDim() int {
	maxLabel := int32(0)
	for _, l := range csr.Labels {
		if l > maxLabel {
			maxLabel = l
		}
	}
	return int(maxLabel + 1)
}

// LabelDim returns the maximum label + 1
func (coo *COO) LabelDim() int {
	maxLabel := int32(0)
	for _, l := range coo.Labels {
		if l > maxLabel {
			maxLabel = l
		}
	}
	return int(maxLabel + 1)
}

// Validate checks CSR consistency
func (csr *CSR) Validate() error {
	if len(csr.RowPtr) != csr.NumStates+1 {
		return fmt.Errorf("RowPtr length %d != NumStates+1 (%d)", len(csr.RowPtr), csr.NumStates+1)
	}
	if len(csr.ColIdx) != csr.NumArcs {
		return fmt.Errorf("ColIdx length %d != NumArcs %d", len(csr.ColIdx), csr.NumArcs)
	}
	if len(csr.Labels) != csr.NumArcs {
		return fmt.Errorf("Labels length %d != NumArcs %d", len(csr.Labels), csr.NumArcs)
	}
	if len(csr.Weights) != csr.NumArcs {
		return fmt.Errorf("Weights length %d != NumArcs %d", len(csr.Weights), csr.NumArcs)
	}

	// Check RowPtr monotonicity
	for i := 0; i < csr.NumStates; i++ {
		if csr.RowPtr[i] > csr.RowPtr[i+1] {
			return fmt.Errorf("RowPtr not monotonic at state %d: %d > %d", i, csr.RowPtr[i], csr.RowPtr[i+1])
		}
	}
	if csr.RowPtr[csr.NumStates] != int32(csr.NumArcs) {
		return fmt.Errorf("RowPtr[last] = %d != NumArcs %d", csr.RowPtr[csr.NumStates], csr.NumArcs)
	}

	// Check col indices in range
	for j := 0; j < csr.NumArcs; j++ {
		if csr.ColIdx[j] < 0 || int(csr.ColIdx[j]) >= csr.NumStates {
			return fmt.Errorf("ColIdx[%d] = %d out of range [0, %d)", j, csr.ColIdx[j], csr.NumStates)
		}
	}

	return nil
}
