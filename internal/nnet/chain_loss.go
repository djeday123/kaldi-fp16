package nnet

/*
#cgo CFLAGS: -I${SRCDIR}/../../cpp/include
#cgo LDFLAGS: -L${SRCDIR}/../../cpp/build -lkaldi_fp16 -lkaldi_fp16_cgo -lcublas -lcudart -lstdc++

#include "chain.h"
#include "bridge.h"
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"unsafe"

	"kaldi-fp16/internal/gpu"
	"kaldi-fp16/internal/sparse"
)

// ============================================================
// ChainLoss — LF-MMI loss computation on GPU
// ============================================================

// ChainLossResult holds the result of chain loss computation
type ChainLossResult struct {
	NumLogprob float32 // numerator log-probability
	DenLogprob float32 // denominator log-probability
	Loss       float32 // -(num - den)
}

// ChainFstGPU holds an FST on GPU for chain computation
type ChainFstGPU struct {
	cFst       C.ChainFstGPU
	rowPtrBuf  unsafe.Pointer
	colIdxBuf  unsafe.Pointer
	labelsBuf  unsafe.Pointer
	weightsBuf unsafe.Pointer
	finalSBuf  unsafe.Pointer
	finalWBuf  unsafe.Pointer
}

// NewChainFstGPU uploads a CSR FST to GPU
func NewChainFstGPU(csr *sparse.CSR) (*ChainFstGPU, error) {
	f := &ChainFstGPU{}

	var err error

	f.rowPtrBuf, err = uploadInt32(csr.RowPtr)
	if err != nil {
		return nil, fmt.Errorf("row_ptr: %w", err)
	}

	f.colIdxBuf, err = uploadInt32(csr.ColIdx)
	if err != nil {
		f.Free()
		return nil, fmt.Errorf("col_idx: %w", err)
	}

	f.labelsBuf, err = uploadInt32(csr.Labels)
	if err != nil {
		f.Free()
		return nil, fmt.Errorf("labels: %w", err)
	}

	f.weightsBuf, err = uploadFloat32(csr.Weights)
	if err != nil {
		f.Free()
		return nil, fmt.Errorf("weights: %w", err)
	}

	if len(csr.FinalStates) > 0 {
		f.finalSBuf, err = uploadInt32(csr.FinalStates)
		if err != nil {
			f.Free()
			return nil, fmt.Errorf("final_states: %w", err)
		}
		f.finalWBuf, err = uploadFloat32(csr.FinalWeights)
		if err != nil {
			f.Free()
			return nil, fmt.Errorf("final_weights: %w", err)
		}
	}

	f.cFst.row_ptr = (*C.int32_t)(f.rowPtrBuf)
	f.cFst.col_idx = (*C.int32_t)(f.colIdxBuf)
	f.cFst.labels = (*C.int32_t)(f.labelsBuf)
	f.cFst.weights = (*C.float)(f.weightsBuf)
	f.cFst.final_states = (*C.int32_t)(f.finalSBuf)
	f.cFst.final_weights = (*C.float)(f.finalWBuf)
	f.cFst.num_states = C.int(csr.NumStates)
	f.cFst.num_arcs = C.int(csr.NumArcs)
	f.cFst.num_final = C.int(len(csr.FinalStates))
	f.cFst.start_state = C.int(csr.StartState)

	return f, nil
}

// NewChainFstGPUFromBatch creates a ChainFstGPU reusing GPU pointers
func NewChainFstGPUFromBatch(gb *gpu.GPUBatch, csr *sparse.CSR) (*ChainFstGPU, error) {
	f := &ChainFstGPU{}
	f.cFst.row_ptr = (*C.int32_t)(gb.CSRRowPtr())
	f.cFst.col_idx = (*C.int32_t)(gb.CSRColIdx())
	f.cFst.labels = (*C.int32_t)(gb.CSRLabels())
	f.cFst.weights = (*C.float)(gb.CSRWeights())
	f.cFst.num_states = C.int(gb.NumStates)
	f.cFst.num_arcs = C.int(gb.NumArcs)
	f.cFst.start_state = C.int(csr.StartState)

	var err error
	if len(csr.FinalStates) > 0 {
		f.finalSBuf, err = uploadInt32(csr.FinalStates)
		if err != nil {
			return nil, fmt.Errorf("final_states: %w", err)
		}
		f.finalWBuf, err = uploadFloat32(csr.FinalWeights)
		if err != nil {
			f.Free()
			return nil, fmt.Errorf("final_weights: %w", err)
		}
		f.cFst.final_states = (*C.int32_t)(f.finalSBuf)
		f.cFst.final_weights = (*C.float)(f.finalWBuf)
		f.cFst.num_final = C.int(len(csr.FinalStates))
	}

	return f, nil
}

// Free releases GPU memory
func (f *ChainFstGPU) Free() {
	if f.rowPtrBuf != nil {
		C.bridge_gpu_free(f.rowPtrBuf)
	}
	if f.colIdxBuf != nil {
		C.bridge_gpu_free(f.colIdxBuf)
	}
	if f.labelsBuf != nil {
		C.bridge_gpu_free(f.labelsBuf)
	}
	if f.weightsBuf != nil {
		C.bridge_gpu_free(f.weightsBuf)
	}
	if f.finalSBuf != nil {
		C.bridge_gpu_free(f.finalSBuf)
	}
	if f.finalWBuf != nil {
		C.bridge_gpu_free(f.finalWBuf)
	}
	*f = ChainFstGPU{}
}

// GetCSROnGPU extracts raw GPU pointers from ChainFstGPU into CSROnGPU
// for use with chain_num_forward_backward / chain_num_forward_backward_det
func GetCSROnGPU(f *ChainFstGPU) *CSROnGPU {
	return &CSROnGPU{
		RowPtr:       unsafe.Pointer(f.cFst.row_ptr),
		ColIdx:       unsafe.Pointer(f.cFst.col_idx),
		Weights:      unsafe.Pointer(f.cFst.weights),
		PdfIds:       unsafe.Pointer(f.cFst.labels),
		FinalStates:  unsafe.Pointer(f.cFst.final_states),
		FinalWeights: unsafe.Pointer(f.cFst.final_weights),
		NumStates:    int(f.cFst.num_states),
		NumArcs:      int(f.cFst.num_arcs),
		NumFinal:     int(f.cFst.num_final),
	}
}

// ============================================================
// ComputeChainLoss — single sequence
// ============================================================

func ComputeChainLoss(
	nnetOutput *gpu.Tensor,
	numFst *ChainFstGPU,
	denFst *ChainFstGPU,
	grad *gpu.Tensor,
) (*ChainLossResult, error) {

	T := nnetOutput.Rows
	P := nnetOutput.Cols

	var gradPtr unsafe.Pointer
	if grad != nil {
		gradPtr = grad.Ptr
	}

	var cResult C.ChainLossResult

	ret := C.chain_compute_loss(
		nnetOutput.Ptr,
		&numFst.cFst,
		&denFst.cFst,
		C.int(T),
		C.int(P),
		gradPtr,
		&cResult,
	)

	if ret != 0 {
		errStr := C.chain_last_error()
		if errStr != nil {
			defer C.chain_clear_error()
			return nil, fmt.Errorf("chain loss: %s", C.GoString(errStr))
		}
		return nil, fmt.Errorf("chain loss failed")
	}

	return &ChainLossResult{
		NumLogprob: float32(cResult.num_logprob),
		DenLogprob: float32(cResult.den_logprob),
		Loss:       float32(cResult.loss),
	}, nil
}

// ============================================================
// ComputeChainLossBatch — per-sequence chain loss with subsampling
//
// framesPerSeq: []int — per-sequence number of output frames after subsampling
// ============================================================

func ComputeChainLossBatch(
	nnetOutput *gpu.Tensor,
	perSeqCSRs []*sparse.CSR,
	denFst *ChainFstGPU,
	frameOffsets []int,
	numFrames []int,
	framesPerSeq []int,
	subsamplingFactor int,
	leftContext int,
	grad *gpu.Tensor,
) (*ChainLossResult, error) {

	batchSize := len(perSeqCSRs)
	P := nnetOutput.Cols

	var totalNumLogprob, totalDenLogprob, totalLoss float32
	gradOffset := 0

	for i := 0; i < batchSize; i++ {
		T := numFrames[i]
		offset := frameOffsets[i]
		fps := framesPerSeq[i]

		// How many input rows to produce exactly fps output rows
		effectiveInRows := leftContext + fps*subsamplingFactor
		if effectiveInRows > T {
			effectiveInRows = T
		}

		// View into nnet output for this sequence
		seqView := nnetOutput.View(offset, effectiveInRows, P)

		// Subsample: every stride-th frame starting at leftContext
		subsampled, err := gpu.SubsampleRows(seqView, subsamplingFactor, leftContext)
		if err != nil {
			return nil, fmt.Errorf("seq %d: subsample: %w", i, err)
		}

		fmt.Printf("  seq %d: input=%d, effective=%d, subsampled=%d x %d, fps=%d, fst_states=%d\n",
			i, T, effectiveInRows, subsampled.Rows, subsampled.Cols, fps, perSeqCSRs[i].NumStates)

		numFst, err := NewChainFstGPU(perSeqCSRs[i])
		if err != nil {
			subsampled.Free()
			return nil, fmt.Errorf("seq %d: upload num FST: %w", i, err)
		}

		var seqGrad *gpu.Tensor
		if grad != nil {
			seqGrad = grad.View(gradOffset, fps, P)
			gradOffset += fps
		}

		result, err := ComputeChainLoss(subsampled, numFst, denFst, seqGrad)
		numFst.Free()
		subsampled.Free()
		if err != nil {
			return nil, fmt.Errorf("seq %d: chain loss: %w", i, err)
		}

		fmt.Printf("  seq %d: num_lp=%.4f den_lp=%.4f loss=%.4f\n",
			i, result.NumLogprob, result.DenLogprob, result.Loss)

		totalNumLogprob += result.NumLogprob
		totalDenLogprob += result.DenLogprob
		totalLoss += result.Loss
	}

	return &ChainLossResult{
		NumLogprob: totalNumLogprob / float32(batchSize),
		DenLogprob: totalDenLogprob / float32(batchSize),
		Loss:       totalLoss / float32(batchSize),
	}, nil
}

// ============================================================
// ForwardBackward — just the forward-backward (for debugging)
// ============================================================

func ForwardBackward(
	nnetOutput *gpu.Tensor,
	fst *ChainFstGPU,
) (totalLogprob float32, err error) {

	T := nnetOutput.Rows
	P := nnetOutput.Cols
	S := int(fst.cFst.num_states)

	wsBytes := C.chain_workspace_bytes(C.int(T), C.int(S))
	halfWs := wsBytes / 2

	alphaPtr := C.bridge_gpu_malloc(C.size_t(halfWs))
	if alphaPtr == nil {
		return 0, fmt.Errorf("alloc alpha workspace")
	}
	defer C.bridge_gpu_free(alphaPtr)

	betaPtr := C.bridge_gpu_malloc(C.size_t(halfWs))
	if betaPtr == nil {
		return 0, fmt.Errorf("alloc beta workspace")
	}
	defer C.bridge_gpu_free(betaPtr)

	var logprobC C.float
	ret := C.chain_forward_backward(
		nnetOutput.Ptr,
		&fst.cFst,
		C.int(T), C.int(P),
		(*C.float)(alphaPtr),
		(*C.float)(betaPtr),
		&logprobC,
	)

	if ret != 0 {
		errStr := C.chain_last_error()
		if errStr != nil {
			defer C.chain_clear_error()
			return 0, fmt.Errorf("forward-backward: %s", C.GoString(errStr))
		}
		return 0, fmt.Errorf("forward-backward failed")
	}

	return float32(logprobC), nil
}

// ============================================================
// Helper: upload data to GPU
// ============================================================

func uploadInt32(data []int32) (unsafe.Pointer, error) {
	if len(data) == 0 {
		return nil, nil
	}
	bytes := C.size_t(len(data) * 4)
	ptr := C.bridge_gpu_malloc(bytes)
	if ptr == nil {
		return nil, fmt.Errorf("gpu malloc int32[%d]", len(data))
	}
	ret := C.bridge_transfer_int32(ptr, (*C.int32_t)(unsafe.Pointer(&data[0])), C.size_t(len(data)))
	if ret != 0 {
		C.bridge_gpu_free(ptr)
		return nil, fmt.Errorf("transfer int32")
	}
	return ptr, nil
}

func uploadFloat32(data []float32) (unsafe.Pointer, error) {
	if len(data) == 0 {
		return nil, nil
	}
	bytes := C.size_t(len(data) * 4)
	ptr := C.bridge_gpu_malloc(bytes)
	if ptr == nil {
		return nil, fmt.Errorf("gpu malloc float32[%d]", len(data))
	}
	ret := C.bridge_transfer_float32(ptr, (*C.float)(unsafe.Pointer(&data[0])), C.size_t(len(data)))
	if ret != 0 {
		C.bridge_gpu_free(ptr)
		return nil, fmt.Errorf("transfer float32")
	}
	return ptr, nil
}
