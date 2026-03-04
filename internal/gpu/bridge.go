package gpu

/*
#cgo CFLAGS: -I${SRCDIR}/../../cpp/include
#cgo LDFLAGS: -L${SRCDIR}/../../cpp/build -lkaldi_fp16 -lkaldi_fp16_cgo -L/usr/local/cuda-12.8/lib64 -lcublas -lcudart -lstdc++

#include "bridge.h"
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"unsafe"

	"kaldi-fp16/internal/fp16"
	"kaldi-fp16/internal/loader"
)

// ============================================================
// Error handling
// ============================================================

func lastError() error {
	s := C.bridge_last_error()
	if s == nil {
		return nil
	}
	defer C.bridge_clear_error()
	return fmt.Errorf("%s", C.GoString(s))
}

func checkErr(ret C.int) error {
	if ret != 0 {
		return lastError()
	}
	return nil
}

// ============================================================
// GPU Device
// ============================================================

// Init sets the active GPU device
func Init(deviceID int) error {
	return checkErr(C.bridge_gpu_init(C.int(deviceID)))
}

// MemoryInfo returns free and total GPU memory in bytes
func MemoryInfo() (free, total uint64, err error) {
	var f, t C.size_t
	if err := checkErr(C.bridge_gpu_get_free_memory(&f, &t)); err != nil {
		return 0, 0, err
	}
	return uint64(f), uint64(t), nil
}

// Sync waits for all GPU operations to complete
func Sync() error {
	return checkErr(C.bridge_gpu_sync())
}

// ============================================================
// GPUBatch — training batch on GPU
// ============================================================

// GPUBatch holds all batch data on GPU in a single allocation
type GPUBatch struct {
	ptrs C.GPUBatchPtrs

	// Dimensions (for compute kernels)
	TotalFrames int
	FeatDim     int
	BatchSize   int
	IvecDim     int
	NumStates   int
	NumArcs     int
	LabelDim    int

	// Supervision metadata
	FrameOffsets []int
	NumFrames    []int
	FramesPerSeq []int
	Weight       float32
	StateOffsets []int32
}

// Features returns device pointer to FP16 features [total_frames × feat_dim]
func (b *GPUBatch) Features() unsafe.Pointer { return b.ptrs.d_features }

// Ivectors returns device pointer to FP16 ivectors [batch_size × ivec_dim]
func (b *GPUBatch) Ivectors() unsafe.Pointer { return b.ptrs.d_ivectors }

// CSRRowPtr returns device pointer to CSR row_ptr [num_states + 1]
func (b *GPUBatch) CSRRowPtr() unsafe.Pointer { return b.ptrs.d_csr_row_ptr }

// CSRColIdx returns device pointer to CSR col_idx [num_arcs]
func (b *GPUBatch) CSRColIdx() unsafe.Pointer { return b.ptrs.d_csr_col_idx }

// CSRLabels returns device pointer to CSR labels [num_arcs]
func (b *GPUBatch) CSRLabels() unsafe.Pointer { return b.ptrs.d_csr_labels }

// CSRWeights returns device pointer to CSR weights [num_arcs]
func (b *GPUBatch) CSRWeights() unsafe.Pointer { return b.ptrs.d_csr_weights }

// TotalBytes returns GPU memory used by this batch
func (b *GPUBatch) TotalBytes() uint64 { return uint64(b.ptrs.total_bytes) }

// Free releases GPU memory
func (b *GPUBatch) Free() {
	C.bridge_batch_free(&b.ptrs)
}

// ============================================================
// TransferBatch — main entry point
// DataLoader TrainingBatch → GPUBatch
// ============================================================

// TransferBatch converts a CPU TrainingBatch to GPU:
//  1. FP32 features/ivectors → FP16 on CPU
//  2. Pack everything into one host buffer
//  3. Single cudaMalloc + single cudaMemcpy
func TransferBatch(tb *loader.TrainingBatch) (*GPUBatch, error) {
	if tb == nil {
		return nil, fmt.Errorf("nil TrainingBatch")
	}

	totalFrames := tb.Features.Rows
	featDim := tb.Features.Cols
	batchSize := tb.BatchSize

	ivecDim := 0
	if tb.Ivectors != nil {
		ivecDim = tb.Ivectors.Cols
	}

	numStates := tb.FstCSR.NumStates
	numArcs := tb.FstCSR.NumArcs

	// 1. Convert features and ivectors to FP16 on CPU
	featFP16 := fp16.ConvertFloat32ToFloat16(tb.Features.Data)

	var ivecFP16 []uint16
	if tb.Ivectors != nil {
		ivecFP16 = fp16.ConvertFloat32ToFloat16(tb.Ivectors.Data)
	} else {
		ivecFP16 = make([]uint16, 0)
	}

	// 2. Allocate combined GPU buffer
	gb := &GPUBatch{
		TotalFrames:  totalFrames,
		FeatDim:      featDim,
		BatchSize:    batchSize,
		IvecDim:      ivecDim,
		NumStates:    numStates,
		NumArcs:      numArcs,
		LabelDim:     tb.LabelDim,
		FrameOffsets: tb.FrameOffsets,
		NumFrames:    tb.NumFrames,
		FramesPerSeq: tb.FramesPerSeq,
		Weight:       tb.Weight,
		StateOffsets: tb.StateOffsets,
	}

	ret := C.bridge_batch_alloc(
		C.int(totalFrames), C.int(featDim),
		C.int(batchSize), C.int(ivecDim),
		C.int(numStates), C.int(numArcs),
		&gb.ptrs,
	)
	if ret != 0 {
		return nil, fmt.Errorf("GPU alloc failed: %w", lastError())
	}

	// 3. Pack into host buffer and transfer
	//    Same layout as GPUBatchPtrs: [feat | ivec | rowptr | colidx | labels | weights]
	//    Each section aligned to 256 bytes
	totalBytes := gb.ptrs.total_bytes
	hostBuf := make([]byte, totalBytes)

	offset := uint64(0)

	// Features FP16
	copyU16(hostBuf, offset, featFP16)
	offset += uint64(gb.ptrs.features_bytes)

	// Ivectors FP16
	if len(ivecFP16) > 0 {
		copyU16(hostBuf, offset, ivecFP16)
	}
	offset += uint64(gb.ptrs.ivectors_bytes)

	// CSR row_ptr
	copyI32(hostBuf, offset, tb.FstCSR.RowPtr)
	offset += uint64(gb.ptrs.csr_rowptr_bytes)

	// CSR col_idx
	copyI32(hostBuf, offset, tb.FstCSR.ColIdx)
	offset += uint64(gb.ptrs.csr_colidx_bytes)

	// CSR labels
	copyI32(hostBuf, offset, tb.FstCSR.Labels)
	offset += uint64(gb.ptrs.csr_labels_bytes)

	// CSR weights
	copyF32(hostBuf, offset, tb.FstCSR.Weights)

	// 4. Single cudaMemcpy
	ret = C.bridge_batch_transfer(
		&gb.ptrs,
		unsafe.Pointer(&hostBuf[0]),
		C.size_t(totalBytes),
	)
	if ret != 0 {
		gb.Free()
		return nil, fmt.Errorf("GPU transfer failed: %w", lastError())
	}

	return gb, nil
}

// ============================================================
// TransferBatchPinned — same as TransferBatch but uses
// pinned (page-locked) memory for ~2x faster PCIe transfer
// ============================================================

// PinnedBuffer is a reusable pinned host buffer
type PinnedBuffer struct {
	ptr  unsafe.Pointer
	size uint64
}

// NewPinnedBuffer allocates pinned memory
func NewPinnedBuffer(size uint64) (*PinnedBuffer, error) {
	ptr := C.bridge_host_alloc(C.size_t(size))
	if ptr == nil {
		return nil, fmt.Errorf("pinned alloc failed: %w", lastError())
	}
	return &PinnedBuffer{ptr: ptr, size: size}, nil
}

// Free releases pinned memory
func (p *PinnedBuffer) Free() {
	if p.ptr != nil {
		C.bridge_host_free(p.ptr)
		p.ptr = nil
	}
}

// Grow ensures buffer is at least size bytes
func (p *PinnedBuffer) Grow(size uint64) error {
	if size <= p.size {
		return nil
	}
	p.Free()
	ptr := C.bridge_host_alloc(C.size_t(size))
	if ptr == nil {
		return fmt.Errorf("pinned realloc failed: %w", lastError())
	}
	p.ptr = ptr
	p.size = size
	return nil
}

// Bytes returns a Go byte slice backed by pinned memory (zero-copy)
func (p *PinnedBuffer) Bytes() []byte {
	return unsafe.Slice((*byte)(p.ptr), p.size)
}

// TransferBatchPinned uses a PinnedBuffer for faster transfer
// The PinnedBuffer should be reused across batches
func TransferBatchPinned(tb *loader.TrainingBatch, pinned *PinnedBuffer) (*GPUBatch, error) {
	if tb == nil {
		return nil, fmt.Errorf("nil TrainingBatch")
	}

	totalFrames := tb.Features.Rows
	featDim := tb.Features.Cols
	batchSize := tb.BatchSize

	ivecDim := 0
	if tb.Ivectors != nil {
		ivecDim = tb.Ivectors.Cols
	}

	numStates := tb.FstCSR.NumStates
	numArcs := tb.FstCSR.NumArcs

	// FP16 conversion on CPU
	featFP16 := fp16.ConvertFloat32ToFloat16(tb.Features.Data)

	var ivecFP16 []uint16
	if tb.Ivectors != nil {
		ivecFP16 = fp16.ConvertFloat32ToFloat16(tb.Ivectors.Data)
	}

	// Allocate GPU buffer
	gb := &GPUBatch{
		TotalFrames:  totalFrames,
		FeatDim:      featDim,
		BatchSize:    batchSize,
		IvecDim:      ivecDim,
		NumStates:    numStates,
		NumArcs:      numArcs,
		LabelDim:     tb.LabelDim,
		FrameOffsets: tb.FrameOffsets,
		NumFrames:    tb.NumFrames,
		FramesPerSeq: tb.FramesPerSeq,
		Weight:       tb.Weight,
		StateOffsets: tb.StateOffsets,
	}

	ret := C.bridge_batch_alloc(
		C.int(totalFrames), C.int(featDim),
		C.int(batchSize), C.int(ivecDim),
		C.int(numStates), C.int(numArcs),
		&gb.ptrs,
	)
	if ret != 0 {
		return nil, fmt.Errorf("GPU alloc failed: %w", lastError())
	}

	// Grow pinned buffer if needed
	totalBytes := uint64(gb.ptrs.total_bytes)
	if err := pinned.Grow(totalBytes); err != nil {
		gb.Free()
		return nil, err
	}

	// Pack into pinned buffer
	hostBuf := pinned.Bytes()
	offset := uint64(0)

	copyU16(hostBuf, offset, featFP16)
	offset += uint64(gb.ptrs.features_bytes)

	if len(ivecFP16) > 0 {
		copyU16(hostBuf, offset, ivecFP16)
	}
	offset += uint64(gb.ptrs.ivectors_bytes)

	copyI32(hostBuf, offset, tb.FstCSR.RowPtr)
	offset += uint64(gb.ptrs.csr_rowptr_bytes)

	copyI32(hostBuf, offset, tb.FstCSR.ColIdx)
	offset += uint64(gb.ptrs.csr_colidx_bytes)

	copyI32(hostBuf, offset, tb.FstCSR.Labels)
	offset += uint64(gb.ptrs.csr_labels_bytes)

	copyF32(hostBuf, offset, tb.FstCSR.Weights)

	// Single cudaMemcpy from pinned memory (DMA, ~2x faster)
	ret = C.bridge_batch_transfer(
		&gb.ptrs,
		pinned.ptr,
		C.size_t(totalBytes),
	)
	if ret != 0 {
		gb.Free()
		return nil, fmt.Errorf("GPU transfer failed: %w", lastError())
	}

	return gb, nil
}

// ============================================================
// ReadBack — read FP16 data from GPU back to CPU as FP32
// For debugging and verification
// ============================================================

// ReadFeatures reads features from GPU back as FP32
func (b *GPUBatch) ReadFeatures() ([]float32, error) {
	count := b.TotalFrames * b.FeatDim
	fp16Data := make([]uint16, count)

	ret := C.bridge_read_fp16(
		(*C.uint16_t)(unsafe.Pointer(&fp16Data[0])),
		b.ptrs.d_features,
		C.size_t(count),
	)
	if ret != 0 {
		return nil, fmt.Errorf("read features failed: %w", lastError())
	}

	return fp16.ConvertFloat16ToFloat32(fp16Data), nil
}

// ReadIvectors reads ivectors from GPU back as FP32
func (b *GPUBatch) ReadIvectors() ([]float32, error) {
	if b.IvecDim == 0 {
		return nil, nil
	}
	count := b.BatchSize * b.IvecDim
	fp16Data := make([]uint16, count)

	ret := C.bridge_read_fp16(
		(*C.uint16_t)(unsafe.Pointer(&fp16Data[0])),
		b.ptrs.d_ivectors,
		C.size_t(count),
	)
	if ret != 0 {
		return nil, fmt.Errorf("read ivectors failed: %w", lastError())
	}

	return fp16.ConvertFloat16ToFloat32(fp16Data), nil
}

// ============================================================
// Helper functions for packing into byte buffers
// ============================================================

func copyU16(dst []byte, offset uint64, src []uint16) {
	if len(src) == 0 {
		return
	}
	srcBytes := unsafe.Slice((*byte)(unsafe.Pointer(&src[0])), len(src)*2)
	copy(dst[offset:], srcBytes)
}

func copyI32(dst []byte, offset uint64, src []int32) {
	if len(src) == 0 {
		return
	}
	srcBytes := unsafe.Slice((*byte)(unsafe.Pointer(&src[0])), len(src)*4)
	copy(dst[offset:], srcBytes)
}

func copyF32(dst []byte, offset uint64, src []float32) {
	if len(src) == 0 {
		return
	}
	srcBytes := unsafe.Slice((*byte)(unsafe.Pointer(&src[0])), len(src)*4)
	copy(dst[offset:], srcBytes)
}
