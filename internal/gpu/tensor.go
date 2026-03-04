package gpu

/*
#cgo CFLAGS: -I${SRCDIR}/../../cpp/include
#cgo LDFLAGS: -L${SRCDIR}/../../cpp/build -lkaldi_fp16 -lkaldi_fp16_cgo -L/usr/local/cuda-12.8/lib64 -lcublas -lcudart -lstdc++

#include "bridge.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// Tensor represents an FP16 tensor on GPU
type Tensor struct {
	Ptr   unsafe.Pointer // device memory
	Rows  int
	Cols  int
	Owned bool // if true, Free() will release memory
}

// Numel returns total number of elements
func (t *Tensor) Numel() int { return t.Rows * t.Cols }

// Bytes returns size in bytes (FP16 = 2 bytes per element)
func (t *Tensor) Bytes() int { return t.Numel() * 2 }

// Free releases GPU memory if owned
func (t *Tensor) Free() {
	if t.Owned && t.Ptr != nil {
		C.bridge_gpu_free(t.Ptr)
		t.Ptr = nil
	}
}

// NewTensor allocates an uninitialized FP16 tensor on GPU
func NewTensor(rows, cols int) (*Tensor, error) {
	bytes := rows * cols * 2 // FP16
	ptr := C.bridge_gpu_malloc(C.size_t(bytes))
	if ptr == nil {
		return nil, fmt.Errorf("GPU alloc %dx%d (%d bytes): %w", rows, cols, bytes, lastError())
	}
	return &Tensor{Ptr: ptr, Rows: rows, Cols: cols, Owned: true}, nil
}

// ZeroTensor allocates an FP16 tensor and returns it (caller should call Fill(t, 0) to zero it)
// Note: memory is uninitialized. Use gpu.Fill(t, 0) after creation.
func ZeroTensor(rows, cols int) (*Tensor, error) {
	t, err := NewTensor(rows, cols)
	if err != nil {
		return nil, err
	}
	// Zero via memset (2 bytes per FP16 element, all-zeros = FP16 zero)
	size := rows * cols * 2
	zeroData := make([]byte, size)
	ret := C.bridge_transfer_fp16(t.Ptr, (*C.uint16_t)(unsafe.Pointer(&zeroData[0])), C.size_t(rows*cols))
	if ret != 0 {
		t.Free()
		return nil, fmt.Errorf("zero tensor: %w", lastError())
	}
	return t, nil
}

// TensorFromFP32 creates GPU FP16 tensor from CPU FP32 data
func TensorFromFP32(data []float32, rows, cols int) (*Tensor, error) {
	if len(data) != rows*cols {
		return nil, fmt.Errorf("data length %d != %d×%d", len(data), rows, cols)
	}

	// Convert to FP16 on CPU
	fp16Data := make([]uint16, len(data))
	for i, v := range data {
		fp16Data[i] = float32ToFP16Bits(v)
	}

	t, err := NewTensor(rows, cols)
	if err != nil {
		return nil, err
	}

	// Transfer FP16 to GPU
	ret := C.bridge_transfer_fp16(t.Ptr, (*C.uint16_t)(unsafe.Pointer(&fp16Data[0])), C.size_t(len(fp16Data)))
	if ret != 0 {
		t.Free()
		return nil, fmt.Errorf("transfer failed: %w", lastError())
	}

	return t, nil
}

// ToFP32 reads GPU FP16 tensor back as CPU FP32
func (t *Tensor) ToFP32() ([]float32, error) {
	count := t.Numel()
	fp16Data := make([]uint16, count)

	ret := C.bridge_read_fp16(
		(*C.uint16_t)(unsafe.Pointer(&fp16Data[0])),
		t.Ptr,
		C.size_t(count),
	)
	if ret != 0 {
		return nil, fmt.Errorf("read failed: %w", lastError())
	}

	result := make([]float32, count)
	for i, v := range fp16Data {
		result[i] = fp16BitsToFloat32(v)
	}
	return result, nil
}

// View creates a non-owning view into part of a tensor
// offset is in elements, not bytes
func (t *Tensor) View(rowOffset, rows, cols int) *Tensor {
	byteOffset := rowOffset * t.Cols * 2
	ptr := unsafe.Pointer(uintptr(t.Ptr) + uintptr(byteOffset))
	return &Tensor{Ptr: ptr, Rows: rows, Cols: cols, Owned: false}
}

// String returns a description
func (t *Tensor) String() string {
	return fmt.Sprintf("Tensor[%d×%d, %d bytes, ptr=%p]", t.Rows, t.Cols, t.Bytes(), t.Ptr)
}

// ============================================================
// TensorInt32 — for integer data on GPU (indices, CSR)
// ============================================================

type TensorInt32 struct {
	Ptr   unsafe.Pointer
	Count int
	Owned bool
}

func NewTensorInt32(count int) (*TensorInt32, error) {
	bytes := count * 4
	ptr := C.bridge_gpu_malloc(C.size_t(bytes))
	if ptr == nil {
		return nil, fmt.Errorf("GPU alloc int32[%d]: %w", count, lastError())
	}
	return &TensorInt32{Ptr: ptr, Count: count, Owned: true}, nil
}

func (t *TensorInt32) Free() {
	if t.Owned && t.Ptr != nil {
		C.bridge_gpu_free(t.Ptr)
		t.Ptr = nil
	}
}

// ============================================================
// FP16 bit conversion (for tensor creation)
// Matches IEEE 754 half-precision
// ============================================================

func float32ToFP16Bits(f float32) uint16 {
	// Use the fp16 package for accurate conversion
	// Inline simplified version for speed
	bits := *(*uint32)(unsafe.Pointer(&f))
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits>>23)&0xFF) - 127
	frac := bits & 0x7FFFFF

	switch {
	case exp > 15:
		return sign | 0x7C00 // Inf
	case exp < -14:
		return sign // zero (flush denorms)
	default:
		return sign | uint16(exp+15)<<10 | uint16(frac>>13)
	}
}

func fp16BitsToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1F
	frac := uint32(h & 0x03FF)

	var bits uint32
	switch {
	case exp == 0:
		if frac == 0 {
			bits = sign
		} else {
			// Denorm
			exp = 1
			for frac&0x400 == 0 {
				frac <<= 1
				exp--
			}
			frac &= 0x3FF
			bits = sign | (uint32(127-15+exp) << 23) | (frac << 13)
		}
	case exp == 31:
		bits = sign | 0x7F800000 | (frac << 13)
	default:
		bits = sign | (uint32(exp-15+127) << 23) | (frac << 13)
	}

	return *(*float32)(unsafe.Pointer(&bits))
}
