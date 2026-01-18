package kaldibridge

/*
#cgo CFLAGS: -I${SRCDIR}/../../cpp/include
#cgo LDFLAGS: -L${SRCDIR}/../../cpp/build -lkaldi_fp16 -lcublas -lcudart -lstdc++

#include <stdlib.h>
#include <stdint.h>

// C interface to C++ TensorFP16
typedef void* TensorHandle;
typedef void* CuBLASHandlePtr;

// Handle management
extern CuBLASHandlePtr kaldi_cublas_create();
extern void kaldi_cublas_destroy(CuBLASHandlePtr handle);
extern void kaldi_cublas_enable_tensor_cores(CuBLASHandlePtr handle);

// Tensor creation
extern TensorHandle kaldi_tensor_create(int rows, int cols);
extern TensorHandle kaldi_tensor_zeros(int rows, int cols);
extern TensorHandle kaldi_tensor_ones(int rows, int cols);
extern void kaldi_tensor_free(TensorHandle t);

// Tensor properties
extern int kaldi_tensor_rows(TensorHandle t);
extern int kaldi_tensor_cols(TensorHandle t);
extern size_t kaldi_tensor_size(TensorHandle t);

// Data transfer
extern void kaldi_tensor_copy_from_host_fp32(TensorHandle t, const float* data, size_t count);
extern void kaldi_tensor_copy_to_host_fp32(TensorHandle t, float* data, size_t count);

// Operations (use Tensor Cores)
extern void kaldi_gemm(CuBLASHandlePtr handle, TensorHandle A, TensorHandle B, TensorHandle C,
                       float alpha, float beta, int transA, int transB);
extern void kaldi_relu(TensorHandle t);
extern void kaldi_sigmoid(TensorHandle t);
extern void kaldi_tanh(TensorHandle t);
extern void kaldi_softmax(TensorHandle t);
extern void kaldi_add(TensorHandle a, TensorHandle b);
extern void kaldi_scale(TensorHandle t, float alpha);

// Loss scaler
typedef void* LossScalerHandle;
extern LossScalerHandle kaldi_loss_scaler_create(float initial_scale);
extern void kaldi_loss_scaler_free(LossScalerHandle ls);
extern float kaldi_loss_scaler_get_scale(LossScalerHandle ls);
extern void kaldi_loss_scaler_update(LossScalerHandle ls, int overflow);

// Error handling
extern const char* kaldi_get_last_error();
extern void kaldi_clear_error();
*/
import "C"

import (
	"errors"
	"runtime"
	"sync"
	"unsafe"
)

// getLastError returns last CUDA/cuBLAS error if any
func getLastError() error {
	errStr := C.kaldi_get_last_error()
	if errStr == nil {
		return nil
	}
	defer C.kaldi_clear_error()
	return errors.New(C.GoString(errStr))
}

// CuBLASHandle wraps cuBLAS handle with Tensor Core support
type CuBLASHandle struct {
	ptr C.CuBLASHandlePtr
	mu  sync.Mutex
}

// NewCuBLASHandle creates cuBLAS handle with Tensor Cores enabled
func NewCuBLASHandle() (*CuBLASHandle, error) {
	ptr := C.kaldi_cublas_create()
	if ptr == nil {
		return nil, getLastError()
	}

	h := &CuBLASHandle{ptr: ptr}

	// Enable Tensor Cores by default
	C.kaldi_cublas_enable_tensor_cores(ptr)

	runtime.SetFinalizer(h, func(h *CuBLASHandle) {
		h.Close()
	})

	return h, nil
}

// Close releases cuBLAS handle
func (h *CuBLASHandle) Close() {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.ptr != nil {
		C.kaldi_cublas_destroy(h.ptr)
		h.ptr = nil
	}
}

// TensorGPU represents FP16 tensor on GPU
type TensorGPU struct {
	handle C.TensorHandle
	rows   int
	cols   int
}

// NewTensorGPU creates new FP16 tensor on GPU
func NewTensorGPU(rows, cols int) (*TensorGPU, error) {
	handle := C.kaldi_tensor_create(C.int(rows), C.int(cols))
	if handle == nil {
		return nil, getLastError()
	}

	t := &TensorGPU{
		handle: handle,
		rows:   rows,
		cols:   cols,
	}

	runtime.SetFinalizer(t, func(t *TensorGPU) {
		t.Free()
	})

	return t, nil
}

// ZerosGPU creates zero-initialized FP16 tensor on GPU
func ZerosGPU(rows, cols int) (*TensorGPU, error) {
	handle := C.kaldi_tensor_zeros(C.int(rows), C.int(cols))
	if handle == nil {
		return nil, getLastError()
	}

	t := &TensorGPU{
		handle: handle,
		rows:   rows,
		cols:   cols,
	}

	runtime.SetFinalizer(t, func(t *TensorGPU) {
		t.Free()
	})

	return t, nil
}

// OnesGPU creates FP16 tensor filled with ones on GPU
func OnesGPU(rows, cols int) (*TensorGPU, error) {
	handle := C.kaldi_tensor_ones(C.int(rows), C.int(cols))
	if handle == nil {
		return nil, getLastError()
	}

	t := &TensorGPU{
		handle: handle,
		rows:   rows,
		cols:   cols,
	}

	runtime.SetFinalizer(t, func(t *TensorGPU) {
		t.Free()
	})

	return t, nil
}

// Free releases GPU memory
func (t *TensorGPU) Free() {
	if t.handle != nil {
		C.kaldi_tensor_free(t.handle)
		t.handle = nil
	}
}

// Rows returns number of rows
func (t *TensorGPU) Rows() int { return t.rows }

// Cols returns number of columns
func (t *TensorGPU) Cols() int { return t.cols }

// Size returns total number of elements
func (t *TensorGPU) Size() int { return t.rows * t.cols }

// CopyFromHost copies FP32 data from host to GPU (converts to FP16)
func (t *TensorGPU) CopyFromHost(data []float32) error {
	if len(data) != t.Size() {
		return errors.New("data size doesn't match tensor size")
	}

	C.kaldi_tensor_copy_from_host_fp32(
		t.handle,
		(*C.float)(unsafe.Pointer(&data[0])),
		C.size_t(len(data)),
	)

	return getLastError()
}

// CopyToHost copies FP16 data from GPU to host (converts to FP32)
func (t *TensorGPU) CopyToHost() ([]float32, error) {
	data := make([]float32, t.Size())

	C.kaldi_tensor_copy_to_host_fp32(
		t.handle,
		(*C.float)(unsafe.Pointer(&data[0])),
		C.size_t(len(data)),
	)

	if err := getLastError(); err != nil {
		return nil, err
	}

	return data, nil
}

// FromHostGPU creates GPU tensor from host FP32 data
func FromHostGPU(data []float32, rows, cols int) (*TensorGPU, error) {
	if len(data) != rows*cols {
		return nil, errors.New("data size doesn't match dimensions")
	}

	t, err := NewTensorGPU(rows, cols)
	if err != nil {
		return nil, err
	}

	if err := t.CopyFromHost(data); err != nil {
		t.Free()
		return nil, err
	}

	return t, nil
}

// ============================================================================
// GPU Operations (Tensor Cores)
// ============================================================================

// GEMM performs C = alpha * A @ B + beta * C using Tensor Cores
func GEMM(handle *CuBLASHandle, A, B, C *TensorGPU, alpha, beta float32, transA, transB bool) error {
	handle.mu.Lock()
	defer handle.mu.Unlock()

	tA := 0
	tB := 0
	if transA {
		tA = 1
	}
	if transB {
		tB = 1
	}

	C.kaldi_gemm(
		handle.ptr,
		A.handle, B.handle, C.handle,
		C.float(alpha), C.float(beta),
		C.int(tA), C.int(tB),
	)

	return getLastError()
}

// MatMulGPU performs matrix multiplication using Tensor Cores
func MatMulGPU(handle *CuBLASHandle, A, B *TensorGPU) (*TensorGPU, error) {
	if A.Cols() != B.Rows() {
		return nil, errors.New("incompatible dimensions for matmul")
	}

	C, err := NewTensorGPU(A.Rows(), B.Cols())
	if err != nil {
		return nil, err
	}

	if err := GEMM(handle, A, B, C, 1.0, 0.0, false, false); err != nil {
		C.Free()
		return nil, err
	}

	return C, nil
}

// ReLUGPU applies ReLU activation in place
func (t *TensorGPU) ReLUGPU() error {
	C.kaldi_relu(t.handle)
	return getLastError()
}

// SigmoidGPU applies sigmoid activation in place
func (t *TensorGPU) SigmoidGPU() error {
	C.kaldi_sigmoid(t.handle)
	return getLastError()
}

// TanhGPU applies tanh activation in place
func (t *TensorGPU) TanhGPU() error {
	C.kaldi_tanh(t.handle)
	return getLastError()
}

// SoftmaxGPU applies softmax in place
func (t *TensorGPU) SoftmaxGPU() error {
	C.kaldi_softmax(t.handle)
	return getLastError()
}

// AddGPU performs a += b in place
func (a *TensorGPU) AddGPU(b *TensorGPU) error {
	if a.Rows() != b.Rows() || a.Cols() != b.Cols() {
		return errors.New("tensor dimensions must match")
	}
	C.kaldi_add(a.handle, b.handle)
	return getLastError()
}

// ScaleGPU multiplies tensor by scalar in place
func (t *TensorGPU) ScaleGPU(alpha float32) error {
	C.kaldi_scale(t.handle, C.float(alpha))
	return getLastError()
}

// ============================================================================
// Loss Scaler for Mixed Precision Training
// ============================================================================

// LossScaler manages dynamic loss scaling for FP16 training
type LossScaler struct {
	handle C.LossScalerHandle
}

// NewLossScaler creates loss scaler with initial scale
func NewLossScaler(initialScale float32) *LossScaler {
	handle := C.kaldi_loss_scaler_create(C.float(initialScale))

	ls := &LossScaler{handle: handle}

	runtime.SetFinalizer(ls, func(ls *LossScaler) {
		ls.Free()
	})

	return ls
}

// Free releases loss scaler
func (ls *LossScaler) Free() {
	if ls.handle != nil {
		C.kaldi_loss_scaler_free(ls.handle)
		ls.handle = nil
	}
}

// GetScale returns current scale
func (ls *LossScaler) GetScale() float32 {
	return float32(C.kaldi_loss_scaler_get_scale(ls.handle))
}

// Update updates scale based on overflow status
func (ls *LossScaler) Update(overflow bool) {
	ovf := 0
	if overflow {
		ovf = 1
	}
	C.kaldi_loss_scaler_update(ls.handle, C.int(ovf))
}
