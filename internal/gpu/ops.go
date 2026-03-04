package gpu

/*
#cgo CFLAGS: -I${SRCDIR}/../../cpp/include
#cgo LDFLAGS: -L${SRCDIR}/../../cpp/build -lkaldi_fp16 -lkaldi_fp16_cgo -L/usr/local/cuda-12.8/lib64 -lcublas -lcudart -lstdc++

#include "bridge.h"
#include "ops.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// ============================================================
// cuBLAS Handle
// ============================================================

type Handle struct {
	h unsafe.Pointer
}

func NewHandle() (*Handle, error) {
	h := C.ops_cublas_create()
	if h == nil {
		return nil, fmt.Errorf("cublas create: %s", C.GoString(C.ops_last_error()))
	}
	return &Handle{h: h}, nil
}

func (h *Handle) Destroy() {
	if h.h != nil {
		C.ops_cublas_destroy(h.h)
		h.h = nil
	}
}

func opsErr() error {
	s := C.ops_last_error()
	if s == nil {
		return fmt.Errorf("unknown ops error")
	}
	defer C.ops_clear_error()
	return fmt.Errorf("%s", C.GoString(s))
}

// ============================================================
// GEMM: C = alpha * A * B + beta * C
// A: [M x K], B: [K x N], C: [M x N], all FP16 row-major
// Uses Tensor Cores with FP32 accumulation
// ============================================================

func GEMM(h *Handle, M, N, K int, alpha float32, A, B *Tensor, beta float32, C_ *Tensor) error {
	ret := C.ops_gemm(
		h.h,
		C.int(M), C.int(N), C.int(K),
		C.float(alpha),
		A.Ptr, C.int(K), // A: MxK, lda=K
		B.Ptr, C.int(N), // B: KxN, ldb=N
		C.float(beta),
		C_.Ptr, C.int(N), // C: MxN, ldc=N
	)
	if ret != 0 {
		return fmt.Errorf("GEMM: %w", opsErr())
	}
	return nil
}

// GEMMSimple: C = A * B (alpha=1, beta=0)
func GEMMSimple(h *Handle, A, B, C_ *Tensor) error {
	return GEMM(h, A.Rows, B.Cols, A.Cols, 1.0, A, B, 0.0, C_)
}

// GEMMAcc: C += A * B (alpha=1, beta=1)
func GEMMAcc(h *Handle, A, B, C_ *Tensor) error {
	return GEMM(h, A.Rows, B.Cols, A.Cols, 1.0, A, B, 1.0, C_)
}

// ============================================================
// Activations (in-place)
// ============================================================

func ReLU(t *Tensor) error {
	ret := C.ops_relu(t.Ptr, C.int(t.Numel()))
	if ret != 0 {
		return fmt.Errorf("relu: %w", opsErr())
	}
	return nil
}

func Sigmoid(t *Tensor) error {
	ret := C.ops_sigmoid(t.Ptr, C.int(t.Numel()))
	if ret != 0 {
		return fmt.Errorf("sigmoid: %w", opsErr())
	}
	return nil
}

func Tanh(t *Tensor) error {
	ret := C.ops_tanh_act(t.Ptr, C.int(t.Numel()))
	if ret != 0 {
		return fmt.Errorf("tanh: %w", opsErr())
	}
	return nil
}

// ============================================================
// Softmax / Log-softmax (per-row)
// ============================================================

func Softmax(t *Tensor) error {
	ret := C.ops_softmax(t.Ptr, C.int(t.Rows), C.int(t.Cols))
	if ret != 0 {
		return fmt.Errorf("softmax: %w", opsErr())
	}
	return nil
}

func LogSoftmax(t *Tensor) error {
	ret := C.ops_log_softmax(t.Ptr, C.int(t.Rows), C.int(t.Cols))
	if ret != 0 {
		return fmt.Errorf("log_softmax: %w", opsErr())
	}
	return nil
}

// ============================================================
// BatchNorm
// ============================================================

// BNParams holds batch normalization parameters on GPU (FP32)
type BNParams struct {
	Mean    unsafe.Pointer // [D] float32
	Var     unsafe.Pointer // [D] float32
	Gamma   unsafe.Pointer // [D] float32
	Beta    unsafe.Pointer // [D] float32
	Dim     int
	Epsilon float32 // small constant for numerical stability
}

// NewBNParams creates batchnorm parameters from CPU float32 arrays
func NewBNParams(mean, variance, gamma, beta []float32) (*BNParams, error) {
	D := len(mean)
	bytes := C.size_t(D * 4)

	bn := &BNParams{Dim: D}
	var err error

	// Allocate GPU memory for FP32 params
	allocF32 := func(data []float32) (unsafe.Pointer, error) {
		ptr := C.bridge_gpu_malloc(bytes)
		if ptr == nil {
			return nil, fmt.Errorf("bn alloc: %w", lastError())
		}
		ret := C.bridge_transfer_float32(ptr, (*C.float)(unsafe.Pointer(&data[0])), C.size_t(D))
		if ret != 0 {
			C.bridge_gpu_free(ptr)
			return nil, fmt.Errorf("bn transfer: %w", lastError())
		}
		return ptr, nil
	}

	if bn.Mean, err = allocF32(mean); err != nil {
		return nil, err
	}
	if bn.Var, err = allocF32(variance); err != nil {
		bn.Free()
		return nil, err
	}
	if bn.Gamma, err = allocF32(gamma); err != nil {
		bn.Free()
		return nil, err
	}
	if bn.Beta, err = allocF32(beta); err != nil {
		bn.Free()
		return nil, err
	}

	return bn, nil
}

func (bn *BNParams) Free() {
	if bn.Mean != nil {
		C.bridge_gpu_free(bn.Mean)
		bn.Mean = nil
	}
	if bn.Var != nil {
		C.bridge_gpu_free(bn.Var)
		bn.Var = nil
	}
	if bn.Gamma != nil {
		C.bridge_gpu_free(bn.Gamma)
		bn.Gamma = nil
	}
	if bn.Beta != nil {
		C.bridge_gpu_free(bn.Beta)
		bn.Beta = nil
	}
}

// BatchNormForward applies batchnorm in-place on FP16 tensor
func BatchNormForward(x *Tensor, bn *BNParams, epsilon float32) error {
	ret := C.ops_batchnorm_forward(
		x.Ptr, C.int(x.Rows), C.int(x.Cols),
		(*C.float)(bn.Mean), (*C.float)(bn.Var),
		(*C.float)(bn.Gamma), (*C.float)(bn.Beta),
		C.float(epsilon),
	)
	if ret != 0 {
		return fmt.Errorf("batchnorm: %w", opsErr())
	}
	return nil
}

// BatchNormForwardRMS applies Kaldi-style batchnorm with target_rms
func BatchNormForwardRMS(x *Tensor, mean, variance unsafe.Pointer, targetRMS, epsilon float32) error {
	ret := C.ops_batchnorm_forward_rms(
		x.Ptr, C.int(x.Rows), C.int(x.Cols),
		(*C.float)(mean), (*C.float)(variance),
		C.float(targetRMS), C.float(epsilon),
	)
	if ret != 0 {
		return fmt.Errorf("batchnorm_rms: %w", opsErr())
	}
	return nil
}

// ============================================================
// Element-wise operations
// ============================================================

// AddScaled: dst = alpha * src + beta * dst
func AddScaled(dst, src *Tensor, alpha, beta float32) error {
	if dst.Numel() != src.Numel() {
		return fmt.Errorf("AddScaled size mismatch: %d vs %d", dst.Numel(), src.Numel())
	}
	ret := C.ops_add_scaled(dst.Ptr, src.Ptr, C.int(dst.Numel()),
		C.float(alpha), C.float(beta))
	if ret != 0 {
		return fmt.Errorf("add_scaled: %w", opsErr())
	}
	return nil
}

// Add: dst += src
func Add(dst, src *Tensor) error {
	ret := C.ops_add(dst.Ptr, src.Ptr, C.int(dst.Numel()))
	if ret != 0 {
		return fmt.Errorf("add: %w", opsErr())
	}
	return nil
}

// Copy: dst = src (device to device)
func Copy(dst, src *Tensor) error {
	ret := C.ops_copy(dst.Ptr, src.Ptr, C.int(src.Numel()))
	if ret != 0 {
		return fmt.Errorf("copy: %w", opsErr())
	}
	return nil
}

// Fill: tensor = val
func Fill(t *Tensor, val float32) error {
	ret := C.ops_fill(t.Ptr, C.int(t.Numel()), C.float(val))
	if ret != 0 {
		return fmt.Errorf("fill: %w", opsErr())
	}
	return nil
}

// ============================================================
// Concat and slice columns (for Append in xconfig)
// ============================================================

// ConcatCols copies src columns into dst at column offset
// dst[t, offset:offset+src.Cols] = src[t, :]
func ConcatCols(dst *Tensor, src *Tensor, colOffset int) error {
	if dst.Rows != src.Rows {
		return fmt.Errorf("ConcatCols row mismatch: %d vs %d", dst.Rows, src.Rows)
	}
	ret := C.ops_concat_cols(
		dst.Ptr, C.int(dst.Rows), C.int(dst.Cols),
		src.Ptr, C.int(src.Cols),
		C.int(colOffset),
	)
	if ret != 0 {
		return fmt.Errorf("concat_cols: %w", opsErr())
	}
	return nil
}

// SliceCols copies columns [colOffset, colOffset+dst.Cols) from src into dst
func SliceCols(dst *Tensor, src *Tensor, colOffset int) error {
	if dst.Rows != src.Rows {
		return fmt.Errorf("SliceCols row mismatch: %d vs %d", dst.Rows, src.Rows)
	}
	ret := C.ops_slice_cols(
		src.Ptr, C.int(src.Rows), C.int(src.Cols),
		dst.Ptr, C.int(dst.Cols),
		C.int(colOffset),
	)
	if ret != 0 {
		return fmt.Errorf("slice_cols: %w", opsErr())
	}
	return nil
}

// CombineFeatureMaps reorders concatenated features for CNN input
func CombineFeatureMaps(t *Tensor, height, nf1, nf2 int) error {
	ret := C.ops_combine_feature_maps(
		t.Ptr, C.int(t.Rows), C.int(t.Cols),
		C.int(height), C.int(nf1), C.int(nf2),
	)
	if ret != 0 {
		return fmt.Errorf("combine_feature_maps: %w", opsErr())
	}
	return nil
}

// ============================================================
// Bias add: adds a bias vector to each row
// x[t, :] += bias[:]
// ============================================================

// AddBias is implemented as GEMM with ones vector:
// x += ones * bias (where ones is [T x 1] and bias is [1 x D])
// But simpler to use a custom kernel — let's use AddScaled approach
// Actually the simplest: just use ops_add on repeated slices

// For now, implement via a simple loop of pointer arithmetic
// TODO: custom CUDA kernel for efficiency
func AddBias(h *Handle, x *Tensor, bias *Tensor) error {
	// bias: [1 x D], x: [T x D]
	// We need a custom kernel. For now use GEMM trick:
	// Allocate ones vector [T x 1], then: x += ones * bias
	ones, err := NewTensor(x.Rows, 1)
	if err != nil {
		return err
	}
	defer ones.Free()

	if err := Fill(ones, 1.0); err != nil {
		return err
	}

	// x += ones * bias (accumulate)
	return GEMM(h, x.Rows, x.Cols, 1, 1.0, ones, bias, 1.0, x)
}

// SubsampleRows copies every stride-th row starting at rowOffset
// Returns new tensor [outRows x cols]
func SubsampleRows(src *Tensor, stride, rowOffset int) (*Tensor, error) {
	outRows := (src.Rows - rowOffset + stride - 1) / stride
	dst, err := NewTensor(outRows, src.Cols)
	if err != nil {
		return nil, err
	}

	C.ops_subsample_rows(dst.Ptr, src.Ptr,
		C.int(src.Rows), C.int(src.Cols),
		C.int(stride), C.int(rowOffset))
	return dst, nil
}
