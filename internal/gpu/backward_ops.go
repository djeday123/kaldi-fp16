// backward_ops.go — GPU backward pass operations
//
// Provides:
//   - Activation backward: ReLU, Sigmoid, Tanh (in-place on grad)
//   - MaxPool1D backward (scatter via saved indices)
//   - Affine backward (2× GEMM + transpose, no new CUDA kernel)
//   - Transpose helper

package gpu

/*
#cgo CFLAGS: -I${SRCDIR}/../../cpp/include
#cgo LDFLAGS: -L${SRCDIR}/../../cpp/build -lkaldi_fp16 -lkaldi_fp16_cgo -L/usr/local/cuda-12.8/lib64 -lcublas -lcudart -lstdc++

#include "ops.h"
#include "bridge.h"

// Manual declaration — cnn_fp16.h can't be included directly (pulls cuda_runtime.h)
void launch_maxpool1d_backward_fp16(
    const void* grad_output,
    const void* indices,
    void* grad_input,
    int batch_size,
    int time_in,
    int time_out,
    int channels,
    void* stream
);
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// ============================================================
// Activation backward passes (in-place on grad)
// ============================================================

// ReLUBackward: grad[i] = x[i] > 0 ? grad[i] : 0
func ReLUBackward(x, grad *Tensor) error {
	if x.Numel() != grad.Numel() {
		return fmt.Errorf("ReLUBackward size mismatch: x=%d grad=%d", x.Numel(), grad.Numel())
	}
	ret := C.ops_relu_backward(x.Ptr, grad.Ptr, C.int(grad.Numel()))
	if ret != 0 {
		return fmt.Errorf("relu_backward: %w", opsErr())
	}
	return nil
}

// SigmoidBackward: grad[i] *= output[i] * (1 - output[i])
func SigmoidBackward(output, grad *Tensor) error {
	if output.Numel() != grad.Numel() {
		return fmt.Errorf("SigmoidBackward size mismatch: output=%d grad=%d", output.Numel(), grad.Numel())
	}
	ret := C.ops_sigmoid_backward(output.Ptr, grad.Ptr, C.int(grad.Numel()))
	if ret != 0 {
		return fmt.Errorf("sigmoid_backward: %w", opsErr())
	}
	return nil
}

// TanhBackward: grad[i] *= (1 - output[i]^2)
func TanhBackward(output, grad *Tensor) error {
	if output.Numel() != grad.Numel() {
		return fmt.Errorf("TanhBackward size mismatch: output=%d grad=%d", output.Numel(), grad.Numel())
	}
	ret := C.ops_tanh_backward(output.Ptr, grad.Ptr, C.int(grad.Numel()))
	if ret != 0 {
		return fmt.Errorf("tanh_backward: %w", opsErr())
	}
	return nil
}

// BatchNormBackward: gradIn = gradOut * gamma / sqrt(var + eps)
func BatchNormBackward(gradOut, gradIn *Tensor, bn *BNParams, eps float32) error {
	if gradOut.Rows != gradIn.Rows || gradOut.Cols != gradIn.Cols {
		return fmt.Errorf("batchnorm backward: shape mismatch")
	}
	ret := C.ops_batchnorm_backward(
		gradOut.Ptr, gradIn.Ptr,
		(*C.float)(bn.Gamma), (*C.float)(bn.Var),
		C.float(eps),
		C.int(gradOut.Rows), C.int(gradOut.Cols))
	if ret != 0 {
		return fmt.Errorf("batchnorm backward failed")
	}
	return nil
}

// ============================================================
// Transpose: dst[N×M] = src[M×N]^T
// ============================================================

func Transpose(src, dst *Tensor) error {
	if src.Rows != dst.Cols || src.Cols != dst.Rows {
		return fmt.Errorf("Transpose: src [%d×%d] dst [%d×%d] mismatch",
			src.Rows, src.Cols, dst.Rows, dst.Cols)
	}
	ret := C.ops_transpose(src.Ptr, dst.Ptr, C.int(src.Rows), C.int(src.Cols))
	if ret != 0 {
		return fmt.Errorf("transpose: %w", opsErr())
	}
	return nil
}

// ============================================================
// MaxPool1D backward
// ============================================================

// MaxPoolIndices holds int32 indices tensor on GPU (saved from MaxPool forward)
type MaxPoolIndices struct {
	Ptr unsafe.Pointer // device int32*
}

// NewMaxPoolIndices allocates int32 indices buffer on GPU
func NewMaxPoolIndices(count int) (*MaxPoolIndices, error) {
	bytes := C.size_t(count * 4)
	ptr := C.bridge_gpu_malloc(bytes)
	if ptr == nil {
		return nil, fmt.Errorf("maxpool indices alloc: %w", lastError())
	}
	return &MaxPoolIndices{Ptr: ptr}, nil
}

func (idx *MaxPoolIndices) Free() {
	if idx.Ptr != nil {
		C.bridge_gpu_free(idx.Ptr)
		idx.Ptr = nil
	}
}

// MaxPool1DBackward scatters gradients from output back to input positions.
func MaxPool1DBackward(gradOutput *Tensor, indices *MaxPoolIndices, gradInput *Tensor,
	batchSize, timeIn, timeOut, channels int) error {

	if err := Fill(gradInput, 0.0); err != nil {
		return fmt.Errorf("maxpool_backward zero: %w", err)
	}

	C.launch_maxpool1d_backward_fp16(
		gradOutput.Ptr,
		indices.Ptr,
		gradInput.Ptr,
		C.int(batchSize),
		C.int(timeIn),
		C.int(timeOut),
		C.int(channels),
		nil,
	)

	return nil
}

// ============================================================
// Affine backward (via GEMM)
// ============================================================

// AffineBackwardData: gradInput[T×M] = gradOutput[T×K] × W^T[K×M]
func AffineBackwardData(h *Handle, gradOutput, weight *Tensor) (*Tensor, error) {
	T := gradOutput.Rows
	K := gradOutput.Cols
	M := weight.Rows

	if weight.Cols != K {
		return nil, fmt.Errorf("AffineBackwardData: weight cols %d != gradOutput cols %d", weight.Cols, K)
	}

	wt, err := NewTensor(K, M)
	if err != nil {
		return nil, fmt.Errorf("alloc W^T: %w", err)
	}
	defer wt.Free()

	if err := Transpose(weight, wt); err != nil {
		return nil, fmt.Errorf("transpose W: %w", err)
	}

	gradInput, err := NewTensor(T, M)
	if err != nil {
		return nil, fmt.Errorf("alloc gradInput: %w", err)
	}

	if err := GEMMSimple(h, gradOutput, wt, gradInput); err != nil {
		gradInput.Free()
		return nil, fmt.Errorf("GEMM gradInput: %w", err)
	}

	return gradInput, nil
}

// AffineBackwardWeights: gradW[M×K] = input^T[M×T] × gradOutput[T×K]
func AffineBackwardWeights(h *Handle, input, gradOutput *Tensor) (*Tensor, error) {
	T := gradOutput.Rows
	K := gradOutput.Cols
	M := input.Cols

	if input.Rows != T {
		return nil, fmt.Errorf("AffineBackwardWeights: input rows %d != gradOutput rows %d", input.Rows, T)
	}

	it, err := NewTensor(M, T)
	if err != nil {
		return nil, fmt.Errorf("alloc input^T: %w", err)
	}
	defer it.Free()

	if err := Transpose(input, it); err != nil {
		return nil, fmt.Errorf("transpose input: %w", err)
	}

	gradW, err := NewTensor(M, K)
	if err != nil {
		return nil, fmt.Errorf("alloc gradW: %w", err)
	}

	if err := GEMMSimple(h, it, gradOutput, gradW); err != nil {
		gradW.Free()
		return nil, fmt.Errorf("GEMM gradW: %w", err)
	}

	return gradW, nil
}

// AffineBackwardBias: gradBias[1×K] = sum(gradOutput, dim=0)
func AffineBackwardBias(h *Handle, gradOutput *Tensor) (*Tensor, error) {
	T := gradOutput.Rows
	K := gradOutput.Cols

	ones, err := NewTensor(1, T)
	if err != nil {
		return nil, fmt.Errorf("alloc ones: %w", err)
	}
	defer ones.Free()

	if err := Fill(ones, 1.0); err != nil {
		return nil, err
	}

	gradBias, err := NewTensor(1, K)
	if err != nil {
		return nil, fmt.Errorf("alloc gradBias: %w", err)
	}

	if err := GEMMSimple(h, ones, gradOutput, gradBias); err != nil {
		gradBias.Free()
		return nil, fmt.Errorf("GEMM gradBias: %w", err)
	}

	return gradBias, nil
}
