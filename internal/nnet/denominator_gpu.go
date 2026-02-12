// denominator.go — Unified Denominator interface for chain loss backward pass.
//
// The backward pass (backward.go) needs GPU-pointer-based denominator computation.
// The existing NativeDenominator (chain_den_native.go) uses Go slices (CPU↔GPU copies).
//
// This file provides:
//  1. Denominator interface — GPU-pointer based, used by ComputeChainObjfAndDeriv
//  2. GPUDenominator — adapter that wraps NativeDenominator for GPU-pointer usage
//
// Migration path:
//   - Phase 1 (now):  GPUDenominator copies GPU→CPU, calls NativeDenominator, copies CPU→GPU
//   - Phase 2 (soon): NativeDenominator gains native GPU-pointer ForwardBackward, no copies
package nnet

/*
#cgo CFLAGS: -I/usr/local/cuda-12.8/include
#include <cuda_runtime.h>
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// DenominatorGPU is the interface for GPU-pointer-based denominator computation.
// Used by ComputeChainObjfAndDeriv (backward pass).
// All pointers are GPU device pointers. The implementation manages its own GPU memory.
type DenominatorGPU interface {
	// ForwardBackwardGPU computes the denominator log-prob and posteriors on GPU.
	//
	// Args:
	//   nnetOutput: [T × numPDFs] FP32 on GPU (read-only)
	//   T:          number of time frames
	//   numPDFs:    output dimension
	//
	// Returns:
	//   logprob:    scalar log-probability (on host)
	//   denPost:    [T × numPDFs] FP32 posteriors on GPU
	//               Caller must NOT free — owned by implementation
	//   error:      nil on success
	ForwardBackwardGPU(nnetOutput unsafe.Pointer, T, numPDFs int) (float64, unsafe.Pointer, error)
}

// Compile-time check: GPUDenominator implements DenominatorGPU
var _ DenominatorGPU = (*GPUDenominator)(nil)

// GPUDenominator wraps NativeDenominator to satisfy the DenominatorGPU interface.
// Phase 1: copies GPU→CPU→GPU (works but adds latency).
// Phase 2: will call chain_den.cu directly with GPU pointers (zero-copy).
type GPUDenominator struct {
	native      *NativeDenominator
	numPDFs     int
	derivWeight float32

	// Reusable GPU buffer for denominator posteriors
	denPostGPU  unsafe.Pointer // float* on GPU
	denPostSize int            // current allocation size in bytes
}

// NewGPUDenominator wraps an existing NativeDenominator.
func NewGPUDenominator(native *NativeDenominator, numPDFs int, derivWeight float32) *GPUDenominator {
	return &GPUDenominator{
		native:      native,
		numPDFs:     numPDFs,
		derivWeight: derivWeight,
	}
}

// ForwardBackwardGPU implements the Denominator interface.
// Phase 1: GPU→CPU copy, call NativeDenominator, CPU→GPU copy.
func (gd *GPUDenominator) ForwardBackwardGPU(nnetOutput unsafe.Pointer, T, numPDFs int) (float64, unsafe.Pointer, error) {
	totalElements := T * numPDFs
	totalBytes := totalElements * 4 // float32

	// 1. Copy nnet_output from GPU to CPU
	hostOutput := make([]float32, totalElements)
	ret := C.cudaMemcpy(
		unsafe.Pointer(&hostOutput[0]),
		nnetOutput,
		C.size_t(totalBytes),
		C.cudaMemcpyDeviceToHost,
	)
	if ret != 0 {
		return 0, nil, fmt.Errorf("cudaMemcpy D2H failed: %d", int(ret))
	}

	// 2. Call existing NativeDenominator (CPU-based)
	logprob, hostGrad, err := gd.native.ForwardBackward(hostOutput, T, 1, gd.derivWeight)
	if err != nil {
		return 0, nil, fmt.Errorf("NativeDenominator.ForwardBackward: %w", err)
	}

	// 3. Ensure GPU buffer is allocated
	if err := gd.ensureGPUBuffer(totalBytes); err != nil {
		return 0, nil, err
	}

	// 4. Copy posteriors from CPU to GPU
	ret = C.cudaMemcpy(
		gd.denPostGPU,
		unsafe.Pointer(&hostGrad[0]),
		C.size_t(totalBytes),
		C.cudaMemcpyHostToDevice,
	)
	if ret != 0 {
		return 0, nil, fmt.Errorf("cudaMemcpy H2D failed: %d", int(ret))
	}

	return float64(logprob), gd.denPostGPU, nil
}

// ensureGPUBuffer allocates or reallocates the GPU posterior buffer.
func (gd *GPUDenominator) ensureGPUBuffer(needed int) error {
	if gd.denPostSize >= needed {
		return nil // already large enough
	}

	// Free old buffer
	if gd.denPostGPU != nil {
		C.cudaFree(gd.denPostGPU)
		gd.denPostGPU = nil
		gd.denPostSize = 0
	}

	// Allocate new
	ret := C.cudaMalloc(&gd.denPostGPU, C.size_t(needed))
	if ret != 0 {
		return fmt.Errorf("cudaMalloc denPost %d bytes: error %d", needed, int(ret))
	}
	gd.denPostSize = needed
	return nil
}

// Free releases GPU resources.
func (gd *GPUDenominator) Free() {
	if gd.denPostGPU != nil {
		C.cudaFree(gd.denPostGPU)
		gd.denPostGPU = nil
		gd.denPostSize = 0
	}
	// Note: does NOT free the underlying NativeDenominator
}
