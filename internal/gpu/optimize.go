// internal/gpu/optimizer.go — SGD optimizer with FP32 master weights
//
// Kaldi SGD with momentum:
//   velocity = momentum * velocity + grad
//   w_fp32 -= lr * velocity
//   w_fp16 = fp16(w_fp32)
//
// Master weights in FP32 avoid precision loss from small updates.

package gpu

/*
#cgo CFLAGS: -I${SRCDIR}/../../cpp/include
#cgo LDFLAGS: -L${SRCDIR}/../../cpp/build -lkaldi_fp16 -lkaldi_fp16_cgo -L/usr/local/cuda-12.8/lib64 -lcublas -lcudart -lstdc++
#include "ops.h"
#include "bridge.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// ============================================================
// SGDOptimizer — SGD with momentum and FP32 master weights
// ============================================================

// SGDOptimizer holds per-parameter state for SGD with momentum
type SGDOptimizer struct {
	LR       float32
	Momentum float32

	// Per-parameter state
	MasterWeights map[string]map[string]unsafe.Pointer // layer → param → FP32 GPU ptr
	Velocities    map[string]map[string]unsafe.Pointer // layer → param → FP32 GPU ptr
	ParamSizes    map[string]map[string]int            // layer → param → element count
}

// NewSGDOptimizer creates a new SGD optimizer
func NewSGDOptimizer(lr, momentum float32) *SGDOptimizer {
	return &SGDOptimizer{
		LR:            lr,
		Momentum:      momentum,
		MasterWeights: make(map[string]map[string]unsafe.Pointer),
		Velocities:    make(map[string]map[string]unsafe.Pointer),
		ParamSizes:    make(map[string]map[string]int),
	}
}

// RegisterParam creates FP32 master copy and zero velocity for a weight tensor
func (opt *SGDOptimizer) RegisterParam(layerName, paramName string, w *Tensor) error {
	if w == nil {
		return nil
	}

	count := w.Rows * w.Cols
	bytes := C.size_t(count * 4) // FP32

	// Allocate FP32 master weight on GPU
	masterPtr := C.bridge_gpu_malloc(bytes)
	if masterPtr == nil {
		return fmt.Errorf("alloc master weight %s/%s", layerName, paramName)
	}

	// Convert current FP16 weights → FP32 master
	ret := C.ops_fp16_to_fp32(w.Ptr, (*C.float)(masterPtr), C.int(count))
	if ret != 0 {
		C.bridge_gpu_free(masterPtr)
		return fmt.Errorf("fp16→fp32 %s/%s", layerName, paramName)
	}

	// Allocate velocity (zero-initialized)
	velPtr := C.bridge_gpu_malloc(bytes)
	if velPtr == nil {
		C.bridge_gpu_free(masterPtr)
		return fmt.Errorf("alloc velocity %s/%s", layerName, paramName)
	}
	C.bridge_gpu_memset(velPtr, 0, bytes)

	// Store
	if opt.MasterWeights[layerName] == nil {
		opt.MasterWeights[layerName] = make(map[string]unsafe.Pointer)
		opt.Velocities[layerName] = make(map[string]unsafe.Pointer)
		opt.ParamSizes[layerName] = make(map[string]int)
	}
	opt.MasterWeights[layerName][paramName] = masterPtr
	opt.Velocities[layerName][paramName] = velPtr
	opt.ParamSizes[layerName][paramName] = count

	return nil
}

// Update applies one SGD step: velocity = momentum*velocity + grad; w -= lr*velocity
func (opt *SGDOptimizer) Update(layerName, paramName string, w *Tensor, grad *Tensor) error {
	masters, ok := opt.MasterWeights[layerName]
	if !ok {
		return fmt.Errorf("layer not registered: %s", layerName)
	}
	master, ok := masters[paramName]
	if !ok {
		return fmt.Errorf("param not registered: %s/%s", layerName, paramName)
	}
	vel := opt.Velocities[layerName][paramName]
	count := opt.ParamSizes[layerName][paramName]

	ret := C.ops_sgd_update(
		(*C.float)(master),
		w.Ptr,
		grad.Ptr,
		(*C.float)(vel),
		C.float(opt.LR),
		C.float(opt.Momentum),
		C.int(count),
	)
	if ret != 0 {
		return fmt.Errorf("sgd_update %s/%s failed", layerName, paramName)
	}
	return nil
}

// SetLR updates the learning rate
func (opt *SGDOptimizer) SetLR(lr float32) {
	opt.LR = lr
}

// Free releases all GPU memory
func (opt *SGDOptimizer) Free() {
	for _, params := range opt.MasterWeights {
		for _, ptr := range params {
			C.bridge_gpu_free(ptr)
		}
	}
	for _, params := range opt.Velocities {
		for _, ptr := range params {
			C.bridge_gpu_free(ptr)
		}
	}
	opt.MasterWeights = nil
	opt.Velocities = nil
	opt.ParamSizes = nil
}
