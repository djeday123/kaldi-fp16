// CNN Bridge - Go interface to CUDA CNN kernels
// Part of Kaldi-FP16 CNN-TDNN implementation

package kaldibridge

/*
#cgo CFLAGS: -I${SRCDIR}/../../cpp/include
#cgo LDFLAGS: -L${SRCDIR}/../../cpp/build -lkaldi_fp16_cgo -lkaldi_fp16 -lcublas -lcudart -lstdc++

#include <cuda_runtime.h>
#include <stdint.h>

// Conv1D forward
void launch_conv1d_forward_fp16(
    const void* input, const void* weight, const void* bias,
    void* output,
    int batch_size, int time_in, int in_channels,
    int out_channels, int kernel_size, int stride, int padding, int dilation,
    void* stream
);

// Conv1D backward
void launch_conv1d_backward_fp16(
    const void* input, const void* grad_output, const void* weight,
    void* grad_input, void* grad_weight, void* grad_bias,
    int batch_size, int time_in, int in_channels,
    int out_channels, int kernel_size, int stride, int padding, int dilation,
    void* stream
);

// Max pooling
void launch_maxpool1d_forward_fp16(
    const void* input, void* output, void* indices,
    int batch_size, int time_in, int channels,
    int kernel_size, int stride,
    void* stream
);

// Stats pooling
void launch_stats_pooling_fp16(
    const void* input, void* output,
    int batch_size, int time_steps, int channels,
    void* stream
);

// BatchNorm
void launch_batchnorm1d_forward_fp16(
    const void* input, const void* gamma, const void* beta,
    void* running_mean, void* running_var,
    void* output, void* save_mean, void* save_invstd,
    int batch_size, int time_steps, int channels,
    float momentum, float eps, int training,
    void* stream
);

// Depthwise conv
void launch_depthwise_conv1d_fp16(
    const void* input, const void* weight, const void* bias,
    void* output,
    int batch_size, int time_in, int channels,
    int kernel_size, int stride, int padding,
    void* stream
);

// Pointwise conv
void launch_pointwise_conv1d_fp16(
    const void* input, const void* weight, const void* bias,
    void* output,
    int batch_size, int time_steps, int in_channels, int out_channels,
    void* stream
);

// CUDA memory wrappers for CGO
static inline void* cgo_cuda_malloc(size_t size) {
    void* ptr = NULL;
    cudaMalloc(&ptr, size);
    return ptr;
}

static inline void cgo_cuda_free(void* ptr) {
    cudaFree(ptr);
}

static inline void cgo_cuda_memset(void* ptr, int value, size_t count) {
    cudaMemset(ptr, value, count);
}

static inline void cgo_cuda_sync() {
    cudaDeviceSynchronize();
}
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"
)

// ============================================================================
// Conv1D Layer on GPU
// ============================================================================

// Conv1DConfig holds convolution parameters
type Conv1DConfig struct {
	InChannels  int
	OutChannels int
	KernelSize  int
	Stride      int
	Padding     int
	Dilation    int
}

// Conv1DGPU represents a GPU-based 1D convolution layer
type Conv1DGPU struct {
	Config     Conv1DConfig
	Weight     *TensorGPU // [OutChannels, InChannels, KernelSize]
	Bias       *TensorGPU // [OutChannels, 1]
	GradWeight *TensorGPU
	GradBias   *TensorGPU
}

// NewConv1DGPU creates a new GPU convolution layer
func NewConv1DGPU(config Conv1DConfig) (*Conv1DGPU, error) {
	weightRows := config.OutChannels
	weightCols := config.InChannels * config.KernelSize

	weight, err := NewTensorGPU(weightRows, weightCols)
	if err != nil {
		return nil, fmt.Errorf("failed to allocate weight: %w", err)
	}

	bias, err := NewTensorGPU(config.OutChannels, 1)
	if err != nil {
		weight.Free()
		return nil, fmt.Errorf("failed to allocate bias: %w", err)
	}

	gradWeight, err := NewTensorGPU(weightRows, weightCols)
	if err != nil {
		weight.Free()
		bias.Free()
		return nil, fmt.Errorf("failed to allocate grad_weight: %w", err)
	}

	gradBias, err := NewTensorGPU(config.OutChannels, 1)
	if err != nil {
		weight.Free()
		bias.Free()
		gradWeight.Free()
		return nil, fmt.Errorf("failed to allocate grad_bias: %w", err)
	}

	return &Conv1DGPU{
		Config:     config,
		Weight:     weight,
		Bias:       bias,
		GradWeight: gradWeight,
		GradBias:   gradBias,
	}, nil
}

// Forward performs convolution on GPU
// Input shape: [batch, timeIn, inChannels]
// Output shape: [batch, timeOut, outChannels]
func (c *Conv1DGPU) Forward(input *TensorGPU, batch, timeIn int) (*TensorGPU, error) {
	timeOut := (timeIn+2*c.Config.Padding-c.Config.Dilation*(c.Config.KernelSize-1)-1)/c.Config.Stride + 1

	output, err := NewTensorGPU(batch*timeOut, c.Config.OutChannels)
	if err != nil {
		return nil, err
	}

	C.launch_conv1d_forward_fp16(
		unsafe.Pointer(input.handle),
		unsafe.Pointer(c.Weight.handle),
		unsafe.Pointer(c.Bias.handle),
		unsafe.Pointer(output.handle),
		C.int(batch), C.int(timeIn), C.int(c.Config.InChannels),
		C.int(c.Config.OutChannels), C.int(c.Config.KernelSize),
		C.int(c.Config.Stride), C.int(c.Config.Padding), C.int(c.Config.Dilation),
		nil, // default stream
	)

	return output, nil
}

// Backward computes gradients for convolution
func (c *Conv1DGPU) Backward(input, gradOutput *TensorGPU, batch, timeIn, timeOut int) (*TensorGPU, error) {
	gradInput, err := NewTensorGPU(batch*timeIn, c.Config.InChannels)
	if err != nil {
		return nil, err
	}

	// Zero gradients
	C.cgo_cuda_memset(unsafe.Pointer(c.GradWeight.handle), 0, C.size_t(c.GradWeight.Size()*2))
	C.cgo_cuda_memset(unsafe.Pointer(c.GradBias.handle), 0, C.size_t(c.GradBias.Size()*2))

	C.launch_conv1d_backward_fp16(
		unsafe.Pointer(input.handle),
		unsafe.Pointer(gradOutput.handle),
		unsafe.Pointer(c.Weight.handle),
		unsafe.Pointer(gradInput.handle),
		unsafe.Pointer(c.GradWeight.handle),
		unsafe.Pointer(c.GradBias.handle),
		C.int(batch), C.int(timeIn), C.int(c.Config.InChannels),
		C.int(c.Config.OutChannels), C.int(c.Config.KernelSize),
		C.int(c.Config.Stride), C.int(c.Config.Padding), C.int(c.Config.Dilation),
		nil,
	)

	return gradInput, nil
}

// Free releases GPU memory
func (c *Conv1DGPU) Free() {
	if c.Weight != nil {
		c.Weight.Free()
	}
	if c.Bias != nil {
		c.Bias.Free()
	}
	if c.GradWeight != nil {
		c.GradWeight.Free()
	}
	if c.GradBias != nil {
		c.GradBias.Free()
	}
}

// ============================================================================
// MaxPool1D on GPU
// ============================================================================

// TensorGPUInt32 holds int32 data on GPU (for pooling indices)
type TensorGPUInt32 struct {
	ptr  unsafe.Pointer
	rows int
	cols int
}

// NewTensorGPUInt32 allocates int32 tensor on GPU
func NewTensorGPUInt32(rows, cols int) (*TensorGPUInt32, error) {
	size := rows * cols * 4 // int32 = 4 bytes
	ptr := C.cgo_cuda_malloc(C.size_t(size))
	if ptr == nil {
		return nil, fmt.Errorf("cudaMalloc failed")
	}

	t := &TensorGPUInt32{ptr: ptr, rows: rows, cols: cols}
	runtime.SetFinalizer(t, func(t *TensorGPUInt32) { t.Free() })
	return t, nil
}

func (t *TensorGPUInt32) Free() {
	if t.ptr != nil {
		C.cgo_cuda_free(t.ptr)
		t.ptr = nil
	}
}

func (t *TensorGPUInt32) Size() int {
	return t.rows * t.cols
}

// MaxPool1DGPU represents max pooling on GPU
type MaxPool1DGPU struct {
	KernelSize int
	Stride     int
	indices    *TensorGPUInt32 // For backward pass
}

// NewMaxPool1DGPU creates max pooling layer
func NewMaxPool1DGPU(kernelSize, stride int) *MaxPool1DGPU {
	return &MaxPool1DGPU{
		KernelSize: kernelSize,
		Stride:     stride,
	}
}

// Forward performs max pooling
func (p *MaxPool1DGPU) Forward(input *TensorGPU, batch, timeIn, channels int) (*TensorGPU, error) {
	timeOut := (timeIn-p.KernelSize)/p.Stride + 1

	output, err := NewTensorGPU(batch*timeOut, channels)
	if err != nil {
		return nil, err
	}

	indices, err := NewTensorGPUInt32(batch*timeOut, channels)
	if err != nil {
		output.Free()
		return nil, err
	}
	p.indices = indices

	C.launch_maxpool1d_forward_fp16(
		unsafe.Pointer(input.handle),
		unsafe.Pointer(output.handle),
		indices.ptr,
		C.int(batch), C.int(timeIn), C.int(channels),
		C.int(p.KernelSize), C.int(p.Stride),
		nil,
	)

	return output, nil
}

// ============================================================================
// Statistics Pooling (for x-vector)
// ============================================================================

// StatsPoolingGPU computes mean and std over time dimension
func StatsPoolingGPU(input *TensorGPU, batch, timeSteps, channels int) (*TensorGPU, error) {
	// Output: [batch, 2*channels]
	output, err := NewTensorGPU(batch, 2*channels)
	if err != nil {
		return nil, err
	}

	C.launch_stats_pooling_fp16(
		unsafe.Pointer(input.handle),
		unsafe.Pointer(output.handle),
		C.int(batch), C.int(timeSteps), C.int(channels),
		nil,
	)

	return output, nil
}

// ============================================================================
// BatchNorm1D on GPU
// ============================================================================

// BatchNorm1DGPU represents batch normalization on GPU
type BatchNorm1DGPU struct {
	Channels    int
	Gamma       *TensorGPU
	Beta        *TensorGPU
	RunningMean *TensorGPU
	RunningVar  *TensorGPU
	SaveMean    *TensorGPU
	SaveInvStd  *TensorGPU
	Momentum    float32
	Eps         float32
	Training    bool
}

// NewBatchNorm1DGPU creates batch norm layer
func NewBatchNorm1DGPU(channels int) (*BatchNorm1DGPU, error) {
	bn := &BatchNorm1DGPU{
		Channels: channels,
		Momentum: 0.1,
		Eps:      1e-5,
		Training: true,
	}

	var err error

	bn.Gamma, err = NewTensorGPU(channels, 1)
	if err != nil {
		return nil, err
	}
	// Initialize gamma to 1
	ones := make([]float32, channels)
	for i := range ones {
		ones[i] = 1.0
	}
	bn.Gamma.CopyFromHost(ones)

	bn.Beta, err = NewTensorGPU(channels, 1)
	if err != nil {
		bn.Free()
		return nil, err
	}

	bn.RunningMean, err = NewTensorGPU(channels, 1)
	if err != nil {
		bn.Free()
		return nil, err
	}

	bn.RunningVar, err = NewTensorGPU(channels, 1)
	if err != nil {
		bn.Free()
		return nil, err
	}
	// Initialize running_var to 1
	bn.RunningVar.CopyFromHost(ones)

	bn.SaveMean, err = NewTensorGPU(channels, 1)
	if err != nil {
		bn.Free()
		return nil, err
	}

	bn.SaveInvStd, err = NewTensorGPU(channels, 1)
	if err != nil {
		bn.Free()
		return nil, err
	}

	return bn, nil
}

// Forward performs batch normalization
func (bn *BatchNorm1DGPU) Forward(input *TensorGPU, batch, timeSteps int) (*TensorGPU, error) {
	output, err := NewTensorGPU(batch*timeSteps, bn.Channels)
	if err != nil {
		return nil, err
	}

	training := 0
	if bn.Training {
		training = 1
	}

	C.launch_batchnorm1d_forward_fp16(
		unsafe.Pointer(input.handle),
		unsafe.Pointer(bn.Gamma.handle),
		unsafe.Pointer(bn.Beta.handle),
		unsafe.Pointer(bn.RunningMean.handle),
		unsafe.Pointer(bn.RunningVar.handle),
		unsafe.Pointer(output.handle),
		unsafe.Pointer(bn.SaveMean.handle),
		unsafe.Pointer(bn.SaveInvStd.handle),
		C.int(batch), C.int(timeSteps), C.int(bn.Channels),
		C.float(bn.Momentum), C.float(bn.Eps), C.int(training),
		nil,
	)

	return output, nil
}

// Free releases GPU memory
func (bn *BatchNorm1DGPU) Free() {
	if bn.Gamma != nil {
		bn.Gamma.Free()
	}
	if bn.Beta != nil {
		bn.Beta.Free()
	}
	if bn.RunningMean != nil {
		bn.RunningMean.Free()
	}
	if bn.RunningVar != nil {
		bn.RunningVar.Free()
	}
	if bn.SaveMean != nil {
		bn.SaveMean.Free()
	}
	if bn.SaveInvStd != nil {
		bn.SaveInvStd.Free()
	}
}

// ============================================================================
// Depthwise Separable Conv1D on GPU
// ============================================================================

// DepthwiseSeparableConv1DGPU combines depthwise and pointwise convolutions
type DepthwiseSeparableConv1DGPU struct {
	Channels    int
	OutChannels int
	KernelSize  int
	Stride      int
	Padding     int
	DepthwiseW  *TensorGPU
	DepthwiseB  *TensorGPU
	PointwiseW  *TensorGPU
	PointwiseB  *TensorGPU
}

// NewDepthwiseSeparableConv1DGPU creates depthwise separable conv layer
func NewDepthwiseSeparableConv1DGPU(channels, outChannels, kernelSize, stride int) (*DepthwiseSeparableConv1DGPU, error) {
	padding := kernelSize / 2

	dsc := &DepthwiseSeparableConv1DGPU{
		Channels:    channels,
		OutChannels: outChannels,
		KernelSize:  kernelSize,
		Stride:      stride,
		Padding:     padding,
	}

	var err error

	// Depthwise: [channels, kernel_size]
	dsc.DepthwiseW, err = NewTensorGPU(channels, kernelSize)
	if err != nil {
		return nil, err
	}

	dsc.DepthwiseB, err = NewTensorGPU(channels, 1)
	if err != nil {
		dsc.Free()
		return nil, err
	}

	// Pointwise: [out_channels, channels]
	dsc.PointwiseW, err = NewTensorGPU(outChannels, channels)
	if err != nil {
		dsc.Free()
		return nil, err
	}

	dsc.PointwiseB, err = NewTensorGPU(outChannels, 1)
	if err != nil {
		dsc.Free()
		return nil, err
	}

	return dsc, nil
}

// Forward performs depthwise separable convolution
func (dsc *DepthwiseSeparableConv1DGPU) Forward(input *TensorGPU, batch, timeIn int) (*TensorGPU, error) {
	timeOut := (timeIn+2*dsc.Padding-dsc.KernelSize)/dsc.Stride + 1

	// Depthwise convolution
	depthwiseOut, err := NewTensorGPU(batch*timeOut, dsc.Channels)
	if err != nil {
		return nil, err
	}

	C.launch_depthwise_conv1d_fp16(
		unsafe.Pointer(input.handle),
		unsafe.Pointer(dsc.DepthwiseW.handle),
		unsafe.Pointer(dsc.DepthwiseB.handle),
		unsafe.Pointer(depthwiseOut.handle),
		C.int(batch), C.int(timeIn), C.int(dsc.Channels),
		C.int(dsc.KernelSize), C.int(dsc.Stride), C.int(dsc.Padding),
		nil,
	)

	// Pointwise convolution
	output, err := NewTensorGPU(batch*timeOut, dsc.OutChannels)
	if err != nil {
		depthwiseOut.Free()
		return nil, err
	}

	C.launch_pointwise_conv1d_fp16(
		unsafe.Pointer(depthwiseOut.handle),
		unsafe.Pointer(dsc.PointwiseW.handle),
		unsafe.Pointer(dsc.PointwiseB.handle),
		unsafe.Pointer(output.handle),
		C.int(batch), C.int(timeOut), C.int(dsc.Channels), C.int(dsc.OutChannels),
		nil,
	)

	depthwiseOut.Free()
	return output, nil
}

// Free releases GPU memory
func (dsc *DepthwiseSeparableConv1DGPU) Free() {
	if dsc.DepthwiseW != nil {
		dsc.DepthwiseW.Free()
	}
	if dsc.DepthwiseB != nil {
		dsc.DepthwiseB.Free()
	}
	if dsc.PointwiseW != nil {
		dsc.PointwiseW.Free()
	}
	if dsc.PointwiseB != nil {
		dsc.PointwiseB.Free()
	}
}

// ============================================================================
// Utility
// ============================================================================

// Synchronize waits for all GPU operations to complete
func Synchronize() {
	C.cgo_cuda_sync()
}
