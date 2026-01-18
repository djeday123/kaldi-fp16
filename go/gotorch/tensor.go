// Package gotorch provides native Go tensor operations with optional GPU acceleration.
package gotorch

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
)

// DType represents tensor data type
type DType int

const (
	Float64 DType = iota
	Float32
	Float16
	Int32
	Int64
)

// Device represents computation device
type Device int

const (
	CPU Device = iota
	CUDA
)

// Tensor represents a multi-dimensional array
type Tensor struct {
	Data    []float64
	Shape   []int
	Strides []int
	Dtype   DType
	Device  Device
	GpuPtr  uintptr
	mu      sync.RWMutex
}

// computeStrides calculates strides for row-major layout
func computeStrides(shape []int) []int {
	if len(shape) == 0 {
		return nil
	}
	strides := make([]int, len(shape))
	strides[len(shape)-1] = 1
	for i := len(shape) - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * shape[i+1]
	}
	return strides
}

// NewTensor creates a new tensor with given shape
func NewTensor(shape []int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	return &Tensor{
		Data:    make([]float64, size),
		Shape:   append([]int{}, shape...),
		Strides: computeStrides(shape),
		Dtype:   Float64,
		Device:  CPU,
	}
}

// Zeros creates zero-initialized tensor
func Zeros(shape []int) *Tensor {
	return NewTensor(shape)
}

// Ones creates tensor filled with ones
func Ones(shape []int) *Tensor {
	t := NewTensor(shape)
	for i := range t.Data {
		t.Data[i] = 1.0
	}
	return t
}

// Rand creates tensor with uniform random values [0, 1)
func Rand(shape []int) *Tensor {
	t := NewTensor(shape)
	for i := range t.Data {
		t.Data[i] = rand.Float64()
	}
	return t
}

// Randn creates tensor with normal distribution (mean=0, std=1)
func Randn(shape []int) *Tensor {
	t := NewTensor(shape)
	for i := range t.Data {
		t.Data[i] = rand.NormFloat64()
	}
	return t
}

// Eye creates identity matrix
func Eye(n int) *Tensor {
	t := Zeros([]int{n, n})
	for i := 0; i < n; i++ {
		t.Data[i*n+i] = 1.0
	}
	return t
}

// Arange creates 1D tensor with values from start to end
func Arange(start, end, step float64) *Tensor {
	if step == 0 {
		panic("step cannot be zero")
	}
	n := int(math.Ceil((end - start) / step))
	if n <= 0 {
		return NewTensor([]int{0})
	}
	t := NewTensor([]int{n})
	for i := 0; i < n; i++ {
		t.Data[i] = start + float64(i)*step
	}
	return t
}

// Size returns total number of elements
func (t *Tensor) Size() int {
	if len(t.Shape) == 0 {
		return 0
	}
	size := 1
	for _, dim := range t.Shape {
		size *= dim
	}
	return size
}

// Clone creates a deep copy
func (t *Tensor) Clone() *Tensor {
	t.mu.RLock()
	defer t.mu.RUnlock()

	clone := &Tensor{
		Shape:   append([]int{}, t.Shape...),
		Strides: append([]int{}, t.Strides...),
		Dtype:   t.Dtype,
		Device:  t.Device,
		Data:    make([]float64, len(t.Data)),
	}
	copy(clone.Data, t.Data)
	return clone
}

// Scale multiplies all elements by scalar
func (t *Tensor) Scale(s float64) {
	for i := range t.Data {
		t.Data[i] *= s
	}
}

// Fill sets all elements to value
func (t *Tensor) Fill(value float64) {
	for i := range t.Data {
		t.Data[i] = value
	}
}

// Reshape returns new view with different shape
func (t *Tensor) Reshape(newShape []int) (*Tensor, error) {
	totalSize := t.Size()
	newSize := 1
	unknownIdx := -1
	
	for i, dim := range newShape {
		if dim == -1 {
			unknownIdx = i
		} else {
			newSize *= dim
		}
	}
	
	if unknownIdx != -1 {
		newShape[unknownIdx] = totalSize / newSize
		newSize = totalSize
	}
	
	if newSize != totalSize {
		return nil, fmt.Errorf("cannot reshape tensor of size %d into shape %v", totalSize, newShape)
	}

	return &Tensor{
		Data:    t.Data,
		Shape:   append([]int{}, newShape...),
		Strides: computeStrides(newShape),
		Dtype:   t.Dtype,
		Device:  t.Device,
	}, nil
}

// Flatten returns 1D view
func (t *Tensor) Flatten() *Tensor {
	result, _ := t.Reshape([]int{t.Size()})
	return result
}

// Transpose returns transposed 2D tensor
func (t *Tensor) Transpose() (*Tensor, error) {
	if len(t.Shape) != 2 {
		return nil, fmt.Errorf("transpose only for 2D tensors")
	}

	rows, cols := t.Shape[0], t.Shape[1]
	result := NewTensor([]int{cols, rows})

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result.Data[j*rows+i] = t.Data[i*cols+j]
		}
	}
	return result, nil
}

// T is shorthand for Transpose
func (t *Tensor) T() *Tensor {
	result, _ := t.Transpose()
	return result
}

// At returns element at given indices
func (t *Tensor) At(indices ...int) float64 {
	offset := 0
	for i, idx := range indices {
		offset += idx * t.Strides[i]
	}
	return t.Data[offset]
}

// Set sets element at given indices
func (t *Tensor) Set(value float64, indices ...int) {
	offset := 0
	for i, idx := range indices {
		offset += idx * t.Strides[i]
	}
	t.Data[offset] = value
}

// String returns string representation
func (t *Tensor) String() string {
	return fmt.Sprintf("Tensor%v", t.Shape)
}

// Rows returns first dimension
func (t *Tensor) Rows() int {
	if len(t.Shape) < 1 {
		return 0
	}
	return t.Shape[0]
}

// Cols returns second dimension
func (t *Tensor) Cols() int {
	if len(t.Shape) < 2 {
		return 1
	}
	return t.Shape[1]
}
