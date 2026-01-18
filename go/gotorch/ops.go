package gotorch

import (
	"fmt"
	"math"
	"runtime"
	"sync"
)

// ============================================================================
// Matrix Multiplication
// ============================================================================

// MatMul performs matrix multiplication: C = A @ B
func MatMul(a, b *Tensor) (*Tensor, error) {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		return nil, fmt.Errorf("matmul requires 2D tensors")
	}
	if a.Shape[1] != b.Shape[0] {
		return nil, fmt.Errorf("incompatible shapes: %v @ %v", a.Shape, b.Shape)
	}

	M, K := a.Shape[0], a.Shape[1]
	N := b.Shape[1]

	c := Zeros([]int{M, N})

	if M*N*K > 10000 {
		matmulParallel(a.Data, b.Data, c.Data, M, K, N)
	} else {
		matmulNaive(a.Data, b.Data, c.Data, M, K, N)
	}

	return c, nil
}

func matmulNaive(a, b, c []float64, M, K, N int) {
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			sum := 0.0
			for k := 0; k < K; k++ {
				sum += a[i*K+k] * b[k*N+j]
			}
			c[i*N+j] = sum
		}
	}
}

func matmulParallel(a, b, c []float64, M, K, N int) {
	numWorkers := runtime.NumCPU()
	if numWorkers > M {
		numWorkers = M
	}

	var wg sync.WaitGroup
	rowsPerWorker := (M + numWorkers - 1) / numWorkers

	for w := 0; w < numWorkers; w++ {
		startRow := w * rowsPerWorker
		endRow := startRow + rowsPerWorker
		if endRow > M {
			endRow = M
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				for j := 0; j < N; j++ {
					sum := 0.0
					for k := 0; k < K; k++ {
						sum += a[i*K+k] * b[k*N+j]
					}
					c[i*N+j] = sum
				}
			}
		}(startRow, endRow)
	}

	wg.Wait()
}

// BatchedMatMul for 3D tensors: [batch, M, K] @ [batch, K, N]
func BatchedMatMul(a, b *Tensor) (*Tensor, error) {
	if len(a.Shape) != 3 || len(b.Shape) != 3 {
		return nil, fmt.Errorf("batched matmul requires 3D tensors")
	}
	if a.Shape[0] != b.Shape[0] {
		return nil, fmt.Errorf("batch sizes must match")
	}
	if a.Shape[2] != b.Shape[1] {
		return nil, fmt.Errorf("incompatible shapes")
	}

	batch := a.Shape[0]
	M, K := a.Shape[1], a.Shape[2]
	N := b.Shape[2]

	c := Zeros([]int{batch, M, N})

	for bIdx := 0; bIdx < batch; bIdx++ {
		aOffset := bIdx * M * K
		bOffset := bIdx * K * N
		cOffset := bIdx * M * N

		for i := 0; i < M; i++ {
			for j := 0; j < N; j++ {
				sum := 0.0
				for k := 0; k < K; k++ {
					sum += a.Data[aOffset+i*K+k] * b.Data[bOffset+k*N+j]
				}
				c.Data[cOffset+i*N+j] = sum
			}
		}
	}

	return c, nil
}

// ============================================================================
// Element-wise Operations
// ============================================================================

func Add(a, b *Tensor) *Tensor {
	if len(a.Data) != len(b.Data) {
		panic("tensor size mismatch")
	}
	c := NewTensor(a.Shape)
	for i := range a.Data {
		c.Data[i] = a.Data[i] + b.Data[i]
	}
	return c
}

func Sub(a, b *Tensor) *Tensor {
	if len(a.Data) != len(b.Data) {
		panic("tensor size mismatch")
	}
	c := NewTensor(a.Shape)
	for i := range a.Data {
		c.Data[i] = a.Data[i] - b.Data[i]
	}
	return c
}

func Mul(a, b *Tensor) *Tensor {
	if len(a.Data) != len(b.Data) {
		panic("tensor size mismatch")
	}
	c := NewTensor(a.Shape)
	for i := range a.Data {
		c.Data[i] = a.Data[i] * b.Data[i]
	}
	return c
}

func Div(a, b *Tensor) *Tensor {
	if len(a.Data) != len(b.Data) {
		panic("tensor size mismatch")
	}
	c := NewTensor(a.Shape)
	for i := range a.Data {
		c.Data[i] = a.Data[i] / b.Data[i]
	}
	return c
}

func ScaleTensor(t *Tensor, s float64) *Tensor {
	c := NewTensor(t.Shape)
	for i := range t.Data {
		c.Data[i] = t.Data[i] * s
	}
	return c
}

func AddScalar(t *Tensor, s float64) *Tensor {
	c := NewTensor(t.Shape)
	for i := range t.Data {
		c.Data[i] = t.Data[i] + s
	}
	return c
}

// ============================================================================
// Activation Functions
// ============================================================================

func ReLU(t *Tensor) *Tensor {
	c := NewTensor(t.Shape)
	for i, v := range t.Data {
		if v > 0 {
			c.Data[i] = v
		}
	}
	return c
}

func ReLUBackward(gradOutput, input *Tensor) *Tensor {
	c := NewTensor(gradOutput.Shape)
	for i := range gradOutput.Data {
		if input.Data[i] > 0 {
			c.Data[i] = gradOutput.Data[i]
		}
	}
	return c
}

func Sigmoid(t *Tensor) *Tensor {
	c := NewTensor(t.Shape)
	for i, v := range t.Data {
		c.Data[i] = 1.0 / (1.0 + math.Exp(-v))
	}
	return c
}

func SigmoidBackward(gradOutput, output *Tensor) *Tensor {
	c := NewTensor(gradOutput.Shape)
	for i := range gradOutput.Data {
		c.Data[i] = gradOutput.Data[i] * output.Data[i] * (1.0 - output.Data[i])
	}
	return c
}

func Tanh(t *Tensor) *Tensor {
	c := NewTensor(t.Shape)
	for i, v := range t.Data {
		c.Data[i] = math.Tanh(v)
	}
	return c
}

func TanhBackward(gradOutput, output *Tensor) *Tensor {
	c := NewTensor(gradOutput.Shape)
	for i := range gradOutput.Data {
		c.Data[i] = gradOutput.Data[i] * (1.0 - output.Data[i]*output.Data[i])
	}
	return c
}

// ============================================================================
// Softmax and LogSoftmax
// ============================================================================

// Softmax along last dimension
func Softmax(t *Tensor) *Tensor {
	if len(t.Shape) < 2 {
		panic("softmax requires at least 2D tensor")
	}

	c := NewTensor(t.Shape)
	lastDim := t.Shape[len(t.Shape)-1]
	batchSize := t.Size() / lastDim

	for b := 0; b < batchSize; b++ {
		offset := b * lastDim

		// Find max for numerical stability
		maxVal := t.Data[offset]
		for i := 1; i < lastDim; i++ {
			if t.Data[offset+i] > maxVal {
				maxVal = t.Data[offset+i]
			}
		}

		// Compute exp and sum
		sum := 0.0
		for i := 0; i < lastDim; i++ {
			c.Data[offset+i] = math.Exp(t.Data[offset+i] - maxVal)
			sum += c.Data[offset+i]
		}

		// Normalize
		for i := 0; i < lastDim; i++ {
			c.Data[offset+i] /= sum
		}
	}

	return c
}

func LogSoftmax(t *Tensor) *Tensor {
	c := Softmax(t)
	for i := range c.Data {
		c.Data[i] = math.Log(c.Data[i] + 1e-10)
	}
	return c
}

// ============================================================================
// Loss Functions
// ============================================================================

// CrossEntropyLoss computes cross-entropy from logits and targets
// pred: [batch, time, classes] after softmax
// target: [batch, time] with class indices
func CrossEntropyLoss(pred, target *Tensor) float64 {
	batch := pred.Shape[0]
	timeSteps := pred.Shape[1]
	classes := pred.Shape[2]

	loss := 0.0
	count := 0

	for b := 0; b < batch; b++ {
		for t := 0; t < timeSteps; t++ {
			targetIdx := int(target.Data[b*timeSteps+t])
			if targetIdx >= 0 && targetIdx < classes {
				idx := b*timeSteps*classes + t*classes + targetIdx
				prob := pred.Data[idx]
				if prob > 1e-10 {
					loss -= math.Log(prob)
				} else {
					loss -= math.Log(1e-10)
				}
				count++
			}
		}
	}

	if count > 0 {
		return loss / float64(count)
	}
	return 0
}

// MSELoss computes mean squared error
func MSELoss(pred, target *Tensor) float64 {
	if len(pred.Data) != len(target.Data) {
		panic("size mismatch")
	}

	sum := 0.0
	for i := range pred.Data {
		diff := pred.Data[i] - target.Data[i]
		sum += diff * diff
	}
	return sum / float64(len(pred.Data))
}

// ============================================================================
// Reduction Operations
// ============================================================================

func Sum(t *Tensor) float64 {
	sum := 0.0
	for _, v := range t.Data {
		sum += v
	}
	return sum
}

func Mean(t *Tensor) float64 {
	return Sum(t) / float64(len(t.Data))
}

func Max(t *Tensor) float64 {
	if len(t.Data) == 0 {
		return 0
	}
	maxVal := t.Data[0]
	for _, v := range t.Data[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	return maxVal
}

func Min(t *Tensor) float64 {
	if len(t.Data) == 0 {
		return 0
	}
	minVal := t.Data[0]
	for _, v := range t.Data[1:] {
		if v < minVal {
			minVal = v
		}
	}
	return minVal
}

func Argmax(t *Tensor) int {
	if len(t.Data) == 0 {
		return -1
	}
	maxIdx := 0
	maxVal := t.Data[0]
	for i, v := range t.Data[1:] {
		if v > maxVal {
			maxVal = v
			maxIdx = i + 1
		}
	}
	return maxIdx
}

// SumAxis sums along specified axis
func SumAxis(t *Tensor, axis int) *Tensor {
	if axis < 0 || axis >= len(t.Shape) {
		panic("invalid axis")
	}

	newShape := make([]int, 0, len(t.Shape)-1)
	for i, dim := range t.Shape {
		if i != axis {
			newShape = append(newShape, dim)
		}
	}

	if len(newShape) == 0 {
		newShape = []int{1}
	}

	result := Zeros(newShape)

	// Generic implementation
	axisSize := t.Shape[axis]
	outerSize := 1
	innerSize := 1

	for i := 0; i < axis; i++ {
		outerSize *= t.Shape[i]
	}
	for i := axis + 1; i < len(t.Shape); i++ {
		innerSize *= t.Shape[i]
	}

	for outer := 0; outer < outerSize; outer++ {
		for inner := 0; inner < innerSize; inner++ {
			sum := 0.0
			for a := 0; a < axisSize; a++ {
				idx := outer*axisSize*innerSize + a*innerSize + inner
				sum += t.Data[idx]
			}
			resultIdx := outer*innerSize + inner
			result.Data[resultIdx] = sum
		}
	}

	return result
}

// ============================================================================
// Vector Operations
// ============================================================================

func Dot(a, b *Tensor) float64 {
	if len(a.Data) != len(b.Data) {
		panic("size mismatch")
	}
	sum := 0.0
	for i := range a.Data {
		sum += a.Data[i] * b.Data[i]
	}
	return sum
}

func Norm(t *Tensor) float64 {
	return math.Sqrt(Dot(t, t))
}

func Normalize(t *Tensor) *Tensor {
	norm := Norm(t)
	if norm < 1e-10 {
		return t.Clone()
	}
	return ScaleTensor(t, 1.0/norm)
}

// ============================================================================
// Clipping
// ============================================================================

func Clip(t *Tensor, minVal, maxVal float64) *Tensor {
	c := NewTensor(t.Shape)
	for i, v := range t.Data {
		if v < minVal {
			c.Data[i] = minVal
		} else if v > maxVal {
			c.Data[i] = maxVal
		} else {
			c.Data[i] = v
		}
	}
	return c
}

func ClipGradNorm(grads []*Tensor, maxNorm float64) float64 {
	totalNorm := 0.0
	for _, g := range grads {
		for _, v := range g.Data {
			totalNorm += v * v
		}
	}
	totalNorm = math.Sqrt(totalNorm)

	if totalNorm > maxNorm {
		scale := maxNorm / totalNorm
		for _, g := range grads {
			for i := range g.Data {
				g.Data[i] *= scale
			}
		}
	}

	return totalNorm
}
