package gotorch

import (
	"math"
	"math/rand"
)

// ============================================================================
// Layer Interface
// ============================================================================

type Layer interface {
	Forward(input *Tensor) *Tensor
	Backward(gradOutput *Tensor) *Tensor
	Parameters() []*Tensor
	Gradients() []*Tensor
	ZeroGrad()
	Name() string
}

// ============================================================================
// Affine (Linear) Layer
// ============================================================================

type AffineLayer struct {
	name       string
	InputDim   int
	OutputDim  int
	Weight     *Tensor // [InputDim x OutputDim]
	Bias       *Tensor // [OutputDim]
	GradW      *Tensor
	GradB      *Tensor
	inputCache *Tensor
}

func NewAffineLayer(inputDim, outputDim int) *AffineLayer {
	scale := math.Sqrt(2.0 / float64(inputDim+outputDim))

	weights := NewTensor([]int{inputDim, outputDim})
	for i := range weights.Data {
		weights.Data[i] = scale * rand.NormFloat64()
	}

	return &AffineLayer{
		name:      "affine",
		InputDim:  inputDim,
		OutputDim: outputDim,
		Weight:    weights,
		Bias:      Zeros([]int{outputDim}),
		GradW:     Zeros([]int{inputDim, outputDim}),
		GradB:     Zeros([]int{outputDim}),
	}
}

func (l *AffineLayer) Name() string { return l.name }

func (l *AffineLayer) Forward(input *Tensor) *Tensor {
	l.inputCache = input.Clone()

	output, _ := MatMul(input, l.Weight)

	batchSize := input.Shape[0]
	for i := 0; i < batchSize; i++ {
		for j := 0; j < l.OutputDim; j++ {
			output.Data[i*l.OutputDim+j] += l.Bias.Data[j]
		}
	}

	return output
}

func (l *AffineLayer) Backward(gradOutput *Tensor) *Tensor {
	batchSize := l.inputCache.Shape[0]

	// Gradient w.r.t. bias: sum over batch
	for i := range l.GradB.Data {
		l.GradB.Data[i] = 0
	}
	for i := 0; i < batchSize; i++ {
		for j := 0; j < l.OutputDim; j++ {
			l.GradB.Data[j] += gradOutput.Data[i*l.OutputDim+j]
		}
	}

	// Gradient w.r.t. weights: input^T @ gradOutput
	for i := range l.GradW.Data {
		l.GradW.Data[i] = 0
	}
	for b := 0; b < batchSize; b++ {
		for i := 0; i < l.InputDim; i++ {
			for j := 0; j < l.OutputDim; j++ {
				l.GradW.Data[i*l.OutputDim+j] += l.inputCache.Data[b*l.InputDim+i] * gradOutput.Data[b*l.OutputDim+j]
			}
		}
	}

	// Gradient w.r.t. input: gradOutput @ weights^T
	gradInput := Zeros([]int{batchSize, l.InputDim})
	for b := 0; b < batchSize; b++ {
		for i := 0; i < l.InputDim; i++ {
			sum := 0.0
			for j := 0; j < l.OutputDim; j++ {
				sum += gradOutput.Data[b*l.OutputDim+j] * l.Weight.Data[i*l.OutputDim+j]
			}
			gradInput.Data[b*l.InputDim+i] = sum
		}
	}

	return gradInput
}

func (l *AffineLayer) Parameters() []*Tensor { return []*Tensor{l.Weight, l.Bias} }
func (l *AffineLayer) Gradients() []*Tensor  { return []*Tensor{l.GradW, l.GradB} }
func (l *AffineLayer) ZeroGrad() {
	for i := range l.GradW.Data {
		l.GradW.Data[i] = 0
	}
	for i := range l.GradB.Data {
		l.GradB.Data[i] = 0
	}
}

// ============================================================================
// Activation Layers
// ============================================================================

type ReLULayer struct {
	name       string
	inputCache *Tensor
}

func (l *ReLULayer) Name() string { return l.name }

func (l *ReLULayer) Forward(input *Tensor) *Tensor {
	l.inputCache = input.Clone()
	output := NewTensor(input.Shape)
	for i, v := range input.Data {
		if v > 0 {
			output.Data[i] = v
		}
	}
	return output
}

func (l *ReLULayer) Backward(gradOutput *Tensor) *Tensor {
	gradInput := NewTensor(gradOutput.Shape)
	for i := range gradOutput.Data {
		if l.inputCache.Data[i] > 0 {
			gradInput.Data[i] = gradOutput.Data[i]
		}
	}
	return gradInput
}

func (l *ReLULayer) Parameters() []*Tensor { return nil }
func (l *ReLULayer) Gradients() []*Tensor  { return nil }
func (l *ReLULayer) ZeroGrad()             {}

type SigmoidLayer struct {
	name        string
	outputCache *Tensor
}

func (l *SigmoidLayer) Name() string { return l.name }

func (l *SigmoidLayer) Forward(input *Tensor) *Tensor {
	output := NewTensor(input.Shape)
	for i, v := range input.Data {
		output.Data[i] = 1.0 / (1.0 + math.Exp(-v))
	}
	l.outputCache = output
	return output
}

func (l *SigmoidLayer) Backward(gradOutput *Tensor) *Tensor {
	gradInput := NewTensor(gradOutput.Shape)
	for i := range gradOutput.Data {
		s := l.outputCache.Data[i]
		gradInput.Data[i] = gradOutput.Data[i] * s * (1.0 - s)
	}
	return gradInput
}

func (l *SigmoidLayer) Parameters() []*Tensor { return nil }
func (l *SigmoidLayer) Gradients() []*Tensor  { return nil }
func (l *SigmoidLayer) ZeroGrad()             {}

type SoftmaxLayer struct {
	name        string
	outputCache *Tensor
}

func (l *SoftmaxLayer) Name() string { return l.name }

func (l *SoftmaxLayer) Forward(input *Tensor) *Tensor {
	l.outputCache = Softmax(input)
	return l.outputCache
}

func (l *SoftmaxLayer) Backward(gradOutput *Tensor) *Tensor {
	// Simplified: assumes cross-entropy loss follows
	return gradOutput.Clone()
}

func (l *SoftmaxLayer) Parameters() []*Tensor { return nil }
func (l *SoftmaxLayer) Gradients() []*Tensor  { return nil }
func (l *SoftmaxLayer) ZeroGrad()             {}

// ============================================================================
// BatchNorm Layer
// ============================================================================

type BatchNormLayer struct {
	name        string
	NumFeatures int
	Gamma       *Tensor
	Beta        *Tensor
	RunningMean *Tensor
	RunningVar  *Tensor
	GradGamma   *Tensor
	GradBeta    *Tensor
	Momentum    float64
	Eps         float64
	Training    bool

	// Cache
	inputCache    *Tensor
	meanCache     *Tensor
	varCache      *Tensor
	normCache     *Tensor
}

func NewBatchNormLayer(numFeatures int) *BatchNormLayer {
	return &BatchNormLayer{
		name:        "batchnorm",
		NumFeatures: numFeatures,
		Gamma:       Ones([]int{numFeatures}),
		Beta:        Zeros([]int{numFeatures}),
		RunningMean: Zeros([]int{numFeatures}),
		RunningVar:  Ones([]int{numFeatures}),
		GradGamma:   Zeros([]int{numFeatures}),
		GradBeta:    Zeros([]int{numFeatures}),
		Momentum:    0.1,
		Eps:         1e-5,
		Training:    true,
	}
}

func (l *BatchNormLayer) Name() string { return l.name }

func (l *BatchNormLayer) Forward(input *Tensor) *Tensor {
	l.inputCache = input.Clone()

	batchSize := input.Shape[0]
	features := l.NumFeatures

	output := NewTensor(input.Shape)

	if l.Training {
		// Compute batch mean and variance
		l.meanCache = Zeros([]int{features})
		l.varCache = Zeros([]int{features})

		for f := 0; f < features; f++ {
			sum := 0.0
			for b := 0; b < batchSize; b++ {
				sum += input.Data[b*features+f]
			}
			mean := sum / float64(batchSize)
			l.meanCache.Data[f] = mean

			varSum := 0.0
			for b := 0; b < batchSize; b++ {
				diff := input.Data[b*features+f] - mean
				varSum += diff * diff
			}
			l.varCache.Data[f] = varSum / float64(batchSize)

			// Update running stats
			l.RunningMean.Data[f] = (1-l.Momentum)*l.RunningMean.Data[f] + l.Momentum*mean
			l.RunningVar.Data[f] = (1-l.Momentum)*l.RunningVar.Data[f] + l.Momentum*l.varCache.Data[f]
		}

		// Normalize
		l.normCache = NewTensor(input.Shape)
		for b := 0; b < batchSize; b++ {
			for f := 0; f < features; f++ {
				norm := (input.Data[b*features+f] - l.meanCache.Data[f]) / math.Sqrt(l.varCache.Data[f]+l.Eps)
				l.normCache.Data[b*features+f] = norm
				output.Data[b*features+f] = l.Gamma.Data[f]*norm + l.Beta.Data[f]
			}
		}
	} else {
		// Use running stats
		for b := 0; b < batchSize; b++ {
			for f := 0; f < features; f++ {
				norm := (input.Data[b*features+f] - l.RunningMean.Data[f]) / math.Sqrt(l.RunningVar.Data[f]+l.Eps)
				output.Data[b*features+f] = l.Gamma.Data[f]*norm + l.Beta.Data[f]
			}
		}
	}

	return output
}

func (l *BatchNormLayer) Backward(gradOutput *Tensor) *Tensor {
	batchSize := l.inputCache.Shape[0]
	features := l.NumFeatures

	gradInput := Zeros(l.inputCache.Shape)

	for f := 0; f < features; f++ {
		l.GradGamma.Data[f] = 0
		l.GradBeta.Data[f] = 0

		for b := 0; b < batchSize; b++ {
			l.GradGamma.Data[f] += gradOutput.Data[b*features+f] * l.normCache.Data[b*features+f]
			l.GradBeta.Data[f] += gradOutput.Data[b*features+f]
		}
	}

	// Gradient w.r.t. input (simplified)
	for b := 0; b < batchSize; b++ {
		for f := 0; f < features; f++ {
			invStd := 1.0 / math.Sqrt(l.varCache.Data[f]+l.Eps)
			gradInput.Data[b*features+f] = gradOutput.Data[b*features+f] * l.Gamma.Data[f] * invStd
		}
	}

	return gradInput
}

func (l *BatchNormLayer) Parameters() []*Tensor { return []*Tensor{l.Gamma, l.Beta} }
func (l *BatchNormLayer) Gradients() []*Tensor  { return []*Tensor{l.GradGamma, l.GradBeta} }
func (l *BatchNormLayer) ZeroGrad() {
	for i := range l.GradGamma.Data {
		l.GradGamma.Data[i] = 0
	}
	for i := range l.GradBeta.Data {
		l.GradBeta.Data[i] = 0
	}
}

// ============================================================================
// Dropout Layer
// ============================================================================

type DropoutLayer struct {
	name     string
	P        float64 // Dropout probability
	Training bool
	mask     []bool
}

func NewDropoutLayer(p float64) *DropoutLayer {
	return &DropoutLayer{
		name:     "dropout",
		P:        p,
		Training: true,
	}
}

func (l *DropoutLayer) Name() string { return l.name }

func (l *DropoutLayer) Forward(input *Tensor) *Tensor {
	if !l.Training || l.P == 0 {
		return input.Clone()
	}

	output := NewTensor(input.Shape)
	l.mask = make([]bool, len(input.Data))
	scale := 1.0 / (1.0 - l.P)

	for i := range input.Data {
		if rand.Float64() > l.P {
			l.mask[i] = true
			output.Data[i] = input.Data[i] * scale
		}
	}

	return output
}

func (l *DropoutLayer) Backward(gradOutput *Tensor) *Tensor {
	if !l.Training || l.P == 0 {
		return gradOutput.Clone()
	}

	gradInput := NewTensor(gradOutput.Shape)
	scale := 1.0 / (1.0 - l.P)

	for i := range gradOutput.Data {
		if l.mask[i] {
			gradInput.Data[i] = gradOutput.Data[i] * scale
		}
	}

	return gradInput
}

func (l *DropoutLayer) Parameters() []*Tensor { return nil }
func (l *DropoutLayer) Gradients() []*Tensor  { return nil }
func (l *DropoutLayer) ZeroGrad()             {}

// ============================================================================
// TDNN Layer (Time-Delay Neural Network)
// ============================================================================

type TDNNLayer struct {
	name       string
	InputDim   int
	OutputDim  int
	Context    []int // e.g., [-2, -1, 0, 1, 2]
	Weight     *Tensor
	Bias       *Tensor
	GradW      *Tensor
	GradB      *Tensor
	inputCache *Tensor
}

func NewTDNNLayer(inputDim, outputDim int, context []int) *TDNNLayer {
	splicedDim := inputDim * len(context)
	scale := math.Sqrt(2.0 / float64(splicedDim+outputDim))

	weights := NewTensor([]int{splicedDim, outputDim})
	for i := range weights.Data {
		weights.Data[i] = scale * rand.NormFloat64()
	}

	return &TDNNLayer{
		name:      "tdnn",
		InputDim:  inputDim,
		OutputDim: outputDim,
		Context:   context,
		Weight:    weights,
		Bias:      Zeros([]int{outputDim}),
		GradW:     Zeros([]int{splicedDim, outputDim}),
		GradB:     Zeros([]int{outputDim}),
	}
}

func (l *TDNNLayer) Name() string { return l.name }

func (l *TDNNLayer) Forward(input *Tensor) *Tensor {
	l.inputCache = input.Clone()

	batch := input.Shape[0]
	timeSteps := input.Shape[1]

	output := Zeros([]int{batch, timeSteps, l.OutputDim})

	for b := 0; b < batch; b++ {
		for t := 0; t < timeSteps; t++ {
			for o := 0; o < l.OutputDim; o++ {
				sum := l.Bias.Data[o]

				for ci, offset := range l.Context {
					tCtx := t + offset
					if tCtx < 0 {
						tCtx = 0
					} else if tCtx >= timeSteps {
						tCtx = timeSteps - 1
					}

					for i := 0; i < l.InputDim; i++ {
						inIdx := b*timeSteps*l.InputDim + tCtx*l.InputDim + i
						wIdx := (ci*l.InputDim+i)*l.OutputDim + o
						sum += input.Data[inIdx] * l.Weight.Data[wIdx]
					}
				}

				outIdx := b*timeSteps*l.OutputDim + t*l.OutputDim + o
				output.Data[outIdx] = sum
			}
		}
	}

	return output
}

func (l *TDNNLayer) Backward(gradOutput *Tensor) *Tensor {
	batch := l.inputCache.Shape[0]
	timeSteps := l.inputCache.Shape[1]

	gradInput := Zeros(l.inputCache.Shape)

	// Zero gradients
	for i := range l.GradW.Data {
		l.GradW.Data[i] = 0
	}
	for i := range l.GradB.Data {
		l.GradB.Data[i] = 0
	}

	for b := 0; b < batch; b++ {
		for t := 0; t < timeSteps; t++ {
			for o := 0; o < l.OutputDim; o++ {
				gradIdx := b*timeSteps*l.OutputDim + t*l.OutputDim + o
				grad := gradOutput.Data[gradIdx]

				l.GradB.Data[o] += grad

				for ci, offset := range l.Context {
					tCtx := t + offset
					if tCtx < 0 {
						tCtx = 0
					} else if tCtx >= timeSteps {
						tCtx = timeSteps - 1
					}

					for i := 0; i < l.InputDim; i++ {
						inIdx := b*timeSteps*l.InputDim + tCtx*l.InputDim + i
						wIdx := (ci*l.InputDim+i)*l.OutputDim + o

						l.GradW.Data[wIdx] += l.inputCache.Data[inIdx] * grad
						gradInput.Data[inIdx] += l.Weight.Data[wIdx] * grad
					}
				}
			}
		}
	}

	return gradInput
}

func (l *TDNNLayer) Parameters() []*Tensor { return []*Tensor{l.Weight, l.Bias} }
func (l *TDNNLayer) Gradients() []*Tensor  { return []*Tensor{l.GradW, l.GradB} }
func (l *TDNNLayer) ZeroGrad() {
	for i := range l.GradW.Data {
		l.GradW.Data[i] = 0
	}
	for i := range l.GradB.Data {
		l.GradB.Data[i] = 0
	}
}
