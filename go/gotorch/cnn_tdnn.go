// CNN-TDNN Architecture for Kaldi FP16
// Combines CNN for local feature extraction with TDNN for temporal modeling
// Compatible with pre-trained HMM-DNN alignments

package gotorch

import (
	"fmt"
	"math"
	"math/rand"
)

// Conv1DLayer implements 1D convolution for speech features
type Conv1DLayer struct {
	InChannels  int
	OutChannels int
	KernelSize  int
	Stride      int
	Padding     int
	Dilation    int

	Weight *Tensor // [OutChannels, InChannels, KernelSize]
	Bias   *Tensor // [OutChannels]

	GradWeight *Tensor
	GradBias   *Tensor

	// Cache for backward
	inputCache *Tensor
	name       string
}

// NewConv1DLayer creates a new 1D convolution layer
func NewConv1DLayer(inChannels, outChannels, kernelSize int, opts ...Conv1DOption) *Conv1DLayer {
	layer := &Conv1DLayer{
		InChannels:  inChannels,
		OutChannels: outChannels,
		KernelSize:  kernelSize,
		Stride:      1,
		Padding:     0,
		Dilation:    1,
		name:        "conv1d",
	}

	for _, opt := range opts {
		opt(layer)
	}

	// Initialize weights with Kaiming/He initialization
	fanIn := float64(inChannels * kernelSize)
	std := math.Sqrt(2.0 / fanIn)

	layer.Weight = Randn([]int{outChannels, inChannels, kernelSize})
	layer.Weight.Scale(std)

	layer.Bias = Zeros([]int{outChannels})

	layer.GradWeight = Zeros([]int{outChannels, inChannels, kernelSize})
	layer.GradBias = Zeros([]int{outChannels})

	return layer
}

type Conv1DOption func(*Conv1DLayer)

func WithStride(stride int) Conv1DOption {
	return func(l *Conv1DLayer) { l.Stride = stride }
}

func WithPadding(padding int) Conv1DOption {
	return func(l *Conv1DLayer) { l.Padding = padding }
}

func WithDilation(dilation int) Conv1DOption {
	return func(l *Conv1DLayer) { l.Dilation = dilation }
}

func WithConvName(name string) Conv1DOption {
	return func(l *Conv1DLayer) { l.name = name }
}

// Forward pass for Conv1D
// Input: [batch, time, channels] or [batch, channels, time] depending on format
// We use [batch, time, channels] to match Kaldi feature format
func (l *Conv1DLayer) Forward(input *Tensor) *Tensor {
	l.inputCache = input.Clone()

	batch := input.Shape[0]
	timeIn := input.Shape[1]

	// Calculate output time dimension
	timeOut := (timeIn+2*l.Padding-l.Dilation*(l.KernelSize-1)-1)/l.Stride + 1

	output := Zeros([]int{batch, timeOut, l.OutChannels})

	// Convolution operation
	for b := 0; b < batch; b++ {
		for tOut := 0; tOut < timeOut; tOut++ {
			for oc := 0; oc < l.OutChannels; oc++ {
				sum := float64(0)

				for k := 0; k < l.KernelSize; k++ {
					tIn := tOut*l.Stride - l.Padding + k*l.Dilation

					if tIn >= 0 && tIn < timeIn {
						for ic := 0; ic < l.InChannels; ic++ {
							inputIdx := b*timeIn*l.InChannels + tIn*l.InChannels + ic
							weightIdx := oc*l.InChannels*l.KernelSize + ic*l.KernelSize + k
							sum += input.Data[inputIdx] * l.Weight.Data[weightIdx]
						}
					}
				}

				// Add bias
				sum += l.Bias.Data[oc]

				outputIdx := b*timeOut*l.OutChannels + tOut*l.OutChannels + oc
				output.Data[outputIdx] = sum
			}
		}
	}

	return output
}

// Backward pass for Conv1D
func (l *Conv1DLayer) Backward(gradOutput *Tensor) *Tensor {
	batch := l.inputCache.Shape[0]
	timeIn := l.inputCache.Shape[1]
	timeOut := gradOutput.Shape[1]

	gradInput := Zeros(l.inputCache.Shape)

	// Zero gradients
	for i := range l.GradWeight.Data {
		l.GradWeight.Data[i] = 0
	}
	for i := range l.GradBias.Data {
		l.GradBias.Data[i] = 0
	}

	for b := 0; b < batch; b++ {
		for tOut := 0; tOut < timeOut; tOut++ {
			for oc := 0; oc < l.OutChannels; oc++ {
				gradIdx := b*timeOut*l.OutChannels + tOut*l.OutChannels + oc
				grad := gradOutput.Data[gradIdx]

				// Gradient w.r.t. bias
				l.GradBias.Data[oc] += grad

				for k := 0; k < l.KernelSize; k++ {
					tIn := tOut*l.Stride - l.Padding + k*l.Dilation

					if tIn >= 0 && tIn < timeIn {
						for ic := 0; ic < l.InChannels; ic++ {
							inputIdx := b*timeIn*l.InChannels + tIn*l.InChannels + ic
							weightIdx := oc*l.InChannels*l.KernelSize + ic*l.KernelSize + k

							// Gradient w.r.t. weight
							l.GradWeight.Data[weightIdx] += l.inputCache.Data[inputIdx] * grad

							// Gradient w.r.t. input
							gradInput.Data[inputIdx] += l.Weight.Data[weightIdx] * grad
						}
					}
				}
			}
		}
	}

	return gradInput
}

func (l *Conv1DLayer) Parameters() []*Tensor { return []*Tensor{l.Weight, l.Bias} }
func (l *Conv1DLayer) Gradients() []*Tensor  { return []*Tensor{l.GradWeight, l.GradBias} }
func (l *Conv1DLayer) ZeroGrad() {
	for i := range l.GradWeight.Data {
		l.GradWeight.Data[i] = 0
	}
	for i := range l.GradBias.Data {
		l.GradBias.Data[i] = 0
	}
}
func (l *Conv1DLayer) Name() string { return l.name }

// ============================================================================
// MaxPool1D Layer
// ============================================================================

type MaxPool1DLayer struct {
	KernelSize int
	Stride     int

	indicesCache []int
	inputShape   []int
	name         string
}

func NewMaxPool1DLayer(kernelSize, stride int) *MaxPool1DLayer {
	return &MaxPool1DLayer{
		KernelSize: kernelSize,
		Stride:     stride,
		name:       "maxpool1d",
	}
}

func (l *MaxPool1DLayer) Forward(input *Tensor) *Tensor {
	batch := input.Shape[0]
	timeIn := input.Shape[1]
	channels := input.Shape[2]

	l.inputShape = input.Shape

	timeOut := (timeIn-l.KernelSize)/l.Stride + 1

	output := Zeros([]int{batch, timeOut, channels})
	l.indicesCache = make([]int, batch*timeOut*channels)

	for b := 0; b < batch; b++ {
		for tOut := 0; tOut < timeOut; tOut++ {
			for c := 0; c < channels; c++ {
				maxVal := math.Inf(-1)
				maxIdx := 0

				for k := 0; k < l.KernelSize; k++ {
					tIn := tOut*l.Stride + k
					idx := b*timeIn*channels + tIn*channels + c
					val := input.Data[idx]

					if val > maxVal {
						maxVal = val
						maxIdx = tIn
					}
				}

				outIdx := b*timeOut*channels + tOut*channels + c
				output.Data[outIdx] = maxVal
				l.indicesCache[outIdx] = maxIdx
			}
		}
	}

	return output
}

func (l *MaxPool1DLayer) Backward(gradOutput *Tensor) *Tensor {
	batch := gradOutput.Shape[0]
	timeOut := gradOutput.Shape[1]
	channels := gradOutput.Shape[2]
	timeIn := l.inputShape[1]

	gradInput := Zeros(l.inputShape)

	for b := 0; b < batch; b++ {
		for tOut := 0; tOut < timeOut; tOut++ {
			for c := 0; c < channels; c++ {
				outIdx := b*timeOut*channels + tOut*channels + c
				maxT := l.indicesCache[outIdx]
				inIdx := b*timeIn*channels + maxT*channels + c
				gradInput.Data[inIdx] += gradOutput.Data[outIdx]
			}
		}
	}

	return gradInput
}

func (l *MaxPool1DLayer) Parameters() []*Tensor { return nil }
func (l *MaxPool1DLayer) Gradients() []*Tensor  { return nil }
func (l *MaxPool1DLayer) ZeroGrad()             {}
func (l *MaxPool1DLayer) Name() string          { return l.name }

// ============================================================================
// Statistics Pooling for x-vector style embeddings
// ============================================================================

type StatsPoolingLayer struct {
	inputCache *Tensor
	name       string
}

func NewStatsPoolingLayer() *StatsPoolingLayer {
	return &StatsPoolingLayer{name: "stats_pooling"}
}

// Forward computes mean and std over time dimension
// Input: [batch, time, channels]
// Output: [batch, 2*channels] (concatenated mean and std)
func (l *StatsPoolingLayer) Forward(input *Tensor) *Tensor {
	l.inputCache = input.Clone()

	batch := input.Shape[0]
	timeSteps := input.Shape[1]
	channels := input.Shape[2]

	output := Zeros([]int{batch, 2 * channels})

	for b := 0; b < batch; b++ {
		for c := 0; c < channels; c++ {
			// Compute mean
			sum := float64(0)
			for t := 0; t < timeSteps; t++ {
				idx := b*timeSteps*channels + t*channels + c
				sum += input.Data[idx]
			}
			mean := sum / float64(timeSteps)

			// Compute variance
			varSum := float64(0)
			for t := 0; t < timeSteps; t++ {
				idx := b*timeSteps*channels + t*channels + c
				diff := input.Data[idx] - mean
				varSum += diff * diff
			}
			std := math.Sqrt(varSum/float64(timeSteps) + 1e-10)

			// Output: [mean, std]
			output.Data[b*2*channels+c] = mean
			output.Data[b*2*channels+channels+c] = std
		}
	}

	return output
}

func (l *StatsPoolingLayer) Backward(gradOutput *Tensor) *Tensor {
	batch := l.inputCache.Shape[0]
	timeSteps := l.inputCache.Shape[1]
	channels := l.inputCache.Shape[2]

	gradInput := Zeros(l.inputCache.Shape)

	for b := 0; b < batch; b++ {
		for c := 0; c < channels; c++ {
			// Recompute mean and std
			sum := float64(0)
			for t := 0; t < timeSteps; t++ {
				idx := b*timeSteps*channels + t*channels + c
				sum += l.inputCache.Data[idx]
			}
			mean := sum / float64(timeSteps)

			varSum := float64(0)
			for t := 0; t < timeSteps; t++ {
				idx := b*timeSteps*channels + t*channels + c
				diff := l.inputCache.Data[idx] - mean
				varSum += diff * diff
			}
			std := math.Sqrt(varSum/float64(timeSteps) + 1e-10)

			// Gradients
			gradMean := gradOutput.Data[b*2*channels+c]
			gradStd := gradOutput.Data[b*2*channels+channels+c]

			for t := 0; t < timeSteps; t++ {
				idx := b*timeSteps*channels + t*channels + c
				diff := l.inputCache.Data[idx] - mean

				// Gradient from mean
				grad := gradMean / float64(timeSteps)

				// Gradient from std
				grad += gradStd * diff / (float64(timeSteps) * std)

				gradInput.Data[idx] = grad
			}
		}
	}

	return gradInput
}

func (l *StatsPoolingLayer) Parameters() []*Tensor { return nil }
func (l *StatsPoolingLayer) Gradients() []*Tensor  { return nil }
func (l *StatsPoolingLayer) ZeroGrad()             {}
func (l *StatsPoolingLayer) Name() string          { return l.name }

// ============================================================================
// CNN-TDNN Model Builder
// ============================================================================

// CNNTDNNConfig defines the CNN-TDNN architecture
type CNNTDNNConfig struct {
	InputDim  int // Input feature dimension (e.g., 40 for MFCC)
	OutputDim int // Number of output classes (e.g., num_pdfs)

	// CNN layers configuration
	CNNChannels []int // Output channels for each CNN layer
	CNNKernels  []int // Kernel sizes for each CNN layer
	CNNStrides  []int // Strides for each CNN layer

	// TDNN layers configuration
	TDNNDims     []int   // Hidden dimensions for TDNN layers
	TDNNContexts [][]int // Context offsets for each TDNN layer

	// Regularization
	Dropout      float64
	UseBatchNorm bool

	// For x-vector style
	UseStatsPool bool
	EmbeddingDim int
}

// DefaultCNNTDNNConfig returns a standard CNN-TDNN configuration
// Based on Kaldi's chain/TDNN recipes
func DefaultCNNTDNNConfig(inputDim, outputDim int) *CNNTDNNConfig {
	return &CNNTDNNConfig{
		InputDim:  inputDim,
		OutputDim: outputDim,

		// CNN front-end: 2 conv layers
		CNNChannels: []int{64, 128},
		CNNKernels:  []int{3, 3},
		CNNStrides:  []int{1, 1},

		// TDNN layers with increasing context
		TDNNDims: []int{256, 256, 256, 256, 256},
		TDNNContexts: [][]int{
			{-1, 0, 1},        // Layer 1: ±1
			{-1, 0, 1},        // Layer 2: ±1
			{-3, 0, 3},        // Layer 3: ±3 (subsampled)
			{-3, 0, 3},        // Layer 4: ±3
			{-6, -3, 0, 3, 6}, // Layer 5: wide context
		},

		Dropout:      0.1,
		UseBatchNorm: true,
		UseStatsPool: false,
		EmbeddingDim: 256,
	}
}

// XVectorConfig returns configuration for x-vector style speaker embeddings
func XVectorConfig(inputDim, numSpeakers int) *CNNTDNNConfig {
	return &CNNTDNNConfig{
		InputDim:  inputDim,
		OutputDim: numSpeakers,

		CNNChannels: []int{}, // No CNN for standard x-vector
		CNNKernels:  []int{},
		CNNStrides:  []int{},

		// Standard x-vector TDNN layers
		TDNNDims: []int{512, 512, 512, 512, 1500},
		TDNNContexts: [][]int{
			{-2, -1, 0, 1, 2}, // Frame-level
			{-2, 0, 2},
			{-3, 0, 3},
			{0}, // No context
			{0},
		},

		Dropout:      0.0,
		UseBatchNorm: true,
		UseStatsPool: true, // Statistics pooling for segment-level
		EmbeddingDim: 512,
	}
}

// BuildCNNTDNN creates a CNN-TDNN model
func BuildCNNTDNN(config *CNNTDNNConfig) *Sequential {
	model := NewSequential()

	currentDim := config.InputDim

	// CNN front-end
	for i := 0; i < len(config.CNNChannels); i++ {
		outChannels := config.CNNChannels[i]
		kernelSize := config.CNNKernels[i]
		stride := config.CNNStrides[i]
		padding := kernelSize / 2 // Same padding

		// Conv1D
		conv := NewConv1DLayer(currentDim, outChannels, kernelSize,
			WithStride(stride),
			WithPadding(padding),
			WithConvName(fmt.Sprintf("cnn%d", i+1)))
		model.Add(conv)

		// BatchNorm
		if config.UseBatchNorm {
			bn := NewBatchNormLayer(outChannels)
			bn.name = fmt.Sprintf("bn_cnn%d", i+1)
			model.Add(bn)
		}

		// ReLU
		model.Add(&ReLULayer{name: fmt.Sprintf("relu_cnn%d", i+1)})

		// Dropout
		if config.Dropout > 0 {
			model.Add(NewDropoutLayer(config.Dropout))
		}

		currentDim = outChannels
	}

	// TDNN layers
	for i := 0; i < len(config.TDNNDims); i++ {
		hiddenDim := config.TDNNDims[i]
		context := config.TDNNContexts[i]

		// TDNN layer (affine with context splicing)
		tdnn := NewTDNNLayer(currentDim, hiddenDim, context)
		tdnn.name = fmt.Sprintf("tdnn%d", i+1)
		model.Add(tdnn)

		// BatchNorm
		if config.UseBatchNorm {
			bn := NewBatchNormLayer(hiddenDim)
			bn.name = fmt.Sprintf("bn_tdnn%d", i+1)
			model.Add(bn)
		}

		// ReLU
		model.Add(&ReLULayer{name: fmt.Sprintf("relu_tdnn%d", i+1)})

		// Dropout
		if config.Dropout > 0 {
			model.Add(NewDropoutLayer(config.Dropout))
		}

		currentDim = hiddenDim
	}

	// Statistics pooling for x-vector style
	if config.UseStatsPool {
		model.Add(NewStatsPoolingLayer())
		currentDim = currentDim * 2 // Mean + Std

		// Embedding layer
		embedding := NewAffineLayer(currentDim, config.EmbeddingDim)
		embedding.name = "embedding"
		model.Add(embedding)

		if config.UseBatchNorm {
			bn := NewBatchNormLayer(config.EmbeddingDim)
			bn.name = "bn_embedding"
			model.Add(bn)
		}

		model.Add(&ReLULayer{name: "relu_embedding"})

		currentDim = config.EmbeddingDim
	}

	// Output layer
	output := NewAffineLayer(currentDim, config.OutputDim)
	output.name = "output"
	model.Add(output)

	return model
}

// ============================================================================
// Pretrained HMM-DNN Loader
// For fine-tuning from existing Kaldi models
// ============================================================================

// LoadKaldiNnet3 loads weights from Kaldi nnet3 format
// This is a simplified version - full implementation would parse nnet3 text format
func LoadKaldiNnet3(model *Sequential, nnet3Path string) error {
	// In production, this would:
	// 1. Parse Kaldi nnet3 text or binary format
	// 2. Map layer names to model layers
	// 3. Convert FP32 weights to our format
	// 4. Handle layer type conversions (NaturalGradientAffineComponent, etc.)

	// Placeholder - actual implementation depends on Kaldi nnet3 format
	fmt.Printf("Loading pretrained weights from: %s\n", nnet3Path)
	fmt.Println("Note: Full nnet3 parsing not implemented - using random init")

	return nil
}

// FreezeLayer freezes a layer by name (no gradient updates)
type FrozenLayer struct {
	Layer
	frozen bool
}

func (f *FrozenLayer) Backward(gradOutput *Tensor) *Tensor {
	if f.frozen {
		// Still propagate gradients, but don't accumulate
		return f.Layer.Backward(gradOutput)
	}
	return f.Layer.Backward(gradOutput)
}

func (f *FrozenLayer) Gradients() []*Tensor {
	if f.frozen {
		return nil // No gradients to update
	}
	return f.Layer.Gradients()
}

// FreezeUpTo freezes all layers up to (but not including) the named layer
func FreezeUpTo(model *Sequential, layerName string) {
	for i, layer := range model.Layers {
		if layer.Name() == layerName {
			break
		}
		model.Layers[i] = &FrozenLayer{Layer: layer, frozen: true}
	}
}

// ============================================================================
// Data Augmentation for CNN-TDNN
// ============================================================================

// SpecAugment applies SpecAugment-style augmentation
type SpecAugment struct {
	FreqMaskParam int     // Maximum frequency mask width
	TimeMaskParam int     // Maximum time mask width
	NumFreqMasks  int     // Number of frequency masks
	NumTimeMasks  int     // Number of time masks
	MaskValue     float64 // Value to use for masking (usually 0)
}

func NewSpecAugment(freqMask, timeMask, numFreq, numTime int) *SpecAugment {
	return &SpecAugment{
		FreqMaskParam: freqMask,
		TimeMaskParam: timeMask,
		NumFreqMasks:  numFreq,
		NumTimeMasks:  numTime,
		MaskValue:     0.0,
	}
}

// Apply augmentation to input features
// Input: [batch, time, freq]
func (s *SpecAugment) Apply(input *Tensor) *Tensor {
	output := input.Clone()

	batch := input.Shape[0]
	timeSteps := input.Shape[1]
	freqBins := input.Shape[2]

	for b := 0; b < batch; b++ {
		// Frequency masking
		for m := 0; m < s.NumFreqMasks; m++ {
			f := rand.Intn(s.FreqMaskParam + 1)
			f0 := rand.Intn(freqBins - f + 1)

			for t := 0; t < timeSteps; t++ {
				for freq := f0; freq < f0+f && freq < freqBins; freq++ {
					idx := b*timeSteps*freqBins + t*freqBins + freq
					output.Data[idx] = s.MaskValue
				}
			}
		}

		// Time masking
		for m := 0; m < s.NumTimeMasks; m++ {
			t := rand.Intn(s.TimeMaskParam + 1)
			t0 := rand.Intn(timeSteps - t + 1)

			for time := t0; time < t0+t && time < timeSteps; time++ {
				for freq := 0; freq < freqBins; freq++ {
					idx := b*timeSteps*freqBins + time*freqBins + freq
					output.Data[idx] = s.MaskValue
				}
			}
		}
	}

	return output
}

// ============================================================================
// Learning Rate Warmup for stable FP16 training
// ============================================================================

type WarmupScheduler struct {
	BaseScheduler LRScheduler
	WarmupSteps   int
	WarmupFactor  float64
	currentStep   int
}

func NewWarmupScheduler(base LRScheduler, warmupSteps int, warmupFactor float64) *WarmupScheduler {
	return &WarmupScheduler{
		BaseScheduler: base,
		WarmupSteps:   warmupSteps,
		WarmupFactor:  warmupFactor,
		currentStep:   0,
	}
}

func (w *WarmupScheduler) Step() {
	w.currentStep++
	if w.currentStep > w.WarmupSteps {
		w.BaseScheduler.Step()
	}
}

func (w *WarmupScheduler) GetLR() float64 {
	if w.currentStep < w.WarmupSteps {
		// Linear warmup
		alpha := float64(w.currentStep) / float64(w.WarmupSteps)
		return w.WarmupFactor + alpha*(w.BaseScheduler.GetLR()-w.WarmupFactor)
	}
	return w.BaseScheduler.GetLR()
}
