package nnet

import (
	"fmt"
	"math"
	"math/rand"

	"kaldi-fp16/internal/gpu"
)

// ============================================================
// Network — GPU-allocated network ready for forward pass
// ============================================================

// Network holds model weights on GPU and executes forward pass
type Network struct {
	Model    *Model
	Handle   *gpu.Handle
	Weights  map[string]*LayerWeights // layer name → weights
	Training bool
}

// LayerWeights holds GPU tensors for one layer's parameters
type LayerWeights struct {
	// Common
	W    *gpu.Tensor   // weight matrix (FP16)
	Bias *gpu.Tensor   // bias vector [1 x D] (FP16)
	BN   *gpu.BNParams // batchnorm parameters (FP32)

	// TDNN-F specific
	LinearW    *gpu.Tensor // bottleneck projection [in x bn_dim]
	AffineW    *gpu.Tensor // expansion [bn_dim x out]
	AffineBias *gpu.Tensor // [1 x out]
	AffBN      *gpu.BNParams

	// Prefinal specific
	SmallW  *gpu.Tensor // [in x small]
	BigW    *gpu.Tensor // [small x big]
	BigBias *gpu.Tensor // [1 x big]
	PfBN    *gpu.BNParams

	// IDCT matrix
	IDCTMat *gpu.Tensor // [dim x dim]
}

// Free releases all GPU memory for layer weights
func (lw *LayerWeights) Free() {
	if lw.W != nil {
		lw.W.Free()
	}
	if lw.Bias != nil {
		lw.Bias.Free()
	}
	if lw.BN != nil {
		lw.BN.Free()
	}
	if lw.LinearW != nil {
		lw.LinearW.Free()
	}
	if lw.AffineW != nil {
		lw.AffineW.Free()
	}
	if lw.AffineBias != nil {
		lw.AffineBias.Free()
	}
	if lw.AffBN != nil {
		lw.AffBN.Free()
	}
	if lw.SmallW != nil {
		lw.SmallW.Free()
	}
	if lw.BigW != nil {
		lw.BigW.Free()
	}
	if lw.BigBias != nil {
		lw.BigBias.Free()
	}
	if lw.PfBN != nil {
		lw.PfBN.Free()
	}
	if lw.IDCTMat != nil {
		lw.IDCTMat.Free()
	}
}

// ============================================================
// ForwardState — activation buffers during forward pass
// ============================================================

// ForwardState holds intermediate activations for each layer
type ForwardState struct {
	Activations map[string]*gpu.Tensor // layer name → output activation
	Output      *gpu.Tensor            // chain output [T x num_pdfs]
	OutputXent  *gpu.Tensor            // xent output [T x num_pdfs] (if exists)
	NumFrames   int
}

// Free releases all activation buffers
func (fs *ForwardState) Free() {
	for _, t := range fs.Activations {
		if t != nil && t.Owned {
			t.Free()
		}
	}
}

// ============================================================
// NewNetwork — create network on GPU from model
// ============================================================

func NewNetwork(model *Model, handle *gpu.Handle) (*Network, error) {
	net := &Network{
		Model:   model,
		Handle:  handle,
		Weights: make(map[string]*LayerWeights),
	}

	// Allocate and initialize weights for each layer
	for _, layer := range model.Layers {
		lw, err := allocWeights(layer)
		if err != nil {
			net.Free()
			return nil, fmt.Errorf("alloc weights %s: %w", layer.Name, err)
		}
		if lw != nil {
			net.Weights[layer.Name] = lw
		}
	}

	return net, nil
}

// Free releases all GPU resources
func (net *Network) Free() {
	for _, lw := range net.Weights {
		lw.Free()
	}
}

// ============================================================
// Forward — run forward pass
// ============================================================

// Forward runs the full forward pass
// features: [T x feat_dim] FP16 on GPU
// ivectors: [B x ivec_dim] FP16 on GPU (can be nil)
// Returns ForwardState with output activations
func (net *Network) Forward(features, ivectors *gpu.Tensor) (*ForwardState, error) {
	T := features.Rows

	state := &ForwardState{
		Activations: make(map[string]*gpu.Tensor),
		NumFrames:   T,
	}

	// Set input activations
	state.Activations["input"] = features
	if ivectors != nil {
		state.Activations["ivector"] = ivectors
	}

	// Execute in order
	for _, layer := range net.Model.ExecutionOrder() {
		if err := net.forwardLayer(layer, state); err != nil {
			state.Free()
			return nil, fmt.Errorf("forward %s: %w", layer.Name, err)
		}
	}

	// Set outputs
	if out := net.Model.ChainOutput(); out != nil {
		state.Output = state.Activations[out.Name]
	}
	if xent := net.Model.XentOutput(); xent != nil {
		state.OutputXent = state.Activations[xent.Name]
	}

	return state, nil
}

// forwardLayer executes one layer
func (net *Network) forwardLayer(layer *Layer, state *ForwardState) error {
	// Get input activation(s)
	inputs, err := net.getInputs(layer, state)
	if err != nil {
		return err
	}

	T := state.NumFrames
	var output *gpu.Tensor

	switch layer.Type {
	case LayerIDCT:
		output, err = net.forwardIDCT(layer, inputs[0], T)

	case LayerLinearComponent:
		output, err = net.forwardLinear(layer, inputs[0], T)

	case LayerBatchnormComponent:
		output, err = net.forwardBatchNorm(layer, inputs[0], T)

	case LayerSpecAugment:
		// Pass-through during inference, apply masks during training
		if net.Training {
			output, err = net.forwardSpecAugment(layer, inputs[0], T)
		} else {
			output = inputs[0] // no-op during inference
		}

	case LayerCombineFeatureMaps:
		output, err = net.forwardCombineFeatureMaps(layer, inputs, T)

	case LayerConvReluBatchnorm:
		output, err = net.forwardConvReluBN(layer, inputs[0], T)

	case LayerTDNNF:
		output, err = net.forwardTDNNF(layer, inputs[0], T)

	case LayerAttentionReluBatchnorm:
		output, err = net.forwardAttention(layer, inputs[0], T)

	case LayerPrefinal:
		output, err = net.forwardPrefinal(layer, inputs[0], T)

	case LayerOutput:
		output, err = net.forwardOutput(layer, inputs[0], T)

	default:
		return fmt.Errorf("unsupported layer type: %s", layer.Type)
	}

	if err != nil {
		return err
	}

	state.Activations[layer.Name] = output
	return nil
}

// getInputs resolves layer inputs from activation state
func (net *Network) getInputs(layer *Layer, state *ForwardState) ([]*gpu.Tensor, error) {
	if layer.Input.Type == InputAppend {
		// Append multiple inputs → concat columns
		var tensors []*gpu.Tensor
		for _, name := range layer.InputNames {
			t, ok := state.Activations[name]
			if !ok {
				return nil, fmt.Errorf("input %q not found in state", name)
			}
			tensors = append(tensors, t)
		}

		// Concat into one tensor
		totalCols := 0
		T := tensors[0].Rows
		for _, t := range tensors {
			totalCols += t.Cols
		}

		concat, err := gpu.NewTensor(T, totalCols)
		if err != nil {
			return nil, err
		}

		offset := 0
		for _, t := range tensors {
			if err := gpu.ConcatCols(concat, t, offset); err != nil {
				concat.Free()
				return nil, err
			}
			offset += t.Cols
		}

		return []*gpu.Tensor{concat}, nil
	}

	// Single input or previous
	if len(layer.InputNames) == 0 {
		return nil, fmt.Errorf("no inputs resolved")
	}

	t, ok := state.Activations[layer.InputNames[0]]
	if !ok {
		return nil, fmt.Errorf("input %q not found", layer.InputNames[0])
	}
	return []*gpu.Tensor{t}, nil
}

// ============================================================
// Layer-specific forward implementations
// ============================================================

// IDCT: fixed matrix multiply
func (net *Network) forwardIDCT(layer *Layer, input *gpu.Tensor, T int) (*gpu.Tensor, error) {
	lw := net.Weights[layer.Name]
	output, err := gpu.NewTensor(T, layer.OutputDim)
	if err != nil {
		return nil, err
	}
	// output = input * IDCT_matrix
	err = gpu.GEMMSimple(net.Handle, input, lw.IDCTMat, output)
	if err != nil {
		output.Free()
		return nil, err
	}
	return output, nil
}

// Linear: y = x * W (no bias)
func (net *Network) forwardLinear(layer *Layer, input *gpu.Tensor, T int) (*gpu.Tensor, error) {
	lw := net.Weights[layer.Name]
	output, err := gpu.NewTensor(T, layer.OutputDim)
	if err != nil {
		return nil, err
	}
	// output = input * W
	err = gpu.GEMMSimple(net.Handle, input, lw.W, output)
	if err != nil {
		output.Free()
		return nil, err
	}
	return output, nil
}

// BatchNorm: normalize using running stats
func (net *Network) forwardBatchNorm(layer *Layer, input *gpu.Tensor, T int) (*gpu.Tensor, error) {
	lw := net.Weights[layer.Name]
	spec := layer.Spec.(*BatchnormSpec)

	// Clone input (batchnorm is in-place)
	output, err := gpu.NewTensor(T, layer.OutputDim)
	if err != nil {
		return nil, err
	}
	if err := gpu.Copy(output, input); err != nil {
		output.Free()
		return nil, err
	}

	if spec.TargetRMS != 1.0 {
		err = gpu.BatchNormForwardRMS(output, lw.BN.Mean, lw.BN.Var,
			float32(spec.TargetRMS), 1e-5)
	} else {
		err = gpu.BatchNormForward(output, lw.BN, 1e-5)
	}
	if err != nil {
		output.Free()
		return nil, err
	}
	return output, nil
}

// SpecAugment: mask features during training
func (net *Network) forwardSpecAugment(layer *Layer, input *gpu.Tensor, T int) (*gpu.Tensor, error) {
	// TODO: implement frequency and time masking on GPU
	// For now, pass-through (masking is a training-only augmentation)
	return input, nil
}

// CombineFeatureMaps: reorder concatenated features
func (net *Network) forwardCombineFeatureMaps(layer *Layer, inputs []*gpu.Tensor, T int) (*gpu.Tensor, error) {
	spec := layer.Spec.(*CombineFeatureMapsSpec)
	input := inputs[0] // already concatenated by getInputs

	output, err := gpu.NewTensor(T, layer.OutputDim)
	if err != nil {
		return nil, err
	}
	if err := gpu.Copy(output, input); err != nil {
		output.Free()
		return nil, err
	}

	err = gpu.CombineFeatureMaps(output, spec.Height, spec.NumFilters1, spec.NumFilters2)
	if err != nil {
		output.Free()
		return nil, err
	}
	return output, nil
}

// Conv-ReLU-BatchNorm
//
// Kaldi CNN layout: input [T x (height * filters)]
// Conv kernel: [filters_out x filters_in x time_offsets x height_offsets]
// For simplicity, implement as a lowered GEMM approach:
//  1. Gather patches (im2col-like)
//  2. GEMM: patches * kernel_matrix
//  3. ReLU
//  4. BatchNorm
func (net *Network) forwardConvReluBN(layer *Layer, input *gpu.Tensor, T int) (*gpu.Tensor, error) {
	lw := net.Weights[layer.Name]
	spec := layer.Spec.(*ConvReluBNSpec)

	// For 2D conv in Kaldi format:
	// patch_dim = num_filters_in * len(time_offsets) * len(height_offsets)
	// output = patches * W + bias, then ReLU, then BN
	//
	// Simplified: treat as affine transform on gathered patches
	// W: [patch_dim x output_dim]
	patchDim := spec.NumFiltersIn * len(spec.TimeOffsets) * len(spec.HeightOffsets)
	_ = patchDim // TODO: im2col gathering

	// For now: use W as [input_dim x output_dim] affine
	// This works for height-preserving convs but not for subsampling
	// Full implementation needs im2col with height/time offsets

	output, err := gpu.NewTensor(T, layer.OutputDim)
	if err != nil {
		return nil, err
	}

	// Affine: output = input * W + bias
	if err := gpu.GEMMSimple(net.Handle, input, lw.W, output); err != nil {
		output.Free()
		return nil, err
	}
	if lw.Bias != nil {
		if err := gpu.AddBias(net.Handle, output, lw.Bias); err != nil {
			output.Free()
			return nil, err
		}
	}

	// ReLU
	if err := gpu.ReLU(output); err != nil {
		output.Free()
		return nil, err
	}

	// BatchNorm
	if lw.BN != nil {
		if err := gpu.BatchNormForward(output, lw.BN, 1e-5); err != nil {
			output.Free()
			return nil, err
		}
	}

	return output, nil
}

// TDNN-F: factorized TDNN layer
//
// Components:
//  1. linear: x * LinearW → [T x bn_dim] (no bias, constrained)
//  2. affine: linear_out * AffineW + AffineBias → [T x dim]
//     (with time-stride: selects frames at stride intervals)
//  3. relu
//  4. batchnorm
//  5. bypass: output = batchnorm_out + bypass_scale * input
//     (only if bypass_scale > 0 and dims match)
func (net *Network) forwardTDNNF(layer *Layer, input *gpu.Tensor, T int) (*gpu.Tensor, error) {
	lw := net.Weights[layer.Name]
	spec := layer.Spec.(*TDNNFSpec)

	// 1. Linear projection: [T x in] * [in x bn] → [T x bn]
	bottleneck, err := gpu.NewTensor(T, spec.BottleneckDim)
	if err != nil {
		return nil, err
	}
	if err := gpu.GEMMSimple(net.Handle, input, lw.LinearW, bottleneck); err != nil {
		bottleneck.Free()
		return nil, err
	}

	// 2. Affine: [T x bn] * [bn x dim] + bias → [T x dim]
	// TODO: time-stride (select frames at stride intervals before GEMM)
	// For stride=0, no splicing needed
	// For stride>0, would need to gather frames at t-stride and t+stride
	output, err := gpu.NewTensor(T, spec.OutputDim)
	if err != nil {
		bottleneck.Free()
		return nil, err
	}

	if err := gpu.GEMMSimple(net.Handle, bottleneck, lw.AffineW, output); err != nil {
		bottleneck.Free()
		output.Free()
		return nil, err
	}
	bottleneck.Free()

	if lw.AffineBias != nil {
		if err := gpu.AddBias(net.Handle, output, lw.AffineBias); err != nil {
			output.Free()
			return nil, err
		}
	}

	// 3. ReLU
	if err := gpu.ReLU(output); err != nil {
		output.Free()
		return nil, err
	}

	// 4. BatchNorm
	if lw.AffBN != nil {
		if err := gpu.BatchNormForward(output, lw.AffBN, 1e-5); err != nil {
			output.Free()
			return nil, err
		}
	}

	// 5. Bypass: output = output + bypass_scale * input
	if spec.BypassScale > 0 && spec.InputDim == spec.OutputDim {
		if err := gpu.AddScaled(output, input, float32(spec.BypassScale), 1.0); err != nil {
			output.Free()
			return nil, err
		}
	}

	return output, nil
}

// Attention-ReLU-BatchNorm
// TODO: full restricted self-attention implementation
// For now: placeholder affine transform
func (net *Network) forwardAttention(layer *Layer, input *gpu.Tensor, T int) (*gpu.Tensor, error) {
	lw := net.Weights[layer.Name]

	output, err := gpu.NewTensor(T, layer.OutputDim)
	if err != nil {
		return nil, err
	}

	// Simplified: affine + relu + bn
	if err := gpu.GEMMSimple(net.Handle, input, lw.W, output); err != nil {
		output.Free()
		return nil, err
	}
	if lw.Bias != nil {
		if err := gpu.AddBias(net.Handle, output, lw.Bias); err != nil {
			output.Free()
			return nil, err
		}
	}
	if err := gpu.ReLU(output); err != nil {
		output.Free()
		return nil, err
	}
	if lw.BN != nil {
		if err := gpu.BatchNormForward(output, lw.BN, 1e-5); err != nil {
			output.Free()
			return nil, err
		}
	}
	return output, nil
}

// Prefinal: linear(small) → affine(big) → relu → bn
func (net *Network) forwardPrefinal(layer *Layer, input *gpu.Tensor, T int) (*gpu.Tensor, error) {
	lw := net.Weights[layer.Name]
	spec := layer.Spec.(*PrefinalSpec)

	// 1. Linear: [T x in] → [T x small]
	small, err := gpu.NewTensor(T, spec.SmallDim)
	if err != nil {
		return nil, err
	}
	if err := gpu.GEMMSimple(net.Handle, input, lw.SmallW, small); err != nil {
		small.Free()
		return nil, err
	}

	// 2. Affine: [T x small] → [T x big]
	output, err := gpu.NewTensor(T, spec.BigDim)
	if err != nil {
		small.Free()
		return nil, err
	}
	if err := gpu.GEMMSimple(net.Handle, small, lw.BigW, output); err != nil {
		small.Free()
		output.Free()
		return nil, err
	}
	small.Free()

	if lw.BigBias != nil {
		if err := gpu.AddBias(net.Handle, output, lw.BigBias); err != nil {
			output.Free()
			return nil, err
		}
	}

	// 3. ReLU
	if err := gpu.ReLU(output); err != nil {
		output.Free()
		return nil, err
	}

	// 4. BatchNorm
	if lw.PfBN != nil {
		if err := gpu.BatchNormForward(output, lw.PfBN, 1e-5); err != nil {
			output.Free()
			return nil, err
		}
	}

	return output, nil
}

// Output: affine → optional log-softmax
func (net *Network) forwardOutput(layer *Layer, input *gpu.Tensor, T int) (*gpu.Tensor, error) {
	lw := net.Weights[layer.Name]
	spec := layer.Spec.(*OutputSpec)

	output, err := gpu.NewTensor(T, spec.OutputDim)
	if err != nil {
		return nil, err
	}

	// Affine: [T x in] * W + bias → [T x out]
	if err := gpu.GEMMSimple(net.Handle, input, lw.W, output); err != nil {
		output.Free()
		return nil, err
	}
	if lw.Bias != nil {
		if err := gpu.AddBias(net.Handle, output, lw.Bias); err != nil {
			output.Free()
			return nil, err
		}
	}

	// Log-softmax (for xent output)
	if spec.IncludeLogSoftmax {
		if err := gpu.LogSoftmax(output); err != nil {
			output.Free()
			return nil, err
		}
	}

	return output, nil
}

// ============================================================
// Weight allocation with random initialization
// (For testing — real weights loaded from Kaldi model later)
// ============================================================

func allocWeights(layer *Layer) (*LayerWeights, error) {
	switch layer.Type {
	case LayerInput:
		return nil, nil // no weights

	case LayerIDCT:
		spec := layer.Spec.(*IDCTSpec)
		mat := makeIDCTMatrix(spec.Dim, spec.CepstralLifter)
		t, err := gpu.TensorFromFP32(mat, spec.Dim, spec.Dim)
		if err != nil {
			return nil, err
		}
		return &LayerWeights{IDCTMat: t}, nil

	case LayerLinearComponent:
		spec := layer.Spec.(*LinearSpec)
		w, err := randTensor(spec.InputDim, spec.OutputDim)
		if err != nil {
			return nil, err
		}
		return &LayerWeights{W: w}, nil

	case LayerBatchnormComponent:
		spec := layer.Spec.(*BatchnormSpec)
		bn, err := identityBN(spec.Dim)
		if err != nil {
			return nil, err
		}
		return &LayerWeights{BN: bn}, nil

	case LayerSpecAugment:
		return nil, nil // no weights

	case LayerCombineFeatureMaps:
		return nil, nil // no weights (just reordering)

	case LayerConvReluBatchnorm:
		spec := layer.Spec.(*ConvReluBNSpec)
		w, err := randTensor(spec.InputDim, spec.OutputDim)
		if err != nil {
			return nil, err
		}
		bias, err := gpu.ZeroTensor(1, spec.OutputDim)
		if err != nil {
			w.Free()
			return nil, err
		}
		bn, err := identityBN(spec.OutputDim)
		if err != nil {
			w.Free()
			bias.Free()
			return nil, err
		}
		return &LayerWeights{W: w, Bias: bias, BN: bn}, nil

	case LayerTDNNF:
		spec := layer.Spec.(*TDNNFSpec)
		linW, err := randTensor(spec.InputDim, spec.BottleneckDim)
		if err != nil {
			return nil, err
		}
		affW, err := randTensor(spec.BottleneckDim, spec.OutputDim)
		if err != nil {
			linW.Free()
			return nil, err
		}
		affBias, err := gpu.ZeroTensor(1, spec.OutputDim)
		if err != nil {
			linW.Free()
			affW.Free()
			return nil, err
		}
		bn, err := identityBN(spec.OutputDim)
		if err != nil {
			linW.Free()
			affW.Free()
			affBias.Free()
			return nil, err
		}
		return &LayerWeights{
			LinearW:    linW,
			AffineW:    affW,
			AffineBias: affBias,
			AffBN:      bn,
		}, nil

	case LayerAttentionReluBatchnorm:
		// Simplified: single affine + bn
		w, err := randTensor(layer.InputDim, layer.OutputDim)
		if err != nil {
			return nil, err
		}
		bias, err := gpu.ZeroTensor(1, layer.OutputDim)
		if err != nil {
			w.Free()
			return nil, err
		}
		bn, err := identityBN(layer.OutputDim)
		if err != nil {
			w.Free()
			bias.Free()
			return nil, err
		}
		return &LayerWeights{W: w, Bias: bias, BN: bn}, nil

	case LayerPrefinal:
		spec := layer.Spec.(*PrefinalSpec)
		smallW, err := randTensor(spec.InputDim, spec.SmallDim)
		if err != nil {
			return nil, err
		}
		bigW, err := randTensor(spec.SmallDim, spec.BigDim)
		if err != nil {
			smallW.Free()
			return nil, err
		}
		bigBias, err := gpu.ZeroTensor(1, spec.BigDim)
		if err != nil {
			smallW.Free()
			bigW.Free()
			return nil, err
		}
		bn, err := identityBN(spec.BigDim)
		if err != nil {
			smallW.Free()
			bigW.Free()
			bigBias.Free()
			return nil, err
		}
		return &LayerWeights{SmallW: smallW, BigW: bigW, BigBias: bigBias, PfBN: bn}, nil

	case LayerOutput:
		spec := layer.Spec.(*OutputSpec)
		w, err := randTensor(spec.InputDim, spec.OutputDim)
		if err != nil {
			return nil, err
		}
		bias, err := gpu.ZeroTensor(1, spec.OutputDim)
		if err != nil {
			w.Free()
			return nil, err
		}
		return &LayerWeights{W: w, Bias: bias}, nil

	default:
		return nil, nil
	}
}

// ============================================================
// Helper: random tensor (Xavier initialization)
// ============================================================

func randTensor(rows, cols int) (*gpu.Tensor, error) {
	scale := float32(math.Sqrt(2.0 / float64(rows+cols)))
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = float32(rand.NormFloat64()) * scale
	}
	return gpu.TensorFromFP32(data, rows, cols)
}

// identityBN creates batchnorm params that are identity transform
// mean=0, var=1, gamma=1, beta=0
func identityBN(dim int) (*gpu.BNParams, error) {
	mean := make([]float32, dim) // zeros
	variance := make([]float32, dim)
	gamma := make([]float32, dim)
	beta := make([]float32, dim) // zeros

	for i := range variance {
		variance[i] = 1.0
		gamma[i] = 1.0
	}

	return gpu.NewBNParams(mean, variance, gamma, beta)
}

// makeIDCTMatrix generates the IDCT matrix for Kaldi MFCC→filterbank
func makeIDCTMatrix(dim int, cepstralLifter float64) []float32 {
	mat := make([]float32, dim*dim)
	for i := 0; i < dim; i++ {
		for j := 0; j < dim; j++ {
			// DCT-II inverse
			val := math.Cos(math.Pi * float64(j) * (float64(i) + 0.5) / float64(dim))
			if j == 0 {
				val *= math.Sqrt(1.0 / float64(dim))
			} else {
				val *= math.Sqrt(2.0 / float64(dim))
			}
			// Apply cepstral liftering
			if cepstralLifter > 0 && j > 0 {
				lifter := 1.0 + (cepstralLifter/2.0)*math.Sin(math.Pi*float64(j)/cepstralLifter)
				val *= lifter
			}
			mat[i*dim+j] = float32(val)
		}
	}
	return mat
}
