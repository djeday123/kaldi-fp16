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
		// DEBUG: NaN check
		if act, ok := state.Activations[layer.Name]; ok && act != nil {
			sample, _ := act.ToFP32()
			if len(sample) > 0 {
				nans := 0
				for _, v := range sample {
					if v != v {
						nans++
					}
				}
				if nans > 0 {
					fmt.Printf("  NaN! %s (%dx%d): %d/%d NaN\n",
						layer.Name, act.Rows, act.Cols, nans, len(sample))
				} else {
					mn, mx := sample[0], sample[0]
					for _, v := range sample {
						if v < mn {
							mn = v
						}
						if v > mx {
							mx = v
						}
					}
					fmt.Printf("  OK   %s (%dx%d) range=[%.4g, %.4g]\n",
						layer.Name, act.Rows, act.Cols, mn, mx)
				}
			}
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
			float32(spec.TargetRMS), lw.BN.Epsilon)
	} else {
		err = gpu.BatchNormForward(output, lw.BN, lw.BN.Epsilon)
	}
	if err != nil {
		output.Free()
		return nil, err
	}
	return output, nil
}

// SpecAugment: mask features during training
func (net *Network) forwardSpecAugment(layer *Layer, input *gpu.Tensor, T int) (*gpu.Tensor, error) {
	_ = layer
	_ = T
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
// Замени forwardConvReluBN в internal/nnet/forward.go:

func (net *Network) forwardConvReluBN(layer *Layer, input *gpu.Tensor, T int) (*gpu.Tensor, error) {
	lw := net.Weights[layer.Name]
	spec := layer.Spec.(*ConvReluBNSpec)

	heightIn := spec.HeightIn
	heightOut := spec.HeightOut
	nfIn := spec.NumFiltersIn
	nfOut := spec.NumFiltersOut
	numOffsets := len(spec.TimeOffsets)
	patchDim := nfIn * numOffsets

	// Read input to CPU for im2col
	inputCPU, err := input.ToFP32()
	if err != nil {
		return nil, fmt.Errorf("im2col read: %w", err)
	}

	// im2col: build patches [T * heightOut, patchDim]
	numPatches := T * heightOut
	patches := make([]float32, numPatches*patchDim)

	for t := 0; t < T; t++ {
		for hOut := 0; hOut < heightOut; hOut++ {
			patchRow := t*heightOut + hOut
			for off := 0; off < numOffsets; off++ {
				tOff := spec.TimeOffsets[off]
				hOff := spec.HeightOffsets[off]
				tSrc := t + tOff
				hSrc := hOut*spec.HeightSubsample + hOff
				for f := 0; f < nfIn; f++ {
					patchIdx := patchRow*patchDim + off*nfIn + f
					if tSrc >= 0 && tSrc < T && hSrc >= 0 && hSrc < heightIn {
						inputIdx := tSrc*(heightIn*nfIn) + hSrc*nfIn + f
						patches[patchIdx] = inputCPU[inputIdx]
					}
				}
			}
		}
	}

	// Upload patches to GPU
	patchTensor, err := gpu.TensorFromFP32(patches, numPatches, patchDim)
	if err != nil {
		return nil, fmt.Errorf("im2col upload: %w", err)
	}
	defer patchTensor.Free()

	// GEMM: [numPatches x patchDim] * W[patchDim x nfOut] → [numPatches x nfOut]
	convOut, err := gpu.NewTensor(numPatches, nfOut)
	if err != nil {
		return nil, err
	}
	if err := gpu.GEMMSimple(net.Handle, patchTensor, lw.W, convOut); err != nil {
		convOut.Free()
		return nil, fmt.Errorf("conv GEMM: %w", err)
	}

	// Add bias
	if lw.Bias != nil {
		if err := gpu.AddBias(net.Handle, convOut, lw.Bias); err != nil {
			convOut.Free()
			return nil, err
		}
	}

	// ReLU
	if err := gpu.ReLU(convOut); err != nil {
		convOut.Free()
		return nil, err
	}

	// Reorder: height-major → filter-major
	// convOut is [T*heightOut, nfOut], row (t*heightOut+h) has filters for position (t,h)
	// Kaldi expects [T, nfOut*heightOut] in filter-major: [f0_h0, f0_h1, ..., f1_h0, ...]
	src, err := convOut.ToFP32()
	if err != nil {
		convOut.Free()
		return nil, err
	}
	convOut.Free()

	dst := make([]float32, T*heightOut*nfOut)
	for t := 0; t < T; t++ {
		for h := 0; h < heightOut; h++ {
			for f := 0; f < nfOut; f++ {
				srcIdx := (t*heightOut+h)*nfOut + f
				dstIdx := t*(nfOut*heightOut) + f*heightOut + h
				dst[dstIdx] = src[srcIdx]
			}
		}
	}

	output, err := gpu.TensorFromFP32(dst, T, heightOut*nfOut)
	if err != nil {
		return nil, err
	}

	// BatchNorm
	if lw.BN != nil {
		if err := gpu.BatchNormForward(output, lw.BN, lw.BN.Epsilon); err != nil {
			output.Free()
			return nil, err
		}
	}

	return output, nil
}

func spliceTensor(h *gpu.Handle, input *gpu.Tensor, T, stride int) (*gpu.Tensor, error) {
	cols := input.Cols
	// Create shifted copy: row t gets input row max(t-stride, 0)
	shifted, err := gpu.NewTensor(T, cols)
	if err != nil {
		return nil, err
	}

	// Copy the valid part: rows [stride..T) ← input rows [0..T-stride)
	if T > stride {
		src := input.View(0, T-stride, cols)
		dst := shifted.View(stride, T-stride, cols)
		if err := gpu.Copy(dst, src); err != nil {
			shifted.Free()
			return nil, err
		}
	}
	// Pad first 'stride' rows with row 0 (edge padding)
	row0 := input.View(0, 1, cols)
	for t := 0; t < stride && t < T; t++ {
		dst := shifted.View(t, 1, cols)
		if err := gpu.Copy(dst, row0); err != nil {
			shifted.Free()
			return nil, err
		}
	}

	// Concat: [T x cols*2]
	spliced, err := gpu.ZeroTensor(T, cols*2)
	if err != nil {
		shifted.Free()
		return nil, err
	}
	// Left half: shifted (t-stride)
	if err := gpu.ConcatCols(spliced, shifted, 0); err != nil {
		shifted.Free()
		spliced.Free()
		return nil, err
	}
	shifted.Free()
	// Right half: original (t)
	if err := gpu.ConcatCols(spliced, input, cols); err != nil {
		spliced.Free()
		return nil, err
	}

	return spliced, nil
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
//
// spliceTensor creates [T x cols*2] by concatenating shifted and current frames.
// For offset [-stride, 0]: left half = input[t-stride], right half = input[t]
// Frames before t=stride use t=0 (edge padding).
func (net *Network) forwardTDNNF(layer *Layer, input *gpu.Tensor, T int) (*gpu.Tensor, error) {
	lw := net.Weights[layer.Name]
	spec := layer.Spec.(*TDNNFSpec)
	stride := spec.TimeStride

	// 1. Linear projection with splice [-stride, 0]
	var linInput *gpu.Tensor
	freeLinInput := false
	if stride > 0 {
		// splice [-stride, 0]: concat input[t-stride] and input[t]
		s, err := spliceBackward(net.Handle, input, T, stride)
		if err != nil {
			return nil, fmt.Errorf("splice linear: %w", err)
		}
		linInput = s
		freeLinInput = true
	} else {
		linInput = input
	}

	bottleneck, err := gpu.NewTensor(T, spec.BottleneckDim)
	if err != nil {
		if freeLinInput {
			linInput.Free()
		}
		return nil, err
	}
	if err := gpu.GEMMSimple(net.Handle, linInput, lw.LinearW, bottleneck); err != nil {
		if freeLinInput {
			linInput.Free()
		}
		bottleneck.Free()
		return nil, fmt.Errorf("GEMM linear: %w", err)
	}
	if freeLinInput {
		linInput.Free()
	}

	// 2. Affine with splice [0, +stride]
	var affInput *gpu.Tensor
	freeAffInput := false
	if stride > 0 {
		// splice [0, +stride]: concat input[t] and input[t+stride]
		s, err := spliceForward(net.Handle, bottleneck, T, stride)
		if err != nil {
			bottleneck.Free()
			return nil, fmt.Errorf("splice affine: %w", err)
		}
		affInput = s
		freeAffInput = true
		bottleneck.Free()
	} else {
		affInput = bottleneck
	}

	output, err := gpu.NewTensor(T, spec.OutputDim)
	if err != nil {
		if freeAffInput {
			affInput.Free()
		} else {
			bottleneck.Free()
		}
		return nil, err
	}
	if err := gpu.GEMMSimple(net.Handle, affInput, lw.AffineW, output); err != nil {
		if freeAffInput {
			affInput.Free()
		} else {
			bottleneck.Free()
		}
		output.Free()
		return nil, fmt.Errorf("GEMM affine: %w", err)
	}
	if freeAffInput {
		affInput.Free()
	} else {
		bottleneck.Free()
	}

	// 3. Bias
	if lw.AffineBias != nil {
		if err := gpu.AddBias(net.Handle, output, lw.AffineBias); err != nil {
			output.Free()
			return nil, err
		}
	}
	// 4. ReLU
	if err := gpu.ReLU(output); err != nil {
		output.Free()
		return nil, err
	}
	// 5. BatchNorm
	if lw.AffBN != nil {
		if err := gpu.BatchNormForward(output, lw.AffBN, lw.AffBN.Epsilon); err != nil {
			output.Free()
			return nil, err
		}
	}
	// 6. Bypass
	if spec.BypassScale > 0 && spec.InputDim == spec.OutputDim {
		if err := gpu.AddScaled(output, input, float32(spec.BypassScale), 1.0); err != nil {
			output.Free()
			return nil, err
		}
	}
	return output, nil
}

// spliceBackward: offsets [-stride, 0] → concat input[t-stride] and input[t]
// Edge: t < stride uses t=0
func spliceBackward(h *gpu.Handle, input *gpu.Tensor, T, stride int) (*gpu.Tensor, error) {
	cols := input.Cols
	shifted, err := gpu.NewTensor(T, cols)
	if err != nil {
		return nil, err
	}
	// rows [stride..T) ← input rows [0..T-stride)
	if T > stride {
		src := input.View(0, T-stride, cols)
		dst := shifted.View(stride, T-stride, cols)
		if err := gpu.Copy(dst, src); err != nil {
			shifted.Free()
			return nil, err
		}
	}
	// Pad first 'stride' rows with row 0
	row0 := input.View(0, 1, cols)
	for t := 0; t < stride && t < T; t++ {
		dst := shifted.View(t, 1, cols)
		if err := gpu.Copy(dst, row0); err != nil {
			shifted.Free()
			return nil, err
		}
	}

	// Concat: [shifted | original] = [t-stride, t]
	spliced, err := gpu.ZeroTensor(T, cols*2)
	if err != nil {
		shifted.Free()
		return nil, err
	}
	if err := gpu.ConcatCols(spliced, shifted, 0); err != nil {
		shifted.Free()
		spliced.Free()
		return nil, err
	}
	shifted.Free()
	if err := gpu.ConcatCols(spliced, input, cols); err != nil {
		spliced.Free()
		return nil, err
	}
	return spliced, nil
}

// spliceForward: offsets [0, +stride] → concat input[t] and input[t+stride]
// Edge: t+stride >= T uses t=T-1
func spliceForward(h *gpu.Handle, input *gpu.Tensor, T, stride int) (*gpu.Tensor, error) {
	cols := input.Cols
	shifted, err := gpu.NewTensor(T, cols)
	if err != nil {
		return nil, err
	}
	// rows [0..T-stride) ← input rows [stride..T)
	if T > stride {
		src := input.View(stride, T-stride, cols)
		dst := shifted.View(0, T-stride, cols)
		if err := gpu.Copy(dst, src); err != nil {
			shifted.Free()
			return nil, err
		}
	}
	// Pad last 'stride' rows with row T-1
	lastRow := input.View(T-1, 1, cols)
	for t := T - stride; t < T; t++ {
		if t >= 0 {
			dst := shifted.View(t, 1, cols)
			if err := gpu.Copy(dst, lastRow); err != nil {
				shifted.Free()
				return nil, err
			}
		}
	}

	// Concat: [original | shifted] = [t, t+stride]
	spliced, err := gpu.ZeroTensor(T, cols*2)
	if err != nil {
		shifted.Free()
		return nil, err
	}
	if err := gpu.ConcatCols(spliced, input, 0); err != nil {
		shifted.Free()
		spliced.Free()
		return nil, err
	}
	if err := gpu.ConcatCols(spliced, shifted, cols); err != nil {
		shifted.Free()
		spliced.Free()
		return nil, err
	}
	shifted.Free()
	return spliced, nil
}

// Attention-ReLU-BatchNorm
// TODO: full restricted self-attention implementation
// For now: placeholder affine transform
func (net *Network) forwardAttention(layer *Layer, input *gpu.Tensor, T int) (*gpu.Tensor, error) {
	lw := net.Weights[layer.Name]
	spec := layer.Spec.(*AttentionSpec)

	numHeads := spec.NumHeads
	keyDim := spec.KeyDim
	valueDim := spec.ValueDim
	contextDim := spec.ContextDim
	timeStride := spec.TimeStride
	keyScale := float32(spec.KeyScale)
	numLeft := spec.NumLeftInputs

	queryDim := keyDim + contextDim
	inputDimPerHead := keyDim + valueDim + queryDim
	outputDimPerHead := valueDim + contextDim
	affineDim := numHeads * inputDimPerHead

	// 1. Affine: [T x InputDim] → [T x affineDim]
	proj, err := gpu.NewTensor(T, affineDim)
	if err != nil {
		return nil, err
	}
	if err := gpu.GEMMSimple(net.Handle, input, lw.W, proj); err != nil {
		proj.Free()
		return nil, err
	}
	if lw.Bias != nil {
		if err := gpu.AddBias(net.Handle, proj, lw.Bias); err != nil {
			proj.Free()
			return nil, err
		}
	}
	projCPU, err := proj.ToFP32()
	if err != nil {
		proj.Free()
		return nil, err
	}
	proj.Free()

	// 2. Pad with zeros: leftCtx frames before, rightCtx frames after
	leftCtx := numLeft * timeStride
	rightCtx := spec.NumRightInputs * timeStride
	Tin := T + leftCtx + rightCtx
	Tout := T
	rowShift := timeStride

	padded := make([]float32, Tin*affineDim)
	for t := 0; t < T; t++ {
		copy(padded[(t+leftCtx)*affineDim:(t+leftCtx+1)*affineDim],
			projCPU[t*affineDim:(t+1)*affineDim])
	}

	// 3. Per-head attention on CPU
	outDim := numHeads * outputDimPerHead
	outputCPU := make([]float32, Tout*outDim)

	for h := 0; h < numHeads; h++ {
		hIn := h * inputDimPerHead
		hOut := h * outputDimPerHead

		for t := 0; t < Tout; t++ {
			qBase := (t+leftCtx)*affineDim + hIn
			qKeyOff := qBase + keyDim + valueDim // query key part
			qCtxOff := qKeyOff + keyDim          // query context part

			// Compute b[o] and softmax
			b := make([]float32, contextDim)
			maxB := float32(-1e30)
			for o := 0; o < contextDim; o++ {
				kBase := (t+o*rowShift)*affineDim + hIn // key row
				dot := float32(0)
				for d := 0; d < keyDim; d++ {
					dot += padded[qKeyOff+d] * padded[kBase+d]
				}
				b[o] = padded[qCtxOff+o] + keyScale*dot
				if b[o] > maxB {
					maxB = b[o]
				}
			}
			sumExp := float32(0)
			for o := 0; o < contextDim; o++ {
				b[o] = float32(math.Exp(float64(b[o] - maxB)))
				sumExp += b[o]
			}

			outBase := t*outDim + hOut
			// Weighted sum of values + context weights
			for o := 0; o < contextDim; o++ {
				w := b[o] / sumExp
				vBase := (t+o*rowShift)*affineDim + hIn + keyDim
				for d := 0; d < valueDim; d++ {
					outputCPU[outBase+d] += w * padded[vBase+d]
				}
				outputCPU[outBase+valueDim+o] = w
			}
		}
	}

	// Upload + ReLU + BN
	output, err := gpu.TensorFromFP32(outputCPU, Tout, outDim)
	if err != nil {
		return nil, err
	}
	if err := gpu.ReLU(output); err != nil {
		output.Free()
		return nil, err
	}
	if lw.BN != nil {
		if err := gpu.BatchNormForward(output, lw.BN, lw.BN.Epsilon); err != nil {
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

	// 1. Affine: [T x in] → [T x big]
	big, err := gpu.NewTensor(T, spec.BigDim)
	if err != nil {
		return nil, err
	}
	if err := gpu.GEMMSimple(net.Handle, input, lw.BigW, big); err != nil {
		big.Free()
		return nil, err
	}
	if lw.BigBias != nil {
		if err := gpu.AddBias(net.Handle, big, lw.BigBias); err != nil {
			big.Free()
			return nil, err
		}
	}

	// 2. ReLU
	if err := gpu.ReLU(big); err != nil {
		big.Free()
		return nil, err
	}

	// 3. BatchNorm1 (big dim)
	if lw.PfBN != nil {
		if err := gpu.BatchNormForward(big, lw.PfBN, lw.PfBN.Epsilon); err != nil {
			big.Free()
			return nil, err
		}
	}

	// 4. Linear: [T x big] → [T x small]
	output, err := gpu.NewTensor(T, spec.SmallDim)
	if err != nil {
		big.Free()
		return nil, err
	}
	if err := gpu.GEMMSimple(net.Handle, big, lw.SmallW, output); err != nil {
		big.Free()
		output.Free()
		return nil, err
	}
	big.Free()

	// 5. BatchNorm2 (small dim) — this is the output
	if lw.BN != nil {
		if err := gpu.BatchNormForward(output, lw.BN, lw.BN.Epsilon); err != nil {
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
	mean := make([]float32, dim)
	variance := make([]float32, dim)
	gamma := make([]float32, dim)
	beta := make([]float32, dim)
	for i := range variance {
		variance[i] = 1.0
		gamma[i] = 1.0
	}
	bn, err := gpu.NewBNParams(mean, variance, gamma, beta)
	if err != nil {
		return nil, err
	}
	bn.Epsilon = 0.001
	return bn, nil
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
