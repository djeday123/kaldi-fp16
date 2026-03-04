// network_backward.go — Backpropagation through network layers
//
// Implements Step 3 of TrainStep: propagate outputGrad backward through
// all layers in reverse ExecutionOrder, computing weight gradients.
//
// For each layer:
//   1. Compute gradInput (gradient w.r.t. layer input) → pass to previous layer
//   2. Compute gradWeights (gradient w.r.t. layer weights) → store for optimizer
//
// Layer types and their backward ops:
//   - Output:     affine backward (GEMM)
//   - Prefinal:   linear backward + affine backward + relu backward + bn backward
//   - TDNN-F:     bypass split + bn backward + relu backward + affine backward + linear backward
//   - Conv-ReLU-BN: bn backward + relu backward + affine backward
//   - Attention:  bn backward + relu backward + affine backward
//   - BatchNorm:  bn backward (TODO: proper kernel, identity passthrough for now)
//   - Linear:     GEMM backward
//   - IDCT:       GEMM backward (fixed weights, no weight gradient)
//   - SpecAugment: passthrough
//   - CombineFeatureMaps: passthrough (reordering only)

package nnet

import (
	"fmt"

	"kaldi-fp16/internal/gpu"
)

// ============================================================
// BackwardState — holds weight gradients from backward pass
// ============================================================

// BackwardState holds weight gradients for each layer
type BackwardState struct {
	WeightGrads map[string]*LayerWeightGrads
}

// LayerWeightGrads holds gradients for one layer's parameters
type LayerWeightGrads struct {
	GradW    *gpu.Tensor // for Linear, Conv, Output, Attention
	GradBias *gpu.Tensor

	// TDNN-F
	GradLinearW    *gpu.Tensor
	GradAffineW    *gpu.Tensor
	GradAffineBias *gpu.Tensor

	// Prefinal
	GradSmallW  *gpu.Tensor
	GradBigW    *gpu.Tensor
	GradBigBias *gpu.Tensor
}

// Free releases all gradient tensors
func (g *LayerWeightGrads) Free() {
	if g == nil {
		return
	}
	freeT(g.GradW)
	freeT(g.GradBias)
	freeT(g.GradLinearW)
	freeT(g.GradAffineW)
	freeT(g.GradAffineBias)
	freeT(g.GradSmallW)
	freeT(g.GradBigW)
	freeT(g.GradBigBias)
}

// Free releases all backward state
func (bs *BackwardState) Free() {
	for _, g := range bs.WeightGrads {
		g.Free()
	}
}

func freeT(t *gpu.Tensor) {
	if t != nil {
		t.Free()
	}
}

// ============================================================
// Network.Backward — main entry point
// ============================================================

// Backward propagates gradients from outputGrad back through all layers.
//
// Parameters:
//   - outputGrad: [T × num_pdfs] gradient from chain loss
//   - fwdState: saved activations from Forward()
//
// Returns BackwardState with weight gradients for each trainable layer.
func (net *Network) Backward(outputGrad *gpu.Tensor, fwdState *ForwardState) (*BackwardState, error) {
	bs := &BackwardState{
		WeightGrads: make(map[string]*LayerWeightGrads),
	}

	order := net.Model.ExecutionOrder()

	// Map: layer name → gradient of that layer's output
	gradMap := make(map[string]*gpu.Tensor)

	// Seed: output layer gets outputGrad
	if out := net.Model.ChainOutput(); out != nil {
		gradMap[out.Name] = outputGrad
	}

	// Reverse iteration
	for i := len(order) - 1; i >= 0; i-- {
		layer := order[i]

		gradOut, ok := gradMap[layer.Name]
		if !ok {
			// No gradient flows to this layer (e.g. unused branch)
			continue
		}

		gradInput, weightGrads, err := net.backwardLayer(layer, gradOut, fwdState)
		if err != nil {
			bs.Free()
			return nil, fmt.Errorf("backward %s: %w", layer.Name, err)
		}

		// Store weight gradients
		if weightGrads != nil {
			bs.WeightGrads[layer.Name] = weightGrads
		}

		// Route gradInput to the input layer(s)
		if gradInput != nil {
			net.routeGradient(layer, gradInput, gradMap)
		}

		// Free intermediate gradOutput if it was created during backward
		// (don't free outputGrad itself or accumulated grads)
		if gradOut != outputGrad && gradOut.Owned {
			// Will be freed by caller or accumulated
		}
	}

	return bs, nil
}

// routeGradient sends gradInput to the appropriate upstream layer(s)
func (net *Network) routeGradient(layer *Layer, gradInput *gpu.Tensor, gradMap map[string]*gpu.Tensor) {
	if layer.Input.Type == InputAppend {
		// Split gradInput columns back to each source layer
		offset := 0
		for _, name := range layer.InputNames {
			src := net.Model.GetLayer(name)
			if src == nil {
				continue
			}
			cols := src.OutputDim

			// Slice columns for this input
			sliced, err := gpu.NewTensor(gradInput.Rows, cols)
			if err != nil {
				continue
			}
			if err := gpu.SliceCols(sliced, gradInput, offset); err != nil {
				sliced.Free()
				continue
			}

			accumGrad(gradMap, name, sliced)
			offset += cols
		}
		// gradInput itself can be freed since we sliced it
		if gradInput.Owned {
			gradInput.Free()
		}
	} else if len(layer.InputNames) > 0 {
		accumGrad(gradMap, layer.InputNames[0], gradInput)
	}
}

// accumGrad accumulates gradient for a layer (handles bypass connections)
func accumGrad(gradMap map[string]*gpu.Tensor, name string, grad *gpu.Tensor) {
	if existing, ok := gradMap[name]; ok {
		// Accumulate: existing += grad
		gpu.Add(existing, grad)
		if grad.Owned {
			grad.Free()
		}
	} else {
		gradMap[name] = grad
	}
}

// ============================================================
// backwardLayer — dispatch to per-type backward
// ============================================================

func (net *Network) backwardLayer(layer *Layer, gradOut *gpu.Tensor, fwd *ForwardState) (
	gradInput *gpu.Tensor, wg *LayerWeightGrads, err error) {

	switch layer.Type {
	case LayerOutput:
		return net.backwardOutput(layer, gradOut, fwd)
	case LayerPrefinal:
		return net.backwardPrefinal(layer, gradOut, fwd)
	case LayerTDNNF:
		return net.backwardTDNNF(layer, gradOut, fwd)
	case LayerConvReluBatchnorm:
		return net.backwardConvReluBN(layer, gradOut, fwd)
	case LayerAttentionReluBatchnorm:
		return net.backwardAttention(layer, gradOut, fwd)
	case LayerBatchnormComponent:
		return net.backwardBatchNorm(layer, gradOut, fwd)
	case LayerLinearComponent:
		return net.backwardLinear(layer, gradOut, fwd)
	case LayerIDCT:
		return net.backwardIDCT(layer, gradOut, fwd)
	case LayerSpecAugment:
		// Passthrough
		return gradOut, nil, nil
	case LayerCombineFeatureMaps:
		// Passthrough (reordering is its own inverse for gradient)
		return gradOut, nil, nil
	default:
		return gradOut, nil, nil
	}
}

// ============================================================
// Per-layer backward implementations
// ============================================================

// Output: affine → (optional log-softmax)
// Forward: output = input × W + bias
// Backward: gradInput = gradOut × W^T, gradW = input^T × gradOut, gradBias = sum(gradOut)
func (net *Network) backwardOutput(layer *Layer, gradOut *gpu.Tensor, fwd *ForwardState) (
	*gpu.Tensor, *LayerWeightGrads, error) {

	lw := net.Weights[layer.Name]
	input := net.getLayerInput(layer, fwd)
	if input == nil {
		return nil, nil, fmt.Errorf("input activation not found")
	}

	// Note: if IncludeLogSoftmax, gradOut already accounts for it
	// (chain loss gradient is computed post-softmax)

	gradInput, err := gpu.AffineBackwardData(net.Handle, gradOut, lw.W)
	if err != nil {
		return nil, nil, fmt.Errorf("gradInput: %w", err)
	}

	gradW, err := gpu.AffineBackwardWeights(net.Handle, input, gradOut)
	if err != nil {
		gradInput.Free()
		return nil, nil, fmt.Errorf("gradW: %w", err)
	}

	gradBias, err := gpu.AffineBackwardBias(net.Handle, gradOut)
	if err != nil {
		gradInput.Free()
		gradW.Free()
		return nil, nil, fmt.Errorf("gradBias: %w", err)
	}

	return gradInput, &LayerWeightGrads{GradW: gradW, GradBias: gradBias}, nil
}

// Linear: y = x × W (no bias)
func (net *Network) backwardLinear(layer *Layer, gradOut *gpu.Tensor, fwd *ForwardState) (
	*gpu.Tensor, *LayerWeightGrads, error) {

	lw := net.Weights[layer.Name]
	input := net.getLayerInput(layer, fwd)
	if input == nil {
		return nil, nil, fmt.Errorf("input activation not found")
	}

	gradInput, err := gpu.AffineBackwardData(net.Handle, gradOut, lw.W)
	if err != nil {
		return nil, nil, err
	}

	gradW, err := gpu.AffineBackwardWeights(net.Handle, input, gradOut)
	if err != nil {
		gradInput.Free()
		return nil, nil, err
	}

	return gradInput, &LayerWeightGrads{GradW: gradW}, nil
}

// IDCT: y = x × IDCTMat (fixed, no weight gradient)
func (net *Network) backwardIDCT(layer *Layer, gradOut *gpu.Tensor, fwd *ForwardState) (
	*gpu.Tensor, *LayerWeightGrads, error) {

	lw := net.Weights[layer.Name]

	// gradInput = gradOut × IDCTMat^T
	gradInput, err := gpu.AffineBackwardData(net.Handle, gradOut, lw.IDCTMat)
	if err != nil {
		return nil, nil, err
	}

	return gradInput, nil, nil // no weight gradient for fixed transform
}

// BatchNorm backward
// TODO: proper GPU kernel for training mode (compute stats + backprop)
// For now: passthrough (identity BN during testing)
func (net *Network) backwardBatchNorm(layer *Layer, gradOut *gpu.Tensor, fwd *ForwardState) (
	*gpu.Tensor, *LayerWeightGrads, error) {

	lw := net.Weights[layer.Name]
	if lw == nil || lw.BN == nil {
		return gradOut, nil, nil
	}

	gradIn, err := gpu.NewTensor(gradOut.Rows, gradOut.Cols)
	if err != nil {
		return nil, nil, err
	}

	err = gpu.BatchNormBackward(gradOut, gradIn, lw.BN, 1e-5)
	if err != nil {
		gradIn.Free()
		return nil, nil, err
	}

	return gradIn, nil, nil
}

// TDNN-F backward
// Forward: linear → affine+bias → relu → bn → bypass
// Backward (reverse): bypass split → bn backward → relu backward →
//
//	affine backward → linear backward
func (net *Network) backwardTDNNF(layer *Layer, gradOut *gpu.Tensor, fwd *ForwardState) (
	*gpu.Tensor, *LayerWeightGrads, error) {

	lw := net.Weights[layer.Name]
	spec := layer.Spec.(*TDNNFSpec)
	input := net.getLayerInput(layer, fwd)
	if input == nil {
		return nil, nil, fmt.Errorf("input activation not found")
	}

	grad := gradOut

	// 5→1 reverse: Bypass split
	// Forward: output = bn_out + bypass_scale * input
	// Backward: gradBnOut = grad, gradInput_bypass = bypass_scale * grad
	var gradBypass *gpu.Tensor
	if spec.BypassScale > 0 && spec.InputDim == spec.OutputDim {
		// gradInput gets bypass_scale * gradOut added later
		gradBypass, _ = gpu.NewTensor(grad.Rows, grad.Cols)
		if gradBypass != nil {
			gpu.Copy(gradBypass, grad)
			// Scale by bypass_scale: gradBypass = bypass_scale * gradOut
			gpu.Fill(gradBypass, 0)
			gpu.AddScaled(gradBypass, grad, float32(spec.BypassScale), 0.0)
		}
	}

	// 4→: BatchNorm backward
	if lw.AffBN != nil {
		bnGrad, err := gpu.NewTensor(grad.Rows, grad.Cols)
		if err != nil {
			return nil, nil, err
		}
		gpu.BatchNormBackward(grad, bnGrad, lw.AffBN, 1e-5)
		if grad != gradOut {
			grad.Free()
		}
		grad = bnGrad
	}

	// 3→: ReLU backward
	// Need post-affine pre-relu activation — but we didn't save it
	// Workaround: use post-relu output, noting relu backward needs pre-activation
	// Actually, for ReLU: grad *= (preActivation > 0)
	// We can use the post-BN output since BN doesn't change sign for identity BN
	bnOutput := fwd.Activations[layer.Name]
	if bnOutput != nil && bnOutput.Rows == grad.Rows {
		// For ReLU backward we need pre-activation input
		// With identity BN, post-BN ≈ post-ReLU, and ReLU(x) > 0 iff x > 0
		// So we can use the activation as proxy for the mask
		reluGrad, err := gpu.NewTensor(grad.Rows, grad.Cols)
		if err != nil {
			return nil, nil, err
		}
		gpu.Copy(reluGrad, grad)
		// Use output as mask: where output > 0, pass gradient
		gpu.ReLUBackward(bnOutput, reluGrad)
		grad = reluGrad
	}

	// 2→: Affine backward: bottleneck × AffineW + bias → output_dim
	// Need bottleneck activation (output of linear projection)
	// We don't save intermediate activations within TDNN-F currently
	// Approximate: recompute bottleneck from input
	bottleneck, err := gpu.NewTensor(input.Rows, spec.BottleneckDim)
	if err != nil {
		return nil, nil, err
	}
	gpu.GEMMSimple(net.Handle, input, lw.LinearW, bottleneck)

	gradBottleneck, err := gpu.AffineBackwardData(net.Handle, grad, lw.AffineW)
	if err != nil {
		bottleneck.Free()
		return nil, nil, err
	}

	gradAffineW, err := gpu.AffineBackwardWeights(net.Handle, bottleneck, grad)
	if err != nil {
		bottleneck.Free()
		gradBottleneck.Free()
		return nil, nil, err
	}

	gradAffineBias, err := gpu.AffineBackwardBias(net.Handle, grad)
	if err != nil {
		bottleneck.Free()
		gradBottleneck.Free()
		gradAffineW.Free()
		return nil, nil, err
	}
	bottleneck.Free()

	// Free relu grad if we allocated it
	if grad != gradOut {
		grad.Free()
	}

	// 1→: Linear backward: input × LinearW → bottleneck
	gradInput, err := gpu.AffineBackwardData(net.Handle, gradBottleneck, lw.LinearW)
	if err != nil {
		gradBottleneck.Free()
		gradAffineW.Free()
		gradAffineBias.Free()
		return nil, nil, err
	}

	gradLinearW, err := gpu.AffineBackwardWeights(net.Handle, input, gradBottleneck)
	if err != nil {
		gradInput.Free()
		gradBottleneck.Free()
		gradAffineW.Free()
		gradAffineBias.Free()
		return nil, nil, err
	}
	gradBottleneck.Free()

	// Add bypass gradient to gradInput
	if gradBypass != nil {
		gpu.Add(gradInput, gradBypass)
		gradBypass.Free()
	}

	return gradInput, &LayerWeightGrads{
		GradLinearW:    gradLinearW,
		GradAffineW:    gradAffineW,
		GradAffineBias: gradAffineBias,
	}, nil
}

// Conv-ReLU-BN backward
// Forward: affine → relu → bn
// Backward: bn → relu → affine
func (net *Network) backwardConvReluBN(layer *Layer, gradOut *gpu.Tensor, fwd *ForwardState) (
	*gpu.Tensor, *LayerWeightGrads, error) {

	lw := net.Weights[layer.Name]
	input := net.getLayerInput(layer, fwd)
	if input == nil {
		return nil, nil, fmt.Errorf("input activation not found")
	}

	grad := gradOut

	// BN backward
	if lw.BN != nil {
		bnGrad, err := gpu.NewTensor(grad.Rows, grad.Cols)
		if err != nil {
			return nil, nil, err
		}
		gpu.BatchNormBackward(grad, bnGrad, lw.BN, 1e-5)
		grad = bnGrad
	}

	// ReLU backward
	layerOutput := fwd.Activations[layer.Name]
	if layerOutput != nil {
		reluGrad, err := gpu.NewTensor(grad.Rows, grad.Cols)
		if err != nil {
			return nil, nil, err
		}
		gpu.Copy(reluGrad, grad)
		gpu.ReLUBackward(layerOutput, reluGrad)
		grad = reluGrad
	}

	// Affine backward
	gradInput, err := gpu.AffineBackwardData(net.Handle, grad, lw.W)
	if err != nil {
		if grad != gradOut {
			grad.Free()
		}
		return nil, nil, err
	}

	gradW, err := gpu.AffineBackwardWeights(net.Handle, input, grad)
	if err != nil {
		gradInput.Free()
		if grad != gradOut {
			grad.Free()
		}
		return nil, nil, err
	}

	var gradBias *gpu.Tensor
	if lw.Bias != nil {
		gradBias, err = gpu.AffineBackwardBias(net.Handle, grad)
		if err != nil {
			gradInput.Free()
			gradW.Free()
			if grad != gradOut {
				grad.Free()
			}
			return nil, nil, err
		}
	}

	if grad != gradOut {
		grad.Free()
	}

	return gradInput, &LayerWeightGrads{GradW: gradW, GradBias: gradBias}, nil
}

// Attention backward (simplified — same structure as Conv-ReLU-BN)
func (net *Network) backwardAttention(layer *Layer, gradOut *gpu.Tensor, fwd *ForwardState) (
	*gpu.Tensor, *LayerWeightGrads, error) {
	// Same as ConvReluBN — simplified affine + relu + bn
	return net.backwardConvReluBN(layer, gradOut, fwd)
}

// Prefinal backward
// Forward: linear(small) → affine(big) + bias → relu → bn
// Backward: bn → relu → affine → linear
func (net *Network) backwardPrefinal(layer *Layer, gradOut *gpu.Tensor, fwd *ForwardState) (
	*gpu.Tensor, *LayerWeightGrads, error) {

	lw := net.Weights[layer.Name]
	spec := layer.Spec.(*PrefinalSpec)
	input := net.getLayerInput(layer, fwd)
	if input == nil {
		return nil, nil, fmt.Errorf("input activation not found")
	}

	grad := gradOut

	// BN backward
	if lw.PfBN != nil {
		bnGrad, err := gpu.NewTensor(grad.Rows, grad.Cols)
		if err != nil {
			return nil, nil, err
		}
		gpu.BatchNormBackward(grad, bnGrad, lw.PfBN, 1e-5)
		if grad != gradOut {
			grad.Free()
		}
		grad = bnGrad
	}

	// ReLU backward
	layerOutput := fwd.Activations[layer.Name]
	if layerOutput != nil {
		reluGrad, err := gpu.NewTensor(grad.Rows, grad.Cols)
		if err != nil {
			return nil, nil, err
		}
		gpu.Copy(reluGrad, grad)
		gpu.ReLUBackward(layerOutput, reluGrad)
		grad = reluGrad
	}

	// Affine backward (big): small → big
	// Recompute small activation
	small, err := gpu.NewTensor(input.Rows, spec.SmallDim)
	if err != nil {
		return nil, nil, err
	}
	gpu.GEMMSimple(net.Handle, input, lw.SmallW, small)

	gradSmall, err := gpu.AffineBackwardData(net.Handle, grad, lw.BigW)
	if err != nil {
		small.Free()
		if grad != gradOut {
			grad.Free()
		}
		return nil, nil, err
	}

	gradBigW, err := gpu.AffineBackwardWeights(net.Handle, small, grad)
	if err != nil {
		small.Free()
		gradSmall.Free()
		if grad != gradOut {
			grad.Free()
		}
		return nil, nil, err
	}

	var gradBigBias *gpu.Tensor
	if lw.BigBias != nil {
		gradBigBias, err = gpu.AffineBackwardBias(net.Handle, grad)
		if err != nil {
			small.Free()
			gradSmall.Free()
			gradBigW.Free()
			if grad != gradOut {
				grad.Free()
			}
			return nil, nil, err
		}
	}
	small.Free()

	if grad != gradOut {
		grad.Free()
	}

	// Linear backward (small): input → small
	gradInput, err := gpu.AffineBackwardData(net.Handle, gradSmall, lw.SmallW)
	if err != nil {
		gradSmall.Free()
		gradBigW.Free()
		freeT(gradBigBias)
		return nil, nil, err
	}

	gradSmallW, err := gpu.AffineBackwardWeights(net.Handle, input, gradSmall)
	if err != nil {
		gradInput.Free()
		gradSmall.Free()
		gradBigW.Free()
		freeT(gradBigBias)
		return nil, nil, err
	}
	gradSmall.Free()

	return gradInput, &LayerWeightGrads{
		GradSmallW:  gradSmallW,
		GradBigW:    gradBigW,
		GradBigBias: gradBigBias,
	}, nil
}

// ============================================================
// Helper: get layer's input activation from ForwardState
// ============================================================

func (net *Network) getLayerInput(layer *Layer, fwd *ForwardState) *gpu.Tensor {
	if len(layer.InputNames) == 0 {
		return nil
	}

	if layer.Input.Type == InputAppend {
		// For Append layers, the input was concatenated during forward
		// We need the concatenated version — check if it was stored
		// The concatenated tensor is created in getInputs() during forward
		// but not stored separately. We need to reconstruct it.
		//
		// For now: reconstruct by re-concatenating
		var tensors []*gpu.Tensor
		totalCols := 0
		T := 0
		for _, name := range layer.InputNames {
			t := fwd.Activations[name]
			if t == nil {
				return nil
			}
			tensors = append(tensors, t)
			totalCols += t.Cols
			T = t.Rows
		}

		concat, err := gpu.NewTensor(T, totalCols)
		if err != nil {
			return nil
		}
		offset := 0
		for _, t := range tensors {
			gpu.ConcatCols(concat, t, offset)
			offset += t.Cols
		}
		return concat
	}

	return fwd.Activations[layer.InputNames[0]]
}
