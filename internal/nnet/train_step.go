// train_step.go — Single training step (mirrors Kaldi's TrainInternal)
//
// Kaldi's TrainInternal():
//   1. computer.AcceptInputs → forward through network
//   2. computer.Run()         → execute forward computation
//   3. ProcessOutputs()       → compute chain loss + gradients
//   4. computer.Run()         → backpropagate gradients through network
//   5. ApplyL2Regularization  → add L2 to weight gradients
//   6. UpdateNnetWithMaxChange → apply weight updates (SGD/Adam)

package nnet

import (
	"fmt"

	"kaldi-fp16/internal/gpu"
	"kaldi-fp16/internal/loader"
)

// TrainConfig holds training configuration
type TrainConfig struct {
	ChainOpts         ChainTrainingOpts
	LearningRate      float32
	Momentum          float32
	MaxParamChange    float32 // max parameter change per minibatch (Kaldi default: 2.0)
	SubsamplingFactor int     // typically 3
	LeftContext       int
}

// TrainStepResult holds results from one training step
type TrainStepResult struct {
	ChainObjf    float64 // chain objective (higher = better)
	L2Term       float64 // L2 regularization term
	TotalWeight  float64 // total frame weight
	ObjfPerFrame float64 // chain objective per frame
	NumLogprob   float32
	DenLogprob   float32
}

// TrainStep executes one training step on a minibatch.
//
// This function mirrors Kaldi's NnetChainTrainer::TrainInternal().
//
// Parameters:
//   - net: the neural network with weights on GPU
//   - batch: training batch from DataLoader
//   - denFst: denominator FST on GPU
//   - config: training configuration
func TrainStep(
	net *Network,
	batch *loader.TrainingBatch,
	denFst *ChainFstGPU,
	config TrainConfig,
) (TrainStepResult, error) {

	result := TrainStepResult{}

	// ================================================================
	// Step 1: Transfer batch to GPU and run forward pass
	// ================================================================
	// TransferBatch: FP32→FP16 conversion + single cudaMemcpy
	gpuBatch, err := gpu.TransferBatch(batch)
	if err != nil {
		return result, fmt.Errorf("transfer batch: %w", err)
	}
	defer gpuBatch.Free()

	// Wrap GPU pointers as Tensors for Network.Forward()
	features := &gpu.Tensor{
		Ptr:   gpuBatch.Features(),
		Rows:  gpuBatch.TotalFrames,
		Cols:  gpuBatch.FeatDim,
		Owned: false, // owned by gpuBatch
	}

	var ivectors *gpu.Tensor
	if gpuBatch.IvecDim > 0 {
		ivectors = &gpu.Tensor{
			Ptr:   gpuBatch.Ivectors(),
			Rows:  gpuBatch.BatchSize,
			Cols:  gpuBatch.IvecDim,
			Owned: false,
		}
	}

	state, err := net.Forward(features, ivectors)
	if err != nil {
		return result, fmt.Errorf("forward pass: %w", err)
	}
	defer state.Free()

	// state.Output: [T × num_pdfs] — network output

	// ================================================================
	// Step 2: Compute chain loss and gradients w.r.t. nnet_output
	//         (This is ProcessOutputs in Kaldi)
	// ================================================================

	// Allocate gradient tensor (same shape as output)
	outputGrad, err := gpu.NewTensor(state.Output.Rows, state.Output.Cols)
	if err != nil {
		return result, fmt.Errorf("alloc output grad: %w", err)
	}
	defer outputGrad.Free()

	// Compute per-sequence chain loss + fill outputGrad
	lossResult, err := ComputeChainLossBatch(
		state.Output,
		batch.PerSeqCSRs,
		denFst,
		batch.FrameOffsets,
		batch.NumFrames,
		batch.FramesPerSeq,
		config.SubsamplingFactor,
		config.LeftContext,
		outputGrad,
	)
	if err != nil {
		return result, fmt.Errorf("chain loss: %w", err)
	}

	// ================================================================
	// Step 3: Backpropagate gradients through the network layers
	//         (This is computer.Run() backward in Kaldi)
	// ================================================================
	//
	// TODO: Implement layer-by-layer backward pass
	// For each layer in reverse ExecutionOrder():
	//   gradInput, gradWeights = layer.Backward(gradOutput)
	// This produces weight gradients for each trainable layer.
	//
	// outputGrad now contains dLoss/dOutput =
	//   sup_weight * (num_post - den_post)

	// ================================================================
	// Step 4: Update weights (SGD with momentum + MaxParamChange)
	//         (This is UpdateNnetWithMaxChange in Kaldi)
	// ================================================================
	//
	// TODO: Apply weight updates using gradients from Step 3
	// For each layer with weights:
	//   delta_w = -lr * grad_w + momentum * delta_w_prev
	//   w += clip(delta_w, max_param_change)

	result = TrainStepResult{
		NumLogprob:   lossResult.NumLogprob,
		DenLogprob:   lossResult.DenLogprob,
		ChainObjf:    float64(lossResult.NumLogprob - lossResult.DenLogprob),
		ObjfPerFrame: float64(lossResult.NumLogprob-lossResult.DenLogprob) / float64(state.NumFrames),
	}

	return result, nil
}
