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

// Trainer holds persistent training state across steps
type Trainer struct {
	Net       *Network
	Optimizer *gpu.SGDOptimizer
	Config    TrainConfig
	DenFst    *ChainFstGPU

	StepCount int
}

// NewTrainer creates a Trainer and registers all network weights with the optimizer
func NewTrainer(net *Network, denFst *ChainFstGPU, config TrainConfig) (*Trainer, error) {
	opt := gpu.NewSGDOptimizer(config.LearningRate, config.Momentum)

	// Register all trainable weights
	for name, lw := range net.Weights {
		if lw.W != nil {
			if err := opt.RegisterParam(name, "W", lw.W); err != nil {
				opt.Free()
				return nil, fmt.Errorf("register %s/W: %w", name, err)
			}
		}
		if lw.Bias != nil {
			if err := opt.RegisterParam(name, "Bias", lw.Bias); err != nil {
				opt.Free()
				return nil, fmt.Errorf("register %s/Bias: %w", name, err)
			}
		}
		if lw.LinearW != nil {
			if err := opt.RegisterParam(name, "LinearW", lw.LinearW); err != nil {
				opt.Free()
				return nil, fmt.Errorf("register %s/LinearW: %w", name, err)
			}
		}
		if lw.AffineW != nil {
			if err := opt.RegisterParam(name, "AffineW", lw.AffineW); err != nil {
				opt.Free()
				return nil, fmt.Errorf("register %s/AffineW: %w", name, err)
			}
		}
		if lw.AffineBias != nil {
			if err := opt.RegisterParam(name, "AffineBias", lw.AffineBias); err != nil {
				opt.Free()
				return nil, fmt.Errorf("register %s/AffineBias: %w", name, err)
			}
		}
		if lw.SmallW != nil {
			if err := opt.RegisterParam(name, "SmallW", lw.SmallW); err != nil {
				opt.Free()
				return nil, fmt.Errorf("register %s/SmallW: %w", name, err)
			}
		}
		if lw.BigW != nil {
			if err := opt.RegisterParam(name, "BigW", lw.BigW); err != nil {
				opt.Free()
				return nil, fmt.Errorf("register %s/BigW: %w", name, err)
			}
		}
		if lw.BigBias != nil {
			if err := opt.RegisterParam(name, "BigBias", lw.BigBias); err != nil {
				opt.Free()
				return nil, fmt.Errorf("register %s/BigBias: %w", name, err)
			}
		}
	}

	return &Trainer{
		Net:       net,
		Optimizer: opt,
		Config:    config,
		DenFst:    denFst,
	}, nil
}

// Free releases optimizer resources
func (t *Trainer) Free() {
	if t.Optimizer != nil {
		t.Optimizer.Free()
	}
}

// SetLR updates the learning rate (for LR scheduling)
func (t *Trainer) SetLR(lr float32) {
	t.Config.LearningRate = lr
	t.Optimizer.SetLR(lr)
}

// Step executes one training step on a minibatch.
func (t *Trainer) Step(batch *loader.TrainingBatch) (TrainStepResult, error) {
	return TrainStep(t.Net, batch, t.DenFst, t.Config, t.Optimizer)
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
//   - optimizer: SGD optimizer (nil = skip weight update)
func TrainStep(
	net *Network,
	batch *loader.TrainingBatch,
	denFst *ChainFstGPU,
	config TrainConfig,
	optimizer *gpu.SGDOptimizer,
) (TrainStepResult, error) {

	result := TrainStepResult{}

	// ================================================================
	// Step 1: Transfer batch to GPU and run forward pass
	// ================================================================
	gpuBatch, err := gpu.TransferBatch(batch)
	if err != nil {
		return result, fmt.Errorf("transfer batch: %w", err)
	}
	defer gpuBatch.Free()

	features := &gpu.Tensor{
		Ptr:   gpuBatch.Features(),
		Rows:  gpuBatch.TotalFrames,
		Cols:  gpuBatch.FeatDim,
		Owned: false,
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

	net.Training = true
	state, err := net.Forward(features, ivectors)
	if err != nil {
		return result, fmt.Errorf("forward pass: %w", err)
	}
	defer state.Free()

	// ================================================================
	// Step 2: Compute chain loss and gradients w.r.t. nnet_output
	// ================================================================
	outputGrad, err := gpu.NewTensor(state.Output.Rows, state.Output.Cols)
	if err != nil {
		return result, fmt.Errorf("alloc output grad: %w", err)
	}
	defer outputGrad.Free()

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
	// ================================================================
	bwdState, err := net.Backward(outputGrad, state)
	if err != nil {
		return result, fmt.Errorf("backward pass: %w", err)
	}
	defer bwdState.Free()

	// ================================================================
	// Step 4: Update weights (SGD with momentum)
	// ================================================================
	if optimizer != nil {
		for name, wg := range bwdState.WeightGrads {
			lw := net.Weights[name]
			if lw == nil {
				continue
			}

			// Update each parameter that has a gradient
			if wg.GradW != nil && lw.W != nil {
				if err := optimizer.Update(name, "W", lw.W, wg.GradW); err != nil {
					return result, fmt.Errorf("update %s/W: %w", name, err)
				}
			}
			if wg.GradBias != nil && lw.Bias != nil {
				if err := optimizer.Update(name, "Bias", lw.Bias, wg.GradBias); err != nil {
					return result, fmt.Errorf("update %s/Bias: %w", name, err)
				}
			}
			if wg.GradLinearW != nil && lw.LinearW != nil {
				if err := optimizer.Update(name, "LinearW", lw.LinearW, wg.GradLinearW); err != nil {
					return result, fmt.Errorf("update %s/LinearW: %w", name, err)
				}
			}
			if wg.GradAffineW != nil && lw.AffineW != nil {
				if err := optimizer.Update(name, "AffineW", lw.AffineW, wg.GradAffineW); err != nil {
					return result, fmt.Errorf("update %s/AffineW: %w", name, err)
				}
			}
			if wg.GradAffineBias != nil && lw.AffineBias != nil {
				if err := optimizer.Update(name, "AffineBias", lw.AffineBias, wg.GradAffineBias); err != nil {
					return result, fmt.Errorf("update %s/AffineBias: %w", name, err)
				}
			}
			if wg.GradSmallW != nil && lw.SmallW != nil {
				if err := optimizer.Update(name, "SmallW", lw.SmallW, wg.GradSmallW); err != nil {
					return result, fmt.Errorf("update %s/SmallW: %w", name, err)
				}
			}
			if wg.GradBigW != nil && lw.BigW != nil {
				if err := optimizer.Update(name, "BigW", lw.BigW, wg.GradBigW); err != nil {
					return result, fmt.Errorf("update %s/BigW: %w", name, err)
				}
			}
			if wg.GradBigBias != nil && lw.BigBias != nil {
				if err := optimizer.Update(name, "BigBias", lw.BigBias, wg.GradBigBias); err != nil {
					return result, fmt.Errorf("update %s/BigBias: %w", name, err)
				}
			}
		}
	}

	// ================================================================
	// Populate result
	// ================================================================
	result = TrainStepResult{
		NumLogprob:   lossResult.NumLogprob,
		DenLogprob:   lossResult.DenLogprob,
		ChainObjf:    float64(lossResult.NumLogprob - lossResult.DenLogprob),
		ObjfPerFrame: float64(lossResult.NumLogprob-lossResult.DenLogprob) / float64(state.NumFrames),
	}

	return result, nil
}
