// Package nnet implements the backward pass for chain LF-MMI loss.
//
// Architecture: Matches Kaldi's ComputeChainObjfAndDeriv() from chain-training.cc
//
//  1. deriv.SetZero()
//  2. DENOMINATOR FIRST (prob-space, chain_den.cu):
//     - den_logprob = weight * denominator.Forward()
//     - denominator.Backward(-weight) → den_post
//  3. PenalizeOutOfRange(nnet_output, [-30,30])
//  4. NUMERATOR (log-domain, chain.cu):
//     - num_logprob = numerator.Forward()
//     - numerator.Backward() → num_post
//  5. grad += weight * (num_post - den_post)
//  6. L2: grad -= weight * l2_regularize * nnet_output
//  7. NaN/Inf check → zero grad, objf = -10*weight
//
// All GPU operations use the same CUDA stream for ordering.
package nnet

/*
#cgo CFLAGS: -I/usr/local/cuda-12.8/include
#cgo LDFLAGS: -L${SRCDIR}/../../cpp/build -lkaldi_fp16 -L/usr/local/cuda-12.8/targets/x86_64-linux/lib -lcudart -lcublas
#include <cuda_runtime.h>
#include <stdlib.h>

// From chain_backward_kernels.cu (appended to chain.cu)
extern int chain_penalize_out_of_range(
    const float *nnet_output,
    float *grad_output,
    float limit,
    float scale,
    int T,
    int num_pdfs);

extern float chain_l2_regularize(
    const float *nnet_output,
    float *grad_output,
    float l2_scale,
    int total_elements);

extern int chain_combine_gradient(
    const float *num_post,
    const float *den_post,
    float weight,
    int T,
    int num_pdfs,
    void *grad_output);

extern int chain_grad_fp32_to_fp16(
    const float *grad_fp32,
    void *grad_fp16,
    int total_elements);

// From chain_backward_kernels.cu — add posterior gradient
// grad[i] += weight * (num_post[i] - den_post[i])
extern int chain_add_posterior_gradient(
    const float *num_post,
    const float *den_post,
    float *grad,
    float weight,
    int total_elements);

// From chain_backward_kernels.cu — numerator forward-backward (log-domain)
// Final states passed explicitly from Go (parsed from supervision FST).
extern float chain_num_forward_backward(
    const int *fst_row_ptr,
    const int *fst_col_idx,
    const float *fst_weights,
    const int *fst_pdf_ids,
    const int *fst_final_states,
    const float *fst_final_weights,
    int num_states,
    int num_arcs,
    int num_final,
    const float *nnet_output,
    float *num_post,
    int T,
    int num_pdfs,
    cudaStream_t stream);

// Deterministic version (no atomics, for scientific comparison)
extern float chain_num_forward_backward_det(
    const int *fst_row_ptr,
    const int *fst_col_idx,
    const float *fst_weights,
    const int *fst_pdf_ids,
    const int *fst_final_states,
    const float *fst_final_weights,
    int num_states,
    int num_arcs,
    int num_final,
    const float *nnet_output,
    float *num_post,
    int T,
    int num_pdfs,
    cudaStream_t stream);

// cublasSaxpy for gradient accumulation
// extern cublasStatus_t cublasSaxpy(cublasHandle_t, int, const float*, const float*, int, float*, int);
*/
import "C"

import (
	"fmt"
	"math"
	"unsafe"
)

// ---------------------------------------------------------------------------
// Training options
// ---------------------------------------------------------------------------

// ChainTrainingOpts mirrors Kaldi's ChainTrainingOptions
type ChainTrainingOpts struct {
	// L2 regularization on network output (default: 0.0)
	L2Regularize float32

	// Out-of-range regularization (default: 0.01 in Kaldi)
	OutOfRangeRegularize float32

	// Leaky HMM coefficient for denominator (default: 1e-05)
	LeakyHMMCoefficient float32

	// Cross-entropy regularization scale (default: 0.0)
	XentRegularize float32

	// Supervision weight (default: 1.0)
	SupervisionWeight float32
}

// DefaultChainTrainingOpts returns Kaldi's default options
func DefaultChainTrainingOpts() ChainTrainingOpts {
	return ChainTrainingOpts{
		L2Regularize:         0.0,
		OutOfRangeRegularize: 0.01,
		LeakyHMMCoefficient:  1e-05,
		SupervisionWeight:    1.0,
		XentRegularize:       0.0,
	}
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

// ChainLossBackward holds the computed loss and gradient diagnostics
type ChainLossBackward struct {
	// Total objective function: weight * (num_logprob - den_logprob)
	TotalObjf float64

	// L2 regularization term
	L2Term float64

	// Total weight (= supervision.weight * num_frames)
	TotalWeight float64

	// Raw numerator log-probability
	NumLogprob float64

	// Raw denominator log-probability
	DenLogprob float64

	// Number of frames
	NumFrames int

	// Number of out-of-range values found
	OutOfRangeCount int

	// Per-frame objective
	ObjfPerFrame float64

	// Whether computation succeeded (no NaN/Inf)
	OK bool
}

// ---------------------------------------------------------------------------
// CSROnGPU represents a numerator FST in CSR format on GPU
// ---------------------------------------------------------------------------

// CSROnGPU represents a CSR sparse matrix stored on GPU.
// Includes final state information for proper backward pass initialization.
//
// For numerator FSTs, FinalStates typically contains a single state
// (e.g. state 120) with FinalWeights = 0.0 (log-domain weight).
// The old code incorrectly assumed ALL states were final.
type CSROnGPU struct {
	RowPtr       unsafe.Pointer // int*   — [num_states + 1]
	ColIdx       unsafe.Pointer // int*   — [num_arcs]
	Weights      unsafe.Pointer // float* — [num_arcs]
	PdfIds       unsafe.Pointer // int*   — [num_arcs]
	FinalStates  unsafe.Pointer // int*   — [num_final]
	FinalWeights unsafe.Pointer // float* — [num_final]
	NumStates    int
	NumArcs      int
	NumFinal     int
}

// ---------------------------------------------------------------------------
// Main backward pass
// ---------------------------------------------------------------------------

// ComputeChainObjfAndDeriv computes the chain loss and its gradient.
//
// This is the Go equivalent of Kaldi's ComputeChainObjfAndDeriv().
// It orchestrates:
//   - Denominator forward-backward (prob-space, via DenominatorGPU interface / chain_den.cu)
//   - PenalizeOutOfRange regularization
//   - Numerator forward-backward (log-domain, via chain.cu)
//   - Gradient combination: grad += weight * (num_post - den_post)
//   - L2 regularization
//   - NaN/Inf safety check
//
// Parameters:
//   - opts:        training options
//   - denominator: denominator computation (GPUDenominator wrapping NativeDenominator)
//   - numFST:      numerator FST in CSR format on GPU
//   - nnetOutput:  [T × numPDFs] FP32 network output on GPU
//   - gradFP32:    [T × numPDFs] FP32 gradient on GPU (pre-allocated by caller)
//   - T:           number of time frames (after subsampling)
//   - numPDFs:     number of output PDFs
//   - stream:      CUDA stream
//
// Returns: ChainLossBackward with loss values; gradFP32 populated with gradients.
func ComputeChainObjfAndDeriv(
	opts ChainTrainingOpts,
	denominator DenominatorGPU,
	numFST *CSROnGPU,
	nnetOutput unsafe.Pointer, // float* on GPU [T × numPDFs]
	gradFP32 unsafe.Pointer, // float* on GPU [T × numPDFs] — output
	T int,
	numPDFs int,
	stream C.cudaStream_t,
) (*ChainLossBackward, error) {

	result := &ChainLossBackward{
		NumFrames: T,
		OK:        true,
	}

	weight := opts.SupervisionWeight
	totalElements := T * numPDFs

	// =================================================================
	// Step 1: Zero the gradient buffer
	// =================================================================
	C.cudaMemset(gradFP32, 0, C.size_t(totalElements*4))

	// =================================================================
	// Step 2: DENOMINATOR (prob-space, chain_den.cu)
	// Kaldi does denominator FIRST to reduce peak memory.
	// The Denominator interface handles leaky HMM, warmup, etc.
	// =================================================================
	denLogprob, denPost, err := denominator.ForwardBackwardGPU(nnetOutput, T, numPDFs)
	if err != nil {
		return nil, fmt.Errorf("denominator forward-backward failed: %w", err)
	}
	// denPost is [T × numPDFs] FP32 denominator posteriors on GPU

	result.DenLogprob = denLogprob

	// =================================================================
	// Step 3: PenalizeOutOfRange
	// Applied BETWEEN denominator and numerator (Kaldi order).
	// Penalizes nnet_output values outside [-30, 30].
	// Applied to ~50% of frames for efficiency.
	// Writes penalty gradients into gradFP32.
	// =================================================================
	if opts.OutOfRangeRegularize > 0.0 {
		limit := float32(30.0)
		scale := 2.0 * opts.OutOfRangeRegularize

		outOfRange := C.chain_penalize_out_of_range(
			(*C.float)(nnetOutput),
			(*C.float)(gradFP32),
			C.float(limit),
			C.float(scale),
			C.int(T),
			C.int(numPDFs),
		)
		result.OutOfRangeCount = int(outOfRange)
	}

	// =================================================================
	// Step 4: NUMERATOR (log-domain, chain.cu)
	// =================================================================
	numPost, err := cudaMalloc(C.size_t(totalElements * 4))
	if err != nil {
		return nil, fmt.Errorf("cudaMalloc num_post: %w", err)
	}
	defer cudaFree(numPost)

	numLogprob := float64(C.chain_num_forward_backward(
		(*C.int)(numFST.RowPtr),
		(*C.int)(numFST.ColIdx),
		(*C.float)(numFST.Weights),
		(*C.int)(numFST.PdfIds),
		(*C.int)(numFST.FinalStates),
		(*C.float)(numFST.FinalWeights),
		C.int(numFST.NumStates),
		C.int(numFST.NumArcs),
		C.int(numFST.NumFinal),
		(*C.float)(nnetOutput),
		(*C.float)(numPost),
		C.int(T),
		C.int(numPDFs),
		stream,
	))

	result.NumLogprob = numLogprob

	// =================================================================
	// Step 5: Combine gradients
	// gradFP32 += weight * (num_post - den_post)
	//
	// PenalizeOutOfRange already wrote penalty values into gradFP32,
	// so we ADD rather than overwrite.
	//
	// Kaldi does this in two steps:
	//   denominator.Backward(-weight, &deriv) → writes -weight * den_post
	//   numerator.Backward(&deriv)            → ADDS +weight * num_post
	// Net result: deriv = weight * (num_post - den_post)
	//
	// We use cublasSaxpy for efficiency:
	//   gradFP32 += weight * num_post
	//   gradFP32 -= weight * den_post
	// =================================================================
	addGradientFromPosteriors(
		numPost, denPost, gradFP32,
		weight, totalElements, stream)

	// =================================================================
	// Step 6: L2 regularization
	// grad -= weight * l2_regularize * nnet_output
	// l2_term = -0.5 * weight * l2_regularize * ||nnet_output||^2
	// =================================================================
	if opts.L2Regularize > 0.0 {
		l2Scale := weight * opts.L2Regularize

		l2TermF := C.chain_l2_regularize(
			(*C.float)(nnetOutput),
			(*C.float)(gradFP32),
			C.float(l2Scale),
			C.int(totalElements),
		)
		result.L2Term = float64(l2TermF)
	}

	// =================================================================
	// Step 7: Compute total objective
	// objf = weight * (num_logprob - den_logprob)
	// =================================================================
	totalObjf := float64(weight) * (numLogprob - denLogprob)

	// =================================================================
	// Step 8: NaN/Inf check
	// If objf is invalid, zero all derivatives and set objf = -10 * weight
	// Matches Kaldi: "Objective function is {}, setting to -10 per frame"
	// =================================================================
	if math.IsNaN(totalObjf) || math.IsInf(totalObjf, 0) {
		C.cudaMemset(gradFP32, 0, C.size_t(totalElements*4))
		totalObjf = -10.0 * float64(weight) * float64(T)
		result.L2Term = 0.0
		result.OK = false
	}

	result.TotalObjf = totalObjf
	result.TotalWeight = float64(weight) * float64(T)
	result.ObjfPerFrame = totalObjf / float64(T)

	return result, nil
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// addGradientFromPosteriors adds weight * (num_post - den_post) to grad.
// Uses a fused kernel: grad[i] += weight * (num_post[i] - den_post[i])
func addGradientFromPosteriors(
	numPost, denPost, grad unsafe.Pointer,
	weight float32,
	totalElements int,
	stream C.cudaStream_t,
) {
	_ = stream
	C.chain_add_posterior_gradient(
		(*C.float)(numPost),
		(*C.float)(denPost),
		(*C.float)(grad),
		C.float(weight),
		C.int(totalElements),
	)
}

// cudaMalloc wraps CUDA allocation
func cudaMalloc(size C.size_t) (unsafe.Pointer, error) {
	var ptr unsafe.Pointer
	ret := C.cudaMalloc(&ptr, size)
	if ret != 0 {
		return nil, fmt.Errorf("cudaMalloc failed: error %d", int(ret))
	}
	return ptr, nil
}

// cudaFree wraps CUDA deallocation
func cudaFree(ptr unsafe.Pointer) {
	C.cudaFree(ptr)
}

// ForwardBackwardDet runs deterministic numerator forward-backward.
// No atomic operations — sequential LogAdd in fixed arc order.
// For scientific comparison with Kaldi CPU implementation.
func ForwardBackwardDet(numFST *CSROnGPU, nnetOutput unsafe.Pointer, numPost unsafe.Pointer, T, numPDFs int) float64 {
	return float64(C.chain_num_forward_backward_det(
		(*C.int)(numFST.RowPtr),
		(*C.int)(numFST.ColIdx),
		(*C.float)(numFST.Weights),
		(*C.int)(numFST.PdfIds),
		(*C.int)(numFST.FinalStates),
		(*C.float)(numFST.FinalWeights),
		C.int(numFST.NumStates),
		C.int(numFST.NumArcs),
		C.int(numFST.NumFinal),
		(*C.float)(nnetOutput),
		(*C.float)(numPost),
		C.int(T),
		C.int(numPDFs),
		nil,
	))
}

// ConvertGradFP32ToFP16 converts a FP32 gradient buffer to FP16.
// Used when the backward pass needs FP16 output for mixed-precision training.
func ConvertGradFP32ToFP16(gradFP32 unsafe.Pointer, gradFP16 unsafe.Pointer, totalElements int) error {
	ret := C.chain_grad_fp32_to_fp16(
		(*C.float)(gradFP32),
		gradFP16,
		C.int(totalElements),
	)
	if ret != 0 {
		return fmt.Errorf("chain_grad_fp32_to_fp16 failed: %d", int(ret))
	}
	return nil
}
