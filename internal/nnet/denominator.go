package nnet

/*
#cgo CFLAGS: -I${SRCDIR}/../../cpp/include
#cgo CFLAGS: -I/projects/pr2/kaldi-fp16/test_system

#cgo LDFLAGS: -L${SRCDIR}/../../cpp/build -lkaldi_fp16 -lkaldi_fp16_cgo -lcublas -lcudart -lstdc++

#cgo LDFLAGS: -L/projects/pr2/kaldi-fp16/test_system -lkaldi_den -Wl,-rpath,/projects/pr2/kaldi-fp16/test_system
#cgo LDFLAGS: -L/opt/kaldi/src/lib -Wl,-rpath,/opt/kaldi/src/lib
#cgo LDFLAGS: -lkaldi-chain -lkaldi-cudamatrix -lkaldi-matrix -lkaldi-util -lkaldi-base -lkaldi-fstext -lkaldi-lat -lkaldi-hmm -lkaldi-tree
#cgo LDFLAGS: -L/opt/kaldi/tools/openfst-1.7.2/lib -Wl,-rpath,/opt/kaldi/tools/openfst-1.7.2/lib -lfst
#cgo LDFLAGS: -L/usr/local/cuda-12.8/targets/x86_64-linux/lib -Wl,-rpath,/usr/local/cuda-12.8/targets/x86_64-linux/lib
#cgo LDFLAGS: -lcudart -lcublas -lstdc++ -lm

#include "chain_den.h"
#include "kaldi_den_wrapper.h"
#include <stdlib.h>
*/
import "C"

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"unsafe"

	"kaldi-fp16/internal/parser"
)

// ============================================================
// Denominator interface — both implementations satisfy this
// ============================================================

// Denominator computes denominator log-prob and gradients for Chain LF-MMI
type Denominator interface {
	Forward(nnetOutput []float32, T, numSequences int) (float32, error)
	ForwardBackward(nnetOutput []float32, T, numSequences int, derivWeight float32) (float32, []float32, error)
	Free()
}

// ============================================================
// NativeDenominator — pure Go + CUDA (no Kaldi dependency)
// ============================================================

// DenTransition represents one arc in the denominator HMM
type DenTransition struct {
	Src       int32
	Dst       int32
	PdfId     int32   // 0-indexed
	TransProb float32 // exp(-arc.weight) — probability space
}

// NativeDenominator holds denominator graph and GPU data
type NativeDenominator struct {
	transitions  []DenTransition
	initialProbs []float32 // [numStates] from 100-iter HMM warmup
	numStates    int
	numPdfs      int
	startState   int

	gpuFst   C.DenFstGPU
	uploaded bool
}

// NewNativeDenominator loads den.fst and computes initial probs
func NewNativeDenominator(denFstPath string, numPdfs int) (*NativeDenominator, error) {
	f, err := os.Open(denFstPath)
	if err != nil {
		return nil, fmt.Errorf("open den.fst: %w", err)
	}
	defer f.Close()

	fst := parser.ReadFst(bufio.NewReader(f))
	if fst == nil {
		return nil, fmt.Errorf("failed to parse den.fst (unsupported format?)")
	}

	fmt.Printf("[den-native] Loaded den.fst: %d states, %d arcs, start=%d\n",
		fst.NumStates, fst.NumArcs, fst.Start)

	// Extract transitions: Label is 1-indexed in Kaldi → subtract 1
	// Weight is tropical semiring (-log_prob) → exp(-weight) for probability
	transitions := make([]DenTransition, 0, fst.NumArcs)
	for stateIdx, state := range fst.States {
		for _, arc := range state.Arcs {
			pdfId := arc.Label - 1
			if pdfId < 0 {
				continue
			}
			tp := float32(math.Exp(float64(-arc.Weight)))
			transitions = append(transitions, DenTransition{
				Src:       int32(stateIdx),
				Dst:       arc.NextState,
				PdfId:     pdfId,
				TransProb: tp,
			})
		}
	}

	nd := &NativeDenominator{
		transitions: transitions,
		numStates:   int(fst.NumStates),
		numPdfs:     numPdfs,
		startState:  int(fst.Start),
	}

	fmt.Printf("[den-native] Extracted %d transitions (from %d arcs)\n",
		len(transitions), fst.NumArcs)

	nd.computeInitialProbs()

	sum := sumFloat32(nd.initialProbs)
	mx := maxFloat32(nd.initialProbs)
	mn := minNonZeroFloat32(nd.initialProbs)
	nz := countNonZero(nd.initialProbs)
	fmt.Printf("[den-native] Initial probs: sum=%.6f, max=%.6e, min_nonzero=%.6e, nonzero=%d/%d\n",
		sum, mx, mn, nz, nd.numStates)

	if err := nd.uploadToGPU(); err != nil {
		return nil, fmt.Errorf("upload to GPU: %w", err)
	}

	return nd, nil
}

// computeInitialProbs implements Kaldi's DenominatorGraph::SetInitialProbs()
// 100 iterations of HMM propagation, averaged. Uses float64.
// Reference: Kaldi src/chain/chain-den-graph.cc
func (nd *NativeDenominator) computeInitialProbs() {
	S := nd.numStates

	curProb := make([]float64, S)
	nextProb := make([]float64, S)
	avgProb := make([]float64, S)

	curProb[nd.startState] = 1.0

	for iter := 0; iter < 100; iter++ {
		for s := 0; s < S; s++ {
			avgProb[s] += curProb[s] / 100.0
		}

		for s := range nextProb {
			nextProb[s] = 0
		}

		for _, t := range nd.transitions {
			nextProb[t.Dst] += curProb[t.Src] * float64(t.TransProb)
		}

		total := 0.0
		for _, v := range nextProb {
			total += v
		}
		if total > 0 {
			invTotal := 1.0 / total
			for s := range nextProb {
				nextProb[s] *= invTotal
			}
		}

		curProb, nextProb = nextProb, curProb
	}

	nd.initialProbs = make([]float32, S)
	for s := 0; s < S; s++ {
		nd.initialProbs[s] = float32(avgProb[s])
	}
}

func (nd *NativeDenominator) uploadToGPU() error {
	n := len(nd.transitions)
	if n == 0 {
		return fmt.Errorf("no transitions to upload")
	}

	src := make([]int32, n)
	dst := make([]int32, n)
	pdf := make([]int32, n)
	tp := make([]float32, n)

	for i, t := range nd.transitions {
		src[i] = t.Src
		dst[i] = t.Dst
		pdf[i] = t.PdfId
		tp[i] = t.TransProb
	}

	ret := C.den_fst_upload(
		&nd.gpuFst,
		(*C.int32_t)(unsafe.Pointer(&src[0])),
		(*C.int32_t)(unsafe.Pointer(&dst[0])),
		(*C.int32_t)(unsafe.Pointer(&pdf[0])),
		(*C.float)(unsafe.Pointer(&tp[0])),
		C.int(n),
		C.int(nd.numStates),
		C.int(nd.numPdfs),
	)
	if ret != 0 {
		errStr := C.den_last_error()
		if errStr != nil {
			defer C.den_clear_error()
			return fmt.Errorf("den_fst_upload: %s", C.GoString(errStr))
		}
		return fmt.Errorf("den_fst_upload failed")
	}

	nd.uploaded = true
	fmt.Printf("[den-native] Uploaded %d transitions to GPU\n", n)
	return nil
}

func (nd *NativeDenominator) Forward(nnetOutput []float32, T, numSequences int) (float32, error) {
	if !nd.uploaded {
		return 0, fmt.Errorf("GPU data not uploaded")
	}

	expected := T * numSequences * nd.numPdfs
	if len(nnetOutput) != expected {
		return 0, fmt.Errorf("nnetOutput size %d != %d (T=%d × seq=%d × pdfs=%d)",
			len(nnetOutput), expected, T, numSequences, nd.numPdfs)
	}

	if numSequences != 1 {
		return 0, fmt.Errorf("multi-sequence not yet supported (got %d)", numSequences)
	}

	logprob := C.den_forward(
		&nd.gpuFst,
		(*C.float)(unsafe.Pointer(&nnetOutput[0])),
		(*C.float)(unsafe.Pointer(&nd.initialProbs[0])),
		C.int(T),
		C.float(1e-05),
	)

	return float32(logprob), nil
}

func (nd *NativeDenominator) ForwardBackward(nnetOutput []float32, T, numSequences int, derivWeight float32) (float32, []float32, error) {
	if !nd.uploaded {
		return 0, nil, fmt.Errorf("GPU data not uploaded")
	}

	expected := T * numSequences * nd.numPdfs
	if len(nnetOutput) != expected {
		return 0, nil, fmt.Errorf("nnetOutput size %d != %d", len(nnetOutput), expected)
	}

	if numSequences != 1 {
		return 0, nil, fmt.Errorf("multi-sequence not yet supported (got %d)", numSequences)
	}

	grad := make([]float32, T*numSequences*nd.numPdfs)

	logprob := C.den_forward_backward(
		&nd.gpuFst,
		(*C.float)(unsafe.Pointer(&nnetOutput[0])),
		(*C.float)(unsafe.Pointer(&nd.initialProbs[0])),
		C.int(T),
		C.float(1e-05),
		(*C.float)(unsafe.Pointer(&grad[0])),
	)

	if derivWeight != 1.0 {
		for i := range grad {
			grad[i] *= derivWeight
		}
	}

	return float32(logprob), grad, nil
}

func (nd *NativeDenominator) Free() {
	if nd.uploaded {
		C.den_fst_free(&nd.gpuFst)
		nd.uploaded = false
	}
}

func (nd *NativeDenominator) NumStates() int          { return nd.numStates }
func (nd *NativeDenominator) InitialProbs() []float32 { return nd.initialProbs }

// ============================================================
// KaldiDenominator — Kaldi C++ wrapper (reference implementation)
// ============================================================

// KaldiDenominator wraps Kaldi's DenominatorComputation
type KaldiDenominator struct {
	handle  unsafe.Pointer
	numPdfs int
}

// NewKaldiDenominator loads den.fst and initializes Kaldi denominator
func NewKaldiDenominator(denFstPath string, numPdfs int) (*KaldiDenominator, error) {
	cPath := C.CString(denFstPath)
	defer C.free(unsafe.Pointer(cPath))

	h := C.kaldi_den_init(cPath, C.int(numPdfs))
	if h == nil {
		return nil, fmt.Errorf("kaldi_den_init failed")
	}

	return &KaldiDenominator{
		handle:  h,
		numPdfs: numPdfs,
	}, nil
}

func (kd *KaldiDenominator) Forward(nnetOutput []float32, T, numSequences int) (float32, error) {
	numRows := T * numSequences
	if len(nnetOutput) != numRows*kd.numPdfs {
		return 0, fmt.Errorf("nnetOutput size %d != %d*%d", len(nnetOutput), numRows, kd.numPdfs)
	}

	logprob := C.kaldi_den_forward(
		kd.handle,
		(*C.float)(unsafe.Pointer(&nnetOutput[0])),
		C.int(numRows),
		C.int(numSequences),
		C.float(1e-05),
	)

	return float32(logprob), nil
}

func (kd *KaldiDenominator) ForwardBackward(nnetOutput []float32, T, numSequences int, derivWeight float32) (float32, []float32, error) {
	numRows := T * numSequences
	if len(nnetOutput) != numRows*kd.numPdfs {
		return 0, nil, fmt.Errorf("nnetOutput size mismatch")
	}

	grad := make([]float32, numRows*kd.numPdfs)

	logprob := C.kaldi_den_forward_backward(
		kd.handle,
		(*C.float)(unsafe.Pointer(&nnetOutput[0])),
		C.int(numRows),
		C.int(numSequences),
		C.float(1e-05),
		C.float(derivWeight),
		(*C.float)(unsafe.Pointer(&grad[0])),
	)

	return float32(logprob), grad, nil
}

func (kd *KaldiDenominator) Free() {
	if kd.handle != nil {
		C.kaldi_den_free(kd.handle)
		kd.handle = nil
	}
}

// ============================================================
// Helpers
// ============================================================

func sumFloat32(a []float32) float64 {
	s := 0.0
	for _, v := range a {
		s += float64(v)
	}
	return s
}

func maxFloat32(a []float32) float64 {
	if len(a) == 0 {
		return 0
	}
	m := float64(a[0])
	for _, v := range a[1:] {
		if float64(v) > m {
			m = float64(v)
		}
	}
	return m
}

func minNonZeroFloat32(a []float32) float64 {
	m := math.MaxFloat64
	for _, v := range a {
		fv := float64(v)
		if fv > 0 && fv < m {
			m = fv
		}
	}
	if m == math.MaxFloat64 {
		return 0
	}
	return m
}

func countNonZero(a []float32) int {
	n := 0
	for _, v := range a {
		if v > 0 {
			n++
		}
	}
	return n
}
