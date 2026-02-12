package main

/*
#cgo CFLAGS: -I/usr/local/cuda-12.8/include
#cgo LDFLAGS: -L${SRCDIR}/../../cpp/build -lkaldi_fp16 -L/usr/local/cuda-12.8/targets/x86_64-linux/lib -lcudart -lcublas
#include <cuda_runtime.h>
*/
import "C"

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
	"unsafe"

	"kaldi-fp16/internal/gpu"
	"kaldi-fp16/internal/loader"
	"kaldi-fp16/internal/nnet"
	"kaldi-fp16/internal/parser"
	"kaldi-fp16/internal/sparse"
)

const refDir = "/tmp/chain_ref"

func main() {
	fmt.Println("=== Chain LF-MMI Verification ===")
	fmt.Println()

	gpu.Init(0)

	dl, err := loader.NewDataLoaderFromPaths(
		[]string{"/opt/kaldi/egs/work_3/s5/exp/chain_cnn_v2/cnn_tdnn1d_v2_sp/egs/cegs.15.ark"},
		8, false)
	if err != nil {
		log.Fatal(err)
	}
	tb, err := dl.NextBatch()
	if err != nil {
		log.Fatal(err)
	}

	fps := tb.FramesPerSeq[0]
	numPdfs := 3080
	numCSR := tb.PerSeqCSRs[0]

	fmt.Printf("Seq 0: fps=%d, numPdfs=%d, fst_states=%d, fst_arcs=%d, fst_final=%d\n",
		fps, numPdfs, numCSR.NumStates, numCSR.NumArcs, len(numCSR.FinalStates))

	kaldiDen, err := nnet.NewKaldiDenominator(
		"/opt/kaldi/egs/work_3/s5/exp/chain_cnn_v2/cnn_tdnn1d_v2_sp/den.fst",
		numPdfs)
	if err != nil {
		log.Fatal("kaldi den:", err)
	}
	defer kaldiDen.Free()

	// ========================================
	// Phase 1: Zero-input test
	// ========================================
	fmt.Println("\n--- Phase 1: Zero nnet_output ---")

	zeroOutput, err := gpu.ZeroTensor(fps, numPdfs)
	if err != nil {
		log.Fatal(err)
	}
	defer zeroOutput.Free()

	seq0FstGPU, err := nnet.NewChainFstGPU(numCSR)
	if err != nil {
		log.Fatal("num fst:", err)
	}
	defer seq0FstGPU.Free()

	numLP, err := nnet.ForwardBackward(zeroOutput, seq0FstGPU)
	if err != nil {
		log.Fatal("num FB:", err)
	}
	fmt.Printf("num_logprob = %.6f  (Kaldi ref: -60.280000)\n", numLP)
	checkClose("num_logprob_zero", float64(numLP), -60.28, 0.01)

	zeroCPU := make([]float32, fps*numPdfs)
	denLP, err := kaldiDen.Forward(zeroCPU, fps, 1)
	if err != nil {
		log.Fatal("den forward:", err)
	}
	fmt.Printf("den_logprob = %.6f  (Kaldi ref: -0.436088)\n", denLP)
	checkClose("den_logprob_zero", float64(denLP), -0.436088, 1e-4)

	objf := float64(numLP) - float64(denLP)
	fmt.Printf("objf_per_frame = %.6f  (Kaldi ref: -1.273276)\n", objf/float64(fps))
	checkClose("objf_per_frame_zero", objf/float64(fps), -1.27328, 1e-3)

	// ========================================
	// Phase 2: Random-input test (reference)
	// ========================================
	fmt.Println("\n--- Phase 2: Random nnet_output (vs Kaldi reference) ---")

	meta, err := loadMeta(refDir + "/meta.txt")
	if err != nil {
		fmt.Printf("  Cannot load %s/meta.txt: %v\n", refDir, err)
		fmt.Println("  Run dump_chain_ref first.")
		return
	}

	refT := int(meta["T"])
	refPdfs := int(meta["D"])
	refNumLP := meta["num_logprob_weighted"]
	refDenLP := meta["den_logprob_weighted"]
	refObjf := meta["objf"]

	fmt.Printf("Reference: T=%d, D=%d\n", refT, refPdfs)
	fmt.Printf("Reference: num_logprob=%.6f, den_logprob=%.6f, objf=%.6f\n",
		refNumLP, refDenLP, refObjf)

	nnetOutputData, rows, cols, err := loadBinaryMatrix(refDir + "/nnet_output.bin")
	if err != nil {
		log.Fatal("load nnet_output.bin:", err)
	}
	fmt.Printf("Loaded nnet_output: [%d x %d]\n", rows, cols)

	if rows != refT || cols != refPdfs {
		log.Fatalf("shape mismatch: [%d x %d] vs [%d x %d]", rows, cols, refT, refPdfs)
	}

	fmt.Printf("nnet_output[0][0:5] = %.6f %.6f %.6f %.6f %.6f\n",
		nnetOutputData[0], nnetOutputData[1], nnetOutputData[2],
		nnetOutputData[3], nnetOutputData[4])

	nnetTensor, err := gpu.TensorFromFP32(nnetOutputData, refT, refPdfs)
	if err != nil {
		log.Fatal("upload nnet_output:", err)
	}
	defer nnetTensor.Free()

	numFstGPU, err := nnet.NewChainFstGPU(numCSR)
	if err != nil {
		log.Fatal("upload num FST:", err)
	}
	defer numFstGPU.Free()

	fmt.Printf("Num FST: %d states, %d arcs, %d final, start=%d\n",
		numCSR.NumStates, numCSR.NumArcs, len(numCSR.FinalStates), numCSR.StartState)

	goNumLP, err := nnet.ForwardBackward(nnetTensor, numFstGPU)
	if err != nil {
		log.Fatal("num FB:", err)
	}
	fmt.Printf("\nAtomic numerator logprob:\n")
	fmt.Printf("  Go:    %.6f\n", goNumLP)
	fmt.Printf("  Kaldi: %.6f\n", refNumLP)
	fmt.Printf("  Diff:  %.2e\n", math.Abs(float64(goNumLP)-refNumLP))
	checkClose("num_logprob_atomic", float64(goNumLP), refNumLP, 2.0)

	denCPU := make([]float32, refT*refPdfs)
	copy(denCPU, nnetOutputData)
	goDenLP, err := kaldiDen.Forward(denCPU, refT, 1)
	if err != nil {
		log.Fatal("den forward:", err)
	}
	fmt.Printf("\nDenominator logprob:\n")
	fmt.Printf("  Go:    %.6f\n", goDenLP)
	fmt.Printf("  Kaldi: %.6f\n", refDenLP)
	fmt.Printf("  Diff:  %.2e\n", math.Abs(float64(goDenLP)-refDenLP))
	checkClose("den_logprob_ref", float64(goDenLP), refDenLP, 1e-3)

	goObjf := float64(goNumLP) - float64(goDenLP)
	fmt.Printf("\nObjective:\n")
	fmt.Printf("  Go:    %.6f (per_frame: %.6f)\n", goObjf, goObjf/float64(refT))
	fmt.Printf("  Kaldi: %.6f (per_frame: %.6f)\n", refObjf, refObjf/float64(refT))
	fmt.Printf("  Diff:  %.2e\n", math.Abs(goObjf-refObjf))

	// ========================================
	// Phase 3: Deterministic vs Atomic
	// ========================================
	fmt.Println("\n--- Phase 3: Deterministic vs Atomic (scientific comparison) ---")

	// Upload nnet_output as FP32 to GPU
	nnetFP32Size := refT * refPdfs * 4
	var nnetFP32GPU unsafe.Pointer
	cRet := C.cudaMalloc(&nnetFP32GPU, C.size_t(nnetFP32Size))
	if cRet != 0 {
		log.Fatal("cudaMalloc nnetFP32:", cRet)
	}
	defer C.cudaFree(nnetFP32GPU)
	C.cudaMemcpy(nnetFP32GPU, unsafe.Pointer(&nnetOutputData[0]),
		C.size_t(nnetFP32Size), C.cudaMemcpyHostToDevice)

	// Allocate num_post buffer
	var numPostGPU unsafe.Pointer
	cRet = C.cudaMalloc(&numPostGPU, C.size_t(nnetFP32Size))
	if cRet != 0 {
		log.Fatal("cudaMalloc numPost:", cRet)
	}
	defer C.cudaFree(numPostGPU)

	csrGPU := nnet.GetCSROnGPU(numFstGPU)

	detNumLP := nnet.ForwardBackwardDet(csrGPU, nnetFP32GPU, numPostGPU, refT, refPdfs)

	fmt.Printf("\nNumerator logprob comparison:\n")
	fmt.Printf("  Kaldi (CPU, FP32):        %.10f\n", refNumLP)
	fmt.Printf("  Go atomic (GPU, FP16):    %.10f\n", float64(goNumLP))
	fmt.Printf("  Go det (GPU, FP16):       %.10f\n", detNumLP)
	fmt.Printf("\n")
	fmt.Printf("  |Kaldi - atomic|:  %.2e\n", math.Abs(float64(goNumLP)-refNumLP))
	fmt.Printf("  |Kaldi - det|:     %.2e\n", math.Abs(detNumLP-refNumLP))
	fmt.Printf("  |atomic - det|:    %.2e\n", math.Abs(float64(goNumLP)-detNumLP))

	fmt.Printf("\nDeterministic reproducibility (3 runs):\n")
	for i := 0; i < 3; i++ {
		lp := nnet.ForwardBackwardDet(csrGPU, nnetFP32GPU, numPostGPU, refT, refPdfs)
		fmt.Printf("  Run %d: %.10f  diff_from_first: %.2e\n", i+1, lp, math.Abs(lp-detNumLP))
	}

	// Derivative stats
	refNumDeriv, _, _, err := loadBinaryMatrix(refDir + "/num_deriv.bin")
	if err == nil {
		fmt.Printf("\nNum deriv reference: sum=%.4f, absmax=%.6f\n",
			sumFloat32(refNumDeriv), maxAbsFloat32(refNumDeriv))
	}
	refDenDeriv, _, _, err := loadBinaryMatrix(refDir + "/den_deriv.bin")
	if err == nil {
		fmt.Printf("Den deriv reference: sum=%.4f, absmax=%.6f\n",
			sumFloat32(refDenDeriv), maxAbsFloat32(refDenDeriv))
	}
	refTotalDeriv, _, _, err := loadBinaryMatrix(refDir + "/total_deriv.bin")
	if err == nil {
		fmt.Printf("Total deriv reference: sum=%.4f, absmax=%.6f\n",
			sumFloat32(refTotalDeriv), maxAbsFloat32(refTotalDeriv))
	}

	fmt.Println("\n=== Done ===")
}

// ========================================
// Helpers
// ========================================

func loadBinaryMatrix(path string) ([]float32, int, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, err
	}
	defer f.Close()
	var rows, cols int32
	binary.Read(f, binary.LittleEndian, &rows)
	binary.Read(f, binary.LittleEndian, &cols)
	data := make([]float32, int(rows)*int(cols))
	binary.Read(f, binary.LittleEndian, data)
	return data, int(rows), int(cols), nil
}

func loadMeta(path string) (map[string]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	meta := make(map[string]float64)
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		idx := strings.Index(line, "=")
		if idx < 0 {
			continue
		}
		val, err := strconv.ParseFloat(line[idx+1:], 64)
		if err != nil {
			continue
		}
		meta[line[:idx]] = val
	}
	return meta, scanner.Err()
}

func checkClose(name string, got, want, tol float64) {
	diff := math.Abs(got - want)
	if diff > tol {
		fmt.Printf("  FAIL %s: got=%.6f want=%.6f diff=%.2e > tol=%.2e\n",
			name, got, want, diff, tol)
	} else {
		fmt.Printf("  PASS %s (diff=%.2e)\n", name, diff)
	}
}

func sumFloat32(data []float32) float64 {
	var s float64
	for _, v := range data {
		s += float64(v)
	}
	return s
}

func maxAbsFloat32(data []float32) float64 {
	var m float64
	for _, v := range data {
		a := math.Abs(float64(v))
		if a > m {
			m = a
		}
	}
	return m
}

// suppress unused import warnings
var _ = parser.ReadFst
var _ = sparse.FstToCSR
