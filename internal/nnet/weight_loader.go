// weight_loader.go — Load Kaldi nnet3 weights from text format
//
// Usage on gpu4:
//   gpu.Init(0)
//   handle, _ := gpu.NewHandle()
//   model, _ := nnet.BuildModel(".../network.xconfig")
//   net, _ := nnet.NewNetwork(model, handle)
//   err := nnet.LoadWeightsFromFile(net, ".../final.mdl")
//
// Parses: nnet3-copy --binary=false final.mdl -

package nnet

import (
	"bufio"
	"fmt"
	"math"
	"os/exec"
	"strconv"
	"strings"

	"kaldi-fp16/internal/gpu"
)

// ============================================================
// Parsed component data
// ============================================================

// KaldiComponent holds parsed parameters from one nnet3 component
type KaldiComponent struct {
	Name string // e.g. "tdnnf7.linear", "cnn1.conv"
	Type string // e.g. "TdnnComponent", "BatchNormComponent"

	// Weight matrices (row-major FP32)
	LinearParams []float32
	LinearRows   int
	LinearCols   int
	BiasParams   []float32 // may be empty

	// BatchNorm specific
	StatsMean []float32
	StatsVar  []float32
	Count     float64
	Epsilon   float32
	TargetRms float32

	// Conv specific
	NumFiltersIn  int
	NumFiltersOut int
	HeightIn      int
	HeightOut     int

	// Attention specific
	NumHeads int
	KeyDim   int
	ValueDim int
	KeyScale float32

	// Metadata
	LearningRate float32
	MaxChange    float32
	L2Regularize float32
}

// NewNetworkFromKaldi creates Network on GPU directly from parsed Kaldi components.
// Avoids double-allocation (no random init then replace).
func NewNetworkFromKaldi(model *Model, handle *gpu.Handle, components map[string]*KaldiComponent) (*Network, error) {
	net := &Network{
		Model:   model,
		Handle:  handle,
		Weights: make(map[string]*LayerWeights),
	}

	for _, layer := range model.Layers {
		lw, err := allocWeightsFromKaldi(layer, components)
		if err != nil {
			net.Free()
			return nil, fmt.Errorf("layer %s: %w", layer.Name, err)
		}
		if lw != nil {
			net.Weights[layer.Name] = lw
		}
	}

	return net, nil
}

// allocWeightsFromKaldi allocates GPU tensors directly from Kaldi data
func allocWeightsFromKaldi(layer *Layer, comps map[string]*KaldiComponent) (*LayerWeights, error) {
	switch layer.Type {
	case LayerInput, LayerSpecAugment, LayerCombineFeatureMaps:
		return nil, nil

	case LayerIDCT:
		comp := comps["idct"]
		if comp == nil {
			return nil, fmt.Errorf("idct not found")
		}
		data, r, c := transposeF32(comp.LinearParams, comp.LinearRows, comp.LinearCols)
		t, err := gpu.TensorFromFP32(data, r, c)
		if err != nil {
			return nil, err
		}
		return &LayerWeights{IDCTMat: t}, nil

	case LayerLinearComponent:
		comp := comps[layer.Name]
		if comp == nil {
			return nil, fmt.Errorf("%s not found", layer.Name)
		}
		data, r, c := transposeF32(comp.LinearParams, comp.LinearRows, comp.LinearCols)
		w, err := gpu.TensorFromFP32(data, r, c)
		if err != nil {
			return nil, err
		}
		return &LayerWeights{W: w}, nil

	case LayerBatchnormComponent:
		comp := comps[layer.Name]
		if comp == nil {
			return nil, fmt.Errorf("%s not found", layer.Name)
		}
		bn, err := makeBN(comp)
		if err != nil {
			return nil, err
		}
		return &LayerWeights{BN: bn}, nil

	case LayerConvReluBatchnorm:
		base := layer.Name
		conv := comps[base+".conv"]
		if conv == nil {
			return nil, fmt.Errorf("%s.conv not found", base)
		}
		convData, convR, convC := transposeF32(conv.LinearParams, conv.LinearRows, conv.LinearCols)
		w, err := gpu.TensorFromFP32(convData, convR, convC)
		if err != nil {
			return nil, err
		}
		var bias *gpu.Tensor
		if len(conv.BiasParams) > 0 {
			bias, err = gpu.TensorFromFP32(conv.BiasParams, 1, len(conv.BiasParams))
			if err != nil {
				w.Free()
				return nil, err
			}
		}
		bnComp := comps[base+".batchnorm"]
		if bnComp == nil {
			w.Free()
			if bias != nil {
				bias.Free()
			}
			return nil, fmt.Errorf("%s.batchnorm not found", base)
		}
		bn, err := makeBlockBN(bnComp, layer.Spec.(*ConvReluBNSpec).HeightOut)
		if err != nil {
			w.Free()
			if bias != nil {
				bias.Free()
			}
			return nil, err
		}
		return &LayerWeights{W: w, Bias: bias, BN: bn}, nil

	case LayerTDNNF:
		base := layer.Name
		lin := comps[base+".linear"]
		if lin == nil {
			return nil, fmt.Errorf("%s.linear not found", base)
		}
		linData, linR, linC := transposeF32(lin.LinearParams, lin.LinearRows, lin.LinearCols)
		linW, err := gpu.TensorFromFP32(linData, linR, linC)
		if err != nil {
			return nil, err
		}
		aff := comps[base+".affine"]
		if aff == nil {
			linW.Free()
			return nil, fmt.Errorf("%s.affine not found", base)
		}
		affData, affR, affC := transposeF32(aff.LinearParams, aff.LinearRows, aff.LinearCols)
		affW, err := gpu.TensorFromFP32(affData, affR, affC)
		if err != nil {
			linW.Free()
			return nil, err
		}
		var affBias *gpu.Tensor
		if len(aff.BiasParams) > 0 {
			affBias, err = gpu.TensorFromFP32(aff.BiasParams, 1, len(aff.BiasParams))
			if err != nil {
				linW.Free()
				affW.Free()
				return nil, err
			}
		} else {
			// Empty bias — allocate zeros
			affBias, err = gpu.ZeroTensor(1, layer.Spec.(*TDNNFSpec).OutputDim)
			if err != nil {
				linW.Free()
				affW.Free()
				return nil, err
			}
		}
		bnComp := comps[base+".batchnorm"]
		if bnComp == nil {
			linW.Free()
			affW.Free()
			affBias.Free()
			return nil, fmt.Errorf("%s.batchnorm not found", base)
		}
		bn, err := makeBN(bnComp)
		if err != nil {
			linW.Free()
			affW.Free()
			affBias.Free()
			return nil, err
		}
		return &LayerWeights{LinearW: linW, AffineW: affW, AffineBias: affBias, AffBN: bn}, nil

	case LayerAttentionReluBatchnorm:
		base := layer.Name
		aff := comps[base+".affine"]
		if aff == nil {
			return nil, fmt.Errorf("%s.affine not found", base)
		}
		spec := layer.Spec.(*AttentionSpec)
		// Affine: [InputDim → num_heads * (key_dim + value_dim + query_dim)]
		// query_dim = key_dim + context_dim
		affDim := spec.NumHeads * (spec.KeyDim + spec.ValueDim + spec.KeyDim + spec.ContextDim)
		_ = affDim // for documentation; actual dims come from parsed weights

		affData, affR, affC := transposeF32(aff.LinearParams, aff.LinearRows, aff.LinearCols)
		w, err := gpu.TensorFromFP32(affData, affR, affC)
		if err != nil {
			return nil, err
		}
		var bias *gpu.Tensor
		if len(aff.BiasParams) > 0 {
			bias, err = gpu.TensorFromFP32(aff.BiasParams, 1, len(aff.BiasParams))
			if err != nil {
				w.Free()
				return nil, err
			}
		}

		// Batchnorm
		bnComp := comps[base+".batchnorm"]
		if bnComp == nil {
			w.Free()
			if bias != nil {
				bias.Free()
			}
			return nil, fmt.Errorf("%s.batchnorm not found", base)
		}
		bn, err := makeBN(bnComp)
		if err != nil {
			w.Free()
			if bias != nil {
				bias.Free()
			}
			return nil, err
		}

		// Store attention component metadata for forward pass
		attComp := comps[base+".attention"]
		if attComp != nil && attComp.KeyScale > 0 {
			spec.KeyScale = float64(attComp.KeyScale)
		} else {
			spec.KeyScale = 1.0 / math.Sqrt(float64(spec.KeyDim))
		}

		fmt.Printf("  attention %s: affine %dx%d, bn dim=%d, key_scale=%.4f\n",
			base, aff.LinearRows, aff.LinearCols, len(bnComp.StatsMean), spec.KeyScale)
		return &LayerWeights{W: w, Bias: bias, BN: bn}, nil

	case LayerPrefinal:
		prefix := "prefinal-chain"
		if strings.Contains(layer.Name, "xent") {
			prefix = "prefinal-xent"
		}
		aff := comps[prefix+".affine"]
		if aff == nil {
			return nil, fmt.Errorf("%s.affine not found", prefix)
		}
		bigData, bigR, bigC := transposeF32(aff.LinearParams, aff.LinearRows, aff.LinearCols)
		bigW, err := gpu.TensorFromFP32(bigData, bigR, bigC)
		if err != nil {
			return nil, err
		}
		var bigBias *gpu.Tensor
		if len(aff.BiasParams) > 0 {
			bigBias, err = gpu.TensorFromFP32(aff.BiasParams, 1, len(aff.BiasParams))
			if err != nil {
				bigW.Free()
				return nil, err
			}
		} else {
			bigBias, err = gpu.ZeroTensor(1, layer.Spec.(*PrefinalSpec).BigDim)
			if err != nil {
				bigW.Free()
				return nil, err
			}
		}
		lin := comps[prefix+".linear"]
		if lin == nil {
			bigW.Free()
			bigBias.Free()
			return nil, fmt.Errorf("%s.linear not found", prefix)
		}
		smallData, smallR, smallC := transposeF32(lin.LinearParams, lin.LinearRows, lin.LinearCols)
		smallW, err := gpu.TensorFromFP32(smallData, smallR, smallC)
		if err != nil {
			bigW.Free()
			bigBias.Free()
			return nil, err
		}
		bnComp := comps[prefix+".batchnorm1"]
		if bnComp == nil {
			bigW.Free()
			bigBias.Free()
			smallW.Free()
			return nil, fmt.Errorf("%s.batchnorm1 not found", prefix)
		}
		bn, err := makeBN(bnComp)
		if err != nil {
			bigW.Free()
			bigBias.Free()
			smallW.Free()
			return nil, err
		}
		// batchnorm2 (after small linear, dim=small)
		var bn2 *gpu.BNParams
		bn2Comp := comps[prefix+".batchnorm2"]
		if bn2Comp != nil {
			bn2, err = makeBN(bn2Comp)
			if err != nil {
				bigW.Free()
				bigBias.Free()
				smallW.Free()
				bn.Free()
				return nil, err
			}
		}
		return &LayerWeights{SmallW: smallW, BigW: bigW, BigBias: bigBias, PfBN: bn, BN: bn2}, nil

	case LayerOutput:
		name := layer.Name + ".affine"
		comp := comps[name]
		if comp == nil {
			return nil, fmt.Errorf("%s not found", name)
		}
		data, r, c := transposeF32(comp.LinearParams, comp.LinearRows, comp.LinearCols)
		w, err := gpu.TensorFromFP32(data, r, c)
		if err != nil {
			return nil, err
		}
		var bias *gpu.Tensor
		if len(comp.BiasParams) > 0 {
			bias, err = gpu.TensorFromFP32(comp.BiasParams, 1, len(comp.BiasParams))
			if err != nil {
				w.Free()
				return nil, err
			}
		} else {
			bias, err = gpu.ZeroTensor(1, layer.Spec.(*OutputSpec).OutputDim)
			if err != nil {
				w.Free()
				return nil, err
			}
		}
		return &LayerWeights{W: w, Bias: bias}, nil

	default:
		return nil, nil
	}
}

// makeBN computes gamma/beta from Kaldi stats and creates gpu.BNParams
// func makeBN(comp *KaldiComponent) (*gpu.BNParams, error) {
// 	dim := len(comp.StatsMean)
// 	if dim == 0 {
// 		return nil, fmt.Errorf("empty StatsMean")
// 	}
// 	targetRms := comp.TargetRms
// 	if targetRms <= 0 {
// 		targetRms = 1.0
// 	}
// 	eps := comp.Epsilon
// 	if eps <= 0 {
// 		eps = 0.001
// 	}
// 	mean := make([]float32, dim)
// 	variance := make([]float32, dim)
// 	gamma := make([]float32, dim)
// 	beta := make([]float32, dim)
// 	for i := 0; i < dim; i++ {
// 		m := comp.StatsMean[i]
// 		v := float32(0)
// 		if i < len(comp.StatsVar) {
// 			v = comp.StatsVar[i]
// 		}
// 		if v < 0 {
// 			v = 0
// 		}
// 		invStd := float32(1.0 / math.Sqrt(float64(v+eps)))
// 		mean[i] = m
// 		variance[i] = v
// 		gamma[i] = targetRms * invStd
// 		beta[i] = -m * gamma[i]
// 	}
// 	return gpu.NewBNParams(mean, variance, gamma, beta)
// }

// func makeBN(comp *KaldiComponent) (*gpu.BNParams, error) {
// 	dim := len(comp.StatsMean)
// 	if dim == 0 {
// 		return nil, fmt.Errorf("empty StatsMean")
// 	}
// 	targetRms := comp.TargetRms
// 	if targetRms <= 0 {
// 		targetRms = 1.0
// 	}
// 	eps := comp.Epsilon
// 	if eps <= 0 {
// 		eps = 0.001
// 	}

// 	gamma := make([]float32, dim)
// 	beta := make([]float32, dim)
// 	for i := 0; i < dim; i++ {
// 		gamma[i] = targetRms
// 		beta[i] = 0
// 	}
// 	return gpu.NewBNParams(comp.StatsMean, comp.StatsVar, gamma, beta)
// }

func makeBN(comp *KaldiComponent) (*gpu.BNParams, error) {
	dim := len(comp.StatsMean)
	if dim == 0 {
		return nil, fmt.Errorf("empty StatsMean")
	}
	targetRms := comp.TargetRms
	if targetRms <= 0 {
		targetRms = 1.0
	}
	eps := comp.Epsilon
	if eps <= 0 {
		eps = 0.001
	}
	gamma := make([]float32, dim)
	beta := make([]float32, dim)
	for i := 0; i < dim; i++ {
		gamma[i] = targetRms
		beta[i] = 0
	}
	bn, err := gpu.NewBNParams(comp.StatsMean, comp.StatsVar, gamma, beta)
	if err != nil {
		return nil, err
	}
	bn.Epsilon = eps
	return bn, nil
}

// makeBlockBN creates BNParams by repeating block params across height positions.
// Kaldi CNN uses BlockDim=num_filters, actual dim = height * num_filters.
// BN params are [num_filters], we tile them to [height * num_filters].
// makeBlockBN — tile params to match height-major layout [h0_f0, h0_f1, ..., h1_f0, ...]
// BN block params are per-filter, repeat for each height position
// func makeBlockBN(comp *KaldiComponent, height int) (*gpu.BNParams, error) {
// 	blockDim := len(comp.StatsMean)
// 	if blockDim == 0 {
// 		return nil, fmt.Errorf("empty StatsMean")
// 	}
// 	fullDim := blockDim * height

// 	targetRms := comp.TargetRms
// 	if targetRms <= 0 {
// 		targetRms = 1.0
// 	}
// 	eps := comp.Epsilon
// 	if eps <= 0 {
// 		eps = 0.001
// 	}

// 	mean := make([]float32, fullDim)
// 	variance := make([]float32, fullDim)
// 	gamma := make([]float32, fullDim)
// 	beta := make([]float32, fullDim)

// 	for i := 0; i < blockDim; i++ {
// 		m := comp.StatsMean[i]
// 		v := float32(0)
// 		if i < len(comp.StatsVar) {
// 			v = comp.StatsVar[i]
// 		}
// 		if v < 0 {
// 			v = 0
// 		}
// 		invStd := float32(1.0 / math.Sqrt(float64(v+eps)))
// 		g := targetRms * invStd
// 		b := -m * g

// 		// Filter-major layout: [f0_h0, f0_h1, ..., f0_hn, f1_h0, ...]
// 		for h := 0; h < height; h++ {
// 			idx := i*height + h
// 			mean[idx] = m
// 			variance[idx] = v
// 			gamma[idx] = g
// 			beta[idx] = b
// 		}
// 	}

// 	return gpu.NewBNParams(mean, variance, gamma, beta)
// }

// func makeBlockBN(comp *KaldiComponent, height int) (*gpu.BNParams, error) {
// 	blockDim := len(comp.StatsMean)
// 	if blockDim == 0 {
// 		return nil, fmt.Errorf("empty StatsMean")
// 	}
// 	fullDim := blockDim * height

// 	targetRms := comp.TargetRms
// 	if targetRms <= 0 {
// 		targetRms = 1.0
// 	}

// 	mean := make([]float32, fullDim)
// 	variance := make([]float32, fullDim)
// 	gamma := make([]float32, fullDim)
// 	beta := make([]float32, fullDim)

// 	for i := 0; i < blockDim; i++ {
// 		m := comp.StatsMean[i]
// 		v := float32(0)
// 		if i < len(comp.StatsVar) {
// 			v = comp.StatsVar[i]
// 		}
// 		if v < 0 {
// 			v = 0
// 		}
// 		for h := 0; h < height; h++ {
// 			idx := i*height + h
// 			mean[idx] = m
// 			variance[idx] = v
// 			gamma[idx] = targetRms
// 			beta[idx] = 0
// 		}
// 	}
// 	return gpu.NewBNParams(mean, variance, gamma, beta)
// }

func makeBlockBN(comp *KaldiComponent, height int) (*gpu.BNParams, error) {
	blockDim := len(comp.StatsMean)
	if blockDim == 0 {
		return nil, fmt.Errorf("empty StatsMean")
	}
	fullDim := blockDim * height

	targetRms := comp.TargetRms
	if targetRms <= 0 {
		targetRms = 1.0
	}
	eps := comp.Epsilon
	if eps <= 0 {
		eps = 0.001
	}

	mean := make([]float32, fullDim)
	variance := make([]float32, fullDim)
	gamma := make([]float32, fullDim)
	beta := make([]float32, fullDim)

	for i := 0; i < blockDim; i++ {
		m := comp.StatsMean[i]
		v := float32(0)
		if i < len(comp.StatsVar) {
			v = comp.StatsVar[i]
		}
		if v < 0 {
			v = 0
		}
		for h := 0; h < height; h++ {
			idx := i*height + h
			mean[idx] = m
			variance[idx] = v
			gamma[idx] = targetRms
			beta[idx] = 0
		}
	}
	bn, err := gpu.NewBNParams(mean, variance, gamma, beta)
	if err != nil {
		return nil, err
	}
	bn.Epsilon = eps
	return bn, nil
}

// ============================================================
// Export + Parse
// ============================================================

// ExportModelText runs nnet3-copy --binary=false and returns text
func ExportModelText(modelPath string) (string, error) {
	cmd := exec.Command("nnet3-copy", "--binary=false", modelPath, "-")
	cmd.Stderr = nil
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("nnet3-copy failed: %w", err)
	}
	return string(out), nil
}

// ParseNnet3Text parses nnet3-copy --binary=false text output.
// Returns map of component name → KaldiComponent.
func ParseNnet3Text(text string) (map[string]*KaldiComponent, error) {
	components := make(map[string]*KaldiComponent)

	scanner := bufio.NewScanner(strings.NewReader(text))
	buf := make([]byte, 0, 1024*1024)
	scanner.Buffer(buf, 100*1024*1024)

	var current *KaldiComponent
	var matrixBuf []float32
	var matrixRows int
	inMatrix := false
	matrixTag := ""

	for scanner.Scan() {
		line := scanner.Text()

		// New component — fall through to check matrix tags on same line
		if strings.Contains(line, "<ComponentName>") {
			if current != nil && inMatrix {
				finishMatrix(current, matrixTag, matrixBuf, matrixRows)
				inMatrix = false
			}
			if current != nil {
				components[current.Name] = current
			}
			current = parseComponentHeader(line)
			matrixBuf = nil
			matrixRows = 0
			inMatrix = false
			matrixTag = ""
		}

		if current == nil {
			continue
		}

		// Continuation-line scalar tags
		if strings.Contains(line, "<Count>") {
			current.Count = parseFloat64(line, "<Count>")
		}
		if strings.Contains(line, "<Epsilon>") && current.Epsilon == 0 {
			current.Epsilon = parseFloat32Tag(line, "<Epsilon>")
		}
		if strings.Contains(line, "<TargetRms>") && current.TargetRms == 0 {
			current.TargetRms = parseFloat32Tag(line, "<TargetRms>")
		}

		// Detect matrix/vector start
		for _, tag := range []string{"<LinearParams>", "<Params>", "<BiasParams>", "<StatsMean>", "<StatsVar>"} {
			if !strings.Contains(line, tag) {
				continue
			}
			if inMatrix {
				finishMatrix(current, matrixTag, matrixBuf, matrixRows)
			}
			matrixTag = tag
			matrixBuf = nil
			matrixRows = 0
			inMatrix = true

			bracketIdx := strings.Index(line, "[")
			if bracketIdx >= 0 {
				after := line[bracketIdx+1:]
				if strings.Contains(after, "]") {
					after = after[:strings.Index(after, "]")]
					vals := parseFloatLine(after)
					if len(vals) > 0 {
						matrixBuf = append(matrixBuf, vals...)
						matrixRows = 1
					}
					finishMatrix(current, matrixTag, matrixBuf, matrixRows)
					inMatrix = false
				}
			}
			break
		}

		// Matrix data lines
		if inMatrix && !strings.Contains(line, "<") {
			trimmed := strings.TrimSpace(line)
			if trimmed == "" {
				continue
			}
			closeBracket := strings.Contains(trimmed, "]")
			if closeBracket {
				trimmed = strings.Replace(trimmed, "]", "", 1)
			}
			vals := parseFloatLine(trimmed)
			if len(vals) > 0 {
				matrixBuf = append(matrixBuf, vals...)
				matrixRows++
			}
			if closeBracket {
				finishMatrix(current, matrixTag, matrixBuf, matrixRows)
				inMatrix = false
			}
		}
	}

	if current != nil {
		if inMatrix {
			finishMatrix(current, matrixTag, matrixBuf, matrixRows)
		}
		components[current.Name] = current
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scanner: %w", err)
	}

	return components, nil
}

// ============================================================
// Load into Network
// ============================================================

// LoadWeightsFromFile exports Kaldi model to text, parses, and loads into Network
func LoadWeightsFromFile(net *Network, modelPath string) error {
	fmt.Printf("LoadWeights: exporting %s...\n", modelPath)
	text, err := ExportModelText(modelPath)
	if err != nil {
		return err
	}
	fmt.Printf("LoadWeights: parsing %d bytes...\n", len(text))

	components, err := ParseNnet3Text(text)
	if err != nil {
		return err
	}
	fmt.Printf("LoadWeights: %d components parsed\n", len(components))

	return LoadWeights(net, components)
}

// LoadWeights loads parsed Kaldi components into Network GPU weights.
// Replaces random-init tensors with actual Kaldi weights.
func LoadWeights(net *Network, components map[string]*KaldiComponent) error {
	loaded := 0
	skipped := 0
	totalParams := 0

	for _, layer := range net.Model.Layers {
		lw := net.Weights[layer.Name]
		if lw == nil && needsWeights(layer.Type) {
			return fmt.Errorf("no LayerWeights for %s", layer.Name)
		}

		switch layer.Type {
		case LayerIDCT:
			comp := components["idct"]
			if comp == nil {
				return fmt.Errorf("idct component not found")
			}
			n, err := replaceMatrix(&lw.IDCTMat, comp.LinearParams, comp.LinearRows, comp.LinearCols)
			if err != nil {
				return fmt.Errorf("idct: %w", err)
			}
			totalParams += n
			loaded++

		case LayerLinearComponent:
			comp := components[layer.Name]
			if comp == nil {
				return fmt.Errorf("%s not found", layer.Name)
			}
			n, err := replaceMatrix(&lw.W, comp.LinearParams, comp.LinearRows, comp.LinearCols)
			if err != nil {
				return fmt.Errorf("%s: %w", layer.Name, err)
			}
			totalParams += n
			loaded++

		case LayerBatchnormComponent:
			comp := components[layer.Name]
			if comp == nil {
				return fmt.Errorf("%s not found", layer.Name)
			}
			n, err := replaceBN(&lw.BN, comp)
			if err != nil {
				return fmt.Errorf("%s: %w", layer.Name, err)
			}
			totalParams += n
			loaded++

		case LayerConvReluBatchnorm:
			base := layer.Name // e.g. "cnn1"
			conv := components[base+".conv"]
			if conv == nil {
				return fmt.Errorf("%s.conv not found", base)
			}
			n, err := replaceMatrix(&lw.W, conv.LinearParams, conv.LinearRows, conv.LinearCols)
			if err != nil {
				return fmt.Errorf("%s.conv W: %w", base, err)
			}
			totalParams += n
			if len(conv.BiasParams) > 0 {
				n, err = replaceVector(&lw.Bias, conv.BiasParams)
				if err != nil {
					return fmt.Errorf("%s.conv bias: %w", base, err)
				}
				totalParams += n
			}
			bn := components[base+".batchnorm"]
			if bn == nil {
				return fmt.Errorf("%s.batchnorm not found", base)
			}
			n, err = replaceBN(&lw.BN, bn)
			if err != nil {
				return fmt.Errorf("%s.batchnorm: %w", base, err)
			}
			totalParams += n
			loaded++

		case LayerTDNNF:
			base := layer.Name // e.g. "tdnnf7"
			lin := components[base+".linear"]
			if lin == nil {
				return fmt.Errorf("%s.linear not found", base)
			}
			n, err := replaceMatrix(&lw.LinearW, lin.LinearParams, lin.LinearRows, lin.LinearCols)
			if err != nil {
				return fmt.Errorf("%s.linear: %w", base, err)
			}
			totalParams += n

			aff := components[base+".affine"]
			if aff == nil {
				return fmt.Errorf("%s.affine not found", base)
			}
			n, err = replaceMatrix(&lw.AffineW, aff.LinearParams, aff.LinearRows, aff.LinearCols)
			if err != nil {
				return fmt.Errorf("%s.affine W: %w", base, err)
			}
			totalParams += n
			if len(aff.BiasParams) > 0 {
				n, err = replaceVector(&lw.AffineBias, aff.BiasParams)
				if err != nil {
					return fmt.Errorf("%s.affine bias: %w", base, err)
				}
				totalParams += n
			}

			bnComp := components[base+".batchnorm"]
			if bnComp == nil {
				return fmt.Errorf("%s.batchnorm not found", base)
			}
			n, err = replaceBN(&lw.AffBN, bnComp)
			if err != nil {
				return fmt.Errorf("%s.batchnorm: %w", base, err)
			}
			totalParams += n
			loaded++

		case LayerPrefinal:
			prefix := "prefinal-chain"
			if strings.Contains(layer.Name, "xent") {
				prefix = "prefinal-xent"
			}

			aff := components[prefix+".affine"]
			if aff == nil {
				return fmt.Errorf("%s.affine not found", prefix)
			}
			n, err := replaceMatrix(&lw.BigW, aff.LinearParams, aff.LinearRows, aff.LinearCols)
			if err != nil {
				return fmt.Errorf("%s.affine: %w", prefix, err)
			}
			totalParams += n
			if len(aff.BiasParams) > 0 {
				n, err = replaceVector(&lw.BigBias, aff.BiasParams)
				if err != nil {
					return fmt.Errorf("%s.affine bias: %w", prefix, err)
				}
				totalParams += n
			}

			lin := components[prefix+".linear"]
			if lin == nil {
				return fmt.Errorf("%s.linear not found", prefix)
			}
			n, err = replaceMatrix(&lw.SmallW, lin.LinearParams, lin.LinearRows, lin.LinearCols)
			if err != nil {
				return fmt.Errorf("%s.linear: %w", prefix, err)
			}
			totalParams += n

			bnComp := components[prefix+".batchnorm1"]
			if bnComp == nil {
				return fmt.Errorf("%s.batchnorm1 not found", prefix)
			}
			n, err = replaceBN(&lw.PfBN, bnComp)
			if err != nil {
				return fmt.Errorf("%s.batchnorm1: %w", prefix, err)
			}
			totalParams += n
			loaded++

		case LayerOutput:
			name := layer.Name + ".affine"
			comp := components[name]
			if comp == nil {
				return fmt.Errorf("%s not found", name)
			}
			n, err := replaceMatrix(&lw.W, comp.LinearParams, comp.LinearRows, comp.LinearCols)
			if err != nil {
				return fmt.Errorf("%s W: %w", name, err)
			}
			totalParams += n
			if len(comp.BiasParams) > 0 {
				n, err = replaceVector(&lw.Bias, comp.BiasParams)
				if err != nil {
					return fmt.Errorf("%s bias: %w", name, err)
				}
				totalParams += n
			}
			loaded++

		case LayerAttentionReluBatchnorm:
			fmt.Printf("  WARNING: %s attention loading not implemented\n", layer.Name)
			skipped++

		default:
			skipped++
		}
	}

	fmt.Printf("LoadWeights: %d layers, %d params, %d skipped\n", loaded, totalParams, skipped)
	return nil
}

func needsWeights(lt LayerType) bool {
	switch lt {
	case LayerIDCT, LayerLinearComponent, LayerBatchnormComponent,
		LayerConvReluBatchnorm, LayerTDNNF, LayerPrefinal, LayerOutput,
		LayerAttentionReluBatchnorm:
		return true
	}
	return false
}

func transposeF32(data []float32, rows, cols int) ([]float32, int, int) {
	out := make([]float32, len(data))
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out[c*rows+r] = data[r*cols+c]
		}
	}
	return out, cols, rows // новые размеры: cols × rows
}

// ============================================================
// GPU tensor replacement
// ============================================================

// replaceMatrix frees old tensor, creates new FP16 from FP32 data via gpu.TensorFromFP32
func replaceMatrix(tp **gpu.Tensor, data []float32, rows, cols int) (int, error) {
	if len(data) == 0 {
		return 0, nil
	}
	if len(data) != rows*cols {
		return 0, fmt.Errorf("data %d != %dx%d", len(data), rows, cols)
	}
	if *tp != nil {
		(*tp).Free()
	}
	tData, tR, tC := transposeF32(data, rows, cols)
	t, err := gpu.TensorFromFP32(tData, tR, tC)
	if err != nil {
		return 0, err
	}
	*tp = t
	return rows * cols, nil
}

// replaceVector creates [1 x N] FP16 tensor from FP32 vector
func replaceVector(tp **gpu.Tensor, data []float32) (int, error) {
	if len(data) == 0 {
		return 0, nil
	}
	if *tp != nil {
		(*tp).Free()
	}
	t, err := gpu.TensorFromFP32(data, 1, len(data))
	if err != nil {
		return 0, err
	}
	*tp = t
	return len(data), nil
}

// replaceBN computes gamma/beta from Kaldi running stats, creates gpu.BNParams
// func replaceBN(bnp **gpu.BNParams, comp *KaldiComponent) (int, error) {
// 	dim := len(comp.StatsMean)
// 	if dim == 0 {
// 		return 0, fmt.Errorf("empty StatsMean")
// 	}

// 	targetRms := comp.TargetRms
// 	if targetRms <= 0 {
// 		targetRms = 1.0
// 	}
// 	eps := comp.Epsilon
// 	if eps <= 0 {
// 		eps = 0.001
// 	}

// 	mean := make([]float32, dim)
// 	variance := make([]float32, dim)
// 	gamma := make([]float32, dim)
// 	beta := make([]float32, dim)

// 	for i := 0; i < dim; i++ {
// 		m := comp.StatsMean[i]
// 		ex2 := float32(0)
// 		if i < len(comp.StatsVar) {
// 			ex2 = comp.StatsVar[i]
// 		}
// 		v := ex2 // StatsVar already contains variance, not E[x^2]
// 		if v < 0 {
// 			v = 0
// 		}
// 		invStd := float32(1.0 / math.Sqrt(float64(v+eps)))
// 		mean[i] = m
// 		variance[i] = v
// 		gamma[i] = targetRms * invStd
// 		beta[i] = -m * gamma[i]
// 	}

// 	if *bnp != nil {
// 		(*bnp).Free()
// 	}
// 	bn, err := gpu.NewBNParams(mean, variance, gamma, beta)
// 	if err != nil {
// 		return 0, err
// 	}
// 	*bnp = bn
// 	return dim * 4, nil
// }

func replaceBN(bnp **gpu.BNParams, comp *KaldiComponent) (int, error) {
	dim := len(comp.StatsMean)
	if dim == 0 {
		return 0, fmt.Errorf("empty StatsMean")
	}

	targetRms := comp.TargetRms
	if targetRms <= 0 {
		targetRms = 1.0
	}
	eps := comp.Epsilon
	if eps <= 0 {
		eps = 0.001
	}

	mean := make([]float32, dim)
	variance := make([]float32, dim)
	gamma := make([]float32, dim)
	beta := make([]float32, dim)

	for i := 0; i < dim; i++ {
		m := comp.StatsMean[i]
		ex2 := float32(0)
		if i < len(comp.StatsVar) {
			ex2 = comp.StatsVar[i]
		}
		v := ex2
		if v < 0 {
			v = 0
		}
		invStd := float32(1.0 / math.Sqrt(float64(v+eps)))
		mean[i] = m
		variance[i] = v
		gamma[i] = targetRms * invStd
		beta[i] = -m * gamma[i]
	}

	if *bnp != nil {
		(*bnp).Free()
	}
	bn, err := gpu.NewBNParams(mean, variance, gamma, beta)
	if err != nil {
		return 0, err
	}
	bn.Epsilon = eps
	*bnp = bn
	return dim * 4, nil
}

// ============================================================
// Text parsing helpers
// ============================================================

func parseComponentHeader(line string) *KaldiComponent {
	comp := &KaldiComponent{}
	idx := strings.Index(line, "<ComponentName>")
	if idx < 0 {
		return comp
	}
	parts := strings.Fields(line[idx+len("<ComponentName>"):])
	if len(parts) < 2 {
		return comp
	}
	comp.Name = parts[0]
	comp.Type = strings.Trim(parts[1], "<>")
	comp.LearningRate = parseFloat32Tag(line, "<LearningRate>")
	comp.MaxChange = parseFloat32Tag(line, "<MaxChange>")
	comp.L2Regularize = parseFloat32Tag(line, "<L2Regularize>")
	comp.Epsilon = parseFloat32Tag(line, "<Epsilon>")
	comp.TargetRms = parseFloat32Tag(line, "<TargetRms>")
	comp.Count = parseFloat64(line, "<Count>")
	comp.NumFiltersIn = parseIntTag(line, "<NumFiltersIn>")
	comp.NumFiltersOut = parseIntTag(line, "<NumFiltersOut>")
	comp.HeightIn = parseIntTag(line, "<HeightIn>")
	comp.HeightOut = parseIntTag(line, "<HeightOut>")
	comp.NumHeads = parseIntTag(line, "<NumHeads>")
	comp.KeyDim = parseIntTag(line, "<KeyDim>")
	comp.ValueDim = parseIntTag(line, "<ValueDim>")
	comp.KeyScale = parseFloat32Tag(line, "<KeyScale>")
	return comp
}

func finishMatrix(comp *KaldiComponent, tag string, data []float32, rows int) {
	if len(data) == 0 {
		return
	}
	cols := 0
	if rows > 0 {
		cols = len(data) / rows
	}
	switch tag {
	case "<LinearParams>", "<Params>":
		comp.LinearParams = data
		comp.LinearRows = rows
		comp.LinearCols = cols
	case "<BiasParams>":
		comp.BiasParams = data
	case "<StatsMean>":
		comp.StatsMean = data
	case "<StatsVar>":
		comp.StatsVar = data
	}
}

func parseFloatLine(line string) []float32 {
	fields := strings.Fields(line)
	vals := make([]float32, 0, len(fields))
	for _, f := range fields {
		v, err := strconv.ParseFloat(f, 32)
		if err != nil {
			continue
		}
		vals = append(vals, float32(v))
	}
	return vals
}

func parseFloat32Tag(line, tag string) float32 {
	idx := strings.Index(line, tag)
	if idx < 0 {
		return 0
	}
	fields := strings.Fields(line[idx+len(tag):])
	if len(fields) == 0 || strings.HasPrefix(fields[0], "<") {
		return 0
	}
	v, _ := strconv.ParseFloat(fields[0], 32)
	return float32(v)
}

func parseFloat64(line, tag string) float64 {
	idx := strings.Index(line, tag)
	if idx < 0 {
		return 0
	}
	fields := strings.Fields(line[idx+len(tag):])
	if len(fields) == 0 || strings.HasPrefix(fields[0], "<") {
		return 0
	}
	v, _ := strconv.ParseFloat(fields[0], 64)
	return v
}

func parseIntTag(line, tag string) int {
	idx := strings.Index(line, tag)
	if idx < 0 {
		return 0
	}
	fields := strings.Fields(line[idx+len(tag):])
	if len(fields) == 0 || strings.HasPrefix(fields[0], "<") {
		return 0
	}
	v, _ := strconv.Atoi(fields[0])
	return v
}
