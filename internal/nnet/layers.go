package nnet

import (
	"fmt"
)

// ============================================================
// Layer — resolved layer with computed dimensions
// ============================================================

// Layer represents a resolved network layer with known dimensions
type Layer struct {
	Config    *LayerConfig
	Name      string
	Type      LayerType
	InputDim  int
	OutputDim int

	// Input references (resolved)
	Input      InputRef
	InputNames []string // resolved list of input layer names

	// Type-specific parameters (resolved)
	Spec interface{} // one of the *Spec types below
}

// ============================================================
// Layer specs — type-specific resolved parameters
// ============================================================

type InputLayerSpec struct {
	Dim int
}

type IDCTSpec struct {
	Dim            int
	CepstralLifter float64
	AffineFile     string
}

type LinearSpec struct {
	InputDim              int
	OutputDim             int
	L2Reg                 float64
	OrthonormalConstraint float64
}

type BatchnormSpec struct {
	Dim       int
	TargetRMS float64
}

type SpecAugmentSpec struct {
	Dim                  int
	FreqMaxProportion    float64
	TimeZeroedProportion float64
	TimeMaskMaxFrames    int
}

type CombineFeatureMapsSpec struct {
	NumFilters1 int
	NumFilters2 int
	Height      int
	InputDim    int
	OutputDim   int
}

type ConvReluBNSpec struct {
	HeightIn           int
	HeightOut          int
	HeightSubsample    int
	TimeOffsets        []int
	HeightOffsets      []int
	NumFiltersIn       int
	NumFiltersOut      int
	InputDim           int
	OutputDim          int
	L2Reg              float64
	LearningRateFactor float64
	MaxChange          float64
}

type TDNNFSpec struct {
	InputDim      int
	OutputDim     int
	BottleneckDim int
	TimeStride    int
	BypassScale   float64
	L2Reg         float64
}

type AttentionSpec struct {
	InputDim       int
	OutputDim      int
	NumHeads       int
	ValueDim       int
	KeyDim         int
	NumLeftInputs  int
	NumRightInputs int
	ContextDim     int
	TimeStride     int
	KeyScale       float64
	L2Reg          float64
}

type PrefinalSpec struct {
	InputDim  int
	SmallDim  int
	BigDim    int
	OutputDim int
	L2Reg     float64
}

type OutputSpec struct {
	InputDim           int
	OutputDim          int
	IncludeLogSoftmax  bool
	L2Reg              float64
	LearningRateFactor float64
}

// ============================================================
// Resolve layers — compute dimensions from xconfig
// ============================================================

func ResolveLayers(configs []*LayerConfig) ([]*Layer, error) {
	layerMap := make(map[string]*Layer)
	var layers []*Layer

	for i, cfg := range configs {
		layer, err := resolveLayer(cfg, layerMap, layers, i)
		if err != nil {
			return nil, fmt.Errorf("layer %q (line %d): %w", cfg.Name, cfg.Line, err)
		}
		layers = append(layers, layer)
		layerMap[layer.Name] = layer
	}

	return layers, nil
}

func resolveLayer(cfg *LayerConfig, layerMap map[string]*Layer, prev []*Layer, idx int) (*Layer, error) {
	layer := &Layer{
		Config: cfg,
		Name:   cfg.Name,
		Type:   cfg.Type,
		Input:  ParseInput(cfg.InputSpec()),
	}

	// Resolve input dimensions
	switch layer.Input.Type {
	case InputPrevious:
		if idx > 0 {
			prevLayer := prev[idx-1]
			layer.InputDim = prevLayer.OutputDim
			layer.InputNames = []string{prevLayer.Name}
		}
	case InputSimple:
		if src, ok := layerMap[layer.Input.Name]; ok {
			layer.InputDim = src.OutputDim
			layer.InputNames = []string{layer.Input.Name}
		} else {
			resolved := resolveLayerName(layer.Input.Name, layerMap)
			if resolved != nil {
				layer.InputDim = resolved.OutputDim
				layer.InputNames = []string{resolved.Name}
			} else {
				return nil, fmt.Errorf("input %q not found", layer.Input.Name)
			}
		}
	case InputAppend:
		totalDim := 0
		for _, name := range layer.Input.Names {
			src := resolveLayerName(name, layerMap)
			if src == nil {
				return nil, fmt.Errorf("append input %q not found", name)
			}
			totalDim += src.OutputDim
			layer.InputNames = append(layer.InputNames, src.Name)
		}
		layer.InputDim = totalDim
	case InputReplace:
		src := resolveLayerName(layer.Input.Source, layerMap)
		if src == nil {
			return nil, fmt.Errorf("replace input %q not found", layer.Input.Source)
		}
		layer.InputDim = src.OutputDim
		layer.InputNames = []string{src.Name}
	}

	// Resolve type-specific parameters
	switch cfg.Type {
	case LayerInput:
		dim := cfg.GetInt("dim", 0)
		if dim <= 0 {
			return nil, fmt.Errorf("input layer missing dim")
		}
		layer.OutputDim = dim
		layer.InputDim = dim
		layer.Spec = &InputLayerSpec{Dim: dim}

	case LayerIDCT:
		dim := cfg.GetInt("dim", layer.InputDim)
		layer.OutputDim = dim
		layer.Spec = &IDCTSpec{
			Dim:            dim,
			CepstralLifter: cfg.GetFloat("cepstral-lifter", 22),
			AffineFile:     cfg.GetString("affine-transform-file", ""),
		}

	case LayerLinearComponent:
		dim := cfg.GetInt("dim", 0)
		if dim <= 0 {
			return nil, fmt.Errorf("linear-component missing dim")
		}
		layer.OutputDim = dim
		layer.Spec = &LinearSpec{
			InputDim:              layer.InputDim,
			OutputDim:             dim,
			L2Reg:                 cfg.GetFloat("l2-regularize", 0),
			OrthonormalConstraint: cfg.GetFloat("orthonormal-constraint", 0),
		}

	case LayerBatchnormComponent:
		layer.OutputDim = layer.InputDim
		layer.Spec = &BatchnormSpec{
			Dim:       layer.InputDim,
			TargetRMS: cfg.GetFloat("target-rms", 1.0),
		}

	case LayerSpecAugment:
		layer.OutputDim = layer.InputDim
		layer.Spec = &SpecAugmentSpec{
			Dim:                  layer.InputDim,
			FreqMaxProportion:    cfg.GetFloat("freq-max-proportion", 0.5),
			TimeZeroedProportion: cfg.GetFloat("time-zeroed-proportion", 0),
			TimeMaskMaxFrames:    cfg.GetInt("time-mask-max-frames", 20),
		}

	case LayerCombineFeatureMaps:
		height := cfg.GetInt("height", 0)
		nf1 := cfg.GetInt("num-filters1", 1)
		nf2 := cfg.GetInt("num-filters2", 1)
		layer.OutputDim = layer.InputDim
		layer.Spec = &CombineFeatureMapsSpec{
			NumFilters1: nf1,
			NumFilters2: nf2,
			Height:      height,
			InputDim:    layer.InputDim,
			OutputDim:   layer.InputDim,
		}

	case LayerConvReluBatchnorm:
		heightIn := cfg.GetInt("height-in", 0)
		heightOut := cfg.GetInt("height-out", heightIn)
		heightSubsample := cfg.GetInt("height-subsample-out", 1)
		numFiltersOut := cfg.GetInt("num-filters-out", 0)
		timeOffsets := cfg.GetIntSlice("time-offsets")
		heightOffsets := cfg.GetIntSlice("height-offsets")

		numFiltersIn := 0
		if heightIn > 0 {
			numFiltersIn = layer.InputDim / heightIn
		}

		layer.OutputDim = heightOut * numFiltersOut
		layer.Spec = &ConvReluBNSpec{
			HeightIn:           heightIn,
			HeightOut:          heightOut,
			HeightSubsample:    heightSubsample,
			TimeOffsets:        timeOffsets,
			HeightOffsets:      heightOffsets,
			NumFiltersIn:       numFiltersIn,
			NumFiltersOut:      numFiltersOut,
			InputDim:           layer.InputDim,
			OutputDim:          layer.OutputDim,
			L2Reg:              cfg.GetFloat("l2-regularize", 0),
			LearningRateFactor: cfg.GetFloat("learning-rate-factor", 1.0),
			MaxChange:          cfg.GetFloat("max-change", 0.75),
		}

	case LayerTDNNF:
		dim := cfg.GetInt("dim", 0)
		bnDim := cfg.GetInt("bottleneck-dim", 0)
		if dim <= 0 || bnDim <= 0 {
			return nil, fmt.Errorf("tdnnf-layer missing dim or bottleneck-dim")
		}
		layer.OutputDim = dim
		layer.Spec = &TDNNFSpec{
			InputDim:      layer.InputDim,
			OutputDim:     dim,
			BottleneckDim: bnDim,
			TimeStride:    cfg.GetInt("time-stride", 3),
			BypassScale:   cfg.GetFloat("bypass-scale", 0.66),
			L2Reg:         cfg.GetFloat("l2-regularize", 0),
		}

	case LayerAttentionReluBatchnorm:
		numHeads := cfg.GetInt("num-heads", 1)
		valueDim := cfg.GetInt("value-dim", 0)
		keyDim := cfg.GetInt("key-dim", 0)
		numLeftInputs := cfg.GetInt("num-left-inputs", 0)
		numRightInputs := cfg.GetInt("num-right-inputs", 0)
		contextDim := 1 + numLeftInputs + numRightInputs
		// Kaldi: output = num_heads * (value_dim + context_dim)
		layer.OutputDim = numHeads * (valueDim + contextDim)
		layer.Spec = &AttentionSpec{
			InputDim:       layer.InputDim,
			OutputDim:      numHeads * (valueDim + contextDim),
			NumHeads:       numHeads,
			ValueDim:       valueDim,
			KeyDim:         keyDim,
			NumLeftInputs:  numLeftInputs,
			NumRightInputs: numRightInputs,
			ContextDim:     contextDim,
			TimeStride:     cfg.GetInt("time-stride", 1),
			L2Reg:          cfg.GetFloat("l2-regularize", 0),
		}

	case LayerPrefinal:
		smallDim := cfg.GetInt("small-dim", 0)
		bigDim := cfg.GetInt("big-dim", 0)
		if smallDim <= 0 || bigDim <= 0 {
			return nil, fmt.Errorf("prefinal-layer missing small-dim or big-dim")
		}
		// Kaldi prefinal output = batchnorm2 (small dim)
		layer.OutputDim = smallDim
		layer.Spec = &PrefinalSpec{
			InputDim:  layer.InputDim,
			SmallDim:  smallDim,
			BigDim:    bigDim,
			OutputDim: smallDim,
			L2Reg:     cfg.GetFloat("l2-regularize", 0),
		}

	case LayerOutput:
		dim := cfg.GetInt("dim", 0)
		if dim <= 0 {
			return nil, fmt.Errorf("output-layer missing dim")
		}
		layer.OutputDim = dim
		layer.Spec = &OutputSpec{
			InputDim:           layer.InputDim,
			OutputDim:          dim,
			IncludeLogSoftmax:  cfg.GetBool("include-log-softmax", true),
			L2Reg:              cfg.GetFloat("l2-regularize", 0),
			LearningRateFactor: cfg.GetFloat("learning-rate-factor", 1.0),
		}

	default:
		return nil, fmt.Errorf("unsupported layer type: %s", cfg.Type)
	}

	return layer, nil
}

func resolveLayerName(name string, layerMap map[string]*Layer) *Layer {
	if l, ok := layerMap[name]; ok {
		return l
	}
	var best *Layer
	for lname, l := range layerMap {
		if lname == name || hasPrefix(lname, name) {
			if best == nil || l.Config.Line > best.Config.Line {
				best = l
			}
		}
	}
	return best
}

func hasPrefix(s, prefix string) bool {
	return len(s) > len(prefix) && s[:len(prefix)] == prefix && s[len(prefix)] == '.'
}
