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

// InputSpec defines an input layer
type InputLayerSpec struct {
	Dim int
}

// IDCTSpec — IDCT transform (fixed matrix, not trainable)
type IDCTSpec struct {
	Dim            int
	CepstralLifter float64
	AffineFile     string // path to idct.mat
}

// LinearSpec — linear projection W*x (no bias)
type LinearSpec struct {
	InputDim              int
	OutputDim             int
	L2Reg                 float64
	OrthonormalConstraint float64
}

// BatchnormSpec — batch normalization
type BatchnormSpec struct {
	Dim       int
	TargetRMS float64 // 0 = default (1.0)
}

// SpecAugmentSpec — SpecAugment (training only)
type SpecAugmentSpec struct {
	Dim                  int
	FreqMaxProportion    float64
	TimeZeroedProportion float64
	TimeMaskMaxFrames    int
}

// CombineFeatureMapsSpec — interleave feature maps
type CombineFeatureMapsSpec struct {
	NumFilters1 int // filters in first input
	NumFilters2 int // filters in second input
	Height      int
	InputDim    int
	OutputDim   int
}

// ConvReluBNSpec — conv-relu-batchnorm layer
type ConvReluBNSpec struct {
	HeightIn           int
	HeightOut          int
	HeightSubsample    int // 1 = no subsampling
	TimeOffsets        []int
	HeightOffsets      []int
	NumFiltersIn       int
	NumFiltersOut      int
	InputDim           int // height_in * num_filters_in * len(time_offsets)
	OutputDim          int // height_out * num_filters_out
	L2Reg              float64
	LearningRateFactor float64
	MaxChange          float64
}

// TDNNFSpec — factorized TDNN layer
type TDNNFSpec struct {
	InputDim      int
	OutputDim     int // = Dim
	BottleneckDim int
	TimeStride    int
	BypassScale   float64 // 0.66 default, 0.0 = no bypass
	L2Reg         float64
	// Sub-components:
	// linear: InputDim → BottleneckDim (no bias)
	// affine: BottleneckDim * num_splices → OutputDim (with bias)
	// relu
	// batchnorm
	// bypass: Scale(bypass_scale, input) + output
}

// AttentionSpec — restricted self-attention + relu + batchnorm
type AttentionSpec struct {
	InputDim       int
	OutputDim      int
	NumHeads       int
	ValueDim       int
	KeyDim         int
	NumLeftInputs  int
	NumRightInputs int
	TimeStride     int
	L2Reg          float64
}

// PrefinalSpec — prefinal layer: linear(small) → affine(big) → relu → bn
type PrefinalSpec struct {
	InputDim  int
	SmallDim  int
	BigDim    int
	OutputDim int // = BigDim
	L2Reg     float64
}

// OutputSpec — output layer: affine → optional log-softmax
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

// ResolveLayers takes parsed xconfig and resolves all dimensions
func ResolveLayers(configs []*LayerConfig) ([]*Layer, error) {
	// Map of layer name → resolved layer
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
			// Try matching with suffix (e.g. "cnn6" matches "cnn6.batchnorm")
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
		// Interleaves feature maps: rearranges [feat1 | feat2] into interleaved order
		// Output dim = input dim (just reordering)
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
		if heightIn > 0 && heightOut > 0 && heightSubsample > 1 {
			// Verify: heightOut = heightIn / heightSubsample
			expected := heightIn / heightSubsample
			if heightOut != expected {
				// Trust the config
			}
		}
		numFiltersOut := cfg.GetInt("num-filters-out", 0)
		timeOffsets := cfg.GetIntSlice("time-offsets")
		heightOffsets := cfg.GetIntSlice("height-offsets")

		// Input num_filters = InputDim / heightIn
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
		// Output dim = num_heads * (value_dim + key_dim) ... actually in Kaldi
		// it's the input dim (the attention output has same dim as input)
		layer.OutputDim = layer.InputDim
		layer.Spec = &AttentionSpec{
			InputDim:       layer.InputDim,
			OutputDim:      layer.InputDim,
			NumHeads:       numHeads,
			ValueDim:       valueDim,
			KeyDim:         keyDim,
			NumLeftInputs:  cfg.GetInt("num-left-inputs", 0),
			NumRightInputs: cfg.GetInt("num-right-inputs", 0),
			TimeStride:     cfg.GetInt("time-stride", 1),
			L2Reg:          cfg.GetFloat("l2-regularize", 0),
		}

	case LayerPrefinal:
		smallDim := cfg.GetInt("small-dim", 0)
		bigDim := cfg.GetInt("big-dim", 0)
		if smallDim <= 0 || bigDim <= 0 {
			return nil, fmt.Errorf("prefinal-layer missing small-dim or big-dim")
		}
		layer.OutputDim = bigDim
		layer.Spec = &PrefinalSpec{
			InputDim:  layer.InputDim,
			SmallDim:  smallDim,
			BigDim:    bigDim,
			OutputDim: bigDim,
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

// resolveLayerName finds a layer by name, supporting Kaldi's naming convention
// where "cnn6" refers to the last sub-component "cnn6.batchnorm"
func resolveLayerName(name string, layerMap map[string]*Layer) *Layer {
	// Exact match first
	if l, ok := layerMap[name]; ok {
		return l
	}

	// Try as prefix — find the last layer starting with "name."
	// This handles "cnn6" → "cnn6.batchnorm"
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
