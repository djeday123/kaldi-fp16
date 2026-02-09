package nnet

import (
	"fmt"
	"strings"
)

// ============================================================
// Model — network assembled from xconfig layers
// ============================================================

// Model represents a complete neural network parsed from xconfig
type Model struct {
	Layers    []*Layer
	LayerMap  map[string]*Layer
	Inputs    []*Layer // input layers
	Outputs   []*Layer // output layers
	NumParams int      // estimated number of parameters
}

// BuildModel parses an xconfig file and builds a model
func BuildModel(xconfigPath string) (*Model, error) {
	configs, err := ParseXConfig(xconfigPath)
	if err != nil {
		return nil, fmt.Errorf("parse xconfig: %w", err)
	}
	return BuildModelFromConfigs(configs)
}

// BuildModelFromString parses xconfig from string and builds a model
func BuildModelFromString(xconfig string) (*Model, error) {
	configs, err := ParseXConfigString(xconfig)
	if err != nil {
		return nil, fmt.Errorf("parse xconfig: %w", err)
	}
	return BuildModelFromConfigs(configs)
}

// BuildModelFromConfigs builds a model from parsed layer configs
func BuildModelFromConfigs(configs []*LayerConfig) (*Model, error) {
	layers, err := ResolveLayers(configs)
	if err != nil {
		return nil, fmt.Errorf("resolve layers: %w", err)
	}

	m := &Model{
		Layers:   layers,
		LayerMap: make(map[string]*Layer),
	}

	for _, l := range layers {
		m.LayerMap[l.Name] = l

		if l.Type == LayerInput {
			m.Inputs = append(m.Inputs, l)
		}
		if l.Type == LayerOutput {
			m.Outputs = append(m.Outputs, l)
		}
	}

	m.NumParams = m.estimateParams()

	return m, nil
}

// GetLayer returns a layer by name
func (m *Model) GetLayer(name string) *Layer {
	if l, ok := m.LayerMap[name]; ok {
		return l
	}
	return resolveLayerName(name, m.LayerMap)
}

// ============================================================
// Summary
// ============================================================

// Summary prints a human-readable model summary
func (m *Model) Summary() string {
	var sb strings.Builder

	sb.WriteString("============================================================\n")
	sb.WriteString("Network Architecture (from xconfig)\n")
	sb.WriteString("============================================================\n\n")

	// Inputs
	sb.WriteString("Inputs:\n")
	for _, l := range m.Inputs {
		fmt.Fprintf(&sb, "  %-20s dim=%d\n", l.Name, l.OutputDim)
	}
	sb.WriteString("\n")

	// Layers table
	fmt.Fprintf(&sb, "%-4s %-32s %-12s %8s → %-8s  %s\n",
		"#", "Name", "Type", "InDim", "OutDim", "Details")
	sb.WriteString(strings.Repeat("-", 90) + "\n")

	for i, l := range m.Layers {
		if l.Type == LayerInput {
			continue
		}
		detail := layerDetail(l)
		typeName := shortTypeName(l.Type)
		fmt.Fprintf(&sb, "%-4d %-32s %-12s %8d → %-8d %s\n",
			i, l.Name, typeName, l.InputDim, l.OutputDim, detail)
	}

	// Outputs
	sb.WriteString("\nOutputs:\n")
	for _, l := range m.Outputs {
		spec := l.Spec.(*OutputSpec)
		logSoftmax := "no"
		if spec.IncludeLogSoftmax {
			logSoftmax = "yes"
		}
		fmt.Fprintf(&sb, "  %-20s dim=%d  log-softmax=%s\n", l.Name, l.OutputDim, logSoftmax)
	}

	fmt.Fprintf(&sb, "\nEstimated parameters: %s\n", formatCount(m.NumParams))

	return sb.String()
}

func shortTypeName(lt LayerType) string {
	switch lt {
	case LayerIDCT:
		return "idct"
	case LayerLinearComponent:
		return "linear"
	case LayerBatchnormComponent:
		return "batchnorm"
	case LayerSpecAugment:
		return "specaug"
	case LayerCombineFeatureMaps:
		return "combine"
	case LayerConvReluBatchnorm:
		return "conv+relu+bn"
	case LayerTDNNF:
		return "tdnnf"
	case LayerAttentionReluBatchnorm:
		return "attention"
	case LayerPrefinal:
		return "prefinal"
	case LayerOutput:
		return "output"
	default:
		return lt.String()
	}
}

func layerDetail(l *Layer) string {
	switch s := l.Spec.(type) {
	case *ConvReluBNSpec:
		return fmt.Sprintf("h:%d→%d filt:%d→%d t:%v h:%v",
			s.HeightIn, s.HeightOut, s.NumFiltersIn, s.NumFiltersOut,
			s.TimeOffsets, s.HeightOffsets)
	case *TDNNFSpec:
		bypass := ""
		if s.BypassScale > 0 {
			bypass = fmt.Sprintf(" bypass=%.2f", s.BypassScale)
		}
		return fmt.Sprintf("bn=%d stride=%d%s", s.BottleneckDim, s.TimeStride, bypass)
	case *AttentionSpec:
		return fmt.Sprintf("heads=%d val=%d key=%d ctx=%d+%d",
			s.NumHeads, s.ValueDim, s.KeyDim, s.NumLeftInputs, s.NumRightInputs)
	case *PrefinalSpec:
		return fmt.Sprintf("small=%d big=%d", s.SmallDim, s.BigDim)
	case *OutputSpec:
		ls := ""
		if !s.IncludeLogSoftmax {
			ls = " no-log-softmax"
		}
		return fmt.Sprintf("dim=%d%s", s.OutputDim, ls)
	case *LinearSpec:
		return fmt.Sprintf("%d→%d", s.InputDim, s.OutputDim)
	case *BatchnormSpec:
		if s.TargetRMS != 1.0 {
			return fmt.Sprintf("target-rms=%.3f", s.TargetRMS)
		}
		return ""
	case *IDCTSpec:
		return fmt.Sprintf("lifter=%.0f", s.CepstralLifter)
	case *CombineFeatureMapsSpec:
		return fmt.Sprintf("nf1=%d nf2=%d h=%d", s.NumFilters1, s.NumFilters2, s.Height)
	default:
		return ""
	}
}

// ============================================================
// Parameter estimation
// ============================================================

func (m *Model) estimateParams() int {
	total := 0
	for _, l := range m.Layers {
		total += layerParams(l)
	}
	return total
}

func layerParams(l *Layer) int {
	switch s := l.Spec.(type) {
	case *LinearSpec:
		return s.InputDim * s.OutputDim // no bias
	case *BatchnormSpec:
		return s.Dim * 2 // scale + offset (mean/var are running stats)
	case *ConvReluBNSpec:
		// Conv kernel: filters_out * filters_in * len(time_offsets) * len(height_offsets) + bias
		kernelSize := len(s.TimeOffsets) * len(s.HeightOffsets)
		convParams := s.NumFiltersOut*s.NumFiltersIn*kernelSize + s.NumFiltersOut
		// BatchNorm: 2 * output_dim
		bnParams := s.OutputDim * 2
		return convParams + bnParams
	case *TDNNFSpec:
		// linear: input_dim * bottleneck_dim
		linear := s.InputDim * s.BottleneckDim
		// affine: bottleneck_dim * output_dim + output_dim (bias)
		// (with time-stride, input to affine is still bottleneck_dim per frame)
		affine := s.BottleneckDim*s.OutputDim + s.OutputDim
		// batchnorm: 2 * output_dim
		bn := s.OutputDim * 2
		return linear + affine + bn
	case *AttentionSpec:
		// Simplified estimate
		// key/query/value projections
		return s.InputDim*(s.KeyDim+s.KeyDim+s.ValueDim)*s.NumHeads +
			s.NumHeads*s.ValueDim*s.InputDim + s.InputDim*2
	case *PrefinalSpec:
		// linear: input→small + affine: small→big + bias + bn
		return s.InputDim*s.SmallDim + s.SmallDim*s.BigDim + s.BigDim + s.BigDim*2
	case *OutputSpec:
		// affine: input→output + bias
		return s.InputDim*s.OutputDim + s.OutputDim
	case *IDCTSpec:
		return s.Dim * s.Dim // IDCT matrix
	default:
		return 0
	}
}

func formatCount(n int) string {
	if n >= 1_000_000 {
		return fmt.Sprintf("%dM (%.1fM)", n, float64(n)/1_000_000)
	}
	if n >= 1_000 {
		return fmt.Sprintf("%dK", n/1000)
	}
	return fmt.Sprintf("%d", n)
}

// ============================================================
// Execution order (topological sort)
// ============================================================

// ExecutionOrder returns layers in forward-pass order
// (already topologically sorted since xconfig is written top-to-bottom)
func (m *Model) ExecutionOrder() []*Layer {
	// xconfig layers are already in dependency order
	// (each layer only references earlier layers)
	var order []*Layer
	for _, l := range m.Layers {
		if l.Type != LayerInput {
			order = append(order, l)
		}
	}
	return order
}

// ChainOutput returns the "output" layer (for chain training)
func (m *Model) ChainOutput() *Layer {
	for _, l := range m.Outputs {
		if l.Name == "output" {
			return l
		}
	}
	if len(m.Outputs) > 0 {
		return m.Outputs[0]
	}
	return nil
}

// XentOutput returns the "output-xent" layer (for xent regularization)
func (m *Model) XentOutput() *Layer {
	for _, l := range m.Outputs {
		if l.Name == "output-xent" {
			return l
		}
	}
	return nil
}
