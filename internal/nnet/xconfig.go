package nnet

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// ============================================================
// xconfig parser — reads Kaldi-style network.xconfig files
// ============================================================

// LayerType enumerates supported xconfig layer types
type LayerType int

const (
	LayerInput LayerType = iota
	LayerIDCT
	LayerLinearComponent
	LayerBatchnormComponent
	LayerSpecAugment
	LayerCombineFeatureMaps
	LayerConvReluBatchnorm
	LayerTDNNF
	LayerAttentionReluBatchnorm
	LayerPrefinal
	LayerOutput
)

var layerTypeNames = map[LayerType]string{
	LayerInput:                  "input",
	LayerIDCT:                   "idct-layer",
	LayerLinearComponent:        "linear-component",
	LayerBatchnormComponent:     "batchnorm-component",
	LayerSpecAugment:            "spec-augment-layer",
	LayerCombineFeatureMaps:     "combine-feature-maps-layer",
	LayerConvReluBatchnorm:      "conv-relu-batchnorm-layer",
	LayerTDNNF:                  "tdnnf-layer",
	LayerAttentionReluBatchnorm: "attention-relu-batchnorm-layer",
	LayerPrefinal:               "prefinal-layer",
	LayerOutput:                 "output-layer",
}

func (lt LayerType) String() string {
	if s, ok := layerTypeNames[lt]; ok {
		return s
	}
	return fmt.Sprintf("unknown(%d)", int(lt))
}

var layerTypeFromString = map[string]LayerType{
	"input":                          LayerInput,
	"idct-layer":                     LayerIDCT,
	"linear-component":               LayerLinearComponent,
	"batchnorm-component":            LayerBatchnormComponent,
	"spec-augment-layer":             LayerSpecAugment,
	"combine-feature-maps-layer":     LayerCombineFeatureMaps,
	"conv-relu-batchnorm-layer":      LayerConvReluBatchnorm,
	"tdnnf-layer":                    LayerTDNNF,
	"attention-relu-batchnorm-layer": LayerAttentionReluBatchnorm,
	"prefinal-layer":                 LayerPrefinal,
	"output-layer":                   LayerOutput,
}

// LayerConfig holds parsed parameters for one xconfig line
type LayerConfig struct {
	Type   LayerType
	Name   string
	Params map[string]string // raw key=value pairs
	Line   int               // source line number
}

// ============================================================
// Convenience getters
// ============================================================

func (lc *LayerConfig) GetString(key, defaultVal string) string {
	if v, ok := lc.Params[key]; ok {
		return v
	}
	return defaultVal
}

func (lc *LayerConfig) GetInt(key string, defaultVal int) int {
	if v, ok := lc.Params[key]; ok {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return defaultVal
}

func (lc *LayerConfig) GetFloat(key string, defaultVal float64) float64 {
	if v, ok := lc.Params[key]; ok {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return defaultVal
}

func (lc *LayerConfig) GetBool(key string, defaultVal bool) bool {
	if v, ok := lc.Params[key]; ok {
		switch strings.ToLower(v) {
		case "true", "1", "yes":
			return true
		case "false", "0", "no":
			return false
		}
	}
	return defaultVal
}

// GetIntSlice parses comma-separated ints like "-1,0,1"
func (lc *LayerConfig) GetIntSlice(key string) []int {
	v, ok := lc.Params[key]
	if !ok || v == "" {
		return nil
	}
	parts := strings.Split(v, ",")
	result := make([]int, 0, len(parts))
	for _, p := range parts {
		if n, err := strconv.Atoi(strings.TrimSpace(p)); err == nil {
			result = append(result, n)
		}
	}
	return result
}

// InputSpec returns the "input" field, or "" if not specified
// Handles: simple name, Append(a,b,c), ReplaceIndex(x,t,0)
func (lc *LayerConfig) InputSpec() string {
	return lc.GetString("input", "")
}

// ============================================================
// Parse xconfig file
// ============================================================

// ParseXConfig reads an xconfig file and returns layer configs
func ParseXConfig(path string) ([]*LayerConfig, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open xconfig: %w", err)
	}
	defer f.Close()

	return ParseXConfigReader(bufio.NewReader(f))
}

// ParseXConfigString parses xconfig from a string
func ParseXConfigString(s string) ([]*LayerConfig, error) {
	return ParseXConfigReader(bufio.NewReader(strings.NewReader(s)))
}

// ParseXConfigReader parses xconfig from a reader
func ParseXConfigReader(r *bufio.Reader) ([]*LayerConfig, error) {
	var layers []*LayerConfig
	scanner := bufio.NewScanner(r)
	lineNum := 0

	for scanner.Scan() {
		lineNum++
		line := strings.TrimSpace(scanner.Text())

		// Skip empty lines and comments
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		lc, err := parseLine(line, lineNum)
		if err != nil {
			return nil, fmt.Errorf("line %d: %w", lineNum, err)
		}
		if lc != nil {
			layers = append(layers, lc)
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("reading xconfig: %w", err)
	}

	return layers, nil
}

// parseLine parses a single xconfig line like:
//
//	conv-relu-batchnorm-layer name=cnn1 l2-regularize=0.03 height-in=40 ...
func parseLine(line string, lineNum int) (*LayerConfig, error) {
	// Tokenize carefully — values can contain parens like Append(a, b)
	tokens := tokenize(line)
	if len(tokens) == 0 {
		return nil, nil
	}

	// First token is layer type
	typeName := tokens[0]
	lt, ok := layerTypeFromString[typeName]
	if !ok {
		return nil, fmt.Errorf("unknown layer type: %q", typeName)
	}

	lc := &LayerConfig{
		Type:   lt,
		Params: make(map[string]string),
		Line:   lineNum,
	}

	// Parse remaining key=value pairs
	for _, tok := range tokens[1:] {
		idx := strings.Index(tok, "=")
		if idx < 0 {
			// Bare token — could be a variable like $cnn_opts that wasn't expanded
			// Skip silently (in real Kaldi, shell expands these)
			continue
		}
		key := tok[:idx]
		val := tok[idx+1:]
		lc.Params[key] = val
	}

	lc.Name = lc.GetString("name", "")
	if lc.Name == "" && lt != LayerInput {
		return nil, fmt.Errorf("layer missing name")
	}

	// For input layers, "name" might be the name or we generate one
	if lt == LayerInput && lc.Name == "" {
		lc.Name = fmt.Sprintf("input_%d", lineNum)
	}

	return lc, nil
}

// tokenize splits an xconfig line into tokens, respecting parentheses
// "conv-relu-batchnorm-layer name=cnn1 input=Append(a, b)" →
//
//	["conv-relu-batchnorm-layer", "name=cnn1", "input=Append(a, b)"]
func tokenize(line string) []string {
	var tokens []string
	var current strings.Builder
	depth := 0

	for _, ch := range line {
		switch {
		case ch == '(':
			depth++
			current.WriteRune(ch)
		case ch == ')':
			depth--
			current.WriteRune(ch)
		case ch == ' ' || ch == '\t':
			if depth > 0 {
				// Inside parens — keep spaces
				current.WriteRune(ch)
			} else if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
		default:
			current.WriteRune(ch)
		}
	}
	if current.Len() > 0 {
		tokens = append(tokens, current.String())
	}
	return tokens
}

// ============================================================
// Input parsing helpers
// ============================================================

// InputType describes how a layer gets its input
type InputType int

const (
	InputSimple   InputType = iota // name of previous layer
	InputAppend                    // Append(a, b, c)
	InputReplace                   // ReplaceIndex(x, t, 0)
	InputPrevious                  // implicit — use previous layer
)

// InputRef describes a layer's input
type InputRef struct {
	Type   InputType
	Name   string   // for InputSimple
	Names  []string // for InputAppend
	Source string   // for InputReplace — the source name
}

// ParseInput parses input specification from xconfig
func ParseInput(spec string) InputRef {
	spec = strings.TrimSpace(spec)

	if spec == "" {
		return InputRef{Type: InputPrevious}
	}

	// Append(a, b, c)
	if strings.HasPrefix(spec, "Append(") && strings.HasSuffix(spec, ")") {
		inner := spec[7 : len(spec)-1]
		parts := strings.Split(inner, ",")
		names := make([]string, len(parts))
		for i, p := range parts {
			names[i] = strings.TrimSpace(p)
		}
		return InputRef{Type: InputAppend, Names: names}
	}

	// ReplaceIndex(name, t, 0)
	if strings.HasPrefix(spec, "ReplaceIndex(") && strings.HasSuffix(spec, ")") {
		inner := spec[13 : len(spec)-1]
		parts := strings.Split(inner, ",")
		if len(parts) >= 1 {
			return InputRef{Type: InputReplace, Source: strings.TrimSpace(parts[0])}
		}
	}

	// Simple name
	return InputRef{Type: InputSimple, Name: spec}
}

// ============================================================
// Printing / debugging
// ============================================================

// String returns a human-readable summary of layer config
func (lc *LayerConfig) String() string {
	s := fmt.Sprintf("%-30s name=%-20s", lc.Type, lc.Name)
	if input := lc.InputSpec(); input != "" {
		s += fmt.Sprintf(" input=%s", input)
	}
	if dim := lc.GetInt("dim", 0); dim > 0 {
		s += fmt.Sprintf(" dim=%d", dim)
	}
	return s
}

// PrintConfigs prints all layer configs for debugging
func PrintConfigs(layers []*LayerConfig) {
	for i, lc := range layers {
		fmt.Printf("  [%2d] %s\n", i, lc)
	}
}
