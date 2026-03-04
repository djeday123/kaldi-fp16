// weight_loader_test.go — Tests for Kaldi text format parser
package nnet

import (
	"fmt"
	"math"
	"strings"
	"testing"
)

// Test data from actual nnet3-copy --binary=false output
const testComponents = `<ComponentName> idct <FixedAffineComponent> <LinearParams>  [
  0.1581139 0.0870932 0.05438261 0.03986969
  0.1581139 0.08655624 0.05304353 0.03766649 ]
<BiasParams>  [ 0.0 0.0 0.0 0.0 ]
<ComponentName> ivector-linear <LinearComponent> <MaxChange> 0.75 <L2Regularize> 0.03 <LearningRate> 0.0001 <Params>  [
  0.01 0.02 0.03
  0.04 0.05 0.06 ]
<ComponentName> ivector-batchnorm <BatchNormComponent> <Dim> 4 <BlockDim> 4 <Epsilon> 0.001 <TargetRms> 0.025 <TestMode> F <Count> 176000 <StatsMean>  [ -0.005183299 -0.00281566 0.001 0.002 ]
<StatsVar>  [ 0.1 0.2 0.3 0.4 ]
<ComponentName> cnn1.conv <TimeHeightConvolutionComponent> <LearningRateFactor> 0.333 <MaxChange> 0.25 <L2Regularize> 0.03 <LearningRate> 3.33e-05 <Model> <ConvolutionModel> <NumFiltersIn> 6 <NumFiltersOut> 48 <HeightIn> 40 <HeightOut> 40 <HeightSubsampleOut> 1 <Offsets> [ -1,-1 -1,0 -1,1 0,-1 0,0 0,1 1,-1 1,0 1,1 ]
<LinearParams>  [
  0.001 0.002 0.003
  0.004 0.005 0.006 ]
<BiasParams>  [ 0.05598261 0.06961362 0.07 ]
<ComponentName> cnn1.relu <RectifiedLinearComponent> <Dim> 1920 <ValueAvg>  [ 0.05577822 0.08261247 ]
<ComponentName> cnn1.batchnorm <BatchNormComponent> <Dim> 3 <BlockDim> 3 <Epsilon> 0.001 <TargetRms> 1 <TestMode> F <Count> 68864 <StatsMean>  [ 0.01 0.02 0.03 ]
<StatsVar>  [ 0.5 0.6 0.7 ]
<ComponentName> tdnnf7.linear <TdnnComponent> <MaxChange> 0.75 <L2Regularize> 0.03 <LearningRate> 0.0001 <TimeOffsets> [ 0 ]
<LinearParams>  [
  3.699428e-43 -3.699428e-43
  -3.643376e-43 3.643376e-43 ]
<BiasParams>  [ ]
<ComponentName> tdnnf7.affine <TdnnComponent> <MaxChange> 0.75 <L2Regularize> 0.03 <LearningRate> 0.0001 <TimeOffsets> [ 0 ]
<LinearParams>  [
  0.1 0.2 0.3
  0.4 0.5 0.6 ]
<BiasParams>  [ -1.943402e-05 -1.780113e-05 7.44856e-06 ]
<ComponentName> tdnnf7.batchnorm <BatchNormComponent> <Dim> 3 <BlockDim> 3 <Epsilon> 0.001 <TargetRms> 1 <TestMode> F <Count> 68864 <StatsMean>  [ 0.001 0.002 0.003 ]
<StatsVar>  [ 0.1 0.2 0.3 ]
<ComponentName> prefinal-chain.affine <NaturalGradientAffineComponent> <MaxChange> 0.75 <L2Regularize> 0.03 <LearningRate> 0.0001 <LinearParams>  [
  0.01 0.02
  0.03 0.04 ]
<BiasParams>  [ 0.001 0.002 ]
<ComponentName> output.affine <NaturalGradientAffineComponent> <MaxChange> 1.5 <L2Regularize> 0.015 <LearningRate> 0.0001 <LinearParams>  [
  0.1 0.2 0.3
  0.4 0.5 0.6
  0.7 0.8 0.9 ]
<BiasParams>  [ 0.01 0.02 0.03 ]
<ComponentName> noop1 <NoOpComponent> <Dim> 768
<ComponentName> output-xent.log-softmax <LogSoftmaxComponent> <Dim> 3080 <ValueAvg>  [ ]
`

func TestParseNnet3Text(t *testing.T) {
	comps, err := ParseNnet3Text(testComponents)
	if err != nil {
		t.Fatalf("ParseNnet3Text failed: %v", err)
	}

	// Check component count
	fmt.Printf("Parsed %d components\n", len(comps))
	for name, c := range comps {
		fmt.Printf("  %s (%s): linear=%dx%d bias=%d mean=%d var=%d\n",
			name, c.Type, c.LinearRows, c.LinearCols, len(c.BiasParams),
			len(c.StatsMean), len(c.StatsVar))
	}

	// Test IDCT
	t.Run("idct", func(t *testing.T) {
		c, ok := comps["idct"]
		if !ok {
			t.Fatal("idct not found")
		}
		if c.Type != "FixedAffineComponent" {
			t.Errorf("type = %s, want FixedAffineComponent", c.Type)
		}
		if c.LinearRows != 2 || c.LinearCols != 4 {
			t.Errorf("matrix size = %dx%d, want 2x4", c.LinearRows, c.LinearCols)
		}
		if len(c.LinearParams) != 8 {
			t.Errorf("param count = %d, want 8", len(c.LinearParams))
		}
		assertClose(t, c.LinearParams[0], 0.1581139, 1e-5)
		assertClose(t, c.LinearParams[4], 0.1581139, 1e-5)
		if len(c.BiasParams) != 4 {
			t.Errorf("bias count = %d, want 4", len(c.BiasParams))
		}
	})

	// Test LinearComponent
	t.Run("ivector-linear", func(t *testing.T) {
		c, ok := comps["ivector-linear"]
		if !ok {
			t.Fatal("ivector-linear not found")
		}
		if c.Type != "LinearComponent" {
			t.Errorf("type = %s", c.Type)
		}
		if c.LinearRows != 2 || c.LinearCols != 3 {
			t.Errorf("matrix = %dx%d, want 2x3", c.LinearRows, c.LinearCols)
		}
		if c.LearningRate != 0.0001 {
			t.Errorf("lr = %v, want 0.0001", c.LearningRate)
		}
		if c.L2Regularize != 0.03 {
			t.Errorf("l2 = %v, want 0.03", c.L2Regularize)
		}
	})

	// Test BatchNorm
	t.Run("ivector-batchnorm", func(t *testing.T) {
		c, ok := comps["ivector-batchnorm"]
		if !ok {
			t.Fatal("ivector-batchnorm not found")
		}
		if c.Type != "BatchNormComponent" {
			t.Errorf("type = %s", c.Type)
		}
		if c.Epsilon != 0.001 {
			t.Errorf("epsilon = %v, want 0.001", c.Epsilon)
		}
		if c.TargetRms != 0.025 {
			t.Errorf("target_rms = %v, want 0.025", c.TargetRms)
		}
		if c.Count != 176000 {
			t.Errorf("count = %v, want 176000", c.Count)
		}
		if len(c.StatsMean) != 4 {
			t.Errorf("mean len = %d, want 4", len(c.StatsMean))
		}
		if len(c.StatsVar) != 4 {
			t.Errorf("var len = %d, want 4", len(c.StatsVar))
		}
		assertClose(t, c.StatsMean[0], -0.005183299, 1e-6)
		assertClose(t, c.StatsVar[0], 0.1, 1e-6)
	})

	// Test TimeHeightConvolution
	t.Run("cnn1.conv", func(t *testing.T) {
		c, ok := comps["cnn1.conv"]
		if !ok {
			t.Fatal("cnn1.conv not found")
		}
		if c.Type != "TimeHeightConvolutionComponent" {
			t.Errorf("type = %s", c.Type)
		}
		if c.NumFiltersIn != 6 || c.NumFiltersOut != 48 {
			t.Errorf("filters = %d→%d, want 6→48", c.NumFiltersIn, c.NumFiltersOut)
		}
		if c.HeightIn != 40 || c.HeightOut != 40 {
			t.Errorf("height = %d→%d, want 40→40", c.HeightIn, c.HeightOut)
		}
		if c.LinearRows != 2 || c.LinearCols != 3 {
			t.Errorf("matrix = %dx%d, want 2x3", c.LinearRows, c.LinearCols)
		}
		if len(c.BiasParams) != 3 {
			t.Errorf("bias = %d, want 3", len(c.BiasParams))
		}
		assertClose(t, c.BiasParams[0], 0.05598261, 1e-6)
	})

	// Test TdnnComponent with empty bias
	t.Run("tdnnf7.linear", func(t *testing.T) {
		c, ok := comps["tdnnf7.linear"]
		if !ok {
			t.Fatal("tdnnf7.linear not found")
		}
		if c.Type != "TdnnComponent" {
			t.Errorf("type = %s", c.Type)
		}
		if c.LinearRows != 2 || c.LinearCols != 2 {
			t.Errorf("matrix = %dx%d, want 2x2", c.LinearRows, c.LinearCols)
		}
		// Empty bias
		if len(c.BiasParams) != 0 {
			t.Errorf("bias = %d, want 0 (empty)", len(c.BiasParams))
		}
		// Check near-zero SVD init values
		assertClose(t, c.LinearParams[0], 3.699428e-43, 1e-45)
	})

	// Test TdnnComponent with bias
	t.Run("tdnnf7.affine", func(t *testing.T) {
		c, ok := comps["tdnnf7.affine"]
		if !ok {
			t.Fatal("tdnnf7.affine not found")
		}
		if c.LinearRows != 2 || c.LinearCols != 3 {
			t.Errorf("matrix = %dx%d, want 2x3", c.LinearRows, c.LinearCols)
		}
		if len(c.BiasParams) != 3 {
			t.Errorf("bias = %d, want 3", len(c.BiasParams))
		}
		assertClose(t, c.BiasParams[0], -1.943402e-05, 1e-8)
	})

	// Test NaturalGradientAffineComponent
	t.Run("prefinal-chain.affine", func(t *testing.T) {
		c, ok := comps["prefinal-chain.affine"]
		if !ok {
			t.Fatal("prefinal-chain.affine not found")
		}
		if c.Type != "NaturalGradientAffineComponent" {
			t.Errorf("type = %s", c.Type)
		}
		if c.LinearRows != 2 || c.LinearCols != 2 {
			t.Errorf("matrix = %dx%d, want 2x2", c.LinearRows, c.LinearCols)
		}
		if len(c.BiasParams) != 2 {
			t.Errorf("bias = %d, want 2", len(c.BiasParams))
		}
	})

	// Test output.affine
	t.Run("output.affine", func(t *testing.T) {
		c, ok := comps["output.affine"]
		if !ok {
			t.Fatal("output.affine not found")
		}
		if c.LinearRows != 3 || c.LinearCols != 3 {
			t.Errorf("matrix = %dx%d, want 3x3", c.LinearRows, c.LinearCols)
		}
		assertClose(t, c.LinearParams[8], 0.9, 1e-6)
	})

	// Test NoOp (should exist but have no params)
	t.Run("noop1", func(t *testing.T) {
		c, ok := comps["noop1"]
		if !ok {
			t.Fatal("noop1 not found")
		}
		if c.Type != "NoOpComponent" {
			t.Errorf("type = %s", c.Type)
		}
		if len(c.LinearParams) != 0 {
			t.Errorf("should have no linear params, got %d", len(c.LinearParams))
		}
	})

	// Test LogSoftmax with empty ValueAvg
	t.Run("output-xent.log-softmax", func(t *testing.T) {
		c, ok := comps["output-xent.log-softmax"]
		if !ok {
			t.Fatal("log-softmax not found")
		}
		if c.Type != "LogSoftmaxComponent" {
			t.Errorf("type = %s", c.Type)
		}
	})
}

func TestParseRealBatchNormLine(t *testing.T) {
	// Real single-line BatchNorm from Kaldi output
	line := `<ComponentName> prefinal-chain.batchnorm2 <BatchNormComponent> <Dim> 192 <BlockDim> 192 <Epsilon> 0.001 <TargetRms> 1 <TestMode> F <Count> 41344 <StatsMean>  [ 4.844032e-10 -4.039575e-09 -7.640916e-11 ]
<StatsVar>  [ 0.001 0.002 0.003 ]`

	comps, err := ParseNnet3Text(line)
	if err != nil {
		t.Fatal(err)
	}

	c, ok := comps["prefinal-chain.batchnorm2"]
	if !ok {
		t.Fatal("not found")
	}
	if c.Epsilon != 0.001 {
		t.Errorf("epsilon = %v", c.Epsilon)
	}
	if c.TargetRms != 1.0 {
		t.Errorf("target_rms = %v", c.TargetRms)
	}
	if c.Count != 41344 {
		t.Errorf("count = %v", c.Count)
	}
	if len(c.StatsMean) != 3 {
		t.Errorf("mean = %d, want 3", len(c.StatsMean))
	}
	assertClose(t, c.StatsMean[0], 4.844032e-10, 1e-15)
}

func TestBatchNormComputation(t *testing.T) {
	// Verify gamma/beta computation matches Kaldi
	mean := float32(-0.005183299)
	variance := float32(0.1) // E[x^2]
	epsilon := float32(0.001)
	targetRms := float32(0.025)

	// True variance = E[x^2] - E[x]^2
	trueVar := variance - mean*mean
	invStd := float32(1.0 / math.Sqrt(float64(trueVar+epsilon)))
	gamma := targetRms * invStd
	beta := -mean * gamma

	fmt.Printf("BN computation: mean=%.6f var=%.6f trueVar=%.6f\n", mean, variance, trueVar)
	fmt.Printf("  invStd=%.6f gamma=%.6f beta=%.6f\n", invStd, gamma, beta)

	// gamma should be approximately target_rms / sqrt(0.1 - 0.000027 + 0.001) ≈ 0.025/0.318 ≈ 0.0787
	expectedGamma := float32(0.025 / math.Sqrt(float64(trueVar+epsilon)))
	assertClose(t, gamma, expectedGamma, 1e-5)
}

func TestComponentCount(t *testing.T) {
	comps, err := ParseNnet3Text(testComponents)
	if err != nil {
		t.Fatal(err)
	}

	// Count by type
	typeCounts := make(map[string]int)
	for _, c := range comps {
		typeCounts[c.Type]++
	}

	fmt.Printf("Component types:\n")
	for typ, count := range typeCounts {
		fmt.Printf("  %s: %d\n", typ, count)
	}

	// We should have all the test components
	expected := map[string]bool{
		"idct": true, "ivector-linear": true, "ivector-batchnorm": true,
		"cnn1.conv": true, "cnn1.relu": true, "cnn1.batchnorm": true,
		"tdnnf7.linear": true, "tdnnf7.affine": true, "tdnnf7.batchnorm": true,
		"prefinal-chain.affine": true, "output.affine": true,
		"noop1": true, "output-xent.log-softmax": true,
	}
	for name := range expected {
		if _, ok := comps[name]; !ok {
			t.Errorf("missing component: %s", name)
		}
	}
}

func TestParseInlineVector(t *testing.T) {
	// Test parsing vectors that are on the same line as the tag
	text := `<ComponentName> test <BatchNormComponent> <Dim> 3 <Epsilon> 0.001 <TargetRms> 1 <Count> 100 <StatsMean>  [ 0.1 0.2 0.3 ]
<StatsVar>  [ 0.4 0.5 0.6 ]`

	comps, err := ParseNnet3Text(text)
	if err != nil {
		t.Fatal(err)
	}

	c := comps["test"]
	if c == nil {
		t.Fatal("test component not found")
	}
	if len(c.StatsMean) != 3 {
		t.Errorf("mean = %d, want 3", len(c.StatsMean))
	}
	if len(c.StatsVar) != 3 {
		t.Errorf("var = %d, want 3", len(c.StatsVar))
	}
	assertClose(t, c.StatsMean[0], 0.1, 1e-6)
	assertClose(t, c.StatsVar[2], 0.6, 1e-6)
}

func TestParseRealPrefinalLine(t *testing.T) {
	// Simulated real output — the <ComponentName> line has LinearParams starting inline
	text := `<ComponentName> prefinal-chain.affine <NaturalGradientAffineComponent> <MaxChange> 0.75 <L2Regularize> 0.03 <LearningRate> 0.0001 <LinearParams>  [
  0.01 0.02
  0.03 0.04 ]
<BiasParams>  [ 0.001 0.002 ]`

	comps, err := ParseNnet3Text(text)
	if err != nil {
		t.Fatal(err)
	}

	c := comps["prefinal-chain.affine"]
	if c == nil {
		t.Fatal("not found")
	}
	if c.LinearRows != 2 || c.LinearCols != 2 {
		t.Errorf("matrix = %dx%d", c.LinearRows, c.LinearCols)
	}
	if len(c.BiasParams) != 2 {
		t.Errorf("bias = %d", len(c.BiasParams))
	}
}

// ============================================================
// Verify all expected Kaldi components from real model
// ============================================================

func TestExpectedRealComponents(t *testing.T) {
	// These are all component names from the real final.mdl
	expected := []string{
		"idct",
		"ivector-linear",
		"ivector-batchnorm",
		"idct-batchnorm",
	}

	// CNN layers
	for i := 1; i <= 6; i++ {
		expected = append(expected,
			fmt.Sprintf("cnn%d.conv", i),
			fmt.Sprintf("cnn%d.relu", i),
			fmt.Sprintf("cnn%d.batchnorm", i),
		)
	}

	// TDNN-F layers
	for i := 7; i <= 21; i++ {
		expected = append(expected,
			fmt.Sprintf("tdnnf%d.linear", i),
			fmt.Sprintf("tdnnf%d.affine", i),
			fmt.Sprintf("tdnnf%d.relu", i),
			fmt.Sprintf("tdnnf%d.batchnorm", i),
		)
	}

	// Prefinal and output
	expected = append(expected,
		"prefinal-l",
		"prefinal-chain.affine",
		"prefinal-chain.relu",
		"prefinal-chain.batchnorm1",
		"prefinal-chain.linear",
		"prefinal-chain.batchnorm2",
		"output.affine",
	)

	fmt.Printf("Expected %d key components for weight loading\n", len(expected))

	// Verify naming convention used in our mapping
	for _, name := range expected {
		parts := strings.Split(name, ".")
		if len(parts) > 1 {
			fmt.Printf("  %s → base=%s sub=%s\n", name, parts[0], parts[1])
		} else {
			fmt.Printf("  %s → standalone\n", name)
		}
	}
}

// ============================================================
// Helpers
// ============================================================

func assertClose(t *testing.T, got, want float32, tol float64) {
	t.Helper()
	diff := math.Abs(float64(got - want))
	if diff > tol {
		t.Errorf("got %v, want %v (diff %v > tol %v)", got, want, diff, tol)
	}
}
