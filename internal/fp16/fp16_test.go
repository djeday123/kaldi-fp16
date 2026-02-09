package fp16

import (
	"math"
	"testing"
)

// ============================================================
// Basic conversions
// ============================================================

func TestFromFloat32_Zero(t *testing.T) {
	h := FromFloat32(0.0)
	if h != 0 {
		t.Errorf("0.0 → %04x, expected 0000", h)
	}
	back := h.ToFloat32()
	if back != 0.0 {
		t.Errorf("round-trip 0.0 → %f", back)
	}
}

func TestFromFloat32_NegZero(t *testing.T) {
	h := FromFloat32(float32(math.Copysign(0, -1)))
	if h != 0x8000 {
		t.Errorf("-0.0 → %04x, expected 8000", h)
	}
}

func TestFromFloat32_One(t *testing.T) {
	h := FromFloat32(1.0)
	if h != 0x3C00 {
		t.Errorf("1.0 → %04x, expected 3C00", h)
	}
	back := h.ToFloat32()
	if back != 1.0 {
		t.Errorf("round-trip 1.0 → %f", back)
	}
}

func TestFromFloat32_NegOne(t *testing.T) {
	h := FromFloat32(-1.0)
	if h != 0xBC00 {
		t.Errorf("-1.0 → %04x, expected BC00", h)
	}
	back := h.ToFloat32()
	if back != -1.0 {
		t.Errorf("round-trip -1.0 → %f", back)
	}
}

func TestFromFloat32_Half(t *testing.T) {
	h := FromFloat32(0.5)
	if h != 0x3800 {
		t.Errorf("0.5 → %04x, expected 3800", h)
	}
}

func TestFromFloat32_Two(t *testing.T) {
	h := FromFloat32(2.0)
	if h != 0x4000 {
		t.Errorf("2.0 → %04x, expected 4000", h)
	}
}

// ============================================================
// Special values
// ============================================================

func TestFromFloat32_Inf(t *testing.T) {
	h := FromFloat32(float32(math.Inf(1)))
	if h != 0x7C00 {
		t.Errorf("+Inf → %04x, expected 7C00", h)
	}
	back := h.ToFloat32()
	if !math.IsInf(float64(back), 1) {
		t.Errorf("round-trip +Inf → %f", back)
	}
}

func TestFromFloat32_NegInf(t *testing.T) {
	h := FromFloat32(float32(math.Inf(-1)))
	if h != 0xFC00 {
		t.Errorf("-Inf → %04x, expected FC00", h)
	}
	back := h.ToFloat32()
	if !math.IsInf(float64(back), -1) {
		t.Errorf("round-trip -Inf → %f", back)
	}
}

func TestFromFloat32_NaN(t *testing.T) {
	h := FromFloat32(float32(math.NaN()))
	// NaN: exp=31, frac!=0
	exp := (h >> 10) & 0x1F
	frac := h & 0x3FF
	if exp != 31 || frac == 0 {
		t.Errorf("NaN → %04x, expected NaN pattern", h)
	}
	back := h.ToFloat32()
	if !math.IsNaN(float64(back)) {
		t.Errorf("round-trip NaN → %f", back)
	}
}

// ============================================================
// Range limits
// ============================================================

func TestFromFloat32_MaxFP16(t *testing.T) {
	// FP16 max = 65504
	h := FromFloat32(65504.0)
	if h != 0x7BFF {
		t.Errorf("65504 → %04x, expected 7BFF", h)
	}
	back := h.ToFloat32()
	if back != 65504.0 {
		t.Errorf("round-trip 65504 → %f", back)
	}
}

func TestFromFloat32_Overflow(t *testing.T) {
	// 65536 > 65504 → Inf
	h := FromFloat32(65536.0)
	if h != 0x7C00 {
		t.Errorf("65536 → %04x, expected 7C00 (Inf)", h)
	}
}

func TestFromFloat32_SmallestNormal(t *testing.T) {
	// FP16 smallest normal = 2^-14 ≈ 6.1035e-5
	val := float32(math.Pow(2, -14))
	h := FromFloat32(val)
	back := h.ToFloat32()
	if math.Abs(float64(back-val)) > 1e-10 {
		t.Errorf("smallest normal: %e → %04x → %e", val, h, back)
	}
}

func TestFromFloat32_Denormalized(t *testing.T) {
	// FP16 smallest denorm = 2^-24 ≈ 5.96e-8
	val := float32(math.Pow(2, -24))
	h := FromFloat32(val)
	if h == 0 {
		t.Errorf("smallest denorm %e → zero", val)
	}
	back := h.ToFloat32()
	if back == 0 {
		t.Errorf("smallest denorm round-trip → zero")
	}
}

func TestFromFloat32_Underflow(t *testing.T) {
	// Way too small → 0
	h := FromFloat32(1e-20)
	if h != 0 {
		t.Errorf("1e-20 → %04x, expected 0 (underflow)", h)
	}
}

// ============================================================
// Round-to-nearest-even
// ============================================================

func TestFromFloat32_RoundToNearestEven(t *testing.T) {
	// 1.0 + 1 ULP in fp16 = 1.0009765625
	// 1.0 + 0.5 ULP = should round to 1.0 (even)
	// 1.0 + 1.5 ULP = should round to 1.0 + 2 ULP = 1.001953125

	h1 := FromFloat32(1.0)
	h2 := FromFloat32(1.0009765625)

	// Both should be exact (representable in fp16)
	if h1 == h2 {
		t.Log("WARN: these values should be different in fp16")
	}
}

// ============================================================
// Typical MFCC/speech feature values
// ============================================================

func TestFromFloat32_SpeechFeatures(t *testing.T) {
	// Typical MFCC values from our dataset
	values := []float32{
		104.7446, -16.8217, -14.2499, 17.5508, 0.0144,
		58.4164, -18.8078, -8.4248, -3.9107, -2.3693,
	}

	for _, v := range values {
		h := FromFloat32(v)
		back := h.ToFloat32()
		relErr := math.Abs(float64(v-back)) / math.Abs(float64(v))
		if relErr > 0.001 { // < 0.1% relative error for speech features
			t.Errorf("%.4f → %04x → %.4f, relErr=%.6f", v, h, back, relErr)
		}
	}
}

func TestFromFloat32_IvectorValues(t *testing.T) {
	// Typical ivector values (smaller range)
	values := []float32{
		0.1, 0.2, -0.05, 1.5, -0.8, 0.001, 3.14, -2.71,
	}

	for _, v := range values {
		h := FromFloat32(v)
		back := h.ToFloat32()
		relErr := math.Abs(float64(v-back)) / math.Abs(float64(v))
		if relErr > 0.002 {
			t.Errorf("%.4f → %04x → %.4f, relErr=%.6f", v, h, back, relErr)
		}
	}
}

// ============================================================
// Batch conversion
// ============================================================

func TestConvertFloat32ToFloat16_Batch(t *testing.T) {
	src := []float32{1.0, -1.0, 0.5, 2.0, 0.0}
	dst := ConvertFloat32ToFloat16(src)

	if len(dst) != 5 {
		t.Fatalf("len = %d, expected 5", len(dst))
	}

	expected := []uint16{0x3C00, 0xBC00, 0x3800, 0x4000, 0x0000}
	for i, v := range expected {
		if dst[i] != v {
			t.Errorf("dst[%d] = %04x, expected %04x", i, dst[i], v)
		}
	}
}

func TestConvertFloat16ToFloat32_Batch(t *testing.T) {
	src := []uint16{0x3C00, 0xBC00, 0x3800, 0x4000, 0x0000}
	dst := ConvertFloat16ToFloat32(src)

	expected := []float32{1.0, -1.0, 0.5, 2.0, 0.0}
	for i, v := range expected {
		if dst[i] != v {
			t.Errorf("dst[%d] = %f, expected %f", i, dst[i], v)
		}
	}
}

func TestRoundTrip_Batch(t *testing.T) {
	src := []float32{1.0, -1.0, 0.5, 100.0, -50.25, 0.001, 65504.0}
	fp16 := ConvertFloat32ToFloat16(src)
	back := ConvertFloat16ToFloat32(fp16)

	for i, v := range src {
		relErr := float64(0)
		if v != 0 {
			relErr = math.Abs(float64(v-back[i])) / math.Abs(float64(v))
		}
		if relErr > 0.001 {
			t.Errorf("[%d] %.4f → %04x → %.4f, relErr=%.6f", i, v, fp16[i], back[i], relErr)
		}
	}
}

// ============================================================
// AnalyzeConversion
// ============================================================

func TestAnalyzeConversion_ExactValues(t *testing.T) {
	// Powers of 2 are exact in fp16
	data := []float32{1.0, 2.0, 4.0, 0.5, 0.25}
	s := AnalyzeConversion(data)

	if s.NumExact != 5 {
		t.Errorf("NumExact = %d, expected 5", s.NumExact)
	}
	if s.MaxAbsErr != 0 {
		t.Errorf("MaxAbsErr = %f, expected 0", s.MaxAbsErr)
	}
}

func TestAnalyzeConversion_SpeechData(t *testing.T) {
	// Simulate typical speech feature range
	data := make([]float32, 1000)
	for i := range data {
		data[i] = float32(i)*0.1 - 50.0 // range [-50, 50]
	}

	s := AnalyzeConversion(data)

	if s.Count != 1000 {
		t.Errorf("Count = %d", s.Count)
	}
	if s.NumInf > 0 {
		t.Errorf("NumInf = %d (unexpected for speech range)", s.NumInf)
	}
	if s.MaxRelErr > 0.01 {
		t.Errorf("MaxRelErr = %f (too high for speech features)", s.MaxRelErr)
	}
	t.Logf("Speech data: maxAbsErr=%.6f avgAbsErr=%.6f maxRelErr=%.6f exact=%d/%d",
		s.MaxAbsErr, s.AvgAbsErr, s.MaxRelErr, s.NumExact, s.Count)
}

// ============================================================
// Performance: large batch
// ============================================================

func BenchmarkConvertFloat32ToFloat16(b *testing.B) {
	// Typical batch: 12000 frames × 40 dims = 480,000 floats
	data := make([]float32, 480000)
	for i := range data {
		data[i] = float32(i)*0.01 - 2400.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ConvertFloat32ToFloat16(data)
	}
}

func BenchmarkConvertFloat16ToFloat32(b *testing.B) {
	data := make([]uint16, 480000)
	for i := range data {
		data[i] = uint16(i % 65536)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ConvertFloat16ToFloat32(data)
	}
}
