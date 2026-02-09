package fp16

import (
	"math"
)

// Float16 represents IEEE 754 half-precision floating point (16 bit)
// Used for Tensor Core operations on GPU
type Float16 uint16

// FromFloat32 converts float32 to float16
// Uses round-to-nearest-even (banker's rounding)
func FromFloat32(f float32) Float16 {
	b := math.Float32bits(f)

	sign := (b >> 31) & 1
	exp := int((b >> 23) & 0xFF)
	frac := b & 0x7FFFFF

	switch {
	case exp == 255:
		// Inf or NaN
		if frac == 0 {
			// Inf
			return Float16(sign<<15 | 0x7C00)
		}
		// NaN — preserve some bits
		return Float16(sign<<15 | 0x7C00 | (frac >> 13))

	case exp > 142:
		// Overflow → Inf
		return Float16(sign<<15 | 0x7C00)

	case exp > 112:
		// Normalized number
		// FP32 bias=127, FP16 bias=15 → offset=112
		newExp := uint32(exp - 112)
		// Round-to-nearest-even
		round := frac & 0x1FFF // bits that will be dropped
		frac >>= 13
		if round > 0x1000 || (round == 0x1000 && (frac&1) != 0) {
			frac++
			if frac > 0x3FF {
				frac = 0
				newExp++
				if newExp > 30 {
					return Float16(sign<<15 | 0x7C00) // overflow
				}
			}
		}
		return Float16(sign<<15 | newExp<<10 | frac)

	case exp > 101:
		// Denormalized (subnormal in fp16)
		shift := uint(113 - exp) // how many extra bits to shift right
		frac |= 0x800000         // add implicit leading 1
		// Round-to-nearest-even
		round := frac & ((1 << (shift + 13)) - 1)
		half := uint32(1 << (shift + 12))
		frac >>= (shift + 13)
		if round > half || (round == half && (frac&1) != 0) {
			frac++
		}
		return Float16(sign<<15 | frac)

	default:
		// Too small → zero (preserve sign)
		return Float16(sign << 15)
	}
}

// ToFloat32 converts float16 back to float32
func (h Float16) ToFloat32() float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1F
	frac := uint32(h) & 0x3FF

	switch {
	case exp == 31:
		// Inf or NaN
		if frac == 0 {
			return math.Float32frombits(sign<<31 | 0x7F800000)
		}
		return math.Float32frombits(sign<<31 | 0x7F800000 | (frac << 13))

	case exp == 0:
		if frac == 0 {
			// Zero
			return math.Float32frombits(sign << 31)
		}
		// Denormalized → normalize
		for frac&0x400 == 0 {
			frac <<= 1
			exp--
		}
		frac &= 0x3FF
		exp++
		return math.Float32frombits(sign<<31 | (exp+112)<<23 | frac<<13)

	default:
		// Normalized
		return math.Float32frombits(sign<<31 | (exp+112)<<23 | frac<<13)
	}
}

// ============================================================
// Batch conversion (for training pipeline)
// ============================================================

// ConvertFloat32ToFloat16 converts a float32 slice to float16 in-place
// Returns the float16 data as uint16 slice (same memory layout as GPU expects)
func ConvertFloat32ToFloat16(src []float32) []uint16 {
	dst := make([]uint16, len(src))
	for i, v := range src {
		dst[i] = uint16(FromFloat32(v))
	}
	return dst
}

// ConvertFloat16ToFloat32 converts float16 (uint16) back to float32
func ConvertFloat16ToFloat32(src []uint16) []float32 {
	dst := make([]float32, len(src))
	for i, v := range src {
		dst[i] = Float16(v).ToFloat32()
	}
	return dst
}

// Stats reports precision loss statistics for a conversion
type Stats struct {
	Count     int
	MaxAbsErr float32 // max |fp32 - fp16→fp32|
	AvgAbsErr float32
	MaxRelErr float32 // max relative error (for non-zero values)
	NumInf    int     // values that became Inf
	NumZero   int     // values that became zero (underflow)
	NumExact  int     // values with no precision loss

	// Worst-case values (the ones that caused MaxRelErr)
	WorstOriginal  float32 // original FP32 value
	WorstConverted float32 // after FP16 round-trip
}

// AnalyzeConversion compares original float32 with round-tripped fp16→fp32
func AnalyzeConversion(data []float32) Stats {
	s := Stats{Count: len(data)}
	var totalAbsErr float64

	for _, v := range data {
		h := FromFloat32(v)
		back := h.ToFloat32()

		absErr := float32(math.Abs(float64(v - back)))
		totalAbsErr += float64(absErr)

		if absErr > s.MaxAbsErr {
			s.MaxAbsErr = absErr
		}

		if v != 0 {
			relErr := absErr / float32(math.Abs(float64(v)))
			if relErr > s.MaxRelErr {
				s.MaxRelErr = relErr
				s.WorstOriginal = v
				s.WorstConverted = back
			}
		}

		if math.IsInf(float64(back), 0) && !math.IsInf(float64(v), 0) {
			s.NumInf++
		}
		if back == 0 && v != 0 {
			s.NumZero++
		}
		if back == v {
			s.NumExact++
		}
	}

	if s.Count > 0 {
		s.AvgAbsErr = float32(totalAbsErr / float64(s.Count))
	}
	return s
}
