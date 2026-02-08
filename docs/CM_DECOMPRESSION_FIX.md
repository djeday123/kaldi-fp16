# Kaldi CM Decompression Fix

## Проблема

CM матрицы (MFCC фичи) используют format `kOneByteWithColHeaders` с:
1. Per-column headers (percentiles)
2. Column-major byte data
3. Piecewise linear interpolation (не простая линейная!)

## Структура CM формата

```
После маркера "CM ":

GlobalHeader (20 bytes):
├── int32   format     = 1 (kOneByteWithColHeaders)
├── float32 min_value  
├── float32 range
├── int32   num_rows
└── int32   num_cols

PerColHeader × num_cols (8 bytes each):
├── uint16 percentile_0    (min)
├── uint16 percentile_25   (25th percentile)
├── uint16 percentile_75   (75th percentile)
└── uint16 percentile_100  (max)

ByteData (num_rows × num_cols bytes, COLUMN-MAJOR):
└── [col0_row0][col0_row1]...[col0_rowN][col1_row0]...[colM_rowN]
```

## Формулы декомпрессии

### 1. Конвертация percentile uint16 → float32

```go
func Uint16ToFloat(minValue, rangeVal float32, value uint16) float32 {
    // 1/65535 = 1.52590218966964e-05
    return minValue + rangeVal * 1.52590218966964e-05 * float32(value)
}
```

### 2. Конвертация byte → float32 (CharToFloat)

```go
func CharToFloat(p0, p25, p75, p100 float32, value uint8) float32 {
    if value <= 64 {
        // Range [0, 64] → linear from p0 to p25
        return p0 + (p25-p0)*float32(value)*(1.0/64.0)
    } else if value <= 192 {
        // Range [64, 192] → linear from p25 to p75
        return p25 + (p75-p25)*float32(value-64)*(1.0/128.0)
    } else {
        // Range [192, 255] → linear from p75 to p100
        return p75 + (p100-p75)*float32(value-192)*(1.0/63.0)
    }
}
```

## Полный код readCompressedMatrix

```go
// PerColHeader structure
type PerColHeader struct {
    Percentile0   uint16
    Percentile25  uint16
    Percentile75  uint16
    Percentile100 uint16
}

func (ar *ArkReader) readCompressedMatrix() *Matrix {
    // After "CM" marker, read space
    ar.binary.ReadSingleByte()

    // Read GlobalHeader (20 bytes)
    format := ar.binary.ReadInt32()      // should be 1
    minValue := ar.binary.ReadFloat32()
    rangeVal := ar.binary.ReadFloat32()
    numRows := ar.binary.ReadInt32()
    numCols := ar.binary.ReadInt32()

    if ar.binary.Err() != nil {
        return nil
    }

    // Validate
    if format != 1 {
        ar.binary.SetErr(fmt.Errorf("CM expected format=1, got %d", format))
        return nil
    }
    if numRows <= 0 || numCols <= 0 || numRows > 100000 || numCols > 10000 {
        ar.binary.SetErr(fmt.Errorf("invalid dims: %dx%d", numRows, numCols))
        return nil
    }

    // Read PerColHeaders (8 bytes each)
    colHeaders := make([]PerColHeader, numCols)
    for i := int32(0); i < numCols; i++ {
        colHeaders[i].Percentile0 = ar.binary.ReadUint16()
        colHeaders[i].Percentile25 = ar.binary.ReadUint16()
        colHeaders[i].Percentile75 = ar.binary.ReadUint16()
        colHeaders[i].Percentile100 = ar.binary.ReadUint16()
    }

    if ar.binary.Err() != nil {
        return nil
    }

    // Read byte data (column-major)
    dataSize := int(numRows) * int(numCols)
    byteData := make([]byte, dataSize)
    n, _ := io.ReadFull(ar.reader, byteData)
    if n != dataSize {
        ar.binary.SetErr(fmt.Errorf("short read: got %d, want %d", n, dataSize))
        return nil
    }

    // Helper: convert uint16 percentile to float
    uint16ToFloat := func(val uint16) float32 {
        return minValue + rangeVal*1.52590218966964e-05*float32(val)
    }

    // Helper: convert byte to float using percentiles
    charToFloat := func(p0, p25, p75, p100 float32, value uint8) float32 {
        if value <= 64 {
            return p0 + (p25-p0)*float32(value)*(1.0/64.0)
        } else if value <= 192 {
            return p25 + (p75-p25)*float32(value-64)*(1.0/128.0)
        } else {
            return p75 + (p100-p75)*float32(value-192)*(1.0/63.0)
        }
    }

    // Decompress to row-major matrix
    m := NewMatrix(int(numRows), int(numCols))

    for col := int32(0); col < numCols; col++ {
        // Get float percentiles for this column
        p0 := uint16ToFloat(colHeaders[col].Percentile0)
        p25 := uint16ToFloat(colHeaders[col].Percentile25)
        p75 := uint16ToFloat(colHeaders[col].Percentile75)
        p100 := uint16ToFloat(colHeaders[col].Percentile100)

        // Decompress column (input is column-major)
        for row := int32(0); row < numRows; row++ {
            byteIdx := col*numRows + row  // column-major index
            val := charToFloat(p0, p25, p75, p100, byteData[byteIdx])
            m.Set(int(row), int(col), val)  // output is row-major
        }
    }

    return m
}
```

## CM2 (kTwoByte) — проще

```go
func (ar *ArkReader) readCompressedMatrix2() *Matrix {
    // After "CM2" marker, read space
    ar.binary.ReadSingleByte()

    // Read GlobalHeader
    format := ar.binary.ReadInt32()      // should be 2
    minValue := ar.binary.ReadFloat32()
    rangeVal := ar.binary.ReadFloat32()
    numRows := ar.binary.ReadInt32()
    numCols := ar.binary.ReadInt32()

    if format != 2 || numRows <= 0 || numCols <= 0 {
        ar.binary.SetErr(fmt.Errorf("invalid CM2: format=%d, dims=%dx%d", 
            format, numRows, numCols))
        return nil
    }

    // Read uint16 data (row-major, no per-column headers)
    dataSize := int(numRows) * int(numCols)
    m := NewMatrix(int(numRows), int(numCols))
    increment := rangeVal / 65535.0

    for i := 0; i < dataSize; i++ {
        val := ar.binary.ReadUint16()
        m.Data[i] = minValue + float32(val)*increment
    }

    return m
}
```

## Тест: ожидаемые значения

Для первой строки MFCC (которую показывает Kaldi как `[123.15, 13.59, -3.31, ...]`):
- Если получаешь NaN или огромные числа → проблема в чтении percentiles или формуле
- Tolerance для CM: ~0.5 (8-bit квантизация)
- Tolerance для CM2: ~0.01 (16-bit квантизация)

## Команда для проверки

```bash
cd /projects/pr2/kaldi-fp16
# После исправления кода:
./egstools compare -n 1 /data/kaldi-data/exp/chain_cnn_v2/cnn_tdnn_50h_bn192_l2_001/egs/cegs.1.ark
```

Должно показать совпадение значений в пределах tolerance.