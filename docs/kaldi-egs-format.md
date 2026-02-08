# Kaldi EGS: Структуры и форматы данных

## 1. Иерархия структур Kaldi
```
NnetChainExample
├── inputs: vector<NnetIo>          // обычно 2 элемента
│   ├── [0] NnetIo "input"          // MFCC фичи [frames × 40]
│   │   ├── name: "input"
│   │   ├── indexes: vector<Index>  // t=-31..171 (с контекстом)
│   │   └── features: GeneralMatrix // CM compressed [224×40]
│   └── [1] NnetIo "ivector"        // i-vector [1 × 100]
│       ├── name: "ivector"
│       ├── indexes: vector<Index>  // t=0 (один фрейм)
│       └── features: GeneralMatrix // CM2 compressed [1×100]
└── outputs: vector<NnetChainSupervision>  // обычно 1 элемент
    └── [0] NnetChainSupervision "output"
        ├── name: "output"
        ├── indexes: vector<Index>  // subsampled frames
        ├── supervision: chain::Supervision (FST граф)
        └── deriv_weights: Vector   // веса для backprop

Index (информация о фрейме):
├── n: int32  // индекс в минибатче (0 для отдельных примеров)
├── t: int32  // временной индекс (frame number)
└── x: int32  // дополнительный индекс (обычно 0)
```

## 2. Бинарный формат файла (.ark)
```
[key]\x20\x00B                    // "example_id" + space + \x00 + 'B' (binary)
<Nnet3ChainEg>                    // пример начался
<NumInputs>\x20\x04[int32]        // количество inputs (обычно 2)

┌─ Input 0: MFCC features ─────────────────────────────────────────────┐
│ <NnetIo>\x20"input"\x20         // тег + имя                         │
│ <I1V>\x20\x04[int32=224]        // количество индексов               │
│ [224 bytes delta-encoded]       // индексы (дельта-сжатие)           │
│   • byte[0]: signed_char(t[0])  // первый t напрямую (если |t|<125)  │
│   • byte[i]: delta = t[i]-t[i-1]// последующие - дельты              │
│   • если byte==127: читать 15 bytes (size+n, size+t, size+x)         │
│ CM\x20[min:f32][range:f32][rows:i32][cols:i32]  // GlobalHeader      │
│ [cols×8 bytes PerColHeaders]    // p0,p25,p75,p100 для каждой колонки│
│ [rows×cols bytes data]          // column-major, 1 byte per value    │
│ </NnetIo>                                                            │
└──────────────────────────────────────────────────────────────────────┘

┌─ Input 1: i-vector ──────────────────────────────────────────────────┐
│ <NnetIo>\x20"ivector"\x20                                            │
│ <I1V>\x20\x04[int32=1]          // 1 индекс                          │
│ [1 byte: 0x00]                  // t=0                               │
│ CM2\x20[min:f32][range:f32][rows:i32][cols:i32]  // GlobalHeader     │
│ [rows×cols×2 bytes]             // uint16 per value, row-major       │
│ </NnetIo>                                                            │
└──────────────────────────────────────────────────────────────────────┘

<NumOutputs>\x20\x04[int32=1]
<NnetChainSup>\x20"output"\x20<I1V>...<Supervision>...<DW2>...</NnetChainSup>
</Nnet3ChainEg>
```

## 3. Форматы сжатия матриц

### CM (kOneByteWithColHeaders)
- **Используется для**: MFCC features [frames × 40]
- **Размер**: 16 + cols×8 + rows×cols bytes
```
GlobalHeader: min(f32) + range(f32) + rows(i32) + cols(i32)
PerColHeaders[cols]: p0(u16) + p25(u16) + p75(u16) + p100(u16)
Data[rows×cols]: 1 byte per value, column-major order
```

**Декомпрессия**:
```
percentile = min + range × (uint16_value / 65535)
value = piecewise_linear(p0, p25, p75, p100, byte_value)
  • [0-64]    → linear(p0, p25)
  • [64-192]  → linear(p25, p75)
  • [192-255] → linear(p75, p100)
```

### CM2 (kTwoByte)
- **Используется для**: i-vectors [1 × 100]
- **Размер**: 16 + rows×cols×2 bytes
```
GlobalHeader: min(f32) + range(f32) + rows(i32) + cols(i32)
Data[rows×cols]: 2 bytes (uint16) per value, row-major order
```

**Декомпрессия**: `value = min + (uint16_value / 65535) × range`

### CM3 (kOneByte)
- **Размер**: 16 + rows×cols bytes
- **Декомпрессия**: `value = min + (uint8_value / 255) × range`

### FM (Full Matrix)
- **Размер**: 1 + 4 + 4 + rows×cols×4 bytes
- **Header**: size_byte(1) + rows(i32) + cols(i32)
- **Data**: float32 per value, row-major order

## 4. Сравнение: Kaldi vs Наш парсер

| Компонент | Kaldi C++ | Наш Go парсер |
|-----------|-----------|---------------|
| Основная структура | `NnetChainExample` | `Example` |
| Входы | `vector<NnetIo> inputs` | `[]IoBlock Inputs` |
| Индексы фреймов | `vector<Index> indexes` | `int Size` (только count) |
| Матрица фич | `GeneralMatrix features` | `MatrixInfo Matrix` |
| Supervision | `NnetChainSupervision` | `SupervisionInfo Supervision` |
| Чтение индексов | `ReadIndexVector()` - полное | `skipIndexVector()` - пропуск |
| Чтение матрицы | `GeneralMatrix::Read()` | `ReadCompressedMatrix/2/3/Full()` |
| Декомпрессия | `CopyToMat()` - полная | Полная (Data []float32) ✅ |
| Использование | Training loop (nnet3-train) | Валидация + будет DataLoader |

## 5. Pipeline создания и использования EGS

### Создание (nnet3-chain-get-egs)
1. Загрузка: `feats.scp` + `ivectors.scp` + supervision (lattice FST)
2. Нарезка: `UtteranceSplitter` делит utterance на chunks
3. Сжатие: `GeneralMatrix.Compress()` → CM/CM2
4. Запись: `NnetChainExample.Write()` → binary ark

### Merge (nnet3-chain-merge-egs)
1. Группировка по структуре (`NnetChainExampleStructureHasher`)
2. `MergeChainExamples()` - объединение в минибатчи
3. n-индексы становятся > 0 (batch index)

### Использование (nnet3-chain-train)
1. `NnetChainExample.Read()` - чтение бинарного формата
2. `GeneralMatrix.GetMatrix()` - декомпрессия в `Matrix<float>`
3. `CuMatrix.CopyFromMat()` - копирование на GPU
4. Forward/Backward pass

### Наш Pipeline (Go + CUDA)
1. `parser.ReadExample()` - чтение бинарного формата
2. GPU декомпрессия CM→FP16 (Tensor Cores ready)
3. Batching на GPU
4. Forward/Backward с FP16 precision

## 6. Статистика датасета (2600h)

- **Файлов**: 297 (cegs.1.ark - cegs.297.ark)
- **Примеров**: 6,534,076
- **Valid**: 100%
- **Frame sizes**: 164, 203, 224 frames

## 7. Найденные баги при разработке парсера

1. **`io.ReadFull` vs `Read`**: `bufio.Reader.Read()` может вернуть меньше байт чем запрошено
2. **Index delta encoding**: длинный формат (byte==127) требует 15 байт (3×(1+4)), не 12
3. **Column-major vs Row-major**: CM использует column-major, CM2/CM3 - row-major
