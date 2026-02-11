# Native Denominator — Integration Guide

## Что это

Замена Kaldi C++ wrapper (`kaldi_den.go` + `libkaldi_den.so`) на нативную Go + CUDA реализацию.
**Убирает зависимость** от Kaldi C++ библиотек для denominator computation.

## Файлы

```
cpp/include/chain_den.h              ← C header (новый)
cpp/cuda/chain_den.cu                ← CUDA kernels (новый)
internal/nnet/chain_den_native.go    ← Go wrapper (новый, заменяет kaldi_den.go)
cmd/denverify/main.go                ← Верификация native vs Kaldi wrapper
build_den.sh                         ← Скрипт компиляции
```

## Шаги интеграции

### 1. Скопировать файлы

```bash
cd /projects/pr2/kaldi-fp16

# CUDA
cp chain_den.h   cpp/include/
cp chain_den.cu  cpp/cuda/

# Go
cp chain_den_native.go  internal/nnet/

# Verification tool
mkdir -p cmd/denverify
cp main.go  cmd/denverify/

# Build script
cp build_den.sh .
chmod +x build_den.sh
```

### 2. Скомпилировать CUDA

**Вариант А — отдельная библиотека:**
```bash
./build_den.sh
```

Потом добавить в `chain_den_native.go` CGO directive:
```go
#cgo LDFLAGS: -L${SRCDIR}/../../cpp/build -lkaldi_fp16_den
```

**Вариант Б — включить в libkaldi_fp16.so (рекомендуется):**
```bash
cd cpp/build
nvcc -shared -o libkaldi_fp16.so \
    ../cuda/kernels.cu \
    ../cuda/cnn_kernels.cu \
    ../cuda/chain.cu \
    ../cuda/chain_den.cu \
    ../cuda/cgo_interface.cu \
    -I../include \
    -Xcompiler -fPIC \
    -arch=sm_89 \
    -O2 \
    --expt-relaxed-constexpr \
    -lcublas -lcudart
```

При Варианте Б не нужно менять CGO directives — `chain_den_native.go` уже линкует `-lkaldi_fp16`.

### 3. Верификация

```bash
# Сравнить с Kaldi wrapper на zero output
go run cmd/denverify/main.go \
    -den /opt/kaldi/egs/work_3/s5/exp/chain_cnn_v2/cnn_tdnn1d_v2_sp/den.fst \
    -egs /opt/kaldi/egs/work_3/s5/exp/chain_cnn_v2/cnn_tdnn1d_v2_sp/egs/cegs.1.ark \
    -pdfs 3080
```

Ожидаемый результат:
```
native  log_prob = -0.313154
kaldi   log_prob = -0.313154
abs_diff = X.XXXe-0X
✅ MATCH
```

### 4. Переключить chain_loss.go на NativeDenominator

В `chain_loss.go` или в `chaintest/main.go` заменить:
```go
// Было (Kaldi wrapper):
den, err := nnet.NewKaldiDenominator(denFstPath, numPdfs)

// Стало (native):
den, err := nnet.NewNativeDenominator(denFstPath, numPdfs)
```

API одинаковый: `.Forward()`, `.ForwardBackward()`, `.Free()`.

### 5. (Опционально) Удалить Kaldi wrapper

После верификации можно удалить:
- `internal/nnet/kaldi_den.go`
- `test_system/kaldi_den_wrapper.{h,cc}`
- `test_system/libkaldi_den.so`

И убрать из CGO все `-lkaldi-chain -lkaldi-cudamatrix ...` ссылки на Kaldi.

## Архитектура

```
Было:
  Go → CGO → libkaldi_den.so → Kaldi C++ → CUDA
  (зависимость: 8 Kaldi .so, OpenFst, CUDA)

Стало:
  Go → CGO → libkaldi_fp16.so → наши CUDA kernels
  (зависимость: только CUDA runtime)
```

## Алгоритм (6 особенностей Kaldi)

1. **Probability space** — `exp(nnet_output)`, не log-domain
2. **Initial probs** — 100 итераций HMM warmup (Go, CPU, float64)
3. **Leaky HMM** — `alpha += tot * 1e-5 * initial_probs`
4. **Arbitrary scaling** — `alpha /= sum(alpha)` каждый фрейм
5. **All states final** — `beta[T] = 1/S` (uniform)
6. **Transition probs** — `exp(-arc.weight)` из tropical semiring

## Потенциальные проблемы

1. **pdf indexing** — Kaldi FST labels 1-indexed, мы делаем `-1` в Go при загрузке.
   Если den.fst имеет 0-indexed labels — будет off-by-one. Проверить!

2. **Epsilon arcs** — пропускаем arcs с label=0 (epsilon). В den.fst их быть не должно.

3. **GPU sum precision** — reduction через partial sums. Для 7052 состояний
   достаточно точно, но при ~100K состояний может потребоваться Kahan summation.

4. **Multi-sequence** — пока только single sequence. Для batch нужно либо
   цикл per-sequence (как сейчас в chain_loss.go), либо batched kernel.