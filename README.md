# Kaldi-FP16: Классический Kaldi на Tensor Cores

Модернизация классического Kaldi для современных GPU с поддержкой FP16 и Tensor Cores.

## Философия

```
┌─────────────────────────────────────────────────────────────┐
│                    СОХРАНЯЕМ                                │
│  • HMM-DNN архитектуру (не end-to-end!)                    │
│  • Классические рецепты Kaldi                               │
│  • Модульность и контроль над пайплайном                   │
│  • WFST декодирование                                       │
└─────────────────────────────────────────────────────────────┘
                            +
┌─────────────────────────────────────────────────────────────┐
│                    ДОБАВЛЯЕМ                                │
│  • FP16 Tensor Cores (2-4x ускорение на Ampere+)           │
│  • GoTorch для параллелизма и продакшена                   │
│  • C++ ядро с минимальными зависимостями                   │
│  • Mixed Precision Training/Inference                       │
└─────────────────────────────────────────────────────────────┘
```

## Структура

```
kaldi-fp16/
├── cpp/                    # C++ ядро (CUDA + cuBLAS FP16)
│   ├── include/           # Заголовки
│   ├── src/               # Реализация
│   └── cuda/              # CUDA кернелы для Tensor Cores
├── go/                    # Go часть
│   ├── gotorch/           # GoTorch - нативные ML операции
│   └── kaldibridge/       # CGO мост к C++ ядру
├── scripts/               # Bash скрипты в стиле Kaldi
├── examples/              # Примеры использования
└── docs/                  # Документация
```

## Требования к GPU

| GPU Architecture | Tensor Core Support | Рекомендация |
|------------------|---------------------|--------------|
| Volta (V100)     | FP16 ✓              | Поддержка    |
| Turing (RTX 20xx)| FP16 ✓              | Поддержка    |
| Ampere (A100/RTX 30xx) | FP16/TF32/BF16 ✓ | **Оптимально** |
| Ada (RTX 40xx)   | FP16/TF32/BF16/FP8 ✓| **Оптимально** |
| Hopper (H100)    | FP16/TF32/BF16/FP8 ✓| **Максимум** |

## Быстрый старт

### C++ (основное ядро)
```bash
cd cpp && mkdir build && cd build
cmake .. -DCUDA_ARCH=86  # RTX 3090
make -j$(nproc)
```

### Go (GoTorch + Bridge)
```bash
cd go/gotorch
go build ./...
go test ./...
```

## Приоритеты реализации

1. **Фаза 1**: C++ FP16 GEMM через cuBLAS (замена FP32 матричных операций)
2. **Фаза 2**: GoTorch базовые операции (tensor, matmul, activations)
3. **Фаза 3**: CGO мост Kaldi ↔ Go
4. **Фаза 4**: Полный training loop на FP16

## Зависимости

### Минимальные (без PyTorch)
- CUDA Toolkit 11.8+
- cuBLAS, cuDNN 8+
- Go 1.21+
- GCC 11+ / Clang 14+

### Опциональные
- PyTorch (только если нужен autograd для экспериментов)
- Gorgonia / Gonum (Go ML библиотеки)
