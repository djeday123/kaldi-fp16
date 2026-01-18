# Kaldi-FP16: Classic Kaldi on Tensor Cores

## 🌍 Multilingual Description

**🇬🇧 English:**  
A modernization of the classic Kaldi speech recognition toolkit for modern GPUs with FP16 and Tensor Core support. This project enables significant performance improvements through half-precision floating-point operations and NVIDIA Tensor Core acceleration.

**🇹🇷 Türkçe:**  
FP16 ve Tensor Çekirdek desteğiyle modern GPU'lar için klasik Kaldi konuşma tanıma araç setinin modernleştirilmiş bir sürümü. Bu proje, yarı hassasiyetli kayan nokta işlemleri ve NVIDIA Tensor Çekirdek hızlandırması ile önemli performans iyileştirmeleri sağlar.

**🇦🇿 Azərbaycanca:**  
FP16 və Tensor Nüvə dəstəyi ilə müasir qrafik kartları üçün klassik Kaldi nitq tanıma alətlər dəstinin modernləşdirilməsi. Bu layihə yarım dəqiqlikli kayan nöqtə əməliyyatları və NVIDIA Tensor Nüvə sürətləndirilməsi vasitəsilə əhəmiyyətli performans təkmilləşdirmələri təmin edir.

**🇷🇺 Русский:**  
Модернизация классического набора инструментов распознавания речи Kaldi для современных GPU с поддержкой FP16 и Tensor Cores. Этот проект обеспечивает значительное улучшение производительности за счет операций с плавающей запятой половинной точности и ускорения NVIDIA Tensor Core.

## ✨ Features

- **🚀 FP16 Support**: Half-precision floating-point operations for faster computation
- **⚡ Tensor Core Acceleration**: Leverages NVIDIA Tensor Cores for matrix operations
- **🔧 Classic Kaldi Compatible**: Maintains compatibility with classic Kaldi workflows
- **📊 Performance Optimized**: Significant speedup on modern GPUs (RTX 20xx/30xx/40xx, A100, H100)
- **💾 Memory Efficient**: Reduced memory footprint with FP16 representations
- **🔄 Mixed Precision**: Automatic mixed precision training support

## 📋 Requirements

### Hardware
- NVIDIA GPU with Tensor Core support (Compute Capability 7.0+)
  - Volta (V100), Turing (RTX 20xx), Ampere (RTX 30xx, A100), Ada Lovelace (RTX 40xx), Hopper (H100)

### Software
- CUDA Toolkit 11.0 or later
- cuBLAS library
- cuDNN (optional, for neural network operations)
- CMake 3.18 or later
- C++14 compatible compiler (GCC 7+, Clang 9+)
- Python 3.7+ (for utilities and examples)

## 🛠️ Installation

### Build from Source

```bash
# Clone the repository
git clone https://github.com/djeday123/kaldi-fp16.git
cd kaldi-fp16

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCUDA_ARCH="70;75;80;86;89;90"

# Build
make -j$(nproc)

# Install (optional)
sudo make install
```

### Build Options

- `CUDA_ARCH`: Target CUDA architectures (default: auto-detect)
- `ENABLE_TENSOR_CORES`: Enable Tensor Core operations (default: ON)
- `ENABLE_FP16`: Enable FP16 support (default: ON)
- `BUILD_EXAMPLES`: Build example programs (default: ON)
- `BUILD_TESTS`: Build unit tests (default: ON)

## 🎯 Quick Start

### Basic FP16 Matrix Multiplication

```cpp
#include "kaldi-fp16/matrix-fp16.h"

// Create FP16 matrices
MatrixFP16 A(1024, 1024);
MatrixFP16 B(1024, 1024);
MatrixFP16 C(1024, 1024);

// Initialize matrices
A.SetRandn();
B.SetRandn();

// Perform matrix multiplication using Tensor Cores
C.AddMatMat(1.0, A, kNoTrans, B, kNoTrans, 0.0);
```

### Performance Benchmark

```bash
# Run performance benchmarks
./build/examples/benchmark_fp16

# Compare FP32 vs FP16 performance
./build/examples/compare_precision
```

## 📚 Documentation

- [API Reference](docs/api-reference.md)
- [Performance Guide](docs/performance-guide.md)
- [Migration from Classic Kaldi](docs/migration-guide.md)
- [Tensor Core Programming](docs/tensor-cores.md)

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original Kaldi project: https://github.com/kaldi-asr/kaldi
- NVIDIA CUDA and cuBLAS teams for Tensor Core support
- The speech recognition research community

## 📞 Contact

For questions and support, please open an issue on GitHub.

## 🔗 Related Projects

- [Kaldi](https://github.com/kaldi-asr/kaldi) - Original Kaldi speech recognition toolkit
- [PyTorch-Kaldi](https://github.com/mravanelli/pytorch-kaldi) - PyTorch integration for Kaldi
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) - Conversational AI toolkit
