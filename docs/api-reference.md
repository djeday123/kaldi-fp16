# Kaldi-FP16 API Reference

## Core Classes

### MatrixFP16

The `MatrixFP16` class provides half-precision floating-point matrix operations optimized for NVIDIA Tensor Cores.

#### Constructor

```cpp
MatrixFP16();
MatrixFP16(size_t rows, size_t cols);
```

Creates a matrix with specified dimensions. Memory is allocated on the GPU.

#### Matrix Operations

##### AddMatMat

```cpp
void AddMatMat(float alpha,
               const MatrixFP16& A, MatrixTransposeType transA,
               const MatrixFP16& B, MatrixTransposeType transB,
               float beta);
```

Performs matrix multiplication with accumulation: `C = alpha * A * B + beta * C`

**Parameters:**
- `alpha`: Scalar multiplier for the product A*B
- `A`: First input matrix
- `transA`: Transpose option for A (`kNoTrans` or `kTrans`)
- `B`: Second input matrix
- `transB`: Transpose option for B (`kNoTrans` or `kTrans`)
- `beta`: Scalar multiplier for C (this matrix)

**Example:**
```cpp
MatrixFP16 A(1024, 512);
MatrixFP16 B(512, 1024);
MatrixFP16 C(1024, 1024);

A.SetRandn();
B.SetRandn();
C.SetZero();

// C = A * B
C.AddMatMat(1.0, A, kNoTrans, B, kNoTrans, 0.0);
```

#### Initialization Methods

##### SetZero

```cpp
void SetZero();
```

Sets all elements to zero.

##### SetRandn

```cpp
void SetRandn();
```

Initializes matrix with random values from normal distribution (mean=0, std=1).

#### Data Transfer

##### CopyFromHost / CopyToHost

```cpp
void CopyFromHost(const float* data);
void CopyToHost(float* data) const;
```

Copy data between host (CPU) and device (GPU) with automatic FP32 ↔ FP16 conversion.

##### CopyFromHostFP16 / CopyToHostFP16

```cpp
void CopyFromHostFP16(const half* data);
void CopyToHostFP16(half* data) const;
```

Copy FP16 data directly without conversion.

#### Accessors

```cpp
size_t NumRows() const;
size_t NumCols() const;
size_t Stride() const;
half* Data();
const half* Data() const;
bool IsAllocated() const;
```

#### Memory Management

```cpp
void Resize(size_t rows, size_t cols);
```

Resizes the matrix, reallocating memory if necessary.

## Tensor Core Operations

### Initialization

```cpp
void InitTensorCoreOps();
```

Initializes Tensor Core operations. Should be called once at program startup.

### Query Functions

```cpp
bool TensorCoresAvailable();
int GetTensorCoreComputeCapability();
```

Check if Tensor Cores are available and get compute capability.

### Performance Optimization

```cpp
void EnableTensorCoreMath(cublasHandle_t handle);
void DisableTensorCoreMath(cublasHandle_t handle);
```

Control whether cuBLAS uses Tensor Cores.

### Benchmarking

```cpp
float BenchmarkMatMul(int m, int n, int k, 
                     bool use_tensor_cores,
                     int num_iterations = 100);

float GetTFLOPS(int m, int n, int k, float time_ms);
```

Performance measurement utilities.

## Type Definitions

```cpp
enum MatrixTransposeType {
  kNoTrans,  // No transpose
  kTrans     // Transpose
};
```

## Utility Functions

### Type Conversion

```cpp
void ConvertFP32ToFP16(half* dst, const float* src, size_t size);
void ConvertFP16ToFP32(float* dst, const half* src, size_t size);
```

Convert between FP32 and FP16 on device.

## Error Handling

All functions may throw `std::runtime_error` on CUDA or cuBLAS errors.

## Thread Safety

- Matrix objects are not thread-safe
- Multiple matrices can be used concurrently on different streams
- cuBLAS handle is internally managed with a singleton pattern

## Memory Layout

- Matrices use row-major layout with padding for alignment
- Stride is aligned to 8 elements for optimal Tensor Core performance
- Memory is allocated on GPU device

## Performance Tips

1. **Matrix Sizes**: Use sizes that are multiples of 8 for best Tensor Core performance
2. **Batch Operations**: Reuse matrices to avoid allocation overhead
3. **Mixed Precision**: Use FP32 accumulation for better numerical stability
4. **Memory Access**: Aligned memory access patterns improve performance

## Example: Complete Workflow

```cpp
#include "kaldi-fp16/matrix-fp16.h"
#include "kaldi-fp16/tensor-ops.h"

using namespace kaldi::fp16;

int main() {
  // Initialize
  InitTensorCoreOps();
  
  // Check device capabilities
  if (!TensorCoresAvailable()) {
    std::cerr << "Tensor Cores not available\n";
    return 1;
  }
  
  // Create matrices
  const int M = 2048, K = 2048, N = 2048;
  MatrixFP16 A(M, K);
  MatrixFP16 B(K, N);
  MatrixFP16 C(M, N);
  
  // Initialize
  A.SetRandn();
  B.SetRandn();
  C.SetZero();
  
  // Compute C = A * B
  C.AddMatMat(1.0f, A, kNoTrans, B, kNoTrans, 0.0f);
  
  // Copy result to host
  std::vector<float> result(M * N);
  C.CopyToHost(result.data());
  
  return 0;
}
```

## Compiler Requirements

- C++14 or later
- CUDA 11.0 or later
- NVIDIA GPU with compute capability 7.0 or higher

## See Also

- [Performance Guide](performance-guide.md)
- [Migration Guide](migration-guide.md)
- [Tensor Cores Programming](tensor-cores.md)
