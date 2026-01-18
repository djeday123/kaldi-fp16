# NVIDIA Tensor Cores Programming Guide

## Introduction

NVIDIA Tensor Cores are specialized hardware units designed to accelerate matrix multiplication operations. This guide explains how Kaldi-FP16 leverages Tensor Cores for maximum performance.

## What are Tensor Cores?

Tensor Cores are dedicated hardware units that perform mixed-precision matrix multiply-accumulate operations:

```
D = A × B + C
```

Where:
- A, B, C, D are matrices
- A and B are FP16 (half precision)
- Accumulation happens in FP32 (single precision)
- Output D can be FP16 or FP32

### Hardware Specifications

#### Operation per Clock

| Architecture | Operation | Size | TFLOPS (FP16) |
|--------------|-----------|------|---------------|
| Volta (V100) | FP16×FP16+FP32 | 4×4×4 | 125 |
| Turing (RTX 2080 Ti) | FP16×FP16+FP32 | 8×8×4 | 54 |
| Ampere (A100) | FP16×FP16+FP32 | 8×8×4 | 312 |
| Ada (RTX 4090) | FP16×FP16+FP32 | 8×8×4 | 165 |
| Hopper (H100) | FP16×FP16+FP32 | 8×8×4 | 756 |

## How Kaldi-FP16 Uses Tensor Cores

### 1. Automatic Tensor Core Usage

Kaldi-FP16 automatically uses Tensor Cores through cuBLAS:

```cpp
MatrixFP16 A(M, K);
MatrixFP16 B(K, N);
MatrixFP16 C(M, N);

// Automatically uses Tensor Cores if:
// - GPU supports Tensor Cores (compute capability ≥ 7.0)
// - Matrices are FP16
// - cuBLAS is configured for Tensor Core math
C.AddMatMat(1.0f, A, kNoTrans, B, kNoTrans, 0.0f);
```

### 2. Under the Hood

When you call `AddMatMat`, Kaldi-FP16:

1. Validates matrix dimensions
2. Gets cuBLAS handle (configured for Tensor Core math)
3. Calls `cublasGemmEx` with:
   - Input type: `CUDA_R_16F` (FP16)
   - Output type: `CUDA_R_16F` (FP16)
   - Compute type: `CUBLAS_COMPUTE_32F` (FP32 accumulation)
   - Algorithm: `CUBLAS_GEMM_DEFAULT_TENSOR_OP`

### 3. Math Mode

Tensor Core math mode is enabled by default:

```cpp
cublasHandle_t handle = GetCublasHandle();
cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
```

You can control this manually if needed:

```cpp
#include "kaldi-fp16/tensor-ops.h"

cublasHandle_t handle;
cublasCreate(&handle);

// Enable Tensor Cores
EnableTensorCoreMath(handle);

// Disable Tensor Cores (use CUDA cores only)
DisableTensorCoreMath(handle);
```

## Performance Characteristics

### Matrix Size Requirements

Tensor Cores have optimal performance for specific matrix sizes:

#### Optimal Sizes (Multiple of 8)

```cpp
// Excellent performance
MatrixFP16 A(1024, 1024);  // 1024 = 128 × 8
MatrixFP16 B(2048, 2048);  // 2048 = 256 × 8
MatrixFP16 C(4096, 4096);  // 4096 = 512 × 8
```

#### Suboptimal Sizes (Not multiple of 8)

```cpp
// Will work but with padding overhead
MatrixFP16 A(1000, 1000);  // Padded to 1008
MatrixFP16 B(1500, 1500);  // Padded to 1504
```

### Memory Alignment

Kaldi-FP16 automatically aligns memory for Tensor Core operations:

```cpp
// stride_ is automatically set to multiple of 8
MatrixFP16 A(100, 100);  // stride = 104
MatrixFP16 B(250, 250);  // stride = 256
```

## Computation Patterns

### Basic Matrix Multiplication

```cpp
// C = A × B
C.AddMatMat(1.0f, A, kNoTrans, B, kNoTrans, 0.0f);
```

**Tensor Core computation:**
- Tiles A into 8×8 blocks
- Tiles B into 8×8 blocks
- Computes 8×8×8 products using Tensor Cores
- Accumulates in FP32
- Outputs FP16

### Matrix Multiplication with Accumulation

```cpp
// C = alpha * A × B + beta * C
C.AddMatMat(alpha, A, kNoTrans, B, kNoTrans, beta);
```

**Use cases:**
- Fused multiply-add operations
- Accumulating results over multiple batches
- Implementing attention mechanisms

### Transposed Operations

```cpp
// C = A^T × B
C.AddMatMat(1.0f, A, kTrans, B, kNoTrans, 0.0f);

// C = A × B^T
C.AddMatMat(1.0f, A, kNoTrans, B, kTrans, 0.0f);

// C = A^T × B^T
C.AddMatMat(1.0f, A, kTrans, B, kTrans, 0.0f);
```

## Advanced Usage

### Mixed Precision Workflow

```cpp
// High precision master weights (FP32)
std::vector<float> master_weights(M * K);

// FP16 working copy
MatrixFP16 weights_fp16(M, K);
weights_fp16.CopyFromHost(master_weights.data());

// Forward pass with FP16
MatrixFP16 input(N, M);
MatrixFP16 output(N, K);
output.AddMatMat(1.0f, input, kNoTrans, weights_fp16, kNoTrans, 0.0f);

// Convert back to FP32 for weight updates
weights_fp16.CopyToHost(master_weights.data());
// Update master_weights in FP32...
```

### Batched Operations

For processing multiple independent matrix multiplications:

```cpp
const int num_batches = 10;
const int M = 512, K = 512, N = 512;

// Allocate once
MatrixFP16 A(M, K);
MatrixFP16 B(K, N);
MatrixFP16 C(M, N);

for (int i = 0; i < num_batches; ++i) {
  LoadBatch(i, A, B);
  C.AddMatMat(1.0f, A, kNoTrans, B, kNoTrans, 0.0f);
  ProcessResults(C);
}
```

### Chained Operations

```cpp
// Forward pass through multiple layers
MatrixFP16 layer1_out(batch, hidden1);
MatrixFP16 layer2_out(batch, hidden2);
MatrixFP16 layer3_out(batch, output);

// Layer 1: input -> hidden1
layer1_out.AddMatMat(1.0f, input, kNoTrans, weights1, kNoTrans, 0.0f);

// Layer 2: hidden1 -> hidden2
layer2_out.AddMatMat(1.0f, layer1_out, kNoTrans, weights2, kNoTrans, 0.0f);

// Layer 3: hidden2 -> output
layer3_out.AddMatMat(1.0f, layer2_out, kNoTrans, weights3, kNoTrans, 0.0f);
```

## Performance Optimization

### Tip 1: Maximize Tensor Core Utilization

```cpp
// Good: Large matrices fully utilize Tensor Cores
MatrixFP16 A(4096, 4096);  // Excellent utilization

// Bad: Small matrices have overhead
MatrixFP16 B(64, 64);      // Poor utilization
```

### Tip 2: Minimize Data Transfer

```cpp
// Bad: Frequent transfers
for (int i = 0; i < N; ++i) {
  MatrixFP16 temp(M, K);
  temp.CopyFromHost(data[i]);  // Transfer overhead
  // ... computation ...
}

// Good: Batch transfers
MatrixFP16 all_data(N * M, K);
all_data.CopyFromHost(all_data_host);  // Single transfer
```

### Tip 3: Reuse Allocations

```cpp
// Bad: Repeated allocation
for (int i = 0; i < N; ++i) {
  MatrixFP16 temp(M, K);  // Allocation overhead
  // ... computation ...
}

// Good: Single allocation
MatrixFP16 temp(M, K);  // Allocate once
for (int i = 0; i < N; ++i) {
  // ... computation with temp ...
}
```

## Debugging Tensor Core Usage

### Check Availability

```cpp
#include "kaldi-fp16/tensor-ops.h"

if (TensorCoresAvailable()) {
  int cc = GetTensorCoreComputeCapability();
  std::cout << "Tensor Cores available: CC " << cc / 10 << "." << cc % 10 << "\n";
} else {
  std::cout << "Tensor Cores NOT available\n";
}
```

### Verify Performance

```cpp
// Benchmark with and without Tensor Cores
float time_tensor = BenchmarkMatMul(4096, 4096, 4096, true, 100);
float time_cuda = BenchmarkMatMul(4096, 4096, 4096, false, 100);

float speedup = time_cuda / time_tensor;
std::cout << "Tensor Core speedup: " << speedup << "x\n";
```

### Profile with NVIDIA Tools

```bash
# Use Nsight Systems
nsys profile --trace=cuda,nvtx ./your_app

# Use Nsight Compute for detailed kernel analysis
ncu --set full --target-processes all ./your_app
```

**Look for:**
- `sm__sass_thread_inst_executed_op_tensor_*` metrics
- Tensor Core utilization percentage
- Memory bandwidth utilization

## Common Pitfalls

### 1. Not Actually Using Tensor Cores

**Problem:** Code doesn't show expected speedup

**Check:**
```cpp
// Verify Tensor Cores are available
assert(TensorCoresAvailable());

// Verify math mode is correct
cublasHandle_t handle = GetCublasHandle();
cublasMath_t mode;
cublasGetMathMode(handle, &mode);
assert(mode == CUBLAS_TENSOR_OP_MATH);
```

### 2. Matrix Sizes Too Small

**Problem:** Overhead dominates for small matrices

**Solution:** Use Tensor Cores only for M, N, K > 256

### 3. Memory Bandwidth Bottleneck

**Problem:** Computation faster than data transfer

**Solution:**
- Increase computational intensity
- Fuse operations
- Batch multiple operations

## Best Practices

1. ✅ Use matrix dimensions that are multiples of 8
2. ✅ Reuse allocated matrices
3. ✅ Batch operations to reduce overhead
4. ✅ Profile to verify Tensor Core usage
5. ✅ Use FP32 accumulation for numerical stability
6. ❌ Don't use for very small matrices (< 256)
7. ❌ Don't transfer data unnecessarily
8. ❌ Don't allocate in hot loops

## Further Reading

- [NVIDIA Tensor Cores Whitepaper](https://www.nvidia.com/en-us/data-center/tensor-cores/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
- [Ampere Architecture](https://www.nvidia.com/en-us/data-center/ampere-architecture/)
- [Hopper Architecture](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/)
