# Performance Guide

## Tensor Core Performance Characteristics

### Hardware Overview

#### GPU Architectures with Tensor Cores

| Architecture | Compute Cap. | GPUs | FP16 TFLOPS (Boost) |
|--------------|--------------|------|---------------------|
| Volta        | 7.0          | V100 | 125 (FP16 Tensor)  |
| Turing       | 7.5          | RTX 2080 Ti | 54 (FP16 Tensor) |
| Ampere       | 8.0          | A100 | 312 (FP16 Tensor)  |
| Ampere       | 8.6          | RTX 3090 | 71 (FP16 Tensor) |
| Ada Lovelace | 8.9          | RTX 4090 | 165 (FP16 Tensor) |
| Hopper       | 9.0          | H100 | 756 (FP16 Tensor)  |

### Performance Optimization Tips

#### 1. Matrix Size Selection

**Optimal Sizes:**
- Use matrix dimensions that are multiples of 8 (preferably 16 or 32)
- Tensor Cores operate on 8×8 tiles in FP16
- Padding is automatically handled but can add overhead

```cpp
// Good: Multiple of 8
MatrixFP16 A(1024, 1024);  // Excellent
MatrixFP16 B(2048, 2048);  // Excellent

// Acceptable: Will be padded
MatrixFP16 C(1000, 1000);  // Padded to 1008
```

#### 2. Memory Alignment

The library automatically aligns stride to 8 elements for optimal performance:

```cpp
MatrixFP16 A(100, 100);
// stride_ is set to 104 (next multiple of 8)
```

#### 3. Batch Processing

Process multiple operations together to maximize GPU utilization:

```cpp
// Less efficient: Multiple small operations
for (int i = 0; i < 100; ++i) {
  small_C.AddMatMat(1.0, small_A, kNoTrans, small_B, kNoTrans, 0.0);
}

// More efficient: Single large operation
large_C.AddMatMat(1.0, large_A, kNoTrans, large_B, kNoTrans, 0.0);
```

#### 4. Memory Management

Reuse matrices to avoid allocation overhead:

```cpp
// Inefficient
void process_frames() {
  for (int frame = 0; frame < num_frames; ++frame) {
    MatrixFP16 temp(1024, 1024);  // Repeated allocation
    temp.AddMatMat(/*...*/);
  }
}

// Efficient
void process_frames() {
  MatrixFP16 temp(1024, 1024);  // Single allocation
  for (int frame = 0; frame < num_frames; ++frame) {
    temp.AddMatMat(/*...*/);
  }
}
```

## Performance Benchmarks

### Expected Speedups

FP16 with Tensor Cores vs FP32 on CUDA Cores:

| Matrix Size | RTX 3090 | A100 | H100 |
|-------------|----------|------|------|
| 1024×1024×1024 | 2-3× | 5-7× | 10-15× |
| 2048×2048×2048 | 3-4× | 7-10× | 15-20× |
| 4096×4096×4096 | 4-5× | 10-15× | 20-30× |

### Measuring Performance

Use the provided benchmark tool:

```bash
./build/examples/benchmark_fp16
```

Or use the API directly:

```cpp
#include "kaldi-fp16/tensor-ops.h"

// Benchmark 4096×4096×4096 multiplication
float time_ms = BenchmarkMatMul(4096, 4096, 4096, true, 100);
float tflops = GetTFLOPS(4096, 4096, 4096, time_ms);

std::cout << "Performance: " << tflops << " TFLOPS\n";
```

## Mixed Precision Strategies

### Automatic Mixed Precision

The library uses FP32 accumulation by default for numerical stability:

```cpp
// Input: FP16, Computation: FP16 on Tensor Cores, Accumulation: FP32
C.AddMatMat(1.0f, A, kNoTrans, B, kNoTrans, 0.0f);
```

### When to Use FP16

✅ **Good use cases:**
- Large matrix multiplications (> 512×512)
- Neural network inference
- Signal processing with bounded dynamic range
- Feature extraction

⚠️ **Use caution:**
- Very deep neural networks (consider gradient scaling)
- Algorithms requiring high precision
- Small matrices (< 256×256) - overhead may dominate

❌ **Avoid FP16:**
- Loss computation (use FP32)
- Gradient updates (use FP32 master weights)
- Small vectors (no Tensor Core benefit)

## Profiling and Debugging

### Using NVIDIA Profiler

```bash
# Profile with nsys
nsys profile --trace=cuda,nvtx ./your_app

# Profile with ncu
ncu --set full ./your_app
```

### Performance Counters

Key metrics to monitor:
- **Tensor Core utilization**: Should be > 80% for large matrices
- **Memory bandwidth**: Check for memory bottlenecks
- **Kernel launch overhead**: Minimize for small operations

### Debugging Performance Issues

1. **Check matrix sizes**: Ensure they're multiples of 8
2. **Verify Tensor Cores are enabled**:
   ```cpp
   if (!TensorCoresAvailable()) {
     std::cerr << "Tensor Cores not available!\n";
   }
   ```
3. **Profile with smaller problem sizes first**
4. **Compare against cuBLAS directly** to isolate issues

## Common Performance Pitfalls

### 1. Frequent Host-Device Transfers

```cpp
// Bad: Frequent transfers
for (int i = 0; i < N; ++i) {
  std::vector<float> host_data = get_data(i);
  matrix.CopyFromHost(host_data.data());
  matrix.AddMatMat(/*...*/);
  matrix.CopyToHost(result.data());
}

// Good: Batch transfers
std::vector<float> all_data = get_all_data();
matrix.CopyFromHost(all_data.data());
// Process on GPU
matrix.CopyToHost(all_results.data());
```

### 2. Synchronization Overhead

Avoid unnecessary `cudaDeviceSynchronize()` calls. The library only synchronizes when necessary.

### 3. Small Matrix Operations

For matrices smaller than 256×256, FP32 on CUDA cores may be faster due to Tensor Core overhead.

### 4. Unaligned Accesses

The library handles alignment automatically, but custom kernels should respect alignment.

## Optimization Checklist

- [ ] Matrix sizes are multiples of 8
- [ ] Reusing matrices instead of repeated allocation
- [ ] Minimizing host-device transfers
- [ ] Using appropriate precision for each operation
- [ ] Batching operations when possible
- [ ] Profiled with NVIDIA tools
- [ ] Verified Tensor Core utilization

## Advanced Optimization

### CUDA Streams

For advanced users, multiple streams can be used for concurrent operations:

```cpp
// Create multiple cuBLAS handles with different streams
// (requires manual handle management)
```

### Multi-GPU

For multi-GPU setups, create separate matrix instances per GPU:

```cpp
cudaSetDevice(0);
MatrixFP16 A_gpu0(M, K);

cudaSetDevice(1);
MatrixFP16 A_gpu1(M, K);
```

## Further Reading

- [NVIDIA Tensor Cores Documentation](https://docs.nvidia.com/cuda/cublas/)
- [Mixed Precision Training Paper](https://arxiv.org/abs/1710.03740)
- [cuBLAS Performance Guide](https://docs.nvidia.com/cuda/cublas/index.html#performance)
