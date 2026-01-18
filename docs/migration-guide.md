# Migration Guide: From Classic Kaldi to Kaldi-FP16

This guide helps you migrate existing Kaldi code to use Kaldi-FP16 with Tensor Core acceleration.

## Overview

Kaldi-FP16 provides a modernized API that maintains conceptual compatibility with classic Kaldi while leveraging modern GPU capabilities.

## Key Differences

### 1. Precision

**Classic Kaldi:**
- Uses FP32 (single precision) for matrix operations
- Higher memory usage
- Standard CUDA core computation

**Kaldi-FP16:**
- Uses FP16 (half precision) for matrix storage and computation
- 2× less memory usage
- Tensor Core acceleration for matrix operations
- FP32 accumulation for numerical stability

### 2. Matrix Operations

#### Classic Kaldi API (Example)

```cpp
// Classic Kaldi (pseudo-code)
Matrix<float> A(1024, 1024);
Matrix<float> B(1024, 1024);
Matrix<float> C(1024, 1024);

A.SetRandn();
B.SetRandn();
C.SetZero();

// C = A * B
C.AddMatMat(1.0, A, kNoTrans, B, kNoTrans, 0.0);
```

#### Kaldi-FP16 API

```cpp
// Kaldi-FP16
#include "kaldi-fp16/matrix-fp16.h"

using namespace kaldi::fp16;

MatrixFP16 A(1024, 1024);
MatrixFP16 B(1024, 1024);
MatrixFP16 C(1024, 1024);

A.SetRandn();
B.SetRandn();
C.SetZero();

// C = A * B (using Tensor Cores)
C.AddMatMat(1.0, A, kNoTrans, B, kNoTrans, 0.0);
```

**Key Changes:**
- Replace `Matrix<float>` with `MatrixFP16`
- Add `#include "kaldi-fp16/matrix-fp16.h"`
- Add namespace: `using namespace kaldi::fp16;`
- API remains the same!

## Step-by-Step Migration

### Step 1: Update Build System

Add Kaldi-FP16 to your CMakeLists.txt:

```cmake
find_package(KaldiFP16 REQUIRED)
include_directories(${KaldiFP16_INCLUDE_DIRS})
target_link_libraries(your_target kaldi-fp16)
```

### Step 2: Replace Matrix Types

Search and replace matrix declarations:

```cpp
// Before
Matrix<float> features;
Matrix<float> weights;

// After
MatrixFP16 features;
MatrixFP16 weights;
```

### Step 3: Update Include Statements

```cpp
// Before
#include "matrix/kaldi-matrix.h"

// After
#include "kaldi-fp16/matrix-fp16.h"
using namespace kaldi::fp16;
```

### Step 4: Handle Data Transfer

If you have existing FP32 data:

```cpp
// Load existing FP32 data
std::vector<float> fp32_data = LoadData();

// Create FP16 matrix and copy
MatrixFP16 matrix(rows, cols);
matrix.CopyFromHost(fp32_data.data());

// Process with Tensor Cores...

// Copy back to FP32 if needed
matrix.CopyToHost(fp32_data.data());
```

### Step 5: Initialize Tensor Cores

Add initialization at program startup:

```cpp
#include "kaldi-fp16/tensor-ops.h"

int main() {
  try {
    kaldi::fp16::InitTensorCoreOps();
  } catch (const std::exception& e) {
    std::cerr << "Tensor Core init failed: " << e.what() << "\n";
    return 1;
  }
  
  // Your code...
}
```

## Common Migration Patterns

### Pattern 1: Feature Extraction

**Before (Classic Kaldi):**
```cpp
Matrix<float> features(num_frames, feat_dim);
Matrix<float> transform(feat_dim, output_dim);
Matrix<float> output(num_frames, output_dim);

// Apply transformation
output.AddMatMat(1.0, features, kNoTrans, transform, kNoTrans, 0.0);
```

**After (Kaldi-FP16):**
```cpp
MatrixFP16 features(num_frames, feat_dim);
MatrixFP16 transform(feat_dim, output_dim);
MatrixFP16 output(num_frames, output_dim);

// Apply transformation (with Tensor Cores!)
output.AddMatMat(1.0, features, kNoTrans, transform, kNoTrans, 0.0);
```

### Pattern 2: Neural Network Forward Pass

**Before:**
```cpp
Matrix<float> input(batch_size, input_dim);
Matrix<float> weights(input_dim, hidden_dim);
Matrix<float> hidden(batch_size, hidden_dim);

hidden.AddMatMat(1.0, input, kNoTrans, weights, kNoTrans, 0.0);
// Apply activation...
```

**After:**
```cpp
MatrixFP16 input(batch_size, input_dim);
MatrixFP16 weights(input_dim, hidden_dim);
MatrixFP16 hidden(batch_size, hidden_dim);

hidden.AddMatMat(1.0, input, kNoTrans, weights, kNoTrans, 0.0);
// Apply activation...
```

### Pattern 3: Batch Processing

**Before:**
```cpp
for (int i = 0; i < num_batches; ++i) {
  Matrix<float> batch_data = GetBatch(i);
  Matrix<float> result(batch_size, output_dim);
  
  result.AddMatMat(1.0, batch_data, kNoTrans, weights, kNoTrans, 0.0);
  ProcessResult(result);
}
```

**After (More Efficient):**
```cpp
// Allocate once, reuse
MatrixFP16 batch_data(batch_size, input_dim);
MatrixFP16 weights_fp16(input_dim, output_dim);
MatrixFP16 result(batch_size, output_dim);

// Load weights once
weights_fp16.CopyFromHost(weights.data());

for (int i = 0; i < num_batches; ++i) {
  LoadBatchToGPU(i, batch_data);
  result.AddMatMat(1.0, batch_data, kNoTrans, weights_fp16, kNoTrans, 0.0);
  ProcessResult(result);
}
```

## Precision Considerations

### When FP16 is Safe

✅ **Recommended:**
- Feature extraction
- Matrix multiplication in neural networks
- Transform operations
- Most signal processing

### When to Use Caution

⚠️ **Mixed Precision Recommended:**
- Loss computation (use FP32)
- Gradient computation (use FP32)
- Weight updates (maintain FP32 master weights)
- Very deep networks (>100 layers)

### Numerical Stability Tips

1. **Use FP32 for critical computations:**
```cpp
// Compute loss in FP32
std::vector<float> predictions_fp32(n);
matrix_fp16.CopyToHost(predictions_fp32.data());
float loss = ComputeLoss(predictions_fp32, labels);  // FP32
```

2. **Gradient scaling (for training):**
```cpp
// Scale gradients to prevent underflow
const float loss_scale = 1024.0f;
scaled_loss = loss * loss_scale;
// Backward pass...
// Unscale gradients before update
```

## Performance Optimization

### Before Migration Checklist

- [ ] Profile existing code to identify bottlenecks
- [ ] Check that matrix operations dominate runtime
- [ ] Verify GPU has Tensor Core support (compute capability ≥ 7.0)
- [ ] Ensure matrix sizes are reasonably large (>512×512)

### After Migration Checklist

- [ ] Verify Tensor Cores are being used
- [ ] Benchmark performance improvement
- [ ] Check numerical accuracy
- [ ] Profile memory usage
- [ ] Test on target hardware

## Troubleshooting

### Issue: No performance improvement

**Possible causes:**
- Matrix sizes too small (< 256×256)
- GPU doesn't support Tensor Cores
- Memory transfer overhead dominates
- Not actually using Tensor Cores

**Solutions:**
```cpp
// Check Tensor Core availability
if (!TensorCoresAvailable()) {
  std::cerr << "Tensor Cores not available!\n";
}

// Benchmark to verify speedup
float time_ms = BenchmarkMatMul(m, n, k, true, 100);
```

### Issue: Accuracy loss

**Possible causes:**
- Numerical instability in FP16
- Gradient underflow (training)

**Solutions:**
- Use mixed precision
- Apply loss scaling
- Maintain FP32 master weights

### Issue: Out of memory

**Solutions:**
- Process in smaller batches
- FP16 should use less memory than FP32
- Check for memory leaks

## Complete Example

Here's a complete example showing migration:

```cpp
// Before: Classic Kaldi
#include "matrix/kaldi-matrix.h"
using namespace kaldi;

void ProcessAudio(const std::vector<float>& audio) {
  Matrix<float> features = ExtractFeatures(audio);
  Matrix<float> weights = LoadWeights();
  Matrix<float> output(features.NumRows(), weights.NumCols());
  
  output.AddMatMat(1.0, features, kNoTrans, weights, kNoTrans, 0.0);
  SaveOutput(output);
}

// After: Kaldi-FP16
#include "kaldi-fp16/matrix-fp16.h"
#include "kaldi-fp16/tensor-ops.h"
using namespace kaldi::fp16;

void ProcessAudio(const std::vector<float>& audio) {
  // Initialize Tensor Cores (once at startup)
  static bool initialized = false;
  if (!initialized) {
    InitTensorCoreOps();
    initialized = true;
  }
  
  // Extract features (FP32)
  std::vector<float> features_fp32 = ExtractFeatures(audio);
  std::vector<float> weights_fp32 = LoadWeights();
  
  // Create FP16 matrices
  MatrixFP16 features(num_frames, feat_dim);
  MatrixFP16 weights(feat_dim, output_dim);
  MatrixFP16 output(num_frames, output_dim);
  
  // Copy to GPU with FP32→FP16 conversion
  features.CopyFromHost(features_fp32.data());
  weights.CopyFromHost(weights_fp32.data());
  
  // Process with Tensor Cores
  output.AddMatMat(1.0, features, kNoTrans, weights, kNoTrans, 0.0);
  
  // Copy back with FP16→FP32 conversion
  std::vector<float> output_fp32(num_frames * output_dim);
  output.CopyToHost(output_fp32.data());
  
  SaveOutput(output_fp32);
}
```

## Next Steps

1. Review the [API Reference](api-reference.md)
2. Study the [Performance Guide](performance-guide.md)
3. Run the included examples
4. Benchmark your specific use case
5. Monitor numerical accuracy

## Support

For questions about migration:
- Open an issue on GitHub
- Check existing issues for similar problems
- Review example code in `examples/`
