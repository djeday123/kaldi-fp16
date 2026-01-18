#ifndef KALDI_FP16_MATRIX_FP16_H_
#define KALDI_FP16_MATRIX_FP16_H_

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <vector>
#include <cstddef>

namespace kaldi {
namespace fp16 {

// Matrix transpose options
enum MatrixTransposeType {
  kNoTrans,
  kTrans
};

/// @class MatrixFP16
/// @brief Half-precision floating-point matrix class optimized for Tensor Cores
///
/// This class provides efficient matrix operations using FP16 (half precision)
/// and leverages NVIDIA Tensor Cores for accelerated matrix multiplication.
/// It maintains compatibility with classic Kaldi matrix operations while
/// providing significant performance improvements on modern GPUs.
class MatrixFP16 {
 public:
  /// Constructor: Creates an empty matrix
  MatrixFP16();
  
  /// Constructor: Creates a matrix with specified dimensions
  /// @param rows Number of rows
  /// @param cols Number of columns
  MatrixFP16(size_t rows, size_t cols);
  
  /// Destructor
  ~MatrixFP16();
  
  /// Copy constructor
  MatrixFP16(const MatrixFP16& other);
  
  /// Move constructor
  MatrixFP16(MatrixFP16&& other) noexcept;
  
  /// Copy assignment operator
  MatrixFP16& operator=(const MatrixFP16& other);
  
  /// Move assignment operator
  MatrixFP16& operator=(MatrixFP16&& other) noexcept;
  
  // Dimension accessors
  size_t NumRows() const { return num_rows_; }
  size_t NumCols() const { return num_cols_; }
  size_t Stride() const { return stride_; }
  
  /// Resize the matrix (may reallocate memory)
  void Resize(size_t rows, size_t cols);
  
  /// Initialize matrix with random normal distribution
  void SetRandn();
  
  /// Initialize matrix with zeros
  void SetZero();
  
  /// Copy data from host memory (FP32)
  /// @param data Host pointer to FP32 data
  void CopyFromHost(const float* data);
  
  /// Copy data to host memory (FP32)
  /// @param data Host pointer to receive FP32 data
  void CopyToHost(float* data) const;
  
  /// Copy data from host memory (FP16)
  /// @param data Host pointer to FP16 data
  void CopyFromHostFP16(const half* data);
  
  /// Copy data to host memory (FP16)
  /// @param data Host pointer to receive FP16 data
  void CopyToHostFP16(half* data) const;
  
  /// Matrix multiplication with accumulation: C = alpha * A * B + beta * C
  /// Uses Tensor Cores when available for maximum performance
  /// @param alpha Scalar multiplier for A*B
  /// @param A First input matrix
  /// @param transA Transpose option for A
  /// @param B Second input matrix
  /// @param transB Transpose option for B
  /// @param beta Scalar multiplier for C (this matrix)
  void AddMatMat(float alpha,
                 const MatrixFP16& A, MatrixTransposeType transA,
                 const MatrixFP16& B, MatrixTransposeType transB,
                 float beta);
  
  /// Get raw device pointer (for advanced users)
  half* Data() { return data_; }
  const half* Data() const { return data_; }
  
  /// Check if memory is allocated
  bool IsAllocated() const { return data_ != nullptr; }
  
 private:
  size_t num_rows_;  ///< Number of rows
  size_t num_cols_;  ///< Number of columns
  size_t stride_;    ///< Memory stride (for alignment)
  half* data_;       ///< Device memory pointer
  
  /// Allocate device memory
  void Allocate();
  
  /// Free device memory
  void Deallocate();
  
  /// Get cuBLAS handle (singleton)
  static cublasHandle_t GetCublasHandle();
};

/// Convert FP32 array to FP16 on device
/// @param dst Destination FP16 array (device)
/// @param src Source FP32 array (device)
/// @param size Number of elements
void ConvertFP32ToFP16(half* dst, const float* src, size_t size);

/// Convert FP16 array to FP32 on device
/// @param dst Destination FP32 array (device)
/// @param src Source FP16 array (device)
/// @param size Number of elements
void ConvertFP16ToFP32(float* dst, const half* src, size_t size);

}  // namespace fp16
}  // namespace kaldi

#endif  // KALDI_FP16_MATRIX_FP16_H_
