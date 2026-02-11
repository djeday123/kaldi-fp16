#!/bin/bash
# Build native denominator CUDA library
# Run from: /projects/pr2/kaldi-fp16/
#
# This compiles chain_den.cu into the existing libkaldi_fp16.so
# or as a separate library if preferred.

set -e

CUDA_PATH=/usr/local/cuda-12.8
NVCC=${CUDA_PATH}/bin/nvcc
BUILD_DIR=cpp/build
INCLUDE_DIR=cpp/include
CUDA_DIR=cpp/cuda

echo "=== Building native denominator ==="

# Option A: Compile as separate library
${NVCC} -shared -o ${BUILD_DIR}/libkaldi_fp16_den.so \
    ${CUDA_DIR}/chain_den.cu \
    -I${INCLUDE_DIR} \
    -Xcompiler -fPIC \
    -arch=sm_89 \
    -O2 \
    --expt-relaxed-constexpr \
    -lcudart

echo "Built: ${BUILD_DIR}/libkaldi_fp16_den.so"

# Option B: Rebuild libkaldi_fp16.so including chain_den.cu
# Uncomment if you want everything in one library:
#
# ${NVCC} -shared -o ${BUILD_DIR}/libkaldi_fp16.so \
#     ${CUDA_DIR}/kernels.cu \
#     ${CUDA_DIR}/cnn_kernels.cu \
#     ${CUDA_DIR}/chain.cu \
#     ${CUDA_DIR}/chain_den.cu \
#     ${CUDA_DIR}/cgo_interface.cu \
#     -I${INCLUDE_DIR} \
#     -Xcompiler -fPIC \
#     -arch=sm_89 \
#     -O2 \
#     --expt-relaxed-constexpr \
#     -lcublas -lcudart
#
# echo "Rebuilt: ${BUILD_DIR}/libkaldi_fp16.so (with chain_den)"

echo ""
echo "=== Done ==="
echo ""
echo "To use separate library, add to chain_den_native.go CGO directives:"
echo '  #cgo LDFLAGS: -L${SRCDIR}/../../cpp/build -lkaldi_fp16_den'
echo ""
echo "Or rebuild libkaldi_fp16.so with Option B (uncomment in script)"