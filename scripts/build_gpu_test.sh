#!/bin/bash
set -e

PROJECT_DIR="/projects/pr2/kaldi-fp16"
cd "$PROJECT_DIR"

echo "=== Building C++/CUDA libraries (CMake) ==="
cd cpp/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=89
make -j$(nproc)

echo ""
echo "Libraries:"
ls -la lib*.so 2>/dev/null || ls -la lib*.a 2>/dev/null || true

echo ""
echo "=== Building GPU test (Go) ==="
cd "$PROJECT_DIR"

export CGO_ENABLED=1
export LD_LIBRARY_PATH="${PROJECT_DIR}/cpp/build:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

mkdir -p bin
go build -o bin/gputest ./cmd/gputest

echo ""
echo "=== Running GPU test ==="
./bin/gputest "$@"