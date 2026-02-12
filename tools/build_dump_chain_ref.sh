#!/bin/bash
# Build dump_chain_ref against Kaldi libraries
#
# Usage: bash build_dump_chain_ref.sh

set -e

KALDI=/opt/kaldi/src
SRC=./tools/dump_chain_ref.cc
OUT=./tools/dump_chain_ref

# Get Kaldi build flags
KALDI_MK=$KALDI/kaldi.mk
if [ ! -f "$KALDI_MK" ]; then
    echo "ERROR: $KALDI_MK not found"
    exit 1
fi

# Extract key settings from kaldi.mk
CUDA_INCLUDE=$(grep 'CUDA_INCLUDE' $KALDI_MK | head -1 | sed 's/.*=//' | tr -d ' ')
CUDA_FLAGS=$(grep 'CUDA_FLAGS' $KALDI_MK | head -1 | sed 's/.*=//')

echo "=== Building dump_chain_ref ==="
echo "Kaldi: $KALDI"
echo "Source: $SRC"
echo "Output: $OUT"

g++ -std=c++17 -O2 \
    -I$KALDI \
    -I$KALDI/../tools/openfst/include \
    -I/usr/local/cuda/include \
    -DHAVE_CUDA=1 \
    -DKALDI_PARANOID \
    $SRC \
    -o $OUT \
    -L$KALDI/lib \
    -Wl,-rpath,$KALDI/lib \
    -lkaldi-chain \
    -lkaldi-nnet3 \
    -lkaldi-cudamatrix \
    -lkaldi-matrix \
    -lkaldi-lat \
    -lkaldi-hmm \
    -lkaldi-tree \
    -lkaldi-fstext \
    -lkaldi-gmm \
    -lkaldi-transform \
    -lkaldi-util \
    -lkaldi-base \
    -L/usr/local/cuda/lib64 \
    -lcudart -lcublas -lcusparse -lcurand -lcusolver \
    -L/opt/kaldi/tools/openfst-1.7.2/src/lib/.libs -Wl,-rpath,/opt/kaldi/tools/openfst-1.7.2/src/lib/.libs -lfst \
    -lm -lpthread -ldl

echo "=== Build successful ==="
echo "Binary: $OUT"
echo ""
echo "Usage:"
echo "  $OUT \\"
echo "    /opt/kaldi/egs/work_3/s5/exp/chain_cnn_v2/cnn_tdnn1d_v2_sp/den.fst \\"
echo "    'ark:/opt/kaldi/egs/work_3/s5/exp/chain_cnn_v2/cnn_tdnn1d_v2_sp/egs/cegs.15.ark' \\"
echo "    /tmp/chain_ref"