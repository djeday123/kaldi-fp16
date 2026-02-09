#!/bin/bash
FILE=${1:-/opt/kaldi/egs/work_3/s5/exp/chain_cnn_v2/cnn_tdnn1d_v2_sp/egs/cegs.1.ark}
NUM=${2:-1}

echo "=== Comparing example $NUM from $FILE ==="

# Our output
./bin/egstools totext -n $NUM "$FILE" 2>/dev/null > /tmp/our_full.txt

# Kaldi output - collect until </Nnet3ChainEg>
nnet3-chain-copy-egs "ark:$FILE" ark,t:- 2>/dev/null | awk -v n=$NUM '
  /<Nnet3ChainEg>/ {count++}
  count==n {print}
  count==n && /<\/Nnet3ChainEg>/ {exit}
' > /tmp/kaldi_full.txt

echo ""
echo "=== File sizes ==="
wc -l /tmp/our_full.txt /tmp/kaldi_full.txt
wc -c /tmp/our_full.txt /tmp/kaldi_full.txt

echo ""
echo "=== Index vectors (line 2) ==="
sed -n '2p' /tmp/our_full.txt | cut -c1-200
echo "---"
sed -n '2p' /tmp/kaldi_full.txt | cut -c1-200

echo ""
echo "=== First matrix row (line 3) ==="
sed -n '3p' /tmp/our_full.txt | cut -c1-150
echo "---"
sed -n '3p' /tmp/kaldi_full.txt | cut -c1-150

echo ""
echo "=== Diff summary ==="
diff /tmp/our_full.txt /tmp/kaldi_full.txt > /tmp/diff.txt 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Files are identical!"
else
    echo "❌ Files differ. Lines changed:"
    wc -l /tmp/diff.txt
    echo "First differences:"
    head -20 /tmp/diff.txt
fi
