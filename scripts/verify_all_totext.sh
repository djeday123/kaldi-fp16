#!/bin/bash
# Verify totext output against Kaldi for all .ark files

EGSDIR="/opt/kaldi/egs/work_3/s5/exp/chain_cnn_v2/cnn_tdnn1d_v2_sp/egs"
TOOL="./bin/egstools"
LOGFILE="/projects/pr2/kaldi-fp16/verify_totext.log"

echo "=== Verify totext against Kaldi ===" | tee $LOGFILE
echo "Started: $(date)" | tee -a $LOGFILE
echo "" | tee -a $LOGFILE

PASSED=0
FAILED=0

for ark in $EGSDIR/cegs.*.ark; do
    name=$(basename $ark)
    
    # Our output (example 1)
    $TOOL totext -n 1 "$ark" 2>/dev/null > /tmp/our_ex.txt
    
    # Kaldi output (example 1)
    nnet3-chain-copy-egs "ark:$ark" ark,t:- 2>/dev/null | awk '
      /<Nnet3ChainEg>/ {count++}
      count==1 {print}
      count==1 && /<\/Nnet3ChainEg>/ {exit}
    ' > /tmp/kaldi_ex.txt
    
    # Compare
    if diff -q /tmp/our_ex.txt /tmp/kaldi_ex.txt > /dev/null 2>&1; then
        echo "✅ $name" | tee -a $LOGFILE
        ((PASSED++))
    else
        echo "❌ $name" | tee -a $LOGFILE
        # Show first difference
        diff /tmp/our_ex.txt /tmp/kaldi_ex.txt | head -5 >> $LOGFILE
        ((FAILED++))
    fi
done

echo "" | tee -a $LOGFILE
echo "=== Summary ===" | tee -a $LOGFILE
echo "Passed: $PASSED" | tee -a $LOGFILE
echo "Failed: $FAILED" | tee -a $LOGFILE
echo "Finished: $(date)" | tee -a $LOGFILE
