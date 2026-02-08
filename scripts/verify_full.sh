#!/bin/bash
# Полная верификация парсера: все строки, несколько файлов

EGSTOOLS=/projects/pr2/kaldi-fp16/bin/egstools
EGS_DIR=/opt/kaldi/egs/work_3/s5/exp/chain_cnn_v2/cnn_tdnn1d_v2_sp/egs

# Цвета
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

verify_example() {
    local file=$1
    local example_num=$2
    local tolerance=${3:-0.001}
    
    # Получаем данные от Kaldi (пропускаем 2 строки заголовка)
    kaldi_data=$(nnet3-chain-copy-egs "ark:$file" ark,t:- 2>/dev/null | \
        awk -v n=$example_num 'BEGIN{eg=0; indata=0} 
            /^[^ ]/ {eg++; indata=0} 
            eg==n && /\[/ {indata=1; next}
            eg==n && indata && !/\]/ {for(i=1;i<=NF;i++) print $i}
            eg==n && /\]/ {for(i=1;i<NF;i++) print $i; exit}')
    
    # Получаем данные от нашего парсера
    our_data=$($EGSTOOLS dump -n $example_num -r -1 "$file" 2>/dev/null | \
        grep -v "^\[" | grep -v "^$" | awk '{for(i=1;i<=NF;i++) print $i}')
    
    # Сравниваем
    paste <(echo "$kaldi_data") <(echo "$our_data") | \
    awk -v tol=$tolerance -v file="$file" -v ex=$example_num '
    BEGIN {errors=0; total=0}
    {
        total++
        diff = $1 - $2
        if (diff < 0) diff = -diff
        if (diff > tol) {
            errors++
            if (errors <= 5) printf "  Line %d: kaldi=%.7g ours=%.7g diff=%.7g\n", total, $1, $2, diff
        }
    }
    END {
        if (errors > 0) {
            printf "FAIL: %d/%d values differ (>%.6f)\n", errors, total, tol
            exit 1
        } else {
            printf "OK: %d values match\n", total
            exit 0
        }
    }'
}

verify_file() {
    local file=$1
    local num_examples=${2:-3}
    
    echo "=== Verifying: $(basename $file) ==="
    
    for i in $(seq 1 $num_examples); do
        printf "  Example %d: " $i
        if verify_example "$file" $i; then
            echo -e "${GREEN}✓${NC}"
        else
            echo -e "${RED}✗${NC}"
            return 1
        fi
    done
}

# Главная функция
main() {
    local files=${1:-5}        # количество файлов
    local examples=${2:-10}    # примеров на файл
    
    echo "Full verification: $files files × $examples examples"
    echo ""
    
    failed=0
    for f in $(ls $EGS_DIR/cegs.*.ark | shuf | head -$files); do
        if ! verify_file "$f" $examples; then
            ((failed++))
        fi
        echo ""
    done
    
    echo "========================================"
    if [ $failed -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
    else
        echo -e "${RED}$failed files failed${NC}"
        exit 1
    fi
}

main "$@"
