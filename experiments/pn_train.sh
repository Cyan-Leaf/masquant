#!/bin/bash

# 通用路径（按需修改或 export 覆盖）
EXP=${EXP:-$(cd "$(dirname "$0")" && pwd)}
REPO=${REPO:-$(cd "$EXP/.." && pwd)}
PY=${PY:-python3}
MODEL=${MODEL:?set MODEL to Qwen2.5-VL-3B-Instruct}
DATA_ROOT=${DATA_ROOT:-$REPO/data}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
PYCACHE=${PYCACHE:-$REPO/pycache}

# 按模态归一化 loss 的对比实验。用法: bash pn_train.sh <tag> <gpu> <ratio> <rep> <pn:0|1>
set -u
TAG=$1; GPU=$2; RATIO=$3; REP=$4; PN=$5
cd "$REPO"
mkdir -p logs
RTAG=$(echo "$RATIO" | tr -d '.')
OUT="$REPO"/outputs_par/${TAG}_pn${PN}_rt${RTAG}_r${REP}
LG="$REPO"/logs/par_${TAG}_pn${PN}_rt${RTAG}_r${REP}.log
ACT=./act_scales/Qwen2.5-VL-3B-Instruct-text-vision-mix_tr${RATIO}-128.pt
[ -f "$ACT" ] || ACT=./act_scales/Qwen2.5-VL-3B-Instruct-text-vision-mix-128.pt
setsid nohup env --chdir="$REPO" \
    MASQUANT_DATA_ROOT=$DATA_ROOT MASQUANT_WIKITEXT_DIR="$DATA_ROOT"/wikitext \
    HF_HOME=$HF_HOME PYTHONPYCACHEPREFIX=$PYCACHE \
    CUDA_VISIBLE_DEVICES=$GPU inference_mode=split_scales \
    MAS_CALI_TEXT_RATIO=$RATIO MAS_PER_MODALITY_NORM=$PN \
    "$PY" -u main.py --model "$MODEL" \
    --mode train --wbits 4 --abits 8 --symmetric --group_size 0 --cache_dir ./cache \
    --dataset-type text-vision-mix --nsamples 128 --eval_ppl --output_dir "$OUT" \
    --act-scales "$ACT" --epochs 2 --let --loss_multi_modal_mae_alpha \
    > "$LG" 2>&1 < /dev/null &
echo "launched gpu$GPU ratio=$RATIO pn=$PN r$REP"
