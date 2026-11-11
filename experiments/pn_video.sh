#!/bin/bash

# 通用路径（按需修改或 export 覆盖）
EXP=${EXP:-$(cd "$(dirname "$0")" && pwd)}
REPO=${REPO:-$(cd "$EXP/.." && pwd)}
PY=${PY:-python3}
MODEL=${MODEL:?set MODEL to Qwen2.5-VL-3B-Instruct}
DATA_ROOT=${DATA_ROOT:-$REPO/data}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
PYCACHE=${PYCACHE:-$REPO/pycache}

# 视频场景下按模态归一化的验证。用法: bash pn_video.sh <tag> <gpu> <share:0|1> <pn:0|1> <rep>
set -u
TAG=$1; GPU=$2; SHARE=$3; PN=$4; REP=$5
cd "$REPO"
mkdir -p logs
OUT="$REPO"/outputs_par/${TAG}_vshare${SHARE}_pn${PN}_r${REP}
LG="$REPO"/logs/par_${TAG}_vshare${SHARE}_pn${PN}_r${REP}.log
setsid nohup env --chdir="$REPO" \
    MASQUANT_DATA_ROOT=$DATA_ROOT MASQUANT_WIKITEXT_DIR="$DATA_ROOT"/wikitext \
    HF_HOME=$HF_HOME PYTHONPYCACHEPREFIX=$PYCACHE \
    CUDA_VISIBLE_DEVICES=$GPU inference_mode=split_scales \
    MAS_VIDEO_SHARE_VISION=$SHARE MAS_PER_MODALITY_NORM=$PN \
    "$PY" -u main.py --model "$MODEL" \
    --mode train --wbits 4 --abits 8 --symmetric --group_size 0 --cache_dir ./cache \
    --dataset-type text-vision-video --nsamples 64 --eval_ppl --output_dir "$OUT" \
    --epochs 2 --let --loss_multi_modal_mae_alpha \
    > "$LG" 2>&1 < /dev/null &
echo "launched gpu$GPU share=$SHARE pn=$PN r$REP"
