#!/bin/bash

# 通用路径（按需修改或 export 覆盖）
EXP=${EXP:-$(cd "$(dirname "$0")" && pwd)}
REPO=${REPO:-$(cd "$EXP/.." && pwd)}
PY=${PY:-python3}
MODEL=${MODEL:?set MODEL to Qwen2.5-VL-3B-Instruct}
DATA_ROOT=${DATA_ROOT:-$REPO/data}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
PYCACHE=${PYCACHE:-$REPO/pycache}

# 用 C4（非 wikitext 领域）测各 checkpoint 的 PPL，检验混合校准的收益是否只是同域匹配。
# 用法: bash c4_ppl.sh <tag> <gpu> <mode> <ckpt|none> <extra...>
set -u
TAG=$1; GPU=$2; MODE=$3; CK=$4; shift 4
cd "$REPO"
mkdir -p logs
RES=""
[ "$CK" != "none" ] && RES="--resume $CK"
env --chdir="$REPO" \
    MASQUANT_DATA_ROOT=$DATA_ROOT MASQUANT_WIKITEXT_DIR="$DATA_ROOT"/wikitext \
    HF_HOME=$HF_HOME PYTHONPYCACHEPREFIX=$PYCACHE \
    CUDA_VISIBLE_DEVICES=$GPU inference_mode=$MODE \
    MAS_PPL_DATASETS=${MAS_PPL_DATASETS:-c4} \
    "$PY" -u main.py --model "$MODEL" \
    --mode train --epochs 0 --symmetric --group_size 0 --cache_dir ./cache \
    --wbits ${WB:-4} --abits ${AB:-8} --dataset-type text-vision --nsamples 128 \
    --eval_ppl --output_dir "$REPO"/outputs_c4ppl/$TAG $RES "$@" \
    > logs/c4_$TAG.log 2>&1
echo "$TAG $(grep -o 'c4 : [0-9.]*' logs/c4_$TAG.log | tail -1)"
